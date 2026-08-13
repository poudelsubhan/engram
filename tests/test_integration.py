"""Live-cluster tests. These are the ones that actually prove the engine works:
$vectorSearch ranking, the dedupe gate, and the $graphLookup cascade cannot be
faked in-process.

Run against a scratch database (`engram_test`) so they can never clobber demo
state mid-recording:

    uv run pytest -m atlas -q

Skipped automatically when MONGODB_URI is absent or still the placeholder.
"""

from __future__ import annotations

import os

import pytest

from engram import config, embed, store
from engram import trust as T

pytestmark = pytest.mark.atlas

TEST_DB = "engram_test"


def _unavailable() -> str | None:
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        return "MONGODB_URI not set"
    if "cluster.xxxxx.mongodb.net" in uri:
        return "MONGODB_URI is still the .env.example placeholder"
    if embed.provider() == "none":
        return "no embedding provider (set VOYAGE_API_KEY or FIREWORKS_API_KEY)"
    return None


@pytest.fixture(scope="module", autouse=True)
def atlas_db():
    reason = _unavailable()
    if reason:
        pytest.skip(reason, allow_module_level=True)

    original, config.DB_NAME = config.DB_NAME, TEST_DB
    try:
        config.client().admin.command("ping")
    except Exception as exc:
        config.DB_NAME = original
        pytest.skip(f"cannot reach Atlas: {exc}", allow_module_level=True)

    store.wipe()
    result = store.ensure_indexes(wait=True)
    if not result.get("queryable"):
        config.DB_NAME = original
        pytest.skip(f"vector index never became queryable: {result}",
                    allow_module_level=True)
    yield
    store.wipe()
    config.DB_NAME = original


@pytest.fixture(autouse=True)
def clean_population():
    store.wipe()
    yield


def write(text: str, kind: str = "fact", **kw) -> str:
    """Write a memory and block until $vectorSearch can actually see it.

    Atlas Search is eventually consistent, so a bare write_memory() followed by
    an immediate retrieve() races the index and intermittently sees an empty
    store — which is exactly the bug this helper exists to keep out of the
    assertions below.
    """
    mid = store.write_memory(text, kind, **kw)
    store.wait_for_sync([mid])
    return mid


# --------------------------------------------------------------------------
# write gate
# --------------------------------------------------------------------------


def test_a_new_claim_is_born_provisional_at_trust_030():
    mid = write("In sample_mflix, movie release years live in the "
                             "year field as an integer.", "fact")
    doc = store.get(mid)
    assert doc["status"] == T.PROVISIONAL
    assert doc["trust"] == pytest.approx(T.INITIAL_TRUST)
    assert doc["wins"] == 0 and doc["losses"] == 0 and doc["parents"] == []


def test_mids_are_monotonic_and_unique():
    """Claims must be genuinely unrelated — anything closer would (correctly)
    be absorbed by the dedupe gate and never get its own mid."""
    mids = [
        write("Theaters are listed in the theaters collection with "
                           "a geoJSON location field.", "fact"),
        write("User accounts live in the users collection keyed by "
                           "email address.", "fact"),
        write("Unwind the directors array before grouping to count "
                           "movies per director.", "procedure"),
    ]
    assert len(set(mids)) == 3
    assert mids == sorted(mids)


def test_a_near_duplicate_merges_instead_of_inserting():
    """The dedupe gate: fifty paraphrases of one fact would otherwise split its
    trust fifty ways."""
    first = write(
        "In sample_mflix, imdb.rating is a float on a 0 to 10 scale.", "fact")
    second = write(
        "In sample_mflix, imdb.rating is a 0-10 scale float value.", "fact")
    assert second == first
    assert config.memories().count_documents({}) == 1
    assert store.get(first)["uses"] == 1


def test_a_merge_unions_provenance_rather_than_dropping_it():
    parent = write("Genres are stored as an array of strings on "
                                "each movie document.", "fact")
    first = write(
        "In sample_mflix, imdb.rating is a float on a 0 to 10 scale.", "fact")
    merged = write(
        "In sample_mflix, imdb.rating is a 0-10 scale float value.", "fact",
        parents=[parent])
    assert merged == first
    assert parent in store.get(first)["parents"]


def test_an_unrelated_claim_gets_its_own_memory():
    a = write("In sample_mflix, imdb.rating is a 0 to 10 float.", "fact")
    b = write("Theaters are listed in the theaters collection with "
                           "a geoJSON location field.", "fact")
    assert a != b and config.memories().count_documents({}) == 2


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def test_retrieval_finds_the_semantically_relevant_memory():
    write("In sample_mflix, imdb.rating is a float on a 0 to 10 "
                       "scale.", "fact")
    write("Theaters are listed in the theaters collection with a "
                       "geoJSON location field.", "fact")
    hits = store.retrieve("how are movie ratings scaled?", k=2)
    assert hits and "imdb.rating" in hits[0]["text"]


def test_quarantined_memories_are_never_served():
    mid = write("In sample_mflix, imdb.rating is a float on a 0 to "
                             "10 scale.", "fact")
    assert store.retrieve("how are movie ratings scaled?", k=4)
    store.force(mid, status=T.QUARANTINED, trust=0.10)
    assert [d["mid"] for d in store.retrieve("how are movie ratings scaled?", k=4)] == []


def test_trust_outranks_a_closer_but_untrusted_memory():
    """The whole point of `score = vectorScore * (0.4 + 0.6 * trust)`."""
    close = write(
        "In sample_mflix, imdb.rating is a float on a 0 to 10 scale.", "fact")
    earned = write(
        "Movie ratings in this dataset are found under the imdb subdocument.", "fact")

    baseline = [d["mid"] for d in store.retrieve("how are movie ratings scaled?", k=2)]
    assert baseline[0] == close, "expected the closer memory to win on similarity alone"

    store.force(earned, trust=0.95, status=T.TRUSTED)
    store.force(close, trust=T.INITIAL_TRUST)
    reranked = [d["mid"] for d in store.retrieve("how are movie ratings scaled?", k=2)]
    assert reranked[0] == earned, "earned trust should have flipped the ranking"


def test_retrieval_never_ships_embeddings_back():
    write("In sample_mflix, imdb.rating is a 0 to 10 float.", "fact")
    hits = store.retrieve("rating scale", k=1)
    assert hits and "embedding" not in hits[0]


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def test_a_citation_in_a_passing_episode_earns_trust_and_promotes():
    mid = write("In sample_mflix, imdb.rating is a 0 to 10 float.",
                             "fact")
    store.force(mid, trust=0.50, wins=1)
    store.apply_outcome({"retrieved": [mid], "cited": [mid], "outcome": "pass"})
    doc = store.get(mid)
    assert doc["trust"] == pytest.approx(0.625)
    assert doc["wins"] == 2 and doc["status"] == T.TRUSTED


def test_a_citation_in_a_failing_episode_is_slashed():
    mid = write("In sample_mflix, imdb.rating is a 0 to 10 float.",
                             "fact")
    store.force(mid, trust=0.80, status=T.TRUSTED)
    store.apply_outcome({"retrieved": [mid], "cited": [mid], "outcome": "fail"})
    doc = store.get(mid)
    assert doc["trust"] == pytest.approx(0.32) and doc["losses"] == 1


def test_being_retrieved_and_ignored_costs_a_little_trust():
    mid = write("In sample_mflix, imdb.rating is a 0 to 10 float.",
                             "fact")
    store.apply_outcome({"retrieved": [mid], "cited": [], "outcome": "pass"})
    doc = store.get(mid)
    assert doc["trust"] == pytest.approx(0.28)
    assert doc["wins"] == 0 and doc["losses"] == 0


def test_crossing_the_quarantine_line_changes_status():
    mid = write("In sample_mflix, imdb.rating is a 0 to 10 float.",
                             "fact")
    store.force(mid, trust=0.30, status=T.TRUSTED)
    store.apply_outcome({"retrieved": [mid], "cited": [mid], "outcome": "fail"})
    assert store.get(mid)["status"] == T.QUARANTINED


# --------------------------------------------------------------------------
# the graph move
# --------------------------------------------------------------------------


@pytest.fixture
def infected_lineage():
    """lie -> child -> grandchild, plus an unrelated bystander."""
    lie = write("In sample_mflix, imdb.rating is on a 0-100 scale; "
                             "divide by 10 to normalize before comparing.", "fact")
    store.force(lie, trust=0.85, status=T.TRUSTED)
    child = write("Always divide imdb ratings by 10 before applying "
                               "a threshold comparison.", "procedure", parents=[lie])
    # Deliberately distant from `child`: anything closer sits in the classifier
    # band, and a `duplicate` verdict would collapse the two into one node and
    # leave nothing at depth 1 to test.
    grandchild = write("Movie runtimes are recorded in minutes in the runtime "
                       "field.", "fact", parents=[child])
    bystander = write("Theaters are listed in the theaters collection "
                                   "with a geoJSON location field.", "fact")
    return {"lie": lie, "child": child, "grandchild": grandchild,
            "bystander": bystander}


def test_graph_lookup_traces_the_whole_lineage_in_one_pass(infected_lineage):
    result = list(config.memories().aggregate(
        store.contamination_pipeline(infected_lineage["lie"])))
    found = {d["mid"]: d["depth"] for d in result[0]["descendants"]}
    assert found == {infected_lineage["child"]: 0,
                     infected_lineage["grandchild"]: 1}


def test_cascade_halves_direct_children_and_resets_them_to_provisional(infected_lineage):
    store.force(infected_lineage["child"], trust=0.80, status=T.TRUSTED)
    store.cascade_quarantine(infected_lineage["lie"])
    child = store.get(infected_lineage["child"])
    assert child["trust"] == pytest.approx(0.40)
    assert child["status"] == T.PROVISIONAL
    assert child["contaminated_by"] == infected_lineage["lie"]


def test_cascade_reaches_deeper_descendants_more_gently(infected_lineage):
    store.force(infected_lineage["grandchild"], trust=0.80)
    store.cascade_quarantine(infected_lineage["lie"])
    grandchild = store.get(infected_lineage["grandchild"])
    assert grandchild["trust"] == pytest.approx(0.56)  # 0.80 * 0.7
    assert grandchild["contaminated_by"] == infected_lineage["lie"]


def test_cascade_leaves_unrelated_memories_alone(infected_lineage):
    store.cascade_quarantine(infected_lineage["lie"])
    bystander = store.get(infected_lineage["bystander"])
    assert bystander["trust"] == pytest.approx(T.INITIAL_TRUST)
    assert "contaminated_by" not in bystander


def test_quarantine_fires_the_cascade_automatically(infected_lineage):
    """No separate call: apply_outcome notices the transition and traces."""
    lie = infected_lineage["lie"]
    store.force(lie, trust=0.30, status=T.TRUSTED)
    store.apply_outcome({"retrieved": [lie], "cited": [lie], "outcome": "fail"})
    assert store.get(lie)["status"] == T.QUARANTINED
    assert store.get(infected_lineage["child"])["contaminated_by"] == lie


def test_a_provenance_cycle_cannot_hang_the_trace():
    """maxDepth is the guard. A self-referential parents edge is malformed data,
    not a reason for the demo to spin."""
    a = write("Claim A about how ratings are stored in this data.",
                           "fact")
    b = write("Claim B about how runtimes are stored in this data.",
                           "fact", parents=[a])
    config.memories().update_one({"mid": a}, {"$set": {"parents": [b]}})
    assert len(store.cascade_quarantine(a)) <= config.MAX_CASCADE_DEPTH + 1


def test_the_full_poison_arc_end_to_end(infected_lineage):
    """Beat 1: the lie passes a task it cannot corrupt and gains trust, and its
    child is untouched. Beat 2: it is cited in a failure, which falsifies its
    `trusted` standing, quarantines it, and cascades to the child."""
    lie, child = infected_lineage["lie"], infected_lineage["child"]

    store.apply_outcome({"retrieved": [lie], "cited": [lie], "outcome": "pass"})
    assert store.get(lie)["trust"] == pytest.approx(0.8875)
    assert store.get(lie)["status"] == T.TRUSTED
    assert "contaminated_by" not in store.get(child)

    store.apply_outcome({"retrieved": [lie], "cited": [lie], "outcome": "fail"})
    assert store.get(lie)["trust"] == pytest.approx(0.355)
    assert store.get(lie)["status"] == T.QUARANTINED
    assert store.get(child)["contaminated_by"] == lie
    assert store.get(child)["trust"] == pytest.approx(0.15)

    # and the lie is no longer served
    served = [d["mid"] for d in store.retrieve("how should I compare imdb ratings?", k=4)]
    assert lie not in served
