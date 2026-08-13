"""The write-gate bands are measured, not assumed, and two facts make the
obvious numbers wrong. These tests state both so nobody "simplifies" the
constants back into breaking the poison demo.

Reproduce with: uv run engram calibrate
"""

import pytest

from engram import config

# Measured with nomic-embed-text-v1.5 @ 512 dims against the true claim
# "In sample_mflix, imdb.rating is a float on a 0 to 10 scale."
MEASURED = {
    "paraphrase": (0.9184, 0.9542),
    "contradiction": (0.9445, 0.9683),
    "same_topic": (0.7294, 0.8498),
    "unrelated": (0.7540, 0.7651),
}


def raw_cosine(atlas_score: float) -> float:
    """Invert Atlas's normalization: $vectorSearch returns (1 + cos) / 2."""
    return 2 * atlas_score - 1


def band(score: float) -> str:
    if score >= config.MERGE_SIMILARITY:
        return "merge"
    if score >= config.CONTRADICTION_LOW:
        return "classify"
    return "insert"


def test_atlas_scores_are_normalized_not_raw_cosine():
    """The spec's 0.75 gate lower bound is raw cosine 0.50 — 'vaguely the same
    topic'. In normalized space that is far too loose to mean anything."""
    assert raw_cosine(0.75) == pytest.approx(0.50)
    assert raw_cosine(0.92) == pytest.approx(0.84)


def test_a_contradiction_is_not_more_distant_than_a_paraphrase():
    """The finding that sets the merge gate: '0-100, divide by 10' is lexically
    almost identical to '0 to 10 scale', so it scores HIGHER than an honest
    restatement. Embedding distance cannot separate agreement from denial."""
    assert max(MEASURED["contradiction"]) > max(MEASURED["paraphrase"])


def test_the_merge_gate_sits_above_the_entire_contradiction_band():
    """If it didn't, the lie would be silently absorbed into the memory it
    contradicts and would never exist as its own claim — no poison demo."""
    assert config.MERGE_SIMILARITY > max(MEASURED["contradiction"])
    for score in MEASURED["contradiction"]:
        assert band(score) == "classify"


def test_paraphrases_reach_the_classifier_and_not_a_plain_insert():
    """They must be caught as duplicates rather than accumulating as separate
    memories that split one fact's trust between them."""
    for score in MEASURED["paraphrase"]:
        assert band(score) == "classify"


def test_merely_related_claims_never_reach_the_classifier():
    """Otherwise every write would pay for an LLM call and risk a spurious
    `contradicts` edge between two facts that simply share a subject."""
    for kind in ("same_topic", "unrelated"):
        for score in MEASURED[kind]:
            assert band(score) == "insert"


def test_the_band_is_ordered_and_non_empty():
    assert 0.5 < config.CONTRADICTION_LOW < config.MERGE_SIMILARITY <= 1.0


def test_both_embedding_providers_share_one_dimension_count():
    """The vector index is built once. If Voyage and Fireworks disagreed on
    dims, swapping providers would silently break every query."""
    assert config.EMBED_DIMS == 512


def test_retrieval_widths_leave_room_for_trust_reranking_to_matter():
    """The index returns 12 candidates and trust re-ranks them down to 4 — if
    limit == k the trust weighting could never change the outcome."""
    assert config.VECTOR_LIMIT > config.RETRIEVE_K
    assert config.NUM_CANDIDATES >= config.VECTOR_LIMIT
