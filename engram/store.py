"""The Engram engine: write gate, trust-weighted retrieval, lifecycle, cascade.

Two aggregation pipelines carry this project — `RETRIEVAL_PIPELINE` and
`CONTAMINATION_PIPELINE`. Both are built by functions below and reproduced
verbatim in the README.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterable

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from . import config, embed as embedding, events, trust as T

Doc = dict[str, Any]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# ids & indexes
# --------------------------------------------------------------------------


def next_mid() -> str:
    """Monotonic, human-citable id. Atomic so parallel writers can't collide."""
    doc = config.db()["counters"].find_one_and_update(
        {"_id": "mid"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return f"M-{doc['seq']:04d}"


def ensure_indexes(wait: bool = True) -> dict[str, Any]:
    """Standard indexes + the Atlas Vector Search index (`mem_vec`)."""
    mem = config.memories()
    mem.create_index([("mid", ASCENDING)], unique=True, name="mid_unique")
    mem.create_index([("status", ASCENDING)], name="status_idx")
    mem.create_index([("parents", ASCENDING)], name="parents_idx")
    config.episodes().create_index([("task_id", ASCENDING)], name="task_idx")
    config.tasks_col().create_index([("task_id", ASCENDING)], unique=True, name="task_unique")

    result: dict[str, Any] = {"vector_index": "skipped", "provider": embedding.provider()}
    if embedding.is_auto():
        result["vector_index"] = "auto-embeddings (managed by Atlas)"
        return result

    dims = embedding.dims()
    if not dims:
        result["vector_index"] = "no embedding provider — vector index not created"
        return result

    existing = {ix["name"] for ix in mem.list_search_indexes()}
    if config.VECTOR_INDEX not in existing:
        mem.create_search_index(
            SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": dims,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "status"},
                    ]
                },
                name=config.VECTOR_INDEX,
                type="vectorSearch",
            )
        )
        result["vector_index"] = f"created ({dims} dims, cosine)"
    else:
        result["vector_index"] = f"exists ({dims} dims)"

    if wait:
        result["queryable"] = wait_for_index()
    return result


def wait_for_index(timeout: float = 300.0) -> bool:
    """Atlas builds search indexes asynchronously; block until queryable."""
    mem = config.memories()
    deadline = time.time() + timeout
    while time.time() < deadline:
        for ix in mem.list_search_indexes(config.VECTOR_INDEX):
            if ix.get("queryable"):
                return True
        time.sleep(3)
    return False


# --------------------------------------------------------------------------
# the two exhibit pipelines
# --------------------------------------------------------------------------


def retrieval_pipeline(
    query_vector: list[float] | None,
    query_text: str,
    k: int = config.RETRIEVE_K,
) -> list[Doc]:
    """Trust-weighted semantic retrieval. ONE pipeline, no post-processing.

    A memory's rank is its semantic similarity scaled by what it has earned:
    `score = vectorScore * (0.4 + 0.6 * trust)`. Quarantined and dead memories
    never surface at all.
    """
    search: Doc = {
        "index": config.VECTOR_INDEX,
        "path": "embedding",
        "numCandidates": config.NUM_CANDIDATES,
        "limit": config.VECTOR_LIMIT,
    }
    if query_vector is None:
        search["query"] = query_text  # MongoDB Automated Embeddings path
    else:
        search["queryVector"] = query_vector

    return [
        {"$vectorSearch": search},
        {"$addFields": {"vectorScore": {"$meta": "vectorSearchScore"}}},
        {"$match": {"status": {"$nin": [T.QUARANTINED, T.DEAD]}}},
        {
            "$addFields": {
                "score": {
                    "$multiply": [
                        "$vectorScore",
                        {"$add": [0.4, {"$multiply": [0.6, "$trust"]}]},
                    ]
                }
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": k},
        {"$project": {"embedding": 0}},
    ]


def contamination_pipeline(mid: str) -> list[Doc]:
    """Trace everything learned under a quarantined memory's influence.

    `parents` is the provenance edge: a memory records the mids that were
    cited in the episode it was extracted from. Walking `mid -> parents`
    with $graphLookup recovers the entire infected subtree in one round trip.
    """
    return [
        {"$match": {"mid": mid}},
        {
            "$graphLookup": {
                "from": config.MEMORIES,
                "startWith": "$mid",
                "connectFromField": "mid",
                "connectToField": "parents",
                "as": "descendants",
                "depthField": "depth",
            }
        },
        {"$project": {"mid": 1, "text": 1, "trust": 1, "status": 1, "descendants.mid": 1,
                      "descendants.text": 1, "descendants.trust": 1,
                      "descendants.status": 1, "descendants.depth": 1,
                      "descendants.parents": 1}},
    ]


# --------------------------------------------------------------------------
# write path
# --------------------------------------------------------------------------


def _nearest(vector: list[float] | None, text: str, k: int = 3) -> list[Doc]:
    """k nearest existing memories, for the dedupe / contradiction gate."""
    if config.memories().estimated_document_count() == 0:
        return []
    search: Doc = {
        "index": config.VECTOR_INDEX,
        "path": "embedding",
        "numCandidates": 50,
        "limit": k,
    }
    if vector is None:
        search["query"] = text
    else:
        search["queryVector"] = vector
    pipeline = [
        {"$vectorSearch": search},
        {"$addFields": {"similarity": {"$meta": "vectorSearchScore"}}},
        {"$match": {"status": {"$ne": T.DEAD}}},
        {"$project": {"embedding": 0}},
    ]
    try:
        return list(config.memories().aggregate(pipeline))
    except OperationFailure:
        return []  # index still building — fail open, insert rather than block


def _merge(existing: Doc, parents: list[str], source_episode: Any) -> str:
    """Dedupe gate: fold the new claim into the memory that already says it."""
    updated = config.memories().find_one_and_update(
        {"mid": existing["mid"]},
        {
            "$inc": {"uses": 1},
            "$addToSet": {"parents": {"$each": list(parents or [])}},
            "$set": {"last_used_at": _now()},
        },
        return_document=ReturnDocument.AFTER,
    )
    events.emit(
        events.MEMORY_MERGED,
        mid=existing["mid"],
        text=existing.get("text", ""),
        similarity=round(float(existing.get("similarity", 1.0)), 4),
        parents_added=list(parents or []),
        source_episode=str(source_episode) if source_episode else None,
    )
    return (updated or existing)["mid"]


def _insert(
    text: str,
    kind: str,
    source_episode: Any,
    parents: list[str],
    contradicts: list[str] | None = None,
    vector: list[float] | None = None,
) -> str:
    mid = next_mid()
    doc: Doc = {
        "mid": mid,
        "text": text.strip(),
        "kind": kind,
        "status": T.PROVISIONAL,
        "trust": T.INITIAL_TRUST,
        "uses": 0,
        "wins": 0,
        "losses": 0,
        "created_at": _now(),
        "last_used_at": None,
        "source_episode": source_episode,
        "parents": list(parents or []),
    }
    if contradicts:
        doc["contradicts"] = list(contradicts)
    if vector is not None:
        doc["embedding"] = vector
    config.memories().insert_one(doc)

    if contradicts:
        config.memories().update_many(
            {"mid": {"$in": list(contradicts)}},
            {"$addToSet": {"contradicts": mid}},
        )
        events.emit(
            events.CONTRADICTION_DETECTED, mid=mid, text=text, against=list(contradicts)
        )

    events.emit(
        events.MEMORY_WRITTEN,
        mid=mid,
        text=text.strip(),
        kind=kind,
        status=T.PROVISIONAL,
        trust=T.INITIAL_TRUST,
        parents=list(parents or []),
        source_episode=str(source_episode) if source_episode else None,
    )
    return mid


def write_memory(
    text: str,
    kind: str = "fact",
    source_episode: Any = None,
    parents: Iterable[str] | None = None,
) -> str | None:
    """Write gate. Returns the mid the claim now lives under (new or merged).

    1. Embed, then $vectorSearch the existing population (k=3).
    2. similarity >= 0.92  -> merge into the existing memory, no insert.
    3. 0.75 <= sim < 0.92  -> one small-model call: duplicate | compatible |
       contradicts. Fails open to a plain insert.
    4. otherwise           -> insert as provisional, trust 0.30.
    """
    text = (text or "").strip()
    if not text:
        return None
    parents = list(parents or [])

    vector = embedding.embed(text, input_type="document")
    neighbours = _nearest(vector, text, k=3)
    top = neighbours[0] if neighbours else None
    similarity = float(top.get("similarity", 0.0)) if top else 0.0

    if top and similarity >= config.MERGE_SIMILARITY:
        return _merge(top, parents, source_episode)

    if top and config.CONTRADICTION_LOW <= similarity < config.MERGE_SIMILARITY:
        verdict = _classify_pair(top.get("text", ""), text)
        if verdict == "duplicate":
            return _merge(top, parents, source_episode)
        if verdict == "contradicts":
            return _insert(
                text, kind, source_episode, parents,
                contradicts=[top["mid"]], vector=vector,
            )

    return _insert(text, kind, source_episode, parents, vector=vector)


def _classify_pair(existing: str, incoming: str) -> str:
    """Fireworks metabolism tier: the small model governs the write path.

    Never blocks a write — any error or unparseable answer means 'compatible'.
    """
    from .llm import classify_relation

    try:
        return classify_relation(existing, incoming)
    except Exception as exc:  # fail open, always
        events.emit(events.NOTE, note=f"contradiction gate failed open: {exc}")
        return "compatible"


# --------------------------------------------------------------------------
# read path
# --------------------------------------------------------------------------


def retrieve(query: str, k: int = config.RETRIEVE_K) -> list[Doc]:
    """Trust-weighted retrieval, then contradiction arbitration."""
    vector = embedding.embed(query, input_type="query")
    try:
        docs = list(config.memories().aggregate(retrieval_pipeline(vector, query, k)))
    except OperationFailure as exc:
        events.emit(events.NOTE, note=f"retrieval unavailable: {exc}")
        return []

    docs = _suppress_contradictions(docs)
    if docs:
        events.emit(
            events.MEMORY_RETRIEVED,
            query=query[:160],
            mids=[d["mid"] for d in docs],
            scores=[round(float(d.get("score", 0.0)), 4) for d in docs],
        )
        config.memories().update_many(
            {"mid": {"$in": [d["mid"] for d in docs]}},
            {"$set": {"last_used_at": _now()}},
        )
    return docs


def _suppress_contradictions(docs: list[Doc]) -> list[Doc]:
    """If two retrieved memories contradict each other, only the higher-trust
    one reaches the model. The agent never sees both sides of a fight."""
    by_mid = {d["mid"]: d for d in docs}
    dropped: set[str] = set()
    for doc in docs:
        for other_mid in doc.get("contradicts", []) or []:
            other = by_mid.get(other_mid)
            if not other or doc["mid"] in dropped or other_mid in dropped:
                continue
            loser = other if doc.get("trust", 0) >= other.get("trust", 0) else doc
            winner = doc if loser is other else other
            dropped.add(loser["mid"])
            events.emit(
                events.CONTRADICTION_SUPPRESSED,
                kept=winner["mid"],
                suppressed=loser["mid"],
                kept_trust=round(float(winner.get("trust", 0)), 3),
                suppressed_trust=round(float(loser.get("trust", 0)), 3),
            )
    return [d for d in docs if d["mid"] not in dropped]


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


def apply_outcome(episode: Doc) -> list[T.TrustUpdate]:
    """Deterministic trust settlement for one episode. Atomic ops only."""
    retrieved = list(episode.get("retrieved") or [])
    cited = list(episode.get("cited") or [])
    passed = episode.get("outcome") == "pass"
    touched = list(dict.fromkeys(retrieved + cited))
    if not touched:
        return []

    updates: list[T.TrustUpdate] = []
    newly_quarantined: list[str] = []

    for doc in config.memories().find({"mid": {"$in": touched}}):
        update = T.compute_update(
            mid=doc["mid"],
            status=doc.get("status", T.PROVISIONAL),
            trust=float(doc.get("trust", T.INITIAL_TRUST)),
            wins=int(doc.get("wins", 0)),
            losses=int(doc.get("losses", 0)),
            cited=doc["mid"] in cited,
            passed=passed,
        )
        config.memories().find_one_and_update(
            {"mid": update.mid},
            {
                "$set": {
                    "trust": update.trust_after,
                    "status": update.status_after,
                    "last_used_at": _now(),
                },
                "$inc": {
                    "uses": 1,
                    "wins": update.win_delta,
                    "losses": update.loss_delta,
                },
            },
        )
        events.emit(
            events.TRUST_UPDATED,
            mid=update.mid,
            reason=update.reason,
            trust_before=round(update.trust_before, 3),
            trust_after=round(update.trust_after, 3),
            text=doc.get("text", "")[:120],
        )
        if update.status_changed:
            events.emit(
                events.STATUS_CHANGED,
                mid=update.mid,
                before=update.status_before,
                after=update.status_after,
                trust=round(update.trust_after, 3),
                text=doc.get("text", "")[:120],
            )
            if update.status_after == T.QUARANTINED:
                newly_quarantined.append(update.mid)
        updates.append(update)

    for mid in newly_quarantined:
        cascade_quarantine(mid)

    return updates


def cascade_quarantine(mid: str) -> list[Doc]:
    """The graph move: one $graphLookup, then downgrade the infected subtree.

    A quarantined memory doesn't just stop being served — everything learned
    under its influence loses trust and is knocked back to `provisional` so it
    has to re-earn its place.
    """
    result = list(config.memories().aggregate(contamination_pipeline(mid)))
    if not result:
        return []

    root = result[0]
    descendants = [d for d in root.get("descendants", []) if d.get("status") != T.DEAD]
    descendants.sort(key=lambda d: (int(d.get("depth", 0)), d.get("mid", "")))

    infected: list[Doc] = []
    for desc in descendants:
        depth = int(desc.get("depth", 0))
        before = float(desc.get("trust", T.INITIAL_TRUST))
        after = T.clamp(before * T.cascade_multiplier(depth))
        status_before = desc.get("status", T.PROVISIONAL)
        status_after = T.QUARANTINED if after < T.QUARANTINE_TRUST else T.PROVISIONAL
        config.memories().update_one(
            {"mid": desc["mid"]},
            {"$set": {
                "trust": after,
                "status": status_after,
                "contaminated_by": mid,
            }},
        )
        infected.append({
            "mid": desc["mid"],
            "text": desc.get("text", ""),
            "depth": depth,
            "parents": desc.get("parents", []),
            "trust_before": round(before, 3),
            "trust_after": round(after, 3),
            "status_before": status_before,
            "status_after": status_after,
        })
        if status_before != status_after:
            events.emit(
                events.STATUS_CHANGED,
                mid=desc["mid"],
                before=status_before,
                after=status_after,
                trust=round(after, 3),
                text=desc.get("text", "")[:120],
                cause="contamination",
            )

    events.emit(
        events.CONTAMINATION_TRACED,
        root=mid,
        root_text=root.get("text", ""),
        count=len(infected),
        subtree=infected,
    )
    return infected


def decay_tick() -> int:
    """Ambient forgetting: everything untouched this run loses a little trust.

    Stands in for a TTL/cron story — memory that stops proving itself fades.
    """
    result = config.memories().update_many(
        {"status": {"$nin": [T.DEAD]}},
        [{"$set": {"trust": {"$max": [T.TRUST_FLOOR,
                                      {"$multiply": ["$trust", T.DECAY_FACTOR]}]}}}],
    )
    events.emit(events.DECAY_TICK, modified=result.modified_count)
    return result.modified_count


# --------------------------------------------------------------------------
# helpers for the demo / TUI
# --------------------------------------------------------------------------


def all_memories() -> list[Doc]:
    order = {T.TRUSTED: 0, T.PROVISIONAL: 1, T.QUARANTINED: 2, T.DEAD: 3}
    docs = list(config.memories().find({}, {"embedding": 0}))
    docs.sort(key=lambda d: (order.get(d.get("status"), 9), -float(d.get("trust", 0))))
    return docs


def get(mid: str) -> Doc | None:
    return config.memories().find_one({"mid": mid}, {"embedding": 0})


def force(mid: str, **fields: Any) -> None:
    """Set fields directly. Used only by the demo to plant the poisoned memory."""
    config.memories().update_one({"mid": mid}, {"$set": fields})


def wipe() -> None:
    config.memories().delete_many({})
    config.episodes().delete_many({})
    config.db()["counters"].delete_many({})
