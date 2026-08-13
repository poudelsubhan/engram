"""The agent's one tool: read-only queries against `sample_mflix`.

Everything except find/aggregate is rejected, and aggregations are scanned for
write or execution stages before they reach the driver. The agent is allowed to
be wrong about the data; it is not allowed to touch it.
"""

from __future__ import annotations

import json
from typing import Any

from bson import json_util

from . import config

ALLOWED_OPS = {"find", "aggregate"}
ALLOWED_COLLECTIONS = {"movies", "comments", "users", "theaters", "sessions", "embedded_movies"}
FORBIDDEN_STAGES = {
    "$out", "$merge", "$function", "$accumulator", "$where",
    "$lookup", "$unionWith", "$graphLookup",  # keep the agent off other collections
}
MAX_DOCS = 25
MAX_TIME_MS = 15000

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_mongo_query",
        "description": (
            "Run a READ-ONLY MongoDB query against the sample_mflix database. "
            "Use op='aggregate' with a pipeline for counts, groupings and rankings; "
            "op='find' with a filter for simple lookups. Returns at most 25 documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name, e.g. 'movies'.",
                    "default": "movies",
                },
                "op": {"type": "string", "enum": ["find", "aggregate"]},
                "pipeline": {
                    "type": "array",
                    "description": "Aggregation pipeline (required when op='aggregate').",
                    "items": {"type": "object"},
                },
                "filter": {
                    "type": "object",
                    "description": "Query filter (used when op='find').",
                },
                "projection": {"type": "object"},
                "sort": {"type": "object"},
                "limit": {"type": "integer"},
            },
            "required": ["op"],
        },
    },
}


class QueryRejected(ValueError):
    """Raised when a request isn't a plain read."""


def _scan_stages(pipeline: list[dict[str, Any]]) -> None:
    for stage in pipeline:
        if not isinstance(stage, dict):
            raise QueryRejected("Each pipeline stage must be an object.")
        for key in stage:
            if key in FORBIDDEN_STAGES:
                raise QueryRejected(f"Stage {key} is not allowed — read-only queries only.")


def run_mongo_query(spec: dict[str, Any] | str) -> dict[str, Any]:
    """Execute a validated read. Returns {ok, count, docs} or {ok: False, error}."""
    try:
        if isinstance(spec, str):
            spec = json.loads(spec)
        if not isinstance(spec, dict):
            raise QueryRejected("Query must be a JSON object.")

        op = str(spec.get("op", "aggregate")).lower()
        if op not in ALLOWED_OPS:
            raise QueryRejected(f"op must be one of {sorted(ALLOWED_OPS)}, got {op!r}.")

        name = str(spec.get("collection", "movies"))
        if name not in ALLOWED_COLLECTIONS:
            raise QueryRejected(f"Unknown collection {name!r}.")

        limit = min(int(spec.get("limit") or MAX_DOCS), MAX_DOCS)

        # Validate the whole request before opening a connection — a rejected
        # query must never reach the driver, let alone the cluster.
        pipeline: list[dict[str, Any]] = []
        if op == "aggregate":
            stages = spec.get("pipeline") or []
            if not isinstance(stages, list):
                raise QueryRejected("pipeline must be an array of stages.")
            _scan_stages(stages)
            pipeline = list(stages) + [{"$limit": limit}]

        col = config.sample_db()[name]
        if op == "aggregate":
            cursor = col.aggregate(pipeline, maxTimeMS=MAX_TIME_MS)
        else:
            cursor = col.find(
                spec.get("filter") or {},
                spec.get("projection") or None,
                limit=limit,
                sort=list((spec.get("sort") or {}).items()) or None,
                max_time_ms=MAX_TIME_MS,
            )

        docs = json.loads(json_util.dumps(list(cursor)))
        return {"ok": True, "count": len(docs), "docs": docs}

    except QueryRejected as exc:
        return {"ok": False, "error": f"rejected: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
