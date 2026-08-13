"""The demo task suite over `sample_mflix`.

Expected answers are *computed* from the cluster by `seed_tasks()`, never hand
written — a hand-written expectation that's subtly wrong would poison the trust
signal that the whole system runs on.

Three of these tasks depend on the imdb.rating SCALE (they threshold or average
it) and one depends only on rating ORDER. That split is what makes the poison
act work: the divide-by-10 lie is invisible to the ranking task and fatal to the
threshold tasks.
"""

from __future__ import annotations

from typing import Any, Callable

from . import config

# task_id -> (prompt, checker, ground-truth pipeline, formatter)
TaskSpec = dict[str, Any]


def _count_1995_above_8() -> str:
    n = config.sample_db()["movies"].count_documents(
        {"year": 1995, "imdb.rating": {"$gt": 8}}
    )
    return str(n)


def _count_drama_above_85() -> str:
    n = config.sample_db()["movies"].count_documents(
        {"genres": "Drama", "imdb.rating": {"$gte": 8.5}}
    )
    return str(n)


def _avg_rating_1995() -> str:
    rows = list(config.sample_db()["movies"].aggregate([
        {"$match": {"year": 1995, "imdb.rating": {"$type": "number"}}},
        {"$group": {"_id": None, "avg": {"$avg": "$imdb.rating"}}},
    ]))
    return f"{rows[0]['avg']:.2f}" if rows else "0"


def _top3_1999_by_rating() -> str:
    rows = list(config.sample_db()["movies"].aggregate([
        {"$match": {"year": 1999, "imdb.rating": {"$type": "number"},
                    "imdb.votes": {"$gte": 1000}}},
        {"$sort": {"imdb.rating": -1, "title": 1}},
        {"$limit": 3},
        {"$project": {"title": 1}},
    ]))
    return "|".join(r["title"] for r in rows)


def _top3_directors_1990s() -> str:
    rows = list(config.sample_db()["movies"].aggregate([
        {"$match": {"year": {"$gte": 1990, "$lte": 1999}, "directors": {"$exists": True}}},
        {"$unwind": "$directors"},
        {"$group": {"_id": "$directors", "n": {"$sum": 1}}},
        {"$sort": {"n": -1, "_id": 1}},
        {"$limit": 3},
    ]))
    return "|".join(r["_id"] for r in rows)


def _longest_runtime_1990s() -> str:
    rows = list(config.sample_db()["movies"].aggregate([
        {"$match": {"year": {"$gte": 1990, "$lte": 1999},
                    "runtime": {"$type": "number"}}},
        {"$sort": {"runtime": -1, "title": 1}},
        {"$limit": 1},
        {"$project": {"title": 1}},
    ]))
    return rows[0]["title"] if rows else ""


TASKS: list[TaskSpec] = [
    {
        "task_id": "rating_threshold_1995",
        "prompt": (
            "In the sample_mflix movies collection, how many movies released in "
            "1995 have an imdb rating greater than 8? Answer with just the number."
        ),
        "checker": "numeric_tol",
        "rating_scale_sensitive": True,
        "compute": _count_1995_above_8,
    },
    {
        "task_id": "top3_1999_ranking",
        "prompt": (
            "In the sample_mflix movies collection, what are the top 3 movies "
            "released in 1999 by imdb rating, counting only movies with at least "
            "1000 imdb votes? List the three titles."
        ),
        "checker": "contains",
        "rating_scale_sensitive": False,
        "compute": _top3_1999_by_rating,
    },
    {
        "task_id": "drama_threshold",
        "prompt": (
            "In the sample_mflix movies collection, how many movies have Drama "
            "among their genres and an imdb rating of at least 8.5? Answer with "
            "just the number."
        ),
        "checker": "numeric_tol",
        "rating_scale_sensitive": True,
        "compute": _count_drama_above_85,
    },
    {
        "task_id": "avg_rating_1995",
        "prompt": (
            "In the sample_mflix movies collection, what is the average imdb "
            "rating of movies released in 1995? Answer to two decimal places."
        ),
        "checker": "numeric_tol",
        "rating_scale_sensitive": True,
        "compute": _avg_rating_1995,
    },
    {
        "task_id": "top3_directors_1990s",
        "prompt": (
            "In the sample_mflix movies collection, which 3 directors have the "
            "most movies released between 1990 and 1999 inclusive? List the "
            "three names."
        ),
        "checker": "contains",
        "rating_scale_sensitive": False,
        "compute": _top3_directors_1990s,
    },
    {
        "task_id": "longest_1990s",
        "prompt": (
            "In the sample_mflix movies collection, which movie released between "
            "1990 and 1999 has the longest runtime? Give the title."
        ),
        "checker": "contains",
        "rating_scale_sensitive": False,
        "compute": _longest_runtime_1990s,
    },
]

# The infection narrative depends on this exact pair (spec, cut-list item 4).
RANKING_TASK = "top3_1999_ranking"
THRESHOLD_TASKS = ["rating_threshold_1995", "drama_threshold", "avg_rating_1995"]


def seed_tasks(force: bool = False) -> list[dict[str, Any]]:
    """Compute every expected answer from the cluster and upsert the suite."""
    col = config.tasks_col()
    out = []
    for spec in TASKS:
        existing = col.find_one({"task_id": spec["task_id"]})
        if existing and not force:
            out.append(existing)
            continue
        expected = spec["compute"]()
        doc = {
            "task_id": spec["task_id"],
            "prompt": spec["prompt"],
            "expected_answer": expected,
            "checker": spec["checker"],
            "rating_scale_sensitive": spec["rating_scale_sensitive"],
        }
        col.update_one({"task_id": doc["task_id"]}, {"$set": doc}, upsert=True)
        out.append(doc)
    return out


def load_tasks(task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Suite in declaration order, optionally filtered."""
    stored = {d["task_id"]: d for d in config.tasks_col().find({})}
    order = task_ids or [t["task_id"] for t in TASKS]
    return [stored[tid] for tid in order if tid in stored]


def get_task(task_id: str) -> dict[str, Any] | None:
    return config.tasks_col().find_one({"task_id": task_id})
