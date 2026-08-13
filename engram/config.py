"""Environment, Mongo handles, and model selection — one place, no surprises."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

DB_NAME = os.environ.get("ENGRAM_DB", "engram")
SAMPLE_DB = os.environ.get("ENGRAM_SAMPLE_DB", "sample_mflix")

MEMORIES = "memories"
EPISODES = "episodes"
TASKS = "tasks"
VECTOR_INDEX = "mem_vec"

# Model gateway (plumbing — the main agent's brain).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("ENGRAM_MAIN_MODEL", "anthropic/claude-sonnet-5")

# Metabolism tier — a small fast model on the memory write path.
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = os.environ.get(
    "ENGRAM_SMALL_MODEL", "accounts/fireworks/models/deepseek-v4-flash"
)

# Embeddings. Both providers are pinned to the SAME dimension count so the
# vector index survives a provider swap untouched.
EMBED_DIMS = int(os.environ.get("ENGRAM_EMBED_DIMS", "512"))
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = os.environ.get("ENGRAM_VOYAGE_MODEL", "voyage-4-lite")
FIREWORKS_EMBED_MODEL = os.environ.get(
    "ENGRAM_FIREWORKS_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"
)

# Write-gate thresholds — MEASURED, not assumed. Run `engram calibrate` to
# reproduce these numbers against whichever embedding provider is configured.
#
# Two things make the obvious values wrong:
#
# 1. Atlas does not return raw cosine. `$vectorSearch` normalizes it as
#    `score = (1 + cosine) / 2`, so every number here lives in [0.5, 1.0].
#
# 2. Embedding distance cannot tell a paraphrase from a contradiction. Measured
#    against the true claim "imdb.rating is a float on a 0 to 10 scale":
#
#        paraphrase      0.918 – 0.954
#        CONTRADICTION   0.945 – 0.968   <- scores HIGHER than the paraphrase
#        same topic      0.729 – 0.850
#        unrelated       0.754 – 0.765
#
#    "0-100, divide by 10" is lexically almost identical to "0 to 10 scale".
#    A merge gate anywhere in the 0.92 range would silently absorb the lie into
#    the memory it contradicts, and it would never exist as its own claim.
#
# So the merge gate sits above the entire contradiction band, and the band
# below it routes to the classifier — which is exactly why a model, not a
# distance, governs this decision.
MERGE_SIMILARITY = float(os.environ.get("ENGRAM_MERGE_SIM", "0.985"))
CONTRADICTION_LOW = float(os.environ.get("ENGRAM_GATE_LOW", "0.88"))

RETRIEVE_K = 4
NUM_CANDIDATES = 50
VECTOR_LIMIT = 12
MAX_CASCADE_DEPTH = 5  # cycle guard on $graphLookup


def mongodb_uri() -> str:
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Put your Atlas Hackathon Sandbox SRV URI in .env"
        )
    return uri


@lru_cache(maxsize=1)
def client() -> MongoClient:
    return MongoClient(mongodb_uri(), appname="engram", serverSelectionTimeoutMS=20000)


def db() -> Database:
    return client()[DB_NAME]


def sample_db() -> Database:
    return client()[SAMPLE_DB]


def memories():
    return db()[MEMORIES]


def episodes():
    return db()[EPISODES]


def tasks_col():
    return db()[TASKS]


def has(key: str) -> bool:
    return bool(os.environ.get(key, "").strip())


def require(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise RuntimeError(f"{key} is not set — add it to .env")
    return value
