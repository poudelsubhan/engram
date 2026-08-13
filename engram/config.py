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
OPENROUTER_MODEL = os.environ.get("ENGRAM_MAIN_MODEL", "anthropic/claude-sonnet-4.5")

# Metabolism tier — a small fast model on the memory write path.
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = os.environ.get(
    "ENGRAM_SMALL_MODEL", "accounts/fireworks/models/llama4-scout-instruct-basic"
)

# Embeddings.
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = os.environ.get("ENGRAM_VOYAGE_MODEL", "voyage-3.5-lite")
VOYAGE_DIMS = int(os.environ.get("ENGRAM_VOYAGE_DIMS", "1024"))
FIREWORKS_EMBED_MODEL = os.environ.get(
    "ENGRAM_FIREWORKS_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"
)
FIREWORKS_EMBED_DIMS = 768

# Retrieval / write-gate thresholds (spec §Phase 1).
MERGE_SIMILARITY = 0.92
CONTRADICTION_LOW = 0.75
RETRIEVE_K = 4
NUM_CANDIDATES = 50
VECTOR_LIMIT = 12


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
