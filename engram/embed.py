"""One function behind three possible providers.

Order of preference:
  1. MongoDB Automated Embeddings — Atlas embeds the `text` field itself. In
     that mode `embed()` returns None and the retrieval pipeline passes a raw
     `query` string to $vectorSearch instead of a `queryVector`.
  2. Voyage AI (VOYAGE_API_KEY)
  3. Fireworks embeddings (FIREWORKS_API_KEY) — already a required key.

Swapping providers is one env var; the rest of the engine never sees a vector
provider name.
"""

from __future__ import annotations

import os
from typing import Literal

import requests

from . import config

Provider = Literal["auto", "voyage", "fireworks", "none"]

_provider: Provider | None = None
_dims: int | None = None


def _detect() -> tuple[Provider, int]:
    forced = os.environ.get("ENGRAM_EMBED_PROVIDER", "").strip().lower()
    if forced == "auto":
        return "auto", 0
    if forced == "voyage" or (not forced and config.has("VOYAGE_API_KEY")):
        if config.has("VOYAGE_API_KEY"):
            return "voyage", config.VOYAGE_DIMS
    if forced == "fireworks" or config.has("FIREWORKS_API_KEY"):
        if config.has("FIREWORKS_API_KEY"):
            return "fireworks", config.FIREWORKS_EMBED_DIMS
    return "none", 0


def provider() -> Provider:
    global _provider, _dims
    if _provider is None:
        _provider, _dims = _detect()
    return _provider


def dims() -> int:
    if _dims is None:
        provider()
    return _dims or 0


def is_auto() -> bool:
    """True when Atlas owns embedding; retrieval then queries by text."""
    return provider() == "auto"


def _voyage(texts: list[str], input_type: str) -> list[list[float]]:
    resp = requests.post(
        config.VOYAGE_URL,
        headers={
            "Authorization": f"Bearer {config.require('VOYAGE_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "input": texts,
            "model": config.VOYAGE_MODEL,
            "input_type": input_type,
            "output_dimension": config.VOYAGE_DIMS,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def _fireworks(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{config.FIREWORKS_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {config.require('FIREWORKS_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={"input": texts, "model": config.FIREWORKS_EMBED_MODEL},
        timeout=30,
    )
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def embed(text: str, *, input_type: str = "document") -> list[float] | None:
    """Return a vector, or None when Atlas is doing the embedding for us."""
    vectors = embed_many([text], input_type=input_type)
    return vectors[0] if vectors else None


def embed_many(
    texts: list[str], *, input_type: str = "document"
) -> list[list[float]] | None:
    if not texts:
        return []
    prov = provider()
    if prov == "auto":
        return None
    if prov == "voyage":
        return _voyage(texts, input_type)
    if prov == "fireworks":
        return _fireworks(texts)
    raise RuntimeError(
        "No embedding provider available. Set VOYAGE_API_KEY or FIREWORKS_API_KEY, "
        "or ENGRAM_EMBED_PROVIDER=auto to use MongoDB Automated Embeddings."
    )


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
