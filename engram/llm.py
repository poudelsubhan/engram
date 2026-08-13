"""Two tiers of model.

The big model *thinks* (OpenRouter -> Claude Sonnet): it reads the task, calls
the Mongo tool, writes an answer, cites its memories.

The small model *governs memory* (Fireworks): it runs on every write — pulling
reusable claims out of a finished episode, and judging whether a near-duplicate
claim agrees or disagrees with what's already stored. Fast, cheap, and on the
hot path where a big model would be waste.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from . import config

VALID_RELATIONS = {"duplicate", "compatible", "contradicts"}


@lru_cache(maxsize=1)
def main_client() -> OpenAI:
    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.require("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/poudelsubhan/engram",
            "X-Title": "Engram",
        },
        timeout=120.0,
    )


@lru_cache(maxsize=1)
def small_client() -> OpenAI:
    return OpenAI(
        base_url=config.FIREWORKS_BASE_URL,
        api_key=config.require("FIREWORKS_API_KEY"),
        timeout=60.0,
    )


def main_chat(messages: list[dict[str, Any]], tools: list[dict] | None = None):
    kwargs: dict[str, Any] = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1500,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return main_client().chat.completions.create(**kwargs)


# --------------------------------------------------------------------------
# metabolism tier
# --------------------------------------------------------------------------

_CLASSIFY_PROMPT = (
    "Given claim A and claim B, answer exactly one word: "
    "duplicate | compatible | contradicts."
)


def classify_relation(claim_a: str, claim_b: str) -> str:
    """One word out. Anything unparseable is treated as 'compatible' upstream."""
    resp = small_client().chat.completions.create(
        model=config.FIREWORKS_MODEL,
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": f"Claim A: {claim_a}\nClaim B: {claim_b}"},
        ],
        temperature=0,
        max_tokens=6,
    )
    word = (resp.choices[0].message.content or "").strip().lower()
    for candidate in VALID_RELATIONS:
        if candidate in word:
            return candidate
    return "compatible"


_EXTRACT_PROMPT = (
    "You extract durable memories from a finished agent episode.\n"
    "Output ONLY a JSON array, 0 to 3 items, each {\"text\": str, \"kind\": "
    "\"fact\"|\"procedure\"}.\n"
    "Rules:\n"
    "- Only durable, reusable knowledge about the DATA or the METHOD: schema "
    "shapes, field locations, units and scales, query techniques.\n"
    "- Never record episode trivia: no specific answers, counts, titles, or "
    "results from this one task.\n"
    "- One or two sentences each. If nothing is reusable, output [].\n"
    "Output the JSON array and nothing else."
)


def extract_memories(episode_summary: str) -> list[dict[str, str]]:
    """Post-episode extractor. Constrained hard because it runs unattended."""
    resp = small_client().chat.completions.create(
        model=config.FIREWORKS_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": episode_summary[:6000]},
        ],
        temperature=0,
        max_tokens=400,
    )
    return parse_memory_json(resp.choices[0].message.content or "")


def parse_memory_json(raw: str) -> list[dict[str, str]]:
    """Tolerant parse of the extractor's output — pure, so it's unit-tested."""
    raw = (raw or "").strip()
    if not raw:
        return []
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    out: list[dict[str, str]] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if len(text) < 12:
            continue
        kind = str(item.get("kind", "fact")).strip().lower()
        out.append({"text": text, "kind": kind if kind in {"fact", "procedure"} else "fact"})
    return out
