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
            "X-OpenRouter-Title": "Engram",
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


_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": sorted(VALID_RELATIONS)}},
    "required": ["verdict"],
}


def classify_relation(claim_a: str, claim_b: str) -> str:
    """One word out, constrained by a JSON schema at the API layer.

    Anything unparseable is treated as 'compatible' — the write path never
    blocks on this call.
    """
    resp = small_client().chat.completions.create(
        model=config.FIREWORKS_MODEL,
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": f"Claim A: {claim_a}\nClaim B: {claim_b}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Relation", "schema": _VERDICT_SCHEMA},
        },
        temperature=0,
        # Generous: the recommended small models reason before emitting JSON,
        # and a budget that cuts them off mid-thought yields prose, not a verdict.
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    choice = resp.choices[0]
    return parse_relation(choice.message.content or "",
                          truncated=choice.finish_reason == "length")


CLASSIFY_MAX_TOKENS = 400
EXTRACT_MAX_TOKENS = 1200


def parse_relation(raw: str, truncated: bool = False) -> str:
    """Read a verdict out of the response, or fail open to 'compatible'.

    Truncated output is discarded rather than mined for keywords: a model cut
    off mid-sentence can easily emit "does not contradict", and keyword
    scanning would read that as `contradicts` and fork the memory wrongly.
    """
    if truncated:
        return "compatible"
    raw = (raw or "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().lower()
            return verdict if verdict in VALID_RELATIONS else "compatible"
    except (json.JSONDecodeError, ValueError):
        pass
    # A bare one-word answer is fine; anything longer is prose we don't trust.
    word = raw.strip().strip(".\"'").lower()
    return word if word in VALID_RELATIONS else "compatible"


_EXTRACT_PROMPT = (
    "You extract durable memories from a finished agent episode.\n"
    "Output ONLY JSON of the form {\"memories\": [{\"text\": str, \"kind\": "
    "\"fact\"|\"procedure\"}]}, with 0 to 3 items.\n"
    "Rules:\n"
    "- Only durable, reusable knowledge about the DATA or the METHOD: schema "
    "shapes, field locations, units and scales, query techniques.\n"
    "- Never record episode trivia: no specific answers, counts, titles, or "
    "results from this one task.\n"
    "- One or two sentences each. If nothing is reusable, return an empty list.\n"
    "Output the JSON object and nothing else."
)


_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["fact", "procedure"]},
                },
                "required": ["text", "kind"],
            },
        }
    },
    "required": ["memories"],
}


def extract_memories(episode_summary: str) -> list[dict[str, str]]:
    """Post-episode extractor. Schema-constrained at the API layer *and*
    re-validated on the way out — it runs unattended on every episode."""
    resp = small_client().chat.completions.create(
        model=config.FIREWORKS_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": episode_summary[:6000]},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Memories", "schema": _EXTRACT_SCHEMA},
        },
        temperature=0,
        max_tokens=EXTRACT_MAX_TOKENS,
    )
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        return []  # truncated JSON is invalid JSON; write nothing rather than junk
    return parse_memory_json(choice.message.content or "")


def parse_memory_json(raw: str) -> list[dict[str, str]]:
    """Tolerant parse of the extractor's output — pure, so it's unit-tested.

    Accepts the schema-constrained `{"memories": [...]}`, a bare array, or
    either of those wrapped in prose or markdown fences.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    items: Any = None
    try:
        parsed = json.loads(raw)
        items = parsed.get("memories") if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if match:
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
