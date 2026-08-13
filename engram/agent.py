"""The agent loop: retrieve_memories -> act -> record.

A LangGraph graph checkpointed to the same Atlas cluster that holds the
memories. One thread per episode, so a crashed run resumes from its last node
instead of re-burning the task — and agent state and agent memory live on one
platform, with Engram governing the boundary between them.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph

from . import checkers, config, events, store
from .llm import extract_memories, main_chat
from .mongo_tool import TOOL_SCHEMA, run_mongo_query

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are a data analyst answering questions about the MongoDB \
`sample_mflix` sample dataset. You have one tool, `run_mongo_query`, which runs \
read-only find/aggregate queries against that database.

You may be given MEMORY entries: claims a previous run of this system learned and \
stored, each with an id and a trust score. Treat them as useful prior knowledge \
about the data.

CITATION RULES — these are strict:
- When you rely on a memory to decide how to query or how to interpret a result, \
cite its id inline like [M-0007] in your reasoning.
- Do not cite memories you did not use.

Work by querying the database, then give a final answer. Your last message must \
end with a line of exactly this form:

FINAL ANSWER: <the answer, and nothing else on that line>

Keep the final answer minimal: a bare number for counts, or the titles/names \
separated by commas for list questions."""


class EpisodeState(TypedDict, total=False):
    task_id: str
    prompt: str
    expected: str
    checker: str
    thread_id: str
    retrieved: list[str]
    memories: list[dict[str, Any]]
    trace: str
    answer: str
    cited: list[str]
    outcome: str
    episode_id: str
    latency_ms: int
    started_at: float
    tool_calls: int


def render_memories(docs: list[dict[str, Any]]) -> str:
    """The exact shape the model sees. Ids are what make citation possible."""
    if not docs:
        return "(no memories yet — you are working from scratch)"
    return "\n".join(
        f"MEMORY [{d['mid']}] (trust {float(d.get('trust', 0)):.2f}): {d['text']}"
        for d in docs
    )


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


def _checkpointable(doc: dict[str, Any]) -> dict[str, Any]:
    """Reduce a memory doc to the fields the loop actually uses.

    Graph state is serialized into Atlas by MongoDBSaver, and BSON ObjectIds
    (`_id`, `source_episode`) are not msgpack-serializable — carrying a raw
    Mongo document through the state crashes the checkpointer.
    """
    return {
        "mid": doc["mid"],
        "text": doc.get("text", ""),
        "trust": float(doc.get("trust", 0.0)),
        "status": doc.get("status", ""),
        "score": round(float(doc.get("score", 0.0)), 4),
    }


def retrieve_memories(state: EpisodeState) -> EpisodeState:
    docs = store.retrieve(state["prompt"], k=config.RETRIEVE_K)
    return {"memories": [_checkpointable(d) for d in docs],
            "retrieved": [d["mid"] for d in docs],
            "started_at": time.time()}


def act(state: EpisodeState) -> EpisodeState:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{render_memories(state.get('memories') or [])}\n\n"
                f"TASK: {state['prompt']}"
            ),
        },
    ]
    trace_parts: list[str] = []
    tool_calls = 0

    for _ in range(MAX_TOOL_ROUNDS):
        response = main_chat(messages, tools=[TOOL_SCHEMA])
        message = response.choices[0].message
        content = message.content or ""
        if content:
            trace_parts.append(content)

        calls = getattr(message, "tool_calls", None)
        if not calls:
            break

        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ],
        })
        for call in calls:
            tool_calls += 1
            result = run_mongo_query(call.function.arguments)
            trace_parts.append(
                f"[query] {call.function.arguments}\n[result] {json.dumps(result)[:900]}"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)[:6000],
            })

    trace = "\n\n".join(trace_parts)
    return {"trace": trace, "answer": _final_answer(trace), "tool_calls": tool_calls}


def _final_answer(trace: str) -> str:
    for line in reversed((trace or "").splitlines()):
        if "FINAL ANSWER:" in line:
            return line.split("FINAL ANSWER:", 1)[1].strip()
    return (trace or "").strip().splitlines()[-1] if trace else ""


def record(state: EpisodeState) -> EpisodeState:
    trace = state.get("trace", "")
    answer = state.get("answer", "")
    retrieved = list(state.get("retrieved") or [])

    # Only citations of memories actually served this episode count. A model
    # inventing [M-9999] must not create trust signal for a memory it never saw.
    cited = [m for m in checkers.parse_citations(trace) if m in retrieved]
    passed = checkers.check(answer, state.get("expected", ""), state.get("checker", "exact"))
    latency_ms = int((time.time() - state.get("started_at", time.time())) * 1000)

    if cited:
        events.emit(events.MEMORY_CITED, mids=cited, task_id=state["task_id"])

    episode = {
        "task_id": state["task_id"],
        "thread_id": state.get("thread_id", ""),
        "retrieved": retrieved,
        "cited": cited,
        "outcome": "pass" if passed else "fail",
        "answer": answer,
        "expected": state.get("expected", ""),
        "latency_ms": latency_ms,
        "tool_calls": state.get("tool_calls", 0),
        "started_at": datetime.fromtimestamp(
            state.get("started_at", time.time()), tz=timezone.utc
        ),
        "ended_at": datetime.now(timezone.utc),
    }
    episode_id = config.episodes().insert_one(episode).inserted_id
    episode["_id"] = episode_id

    store.apply_outcome(episode)
    _extract(state, episode, episode_id, cited)

    events.emit(
        events.EPISODE_END,
        task_id=state["task_id"],
        outcome=episode["outcome"],
        answer=answer[:200],
        expected=state.get("expected", "")[:200],
        cited=cited,
        retrieved=retrieved,
        latency_ms=latency_ms,
    )
    return {"cited": cited, "outcome": episode["outcome"],
            "episode_id": str(episode_id), "latency_ms": latency_ms}


def _extract(state: EpisodeState, episode: dict, episode_id: Any, cited: list[str]) -> None:
    """Metabolism tier. Every memory born here records the memories that were
    cited in the episode that produced it — that's the provenance edge the
    contamination trace later walks."""
    summary = (
        f"TASK: {state['prompt']}\n"
        f"MEMORIES IN CONTEXT:\n{render_memories(state.get('memories') or [])}\n"
        f"TRANSCRIPT:\n{state.get('trace', '')[:4000]}\n"
        f"FINAL ANSWER: {state.get('answer', '')}\n"
        f"OUTCOME: {episode['outcome']}"
    )
    try:
        candidates = extract_memories(summary)
    except Exception as exc:
        events.emit(events.NOTE, note=f"extractor failed: {exc}")
        return

    for item in candidates:
        try:
            store.write_memory(
                item["text"], item["kind"], source_episode=episode_id, parents=cited
            )
        except Exception as exc:
            events.emit(events.NOTE, note=f"write_memory failed: {exc}")


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def build_graph(checkpointer: Any = None):
    builder = StateGraph(EpisodeState)
    builder.add_node("retrieve_memories", retrieve_memories)
    builder.add_node("act", act)
    builder.add_node("record", record)
    builder.add_edge(START, "retrieve_memories")
    builder.add_edge("retrieve_memories", "act")
    builder.add_edge("act", "record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=checkpointer)


_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        saver = MongoDBSaver(config.client(), db_name=config.DB_NAME)
        _GRAPH = build_graph(saver)
    return _GRAPH


def run_task(task: dict[str, Any], thread_id: str | None = None) -> dict[str, Any]:
    """Run one task end-to-end as a checkpointed LangGraph thread."""
    thread_id = thread_id or f"{task['task_id']}-{uuid.uuid4().hex[:8]}"
    events.emit(events.EPISODE_START, task_id=task["task_id"],
                thread_id=thread_id, prompt=task["prompt"][:200])
    state: EpisodeState = {
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "expected": task.get("expected_answer", ""),
        "checker": task.get("checker", "exact"),
        "thread_id": thread_id,
    }
    return graph().invoke(state, config={"configurable": {"thread_id": thread_id}})
