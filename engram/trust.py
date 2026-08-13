"""Pure trust arithmetic and lifecycle transitions.

Deliberately free of MongoDB imports so the rules that govern a memory's life
can be unit-tested in isolation. `store.py` translates these into atomic
`$set`/`$inc` operations.
"""

from __future__ import annotations

from dataclasses import dataclass

PROVISIONAL = "provisional"
TRUSTED = "trusted"
QUARANTINED = "quarantined"
DEAD = "dead"

INITIAL_TRUST = 0.30
TRUST_FLOOR = 0.05
TRUST_CEILING = 1.0

# Lifecycle thresholds
PROMOTE_TRUST = 0.60
PROMOTE_WINS = 2
QUARANTINE_TRUST = 0.15
DEAD_LOSSES = 3

# Outcome deltas
WIN_LEARNING_RATE = 0.25  # trust += 0.25 * (1 - trust)
LOSS_SLASH = 0.4  # trust *= 0.4
IGNORED_PENALTY = 0.02  # retrieved but not cited
DECAY_FACTOR = 0.98

# Contamination cascade
CASCADE_DIRECT = 0.5  # depth 0 == direct children
CASCADE_DEEP = 0.7


def clamp(trust: float) -> float:
    return max(TRUST_FLOOR, min(TRUST_CEILING, trust))


def on_cited_pass(trust: float) -> float:
    """Cited in a passing episode: asymptotic climb toward 1.0."""
    return clamp(trust + WIN_LEARNING_RATE * (1.0 - trust))


def on_cited_fail(trust: float) -> float:
    """Cited in a failing episode: multiplicative slash. This is the sharp edge."""
    return clamp(trust * LOSS_SLASH)


def on_retrieved_unused(trust: float) -> float:
    """Surfaced by retrieval but the agent didn't lean on it: mild decay."""
    return clamp(trust - IGNORED_PENALTY)


def on_decay(trust: float) -> float:
    return clamp(trust * DECAY_FACTOR)


def cascade_multiplier(depth: int) -> float:
    """Trust multiplier applied to a descendant of a quarantined memory.

    `depth` comes from $graphLookup's depthField: 0 == direct child.
    """
    return CASCADE_DIRECT if depth <= 0 else CASCADE_DEEP


def next_status(
    current: str, trust: float, wins: int, losses: int, falsified: bool = False
) -> str:
    """Deterministic lifecycle transition, evaluated after every trust update.

    Order matters: an already-quarantined memory can only fall further, never
    climb back on its own — it has to be re-earned through a fresh write.

    `falsified` — a memory cited in a failing episode *while it held `trusted`*
    — quarantines immediately rather than decaying toward it. `trusted` is a
    standing claim that a memory has earned the right to be served by default;
    one demonstrated failure under that claim falsifies it. Gradual decay is
    also unsound here: the moment a discredited memory loses retrieval
    arbitration to something it contradicts, it stops being served, stops being
    citable, and can never accumulate the losses that would have condemned it.
    It would sit at middling trust forever, dormant but unaccountable.
    Quarantine is not deletion — the claim can be rewritten and earn its way
    back.
    """
    if current == DEAD:
        return DEAD
    if current == QUARANTINED:
        return DEAD if losses >= DEAD_LOSSES else QUARANTINED
    if falsified and current == TRUSTED:
        return QUARANTINED
    if trust < QUARANTINE_TRUST:
        return QUARANTINED
    if trust >= PROMOTE_TRUST and wins >= PROMOTE_WINS:
        return TRUSTED
    return current


@dataclass(frozen=True)
class TrustUpdate:
    """The computed effect of one episode on one memory."""

    mid: str
    reason: str  # "cited_pass" | "cited_fail" | "retrieved_unused"
    trust_before: float
    trust_after: float
    win_delta: int
    loss_delta: int
    status_before: str
    status_after: str

    @property
    def status_changed(self) -> bool:
        return self.status_before != self.status_after


def compute_update(
    *,
    mid: str,
    status: str,
    trust: float,
    wins: int,
    losses: int,
    cited: bool,
    passed: bool,
) -> TrustUpdate:
    """Given a memory's current state and how it fared in an episode, compute
    the full update. Callers apply it atomically."""
    if cited and passed:
        reason, new_trust, dw, dl = "cited_pass", on_cited_pass(trust), 1, 0
    elif cited and not passed:
        reason, new_trust, dw, dl = "cited_fail", on_cited_fail(trust), 0, 1
    else:
        reason, new_trust, dw, dl = "retrieved_unused", on_retrieved_unused(trust), 0, 0

    falsified = cited and not passed and status == TRUSTED
    new_status = next_status(status, new_trust, wins + dw, losses + dl,
                             falsified=falsified)
    return TrustUpdate(
        mid=mid,
        reason=reason,
        trust_before=trust,
        trust_after=new_trust,
        win_delta=dw,
        loss_delta=dl,
        status_before=status,
        status_after=new_status,
    )


def retrieval_score(vector_score: float, trust: float) -> float:
    """Mirror of the $addFields stage in the retrieval pipeline.

    Trust never fully silences a memory (0.4 floor) but doubles the weight of
    an earned one. Kept here so the Python and the aggregation agree.
    """
    return vector_score * (0.4 + 0.6 * trust)
