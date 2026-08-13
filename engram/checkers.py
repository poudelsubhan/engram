"""Answer checkers and citation parsing.

Pure functions — no I/O — so the pass/fail verdict that drives every trust
update is deterministic and testable.
"""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[(M-\d{1,6})\]")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_citations(text: str) -> list[str]:
    """Pull `[M-0007]`-style citations out of a trace, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for match in CITATION_RE.findall(text or ""):
        seen.setdefault(match, None)
    return list(seen)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .")


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER_RE.findall(text or "")]


def check_exact(answer: str, expected: str) -> bool:
    return _normalize(answer) == _normalize(expected)


def check_numeric_tol(answer: str, expected: str, tol: float = 0.02) -> bool:
    """True if any number in the answer matches the expected number within a
    relative tolerance. Agents pad numbers with prose; we only care that the
    right one is in there."""
    want = _numbers(expected)
    if not want:
        return False
    target = want[0]
    got = _numbers(answer)
    if not got:
        return False
    slack = max(abs(target) * tol, 0.5)
    return any(abs(g - target) <= slack for g in got)


def check_contains(answer: str, expected: str) -> bool:
    """Every `|`-separated fragment of `expected` must appear in the answer.

    Used for ranking tasks where we care about the set and order of titles,
    not the sentence wrapped around them.
    """
    hay = _normalize(answer)
    parts = [_normalize(p) for p in expected.split("|") if p.strip()]
    return bool(parts) and all(p in hay for p in parts)


CHECKERS = {
    "exact": check_exact,
    "numeric_tol": check_numeric_tol,
    "contains": check_contains,
}


def check(answer: str, expected: str, checker: str) -> bool:
    fn = CHECKERS.get(checker, check_exact)
    return bool(fn(answer, expected))
