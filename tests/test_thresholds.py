"""Atlas does not return raw cosine — `$vectorSearch` normalizes it to
`(1 + cosine) / 2`. Every similarity constant in this project lives in that
normalized space, and these tests pin the translation so nobody re-reads the
spec's raw-cosine intuition back into the code."""

import pytest

from engram import config


def raw_cosine(atlas_score: float) -> float:
    """Invert Atlas's normalization: score = (1 + cos) / 2."""
    return 2 * atlas_score - 1


def test_merge_gate_is_a_genuine_near_duplicate():
    assert config.MERGE_SIMILARITY == 0.92
    assert raw_cosine(config.MERGE_SIMILARITY) == pytest.approx(0.84)


def test_contradiction_band_lower_bound_is_not_the_specs_raw_cosine_number():
    """0.75 in normalized space is raw cosine 0.50 — 'vaguely the same topic'.
    That would send nearly every write through the classifier."""
    assert raw_cosine(0.75) == pytest.approx(0.50)
    assert config.CONTRADICTION_LOW == 0.85
    assert raw_cosine(config.CONTRADICTION_LOW) == pytest.approx(0.70)


def test_the_band_is_ordered_and_non_empty():
    assert 0.0 < config.CONTRADICTION_LOW < config.MERGE_SIMILARITY <= 1.0


def test_both_embedding_providers_share_one_dimension_count():
    """The vector index is built once. If Voyage and Fireworks disagreed on
    dims, swapping providers would silently break every query."""
    assert config.EMBED_DIMS == 512


def test_retrieval_widths_leave_room_for_the_trust_reranking_to_matter():
    """The index returns 12 candidates and trust re-ranks them down to 4 —
    if limit == k the trust weighting could never change the outcome."""
    assert config.VECTOR_LIMIT > config.RETRIEVE_K
    assert config.NUM_CANDIDATES >= config.VECTOR_LIMIT
