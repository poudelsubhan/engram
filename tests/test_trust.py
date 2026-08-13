"""The trust rules are the contract. If these drift, the demo stops meaning anything."""

import pytest

from engram import trust as T


def test_initial_constants_match_spec():
    assert T.INITIAL_TRUST == 0.30
    assert T.PROMOTE_TRUST == 0.60 and T.PROMOTE_WINS == 2
    assert T.QUARANTINE_TRUST == 0.15
    assert T.DEAD_LOSSES == 3


def test_cited_pass_climbs_asymptotically():
    assert T.on_cited_pass(0.30) == pytest.approx(0.475)
    assert T.on_cited_pass(0.85) == pytest.approx(0.8875)
    # never exceeds 1.0, no matter how many wins
    trust = 0.5
    for _ in range(100):
        trust = T.on_cited_pass(trust)
    assert trust <= 1.0


def test_cited_fail_slashes_multiplicatively():
    assert T.on_cited_fail(0.80) == pytest.approx(0.32)
    assert T.on_cited_fail(0.30) == pytest.approx(0.12)


def test_retrieved_unused_decays_with_a_floor():
    assert T.on_retrieved_unused(0.30) == pytest.approx(0.28)
    assert T.on_retrieved_unused(0.05) == pytest.approx(T.TRUST_FLOOR)
    assert T.on_retrieved_unused(0.06) >= T.TRUST_FLOOR


def test_trust_never_leaves_bounds():
    for value in (-5.0, 0.0, 0.5, 1.0, 9.0):
        assert T.TRUST_FLOOR <= T.clamp(value) <= T.TRUST_CEILING


class TestTransitions:
    def test_promotion_needs_both_trust_and_wins(self):
        assert T.next_status(T.PROVISIONAL, 0.65, wins=2, losses=0) == T.TRUSTED
        assert T.next_status(T.PROVISIONAL, 0.65, wins=1, losses=0) == T.PROVISIONAL
        assert T.next_status(T.PROVISIONAL, 0.55, wins=5, losses=0) == T.PROVISIONAL

    def test_quarantine_line(self):
        assert T.next_status(T.TRUSTED, 0.14, wins=9, losses=1) == T.QUARANTINED
        assert T.next_status(T.PROVISIONAL, 0.15, wins=0, losses=0) == T.PROVISIONAL

    def test_quarantined_cannot_climb_back_on_its_own(self):
        assert T.next_status(T.QUARANTINED, 0.99, wins=10, losses=0) == T.QUARANTINED

    def test_quarantined_dies_after_three_losses(self):
        assert T.next_status(T.QUARANTINED, 0.10, wins=0, losses=3) == T.DEAD
        assert T.next_status(T.QUARANTINED, 0.10, wins=0, losses=2) == T.QUARANTINED

    def test_dead_is_terminal(self):
        assert T.next_status(T.DEAD, 1.0, wins=99, losses=0) == T.DEAD


class TestComputeUpdate:
    def test_cited_pass_records_a_win(self):
        u = T.compute_update(mid="M-1", status=T.PROVISIONAL, trust=0.30, wins=1,
                             losses=0, cited=True, passed=True)
        assert u.reason == "cited_pass" and u.win_delta == 1 and u.loss_delta == 0
        assert u.trust_after == pytest.approx(0.475)

    def test_cited_fail_records_a_loss(self):
        u = T.compute_update(mid="M-1", status=T.TRUSTED, trust=0.80, wins=3,
                             losses=0, cited=True, passed=False)
        assert u.reason == "cited_fail" and u.loss_delta == 1
        assert u.trust_after == pytest.approx(0.32)

    def test_retrieved_but_not_cited_is_neither_win_nor_loss(self):
        u = T.compute_update(mid="M-1", status=T.PROVISIONAL, trust=0.40, wins=0,
                             losses=0, cited=False, passed=True)
        assert u.reason == "retrieved_unused"
        assert u.win_delta == 0 and u.loss_delta == 0

    def test_promotion_is_reported_as_a_status_change(self):
        u = T.compute_update(mid="M-1", status=T.PROVISIONAL, trust=0.50, wins=1,
                             losses=0, cited=True, passed=True)
        assert u.trust_after == pytest.approx(0.625)
        assert u.status_after == T.TRUSTED and u.status_changed


def test_poison_arc_is_deterministic():
    """The demo depends on this exact arc: a trusted lie survives one failure
    and crosses the quarantine line on the second."""
    trust, status, wins, losses = 0.85, T.TRUSTED, 0, 0

    # Beat 1: the ranking task passes while citing the lie — it gains trust.
    u = T.compute_update(mid="LIE", status=status, trust=trust, wins=wins,
                         losses=losses, cited=True, passed=True)
    trust, status, wins = u.trust_after, u.status_after, wins + u.win_delta
    assert trust == pytest.approx(0.8875) and status == T.TRUSTED

    # Beat 2, first failure: slashed hard but still above the line.
    u = T.compute_update(mid="LIE", status=status, trust=trust, wins=wins,
                         losses=losses, cited=True, passed=False)
    trust, status, losses = u.trust_after, u.status_after, losses + u.loss_delta
    assert trust == pytest.approx(0.355) and status == T.TRUSTED

    # Beat 2, second failure: crosses 0.15 -> quarantined -> cascade fires.
    u = T.compute_update(mid="LIE", status=status, trust=trust, wins=wins,
                         losses=losses, cited=True, passed=False)
    assert u.trust_after == pytest.approx(0.142)
    assert u.status_after == T.QUARANTINED and u.status_changed


def test_cascade_multiplier_hits_direct_children_hardest():
    assert T.cascade_multiplier(0) == 0.5
    assert T.cascade_multiplier(1) == 0.7
    assert T.cascade_multiplier(4) == 0.7


def test_contaminated_child_can_be_knocked_below_the_line():
    """A provisional child of a lie (trust 0.30) survives as provisional;
    a barely-alive one gets quarantined by the cascade itself."""
    assert T.clamp(0.30 * T.cascade_multiplier(0)) == pytest.approx(0.15)
    assert T.clamp(0.20 * T.cascade_multiplier(0)) == pytest.approx(0.10)
    assert T.clamp(0.20 * T.cascade_multiplier(0)) < T.QUARANTINE_TRUST


def test_retrieval_score_matches_the_aggregation_formula():
    assert T.retrieval_score(0.9, 0.0) == pytest.approx(0.36)
    assert T.retrieval_score(0.9, 1.0) == pytest.approx(0.90)
    # a high-trust memory outranks a slightly-closer untrusted one
    assert T.retrieval_score(0.80, 0.90) > T.retrieval_score(0.88, 0.30)


def test_decay_is_gentle():
    assert T.on_decay(0.50) == pytest.approx(0.49)
    assert T.on_decay(T.TRUST_FLOOR) == pytest.approx(T.TRUST_FLOOR)
