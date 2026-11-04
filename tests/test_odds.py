import pytest

from edgesim.odds import (american_to_decimal, decimal_to_american, expected_value, implied_prob,
                          kelly_fraction, no_vig_probs)


def test_conversions_roundtrip():
    for a in (-250, -110, 100, 150, 400):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)
    assert american_to_decimal(-110) == pytest.approx(1.90909, rel=1e-4)
    assert implied_prob(-110) == pytest.approx(0.5238, rel=1e-3)


def test_no_vig_sums_to_one_and_is_fair():
    p = no_vig_probs([-110, -110])
    assert p == pytest.approx([0.5, 0.5])
    p = no_vig_probs([-200, 170])
    assert sum(p) == pytest.approx(1)
    assert p[0] > 0.6


def test_ev_breakeven_and_push():
    assert expected_value(0.5238, -110) == pytest.approx(0, abs=1e-3)
    assert expected_value(0.55, -110) > 0
    # a push returns the stake: EV ignores it
    assert expected_value(0.45, +100, p_push=0.1) == pytest.approx(0.0)


def test_kelly_zero_when_negative_and_capped():
    assert kelly_fraction(0.50, -110) == 0
    assert 0 < kelly_fraction(0.56, -110) <= 0.03
    assert kelly_fraction(0.9, +200, cap=0.03) == 0.03
