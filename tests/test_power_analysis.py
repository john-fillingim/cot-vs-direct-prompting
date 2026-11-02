"""Tests for the McNemar sample-size formula."""

import math

from src.power_analysis import connor_n


def test_connor_planned_n() -> None:
    assert connor_n(p10=0.15, p01=0.05, alpha=0.025, power=0.80) == 188


def test_connor_no_effect_is_infinite() -> None:
    assert math.isinf(connor_n(0.1, 0.1, 0.025, 0.8))
