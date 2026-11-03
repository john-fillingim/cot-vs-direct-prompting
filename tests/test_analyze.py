"""Tests for the paired statistics in src.analyze."""

import math

import numpy as np
import pandas as pd
import pytest

from src.analyze import (
    bootstrap,
    discordant_or,
    exact_mcnemar,
    h1_mcnemar,
    h2_interaction,
    h3_tokens,
    hodges_lehmann_ci,
    newcombe_paired_ci,
    omnibus,
    rank_biserial,
)


def test_exact_mcnemar_known_values() -> None:
    assert exact_mcnemar(10, 0) == pytest.approx(2 * 0.5**10)
    assert exact_mcnemar(5, 5) == pytest.approx(1.0)
    assert exact_mcnemar(0, 0) == 1.0


def test_newcombe_ci_contains_estimate_and_is_bounded() -> None:
    diff, lo, hi = newcombe_paired_ci(a=60, b=30, c=10, d=100)
    assert diff == pytest.approx(0.10)
    assert -1 <= lo < diff < hi <= 1
    # Swapping the two conditions mirrors the interval.
    d2, lo2, hi2 = newcombe_paired_ci(a=60, b=10, c=30, d=100)
    assert (d2, lo2, hi2) == pytest.approx((-diff, -hi, -lo))


def test_newcombe_no_discordance_is_zero() -> None:
    diff, lo, hi = newcombe_paired_ci(a=100, b=0, c=0, d=100)
    assert diff == 0 and lo <= 0 <= hi


def test_discordant_or() -> None:
    or_, lo, hi = discordant_or(30, 10)
    assert or_ == 3.0 and lo < 3.0 < hi
    assert math.isinf(discordant_or(5, 0)[0])


def test_hodges_lehmann_and_rank_biserial() -> None:
    diff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    hl, lo, hi = hodges_lehmann_ci(diff)
    assert hl == pytest.approx(5.5) and lo < hl < hi
    assert rank_biserial(diff) == 1.0
    assert rank_biserial(-diff) == -1.0


def _synthetic(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Long-format results with a CoT gain only for the large model."""
    rng = np.random.default_rng(seed)
    p = {("small", "direct"): 0.4, ("small", "cot"): 0.4, ("large", "direct"): 0.5, ("large", "cot"): 0.85}
    rows = []
    for pid in range(n):
        u = rng.random()  # shared problem difficulty induces pairing
        for (model, style), prob in p.items():
            correct = u < prob
            rows.append({"problem_id": pid, "model_key": model, "prompt_style": style, "correct": correct,
                         "parse_failed": False, "output_tokens": 300 if style == "cot" else 3})
    return pd.DataFrame(rows)


def test_pipeline_on_synthetic_data() -> None:
    df = _synthetic()
    h1 = h1_mcnemar(df, alpha=0.025).set_index("model_key")
    assert h1.loc["large", "significant"] and not h1.loc["small", "significant"]
    assert h1.loc["large", "cot_broke_it"] == 0

    q, pairs = omnibus(df)
    assert q["p_value"] < 0.001 and len(pairs) == 6

    h3 = h3_tokens(df).set_index("model_key")
    assert h3.loc["large", "median_diff"] == 297 and h3.loc["large", "rank_biserial_r"] == 1.0

    cost, acc = bootstrap(df, n_boot=200, seed=1)
    assert set(cost["prompt_style"]) == {"cot", "direct"}
    large = acc.set_index("model_key").loc["large"]
    assert large["ci_low"] <= large["acc_diff"] <= large["ci_high"]


def test_h2_reports_lrt_or_error_without_crashing() -> None:
    rng = np.random.default_rng(3)
    df = _synthetic()
    df["correct"] = df["correct"] ^ (rng.random(len(df)) < 0.1)  # noise so the model is estimable
    out = h2_interaction(df)
    assert out["conditional_logit"]["lrt_interaction"]["p_value"] < 0.05
    assert "cot:large" in out["gee"]
