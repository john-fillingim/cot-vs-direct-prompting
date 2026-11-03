"""Run the preregistered analysis on the cleaned results.

python -m src.analyze           data/clean/results.csv        ->  results/
python -m src.analyze --pilot   data/clean/pilot_results.csv  ->  results/pilot/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from src.config import DEFAULT_CONFIG, load_config, resolve, setup_logging

log = logging.getLogger(__name__)
Z95 = stats.norm.ppf(0.975)


# ---------- paired-proportion helpers ----------

def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def newcombe_paired_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Difference p1 - p2 for paired proportions with Newcombe's (1998, method 10) 95% CI.

    a = both correct, b = only condition 1 correct, c = only condition 2 correct, d = both wrong.
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / denom if denom > 0 else 0.0
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    diff = p1 - p2
    return diff, max(diff - dl, -1.0), min(diff + du, 1.0)


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    return 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)


def discordant_or(b: int, c: int) -> tuple[float, float, float]:
    """Discordant odds ratio b/c with an exact (Clopper-Pearson-based) 95% CI."""
    if b + c == 0:
        return math.nan, math.nan, math.nan
    lo, hi = proportion_confint(b, b + c, alpha=0.05, method="beta")

    def to_or(p: float) -> float:
        return math.inf if p >= 1 else p / (1 - p)

    return (math.inf if c == 0 else b / c), to_or(lo), to_or(hi)


def paired_counts(x: pd.Series, y: pd.Series) -> tuple[int, int, int, int]:
    """(a, b, c, d) for two aligned boolean series: both, only x, only y, neither."""
    x, y = x.astype(bool), y.astype(bool)
    return int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())


# ---------- signed-rank helpers ----------

def hodges_lehmann_ci(diff: np.ndarray) -> tuple[float, float, float]:
    """Hodges-Lehmann estimate (median of Walsh averages) with a 95% CI (normal approximation)."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    i, j = np.triu_indices(n)
    walsh = np.sort((diff[i] + diff[j]) / 2)
    m = len(walsh)
    k = int(math.floor(n * (n + 1) / 4 - Z95 * math.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = min(max(k, 0), m - 1)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[m - 1 - k])


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (zeros dropped)."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


# ---------- analyses ----------

def wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Problems x (model_key, prompt_style) table of `value`, complete cases only."""
    return df.pivot_table(index="problem_id", columns=["model_key", "prompt_style"], values=value,
                          aggfunc="first").dropna()


def h1_mcnemar(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per-model exact McNemar, Newcombe CI on the accuracy difference, discordant OR."""
    rows = []
    for model in sorted(df["model_key"].unique()):
        w = wide(df[df["model_key"] == model], "correct")[model]
        a, b, c, d = paired_counts(w["cot"], w["direct"])
        diff, lo, hi = newcombe_paired_ci(a, b, c, d)
        or_, or_lo, or_hi = discordant_or(b, c)
        p = exact_mcnemar(b, c)
        rows.append({
            "model_key": model, "n_pairs": a + b + c + d, "acc_cot": (a + b) / (a + b + c + d),
            "acc_direct": (a + c) / (a + b + c + d), "both_correct": a, "cot_fixed_it": b, "cot_broke_it": c,
            "both_wrong": d, "acc_diff": diff, "diff_ci_low": lo, "diff_ci_high": hi, "discordant_or": or_,
            "or_ci_low": or_lo, "or_ci_high": or_hi, "p_exact": p, "alpha": alpha, "significant": p < alpha,
        })
    return pd.DataFrame(rows)


def omnibus(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Cochran's Q across the four cells, then Holm-corrected pairwise exact McNemar tests."""
    w = wide(df, "correct").astype(int)
    q = cochrans_q(w.to_numpy())
    rows = []
    for c1, c2 in itertools.combinations(w.columns, 2):
        _, b, c, _ = paired_counts(w[c1], w[c2])
        rows.append({"cell_1": "_".join(c1), "cell_2": "_".join(c2), "only_1_correct": b,
                     "only_2_correct": c, "p_exact": exact_mcnemar(b, c)})
    pairs = pd.DataFrame(rows)
    pairs["p_holm"] = multipletests(pairs["p_exact"], method="holm")[1]
    return {"n_problems": len(w), "cochran_q": float(q.statistic), "df": int(q.df),
            "p_value": float(q.pvalue)}, pairs


def h2_interaction(df: pd.DataFrame) -> dict[str, Any]:
    """Prompt x model interaction: conditional logit by problem (primary, LRT) and GEE (robustness)."""
    d = df.assign(cot=(df["prompt_style"] == "cot").astype(float),
                  large=(df["model_key"] == "large").astype(float),
                  correct=df["correct"].astype(float))
    d["cot_x_large"] = d["cot"] * d["large"]
    # Problems with identical outcomes in all cells carry no information in a conditional likelihood.
    informative = d.groupby("problem_id")["correct"].transform("nunique") > 1
    di = d[informative]
    out: dict[str, Any] = {"n_problems": int(d["problem_id"].nunique()),
                           "n_informative_problems": int(di["problem_id"].nunique())}
    try:
        full = ConditionalLogit(di["correct"], di[["cot", "large", "cot_x_large"]],
                                groups=di["problem_id"]).fit(disp=False)
        reduced = ConditionalLogit(di["correct"], di[["cot", "large"]], groups=di["problem_id"]).fit(disp=False)
        lr = 2 * (full.llf - reduced.llf)
        ci = full.conf_int()
        out["conditional_logit"] = {
            "odds_ratios": {k: {"or": float(np.exp(full.params[k])), "ci_low": float(np.exp(ci.loc[k, 0])),
                                "ci_high": float(np.exp(ci.loc[k, 1])), "p_wald": float(full.pvalues[k])}
                            for k in full.params.index},
            "lrt_interaction": {"chi2": float(lr), "df": 1, "p_value": float(stats.chi2.sf(lr, 1))},
        }
    except Exception as e:  # noqa: BLE001 - small pilots can be separable; report rather than crash
        log.warning("Conditional logit failed: %s", e)
        out["conditional_logit"] = {"error": str(e)}
    try:
        gee = smf.gee("correct ~ cot * large", groups="problem_id", data=d, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = gee.conf_int()
        out["gee"] = {k: {"or": float(np.exp(gee.params[k])), "ci_low": float(np.exp(ci.loc[k, 0])),
                          "ci_high": float(np.exp(ci.loc[k, 1])), "p_wald": float(gee.pvalues[k])}
                      for k in gee.params.index}
    except Exception as e:  # noqa: BLE001
        log.warning("GEE failed: %s", e)
        out["gee"] = {"error": str(e)}
    return out


def h3_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model Wilcoxon signed-rank on output tokens (CoT - Direct)."""
    rows = []
    for model in sorted(df["model_key"].unique()):
        w = wide(df[df["model_key"] == model], "output_tokens")[model]
        diff = (w["cot"] - w["direct"]).to_numpy(dtype=float)
        hl, lo, hi = hodges_lehmann_ci(diff)
        res = stats.wilcoxon(diff)
        rows.append({"model_key": model, "n_pairs": len(diff), "median_tokens_cot": float(w["cot"].median()),
                     "median_tokens_direct": float(w["direct"].median()), "median_diff": float(np.median(diff)),
                     "hodges_lehmann": hl, "hl_ci_low": lo, "hl_ci_high": hi,
                     "wilcoxon_stat": float(res.statistic), "p_value": float(res.pvalue),
                     "rank_biserial_r": rank_biserial(diff)})
    return pd.DataFrame(rows)


def bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Problem-level bootstrap: cost per correct answer per cell, and CoT - Direct accuracy per model."""
    corr = wide(df, "correct").astype(float)
    toks = wide(df, "output_tokens").loc[corr.index].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(corr), size=(n_boot, len(corr)))
    c_arr, t_arr = corr.to_numpy(), toks.to_numpy()
    c_sum = c_arr[idx].sum(axis=1)  # (n_boot, cells)
    t_sum = t_arr[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_boot = np.where(c_sum > 0, t_sum / c_sum, np.nan)

    cost_rows = []
    for j, (model, style) in enumerate(corr.columns):
        n_correct = c_arr[:, j].sum()
        cost_rows.append({
            "model_key": model, "prompt_style": style, "n_problems": len(corr),
            "total_output_tokens": float(t_arr[:, j].sum()), "n_correct": int(n_correct),
            "cost_per_correct": float(t_arr[:, j].sum() / n_correct) if n_correct else math.nan,
            "ci_low": float(np.nanpercentile(cost_boot[:, j], 2.5)),
            "ci_high": float(np.nanpercentile(cost_boot[:, j], 97.5)),
        })

    acc_rows = []
    cols = list(corr.columns)
    for model in sorted({m for m, _ in cols}):
        jc, jd = cols.index((model, "cot")), cols.index((model, "direct"))
        diff_boot = (c_sum[:, jc] - c_sum[:, jd]) / len(corr)
        acc_rows.append({"model_key": model,
                         "acc_diff": float((c_arr[:, jc] - c_arr[:, jd]).mean()),
                         "ci_low": float(np.percentile(diff_boot, 2.5)),
                         "ci_high": float(np.percentile(diff_boot, 97.5))})
    return pd.DataFrame(cost_rows), pd.DataFrame(acc_rows)


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (Wilson CI), parse failures, truncation and token use per cell."""
    rows = []
    for (model, style), g in df.groupby(["model_key", "prompt_style"]):
        k, n = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"model_key": model, "model": g["model"].iloc[0], "prompt_style": style, "n": n,
                     "n_correct": k, "accuracy": k / n, "acc_ci_low": lo, "acc_ci_high": hi,
                     "parse_failed": int(g["parse_failed"].sum()), "truncated": int(g["truncated"].sum()),
                     "median_output_tokens": float(g["output_tokens"].median()),
                     "mean_output_tokens": float(g["output_tokens"].mean()),
                     "median_latency_s": float(g["latency_s"].median())})
    return pd.DataFrame(rows)


def exploratory_steps(df: pd.DataFrame) -> pd.DataFrame:
    """EXPLORATORY: CoT gain by reference-solution step count (2, 3, 4, 5+)."""
    d = df.assign(steps=np.where(df["n_steps"] >= 5, "5+", df["n_steps"].astype(int).astype(str)))
    acc = d.groupby(["model_key", "steps", "prompt_style"])["correct"].mean().unstack("prompt_style")
    n = d[d["prompt_style"] == "cot"].groupby(["model_key", "steps"]).size().rename("n_problems")
    out = acc.join(n).reset_index().rename(columns={"cot": "acc_cot", "direct": "acc_direct"})
    out["cot_gain"] = out["acc_cot"] - out["acc_direct"]
    return out


def _clean_json(obj: Any) -> Any:
    """Make floats JSON-safe (inf/nan -> None)."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    df = pd.read_csv(resolve(cfg, "pilot_clean" if args.pilot else "clean"))
    out_dir: Path = resolve(cfg, "results_dir") / ("pilot" if args.pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha, n_boot = cfg["analysis"]["alpha_h1"], cfg["analysis"]["n_bootstrap"]

    cells = cell_summary(df)
    h1 = h1_mcnemar(df, alpha)
    h1_robust = h1_mcnemar(df[~df.groupby(["problem_id", "model_key"])["parse_failed"].transform("any")], alpha)
    q, pairwise = omnibus(df)
    h2 = h2_interaction(df)
    h3 = h3_tokens(df)
    cost, acc_boot = bootstrap(df, n_boot, cfg["seed"])
    steps = exploratory_steps(df)

    for name, table in [("cell_summary", cells), ("h1_mcnemar", h1), ("h1_robust_no_parse_failures", h1_robust),
                        ("omnibus_pairwise_mcnemar", pairwise), ("h3_tokens", h3), ("cost_per_correct", cost),
                        ("robust_bootstrap_acc_diff", acc_boot), ("exploratory_cot_gain_by_steps", steps)]:
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "omnibus_cochran_q.json").write_text(json.dumps(q, indent=2), encoding="utf-8")
    (out_dir / "h2_interaction.json").write_text(json.dumps(_clean_json(h2), indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    log.info("Cells:\n%s", cells.round(3).to_string(index=False))
    log.info("H1:\n%s", h1[["model_key", "acc_cot", "acc_direct", "acc_diff", "diff_ci_low", "diff_ci_high",
                            "cot_fixed_it", "cot_broke_it", "p_exact"]].round(4).to_string(index=False))
    log.info("Cochran's Q = %.2f (df=%d), p = %.4g", q["cochran_q"], q["df"], q["p_value"])
    lrt = h2.get("conditional_logit", {}).get("lrt_interaction")
    if lrt:
        log.info("H2 interaction LRT: chi2 = %.2f, p = %.4g", lrt["chi2"], lrt["p_value"])
    log.info("Cost per correct:\n%s", cost.round(1).to_string(index=False))
    log.info("Wrote results to %s", out_dir)


if __name__ == "__main__":
    main()
