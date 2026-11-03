"""Make the report figures from the analysis outputs.

python -m src.figures           results/        ->  figures/*.png
python -m src.figures --pilot   results/pilot/  ->  figures/pilot/*.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DEFAULT_CONFIG, load_config, resolve, setup_logging  # noqa: E402

log = logging.getLogger(__name__)

# One color per prompt style, used in every figure (validated for color-vision deficiency).
COLORS = {"direct": "#2a78d6", "cot": "#eb6834"}
STYLE_LABELS = {"direct": "Direct", "cot": "CoT"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = ["small", "large"]
DPI = 300
LABEL_OFFSETS = {("small", "direct"): (20, 35), ("large", "direct"): (20, 60)}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "text.color": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def model_label(name: str) -> str:
    """'qwen2.5:1.5b' -> 'Qwen2.5 1.5B'."""
    family, size = name.split(":")
    return f"{family.capitalize()} {size.upper()}"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_dir / name)


def fig_accuracy(cells: pd.DataFrame, h1: pd.DataFrame, out_dir: Path) -> None:
    """Hero: accuracy per condition with Wilson 95% CIs, Direct and CoT joined per model."""
    labels = {r.model_key: model_label(r.model) for r in cells.itertuples()}
    gains = h1.set_index("model_key")["acc_diff"] * 100
    title = "CoT raises accuracy by " + " and ".join(
        f"{gains[m]:.0f} pts ({labels[m]})" for m in MODEL_ORDER if m in gains)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    offset = {"direct": -0.12, "cot": 0.12}
    for i, model in enumerate(MODEL_ORDER):
        c = cells[cells["model_key"] == model].set_index("prompt_style")
        if c.empty:
            continue
        xs = [i + offset[s] for s in ("direct", "cot")]
        ax.plot(xs, c.loc[["direct", "cot"], "accuracy"] * 100, color=INK_2, lw=1.5, zorder=1)
        for s in ("direct", "cot"):
            r = c.loc[s]
            x, y = i + offset[s], r["accuracy"] * 100
            ax.errorbar(x, y, yerr=[[y - r["acc_ci_low"] * 100], [r["acc_ci_high"] * 100 - y]],
                        fmt="o", ms=9, color=COLORS[s], ecolor=COLORS[s], elinewidth=2, capsize=0,
                        mec="white", mew=1.5, zorder=2, label=STYLE_LABELS[s] if i == 0 else None)
            ax.annotate(f"{y:.0f}%", (x, y), xytext=(-10 if s == "direct" else 10, 0),
                        textcoords="offset points", ha="right" if s == "direct" else "left",
                        va="center", color=INK, fontsize=10)
    ax.set_xticks(range(len(MODEL_ORDER)), [labels.get(m, m) for m in MODEL_ORDER])
    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy on GSM8K (% correct)")
    ax.grid(axis="x", visible=False)
    n = int(cells["n"].max())
    ax.set_title(f"{title}\n", loc="left")
    ax.text(0, 1.02, f"n = {n} problems per condition; whiskers = 95% Wilson CI",
            transform=ax.transAxes, color=INK_2, fontsize=9)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    _save(fig, out_dir, "fig1_accuracy_by_condition.png")


def fig_discordant(h1: pd.DataFrame, cells: pd.DataFrame, out_dir: Path) -> None:
    """Discordant pairs per model: 'CoT fixed it' vs 'CoT broke it'."""
    labels = {r.model_key: model_label(r.model) for r in cells.itertuples()}
    h = h1.set_index("model_key").reindex([m for m in MODEL_ORDER if m in set(h1["model_key"])])
    fixed, broke = int(h["cot_fixed_it"].sum()), int(h["cot_broke_it"].sum())

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bar_h = 0.36
    for i, (_, r) in enumerate(h.iterrows()):
        for j, (col, style, text) in enumerate([("cot_fixed_it", "cot", "CoT fixed it"),
                                                ("cot_broke_it", "direct", "CoT broke it")]):
            y = i + (j - 0.5) * (bar_h + 0.04)
            ax.barh(y, r[col], height=bar_h, color=COLORS[style], label=text if i == 0 else None)
            ax.text(r[col] + 0.5, y, f"{int(r[col])}", va="center", color=INK, fontsize=9)
    ticks = []
    for model, r in h.iterrows():
        p = r["p_exact"]
        ticks.append(f"{labels.get(model, model)}\nMcNemar p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}")
    ax.set_yticks(range(len(h)), ticks)
    ax.set_xlim(0, h[["cot_fixed_it", "cot_broke_it"]].to_numpy().max() * 1.12 + 1)
    ax.invert_yaxis()
    ax.set_xlabel("Problems (count)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"CoT fixed {fixed} answers and broke {broke}\n\n", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    _save(fig, out_dir, "fig2_discordant_pairs.png")


def fig_tokens(cells: pd.DataFrame, cost: pd.DataFrame, out_dir: Path) -> None:
    """Accuracy vs median output tokens, one labelled point per condition."""
    c = cells.merge(cost[["model_key", "prompt_style", "cost_per_correct"]], on=["model_key", "prompt_style"])
    med = c.groupby("prompt_style")["median_output_tokens"].median()
    ratio = med["cot"] / med["direct"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"small": "o", "large": "s"}
    for r in c.itertuples():
        ax.scatter(r.median_output_tokens, r.accuracy * 100, s=90, marker=markers[r.model_key],
                   color=COLORS[r.prompt_style], edgecolor="white", linewidth=1.5, zorder=2)
        # The two Direct points nearly coincide, so their labels are stacked above with leader lines.
        dx, dy = LABEL_OFFSETS.get((r.model_key, r.prompt_style), (8, 0))
        ax.annotate(f"{model_label(r.model)} {STYLE_LABELS[r.prompt_style]}: "
                    f"{r.cost_per_correct:.0f} tokens per correct answer",
                    (r.median_output_tokens, r.accuracy * 100), xytext=(dx, dy), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8, "relpos": (0, 0.5)} if dy else None)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Median output tokens per answer (log scale)")
    ax.set_ylabel("Accuracy on GSM8K (% correct)")
    ax.set_title(f"CoT uses ~{ratio:.0f}x more output tokens for its accuracy gain", loc="left")
    _save(fig, out_dir, "fig3_accuracy_vs_tokens.png")


def fig_steps(steps: pd.DataFrame, cells: pd.DataFrame, out_dir: Path) -> None:
    """EXPLORATORY: CoT gain (CoT - Direct accuracy) by reference-solution step count."""
    labels = {r.model_key: model_label(r.model) for r in cells.itertuples()}
    order = ["2", "3", "4", "5+"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    shades = {"small": INK_2, "large": INK}
    for model in MODEL_ORDER:
        s = steps[steps["model_key"] == model].set_index("steps").reindex(order).dropna(subset=["cot_gain"])
        if s.empty:
            continue
        xs = [order.index(k) for k in s.index]
        ax.plot(xs, s["cot_gain"] * 100, marker="o", ms=8, lw=2, color=shades[model],
                ls="-" if model == "large" else "--", label=labels.get(model, model))
    n = steps.groupby("steps")["n_problems"].max().reindex(order)
    ax.set_xticks(range(len(order)), [f"{k}\n(n={int(v) if pd.notna(v) else 0})" for k, v in n.items()])
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlabel("Steps in reference solution")
    ax.set_ylabel("CoT gain (accuracy points, CoT − Direct)")
    ax.set_title("Exploratory: CoT gain by problem difficulty", loc="left")
    ax.legend(loc="best")
    _save(fig, out_dir, "fig4_exploratory_gain_by_steps.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    sub = "pilot" if args.pilot else ""
    res_dir = resolve(cfg, "results_dir") / sub
    out_dir = resolve(cfg, "figures_dir") / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(res_dir / "cell_summary.csv")
    h1 = pd.read_csv(res_dir / "h1_mcnemar.csv")
    cost = pd.read_csv(res_dir / "cost_per_correct.csv")
    steps = pd.read_csv(res_dir / "exploratory_cot_gain_by_steps.csv", dtype={"steps": str})

    fig_accuracy(cells, h1, out_dir)
    fig_discordant(h1, cells, out_dir)
    fig_tokens(cells, cost, out_dir)
    fig_steps(steps, cells, out_dir)


if __name__ == "__main__":
    main()
