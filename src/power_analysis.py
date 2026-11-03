"""Sample size for a paired McNemar test (Connor, 1987).

Planned:  python -m src.power_analysis
Pilot:    python -m src.power_analysis --pilot   (uses observed discordance from pilot_results.csv)
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import pandas as pd
from scipy.stats import norm

from src.config import DEFAULT_CONFIG, load_config, resolve, setup_logging

log = logging.getLogger(__name__)


def connor_n(p10: float, p01: float, alpha: float, power: float) -> int | float:
    """Number of pairs needed for a two-sided McNemar test (Connor, 1987).

    p10 = P(CoT correct, Direct wrong); p01 = P(CoT wrong, Direct correct).
    Returns math.inf when p10 == p01 (no detectable effect).
    """
    psi = p10 + p01
    delta = p10 - p01
    if delta == 0:
        return math.inf
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    n = (z_a * math.sqrt(psi) + z_b * math.sqrt(psi - delta**2)) ** 2 / delta**2
    return math.ceil(n)


def pilot_discordance(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Observed p10/p01 per model from cleaned pilot results."""
    out = {}
    for model, g in results.groupby("model_key"):
        wide = g.pivot(index="problem_id", columns="prompt_style", values="correct").dropna()
        cot, direct = wide["cot"].astype(bool), wide["direct"].astype(bool)
        out[str(model)] = {
            "n_pairs": int(len(wide)),
            "p10": float((cot & ~direct).mean()),
            "p01": float((~cot & direct).mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pilot", action="store_true", help="also recompute n from pilot discordance")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    p = cfg["power"]

    report: dict = {"planned": {**p, "n_required": connor_n(p["p10"], p["p01"], p["alpha"], p["power"])}}
    log.info("Planned: p10=%.3f p01=%.3f -> n=%s", p["p10"], p["p01"], report["planned"]["n_required"])

    if args.pilot:
        results = pd.read_csv(resolve(cfg, "pilot_clean"))
        report["pilot"] = {}
        for model, d in pilot_discordance(results).items():
            n = connor_n(d["p10"], d["p01"], p["alpha"], p["power"])
            report["pilot"][model] = {**d, "n_required": n if math.isfinite(n) else None}
            log.info("Pilot %s: p10=%.3f p01=%.3f (pairs=%d) -> n=%s",
                     model, d["p10"], d["p01"], d["n_pairs"], n)

    out_dir = resolve(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "power_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
