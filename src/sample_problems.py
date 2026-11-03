"""Draw the main and pilot problem samples from the GSM8K test split.

python -m src.sample_problems  ->  data/raw/problems.csv

One permutation of the test split (seed from config) gives the main sample
(first n_main) and a disjoint pilot sample (next n_pilot).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.config import DEFAULT_CONFIG, load_config, resolve, setup_logging

log = logging.getLogger(__name__)


def parse_gold(answer: str) -> float:
    """Gold numeric answer: the text after '####' in a GSM8K solution."""
    return float(answer.split("####")[-1].strip().replace(",", ""))


def count_steps(answer: str) -> int:
    """Number of non-empty solution lines before '####' (difficulty proxy)."""
    body = answer.split("####")[0]
    return sum(1 for line in body.splitlines() if line.strip())


def build_sample(test: pd.DataFrame, n_main: int, n_pilot: int, seed: int) -> pd.DataFrame:
    """Return main + pilot rows with ids, gold answers and step counts."""
    perm = np.random.default_rng(seed).permutation(len(test))
    idx = perm[: n_main + n_pilot]
    out = test.iloc[idx].reset_index().rename(columns={"index": "problem_id"})
    out["sample"] = ["main"] * n_main + ["pilot"] * n_pilot
    out["gold"] = out["answer"].map(parse_gold)
    out["n_steps"] = out["answer"].map(count_steps)
    return out[["problem_id", "sample", "question", "answer", "gold", "n_steps"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]

    out = resolve(cfg, "problems")
    if out.exists():
        log.info("%s already exists; raw data is never overwritten. Nothing to do.", out)
        return

    log.info("Loading %s (%s, %s)", ds_cfg["name"], ds_cfg["config"], ds_cfg["split"])
    test = load_dataset(ds_cfg["name"], ds_cfg["config"], split=ds_cfg["split"]).to_pandas()
    sample = build_sample(test, ds_cfg["n_main"], ds_cfg["n_pilot"], cfg["seed"])

    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    log.info("Wrote %d problems (%d main, %d pilot) to %s",
             len(sample), ds_cfg["n_main"], ds_cfg["n_pilot"], out)


if __name__ == "__main__":
    main()
