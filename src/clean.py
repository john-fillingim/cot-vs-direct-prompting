"""Extract numeric answers from raw responses and score them.

python -m src.clean --pilot   ->  data/clean/pilot_results.csv
python -m src.clean           ->  data/clean/results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG, load_config, resolve, setup_logging

log = logging.getLogger(__name__)

# A minus sign counts only when it is not a hyphen between words/numbers (e.g. "3-4").
NUMBER_RE = re.compile(r"(?:(?<![\w)])-)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def extract_answer(text: str) -> float | None:
    """Number after the last '####' if present, otherwise the last number in the text.

    `$`, units and other text around the number are ignored; thousands commas are stripped.
    Returns None if no number can be found.
    """
    if not isinstance(text, str):
        return None
    if "####" in text:
        after = NUMBER_RE.search(text.rsplit("####", 1)[1])
        if after:
            return _to_float(after.group())
    numbers = NUMBER_RE.findall(text)
    return _to_float(numbers[-1]) if numbers else None


def is_correct(pred: float | None, gold: float) -> bool:
    """Numeric equality with a small tolerance; unparsed answers are incorrect."""
    return pred is not None and math.isclose(pred, gold, rel_tol=0, abs_tol=1e-6)


def read_jsonl(path: Path) -> pd.DataFrame:
    """Read valid JSON lines, skipping any truncated line."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed line in %s", path.name)
    return pd.DataFrame(rows)


def clean(responses: pd.DataFrame, problems: pd.DataFrame) -> pd.DataFrame:
    """Join responses to gold answers and score each one."""
    keys = ["problem_id", "model_key", "prompt_style"]
    n_dup = responses.duplicated(keys).sum()
    if n_dup:
        log.warning("Dropping %d duplicate responses (keeping first)", n_dup)
    df = responses.drop_duplicates(keys).merge(
        problems[["problem_id", "gold", "n_steps"]], on="problem_id", how="left", validate="many_to_one"
    )
    df["pred"] = df["response"].map(extract_answer)
    df["parse_failed"] = df["pred"].isna()
    df["correct"] = [is_correct(p, g) for p, g in zip(df["pred"], df["gold"], strict=True)]
    df["truncated"] = df["done_reason"].eq("length")
    df = df.rename(columns={"eval_count": "output_tokens"})
    cols = keys + ["model", "model_digest", "gold", "pred", "correct", "parse_failed", "truncated",
                   "output_tokens", "prompt_eval_count", "latency_s", "n_steps"]
    return df[cols].sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    src = resolve(cfg, "pilot_responses" if args.pilot else "responses")
    out = resolve(cfg, "pilot_clean" if args.pilot else "clean")
    results = clean(read_jsonl(src), pd.read_csv(resolve(cfg, "problems")))

    n_expected = cfg["dataset"]["n_pilot" if args.pilot else "n_main"] * len(cfg["models"]) * len(
        cfg["prompt_styles"])
    if len(results) != n_expected:
        log.warning("Expected %d responses, found %d", n_expected, len(results))

    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(results), out)

    summary = results.groupby(["model_key", "prompt_style"]).agg(
        n=("correct", "size"), accuracy=("correct", "mean"), parse_failed=("parse_failed", "sum"),
        truncated=("truncated", "sum"), median_tokens=("output_tokens", "median"))
    log.info("Summary:\n%s", summary.round(3).to_string())


if __name__ == "__main__":
    main()
