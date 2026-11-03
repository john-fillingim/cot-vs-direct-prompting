"""Generate model responses for every problem x model x prompt style.

python -m src.run_experiment --pilot   ->  data/raw/pilot_responses.jsonl
python -m src.run_experiment           ->  data/raw/responses.jsonl

Resumable: jobs already in the output file are skipped. Each response is
appended as one JSON line and flushed to disk immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import DEFAULT_CONFIG, load_config, resolve, setup_logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    problem_id: int
    model_key: str
    prompt_style: str

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.problem_id, self.model_key, self.prompt_style)


def build_jobs(problem_ids: list[int], cfg: dict[str, Any]) -> list[Job]:
    """All jobs, grouped in model blocks; problem x style order is shuffled within each block."""
    rng = random.Random(cfg["seed"])
    jobs: list[Job] = []
    for model_key in cfg["model_order"]:
        block = [Job(pid, model_key, style) for pid in problem_ids for style in cfg["prompt_styles"]]
        rng.shuffle(block)
        jobs.extend(block)
    return jobs


def prompt_hash(cfg: dict[str, Any], style: str) -> str:
    """Short hash of the system message + template, to detect prompt changes between runs."""
    text = cfg["prompts"]["system"] + "\n---\n" + cfg["prompts"][style]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_done(path: Path) -> dict[tuple[int, str, str], str]:
    """Map job key -> prompt hash for every valid line already written."""
    done: dict[tuple[int, str, str], str] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed line %d in %s", lineno, path.name)
                continue
            done[(r["problem_id"], r["model_key"], r["prompt_style"])] = r["prompt_hash"]
    return done


def verify_digests(cfg: dict[str, Any]) -> dict[str, str]:
    """Check installed model digests against the pinned ones; return model_key -> digest."""
    resp = requests.get(f"{cfg['ollama']['host']}/api/tags", timeout=30)
    resp.raise_for_status()
    installed = {m["name"]: m["digest"] for m in resp.json()["models"]}
    digests = {}
    for key, m in cfg["models"].items():
        got = installed.get(m["name"])
        if got is None:
            raise SystemExit(f"Model {m['name']} is not installed. Run: ollama pull {m['name']}")
        if got != m["digest"]:
            raise SystemExit(f"Digest mismatch for {m['name']}: config {m['digest']}, installed {got}")
        digests[key] = got
    return digests


def generate(cfg: dict[str, Any], model: str, user_msg: str) -> tuple[dict[str, Any], float]:
    """Call Ollama /api/chat with retries; return (response json, wall-clock latency in s)."""
    o, g = cfg["ollama"], cfg["generation"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": cfg["prompts"]["system"]},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "keep_alive": o["keep_alive"],
        "options": {k: g[k] for k in ("temperature", "seed", "num_predict", "num_ctx")},
    }
    for attempt in range(1, o["max_retries"] + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.post(f"{o['host']}/api/chat", json=payload, timeout=o["timeout_s"])
            resp.raise_for_status()
            return resp.json(), time.perf_counter() - t0
        except requests.RequestException as e:
            if attempt == o["max_retries"]:
                raise
            wait = 5 * 2**attempt
            log.warning("Request failed (%s); retry %d in %ds", e, attempt, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pilot", action="store_true", help="run the pilot sample")
    parser.add_argument("--limit", type=int, default=None, help="stop after N new jobs (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="show the job count and exit")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    problems = pd.read_csv(resolve(cfg, "problems"))
    problems = problems[problems["sample"] == ("pilot" if args.pilot else "main")]
    questions = dict(zip(problems["problem_id"], problems["question"], strict=True))
    out_path = resolve(cfg, "pilot_responses" if args.pilot else "responses")

    jobs = build_jobs(problems["problem_id"].tolist(), cfg)
    done = load_done(out_path)
    hashes = {s: prompt_hash(cfg, s) for s in cfg["prompt_styles"]}
    stale = {k for k, h in done.items() if h != hashes[k[2]]}
    if stale:
        raise SystemExit(f"{len(stale)} existing responses in {out_path.name} used different prompts. "
                         "Raw data is never overwritten; use a new output file.")
    todo = [j for j in jobs if j.key not in done]
    log.info("%d jobs total, %d done, %d to run -> %s", len(jobs), len(jobs) - len(todo), len(todo), out_path)
    if args.limit is not None:
        todo = todo[: args.limit]
    if args.dry_run or not todo:
        return

    digests = verify_digests(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for i, job in enumerate(todo, 1):
            model = cfg["models"][job.model_key]["name"]
            user_msg = cfg["prompts"][job.prompt_style].format(question=questions[job.problem_id])
            r, latency = generate(cfg, model, user_msg)
            record = {
                "problem_id": job.problem_id,
                "model_key": job.model_key,
                "model": model,
                "model_digest": digests[job.model_key],
                "prompt_style": job.prompt_style,
                "prompt_hash": hashes[job.prompt_style],
                "response": r["message"]["content"],
                "eval_count": r.get("eval_count"),
                "prompt_eval_count": r.get("prompt_eval_count"),
                "done_reason": r.get("done_reason"),
                "latency_s": round(latency, 3),
                "total_duration_ns": r.get("total_duration"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

            elapsed = time.perf_counter() - t_start
            eta_min = elapsed / i * (len(todo) - i) / 60
            log.info("[%d/%d] %s %-6s pid=%d tokens=%s %.1fs | ETA %.0f min",
                     i, len(todo), job.model_key, job.prompt_style, job.problem_id,
                     record["eval_count"], latency, eta_min)
    log.info("Done.")


if __name__ == "__main__":
    main()
