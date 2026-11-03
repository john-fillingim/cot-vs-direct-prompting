"""Tests for job construction and run order."""

from src.config import load_config
from src.run_experiment import build_jobs


def test_build_jobs_is_complete_blocked_and_deterministic() -> None:
    cfg = load_config()
    jobs = build_jobs(list(range(10)), cfg)
    assert len(jobs) == 10 * 2 * 2
    assert len({j.key for j in jobs}) == len(jobs)
    assert [j.model_key for j in jobs[:20]] == [cfg["model_order"][0]] * 20
    assert jobs == build_jobs(list(range(10)), cfg)
