"""Shared helpers: config loading, path resolution, logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Return an absolute path for `cfg['paths'][key]` (relative paths are from the repo root)."""
    return ROOT / cfg["paths"][key]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure timestamped console logging."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
