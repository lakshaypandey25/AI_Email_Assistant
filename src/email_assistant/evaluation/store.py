"""Benchmark result history (JSON file, last 20 runs)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

RESULTS_FILE = Path("data/eval_results.json")


def load_runs(path: Path | None = None) -> list[dict[str, Any]]:
    try:
        data = json.loads((path or RESULTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def save_run(run: dict[str, Any], path: Path | None = None, keep: int = 20) -> None:
    target = path or RESULTS_FILE
    runs = [*load_runs(target), run][-keep:]
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=2, default=str)
    os.replace(tmp, target)