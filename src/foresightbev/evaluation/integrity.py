"""Guards on experiment validity: seed separation, overwrite protection, records."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from foresightbev.errors import ConfigurationError, JobConflict

# Seeds used by the trained forecaster. Half-open ranges.
FORECASTER_SEEDS = {"forecaster_train": (300, 330), "forecaster_val": (330, 336), "forecaster_test": (336, 346)}
# Seeds looked at while tuning thresholds. A final result must not reuse them.
DEVELOPMENT_SEEDS = (100, 110)


def seed_ranges(config: dict) -> dict[str, tuple[int, int]]:
    ranges = {"closed_loop": tuple(config["closed_loop_seeds"]), "calibration_fit": tuple(config["calibration"]["fit_seeds"]),
              "calibration_test": tuple(config["calibration"]["test_seeds"])}
    return ranges | FORECASTER_SEEDS


def check_seed_separation(config: dict, *, final: bool = False) -> dict[str, tuple[int, int]]:
    """Fail if any two splits share a seed, or a final run reuses development seeds."""
    ranges = seed_ranges(config)
    for name, (low, high) in ranges.items():
        if low >= high:
            raise ConfigurationError(f"seed range {name} is empty: {low}..{high}")
    names = list(ranges)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ranges[a][0] < ranges[b][1] and ranges[b][0] < ranges[a][1]:
                raise ConfigurationError(f"seed leakage: {a} {ranges[a]} overlaps {b} {ranges[b]}")
    closed = ranges["closed_loop"]
    if final and closed[0] < DEVELOPMENT_SEEDS[1] and DEVELOPMENT_SEEDS[0] < closed[1]:
        raise ConfigurationError(f"final runs must not use development seeds {DEVELOPMENT_SEEDS}")
    return ranges


def claim_result_directory(directory: Path, force: bool) -> None:
    """Refuse to overwrite stored results unless explicitly forced."""
    if (directory / "summary.json").exists() and not force:
        raise JobConflict(f"results already exist at {directory}. Choose another --name; existing results are not overwritten.")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def append_record(data_dir: Path, record: dict) -> None:
    """Append-only index of every experiment run."""
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "experiments.jsonl").open("a") as stream:
        stream.write(json.dumps(record | {"recorded_unix": time.time()}) + "\n")
