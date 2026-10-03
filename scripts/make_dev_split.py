"""Step 2 — deterministic train-only development-validation split.

Creates a 90/10 development split from official TRAINING references only
(sessions s3/s7/s10 evaluation data is never touched). The split is
stratified per experience and per class with a fixed seed, preserves the
official experience ordering, and is saved as an exact manifest under
data/splits/.

Usage:
    python scripts/make_dev_split.py
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.continual import (  # noqa: E402
    check_development_split,
    load_scenario_cached,
)

SEED = 42
FRACTION = 0.10
MANIFEST_PATH = PROJECT_ROOT / "data/splits/nic_inc_run0_dev10_seed42.json"


def split_experience(exp_id: int, samples) -> tuple[list, list]:
    """Return (train, dev) partitions for one experience."""
    by_class: dict[int, list] = {}
    for record in samples:
        by_class.setdefault(int(record.label), []).append(record)
    train: list = []
    dev: list = []
    for class_id in sorted(by_class):
        records = list(by_class[class_id])
        rng = random.Random(SEED * 1_000_003 + exp_id * 1_009 + class_id)
        rng.shuffle(records)
        n_dev = max(1, int(round(len(records) * FRACTION)))
        dev.extend(records[:n_dev])
        train.extend(records[n_dev:])
    return train, dev


def main() -> int:
    scenario = load_scenario_cached("NIC", "inc", 0)
    evaluation_paths = {
        record.relative_path
        for exp in scenario.iter_experiences()
        for record in exp.evaluation_samples
    }

    manifest_experiences = []
    all_train_paths: set[str] = set()
    all_dev_paths: set[str] = set()
    dev_class_coverage: set[int] = set()

    for exp in scenario.iter_experiences():
        train, dev = split_experience(exp.experience_id, exp.train_samples)
        train_paths = [r.relative_path for r in train]
        dev_paths = [r.relative_path for r in dev]
        overlap = set(train_paths) & set(dev_paths)
        if overlap:
            raise SystemExit(f"train/dev overlap in experience {exp.experience_id}")
        leak = set(dev_paths) & evaluation_paths
        if leak:
            raise SystemExit(f"evaluation data leaked into dev split: {sorted(leak)[:3]}")
        all_train_paths.update(train_paths)
        all_dev_paths.update(dev_paths)
        dev_class_coverage.update(int(r.label) for r in dev)
        manifest_experiences.append(
            {
                "experience_id": exp.experience_id,
                "n_train": len(train_paths),
                "n_dev": len(dev_paths),
                "dev_paths": dev_paths,
            }
        )

    total = len(all_train_paths) + len(all_dev_paths)
    if all_train_paths & all_dev_paths:
        raise SystemExit("global train/dev overlap detected")
    if all_dev_paths & evaluation_paths:
        raise SystemExit("evaluation reference found in dev split")
    if dev_class_coverage != set(range(50)):
        raise SystemExit(f"dev split misses classes: {set(range(50)) - dev_class_coverage}")

    leak = check_development_split(all_dev_paths, scenario)
    if not leak.passed:
        raise SystemExit(f"leakage validation failed: {leak.detail}")

    payload = {
        "format_version": 1,
        "seed": SEED,
        "fraction": FRACTION,
        "policy": (
            "stratified per experience and per class; seeded shuffle "
            "(seed*1000003 + experience_id*1009 + class_id); "
            "n_dev = round(0.10 * class_count), minimum 1"
        ),
        "scenario": "NIC",
        "variant": "inc",
        "run": 0,
        "official_evaluation_sessions": [3, 7, 10],
        "total_training_references": total,
        "total_train": len(all_train_paths),
        "total_dev": len(all_dev_paths),
        "dev_class_coverage": sorted(dev_class_coverage),
        "experiences": manifest_experiences,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    print(f"manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"sha256: {digest}")
    print(f"train refs: {len(all_train_paths)}  dev refs: {len(all_dev_paths)}")
    print(f"dev class coverage: {len(dev_class_coverage)}/50")
    print(f"leakage validation: {leak.detail}")
    print("evaluation-session leakage check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
