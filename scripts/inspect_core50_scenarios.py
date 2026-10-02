#!/usr/bin/env python
"""Lightweight inspection of an official CORe50 continual scenario.

Prints the experience structure (official batch order) plus leakage-check
results. Metadata/filelist based only — no image decoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.data.continual import (  # noqa: E402
    ContinualDataError,
    list_scenarios,
    load_scenario,
    validate_scenario,
)
from src.utils.progress import PhaseProgress  # noqa: E402

STEP_LABELS = [
    "Loading official scenario filelists",
    "Running leakage and integrity checks",
    "Rendering inspection report",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None, help="NI, NC or NIC")
    parser.add_argument("--variant", default=None, help="inc, cum, v2_79, ...")
    parser.add_argument("--run", type=int, default=None, help="official run id")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "continual.yaml"),
        help="YAML defaults (CLI flags override)",
    )
    parser.add_argument(
        "--no-path-check",
        action="store_true",
        help="skip on-disk existence check of referenced images",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list discovered official scenario variants and exit",
    )
    parser.add_argument(
        "--max-experiences",
        type=int,
        default=0,
        help="limit printed experiences (0 = all)",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def list_variants() -> int:
    variants = list_scenarios()
    print("=" * 60)
    print("OFFICIAL CORE50 SCENARIO VARIANTS")
    print("=" * 60)
    for variant in variants:
        print(f"  {variant.name:<14} type={variant.scenario_type} "
              f"variant={variant.variant} runs={list(variant.runs)}")
    print(f"Total: {len(variants)} variants")
    return 0


def print_scenario(scenario, max_experiences: int) -> None:
    print("=" * 60)
    print("CORe50 CONTINUAL SCENARIO")
    print("=" * 60)
    print()
    print(f"Scenario      : {scenario.scenario_type}")
    print(f"Run           : {scenario.run_id}")
    print(f"Variant       : {scenario.variant}")
    print(f"Experiences   : {len(scenario.experiences)}")
    print(f"Label space   : {scenario.metadata['label_min']}"
          f"..{scenario.metadata['label_max']} "
          f"(object-labels: {scenario.metadata['label_equals_object_minus_one']})")
    print()

    count = len(scenario.experiences)
    if max_experiences:
        count = min(count, max_experiences)
    for index in range(count):
        exp = scenario.get_experience(index)
        print(f"Experience {exp.experience_id + 1}")
        print("-" * 12)
        print(f"Train samples          : {exp.train_count}")
        print(f"Evaluation samples     : {exp.evaluation_count}")
        print(f"Introduced classes     : {list(exp.classes_introduced)}")
        print(f"Seen classes (cumul.)  : {list(exp.classes_seen)}")
        print(f"Introduced objects     : {list(exp.objects_introduced)}")
        print(f"Objects seen (cumul.)  : {list(exp.objects_seen)}")
        print(f"Sessions in experience : {list(exp.sessions_present)}")
        print(f"Sessions seen (cumul.) : {list(exp.sessions_seen)}")
        print(f"Official train source  : {exp.train_source}")
        print()
    if count < len(scenario.experiences):
        print(f"... ({len(scenario.experiences) - count} more experiences "
              f"not printed)")
        print()


def print_checks(report) -> None:
    print("=" * 60)
    print("LEAKAGE CHECK")
    print("=" * 60)
    print()
    labels = {
        "train_evaluation_overlap": "Train/evaluation overlap",
        "future_training_leakage": "Future-data leakage",
        "experience_ordering": "Ordering",
        "path_resolution": "Path resolution",
    }
    by_name = {check.name: check for check in report.checks}
    for name, label in labels.items():
        check = by_name.get(name)
        if check is None:
            status = "SKIPPED"
        else:
            status = "PASS" if check.passed else "FAIL"
        print(f"{label:<26}: {status}")
    for check in report.checks:
        if not check.passed:
            print(f"  ! {check.name}: {check.detail}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        return list_variants()

    config = load_config(Path(args.config))
    scenario_name = args.scenario or config.get("scenario", "NIC")
    variant = args.variant or config.get("variant")
    run = args.run if args.run is not None else int(config.get("run", 0))
    check_paths = not args.no_path_check and bool(
        config.get("check_paths_exist", True)
    )

    progress = PhaseProgress("PHASE 3 OVERALL", STEP_LABELS)
    try:
        progress.set_step(1)
        scenario = load_scenario(
            scenario_name,
            variant=variant,
            run=run,
            on_progress=lambda stage, current, total: progress.update(
                current,
                total,
                detail=f"{stage}: batch {current}/{total}",
                force=True,
            ),
        )
        progress.log(
            f"Loaded {scenario.name} run{scenario.run_id}: "
            f"{len(scenario.experiences)} experiences"
        )

        progress.set_step(2)
        report = validate_scenario(
            scenario,
            check_paths_exist=check_paths,
            on_progress=lambda stage, current, total: progress.update(
                current,
                total,
                detail=f"check {current}/{total}",
                force=True,
            ),
        )
        progress.log(
            "Checks: "
            + ", ".join(
                f"{c.name}={'PASS' if c.passed else 'FAIL'}"
                for c in report.checks
            )
        )

        progress.set_step(3)
        print_scenario(scenario, args.max_experiences)
        print_checks(report)
    except ContinualDataError as exc:
        progress.finish(f"ERROR: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    progress.finish("Inspection complete")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
