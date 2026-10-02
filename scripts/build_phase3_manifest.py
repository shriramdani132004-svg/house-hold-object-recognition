#!/usr/bin/env python
"""Build the Phase-3 continual pipeline manifest (all official variants).

Loads every discovered official scenario variant for a single run (default
run 0), runs the metadata-level leakage checks, and writes a JSON manifest
with project-relative references only. No image decoding, no copies.
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
    build_manifest,
    list_scenarios,
    load_scenario,
    validate_scenario,
    write_manifest,
)
from src.utils.progress import PhaseProgress  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="official run per variant (default: config run, else 0)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="restrict to specific variant names (default: all discovered)",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "continual.yaml"),
        help="YAML configuration file",
    )
    parser.add_argument(
        "--no-path-check",
        action="store_true",
        help="skip on-disk existence check of referenced images",
    )
    parser.add_argument("--output", default=None, help="manifest output path")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    run = args.run if args.run is not None else int(config.get("run", 0))
    check_paths = not args.no_path_check and bool(
        config.get("check_paths_exist", True)
    )
    output = args.output or config.get(
        "manifest", "reports/phase3_continual_pipeline_manifest.json"
    )

    variants = list_scenarios()
    if args.variants:
        wanted = set(args.variants)
        variants = tuple(v for v in variants if v.name in wanted)
        missing = wanted - {v.name for v in variants}
        if missing:
            print(f"ERROR: unknown variants: {sorted(missing)}", file=sys.stderr)
            return 2
    if not variants:
        print("ERROR: no official scenario variants discovered", file=sys.stderr)
        return 2

    step_labels = [f"Loading {v.name}" for v in variants] + [
        "Writing manifest",
        "Complete",
    ]
    progress = PhaseProgress("PHASE 3 OVERALL", step_labels)

    scenarios = []
    reports = {}
    failures: list[str] = []
    for index, variant in enumerate(variants, start=1):
        progress.set_step(index, f"Loading {variant.name} (run {run})")
        try:
            chosen_run = run if run in variant.runs else variant.runs[0]
            scenario = load_scenario(
                variant.scenario_type,
                variant=variant.variant,
                run=chosen_run,
                on_progress=lambda stage, current, total: progress.update(
                    current,
                    total,
                    detail=f"{stage} {current}/{total}",
                    force=True,
                ),
            )
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
        except ContinualDataError as exc:
            failures.append(f"{variant.name}: {exc}")
            progress.log(f"SKIP {variant.name}: {exc}")
            continue
        scenarios.append(scenario)
        reports[scenario.name] = report
        status = "PASS" if report.all_passed else "FAIL"
        progress.log(
            f"{scenario.name}: {len(scenario.experiences)} experiences, "
            f"{scenario.metadata['train_samples_total']:,} train refs, "
            f"checks={status}"
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2

    progress.set_step(len(variants) + 1, "Writing manifest")
    manifest = build_manifest(
        scenarios,
        reports,
        requested_run=run,
        check_paths_exist=check_paths,
    )
    written = write_manifest(manifest, PROJECT_ROOT / output)
    progress.set_step(len(variants) + 2, "Complete")
    progress.finish(f"Manifest written: {output}")

    failed_checks = [
        f"{name}:{check.name}"
        for name, report in reports.items()
        for check in report.failed()
    ]
    if failed_checks:
        print(f"FAILED CHECKS: {failed_checks}", file=sys.stderr)
        return 1
    print(f"Manifest: {written.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Scenarios: {len(scenarios)} | run: {run} | all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
