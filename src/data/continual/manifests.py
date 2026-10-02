"""Build and persist Phase-3 pipeline manifests (JSON, project-relative)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.data import core50
from src.data.continual.models import ContinualScenario
from src.data.continual.validation import ValidationReport


def _experience_entry(scenario: ContinualScenario, index: int) -> dict[str, Any]:
    exp = scenario.get_experience(index)
    return {
        "experience_id": exp.experience_id,
        "train_source": exp.train_source,
        "evaluation_source": exp.evaluation_source,
        "train_count": exp.train_count,
        "evaluation_count": exp.evaluation_count,
        "classes_introduced": list(exp.classes_introduced),
        "classes_seen": list(exp.classes_seen),
        "objects_introduced": list(exp.objects_introduced),
        "objects_seen": list(exp.objects_seen),
        "categories_introduced": list(exp.categories_introduced),
        "categories_seen": list(exp.categories_seen),
        "sessions_present": list(exp.sessions_present),
        "sessions_seen": list(exp.sessions_seen),
        "first_train_sample": exp.train_samples[0].to_dict()
        if exp.train_samples
        else None,
        "last_train_sample": exp.train_samples[-1].to_dict()
        if exp.train_samples
        else None,
    }


def build_scenario_manifest(
    scenario: ContinualScenario,
    report: ValidationReport | None = None,
) -> dict[str, Any]:
    """Manifest for a single loaded scenario run (fully JSON-serializable)."""
    entry: dict[str, Any] = {
        "scenario": scenario.scenario_type,
        "variant": scenario.variant,
        "name": scenario.name,
        "run": scenario.run_id,
        "experiences_available": list(scenario.metadata.get("runs_available", [])),
        "experience_count": len(scenario.experiences),
        "metadata": dict(scenario.metadata),
        "experiences": [
            _experience_entry(scenario, i) for i in range(len(scenario.experiences))
        ],
    }
    if report is not None:
        entry["validation"] = report.to_dict()
        unresolved = [
            check.detail
            for check in report.checks
            if check.name == "path_resolution" and not check.passed
        ]
        duplicates = next(
            (
                check.detail
                for check in report.checks
                if check.name == "duplicate_references"
            ),
            None,
        )
        leakage = [
            check.to_dict()
            for check in report.checks
            if check.name
            in ("train_evaluation_overlap", "future_training_leakage")
        ]
        entry["unresolved_references"] = unresolved or []
        entry["duplicate_reference_results"] = duplicates
        entry["leakage_check_results"] = leakage
    return entry


def build_manifest(
    scenarios: Iterable[ContinualScenario],
    reports: dict[str, ValidationReport] | None = None,
    *,
    requested_run: int | None = None,
    check_paths_exist: bool | None = None,
) -> dict[str, Any]:
    """Aggregate manifest across several loaded scenario runs."""
    reports = reports or {}
    scenario_list = list(scenarios)
    entries = [
        build_scenario_manifest(scenario, reports.get(scenario.name))
        for scenario in scenario_list
    ]
    return {
        "phase": 3,
        "title": "Continual data pipeline manifest",
        "dataset": "CORe50",
        "dataset_root": (core50.DATASET_DIR / "core50_128x128").relative_to(
            core50.PROJECT_ROOT
        ).as_posix(),
        "official_source": core50.OFFICIAL_PAGE,
        "official_repository": core50.OFFICIAL_REPO,
        "supported_scenarios": ["NI", "NC", "NIC"],
        "primary_future_scenario": "NIC",
        "image_copies_created": 0,
        "dataset_modified": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_selection": {
            "requested_run": requested_run,
            "runs": {scenario.name: scenario.run_id for scenario in scenario_list},
            "variants": len(scenario_list),
            "path_check": check_paths_exist,
            "all_checks_passed": all(
                reports[name].all_passed for name in reports
            )
            and len(reports) == len(scenario_list),
        },
        "scenarios": entries,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write the manifest as pretty JSON (deterministic key order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path
