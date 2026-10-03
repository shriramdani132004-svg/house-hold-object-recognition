"""Metadata-level leakage and integrity checks for loaded scenarios.

All checks operate on filelist references (paths, labels, ids) — image bytes
are never read. Checks are variant-aware: incremental variants must partition
the training set, cumulative variants must form an official monotone chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data import core50
from src.data.continual.models import ContinualScenario


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one named safeguard."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ValidationReport:
    """Bundle of checks for one scenario run."""

    scenario_name: str
    run_id: int
    checks: tuple[CheckResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def failed(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "run": self.run_id,
            "all_passed": self.all_passed,
            "checks": [check.to_dict() for check in self.checks],
        }


def _paths(records) -> set[str]:
    return {record.relative_path for record in records}


def _check_ordering(scenario: ContinualScenario) -> CheckResult:
    ids = [exp.experience_id for exp in scenario.experiences]
    expected = list(range(len(scenario.experiences)))
    sources = [exp.train_source for exp in scenario.experiences]
    numbered = [f"train_batch_{i:02d}_filelist.txt" for i in expected]
    source_ok = all(
        src.endswith(name) for src, name in zip(sources, numbered)
    )
    passed = ids == expected and source_ok
    detail = (
        f"{len(ids)} experiences in official batch order"
        if passed
        else f"ordering mismatch: ids={ids} sources={sources[:3]}..."
    )
    return CheckResult("experience_ordering", passed, detail)


def _check_train_evaluation_overlap(scenario: ContinualScenario) -> CheckResult:
    overlap: set[str] = set()
    for exp in scenario.experiences:
        overlap |= _paths(exp.train_samples) & _paths(exp.evaluation_samples)
    passed = not overlap
    detail = (
        "no path appears in both training and evaluation"
        if passed
        else f"{len(overlap)} paths leaked between train and evaluation"
    )
    return CheckResult("train_evaluation_overlap", passed, detail)


def _check_evaluation_stable(scenario: ContinualScenario) -> CheckResult:
    if not scenario.experiences:
        return CheckResult("evaluation_stable", False, "scenario has no experiences")
    reference = scenario.experiences[0].evaluation_samples
    stable = all(exp.evaluation_samples is reference for exp in scenario.experiences)
    passed = stable and len(reference) > 0
    detail = (
        f"single fixed evaluation set of {len(reference)} samples shared by "
        "all experiences"
        if passed
        else "evaluation set differs between experiences"
    )
    return CheckResult("evaluation_stable", passed, detail)


def _check_future_leakage(scenario: ContinualScenario) -> CheckResult:
    """No experience may expose data introduced by a later experience.

    Incremental (partition) variants: every training batch must be disjoint
    from all earlier batches. Cumulative variants: the official chain must be
    monotone (``train_{j-1} subset of train_j``) and each batch's *new* data
    must be disjoint from everything exposed earlier.
    """
    cumulative = scenario.variant.lower().startswith("cum")
    exposed: set[str] = set()
    previous: set[str] = set()
    violations: list[str] = []
    for exp in scenario.experiences:
        current = _paths(exp.train_samples)
        introduced = current - previous if cumulative else current
        clash = introduced & exposed
        if clash:
            violations.append(
                f"experience {exp.experience_id}: {len(clash)} future/new paths "
                "already exposed earlier"
            )
        if cumulative and not previous <= current:
            missing = previous - current
            violations.append(
                f"experience {exp.experience_id}: breaks official cumulative "
                f"chain ({len(missing)} earlier paths missing)"
            )
        exposed |= introduced
        previous = current
    passed = not violations
    detail = (
        ("cumulative chain monotone; " if cumulative else "batches disjoint; ")
        + f"{len(exposed)} unique training paths exposed in order"
        if passed
        else "; ".join(violations[:3])
    )
    return CheckResult("future_training_leakage", passed, detail)


def _check_duplicates(scenario: ContinualScenario) -> CheckResult:
    problems: list[str] = []
    for exp in scenario.experiences:
        paths = [record.relative_path for record in exp.train_samples]
        if len(paths) != len(set(paths)):
            problems.append(f"experience {exp.experience_id} train duplicates")
    eval_paths = [
        record.relative_path
        for record in (scenario.experiences[0].evaluation_samples if scenario.experiences else ())
    ]
    if len(eval_paths) != len(set(eval_paths)):
        problems.append("evaluation duplicates")
    passed = not problems
    detail = (
        "no duplicate references within any single filelist"
        if passed
        else "; ".join(problems)
    )
    return CheckResult("duplicate_references", passed, detail)


def _check_path_resolution(
    scenario: ContinualScenario,
    images_root: Path,
) -> CheckResult:
    unique: set[str] = set()
    for exp in scenario.experiences:
        unique |= _paths(exp.train_samples)
    unique |= _paths(scenario.experiences[0].evaluation_samples)
    missing: list[str] = []
    for rel_path in sorted(unique):
        if not (images_root / rel_path).is_file():
            missing.append(rel_path)
            if len(missing) >= 5:
                break
    passed = not missing
    detail = (
        f"all {len(unique):,} unique referenced images exist on disk"
        if passed
        else f"unresolved paths (first {len(missing)}): {missing}"
    )
    return CheckResult("path_resolution", passed, detail)


def _check_object_integrity(scenario: ContinualScenario) -> CheckResult:
    known = set(core50.OBJECT_IDS)
    unknown = {
        record.object_id
        for exp in scenario.experiences
        for record in exp.train_samples
        if record.object_id not in known
    }
    if scenario.experiences:
        unknown |= {
            record.object_id
            for record in scenario.experiences[0].evaluation_samples
            if record.object_id not in known
        }
    passed = not unknown
    detail = (
        "all object ids within official o1..o50"
        if passed
        else f"unknown object ids: {sorted(unknown)}"
    )
    return CheckResult("object_integrity", passed, detail)


def _check_session_integrity(scenario: ContinualScenario) -> CheckResult:
    known = set(core50.SESSION_IDS)
    train_sessions: set[int] = set()
    eval_sessions: set[int] = set()
    unknown: set[int] = set()
    for exp in scenario.experiences:
        for record in exp.train_samples:
            train_sessions.add(record.session_id)
            if record.session_id not in known:
                unknown.add(record.session_id)
        for record in exp.evaluation_samples:
            eval_sessions.add(record.session_id)
            if record.session_id not in known:
                unknown.add(record.session_id)
    overlap = train_sessions & eval_sessions
    passed = not unknown and not overlap
    detail = (
        f"train sessions {sorted(train_sessions)} / evaluation sessions "
        f"{sorted(eval_sessions)} are valid and disjoint"
        if passed
        else f"unknown={sorted(unknown)} overlap={sorted(overlap)}"
    )
    return CheckResult("session_integrity", passed, detail)


def _check_label_integrity(scenario: ContinualScenario) -> CheckResult:
    max_label = max(core50.OBJECT_IDS) - 1
    bad = [
        record.label
        for exp in scenario.experiences
        for record in exp.train_samples
        if record.label < 0 or record.label > max_label
    ]
    passed = not bad
    detail = (
        f"all labels within official 0..{max_label}"
        if passed
        else f"{len(bad)} labels outside 0..{max_label}"
    )
    return CheckResult("label_integrity", passed, detail)


def check_development_split(
    dev_paths: set[str],
    scenario: ContinualScenario,
) -> CheckResult:
    """Development split must come ONLY from official training references.

    Guarantees the selection pipeline can never silently fall back to the
    held-out evaluation sessions: every dev path must be a training
    reference of some experience and none may appear in the fixed
    evaluation set (s3/s7/s10). Purely metadata-level — no image reads.
    """
    train_paths: set[str] = set()
    eval_paths: set[str] = set()
    for exp in scenario.experiences:
        train_paths |= _paths(exp.train_samples)
        eval_paths |= _paths(exp.evaluation_samples)
    leaked = sorted(dev_paths & eval_paths)
    unknown = sorted(dev_paths - train_paths)
    problems: list[str] = []
    if leaked:
        problems.append(
            f"{len(leaked)} development path(s) are official evaluation "
            f"references (first: {leaked[:3]})"
        )
    if unknown:
        problems.append(
            f"{len(unknown)} development path(s) are not training references "
            f"(first: {unknown[:3]})"
        )
    passed = not problems
    detail = (
        f"all {len(dev_paths)} development references are training-only and "
        "disjoint from the held-out evaluation sessions"
        if passed
        else "; ".join(problems)
    )
    return CheckResult("development_split_leakage", passed, detail)


def validate_scenario(
    scenario: ContinualScenario,
    *,
    check_paths_exist: bool = True,
    images_root: Path | None = None,
    on_progress=None,
) -> ValidationReport:
    """Run every metadata-level safeguard and return a structured report."""
    root = Path(images_root) if images_root else Path(scenario.images_root)
    checks: list[CheckResult] = []
    steps = [
        ("experience_ordering", lambda: _check_ordering(scenario)),
        ("train_evaluation_overlap", lambda: _check_train_evaluation_overlap(scenario)),
        ("evaluation_stable", lambda: _check_evaluation_stable(scenario)),
        ("future_training_leakage", lambda: _check_future_leakage(scenario)),
        ("duplicate_references", lambda: _check_duplicates(scenario)),
        ("object_integrity", lambda: _check_object_integrity(scenario)),
        ("session_integrity", lambda: _check_session_integrity(scenario)),
        ("label_integrity", lambda: _check_label_integrity(scenario)),
    ]
    if check_paths_exist:
        steps.append(("path_resolution", lambda: _check_path_resolution(scenario, root)))

    total = len(steps)
    for index, (_, fn) in enumerate(steps, start=1):
        if on_progress:
            on_progress("check", index, total)
        checks.append(fn())
    if on_progress:
        on_progress("check", total, total)

    return ValidationReport(
        scenario_name=scenario.name,
        run_id=scenario.run_id,
        checks=tuple(checks),
    )
