"""Training-readiness gate — run BEFORE any training launch.

Sixteen blocking checks across import health, data guards, code hygiene,
numerical safety, checkpoint contracts, and the frozen baseline. The gate
passes only when every check passes; otherwise the correct answer is

    TRAINING READINESS = NOT READY

and training must not start. The script performs NO training: it only
loads metadata, runs tiny in-memory probes, and hashes files.

Usage:
    python scripts/verify_training_readiness.py
Exit code: 0 = all checks passed, 1 = at least one failed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_BASELINE_SHA256 = (
    "b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351"
)
BASELINE_MODEL = PROJECT_ROOT / "models/continual/final_model.pt"
DEV_MANIFEST = PROJECT_ROOT / "data/splits/nic_inc_run0_dev10_seed42.json"

SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml"}
SCAN_DIRS = ("src", "app", "configs", "scripts")
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"""(?i)(api_key|password|secret|token)\s*[=:]\s*["'][^"'\s]{8,}["']"""),
)
SECRET_ALLOWLIST = re.compile(
    r"""(?i)(environ|getenv|os\.env|\.env|example|placeholder|dummy|dummy_secret)"""
)
ABSOLUTE_PATH = re.compile(
    r"""["'](?:[A-Za-z]:[\\/]|/home/|/Users/)[A-Za-z0-9_. -]"""
)


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str


def _iter_scanned_files():
    for dirname in SCAN_DIRS:
        root = PROJECT_ROOT / dirname
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix in SCAN_SUFFIXES and path.is_file():
                yield path


def check_imports() -> CheckOutcome:
    modules = [
        "src.data",
        "src.data.continual",
        "src.data.preprocessing",
        "src.evaluation",
        "src.evaluation.continual",
        "src.evaluation.metrics",
        "src.training",
        "src.training.improved",
        "src.training.early_stop",
        "src.training.checkpoint_schema",
        "src.utils.run_logging",
    ]
    missing = []
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    return CheckOutcome(
        "imports",
        not missing,
        f"{len(modules)} modules import cleanly" if not missing else "; ".join(missing),
    )


def check_class_mapping() -> CheckOutcome:
    from src.data.class_mapping import ClassMappingError, load_class_mapping

    try:
        mapping = load_class_mapping()
    except (ClassMappingError, FileNotFoundError) as exc:
        return CheckOutcome("class_mapping", False, str(exc))
    names = list(mapping.names)
    problems = []
    if len(names) != 50:
        problems.append(f"expected 50 classes, got {len(names)}")
    if len(set(names)) != len(names):
        problems.append("duplicate class names")
    if tuple(mapping.labels) != tuple(range(50)):
        problems.append("class indices are not exactly 0..49")
    if problems:
        return CheckOutcome("class_mapping", False, "; ".join(problems))
    return CheckOutcome("class_mapping", True, "strict 50-class mapping loads coherently")


def check_no_absolute_paths() -> CheckOutcome:
    offenders = []
    for path in _iter_scanned_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ABSOLUTE_PATH.search(line):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}")
                break
    if offenders:
        return CheckOutcome(
            "no_absolute_paths", False, f"{len(offenders)} file(s): {offenders[:5]}"
        )
    return CheckOutcome("no_absolute_paths", True, "no machine-specific paths in code/configs")


def check_no_secrets() -> CheckOutcome:
    offenders = []
    for path in _iter_scanned_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if SECRET_ALLOWLIST.search(match.group(0)):
                    continue
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {match.group(0)[:24]}")
                break
    if offenders:
        return CheckOutcome("no_secrets", False, f"{len(offenders)} hit(s): {offenders[:5]}")
    return CheckOutcome("no_secrets", True, "no credential-like literals in code/configs")


def check_config_validation() -> CheckOutcome:
    from src.training.config import ContinualTrainConfig

    failures = []
    for bad, needle in (
        ({"learning_rate": 0.001, "totally_unknown_key": 1}, "totally_unknown_key"),
        ({"batch_size": 0}, "batch_size"),
        ({"early_stop_patience": 0}, "early_stop_patience"),
        ({"learning_rate": -1.0}, "learning_rate"),
        ({"image_size": 0}, "image_size"),
    ):
        try:
            ContinualTrainConfig.from_dict(bad)
        except Exception as exc:  # noqa: BLE001
            if needle not in str(exc):
                failures.append(f"{bad} -> error lacks {needle!r}: {exc}")
        else:
            failures.append(f"{bad} was accepted")
    if failures:
        return CheckOutcome("config_validation", False, "; ".join(failures))
    return CheckOutcome("config_validation", True, "invalid configs rejected with named keys")


def check_preprocessing_single_source() -> CheckOutcome:
    offenders = []
    own_path = Path(__file__).resolve()
    for path in _iter_scanned_files():
        if path.resolve() == (PROJECT_ROOT / "src/data/preprocessing.py").resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*(MEAN|STD)\s*=\s*\(", text, re.MULTILINE):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
        if path.resolve() != own_path and "Resampling.BILINEAR" in text:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)} (interpolation)")
    if offenders:
        return CheckOutcome(
            "preprocessing_single_source", False, f"redefined outside SSOT: {offenders}"
        )
    consumers = [
        "src/training/dataset.py",
        "src/training/tensor_cache.py",
        "src/inference/preprocessing.py",
        "src/training/improved.py",
    ]
    not_shared = [
        name
        for name in consumers
        if "data.preprocessing" not in (PROJECT_ROOT / name).read_text(encoding="utf-8")
    ]
    if not_shared:
        return CheckOutcome(
            "preprocessing_single_source", False, f"not using shared pipeline: {not_shared}"
        )
    return CheckOutcome(
        "preprocessing_single_source", True, "one implementation, 4 consumers wired to it"
    )


def check_dev_manifest_integrity() -> CheckOutcome:
    if not DEV_MANIFEST.is_file():
        return CheckOutcome(
            "dev_manifest",
            False,
            "missing data/splits/nic_inc_run0_dev10_seed42.json — run scripts/make_dev_split.py",
        )
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    problems = []
    experiences = payload.get("experiences", [])
    if len(experiences) != 79:
        problems.append(f"expected 79 experiences, got {len(experiences)}")
    total_dev = payload.get("total_dev")
    summed = sum(len(entry.get("dev_paths", [])) for entry in experiences)
    if total_dev != summed:
        problems.append(f"total_dev {total_dev} != sum of per-experience dev {summed}")
    if payload.get("official_evaluation_sessions") != [3, 7, 10]:
        problems.append("official evaluation sessions must be recorded as [3, 7, 10]")
    digest = hashlib.sha256(DEV_MANIFEST.read_bytes()).hexdigest()
    if len(digest) != 64:
        problems.append("sha256 unreadable")
    if problems:
        return CheckOutcome("dev_manifest", False, "; ".join(problems))
    return CheckOutcome(
        "dev_manifest",
        True,
        f"internally consistent, {summed} dev refs, sha256={digest[:12]}",
    )


def check_dev_leakage() -> CheckOutcome:
    from src.data.continual import check_development_split, load_scenario_cached

    if not DEV_MANIFEST.is_file():
        return CheckOutcome("dev_leakage", False, "manifest missing")
    payload = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    dev_paths = {
        path for entry in payload["experiences"] for path in entry.get("dev_paths", [])
    }
    scenario = load_scenario_cached("NIC", "inc", 0)
    result = check_development_split(dev_paths, scenario)
    return CheckOutcome("dev_leakage", result.passed, result.detail)


def check_selection_guards() -> CheckOutcome:
    source = (PROJECT_ROOT / "scripts/train_candidates.py").read_text(encoding="utf-8")
    problems = []
    if "check_development_split" not in source:
        problems.append("train_candidates does not call check_development_split")
    if "create it first: python scripts/make_dev_split.py" not in source:
        problems.append("missing explicit missing-manifest error message")
    spec = importlib.util.spec_from_file_location(
        "_train_candidates_probe", PROJECT_ROOT / "scripts/train_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
        try:
            module.dev_manifest_paths(PROJECT_ROOT / "data/splits/does_not_exist.json")
        except SystemExit:
            pass
        else:
            problems.append("missing manifest did not abort")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"module import failed: {exc}")
    if problems:
        return CheckOutcome("selection_guards", False, "; ".join(problems))
    return CheckOutcome(
        "selection_guards", True, "manifest-missing aborts; leakage check wired in"
    )


def check_early_stop_wiring() -> CheckOutcome:
    from src.training.base import BaseContinualTrainer
    from src.training.early_stop import EarlyStopping
    from src.training.improved import ImprovedReplayTrainer

    problems = []
    if not issubclass(ImprovedReplayTrainer, BaseContinualTrainer):
        problems.append("trainer does not extend the base class")
    if ImprovedReplayTrainer._finalize_experience_weights is BaseContinualTrainer._finalize_experience_weights:
        problems.append("best-epoch restore hook not overridden")
    if ImprovedReplayTrainer._on_experience_start is BaseContinualTrainer._on_experience_start:
        problems.append("per-experience stopper hook not overridden")
    stopper = EarlyStopping(patience=2)
    first = stopper.observe(0.5, 0)
    second = stopper.observe(0.4, 1)
    third = stopper.observe(0.3, 2)
    if (first, second, third) != (False, False, True):
        problems.append(f"patience semantics wrong: {(first, second, third)}")
    if stopper.best_value != 0.5 or stopper.best_epoch != 0:
        problems.append("best-epoch tracking wrong")
    if not stopper.should_restore:
        problems.append("should_restore false after a best exists")
    if problems:
        return CheckOutcome("early_stop_wiring", False, "; ".join(problems))
    return CheckOutcome(
        "early_stop_wiring", True, "stopper hooks overridden; patience/restore semantics verified"
    )


def check_nonfinite_guards() -> CheckOutcome:
    import torch

    from src.training.base import BaseContinualTrainer, ContinualTrainingError

    problems = []
    probe = BaseContinualTrainer.__new__(BaseContinualTrainer)
    for value, label in ((float("nan"), "NaN"), (float("inf"), "Inf")):
        try:
            BaseContinualTrainer._assert_finite_loss(
                torch.tensor([value]), experience_id=0, epoch=0, step=0
            )
        except ContinualTrainingError:
            pass
        else:
            problems.append(f"{label} loss was not rejected")
    model = torch.nn.Linear(2, 2)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    probe._model = model
    try:
        BaseContinualTrainer._assert_finite_gradients(
            probe, experience_id=0, epoch=0, step=0
        )
    except ContinualTrainingError:
        pass
    else:
        problems.append("inf gradients were not rejected")
    if problems:
        return CheckOutcome("nonfinite_guards", False, "; ".join(problems))
    return CheckOutcome("nonfinite_guards", True, "NaN/Inf loss and gradients abort loudly")


def check_baseline_checkpoint() -> CheckOutcome:
    import torch

    from src.training.checkpoint_schema import (
        CheckpointSchemaError,
        load_final_model,
        validate_checkpoint_schema,
    )

    if not BASELINE_MODEL.is_file():
        return CheckOutcome("baseline_checkpoint", False, "final_model.pt missing")
    digest = hashlib.sha256(BASELINE_MODEL.read_bytes()).hexdigest()
    if digest != EXPECTED_BASELINE_SHA256:
        # Post-freeze state: the final model is the frozen run's artifact;
        # the historical baseline must live on in the preservation copy and
        # be pinned by the frozen payload's provenance.
        preserved = PROJECT_ROOT / "models/continual/baseline_replay_phase5.pt"
        if not preserved.is_file():
            return CheckOutcome(
                "baseline_checkpoint",
                False,
                "final_model.pt no longer matches the pinned baseline and the "
                "preservation copy baseline_replay_phase5.pt is missing",
            )
        preserved_digest = hashlib.sha256(preserved.read_bytes()).hexdigest()
        if preserved_digest != EXPECTED_BASELINE_SHA256:
            return CheckOutcome(
                "baseline_checkpoint",
                False,
                "preservation copy changed: sha256="
                f"{preserved_digest} expected {EXPECTED_BASELINE_SHA256}",
            )
        payload = torch.load(BASELINE_MODEL, map_location="cpu", weights_only=False)
        provenance = payload.get("provenance") if isinstance(payload, dict) else None
        pinned = provenance.get("baseline_sha256") if isinstance(provenance, dict) else None
        if pinned != EXPECTED_BASELINE_SHA256:
            return CheckOutcome(
                "baseline_checkpoint",
                False,
                "frozen final_model.pt does not pin the historical baseline "
                f"in provenance (got {pinned!r})",
            )
        problems = validate_checkpoint_schema(BASELINE_MODEL, expected_num_classes=50)
        if problems:
            return CheckOutcome("baseline_checkpoint", False, "; ".join(problems))
        try:
            model, schema = load_final_model()
        except CheckpointSchemaError as exc:
            return CheckOutcome("baseline_checkpoint", False, f"load failed: {exc}")
        if int(schema.get("num_classes", 0)) != 50:
            return CheckOutcome("baseline_checkpoint", False, "num_classes != 50 in schema")
        return CheckOutcome(
            "baseline_checkpoint",
            True,
            "frozen final model; baseline preserved at baseline_replay_phase5.pt "
            f"({preserved_digest[:12]}); provenance pinned; schema+contract valid",
        )
    problems = validate_checkpoint_schema(BASELINE_MODEL, expected_num_classes=50)
    if problems:
        return CheckOutcome("baseline_checkpoint", False, "; ".join(problems))
    try:
        model, schema = load_final_model()
    except CheckpointSchemaError as exc:
        return CheckOutcome("baseline_checkpoint", False, f"load failed: {exc}")
    if int(schema.get("num_classes", 0)) != 50:
        return CheckOutcome("baseline_checkpoint", False, "num_classes != 50 in schema")
    return CheckOutcome(
        "baseline_checkpoint",
        True,
        f"sha256={digest[:12]} matches frozen baseline; schema+contract valid",
    )


def check_metrics_spot_check() -> CheckOutcome:
    from src.evaluation.metrics import (
        accuracy_from_confusion,
        average_incremental_accuracy,
        confusion_matrix,
        precision_recall_f1,
    )

    matrix = confusion_matrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], 3)
    accuracy = accuracy_from_confusion(matrix)
    macro = precision_recall_f1(matrix)["macro"]
    ok = (
        abs(accuracy - 4 / 6) < 1e-9
        and abs(macro["precision"] - 13 / 18) < 1e-9
        and abs(macro["recall"] - 2 / 3) < 1e-9
        and abs(average_incremental_accuracy([0.5, None, 0.75, 0.25]) - 0.5) < 1e-9
    )
    return CheckOutcome(
        "metrics_spot_check",
        ok,
        "hand-computed reference values reproduce" if ok else "reference values mismatch",
    )


def check_structured_logging() -> CheckOutcome:
    from src.utils.run_logging import format_event

    line = format_event(
        event="dev_epoch", experience=3, epoch=2, dev_accuracy=0.5, best_dev=0.6
    )
    expected = "event=dev_epoch experience=3 epoch=2 dev_accuracy=0.5 best_dev=0.6"
    ok = line == expected
    return CheckOutcome(
        "structured_logging",
        ok,
        "deterministic key=value formatting" if ok else f"got {line!r}",
    )


def check_dead_code_removed() -> CheckOutcome:
    problems = []
    config_source = (PROJECT_ROOT / "src/training/config.py").read_text(encoding="utf-8")
    if "tensor_cache_dir" in config_source:
        problems.append("config.tensor_cache_dir still present")
    if "def from_yaml" in config_source:
        problems.append("ContinualTrainConfig.from_yaml still present")
    phase5 = (PROJECT_ROOT / "scripts/run_phase5_experiment.py").read_text(encoding="utf-8")
    if "def parse_train_done" in phase5:
        problems.append("run_phase5_experiment.parse_train_done still present")
    improved = (PROJECT_ROOT / "src/training/improved.py").read_text(encoding="utf-8")
    if re.search(r"^\s*def _normalize\s*\(", improved, re.MULTILINE):
        problems.append("duplicate _normalize still in improved.py")
    return CheckOutcome(
        "dead_code_removed",
        not problems,
        "dead helpers removed" if not problems else "; ".join(problems),
    )


def check_authoritative_class_names() -> CheckOutcome:
    source = (PROJECT_ROOT / "scripts/run_phase6_analysis.py").read_text(encoding="utf-8")
    problems = []
    if "def merge_authoritative_class_names" not in source:
        problems.append("merge_authoritative_class_names missing")
    if source.count("merge_authoritative_class_names(") < 2:
        problems.append("not applied at both report/classification sites")
    return CheckOutcome(
        "class_names_precedence",
        not problems,
        "mapping-won precedence enforced" if not problems else "; ".join(problems),
    )


CHECKS = (
    check_imports,
    check_class_mapping,
    check_no_absolute_paths,
    check_no_secrets,
    check_config_validation,
    check_preprocessing_single_source,
    check_dev_manifest_integrity,
    check_dev_leakage,
    check_selection_guards,
    check_early_stop_wiring,
    check_nonfinite_guards,
    check_baseline_checkpoint,
    check_metrics_spot_check,
    check_structured_logging,
    check_dead_code_removed,
    check_authoritative_class_names,
)


def main() -> int:
    if len(CHECKS) != 16:
        print(f"INTERNAL: expected 16 checks, have {len(CHECKS)}", file=sys.stderr)
        return 1
    outcomes: list[CheckOutcome] = []
    for function in CHECKS:
        try:
            outcome = function()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            outcome = CheckOutcome(function.__name__, False, f"crashed: {type(exc).__name__}: {exc}")
        outcomes.append(outcome)
        status = "PASS" if outcome.passed else "FAIL"
        print(f"[{status}] {outcome.name}: {outcome.detail}", flush=True)

    failed = [o for o in outcomes if not o.passed]
    print()
    if failed:
        print(f"TRAINING READINESS = NOT READY ({len(failed)}/{len(outcomes)} checks failed)")
        for outcome in failed:
            print(f"  failed: {outcome.name}")
        return 1
    print(f"TRAINING READINESS = READY ({len(outcomes)}/{len(outcomes)} checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
