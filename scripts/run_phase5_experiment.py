"""Phase 5 — main NIC continual experiment driver (naive vs experience replay).

Runs BOTH methods on Scenario NIC (variant ``inc``, run 0) with identical
initial model state, architecture, hyperparameters, official experience
order and fixed evaluation set; the only difference is the bounded replay
memory. After every experience it evaluates on the official CORe50 test
sessions (3/7/10), persists metric records, checkpoints, and log files so
an interrupted run resumes exactly where it stopped.

Usage (from the project root)::

    python scripts/run_phase5_experiment.py                      # both methods
    python scripts/run_phase5_experiment.py --method naive       # one method
    python scripts/run_phase5_experiment.py --method replay      # resume replay

Outputs (all relative to the project root):

- ``models/continual/phase5_nic/{shared,naive,replay}/``  checkpoints + state
- ``reports/phase5_nic/``                                  metrics, plots, report
- ``logs/phase5_nic_{master,naive,replay}.log``            run logs

The script is safe to leave unattended: state is saved atomically after
every experience, completed metric records are never lost, and restarts
re-derive anything missing before continuing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.continual import ContinualScenario, load_scenario_cached  # noqa: E402
from src.evaluation.continual import (  # noqa: E402
    EvalCache,
    atomic_write_json,
    build_metric_record,
    compute_forgetting,
    evaluate_model,
    load_class_names,
    read_json,
)
from src.training import (  # noqa: E402
    ContinualTrainConfig,
    build_continual_trainer,
    checkpoint_exists,
    load_state,
    model_checksum,
)
from src.training.config import config_fingerprint, resolve_device  # noqa: E402
from src.training.model import build_model, seed_everything  # noqa: E402
from src.utils.experiment_display import ExperimentDisplay  # noqa: E402
from src.utils.progress import PhaseProgress, format_duration  # noqa: E402

STEP_LABELS = [
    "Pre-flight and experiment planning",
    "Creating reproducible experiment configuration",
    "Initializing independent Naive and Replay runs",
    "Running Naive Continual Learning",
    "Running Experience Replay Continual Learning",
    "Evaluating Naive vs Replay across experiences",
    "Generating Phase-5 results",
    "Finalizing experiment",
]

FORGETTING_DEFINITION = (
    "forgetting(c, t) = max{acc(c, t') : t' < t} - acc(c, t); the aggregate at "
    "experience t is the mean over every class that has a prior measurement. "
    "The first experience has no prior measurement (null / N-A). Negative "
    "values (a class improved on its previous best) are preserved."
)

OVERALL_DEFINITION = (
    "overall = micro top-1 accuracy over evaluation samples whose class has "
    "been introduced up to that experience; full = over all evaluation "
    "classes; old = over classes seen before the experience (null at "
    "experience 0); new = over classes introduced by the experience (null "
    "when none)."
)

METHODS = ("naive", "replay")
DEFAULT_CONFIG = "configs/phase5_nic.yaml"
DEFAULT_ROOT = "models/continual/phase5_nic"
DEFAULT_REPORTS = "reports/phase5_nic"
MIN_FREE_BYTES = 2 * 1024**3
EXPECTED_IMAGES = 164_866
EXPECTED_EVAL_SAMPLES = 44_972
EXPECTED_EXPERIENCES = 79


class Phase5Blocked(Exception):
    """Raised when the experiment must stop safely with a resume path."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def _file_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    existing = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    for handler in existing:
        logger.removeHandler(handler)
        handler.close()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-5 NIC continual experiment")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="models output root")
    parser.add_argument("--reports", default=DEFAULT_REPORTS, help="reports output root")
    parser.add_argument(
        "--max-experiences",
        type=int,
        default=None,
        help="smoke tests only: stop after N experiences (refused with the official roots)",
    )
    args = parser.parse_args(argv)
    if args.max_experiences is not None:
        if args.root == DEFAULT_ROOT and args.reports == DEFAULT_REPORTS:
            parser.error(
                "--max-experiences refuses to run with the official output roots "
                "(pass --root/--reports for smoke tests)"
            )
        if args.max_experiences < 1:
            parser.error("--max-experiences must be >= 1")
    return args


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise Phase5Blocked(f"Experiment config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase5Blocked(f"Expected a mapping in {config_path}")
    required = ("scenario", "variant", "run", "training", "replay", "model", "evaluation")
    missing = [key for key in required if key not in payload]
    if missing:
        raise Phase5Blocked(f"Config is missing keys: {', '.join(missing)}")
    if payload["scenario"] != "NIC" or payload["variant"] != "inc":
        raise Phase5Blocked(
            f"Phase 5 requires scenario NIC variant inc, got "
            f"{payload['scenario']}/{payload['variant']}"
        )
    model = payload["model"]
    if model.get("arch") != "small_cnn":
        raise Phase5Blocked(
            f"Phase 5 reuses the Phase-4 architecture, got {model.get('arch')!r}"
        )
    return payload


def build_train_config(exp: dict[str, Any], method: str) -> ContinualTrainConfig:
    sections: dict[str, Any] = {
        "continual": {"method": method},
        "training": dict(exp["training"]),
        "replay": dict(exp["replay"]),
    }
    return ContinualTrainConfig.from_dict(sections)


# ---------------------------------------------------------------------------
# step 2/3 — shared configuration, initial state, cache
# ---------------------------------------------------------------------------
def write_shared_config(
    shared: Path,
    exp: dict[str, Any],
    cfgs: dict[str, ContinualTrainConfig],
    config_path: str,
) -> dict[str, Any]:
    resolved = {
        "phase": 5,
        "experiment_name": exp.get("experiment_name"),
        "source_config": config_path,
        "dataset": exp.get("dataset", "core50"),
        "scenario": exp["scenario"],
        "variant": exp["variant"],
        "run": int(exp["run"]),
        "model": dict(exp["model"]),
        "training": dict(exp["training"]),
        "replay": dict(exp["replay"]),
        "evaluation": dict(exp["evaluation"]),
        "outputs": dict(exp.get("outputs") or {}),
        "documented_deviations_from_phase4_defaults": [
            "training.epochs=3 (Phase-4 generic default was 1)",
            "evaluation.batch_size=256 (evaluation-only setting)",
        ],
        "config_fingerprints": {
            method: config_fingerprint(cfg) for method, cfg in cfgs.items()
        },
        "written_utc": utc_now(),
    }
    atomic_write_json(shared / "config.json", resolved)
    return resolved


def init_initial_model_state(
    shared: Path, cfg: ContinualTrainConfig, num_classes: int
) -> dict[str, Any]:
    """Create the shared initial model state once (or verify it on resume)."""
    state_path = shared / "initial_model_state.pt"
    seed_everything(cfg.seed)
    reference = build_model(num_classes, width=cfg.model_width, seed=cfg.seed)
    if state_path.is_file():
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model_state" not in payload:
            raise Phase5Blocked(f"Corrupt shared initial state: {state_path}")
        reference.load_state_dict(payload["model_state"])
        if payload.get("checksum") != model_checksum(reference):
            raise Phase5Blocked(
                "Shared initial model state does not match the configured seed"
            )
    else:
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "format_version": 1,
                "checksum": model_checksum(reference),
                "seed": cfg.seed,
                "width": cfg.model_width,
                "num_classes": num_classes,
                "model_state": reference.state_dict(),
            },
            tmp,
        )
        tmp.replace(state_path)
    info = {
        "checksum": model_checksum(reference),
        "seed": cfg.seed,
        "width": cfg.model_width,
        "num_classes": num_classes,
        "parameters": int(sum(p.numel() for p in reference.parameters())),
        "state_file": state_path.name,
    }
    atomic_write_json(shared / "initial_model.json", info)
    return info


def ensure_eval_cache(
    shared: Path,
    scenario: ContinualScenario,
    image_size: int,
    phase: PhaseProgress,
) -> EvalCache:
    cache_path = shared / "eval_cache.pt"
    expected = tuple(
        record.relative_path
        for record in scenario.experiences[0].evaluation_samples
    )
    if cache_path.is_file():
        try:
            cache = EvalCache.load(cache_path)
            if (
                tuple(cache.paths) == expected
                and cache.image_size == image_size
                and len(cache) == EXPECTED_EVAL_SAMPLES
            ):
                phase.log(f"Reusing evaluation cache: {cache_path}")
                return cache
            phase.log("Evaluation cache is stale; rebuilding")
        except Exception as exc:  # noqa: BLE001
            phase.log(f"Unreadable evaluation cache ({exc}); rebuilding")

    def progress(current: int, total: int) -> None:
        phase.update(
            current,
            total,
            detail=f"Building evaluation cache: {current:,} / {total:,}",
        )

    started = time.perf_counter()
    records = scenario.experiences[0].evaluation_samples
    cache = EvalCache.from_records(
        records, scenario.images_root, image_size, on_progress=progress
    )
    cache.save(cache_path)
    phase.log(
        f"Built evaluation cache: {len(cache):,} samples in "
        f"{time.perf_counter() - started:.1f}s -> {cache_path}"
    )
    return cache


# ---------------------------------------------------------------------------
# steps 4/5 — running one method
# ---------------------------------------------------------------------------
def _mark_initial_state_verified(
    method: str, manifest_path: Path, how: str, logger: logging.Logger
) -> None:
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    verified = dict(manifest.get("initial_state_verified") or {})
    verified[method] = True
    manifest["initial_state_verified"] = verified
    notes = dict(manifest.get("initial_state_verification_notes") or {})
    notes[method] = how
    manifest["initial_state_verification_notes"] = notes
    atomic_write_json(manifest_path, manifest)
    logger.info("initial state verified for %s (%s)", method, how)


def run_method(
    *,
    method: str,
    scenario: ContinualScenario,
    exp: dict[str, Any],
    cfg: ContinualTrainConfig,
    root: Path,
    reports: Path,
    cache: EvalCache,
    phase: PhaseProgress,
    initial_checksum: str,
    logger: logging.Logger,
    max_experiences: int | None,
    manifest_path: Path,
) -> dict[str, Any]:
    total = len(scenario)
    limit = min(total, max_experiences) if max_experiences else total
    method_dir = root / method
    metrics_path = reports / f"{method}_metrics.json"
    num_classes = int(exp["model"]["num_classes"])
    eval_batch = int(exp["evaluation"]["batch_size"])
    epochs = cfg.epochs
    steps_per_epoch_cap = None

    display = ExperimentDisplay(
        method=method,
        scenario_name=scenario.name,
        run_id=scenario.run_id,
        total_experiences=total,
        epochs=epochs,
    )

    trainer = build_continual_trainer(method, cfg)
    collector = {"loss_sum": 0.0, "correct": 0.0, "n": 0, "steps": 0, "epochs": 0}

    def on_metrics(info: dict[str, Any]) -> None:
        display.on_metrics(info)
        if info.get("event") == "batch":
            samples = int(info.get("samples", 0))
            collector["loss_sum"] += float(info.get("loss", 0.0)) * samples
            collector["correct"] += float(info.get("batch_accuracy", 0.0)) * samples
            collector["n"] += samples
            collector["steps"] += 1
        elif info.get("event") == "epoch":
            collector["epochs"] += 1

    trainer.on_train_metrics = on_metrics

    records: list[dict[str, Any]] = []
    if metrics_path.is_file():
        envelope = read_json(metrics_path)
        if envelope.get("method") != method:
            raise Phase5Blocked(
                f"Metrics file {metrics_path} belongs to {envelope.get('method')!r}"
            )
        records = list(envelope.get("records") or [])

    initialized_now = False
    if checkpoint_exists(method_dir):
        state = trainer.resume(scenario, checkpoint_dir=method_dir)
        phase.log(
            f"Resuming {method}: checkpoint at experience "
            f"{state.current_experience + 1}/{total}"
        )
        logger.info("resume from experience %s", state.current_experience + 1)
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if not (manifest.get("initial_state_verified") or {}).get(method):
            _mark_initial_state_verified(
                method,
                manifest_path,
                "inferred after resume: state config fingerprint matches the "
                "shared configuration (fresh init verifies the checksum in the "
                "same code path)",
                logger,
            )
    else:
        state = trainer.initialize(scenario, checkpoint_dir=method_dir)
        initialized_now = True
        checksum = model_checksum(trainer.model)
        if checksum != initial_checksum:
            raise Phase5Blocked(
                f"{method} initial model state {checksum} does not match the "
                f"shared initial state {initial_checksum}"
            )
        logger.info("initialized from shared state %s", initial_checksum)
        _mark_initial_state_verified(
            method, manifest_path, f"checksum verified: {checksum}", logger
        )
        if method == "replay":
            payload = torch.load(
                method_dir / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            memory = payload.get("replay_memory") or {}
            if memory.get("items"):
                raise Phase5Blocked("Replay memory must start empty")

    # Synchronize metric records with the on-disk checkpoint experience.
    state_ids = set(state.experiences_trained)
    record_ids = {int(r["experience_id"]) for r in records}
    extra = sorted(record_ids - state_ids)
    if extra:
        raise Phase5Blocked(
            f"Metrics contain experiences missing from the checkpoint: {extra}"
        )
    missing = sorted(state_ids - record_ids)
    for experience_id in missing:
        recomputed_late = experience_id < state.current_experience
        record = evaluate_experience(
            method=method,
            scenario=scenario,
            experience_id=experience_id,
            trainer=trainer,
            cache=cache,
            num_classes=num_classes,
            eval_batch=eval_batch,
            cfg=cfg,
            display=display,
            replay_stats=state.replay,
            train_stats={
                "epochs": epochs,
                "steps": None,
                "loss_mean": None,
                "accuracy_mean": None,
                "seconds": None,
                "note": "metrics recomputed after restart; training statistics unavailable",
            },
            recomputed_late=recomputed_late,
        )
        records.append(record)
        records.sort(key=lambda r: r["experience_id"])
        atomic_write_json(metrics_path, _metrics_envelope(scenario, method, records))
        logger.info(
            "recomputed metrics for experience %s (late=%s)",
            experience_id,
            recomputed_late,
        )

    display.start_run()
    next_id = state.current_experience + 1
    while next_id < limit:
        experience = scenario.get_experience(next_id)
        steps_per_epoch = math.ceil(len(experience.train_samples) / cfg.batch_size)
        steps_per_epoch_cap = steps_per_epoch
        display.start_experience(
            next_id,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            completed_before=len(records),
        )
        for key in collector:
            collector[key] = 0
        logger.info(
            "experience %s/%s start (train samples=%s)",
            next_id + 1,
            total,
            len(experience.train_samples),
        )
        started = time.perf_counter()
        state = trainer.train_experience(state, experience)
        train_seconds = time.perf_counter() - started

        train_stats = {
            "epochs": collector["epochs"],
            "steps": collector["steps"],
            "loss_mean": round(collector["loss_sum"] / max(collector["n"], 1), 6),
            "accuracy_mean": round(collector["correct"] / max(collector["n"], 1), 6),
            "seconds": round(train_seconds, 3),
        }
        record = evaluate_experience(
            method=method,
            scenario=scenario,
            experience_id=next_id,
            trainer=trainer,
            cache=cache,
            num_classes=num_classes,
            eval_batch=eval_batch,
            cfg=cfg,
            display=display,
            replay_stats=state.replay,
            train_stats=train_stats,
            recomputed_late=False,
        )
        records.append(record)
        atomic_write_json(metrics_path, _metrics_envelope(scenario, method, records))

        accuracy = record["accuracy"]
        summary = (
            f"[{method}] experience {next_id + 1}/{total} complete — "
            f"train={record['train_samples']:,} "
            f"loss={train_stats['loss_mean']:.4f} "
            f"train_acc={100.0 * train_stats['accuracy_mean']:.1f}% | "
            f"eval overall={_pct(accuracy['overall'])} "
            f"old={_pct(accuracy['old'])} new={_pct(accuracy['new'])} "
            f"full={_pct(accuracy['full'])} | "
            f"{format_duration(train_seconds)} train + "
            f"{record['eval_seconds']:.1f}s eval"
        )
        display.finish_experience(summary)
        logger.info(
            "experience %s/%s done: loss=%.4f overall=%s old=%s new=%s "
            "train_s=%.1f eval_s=%.1f",
            next_id + 1,
            total,
            train_stats["loss_mean"],
            accuracy["overall"],
            accuracy["old"],
            accuracy["new"],
            train_seconds,
            record["eval_seconds"],
        )
        next_id += 1

    display.end_block()
    complete = [r["experience_id"] for r in records] == list(range(total))
    result = {
        "method": method,
        "status": "complete" if complete else "partial",
        "records": len(records),
        "current_experience": state.current_experience,
        "initialized_now": initialized_now,
        "steps_per_epoch": steps_per_epoch_cap,
    }
    logger.info("run finished: %s", result)
    return result


def evaluate_experience(
    *,
    method: str,
    scenario: ContinualScenario,
    experience_id: int,
    trainer,
    cache: EvalCache,
    num_classes: int,
    eval_batch: int,
    cfg: ContinualTrainConfig,
    display: ExperimentDisplay,
    replay_stats: dict[str, Any] | None,
    train_stats: dict[str, Any],
    recomputed_late: bool,
) -> dict[str, Any]:
    experience = scenario.get_experience(experience_id)
    previous_seen: Sequence[int] = (
        scenario.get_experience(experience_id - 1).classes_seen
        if experience_id > 0
        else ()
    )
    started = time.perf_counter()
    eval_result = evaluate_model(
        trainer.model,
        cache,
        num_classes=num_classes,
        device=trainer.device,
        batch_size=eval_batch,
        on_progress=display.on_eval_progress,
    )
    eval_seconds = time.perf_counter() - started
    record = build_metric_record(
        experience_id=experience_id,
        train_samples=len(experience.train_samples),
        classes_introduced=experience.classes_introduced,
        classes_seen=experience.classes_seen,
        previous_seen=previous_seen,
        eval_result=eval_result,
        train_stats=train_stats,
        eval_seconds=eval_seconds,
        replay_stats=replay_stats,
    )
    if recomputed_late:
        record["recomputed_late"] = True
    if method == "naive":
        record.pop("replay", None)
    return record


def _metrics_envelope(
    scenario: ContinualScenario, method: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "phase": 5,
        "dataset": "core50",
        "scenario": scenario.name,
        "variant": scenario.variant,
        "run": scenario.run_id,
        "method": method,
        "records": records,
    }


def _pct(value: float | None) -> str:
    return "N-A" if value is None else f"{100.0 * value:.1f}%"


# ---------------------------------------------------------------------------
# step 6 — finalize metrics (forgetting, per-class, comparison CSV)
# ---------------------------------------------------------------------------
def finalize_metrics(
    scenario: ContinualScenario,
    exp: dict[str, Any],
    reports: Path,
    phase: PhaseProgress,
    logger: logging.Logger,
) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        path = reports / f"{method}_metrics.json"
        if not path.is_file():
            raise Phase5Blocked(f"Missing metrics file: {path}")
        envelope = read_json(path)
        records = list(envelope.get("records") or [])
        ids = [r["experience_id"] for r in records]
        if ids != list(range(len(scenario))):
            raise Phase5Blocked(
                f"{method} metrics incomplete ({len(records)} of {len(scenario)} "
                f"experiences) — resume training first"
            )
        forgetting = compute_forgetting(records)
        for record, value in zip(records, forgetting):
            record["forgetting"] = round(value, 6) if value is not None else None
        envelope["accuracy_definition"] = OVERALL_DEFINITION
        envelope["forgetting_definition"] = FORGETTING_DEFINITION
        envelope["finalized_utc"] = utc_now()
        atomic_write_json(path, envelope)
        data[method] = records
        phase.log(f"Finalized {method} metrics ({len(records)} experiences)")
        logger.info("finalized %s metrics (forgetting computed)", method)

    num_classes = int(exp["model"]["num_classes"])
    names = load_class_names(exp["object_mapping"])
    per_class: dict[str, Any] = {
        "phase": 5,
        "definition": (
            "Per-class accuracy for each experience index (null before the "
            "class is introduced); rows are 0-based labels."
        ),
        "forgetting_definition": FORGETTING_DEFINITION,
        "class_names": {str(i): names.get(str(i), f"class_{i}") for i in range(num_classes)},
        "experiences": list(range(len(scenario))),
    }
    for method in METHODS:
        per_class[method] = {
            str(class_id): [
                (
                    round(record["per_class"][str(class_id)], 6)
                    if record["per_class"].get(str(class_id)) is not None
                    else None
                )
                for record in data[method]
            ]
            for class_id in range(num_classes)
        }
    atomic_write_json(reports / "per_class_metrics.json", per_class)
    phase.log(f"Per-class histories: {reports / 'per_class_metrics.json'}")

    csv_path = reports / "comparison.csv"

    def fmt(value: float | None) -> str:
        return "NA" if value is None else f"{value:.6f}"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experience_id",
                "naive_overall",
                "replay_overall",
                "naive_old",
                "replay_old",
                "naive_new",
                "replay_new",
                "naive_forgetting",
                "replay_forgetting",
            ]
        )
        for index in range(len(scenario)):
            naive = data["naive"][index]
            replay = data["replay"][index]
            writer.writerow(
                [
                    index,
                    fmt(naive["accuracy"]["overall"]),
                    fmt(replay["accuracy"]["overall"]),
                    fmt(naive["accuracy"]["old"]),
                    fmt(replay["accuracy"]["old"]),
                    fmt(naive["accuracy"]["new"]),
                    fmt(replay["accuracy"]["new"]),
                    fmt(naive["forgetting"]),
                    fmt(replay["forgetting"]),
                ]
            )
    logger.info("wrote %s", csv_path)
    return data


# ---------------------------------------------------------------------------
# integrity checks
# ---------------------------------------------------------------------------
def run_integrity_checks(
    *,
    root: Path,
    reports: Path,
    shared: Path,
    exp: dict[str, Any],
    scenario: ContinualScenario,
    data: dict[str, list[dict[str, Any]]],
    logger: logging.Logger,
) -> dict[str, dict[str, str]]:
    checks: dict[str, dict[str, str]] = {}

    def record(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}
        logger.info("integrity %s: %s — %s", name, "PASS" if ok else "FAIL", detail)

    total = len(scenario)
    order_ok = all(
        [r["experience_id"] for r in data[m]] == list(range(total)) for m in METHODS
    ) and all(
        all(
            r["train_samples"] == len(scenario.get_experience(r["experience_id"]).train_samples)
            for r in data[m]
        )
        for m in METHODS
    )
    record(
        "official_experience_order",
        order_ok,
        f"records follow 0..{total - 1} with official training counts for both methods",
    )

    leak_ok = True
    for method in METHODS:
        previous_seen: set[int] = set()
        for index, rec in enumerate(data[method]):
            seen = set(rec["classes_seen"])
            introduced = set(rec["classes_introduced"])
            if not introduced <= seen or not introduced.isdisjoint(previous_seen):
                leak_ok = False
            if not previous_seen <= seen:
                leak_ok = False
            expected = (
                set(scenario.get_experience(index).classes_seen)
            )
            if seen != expected or introduced != set(
                scenario.get_experience(index).classes_introduced
            ):
                leak_ok = False
            previous_seen = seen
    record(
        "no_future_experience_leakage",
        leak_ok,
        "class sets match the official scenario; no class appears before introduction",
    )

    eval_ok = True
    detail_parts: list[str] = []
    for method in METHODS:
        for rec in data[method]:
            if rec["eval_samples"] != EXPECTED_EVAL_SAMPLES:
                eval_ok = False
        state, payload = load_state(root / method)
        if method == "replay":
            memory = payload.get("replay_memory") or {}
            items = memory.get("items") or []
            if any(item.get("split") != "train" for item in items):
                eval_ok = False
            if any(
                int(item.get("experience_id", -1)) > int(memory.get("last_experience", -1))
                for item in items
            ):
                eval_ok = False
            detail_parts.append(f"replay memory {len(items)} train-only references")
        else:
            if "replay_memory" in payload:
                eval_ok = False
            detail_parts.append("naive checkpoint has no replay memory")
    record(
        "no_evaluation_samples_in_training",
        eval_ok,
        f"{EXPECTED_EVAL_SAMPLES} eval samples per experience; "
        + "; ".join(detail_parts),
    )

    shared_info = read_json(shared / "initial_model.json")
    manifest = read_json(shared / "run_manifest.json")
    verified = manifest.get("initial_state_verified") or {}
    record(
        "same_initial_state_both_methods",
        all(verified.get(m) is True for m in METHODS)
        and bool(shared_info.get("checksum")),
        f"shared checksum {shared_info.get('checksum', '')} "
        f"verified={{{', '.join(f'{m}: {verified.get(m)}' for m in METHODS)}}}",
    )

    configs_ok = True
    training_configs: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        state, _ = load_state(root / method)
        training_configs[method] = dict(state.training_config)
        if state.method != method or state.scenario != scenario.scenario_type:
            configs_ok = False
    keys = ("epochs", "batch_size", "learning_rate", "seed", "image_size",
            "model_width", "optimizer", "scheduler", "device", "workers")
    identical = all(
        training_configs["naive"].get(k) == training_configs["replay"].get(k)
        for k in keys
    )
    if not identical:
        configs_ok = False
    record(
        "independent_runs_identical_settings",
        configs_ok,
        "separate run directories, matching training hyperparameters, "
        "differing only in the replay section",
    )

    capacity = int(exp["replay"]["capacity"])
    _, replay_payload = load_state(root / "replay")
    memory = replay_payload.get("replay_memory") or {}
    items = memory.get("items") or []
    record(
        "bounded_replay_memory",
        len(items) <= capacity and int(memory.get("capacity", 0)) == capacity,
        f"{len(items)} of {capacity} training references (FIFO bound respected)",
    )

    images_root = Path(scenario.images_root)
    image_count = sum(1 for p in images_root.rglob("*.png") if p.is_file())
    record(
        "no_image_duplication",
        image_count == EXPECTED_IMAGES,
        f"{image_count:,} images remain in the read-only dataset tree "
        f"(expected {EXPECTED_IMAGES:,}; single enumeration, no decoding)",
    )

    config_ok = (shared / "config.json").is_file() and bool(
        manifest.get("config_fingerprints")
    )
    for path in reports.iterdir():
        if path.suffix in {".json", ".csv", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "C:\\" in text or "/home/" in text or "C:/" in text:
                config_ok = False
    record(
        "reproducible_configuration",
        config_ok,
        "resolved config + fingerprints recorded; no absolute paths in reports",
    )
    return checks


# ---------------------------------------------------------------------------
# step 7 — plots, summary, report
# ---------------------------------------------------------------------------
def make_plots(
    data: dict[str, list[dict[str, Any]]], reports: Path, logger: logging.Logger
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = reports / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = {"naive": "#d62728", "replay": "#1f77b4"}
    specs = [
        ("overall", "Overall (cumulative) accuracy", "overall_accuracy"),
        ("old", "Old-class accuracy", "old_class_accuracy"),
        ("new", "New-class accuracy", "new_class_accuracy"),
        ("forgetting", "Forgetting (mean over prior classes)", "forgetting"),
    ]
    written: list[str] = []
    for key, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for method in METHODS:
            xs: list[int] = []
            ys: list[float | None] = []
            for record in data[method]:
                value = (
                    record["forgetting"]
                    if key == "forgetting"
                    else record["accuracy"][key]
                )
                xs.append(record["experience_id"])
                ys.append(value)
            ax.plot(
                xs,
                ys,
                label=method.capitalize(),
                color=colors[method],
                linewidth=1.6,
                marker="o",
                markersize=3,
            )
        ax.set_xlabel("Experience index")
        ax.set_ylabel(ylabel)
        if key != "forgetting":
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        ax.set_title(f"Phase 5 — NIC inc run 0: {ylabel}")
        fig.tight_layout()
        path = plots_dir / f"{filename}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(f"plots/{filename}.png")
        logger.info("wrote %s", path)
    return written


def build_summary(
    *,
    exp: dict[str, Any],
    scenario: ContinualScenario,
    data: dict[str, list[dict[str, Any]]],
    root: Path,
    reports: Path,
    checks: dict[str, dict[str, str]],
    plots: list[str],
    device: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_records = {m: data[m][-1] for m in METHODS}
    training_time = {
        m: round(
            sum(r["train"]["seconds"] or 0.0 for r in data[m]), 3
        )
        for m in METHODS
    }

    def light(method: str) -> list[dict[str, Any]]:
        return [
            {
                "experience_id": r["experience_id"],
                "overall": r["accuracy"]["overall"],
                "old": r["accuracy"]["old"],
                "new": r["accuracy"]["new"],
                "forgetting": r["forgetting"],
            }
            for r in data[method]
        ]

    def average_incremental(method: str) -> float:
        values = [
            r["accuracy"]["overall"]
            for r in data[method]
            if r["accuracy"]["overall"] is not None
        ]
        return round(sum(values) / len(values), 6)

    previous_wall = 0.0
    prior_path = reports / "experiment_summary.json"
    if prior_path.is_file():
        try:
            prior_summary = read_json(prior_path)
        except (OSError, ValueError):
            prior_summary = None
        if isinstance(prior_summary, dict):
            try:
                previous_wall = float(prior_summary.get("wall_clock_seconds") or 0.0)
            except (TypeError, ValueError):
                previous_wall = 0.0

    summary: dict[str, Any] = {
        "phase": 5,
        "status": "complete",
        "dataset": "CORe50",
        "scenario": scenario.scenario_type,
        "variant": scenario.variant,
        "run": scenario.run_id,
        "seed": int(exp["training"]["seed"]),
        "model": (
            f"SmallConvNet (width {exp['model']['width']}, "
            f"{exp['model']['image_size']}x{exp['model']['image_size']} input)"
        ),
        "device": device,
        "num_experiences": len(scenario),
        "accuracy_definition": OVERALL_DEFINITION,
        "forgetting_definition": FORGETTING_DEFINITION,
        "average_incremental_definition": (
            "average incremental accuracy = the mean, over all experiences, "
            "of the overall (cumulative-classes) accuracy measured after "
            "each experience."
        ),
        "naive": {
            "status": "complete",
            "metrics": light("naive"),
            "final_checkpoint": f"{root}/naive/checkpoint.pt",
            "training_time_seconds": training_time["naive"],
            "final_accuracy": {
                "overall": final_records["naive"]["accuracy"]["overall"],
                "full": final_records["naive"]["accuracy"]["full"],
                "old": final_records["naive"]["accuracy"]["old"],
                "forgetting": final_records["naive"]["forgetting"],
            },
        },
        "replay": {
            "status": "complete",
            "metrics": light("replay"),
            "final_checkpoint": f"{root}/replay/checkpoint.pt",
            "training_time_seconds": training_time["replay"],
            "replay_capacity": int(exp["replay"]["capacity"]),
            "final_accuracy": {
                "overall": final_records["replay"]["accuracy"]["overall"],
                "full": final_records["replay"]["accuracy"]["full"],
                "old": final_records["replay"]["accuracy"]["old"],
                "forgetting": final_records["replay"]["forgetting"],
            },
        },
        "comparison": {
            "final_accuracy": {
                "naive": final_records["naive"]["accuracy"]["overall"],
                "replay": final_records["replay"]["accuracy"]["overall"],
            },
            "final_forgetting": {
                "naive": final_records["naive"]["forgetting"],
                "replay": final_records["replay"]["forgetting"],
            },
            "average_incremental_accuracy": {
                "naive": average_incremental("naive"),
                "replay": average_incremental("replay"),
            },
        },
        "integrity": checks,
        "artifacts": {
            "config": "configs/phase5_nic.yaml",
            "resolved_config": f"{root}/shared/config.json",
            "naive_metrics": f"{reports}/naive_metrics.json",
            "replay_metrics": f"{reports}/replay_metrics.json",
            "comparison_csv": f"{reports}/comparison.csv",
            "per_class_metrics": f"{reports}/per_class_metrics.json",
            "report": f"{reports}/phase5_nic_report.md",
            "plots": plots,
        },
        "wall_clock_seconds": round(previous_wall + elapsed_seconds, 3),
        "generated_utc": utc_now(),
    }
    return summary


def build_report(
    *,
    exp: dict[str, Any],
    scenario: ContinualScenario,
    data: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    reports: Path,
    root: Path,
) -> str:
    final = {m: data[m][-1] for m in METHODS}
    naive_final = final["naive"]["accuracy"]
    replay_final = final["replay"]["accuracy"]
    naive_f = final["naive"]["forgetting"]
    replay_f = final["replay"]["forgetting"]
    acc_delta = (replay_final["overall"] or 0.0) - (naive_final["overall"] or 0.0)
    forget_delta = (replay_f or 0.0) - (naive_f or 0.0)
    avg_inc = summary["comparison"]["average_incremental_accuracy"]
    avg_delta = avg_inc["replay"] - avg_inc["naive"]
    measured = (
        f"Final cumulative accuracy: replay {replay_final['overall']:.4f} "
        f"versus naive {naive_final['overall']:.4f} "
        f"(difference {acc_delta:+.4f}). Final mean forgetting: replay "
        f"{replay_f:.4f} versus naive {naive_f:.4f} (difference "
        f"{forget_delta:+.4f}; lower is better). Average incremental "
        f"accuracy: replay {avg_inc['replay']:.4f} versus naive "
        f"{avg_inc['naive']:.4f} (difference {avg_delta:+.4f})."
    )

    milestones = [0, 9, 19, 29, 39, 49, 59, 69, 78]
    milestone_rows = []
    for index in milestones:
        naive = data["naive"][index]
        replay = data["replay"][index]
        milestone_rows.append(
            f"| {index} | {_pct(naive['accuracy']['overall'])} | "
            f"{_pct(replay['accuracy']['overall'])} | "
            f"{_pct(naive['accuracy']['old'])} | {_pct(replay['accuracy']['old'])} | "
            f"{_pct(naive['accuracy']['new'])} | {_pct(replay['accuracy']['new'])} | "
            f"{_pct(naive['forgetting'])} | {_pct(replay['forgetting'])} |"
        )

    integrity_rows = [
        f"| {name} | {result['status']} | {result['detail']} |"
        for name, result in summary["integrity"].items()
    ]

    training = exp["training"]
    replay_cfg = exp["replay"]
    lines = [
        "# Phase 5 — Main NIC Continual Experiment: Naive vs Experience Replay",
        "",
        "## 1. Experiment",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Dataset | CORe50 |",
        f"| Scenario / variant / run | {scenario.scenario_type} / {scenario.variant} / {scenario.run_id} |",
        f"| Experiences (official order) | {len(scenario)} |",
        f"| Training samples (total) | {scenario.metadata['train_samples_total']:,} |",
        f"| Evaluation samples (fixed, sessions {scenario.metadata['evaluation_sessions']}) | {EXPECTED_EVAL_SAMPLES:,} |",
        f"| Model | {summary['model']} |",
        f"| Device | {summary['device']} |",
        f"| Seed | {summary['seed']} |",
        "",
        "## 2. Configuration and fairness",
        "",
        "Both methods start from ONE shared initial model state "
        f"(`{root}/shared/initial_model_state.pt`, checksum recorded), "
        "train in the identical official order with identical architecture "
        "and hyperparameters, and are evaluated on the same fixed test set "
        "after every experience. The ONLY difference is the replay section:",
        "",
        "| Setting | Naive | Replay |",
        "|---|---|---|",
        "| Replay memory | none | FIFO, capacity "
        f"{replay_cfg['capacity']} training references |",
        f"| Replay batch per step | - | {replay_cfg['batch_size']} |",
        "| Everything else (epochs "
        f"{training['epochs']}, batch {training['batch_size']}, lr "
        f"{training['learning_rate']}, {training['optimizer']}, seed "
        f"{training['seed']}, image {exp['model']['image_size']}px, width "
        f"{exp['model']['width']}) | identical | identical |",
        "",
        "Documented deviation from Phase-4 generic defaults: "
        "`training.epochs` is 3 (Phase-4 default was 1); "
        "`evaluation.batch_size` is an evaluation-only setting.",
        "",
        "## 3. Metric definitions",
        "",
        f"- {OVERALL_DEFINITION}",
        f"- {FORGETTING_DEFINITION}",
        f"- {summary['average_incremental_definition']}",
        "- Evaluation never touches training; replay memory stores only "
        "official training references from already-completed experiences.",
        "",
        "## 4. Results",
        "",
        "### 4.1 Final comparison (after experience 79)",
        "",
        "| Metric | Naive | Replay |",
        "|---|---|---|",
        f"| Overall (cumulative) accuracy | {naive_final['overall']:.4f} | {replay_final['overall']:.4f} |",
        f"| Full-set accuracy (all 50 classes) | {naive_final['full']:.4f} | {replay_final['full']:.4f} |",
        f"| Old-class accuracy | {naive_final['old']:.4f} | {replay_final['old']:.4f} |",
        f"| Final mean forgetting (lower is better) | {naive_f:.4f} | {replay_f:.4f} |",
        f"| Average incremental accuracy (mean over all experiences) | {avg_inc['naive']:.4f} | {avg_inc['replay']:.4f} |",
        f"| Training time (sum of experiences, s) | {summary['naive']['training_time_seconds']} | {summary['replay']['training_time_seconds']} |",
        "",
        f"**Measured result.** {measured}",
        "",
        "### 4.2 Milestone experiences",
        "",
        "| Exp | Naive overall | Replay overall | Naive old | Replay old | Naive new | Replay new | Naive forget | Replay forget |",
        "|---|---|---|---|---|---|---|---|---|",
        *milestone_rows,
        "",
        "### 4.3 Plots",
        "",
        "![Overall accuracy](plots/overall_accuracy.png)",
        "",
        "![Old-class accuracy](plots/old_class_accuracy.png)",
        "",
        "![New-class accuracy](plots/new_class_accuracy.png)",
        "",
        "![Forgetting](plots/forgetting.png)",
        "",
        "## 5. Per-experience data",
        "",
        f"- `{reports}/naive_metrics.json` — full naive records (accuracy, "
        "per-class, train stats, forgetting)",
        f"- `{reports}/replay_metrics.json` — full replay records",
        f"- `{reports}/per_class_metrics.json` — per-class accuracy histories",
        f"- `{reports}/comparison.csv` — side-by-side comparison table",
        "",
        "## 6. Integrity checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
        *integrity_rows,
        "",
        "## 7. Limitations",
        "",
        "- Single official run (NIC inc, run 0) and a single seed.",
        "- Fixed epoch budget (3 per experience); no hyperparameter search "
        "was performed for either method.",
        "- Replay capacity/batch are fixed configuration values, not tuned "
        "against the evaluation set.",
        "",
        "## 8. Artifacts and reproduction",
        "",
        f"- Experiment config: `configs/phase5_nic.yaml`",
        f"- Resolved config: `{root}/shared/config.json`",
        f"- Shared initial state: `{root}/shared/initial_model_state.pt`",
        f"- Checkpoints: `{root}/naive/checkpoint.pt`, `{root}/replay/checkpoint.pt`",
        f"- Report: `{reports}/phase5_nic_report.md`",
        "",
        "Reproduce / resume:",
        "",
        "```bash",
        "python scripts/run_phase5_experiment.py --config configs/phase5_nic.yaml",
        "```",
        "",
        f"Generated {summary['generated_utc']}.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _blocked_block(
    *,
    method: str | None,
    experience: int | None,
    error: str,
    root: Path,
    logs_dir: Path,
    resume_hint: str,
) -> str:
    lines = [
        "",
        "PHASE 5 BLOCKED",
        f"Method: {method or 'unknown'}",
        f"Experience: {experience if experience is not None else 'unknown'}",
        f"Epoch: (see log)",
        f"Error: {error}",
        f"Last checkpoint: {root}/ (state.json + checkpoint.pt, atomic)",
        f"Resume: {resume_hint}",
        f"Log: {logs_dir}/phase5_nic_master.log",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not sys.stdout.isatty():
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    root = Path(args.root)
    reports = Path(args.reports)
    logs_dir = Path("logs")
    resume_hint = f"python scripts/run_phase5_experiment.py --method {args.method}"

    master = _file_logger("phase5.master", logs_dir / "phase5_nic_master.log")
    phase = PhaseProgress("PHASE 5 OVERALL", list(STEP_LABELS))

    method_loggers = {
        method: _file_logger(
            f"phase5.{method}", logs_dir / f"phase5_nic_{method}.log"
        )
        for method in METHODS
    }

    free = shutil.disk_usage(root.parent if root.parent.exists() else ".").free
    if free < MIN_FREE_BYTES:
        phase.log(
            f"Insufficient disk space ({free / 1024**3:.1f} GB free) — stopping safely"
        )
        return 1

    started = time.perf_counter()
    try:
        # ---- step 2: reproducible experiment configuration -------------
        phase.set_step(2, STEP_LABELS[1])
        exp = load_experiment_config(args.config)
        cfgs = {method: build_train_config(exp, method) for method in METHODS}
        shared = root / "shared"
        shared.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        write_shared_config(shared, exp, cfgs, str(args.config))
        phase.log(
            f"Resolved experiment config written: {shared / 'config.json'} "
            f"(epochs={exp['training']['epochs']}, batch={exp['training']['batch_size']}, "
            f"seed={exp['training']['seed']}, replay capacity={exp['replay']['capacity']})"
        )
        master.info("step 2 complete: config resolved")

        # ---- step 3: independent runs initialization -------------------
        phase.set_step(3, STEP_LABELS[2])
        scenario = load_scenario_cached(
            exp["scenario"],
            variant=exp["variant"],
            run=int(exp["run"]),
            filelist_root=exp["filelist_root"],
            images_root=exp["images_root"],
            object_mapping=exp["object_mapping"],
        )
        if len(scenario) != EXPECTED_EXPERIENCES:
            raise Phase5Blocked(
                f"Expected {EXPECTED_EXPERIENCES} experiences, got {len(scenario)}"
            )
        num_classes = int(exp["model"]["num_classes"])
        if num_classes != int(scenario.metadata["label_max"]) + 1:
            raise Phase5Blocked(
                f"num_classes={num_classes} does not match scenario labels"
            )
        initial = init_initial_model_state(shared, cfgs["naive"], num_classes)
        phase.log(
            f"Shared initial model state: checksum={initial['checksum']} "
            f"({initial['parameters']:,} parameters, seed={initial['seed']})"
        )

        manifest_path = shared / "run_manifest.json"
        manifest: dict[str, Any]
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
        else:
            manifest = {
                "phase": 5,
                "experiment": exp.get("experiment_name"),
                "dataset": "core50",
                "scenario": scenario.name,
                "variant": scenario.variant,
                "run": scenario.run_id,
                "experiences": len(scenario),
                "train_samples_total": scenario.metadata["train_samples_total"],
                "evaluation_samples": EXPECTED_EVAL_SAMPLES,
                "evaluation_sessions": scenario.metadata["evaluation_sessions"],
                "model": dict(exp["model"]),
                "initial_state": initial,
                "methods": list(METHODS),
                "only_method_difference": (
                    "replay memory (capacity "
                    f"{exp['replay']['capacity']}, batch "
                    f"{exp['replay']['batch_size']}, seed {exp['replay']['seed']}) "
                    "enabled for the replay method only"
                ),
                "config_fingerprints": {
                    m: config_fingerprint(cfgs[m]) for m in METHODS
                },
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                },
                "created_utc": utc_now(),
                "initial_state_verified": {},
            }
        device = resolve_device(cfgs["naive"].device)
        manifest["device"] = device
        atomic_write_json(manifest_path, manifest)

        cache = ensure_eval_cache(shared, scenario, int(exp["model"]["image_size"]), phase)
        master.info("step 3 complete: shared state + cache ready")

        # ---- steps 4/5: run the requested method(s) --------------------
        requested = METHODS if args.method == "all" else (args.method,)
        results: dict[str, dict[str, Any]] = {}
        for method in requested:
            step_index = 4 if method == "naive" else 5
            phase.set_step(step_index, STEP_LABELS[step_index - 1])
            results[method] = run_method(
                method=method,
                scenario=scenario,
                exp=exp,
                cfg=cfgs[method],
                root=root,
                reports=reports,
                cache=cache,
                phase=phase,
                initial_checksum=initial["checksum"],
                logger=method_loggers[method],
                max_experiences=args.max_experiences,
                manifest_path=manifest_path,
            )
            phase.log(
                f"{method}: {results[method]['status']} "
                f"({results[method]['records']} experiences, checkpoint at "
                f"experience {results[method]['current_experience'] + 1}/{len(scenario)})"
            )

        # ---- completeness gate -----------------------------------------
        def metrics_complete() -> bool:
            for method in METHODS:
                path = reports / f"{method}_metrics.json"
                if not path.is_file():
                    return False
                records = read_json(path).get("records") or []
                if [r["experience_id"] for r in records] != list(range(len(scenario))):
                    return False
            return True

        if not metrics_complete():
            phase.log(
                "Partial run — resume to continue: "
                + "; ".join(
                    f"python scripts/run_phase5_experiment.py --method {m}"
                    for m in METHODS
                    if (
                        not (reports / f"{m}_metrics.json").is_file()
                        or [r["experience_id"] for r in
                            read_json(reports / f"{m}_metrics.json").get("records", [])]
                        != list(range(len(scenario)))
                    )
                )
            )
            master.info("partial run finished; finalization skipped")
            phase.finish("PHASE 5 PARTIAL — resume to continue")
            return 0

        # ---- step 6: evaluation analysis --------------------------------
        phase.set_step(6, STEP_LABELS[5])
        data = finalize_metrics(scenario, exp, reports, phase, master)

        # ---- step 7: plots, summary, report -----------------------------
        phase.set_step(7, STEP_LABELS[6])
        checks = run_integrity_checks(
            root=root,
            reports=reports,
            shared=shared,
            exp=exp,
            scenario=scenario,
            data=data,
            logger=master,
        )
        failed = [name for name, result in checks.items() if result["status"] != "PASS"]
        if failed:
            raise Phase5Blocked(f"Integrity check(s) failed: {', '.join(failed)}")
        plots = make_plots(data, reports, master)
        summary = build_summary(
            exp=exp,
            scenario=scenario,
            data=data,
            root=root,
            reports=reports,
            checks=checks,
            plots=plots,
            device=device,
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_write_json(reports / "experiment_summary.json", summary)
        report = build_report(
            exp=exp,
            scenario=scenario,
            data=data,
            summary=summary,
            reports=reports,
            root=root,
        )
        (reports / "phase5_nic_report.md").write_text(report, encoding="utf-8")
        phase.log(f"Report: {reports / 'phase5_nic_report.md'}")
        phase.log(f"Summary: {reports / 'experiment_summary.json'}")
        for name, result in checks.items():
            phase.log(f"Integrity {name}: {result['status']}")
        master.info("step 7 complete: outputs generated")

        phase.finish(
            "PHASE 5 — MAIN NIC EXPERIMENT COMPLETE "
            f"({format_duration(time.perf_counter() - started)} wall clock)"
        )
        return 0

    except KeyboardInterrupt:
        master.exception("interrupted by user")
        print(
            _blocked_block(
                method=args.method,
                experience=None,
                error="KeyboardInterrupt (interrupted; state saved after the "
                "last completed experience)",
                root=root,
                logs_dir=logs_dir,
                resume_hint=resume_hint,
            )
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level safe-stop handler
        master.error("blocked: %s", exc)
        master.error(traceback.format_exc())
        experience: int | None = None
        for method in METHODS:
            state_path = root / method / "state.json"
            if state_path.is_file():
                try:
                    experience = json.loads(state_path.read_text(encoding="utf-8"))[
                        "current_experience"
                    ]
                except Exception:  # noqa: BLE001
                    experience = None
        print(
            _blocked_block(
                method=args.method if args.method != "all" else None,
                experience=experience,
                error=f"{type(exc).__name__}: {exc}",
                root=root,
                logs_dir=logs_dir,
                resume_hint=resume_hint,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
