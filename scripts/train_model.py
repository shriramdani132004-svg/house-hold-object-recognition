"""Run the Phase 6 custom YOLO training experiment.

Reads configs/train.yaml, validates that the data YAML is train/val only,
manages the run directory (resume from last.pt, rename stale incomplete
runs), and fine-tunes the COCO-pretrained base model. Safe to re-run:
a completed run is reported as done, an interrupted run resumes, and a
stale run without checkpoints is renamed out of the way. Out-of-memory
errors trigger one batch-size reduction and a retry.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line options for the training script."""
    parser = argparse.ArgumentParser(description="Phase 6 custom YOLO training")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs (1-50)")
    parser.add_argument("--resume", action="store_true", help="Force resume from last.pt")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan and exit")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict:
    """Load a YAML file from the project tree."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_model_weights(base: str, root: Path) -> str:
    """Find the base weights locally before letting Ultralytics download."""
    candidates = [root / base, root / "models" / "baseline" / Path(base).name, Path(base)]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return base


def preflight(cfg: dict, root: Path) -> dict:
    """Validate config and dataset layout before touching the trainer."""
    data_yaml = root / cfg["data"]
    dataset = load_yaml(data_yaml)
    if "test" in dataset:
        raise ValueError("Training data YAML must not contain a test split")
    dataset_root = root / dataset["path"]
    for split in ("train", "val"):
        split_dir = dataset_root / dataset[split]
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {split_dir}")
    names = dataset.get("names") or []
    if isinstance(names, dict):
        names = list(names.values())
    if int(dataset.get("nc", 0)) != len(names) or len(names) != 19:
        raise ValueError(f"Expected 19 classes, got nc={dataset.get('nc')} len={len(names)}")
    classes_yaml = load_yaml(root / "configs" / "classes.yaml")
    class_list = classes_yaml.get("classes", classes_yaml.get("names", []))
    if isinstance(class_list, dict):
        class_list = list(class_list.values())
    if [str(c) for c in class_list] != [str(n) for n in names]:
        raise ValueError("Class order in configs/train_data.yaml differs from configs/classes.yaml")
    training = cfg["training"]
    epochs = int(training["epochs"])
    if not 1 <= epochs <= 50:
        raise ValueError(f"epochs must be within 1-50, got {epochs}")
    return {"data_yaml": data_yaml, "names": names, "epochs": epochs}


def build_kwargs(cfg: dict, root: Path, epochs_override: int | None) -> dict:
    """Map the training config onto Ultralytics train arguments."""
    training = cfg["training"]
    epochs = epochs_override if epochs_override is not None else int(training["epochs"])
    if not 1 <= epochs <= 50:
        raise ValueError(f"epochs must be within 1-50, got {epochs}")
    return {
        "data": str(root / cfg["data"]),
        "epochs": epochs,
        "patience": int(training["patience"]),
        "imgsz": int(training["imgsz"]),
        "batch": int(training["batch"]),
        "seed": int(training["seed"]),
        "deterministic": bool(training["deterministic"]),
        "workers": int(training["workers"]),
        "device": str(training["device"]),
        "optimizer": str(training["optimizer"]),
        "val": bool(training["val"]),
        "amp": bool(training["amp"]),
        "close_mosaic": int(training["close_mosaic"]),
        "save_period": int(training["save_period"]),
        "plots": bool(training["plots"]),
        "warmup_epochs": float(training.get("warmup_epochs", 3.0)),
        "project": str(root / cfg["output"]["project"]),
        "name": cfg["output"]["name"],
        "exist_ok": True,
        "resume": False,
    }


def results_rows(results_csv: Path) -> list[dict]:
    """Read the per-epoch metrics CSV written by the trainer."""
    if not results_csv.is_file():
        return []
    with results_csv.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plan_run(kwargs: dict, resume_requested: bool, execute: bool) -> tuple[Path, bool, str | None]:
    """Decide whether to resume, start fresh, or rename a stale run."""
    run_dir = Path(kwargs["project"]) / kwargs["name"]
    weights_last = run_dir / "weights" / "last.pt"
    if weights_last.is_file() or resume_requested:
        if not weights_last.is_file():
            raise FileNotFoundError(f"--resume requested but {weights_last} does not exist")
        return run_dir, True, None
    renamed = None
    if run_dir.is_dir():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stale = run_dir.with_name(f"{run_dir.name}_abandoned_{stamp}")
        if execute:
            shutil.move(str(run_dir), str(stale))
        renamed = str(stale)
    return run_dir, False, renamed


def is_oom(error: BaseException) -> bool:
    """Return True when the failure is a memory-allocation problem."""
    message = str(error).lower()
    return isinstance(error, MemoryError) or "out of memory" in message or "can't allocate memory" in message


def run_training(kwargs: dict, model_path: str) -> dict:
    """Execute one training attempt, retrying once with a smaller batch on OOM."""
    from ultralytics import YOLO

    attempt_kwargs = dict(kwargs)
    attempts = 0
    while True:
        attempts += 1
        started = time.time()
        try:
            model = YOLO(model_path)
            model.train(**attempt_kwargs)
            return {"wall_s": time.time() - started, "batch": attempt_kwargs["batch"], "attempts": attempts}
        except Exception as error:  # noqa: BLE001 - OOM surfaces with several exception types
            if attempts >= 2 or not is_oom(error):
                raise
            reduced = max(4, int(attempt_kwargs["batch"]) // 2)
            print(f"OOM at batch={attempt_kwargs['batch']}, retrying with batch={reduced}", flush=True)
            attempt_kwargs["batch"] = reduced
            last_pt = Path(kwargs["project"]) / kwargs["name"] / "weights" / "last.pt"
            attempt_kwargs["resume"] = last_pt.is_file()


def summarize(status: str, run_dir: Path, failure: str | None, wall_s: float | None) -> dict:
    """Build the machine-readable end-of-run report."""
    results_csv = run_dir / "results.csv"
    rows = results_rows(results_csv)
    weights = run_dir / "weights"
    report: dict = {
        "status": status,
        "epochs_completed": len(rows),
        "run_dir": str(run_dir),
        "best_pt": (weights / "best.pt").is_file(),
        "last_pt": (weights / "last.pt").is_file(),
        "results_csv": str(results_csv),
        "failure": failure,
        "wall_s": round(wall_s, 1) if wall_s else None,
    }
    if rows:
        last = rows[-1]
        report["final_metrics"] = {
            "precision": float(last.get("metrics/precision(B)", 0.0)),
            "recall": float(last.get("metrics/recall(B)", 0.0)),
            "map50": float(last.get("metrics/mAP50(B)", 0.0)),
            "map50_95": float(last.get("metrics/mAP50-95(B)", 0.0)),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point: plan, run and report the training experiment."""
    args = parse_args(argv)
    root = PROJECT_ROOT
    os.chdir(root)
    cfg = load_yaml(root / args.config)
    checked = preflight(cfg, root)
    kwargs = build_kwargs(cfg, root, args.epochs)
    model_path = resolve_model_weights(cfg["model"]["base"], root)
    run_dir, resume, renamed = plan_run(kwargs, args.resume, execute=not args.dry_run)

    results_csv = run_dir / "results.csv"
    completed = results_rows(results_csv)
    plan = {
        "config": args.config,
        "model": model_path,
        "data": cfg["data"],
        "epochs": kwargs["epochs"],
        "batch": kwargs["batch"],
        "seed": kwargs["seed"],
        "device": kwargs["device"],
        "run_dir": str(run_dir),
        "resume": resume,
        "renamed_stale_run": renamed,
        "epochs_already_completed": len(completed),
        "classes": len(checked["names"]),
    }
    print("TRAIN_PLAN " + json.dumps(plan), flush=True)
    if args.dry_run:
        return 0
    if len(completed) >= kwargs["epochs"]:
        print("TRAIN_DONE " + json.dumps(summarize("already_complete", run_dir, None, None)), flush=True)
        return 0

    kwargs["resume"] = resume
    if resume:
        model_path = str(run_dir / "weights" / "last.pt")
    started = time.time()
    status = "completed"
    failure: str | None = None
    try:
        run_training(kwargs, model_path)
    except Exception as error:  # noqa: BLE001 - report failure details before propagating
        status = "failed"
        failure = f"{type(error).__name__}: {error}"
        print(f"TRAIN_FAILED {failure}", flush=True)
        traceback.print_exc()
    report = summarize(status, run_dir, failure, time.time() - started)
    print("TRAIN_DONE " + json.dumps(report), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
