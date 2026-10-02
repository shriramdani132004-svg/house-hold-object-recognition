"""Phase 6 tests: training config, scripts, artifacts and reports.

Synthetic fixtures cover the pure helpers; the real training artifacts
(configs, weights, results.csv, reports) are read-only assertions that run
after Phase 6 validation has completed.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CLASSES = [
    "chair",
    "book",
    "bottle",
    "cup",
    "dining table",
    "bowl",
    "potted plant",
    "wine glass",
    "cell phone",
    "clock",
    "tv",
    "couch",
    "remote",
    "sink",
    "laptop",
    "bed",
    "keyboard",
    "refrigerator",
    "mouse",
]

RUN_DIR = PROJECT_ROOT / "models" / "training" / "household_yolo26n"


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_script = _load_script("train_model")
validate_script = _load_script("validate_training")
bench_script = _load_script("benchmark_training_speed")


def _train_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "train.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _post_state_fixture() -> dict[str, Any]:
    return {
        "splits": {
            "train": {"images": 40890, "labels": 40890, "instances": 184709},
            "val": {"images": 4544, "labels": 4544, "instances": 21100},
            "test": {"images": 1965, "labels": 1965, "instances": 9174},
        },
        "instances_total": 214983,
        "raw_files": 123298,
        "raw_bytes": 41368946252,
    }


def _epoch_record(epoch: int, previous: float) -> dict[str, Any]:
    elapsed = previous + 20000.0
    return {
        "epoch": epoch,
        "wall_s": elapsed,
        "epoch_wall_s": elapsed - previous,
        "train_box_loss": 1.4,
        "train_cls_loss": 3.0,
        "train_l1_loss": 0.01,
        "precision": 0.6,
        "recall": 0.5,
        "map50": 0.54,
        "map50_95": 0.38,
        "val_box_loss": 1.4,
        "val_cls_loss": 2.5,
        "lr_pg0": 0.001,
    }


def _report_ctx() -> dict[str, Any]:
    config = _train_config()
    per_class = [
        {
            "class_id": index,
            "class_name": name,
            "ground_truth": 100,
            "images": 50,
            "precision": 0.6,
            "recall": 0.5,
            "map50": 0.54,
            "map50_95": 0.38,
        }
        for index, name in enumerate(EXPECTED_CLASSES)
    ]
    validation = {
        "overall": {"precision": 0.621, "recall": 0.501, "map50": 0.541, "map50_95": 0.378, "fitness": 0.394},
        "per_class": per_class,
        "speed_ms_per_image": {"inference": 47.6},
        "weights": "models/training/household_yolo26n/weights/best.pt",
        "save_dir": "models/training/household_yolo26n_val_best",
    }
    baseline_val = {
        "overall": {"precision": 0.7, "recall": 0.39, "map50": 0.55, "map50_95": 0.4},
        "per_class": [
            {**row, "precision": 0.7, "recall": 0.39, "map50": 0.55, "map50_95": 0.4}
            for row in per_class
        ],
    }
    deltas = validate_script.compute_deltas(validation["overall"], baseline_val["overall"])
    return {
        "generated_at": "2026-10-02T18:30:00+00:00",
        "config": config,
        "env": validate_script.environment_info(),
        "train_plan": {"model": "yolo26n.pt"},
        "train_done": {"status": "completed", "epochs_completed": 2, "wall_s": 46086.5},
        "epochs": [_epoch_record(1, 0.0), _epoch_record(2, 20000.0)],
        "integrity_post": _post_state_fixture(),
        "integrity_match": True,
        "validation": validation,
        "baseline_val": baseline_val,
        "phase5_test": {
            "metrics": {"precision": 0.6924, "recall": 0.3914, "map50": 0.5485, "map50_95": 0.3961},
            "timing": {"avg_wall_ms_per_image": 415.6, "images_per_second": 2.41},
        },
        "speed": {
            "harness": "per-image model.predict, batch=1, identical to Phase 5 baseline harness",
            "image_count": 1965,
            "model_load_ms": 41.0,
            "first_image_ms": 900.0,
            "avg_wall_ms_per_image": 410.0,
            "warm_avg_wall_ms_per_image": 405.0,
            "avg_model_ms_per_image": 350.0,
            "total_inference_s": 800.0,
            "images_per_second": 2.44,
            "warmup_excluded_images": 20,
        },
        "deltas_val": deltas,
        "comparison_note": validate_script.comparison_note(deltas),
        "console_log_bytes": 1000,
        "figures_copied": ["results.png"],
        "artifacts": [{"path": "models/training/household_yolo26n/weights/best.pt", "bytes": 5375173}],
    }


def test_train_config_values() -> None:
    config = _train_config()
    training = config["training"]
    assert 1 <= training["epochs"] <= 50
    assert training["epochs"] == 2
    assert training["patience"] == 15
    assert training["seed"] == 42
    assert training["deterministic"] is True
    assert training["device"] == "cpu"
    assert training["imgsz"] == 640
    assert training["batch"] >= 1
    assert 0 < training.get("warmup_epochs", 3.0) <= training["epochs"]
    assert config["data"] == "configs/train_data.yaml"
    assert config["output"] == {"project": "models/training", "name": "household_yolo26n"}


def test_train_data_yaml_excludes_test_and_matches_classes() -> None:
    with (PROJECT_ROOT / "configs" / "train_data.yaml").open(encoding="utf-8") as handle:
        dataset = yaml.safe_load(handle)
    assert "test" not in dataset
    assert dataset["nc"] == 19
    names = list(dataset["names"])
    assert names == EXPECTED_CLASSES
    with (PROJECT_ROOT / "configs" / "classes.yaml").open(encoding="utf-8") as handle:
        classes = yaml.safe_load(handle)
    assert list(classes["classes"].values()) == names
    dataset_root = PROJECT_ROOT / dataset["path"]
    assert (dataset_root / dataset["train"]).is_dir()
    assert (dataset_root / dataset["val"]).is_dir()


def test_preflight_accepts_real_config() -> None:
    checked = train_script.preflight(_train_config(), PROJECT_ROOT)
    assert checked["epochs"] == 2
    assert len(checked["names"]) == 19


def test_build_kwargs_enforces_epoch_cap() -> None:
    config = _train_config()
    kwargs = train_script.build_kwargs(config, PROJECT_ROOT, None)
    assert kwargs["epochs"] == 2
    assert kwargs["resume"] is False
    assert kwargs["seed"] == 42
    assert kwargs["warmup_epochs"] == 0.5
    with pytest.raises(ValueError, match="epochs"):
        train_script.build_kwargs(config, PROJECT_ROOT, 51)


def test_plan_run_resumes_and_renames(tmp_path: Path) -> None:
    project = tmp_path / "models"
    name = "run"
    kwargs = {"project": str(project), "name": name}
    run_dir, resume, renamed = train_script.plan_run(kwargs, resume_requested=False, execute=False)
    assert run_dir == project / name
    assert resume is False
    assert renamed is None
    run_dir.mkdir(parents=True)
    _, resume, renamed = train_script.plan_run(kwargs, resume_requested=False, execute=False)
    assert resume is False
    assert renamed is not None and "abandoned" in renamed
    weights = run_dir / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"x")
    _, resume, renamed = train_script.plan_run(kwargs, resume_requested=False, execute=False)
    assert resume is True
    assert renamed is None
    shutil.rmtree(weights)
    with pytest.raises(FileNotFoundError):
        train_script.plan_run(kwargs, resume_requested=True, execute=False)


def test_results_rows_and_oom_helpers(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,time,metrics/mAP50(B)\n1,100.5,0.5\n", encoding="utf-8")
    rows = train_script.results_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["metrics/mAP50(B)"] == "0.5"
    assert train_script.results_rows(tmp_path / "missing.csv") == []
    assert train_script.is_oom(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert train_script.is_oom(MemoryError())
    assert not train_script.is_oom(ValueError("bad value"))


def test_epoch_records_typing() -> None:
    rows = [
        {
            "epoch": "1",
            "time": "20488.9",
            "train/box_loss": "1.43",
            "train/cls_loss": "3.26",
            "train/l1_loss": "0.01",
            "metrics/precision(B)": "0.558",
            "metrics/recall(B)": "0.433",
            "metrics/mAP50(B)": "0.451",
            "metrics/mAP50-95(B)": "0.298",
            "val/box_loss": "1.45",
            "val/cls_loss": "2.48",
            "lr/pg0": "0.00043",
        }
    ]
    records = validate_script.epoch_records(rows)
    assert records[0]["epoch"] == 1
    assert records[0]["map50"] == pytest.approx(0.451)
    assert records[0]["epoch_wall_s"] == pytest.approx(20488.9)


def test_integrity_matches_fixture() -> None:
    assert validate_script.integrity_matches(_post_state_fixture()) is True
    broken = _post_state_fixture()
    broken["splits"]["test"]["images"] = 1964
    assert validate_script.integrity_matches(broken) is False


def test_deltas_and_comparison_note() -> None:
    trained = {"precision": 0.62, "recall": 0.5, "map50": 0.54, "map50_95": 0.38}
    zero_shot = {"precision": 0.69, "recall": 0.39, "map50": 0.55, "map50_95": 0.4}
    deltas = validate_script.compute_deltas(trained, zero_shot)
    assert deltas["precision_delta"] == pytest.approx(-0.07)
    note = validate_script.comparison_note(deltas)
    assert "recall" in note and "mAP50" in note


def test_report_render_contains_all_19_sections() -> None:
    report = validate_script.render_report(_report_ctx())
    sections = [int(value) for value in re.findall(r"^## (\d+)\. ", report, flags=re.MULTILINE)]
    assert sections == list(range(1, 20))
    assert "VALIDATION RESULT" in report
    assert "never used" in report
    assert "Zero-shot mAP50 (val)" in report
    assert report.count("| 18 | mouse |") == 1


def test_summary_and_commands_render() -> None:
    ctx = _report_ctx()
    summary = validate_script.render_summary(ctx)
    assert "Phase 6 - Training Summary" in summary
    assert "Headline metrics" in summary
    commands = validate_script.render_commands(ctx)
    for script_name in ("train_model.py", "validate_training.py", "benchmark_training_speed.py"):
        assert script_name in commands
    assert "pytest tests -q" in commands


def test_benchmark_script_defaults() -> None:
    args = bench_script.parse_args([])
    assert args.workers == 0
    assert args.batch == 16
    assert args.fraction == 0.05
    assert "bench" in str(args.out)


def test_no_absolute_paths_in_committed_sources() -> None:
    targets = [
        PROJECT_ROOT / "configs" / "train.yaml",
        PROJECT_ROOT / "configs" / "train_data.yaml",
        PROJECT_ROOT / "scripts" / "train_model.py",
        PROJECT_ROOT / "scripts" / "validate_training.py",
        PROJECT_ROOT / "scripts" / "benchmark_training_speed.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users" not in text, path
        assert "/home/" not in text, path


def test_gitignore_allows_phase6_figures() -> None:
    text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!reports/figures/phase6/" in text


def test_training_artifacts_exist() -> None:
    best = RUN_DIR / "weights" / "best.pt"
    last = RUN_DIR / "weights" / "last.pt"
    assert best.is_file() and best.stat().st_size > 1_000_000
    assert last.is_file() and last.stat().st_size > 1_000_000
    with (RUN_DIR / "results.csv").open(encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()
    assert len(rows) == 3
    with (RUN_DIR / "args.yaml").open(encoding="utf-8") as handle:
        run_args = yaml.safe_load(handle)
    assert run_args["epochs"] == 2
    assert run_args["seed"] == 42
    assert run_args["deterministic"] is True
    assert run_args["patience"] == 15
    assert run_args["fraction"] == 1.0
    assert "train_data.yaml" in str(run_args["data"])
    console = PROJECT_ROOT / "models" / "training" / "train_console.log"
    assert console.is_file()
    text = console.read_text(encoding="utf-8", errors="replace")
    assert '"status": "completed"' in text
    assert "TRAIN_PLAN" in text


def test_results_json_artifact() -> None:
    path = PROJECT_ROOT / "reports" / "phase6_training_results.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["phase"] == 6
    assert data["status"] == "completed"
    assert data["experiment"]["epochs_completed"] == 2
    assert data["experiment"]["epochs_planned"] <= 50
    assert data["experiment"]["seed"] == 42
    assert data["experiment"]["early_stop_triggered"] is False
    assert data["environment"]["device_used"] == "cpu"
    assert data["environment"]["xpu_available"] is False
    assert data["integrity"]["unchanged"] is True
    assert data["integrity"]["pre_state_recorded_at_phase_start"]["test_images"] == 1965
    validation = data["validation_best_pt"]
    assert validation["image_count"] == 4544
    assert validation["instance_count"] == 21100
    assert len(validation["per_class"]) == 19
    assert set(validation["per_class"][0]) >= {"class_name", "precision", "recall", "map50", "map50_95"}
    overall = validation["overall"]
    assert 0.0 <= overall["map50"] <= 1.0
    assert 0.0 <= overall["map50_95"] <= 1.0
    baseline = data["zero_shot_baseline_val"]
    assert baseline is not None and baseline["image_count"] == 4544
    comparison = data["controlled_comparison_val_split"]["trained_vs_zero_shot"]
    for key in ("precision_delta", "recall_delta", "map50_delta", "map50_95_delta"):
        assert key in comparison
    assert data["training"]["per_epoch"][0]["epoch"] == 1
    assert data["training"]["per_epoch"][-1]["epoch"] == 2
    assert data["speed_benchmark"]["image_count"] == 1965
    assert data["phase5_baseline_test_reference"]["metrics"]["map50"] == pytest.approx(0.548459)


def test_report_markdown_artifact() -> None:
    text = (PROJECT_ROOT / "reports" / "phase6_training_report.md").read_text(encoding="utf-8")
    sections = [int(value) for value in re.findall(r"^## (\d+)\. ", text, flags=re.MULTILINE)]
    assert sections == list(range(1, 20))
    assert "VALIDATION RESULT" in text
    assert "held out" in text or "never used" in text
    for figure in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"):
        assert (PROJECT_ROOT / "reports" / "figures" / "phase6" / figure).is_file()
        assert f"figures/phase6/{figure}" in text


def test_summary_and_commands_artifacts() -> None:
    summary = (PROJECT_ROOT / "reports" / "phase6_training_summary.md").read_text(encoding="utf-8")
    assert "VALIDATION RESULT" in summary
    assert "best.pt" in summary
    commands = (PROJECT_ROOT / "reports" / "phase6_training_commands.md").read_text(encoding="utf-8")
    assert "scripts\\train_model.py" in commands
    assert "scripts\\validate_training.py" in commands


def test_prepared_split_counts_unchanged() -> None:
    dataset_root = PROJECT_ROOT / "data" / "processed" / "household_objects"
    expected = {"train": 40890, "val": 4544, "test": 1965}
    for split, count in expected.items():
        images = [p for p in (dataset_root / "images" / split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = [p for p in (dataset_root / "labels" / split).iterdir() if p.suffix.endswith(".txt")]
        assert len(images) == count, split
        assert len(labels) == count, split


def test_train_script_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train_model.py", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    assert "TRAIN_PLAN" in completed.stdout
    assert '"epochs": 2' in completed.stdout
    assert '"classes": 19' in completed.stdout
