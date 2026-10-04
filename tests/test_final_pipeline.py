"""Tests for the one-shot final pipeline (config, cache plumbing, scripts).

Covers the Phase 14-33 engineering surface that the final run depends on:
the pinned final config file, ``torch_threads`` validation, replay-cache
pass-through, augmentation bounds, driver argument validation, the
freeze script's refusal conditions, and the evaluation script's
held-out metric helpers (pure functions only — the held-out cache
itself is never read here).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts.freeze_final_model import EXPECTED_BASELINE_SHA256, sha256_file
from scripts.evaluate_final import old_new_split, session_accuracy
from scripts.train_candidates import build_config
from src.data.continual import SampleRecord
from src.training.config import ContinualConfigError, ContinualTrainConfig, config_fingerprint
from src.training.dataset import ContinualImageDataset
from src.training.improved import _augment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = PROJECT_ROOT / "configs/final_training.yaml"
BASELINE_PATH = PROJECT_ROOT / "models/continual/baseline_replay_phase5.pt"
SCRIPT = PROJECT_ROOT / "scripts/train_candidates.py"


def _record(relative_path: str = "s1/o1/C_01_01_000.png", label: int = 0) -> SampleRecord:
    return SampleRecord(
        relative_path=relative_path,
        label=label,
        split="train",
        experience_id=0,
        source_filelist="synthetic/train_filelist.txt",
        line_number=1,
        object_id=1,
        session_id=1,
        category_id=0,
        category_name="synthetic",
        object_name="synthetic_object",
    )


class _FakeCache:
    def __init__(self, entries: dict[str, torch.Tensor]) -> None:
        self._entries = entries
        self.gets = 0

    def __contains__(self, path: str) -> bool:
        return path in self._entries

    def get(self, path: str) -> torch.Tensor:
        self.gets += 1
        return self._entries[path]


def test_final_training_config_is_loadable_and_pinned() -> None:
    payload = yaml.safe_load(FINAL_CONFIG.read_text(encoding="utf-8"))
    config = build_config(payload, epochs_override=None)
    assert config.method == "replay"
    assert config.model_arch == "compact_resnet"
    assert config.image_size == 64
    assert config.batch_size == 64
    assert config.optimizer == "adamw"
    assert config.replay_policy == "reservoir"
    assert config.replay_capacity == 10000
    assert config.replay_batch_size == 32
    assert config.label_smoothing == 0.1
    assert config.augment is True
    assert config.epochs == 12
    assert config.early_stop_patience == 5
    assert config.seed == 42
    assert config.torch_threads == 8
    assert config_fingerprint(config) == config_fingerprint(
        build_config(payload, epochs_override=None)
    )


def test_torch_threads_validation() -> None:
    assert ContinualTrainConfig().torch_threads == 8
    with pytest.raises(ContinualConfigError, match="torch_threads"):
        ContinualTrainConfig(torch_threads=0)
    with pytest.raises(ContinualConfigError, match="torch_threads"):
        ContinualTrainConfig(torch_threads=65)


def test_dataset_serves_from_cache_without_touching_disk(tmp_path: Path) -> None:
    record = _record()
    cached = torch.full((3, 8, 8), 7, dtype=torch.uint8)
    cache = _FakeCache({record.relative_path: cached})
    dataset = ContinualImageDataset(
        [record], tmp_path, 8, require_split="train", cache=cache
    )
    tensor = dataset.get_uint8(0)
    assert torch.equal(tensor, cached)
    assert cache.gets == 1
    image, label = dataset[0]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label == 0


def test_augmentation_stays_bounded_and_different() -> None:
    torch.manual_seed(0)
    base = torch.zeros(3, 32, 32)
    seen_change = False
    for _ in range(30):
        out = _augment(base.clone())
        assert out.shape == (3, 32, 32)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= -1.0
        assert float(out.max()) <= 1.0
        if not torch.equal(out, base):
            seen_change = True
    assert seen_change


def test_driver_rejects_ambiguous_arguments() -> None:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )

    result = run("--final")
    assert result.returncode == 2
    assert "--final requires" in result.stderr

    result = run("--config", "configs/final_training.yaml")
    assert result.returncode == 2
    assert "only valid together with --final" in result.stderr

    result = run(
        "--final", "--config", "configs/final_training.yaml", "--from", "c1_corrected_baseline"
    )
    assert result.returncode == 2
    assert "only one of" in result.stderr

    result = run("--final", "--config", "configs/does_not_exist.yaml")
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_freeze_refuses_when_final_run_is_missing(tmp_path: Path) -> None:
    from scripts.freeze_final_model import freeze

    with pytest.raises(SystemExit, match="dev_metrics.json"):
        freeze(tmp_path, tmp_path / "final_model.pt")

    (tmp_path / "dev_metrics.json").write_text(
        '{"records": []}', encoding="utf-8"
    )
    (tmp_path / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="development records are empty"):
        freeze(tmp_path, tmp_path / "final_model.pt")

    (tmp_path / "dev_metrics.json").write_text(
        '{"records": [{"experience_id": 0, "accuracy": {"overall": 0.5},'
        ' "classes_introduced": [0]}]}',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="expected records for all 79"):
        freeze(tmp_path, tmp_path / "final_model.pt")


def test_pinned_baseline_is_unchanged() -> None:
    assert BASELINE_PATH.is_file()
    assert sha256_file(BASELINE_PATH) == EXPECTED_BASELINE_SHA256


def test_session_accuracy_requires_exactly_held_out_sessions() -> None:
    targets = torch.tensor([0, 1, 2, 3])
    predictions = torch.tensor([0, 0, 2, 4])
    paths = (
        "s3/o1/C_01_01_000.png",
        "s7/o2/C_07_01_000.png",
        "s10/o3/C_10_01_000.png",
        "s3/o4/C_03_01_000.png",
    )
    result = session_accuracy(targets, predictions, paths)
    assert set(result) == {"3", "7", "10"}
    assert result["3"]["n"] == 2

    with pytest.raises(SystemExit, match="expected held-out sessions"):
        session_accuracy(
            targets,
            predictions,
            ("s3/o1/x.png", "s7/o2/x.png", "s1/o3/x.png", "s3/o4/x.png"),
        )
    with pytest.raises(SystemExit, match="cannot parse session"):
        session_accuracy(
            targets,
            predictions,
            ("s3/o1/x.png", "s7/o2/x.png", "train/o3/x.png", "s3/o4/x.png"),
        )


def test_old_new_split_definition() -> None:
    targets = torch.tensor([0, 1, 2, 3])
    predictions = torch.tensor([0, 0, 2, 4])
    result = old_new_split(targets, predictions, new_classes={3})
    assert result["new_classes"] == [3]
    assert result["old_n"] == 3
    assert result["new_n"] == 1
    assert result["old_accuracy"] == pytest.approx(2 / 3)
    assert result["new_accuracy"] == pytest.approx(0.0)

    with pytest.raises(SystemExit, match="empty new-class set"):
        old_new_split(targets, predictions, new_classes=set())
    with pytest.raises(SystemExit, match="empty side"):
        old_new_split(targets, predictions, new_classes={0, 1, 2, 3})
