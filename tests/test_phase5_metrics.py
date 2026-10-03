"""Focused Phase-5 tests: continual evaluation metrics + resume support.

Tiny synthetic fixtures only — no CORe50 training runs, no full-dataset
scans. The whole file must finish in seconds.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.data.continual import (
    ContinualExperience,
    ContinualScenario,
    SampleRecord,
)
from src.evaluation.continual import (
    EvalCache,
    EvalCacheError,
    atomic_write_json,
    build_metric_record,
    compute_forgetting,
    evaluate_model,
    load_class_names,
    read_json,
)
from src.training import (
    ContinualStateError,
    ContinualTrainConfig,
    NaiveContinualTrainer,
    checkpoint_exists,
    load_state,
    model_checksum,
)
from src.utils.experiment_display import ExperimentDisplay

IMG_SIZE = 12
NUM_CLASSES = 3
SAMPLES_PER_EXP = 4
EVAL_SAMPLES = 6


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color).save(path)


def make_mini_scenario(root: Path) -> ContinualScenario:
    """2 experiences x 4 train samples + shared 6-sample evaluation set."""
    images_root = root / "images"
    eval_records: list[SampleRecord] = []
    for k in range(EVAL_SAMPLES):
        rel = f"s3/o1/E_{k:02d}.png"
        _png(images_root.joinpath(*rel.split("/")), (180, 90, 40))
        eval_records.append(
            SampleRecord(
                relative_path=rel,
                label=k % NUM_CLASSES,
                split="test",
                experience_id=0,
                source_filelist="synthetic/test_filelist.txt",
                line_number=k + 1,
                object_id=1,
                session_id=3,
                category_id=0,
                category_name="synthetic",
                object_name="synthetic_object_1",
            )
        )

    experiences: list[ContinualExperience] = []
    seen: tuple[int, ...] = ()
    for exp in range(2):
        label = exp
        records: list[SampleRecord] = []
        for j in range(SAMPLES_PER_EXP):
            rel = f"s1/o{exp + 1}/C_{exp:02d}_{j:02d}.png"
            _png(images_root.joinpath(*rel.split("/")), (30 + 20 * exp, 40, 90))
            records.append(
                SampleRecord(
                    relative_path=rel,
                    label=label,
                    split="train",
                    experience_id=exp,
                    source_filelist=f"synthetic/train_batch_{exp:02d}_filelist.txt",
                    line_number=j + 1,
                    object_id=exp + 1,
                    session_id=1,
                    category_id=exp,
                    category_name="synthetic",
                    object_name=f"synthetic_object_{exp + 1}",
                )
            )
        seen = seen + (label,)
        experiences.append(
            ContinualExperience(
                experience_id=exp,
                scenario_name="SYNTH",
                run_id=0,
                train_samples=tuple(records),
                evaluation_samples=tuple(eval_records),
                train_source=f"synthetic/train_batch_{exp:02d}_filelist.txt",
                evaluation_source="synthetic/test_filelist.txt",
                classes_introduced=(label,),
                classes_seen=seen,
                objects_introduced=(exp + 1,),
                objects_seen=(exp + 1,),
                categories_introduced=(exp,),
                categories_seen=tuple(range(exp + 1)),
                sessions_present=(1,),
                sessions_seen=(1,),
            )
        )
    return ContinualScenario(
        name="SYNTH",
        scenario_type="SYNTH",
        variant="inc",
        run_id=0,
        experiences=tuple(experiences),
        metadata={
            "label_min": 0,
            "label_max": NUM_CLASSES - 1,
            "train_samples_total": 2 * SAMPLES_PER_EXP,
            "evaluation_samples": EVAL_SAMPLES,
        },
        images_root=images_root,
    )


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> ContinualScenario:
    root = tmp_path_factory.mktemp("phase5_synth")
    return make_mini_scenario(root)


def mini_config(method: str, checkpoint_root: str) -> ContinualTrainConfig:
    return ContinualTrainConfig(
        method=method,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        seed=11,
        device="cpu",
        workers=0,
        image_size=IMG_SIZE,
        model_width=4,
        max_steps_per_epoch=1,
        checkpoint_root=checkpoint_root,
        replay_capacity=8,
        replay_batch_size=2,
        replay_seed=11,
    )


class AlwaysZeroClassifier(nn.Module):
    """Deterministic model: always predicts class 0."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.bias.view(1, -1).expand(x.shape[0], -1).clone()
        logits[:, 0] = 1.0
        return logits


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------
def test_evaluate_model_counts_per_class() -> None:
    images = torch.randint(0, 255, (EVAL_SAMPLES, 3, 8, 8), dtype=torch.uint8)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)
    cache = EvalCache(images, labels, [f"p{i}" for i in range(EVAL_SAMPLES)])
    model = AlwaysZeroClassifier(NUM_CLASSES)
    result = evaluate_model(
        model, cache, num_classes=NUM_CLASSES, device="cpu", batch_size=4
    )
    assert result["n"] == EVAL_SAMPLES
    assert result["correct"][0] == 2  # both class-0 samples predicted 0
    assert result["correct"][1] == 0
    assert result["correct"][2] == 0
    assert result["total"] == [2, 2, 2]
    assert result["overall_correct"] == 2


def test_build_metric_record_null_conventions() -> None:
    eval_result = {"correct": [8, 4, 0], "total": [10, 10, 10], "n": 30,
                   "overall_correct": 12}
    # Experience 0: introduces classes {0}, no previous classes.
    record = build_metric_record(
        experience_id=0,
        train_samples=5,
        classes_introduced=[0],
        classes_seen=[0],
        previous_seen=(),
        eval_result=eval_result,
        train_stats={"epochs": 1},
        eval_seconds=1.0,
        replay_stats=None,
    )
    assert record["accuracy"]["old"] is None  # no old classes at experience 0
    assert record["accuracy"]["new"] is not None
    assert record["accuracy"]["overall"] == pytest.approx(8 / 10)
    assert record["accuracy"]["full"] == pytest.approx(12 / 30)
    assert record["per_class"]["0"] == pytest.approx(0.8)
    assert record["per_class"]["1"] is None  # not introduced yet
    assert record["per_class"]["2"] is None

    # Later experience that introduces nothing: new is null, old exists.
    record2 = build_metric_record(
        experience_id=1,
        train_samples=5,
        classes_introduced=[],
        classes_seen=[0],
        previous_seen=[0],
        eval_result=eval_result,
        train_stats={"epochs": 1},
        eval_seconds=1.0,
        replay_stats=None,
    )
    assert record2["accuracy"]["new"] is None
    assert record2["accuracy"]["old"] == pytest.approx(0.8)


def test_compute_forgetting_definition() -> None:
    records = [
        {"per_class": {"0": 0.8, "1": None}},
        {"per_class": {"0": 0.5, "1": None}},
        {"per_class": {"0": 0.6, "1": 0.9}},
        {"per_class": {"0": 0.7, "1": 0.4}},
    ]
    forgetting = compute_forgetting(records)
    assert forgetting[0] is None  # first experience: no prior measurement
    assert forgetting[1] == pytest.approx(0.8 - 0.5)  # best prior 0.8 - 0.5
    # class 1 first measured at index 2 -> no prior for class 1 yet
    assert forgetting[2] == pytest.approx((0.8 - 0.6))
    # t=3: class0 max prior 0.8 -> 0.8-0.7; class1 max prior 0.9 -> 0.9-0.4
    assert forgetting[3] == pytest.approx(((0.8 - 0.7) + (0.9 - 0.4)) / 2)


def test_compute_forgetting_negative_preserved() -> None:
    records = [{"per_class": {"0": 0.3}}, {"per_class": {"0": 0.9}}]
    forgetting = compute_forgetting(records)
    assert forgetting[1] == pytest.approx(0.3 - 0.9)  # improvement kept


def test_eval_cache_roundtrip(tmp_path: Path) -> None:
    images = torch.randint(0, 255, (5, 3, 8, 8), dtype=torch.uint8)
    labels = torch.arange(5)
    cache = EvalCache(images, labels, [f"a{i}" for i in range(5)])
    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = EvalCache.load(path)
    assert torch.equal(loaded.images, images)
    assert torch.equal(loaded.labels, labels)
    assert loaded.paths == cache.paths
    batch = loaded.float_batch(1, 3)
    assert batch.dtype == torch.float32
    assert batch.shape == (2, 3, 8, 8)
    assert float(batch.min()) >= -1.0 and float(batch.max()) <= 1.0


def test_eval_cache_rejects_bad_shapes() -> None:
    with pytest.raises(EvalCacheError):
        EvalCache(torch.zeros(4, 3, 8, dtype=torch.uint8), torch.zeros(4), ["a"])


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.json"
    atomic_write_json(path, {"a": [1, 2], "b": None})
    assert read_json(path) == {"a": [1, 2], "b": None}
    assert not list(path.parent.glob("*.tmp"))


def test_load_class_names_from_real_mapping() -> None:
    mapping = Path("data/raw/core50/metadata/object_mapping.json")
    if not mapping.is_file():
        pytest.skip("official object mapping not available")
    names = load_class_names(mapping)
    assert names["0"].startswith("plug_adapter")
    assert len(names) == 50


# ---------------------------------------------------------------------------
# Phase-4 resume support used by the Phase-5 driver
# ---------------------------------------------------------------------------
def test_resume_requires_existing_checkpoint(mini: ContinualScenario, tmp_path: Path) -> None:
    trainer = NaiveContinualTrainer(mini_config("naive", str(tmp_path)))
    with pytest.raises(ContinualStateError):
        trainer.resume(mini, checkpoint_dir=tmp_path / "missing")


def test_resume_restores_model_and_state(mini: ContinualScenario, tmp_path: Path) -> None:
    trainer = NaiveContinualTrainer(mini_config("naive", str(tmp_path)))
    state = trainer.initialize(mini, checkpoint_dir=tmp_path / "run")
    initial_checksum = model_checksum(trainer.model)
    assert checkpoint_exists(tmp_path / "run")

    # A fresh trainer resumes exactly what the first one saved.
    resumed = NaiveContinualTrainer(mini_config("naive", str(tmp_path)))
    restored_state = resumed.resume(mini, checkpoint_dir=tmp_path / "run")
    assert restored_state.current_experience == state.current_experience
    assert model_checksum(resumed.model) == initial_checksum

    # Train experience 0 (single step), then resume again from the new state.
    trained = trainer.train_experience(state, mini.get_experience(0))
    again = NaiveContinualTrainer(mini_config("naive", str(tmp_path)))
    restored = again.resume(mini, checkpoint_dir=tmp_path / "run")
    assert restored.current_experience == trained.current_experience == 0
    assert restored.experiences_trained == (0,)
    on_disk, payload = load_state(tmp_path / "run")
    assert model_checksum(again.model) == model_checksum(trainer.model)
    assert payload["current_experience"] == on_disk.current_experience == 0


# ---------------------------------------------------------------------------
# experiment display
# ---------------------------------------------------------------------------
def test_experiment_display_block_contents() -> None:
    stream = io.StringIO()
    display = ExperimentDisplay(
        method="replay",
        scenario_name="NIC_inc",
        run_id=0,
        total_experiences=79,
        epochs=3,
        stream=stream,
        snapshot_interval=0.0,
    )
    display.start_run()
    display.start_experience(
        4, steps_per_epoch=25, epochs=3, completed_before=4
    )
    display.on_metrics(
        {"event": "batch", "epoch": 2, "step": 10, "loss": 0.5,
         "batch_accuracy": 0.75}
    )
    display.on_eval_progress(100, 44972)
    lines = "\n".join(display.build_lines())
    assert "PHASE 5 — CONTINUAL EXPERIMENT" in lines
    assert "Method: replay" in lines
    assert "NIC_inc" in lines
    assert "Experience: 5/79" in lines
    assert "Epoch: 2/3" in lines
    assert "Loss: 0.5000" in lines
    assert "Accuracy: 75.0%" in lines
    assert "44,972" in lines
    assert "experiences 4/79" in lines
    display.finish_experience("[replay] experience 5/79 complete")
    assert "[replay] experience 5/79 complete" in stream.getvalue()
