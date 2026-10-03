"""Tests for early stopping, best-epoch restoration, and trainer smoke.

Covers the section-13 fix: the development metric stops an experience,
the BEST epoch's weights are restored before the next experience, and
the global best checkpoint is written atomically and re-loaded on resume.
Tiny synthetic PNGs only — finishes in seconds, no CORe50 access.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.continual import ContinualExperience, ContinualScenario, SampleRecord
from src.training import ContinualTrainConfig, NaiveContinualTrainer, build_model
from src.training.early_stop import EarlyStopping
from src.training.improved import ImprovedReplayTrainer

IMG_SIZE = 16
NUM_CLASSES = 50


# ---------------------------------------------------------------------------
# tiny synthetic scenario (2 experiences x 4 train samples)
# ---------------------------------------------------------------------------
def _png(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def make_tiny_scenario(root: Path) -> ContinualScenario:
    images_root = root / "images"
    experiences = []
    seen: tuple[int, ...] = ()
    for exp in range(2):
        label = exp
        records = []
        for j in range(4):
            rel = f"s1/o{exp + 1}/C_{exp:02d}_{j:02d}.png"
            _png(images_root.joinpath(*rel.split("/")), (30 + 40 * exp, 40 + 10 * j, 90))
            records.append(
                SampleRecord(
                    relative_path=rel,
                    label=label,
                    split="train",
                    experience_id=exp,
                    source_filelist=f"synthetic/train_{exp:02d}.txt",
                    line_number=j + 1,
                    object_id=exp + 1,
                    session_id=(1, 5)[exp],
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
                evaluation_samples=tuple(records),
                train_source=f"synthetic/train_{exp:02d}.txt",
                evaluation_source=f"synthetic/train_{exp:02d}.txt",
                classes_introduced=(label,),
                classes_seen=seen,
                objects_introduced=(exp + 1,),
                objects_seen=(exp + 1,),
                categories_introduced=(exp,),
                categories_seen=tuple(range(exp + 1)),
                sessions_present=((1, 5)[exp],),
                sessions_seen=(1, 5)[exp],
            )
        )
    return ContinualScenario(
        name="SYNTH",
        scenario_type="SYNTH",
        variant="inc",
        run_id=0,
        experiences=tuple(experiences),
        metadata={"label_min": 0, "label_max": 1},
        images_root=images_root,
    )


@pytest.fixture(scope="module")
def scenario(tmp_path_factory) -> ContinualScenario:
    return make_tiny_scenario(tmp_path_factory.mktemp("early_stop"))


def _config(**overrides) -> ContinualTrainConfig:
    base = {
        "method": "replay",
        "image_size": IMG_SIZE,
        "epochs": 1,
        "batch_size": 4,
        "early_stop_patience": 2,
        "seed": 7,
    }
    base.update(overrides)
    return ContinualTrainConfig(**base)


# ---------------------------------------------------------------------------
# EarlyStopping decision rule
# ---------------------------------------------------------------------------
def test_stops_after_patience_bad_epochs_keeps_best():
    stopper = EarlyStopping(patience=2)
    assert stopper.observe(0.50, 0) is False
    assert stopper.observe(0.40, 1) is False
    assert stopper.observe(0.30, 2) is True
    assert stopper.best_value == 0.50
    assert stopper.best_epoch == 0
    assert stopper.bad_epochs == 2
    assert stopper.should_restore is True


def test_best_resets_bad_counter():
    stopper = EarlyStopping(patience=2)
    stopper.observe(0.50, 0)
    stopper.observe(0.40, 1)
    assert stopper.observe(0.51, 2) is False
    assert stopper.bad_epochs == 0
    assert stopper.best_epoch == 2
    assert stopper.observe(0.505, 3) is False  # 0.51 + 1e-4 not beaten
    assert stopper.observe(0.505, 4) is True


def test_nonfinite_metric_is_rejected_loudly():
    stopper = EarlyStopping(patience=1)
    with pytest.raises(ValueError, match="finite"):
        stopper.observe(float("nan"), 0)
    with pytest.raises(ValueError, match="finite"):
        stopper.observe(float("inf"), 0)


def test_invalid_patience_and_min_delta():
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=0)
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=-1)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(patience=1, min_delta=-0.1)


# ---------------------------------------------------------------------------
# ImprovedReplayTrainer: best-epoch restore + global best checkpoint
# ---------------------------------------------------------------------------
class _AlwaysZero(torch.nn.Module):
    """Stub head: predicts class 0 for every image (deterministic eval)."""

    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 50
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], self.num_classes)


def test_evaluate_records_accuracy_is_hand_countable(scenario):
    cfg = _config()
    trainer = ImprovedReplayTrainer(cfg, dev_provider=lambda _: ())
    trainer._scenario = scenario
    trainer._model = _AlwaysZero()
    # experience 0 records are all label 0 -> always-zero predictor scores 1.0
    assert trainer._evaluate_records(scenario.get_experience(0).train_samples) == 1.0
    # experience 1 records are all label 1 -> scores 0.0
    assert trainer._evaluate_records(scenario.get_experience(1).train_samples) == 0.0


def test_best_epoch_weights_are_restored_at_experience_end(scenario, tmp_path):
    cfg = _config()
    best_path = tmp_path / "best_model.pt"
    trainer = ImprovedReplayTrainer(
        cfg,
        dev_provider=lambda _: scenario.get_experience(0).train_samples,
        best_checkpoint_path=best_path,
    )
    trainer._scenario = scenario
    trainer._model = build_model(NUM_CLASSES)

    experience = scenario.get_experience(0)
    trainer._on_experience_start(experience)

    # deterministic metric sequence: improve, then decay past patience
    values = iter([0.50, 0.40, 0.30])
    trainer._evaluate_records = lambda records: next(values)  # type: ignore[method-assign]

    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in trainer._model.state_dict().items()
    }
    stops = []
    for epoch in range(3):
        stops.append(
            trainer._on_epoch_end(experience, epoch, {"loss": 1.0, "accuracy": 0.5})
        )
        if epoch == 0:
            # simulate further training corrupting the weights after the best
            with torch.no_grad():
                for parameter in trainer._model.parameters():
                    parameter.add_(1.0)
    assert stops == [False, False, True]
    assert trainer._exp_best_state is not None

    trainer._finalize_experience_weights(experience)
    restored = trainer._model.state_dict()
    for name, tensor in initial_state.items():
        assert torch.equal(restored[name], tensor), name

    assert trainer.best_development_value == 0.50
    assert best_path.is_file()

    # a fresh process reloads the global-best threshold
    resumed = ImprovedReplayTrainer(cfg, best_checkpoint_path=best_path)
    assert resumed.best_development_value == 0.50
    assert resumed.best_development_meta["experience"] == 0


def test_no_dev_provider_means_no_stopping(scenario):
    cfg = _config()
    trainer = ImprovedReplayTrainer(cfg, dev_provider=None)
    trainer._scenario = scenario
    trainer._model = build_model(NUM_CLASSES)
    experience = scenario.get_experience(0)
    trainer._on_experience_start(experience)
    trainer._evaluate_records = lambda records: 0.0  # type: ignore[method-assign]
    assert trainer._on_epoch_end(experience, 0, {"loss": 1.0, "accuracy": 0.5}) is False
    trainer._finalize_experience_weights(experience)
    assert trainer._exp_best_state is None


def test_config_without_patience_skips_stopper(scenario):
    cfg = _config(early_stop_patience=None)
    trainer = ImprovedReplayTrainer(cfg, dev_provider=lambda _: ())
    trainer._on_experience_start(scenario.get_experience(0))
    assert experience_ids(trainer) == []


def experience_ids(trainer) -> list[int]:
    return sorted(trainer._stoppers)


# ---------------------------------------------------------------------------
# one-batch end-to-end smoke: full initialize + one optimizer step
# ---------------------------------------------------------------------------
def test_one_batch_smoke_run(scenario, tmp_path):
    cfg = _config(
        method="naive",
        early_stop_patience=None,
        epochs=1,
        batch_size=4,
        max_steps_per_epoch=1,
    )
    trainer = NaiveContinualTrainer(cfg)
    state = trainer.initialize(scenario, checkpoint_dir=tmp_path)
    events: list[dict] = []
    trainer.on_train_metrics = lambda event: events.append(event)
    state = trainer.train_experience(state, scenario.get_experience(0))
    assert state.steps_completed == 1
    assert state.epochs_completed == 1
    epoch_events = [e for e in events if e.get("event") == "epoch"]
    assert epoch_events and epoch_events[0]["loss"] is not None
    assert epoch_events[0]["loss"] == epoch_events[0]["loss"]  # finite (not NaN)
    assert (tmp_path / "state.json").is_file()


# ---------------------------------------------------------------------------
# tiny overfit: memorizing a fixed batch proves gradients reach the head
# ---------------------------------------------------------------------------
def test_tiny_batch_overfits():
    torch.manual_seed(0)
    model = build_model(3)
    inputs = torch.randn(8, 3, 16, 16)
    targets = torch.randint(0, 3, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    first_loss = None
    final_loss = None
    for step in range(60):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        final_loss = float(loss.detach())

    assert first_loss is not None and final_loss is not None
    assert final_loss < 0.5 * first_loss, (first_loss, final_loss)

    model.eval()
    with torch.inference_mode():
        predictions = model(inputs).argmax(dim=1)
    assert int(predictions.eq(targets).sum()) >= 6
