"""Focused Phase-4 tests: naive continual learning + experience replay.

Tiny synthetic fixtures only — no CORe50 training runs, no full-dataset
scans. The whole file must finish in seconds. Item numbers in docstrings
follow the Step-7 checklist from the phase brief.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.continual import (
    ContinualExperience,
    ContinualScenario,
    ExperienceNotFoundError,
    SampleRecord,
    load_scenario,
)
from src.training import (
    CheckpointMissingError,
    ContinualConfigError,
    ContinualTrainConfig,
    ContinualTrainingError,
    ContinualTrainingState,
    NaiveContinualTrainer,
    ReplayContinualTrainer,
    ReplayMemory,
    ReplayMemoryError,
    StateCompatibilityError,
    build_continual_trainer,
    build_model,
    checkpoint_exists,
    load_state,
    model_checksum,
)
from src.training.config import looks_absolute
from src.training.dataset import ContinualImageDataset
from src.training.state import STATE_VERSION

IMG_SIZE = 16
NUM_CLASSES = 4
SAMPLES_PER_EXP = 6
EVAL_SAMPLES = 4


# ---------------------------------------------------------------------------
# synthetic scenario fixture (tiny generated PNGs + Phase-3-shaped records)
# ---------------------------------------------------------------------------


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def make_synthetic_scenario(root: Path) -> ContinualScenario:
    """3 experiences x 6 train samples + shared 4-sample evaluation set."""
    images_root = root / "images"
    experiences: list[ContinualExperience] = []
    eval_records: list[SampleRecord] = []
    for k in range(EVAL_SAMPLES):
        rel = f"s3/o1/E_{k:02d}.png"
        _png(images_root.joinpath(*rel.split("/")), (200, 100, 50))
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

    seen_classes: tuple[int, ...] = ()
    seen_objects: tuple[int, ...] = ()
    seen_sessions: tuple[int, ...] = ()
    for exp in range(3):
        label = exp % NUM_CLASSES
        session = (1, 5, 11)[exp]
        records: list[SampleRecord] = []
        for j in range(SAMPLES_PER_EXP):
            rel = f"s1/o{exp + 1}/C_{exp:02d}_{j:02d}.png"
            _png(images_root.joinpath(*rel.split("/")), (30 + 20 * exp, 40 + 10 * j, 90))
            records.append(
                SampleRecord(
                    relative_path=rel,
                    label=label,
                    split="train",
                    experience_id=exp,
                    source_filelist=f"synthetic/train_batch_{exp:02d}_filelist.txt",
                    line_number=j + 1,
                    object_id=exp + 1,
                    session_id=session,
                    category_id=exp,
                    category_name="synthetic",
                    object_name=f"synthetic_object_{exp + 1}",
                )
            )
        seen_classes = seen_classes + (label,)
        seen_objects = seen_objects + (exp + 1,)
        seen_sessions = seen_sessions + (session,)
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
                classes_seen=seen_classes,
                objects_introduced=(exp + 1,),
                objects_seen=seen_objects,
                categories_introduced=(exp,),
                categories_seen=tuple(range(exp + 1)),
                sessions_present=(session,),
                sessions_seen=seen_sessions,
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
            "train_samples_total": 3 * SAMPLES_PER_EXP,
            "evaluation_samples": EVAL_SAMPLES,
        },
        images_root=images_root,
    )


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> ContinualScenario:
    root = tmp_path_factory.mktemp("synthetic_core50")
    return make_synthetic_scenario(root)


def tiny_config(
    method: str,
    *,
    max_steps: int | None = 2,
    checkpoint_root: str = "models/continual",
    **overrides,
) -> ContinualTrainConfig:
    values: dict = dict(
        method=method,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        seed=7,
        device="cpu",
        workers=0,
        image_size=IMG_SIZE,
        model_width=4,
        max_steps_per_epoch=max_steps,
        checkpoint_root=checkpoint_root,
        replay_capacity=8,
        replay_batch_size=2,
        replay_seed=11,
    )
    values.update(overrides)
    return ContinualTrainConfig(**values)


def _payload_model_state(run_dir: Path) -> dict:
    _, payload = load_state(run_dir)
    return payload["model_state"]


def _tamper_checkpoint(run_dir: Path, value: float = 0.1234) -> str:
    """Overwrite one saved weight with a sentinel (proves disk reload)."""
    state, payload = load_state(run_dir)
    key = next(k for k in payload["model_state"] if k.endswith("weight"))
    payload["model_state"][key] = torch.full_like(payload["model_state"][key], value)
    torch.save(payload, run_dir / state.checkpoint)
    return key


# ---------------------------------------------------------------------------
# NAIVE (items 1-7)
# ---------------------------------------------------------------------------


def test_01_trainer_initializes(synth, tmp_path):
    """1. Trainer can initialize."""
    trainer = build_continual_trainer("naive", tiny_config("naive"))
    assert trainer.model is None
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    assert isinstance(state, ContinualTrainingState)
    assert state.scenario == "SYNTH"
    assert state.variant == "inc"
    assert state.run_id == 0
    assert state.method == "naive"
    assert state.current_experience == -1
    assert state.experiences_trained == ()
    assert trainer.model is not None
    assert (run_dir / "state.json").is_file()
    assert (run_dir / "checkpoint.pt").is_file()
    assert checkpoint_exists(run_dir)


def test_02_first_experience_creates_checkpoint(synth, tmp_path):
    """2. First experience creates a model/checkpoint."""
    trainer = build_continual_trainer("naive", tiny_config("naive"))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state)
    assert state.current_experience == 0
    assert state.experiences_trained == (0,)
    assert state.epochs_completed == 1
    assert state.steps_completed > 0
    assert checkpoint_exists(run_dir)
    _, payload = load_state(run_dir)
    assert payload["current_experience"] == 0
    assert payload["method"] == "naive"
    assert payload["optimizer_state"]["state"], "optimizer state must be persisted"


def test_03_second_experience_loads_previous_state(synth, tmp_path):
    """3. Second experience loads previous state (from disk, not memory)."""
    trainer = build_continual_trainer("naive", tiny_config("naive", max_steps=0))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    key = _tamper_checkpoint(run_dir, 0.1234)
    state = trainer.train_experience(state, synth.experiences[1])
    assert state.current_experience == 1
    tensor = _payload_model_state(run_dir)[key]
    assert torch.allclose(tensor, torch.full_like(tensor, 0.1234)), (
        "experience 2 must reload the checkpoint written by experience 1"
    )


def test_04_model_state_persists_across_experiences(synth, tmp_path):
    """4. Model state persists and evolves across experiences."""
    trainer = build_continual_trainer("naive", tiny_config("naive"))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    checksum_0 = model_checksum(trainer.model)
    state = trainer.train_experience(state, synth.experiences[1])
    checksum_1 = model_checksum(trainer.model)
    assert checksum_0 != checksum_1
    assert state.epochs_completed == 2
    assert state.experiences_trained == (0, 1)
    loaded, _ = load_state(run_dir)
    assert loaded.experiences_trained == (0, 1)
    assert loaded.epochs_completed == 2


def test_05_no_fresh_model_per_experience(synth, tmp_path):
    """5. Fresh model is NOT silently created for each experience."""
    init_checksum = model_checksum(build_model(NUM_CLASSES, width=4, seed=7))
    trainer = build_continual_trainer("naive", tiny_config("naive"))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    after_first = model_checksum(trainer.model)
    state = trainer.train_experience(state, synth.experiences[1])
    after_second = model_checksum(trainer.model)
    assert after_first != init_checksum, "experience 1 must train (change) the initial model"
    assert after_second != init_checksum, "experience 2 must continue, not re-initialize"


def test_06_state_serialization(synth, tmp_path):
    """6. State serialization works (round-trip + compatibility checks)."""
    trainer = build_continual_trainer("naive", tiny_config("naive"))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    loaded, payload = load_state(run_dir)
    strip = lambda d: {  # noqa: E731 - tiny local helper
        k: v for k, v in d.items() if k not in ("created_utc", "updated_utc")
    }
    assert strip(loaded.to_dict()) == strip(state.to_dict())
    assert loaded.version == STATE_VERSION
    assert payload["format_version"] == 1
    raw = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert raw["experiences_trained"] == [0]
    validate_kwargs = dict(
        scenario="SYNTH", variant="inc", run_id=0, method="naive",
        config_fingerprint=state.config_fingerprint,
    )
    from src.training import validate_state

    validate_state(loaded, **validate_kwargs)
    bad = replace(loaded, config_fingerprint="deadbeef")
    with pytest.raises(StateCompatibilityError):
        trainer.train_experience(bad, synth.experiences[1])


def test_07_invalid_configuration_fails(synth, tmp_path):
    """7. Invalid configuration fails correctly."""
    for override in (
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"replay_capacity": 0},
        {"optimizer": "lion"},
        {"scheduler": "cosine"},
        {"device": "tpu"},
        {"max_steps_per_epoch": -1},
        {"method": "ewc"},
    ):
        with pytest.raises((ContinualConfigError, ValueError)):
            ContinualTrainConfig(**{**tiny_config("naive").to_dict(), **override})
    with pytest.raises(ContinualTrainingError):
        NaiveContinualTrainer(tiny_config("replay"))
    trainer = NaiveContinualTrainer(tiny_config("naive"))
    with pytest.raises(ContinualTrainingError, match="state is required"):
        trainer.train_experience(None)
    other = NaiveContinualTrainer(tiny_config("naive"))
    foreign_state = other.initialize(synth, checkpoint_dir=tmp_path / "a")
    with pytest.raises(ContinualTrainingError, match="not initialized"):
        trainer.train_experience(foreign_state)


# ---------------------------------------------------------------------------
# REPLAY MEMORY (items 8-14)
# ---------------------------------------------------------------------------


def _memory(synth, capacity: int = 8, *, scenario="SYNTH") -> ReplayMemory:
    return ReplayMemory(capacity, seed=11, scenario=scenario, variant="inc", run_id=0)


def test_08_replay_memory_starts_empty(synth):
    """8. Replay memory starts empty."""
    memory = _memory(synth)
    assert len(memory) == 0
    assert memory.size == 0
    assert not memory
    assert memory.sample(3) == []
    assert memory.last_experience == -1


def test_09_replay_memory_adds_current_samples(synth):
    """9. Current samples can be added."""
    memory = _memory(synth)
    added = memory.add(synth.experiences[0].train_samples, experience_index=0)
    assert added == SAMPLES_PER_EXP
    assert len(memory) == SAMPLES_PER_EXP
    assert memory.last_experience == 0
    assert all(r.split == "train" for r in memory.records())


def test_10_replay_memory_bounded(synth):
    """10. Replay memory is bounded (FIFO eviction, references only)."""
    memory = _memory(synth, capacity=4)
    memory.add(synth.experiences[0].train_samples, experience_index=0)
    assert len(memory) == 4
    expected_first = synth.experiences[0].train_samples[2].relative_path
    assert memory.records()[0].relative_path == expected_first
    memory.add(synth.experiences[1].train_samples, experience_index=1)
    assert len(memory) == 4
    assert memory.last_experience == 1


def test_11_replay_memory_sampling(synth):
    """11. Replay sampling works."""
    memory = _memory(synth)
    memory.add(synth.experiences[0].train_samples, experience_index=0)
    picked = memory.sample(3)
    assert len(picked) == 3
    assert len({r.relative_path for r in picked}) == 3
    contents = {r.relative_path for r in memory.records()}
    assert {r.relative_path for r in picked} <= contents
    assert len(memory.sample(100)) == SAMPLES_PER_EXP
    assert memory.sample(0) == []


def test_12_replay_memory_serialization(synth):
    """12. Replay memory serializes/reloads."""
    memory = _memory(synth, capacity=8)
    memory.add(synth.experiences[0].train_samples, experience_index=0)
    state = memory.state_dict()
    restored = _memory(synth, capacity=8)
    restored.load_state_dict(state)
    assert restored.records() == memory.records()
    assert restored.last_experience == 0
    assert restored.sample(3, seed=5) == memory.sample(3, seed=5)

    bad_version = dict(state, format_version=99)
    with pytest.raises(ReplayMemoryError, match="format"):
        _memory(synth).load_state_dict(bad_version)
    too_small = ReplayMemory(2, scenario="SYNTH", variant="inc", run_id=0)
    with pytest.raises(ReplayMemoryError, match="exceeds"):
        too_small.load_state_dict(state)
    wrong_place = dict(state, scenario="OTHER")
    with pytest.raises(ReplayMemoryError, match="provenance"):
        _memory(synth).load_state_dict(wrong_place)


def test_13_evaluation_samples_cannot_enter_memory(synth):
    """13. Evaluation samples cannot enter memory."""
    memory = _memory(synth)
    eval_records = synth.experiences[0].evaluation_samples
    with pytest.raises(ReplayMemoryError, match="[Ee]valuation"):
        memory.add(eval_records, experience_index=0)
    assert len(memory) == 0
    with pytest.raises(ValueError, match="split"):
        ContinualImageDataset(
            eval_records, synth.images_root, IMG_SIZE, require_split="train"
        )


def test_14_future_samples_cannot_enter_memory(synth):
    """14. Future experience samples cannot enter memory."""
    memory = _memory(synth)
    with pytest.raises(ReplayMemoryError, match="official order"):
        memory.add(synth.experiences[2].train_samples, experience_index=2)
    with pytest.raises(ReplayMemoryError, match="Future-experience"):
        memory.add(synth.experiences[1].train_samples, experience_index=0)
    memory.add(synth.experiences[0].train_samples, experience_index=0)
    with pytest.raises(ReplayMemoryError, match="official order"):
        memory.add(synth.experiences[1].train_samples, experience_index=2)
    memory.add(synth.experiences[1].train_samples, experience_index=1)
    assert all(r.experience_id <= memory.last_experience for r in memory.records())


# ---------------------------------------------------------------------------
# REPLAY TRAINER (items 15-17)
# ---------------------------------------------------------------------------


def test_15_replay_trainer_uses_previous_model_state(synth, tmp_path):
    """15. Trainer uses previous model state (disk reload, no reset)."""
    trainer = build_continual_trainer("replay", tiny_config("replay", max_steps=0))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    key = _tamper_checkpoint(run_dir, 0.4321)
    state = trainer.train_experience(state, synth.experiences[1])
    tensor = _payload_model_state(run_dir)[key]
    assert torch.allclose(tensor, torch.full_like(tensor, 0.4321))
    _, payload = load_state(run_dir)
    assert payload["replay_memory"]["last_experience"] == 1


def test_16_replay_trainer_combines_current_and_replay(synth, tmp_path):
    """16. Trainer combines current + replay data (after memory has content)."""
    trainer = build_continual_trainer("replay", tiny_config("replay"))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    assert state.replay is not None
    assert state.replay["steps_with_replay"] == 0, "no replay before any history exists"
    assert state.replay["size"] == SAMPLES_PER_EXP
    state = trainer.train_experience(state, synth.experiences[1])
    assert state.replay["steps_with_replay"] == 2, "every training step of exp 2 uses replay"
    assert state.replay["samples_replayed"] == 4
    assert state.replay["last_experience"] == 1
    assert state.replay["size"] == 8, "capacity stays bounded at 8"


def test_17_replay_trainer_checkpoints_state(synth, tmp_path):
    """17. Replay trainer checkpoints state correctly."""
    trainer = build_continual_trainer("replay", tiny_config("replay"))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, synth.experiences[0])
    state = trainer.train_experience(state, synth.experiences[1])
    assert checkpoint_exists(run_dir)
    loaded, payload = load_state(run_dir)
    memory_state = payload["replay_memory"]
    assert memory_state["format_version"] == 1
    assert len(memory_state["items"]) == 8
    assert all(item["split"] == "train" for item in memory_state["items"])
    assert payload["replay_stats"]["steps_with_replay"] == 2
    assert loaded.replay["size"] == 8
    assert loaded.method == "replay"
    raw = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert raw["replay"]["capacity"] == 8
    with pytest.raises(CheckpointMissingError):
        (run_dir / "checkpoint.pt").unlink()
        load_state(run_dir)


# ---------------------------------------------------------------------------
# UNIFIED API (items 18-22)
# ---------------------------------------------------------------------------


def test_18_factory_selects_naive():
    """18. method=naive selects naive trainer."""
    trainer = build_continual_trainer("naive", tiny_config("replay"))
    assert isinstance(trainer, NaiveContinualTrainer)
    assert trainer.method_name == "naive"
    assert trainer.config.method == "naive"
    assert isinstance(build_continual_trainer("naive"), NaiveContinualTrainer)


def test_19_factory_selects_replay():
    """19. method=replay selects replay trainer."""
    trainer = build_continual_trainer("replay", tiny_config("naive"))
    assert isinstance(trainer, ReplayContinualTrainer)
    assert trainer.method_name == "replay"
    assert trainer.config.method == "replay"
    from src.training import supported_methods

    assert supported_methods() == ("naive", "replay")
    assert isinstance(
        build_continual_trainer(None, tiny_config("replay", replay_enabled=True)),
        ReplayContinualTrainer,
    )
    assert isinstance(build_continual_trainer(None, tiny_config("naive")), NaiveContinualTrainer)


def test_20_invalid_method_fails_clearly():
    """20. Invalid method fails clearly."""
    for bad in ("ewc", "lwf", "distill", "", "NAIVE_X"):
        with pytest.raises(ValueError, match="naive"):
            build_continual_trainer(bad)
    with pytest.raises(ValueError):
        build_continual_trainer(5)


def test_21_deterministic_replay_sampling(synth):
    """21. Deterministic replay sampling with the same seed."""
    first = _memory(synth, capacity=8)
    second = _memory(synth, capacity=8)
    for memory in (first, second):
        memory.add(synth.experiences[0].train_samples, experience_index=0)
        memory.add(synth.experiences[1].train_samples, experience_index=1)
    assert first.sample(4, seed=123) == second.sample(4, seed=123)
    assert first.sample(4, seed=123) == first.sample(4, seed=123)
    assert first.sample(4, seed=123) == first.sample(4, seed=123)


def test_22_no_absolute_paths_in_generated_metadata(synth, tmp_path):
    """22. No absolute personal path in generated metadata."""
    seen: list[Path] = []
    for method in ("naive", "replay"):
        run_dir = tmp_path / method
        trainer = build_continual_trainer(
            method, tiny_config(method, checkpoint_root=str(tmp_path))
        )
        state = trainer.initialize(synth, checkpoint_dir=run_dir)
        state = trainer.train_experience(state, synth.experiences[0])
        seen.append(run_dir / "state.json")
        for value in state.to_dict()["training_config"].values():
            if isinstance(value, str):
                assert not looks_absolute(value), f"absolute path leaked: {value}"
    for state_file in seen:
        text = state_file.read_text(encoding="utf-8")
        assert "C:" not in text, f"drive path leaked into {state_file}"
        assert "/home/" not in text
        payload = json.loads(text)
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, str) and not node.startswith("s1/") and not node.startswith("s3/"):
                assert not looks_absolute(node), f"absolute path in metadata: {node}"


# ---------------------------------------------------------------------------
# official-order guard + Phase-3 integration
# ---------------------------------------------------------------------------


def test_23_official_order_is_enforced(synth, tmp_path):
    """Out-of-order / future / repeated experiences are rejected."""
    trainer = build_continual_trainer("naive", tiny_config("naive", max_steps=0))
    run_dir = tmp_path / "run"
    state = trainer.initialize(synth, checkpoint_dir=run_dir)
    with pytest.raises(ContinualTrainingError, match="Official order"):
        trainer.train_experience(state, synth.experiences[2])
    with pytest.raises(ContinualTrainingError, match="Official order"):
        trainer.train_experience(state, synth.experiences[1])
    state = trainer.train_experience(state, synth.experiences[0])
    with pytest.raises(ContinualTrainingError, match="Official order"):
        trainer.train_experience(state, synth.experiences[0])
    state = trainer.train_experience(state, synth.experiences[1])
    state = trainer.train_experience(state, synth.experiences[2])
    assert state.current_experience == 2
    with pytest.raises(ExperienceNotFoundError):
        trainer.train_experience(state)


@pytest.mark.skipif(
    not Path("data/raw/core50/filelists").is_dir(),
    reason="official CORe50 filelists not present",
)
def test_24_real_nic_first_experience_smoke(tmp_path):
    """Phase-3 integration: official NIC run 0, experience 0, two tiny steps."""
    scenario = load_scenario("NIC", variant="inc", run=0)
    assert scenario.experiences[0].experience_id == 0
    assert scenario.name == "NIC_inc"
    assert scenario.scenario_type == "NIC"
    assert scenario.experiences[0].train_source.startswith("data/raw/")
    assert scenario.metadata["evaluation_sessions"] == [3, 7, 10]
    trainer = build_continual_trainer(
        "naive", tiny_config("naive", max_steps=2, checkpoint_root=str(tmp_path))
    )
    state = trainer.initialize(scenario, checkpoint_dir=tmp_path / "nic")
    state = trainer.train_experience(state)
    assert state.scenario == "NIC"
    assert state.current_experience == 0
    assert state.steps_completed == 2
    assert checkpoint_exists(tmp_path / "nic")
