"""Focused repair-integrity tests (Phase 1-6 repair pass).

Each test pins one contract the repair had to establish: the semantic
CORe50 object mapping, the public ``NaiveContinual`` API name, naive
sequential semantics, replay-memory bounds, the documented final-checkpoint
loader, final-model metadata/hash, the leakage-free training configuration,
and the locked Phase-5 artifacts. Tiny fixtures only — no training runs on
CORe50, no full-dataset scans.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.data.continual import SampleRecord
from src.data.core50 import (
    OBJECT_CATEGORY,
    OFFICIAL_CATEGORY_ORDER,
    TRAIN_SESSIONS,
    validate_core50_object_mapping,
)
from src.evaluation.continual import load_class_names
from src.training import (
    CheckpointSchemaError,
    NaiveContinual,
    NaiveContinualTrainer,
    ReplayContinualTrainer,
    ReplayMemory,
    ReplayMemoryError,
    build_continual_trainer,
    load_final_model,
    model_checksum,
    read_checkpoint_schema,
    validate_checkpoint_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = (
    PROJECT_ROOT / "data" / "raw" / "core50" / "metadata" / "object_mapping.json"
)
FINAL_PT = PROJECT_ROOT / "models" / "continual" / "final_model.pt"
FINAL_JSON = PROJECT_ROOT / "models" / "continual" / "final_model.json"
REPLAY_PT = (
    PROJECT_ROOT / "models" / "continual" / "phase5_nic" / "replay" / "checkpoint.pt"
)
LOCKED_FINAL_SHA256 = (
    "b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351"
)
LOCKED_NAIVE_FINAL = 0.0234590411811794
LOCKED_NAIVE_FORGETTING = 0.620896
LOCKED_NAIVE_AVG_INCR = 0.045834
LOCKED_REPLAY_FINAL = 0.05378902428177533
LOCKED_REPLAY_FORGETTING = 0.536408
LOCKED_REPLAY_AVG_INCR = 0.063249


# ---------------------------------------------------------------------------
# 1. CORe50 object mapping resolves 50 semantic identities
# ---------------------------------------------------------------------------


def test_object_mapping_resolves_50_official_identities() -> None:
    result = validate_core50_object_mapping()
    assert result["errors"] == []
    assert result["identity_count"] == 50
    assert result["category_count"] == 10

    identities = result["identities"]
    assert [item["object_id"] for item in identities] == list(range(1, 51))
    assert [item["label"] for item in identities] == list(range(50))
    assert len({item["name"] for item in identities}) == 50
    assert len({item["directory"] for item in identities}) == 50
    for item in identities:
        assert item["category"] == OBJECT_CATEGORY[item["object_id"]]
    assert set(result["categories"]) == set(OFFICIAL_CATEGORY_ORDER)

    class_names = load_class_names(MAPPING_PATH)
    assert len(class_names) == 50
    for item in identities:
        assert class_names[str(item["label"])] == item["name"]


# ---------------------------------------------------------------------------
# 2. NaiveContinual is the public naive API
# ---------------------------------------------------------------------------


def test_naive_continual_is_public_naive_api() -> None:
    from src.training import supported_methods

    assert issubclass(NaiveContinual, NaiveContinualTrainer)
    assert NaiveContinual.method_name == "naive"
    assert "naive" in supported_methods()

    trainer = build_continual_trainer("naive")
    assert isinstance(trainer, NaiveContinualTrainer)
    assert not isinstance(trainer, ReplayContinualTrainer)
    assert trainer.method_name == "naive"
    assert isinstance(NaiveContinual(), NaiveContinualTrainer)


# ---------------------------------------------------------------------------
# 3. Naive method trains sequentially without re-initializing
# ---------------------------------------------------------------------------


def test_naive_sequential_state_semantics(tmp_path: Path) -> None:
    from test_phase4_continual_training import make_synthetic_scenario, tiny_config

    scenario = make_synthetic_scenario(tmp_path / "synthetic")
    trainer = build_continual_trainer("naive", tiny_config("naive"))
    run_dir = tmp_path / "run"

    state = trainer.initialize(scenario, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, scenario.experiences[0])
    after_first = model_checksum(trainer.model)
    state = trainer.train_experience(state, scenario.experiences[1])
    after_second = model_checksum(trainer.model)

    assert state.experiences_trained == (0, 1)
    assert after_first != after_second, "experience 2 must continue training"
    from src.training import build_model, load_state

    init_checksum = model_checksum(build_model(4, width=4, seed=7))
    assert after_first != init_checksum, "experience 1 must train, not re-init"
    loaded, _ = load_state(run_dir)
    assert loaded.experiences_trained == (0, 1)


# ---------------------------------------------------------------------------
# 4. ReplayMemory stays bounded, ordered, and train-only
# ---------------------------------------------------------------------------


def _record(
    index: int,
    *,
    split: str = "train",
    experience_id: int = 0,
) -> SampleRecord:
    return SampleRecord(
        relative_path=f"s1/o1/C_00_{index:03d}.png",
        label=0,
        split=split,
        experience_id=experience_id,
        source_filelist="synthetic/train_batch_00_filelist.txt",
        line_number=index + 1,
        object_id=1,
        session_id=1,
        category_id=0,
        category_name="synthetic",
        object_name="synthetic_object_1",
    )


def test_replay_memory_bounds_and_train_only() -> None:
    memory = ReplayMemory(capacity=3, seed=11)

    added = memory.add([_record(i) for i in range(5)], experience_index=0)
    assert added == 5
    assert memory.size == 3
    kept = [record.relative_path for record in memory.records()]
    assert kept == [
        "s1/o1/C_00_002.png",
        "s1/o1/C_00_003.png",
        "s1/o1/C_00_004.png",
    ], "FIFO eviction must drop the oldest references"

    with pytest.raises(ReplayMemoryError, match="cannot enter replay memory"):
        memory.add([_record(99, split="test")], experience_index=1)
    with pytest.raises(ReplayMemoryError, match="Future-experience sample"):
        memory.add([_record(50, experience_id=2)], experience_index=1)
    with pytest.raises(ReplayMemoryError, match="official order"):
        memory.add([_record(60)], experience_index=5)

    assert memory.add([_record(70)], experience_index=1) == 1
    assert memory.size == 3
    assert memory.sample(2, seed=5) == memory.sample(2, seed=5)


# ---------------------------------------------------------------------------
# 5. Final checkpoint loads through the documented schema reader
# ---------------------------------------------------------------------------


def test_final_checkpoint_loader() -> None:
    assert validate_checkpoint_schema(
        expected_num_classes=50, expected_method="replay"
    ) == []
    model, schema = load_final_model()
    assert schema["method"] == "replay"
    assert schema["num_classes"] == 50
    assert schema["model_state_tensors"] == len(model.state_dict())
    assert "state_dict" not in schema["top_level_keys"]
    assert "model_state" in schema["top_level_keys"]

    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 50)
    assert bool(torch.isfinite(output).all())

    with pytest.raises(CheckpointSchemaError):
        load_final_model(num_classes=19)


# ---------------------------------------------------------------------------
# 6. Final model metadata matches the frozen checkpoint bytes
# ---------------------------------------------------------------------------


def test_final_model_metadata_and_hash() -> None:
    digest = hashlib.sha256(FINAL_PT.read_bytes()).hexdigest()
    meta = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
    assert meta["sha256"] == digest
    assert meta["method"] == "replay"
    assert meta["num_classes"] == 50
    assert meta["phase"] in (6, 7)
    if meta["phase"] == 6:
        assert digest == LOCKED_FINAL_SHA256
        assert meta["source_checkpoint"].replace("\\", "/").endswith(
            "models/continual/phase5_nic/replay/checkpoint.pt"
        )
        assert FINAL_PT.read_bytes() == REPLAY_PT.read_bytes(), (
            "final model must be byte-identical to the selected replay checkpoint"
        )
    else:
        # Phase-7 one-shot freeze: the final model is the development-
        # selected final-run artifact, and the historical baseline must
        # live on byte-identical in the preservation copy.
        assert meta["baseline_sha256"] == LOCKED_FINAL_SHA256
        payload = torch.load(FINAL_PT, map_location="cpu", weights_only=False)
        provenance = payload["provenance"]
        assert provenance["baseline_sha256"] == LOCKED_FINAL_SHA256
        assert payload["selection"]["held_out_sessions_used"] is False
        preserved = PROJECT_ROOT / "models/continual/baseline_replay_phase5.pt"
        assert preserved.is_file()
        assert preserved.read_bytes() == REPLAY_PT.read_bytes(), (
            "preservation copy must be byte-identical to the phase-5 replay checkpoint"
        )


# ---------------------------------------------------------------------------
# 7. Training configuration never sees evaluation data
# ---------------------------------------------------------------------------


def test_no_evaluation_data_in_training_configuration() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase5_nic.yaml").read_text(encoding="utf-8")
    )
    assert config["scenario"] == "NIC"
    assert config["variant"] == "inc"
    assert config["run"] == 0
    assert config["training"]["seed"] == 42
    eval_sessions = config["evaluation"]["sessions"]
    assert eval_sessions == [3, 7, 10]
    assert not set(eval_sessions) & set(TRAIN_SESSIONS), (
        "official test sessions must never appear in the training sessions"
    )

    manifest = json.loads(
        (
            PROJECT_ROOT / "reports" / "phase3_continual_pipeline_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["image_copies_created"] == 0
    assert manifest["dataset_modified"] is False
    assert manifest["run_selection"]["all_checks_passed"] is True
    assert manifest["supported_scenarios"] == ["NI", "NC", "NIC"]
    for scenario in manifest["scenarios"]:
        checks = scenario["leakage_check_results"]
        assert checks, f"{scenario['name']}: no leakage checks recorded"
        for check in checks:
            assert check["passed"] is True, f"{scenario['name']}: {check}"


# ---------------------------------------------------------------------------
# 8. Phase-5 artifacts parse and match the locked results
# ---------------------------------------------------------------------------


def test_phase5_artifacts_parse_with_locked_values() -> None:
    reports = PROJECT_ROOT / "reports" / "phase5_nic"
    naive = json.loads((reports / "naive_metrics.json").read_text(encoding="utf-8"))
    replay = json.loads((reports / "replay_metrics.json").read_text(encoding="utf-8"))
    for payload in (naive, replay):
        assert payload["phase"] == 5
        assert payload["scenario"] == "NIC_inc"
        assert payload["variant"] == "inc"
        assert payload["run"] == 0
        assert len(payload["records"]) == 79
        assert [row["experience_id"] for row in payload["records"]] == list(range(79))

    naive_final = naive["records"][-1]["accuracy"]["overall"]
    replay_final = replay["records"][-1]["accuracy"]["overall"]
    assert naive_final == LOCKED_NAIVE_FINAL
    assert naive["records"][-1]["forgetting"] == LOCKED_NAIVE_FORGETTING
    assert replay_final == LOCKED_REPLAY_FINAL
    assert replay["records"][-1]["forgetting"] == LOCKED_REPLAY_FORGETTING

    selection = json.loads(
        (PROJECT_ROOT / "reports" / "phase6_analysis" / "final_model_selection.json")
        .read_text(encoding="utf-8")
    )
    assert selection["selected_method"] == "replay"
    assert selection["sha256"] == LOCKED_FINAL_SHA256
    assert selection["naive_metrics"]["final_accuracy"] == LOCKED_NAIVE_FINAL
    assert (
        selection["naive_metrics"]["average_incremental_accuracy"]
        == LOCKED_NAIVE_AVG_INCR
    )
    assert selection["replay_metrics"]["final_accuracy"] == LOCKED_REPLAY_FINAL
    assert (
        selection["replay_metrics"]["average_incremental_accuracy"]
        == LOCKED_REPLAY_AVG_INCR
    )

    summary = json.loads(
        (reports / "experiment_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "complete"
    assert summary["num_experiences"] == 79
    assert summary["seed"] == 42
