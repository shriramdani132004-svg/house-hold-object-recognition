"""ContinualTrainingState persistence tests (Phase-4 state contract).

Synthetic payloads and tiny generated-PNG scenarios only — no CORe50
filelists, no ``models/*.pt`` reads, no training runs beyond a few
zero-step experiences. Pins save/load round-trips, ``validate_state``
mismatch reporting, corrupt ``state.json`` error types, and the base
trainer's state/payload consistency plus official-order guards.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from src.training import (
    CheckpointMissingError,
    ContinualStateError,
    ContinualTrainingError,
    ContinualTrainingState,
    StateCompatibilityError,
    build_continual_trainer,
    checkpoint_exists,
    load_state,
    save_state,
    validate_state,
)
from src.training.state import PAYLOAD_VERSION, STATE_VERSION

STATE_KWARGS: dict[str, Any] = dict(
    scenario="NIC",
    variant="inc",
    run_id=0,
    method="replay",
    current_experience=2,
    experiences_trained=(0, 1, 2),
    epochs_completed=3,
    steps_completed=42,
    seed=42,
    config_fingerprint="feedfacecafe0000",
    training_config={"epochs": 1, "batch_size": 8},
    replay={"size": 12, "last_experience": 2},
    created_utc="2026-01-02T03:04:05+00:00",
)


def _state(**overrides: Any) -> ContinualTrainingState:
    return ContinualTrainingState(**{**STATE_KWARGS, **overrides})


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": PAYLOAD_VERSION,
        "method": "replay",
        "current_experience": 2,
        "seed": 42,
        "model_state": {"classifier.weight": torch.zeros(4, 8)},
    }
    payload.update(overrides)
    return payload


def _write_run(
    directory: Path,
    *,
    state: ContinualTrainingState | None = None,
    payload: dict[str, Any] | None = None,
) -> Path:
    save_state(state or _state(), directory, payload=payload or _payload())
    return directory


def _mutate_state_json(
    directory: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    path = directory / "state.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _drop_seed(raw: dict[str, Any]) -> None:
    del raw["seed"]


def _bump_version(raw: dict[str, Any]) -> None:
    raw["version"] = STATE_VERSION + 1


def _set_current_experience(value: int) -> Callable[[dict[str, Any]], None]:
    def mutate(raw: dict[str, Any]) -> None:
        raw["current_experience"] = value

    return mutate


def test_save_load_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    original = _state()
    state_path = save_state(original, run_dir, payload=_payload())
    assert state_path == run_dir / "state.json"
    assert (run_dir / "state.json").is_file()
    assert (run_dir / "checkpoint.pt").is_file()

    loaded, payload = load_state(run_dir)
    assert {
        key: value for key, value in loaded.to_dict().items() if key != "updated_utc"
    } == {
        key: value for key, value in original.to_dict().items() if key != "updated_utc"
    }
    assert loaded.created_utc == original.created_utc
    assert datetime.fromisoformat(loaded.updated_utc).tzinfo is not None
    assert loaded.experiences_trained == (0, 1, 2)
    assert isinstance(loaded.experiences_trained, tuple)
    assert loaded.version == STATE_VERSION

    raw = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert raw["created_utc"] == loaded.created_utc
    assert raw["updated_utc"] == loaded.updated_utc

    assert payload["format_version"] == PAYLOAD_VERSION
    assert payload["current_experience"] == 2
    assert torch.equal(payload["model_state"]["classifier.weight"], torch.zeros(4, 8))
    assert checkpoint_exists(run_dir) is True


def test_validate_state_accepts_matching_expectations() -> None:
    state = _state()
    validate_state(state)
    validate_state(
        state,
        scenario="NIC",
        variant="inc",
        run_id=0,
        method="replay",
        config_fingerprint="feedfacecafe0000",
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("scenario", "NC"),
        ("variant", "dec"),
        ("run_id", 1),
        ("method", "naive"),
        ("config_fingerprint", "deadbeef00000000"),
    ],
)
def test_validate_state_reports_each_mismatch(
    field: str, wrong_value: Any
) -> None:
    state = _state()
    with pytest.raises(
        StateCompatibilityError, match="State compatibility check failed"
    ) as excinfo:
        validate_state(state, **{field: wrong_value})
    message = str(excinfo.value)
    assert field in message
    assert f"expected {wrong_value!r}" in message
    assert "state has" in message


@pytest.mark.parametrize(
    ("case_id", "corrupt", "match"),
    [
        ("missing_field", _drop_seed, "missing required key"),
        ("wrong_version", _bump_version, "Unsupported state version"),
    ],
)
def test_corrupt_state_json_raises_state_compatibility_error(
    tmp_path: Path, case_id: str, corrupt: Callable[[dict[str, Any]], None], match: str
) -> None:
    run_dir = _write_run(tmp_path / case_id)
    _mutate_state_json(run_dir, corrupt)
    with pytest.raises(StateCompatibilityError, match=match):
        load_state(run_dir)
    assert checkpoint_exists(run_dir) is False


def test_invalid_json_raises_state_error_not_raw_json_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_json"
    run_dir.mkdir()
    (run_dir / "state.json").write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(StateCompatibilityError, match="not valid JSON") as excinfo:
        load_state(run_dir)
    assert not isinstance(excinfo.value, json.JSONDecodeError)
    assert isinstance(excinfo.value, ContinualStateError)


def test_missing_state_and_checkpoint_files_raise_documented_errors(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ContinualStateError, match="State file not found"):
        load_state(empty)
    assert checkpoint_exists(empty) is False

    run_dir = _write_run(tmp_path / "no_ckpt")
    (run_dir / "checkpoint.pt").unlink()
    with pytest.raises(
        CheckpointMissingError, match="Checkpoint referenced by state is missing"
    ):
        load_state(run_dir)
    assert checkpoint_exists(run_dir) is False


def test_checkpoint_payload_problems_raise_documented_errors(tmp_path: Path) -> None:
    run_a = _write_run(tmp_path / "payload_version", payload=_payload(format_version=99))
    with pytest.raises(
        StateCompatibilityError, match="Unsupported checkpoint payload format"
    ):
        load_state(run_a)

    run_b = _write_run(tmp_path / "payload_not_mapping")
    torch.save([1, 2, 3], run_b / "checkpoint.pt")
    with pytest.raises(StateCompatibilityError, match="Checkpoint payload is malformed"):
        load_state(run_b)

    run_c = _write_run(tmp_path / "payload_unreadable")
    (run_c / "checkpoint.pt").write_bytes(b"garbage bytes")
    with pytest.raises(
        StateCompatibilityError, match="Could not load checkpoint payload"
    ):
        load_state(run_c)


def test_save_state_rejects_payload_without_required_keys(tmp_path: Path) -> None:
    with pytest.raises(
        StateCompatibilityError, match="format_version"
    ):
        save_state(_state(), tmp_path / "bad_payload", payload={"model_state": {}})
    with pytest.raises(
        StateCompatibilityError, match="'format_version' and 'model_state'"
    ):
        save_state(_state(), tmp_path / "bad_payload_2", payload="not-a-mapping")


def test_from_dict_rejects_non_mapping_payload() -> None:
    with pytest.raises(StateCompatibilityError, match="must be a mapping"):
        ContinualTrainingState.from_dict([1, 2, 3])


def test_resume_rejects_state_json_vs_payload_experience_mismatch(
    tmp_path: Path,
) -> None:
    from test_phase4_continual_training import make_synthetic_scenario, tiny_config

    scenario = make_synthetic_scenario(tmp_path / "scenario")
    run_dir = tmp_path / "resume_run"
    trainer = build_continual_trainer("naive", tiny_config("naive", max_steps=0))
    state = trainer.initialize(scenario, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, scenario.experiences[0])
    assert state.current_experience == 0

    _mutate_state_json(run_dir, _set_current_experience(1))
    resumed = build_continual_trainer("naive", tiny_config("naive", max_steps=0))
    with pytest.raises(
        StateCompatibilityError, match="Checkpoint payload is at experience"
    ):
        resumed.resume(scenario, checkpoint_dir=run_dir)


def test_train_experience_rejects_on_disk_state_drift(tmp_path: Path) -> None:
    from test_phase4_continual_training import make_synthetic_scenario, tiny_config

    scenario = make_synthetic_scenario(tmp_path / "scenario")
    run_dir = tmp_path / "drift_run"
    trainer = build_continual_trainer("naive", tiny_config("naive", max_steps=0))
    state = trainer.initialize(scenario, checkpoint_dir=run_dir)
    state = trainer.train_experience(state, scenario.experiences[0])

    _mutate_state_json(run_dir, _set_current_experience(5))
    with pytest.raises(StateCompatibilityError, match="On-disk state is at experience"):
        trainer.train_experience(state, scenario.experiences[1])


def test_train_experience_enforces_official_order(tmp_path: Path) -> None:
    from test_phase4_continual_training import make_synthetic_scenario, tiny_config

    scenario = make_synthetic_scenario(tmp_path / "scenario")
    trainer = build_continual_trainer("naive", tiny_config("naive", max_steps=0))
    state = trainer.initialize(scenario, checkpoint_dir=tmp_path / "order_run")

    with pytest.raises(ContinualTrainingError, match="Official order violated"):
        trainer.train_experience(state, scenario.experiences[1])
    state = trainer.train_experience(state, scenario.experiences[0])
    with pytest.raises(ContinualTrainingError, match="Official order violated"):
        trainer.train_experience(state, scenario.experiences[0])

    assert state.current_experience == 0
    assert state.experiences_trained == (0,)
