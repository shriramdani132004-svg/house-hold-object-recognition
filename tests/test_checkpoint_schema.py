"""Checkpoint schema tests with synthetic payloads only (Phase 5/6 reader).

Builds a tiny ``SmallConvNet`` and ``torch.save``'s synthetic checkpoint
dictionaries into ``tmp_path`` to pin the documented behaviour of
``src/training/checkpoint_schema.py``: required keys, scalar field
validation, expected-value mismatches, missing/garbage files, the final
model loader, and SHA-256 reporting. Never reads ``models/*.pt`` or any
CORe50 artifact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from src.training import (
    CheckpointSchemaError,
    build_model,
    load_final_model,
    read_checkpoint_schema,
    validate_checkpoint_schema,
)
from src.training.checkpoint_schema import FORMAT_VERSION, REQUIRED_KEYS

NUM_CLASSES = 50
WIDTH = 8


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    """Small synthetic classifier matching the payload used by every test."""
    return build_model(NUM_CLASSES, width=WIDTH)


def _payload(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "method": "naive",
        "num_classes": NUM_CLASSES,
        "current_experience": 0,
        "seed": 42,
        "model_state": model.state_dict(),
    }


def _save(tmp_path: Path, payload: Any, name: str = "checkpoint.pt") -> Path:
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_valid_payload_passes_validation(tmp_path: Path, model: torch.nn.Module) -> None:
    path = _save(tmp_path, _payload(model))
    assert validate_checkpoint_schema(path) == []
    assert (
        validate_checkpoint_schema(
            path, expected_num_classes=NUM_CLASSES, expected_method="naive"
        )
        == []
    )


def test_read_schema_reports_documented_fields(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    payload = _payload(model)
    path = _save(tmp_path, payload)
    schema = read_checkpoint_schema(path)

    assert schema["path"].endswith("checkpoint.pt")
    assert schema["size_bytes"] == path.stat().st_size
    assert schema["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert schema["top_level_keys"] == sorted(payload)
    assert schema["format_version"] == FORMAT_VERSION
    assert schema["method"] == "naive"
    assert schema["num_classes"] == NUM_CLASSES
    assert schema["current_experience"] == 0
    assert schema["seed"] == 42
    assert schema["model_state_keys"] == sorted(payload["model_state"])
    assert schema["model_state_tensors"] == len(payload["model_state"])
    assert schema["model_state_all_tensors"] is True
    assert schema["has_optimizer_state"] is False
    assert schema["has_scheduler_state"] is False
    assert schema["has_replay_memory"] is False
    assert schema["has_replay_stats"] is False


def test_optional_sections_are_reported_when_present(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    payload = _payload(model)
    payload["optimizer_state"] = {"state": {}}
    payload["scheduler_state"] = None
    payload["replay_memory"] = {"format_version": FORMAT_VERSION}
    payload["replay_stats"] = {"steps_with_replay": 0}
    path = _save(tmp_path, payload, "replay.pt")
    schema = read_checkpoint_schema(path)
    assert schema["has_optimizer_state"] is True
    assert schema["has_scheduler_state"] is False
    assert schema["has_replay_memory"] is True
    assert schema["has_replay_stats"] is True
    assert validate_checkpoint_schema(path) == []


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_missing_required_key_is_reported(
    tmp_path: Path, model: torch.nn.Module, key: str
) -> None:
    payload = _payload(model)
    del payload[key]
    problems = validate_checkpoint_schema(_save(tmp_path, payload, f"missing_{key}.pt"))
    assert problems, f"missing {key!r} must be reported as a problem"
    assert any(key in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"format_version": FORMAT_VERSION + 1},
            f"format_version {FORMAT_VERSION + 1} != {FORMAT_VERSION}",
        ),
        ({"num_classes": 0}, "'num_classes' must be a positive int, got 0"),
        ({"num_classes": -3}, "'num_classes' must be a positive int, got -3"),
        (
            {"current_experience": -1},
            "'current_experience' must be a non-negative int, got -1",
        ),
        ({"model_state": {}}, "'model_state' must be a non-empty dict of tensors"),
        (
            {"model_state": {"not_a_tensor": 7}},
            "'model_state' has non-tensor entries",
        ),
        ({"method": 5}, "'method' must be a non-empty string, got 5"),
        ({"method": ""}, "'method' must be a non-empty string, got ''"),
    ],
    ids=[
        "format_version",
        "num_classes_zero",
        "num_classes_negative",
        "current_experience_negative",
        "model_state_empty",
        "model_state_non_tensor",
        "method_non_string",
        "method_empty",
    ],
)
def test_specific_problem_string_is_produced(
    tmp_path: Path,
    model: torch.nn.Module,
    changes: dict[str, Any],
    expected: str,
) -> None:
    payload = {**_payload(model), **changes}
    problems = validate_checkpoint_schema(_save(tmp_path, payload, "invalid.pt"))
    assert any(expected in problem for problem in problems), problems


def test_expected_num_classes_mismatch_is_reported(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    path = _save(tmp_path, _payload(model))
    problems = validate_checkpoint_schema(path, expected_num_classes=10)
    assert any("num_classes 50 != expected 10" in problem for problem in problems)


def test_expected_method_mismatch_is_reported(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    path = _save(tmp_path, _payload(model))
    problems = validate_checkpoint_schema(path, expected_method="replay")
    assert any(
        "method 'naive' != expected 'replay'" in problem for problem in problems
    ), problems


def test_missing_file_yields_problem_list_and_schema_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pt"
    problems = validate_checkpoint_schema(missing)
    assert problems
    assert any("checkpoint not found" in problem for problem in problems)
    with pytest.raises(CheckpointSchemaError, match="checkpoint not found"):
        read_checkpoint_schema(missing)


def test_garbage_files_never_leak_raw_exceptions(tmp_path: Path) -> None:
    text_path = _save(tmp_path, "not a dict payload", "text.pt")
    problems = validate_checkpoint_schema(text_path)
    assert any("dict payload" in problem for problem in problems)
    with pytest.raises(CheckpointSchemaError, match="dict payload"):
        read_checkpoint_schema(text_path)

    raw_path = tmp_path / "raw.pt"
    raw_path.write_bytes(b"definitely not a torch checkpoint")
    problems = validate_checkpoint_schema(raw_path)
    assert any("unreadable" in problem for problem in problems)
    with pytest.raises(CheckpointSchemaError, match="unreadable"):
        read_checkpoint_schema(raw_path)


def test_load_final_model_returns_eval_model_with_50_way_head(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    path = _save(tmp_path, _payload(model))
    loaded, schema = load_final_model(path, num_classes=NUM_CLASSES, width=WIDTH)
    assert loaded.training is False
    with torch.no_grad():
        output = loaded(torch.zeros(2, 3, 32, 32))
    assert tuple(output.shape) == (2, NUM_CLASSES)
    assert bool(torch.isfinite(output).all())
    assert schema["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_final_model_rejects_wrong_num_classes(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    path = _save(tmp_path, _payload(model))
    with pytest.raises(CheckpointSchemaError, match="num_classes"):
        load_final_model(path, num_classes=19, width=WIDTH)


def test_load_final_model_raises_on_corrupt_weights(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    payload = _payload(model)
    broken = dict(payload["model_state"])
    broken["classifier.weight"] = torch.zeros(1, 1)
    payload["model_state"] = broken
    path = _save(tmp_path, payload, "corrupt.pt")
    with pytest.raises((RuntimeError, CheckpointSchemaError)) as excinfo:
        load_final_model(path, num_classes=NUM_CLASSES, width=WIDTH)
    message = str(excinfo.value)
    assert "state_dict" in message or "checkpoint" in message
