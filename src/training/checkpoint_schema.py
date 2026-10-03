"""Documented reader for the Phase-5/6 continual checkpoint schema.

Phase-5/6 checkpoints are ``torch.save``-d dictionaries with one canonical
weights key: **``model_state``**. There is intentionally no top-level
``state_dict`` key, so generic loaders that require one fail by design;
code must use the helpers here instead of guessing key names.

Full payload layout (``format_version == 1``)::

    {
      "format_version": 1,
      "method":          "naive" | "replay",
      "num_classes":     50,
      "current_experience": int,       # last trained experience id
      "seed":            int,
      "image_size":      int,          # model input resolution (new checkpoints)
      "arch":            str,          # model architecture id (new checkpoints)
      "model_state":     OrderedDict[str, Tensor],   # SmallConvNet weights
      "optimizer_state": dict,                        # Adam state
      "scheduler_state": dict | None,
      "replay_memory":   dict,                        # replay method only
      "replay_stats":    dict,                        # replay method only
    }

Legacy checkpoints (written before ``image_size``/``arch`` existed) remain
loadable; readers treat those keys as optional and validate them only when
present. Reading or validating a checkpoint never modifies the file on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from src.training.model import assert_model_contract, build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "continual" / "final_model.pt"

FORMAT_VERSION = 1
REQUIRED_KEYS: tuple[str, ...] = (
    "format_version",
    "method",
    "num_classes",
    "current_experience",
    "model_state",
)
REPLAY_KEYS: tuple[str, ...] = ("replay_memory", "replay_stats")


class CheckpointSchemaError(ValueError):
    """Checkpoint missing, unreadable, or violating the documented schema."""


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointSchemaError(f"checkpoint not found: {_relative(path)}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch raises many load-time error types
        raise CheckpointSchemaError(
            f"checkpoint unreadable ({_relative(path)}): {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointSchemaError(
            f"checkpoint must be a dict payload, got {type(payload).__name__}"
        )
    return payload


def read_checkpoint_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Summarise a checkpoint's schema without touching model weights.

    Returns project-relative path, size, SHA-256, the documented scalar
    fields, the ``model_state`` tensor inventory, and which optional
    sections are present. Raises :class:`CheckpointSchemaError` when the
    file is missing, unreadable, or not a dict payload.
    """
    checkpoint = Path(path) if path is not None else FINAL_MODEL_PATH
    payload = _load_payload(checkpoint)
    model_state = payload.get("model_state")
    tensors = (
        {name: tensor for name, tensor in model_state.items()}
        if isinstance(model_state, dict)
        else {}
    )
    all_tensors = all(isinstance(value, torch.Tensor) for value in tensors.values())
    return {
        "path": _relative(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "top_level_keys": sorted(payload),
        "format_version": payload.get("format_version"),
        "method": payload.get("method"),
        "num_classes": payload.get("num_classes"),
        "current_experience": payload.get("current_experience"),
        "seed": payload.get("seed"),
        "image_size": payload.get("image_size"),
        "arch": payload.get("arch"),
        "model_state_keys": sorted(tensors),
        "model_state_tensors": len(tensors),
        "model_state_all_tensors": all_tensors,
        "has_optimizer_state": "optimizer_state" in payload,
        "has_scheduler_state": payload.get("scheduler_state") is not None,
        "has_replay_memory": "replay_memory" in payload,
        "has_replay_stats": "replay_stats" in payload,
    }


def validate_checkpoint_schema(
    path: str | Path | None = None,
    *,
    expected_num_classes: int | None = None,
    expected_method: str | None = None,
) -> list[str]:
    """Validate a checkpoint against the documented schema.

    Returns a list of human-readable problems; an empty list means the
    checkpoint fully conforms. The file itself is only read.
    """
    checkpoint = Path(path) if path is not None else FINAL_MODEL_PATH
    try:
        payload = _load_payload(checkpoint)
    except CheckpointSchemaError as exc:
        return [str(exc)]

    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(
                f"missing required key {key!r} (canonical weights key is "
                "'model_state', not 'state_dict')"
            )
    if payload.get("format_version") != FORMAT_VERSION:
        errors.append(
            f"format_version {payload.get('format_version')!r} != {FORMAT_VERSION}"
        )
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        errors.append(f"'method' must be a non-empty string, got {method!r}")
    elif expected_method is not None and method != expected_method:
        errors.append(f"method {method!r} != expected {expected_method!r}")
    num_classes = payload.get("num_classes")
    if not isinstance(num_classes, int) or num_classes <= 0:
        errors.append(f"'num_classes' must be a positive int, got {num_classes!r}")
    elif expected_num_classes is not None and num_classes != expected_num_classes:
        errors.append(
            f"num_classes {num_classes} != expected {expected_num_classes}"
        )
    current = payload.get("current_experience")
    if not isinstance(current, int) or current < 0:
        errors.append(
            f"'current_experience' must be a non-negative int, got {current!r}"
        )
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        errors.append("'model_state' must be a non-empty dict of tensors")
    else:
        non_tensors = [
            name for name, value in model_state.items() if not isinstance(value, torch.Tensor)
        ]
        if non_tensors:
            errors.append(f"'model_state' has non-tensor entries: {non_tensors[:5]}")
    image_size = payload.get("image_size")
    if image_size is not None and (not isinstance(image_size, int) or image_size <= 0):
        errors.append(f"'image_size' must be a positive int when present, got {image_size!r}")
    arch = payload.get("arch")
    if arch is not None and (not isinstance(arch, str) or not arch):
        errors.append(f"'arch' must be a non-empty string when present, got {arch!r}")
    return errors


def load_final_model(
    path: str | Path | None = None,
    *,
    num_classes: int = 50,
    width: int = 32,
    image_size: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a continual checkpoint for inference with full validation.

    Validates the documented schema, builds the architecture recorded in
    the payload (``small_cnn`` for legacy checkpoints without ``arch``),
    loads ``model_state`` with ``strict=True``, verifies any recorded
    ``image_size`` against the requested one, and runs
    :func:`src.training.model.assert_model_contract` so a structurally
    wrong or non-finite model fails here instead of at predict time.
    Returns ``(model, schema)`` with the model in eval mode. Raises
    :class:`CheckpointSchemaError` when validation fails.
    """
    checkpoint = Path(path) if path is not None else FINAL_MODEL_PATH
    problems = validate_checkpoint_schema(
        checkpoint, expected_num_classes=num_classes
    )
    if problems:
        raise CheckpointSchemaError(
            f"invalid checkpoint {_relative(checkpoint)}: " + "; ".join(problems)
        )
    payload = _load_payload(checkpoint)
    recorded_size = payload.get("image_size")
    if (
        image_size is not None
        and recorded_size is not None
        and int(recorded_size) != int(image_size)
    ):
        raise CheckpointSchemaError(
            f"checkpoint {_relative(checkpoint)} was trained at "
            f"image_size={recorded_size} but image_size={image_size} was requested"
        )
    arch = payload.get("arch") or "small_cnn"
    model = build_model(num_classes, width=width, arch=arch)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise CheckpointSchemaError(
            f"weight shapes in {_relative(checkpoint)} do not match arch "
            f"{arch!r} (width {width}): {exc}"
        ) from exc
    model.eval()
    assert_model_contract(
        model,
        num_classes=num_classes,
        image_size=int(image_size if image_size is not None else recorded_size or 64),
    )
    return model, read_checkpoint_schema(checkpoint)
