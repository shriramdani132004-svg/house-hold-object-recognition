"""CPU inference engine over the frozen Phase-6 continual model.

Design contract (Phase 7, Part F):

- the checkpoint is loaded **once** per engine instance (and once per
  process via :func:`get_engine`), never per frame;
- the model stays in ``eval()`` mode and every forward pass runs under
  ``torch.no_grad()`` on the CPU;
- preprocessing is delegated to :mod:`src.inference.preprocessing`, which
  mirrors the Phase-5 training transform;
- class names come from the existing official object mapping via
  :func:`src.evaluation.continual.load_class_names`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.core50 import OBJECT_MAPPING_PATH
from src.evaluation.continual import load_class_names
from src.inference.preprocessing import IMAGE_SIZE, preprocess_image
from src.inference.types import (
    SUPPORTS_BOUNDING_BOXES,
    InferenceError,
    InvalidImageError,
    ModelLoadError,
    Prediction,
)
from src.training.checkpoint_schema import (
    CheckpointSchemaError,
    FINAL_MODEL_PATH,
    load_final_model,
)

TOP_K = 3


def _relative(path: Path) -> str:
    try:
        return path.relative_to(FINAL_MODEL_PATH.parents[2]).as_posix()
    except ValueError:
        return path.as_posix()


class InferenceEngine:
    """Loads the frozen model once and serves repeated predictions.

    Args:
        checkpoint_path: Phase-5/6 checkpoint (defaults to
            ``models/continual/final_model.pt``).
        class_mapping_path: official CORe50 object mapping JSON used for
            label -> name resolution.
        image_size: model input resolution (Phase-5 config: 64).
        width: ``SmallConvNet`` base width (Phase-5 config: 32).
        num_classes: expected output classes (CORe50: 50).
    """

    supports_bounding_boxes = SUPPORTS_BOUNDING_BOXES

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        class_mapping_path: str | Path | None = None,
        *,
        image_size: int = IMAGE_SIZE,
        width: int = 32,
        num_classes: int = 50,
    ) -> None:
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else FINAL_MODEL_PATH
        )
        self.class_mapping_path = (
            Path(class_mapping_path)
            if class_mapping_path is not None
            else OBJECT_MAPPING_PATH
        )
        self.image_size = int(image_size)
        self.width = int(width)
        self.num_classes = int(num_classes)
        self.device = torch.device("cpu")
        self._model: torch.nn.Module | None = None
        self._class_names: dict[str, str] = {}
        self._load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise ModelLoadError(
                "Inference engine is not loaded; call load() first."
            )
        return self._model

    @property
    def class_names(self) -> dict[str, str]:
        return dict(self._class_names)

    @property
    def load_count(self) -> int:
        """Number of checkpoint loads performed by this instance."""
        return self._load_count

    def load(self) -> "InferenceEngine":
        """Load and validate the checkpoint + class mapping once."""
        if self._model is not None:
            return self
        if not self.class_mapping_path.is_file():
            raise ModelLoadError(
                f"Class mapping not found: {_relative(self.class_mapping_path)}"
            )
        names = load_class_names(self.class_mapping_path)
        if len(names) != self.num_classes:
            raise ModelLoadError(
                f"Incorrect class mapping: expected {self.num_classes} labels, "
                f"found {len(names)} in "
                f"{_relative(self.class_mapping_path)}"
            )
        try:
            model, _schema = load_final_model(
                self.checkpoint_path,
                num_classes=self.num_classes,
                width=self.width,
            )
        except CheckpointSchemaError as exc:
            raise ModelLoadError(str(exc)) from exc
        model.to(self.device)
        model.eval()
        self._model = model
        self._class_names = names
        self._load_count += 1
        return self

    def predict(self, image: object) -> Prediction:
        """Classify a PIL image, NumPy image, or camera frame."""
        self.load()
        batch = preprocess_image(image, self.image_size)
        return self._forward(batch)

    def predict_pil(self, image: object) -> Prediction:
        return self.predict(image)

    def predict_numpy(self, image: object) -> Prediction:
        return self.predict(image)

    def predict_frame(self, frame: object) -> Prediction:
        """Camera-frame entry point; same backend as every other input."""
        return self.predict(frame)

    def _forward(self, batch: torch.Tensor) -> Prediction:
        try:
            with torch.no_grad():
                logits = self.model(batch.to(self.device))
        except InferenceError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean error
            raise InferenceError(f"Prediction failed: {exc}") from exc
        scores = logits[0]
        if scores.numel() != self.num_classes:
            raise InferenceError(
                f"Model returned {scores.numel()} classes; "
                f"expected {self.num_classes}."
            )
        if not torch.isfinite(scores).all():
            raise InferenceError("Model produced non-finite outputs.")
        probs = torch.softmax(scores, dim=0)
        confidence, index = torch.max(probs, dim=0)
        class_index = int(index.item())
        name = self._class_names.get(str(class_index))
        if name is None:
            raise ModelLoadError(
                f"Class mapping has no name for label {class_index}."
            )
        top_values, top_indices = torch.topk(probs, k=min(TOP_K, probs.numel()))
        top3 = tuple(
            (self._class_names[str(int(i))], float(v))
            for v, i in zip(top_values, top_indices)
            if str(int(i)) in self._class_names
        )
        return Prediction(
            object_name=name,
            confidence=max(0.0, min(1.0, float(confidence.item()))),
            class_index=class_index,
            top3=top3,
        )


_ENGINE: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """Process-wide cached engine: one checkpoint load, reused forever."""
    global _ENGINE
    if _ENGINE is None:
        engine = InferenceEngine()
        engine.load()
        _ENGINE = engine
    return _ENGINE


def reset_engine() -> None:
    """Drop the cached engine (tests/tools only; not used by the app)."""
    global _ENGINE
    _ENGINE = None
