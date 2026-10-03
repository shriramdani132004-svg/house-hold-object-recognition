"""Prediction result types and error hierarchy for Phase-7 inference.

The frozen Phase-6 selection is a CORe50 *classification* model
(SmallConvNet, 50 classes), so a prediction carries an object name, a
confidence, and a class index — never detection coordinates. The
``bounding_box`` field exists only so callers can check it is always
``None`` for this model; boxes are unsupported, not fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTS_BOUNDING_BOXES = False

TopPrediction = tuple[str, float]


class InferenceError(Exception):
    """Base class for inference failures with a user-readable message."""


class InvalidImageError(InferenceError):
    """Input image is missing, empty, or of an unsupported type/shape."""


class ModelLoadError(InferenceError):
    """Frozen checkpoint or class mapping could not be loaded/validated."""


@dataclass(frozen=True)
class Prediction:
    """One classification result from the frozen continual model.

    ``confidence`` is the softmax score over the 50 CORe50 output
    classes: a *model confidence*, NOT a calibrated probability that the
    prediction is correct. The classifier is closed-set — every input is
    forced to one of the 50 learned identities, even when the true object
    is out of scope.

    Attributes:
        object_name: CORe50 object identity name (e.g. ``plug_adapter1``).
        confidence: softmax score over the 50 output classes, in
            ``[0.0, 1.0]`` (model confidence, not a calibrated
            probability of correctness).
        class_index: 0-based label in ``[0, 49]``.
        top3: optional ``(name, score)`` triples for debugging.
        bounding_box: always ``None`` — this classifier emits no boxes.
        uncertain: ``True`` when the engine's ``min_confidence``
            threshold rejected the top score; the other fields still
            describe the argmax prediction.
    """

    object_name: str
    confidence: float
    class_index: int
    top3: tuple[TopPrediction, ...] = ()
    bounding_box: tuple[float, float, float, float] | None = None
    uncertain: bool = False

    @property
    def supports_bounding_boxes(self) -> bool:
        return SUPPORTS_BOUNDING_BOXES

    @property
    def formatted_confidence(self) -> str:
        """Confidence as ``0.00%`` … ``100.00%``."""
        return f"{self.confidence * 100:.2f}%"

    def format_block(self) -> str:
        """The Phase-7 output contract: OBJECT / CONFIDENCE / CLASS ID."""
        return (
            f"OBJECT: {self.object_name}\n"
            f"CONFIDENCE: {self.formatted_confidence}\n"
            f"CLASS ID: {self.class_index}"
        )
