"""Reusable Phase-7 inference layer over the frozen continual model.

Public API::

    from src.inference import InferenceEngine, Prediction, get_engine

    engine = get_engine()                     # checkpoint loaded once
    prediction = engine.predict(pil_image)    # PIL / NumPy / camera frame
    print(prediction.format_block())          # OBJECT / CONFIDENCE / CLASS ID

Supported inputs: ``PIL.Image.Image``, RGB/RGBA/grayscale NumPy arrays
(``uint8``, or ``float`` in ``[0, 1]``), and Gradio camera frames (passed
through the same path). Invalid input raises ``InvalidImageError`` with a
user-readable message; load failures raise ``ModelLoadError``.

The selected model is a CORe50 *classifier*: predictions never include
bounding boxes (``SUPPORTS_BOUNDING_BOXES is False``).
"""

from __future__ import annotations

from src.inference.engine import InferenceEngine, get_engine, reset_engine
from src.inference.preprocessing import IMAGE_SIZE, preprocess_image, to_pil_rgb
from src.inference.types import (
    SUPPORTS_BOUNDING_BOXES,
    InferenceError,
    InvalidImageError,
    ModelLoadError,
    Prediction,
)

__all__ = [
    "IMAGE_SIZE",
    "SUPPORTS_BOUNDING_BOXES",
    "InferenceEngine",
    "InferenceError",
    "InvalidImageError",
    "ModelLoadError",
    "Prediction",
    "get_engine",
    "preprocess_image",
    "reset_engine",
    "to_pil_rgb",
]
