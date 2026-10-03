"""Hardening tests: confidence threshold, closed-set flag, input contracts.

All inputs are small synthetic images; the frozen Phase-6 checkpoint is
only read, never modified. No CORe50 data, training, camera hardware,
network, or deployment is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.inference import (
    IMAGE_SIZE,
    InferenceEngine,
    InvalidImageError,
    Prediction,
    get_engine,
    reset_engine,
)
from src.training.checkpoint_schema import FINAL_MODEL_PATH

MODEL_PRESENT = FINAL_MODEL_PATH.is_file()
needs_model = pytest.mark.skipif(
    not MODEL_PRESENT, reason="frozen final model not present"
)


@pytest.fixture(scope="module")
def rgb_pil() -> Image.Image:
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (180, 40, 90))


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    return InferenceEngine().load()


# --- prediction contract on a synthetic image ---


@needs_model
def test_predict_returns_typed_closed_set_prediction(
    engine: InferenceEngine, rgb_pil: Image.Image
) -> None:
    prediction = engine.predict(rgb_pil)
    assert isinstance(prediction, Prediction)
    assert 0 <= prediction.class_index <= 49
    assert prediction.object_name in engine.class_names.values()
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.top3) <= 3
    scores = [score for _, score in prediction.top3]
    assert scores == sorted(scores, reverse=True)
    assert prediction.bounding_box is None
    assert prediction.uncertain is False


# --- min_confidence validation ---


@pytest.mark.parametrize("bad_value", [1.1, -0.1])
def test_min_confidence_out_of_range_raises(bad_value: float) -> None:
    with pytest.raises(ValueError):
        InferenceEngine(min_confidence=bad_value)


def test_min_confidence_defaults_to_none() -> None:
    assert InferenceEngine().min_confidence is None
    assert InferenceEngine(min_confidence=0.25).min_confidence == 0.25


@needs_model
def test_impossible_threshold_marks_prediction_uncertain(
    rgb_pil: Image.Image,
) -> None:
    strict = InferenceEngine(min_confidence=1.0).load()
    prediction = strict.predict(rgb_pil)
    assert prediction.uncertain is True
    assert 0 <= prediction.class_index <= 49
    assert prediction.object_name in strict.class_names.values()
    assert 0.0 <= prediction.confidence < 1.0
    floor = InferenceEngine(min_confidence=0.0).load()
    assert floor.predict(rgb_pil).uncertain is False


# --- supported and rejected input types ---


@needs_model
def test_supported_input_types_return_predictions(
    engine: InferenceEngine,
) -> None:
    rgb_np = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    float_np = rgb_np.astype(np.float32) / 255.0
    rgba_np = np.zeros((32, 48, 4), dtype=np.uint8)
    rgba_np[..., 1] = 200
    rgba_np[..., 3] = 255
    gray_np = np.full((40, 40), 77, dtype=np.uint8)
    gray_pil = Image.new("L", (40, 40), 77)
    for image in (rgb_np, float_np, rgba_np, gray_np, gray_pil):
        assert isinstance(engine.predict(image), Prediction)


@needs_model
@pytest.mark.parametrize("bad_input", [None, "not-an-image", object()])
def test_invalid_input_types_raise_invalid_image_error(
    engine: InferenceEngine, bad_input: object
) -> None:
    with pytest.raises(InvalidImageError):
        engine.predict(bad_input)


@needs_model
def test_float_array_above_one_raises(engine: InferenceEngine) -> None:
    with pytest.raises(InvalidImageError):
        engine.predict(np.full((8, 8, 3), 2.0, dtype=np.float32))


# --- process-wide singleton ---


@needs_model
def test_singleton_loads_once_and_reset_drops_it() -> None:
    reset_engine()
    try:
        first = get_engine()
        second = get_engine()
        assert first is second
        assert first.load_count == 1
        reset_engine()
        third = get_engine()
        assert third is not first
        assert third.load_count == 1
    finally:
        reset_engine()


# --- eval mode + no_grad ---


@needs_model
def test_predict_keeps_eval_mode_and_creates_no_grads(
    engine: InferenceEngine, rgb_pil: Image.Image
) -> None:
    engine.predict(rgb_pil)
    model = engine.model
    assert model.training is False
    params = list(model.parameters())
    assert params
    for param in params[:5]:
        assert param.grad is None
