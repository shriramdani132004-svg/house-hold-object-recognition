"""Phase 7 tests: frozen-model inference, input handling, and the Gradio app.

All inputs are small synthetic images; no CORe50 data, training, camera
hardware, network, or deployment is required.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.core50 import OBJECT_MAPPING_PATH
from src.evaluation.continual import load_class_names
from src.inference import (
    IMAGE_SIZE,
    SUPPORTS_BOUNDING_BOXES,
    InferenceEngine,
    InferenceError,
    InvalidImageError,
    ModelLoadError,
    Prediction,
    get_engine,
)
from src.training.checkpoint_schema import FINAL_MODEL_PATH, read_checkpoint_schema

LOCKED_SHA256 = (
    "b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_RE = re.compile(r"^\d{1,3}\.\d{2}%$")


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    return InferenceEngine().load()


@pytest.fixture(scope="module")
def rgb_pil() -> Image.Image:
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (180, 40, 90))


@pytest.fixture(scope="module")
def rgb_np(rgb_pil: Image.Image) -> np.ndarray:
    return np.asarray(rgb_pil)


def _load_app():
    return importlib.import_module("app")


def _tree_state(roots: list[tuple[Path, bool]]) -> list[tuple]:
    state: list[tuple] = []
    for root, deep in roots:
        if not root.exists():
            state.append((root.name, "missing"))
            continue
        entries = sorted(root.rglob("*")) if deep else sorted(root.iterdir())
        for path in entries:
            rel = path.relative_to(root).as_posix()
            if path.is_file():
                stat = path.stat()
                state.append((rel, stat.st_size, stat.st_mtime_ns))
            else:
                state.append((rel, "dir"))
    return state


# --- 1-4: model loading, classes, mapping, engine initialization ---


def test_frozen_model_loads(engine: InferenceEngine) -> None:
    assert engine.is_loaded
    assert isinstance(engine.model, torch.nn.Module)
    assert engine.model.training is False
    assert engine.load_count == 1
    assert engine.device.type == "cpu"


def test_model_has_50_output_classes(engine: InferenceEngine) -> None:
    with torch.no_grad():
        output = engine.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert tuple(output.shape) == (1, 50)
    schema = read_checkpoint_schema(FINAL_MODEL_PATH)
    assert schema["num_classes"] == 50


def test_class_mapping_resolves(engine: InferenceEngine) -> None:
    names = engine.class_names
    assert len(names) == 50
    assert set(names) == {str(i) for i in range(50)}
    assert names == load_class_names(OBJECT_MAPPING_PATH)
    assert names["0"] == "plug_adapter1"
    assert names[str(engine.predict(Image.new("RGB", (64, 64), (1, 2, 3))).class_index)]


def test_engine_initializes_with_project_defaults() -> None:
    fresh = InferenceEngine()
    assert not fresh.is_loaded
    assert fresh.checkpoint_path == FINAL_MODEL_PATH
    assert fresh.checkpoint_path.is_file()
    assert fresh.class_mapping_path == OBJECT_MAPPING_PATH
    assert fresh.image_size == 64
    assert fresh.width == 32
    assert fresh.num_classes == 50
    assert fresh.supports_bounding_boxes is False


# --- 5-7: PIL / NumPy / RGBA inputs ---


def test_pil_image_inference(engine: InferenceEngine, rgb_pil: Image.Image) -> None:
    prediction = engine.predict(rgb_pil)
    assert isinstance(prediction, Prediction)
    assert prediction.object_name in engine.class_names.values()


def test_numpy_rgb_image_inference(
    engine: InferenceEngine, rgb_np: np.ndarray, rgb_pil: Image.Image
) -> None:
    from_array = engine.predict_numpy(rgb_np)
    from_pil = engine.predict_pil(rgb_pil)
    assert from_array == from_pil


def test_rgba_and_grayscale_inputs_are_handled(engine: InferenceEngine) -> None:
    rgba_pil = Image.new("RGBA", (80, 60), (10, 20, 30, 200))
    assert isinstance(engine.predict(rgba_pil), Prediction)
    rgba_np = np.zeros((60, 80, 4), dtype=np.uint8)
    rgba_np[..., 0] = 200
    rgba_np[..., 3] = 255
    assert isinstance(engine.predict(rgba_np), Prediction)
    gray_pil = Image.new("L", (50, 50), 77)
    assert isinstance(engine.predict(gray_pil), Prediction)
    gray_np = np.full((50, 50), 77, dtype=np.uint8)
    assert isinstance(engine.predict(gray_np), Prediction)


# --- 8: invalid inputs ---


@pytest.mark.parametrize(
    "bad_input, label",
    [
        (None, "missing"),
        ("not-an-image", "string"),
        (np.zeros((0,), dtype=np.uint8), "empty"),
        (np.zeros((5,), dtype=np.uint8), "1-D"),
        (np.zeros((8, 8, 2), dtype=np.uint8), "channels"),
        (np.zeros((8, 8), dtype=np.int32), "dtype"),
        (np.full((8, 8, 3), 200.0, dtype=np.float64), "float-range"),
        (np.full((6, 6, 3), np.nan, dtype=np.float32), "nan"),
        (np.zeros((2, 4, 6, 8), dtype=np.uint8), "4-D"),
    ],
)
def test_invalid_inputs_raise_readable_errors(
    engine: InferenceEngine, bad_input: object, label: str
) -> None:
    with pytest.raises(InvalidImageError) as excinfo:
        engine.predict(bad_input)
    assert isinstance(excinfo.value, InferenceError)
    assert str(excinfo.value).strip(), label


def test_zero_size_pil_raises(engine: InferenceEngine) -> None:
    with pytest.raises(InvalidImageError):
        engine.predict(Image.new("RGB", (0, 0)))


def test_unloaded_engine_error_and_model_load_error_paths() -> None:
    with pytest.raises(ModelLoadError):
        InferenceEngine().model
    with pytest.raises(ModelLoadError):
        InferenceEngine(checkpoint_path=PROJECT_ROOT / "missing_ckpt.pt").load()
    with pytest.raises(ModelLoadError):
        InferenceEngine(class_mapping_path=PROJECT_ROOT / "missing_map.json").load()


# --- 9-12: prediction contract ---


def test_prediction_returns_name_confidence_class_index(
    engine: InferenceEngine, rgb_pil: Image.Image
) -> None:
    prediction = engine.predict(rgb_pil)
    assert isinstance(prediction.object_name, str) and prediction.object_name
    assert isinstance(prediction.confidence, float)
    assert isinstance(prediction.class_index, int)
    block = prediction.format_block().splitlines()
    assert block == [
        f"OBJECT: {prediction.object_name}",
        f"CONFIDENCE: {prediction.formatted_confidence}",
        f"CLASS ID: {prediction.class_index}",
    ]
    assert CONFIDENCE_RE.match(prediction.formatted_confidence)
    assert 0.0 <= prediction.confidence <= 1.0


def test_confidence_is_finite_and_within_bounds(engine: InferenceEngine) -> None:
    for color in [(0, 0, 0), (255, 255, 255), (12, 200, 90)]:
        prediction = engine.predict(Image.new("RGB", (64, 64), color))
        assert math.isfinite(prediction.confidence)
        assert 0.0 <= prediction.confidence <= 1.0


def test_class_index_is_in_valid_range(engine: InferenceEngine) -> None:
    for color in [(0, 0, 0), (255, 0, 0), (30, 30, 200)]:
        prediction = engine.predict(Image.new("RGB", (96, 48), color))
        assert 0 <= prediction.class_index <= 49


# --- 13: model reuse ---


def test_repeated_predictions_reuse_loaded_model() -> None:
    fresh = InferenceEngine()
    first = fresh.predict(Image.new("RGB", (64, 64), (5, 6, 7)))
    model_id = id(fresh.model)
    second = fresh.predict(Image.new("RGB", (32, 80), (250, 3, 90)))
    assert fresh.load_count == 1
    assert id(fresh.model) == model_id
    assert first.class_index >= 0 and second.class_index >= 0


def test_get_engine_returns_cached_singleton() -> None:
    first = get_engine()
    second = get_engine()
    assert first is second
    assert first.load_count == 1


def test_set_engine_installs_shared_singleton() -> None:
    from src.inference import set_engine

    original = get_engine()
    replacement = InferenceEngine().load()
    set_engine(replacement)
    try:
        assert get_engine() is replacement
    finally:
        set_engine(original)
    assert get_engine() is original
    with pytest.raises(TypeError):
        set_engine(object())  # type: ignore[arg-type]


# --- 14: frozen checkpoint integrity ---


def test_final_checkpoint_hash_unchanged() -> None:
    digest = hashlib.sha256(FINAL_MODEL_PATH.read_bytes()).hexdigest()
    assert digest == LOCKED_SHA256


# --- 15: no bounding-box claims ---


def test_classifier_does_not_claim_bounding_boxes(
    engine: InferenceEngine, rgb_pil: Image.Image
) -> None:
    assert SUPPORTS_BOUNDING_BOXES is False
    assert engine.supports_bounding_boxes is False
    prediction = engine.predict(rgb_pil)
    assert prediction.bounding_box is None
    assert prediction.supports_bounding_boxes is False
    prefixes = [line.split(":", 1)[0] for line in prediction.format_block().splitlines()]
    assert prefixes == ["OBJECT", "CONFIDENCE", "CLASS ID"]


# --- 16-18: Gradio app import, construction, callbacks ---


def test_gradio_app_imports() -> None:
    app_module = _load_app()
    module_file = Path(app_module.__file__)
    assert module_file is not None
    assert module_file.name == "__init__.py"
    assert module_file.parent.name == "app"
    main_module = importlib.import_module("app.main")
    assert callable(main_module.main)
    assert (module_file.parent / "__main__.py").is_file()  # `python -m app`
    assert callable(app_module.predict_callback)
    assert callable(app_module.build_app)


def test_gradio_interface_construction() -> None:
    app_module = _load_app()
    demo = app_module.build_app()
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    assert len(config["components"]) >= 8
    dependencies = config["dependencies"]
    assert len(dependencies) == 2  # image.change + button.click
    triggers = {dep["targets"][0][1] for dep in dependencies}
    assert triggers == {"change", "click"}


def test_upload_callback_works_with_synthetic_image(
    rgb_pil: Image.Image, rgb_np: np.ndarray
) -> None:
    app_module = _load_app()
    objects, confidence, class_id, error = app_module.predict_callback(rgb_pil)
    assert objects
    assert CONFIDENCE_RE.match(confidence)
    assert 0 <= int(class_id) <= 49
    assert error == ""
    assert app_module.predict_callback(rgb_np) == (
        objects,
        confidence,
        class_id,
        error,
    )


def test_camera_callback_uses_same_inference_backend(
    rgb_pil: Image.Image, rgb_np: np.ndarray
) -> None:
    app_module = _load_app()
    frame_result = app_module.predict_callback(rgb_np)
    upload_result = app_module.predict_callback(rgb_pil)
    assert frame_result == upload_result
    demo = app_module.build_app()
    dependencies = demo.get_config_file()["dependencies"]
    handlers = [demo.fns[dep["id"]].fn for dep in dependencies]
    assert handlers[0] is handlers[1] is app_module.predict_callback


# --- 20: app construction touches no data ---


def test_app_initialization_does_not_touch_data() -> None:
    app_module = _load_app()
    deep_roots = [
        (PROJECT_ROOT / "data" / "splits", True),
        (PROJECT_ROOT / "models" / "continual", True),
        (PROJECT_ROOT / "reports" / "phase5_nic", True),
    ]
    shallow_roots = [
        (PROJECT_ROOT / "data" / "raw", False),
        (PROJECT_ROOT / "data" / "processed", False),
    ]
    before = _tree_state(deep_roots + shallow_roots)
    app_module.build_app()
    assert _tree_state(deep_roots + shallow_roots) == before


# --- Part O: application smoke test ---


def test_app_smoke_test(rgb_pil: Image.Image) -> None:
    app_module = _load_app()  # 1. module imports
    engine = get_engine()  # 2. model initializes (once, cached)
    assert engine.is_loaded
    demo = app_module.build_app()  # 3. Gradio UI constructs
    assert isinstance(demo, gr.Blocks)
    objects, confidence, class_id, error = app_module.predict_callback(
        rgb_pil
    )  # 4-5. callback + synthetic image prediction
    assert objects and CONFIDENCE_RE.match(confidence)
    assert 0 <= int(class_id) <= 49
    assert error == ""
    bad = app_module.predict_callback(None)  # 6. invalid input -> clear error
    assert bad[:3] == ("", "", "")
    assert bad[3].strip()
