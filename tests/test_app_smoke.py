"""App smoke tests: import, shared callback output, and UI construction.

Synthetic images only; the frozen checkpoint is read through the normal
singleton path. No camera hardware, network, dataset, or launch is used.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator
from types import ModuleType

import gradio as gr
import pytest
from PIL import Image

from src.inference import IMAGE_SIZE, reset_engine

CONFIDENCE_RE = re.compile(r"^Model confidence: \d{1,3}\.\d{2}%$")


@pytest.fixture(scope="module")
def rgb_pil() -> Image.Image:
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (180, 40, 90))


@pytest.fixture(scope="module", autouse=True)
def clean_singleton() -> Iterator[None]:
    reset_engine()
    yield
    reset_engine()


@pytest.fixture(scope="module")
def app_main() -> ModuleType:
    return importlib.import_module("app.main")


def test_app_main_module_imports(app_main: ModuleType) -> None:
    assert callable(app_main.predict_callback)
    assert callable(app_main.build_app)
    assert callable(app_main.main)
    assert app_main.DEFAULT_MIN_CONFIDENCE is None


def test_predict_callback_returns_confidence_and_no_error(
    app_main: ModuleType, rgb_pil: Image.Image
) -> None:
    objects, confidence, class_id, error = app_main.predict_callback(rgb_pil)
    assert objects
    assert CONFIDENCE_RE.match(confidence)
    assert 0 <= int(class_id) <= 49
    assert error == ""


def test_predict_callback_handles_missing_image(app_main: ModuleType) -> None:
    objects, confidence, class_id, error = app_main.predict_callback(None)
    assert objects == ""
    assert confidence == ""
    assert class_id == ""
    assert error.strip()


def test_build_app_constructs_gradio_blocks(app_main: ModuleType) -> None:
    demo = app_main.build_app()
    assert isinstance(demo, gr.Blocks)
