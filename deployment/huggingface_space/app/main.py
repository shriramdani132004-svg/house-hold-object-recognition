r"""Phone-first Gradio web app for the frozen continual model (Phase 7).

Local launch, from the project root::

    .\.venv\Scripts\python.exe -m app

Works from any working directory as well::

    .\.venv\Scripts\python.exe <project>\app\main.py

Optional public sharing (public-link verification belongs to Phase 8)::

    $env:GRADIO_SHARE = "true"
    .\.venv\Scripts\python.exe -m app

Camera frames and uploaded images both flow through the single
:func:`predict_callback` -> :func:`src.inference.get_engine` backend, so
there is exactly one inference path and one checkpoint load per process.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from src.inference import InferenceError, get_engine

LOGGER = logging.getLogger("household_object.app")

TITLE = "Continual Household Object Recognition"
DESCRIPTION = (
    "Recognizes supported CORe50 household objects using the frozen "
    "continual-learning model (Experience Replay, 50 classes, 64x64 input)."
)
BOUNDING_BOX_NOTE = (
    "**Bounding boxes: not supported.** The selected model is a 50-class "
    "CORe50 *classifier* — it predicts the object name and confidence only."
)
PREDICTION_OUTPUTS = ("OBJECT", "CONFIDENCE", "CLASS ID", "ERROR")


def predict_callback(image: object) -> tuple[str, str, str, str]:
    """Shared prediction backend for camera frames and uploaded images."""
    try:
        prediction = get_engine().predict(image)
    except InferenceError as exc:
        return "", "", "", str(exc)
    except Exception:  # noqa: BLE001 - keep tracebacks out of the UI
        LOGGER.exception("unexpected prediction failure")
        return (
            "",
            "",
            "",
            "Prediction failed unexpectedly. Check the server log for details.",
        )
    return (
        prediction.object_name,
        prediction.formatted_confidence,
        str(prediction.class_index),
        "",
    )


def build_app() -> gr.Blocks:
    """Construct the single-page phone-first interface (no server start)."""
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESCRIPTION}")
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    sources=["webcam", "upload"],
                    type="pil",
                    label="Live camera / image upload",
                )
                recognize = gr.Button("Recognize", variant="primary")
            with gr.Column():
                object_out = gr.Textbox(label="OBJECT", interactive=False)
                confidence_out = gr.Textbox(label="CONFIDENCE", interactive=False)
                class_out = gr.Textbox(label="CLASS ID", interactive=False)
                error_out = gr.Textbox(label="ERROR", interactive=False)
        gr.Markdown(BOUNDING_BOX_NOTE)
        inputs = [image]
        outputs = [object_out, confidence_out, class_out, error_out]
        image.change(predict_callback, inputs=inputs, outputs=outputs)
        recognize.click(predict_callback, inputs=inputs, outputs=outputs)
    return demo


def main() -> None:
    share = os.environ.get("GRADIO_SHARE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    get_engine()  # fail fast on startup; the checkpoint is loaded once
    demo = build_app()
    demo.launch(share=share)


if __name__ == "__main__":
    main()
