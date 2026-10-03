"""Phone-first Gradio web application for household-object recognition.

Import surface used by tests and tooling::

    import app
    demo = app.build_app()
    app.predict_callback(image)

Launch from the project root with ``python -m app``.
"""

from __future__ import annotations

from app.main import PREDICTION_OUTPUTS, build_app, main, predict_callback

__all__ = ["PREDICTION_OUTPUTS", "build_app", "main", "predict_callback"]
