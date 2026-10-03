r"""Hugging Face Space entrypoint (Phase 8).

Wires the frozen model and the official CORe50 class mapping into the
shared Phase-7 inference singleton, then serves the same phone-first
Gradio application used locally (``python -m app`` in the project).

Run from this directory::

    python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SPACE_ROOT = Path(__file__).resolve().parent
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from src.inference import InferenceEngine, set_engine

set_engine(
    InferenceEngine(
        checkpoint_path=SPACE_ROOT / "models" / "continual" / "final_model.pt",
        class_mapping_path=SPACE_ROOT
        / "models"
        / "continual"
        / "object_mapping.json",
    ).load()
)

from app.main import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
