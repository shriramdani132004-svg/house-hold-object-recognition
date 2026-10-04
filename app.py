"""Root launcher for the frozen-model web app (Phase 45).

Run from the project root::

    .\\.venv\\Scripts\\python.exe app.py

Equivalent to ``python -m app``; both go through
:func:`app.main.main`, so the app has exactly one entry point and one
inference path (:func:`app.main.predict_callback` ->
:func:`src.inference.get_engine`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.main import main

if __name__ == "__main__":
    main()
