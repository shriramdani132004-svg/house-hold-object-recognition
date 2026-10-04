"""Phase 45 — application verification against the frozen final model.

Checks, in one process:
  1. the root launcher (``app.py``) and ``python -m app`` import paths;
  2. the shared prediction callback on a real official training image;
  3. engine-vs-metadata consistency (arch / image size / class count);
  4. a real HTTP launch of the Gradio app on a free localhost port
     (no public share, no camera hardware), then a clean shutdown.

Usage:
    python scripts/verify_app.py
"""
from __future__ import annotations

import importlib
import json
import re
import socket
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.data.continual import load_scenario_cached  # noqa: E402
from src.inference import reset_engine  # noqa: E402

METADATA_PATH = PROJECT_ROOT / "models/continual/final_model.json"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    if not METADATA_PATH.is_file():
        raise SystemExit(f"final model metadata missing: {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    # 1. entry points
    root_module = importlib.import_module("app")
    app_main = importlib.import_module("app.main")
    if not callable(getattr(root_module, "main", None)) and not callable(
        getattr(app_main, "main", None)
    ):
        raise SystemExit("root app launcher does not expose main()")
    for name in ("build_app", "predict_callback", "main"):
        if not callable(getattr(app_main, name, None)):
            raise SystemExit(f"app.main.{name} is missing or not callable")

    # 2. engine consistency with frozen metadata
    reset_engine()
    engine = app_main.get_engine()
    if int(engine.image_size) != int(metadata["image_size"]):
        raise SystemExit(
            f"engine image_size {engine.image_size} != metadata "
            f"{metadata['image_size']}"
        )
    if int(engine.model.num_classes) != int(metadata["num_classes"]):
        raise SystemExit("engine class count does not match final metadata")
    schema_arch = metadata["arch"]
    engine_arch = type(engine.model).__name__
    expected_arch = (
        "CompactResNet" if schema_arch == "compact_resnet" else "SmallConvNet"
    )
    if engine_arch != expected_arch:
        raise SystemExit(
            f"engine architecture {engine_arch} != metadata arch {schema_arch}"
        )

    # 3. shared callback on a real official image
    scenario = load_scenario_cached("NIC", "inc", 0)
    record = next(
        r
        for exp in scenario.iter_experiences()
        for r in exp.train_samples
        if r.split == "train"
    )
    image_path = scenario.images_root.joinpath(*record.relative_path.split("/"))
    image = Image.open(image_path).convert("RGB")
    objects, confidence, class_id, error = app_main.predict_callback(image)
    if error:
        raise SystemExit(f"predict_callback reported an error: {error}")
    if not isinstance(objects, str) or not objects.strip():
        raise SystemExit("predict_callback returned an empty object name")
    match = re.fullmatch(r"Model confidence: (\d{1,3})\.(\d{2})%", confidence)
    if match is None:
        raise SystemExit(f"unexpected confidence format: {confidence!r}")
    confidence_value = int(match.group(1)) / 100.0 + int(match.group(2)) / 10000.0
    if not 0.0 <= confidence_value <= 1.0:
        raise SystemExit(f"confidence out of range: {confidence!r}")
    if not str(class_id).isdigit() or not 0 <= int(class_id) < int(
        metadata["num_classes"]
    ):
        raise SystemExit(f"predict_callback class id invalid: {class_id!r}")

    # 4. real local launch on a free port
    port = _free_port()
    demo = app_main.build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        share=False,
        quiet=True,
    )
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/", timeout=30
        ) as response:
            status = int(response.status)
            body = response.read(4096)
        if status != 200:
            raise SystemExit(f"app returned HTTP {status}")
        if not body:
            raise SystemExit("app returned an empty page")
    finally:
        demo.close()
        reset_engine()

    print(
        f"APP OK: callback pred={objects} conf={confidence} | "
        f"arch={schema_arch} @ {metadata['image_size']}px | "
        f"HTTP 200 on 127.0.0.1:{port}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
