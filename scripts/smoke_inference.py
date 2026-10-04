"""Phase 44 — smoke inference for the frozen model (training-independent).

Runs the single inference path (``src.inference``) on official TRAINING
images with known official labels and asserts the output contract:
50 finite softmax scores summing to 1, a label in range, an object name
from the authoritative class mapping, and a well-formed formatted block.

Held-out sessions are not touched; nothing is written to the dataset.

Usage:
    python scripts/smoke_inference.py
    python scripts/smoke_inference.py --checkpoint models/continual/final_model.pt --limit 8
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapping import load_class_mapping  # noqa: E402
from src.data.continual import load_scenario_cached  # noqa: E402
from src.inference import InferenceEngine  # noqa: E402

NUM_CLASSES = 50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint to smoke (default: models/continual/final_model.pt)",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--size", type=int, default=64, help="expected model input resolution"
    )
    args = parser.parse_args()

    mapping = load_class_mapping()
    engine = InferenceEngine(checkpoint_path=args.checkpoint, image_size=args.size)

    scenario = load_scenario_cached("NIC", "inc", 0)
    records = []
    for exp in scenario.iter_experiences():
        for record in exp.train_samples:
            if record.split == "train":
                records.append(record)
        if len(records) >= args.limit:
            break
    if len(records) < args.limit:
        raise SystemExit(f"only {len(records)} training records found")

    print(
        f"smoke checkpoint: {args.checkpoint or 'models/continual/final_model.pt'}"
    )
    for record in records[: args.limit]:
        path = scenario.images_root.joinpath(*record.relative_path.split("/"))
        image = Image.open(path).convert("RGB")
        prediction = engine.predict(image)

        if prediction.class_index not in range(NUM_CLASSES):
            raise SystemExit(f"class index out of range: {prediction.class_index}")
        if prediction.object_name != mapping.names[prediction.class_index]:
            raise SystemExit(
                f"name mismatch: {prediction.object_name!r} != "
                f"{mapping.names[prediction.class_index]!r}"
            )
        if not 0.0 <= prediction.confidence <= 1.0:
            raise SystemExit(f"confidence out of range: {prediction.confidence}")
        if prediction.top3 and any(
            not math.isfinite(score) for _, score in prediction.top3
        ):
            raise SystemExit("non-finite score in top3")
        block = prediction.format_block()
        for line in ("OBJECT:", "CONFIDENCE:", "CLASS ID:"):
            if line not in block:
                raise SystemExit(f"formatted block missing {line!r}")

        gold = mapping.names[int(record.label)]
        print(
            f"{record.relative_path}: pred={prediction.object_name} "
            f"({prediction.confidence:.3f}) gold={gold}"
        )

    print(f"SMOKE OK: {args.limit} predictions, {NUM_CLASSES} classes, finite scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
