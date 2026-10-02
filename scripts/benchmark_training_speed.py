"""Benchmark CPU training speed under different Ultralytics tuning knobs.

Runs one epoch on a small fraction of the train split (train only: no
validation, no plots, no weights) and prints BENCH_RESULT {json} with the
wall time and average seconds/iteration. Used to fix the Phase 6 epoch
schedule before launch; re-runnable at any time to re-measure a machine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description="CPU training speed benchmark")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (Ultralytics forces 0 on CPU)")
    parser.add_argument("--threads", type=int, default=16, help="torch CPU threads")
    parser.add_argument("--batch", type=int, default=16, help="Train batch size")
    parser.add_argument("--fraction", type=float, default=0.05, help="Train split fraction")
    parser.add_argument("--tag", default="bench", help="Run tag for the output directory")
    parser.add_argument("--channels-last", action="store_true", help="Enable channels_last memory format")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--data", default="configs/train_data.yaml", help="Dataset YAML")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "models" / "training" / "bench"), help="Output root")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to run (benchmark: keep at 1)")
    return parser.parse_args(argv)


def resolve_model_weights(base: str, root: Path) -> str:
    """Find the base weights locally before letting Ultralytics download."""
    candidates = [root / base, root / "models" / "baseline" / Path(base).name, Path(base)]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return base


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a machine-readable result."""
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)

    import torch
    import ultralytics.utils.torch_utils as tu

    tu.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)

    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.models.yolo.detect import DetectionTrainer

    class BenchTrainer(DetectionTrainer):
        """Trainer that honors the requested worker/thread settings on CPU."""

        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            super().__init__(cfg, overrides, _callbacks)
            self.args.workers = args.workers
            torch.set_num_threads(args.threads)

    model_path = resolve_model_weights("yolo26n.pt", PROJECT_ROOT)
    model = YOLO(model_path)
    started = time.time()
    model.train(
        trainer=BenchTrainer,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        fraction=args.fraction,
        workers=args.workers,
        device="cpu",
        val=False,
        plots=False,
        save=False,
        deterministic=True,
        seed=42,
        channels_last=args.channels_last,
        compile=args.compile,
        project=args.out,
        name=f"bench_{args.tag}",
        exist_ok=True,
    )
    wall = time.time() - started

    trainer = model.trainer
    iterations = len(getattr(trainer, "train_loader", []) or [])
    epoch_time = getattr(trainer, "epoch_time", None) or wall
    result = {
        "tag": args.tag,
        "workers": args.workers,
        "threads": args.threads,
        "batch": args.batch,
        "fraction": args.fraction,
        "channels_last": args.channels_last,
        "compile": args.compile,
        "torch_threads": torch.get_num_threads(),
        "loader_workers": getattr(getattr(trainer, "train_loader", None), "num_workers", None),
        "amp": getattr(trainer, "amp", None),
        "iterations": iterations,
        "wall_s": round(wall, 1),
        "epoch_time": round(float(epoch_time), 1),
        "s_per_iteration": round(float(epoch_time) / iterations, 2) if iterations else None,
        "projected_full_epoch_h": round(float(epoch_time) / max(args.fraction, 1e-9) / 3600.0, 2),
    }
    print("BENCH_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
