"""Candidate training driver for the model-improvement work package (Steps 3-11).

Trains Experience Replay candidates on the official NIC ordering using a
train-only development validation split for early stopping and model
selection. The official held-out sessions (s3/s7/s10) are never read.

Usage:
    python scripts/train_candidates.py --list
    python scripts/train_candidates.py --candidate c1_corrected_baseline
    python scripts/train_candidates.py --all
    python scripts/train_candidates.py --smoke c1_corrected_baseline --max-experiences 2
    python scripts/train_candidates.py --final --from c2_replay_strong --epochs 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.continual import check_development_split, load_scenario_cached  # noqa: E402
from src.evaluation.continual import (  # noqa: E402
    atomic_write_json,
    build_metric_record,
    compute_forgetting,
    evaluate_dataset,
    read_json,
)
from src.training.config import ContinualTrainConfig  # noqa: E402
from src.training.improved import AugmentedCachedDataset, ImprovedReplayTrainer  # noqa: E402
from src.training.tensor_cache import TensorCache  # noqa: E402
from src.utils.run_logging import get_logger, log_event  # noqa: E402

RECIPE_PATH = PROJECT_ROOT / "configs/model_improvement_candidates.yaml"
NUM_CLASSES = 50
LOG_PATH = Path("logs") / "train_candidates.log"

_META_KEYS = {"id", "description"}


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def load_recipes() -> dict:
    payload = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8")) or {}
    recipes = {entry["id"]: entry for entry in payload["candidates"]}
    return {"global": payload, "recipes": recipes}


def build_config(recipe: dict, *, method: str, final: bool, epochs_override: int | None):
    flat = {k: v for k, v in recipe.items() if k not in _META_KEYS and k != "method"}
    if final:
        flat["early_stop_patience"] = None
        if epochs_override is not None:
            flat["epochs"] = int(epochs_override)
    if epochs_override is not None and not final:
        flat["epochs"] = int(epochs_override)
    flat["method"] = recipe.get("method", "replay")
    return ContinualTrainConfig.from_dict(flat)


def dev_manifest_paths(manifest_path: Path) -> tuple[list[set[str]], str]:
    if not manifest_path.is_file():
        raise SystemExit(
            f"development manifest missing: "
            f"{manifest_path.relative_to(PROJECT_ROOT)}\n"
            "create it first: python scripts/make_dev_split.py"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_exp = [
        set(entry["dev_paths"]) for entry in sorted(payload["experiences"], key=lambda e: e["experience_id"])
    ]
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return per_exp, digest


class CandidateRunner:
    def __init__(
        self,
        recipe: dict,
        *,
        recipe_index: int,
        recipe_total: int,
        final: bool,
        epochs_override: int | None,
        max_experiences: int | None,
        smoke: bool,
    ) -> None:
        self.recipe = recipe
        self.candidate_id = recipe["id"] + ("_final" if final else "")
        self.final = final
        self.max_experiences = max_experiences
        self.started = time.monotonic()
        self.last_validation: float | None = None
        self.recipe_index = recipe_index
        self.recipe_total = recipe_total

        global_cfg = load_recipes()["global"]
        self.manifest_path = PROJECT_ROOT / global_cfg["manifest"]
        self.cache_dir = PROJECT_ROOT / global_cfg["cache_dir"]

        suffix = "_smoke" if smoke else ""
        self.ckpt_dir = PROJECT_ROOT / "models/continual/candidates" / f"{self.candidate_id}{suffix}"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.ckpt_dir / "dev_metrics.json"
        self.summary_path = self.ckpt_dir / "candidate.json"

        self.scenario = load_scenario_cached("NIC", "inc", 0)
        self.dev_sets, self.manifest_sha = dev_manifest_paths(self.manifest_path)
        leak = check_development_split(
            set().union(*self.dev_sets) if self.dev_sets else set(),
            self.scenario,
        )
        if not leak.passed:
            raise SystemExit(
                f"development split failed leakage validation: {leak.detail}\n"
                "regenerate it: python scripts/make_dev_split.py"
            )
        self.images_root = self.scenario.images_root
        self.logger = get_logger("train_candidates", log_file=LOG_PATH)
        self.config = build_config(
            recipe,
            method=recipe.get("method", "replay"),
            final=final,
            epochs_override=epochs_override,
        )

    # ------------------------------------------------------------------
    def _banner(self, event: str, current: int, total: int) -> None:
        """One structured line per lifecycle/epoch event (never per image)."""
        if event == "batch":
            return
        log_event(
            self.logger,
            event=event,
            candidate=self.candidate_id,
            seed=self.config.seed,
            progress=f"{current}/{total}",
        )

    def _print_block(self, experience_id: int, epoch: int | None, loss: float | None) -> None:
        """Print one progress block; ``experience_id`` is 0-based."""
        elapsed = time.monotonic() - self.started
        done = experience_id
        remaining = max(0, 79 - done)
        eta = (elapsed / max(1, done + 1)) * remaining
        progress = 100.0 * done / 79
        header = (
            f"[CANDIDATE {self.recipe_index}/{self.recipe_total}] {self.candidate_id}"
        )
        validation = (
            f"{self.last_validation:.4f}" if self.last_validation is not None else "-"
        )
        loss_text = f"{loss:.4f}" if loss is not None else "-"
        print(header, flush=True)
        print(f"experience {experience_id + 1}/79", flush=True)
        if epoch is not None:
            print(f"epoch {epoch}/{self.config.epochs}", flush=True)
        print(f"progress {progress:.0f}%", flush=True)
        print(f"elapsed {fmt_hms(elapsed)}", flush=True)
        print(f"ETA {fmt_hms(eta)}", flush=True)
        print(f"train loss {loss_text}", flush=True)
        print(f"validation accuracy {validation}", flush=True)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.config
        print(
            f"[CANDIDATE {self.recipe_index}/{self.recipe_total}] {self.candidate_id}: "
            f"{self.recipe.get('description', '').strip()}",
            flush=True,
        )
        print(
            f"config: arch={cfg.model_arch} w={cfg.model_width} size={cfg.image_size} "
            f"opt={cfg.optimizer} lr={cfg.learning_rate} wd={cfg.weight_decay} "
            f"epochs={cfg.epochs} patience={cfg.early_stop_patience} "
            f"replay={cfg.replay_policy}/{cfg.replay_capacity}/"
            f"batch{cfg.replay_batch_size} augment={cfg.augment}",
            flush=True,
        )

        all_train_records = [
            record
            for exp in self.scenario.iter_experiences()
            for record in exp.train_samples
        ]
        cache = TensorCache.load_or_build(
            all_train_records,
            self.images_root,
            cfg.image_size,
            self.cache_dir,
            on_progress=lambda cur, tot: print(
                f"tensor cache: {cur}/{tot} decoded", flush=True
            )
            if cur == tot or cur % 30000 == 0
            else None,
        )
        print(f"tensor cache ready: {len(cache)} images @ {cfg.image_size}px", flush=True)

        dev_records = tuple(
            record
            for exp in self.scenario.iter_experiences()
            for record in exp.train_samples
            if record.relative_path in self.dev_sets[exp.experience_id]
        )

        # Early-stopping signal: a fixed stratified dev subset (~25 refs per
        # class, development data only), restricted at evaluation time to
        # classes already seen — the development analogue of the official
        # cumulative-classes accuracy definition.
        by_class: dict[int, list[str]] = {}
        for record in dev_records:
            by_class.setdefault(int(record.label), []).append(record.relative_path)
        subset_paths = {
            path
            for paths in by_class.values()
            for path in sorted(paths)[:25]
        }
        self._subset_records = tuple(
            record for record in dev_records if record.relative_path in subset_paths
        )
        print(
            f"early-stop dev subset: {len(self._subset_records)} refs "
            f"({len(subset_paths)} paths)",
            flush=True,
        )
        dev_dataset = None
        if not self.final:
            dev_dataset = AugmentedCachedDataset(
                dev_records, self.images_root, cfg.image_size,
                cache=cache, augment=False, require_split="train",
            )
            print(
                f"train refs {len(all_train_records) - len(dev_records)} | "
                f"dev refs {len(dev_records)}",
                flush=True,
            )
        else:
            print(f"training on ALL {len(all_train_records)} references (final run)", flush=True)

        def dev_provider(experience_id: int):
            seen = set(self.scenario.get_experience(experience_id).classes_seen)
            return tuple(
                record for record in self._subset_records if int(record.label) in seen
            )

        trainer = ImprovedReplayTrainer(
            cfg,
            on_progress=self._banner,
            cache=cache,
            dev_provider=None if self.final else dev_provider,
            best_checkpoint_path=self.ckpt_dir / "best_model.pt",
        )
        log_event(
            self.logger,
            event="run_start",
            candidate=self.candidate_id,
            scenario="NIC-inc-0",
            seed=cfg.seed,
            manifest=str(self.manifest_path.relative_to(PROJECT_ROOT)),
            checkpoint=str(self.ckpt_dir.relative_to(PROJECT_ROOT)),
        )

        epoch_state = {"losses": [], "accs": []}

        def on_metrics(event: dict) -> None:
            if event.get("event") == "epoch":
                epoch_state["losses"].append(event.get("loss"))
                epoch_state["accs"].append(event.get("accuracy"))
                self._print_block(
                    event["experience"], event["epoch"], event.get("loss")
                )

        def on_dev(event: dict) -> None:
            acc = event["dev_accuracy"]
            print(
                f"validation accuracy {acc:.4f} "
                f"(experience {event['experience']}, epoch {event['epoch']})",
                flush=True,
            )
            self.last_validation = acc
            log_event(
                self.logger,
                event="dev_epoch",
                candidate=self.candidate_id,
                experience=event["experience"],
                epoch=event["epoch"],
                dev_accuracy=acc,
                best_dev=event["best_dev_accuracy"],
                replay_size=event["non_improving_epochs"],
                checkpoint="best" if event["is_best"] else None,
            )

        trainer.on_train_metrics = on_metrics
        trainer.on_dev_metrics = on_dev

        state_file = self.ckpt_dir / "state.json"
        if state_file.is_file():
            state = trainer.resume(self.scenario, checkpoint_dir=self.ckpt_dir)
            print(f"resumed at experience {state.current_experience + 1}", flush=True)
        else:
            state = trainer.initialize(self.scenario, checkpoint_dir=self.ckpt_dir)

        records: list[dict] = []
        if self.metrics_path.is_file() and state.current_experience >= 0:
            records = read_json(self.metrics_path).get("records", [])
            records = [
                r for r in records if r["experience_id"] <= state.current_experience
            ]

        start_id = state.current_experience + 1
        stop_id = 79 if self.max_experiences is None else self.max_experiences
        previous_seen: set[int] = set()

        for exp in self.scenario.iter_experiences():
            if exp.experience_id < start_id:
                previous_seen = set(exp.classes_seen)
                continue
            if exp.experience_id >= stop_id:
                break

            if self.final:
                filtered = exp
            else:
                dev_set = self.dev_sets[exp.experience_id]
                train_samples = tuple(
                    r for r in exp.train_samples if r.relative_path not in dev_set
                )
                filtered = replace(exp, train_samples=train_samples)

            self._print_block(exp.experience_id, None, None)
            epoch_state["losses"].clear()
            epoch_state["accs"].clear()
            t0 = time.monotonic()
            prev_epochs = state.epochs_completed
            prev_steps = state.steps_completed
            state = trainer.train_experience(state, filtered)
            train_seconds = time.monotonic() - t0
            epochs_run = state.epochs_completed - prev_epochs
            steps_run = state.steps_completed - prev_steps

            if self.final:
                continue

            eval_t0 = time.monotonic()
            eval_result = evaluate_dataset(
                trainer.model,
                dev_dataset,
                num_classes=NUM_CLASSES,
                device=trainer.device,
                batch_size=256,
            )
            eval_seconds = time.monotonic() - eval_t0
            record = build_metric_record(
                experience_id=exp.experience_id,
                train_samples=len(filtered.train_samples),
                classes_introduced=exp.classes_introduced,
                classes_seen=exp.classes_seen,
                previous_seen=sorted(previous_seen),
                eval_result=eval_result,
                train_stats={
                    "loss_mean": (
                        sum(x for x in epoch_state["losses"] if x is not None)
                        / len(epoch_state["losses"])
                        if epoch_state["losses"] else None
                    ),
                    "accuracy_mean": (
                        sum(x for x in epoch_state["accs"] if x is not None)
                        / len(epoch_state["accs"])
                        if epoch_state["accs"] else None
                    ),
                    "epochs": epochs_run,
                    "steps": steps_run,
                    "seconds": round(train_seconds, 3),
                },
                eval_seconds=eval_seconds,
                replay_stats=state.replay,
            )
            records.append(record)
            record["forgetting"] = compute_forgetting(records)[-1]
            previous_seen = set(exp.classes_seen)
            atomic_write_json(
                self.metrics_path, {"candidate": self.candidate_id, "records": records}
            )
            overall = record["accuracy"]["overall"]
            self.last_validation = overall
            forgetting = compute_forgetting(records)
            print(
                f"validation accuracy {overall:.4f} | dev forgetting "
                f"{forgetting[-1]} | experience seconds {train_seconds:.1f}",
                flush=True,
            )
            log_event(
                self.logger,
                event="experience_eval",
                candidate=self.candidate_id,
                experience=exp.experience_id,
                dev_accuracy=overall,
                best_dev=trainer.best_development_value,
                replay_size=state.replay.get("size") if state.replay else None,
                elapsed=round(train_seconds, 1),
            )

        summary = self._summarize(state, records)
        atomic_write_json(self.summary_path, summary)
        print(f"candidate complete: {self.summary_path.relative_to(PROJECT_ROOT)}", flush=True)
        return summary

    def _summarize(self, state, records: list[dict]) -> dict:
        overalls = [r["accuracy"]["overall"] for r in records if r["accuracy"]["overall"] is not None]
        forgetting = compute_forgetting(records) if records else []
        forg_values = [f for f in forgetting if f is not None]
        avg_incremental = sum(overalls) / len(overalls) if overalls else None
        best_dev = max(overalls) if overalls else None
        cfg = self.config
        return {
            "candidate": self.candidate_id,
            "recipe": {k: v for k, v in self.recipe.items()},
            "final_run": self.final,
            "manifest": str(self.manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": self.manifest_sha,
            "training_config": cfg.to_dict(),
            "checkpoint_dir": str(self.ckpt_dir.relative_to(PROJECT_ROOT)),
            "experiences_trained": len(state.experiences_trained),
            "epochs_completed": state.epochs_completed,
            "steps_completed": state.steps_completed,
            "dev_final_accuracy": overalls[-1] if overalls else None,
            "dev_best_accuracy": best_dev,
            "dev_average_incremental_accuracy": avg_incremental,
            "dev_final_forgetting": forg_values[-1] if forg_values else None,
            "wall_clock_seconds": round(time.monotonic() - self.started, 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", help="run a single candidate id")
    parser.add_argument("--all", action="store_true", help="run every candidate")
    parser.add_argument("--list", action="store_true", help="list candidate ids")
    parser.add_argument("--smoke", help="short smoke run of one candidate")
    parser.add_argument("--max-experiences", type=int, default=None)
    parser.add_argument("--final", action="store_true", help="final full training run")
    parser.add_argument("--from", dest="from_recipe", help="recipe id for --final")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch budget")
    args = parser.parse_args()

    recipes = load_recipes()["recipes"]
    if args.list:
        for rid in recipes:
            print(rid)
        return 0

    if args.final:
        if not args.from_recipe or args.from_recipe not in recipes:
            print("--final requires --from <recipe-id>", file=sys.stderr)
            return 2
        selected = [args.from_recipe]
        final = True
    elif args.smoke:
        if args.smoke not in recipes:
            print(f"unknown candidate {args.smoke!r}", file=sys.stderr)
            return 2
        selected = [args.smoke]
        final = False
    elif args.candidate:
        if args.candidate not in recipes:
            print(f"unknown candidate {args.candidate!r}", file=sys.stderr)
            return 2
        selected = [args.candidate]
        final = False
    elif args.all:
        selected = list(recipes)
        final = False
    else:
        parser.print_help()
        return 2

    summaries = []
    for index, candidate_id in enumerate(selected, start=1):
        runner = CandidateRunner(
            recipes[candidate_id],
            recipe_index=index,
            recipe_total=len(selected),
            final=final,
            epochs_override=args.epochs,
            max_experiences=(
                args.max_experiences
                if args.max_experiences is not None
                else (2 if args.smoke else None)
            ),
            smoke=bool(args.smoke),
        )
        summaries.append(runner.run())

    print("\n=== selection table (development validation only) ===")
    for summary in summaries:
        print(
            f"{summary['candidate']}: dev_final={summary['dev_final_accuracy']} "
            f"dev_best={summary['dev_best_accuracy']} "
            f"dev_forgetting={summary['dev_final_forgetting']} "
            f"avg_incr={summary['dev_average_incremental_accuracy']} "
            f"wall={summary['wall_clock_seconds']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
