"""Fixed experiment progress block for the Phase-5 continual runs.

Shows method/scenario, current experience and epoch, an experience-level
bar, per-experience and total elapsed time, ETA, live training loss and
accuracy, an overall progress bar, and evaluation progress while the fixed
official test set is being scored.

Renders in place on a TTY and falls back to throttled plain snapshots when
stdout is redirected, so log files stay readable (no thousands of lines).
"""

from __future__ import annotations

import sys
import time
from typing import Any

from src.utils.progress import EMPTY, FILL, _enable_ansi, _ensure_utf8, format_duration

BAR_WIDTH = 20
REDRAW_INTERVAL = 0.25
SNAPSHOT_INTERVAL = 10.0


def _bar(fraction: float) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = round(fraction * BAR_WIDTH)
    return FILL * filled + EMPTY * (BAR_WIDTH - filled)


class ExperimentDisplay:
    """Fixed experiment block for one method's run (naive or replay)."""

    def __init__(
        self,
        *,
        method: str,
        scenario_name: str,
        run_id: int,
        total_experiences: int,
        epochs: int,
        stream=None,
        snapshot_interval: float = SNAPSHOT_INTERVAL,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        _ensure_utf8(self.stream)
        _enable_ansi()
        self.method = method
        self.scenario_name = scenario_name
        self.run_id = run_id
        self.total_experiences = int(total_experiences)
        self.epochs = int(epochs)
        self.snapshot_interval = float(snapshot_interval)
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())

        self._run_started = time.monotonic()
        self._exp_started = 0.0
        self._exp_index: int | None = None
        self._steps_per_epoch = 0
        self._exp_steps_total = 0
        self._exp_steps_done = 0
        self._epoch = 0
        self._loss: float | None = None
        self._accuracy: float | None = None
        self._eval_detail = ""
        self._completed_before = 0
        self._rendered_lines = 0
        self._last_render = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start_run(self) -> None:
        self._run_started = time.monotonic()
        self._completed_before = 0
        self._exp_index = None
        self._eval_detail = ""
        self._render(force=True)

    def start_experience(
        self,
        experience_id: int,
        *,
        steps_per_epoch: int,
        epochs: int,
        completed_before: int,
    ) -> None:
        self._exp_index = int(experience_id)
        self._steps_per_epoch = int(steps_per_epoch)
        self._exp_steps_total = int(steps_per_epoch) * int(epochs)
        self._exp_steps_done = 0
        self._completed_before = int(completed_before)
        self._epoch = 0
        self._loss = None
        self._accuracy = None
        self._eval_detail = ""
        self._exp_started = time.monotonic()
        self._render(force=True)

    def on_metrics(self, info: dict[str, Any]) -> None:
        """Trainer ``on_train_metrics`` callback (batch and epoch events)."""
        if info.get("event") == "batch":
            epoch = int(info.get("epoch", 1))
            step = int(info.get("step", 0))
            self._epoch = epoch
            self._exp_steps_done = (epoch - 1) * self._steps_per_epoch + step
            if info.get("loss") is not None:
                self._loss = float(info["loss"])
            if info.get("batch_accuracy") is not None:
                self._accuracy = float(info["batch_accuracy"])
        elif info.get("event") == "epoch":
            self._epoch = int(info.get("epoch", 0))
            self._exp_steps_done = self._epoch * self._steps_per_epoch
            if info.get("loss") is not None:
                self._loss = float(info["loss"])
            if info.get("accuracy") is not None:
                self._accuracy = float(info["accuracy"])
        self._render()

    def on_eval_progress(self, current: int, total: int) -> None:
        self._eval_detail = f"Evaluating official test set: {current:,} / {total:,}"
        self._render()

    def finish_experience(self, summary: str) -> None:
        self._completed_before += 1
        self._eval_detail = ""
        self._exp_index = None
        self._epoch = 0
        self._exp_steps_total = 0
        self._exp_steps_done = 0
        self._loss = None
        self._accuracy = None
        self.log(summary)

    def end_block(self) -> None:
        """Erase the fixed block (TTY) before the next phase-step header."""
        if not (self._tty and self._rendered_lines):
            self._rendered_lines = 0
            return
        self._move_to_block_top()
        for index in range(self._rendered_lines):
            self._write("\033[2K")
            if index < self._rendered_lines - 1:
                self._write("\n")
        self._move_to_block_top()
        self._rendered_lines = 0

    def log(self, message: str) -> None:
        """Print a plain line above the fixed block (block is redrawn after)."""
        if self._tty and self._rendered_lines:
            rendered = self._rendered_lines
            self._move_to_block_top()
            for index in range(rendered):
                self._write("\033[2K")
                if index < rendered - 1:
                    self._write("\n")
            self._move_to_block_top()
            self._rendered_lines = 0
        for line in str(message).splitlines() or [""]:
            self._write(line + "\n")
        self._render(force=True)

    # ------------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------------
    @property
    def elapsed_total(self) -> float:
        return time.monotonic() - self._run_started

    @property
    def elapsed_experience(self) -> float:
        if self._exp_started == 0.0:
            return 0.0
        return time.monotonic() - self._exp_started

    def _fractions(self) -> tuple[float, float]:
        exp_fraction = 0.0
        if self._exp_index is not None and self._exp_steps_total > 0:
            exp_fraction = min(self._exp_steps_done / self._exp_steps_total, 1.0)
        overall = (self._completed_before + exp_fraction) / max(
            self.total_experiences, 1
        )
        return exp_fraction, min(overall, 1.0)

    def build_lines(self) -> list[str]:
        exp_fraction, overall = self._fractions()
        experience_label = (
            f"{self._exp_index + 1}/{self.total_experiences}"
            if self._exp_index is not None
            else f"-/{self.total_experiences}"
        )
        eta = "--:--:--"
        if overall > 0 and self.elapsed_total > 0:
            eta = format_duration((1.0 - overall) * self.elapsed_total / overall)
        loss = f"{self._loss:.4f}" if self._loss is not None else "-"
        accuracy = (
            f"{100.0 * self._accuracy:.1f}%"
            if self._accuracy is not None
            else "-"
        )
        lines = [
            "PHASE 5 — CONTINUAL EXPERIMENT",
            f"Method: {self.method}   Scenario: {self.scenario_name}   "
            f"Run: {self.run_id}",
            f"Experience: {experience_label}   Epoch: {self._epoch or '-'}/{self.epochs}",
            f"[{_bar(exp_fraction)}] {100.0 * exp_fraction:5.1f}%  "
            f"steps {self._exp_steps_done}/{self._exp_steps_total}",
            f"Experience elapsed: {format_duration(self.elapsed_experience)}   "
            f"Total elapsed: {format_duration(self.elapsed_total)}   ETA: {eta}",
            f"Loss: {loss}   Accuracy: {accuracy}",
        ]
        if self._eval_detail:
            lines.append(self._eval_detail)
        lines.append(
            f"Overall [{_bar(overall)}] {100.0 * overall:5.1f}%  "
            f"experiences {self._completed_before}/{self.total_experiences}"
        )
        return lines

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _move_to_block_top(self) -> None:
        if self._rendered_lines:
            self._write(f"\033[{self._rendered_lines}A")

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self._tty:
            if not force and now - self._last_render < REDRAW_INTERVAL:
                return
            self._last_render = now
            self._render_tty()
            return
        if not force and now - self._last_render < self.snapshot_interval:
            return
        self._last_render = now
        for line in self.build_lines():
            self._write(line + "\n")

    def _render_tty(self) -> None:
        lines = self.build_lines()
        self._move_to_block_top()
        for index, line in enumerate(lines):
            self._write("\033[2K" + line)
            if index < len(lines) - 1:
                self._write("\n")
        self._write("\n")
        self._rendered_lines = len(lines)
