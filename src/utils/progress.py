"""Fixed-header phase progress display for long-running phase scripts.

Renders an overall phase progress block at the top of the terminal and
redraws it in place while detailed operation output is printed below.
Falls back to periodic plain snapshots when stdout is not a TTY so logs
stay readable when redirected to a file.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

BAR_WIDTH = 20
REDRAW_INTERVAL = 0.25
PLAIN_INTERVAL = 5.0
FILL = "█"
EMPTY = "░"


def _enable_ansi() -> None:
    if os.name == "nt":
        os.system("")


def _ensure_utf8(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort; falls back to ASCII-safe output
        pass


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class PhaseProgress:
    """Overall phase progress block kept above detailed logging."""

    def __init__(
        self,
        title: str,
        step_labels: Sequence[str],
        *,
        stream=None,
        eta_own_line: bool = False,
        sample_wording: bool = False,
    ) -> None:
        if not step_labels:
            raise ValueError("step_labels must not be empty")
        self.title = title
        self.step_labels = list(step_labels)
        self.total_steps = len(step_labels)
        self.eta_own_line = eta_own_line
        self.sample_wording = sample_wording
        self.stream = stream if stream is not None else sys.stdout
        _ensure_utf8(self.stream)
        self.step_index = 1
        self.current = 0
        self.total = 0
        self.detail = ""
        self.started = time.monotonic()
        self._rendered_lines = 0
        self._last_render = 0.0
        self._last_plain = 0.0
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())
        _enable_ansi()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def overall_fraction(self) -> float:
        done_steps = self.step_index - 1
        sub = 0.0
        if self.total > 0:
            sub = min(max(self.current / self.total, 0.0), 1.0)
        return min((done_steps + sub) / self.total_steps, 1.0)

    def set_step(self, index: int, label: str | None = None) -> None:
        self.step_index = min(max(index, 1), self.total_steps)
        if label is not None:
            self.step_labels[self.step_index - 1] = label
        self.current = 0
        self.total = 0
        self.detail = ""
        self.render(force=True)

    def update(
        self,
        current: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        *,
        force: bool = False,
    ) -> None:
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        if detail is not None:
            self.detail = detail
        self.render(force=force)

    def build_lines(self) -> list[str]:
        fraction = self.overall_fraction
        filled = round(fraction * BAR_WIDTH)
        bar = FILL * filled + EMPTY * (BAR_WIDTH - filled)
        lines = [
            self.title,
            f"[{bar}] {fraction * 100:5.1f}%",
            f"Step {self.step_index}/{self.total_steps} — "
            f"{self.step_labels[self.step_index - 1]}",
        ]
        if self.total > 0:
            pct = 100.0 * min(self.current, self.total) / self.total
            if self.sample_wording:
                lines.append(f"Current sample: {self.current:,} / {self.total:,}")
                lines.append(f"Progress: {pct:.1f}%")
            else:
                lines.append(
                    f"Current: {self.current:,} / {self.total:,}  ({pct:.1f}%)"
                )
        elif self.detail:
            lines.append(self.detail)
        elapsed = self.elapsed
        if self.eta_own_line:
            lines.append(f"Elapsed: {format_duration(elapsed)}")
            lines.append(f"ETA: {self._eta_text(elapsed)}")
            return lines
        rate = self.current / elapsed if elapsed > 0 and self.total else 0.0
        eta = ""
        if self.total and rate > 0:
            eta = f"  ETA {format_duration((self.total - self.current) / rate)}"
        lines.append(f"Elapsed: {format_duration(elapsed)}{eta}")
        return lines

    def _eta_text(self, elapsed: float) -> str:
        if self.total > 0 and self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            return format_duration((self.total - self.current) / rate)
        fraction = self.overall_fraction
        if fraction > 0 and elapsed > 0:
            return format_duration(elapsed * (1.0 - fraction) / fraction)
        return "--:--:--"

    def render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not self._tty:
            self._render_plain(force=force)
            return
        if not force and now - self._last_render < REDRAW_INTERVAL:
            return
        self._last_render = now
        self._render_tty()

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _render_tty(self) -> None:
        lines = self.build_lines()
        if self._rendered_lines:
            self._write(f"\033[{self._rendered_lines}A")
        for index, line in enumerate(lines):
            self._write("\033[2K" + line)
            if index < len(lines) - 1:
                self._write("\n")
        self._write("\n")
        self._rendered_lines = len(lines)

    def _render_plain(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_plain < PLAIN_INTERVAL:
            return
        self._last_plain = now
        for line in self.build_lines():
            self._write(line + "\n")

    def log(self, message: str) -> None:
        """Print a message above the progress block."""
        if self._tty and self._rendered_lines:
            self._write(f"\033[{self._rendered_lines}A")
            for line in str(message).splitlines() or [""]:
                self._write("\033[2K" + line + "\n")
            lines = self.build_lines()
            for index, line in enumerate(lines):
                self._write("\033[2K" + line)
                if index < len(lines) - 1:
                    self._write("\n")
            self._write("\n")
            self._last_render = time.monotonic()
        else:
            self._write(str(message) + "\n")

    def finish(self, message: str | None = None) -> None:
        self.render(force=True)
        if message:
            self.log(message)
        self._rendered_lines = 0
