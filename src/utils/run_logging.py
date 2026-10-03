"""Structured one-line logging for training runs.

Emit exactly one line per epoch or per experience — never per image — and
never log secrets (tokens, API keys, passwords, credentials or ``.env``
values).  Log files are append-only and live under the project tree unless
an absolute path is requested explicitly.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SINGLE_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

EVENT_FIELD_ORDER: tuple[str, ...] = (
    "event",
    "candidate",
    "scenario",
    "run",
    "seed",
    "experience",
    "epoch",
    "loss",
    "dev_accuracy",
    "best_dev",
    "checkpoint",
    "replay_size",
    "elapsed",
    "eta",
)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return str(round(value, 6))
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def format_event(**fields: Any) -> str:
    """Render fields as one deterministic ``key=value`` line.

    Fields declared in ``EVENT_FIELD_ORDER`` come first in that order,
    any remaining field follows in the order it was given, so the same
    call always produces the same line.
    """
    ordered = [name for name in EVENT_FIELD_ORDER if name in fields]
    ordered.extend(name for name in fields if name not in EVENT_FIELD_ORDER)
    return " ".join(f"{name}={_format_value(fields[name])}" for name in ordered)


def _resolve_log_file(log_file: str | Path) -> Path:
    target = Path(log_file).expanduser()
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve()


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first)) == os.path.normcase(str(second))


def _is_console_handler(handler: logging.Handler) -> bool:
    return isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, logging.FileHandler
    )


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return the logger called ``name``, ready to use.

    The logger always owns exactly one console handler on stderr and, when
    ``log_file`` is given, exactly one append-mode file handler (parent
    directories are created).  Repeated calls are idempotent: existing
    handlers are reused instead of duplicated, and a file handler for a
    previous file is closed and replaced rather than leaked.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(_is_console_handler(handler) for handler in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(SINGLE_LINE_FORMAT, DATE_FORMAT))
        logger.addHandler(console)

    if log_file is not None:
        target = _resolve_log_file(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handlers = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        reused = any(
            _same_path(Path(handler.baseFilename).resolve(), target)
            for handler in file_handlers
        )
        if not reused:
            for handler in file_handlers:
                logger.removeHandler(handler)
                handler.close()
            file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(SINGLE_LINE_FORMAT, DATE_FORMAT))
            logger.addHandler(file_handler)

    return logger


def log_event(
    logger: logging.Logger, level: int = logging.INFO, **fields: Any
) -> None:
    """Log one structured ``key=value`` line at ``level``."""
    logger.log(level, format_event(**fields))
