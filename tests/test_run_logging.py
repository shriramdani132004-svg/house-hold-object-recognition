"""Tests for the structured training-run logger (``src.utils.run_logging``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils.run_logging import format_event, get_logger, log_event


@pytest.fixture
def logger_name(request: pytest.FixtureRequest):
    name = f"test_run_logging.{request.node.name}"
    yield name
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _console_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]


def _file_handlers(logger: logging.Logger) -> list[logging.FileHandler]:
    return [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]


def test_format_event_uses_stable_field_order() -> None:
    first = format_event(loss=0.5, event="epoch", seed=42, note="tail")
    second = format_event(loss=0.5, event="epoch", seed=42, note="tail")
    assert first == second
    assert first == "event=epoch seed=42 loss=0.5 note=tail"


def test_format_event_order_covers_full_example_field_set() -> None:
    line = format_event(
        eta="00:03:00",
        elapsed=12.5,
        replay_size=2000,
        checkpoint="models/x/checkpoint.pt",
        best_dev=0.8123456789,
        dev_accuracy=0.5,
        loss=1.25,
        epoch=2,
        experience=4,
        seed=42,
        run=0,
        scenario="NIC",
        candidate="c1",
        event="epoch",
    )
    assert line == (
        "event=epoch candidate=c1 scenario=NIC run=0 seed=42 experience=4 "
        "epoch=2 loss=1.25 dev_accuracy=0.5 best_dev=0.812346 "
        "checkpoint=models/x/checkpoint.pt replay_size=2000 elapsed=12.5 "
        "eta=00:03:00"
    )


def test_format_event_rounds_floats_and_renders_none() -> None:
    line = format_event(event="epoch", loss=0.123456789, dev_accuracy=None, elapsed=2.0)
    assert line == "event=epoch loss=0.123457 dev_accuracy=- elapsed=2.0"


def test_format_event_renders_int_bool_and_path() -> None:
    line = format_event(event="resume", experience=3, augment=False, checkpoint=Path("a/b.pt"))
    assert line == "event=resume experience=3 checkpoint=a/b.pt augment=False"


def test_log_event_writes_expected_line_to_log_file(
    tmp_path: Path, logger_name: str
) -> None:
    log_path = tmp_path / "nested" / "run.log"
    logger = get_logger(logger_name, log_path)
    log_event(logger, event="epoch", candidate="c1", epoch=3, loss=0.25, checkpoint=None)
    text = log_path.read_text(encoding="utf-8")
    assert "INFO" in text
    assert "event=epoch" in text
    assert "candidate=c1" in text
    assert "epoch=3" in text
    assert "loss=0.25" in text
    assert "checkpoint=-" in text


def test_get_logger_is_idempotent(logger_name: str) -> None:
    first = get_logger(logger_name)
    second = get_logger(logger_name)
    assert first is second
    assert len(_console_handlers(first)) == 1
    assert not _file_handlers(first)


def test_get_logger_keeps_single_console_handler_with_log_file(
    tmp_path: Path, logger_name: str
) -> None:
    log_path = tmp_path / "run.log"
    for _ in range(3):
        logger = get_logger(logger_name, log_path)
    assert len(_console_handlers(logger)) == 1
    assert len(_file_handlers(logger)) == 1
    assert len(logger.handlers) == 2


def test_get_logger_replaces_log_file_without_leaking_handlers(
    tmp_path: Path, logger_name: str
) -> None:
    first_file = tmp_path / "first.log"
    second_file = tmp_path / "second.log"
    get_logger(logger_name, first_file)
    logger = get_logger(logger_name, second_file)
    file_handlers = _file_handlers(logger)
    assert len(logger.handlers) == 2
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename).name == "second.log"
    log_event(logger, event="epoch", epoch=1)
    assert "event=epoch" in second_file.read_text(encoding="utf-8")
    assert first_file.read_text(encoding="utf-8") == ""


def test_relative_log_file_resolves_under_project_root(
    tmp_path: Path, logger_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.utils.run_logging as run_logging

    monkeypatch.setattr(run_logging, "PROJECT_ROOT", tmp_path)
    logger = get_logger(logger_name, "logs/relative_test.log")
    file_handlers = _file_handlers(logger)
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == tmp_path / "logs" / "relative_test.log"
    assert (tmp_path / "logs").is_dir()
