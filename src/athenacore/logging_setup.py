"""Logging setup.

Two formats: a compact human line for terminals, and single-line JSON for
anything that ships logs. Configuration is idempotent so the CLI, the API server
and the Streamlit UI can each call :func:`configure_logging` at startup without
stacking handlers.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_CONFIGURED = False

_RESERVED = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "stacklevel",
        "thread",
        "threadName",
        "taskName",
    ]
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    _COLOURS = {
        "DEBUG": "\033[38;5;245m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[48;5;203m\033[38;5;231m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname[:4]
        if self.colour:
            tint = self._COLOURS.get(record.levelname, "")
            level = f"{tint}{level}{self._RESET}"
        name = record.name.removeprefix("athenacore.")
        extras = " ".join(
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        )
        line = f"{ts} {level} {name}: {record.getMessage()}"
        if extras:
            line = f"{line}  {extras}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(
    level: str = "INFO", *, json_output: bool = False, force: bool = False
) -> None:
    """Install a single stderr handler on the ``athenacore`` logger tree."""
    global _CONFIGURED
    root = logging.getLogger("athenacore")
    if _CONFIGURED and not force:
        root.setLevel(level.upper())
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanFormatter(colour=sys.stderr.isatty()))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``get_logger(__name__)`` is the idiom)."""
    if not name.startswith("athenacore"):
        name = f"athenacore.{name}"
    return logging.getLogger(name)
