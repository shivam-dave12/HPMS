"""
utils/logger.py — Structured logger for HL SDK modules.

stdlib Logger._log() only accepts (exc_info, stack_info, stacklevel, extra)
as keyword arguments. Every call site in api_client.py and ws_client.py
passes structured key=value kwargs instead, which raises TypeError at runtime.

StructuredLogger intercepts those kwargs, formats them as "key=value" pairs
appended to the message string, then delegates to the underlying stdlib logger
with zero extra kwargs. Drop-in replacement — no call sites need changes.
"""
from __future__ import annotations

import logging


class StructuredLogger:
    """Wraps a stdlib Logger; formats key=value kwargs into the message."""

    __slots__ = ("_log",)

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    # ── Private helper ────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(msg: str, kwargs: dict) -> str:
        if not kwargs:
            return msg
        pairs = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{msg} {pairs}"

    # ── Log-level methods ─────────────────────────────────────────────────────

    def debug(self, msg: str, **kwargs) -> None:
        if self._log.isEnabledFor(logging.DEBUG):
            self._log.debug(self._fmt(msg, kwargs))

    def info(self, msg: str, **kwargs) -> None:
        if self._log.isEnabledFor(logging.INFO):
            self._log.info(self._fmt(msg, kwargs))

    def warning(self, msg: str, **kwargs) -> None:
        self._log.warning(self._fmt(msg, kwargs))

    def error(self, msg: str, **kwargs) -> None:
        self._log.error(self._fmt(msg, kwargs))

    def exception(self, msg: str, **kwargs) -> None:
        self._log.exception(self._fmt(msg, kwargs))

    # ── Passthrough for anything that uses the raw logger directly ────────────

    @property
    def name(self) -> str:
        return self._log.name

    def isEnabledFor(self, level: int) -> bool:
        return self._log.isEnabledFor(level)

    def setLevel(self, level) -> None:
        self._log.setLevel(level)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
