"""
logger_core.py — Structured Event Logger (elog)
================================================
Provides the `elog` singleton used by hpms_engine.py and main.py for
structured decision-trace logging.

Every call emits a single JSON line so the trace is available when
LOG_LEVEL=DEBUG but invisible at INFO (the default).

Usage:
    from logger_core import elog
    elog.log("ENGINE_SIGNAL", signal="LONG", price=84500.0)
    elog.warn("SYSTEM_HEALTH", issue="price_stale_gt_120s")
    elog.error("ENGINE_KDE_REBUILD", error="singular matrix", stage="build_failed")
"""

from __future__ import annotations

import json
import logging

_elog_logger = logging.getLogger("hpms.elog")


class _ELog:
    """Lightweight structured JSON event logger."""

    # Map event prefixes → log level.
    # SYSTEM_ events are INFO (startup/shutdown milestones — always visible).
    # RISK_HALT is WARNING (circuit-breaker — always visible).
    # ENGINE_ / RISK_ / ORDER_ are DEBUG (verbose per-bar diagnostics shown
    # when LOG_LEVEL=DEBUG, which is now the default).
    _LEVEL_MAP = {
        "SYSTEM_":   logging.INFO,
        "RISK_HALT": logging.WARNING,
        "ENGINE_":   logging.DEBUG,
        "RISK_":     logging.DEBUG,
        "ORDER_":    logging.DEBUG,
    }

    def _level_for(self, event: str) -> int:
        for prefix, level in self._LEVEL_MAP.items():
            if event.startswith(prefix):
                return level
        return logging.DEBUG

    def log(self, event: str, **kwargs) -> None:
        """Emit a structured event.

        The old isEnabledFor(DEBUG) guard was removed.  It was evaluated
        against the *root* effective level, which silently dropped all
        ENGINE_/RISK_/ORDER_ events when LOG_LEVEL=INFO even though
        hpms.elog was intended to be verbose.  The logger's own level
        (set explicitly to DEBUG in setup_logging) is the correct gate.
        """
        payload = {"event": event, **kwargs}
        _elog_logger.log(
            self._level_for(event),
            json.dumps(payload, default=str),
        )

    def warn(self, event: str, **kwargs) -> None:
        """Emit a structured warning event at WARNING level."""
        payload = {"event": event, **kwargs}
        _elog_logger.warning(json.dumps(payload, default=str))

    def error(self, event: str, **kwargs) -> None:
        """Emit a structured error event at WARNING level."""
        payload = {"event": event, **kwargs}
        _elog_logger.warning(json.dumps(payload, default=str))


# Singleton — imported as `from logger_core import elog`
elog = _ELog()
