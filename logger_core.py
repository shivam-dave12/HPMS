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

    # Map event prefixes → log level so critical events surface higher.
    _LEVEL_MAP = {
        "SYSTEM_":   logging.INFO,
        "RISK_HALT": logging.WARNING,
    }

    def _level_for(self, event: str) -> int:
        for prefix, level in self._LEVEL_MAP.items():
            if event.startswith(prefix):
                return level
        return logging.DEBUG

    def log(self, event: str, **kwargs) -> None:
        """Emit a structured event at DEBUG (or INFO for SYSTEM_ events)."""
        if _elog_logger.isEnabledFor(logging.DEBUG):
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
