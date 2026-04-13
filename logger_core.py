"""
logger_core.py — Structured Event Logger (elog)
================================================
Provides the `elog` singleton used by hpms_engine.py and main.py for
structured decision-trace logging.

Every call emits a single human-readable line with an icon prefix and
key=value pairs.  The format is intentionally terse but self-describing:

  🎯 ENGINE_SIGNAL   │ signal=LONG  price=84,500.00  conf=85.0%  Δq=+0.00234
  🔄 ENGINE_KDE_REBUILD │ bars_since=50  history=200  bandwidth=0.3000
  🚨 RISK_HALT       │ reason=max_daily_loss  pnl=-200.00
  🏥 SYSTEM_HEALTH   │ ⚠  issue=price_stale_gt_120s

Usage:
    from logger_core import elog
    elog.log("ENGINE_SIGNAL", signal="LONG", price=84500.0)
    elog.warn("SYSTEM_HEALTH", issue="price_stale_gt_120s")
    elog.error("ENGINE_KDE_REBUILD", error="singular matrix", stage="build_failed")
"""

from __future__ import annotations

import logging
from typing import Any

_elog_logger = logging.getLogger("hpms.elog")


class _ELog:
    """Lightweight structured event logger with human-readable output."""

    # ── Per-event-name icons (exact match checked first) ─────────────────────
    _ICONS: dict[str, str] = {
        "ENGINE_SIGNAL":         "🎯",
        "ENGINE_SKIP":           "⏭ ",
        "ENGINE_KDE_REBUILD":    "🔄",
        "ENGINE_PHASE_STATE":    "🌀",
        "ENGINE_CRITERIA":       "🔬",
        "ENGINE_TRAJECTORY":     "📐",
        "ENGINE_INTEGRATOR":     "∫ ",
        "RISK_HALT":             "🚨",
        "RISK_PARAM_UPDATE":     "⚙️ ",
        "RISK_TRADE":            "💱",
        "RISK_POSITION":         "📍",
        "ORDER_ENTRY":           "📥",
        "ORDER_EXIT":            "📤",
        "ORDER_CANCEL":          "❌",
        "ORDER_FILL":            "✅",
        "SYSTEM_START":          "🚀",
        "SYSTEM_SHUTDOWN":       "🛑",
        "SYSTEM_HEALTH":         "🏥",
        "SYSTEM_MAIN_LOOP":      "🔁",
    }

    # ── Prefix-level fallback icons (checked when no exact match) ────────────
    _PREFIX_ICONS: dict[str, str] = {
        "ENGINE_":  "⚙️ ",
        "RISK_":    "🛡 ",
        "ORDER_":   "📋",
        "SYSTEM_":  "🔧",
    }

    # ── Map event prefixes → log level ────────────────────────────────────────
    # SYSTEM_ events are INFO  (startup/shutdown milestones — always visible).
    # RISK_HALT is WARNING     (circuit-breaker — always visible).
    # ENGINE_ / RISK_ / ORDER_ are DEBUG (per-bar diagnostics).
    _LEVEL_MAP = {
        "SYSTEM_":   logging.INFO,
        "RISK_HALT": logging.WARNING,
        "ENGINE_":   logging.DEBUG,
        "RISK_":     logging.DEBUG,
        "ORDER_":    logging.DEBUG,
    }

    # ── Friendly field-name substitutions for display ─────────────────────────
    _FIELD_ALIAS: dict[str, str] = {
        "predicted_delta_q": "Δq",
        "delta_q":           "Δq",
        "dH_dt":             "dH/dt",
        "confidence":        "conf",
        "signal_type":       "signal",
        "compute_time_us":   "μs",
        "bars_since_kde":    "kde_age",
    }

    # ─────────────────────────────────────────────────────────────────────────

    def _level_for(self, event: str) -> int:
        for prefix, level in self._LEVEL_MAP.items():
            if event.startswith(prefix):
                return level
        return logging.DEBUG

    def _icon_for(self, event: str) -> str:
        if event in self._ICONS:
            return self._ICONS[event]
        for prefix, icon in self._PREFIX_ICONS.items():
            if event.startswith(prefix):
                return icon
        return "·  "

    def _fmt_val(self, v: Any) -> str:
        """Format a single value for display."""
        if isinstance(v, float):
            av = abs(v)
            if av == 0.0:
                return "0"
            if av >= 10_000:
                return f"{v:,.2f}"
            if av >= 1_000:
                return f"{v:,.1f}"
            if av >= 1.0:
                return f"{v:.4f}"
            if av >= 0.001:
                return f"{v:+.5f}" if v < 0 else f"{v:.5f}"
            return f"{v:.6f}"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 10_000 else str(v)
        return str(v)

    def _build_line(self, event: str, kwargs: dict) -> str:
        """Assemble `icon EVENT │ k=v  k=v  k=v`."""
        icon = self._icon_for(event)
        parts = []
        for k, v in kwargs.items():
            display_key = self._FIELD_ALIAS.get(k, k)
            parts.append(f"{display_key}={self._fmt_val(v)}")
        body = "  ".join(parts)
        sep  = "│" if parts else ""
        return f"{icon} {event:<28} {sep} {body}".rstrip()

    # ─────────────────────────────────────────────────────────────────────────

    def log(self, event: str, **kwargs) -> None:
        """Emit a structured event at the appropriate level.

        The old isEnabledFor(DEBUG) guard was removed.  It was evaluated
        against the *root* effective level, which silently dropped all
        ENGINE_/RISK_/ORDER_ events when LOG_LEVEL=INFO even though
        hpms.elog was intended to be verbose.  The logger's own level
        (set explicitly to DEBUG in setup_logging) is the correct gate.
        """
        _elog_logger.log(
            self._level_for(event),
            self._build_line(event, kwargs),
        )

    def warn(self, event: str, **kwargs) -> None:
        """Emit a structured WARNING event (always visible)."""
        line = self._build_line(event, kwargs)
        _elog_logger.warning(f"⚠  {line}")

    def error(self, event: str, **kwargs) -> None:
        """Emit a structured ERROR event (always visible)."""
        line = self._build_line(event, kwargs)
        _elog_logger.warning(f"💥 {line}")


# Singleton — imported as `from logger_core import elog`
elog = _ELog()
