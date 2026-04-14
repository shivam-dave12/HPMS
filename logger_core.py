"""
logger_core.py — Structured Event Logger (elog)
================================================
Provides the `elog` singleton used by hpms_engine.py, strategy.py, and
main.py for structured decision-trace logging.

Every call emits a single human-readable line with an icon prefix and
key=value pairs.  The format is intentionally terse but self-describing:

  🎯 ENGINE_SIGNAL             │ signal=LONG  price=84,500.00  conf=87.3%  Δq=+0.00234
  🔄 ENGINE_KDE_REBUILD        │ bars_since=50  history=200  bandwidth=0.2500
  🚨 RISK_HALT                 │ reason=CONSECUTIVE_LOSSES  consec=5.0  daily_pnl=-45.20
  🚀 TRADE_ENTRY               │ side=LONG  size=50  price=84,500.00  tp=84,912.00  sl=84,253.00
  💵 TRADE_EXIT                │ reason=TP_HIT  exit=84,912.00  net=+0.4075  roe=+0.48%
  🏥 SYSTEM_HEALTH             │ ⚠  issue=price_stale_gt_120s

Usage:
    from logger_core import elog
    elog.log("ENGINE_SIGNAL", signal="LONG", price=84500.0, confidence=0.873)
    elog.log("TRADE_ENTRY",   side="LONG", size=50, price=84500.0)
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
        # Engine
        "ENGINE_SIGNAL":         "🎯",
        "ENGINE_SKIP":           "⏭ ",
        "ENGINE_KDE_REBUILD":    "🔄",
        "ENGINE_PHASE_STATE":    "🌀",
        "ENGINE_CRITERIA":       "🔬",
        "ENGINE_TRAJECTORY":     "📐",
        "ENGINE_INTEGRATOR":     "∫ ",
        # Trade lifecycle
        "TRADE_ENTRY":           "🚀",
        "TRADE_EXIT":            "💵",
        "TRADE_TRAIL":           "📏",
        "TRADE_BLOCKED":         "🚧",
        # Risk
        "RISK_HALT":             "🚨",
        "RISK_RESUME":           "✅",
        "RISK_PARAM_UPDATE":     "⚙️ ",
        "RISK_TRADE":            "💱",
        "RISK_POSITION":         "📍",
        "RISK_SIZE":             "📐",
        # Orders
        "ORDER_ENTRY":           "📥",
        "ORDER_EXIT":            "📤",
        "ORDER_CANCEL":          "❌",
        "ORDER_FILL":            "✅",
        # System
        "SYSTEM_START":          "🚀",
        "SYSTEM_SHUTDOWN":       "🛑",
        "SYSTEM_HEALTH":         "🏥",
        "SYSTEM_MAIN_LOOP":      "🔁",
        "SYSTEM_DAILY_RESET":    "🗓 ",
    }

    # ── Prefix-level fallback icons ───────────────────────────────────────────
    _PREFIX_ICONS: dict[str, str] = {
        "ENGINE_":  "⚙️ ",
        "TRADE_":   "💹",
        "RISK_":    "🛡 ",
        "ORDER_":   "📋",
        "SYSTEM_":  "🔧",
    }

    # ── Map event prefixes → log level ────────────────────────────────────────
    # SYSTEM_  events → INFO   (startup/shutdown milestones — always visible)
    # TRADE_   events → INFO   (entries/exits — always visible in live operation)
    # RISK_HALT → WARNING      (circuit-breaker — always visible)
    # ENGINE_ / RISK_ / ORDER_ → DEBUG (per-bar diagnostics, verbose)
    _LEVEL_MAP = {
        "SYSTEM_":   logging.INFO,
        "TRADE_":    logging.INFO,
        "RISK_HALT": logging.WARNING,
        "RISK_RESUME": logging.INFO,
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
        "entry_price":       "entry",
        "exit_price":        "exit",
        "tp_price":          "tp",
        "sl_price":          "sl",
        "net_pnl":           "net",
        "gross_pnl":         "gross",
        "roe_pct":           "roe",
        "bars_held":         "bars",
        "position_size":     "size",
    }

    # ── Fields that represent proportions (0–1) → display as percentage ──────
    # These are raw float values in [0, 1] that should render as "87.3%".
    _PCT_PROPORTION_FIELDS: frozenset[str] = frozenset({
        "confidence", "conf",           # signal confidence
        "tp_progress",                  # how far from entry to TP
        "vol_scale", "conf_scale",      # sizing scale factors
    })

    # ── Fields that are already in percent (0–100) → display with % suffix ───
    _PCT_FIELDS: frozenset[str] = frozenset({
        "roe", "roe_pct",
        "spread_pct", "atr_pct",
        "dd_pct", "equity_pct",
    })

    # ─────────────────────────────────────────────────────────────────────────

    def _level_for(self, event: str) -> int:
        # Exact-key matches first (e.g. RISK_HALT before RISK_)
        for prefix, level in self._LEVEL_MAP.items():
            if event == prefix or event.startswith(prefix):
                return level
        return logging.DEBUG

    def _icon_for(self, event: str) -> str:
        if event in self._ICONS:
            return self._ICONS[event]
        for prefix, icon in self._PREFIX_ICONS.items():
            if event.startswith(prefix):
                return icon
        return "·  "

    def _fmt_val(self, display_key: str, v: Any) -> str:
        """Format a single value for display, with context-aware rendering."""
        if isinstance(v, float):
            # Proportion → percentage (e.g. confidence=0.873 → "87.3%")
            if display_key in self._PCT_PROPORTION_FIELDS:
                return f"{v:.1%}"
            # Already-percent field
            if display_key in self._PCT_FIELDS:
                return f"{v:+.2f}%" if v != 0 else "0.00%"

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
                # Sign prefix for signed values (momentum, Δq, PnL deltas)
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
            parts.append(f"{display_key}={self._fmt_val(display_key, v)}")
        body = "  ".join(parts)
        sep  = "│" if parts else ""
        return f"{icon} {event:<28} {sep} {body}".rstrip()

    # ─────────────────────────────────────────────────────────────────────────

    def log(self, event: str, **kwargs) -> None:
        """Emit a structured event at the appropriate level.

        The hpms.elog logger is explicitly set to DEBUG in setup_logging(),
        so all ENGINE_/RISK_/ORDER_ events are always written regardless of
        the root logger's effective level.
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
        """Emit a structured ERROR event (always visible at WARNING level)."""
        line = self._build_line(event, kwargs)
        _elog_logger.error(f"💥 {line}")


# Singleton — imported as `from logger_core import elog`
elog = _ELog()
