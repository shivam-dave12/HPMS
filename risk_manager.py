"""
risk_manager.py — Institutional Risk Management Layer
======================================================
Position sizing, circuit breakers, drawdown protection, daily limits.

Supports an optional notify_fn (e.g. Telegram send_message) so that every
circuit-breaker event fires an immediate push notification.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp:   float
    side:        str
    entry_price: float
    exit_price:  float = 0.0
    pnl_usd:     float = 0.0
    size:        int   = 0
    hold_bars:   int   = 0
    reason:      str   = ""
    closed:      bool  = False


class RiskManager:
    """Real-time risk engine. All checks are thread-safe."""

    def __init__(
        self,
        max_position_usd:        float = 500.0,
        max_position_contracts:  int   = 100,
        leverage:                int   = 10,
        max_daily_loss_usd:      float = 200.0,
        max_daily_trades:        int   = 50,
        max_consecutive_losses:  int   = 5,
        cooldown_seconds:        float = 30.0,
        max_drawdown_pct:        float = 5.0,
        equity_pct_per_trade:    float = 2.0,
    ):
        self._max_pos_usd       = max_position_usd
        self._max_pos_contracts = max_position_contracts
        self._leverage          = leverage
        self._max_daily_loss    = max_daily_loss_usd
        self._max_daily_trades  = max_daily_trades
        self._max_consec_losses = max_consecutive_losses
        self._cooldown          = cooldown_seconds
        self._max_dd_pct        = max_drawdown_pct
        self._equity_pct        = equity_pct_per_trade

        self._lock               = threading.RLock()
        self._trades_today:      List[TradeRecord] = []
        self._daily_pnl:         float = 0.0
        self._session_high_pnl:  float = 0.0
        self._consecutive_losses: int  = 0
        self._last_trade_time:   float = 0.0
        self._is_halted:         bool  = False
        self._halt_reason:       str   = ""
        self._day_start:         str   = ""
        self._open_trade:        Optional[TradeRecord] = None

        # Optional push-notification callback (set by main after Telegram is ready)
        self._notify_fn: Optional[Callable[[str], None]] = None

        self._reset_if_new_day()
        logger.info("RiskManager initialized")

    def set_notify_fn(self, fn: Callable[[str], None]):
        """Wire in Telegram send_message so halts push immediately."""
        self._notify_fn = fn

    def _notify(self, text: str):
        if self._notify_fn:
            try:
                self._notify_fn(text)
            except Exception:
                pass

    # ─── DAILY RESET ──────────────────────────────────────────────────────────

    def _reset_if_new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day_start:
            prev_day = self._day_start
            self._day_start          = today
            self._trades_today       = []
            self._daily_pnl          = 0.0
            self._session_high_pnl   = 0.0
            self._consecutive_losses = 0
            if self._is_halted and "DAILY" in self._halt_reason:
                self._is_halted   = False
                self._halt_reason = ""
            logger.info(f"RiskManager daily reset → {today} (was {prev_day})")
            if prev_day:  # don't notify on first init
                self._notify(f"🗓 *Daily Reset* — new trading day: `{today}`")

    # ─── PRE-TRADE CHECKS ────────────────────────────────────────────────────

    def can_trade(self) -> Tuple[bool, str]:
        with self._lock:
            self._reset_if_new_day()

            if self._is_halted:
                return False, f"HALTED: {self._halt_reason}"

            if self._open_trade and not self._open_trade.closed:
                return False, "POSITION_OPEN"

            elapsed = time.time() - self._last_trade_time
            if elapsed < self._cooldown:
                return False, f"COOLDOWN: {self._cooldown - elapsed:.1f}s"

            if len(self._trades_today) >= self._max_daily_trades:
                self._halt("DAILY_TRADE_LIMIT")
                return False, f"DAILY_TRADE_LIMIT: {len(self._trades_today)}/{self._max_daily_trades}"

            if self._daily_pnl <= -self._max_daily_loss:
                self._halt("DAILY_LOSS_LIMIT")
                return False, f"DAILY_LOSS: ${self._daily_pnl:.2f}"

            if self._consecutive_losses >= self._max_consec_losses:
                self._halt("CONSECUTIVE_LOSSES")
                return False, f"CONSEC_LOSSES: {self._consecutive_losses}"

            if self._session_high_pnl > 0:
                dd = (self._session_high_pnl - self._daily_pnl) / self._session_high_pnl * 100
                if dd > self._max_dd_pct:
                    self._halt("MAX_DRAWDOWN")
                    return False, f"DRAWDOWN: {dd:.1f}%"

            return True, "OK"

    def _halt(self, reason: str):
        self._is_halted   = True
        self._halt_reason = reason
        logger.warning(f"⛔ RiskManager HALTED: {reason}")
        self._notify(
            f"⛔ *RISK HALT: {reason}*\n"
            f"Daily PnL: `${self._daily_pnl:+.2f}`\n"
            f"Trades today: `{len(self._trades_today)}`\n"
            f"Consec losses: `{self._consecutive_losses}`\n"
            f"Use /resume to reset or /halt for emergency flatten"
        )

    # ─── POSITION SIZING ─────────────────────────────────────────────────────

    def compute_size(self, price: float, equity_usd: float) -> int:
        with self._lock:
            risk_usd  = equity_usd * (self._equity_pct / 100.0)
            notional  = min(risk_usd * self._leverage, self._max_pos_usd)
            contracts = int(notional / price) if price > 0 else 0
            contracts = min(contracts, self._max_pos_contracts)
            contracts = max(contracts, 1)
            return contracts

    # ─── TRADE LIFECYCLE ──────────────────────────────────────────────────────

    def on_trade_open(self, side: str, entry_price: float, size: int):
        with self._lock:
            self._open_trade = TradeRecord(
                timestamp=time.time(), side=side, entry_price=entry_price, size=size,
            )
            self._last_trade_time = time.time()

    def on_trade_close(self, exit_price: float, pnl_usd: float, hold_bars: int, reason: str):
        with self._lock:
            if self._open_trade:
                self._open_trade.exit_price = exit_price
                self._open_trade.pnl_usd    = pnl_usd
                self._open_trade.hold_bars   = hold_bars
                self._open_trade.reason      = reason
                self._open_trade.closed      = True
                self._trades_today.append(self._open_trade)
                self._open_trade = None

            self._daily_pnl        += pnl_usd
            self._session_high_pnl  = max(self._session_high_pnl, self._daily_pnl)

            if pnl_usd < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def get_open_trade(self) -> Optional[TradeRecord]:
        with self._lock:
            return self._open_trade

    # ─── EXIT CONDITIONS ──────────────────────────────────────────────────────

    def check_exit_conditions(
        self, dH_dt: float, dH_dt_spike: float, bars_held: int, max_hold: int
    ) -> Optional[str]:
        if bars_held >= max_hold:
            return f"MAX_HOLD: {bars_held} bars"
        if abs(dH_dt) > dH_dt_spike:
            return f"ENERGY_SPIKE: |dH/dt|={abs(dH_dt):.4f} > {dH_dt_spike}"
        return None

    # ─── CONTROLS ─────────────────────────────────────────────────────────────

    def force_halt(self, reason: str = "MANUAL"):
        with self._lock:
            self._halt(reason)

    def resume(self) -> str:
        with self._lock:
            was = self._halt_reason
            self._is_halted          = False
            self._halt_reason        = ""
            self._consecutive_losses = 0
            logger.info(f"RiskManager resumed (was halted: {was})")
            self._notify(f"✅ *Risk resumed* (was: `{was}`) — consecutive losses reset")
            return was

    def update_param(self, key: str, value) -> bool:
        _MAP = {
            "max_position_usd":       ("_max_pos_usd",       float),
            "max_position_contracts": ("_max_pos_contracts",  int),
            "leverage":               ("_leverage",           int),
            "max_daily_loss_usd":     ("_max_daily_loss",     float),
            "max_daily_trades":       ("_max_daily_trades",   int),
            "max_consecutive_losses": ("_max_consec_losses",  int),
            "cooldown_seconds":       ("_cooldown",           float),
            "max_drawdown_pct":       ("_max_dd_pct",         float),
            "equity_pct_per_trade":   ("_equity_pct",         float),
        }
        if key not in _MAP:
            return False
        attr, typ = _MAP[key]
        try:
            setattr(self, attr, typ(value))
            logger.info(f"Risk param updated: {key} = {value}")
            return True
        except Exception:
            return False

    # ─── STATUS ───────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        with self._lock:
            self._reset_if_new_day()
            return {
                "is_halted":          self._is_halted,
                "halt_reason":        self._halt_reason,
                "daily_pnl":          round(self._daily_pnl, 2),
                "session_high_pnl":   round(self._session_high_pnl, 2),
                "trades_today":       len(self._trades_today),
                "consecutive_losses": self._consecutive_losses,
                "last_trade_time":    self._last_trade_time,
                "open_trade":         bool(self._open_trade and not self._open_trade.closed),
                "cooldown_remaining": max(0.0, self._cooldown - (time.time() - self._last_trade_time)),
                "params": {
                    "max_pos_usd":       self._max_pos_usd,
                    "max_pos_contracts": self._max_pos_contracts,
                    "leverage":          self._leverage,
                    "max_daily_loss":    self._max_daily_loss,
                    "max_daily_trades":  self._max_daily_trades,
                    "max_consec_losses": self._max_consec_losses,
                    "cooldown":          self._cooldown,
                    "max_dd_pct":        self._max_dd_pct,
                    "equity_pct":        self._equity_pct,
                },
            }

    def get_trade_log(self, last_n: int = 10) -> List[Dict]:
        with self._lock:
            return [
                {
                    "time":   datetime.fromtimestamp(t.timestamp, tz=timezone.utc).strftime("%H:%M:%S"),
                    "side":   t.side,
                    "entry":  t.entry_price,
                    "exit":   t.exit_price,
                    "pnl":    round(t.pnl_usd, 2),
                    "bars":   t.hold_bars,
                    "reason": t.reason,
                }
                for t in self._trades_today[-last_n:]
            ]
