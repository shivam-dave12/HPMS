"""
risk_manager.py — Institutional Risk Management Layer
======================================================
Position sizing, circuit breakers, drawdown protection, daily limits.

Supports an optional notify_fn (e.g. Telegram send_message) so that every
circuit-breaker event fires an immediate push notification.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

# ── Indian Standard Time (UTC +5:30) ─────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# Persistent trade log — survives daily resets and process restarts.
# Each line is a JSON object (one trade per line / newline-delimited JSON).
_ALL_TRADES_FILE = "hpms_all_trades.json"

from logger_core import elog

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp:   float
    side:        str
    entry_price: float
    exit_price:  float = 0.0
    gross_pnl:   float = 0.0
    fees_usd:    float = 0.0
    net_pnl:     float = 0.0
    size:        int   = 0
    hold_bars:   int   = 0
    reason:      str   = ""
    closed:      bool  = False
    margin_used: float = 0.0  # for ROE% calculation


class RiskManager:
    """Real-time risk engine.  All state mutations are thread-safe (RLock)."""

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
        auto_resume_seconds:     float = 600.0,
        soft_loss_weight:        float = 0.5,
    ):
        self._max_pos_usd       = max_position_usd
        self._max_pos_contracts = max_position_contracts
        self._leverage          = leverage
        self._max_daily_loss    = max_daily_loss_usd
        self._max_daily_trades  = max_daily_trades
        self._max_consec_losses = max_consecutive_losses
        self._max_dd_pct        = max_drawdown_pct
        self._equity_pct        = equity_pct_per_trade

        self._lock               = threading.RLock()
        self._trades_today:      List[TradeRecord] = []
        self._daily_pnl:         float = 0.0
        self._total_fees:        float = 0.0
        self._session_high_pnl:  float = 0.0
        self._consecutive_losses: float = 0.0
        self._last_trade_time:   float = 0.0
        self._is_halted:         bool  = False
        self._halt_reason:       str   = ""
        self._halt_time:         float = 0.0   # unix ts when halt fired
        self._day_start:         str   = ""
        self._open_trade:        Optional[TradeRecord] = None

        # Auto-resume: CONSECUTIVE_LOSSES / MAX_DRAWDOWN halts auto-clear
        # after this many seconds.  DAILY_ halts persist until new day.
        self._auto_resume_seconds: float = auto_resume_seconds

        # Graduated cooldown: actual cooldown = base × (1 + consec_losses × 0.5)
        self._base_cooldown:     float = cooldown_seconds

        # Loss classification: forced exits (ENERGY_SPIKE, MAX_HOLD) count as
        # "soft" losses — only half-weight toward consecutive loss counter.
        self._soft_loss_weight:  float = soft_loss_weight

        # Optional push-notification callback (set by main after Telegram is ready)
        self._notify_fn: Optional[Callable[[str], None]] = None

        self._reset_if_new_day()
        logger.info("RiskManager initialised")

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
        if today == self._day_start:
            return

        prev_day = self._day_start
        self._day_start          = today
        self._trades_today       = []
        self._daily_pnl          = 0.0
        self._total_fees         = 0.0
        self._session_high_pnl   = 0.0
        self._consecutive_losses = 0

        # Clear ALL halts on new day — not just DAILY ones.
        # CONSECUTIVE_LOSSES halts were surviving midnight and blocking
        # the bot for 7+ hours even after the root cause was long gone.
        if self._is_halted:
            old_reason        = self._halt_reason
            self._is_halted   = False
            self._halt_reason = ""
            logger.info("Daily reset cleared halt: %s", old_reason)

        logger.info("RiskManager daily reset → %s (was %s)", today, prev_day or "init")
        elog.log("SYSTEM_DAILY_RESET", new_day=today, prev_day=prev_day or "init")

        if prev_day:  # don't notify on first init
            self._notify(
                f"🗓 *Daily Reset* — `{today}`\n"
                f"All halts cleared.  Counters zeroed.  Trading resumes."
            )

    # ─── PRE-TRADE CHECKS ────────────────────────────────────────────────────

    def can_trade(self) -> Tuple[bool, str]:
        with self._lock:
            self._reset_if_new_day()

            # ── Auto-resume from timed halts ──────────────────────────────────
            # CONSECUTIVE_LOSSES and MAX_DRAWDOWN auto-clear after a pause.
            # DAILY_LOSS_LIMIT and DAILY_TRADE_LIMIT remain until new day.
            # MANUAL / TELEGRAM halts remain until explicit /resume.
            if self._is_halted:
                auto_resumable = self._halt_reason in (
                    "CONSECUTIVE_LOSSES", "MAX_DRAWDOWN",
                )
                if auto_resumable and self._halt_time > 0:
                    elapsed_halt = time.time() - self._halt_time
                    if elapsed_halt >= self._auto_resume_seconds:
                        old_reason               = self._halt_reason
                        self._is_halted          = False
                        self._halt_reason        = ""
                        self._consecutive_losses = 0
                        elapsed_min              = elapsed_halt / 60
                        logger.info(
                            "⏰ Auto-resumed from %s after %.1f min pause",
                            old_reason, elapsed_min,
                        )
                        elog.log("RISK_RESUME", reason=old_reason,
                                 pause_min=round(elapsed_min, 1))
                        self._notify(
                            f"⏰ *Auto-resumed* from `{old_reason}`\n"
                            f"Pause: `{elapsed_min:.1f}` min  │  "
                            f"Consecutive losses reset to `0`\n"
                            f"_Trading resumes on next signal._"
                        )
                    else:
                        remaining = self._auto_resume_seconds - elapsed_halt
                        return False, (
                            f"HALTED: {self._halt_reason} "
                            f"(auto-resume in {remaining:.0f}s)"
                        )
                else:
                    return False, f"HALTED: {self._halt_reason}"

            if self._open_trade and not self._open_trade.closed:
                return False, "POSITION_OPEN"

            # ── Graduated cooldown ────────────────────────────────────────────
            effective_cooldown = self._base_cooldown * (
                1.0 + self._consecutive_losses * 0.5
            )
            elapsed = time.time() - self._last_trade_time
            if elapsed < effective_cooldown:
                return False, (
                    f"COOLDOWN: {effective_cooldown - elapsed:.1f}s "
                    f"(base {self._base_cooldown:.0f}s × "
                    f"{1.0 + self._consecutive_losses * 0.5:.1f})"
                )

            if len(self._trades_today) >= self._max_daily_trades:
                self._halt("DAILY_TRADE_LIMIT")
                return False, (
                    f"DAILY_TRADE_LIMIT: {len(self._trades_today)}/{self._max_daily_trades}"
                )

            if self._daily_pnl <= -self._max_daily_loss:
                self._halt("DAILY_LOSS_LIMIT")
                return False, f"DAILY_LOSS: ${self._daily_pnl:.2f}"

            if self._consecutive_losses >= self._max_consec_losses:
                self._halt("CONSECUTIVE_LOSSES")
                return False, f"CONSEC_LOSSES: {self._consecutive_losses}"

            # ── Drawdown check (equity-relative) ──────────────────────────────
            # Only trigger if session_high_pnl is meaningful (> $1).
            # A $0.50 session high followed by a $0.49 pullback would otherwise
            # read as ~98% "drawdown" — a nonsensical false trigger.
            if self._session_high_pnl > 1.0:
                dd = (
                    (self._session_high_pnl - self._daily_pnl)
                    / self._session_high_pnl * 100
                )
                if dd > self._max_dd_pct:
                    self._halt("MAX_DRAWDOWN")
                    return False, f"DRAWDOWN: {dd:.1f}%"

            return True, "OK"

    def _halt(self, reason: str):
        """Internal halt — updates state and fires push notification."""
        self._is_halted   = True
        self._halt_reason = reason
        self._halt_time   = time.time()

        logger.warning("⛔ RiskManager HALTED: %s", reason)
        elog.log("RISK_HALT", reason=reason,
                 daily_pnl=self._daily_pnl,
                 consecutive_losses=self._consecutive_losses,
                 trades_today=len(self._trades_today))

        auto_resumable = reason in ("CONSECUTIVE_LOSSES", "MAX_DRAWDOWN")

        # Build a precise breach line for each halt type
        if reason == "CONSECUTIVE_LOSSES":
            breach_line = (
                f"   Consecutive losses: `{self._consecutive_losses}` "
                f"/ limit `{self._max_consec_losses}`"
            )
        elif reason == "MAX_DRAWDOWN":
            dd = (
                (self._session_high_pnl - self._daily_pnl)
                / self._session_high_pnl * 100
                if self._session_high_pnl > 0 else 0.0
            )
            breach_line = (
                f"   Drawdown: `{dd:.1f}%` / limit `{self._max_dd_pct}%`\n"
                f"   Session high: `${self._session_high_pnl:+.4f}` → "
                f"now: `${self._daily_pnl:+.4f}`"
            )
        elif reason == "DAILY_LOSS_LIMIT":
            breach_line = (
                f"   Daily loss: `${self._daily_pnl:+.4f}` "
                f"/ limit `$-{self._max_daily_loss:.2f}`"
            )
        elif reason == "DAILY_TRADE_LIMIT":
            breach_line = (
                f"   Trades today: `{len(self._trades_today)}` "
                f"/ limit `{self._max_daily_trades}`"
            )
        else:
            breach_line = f"   Reason: `{reason}`"

        resume_line = (
            f"⏰ Auto-resume in `{self._auto_resume_seconds / 60:.0f}` min"
            if auto_resumable
            else "_Use /resume to clear  │  /halt to emergency flatten_"
        )

        self._notify(
            f"⛔ *RISK HALT: {reason}*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💥 *Breach:*\n{breach_line}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Session at halt:*\n"
            f"   Trades: `{len(self._trades_today)}`  │  "
            f"Net P&L: `${self._daily_pnl:+.4f}`\n"
            f"   Fees: `$-{self._total_fees:.4f}`  │  "
            f"High: `${self._session_high_pnl:+.4f}`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{resume_line}"
        )

    # ─── POSITION SIZING ─────────────────────────────────────────────────────

    def compute_size(
        self,
        price:          float,
        equity_usd:     float,
        contract_value: float = 0.001,
        sl_pct:         float = 0.0,
        confidence:     float = 1.0,
        norm_vol:       float = 0.0,
    ) -> int:
        """
        Compute position size — margin-bounded, confidence-weighted, vol-normalised.

        Sizing hierarchy (each step can only REDUCE the result, never increase it):

        1. MARGIN CAP  — hard physical ceiling:
                         max_by_margin = floor(equity × 0.95 × leverage / (price × cv))
                         This is computed first and is never violated.

        2. RISK BUDGET — Kelly-style: how many contracts keep the dollar loss on
                         a SL hit within equity_pct% of equity.

        3. CONFIDENCE  — scale within [50%, 100%] of the margin cap target.

        4. VOL SCALE   — inverse-vol targeting: reduces size in high-vol regimes.

        5. HARD CAPS   — max_pos_contracts, max_pos_usd (from config).
        """
        with self._lock:
            if price <= 0 or contract_value <= 0 or equity_usd <= 0:
                return 1

            # ── Step 1: Margin cap ────────────────────────────────────────────
            max_by_margin = max(
                1,
                int(equity_usd * 0.95 * self._leverage / (price * contract_value))
            )

            # ── Step 2: Confidence scaling within margin cap ──────────────────
            # conf_scale ∈ [0.50, 1.00]: at minimum confidence → half the margin cap;
            # at maximum confidence → full cap.
            conf_scale = max(0.50, min(1.00, confidence))

            # ── Step 3: Volatility normalisation ─────────────────────────────
            vol_scale = 1.0
            if norm_vol > 0:
                vol_target = 0.001   # baseline: 0.1% per bar
                vol_scale  = min(1.0, max(0.3, vol_target / norm_vol))

            # ── Step 4: Risk-budget sizing (Kelly-style) ──────────────────────
            risk_usd = equity_usd * (self._equity_pct / 100.0)

            if sl_pct > 0:
                per_contract_sl_loss  = price * sl_pct * contract_value
                risk_budget_contracts = (
                    int(risk_usd / per_contract_sl_loss)
                    if per_contract_sl_loss > 0 else max_by_margin
                )
            else:
                # Fallback: notional-based sizing
                notional              = min(risk_usd * self._leverage, self._max_pos_usd)
                per_contract_notional = contract_value * price
                risk_budget_contracts = (
                    int(notional / per_contract_notional)
                    if per_contract_notional > 0 else 0
                )

            # ── Step 5: Combine all constraints ──────────────────────────────
            target    = int(max_by_margin * conf_scale * vol_scale)
            contracts = min(target, risk_budget_contracts, self._max_pos_contracts)

            # Notional cap: never exceed max_pos_usd total exposure
            max_by_notional = (
                int(self._max_pos_usd / (price * contract_value))
                if price > 0 else contracts
            )
            contracts = min(contracts, max_by_notional)

            # Floor at 1 — always take at least one contract if cleared to trade
            contracts = max(contracts, 1)

            # Final sanity: re-enforce margin cap (guards any edge-case drift)
            contracts = min(contracts, max_by_margin)

            logger.debug(
                "Size: equity=$%.2f lev=%dx max_margin=%dc "
                "risk_budget=%dc conf=%.0f%% conf_scale=%.2f "
                "vol_scale=%.2f → %dc (margin_req=$%.2f)",
                equity_usd, self._leverage, max_by_margin,
                risk_budget_contracts, confidence * 100, conf_scale,
                vol_scale, contracts,
                price * contract_value * contracts / self._leverage,
            )
            elog.log("RISK_SIZE",
                     max_margin=max_by_margin, risk_budget=risk_budget_contracts,
                     confidence=confidence, vol_scale=round(vol_scale, 3),
                     result=contracts)
            return contracts

    # ─── TRADE LIFECYCLE ──────────────────────────────────────────────────────

    def on_trade_open(self, side: str, entry_price: float, size: int,
                      margin_used: float = 0.0):
        with self._lock:
            self._open_trade = TradeRecord(
                timestamp=time.time(), side=side,
                entry_price=entry_price, size=size,
                margin_used=margin_used,
            )
            self._last_trade_time = time.time()

    def on_trade_close(self, exit_price: float, gross_pnl: float,
                       fees: float, net_pnl: float,
                       hold_bars: int, reason: str):
        with self._lock:
            # Guard against double-call (race condition where exchange WebSocket
            # fill event and strategy close callback both fire).
            # Bug fix: previously PnL counters updated even when _open_trade was
            # None, causing inflated daily_pnl and incorrect halt triggers.
            if self._open_trade is None:
                logger.warning(
                    "on_trade_close called with no open trade record "
                    "(possible double-close) — reason=%s  net_pnl=%.4f  IGNORED",
                    reason, net_pnl,
                )
                return

            self._open_trade.exit_price = exit_price
            self._open_trade.gross_pnl  = gross_pnl
            self._open_trade.fees_usd   = fees
            self._open_trade.net_pnl    = net_pnl
            self._open_trade.hold_bars  = hold_bars
            self._open_trade.reason     = reason
            self._open_trade.closed     = True
            self._trades_today.append(self._open_trade)

            # ── Persist to disk so /overallpnl survives restarts/daily resets ─
            try:
                rec = self._open_trade
                entry = {
                    "timestamp":   rec.timestamp,
                    "ist_time":    datetime.fromtimestamp(rec.timestamp, tz=_IST)
                                   .strftime("%Y-%m-%d %H:%M:%S IST"),
                    "side":        rec.side,
                    "entry_price": rec.entry_price,
                    "exit_price":  rec.exit_price,
                    "size":        rec.size,
                    "gross_pnl":   round(rec.gross_pnl, 6),
                    "fees_usd":    round(rec.fees_usd, 6),
                    "net_pnl":     round(rec.net_pnl, 6),
                    "roe_pct":     round(
                        rec.net_pnl / rec.margin_used * 100, 3
                    ) if rec.margin_used > 0 else 0.0,
                    "hold_bars":   rec.hold_bars,
                    "reason":      rec.reason,
                    "margin_used": round(rec.margin_used, 4),
                }
                with open(_ALL_TRADES_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except Exception as _e:
                logger.warning("Could not persist trade to %s: %s", _ALL_TRADES_FILE, _e)

            self._open_trade = None

            # Track daily PnL using NET (after fees) — real money
            self._daily_pnl        += net_pnl
            self._total_fees       += fees
            self._session_high_pnl  = max(self._session_high_pnl, self._daily_pnl)

            if net_pnl < 0:
                is_forced = any(
                    tag in reason
                    for tag in ("ENERGY_SPIKE", "MAX_HOLD", "SHUTDOWN", "TELEGRAM")
                )
                if is_forced:
                    self._consecutive_losses += self._soft_loss_weight
                    self._consecutive_losses  = round(self._consecutive_losses, 1)
                    logger.info(
                        "Soft loss (%s): consec_losses=%.1f (+%.1f)",
                        reason, self._consecutive_losses, self._soft_loss_weight,
                    )
                else:
                    self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def get_open_trade(self) -> Optional[TradeRecord]:
        with self._lock:
            return self._open_trade

    # ─── EXIT CONDITIONS ──────────────────────────────────────────────────────

    def check_exit_conditions(
        self,
        dH_dt:       float,
        dH_dt_spike: float,
        bars_held:   int,
        max_hold:    int = 0,
    ) -> Optional[str]:
        """
        Check exit conditions.

        With the trailing stop system, the ONLY forced exit reason is:
          1. Energy spike (Hamiltonian instability — |dH/dt| > adaptive threshold)
          2. Absolute safety ceiling (120 bars) — handled in strategy, only if profitable.

        Time-based exits are removed.  The trailing stop protects profit by
        ratcheting the SL on the exchange.
        """
        if abs(dH_dt) > dH_dt_spike:
            return f"ENERGY_SPIKE: |dH/dt|={abs(dH_dt):.4f} > {dH_dt_spike:.4f}"
        return None

    # ─── CONTROLS ─────────────────────────────────────────────────────────────

    def force_halt(self, reason: str = "MANUAL"):
        with self._lock:
            self._halt(reason)

    def resume(self) -> str:
        with self._lock:
            was                      = self._halt_reason
            self._is_halted          = False
            self._halt_reason        = ""
            self._halt_time          = 0.0
            self._consecutive_losses = 0
            logger.info("RiskManager resumed (was halted: %s)", was)
            elog.log("RISK_RESUME", was=was, note="manual_resume")
            self._notify(
                f"✅ *Risk resumed*\n"
                f"Was halted: `{was or 'none'}`\n"
                f"Consecutive losses → `0`\n"
                f"_Use /start\\_trading to re-enable the strategy._"
            )
            return was

    def update_param(self, key: str, value) -> bool:
        _MAP = {
            "max_position_usd":       ("_max_pos_usd",         float),
            "max_position_contracts": ("_max_pos_contracts",   int),
            "leverage":               ("_leverage",            int),
            "max_daily_loss_usd":     ("_max_daily_loss",      float),
            "max_daily_trades":       ("_max_daily_trades",    int),
            "max_consecutive_losses": ("_max_consec_losses",   int),
            "cooldown_seconds":       ("_base_cooldown",       float),
            "max_drawdown_pct":       ("_max_dd_pct",          float),
            "equity_pct_per_trade":   ("_equity_pct",          float),
            "auto_resume_seconds":    ("_auto_resume_seconds", float),
            "soft_loss_weight":       ("_soft_loss_weight",    float),
        }
        if key not in _MAP:
            return False
        attr, typ = _MAP[key]
        try:
            old_val = getattr(self, attr)
            new_val = typ(value)
            setattr(self, attr, new_val)
            logger.info("Risk param updated: %s = %s (was %s)", key, new_val, old_val)
            elog.log("RISK_PARAM_UPDATE", key=key, old=str(old_val), new=str(new_val))
            return True
        except Exception as e:
            logger.warning("Risk param update failed: %s = %s — %s", key, value, e)
            return False

    # ─── STATUS ───────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        with self._lock:
            self._reset_if_new_day()

            effective_cooldown = self._base_cooldown * (
                1.0 + self._consecutive_losses * 0.5
            )

            auto_resume_remaining = 0.0
            if self._is_halted and self._halt_time > 0:
                auto_resumable = self._halt_reason in (
                    "CONSECUTIVE_LOSSES", "MAX_DRAWDOWN",
                )
                if auto_resumable:
                    elapsed_halt = time.time() - self._halt_time
                    auto_resume_remaining = max(
                        0.0, self._auto_resume_seconds - elapsed_halt
                    )

            daily_gross = sum(t.gross_pnl for t in self._trades_today)

            return {
                "is_halted":          self._is_halted,
                "halt_reason":        self._halt_reason,
                "daily_pnl":          round(self._daily_pnl, 4),
                "daily_gross_pnl":    round(daily_gross, 4),
                "daily_fees":         round(self._total_fees, 4),
                "session_high_pnl":   round(self._session_high_pnl, 4),
                "trades_today":       len(self._trades_today),
                "consecutive_losses": self._consecutive_losses,
                "last_trade_time":    self._last_trade_time,
                "open_trade":         bool(self._open_trade and not self._open_trade.closed),
                "cooldown_remaining": max(
                    0.0, effective_cooldown - (time.time() - self._last_trade_time)
                ),
                "auto_resume_remaining": round(auto_resume_remaining, 0),
                "params": {
                    "max_pos_usd":        self._max_pos_usd,
                    "max_pos_contracts":  self._max_pos_contracts,
                    "leverage":           self._leverage,
                    "max_daily_loss":     self._max_daily_loss,
                    "max_daily_trades":   self._max_daily_trades,
                    "max_consec_losses":  self._max_consec_losses,
                    "cooldown":           self._base_cooldown,
                    "effective_cooldown": round(effective_cooldown, 1),
                    "max_dd_pct":         self._max_dd_pct,
                    "equity_pct":         self._equity_pct,
                    "auto_resume_sec":    self._auto_resume_seconds,
                    "soft_loss_weight":   self._soft_loss_weight,
                },
            }

    def get_trade_log(self, last_n: int = 10) -> List[Dict]:
        with self._lock:
            return [
                {
                    "time":      datetime.fromtimestamp(
                        t.timestamp, tz=_IST
                    ).strftime("%H:%M:%S IST"),
                    "side":      t.side,
                    "entry":     t.entry_price,
                    "exit":      t.exit_price,
                    "gross_pnl": round(t.gross_pnl, 4),
                    "fees":      round(t.fees_usd, 4),
                    "net_pnl":   round(t.net_pnl, 4),
                    "roe_pct":   round(
                        t.net_pnl / t.margin_used * 100, 2
                    ) if t.margin_used > 0 else 0.0,
                    "bars":      t.hold_bars,
                    "size":      t.size,
                    "reason":    t.reason,
                }
                for t in self._trades_today[-last_n:]
            ]

    def get_all_trades_ever(self) -> List[Dict]:
        """
        Return every trade recorded since the bot first ran, loaded from the
        persistent newline-delimited JSON file on disk.  Trades recorded
        during the current session that have not yet been flushed are already
        in the file (written in on_trade_close), so this is always complete.

        Returns an empty list if the file does not exist or cannot be parsed.
        """
        trades: List[Dict] = []
        if not os.path.exists(_ALL_TRADES_FILE):
            return trades
        try:
            with open(_ALL_TRADES_FILE, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # skip corrupted lines
        except Exception as e:
            logger.warning("Could not read %s: %s", _ALL_TRADES_FILE, e)
        return trades
