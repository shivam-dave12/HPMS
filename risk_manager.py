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
    gross_pnl:   float = 0.0
    fees_usd:    float = 0.0
    net_pnl:     float = 0.0
    size:        float = 0.0
    hold_bars:   int   = 0
    reason:      str   = ""
    closed:      bool  = False
    margin_used: float = 0.0  # for ROE% calculation


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
        auto_resume_seconds:     float = 300.0,
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
        self._halt_time:         float = 0.0   # when halt was triggered
        self._day_start:         str   = ""
        self._open_trade:        Optional[TradeRecord] = None

        # Auto-resume: CONSECUTIVE_LOSSES halts auto-clear after this many seconds.
        # Prevents the bot from staying dead for hours over tiny losses.
        self._auto_resume_seconds: float = auto_resume_seconds

        # Graduated cooldown: cooldown increases after consecutive losses.
        # Base cooldown is self._base_cooldown; actual = base * (1 + consec_losses * 0.5)
        # So after 3 consecutive losses: cooldown = 10 * (1 + 1.5) = 25s
        self._base_cooldown:     float = cooldown_seconds

        # Loss classification: forced exits (ENERGY_SPIKE, MAX_HOLD) count as
        # "soft" losses — only half-weight toward consecutive loss counter.
        self._soft_loss_weight:  float = soft_loss_weight

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
            self._total_fees         = 0.0
            self._session_high_pnl   = 0.0
            self._consecutive_losses = 0
            # Clear ALL halts on new day — not just DAILY ones.
            # CONSECUTIVE_LOSSES halts were surviving midnight and blocking
            # the bot for 7+ hours even after the conditions that caused
            # the losses (e.g. ENERGY_SPIKE bug) were long gone.
            if self._is_halted:
                old_reason = self._halt_reason
                self._is_halted   = False
                self._halt_reason = ""
                logger.info(f"Daily reset cleared halt: {old_reason}")
            logger.info(f"RiskManager daily reset → {today} (was {prev_day})")
            if prev_day:  # don't notify on first init
                self._notify(f"🗓 *Daily Reset* — new trading day: `{today}` (all halts cleared)")

    # ─── PRE-TRADE CHECKS ────────────────────────────────────────────────────

    def can_trade(self) -> Tuple[bool, str]:
        with self._lock:
            self._reset_if_new_day()

            # ── Auto-resume from timed halts ──────────────────────────────────
            # CONSECUTIVE_LOSSES and MAX_DRAWDOWN halts auto-clear after a pause.
            # DAILY_LOSS_LIMIT and DAILY_TRADE_LIMIT remain until new day.
            # MANUAL/TELEGRAM halts remain until explicit /resume.
            if self._is_halted:
                auto_resumable = self._halt_reason in (
                    "CONSECUTIVE_LOSSES", "MAX_DRAWDOWN",
                )
                if auto_resumable and self._halt_time > 0:
                    elapsed_halt = time.time() - self._halt_time
                    if elapsed_halt >= self._auto_resume_seconds:
                        old_reason = self._halt_reason
                        self._is_halted          = False
                        self._halt_reason        = ""
                        self._consecutive_losses = 0
                        logger.info(
                            f"⏰ Auto-resumed from {old_reason} after "
                            f"{elapsed_halt:.0f}s pause"
                        )
                        self._notify(
                            f"⏰ *Auto-resumed* from `{old_reason}` "
                            f"after {elapsed_halt/60:.1f} min pause\n"
                            f"Consecutive losses reset. Trading resumes."
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
            # Cooldown increases with consecutive losses to slow down during
            # losing streaks without hard-halting.
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
                return False, f"DAILY_TRADE_LIMIT: {len(self._trades_today)}/{self._max_daily_trades}"

            if self._daily_pnl <= -self._max_daily_loss:
                self._halt("DAILY_LOSS_LIMIT")
                return False, f"DAILY_LOSS: ${self._daily_pnl:.2f}"

            if self._consecutive_losses >= self._max_consec_losses:
                self._halt("CONSECUTIVE_LOSSES")
                return False, f"CONSEC_LOSSES: {self._consecutive_losses}"

            # ── Drawdown check (equity-relative) ──────────────────────────────
            # Old bug: drawdown was calculated as % of session_high_pnl.
            # A $0.50 high followed by $0.49 drop = 98% "drawdown".
            # Fix: only trigger if session_high_pnl is meaningful (> $1).
            if self._session_high_pnl > 1.0:
                dd = (self._session_high_pnl - self._daily_pnl) / self._session_high_pnl * 100
                if dd > self._max_dd_pct:
                    self._halt("MAX_DRAWDOWN")
                    return False, f"DRAWDOWN: {dd:.1f}%"

            return True, "OK"

    def _halt(self, reason: str):
        self._is_halted   = True
        self._halt_reason = reason
        self._halt_time   = time.time()
        logger.warning(f"⛔ RiskManager HALTED: {reason}")

        auto_resumable = reason in ("CONSECUTIVE_LOSSES", "MAX_DRAWDOWN")
        resume_note = (
            f"\n⏰ Auto-resume in {self._auto_resume_seconds/60:.0f} min"
            if auto_resumable else
            "\nUse /resume to reset or /halt for emergency flatten"
        )
        self._notify(
            f"⛔ *RISK HALT: {reason}*\n"
            f"Daily PnL: `${self._daily_pnl:+.2f}`\n"
            f"Trades today: `{len(self._trades_today)}`\n"
            f"Consec losses: `{self._consecutive_losses}`"
            f"{resume_note}"
        )

    # ─── POSITION SIZING ─────────────────────────────────────────────────────

    def compute_size(self, price: float, equity_usd: float,
                     sl_pct: float = 0.0,
                     confidence: float = 1.0,
                     norm_vol: float = 0.0,
                     sz_decimals: int = 5,
                     **kwargs) -> float:
        """
        Compute position size in COIN UNITS (Hyperliquid).

        Returns a float coin size (e.g. 0.001 BTC), NOT integer contracts.
        The result is rounded to sz_decimals (from HL meta for the coin).

        Size = base_risk_size × confidence_scale × vol_scale

        confidence_scale: higher confidence → larger position (0.5x to 1.5x)
        vol_scale: higher volatility → smaller position (inverse vol targeting)

        sl_pct: if > 0, uses true risk-based sizing (Kelly-style).
        """
        with self._lock:
            risk_usd = equity_usd * (self._equity_pct / 100.0)

            # ── Confidence scaling (0.5x at 40% conf, 1.0x at 70%, 1.5x at 100%) ──
            conf_scale = max(0.5, min(1.5, confidence / 0.70))

            # ── Volatility normalization ──────────────────────────────────────
            vol_scale = 1.0
            if norm_vol > 0:
                vol_target = 0.001  # baseline: 0.1% per bar
                vol_scale = min(2.0, max(0.3, vol_target / norm_vol))

            adjusted_risk = risk_usd * conf_scale * vol_scale

            if sl_pct > 0 and price > 0:
                # Risk-based: how many coins can we risk adjusted_risk on?
                sl_loss_per_coin = price * sl_pct
                coin_size = adjusted_risk / sl_loss_per_coin if sl_loss_per_coin > 0 else 0.0
            else:
                # Notional-based: cap by max_pos_usd
                notional = min(adjusted_risk * self._leverage, self._max_pos_usd)
                coin_size = notional / price if price > 0 else 0.0

            # ── Hard cap: notional must not exceed what equity+leverage can support ──
            # The SL path above sizes purely by risk/SL-distance with no notional cap,
            # so a tight SL on a small account can produce a notional that requires
            # more margin than the account holds. Cap to 90% of max leveraged equity
            # (10% headroom so the strategy preflight's 95% check always passes).
            # Also bounded by RISK_MAX_POSITION_USD as a secondary ceiling.
            if equity_usd > 0 and price > 0:
                max_notional = min(equity_usd * self._leverage * 0.90, self._max_pos_usd)
                coin_size = min(coin_size, max_notional / price)

            # Round to exchange precision
            coin_size = round(coin_size, sz_decimals)

            # Enforce minimum (smallest tradeable unit)
            min_size = 10 ** (-sz_decimals)
            coin_size = max(coin_size, min_size)

            logger.debug(
                f"Size: base_risk=${risk_usd:.2f} conf_scale={conf_scale:.2f} "
                f"vol_scale={vol_scale:.2f} → {coin_size} coins"
            )
            return coin_size

    # ─── TRADE LIFECYCLE ──────────────────────────────────────────────────────

    def on_trade_open(self, side: str, entry_price: float, size: float,
                      margin_used: float = 0.0):
        with self._lock:
            self._open_trade = TradeRecord(
                timestamp=time.time(), side=side, entry_price=entry_price,
                size=size, margin_used=margin_used,
            )
            self._last_trade_time = time.time()

    def on_trade_close(self, exit_price: float, gross_pnl: float,
                       fees: float, net_pnl: float,
                       hold_bars: int, reason: str):
        with self._lock:
            # ── FIX: Check for midnight rollover before recording the trade ──
            # Without this, a trade closing at 00:01 UTC gets its PnL added to
            # yesterday's accumulator (which hasn't been reset yet), then wiped
            # when can_trade() triggers the daily reset.  By calling
            # _reset_if_new_day() first, the new-day reset fires before the
            # trade is recorded, so the trade correctly belongs to the new day.
            self._reset_if_new_day()

            if self._open_trade:
                self._open_trade.exit_price = exit_price
                self._open_trade.gross_pnl  = gross_pnl
                self._open_trade.fees_usd   = fees
                self._open_trade.net_pnl    = net_pnl
                self._open_trade.hold_bars  = hold_bars
                self._open_trade.reason     = reason
                self._open_trade.closed     = True
                self._trades_today.append(self._open_trade)
                self._open_trade = None

            # Track daily PnL using NET (after fees) — this is real money
            self._daily_pnl        += net_pnl
            self._total_fees       += fees
            self._session_high_pnl  = max(self._session_high_pnl, self._daily_pnl)

            if net_pnl < 0:
                is_forced = any(tag in reason for tag in (
                    "ENERGY_SPIKE", "MAX_HOLD", "SHUTDOWN", "TELEGRAM",
                ))
                if is_forced:
                    self._consecutive_losses += self._soft_loss_weight
                    self._consecutive_losses = round(self._consecutive_losses, 1)
                    logger.info(
                        f"Soft loss ({reason}): consec_losses={self._consecutive_losses} "
                        f"(weighted +{self._soft_loss_weight})"
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
            self._halt_time          = 0.0
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
            "cooldown_seconds":       ("_base_cooldown",      float),
            "max_drawdown_pct":       ("_max_dd_pct",         float),
            "equity_pct_per_trade":   ("_equity_pct",         float),
            "auto_resume_seconds":    ("_auto_resume_seconds", float),
            "soft_loss_weight":       ("_soft_loss_weight",   float),
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

            # Compute effective cooldown with graduated scaling
            effective_cooldown = self._base_cooldown * (
                1.0 + self._consecutive_losses * 0.5
            )

            # Auto-resume remaining time
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

            # Compute daily gross PnL (before fees)
            daily_gross = sum(t.gross_pnl for t in self._trades_today)

            return {
                "is_halted":          self._is_halted,
                "halt_reason":        self._halt_reason,
                "daily_pnl":          round(self._daily_pnl, 4),      # net (after fees)
                "daily_gross_pnl":    round(daily_gross, 4),
                "daily_fees":         round(self._total_fees, 4),
                "session_high_pnl":   round(self._session_high_pnl, 4),
                "trades_today":       len(self._trades_today),
                "consecutive_losses": self._consecutive_losses,
                "last_trade_time":    self._last_trade_time,
                "open_trade":         bool(self._open_trade and not self._open_trade.closed),
                "cooldown_remaining": max(0.0, effective_cooldown - (time.time() - self._last_trade_time)),
                "auto_resume_remaining": round(auto_resume_remaining, 0),
                "params": {
                    "max_pos_usd":       self._max_pos_usd,
                    "max_pos_contracts": self._max_pos_contracts,
                    "leverage":          self._leverage,
                    "max_daily_loss":    self._max_daily_loss,
                    "max_daily_trades":  self._max_daily_trades,
                    "max_consec_losses": self._max_consec_losses,
                    "cooldown":          self._base_cooldown,
                    "effective_cooldown": effective_cooldown,
                    "max_dd_pct":        self._max_dd_pct,
                    "equity_pct":        self._equity_pct,
                    "auto_resume_sec":   self._auto_resume_seconds,
                    "soft_loss_weight":  self._soft_loss_weight,
                },
            }

    def get_trade_log(self, last_n: int = 10) -> List[Dict]:
        with self._lock:
            return [
                {
                    "time":      datetime.fromtimestamp(t.timestamp, tz=timezone.utc).strftime("%H:%M:%S"),
                    "side":      t.side,
                    "entry":     t.entry_price,
                    "exit":      t.exit_price,
                    "gross_pnl": round(t.gross_pnl, 4),
                    "fees":      round(t.fees_usd, 4),
                    "net_pnl":   round(t.net_pnl, 4),
                    "roe_pct":   round(t.net_pnl / t.margin_used * 100, 2) if t.margin_used > 0 else 0.0,
                    "bars":      t.hold_bars,
                    "size":      t.size,
                    "reason":    t.reason,
                }
                for t in self._trades_today[-last_n:]
            ]
