"""
strategy.py — HPMS Trading Strategy Orchestrator
==================================================
Wires together HPMSEngine, RiskManager, OrderManager, and DataManager.
Processes each 1m bar close and manages the full trade lifecycle.

Logging philosophy:
  - DEBUG: every bar (signal, filter, risk gate outcomes)
  - INFO:  meaningful state changes only (trade open/close, strategy start/stop,
           filter blocks, risk gate blocks — throttled to once per 10 bars)
  - WARNING: unexpected situations that need attention
  - Telegram: every tradeable event, every exit, every halt
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

# ── Indian Standard Time (UTC +5:30) — all user-facing timestamps use IST ────
_IST = timezone(timedelta(hours=5, minutes=30), name="IST")

import numpy as _np                                          # module-level — not per-bar

from hpms_engine import HPMSEngine, HPMSSignal, SignalType   # module-level
from hpms_engine import RegimeType                           # module-level — was inside hot path
from risk_manager import RiskManager, TradeRecord
from order_manager import OrderManager
from state import STATE
from logger_core import elog

logger = logging.getLogger(__name__)

# How often to log a "signal blocked" INFO line when no trade happens (bars)
_BLOCK_LOG_INTERVAL = 10


class HPMSStrategy:
    """
    Full strategy lifecycle:
      1. Receive 1m bar close from DataManager
      2. Run HPMSEngine.on_bar_close() → signal
      3. Pre-trade risk checks (RiskManager.can_trade())
      4. Market microstructure filters (spread, volume, volatility)
      5. If signal passes → OrderManager.open_position()
      6. On each bar: check exit conditions (max hold, energy spike)
      7. On close: update risk metrics, log trade, notify Telegram
    """

    def __init__(
        self,
        engine:    HPMSEngine,
        risk_mgr:  RiskManager,
        order_mgr: OrderManager,
        data_mgr,
        api,
        config,
        notify_fn: Optional[Callable] = None,
    ):
        self._engine  = engine
        self._risk    = risk_mgr
        self._orders  = order_mgr
        self._data    = data_mgr
        self._api     = api
        self._config  = config
        self._notify  = notify_fn

        self._lock               = threading.RLock()
        self._enabled            = False
        self._warming_up         = False   # True during replay — engine runs, orders suppressed
        self._bar_count          = 0
        self._last_signal:       Optional[HPMSSignal] = None

        # Energy spike exit confirmation: require 2 consecutive bars above threshold
        self._consecutive_energy_spikes = 0

        # Entry tracking for ROE% computation and exit notification
        self._last_entry_size:   int   = 0
        self._last_entry_price:  float = 0.0
        self._last_margin_used:  float = 0.0
        self._entry_time:        float = 0.0   # unix seconds at entry (for hold duration)

        # Throttle "blocked" console log lines so they don't spam
        self._last_block_log_bar = 0
        self._last_block_reason  = ""
        self._block_count        = 0   # how many bars since last block log

        self._orders.set_on_close(self._on_trade_closed)
        logger.info("HPMSStrategy initialized")

    # ─── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self):
        self._enabled = True
        STATE.trading_enabled = True
        logger.info("HPMSStrategy STARTED")
        symbol = getattr(self._config, "DELTA_SYMBOL", "BTCUSD")
        self._push(
            "⚡ *Strategy STARTED*\n"
            f"Scanning `{symbol}` for HPMS signals\n"
            "_All filters and risk gates are active_"
        )

    def stop(self):
        self._enabled = False
        STATE.trading_enabled = False
        logger.info("HPMSStrategy STOPPED")
        self._push("⏹ *Strategy STOPPED* — no new entries _(open positions remain)_")

    def set_warming_up(self, value: bool):
        """
        Call True before warm-start replay, False after.
        While warming_up the engine is primed normally but NO orders are placed.
        """
        self._warming_up = value
        label = "ON (replay mode — orders suppressed)" if value else "OFF (live trading active)"
        logger.info("HPMSStrategy warming_up=%s", label)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ─── MAIN BAR PROCESSOR ──────────────────────────────────────────────────

    def on_bar_close(self, candles_1m: List[Dict]) -> Optional[HPMSSignal]:
        if not self._enabled:
            return None

        with self._lock:
            self._bar_count += 1

            valid = [c for c in candles_1m if c.get("c", 0) > 0]
            if not valid:
                return None

            closes        = [c["c"] for c in valid]
            volumes       = [c.get("v", 0.0) for c in valid]
            # Real OHLCV H/L — critical for accurate Fibonacci swing detection.
            # Fallback to close if "h"/"l" keys are absent (data-source safety).
            highs         = [c.get("h", c["c"]) for c in valid]
            lows          = [c.get("l", c["c"]) for c in valid]
            current_price = closes[-1]
            timestamp     = valid[-1].get("t", time.time() * 1000) / 1000.0

            # ── If in position: manage exit ───────────────────────────────────
            if self._orders.is_in_position:
                self._orders.on_bar()

                close_reason = self._orders.reconcile_position()
                if close_reason:
                    return None

                signal = self._engine.on_bar_close(closes, volumes, timestamp,
                                                   highs=highs, lows=lows)
                if signal:
                    # KDE grace period: skip energy spike check for 2 bars after
                    # a KDE rebuild, as dH/dt spikes artificially on rebuild bars.
                    kde_grace      = getattr(self._config, "HPMS_KDE_REBUILD_INTERVAL", 3)
                    bars_since_kde = getattr(signal, "bars_since_kde", kde_grace)
                    in_kde_grace   = bars_since_kde < 2

                    # Adaptive spike threshold — takes the higher of the adaptive
                    # estimate (mean + 3σ of recent |dH/dt| history) and the config floor.
                    adaptive_spike = self._engine.get_adaptive_dH_spike_threshold()
                    config_spike   = getattr(self._config, "TRADE_DH_DT_EXIT_SPIKE", 0.15)
                    exit_spike     = max(adaptive_spike, config_spike)

                    # ── Trailing stop management ─────────────────────────────
                    if getattr(self._config, "TRAILING_ENABLED", True):
                        trail = self._engine.compute_trailing_stop(
                            side=self._orders._position_side,
                            entry_price=self._orders.entry_price,
                            current_price=current_price,
                            current_sl=self._orders._sl_price,
                            tp_price=self._orders._tp_price,
                            entry_fee_usd=self._orders._entry_fee_usd,
                            position_size=self._orders._position_size,
                            contract_value=getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001),
                            bars_held=self._orders.bars_held,
                            config=self._config,
                        )
                        if trail.get("new_sl") is not None:
                            updated = self._orders.update_sl_price(trail["new_sl"])
                            if updated:
                                new_sl    = trail["new_sl"]
                                fib_ratio = trail.get("fib_ratio", 0.0)
                                hwm       = trail.get("high_watermark", 0.0)
                                progress  = trail.get("tp_progress", 0.0)
                                phase     = trail.get("phase", "?")
                                be_move   = trail.get("fee_breakeven_move", 0.0)
                                entry_px  = self._orders.entry_price

                                # SL movement direction and distance from entry
                                sl_dist    = abs(new_sl - entry_px)
                                sl_sign    = "+" if new_sl > entry_px else "-"

                                elog.log("TRADE_TRAIL",
                                         side=self._orders._position_side,
                                         new_sl=new_sl,
                                         phase=phase,
                                         fib_ratio=fib_ratio,
                                         tp_progress=progress,
                                         bars=self._orders.bars_held)

                                self._push(
                                    f"📏 *SL Trailed* → `${new_sl:,.1f}` "
                                    f"(`{sl_sign}${sl_dist:.1f}` from entry)\n"
                                    f"Phase: `{phase}`  │  Fib: `{fib_ratio:.3f}`  │  Bar: `{self._orders.bars_held}`\n"
                                    f"HWM: `${hwm:,.1f}`  │  Progress: `{progress:.0%}`  │  BE≥`${entry_px + be_move:,.1f}`"
                                )

                    # ── Absolute safety ceiling ──────────────────────────────
                    abs_max = getattr(self._config, "TRAILING_ABSOLUTE_MAX_BARS", 120)
                    if self._orders.bars_held >= abs_max:
                        is_long = self._orders._position_side == "long"
                        pnl     = (current_price - self._orders.entry_price if is_long
                                   else self._orders.entry_price - current_price)
                        if pnl > 0:
                            reason = (
                                f"ABSOLUTE_MAX: {self._orders.bars_held}>={abs_max} (profitable)"
                            )
                            logger.info("Safety ceiling exit: %s", reason)
                            self._orders.close_position(
                                reason=reason, current_price=current_price
                            )
                            self._consecutive_energy_spikes = 0
                            return None

                    # ── Energy spike check ───────────────────────────────────
                    exit_reason = self._risk.check_exit_conditions(
                        dH_dt=signal.dH_dt,
                        dH_dt_spike=exit_spike,
                        bars_held=self._orders.bars_held,
                    )
                    if exit_reason:
                        if in_kde_grace:
                            logger.debug(
                                "Energy spike suppressed (KDE grace, "
                                "bars_since_kde=%d): %s", bars_since_kde, exit_reason,
                            )
                            self._consecutive_energy_spikes = 0
                        else:
                            self._consecutive_energy_spikes += 1
                            if self._consecutive_energy_spikes >= 2:
                                logger.info(
                                    "Force-exit triggered (confirmed): %s", exit_reason
                                )
                                self._orders.close_position(
                                    reason=exit_reason, current_price=current_price
                                )
                                self._consecutive_energy_spikes = 0
                            else:
                                logger.info(
                                    "Energy spike detected (%d/2 for confirmation): %s",
                                    self._consecutive_energy_spikes, exit_reason,
                                )
                    else:
                        self._consecutive_energy_spikes = 0
                return None

            # ── Run HPMS engine ───────────────────────────────────────────────
            signal = self._engine.on_bar_close(closes, volumes, timestamp,
                                               highs=highs, lows=lows)
            self._last_signal = signal

            if signal is None or signal.signal_type == SignalType.FLAT:
                return signal

            # ── Signal present: try to trade ──────────────────────────────────
            sig_name = signal.signal_type.name
            logger.info(
                "bar=%d SIGNAL %s | conf=%.1f%% Δq=%+.5f "
                "|dH/dt|=%.5f compute=%.0fµs",
                self._bar_count, sig_name,
                signal.confidence * 100, signal.predicted_delta_q,
                abs(signal.dH_dt), signal.compute_time_us,
            )

            # During warm-start replay the engine is primed but no real orders fire
            if self._warming_up:
                return signal

            # ── Pre-trade risk gate ───────────────────────────────────────────
            can_trade, reason = self._risk.can_trade()
            if not can_trade:
                self._log_blocked(f"RISK GATE: {reason}", signal)
                return signal

            # ── Microstructure filters ────────────────────────────────────────
            filter_reason = self._apply_filters(candles_1m)
            if filter_reason:
                self._log_blocked(f"FILTER: {filter_reason}", signal)
                return signal

            # ── Regime filter ─────────────────────────────────────────────────
            signal_regime  = getattr(signal, "regime",         RegimeType.UNKNOWN)
            # Guard against None — getattr with default returns 0.0 but the
            # attribute could theoretically be explicitly set to None by the engine.
            _raw_trend     = getattr(signal, "trend_strength", 0.0)
            sig_trend_str  = float(_raw_trend) if _raw_trend is not None else 0.0

            if signal_regime == RegimeType.CHOPPY and signal.confidence < 0.55:
                self._log_blocked(
                    f"REGIME: CHOPPY + low conf ({signal.confidence:.1%})", signal
                )
                return signal

            if signal_regime == RegimeType.TRENDING:
                trend_aligns = (
                    (signal.signal_type == SignalType.LONG  and sig_trend_str > 0) or
                    (signal.signal_type == SignalType.SHORT and sig_trend_str < 0)
                )
                if not trend_aligns and signal.confidence < 0.70:
                    self._log_blocked(
                        f"REGIME: TRENDING counter-trend "
                        f"(strength={sig_trend_str:+.3f}) + "
                        f"insufficient conf ({signal.confidence:.1%}<0.70)",
                        signal,
                    )
                    return signal

            # ── Position sizing (confidence-weighted, vol-normalised) ─────────
            balance = self._api.get_balance("USD")
            equity  = balance.get("available", 0.0)
            if equity <= 0:
                logger.warning("No equity available — skipping entry")
                self._push("⚠️ *No equity available* — cannot open position")
                return signal

            # Compute normalized volatility for vol-adjusted sizing
            norm_vol = 0.0
            if len(candles_1m) >= 15:
                recent_closes = [c["c"] for c in candles_1m[-15:] if c.get("c", 0) > 0]
                if len(recent_closes) >= 2:
                    _rc   = _np.array(recent_closes)
                    _atr  = float(_np.mean(_np.abs(_np.diff(_rc))))
                    _mid  = float(_np.mean(_rc))
                    norm_vol = _atr / _mid if _mid > 0 else 0.0

            sl_pct = (
                abs(current_price - signal.sl_price) / current_price
                if signal.sl_price > 0 else 0.0
            )
            size = self._risk.compute_size(
                current_price, equity,
                contract_value=getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001),
                sl_pct=sl_pct,
                confidence=signal.confidence,
                norm_vol=norm_vol,
            )
            side = "long" if signal.signal_type == SignalType.LONG else "short"

            # ── Pre-flight margin safety clamp ────────────────────────────────
            # compute_size() enforces the margin cap internally, so this block
            # should never trigger in normal operation.  Retained as a safety net:
            # if the computed size would exceed 95% of available equity when
            # converted to margin, clamp size down rather than discard the signal.
            contract_value = getattr(self._orders, "_contract_value", 0.001)
            leverage       = getattr(self._config, "RISK_LEVERAGE", 10)
            margin_needed  = (current_price * contract_value * size) / max(leverage, 1)

            if margin_needed > equity * 0.95:
                safe_size = max(
                    1, int(equity * 0.95 * leverage / (current_price * contract_value))
                )
                logger.warning(
                    "MARGIN_PREFLIGHT: clamping size %d→%d "
                    "(need $%.2f, have $%.2f @ %dx) — trade PROCEEDS at safe size",
                    size, safe_size, margin_needed, equity, leverage,
                )
                size          = safe_size
                margin_needed = (current_price * contract_value * size) / max(leverage, 1)

                if margin_needed > equity * 0.95:
                    logger.warning(
                        "MARGIN_PREFLIGHT: even 1 contract requires $%.2f "
                        "but equity=$%.2f — cannot trade, deposit required",
                        margin_needed, equity,
                    )
                    self._push(
                        f"⚠️ *Cannot open position* — 1 contract costs "
                        f"`${margin_needed:.2f}` margin but equity is "
                        f"`${equity:.2f}` @ `{leverage}x`.\n"
                        f"Deposit funds or reduce leverage to continue trading."
                    )
                    return signal

            # ── EXECUTE ENTRY ─────────────────────────────────────────────────
            result = self._orders.open_position(
                side=side,
                size=size,
                price=current_price,
                tp_price=signal.tp_price,
                sl_price=signal.sl_price,
                use_bracket=getattr(self._config, "TRADE_USE_BRACKET_ORDERS", True),
                order_type=getattr(self._config, "TRADE_ORDER_TYPE", "market"),
            )

            if result.get("success"):
                actual_entry   = self._orders.entry_price or current_price
                contract_value = getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001)
                leverage       = getattr(self._config, "RISK_LEVERAGE", 10)
                notional       = actual_entry * contract_value * size
                margin_used    = notional / max(leverage, 1)
                entry_fee      = getattr(self._orders, "_entry_fee_usd", 0.0)

                # Derived trade geometry
                tp_dist    = abs(signal.tp_price - actual_entry)
                sl_dist    = abs(signal.sl_price - actual_entry)
                tp_pct     = tp_dist / actual_entry * 100
                sl_pct_val = sl_dist / actual_entry * 100
                rr         = tp_dist / sl_dist if sl_dist > 0 else 0.0
                tp_sign    = "+" if signal.tp_price > actual_entry else "-"
                sl_sign    = "-" if signal.sl_price < actual_entry else "+"

                self._risk.on_trade_open(side, actual_entry, size, margin_used)
                self._engine.reset_trail_watermark()
                self._consecutive_energy_spikes = 0
                self._last_entry_size   = size
                self._last_entry_price  = actual_entry
                self._last_margin_used  = margin_used
                self._entry_time        = time.time()

                now_ist = datetime.now(_IST).strftime("%H:%M:%S IST")

                # Daily session snapshot for context
                daily = self._risk.get_status()
                trades_before = daily["trades_today"]  # includes this just-opened one

                elog.log("TRADE_ENTRY",
                         side=side, size=size,
                         entry_price=actual_entry,
                         tp_price=signal.tp_price, sl_price=signal.sl_price,
                         confidence=signal.confidence,
                         note=f"rr={rr:.2f}  bar={self._bar_count}")

                logger.info(
                    "TRADE OPEN ▶ %s %dc @ $%s  TP=$%s  SL=$%s  "
                    "RR=%.2f  margin=$%.2f  fee=$-%.4f  conf=%.1f%%  "
                    "regime=%s  trend=%+.3f  id=%s",
                    side.upper(), size, f"{actual_entry:,.1f}",
                    f"{float(signal.tp_price):,.1f}", f"{float(signal.sl_price):,.1f}", rr,
                    margin_used, entry_fee, signal.confidence * 100,
                    signal_regime.name, sig_trend_str,
                    result.get("order_id", ""),
                )

                self._push(
                    f"🚀 *ENTRY {side.upper()}*  ⏱ `{now_ist}`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📍 `{size}c` × `${actual_entry:,.1f}`\n"
                    f"   Notional: `${notional:,.2f}`  │  Margin: `${margin_used:.2f}` @ `{leverage}x`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎯 TP: `${signal.tp_price:,.1f}` (`{tp_sign}${tp_dist:.1f}` / `+{tp_pct:.2f}%`)\n"
                    f"🛑 SL: `${signal.sl_price:,.1f}` (`{sl_sign}${sl_dist:.1f}` / `-{sl_pct_val:.2f}%`)\n"
                    f"⚖️ R:R: `{rr:.2f}:1`  │  Fee in: `$-{entry_fee:.4f}`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔬 Conf: `{signal.confidence:.1%}`  │  Δq: `{signal.predicted_delta_q:+.5f}`\n"
                    f"   dH/dt: `{signal.dH_dt:.5f}`  │  Regime: `{signal_regime.name}` "
                    f"trend `{sig_trend_str:+.3f}`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Session: `{trades_before}` trades  │  Daily net: "
                    f"`${daily['daily_pnl']:+.4f}`\n"
                    f"\n{signal.fib_telegram_section}"
                )
                self._block_count = 0
            else:
                err = result.get("error", "unknown")
                logger.error("Entry FAILED: %s", err)
                self._push(f"❌ *Entry failed*: `{err}`")

            return signal

    # ─── BLOCK LOGGING (throttled) ────────────────────────────────────────────

    def _log_blocked(self, reason: str, signal: HPMSSignal):
        """
        Log that a valid signal was blocked by risk/filter.

        Prints one INFO line immediately, then only every _BLOCK_LOG_INTERVAL
        bars to avoid burying the log in repeated "COOLDOWN" messages.
        """
        self._block_count += 1
        sig_name = signal.signal_type.name

        if (self._block_count == 1
                or self._block_count % _BLOCK_LOG_INTERVAL == 0
                or reason != self._last_block_reason):
            logger.info(
                "bar=%d BLOCKED (%dx) %s conf=%.1f%% | %s",
                self._bar_count, self._block_count,
                sig_name, signal.confidence * 100, reason,
            )
            self._last_block_reason = reason

    # ─── FILTERS ──────────────────────────────────────────────────────────────

    def _apply_filters(self, candles_1m: List[Dict]) -> Optional[str]:
        ob   = self._data.get_orderbook()
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if bids and asks:
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                if best_bid > 0:
                    spread_pct = (best_ask - best_bid) / best_bid
                    # Config value is in percent (e.g. 0.04 means 0.04%)
                    max_spread = getattr(self._config, "FILTER_SPREAD_MAX_PCT", 0.05) / 100.0
                    if spread_pct > max_spread:
                        return f"SPREAD {spread_pct:.4%} > {max_spread:.4%}"
            except (IndexError, ValueError):
                pass

        if candles_1m:
            last_vol = candles_1m[-1].get("v", 0)
            min_vol  = getattr(self._config, "FILTER_MIN_VOLUME_1M", 10.0)
            if last_vol < min_vol:
                return f"VOLUME {last_vol:.1f} < {min_vol}"

        if len(candles_1m) >= 10:
            recent  = candles_1m[-10:]
            atr     = sum(
                c.get("h", c.get("c", 0)) - c.get("l", c.get("c", 0))
                for c in recent
            ) / len(recent)
            mid = candles_1m[-1]["c"]
            if mid > 0:
                atr_pct = atr / mid * 100
                vol_min = getattr(self._config, "FILTER_VOLATILITY_MIN_PCT", 0.01)
                vol_max = getattr(self._config, "FILTER_VOLATILITY_MAX_PCT", 2.0)
                if atr_pct < vol_min:
                    return f"VOL_LOW ATR={atr_pct:.4f}% < {vol_min}%"
                if atr_pct > vol_max:
                    return f"VOL_HIGH ATR={atr_pct:.4f}% > {vol_max}%"

        return None

    # ─── TRADE CLOSE CALLBACK ─────────────────────────────────────────────────

    def _on_trade_closed(self, exit_price: float, gross_pnl: float,
                         fees: float, net_pnl: float,
                         bars_held: int, reason: str):
        self._risk.on_trade_close(exit_price, gross_pnl, fees, net_pnl,
                                  bars_held, reason)

        daily    = self._risk.get_status()
        emoji    = "💰" if net_pnl >= 0 else "🔻"
        roe_pct  = (net_pnl / self._last_margin_used * 100) if self._last_margin_used > 0 else 0.0

        # Price move from entry to exit
        entry_px = self._last_entry_price
        move_usd = exit_price - entry_px
        move_pct = move_usd / entry_px * 100 if entry_px > 0 else 0.0
        # For short trades, a negative price move is a positive result
        side_str  = "LONG" if (net_pnl >= 0) == (move_usd >= 0) else "SHORT"
        # Actual hold time in minutes
        hold_mins = (time.time() - self._entry_time) / 60.0 if self._entry_time > 0 else 0.0

        # Streak emoji — show the last 5 trade outcomes
        trade_log = self._risk.get_trade_log(5)
        streak_icons = " ".join(
            "✅" if t["net_pnl"] >= 0 else "❌"
            for t in trade_log
        )

        now_ist = datetime.now(_IST).strftime("%H:%M:%S IST")

        elog.log("TRADE_EXIT",
                 reason=reason,
                 exit_price=exit_price,
                 net_pnl=net_pnl,
                 roe_pct=roe_pct,
                 bars=bars_held)

        logger.info(
            "TRADE CLOSE ■ reason=%s  exit=$%s  entry=$%s  "
            "gross=$%+.4f  fees=$-%.4f  net=$%+.4f  ROE=%+.2f%%  "
            "bars=%d  daily_net=$%+.4f  trades=%d  consec_loss=%.1f",
            reason, f"{exit_price:,.1f}", f"{entry_px:,.1f}",
            gross_pnl, fees, net_pnl, roe_pct,
            bars_held,
            daily["daily_pnl"], daily["trades_today"], daily["consecutive_losses"],
        )

        self._push(
            f"{emoji} *EXIT: {reason}*  ⏱ `{now_ist}`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry: `${entry_px:,.1f}` → Exit: `${exit_price:,.1f}`\n"
            f"   Move: `{move_usd:+.1f}` USD  /  `{move_pct:+.2f}%`\n"
            f"   Held: `{bars_held}` bars  /  `{hold_mins:.1f}` min\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Gross: `${gross_pnl:+.4f}`  │  Fees: `$-{fees:.4f}`\n"
            f"{'✅' if net_pnl >= 0 else '❌'} *Net P&L: `${net_pnl:+.4f}`*  │  ROE: `{roe_pct:+.2f}%`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Session today:\n"
            f"   Trades: `{daily['trades_today']}`  │  Net: `${daily['daily_pnl']:+.4f}`\n"
            f"   Fees: `$-{daily['daily_fees']:.4f}`  │  High: `${daily['session_high_pnl']:+.4f}`\n"
            f"   Streak: {streak_icons}  │  Consec loss: `{daily['consecutive_losses']}`"
        )

    # ─── NOTIFICATIONS ────────────────────────────────────────────────────────

    def _push(self, text: str):
        if self._notify:
            try:
                self._notify(text)
            except Exception as e:
                logger.debug("Telegram notify error: %s", e)

    # ─── STATUS ───────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "enabled":   self._enabled,
            "bar_count": self._bar_count,
            "last_signal": {
                "type":       self._last_signal.signal_type.name if self._last_signal else None,
                "confidence": self._last_signal.confidence if self._last_signal else None,
                "delta_q":    self._last_signal.predicted_delta_q if self._last_signal else None,
                "reason":     self._last_signal.reason if self._last_signal else None,
                "compute_us": self._last_signal.compute_time_us if self._last_signal else None,
            },
            "position": self._orders.get_status(),
            "risk":     self._risk.get_status(),
            "engine":   self._engine.get_params(),
        }
