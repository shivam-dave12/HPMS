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
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from hpms_engine import HPMSEngine, HPMSSignal, SignalType
from risk_manager import RiskManager, TradeRecord
from order_manager import OrderManager
from state import STATE

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

        # Entry tracking for ROE% computation
        self._last_entry_size:  int   = 0
        self._last_entry_price: float = 0.0
        self._last_margin_used: float = 0.0

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
        self._push("⚡ *Strategy STARTED* — looking for signals on "
                   f"`{getattr(self._config, 'DELTA_SYMBOL', 'BTCUSD')}`")

    def stop(self):
        self._enabled = False
        STATE.trading_enabled = False
        logger.info("HPMSStrategy STOPPED")
        self._push("⏹ *Strategy STOPPED* — no new entries (open positions remain)")

    def set_warming_up(self, value: bool):
        """
        Call with True before warm-start replay, False after.
        While warming_up the engine is primed normally but NO orders are placed.
        """
        self._warming_up = value
        logger.info(f"HPMSStrategy warming_up={'ON (replay mode — orders suppressed)' if value else 'OFF (live trading active)'}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ─── MAIN BAR PROCESSOR ──────────────────────────────────────────────────

    def on_bar_close(self, candles_1m: List[Dict]) -> Optional[HPMSSignal]:
        if not self._enabled:
            return None

        with self._lock:
            self._bar_count += 1

            valid    = [c for c in candles_1m if c.get("c", 0) > 0]
            if not valid:
                return None

            closes       = [c["c"] for c in valid]
            volumes      = [c.get("v", 0.0) for c in valid]
            current_price = closes[-1]
            timestamp     = valid[-1].get("t", time.time() * 1000) / 1000.0

            # ── If in position: manage exit ───────────────────────────────────
            if self._orders.is_in_position:
                self._orders.on_bar()

                close_reason = self._orders.reconcile_position()
                if close_reason:
                    return None

                signal = self._engine.on_bar_close(closes, volumes, timestamp)
                if signal:
                    # KDE grace period: skip energy spike check for 2 bars after
                    # a KDE rebuild, as dH/dt spikes artificially on rebuild bars.
                    kde_grace = getattr(self._config, "HPMS_KDE_REBUILD_INTERVAL", 3)
                    bars_since_kde = getattr(signal, "bars_since_kde", kde_grace)
                    in_kde_grace = bars_since_kde < 2

                    # Use adaptive threshold if available, else config
                    adaptive_spike = self._engine.get_adaptive_dH_spike_threshold()
                    config_spike = getattr(self._config, "TRADE_DH_DT_EXIT_SPIKE", 0.15)
                    exit_spike = max(adaptive_spike, config_spike)

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
                                self._push(
                                    f"🔄 *SL Trailed* → `${trail['new_sl']:,.1f}` "
                                    f"({trail['phase']}) bar={self._orders.bars_held}"
                                )

                    # ── Absolute safety ceiling ──────────────────────────────
                    abs_max = getattr(self._config, "TRAILING_ABSOLUTE_MAX_BARS", 120)
                    if self._orders.bars_held >= abs_max:
                        # Only force exit if profitable (SL handles losses)
                        is_long = self._orders._position_side == "long"
                        if is_long:
                            pnl = current_price - self._orders.entry_price
                        else:
                            pnl = self._orders.entry_price - current_price
                        if pnl > 0:
                            reason = f"ABSOLUTE_MAX: {self._orders.bars_held}>={abs_max} (profitable)"
                            logger.info(f"Safety ceiling exit: {reason}")
                            self._orders.close_position(
                                reason=reason, current_price=current_price
                            )
                            self._consecutive_energy_spikes = 0
                            return None
                        # else: let trailing SL or exchange SL handle it

                    # ── Energy spike check ───────────────────────────────────
                    exit_reason = self._risk.check_exit_conditions(
                        dH_dt=signal.dH_dt,
                        dH_dt_spike=exit_spike,
                        bars_held=self._orders.bars_held,
                    )
                    if exit_reason:
                        if in_kde_grace:
                            logger.debug(
                                f"Energy spike suppressed (KDE grace, "
                                f"bars_since_kde={bars_since_kde}): {exit_reason}"
                            )
                            self._consecutive_energy_spikes = 0
                        else:
                            self._consecutive_energy_spikes += 1
                            if self._consecutive_energy_spikes >= 2:
                                logger.info(f"Force-exit triggered (confirmed): {exit_reason}")
                                self._orders.close_position(
                                    reason=exit_reason, current_price=current_price
                                )
                                self._consecutive_energy_spikes = 0
                            else:
                                logger.info(
                                    f"Energy spike detected ({self._consecutive_energy_spikes}/2 "
                                    f"for confirmation): {exit_reason}"
                                )
                    else:
                        self._consecutive_energy_spikes = 0
                return None

            # ── Run HPMS engine ───────────────────────────────────────────────
            signal = self._engine.on_bar_close(closes, volumes, timestamp)
            self._last_signal = signal

            if signal is None or signal.signal_type == SignalType.FLAT:
                # Engine already emits per-bar INFO diagnostics in on_bar_close
                return signal

            # ── Signal present: try to trade ──────────────────────────────────
            sig_name = signal.signal_type.name
            logger.info(
                f"bar={self._bar_count} SIGNAL {sig_name} | "
                f"conf={signal.confidence:.1%} Δq={signal.predicted_delta_q:+.5f} "
                f"|dH/dt|={abs(signal.dH_dt):.5f} compute={signal.compute_time_us:.0f}µs"
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
            # Two independent checks:
            #
            # 1. CHOPPY + low confidence: edge is insufficient in a directionless
            #    market.  Unchanged from previous logic.
            #
            # 2. TRENDING + signal opposes the measured trend direction: a counter-
            #    trend entry into confirmed momentum requires materially higher
            #    confidence (≥ 0.70) to justify overriding the regime signal.
            #    Production logs showed ALL LONG signals occurring with negative
            #    trend_strength (bearish regime), yet passing the filter because
            #    only CHOPPY was checked.  This adds the missing TRENDING gate.
            from hpms_engine import RegimeType, SignalType as _ST
            signal_regime    = getattr(signal, "regime",          RegimeType.UNKNOWN)
            sig_trend_str    = getattr(signal, "trend_strength",  0.0)

            if signal_regime == RegimeType.CHOPPY and signal.confidence < 0.55:
                self._log_blocked(
                    f"REGIME: CHOPPY + low conf ({signal.confidence:.1%})", signal
                )
                return signal

            if signal_regime == RegimeType.TRENDING:
                trend_aligns = (
                    (signal.signal_type == _ST.LONG  and sig_trend_str > 0) or
                    (signal.signal_type == _ST.SHORT and sig_trend_str < 0)
                )
                if not trend_aligns and signal.confidence < 0.70:
                    self._log_blocked(
                        f"REGIME: TRENDING counter-trend "
                        f"(strength={sig_trend_str:+.3f}) + "
                        f"insufficient conf ({signal.confidence:.1%}<0.70)",
                        signal,
                    )
                    return signal

            # ── Position sizing (confidence-weighted, vol-normalized) ─────────
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
                    import numpy as _np
                    _rc = _np.array(recent_closes)
                    _atr = float(_np.mean(_np.abs(_np.diff(_rc))))
                    _mid = float(_np.mean(_rc))
                    norm_vol = _atr / _mid if _mid > 0 else 0.0

            size = self._risk.compute_size(
                current_price, equity,
                contract_value=getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001),
                sl_pct=abs(current_price - signal.sl_price) / current_price if signal.sl_price > 0 else 0.0,
                confidence=signal.confidence,
                norm_vol=norm_vol,
            )
            side = "long" if signal.signal_type == SignalType.LONG else "short"

            # ── Pre-flight margin safety clamp ────────────────────────────────
            # compute_size() now enforces the margin cap internally, so this
            # block should never trigger in normal operation.  It is retained
            # as a belt-and-suspenders guard: if, for any reason, the computed
            # size would exceed 95% of available equity when converted to margin,
            # we clamp size down to the maximum safe value rather than skipping
            # the signal entirely.  A valid signal is NEVER discarded due to size.
            contract_value = getattr(self._orders, "_contract_value", 0.001)
            leverage       = getattr(self._config, "RISK_LEVERAGE", 10)
            margin_needed  = (current_price * contract_value * size) / max(leverage, 1)

            if margin_needed > equity * 0.95:
                # Compute the largest size that fits within 95% of equity
                safe_size = max(1, int(equity * 0.95 * leverage / (current_price * contract_value)))
                logger.warning(
                    "MARGIN_PREFLIGHT: clamping size %d→%d "
                    "(need $%.2f, have $%.2f @ %dx) — trade PROCEEDS at safe size",
                    size, safe_size, margin_needed, equity, leverage,
                )
                size = safe_size
                margin_needed = (current_price * contract_value * size) / max(leverage, 1)

                # If even 1 contract exceeds margin, the account is too small
                # to open any position — skip with an actionable message.
                if margin_needed > equity * 0.95:
                    logger.warning(
                        "MARGIN_PREFLIGHT: even 1 contract requires $%.2f "
                        "but equity=$%.2f — cannot trade, deposit required",
                        margin_needed, equity,
                    )
                    self._push(
                        f"⚠️ *Cannot open position* — 1 contract costs "
                        f"`${margin_needed:.2f}` margin but available equity is "
                        f"`${equity:.2f}` @ `{leverage}x` leverage.\n"
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
                # Use ACTUAL entry price from order manager (API fill price)
                actual_entry = self._orders.entry_price or current_price
                contract_value = getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001)
                leverage = getattr(self._config, "RISK_LEVERAGE", 10)
                margin_used = (actual_entry * contract_value * size) / max(leverage, 1)
                notional = actual_entry * contract_value * size

                # Use EXACT entry fee from API (stored by order manager)
                entry_fee = getattr(self._orders, "_entry_fee_usd", 0.0)

                self._risk.on_trade_open(side, actual_entry, size, margin_used)
                self._consecutive_energy_spikes = 0
                self._last_entry_size  = size
                self._last_entry_price = actual_entry
                self._last_margin_used = margin_used

                logger.info(
                    f"TRADE OPEN ▶ {side.upper()} {size}c @ ${actual_entry:,.1f} "
                    f"TP=${signal.tp_price:,.1f} SL=${signal.sl_price:,.1f} "
                    f"margin=${margin_used:.2f} fee=$-{entry_fee:.4f} "
                    f"conf={signal.confidence:.1%} regime={signal_regime.name} "
                    f"trend={sig_trend_str:+.3f} "
                    f"id={result.get('order_id', '')}"
                )
                self._push(
                    f"🚀 *ENTRY {side.upper()}*\n"
                    f"Size: `{size}c` | Notional: `${notional:.2f}`\n"
                    f"Entry: `${actual_entry:,.1f}`\n"
                    f"TP: `${signal.tp_price:,.1f}` (`${abs(signal.tp_price - actual_entry):.1f}`)\n"
                    f"SL: `${signal.sl_price:,.1f}` (`${abs(signal.sl_price - actual_entry):.1f}`)\n"
                    f"Margin: `${margin_used:.2f}` @ `{leverage}x`\n"
                    f"Entry fee: `$-{entry_fee:.4f}` _(exact)_\n"
                    f"Conf: `{signal.confidence:.1%}` | Regime: `{signal_regime.name}` "
                    f"trend `{sig_trend_str:+.3f}`\n"
                    f"Δq: `{signal.predicted_delta_q:+.5f}`"
                )
                self._block_count = 0
            else:
                err = result.get("error", "unknown")
                logger.error(f"Entry FAILED: {err}")
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

        if (self._block_count == 1                  # always log the first block
                or self._block_count % _BLOCK_LOG_INTERVAL == 0
                or reason != self._last_block_reason):  # or when reason changes
            logger.info(
                f"bar={self._bar_count} BLOCKED ({self._block_count}x) "
                f"{sig_name} conf={signal.confidence:.1%} | {reason}"
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
            atr     = sum(c.get("h", c.get("c", 0)) - c.get("l", c.get("c", 0)) for c in recent) / len(recent)
            mid     = candles_1m[-1]["c"]
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

        daily = self._risk.get_status()
        emoji = "💰" if net_pnl >= 0 else "🔻"

        # ROE% = net_pnl / margin_used
        roe_pct = (net_pnl / self._last_margin_used * 100) if self._last_margin_used > 0 else 0.0

        logger.info(
            f"TRADE CLOSE ■ reason={reason} exit=${exit_price:,.1f} "
            f"gross=${gross_pnl:+.4f} fees=$-{fees:.4f} net=${net_pnl:+.4f} "
            f"ROE={roe_pct:+.2f}% bars={bars_held} "
            f"daily_net=${daily['daily_pnl']:+.4f} "
            f"trades={daily['trades_today']} consec_loss={daily['consecutive_losses']}"
        )
        self._push(
            f"{emoji} *EXIT: {reason}*\n"
            f"Exit: `${exit_price:,.1f}`\n"
            f"Gross: `${gross_pnl:+.4f}` | Fees: `$-{fees:.4f}`\n"
            f"*Net: `${net_pnl:+.4f}`* | ROE: `{roe_pct:+.2f}%`\n"
            f"Held: `{bars_held}` bars\n"
            f"Daily: gross `${daily.get('daily_gross_pnl', 0):+.4f}` "
            f"net `${daily['daily_pnl']:+.4f}` "
            f"fees `$-{daily.get('daily_fees', 0):.4f}`\n"
            f"Trades: `{daily['trades_today']}`"
        )

    # ─── NOTIFICATIONS ────────────────────────────────────────────────────────

    def _push(self, text: str):
        if self._notify:
            try:
                self._notify(text)
            except Exception as e:
                logger.debug(f"Telegram notify error: {e}")

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
