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
        self._bar_count          = 0
        self._last_signal:       Optional[HPMSSignal] = None

        # Throttle "blocked" console log lines so they don't spam
        self._last_block_log_bar = 0
        self._last_block_reason  = ""
        self._block_count        = 0   # how many bars since last block log

        self._orders.set_on_close(self._on_trade_closed)
        logger.info("HPMSStrategy initialized")

    # ─── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self):
        self._enabled = True
        logger.info("HPMSStrategy STARTED")
        self._push("⚡ *Strategy STARTED* — looking for signals on "
                   f"`{getattr(self._config, 'DELTA_SYMBOL', 'BTCUSD')}`")

    def stop(self):
        self._enabled = False
        logger.info("HPMSStrategy STOPPED")
        self._push("⏹ *Strategy STOPPED* — no new entries (open positions remain)")

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
                    exit_reason = self._risk.check_exit_conditions(
                        dH_dt=signal.dH_dt,
                        dH_dt_spike=getattr(self._config, "TRADE_DH_DT_EXIT_SPIKE", 0.15),
                        bars_held=self._orders.bars_held,
                        max_hold=getattr(self._config, "TRADE_MAX_HOLD_BARS", 5),
                    )
                    if exit_reason:
                        logger.info(f"Force-exit triggered: {exit_reason}")
                        self._orders.close_position(
                            reason=exit_reason, current_price=current_price
                        )
                return None

            # ── Run HPMS engine ───────────────────────────────────────────────
            signal = self._engine.on_bar_close(closes, volumes, timestamp)
            self._last_signal = signal

            if signal is None or signal.signal_type == SignalType.FLAT:
                # Log the flat reason at DEBUG (not INFO) — one line per bar is too spammy
                if signal:
                    logger.debug(f"bar={self._bar_count} FLAT | {signal.reason}")
                return signal

            # ── Signal present: try to trade ──────────────────────────────────
            sig_name = signal.signal_type.name
            logger.info(
                f"bar={self._bar_count} SIGNAL {sig_name} | "
                f"conf={signal.confidence:.1%} Δq={signal.predicted_delta_q:+.5f} "
                f"|dH/dt|={abs(signal.dH_dt):.5f} compute={signal.compute_time_us:.0f}µs"
            )

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

            # ── Position sizing ───────────────────────────────────────────────
            balance = self._api.get_balance("USD")
            equity  = balance.get("available", 0.0)
            if equity <= 0:
                logger.warning("No equity available — skipping entry")
                self._push("⚠️ *No equity available* — cannot open position")
                return signal

            size = self._risk.compute_size(current_price, equity)
            side = "long" if signal.signal_type == SignalType.LONG else "short"

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
                self._risk.on_trade_open(side, current_price, size)
                logger.info(
                    f"TRADE OPEN ▶ {side.upper()} {size}c @ ${current_price:,.1f} "
                    f"TP=${signal.tp_price:,.1f} SL=${signal.sl_price:,.1f} "
                    f"conf={signal.confidence:.1%} id={result.get('order_id', '')}"
                )
                self._push(
                    f"🚀 *ENTRY {side.upper()}*\n"
                    f"Size: `{size}` contracts\n"
                    f"Entry: `${current_price:,.1f}`\n"
                    f"TP: `${signal.tp_price:,.1f}` | SL: `${signal.sl_price:,.1f}`\n"
                    f"Confidence: `{signal.confidence:.1%}`\n"
                    f"Δq: `{signal.predicted_delta_q:+.5f}` | |dH/dt|: `{abs(signal.dH_dt):.5f}`\n"
                    f"Compute: `{signal.compute_time_us:.0f}µs`"
                )
                # Reset block counter after a successful trade
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
            atr     = sum(c["h"] - c["l"] for c in recent) / len(recent)
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

    def _on_trade_closed(self, exit_price: float, pnl_usd: float, bars_held: int, reason: str):
        self._risk.on_trade_close(exit_price, pnl_usd, bars_held, reason)

        daily = self._risk.get_status()
        emoji = "💰" if pnl_usd >= 0 else "🔻"

        logger.info(
            f"TRADE CLOSE ■ reason={reason} exit=${exit_price:,.1f} "
            f"pnl=${pnl_usd:+.4f} bars={bars_held} "
            f"daily_pnl=${daily['daily_pnl']:+.2f} "
            f"trades={daily['trades_today']} consec_loss={daily['consecutive_losses']}"
        )
        self._push(
            f"{emoji} *EXIT: {reason}*\n"
            f"Exit: `${exit_price:,.1f}`\n"
            f"PnL: `${pnl_usd:+.2f}` | Held: `{bars_held}` bars\n"
            f"Daily PnL: `${daily['daily_pnl']:+.2f}` | "
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
