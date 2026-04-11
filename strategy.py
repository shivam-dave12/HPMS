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
import config

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

        # Dynamic hold: TP distance stored at entry for max_hold computation
        self._current_tp_distance: float = 0.0

        # Entry tracking for ROE% computation
        self._last_entry_size:  float = 0.0
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
                   f"`{getattr(self._config, 'HL_SYMBOL', 'BTC')}`")

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

                    # Dynamic max hold based on ATR and TP distance
                    dynamic_max_hold = self._engine.get_dynamic_max_hold(
                        self._current_tp_distance, current_price
                    )

                    exit_reason = self._risk.check_exit_conditions(
                        dH_dt=signal.dH_dt,
                        dH_dt_spike=exit_spike,
                        bars_held=self._orders.bars_held,
                        max_hold=dynamic_max_hold,
                    )
                    if exit_reason:
                        # MAX_HOLD always exits immediately
                        if exit_reason.startswith("MAX_HOLD"):
                            self._consecutive_energy_spikes = 0
                            logger.info(f"Force-exit triggered: {exit_reason}")
                            self._orders.close_position(
                                reason=exit_reason, current_price=current_price
                            )
                        elif in_kde_grace:
                            # Skip energy spike during KDE grace period
                            logger.debug(
                                f"Energy spike suppressed (KDE grace, "
                                f"bars_since_kde={bars_since_kde}): {exit_reason}"
                            )
                            self._consecutive_energy_spikes = 0
                        else:
                            # Require 2 consecutive energy spikes to confirm exit
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
            # In CHOPPY regime with low confidence, skip — edge is insufficient
            from hpms_engine import RegimeType
            signal_regime = getattr(signal, "regime", RegimeType.UNKNOWN)
            if signal_regime == RegimeType.CHOPPY and signal.confidence < 0.55:
                self._log_blocked(
                    f"REGIME: CHOPPY + low conf ({signal.confidence:.1%})", signal
                )
                return signal

            # ── Position sizing (confidence-weighted, vol-normalized) ─────────
            equity = self._get_equity()
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
                sl_pct=abs(current_price - signal.sl_price) / current_price if signal.sl_price > 0 else 0.0,
                confidence=signal.confidence,
                norm_vol=norm_vol,
            )
            side = "long" if signal.signal_type == SignalType.LONG else "short"

            # ── Pre-flight margin check ───────────────────────────────────────
            # Hyperliquid: notional = price × coin_size, margin = notional / leverage
            leverage      = getattr(self._config, "RISK_LEVERAGE", 50)
            notional_est  = current_price * size
            margin_needed = notional_est / max(leverage, 1)
            if margin_needed > equity * 0.95:
                # compute_size already caps at 90% of leveraged equity, so this
                # should rarely fire. When it does (e.g. rounding up), clamp the
                # size down to the largest amount that fits before giving up.
                import math as _math
                max_notional  = equity * 0.90 * leverage
                clamped_size  = _math.floor(
                    (max_notional / current_price) * 1e5
                ) / 1e5                              # floor at 5 dp (BTC default)
                hl_min_notional = 10.0               # Hyperliquid $10 minimum order
                if clamped_size * current_price >= hl_min_notional:
                    logger.info(
                        "MARGIN_PREFLIGHT: clamped size %.5f→%.5f "
                        "(need $%.2f, have $%.2f equity @ %dx)",
                        size, clamped_size, margin_needed, equity, leverage,
                    )
                    size          = clamped_size
                    notional_est  = current_price * size
                    margin_needed = notional_est / max(leverage, 1)
                else:
                    logger.warning(
                        "MARGIN_PREFLIGHT FAIL: equity=$%.2f too small for "
                        "HL $10 minimum even at %dx leverage — skipping entry",
                        equity, leverage,
                    )
                    self._push(
                        f"⚠️ *Margin too low* — equity `${equity:.2f}` cannot "
                        f"meet HL's $10 minimum notional at `{leverage}x` leverage."
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
                leverage = getattr(self._config, "RISK_LEVERAGE", 50)
                notional = actual_entry * size
                margin_used = notional / max(leverage, 1)

                # Use EXACT entry fee from API (stored by order manager)
                entry_fee = getattr(self._orders, "_entry_fee_usd", 0.0)

                self._risk.on_trade_open(side, actual_entry, size, margin_used)
                self._current_tp_distance = abs(signal.tp_price - actual_entry)
                self._consecutive_energy_spikes = 0
                self._last_entry_size  = size
                self._last_entry_price = actual_entry
                self._last_margin_used = margin_used

                logger.info(
                    f"TRADE OPEN ▶ {side.upper()} {size} {getattr(self._config, 'HL_SYMBOL', 'BTC')} "
                    f"@ ${actual_entry:,.1f} "
                    f"TP=${signal.tp_price:,.1f} SL=${signal.sl_price:,.1f} "
                    f"margin=${margin_used:.2f} fee=$-{entry_fee:.4f} "
                    f"conf={signal.confidence:.1%} regime={signal_regime.name} "
                    f"id={result.get('order_id', '')}"
                )
                self._push(
                    f"🚀 *ENTRY {side.upper()}*\n"
                    f"Size: `{size}` {getattr(self._config, 'HL_SYMBOL', 'BTC')} | "
                    f"Notional: `${notional:.2f}`\n"
                    f"Entry: `${actual_entry:,.1f}`\n"
                    f"TP: `${signal.tp_price:,.1f}` (`${abs(signal.tp_price - actual_entry):.1f}`)\n"
                    f"SL: `${signal.sl_price:,.1f}` (`${abs(signal.sl_price - actual_entry):.1f}`)\n"
                    f"Margin: `${margin_used:.2f}` @ `{leverage}x`\n"
                    f"Entry fee: `$-{entry_fee:.4f}` _(from exchange)_\n"
                    f"Conf: `{signal.confidence:.1%}` | Regime: `{signal_regime.name}`\n"
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

    # ─── EQUITY FETCH ─────────────────────────────────────────────────────────

    def _get_equity(self) -> float:
        """
        Fetch account equity via the canonical HyperliquidClient.get_equity()
        waterfall, routed through SyncHLAPI.get_equity().

        Waterfall (handled entirely in api_client.py — do NOT duplicate here):
          1. marginSummary.accountValue    — total NAV (cross + isolated + unrealisedPnL)
          2. crossMarginSummary.accountValue — cross-margin NAV
          3. withdrawable                  — free cash (0 when locked in open positions)
          4. marginSummary.totalRawUsd     — raw deposited USDC (ignores PnL)
          5. Spot USDC balance             — funds not yet transferred to perp
          6. API-wallet cross-check        — detects HL_WALLET_ADDRESS misconfiguration

        IMPORTANT: Do NOT use get_clearinghouse_state() + manual parsing here.
        That approach was responsible for all previous zero-equity bugs:
          - 'withdrawable' is 0.0 whenever funds are tied up in an open position.
          - crossMarginSummary misses isolated-margin equity entirely.
          - Neither checks the spot account.
        All of that is handled correctly in api_client.get_equity().
        """
        try:
            master_wallet = config.get_settings().wallet_address
            if not master_wallet:
                logger.warning("Equity fetch: HL_WALLET_ADDRESS is not set in .env")
                return 0.0

            # Route through SyncHLAPI → HyperliquidClient.get_equity()
            # which runs the full 6-step waterfall including spot fallback.
            equity = self._api.get_equity(master_wallet)

            if equity <= 0:
                logger.warning(
                    f"Equity fetch returned 0 for wallet {master_wallet} — "
                    "check api_client logs for 'clearinghouse_zero_equity' details. "
                    "Possible causes: (1) HL_WALLET_ADDRESS is the API sub-wallet not "
                    "the master account, (2) USDC has not been deposited, "
                    "(3) funds are in spot account — transfer at app.hyperliquid.xyz."
                )
            return equity

        except AttributeError:
            # SyncHLAPI older version without get_equity() — fall back gracefully
            logger.warning(
                "SyncHLAPI.get_equity() not found — update main.py. "
                "Falling back to get_clearinghouse_state() with basic parsing."
            )
            return self._get_equity_legacy()
        except Exception as e:
            logger.warning(f"Equity fetch error: {e}")
            return 0.0

    def _get_equity_legacy(self) -> float:
        """
        Legacy fallback — only used if SyncHLAPI.get_equity() is unavailable.
        Uses marginSummary.accountValue (correct primary field) rather than
        withdrawable (which is 0 when funds are locked in open positions).
        """
        try:
            master_wallet = config.get_settings().wallet_address
            state = self._api.get_clearinghouse_state(master_wallet)
            if not state:
                return 0.0
            ms  = state.get("marginSummary")      or {}
            cms = state.get("crossMarginSummary") or {}

            def _sf(v):
                try:
                    return float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    return 0.0

            # Priority: total NAV → cross NAV → totalRawUsd → withdrawable
            v = _sf(ms.get("accountValue"))
            if v > 0:
                return v
            v = _sf(cms.get("accountValue"))
            if v > 0:
                return v
            v = _sf(ms.get("totalRawUsd"))
            if v > 0:
                return v
            return max(_sf(state.get("withdrawable")), 0.0)
        except Exception as e:
            logger.warning(f"Legacy equity fetch error: {e}")
            return 0.0

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
