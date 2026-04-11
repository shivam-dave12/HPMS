"""
order_manager.py — Hyperliquid Order Execution Manager
========================================================
Handles order placement (market + separate SL/TP via positionTpsl),
position tracking, forced exits, and fill detection.

FEES & P&L — All sourced from Hyperliquid fill data:
  - Entry fee:  `fee` field from the market order fill
  - Exit fee:   `fee` field from the closing fill (SL/TP or force-close)
  - Gross PnL:  `closedPnl` from the closing fill, or computed from prices
  - Prices:     actual fill price from the order response

Hyperliquid fee schedule (base tier, no staking):
  Taker: 0.045%  (market orders always pay taker)
  Maker: 0.015%

If the API does not return a fee value, a conservative taker-fee estimate
is computed locally and flagged as estimated.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Dict, Optional

import config

logger = logging.getLogger(__name__)


def _safe_float(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class OrderManager:
    """
    Manages order lifecycle on Hyperliquid.

    Key differences from Delta:
    - Sizing is in coin units (float), not integer contracts
    - SL/TP are placed as separate positionTpsl orders
    - Fees come from fill events, not order responses
    - P&L uses closedPnl from fills or price-based computation
    """

    def __init__(self, api, symbol: str = "BTC", **kwargs):
        self._api = api        # HyperliquidClient (async) — wrapped via _run()
        self._symbol = symbol
        self._lock = threading.RLock()

        # Async bridge — set by main.py after loop is running
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Active state
        self._position_side:    Optional[str] = None
        self._position_size:    float = 0.0     # coin size (e.g. 0.001 BTC)
        self._entry_price:      float = 0.0
        self._tp_price:         float = 0.0
        self._sl_price:         float = 0.0
        self._entry_bar:        int   = 0
        self._entry_time:       float = 0.0
        self._entry_fee_usd:    float = 0.0
        self._sl_order_id:      Optional[int] = None
        self._tp_order_id:      Optional[int] = None
        self._entry_order_id:   Optional[int] = None

        # Callbacks
        self._on_fill_cb:  Optional[Callable] = None
        self._on_close_cb: Optional[Callable] = None

        logger.info(f"OrderManager initialized: symbol={symbol} (Hyperliquid)")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def _run(self, coro, timeout=30):
        """Run async coroutine from sync context."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("Event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ─── CALLBACKS ────────────────────────────────────────────────────────────

    def set_on_fill(self, cb: Callable):
        self._on_fill_cb = cb

    def set_on_close(self, cb: Callable):
        self._on_close_cb = cb

    # ─── ENTRY ────────────────────────────────────────────────────────────────

    def open_position(
        self,
        side:       str,        # "long" | "short"
        size:       float,      # coin size (e.g. 0.001 BTC)
        price:      float,
        tp_price:   float,
        sl_price:   float,
        use_bracket: bool = True,
        order_type:  str  = "market",
        **kwargs,
    ) -> Dict:
        with self._lock:
            if self._position_side is not None:
                return {"success": False, "error": "ALREADY_IN_POSITION"}

            is_buy = (side == "long")

            try:
                # Set leverage first
                leverage = getattr(config, "RISK_LEVERAGE", 50)
                try:
                    self._run(self._api.set_leverage(self._symbol, leverage))
                except Exception as e:
                    logger.warning(f"Set leverage warning: {e}")

                # Place market entry
                resp = self._run(self._api.place_market_order(
                    coin=self._symbol,
                    is_buy=is_buy,
                    size=size,
                    slippage=0.005,
                ))

                # Parse response
                if not self._is_order_success(resp):
                    error = self._extract_error(resp)
                    logger.error(f"Entry order failed: {error}")
                    return {"success": False, "error": error}

                # Extract fill data
                statuses = self._get_statuses(resp)
                actual_entry_price = price
                entry_fee = 0.0
                order_id = 0

                if statuses:
                    first = statuses[0]
                    if isinstance(first, dict):
                        filled = first.get("filled", {})
                        actual_entry_price = _safe_float(filled.get("avgPx"), price)
                        order_id = int(filled.get("oid", 0))
                        # Fee comes from totalFee in the status
                        entry_fee = abs(_safe_float(filled.get("totalFee"), 0.0))

                # If no fee from response, estimate from taker rate
                if entry_fee <= 0:
                    notional = actual_entry_price * size
                    entry_fee = notional * config.FEE_TAKER_RATE
                    logger.debug(
                        f"Entry fee estimated (taker {config.FEE_TAKER_RATE:.5f}): "
                        f"${entry_fee:.6f}"
                    )

                self._entry_order_id = order_id
                self._position_side = side
                self._position_size = size
                self._entry_price = actual_entry_price
                self._tp_price = tp_price
                self._sl_price = sl_price
                self._entry_time = time.time()
                self._entry_bar = 0
                self._entry_fee_usd = entry_fee

                logger.info(
                    f"✅ ENTRY {side.upper()} {size} {self._symbol} "
                    f"@ ~{actual_entry_price:.1f} "
                    f"TP={tp_price:.1f} SL={sl_price:.1f} "
                    f"fee=${entry_fee:.6f} oid={order_id}"
                )

                # Place SL/TP as positionTpsl orders
                if use_bracket:
                    self._place_sl_tp(side, size, sl_price, tp_price)

                if self._on_fill_cb:
                    self._on_fill_cb(side, size, actual_entry_price, order_id)

                return {"success": True, "order_id": order_id, "error": None}

            except Exception as e:
                logger.error(f"Entry exception: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

    def _place_sl_tp(self, side: str, size: float, sl_price: float, tp_price: float):
        """Place SL and TP as separate positionTpsl trigger orders."""
        # For a LONG position: SL sells (is_buy=False), TP sells (is_buy=False)
        # For a SHORT position: SL buys (is_buy=True), TP buys (is_buy=True)
        exit_is_buy = (side == "short")

        # Stop Loss
        try:
            resp = self._run(self._api.place_stop_loss(
                coin=self._symbol,
                is_buy=exit_is_buy,
                size=size,
                trigger_price=sl_price,
            ))
            statuses = self._get_statuses(resp)
            if statuses:
                first = statuses[0]
                if isinstance(first, dict):
                    resting = first.get("resting", {})
                    self._sl_order_id = int(resting.get("oid", 0)) or None
            if self._sl_order_id:
                logger.info(f"SL order placed: oid={self._sl_order_id} @ {sl_price:.1f}")
            else:
                logger.warning(f"SL order response: {str(resp)[:200]}")
        except Exception as e:
            logger.error(f"SL placement error: {e}")

        # Take Profit
        try:
            resp = self._run(self._api.place_take_profit(
                coin=self._symbol,
                is_buy=exit_is_buy,
                size=size,
                trigger_price=tp_price,
            ))
            statuses = self._get_statuses(resp)
            if statuses:
                first = statuses[0]
                if isinstance(first, dict):
                    resting = first.get("resting", {})
                    self._tp_order_id = int(resting.get("oid", 0)) or None
            if self._tp_order_id:
                logger.info(f"TP order placed: oid={self._tp_order_id} @ {tp_price:.1f}")
            else:
                logger.warning(f"TP order response: {str(resp)[:200]}")
        except Exception as e:
            logger.error(f"TP placement error: {e}")

    # ─── EXIT ─────────────────────────────────────────────────────────────────

    def close_position(self, reason: str = "MANUAL", current_price: float = 0.0) -> Dict:
        with self._lock:
            if self._position_side is None:
                return {"success": False, "error": "NO_POSITION"}

            try:
                # Cancel SL/TP orders first
                self._cancel_sl_tp()

                # Close via market order (reduce_only)
                resp = self._run(self._api.close_position(
                    coin=self._symbol,
                    is_long=(self._position_side == "long"),
                    size=self._position_size,
                ))

                exit_price = current_price
                exit_fee = 0.0

                statuses = self._get_statuses(resp)
                if statuses:
                    first = statuses[0]
                    if isinstance(first, dict):
                        filled = first.get("filled", {})
                        exit_price = _safe_float(filled.get("avgPx"), current_price)
                        exit_fee = abs(_safe_float(filled.get("totalFee"), 0.0))
                        closed_pnl = _safe_float(filled.get("closedPnl"), None)

                if exit_price <= 0:
                    exit_price = current_price

                # Compute PnL
                gross_pnl = self._compute_pnl_from_prices(exit_price)

                # If no exit fee from response, estimate
                if exit_fee <= 0:
                    notional = exit_price * self._position_size
                    exit_fee = notional * config.FEE_TAKER_RATE
                    logger.debug(f"Exit fee estimated: ${exit_fee:.6f}")

                total_fees = self._entry_fee_usd + exit_fee
                net_pnl = gross_pnl - total_fees
                bars_held = self._entry_bar

                logger.info(
                    f"🔻 EXIT {self._position_side.upper()} {self._position_size} "
                    f"@ ~{exit_price:.1f} gross=${gross_pnl:+.4f} "
                    f"fees=$-{total_fees:.4f} net=${net_pnl:+.4f} "
                    f"bars={bars_held} reason={reason}"
                )

                if self._on_close_cb:
                    self._on_close_cb(exit_price, gross_pnl, total_fees,
                                      net_pnl, bars_held, reason)

                self._reset_position()
                return {"success": True, "gross_pnl": gross_pnl,
                        "fees": total_fees, "net_pnl": net_pnl,
                        "reason": reason}

            except Exception as e:
                logger.error(f"Close exception: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

    def _cancel_sl_tp(self):
        for oid in (self._sl_order_id, self._tp_order_id):
            if oid:
                try:
                    self._run(self._api.cancel_order(self._symbol, oid))
                except Exception:
                    pass
        self._sl_order_id = None
        self._tp_order_id = None

    def _reset_position(self):
        self._position_side = None
        self._position_size = 0.0
        self._entry_price = 0.0
        self._tp_price = 0.0
        self._sl_price = 0.0
        self._entry_bar = 0
        self._entry_time = 0.0
        self._entry_fee_usd = 0.0
        self._sl_order_id = None
        self._tp_order_id = None
        self._entry_order_id = None

    # ─── BAR TICK ─────────────────────────────────────────────────────────────

    def on_bar(self):
        with self._lock:
            if self._position_side:
                self._entry_bar += 1

    # ─── FILL DETECTION (called from main loop) ──────────────────────────────

    def check_position_on_exchange(self) -> Optional[str]:
        """
        Check if our position still exists on exchange.
        If SL/TP filled on exchange, detect and sync.
        Returns close reason if position was closed, else None.
        """
        with self._lock:
            if self._position_side is None:
                return None

            try:
                wallet = config.get_settings().wallet_address
                state = self._run(self._api.get_clearinghouse_state(wallet))
                positions = state.get("assetPositions", [])

                # Find our coin
                for pos_wrapper in positions:
                    pos = pos_wrapper.get("position", pos_wrapper)
                    if pos.get("coin") == self._symbol:
                        szi = _safe_float(pos.get("szi"), 0.0)
                        if abs(szi) > 1e-12:
                            # Position still exists
                            return None

                # Position gone — closed by SL/TP on exchange
                # Get fill data for PnL
                exit_price, exit_fee, closed_pnl = self._get_recent_fill_data()
                if exit_price <= 0:
                    exit_price = self._entry_price

                if closed_pnl is not None and abs(closed_pnl) > 0:
                    gross_pnl = closed_pnl
                else:
                    gross_pnl = self._compute_pnl_from_prices(exit_price)

                if exit_fee <= 0:
                    notional = exit_price * self._position_size
                    exit_fee = notional * config.FEE_TAKER_RATE

                total_fees = self._entry_fee_usd + exit_fee
                net_pnl = gross_pnl - total_fees
                reason = self._classify_exit(exit_price)

                logger.info(
                    f"Position closed on exchange ({reason}) "
                    f"exit={exit_price:.1f} gross=${gross_pnl:+.4f} "
                    f"fees=$-{total_fees:.4f} net=${net_pnl:+.4f}"
                )

                if self._on_close_cb:
                    self._on_close_cb(exit_price, gross_pnl, total_fees,
                                      net_pnl, self._entry_bar, reason)
                self._reset_position()
                return reason

            except Exception as e:
                logger.debug(f"Position check error: {e}")
                return None

    def _get_recent_fill_data(self) -> tuple:
        """Get recent fill data for P&L computation."""
        try:
            wallet = config.get_settings().wallet_address
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - 60_000  # last 60 seconds
            fills = self._run(
                self._api.get_user_fills_by_time(wallet, start_ms, now_ms)
            )
            if isinstance(fills, list):
                for f in reversed(fills):
                    if f.get("coin") == self._symbol:
                        px = _safe_float(f.get("px"), 0.0)
                        fee = abs(_safe_float(f.get("fee"), 0.0))
                        closed_pnl = _safe_float(f.get("closedPnl"), 0.0)
                        return px, fee, closed_pnl
        except Exception as e:
            logger.debug(f"_get_recent_fill_data error: {e}")
        return 0.0, 0.0, None

    # ─── RECONCILIATION ──────────────────────────────────────────────────────

    def reconcile_position(self) -> Optional[str]:
        """Alias for check_position_on_exchange for interface compatibility."""
        return self.check_position_on_exchange()

    def register_with_data_manager(self, data_mgr) -> None:
        """Interface compatibility — HL uses exchange polling instead of WS callbacks."""
        logger.info("OrderManager: HL uses position polling (no WS fill callbacks)")

    # ─── PnL HELPERS ──────────────────────────────────────────────────────────

    def _compute_pnl_from_prices(self, exit_price: float) -> float:
        if exit_price <= 0 or self._entry_price <= 0:
            return 0.0
        diff = exit_price - self._entry_price
        if self._position_side == "short":
            diff = -diff
        return diff * self._position_size

    def _classify_exit(self, exit_price: float) -> str:
        if exit_price <= 0:
            return "EXCHANGE_CLOSE"
        if self._tp_price > 0 and self._sl_price > 0:
            tp_dist = abs(exit_price - self._tp_price)
            sl_dist = abs(exit_price - self._sl_price)
            return "TP_HIT" if tp_dist < sl_dist else "SL_HIT"
        if self._position_side == "long":
            return "TP_HIT" if exit_price > self._entry_price else "SL_HIT"
        else:
            return "TP_HIT" if exit_price < self._entry_price else "SL_HIT"

    # ─── RESPONSE PARSING ────────────────────────────────────────────────────

    @staticmethod
    def _is_order_success(resp) -> bool:
        if isinstance(resp, dict):
            status = resp.get("status")
            if status == "ok":
                return True
            response = resp.get("response", {})
            if isinstance(response, dict) and response.get("type") == "order":
                return True
        return False

    @staticmethod
    def _get_statuses(resp) -> list:
        if isinstance(resp, dict):
            response = resp.get("response", {})
            if isinstance(response, dict):
                data = response.get("data", {})
                if isinstance(data, dict):
                    return data.get("statuses", [])
        return []

    @staticmethod
    def _extract_error(resp) -> str:
        if isinstance(resp, dict):
            response = resp.get("response", {})
            if isinstance(response, dict):
                data = response.get("data", {})
                if isinstance(data, dict):
                    statuses = data.get("statuses", [])
                    for s in statuses:
                        if isinstance(s, dict) and "error" in s:
                            return s["error"]
                        if isinstance(s, str):
                            return s
            return str(resp)[:200]
        return str(resp)[:200]

    # ─── STATUS ───────────────────────────────────────────────────────────────

    @property
    def is_in_position(self) -> bool:
        with self._lock:
            return self._position_side is not None

    @property
    def position_side(self) -> Optional[str]:
        with self._lock:
            return self._position_side

    @property
    def bars_held(self) -> int:
        with self._lock:
            return self._entry_bar

    @property
    def entry_price(self) -> float:
        with self._lock:
            return self._entry_price

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "in_position":  self._position_side is not None,
                "side":         self._position_side,
                "size":         self._position_size,
                "entry_price":  self._entry_price,
                "bars_held":    self._entry_bar,
                "entry_time":   self._entry_time,
                "sl_order":     self._sl_order_id,
                "tp_order":     self._tp_order_id,
                "entry_order":  self._entry_order_id,
            }

    # ─── EMERGENCY ────────────────────────────────────────────────────────────

    def emergency_flatten(self) -> Dict:
        results = []
        try:
            wallet = config.get_settings().wallet_address
            open_orders = self._run(self._api.get_open_orders(wallet))
            if isinstance(open_orders, list):
                for o in open_orders:
                    if o.get("coin") == self._symbol:
                        oid = o.get("oid")
                        if oid:
                            self._run(self._api.cancel_order(self._symbol, int(oid)))
                            results.append(f"cancelled {oid}")
        except Exception as e:
            results.append(f"cancel_error: {e}")

        try:
            if self._position_side:
                self._run(self._api.close_position(
                    coin=self._symbol,
                    is_long=(self._position_side == "long"),
                    size=self._position_size,
                ))
                results.append("position_closed")
        except Exception as e:
            results.append(f"close_error: {e}")

        self._reset_position()
        return {"success": True, "actions": results}
