"""
order_manager.py — Delta Exchange Order Execution Manager
==========================================================
Handles atomic order placement (bracket or separate SL/TP), position tracking,
forced exits, and order state reconciliation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages order lifecycle on Delta Exchange.

    - Entry: market or limit with bracket SL/TP
    - Exit: forced close, SL/TP hit detection, max-hold timeout
    - Reconciliation: poll position state to detect fills
    """

    def __init__(self, api, symbol: str = "BTCUSD", contract_value: float = 0.001,
                 taker_fee_pct: float = 0.05, maker_fee_pct: float = 0.02):
        self._api    = api
        self._symbol = symbol
        self._lock   = threading.RLock()
        self._contract_value = contract_value  # BTC per contract (Delta BTCUSD = 0.001)

        # Fee rates (percentage, e.g. 0.05 = 0.05%)
        self._taker_fee_pct = taker_fee_pct
        self._maker_fee_pct = maker_fee_pct

        # Active state
        self._position_side:    Optional[str] = None   # "long" | "short" | None
        self._position_size:    int   = 0
        self._entry_price:      float = 0.0
        self._tp_price:         float = 0.0   # stored at entry for correct TP/SL classification
        self._sl_price:         float = 0.0   # stored at entry for correct TP/SL classification
        self._entry_bar:        int   = 0
        self._entry_time:       float = 0.0
        self._entry_fee_usd:    float = 0.0   # fee paid at entry
        self._sl_order_id:      Optional[str] = None
        self._tp_order_id:      Optional[str] = None
        self._entry_order_id:   Optional[str] = None

        # Callbacks
        self._on_fill_cb:  Optional[Callable] = None
        self._on_close_cb: Optional[Callable] = None

        # Product ID (cached for speed)
        self._product_id: Optional[int] = None
        self._resolve_product_id()

        logger.info(
            f"OrderManager initialized: symbol={symbol} "
            f"contract_value={self._contract_value} product_id={self._product_id}"
        )

    def _resolve_product_id(self):
        """Resolve product_id and auto-detect contract_value from ticker."""
        try:
            self._product_id = self._api.get_product_id(self._symbol)
            if self._product_id:
                logger.info(f"OrderManager product_id: {self._product_id}")

            # Auto-detect contract_value from ticker
            ticker = self._api.get_ticker(self._symbol)
            if ticker.get("success"):
                raw = ticker.get("result", {})
                cv = raw.get("contract_value")
                if cv:
                    self._contract_value = float(cv)
                    logger.info(
                        f"OrderManager contract_value auto-detected: "
                        f"{self._contract_value} (from ticker)"
                    )
        except Exception as e:
            logger.error(f"Failed to resolve product_id/contract_value: {e}")

    # ─── CALLBACKS ────────────────────────────────────────────────────────────

    def set_on_fill(self, cb: Callable):
        self._on_fill_cb = cb

    def set_on_close(self, cb: Callable):
        self._on_close_cb = cb

    # ─── ENTRY ────────────────────────────────────────────────────────────────

    def open_position(
        self,
        side:       str,       # "long" | "short"
        size:       int,       # contracts
        price:      float,     # current price (for limit offset)
        tp_price:   float,
        sl_price:   float,
        use_bracket: bool = True,
        order_type:  str  = "market",
        limit_offset_ticks: int = 1,
    ) -> Dict:
        """
        Open a new position with SL/TP.

        Returns {"success": bool, "order_id": str, "error": str}
        """
        with self._lock:
            if self._position_side is not None:
                return {"success": False, "error": "ALREADY_IN_POSITION"}

            delta_side = "buy" if side == "long" else "sell"

            try:
                if use_bracket and order_type == "market":
                    # Bracket order: entry + SL + TP in one atomic request
                    resp = self._api.place_bracket_order(
                        symbol=self._symbol,
                        side=delta_side,
                        size=size,
                        order_type="market",
                        bracket_stop_loss_price=round(sl_price, 1),
                        bracket_take_profit_price=round(tp_price, 1),
                        product_id=self._product_id,
                    )
                else:
                    # Separate orders
                    resp = self._api.place_market_order(
                        symbol=self._symbol,
                        side=delta_side,
                        size=size,
                        product_id=self._product_id,
                    )

                if not resp.get("success"):
                    error = resp.get("error", "unknown")
                    logger.error(f"Entry order failed: {error}")
                    return {"success": False, "error": error}

                order_id = resp["result"].get("order_id", "")
                self._entry_order_id = order_id
                self._position_side  = side
                self._position_size  = size
                self._entry_price    = price
                self._tp_price       = tp_price
                self._sl_price       = sl_price
                self._entry_time     = time.time()
                self._entry_bar      = 0
                self._entry_fee_usd  = self._compute_fee(
                    price, size, is_taker=(order_type == "market")
                )

                # For bracket orders Delta returns the leg IDs directly in the result.
                # Capture them now so _cancel_sl_tp() can cancel them on a force-exit.
                if use_bracket and order_type == "market":
                    result_data = resp.get("result", {})
                    sl_id = result_data.get("bracket_stop_loss_order_id") or \
                            result_data.get("stop_loss_order", {}).get("order_id")
                    tp_id = result_data.get("bracket_take_profit_order_id") or \
                            result_data.get("take_profit_order", {}).get("order_id")
                    self._sl_order_id = str(sl_id) if sl_id else None
                    self._tp_order_id = str(tp_id) if tp_id else None
                    if self._sl_order_id or self._tp_order_id:
                        logger.info(
                            f"Bracket legs captured: SL={self._sl_order_id} TP={self._tp_order_id}"
                        )
                    else:
                        logger.warning(
                            "Bracket order placed but leg IDs not found in response — "
                            "force-exit will rely on exchange auto-cancel. "
                            f"Response keys: {list(result_data.keys())}"
                        )

                logger.info(
                    f"✅ ENTRY {side.upper()} {size} contracts @ ~{price:.1f} "
                    f"TP={tp_price:.1f} SL={sl_price:.1f} id={order_id}"
                )

                # If not bracket, place SL/TP separately
                if not use_bracket or order_type != "market":
                    self._place_sl_tp(side, size, sl_price, tp_price)

                if self._on_fill_cb:
                    self._on_fill_cb(side, size, price, order_id)

                return {"success": True, "order_id": order_id, "error": None}

            except Exception as e:
                logger.error(f"Entry exception: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

    def _place_sl_tp(self, side: str, size: int, sl_price: float, tp_price: float):
        """Place separate stop-loss and take-profit orders."""
        exit_side = "sell" if side == "long" else "buy"

        try:
            sl_resp = self._api.place_stop_market_order(
                symbol=self._symbol,
                side=exit_side,
                size=size,
                stop_price=round(sl_price, 1),
                reduce_only=True,
                product_id=self._product_id,
            )
            if sl_resp.get("success"):
                self._sl_order_id = sl_resp["result"].get("order_id")
                logger.info(f"SL order placed: {self._sl_order_id} @ {sl_price:.1f}")
            else:
                logger.error(f"SL order failed: {sl_resp.get('error')}")
        except Exception as e:
            logger.error(f"SL placement error: {e}")

        try:
            tp_resp = self._api.place_take_profit_market_order(
                symbol=self._symbol,
                side=exit_side,
                size=size,
                stop_price=round(tp_price, 1),
                reduce_only=True,
                product_id=self._product_id,
            )
            if tp_resp.get("success"):
                self._tp_order_id = tp_resp["result"].get("order_id")
                logger.info(f"TP order placed: {self._tp_order_id} @ {tp_price:.1f}")
            else:
                logger.error(f"TP order failed: {tp_resp.get('error')}")
        except Exception as e:
            logger.error(f"TP placement error: {e}")

    # ─── EXIT ─────────────────────────────────────────────────────────────────

    def close_position(self, reason: str = "MANUAL", current_price: float = 0.0) -> Dict:
        """Force-close the current position at market."""
        with self._lock:
            if self._position_side is None:
                return {"success": False, "error": "NO_POSITION"}

            try:
                # Cancel any outstanding SL/TP orders
                self._cancel_sl_tp()

                # Close via market order (reduce_only)
                exit_side = "sell" if self._position_side == "long" else "buy"
                resp = self._api.place_market_order(
                    symbol=self._symbol,
                    side=exit_side,
                    size=self._position_size,
                    reduce_only=True,
                    product_id=self._product_id,
                )

                if not resp.get("success"):
                    # Try position close endpoint as fallback
                    if self._product_id:
                        resp = self._api.close_position(self._product_id)

                # Compute PnL with fees
                if current_price > 0:
                    gross_pnl, total_fees, net_pnl = self._compute_pnl_full(
                        current_price, exit_is_taker=True  # market exit = taker
                    )
                else:
                    gross_pnl = total_fees = net_pnl = 0.0

                bars_held = self._entry_bar

                logger.info(
                    f"🔻 EXIT {self._position_side.upper()} {self._position_size} "
                    f"@ ~{current_price:.1f} gross=${gross_pnl:+.4f} "
                    f"fees=$-{total_fees:.4f} net=${net_pnl:+.4f} "
                    f"bars={bars_held} reason={reason}"
                )

                if self._on_close_cb:
                    self._on_close_cb(current_price, gross_pnl, total_fees,
                                      net_pnl, bars_held, reason)

                self._reset_position()
                return {"success": True, "gross_pnl": gross_pnl,
                        "fees": total_fees, "net_pnl": net_pnl,
                        "reason": reason}

            except Exception as e:
                logger.error(f"Close exception: {e}", exc_info=True)
                return {"success": False, "error": str(e)}

    def _cancel_sl_tp(self):
        """Cancel outstanding SL/TP orders."""
        for oid in (self._sl_order_id, self._tp_order_id):
            if oid:
                try:
                    self._api.cancel_order(oid)
                except Exception:
                    pass
        self._sl_order_id = None
        self._tp_order_id = None

    def _reset_position(self):
        self._position_side  = None
        self._position_size  = 0
        self._entry_price    = 0.0
        self._tp_price       = 0.0
        self._sl_price       = 0.0
        self._entry_bar      = 0
        self._entry_time     = 0.0
        self._entry_fee_usd  = 0.0
        self._sl_order_id    = None
        self._tp_order_id    = None
        self._entry_order_id = None

    # ─── BAR TICK ─────────────────────────────────────────────────────────────

    def on_bar(self):
        """Called each new bar — increments hold counter."""
        with self._lock:
            if self._position_side:
                self._entry_bar += 1

    # ─── DATA MANAGER INTEGRATION ────────────────────────────────────────────

    def register_with_data_manager(self, data_mgr) -> None:
        """
        Wire real-time order/fill events from DataManager's WebSocket.
        Call once after both objects are constructed.
        """
        data_mgr.register_fill_callback(self._on_ws_fill)
        data_mgr.register_order_callback(self._on_ws_order)
        logger.info("OrderManager registered fill+order callbacks with DataManager")

    def _on_ws_fill(self, data: dict) -> None:
        """
        Called immediately when the WS fills channel fires.
        Detects bracket TP/SL fills and records the actual exit price + PnL.
        """
        with self._lock:
            if self._position_side is None:
                return
            try:
                fill_price = float(data.get("price") or data.get("fill_price") or 0)
                fill_side  = str(data.get("side", "")).lower()   # "buy" | "sell"
                fill_size  = float(data.get("size") or data.get("fill_quantity") or 0)
                role       = str(data.get("role", "")).lower()   # "taker" | "maker"

                if fill_price <= 0 or fill_size <= 0:
                    return

                # Exit fills are on the OPPOSITE side to our position
                expected_exit_side = "buy" if self._position_side == "short" else "sell"
                if fill_side != expected_exit_side:
                    return

                # Determine TP vs SL from price
                reason = self._classify_exit(fill_price)
                is_taker = (role == "taker") if role else True
                gross_pnl, total_fees, net_pnl = self._compute_pnl_full(
                    fill_price, exit_is_taker=is_taker
                )

                logger.info(
                    f"🔔 WS FILL detected: {reason} {self._position_side.upper()} "
                    f"fill_price={fill_price:.1f} gross=${gross_pnl:+.4f} "
                    f"fees=$-{total_fees:.4f} net=${net_pnl:+.4f}"
                )

                if self._on_close_cb:
                    self._on_close_cb(fill_price, gross_pnl, total_fees,
                                      net_pnl, self._entry_bar, reason)
                self._reset_position()

            except Exception as e:
                logger.debug(f"_on_ws_fill error: {e}")

    def _on_ws_order(self, data: dict) -> None:
        """
        Called when WS order state changes. Used as a belt-and-suspenders
        check: if an order in 'closed'/'filled' state arrives and we're still
        tracking a position, trigger reconcile.
        """
        with self._lock:
            if self._position_side is None:
                return
            state = str(data.get("state", "")).lower()
            if state in ("filled", "closed", "cancelled"):
                logger.debug(f"WS order state={state} — scheduling reconcile check")

    # ─── PNL HELPERS ──────────────────────────────────────────────────────────

    def _compute_fee(self, price: float, size: int, is_taker: bool = True) -> float:
        """Compute fee in USD for a fill at given price and size."""
        notional = price * self._contract_value * size
        rate = self._taker_fee_pct if is_taker else self._maker_fee_pct
        return notional * rate / 100.0

    def _compute_pnl(self, exit_price: float) -> float:
        """Compute gross USD PnL (no fees) for current open position."""
        if exit_price <= 0 or self._entry_price <= 0:
            return 0.0
        diff = exit_price - self._entry_price
        if self._position_side == "short":
            diff = -diff
        return diff * self._contract_value * self._position_size

    def _compute_pnl_full(self, exit_price: float, exit_is_taker: bool = True):
        """
        Compute full PnL breakdown: gross, entry_fee, exit_fee, net.
        Returns (gross_pnl, total_fees, net_pnl).
        """
        gross = self._compute_pnl(exit_price)
        entry_fee = self._entry_fee_usd
        exit_fee = self._compute_fee(exit_price, self._position_size, exit_is_taker)
        total_fees = entry_fee + exit_fee
        return gross, total_fees, gross - total_fees

    def _classify_exit(self, exit_price: float) -> str:
        """Return 'TP_HIT', 'SL_HIT', or 'EXCHANGE_CLOSE' based on actual TP/SL levels."""
        if exit_price <= 0:
            return "EXCHANGE_CLOSE"
        # Compare against the stored TP/SL prices set at entry, not just entry_price.
        # Using entry_price as the boundary misclassifies a tiny bounce above entry as
        # TP_HIT even when price never came close to the real TP level (Bug 2 fix).
        if self._tp_price > 0 and self._sl_price > 0:
            if self._position_side == "long":
                tp_dist = abs(exit_price - self._tp_price)
                sl_dist = abs(exit_price - self._sl_price)
                return "TP_HIT" if tp_dist < sl_dist else "SL_HIT"
            else:
                tp_dist = abs(exit_price - self._tp_price)
                sl_dist = abs(exit_price - self._sl_price)
                return "TP_HIT" if tp_dist < sl_dist else "SL_HIT"
        # Fallback if TP/SL prices were never recorded
        if self._position_side == "long":
            return "TP_HIT" if exit_price > self._entry_price else "SL_HIT"
        else:
            return "TP_HIT" if exit_price < self._entry_price else "SL_HIT"

    def _get_exit_price_from_fills(self) -> tuple:
        """Fetch most recent fill from REST API to get actual exit price."""
        try:
            resp = self._api.get_fills(symbol=self._symbol, page_size=5)
            if resp.get("success"):
                raw = resp.get("result", {})
                # Delta wraps fills under 'fills' or 'data' key
                fills = raw if isinstance(raw, list) else \
                        raw.get("fills", raw.get("data", []))
                if fills:
                    latest    = fills[0]
                    price_val = latest.get("price") or latest.get("fill_price") or 0
                    price     = float(price_val)
                    if price > 0:
                        return price, self._classify_exit(price)
        except Exception as e:
            logger.debug(f"get_fills error during reconcile: {e}")
        return 0.0, "EXCHANGE_CLOSE"

    # ─── RECONCILIATION ──────────────────────────────────────────────────────

    def reconcile_position(self) -> Optional[str]:
        """
        Poll exchange for actual position state.
        If position was closed by SL/TP on exchange side, detect and sync.
        Returns close reason if position was closed externally, else None.
        """
        with self._lock:
            if self._position_side is None:
                return None

            try:
                pos = self._api.get_position(self._symbol)
                if pos.get("success"):
                    result = pos.get("result", {})
                    size = result.get("size", 0)
                    if size == 0 or result.get("side") is None:
                        # Position closed on exchange (SL/TP hit) — get real exit price
                        exit_price, reason = self._get_exit_price_from_fills()
                        # SL/TP fills are typically taker on Delta
                        gross_pnl, total_fees, net_pnl = self._compute_pnl_full(
                            exit_price, exit_is_taker=True
                        )

                        logger.info(
                            f"Position closed on exchange ({reason}) "
                            f"exit_price={exit_price:.1f} gross=${gross_pnl:+.4f} "
                            f"fees=$-{total_fees:.4f} net=${net_pnl:+.4f}"
                        )
                        if self._on_close_cb:
                            self._on_close_cb(exit_price, gross_pnl, total_fees,
                                              net_pnl, self._entry_bar, reason)
                        self._reset_position()
                        return reason
            except Exception as e:
                logger.debug(f"Reconciliation error: {e}")

            return None

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
                "in_position":    self._position_side is not None,
                "side":           self._position_side,
                "size":           self._position_size,
                "entry_price":    self._entry_price,
                "bars_held":      self._entry_bar,
                "entry_time":     self._entry_time,
                "sl_order":       self._sl_order_id,
                "tp_order":       self._tp_order_id,
                "entry_order":    self._entry_order_id,
            }

    # ─── EMERGENCY ────────────────────────────────────────────────────────────

    def emergency_flatten(self) -> Dict:
        """Cancel all orders and close all positions immediately."""
        results = []

        # Cancel all open orders
        try:
            open_orders = self._api.get_open_orders(symbol=self._symbol)
            if open_orders.get("success"):
                for o in open_orders.get("result", []):
                    oid = o.get("order_id") or o.get("id")
                    if oid:
                        self._api.cancel_order(str(oid))
                        results.append(f"cancelled {oid}")
        except Exception as e:
            results.append(f"cancel_error: {e}")

        # Close position
        try:
            if self._product_id:
                self._api.close_position(self._product_id)
                results.append("position_closed")
        except Exception as e:
            results.append(f"close_error: {e}")

        self._reset_position()
        return {"success": True, "actions": results}
