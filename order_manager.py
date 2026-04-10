"""
order_manager.py — Delta Exchange Order Execution Manager
==========================================================
Handles atomic order placement (bracket or separate SL/TP), position tracking,
forced exits, and order state reconciliation.

** ALL fees and PnL are sourced from the Delta Exchange API — no estimates. **
  - Entry fee: from order's `paid_commission` field
  - Exit fee: from fill's `commission` field
  - Gross PnL: from position's `realized_pnl` or computed from actual fill prices
  - Entry/exit prices: from `average_fill_price` (orders) or `price` (fills)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ─── MODULE-LEVEL HELPER ──────────────────────────────────────────────────────

def _safe_float(v, default=0.0):
    """Convert to float safely, returning default on failure."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class OrderManager:
    """
    Manages order lifecycle on Delta Exchange.

    - Entry: market or limit with bracket SL/TP
    - Exit: forced close, SL/TP hit detection, max-hold timeout
    - Reconciliation: poll position state to detect fills

    All financial figures (fees, PnL) come from the exchange API,
    never from local estimation.
    """

    def __init__(self, api, symbol: str = "BTCUSD", contract_value: float = 0.001,
                 taker_fee_pct: float = 0.05, maker_fee_pct: float = 0.02):
        self._api    = api
        self._symbol = symbol
        self._lock   = threading.RLock()
        self._contract_value = contract_value  # BTC per contract (Delta BTCUSD = 0.001)

        # Fee rates kept ONLY as last-resort fallback if API doesn't return commission.
        # All normal paths use exact API data.
        self._taker_fee_pct = taker_fee_pct
        self._maker_fee_pct = maker_fee_pct

        # Active state
        self._position_side:    Optional[str] = None   # "long" | "short" | None
        self._position_size:    int   = 0
        self._entry_price:      float = 0.0   # actual fill price from API
        self._tp_price:         float = 0.0
        self._sl_price:         float = 0.0
        self._entry_bar:        int   = 0
        self._entry_time:       float = 0.0
        self._entry_fee_usd:    float = 0.0   # exact fee from API paid_commission
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
        price:      float,     # current price (for limit offset / fallback)
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

                result_data = resp.get("result", {})
                raw = result_data.get("_raw", result_data)
                order_id = result_data.get("order_id", "")

                # ── Extract EXACT entry price from API ────────────────────────
                # average_fill_price is the actual fill price from the exchange.
                actual_entry_price = _safe_float(
                    raw.get("average_fill_price"), 0.0
                )
                if actual_entry_price <= 0:
                    actual_entry_price = price
                    logger.debug(
                        "average_fill_price not in order response — "
                        f"using passed price {price:.1f}"
                    )

                # ── Extract EXACT entry fee from API ──────────────────────────
                # paid_commission is the actual fee charged by the exchange.
                actual_entry_fee = _safe_float(
                    raw.get("paid_commission",
                            raw.get("commission")), 0.0
                )
                if actual_entry_fee <= 0:
                    # Order may not have fee yet — fetch from fills for this order.
                    actual_entry_fee = self._fetch_order_fee(order_id)

                if actual_entry_fee <= 0:
                    # Last resort: estimate (should rarely happen for market orders)
                    actual_entry_fee = self._estimate_fee(
                        actual_entry_price, size, is_taker=(order_type == "market")
                    )
                    logger.warning(
                        f"Could not get exact entry fee from API — "
                        f"using estimate: ${actual_entry_fee:.6f}"
                    )

                self._entry_order_id = order_id
                self._position_side  = side
                self._position_size  = size
                self._entry_price    = actual_entry_price
                self._tp_price       = tp_price
                self._sl_price       = sl_price
                self._entry_time     = time.time()
                self._entry_bar      = 0
                self._entry_fee_usd  = actual_entry_fee

                # Capture bracket leg IDs
                if use_bracket and order_type == "market":
                    sl_id = raw.get("bracket_stop_loss_order_id")
                    if not sl_id and isinstance(raw.get("stop_loss_order"), dict):
                        sl_id = raw["stop_loss_order"].get("order_id")
                    tp_id = raw.get("bracket_take_profit_order_id")
                    if not tp_id and isinstance(raw.get("take_profit_order"), dict):
                        tp_id = raw["take_profit_order"].get("order_id")
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
                            f"Response keys: {list(raw.keys())}"
                        )

                logger.info(
                    f"✅ ENTRY {side.upper()} {size} contracts @ ~{actual_entry_price:.1f} "
                    f"TP={tp_price:.1f} SL={sl_price:.1f} id={order_id}"
                )

                # If not bracket, place SL/TP separately
                if not use_bracket or order_type != "market":
                    self._place_sl_tp(side, size, sl_price, tp_price)

                if self._on_fill_cb:
                    self._on_fill_cb(side, size, actual_entry_price, order_id)

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
                    if self._product_id:
                        resp = self._api.close_position(self._product_id)

                # ── Get EXACT exit data from the API ──────────────────────────
                exit_price, exit_fee = 0.0, 0.0
                if resp.get("success"):
                    result_data = resp.get("result", {})
                    raw = result_data.get("_raw", result_data)
                    exit_price = _safe_float(raw.get("average_fill_price"), 0.0)
                    exit_fee = _safe_float(
                        raw.get("paid_commission", raw.get("commission")), 0.0
                    )

                # Fetch from fills API for exact data if needed
                if exit_price <= 0 or exit_fee <= 0:
                    fill_price, fill_fee, _ = self._get_exit_from_fills()
                    if fill_price > 0 and exit_price <= 0:
                        exit_price = fill_price
                    if fill_fee > 0 and exit_fee <= 0:
                        exit_fee = fill_fee

                if exit_price <= 0:
                    exit_price = current_price

                # ── Compute PnL from EXACT data ──────────────────────────────
                gross_pnl = self._fetch_realized_pnl()
                if gross_pnl is None:
                    gross_pnl = self._compute_pnl_from_prices(exit_price)

                if exit_fee <= 0:
                    exit_fee = self._fetch_latest_fill_commission()
                if exit_fee <= 0:
                    exit_fee = self._estimate_fee(
                        exit_price, self._position_size, is_taker=True
                    )
                    logger.warning(
                        f"Could not get exact exit fee from API — "
                        f"using estimate: ${exit_fee:.6f}"
                    )

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
        Detects bracket TP/SL fills and records the ACTUAL exit price + fees.
        """
        with self._lock:
            if self._position_side is None:
                return
            try:
                fill_price = float(data.get("price") or data.get("fill_price") or
                                   data.get("p") or 0)
                fill_side  = str(data.get("side") or data.get("s") or "").lower()
                fill_size  = float(data.get("size") or data.get("fill_quantity") or
                                   data.get("sz") or 0)

                if fill_price <= 0 or fill_size <= 0:
                    return

                # Exit fills are on the OPPOSITE side to our position
                expected_exit_side = "buy" if self._position_side == "short" else "sell"
                if fill_side != expected_exit_side:
                    return

                # ── EXACT fee from WS fill data ──────────────────────────────
                exit_fee = _safe_float(
                    data.get("commission") or data.get("c_val"), 0.0
                )
                if exit_fee <= 0:
                    exit_fee = self._fetch_latest_fill_commission()
                if exit_fee <= 0:
                    exit_fee = self._estimate_fee(
                        fill_price, self._position_size, is_taker=True
                    )
                    logger.warning(
                        f"WS fill missing commission — using estimate: ${exit_fee:.6f}"
                    )

                reason = self._classify_exit(fill_price)

                # ── EXACT PnL ────────────────────────────────────────────────
                gross_pnl = self._fetch_realized_pnl()
                if gross_pnl is None:
                    gross_pnl = self._compute_pnl_from_prices(fill_price)

                total_fees = self._entry_fee_usd + exit_fee
                net_pnl = gross_pnl - total_fees

                logger.info(
                    f"Position closed on exchange ({reason}) "
                    f"exit_price={fill_price:.1f} gross=${gross_pnl:+.4f} "
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
        Called when WS order state changes. Belt-and-suspenders check.
        """
        with self._lock:
            if self._position_side is None:
                return
            state = str(data.get("state", "")).lower()
            if state in ("filled", "closed", "cancelled"):
                logger.debug(f"WS order state={state} — scheduling reconcile check")

    # ─── API-SOURCED PNL & FEE HELPERS ────────────────────────────────────────

    def _fetch_order_fee(self, order_id: str) -> float:
        """Fetch actual paid commission from order details or its fills."""
        try:
            resp = self._api.get_order(order_id)
            if resp.get("success"):
                raw = resp.get("result", {}).get("_raw", resp.get("result", {}))
                fee = _safe_float(
                    raw.get("paid_commission", raw.get("commission")), 0.0
                )
                if fee > 0:
                    return fee
        except Exception as e:
            logger.debug(f"_fetch_order_fee error: {e}")

        # Try fills for this specific order
        try:
            resp = self._api.get_fills(product_id=self._product_id, page_size=10)
            if resp.get("success"):
                fills = resp.get("result", [])
                if isinstance(fills, dict):
                    fills = fills.get("result", fills.get("data", fills.get("fills", [])))
                for f in (fills if isinstance(fills, list) else []):
                    if str(f.get("order_id", "")) == str(order_id):
                        meta = f.get("meta_data", {}) or {}
                        total_comm = _safe_float(
                            meta.get("total_commission_in_settling_asset"), 0.0
                        )
                        if total_comm > 0:
                            return total_comm
                        fee = _safe_float(f.get("commission"), 0.0)
                        if fee > 0:
                            return fee
        except Exception as e:
            logger.debug(f"_fetch_order_fee fills error: {e}")

        return 0.0

    def _fetch_latest_fill_commission(self) -> float:
        """Fetch commission from the most recent fill for our product."""
        try:
            resp = self._api.get_fills(product_id=self._product_id, page_size=5)
            if resp.get("success"):
                fills = resp.get("result", [])
                if isinstance(fills, dict):
                    fills = fills.get("result", fills.get("data", fills.get("fills", [])))
                if isinstance(fills, list) and fills:
                    latest = fills[0]
                    meta = latest.get("meta_data", {}) or {}
                    total_comm = _safe_float(
                        meta.get("total_commission_in_settling_asset"), 0.0
                    )
                    if total_comm > 0:
                        return total_comm
                    return _safe_float(latest.get("commission"), 0.0)
        except Exception as e:
            logger.debug(f"_fetch_latest_fill_commission error: {e}")
        return 0.0

    def _fetch_realized_pnl(self) -> Optional[float]:
        """
        Fetch realized_pnl from position API.
        Returns None if unavailable (position already gone or API error).
        """
        try:
            resp = self._api.get_positions(product_id=self._product_id)
            if resp.get("success"):
                positions = resp.get("result", [])
                if isinstance(positions, dict):
                    positions = [positions]
                for pos in (positions if isinstance(positions, list) else []):
                    if str(pos.get("product_symbol", "")).upper() == self._symbol.upper():
                        rpnl = pos.get("realized_pnl")
                        if rpnl is not None:
                            val = _safe_float(rpnl, None)
                            if val is not None:
                                logger.debug(f"Fetched realized_pnl from API: ${val:.6f}")
                                return val
        except Exception as e:
            logger.debug(f"_fetch_realized_pnl error: {e}")
        return None

    def _compute_pnl_from_prices(self, exit_price: float) -> float:
        """
        Compute gross PnL from actual entry/exit prices.
        Used as fallback when realized_pnl is not available from API.
        Both entry_price and exit_price should be actual fill prices from API.
        """
        if exit_price <= 0 or self._entry_price <= 0:
            return 0.0
        diff = exit_price - self._entry_price
        if self._position_side == "short":
            diff = -diff
        return diff * self._contract_value * self._position_size

    def _estimate_fee(self, price: float, size: int, is_taker: bool = True) -> float:
        """
        FALLBACK ONLY: Estimate fee when API doesn't return commission.
        This should rarely be needed — all normal paths use exact API data.
        """
        notional = price * self._contract_value * size
        rate = self._taker_fee_pct if is_taker else self._maker_fee_pct
        return notional * rate / 100.0

    def _classify_exit(self, exit_price: float) -> str:
        """Return 'TP_HIT', 'SL_HIT', or 'EXCHANGE_CLOSE' based on actual TP/SL levels."""
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

    def _get_exit_from_fills(self) -> tuple:
        """Fetch most recent fill from REST API to get actual exit price + commission."""
        try:
            resp = self._api.get_fills(product_id=self._product_id, page_size=5)
            if resp.get("success"):
                raw = resp.get("result", {})
                fills = raw if isinstance(raw, list) else \
                        raw.get("fills", raw.get("data", raw.get("result", [])))
                if isinstance(fills, list) and fills:
                    latest = fills[0]
                    price = _safe_float(
                        latest.get("price") or latest.get("fill_price"), 0.0
                    )
                    meta = latest.get("meta_data", {}) or {}
                    total_comm = _safe_float(
                        meta.get("total_commission_in_settling_asset"), 0.0
                    )
                    fee = total_comm if total_comm > 0 else _safe_float(
                        latest.get("commission"), 0.0
                    )
                    reason = self._classify_exit(price) if price > 0 else "EXCHANGE_CLOSE"
                    return price, fee, reason
        except Exception as e:
            logger.debug(f"_get_exit_from_fills error: {e}")
        return 0.0, 0.0, "EXCHANGE_CLOSE"

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
                        # Position closed on exchange — get exact data from fills
                        exit_price, exit_fee, reason = self._get_exit_from_fills()

                        # Try realized_pnl from the position response itself
                        raw_pos = result.get("_raw", result)
                        rpnl = raw_pos.get("realized_pnl")
                        if rpnl is not None:
                            gross_pnl = _safe_float(rpnl, 0.0)
                        else:
                            gross_pnl = self._compute_pnl_from_prices(exit_price)

                        if exit_fee <= 0:
                            exit_fee = self._estimate_fee(
                                exit_price, self._position_size, is_taker=True
                            )
                            logger.warning(
                                f"Reconcile: no exact exit fee — "
                                f"using estimate: ${exit_fee:.6f}"
                            )

                        total_fees = self._entry_fee_usd + exit_fee
                        net_pnl = gross_pnl - total_fees

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

        try:
            if self._product_id:
                self._api.close_position(self._product_id)
                results.append("position_closed")
        except Exception as e:
            results.append(f"close_error: {e}")

        self._reset_position()
        return {"success": True, "actions": results}
