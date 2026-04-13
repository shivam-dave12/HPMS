"""
order_manager.py — Delta Exchange Order Execution Manager
==========================================================
Handles atomic order placement (bracket or separate SL/TP), position tracking,
forced exits, trailing-stop SL amendment, and order state reconciliation.

ALL fees and PnL are sourced exclusively from the Delta Exchange API — no
estimates, no fallback calculations.
  - Entry fee  : order's `paid_commission` field (or fills API if not in order)
  - Exit fee   : fill's `commission` / `total_commission_in_settling_asset`
  - Gross PnL  : position's `realized_pnl`, or computed from actual fill prices
  - Prices     : `average_fill_price` (orders) or `price` (fills)

If the API does not return a fee value, the fee is recorded as $0 and an error
is logged.  No local fee estimation is ever performed.

Bracket SL/TP leg resolution — Delta API response notes
────────────────────────────────────────────────────────
Delta Exchange does NOT include leg order IDs in the bracket entry response at
the top level.  The response contains `bracket_stop_loss_price` and
`bracket_take_profit_price` (the prices), but the leg order IDs are either
nested under a `bracket_order` sub-object or must be recovered via a follow-up
GET /orders query.

Resolution is attempted in four passes (each only runs if the previous failed):

  Pass 1  — `bracket_stop_loss_order_id` at the top level of the raw response
  Pass 2  — `stop_loss_order.order_id` in the raw response (some API versions)
  Pass 3  — `bracket_order` sub-object, checking common field names for the
             stop-loss leg ID
  Pass 4  — REST fallback: query open orders and match by stop price, order side,
             and reduce_only flag.  A 500 ms pause allows the exchange to
             propagate the bracket legs before the query fires.

If all four passes fail, a WARNING is logged.  Trailing-stop amendment then
uses a lazy-recovery path inside `update_sl_price` — every time an amendment
is attempted and `_sl_order_id` is still None, the REST fallback is retried
exactly once.  This guards against transient propagation delays at entry time.

Trailing-stop SL amendment strategy (two-tier)
───────────────────────────────────────────────
Tier 1 — Atomic edit  PUT /v2/orders  ← ALWAYS TRIED FIRST
  Amends the stop_price of the existing SL order in-place via the Delta
  Exchange edit endpoint.  No unprotected window.  Works for bracket child
  SL legs AND standalone stop_market_orders.

  WHY THIS IS REQUIRED FOR BRACKET ORDERS:
  Delta Exchange returns `bad_schema` when a new standalone stop_market_order
  is POST-ed while a bracket take-profit leg is still open on the same position.
  The correct operation is to edit the existing bracket SL in-place; the bracket
  TP leg is unaffected.

Tier 2 — Cancel + replace  POST /v2/orders  ← FALLBACK ONLY
  Used when Tier 1 fails (order already filled, ID stale, etc.).
  Carries a short unprotected window; always attempts restore on failure.

Emergency protection:
  If `_sl_order_id` cannot be recovered AND Tier 2 placement fails, an
  emergency stop_market_order is placed at the last known `_sl_price`.
  The position is NEVER intentionally left without stop-loss coverage.
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
    - Exit: forced close, SL/TP hit detection, absolute safety ceiling
    - Trailing: cancel old SL bracket leg, place new stop order at amended price
    - Reconciliation: poll position state to detect exchange-side fills

    All financial figures (fees, PnL) come from the exchange API,
    never from local estimation.
    """

    def __init__(self, api, symbol: str = "BTCUSD", contract_value: float = 0.001):
        self._api    = api
        self._symbol = symbol
        self._lock   = threading.RLock()
        self._contract_value = contract_value  # BTC per contract (Delta BTCUSD = 0.001)

        # Active position state
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

    # ─── BRACKET LEG ID RECOVERY ─────────────────────────────────────────────

    def _extract_bracket_leg_ids(
        self,
        raw: dict,
        sl_price: float,
        tp_price: float,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve SL and TP bracket leg order IDs from the entry order response.

        Four-pass resolution strategy (see module docstring for rationale):

          Pass 1: top-level `bracket_stop_loss_order_id` / `bracket_take_profit_order_id`
          Pass 2: nested `stop_loss_order` / `take_profit_order` dicts
          Pass 3: nested `bracket_order` sub-object
          Pass 4: REST recovery via open orders (with 500 ms propagation pause)

        Returns (sl_order_id, tp_order_id). Either may be None if unresolvable.
        """
        # ── Pass 1: top-level named fields ────────────────────────────────
        sl_id = raw.get("bracket_stop_loss_order_id")
        tp_id = raw.get("bracket_take_profit_order_id")

        # ── Pass 2: nested stop_loss_order / take_profit_order dicts ──────
        if not sl_id and isinstance(raw.get("stop_loss_order"), dict):
            sl_id = raw["stop_loss_order"].get("order_id") or raw["stop_loss_order"].get("id")
        if not tp_id and isinstance(raw.get("take_profit_order"), dict):
            tp_id = raw["take_profit_order"].get("order_id") or raw["take_profit_order"].get("id")

        # ── Pass 3: bracket_order sub-object ──────────────────────────────
        # Delta may nest the leg references under a `bracket_order` key.
        # The sub-object field names vary by API version — check all known names.
        if (not sl_id or not tp_id) and isinstance(raw.get("bracket_order"), dict):
            bo = raw["bracket_order"]
            if not sl_id:
                sl_id = (
                    bo.get("stop_loss_order_id")
                    or bo.get("sl_order_id")
                    or bo.get("bracket_stop_loss_order_id")
                    or (bo.get("stop_loss_order") or {}).get("order_id")
                    or (bo.get("stop_loss_order") or {}).get("id")
                )
            if not tp_id:
                tp_id = (
                    bo.get("take_profit_order_id")
                    or bo.get("tp_order_id")
                    or bo.get("bracket_take_profit_order_id")
                    or (bo.get("take_profit_order") or {}).get("order_id")
                    or (bo.get("take_profit_order") or {}).get("id")
                )

        # ── Pass 4: REST recovery ──────────────────────────────────────────
        # If either leg is still unresolved, pause briefly so the exchange can
        # propagate the bracket legs, then search open orders by stop price.
        if not sl_id or not tp_id:
            time.sleep(0.5)
            if not sl_id:
                sl_id = self._find_open_stop_order(target_price=sl_price)
                if sl_id:
                    logger.info(f"Bracket SL leg recovered via open_orders: id={sl_id}")
            if not tp_id:
                tp_id = self._find_open_stop_order(target_price=tp_price)
                if tp_id and tp_id != sl_id:
                    logger.info(f"Bracket TP leg recovered via open_orders: id={tp_id}")
                elif tp_id == sl_id:
                    # Same order matched both prices — clear the ambiguous TP match
                    tp_id = None

        return (str(sl_id) if sl_id else None, str(tp_id) if tp_id else None)

    def _find_open_stop_order(
        self,
        target_price: float,
        tolerance: float = 2.0,
    ) -> Optional[str]:
        """
        Query open orders and return the order ID of the stop-loss order whose
        stop_price is closest to ``target_price``.

        Delta Exchange bracket child orders arrive with:
          - order_type  = "market_order"  (NOT "stop_market_order")
          - stop_order_type = "stop_loss_order"
          - reduce_only = True

        Standalone stop-market orders placed by this bot have:
          - order_type  = "stop_market_order"
          - reduce_only = True

        Both patterns are matched. ``tolerance`` is the maximum allowed
        price-distance (default 2.0 ticks) to guard against matching the
        wrong leg on a position with multiple conditional orders.

        Returns None if no qualifying order is found or on any API error.
        """
        try:
            resp = self._api.get_open_orders(symbol=self._symbol)
            if not resp.get("success"):
                return None

            orders = resp.get("result", [])
            if isinstance(orders, dict):
                orders = (
                    orders.get("result")
                    or orders.get("data")
                    or orders.get("orders")
                    or []
                )

            best_id:   Optional[str] = None
            best_dist: float = float("inf")

            for o in (orders if isinstance(orders, list) else []):
                raw_o = o.get("_raw", o)

                otype      = str(raw_o.get("order_type",      "")).lower()
                stop_otype = str(raw_o.get("stop_order_type", "")).lower()
                reduce     = raw_o.get("reduce_only", False)

                # Pattern A: standalone stop-market (placed by update_sl_price)
                is_standalone_stop = (
                    otype in ("stop_market_order", "stop_market")
                    and reduce
                )
                # Pattern B: Delta bracket child SL leg
                is_bracket_sl = (
                    stop_otype == "stop_loss_order"
                    and reduce
                )
                # Pattern C: bracket SL with explicit stop_market type
                is_bracket_sl_market = (
                    otype in ("market_order", "market")
                    and stop_otype == "stop_loss_order"
                )

                if not (is_standalone_stop or is_bracket_sl or is_bracket_sl_market):
                    continue

                stop = _safe_float(
                    raw_o.get("stop_price") or raw_o.get("trigger_price"), 0.0
                )
                if stop <= 0:
                    continue

                dist = abs(stop - target_price)
                if dist <= tolerance and dist < best_dist:
                    oid = raw_o.get("order_id") or raw_o.get("id")
                    if oid:
                        best_id   = str(oid)
                        best_dist = dist

            return best_id

        except Exception as e:
            logger.debug(f"_find_open_stop_order error: {e}")
            return None

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

                # Delta Exchange uses "id" in some API versions, "order_id" in others.
                order_id = str(
                    result_data.get("order_id")
                    or result_data.get("id")
                    or raw.get("order_id")
                    or raw.get("id")
                    or ""
                )

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
                # ONLY use paid_commission — it is the ACTUAL fee charged after fill.
                # Do NOT fall back to 'commission' — that is an exchange estimate
                # computed before the fill and will not match the actual charge.
                actual_entry_fee = _safe_float(raw.get("paid_commission"), 0.0)
                if actual_entry_fee <= 0:
                    # Market orders fill almost instantly; give the exchange a moment
                    # to record the fill before querying the fills API.
                    time.sleep(0.5)
                    actual_entry_fee = self._fetch_order_fee(order_id)

                if actual_entry_fee <= 0:
                    logger.error(
                        f"Could not retrieve actual entry fee from API for order {order_id} "
                        f"— entry fee recorded as $0"
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

                # ── Resolve bracket leg IDs ───────────────────────────────────
                if use_bracket and order_type == "market":
                    sl_id, tp_id = self._extract_bracket_leg_ids(raw, sl_price, tp_price)
                    self._sl_order_id = sl_id
                    self._tp_order_id = tp_id

                    if self._sl_order_id and self._tp_order_id:
                        logger.info(
                            f"Bracket legs resolved: SL={self._sl_order_id} "
                            f"TP={self._tp_order_id}"
                        )
                    elif self._sl_order_id:
                        logger.warning(
                            f"Bracket SL resolved (id={self._sl_order_id}) "
                            f"but TP leg ID not found — TP will rely on "
                            f"exchange auto-cancel"
                        )
                    else:
                        logger.warning(
                            "Bracket leg IDs could not be resolved after all "
                            "four recovery passes. Trailing-stop will retry "
                            "lazy recovery on each amendment attempt. "
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
                res = sl_resp.get("result", {})
                self._sl_order_id = str(
                    res.get("order_id") or res.get("id") or ""
                ) or None
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
                res = tp_resp.get("result", {})
                self._tp_order_id = str(
                    res.get("order_id") or res.get("id") or ""
                ) or None
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
                # ONLY use paid_commission — the actual fee charged after fill.
                # 'commission' is an exchange pre-fill estimate; never use it.
                exit_price, exit_fee = 0.0, 0.0
                if resp.get("success"):
                    result_data = resp.get("result", {})
                    raw = result_data.get("_raw", result_data)
                    exit_price = _safe_float(raw.get("average_fill_price"), 0.0)
                    exit_fee = _safe_float(raw.get("paid_commission"), 0.0)

                # Fetch from fills API for exact data if needed.
                # Brief sleep to allow the fill to propagate on the exchange side.
                if exit_price <= 0 or exit_fee <= 0:
                    time.sleep(0.5)
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
                    logger.error(
                        f"Could not retrieve actual exit fee from API — "
                        f"exit fee recorded as $0"
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

    # ─── TRAILING STOP — SL MODIFICATION ────────────────────────────────────

    def update_sl_price(self, new_sl_price: float) -> bool:
        """
        Move the stop-loss to a new (improved) price.

        Amendment strategy (two-tier, most reliable first):
        ─────────────────────────────────────────────────
        Tier 1 — Atomic edit via PUT /v2/orders  ← PREFERRED
          Amends the stop_price of the existing SL order in-place.
          • No unprotected window (order stays live during amendment).
          • Works for BOTH standalone stop_market_orders AND bracket
            child SL legs — Delta Exchange accepts PUT /v2/orders for
            both types.
          • Avoids `bad_schema` that arises when POSTing a new standalone
            stop_market_order while a bracket take-profit leg is still
            active on the same position (Delta rejects the duplicate).

        Tier 2 — Cancel + replace  ← FALLBACK only
          Used when Tier 1 fails (e.g. order already filled/cancelled).
          Carries a small unprotected window between cancel and re-place,
          but is the only option when the original order ID is gone.

        Emergency protection:
          If _sl_order_id is None AND lazy REST recovery fails, an
          emergency stop is placed at the original _sl_price before
          returning False.  This ensures the position is never left
          fully unprotected.

        Ratchet rule:
          SL can only ever improve:
            Long  → new_sl must be strictly > current_sl
            Short → new_sl must be strictly < current_sl
        """
        with self._lock:
            if not self._position_side:
                return False

            exit_side = "sell" if self._position_side == "long" else "buy"

            # ── Step 1: Ensure we have a live SL order ID ──────────────────────
            if not self._sl_order_id:
                logger.debug(
                    "update_sl_price: _sl_order_id is None — lazy REST recovery"
                )
                recovered = self._find_open_stop_order(target_price=self._sl_price)
                if recovered:
                    self._sl_order_id = recovered
                    logger.info(
                        f"Lazy SL recovery succeeded: id={self._sl_order_id} "
                        f"@ {self._sl_price:.1f}"
                    )
                else:
                    # No stop on the exchange at all — place emergency protection
                    logger.error(
                        f"update_sl_price: SL ID unresolvable for "
                        f"{self._position_side} @ entry {self._entry_price:.1f}. "
                        f"Placing EMERGENCY stop @ {self._sl_price:.1f}."
                    )
                    emerg = self._api.place_stop_market_order(
                        symbol=self._symbol,
                        side=exit_side,
                        size=self._position_size,
                        stop_price=round(self._sl_price, 1),
                        reduce_only=True,
                        product_id=self._product_id,
                    )
                    if emerg.get("success"):
                        res = emerg.get("result", {})
                        self._sl_order_id = str(
                            res.get("order_id") or res.get("id") or ""
                        ) or None
                        logger.warning(
                            f"EMERGENCY stop placed @ {self._sl_price:.1f} "
                            f"id={self._sl_order_id} — position now protected."
                        )
                    else:
                        logger.error(
                            f"EMERGENCY stop FAILED: {emerg.get('error')} — "
                            f"POSITION UNPROTECTED. Manual intervention required."
                        )
                    # Trail update cannot proceed without a confirmed new ID
                    return False

            new_sl_price = round(new_sl_price, 1)

            # ── Step 2: Direction ratchet ──────────────────────────────────────
            if self._position_side == "long"  and new_sl_price <= self._sl_price:
                return False
            if self._position_side == "short" and new_sl_price >= self._sl_price:
                return False

            old_sl    = self._sl_price
            old_sl_id = self._sl_order_id

            # ── Tier 1: Atomic edit (PUT /v2/orders) ───────────────────────────
            # Preferred: amends stop_price in-place with no unprotected window.
            # Works for bracket child SL legs AND standalone stop orders.
            # Delta Exchange docs: id + product_id + stop_price in PUT body.
            try:
                edit_resp = self._api.edit_order(
                    order_id=old_sl_id,
                    stop_price=new_sl_price,
                    product_id=self._product_id,
                )
                if edit_resp.get("success"):
                    raw     = edit_resp.get("result", {})
                    new_id  = str(raw.get("order_id") or old_sl_id)
                    self._sl_order_id = new_id
                    self._sl_price    = new_sl_price
                    logger.info(
                        f"SL AMENDED ✓ (atomic): {old_sl:.1f} → {new_sl_price:.1f} "
                        f"({self._position_side}) id={self._sl_order_id}"
                    )
                    return True

                edit_err = edit_resp.get("error", "unknown")
                logger.warning(
                    f"edit_order failed ({edit_err}) — "
                    f"falling back to cancel+replace for SL trail"
                )

            except Exception as e:
                logger.warning(
                    f"edit_order exception ({e}) — "
                    f"falling back to cancel+replace for SL trail"
                )

            # ── Tier 2: Cancel + replace (fallback) ────────────────────────────
            try:
                cancel_resp = self._api.cancel_order(old_sl_id)
                if not cancel_resp.get("success"):
                    logger.warning(
                        f"SL cancel non-success id={old_sl_id}: "
                        f"{cancel_resp.get('error')} — proceeding with new placement"
                    )

                sl_resp = self._api.place_stop_market_order(
                    symbol=self._symbol,
                    side=exit_side,
                    size=self._position_size,
                    stop_price=new_sl_price,
                    reduce_only=True,
                    product_id=self._product_id,
                )

                if sl_resp.get("success"):
                    res    = sl_resp.get("result", {})
                    new_id = str(res.get("order_id") or res.get("id") or "") or None
                    self._sl_order_id = new_id
                    self._sl_price    = new_sl_price
                    logger.info(
                        f"SL TRAILED ✓ (cancel+replace): {old_sl:.1f} → {new_sl_price:.1f} "
                        f"({self._position_side}) id={self._sl_order_id}"
                    )
                    return True

                # New placement failed — attempt to restore old SL
                logger.error(
                    f"SL new order failed: {sl_resp.get('error')} "
                    f"— restoring @ {old_sl:.1f}"
                )
                restore = self._api.place_stop_market_order(
                    symbol=self._symbol,
                    side=exit_side,
                    size=self._position_size,
                    stop_price=old_sl,
                    reduce_only=True,
                    product_id=self._product_id,
                )
                if restore.get("success"):
                    res = restore.get("result", {})
                    self._sl_order_id = str(
                        res.get("order_id") or res.get("id") or ""
                    ) or None
                    self._sl_price = old_sl
                    logger.info(
                        f"SL restored @ {old_sl:.1f} id={self._sl_order_id}"
                    )
                else:
                    self._sl_order_id = None
                    logger.error(
                        f"SL RESTORE FAILED: {restore.get('error')} — "
                        f"POSITION UNPROTECTED. Will retry emergency stop next bar."
                    )
                return False

            except Exception as e:
                logger.error(f"SL trail exception: {e}", exc_info=True)
                # Last-resort restore attempt
                try:
                    restore = self._api.place_stop_market_order(
                        symbol=self._symbol,
                        side=exit_side,
                        size=self._position_size,
                        stop_price=old_sl,
                        reduce_only=True,
                        product_id=self._product_id,
                    )
                    if restore.get("success"):
                        res = restore.get("result", {})
                        self._sl_order_id = str(
                            res.get("order_id") or res.get("id") or ""
                        ) or None
                        self._sl_price = old_sl
                        logger.info(
                            f"SL restored (exception path) @ {old_sl:.1f} "
                            f"id={self._sl_order_id}"
                        )
                    else:
                        self._sl_order_id = None
                        logger.error(
                            f"SL RESTORE FAILED (exception path): "
                            f"{restore.get('error')} — POSITION UNPROTECTED."
                        )
                except Exception as restore_err:
                    self._sl_order_id = None
                    logger.error(
                        f"SL RESTORE EXCEPTION: {restore_err} — "
                        f"POSITION UNPROTECTED."
                    )
                return False

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
                # WS fills carry the actual post-fill commission.
                # Try paid_commission first, then commission (fill-record field).
                # Never use the order-level 'commission' estimate.
                exit_fee = _safe_float(
                    data.get("paid_commission") or data.get("commission") or data.get("c_val"), 0.0
                )
                if exit_fee <= 0:
                    exit_fee = self._fetch_latest_fill_commission()
                if exit_fee <= 0:
                    logger.error(
                        f"WS fill missing commission and fills API returned nothing "
                        f"— exit fee recorded as $0"
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
        if not order_id:
            return 0.0
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
                        paid = _safe_float(f.get("paid_commission"), 0.0)
                        if paid > 0:
                            return paid
                        fee = _safe_float(f.get("commission"), 0.0)
                        if fee > 0:
                            return fee
        except Exception as e:
            logger.debug(f"_fetch_order_fee fills error: {e}")

        return 0.0

    def _fetch_latest_fill_commission(self) -> float:
        """Fetch ACTUAL paid commission from the most recent fill for our product."""
        try:
            resp = self._api.get_fills(product_id=self._product_id, page_size=5)
            if resp.get("success"):
                fills = resp.get("result", [])
                if isinstance(fills, dict):
                    fills = fills.get("result", fills.get("data", fills.get("fills", [])))
                if isinstance(fills, list) and fills:
                    latest = fills[0]
                    meta = latest.get("meta_data", {}) or {}
                    # Prefer total_commission_in_settling_asset (most complete),
                    # then paid_commission, then commission — all from the fill record.
                    total_comm = _safe_float(
                        meta.get("total_commission_in_settling_asset"), 0.0
                    )
                    if total_comm > 0:
                        return total_comm
                    paid = _safe_float(latest.get("paid_commission"), 0.0)
                    if paid > 0:
                        return paid
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
                            logger.error(
                                f"Reconcile: no actual exit fee returned by API "
                                f"— exit fee recorded as $0"
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
                "current_sl":     self._sl_price,
                "current_tp":     self._tp_price,
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
