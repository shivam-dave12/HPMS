"""
delta_api.py — Delta Exchange Futures REST API Plugin
======================================================
Production-grade REST API client for Delta Exchange.

Covers ALL Delta Exchange Futures API endpoints:
  - Order management (place, cancel, replace, batch, brackets)
  - Position management (fetch, close, margin adjust, mode change)
  - Account / wallet (balances, transaction history, portfolio margin)
  - Market data (products, tickers, orderbook, recent trades, candles, funding)
  - Risk management (leverage, set/get margin mode)
  - Miscellaneous (server time, assets, settlement history)

Authentication: HMAC-SHA256 over (method + timestamp + path + query + body)
Base URL: https://api.delta.exchange (can be overridden for testnet)

Drop-in replacement for futures_api.py. Keeps the same public method names
(place_order, cancel_order, get_positions, get_balance, get_klines, …)
so order_manager.py, data_manager.py, and risk_manager.py need only
minimal config changes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DELTA_LIVE_URL    = "https://api.india.delta.exchange"   # Delta India (delta.exchange)
DELTA_TESTNET_URL = "https://cdn-ind.testnet.deltaex.org" # Delta India Testnet (demo.delta.exchange)

# NOTE: https://api.delta.exchange is Delta GLOBAL — a completely separate platform
# with separate accounts and API keys. Do NOT use that URL if your account is on
# delta.exchange (India). Using the wrong URL will cause invalid_api_key errors.

# Delta order statuses
_OPEN_STATUSES   = {"open", "pending"}
_FILLED_STATUSES = {"closed", "filled"}
_CANCELLED_STATUSES = {"cancelled"}

# Candle resolution map: minutes → resolution string accepted by Delta
# Official Delta API resolution strings: must use "1m","5m","15m" etc (NOT "1","5","15")
# Confirmed from API docs: /v2/history/candles?resolution=5m&symbol=BTCUSD&start=...&end=...
_RESOLUTION_MAP: Dict[int, str] = {
    1:     "1m",
    3:     "3m",
    5:     "5m",
    15:    "15m",
    30:    "30m",
    60:    "1h",
    120:   "2h",
    240:   "4h",
    360:   "6h",
    720:   "12h",
    1440:  "1d",
    10080: "1w",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ts_s() -> int:
    """Current UTC timestamp in SECONDS (Delta Exchange requires seconds)."""
    return int(time.time())


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DeltaAPI:
    """
    Delta Exchange Futures REST API client.

    Constructor
    -----------
    api_key    : str  — Delta API key (or env DELTA_API_KEY)
    secret_key : str  — Delta API secret (or env DELTA_SECRET_KEY)
    testnet    : bool — Use testnet endpoint (default False)
    timeout    : int  — HTTP request timeout seconds (default 10)

    All public methods return a plain dict:
      • success: {"success": True,  "result": <payload>, "error": None}
      • failure: {"success": False, "result": None, "error": "<msg>",
                  "status_code": <int>}

    The "_raw" variants return the full response dict unchanged.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet:    bool          = False,
        timeout:    int           = 10,  # was 30s — 30s × retries = multi-min main-thread freeze
    ):
        self.api_key    = api_key    or os.getenv("DELTA_API_KEY",    "")
        self.secret_key = secret_key or os.getenv("DELTA_SECRET_KEY", "")
        self.base_url   = DELTA_TESTNET_URL if testnet else DELTA_LIVE_URL
        self.timeout    = timeout
        self._session   = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # Clock skew compensation (in SECONDS — Delta timestamps are seconds).
        # Delta's tolerance: ±5 seconds from server time.
        # Only needed if your system clock is badly out of sync.
        # Calibrated automatically on first expired_signature error.
        self._time_offset_s:    int  = 0
        self._clock_calibrated: bool = False
        self._product_id_cache: Dict[str, int] = {}

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Delta API credentials required. "
                "Set DELTA_API_KEY and DELTA_SECRET_KEY in .env or pass explicitly."
            )
        logger.info(
            f"DeltaAPI initialized — endpoint: {self.base_url} "
            f"(testnet={testnet})"
        )

    # =========================================================================
    # AUTH
    # =========================================================================

    def _sign(
        self,
        method:   str,
        path:     str,
        query:    str = "",
        body_str: str = "",
    ) -> Tuple[str, str]:
        """
        Generate HMAC-SHA256 signature per Delta Exchange official docs.

        Official signature string:
          method + timestamp + path + query_string + payload

        Where:
          - timestamp  = str(int(time.time()))  — SECONDS, not milliseconds
          - query_string includes the '?' prefix when present: '?product_id=1&state=open'
          - payload    = JSON body string for POST/DELETE, empty string for GET

        Reference: https://docs.delta.exchange/#signing-a-message
        """
        ts = str(_ts_s() - self._time_offset_s)

        # Build the pre-hash string exactly as documented
        sig_str = method.upper() + ts + path
        if query:
            sig_str += "?" + query     # '?' prefix required per official sample
        sig_str += body_str

        sig = hmac.new(
            self.secret_key.encode("utf-8"),
            sig_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig, ts

    def _calibrate_clock(self, error_body: Dict) -> bool:
        """
        Extract server_time from a Delta expired_signature error and compute
        the local clock offset in seconds.

        Delta error body:
          {"error": {"code": "expired_signature",
                     "context": {"request_time": <seconds_or_ms>, "server_time": <seconds>}}}

        The docs say timestamp = int(time.time()) i.e. seconds.
        server_time in the error context is always seconds.
        request_time may be whatever we sent (could be seconds or ms if old bug).

        Strategy: use server_time directly as the reference and compare to
        current local time in seconds to get the offset. This is the cleanest
        approach since we know server_time is always seconds.
        """
        try:
            ctx = (
                error_body.get("error", {}).get("context", {})
                if isinstance(error_body, dict) else {}
            )
            server_time_s = ctx.get("server_time")
            if server_time_s is None:
                logger.debug(f"Clock calibration: missing server_time in context: {ctx}")
                return False

            server_s = int(server_time_s)
            local_s  = _ts_s()
            offset   = local_s - server_s   # positive = local is ahead in seconds

            if abs(offset) > 1:
                self._time_offset_s    = offset
                self._clock_calibrated = True
                logger.warning(
                    f"⏱️  Clock skew: local={local_s}s server={server_s}s "
                    f"→ offset={offset:+d}s. "
                    f"All future signatures will subtract {offset}s from local time."
                )
                return True
            else:
                logger.info(f"Clock skew {offset}s is within tolerance — no offset applied")
        except Exception as e:
            logger.debug(f"Clock calibration error: {e}")
        return False

    def _headers(self, signature: str, timestamp: str) -> Dict[str, str]:
        return {
            "api-key":   self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "User-Agent": "DeltaBot/1.0",
        }

    # =========================================================================
    # HTTP CORE
    # =========================================================================

    def _request(
        self,
        method:  str,
        path:    str,
        params:  Optional[Dict] = None,
        body:    Optional[Dict] = None,
        _retry_clock: bool = True,
    ) -> Dict:
        """
        Execute an authenticated HTTP request.

        Returns a normalised dict:
          {"success": bool, "result": Any, "error": str|None, "status_code": int}

        Auto-calibrates clock on first expired_signature 401 and retries once.
        """
        params   = params or {}
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        query    = urllib.parse.urlencode(params, doseq=True) if params else ""

        sig, ts = self._sign(method, path, query, body_str)
        headers = {
            **self._headers(sig, ts),
            "Content-Type": "application/json",
        }

        url = self.base_url + path
        if query:
            url += "?" + query

        try:
            resp = self._session.request(
                method  = method.upper(),
                url     = url,
                headers = headers,
                data    = body_str if body_str else None,
                timeout = self.timeout,
            )

            if resp.status_code == 429:
                logger.warning("⚠️ Delta 429 rate limit hit — consider slowing down")

            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text}

            if resp.ok:
                if isinstance(data, dict) and "result" in data:
                    return {
                        "success":     True,
                        "result":      data["result"],
                        "error":       None,
                        "status_code": resp.status_code,
                    }
                return {
                    "success":     True,
                    "result":      data,
                    "error":       None,
                    "status_code": resp.status_code,
                }

            # ── 401 expired_signature — calibrate clock and retry once ───────
            if resp.status_code == 401 and _retry_clock:
                err_code = ""
                try:
                    err_code = data.get("error", {}).get("code", "") if isinstance(data, dict) else ""
                except Exception:
                    pass
                if err_code == "expired_signature":
                    calibrated = self._calibrate_clock(data)
                    if calibrated:
                        logger.info(f"Retrying {method} {path} with corrected timestamp…")
                        return self._request(method, path, params=params, body=body, _retry_clock=False)

            # HTTP error
            error_msg = ""
            if isinstance(data, dict):
                err_obj = data.get("error", {})
                if isinstance(err_obj, dict):
                    error_msg = err_obj.get("message", "") or err_obj.get("code", "")
                error_msg = error_msg or data.get("message", "") or str(data)
            logger.error(
                f"Delta API error {resp.status_code} [{method} {path}]: {error_msg or data}"
            )
            return {
                "success":     False,
                "result":      None,
                "error":       error_msg or f"HTTP {resp.status_code}",
                "status_code": resp.status_code,
            }

        except requests.exceptions.Timeout:
            logger.error(f"Delta API timeout [{method} {path}]")
            return {"success": False, "result": None, "error": "timeout", "status_code": 0}
        except requests.exceptions.RequestException as e:
            logger.error(f"Delta API request error [{method} {path}]: {e}")
            return {"success": False, "result": None, "error": str(e), "status_code": 0}

    # Convenience wrappers
    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: Optional[Dict] = None) -> Dict:
        return self._request("POST", path, body=body)

    def _put(self, path: str, body: Optional[Dict] = None) -> Dict:
        return self._request("PUT", path, body=body)

    def _delete(self, path: str, body: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        return self._request("DELETE", path, params=params, body=body)

    # =========================================================================
    # SERVER / MISC
    # =========================================================================

    def get_server_time(self) -> Dict:
        """
        Get Delta Exchange server time.

        Delta India does not expose a dedicated /v2/time endpoint.
        Server time is read from the ticker's 'timestamp' field which is in
        MICROSECONDS (confirmed from live data: 1773718411081120 µs = 1773718411 s).

        Note: the official docs show a simplified example (1609459200) which looks
        like seconds — but live data shows 16-digit values confirming microseconds.
        """
        for sym in ("BTCUSD", "ETHUSD"):
            try:
                resp = self._session.get(
                    f"{self.base_url}/v2/tickers/{sym}",
                    timeout=5,
                )
                if resp.ok:
                    data   = resp.json()
                    # Guard: data["result"] may be None even when key exists
                    _r     = data.get("result") if isinstance(data, dict) else None
                    result = _r if isinstance(_r, dict) else (data if isinstance(data, dict) else {})
                    ts_raw = result.get("timestamp")
                    if ts_raw:
                        ts_us = int(ts_raw)      # microseconds
                        ts_s  = ts_us // 1_000_000
                        ts_ms = ts_us // 1_000
                        return {
                            "success":     True,
                            "result": {
                                "server_time_us": ts_us,
                                "server_time_ms": ts_ms,
                                "server_time_s":  ts_s,
                            },
                            "error":       None,
                            "status_code": 200,
                        }
            except Exception:
                continue

        return {
            "success":     False,
            "result":      None,
            "error":       "No server time available (ticker unreachable)",
            "status_code": 0,
        }

    def get_assets(self) -> Dict:
        """GET /v2/assets — list all assets (no auth required)."""
        return self._get("/v2/assets")

    def get_products(self, contract_types: Optional[List[str]] = None) -> Dict:
        """
        GET /v2/products — all tradeable products.

        contract_types: e.g. ["perpetual_futures", "futures", "options"]
        """
        params: Dict[str, Any] = {}
        if contract_types:
            params["contract_types"] = ",".join(contract_types)
        return self._get("/v2/products", params=params)

    def get_product(self, symbol: str) -> Dict:
        """GET /v2/products/{symbol} — single product spec."""
        return self._get(f"/v2/products/{symbol}")

    # =========================================================================
    # MARKET DATA — PUBLIC (no auth required)
    # =========================================================================

    def get_ticker(self, symbol: str) -> Dict:
        """
        GET /v2/tickers/{symbol} — best bid/ask, last price, OI, funding rate.

        Also auto-caches product_id from the response to avoid separate
        /v2/products calls in order placement.
        """
        resp = self._get(f"/v2/tickers/{symbol}")
        if resp["success"] and isinstance(resp.get("result"), dict):
            raw = resp["result"]
            pid = raw.get("product_id")
            if pid and symbol.upper() not in self._product_id_cache:
                self._product_id_cache[symbol.upper()] = int(pid)
        return resp

    def get_tickers(self, contract_types: Optional[List[str]] = None) -> Dict:
        """GET /v2/tickers — all tickers optionally filtered by contract type."""
        params: Dict[str, Any] = {}
        if contract_types:
            params["contract_types"] = ",".join(contract_types)
        return self._get("/v2/tickers", params=params)

    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """
        GET /v2/l2orderbook/{symbol} — Level 2 orderbook.

        Returns normalised {"bids": [[price, size], ...], "asks": [...]}
        compatible with the rest of the bot's orderbook interface.
        """
        params: Dict[str, Any] = {"depth": depth}
        resp = self._get(f"/v2/l2orderbook/{symbol}", params=params)
        if not resp["success"]:
            return resp

        raw = resp["result"]
        bids_raw = raw.get("buy", [])
        asks_raw = raw.get("sell", [])

        def _normalise(levels: List[Dict]) -> List[List]:
            out = []
            for lvl in levels:
                try:
                    out.append([
                        str(lvl.get("price", lvl.get("limit_price", 0))),
                        str(lvl.get("size", lvl.get("depth", 0))),
                    ])
                except Exception:
                    pass
            return out

        resp["result"] = {
            "bids":      _normalise(bids_raw),
            "asks":      _normalise(asks_raw),
            "symbol":    symbol,
            "timestamp": _ts_s() * 1000,
        }
        return resp

    def get_recent_trades(self, symbol: str) -> Dict:
        """GET /v2/trades/{symbol} — recent public trades."""
        return self._get(f"/v2/trades/{symbol}")

    def get_candles(
        self,
        symbol:     str,
        resolution: int = 1,
        start_time: Optional[int] = None,
        end_time:   Optional[int] = None,
        limit:      int = 200,
    ) -> Dict:
        """
        GET /v2/history/candles — OHLCV candle history.

        resolution: candle size in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360,
                    720, 1440, 10080)
        start_time / end_time: Unix MILLISECONDS — this function converts to
                    seconds internally (Delta API requires seconds). Callers
                    must pass milliseconds so the interface is consistent with
                    every other timestamp in the bot.
        limit:      max bars to return (default 200, max 2000 per request)

        Returns normalised list of dicts:
          [{"t": <ms>, "o": float, "h": float, "l": float,
            "c": float, "v": float, "close": float}, ...]
        """
        res_str = _RESOLUTION_MAP.get(resolution, str(resolution))

        # start and end are REQUIRED by Delta API (bad_schema if omitted).
        # Compute defaults: fetch `limit` bars ending now.
        now_s = int(time.time())
        _resolution_seconds = resolution * 60
        _default_start = now_s - limit * _resolution_seconds

        # BUG-API: old code used "if start_time" which is falsy for 0 (epoch).
        # Use explicit None check so callers can safely pass 0 if needed.
        params: Dict[str, Any] = {
            "symbol":     symbol,
            "resolution": res_str,
            "start":      int(start_time / 1000) if start_time is not None else _default_start,
            "end":        int(end_time   / 1000) if end_time   is not None else now_s,
        }

        resp = self._get("/v2/history/candles", params=params)
        if not resp["success"]:
            return resp

        raw = resp.get("result", []) or []
        candles: List[Dict] = []
        for c in raw:
            try:
                candles.append({
                    "t": int(c["time"]) * 1000,          # s → ms
                    "o": _safe_float(c.get("open")),
                    "h": _safe_float(c.get("high")),
                    "l": _safe_float(c.get("low")),
                    "c": _safe_float(c.get("close")),
                    "v": _safe_float(c.get("volume")),
                })
            except Exception:
                pass

        resp["result"] = candles
        return resp

    def get_funding_rate(self, symbol: str) -> Dict:
        """GET /v2/funding_rate/{symbol} — current and predicted funding rate."""
        return self._get(f"/v2/funding_rate/{symbol}")

    def get_mark_price(self, symbol: str) -> Dict:
        """GET /v2/mark_price/{symbol} — current mark price."""
        return self._get(f"/v2/mark_price/{symbol}")

    def get_index_price(self, symbol: str) -> Dict:
        """GET /v2/index_price/{symbol} — current index price."""
        return self._get(f"/v2/index_price/{symbol}")

    def get_settlement_history(self, symbol: str, page_size: int = 50) -> Dict:
        """GET /v2/settlement_history/{symbol} — historical settlements."""
        return self._get(
            f"/v2/settlement_history/{symbol}",
            params={"page_size": page_size},
        )

    def get_spot_price(self, symbol: str) -> Dict:
        """GET /v2/spot_price/{symbol} — underlying spot price."""
        return self._get(f"/v2/spot_price/{symbol}")

    # =========================================================================
    # ACCOUNT — WALLET & BALANCES
    # =========================================================================

    def get_wallet_balances(self, asset_id: Optional[int] = None) -> Dict:
        """
        GET /v2/wallet/balances — all wallet balances, optionally filtered.

        Returns raw Delta response; normalised balances available via get_balance().
        """
        params: Dict[str, Any] = {}
        if asset_id is not None:
            params["asset_id"] = asset_id
        return self._get("/v2/wallet/balances", params=params)

    def get_balance(self, currency: str = "USD") -> Dict:
        """
        Normalised balance fetch. Delta India perpetuals are USD-settled — use "USD" not "USDT".

        Returns:
            {"available": float, "locked": float, "currency": str}  on success
            {"available": 0.0, "locked": 0.0, "currency": str, "error": str}  on failure

        NOTE: If you receive available=$0.00 and you have funds, this means
        authentication failed (clock skew or wrong credentials).
        Check logs for ⏱️ clock calibration messages and 401 errors.
        """
        result = {"available": 0.0, "locked": 0.0, "currency": currency}

        try:
            resp = self.get_wallet_balances()

            # Surface auth failures clearly
            if not resp["success"]:
                sc  = resp.get("status_code", 0)
                err = resp.get("error", "unknown")
                if sc == 401:
                    logger.error(
                        f"❌ Balance auth failed (401: {err}). "
                        f"Check DELTA_API_KEY / DELTA_SECRET_KEY and clock sync. "
                        f"Clock offset: {self._time_offset_s}s "
                        f"(calibrated={self._clock_calibrated})"
                    )
                else:
                    logger.error(f"❌ Balance fetch failed ({sc}): {err}")
                return {**result, "error": f"auth_failed: {err}"}

            balances = resp.get("result", [])
            if isinstance(balances, dict):
                balances = [balances]

            for b in balances:
                asset_sym = (
                    b.get("asset_symbol", "")
                    or b.get("currency", "")
                    or b.get("asset", "")
                ).upper()
                if asset_sym == currency.upper():
                    available = float(
                        b.get("available_balance",
                        b.get("available_balance_for_new_orders",
                        b.get("available", 0))) or 0
                    )
                    total_bal = float(
                        b.get("balance",
                        b.get("total_balance", available)) or 0
                    )
                    locked = max(0.0, total_bal - available)
                    logger.info(
                        f"✅ Balance {currency}: available=${available:.2f} "
                        f"locked=${locked:.2f} total=${total_bal:.2f}"
                    )
                    return {
                        "available": available,
                        "locked":    locked,
                        "currency":  currency,
                    }

            # Currency not found in balances — list what we got
            found = [
                (b.get("asset_symbol") or b.get("currency", "?")).upper()
                for b in balances
            ]
            logger.warning(
                f"⚠️  {currency} not found in wallet balances. "
                f"Found: {found}. "
                f"Total balance entries: {len(balances)}"
            )
            return {**result, "error": f"{currency} not found — found: {found}"}

        except Exception as e:
            logger.error(f"Balance exception: {e}", exc_info=True)
            return {**result, "error": f"Exception: {e}"}

    def get_transaction_history(
        self,
        asset_id:         Optional[int]  = None,
        transaction_type: Optional[str]  = None,
        page_size:        int            = 50,
        after:            Optional[str]  = None,
        before:           Optional[str]  = None,
    ) -> Dict:
        """
        GET /v2/wallet/transactions — funding, commissions, realised PnL, etc.

        transaction_type options: commission, pnl, deposit, withdrawal,
                                  realized_pnl, transfer, funding, liquidation
        """
        params: Dict[str, Any] = {"page_size": page_size}
        if asset_id is not None:
            params["asset_id"] = asset_id
        if transaction_type:
            params["transaction_type"] = transaction_type
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return self._get("/v2/wallet/transactions", params=params)

    def get_portfolio_margin(self, body: Optional[Dict] = None) -> Dict:
        """
        GET /v2/wallet/portfolio_margin — portfolio margin details.
        Optionally pass body to simulate a new position's margin impact.
        """
        if body:
            return self._post("/v2/wallet/portfolio_margin", body=body)
        return self._get("/v2/wallet/portfolio_margin")

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def get_positions(
        self,
        product_id:     Optional[int] = None,
        product_symbol: Optional[str] = None,
        state:          Optional[str] = None,
    ) -> Dict:
        """
        GET /v2/positions/margined — all open margined positions.

        state: "open" | "closed"
        """
        params: Dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        if product_symbol:
            params["product_symbol"] = product_symbol
        if state:
            params["product_contract_type"] = state
        return self._get("/v2/positions/margined", params=params)

    def get_position(self, symbol: str) -> Dict:
        """
        Fetch a single position by symbol. Compatible with existing bot interface.

        Returns normalised dict:
          {"side": "long"|"short", "size": float, "entry_price": float,
           "unrealized_pnl": float, "leverage": float, "margin": float}
        """
        resp = self.get_positions(product_symbol=symbol)
        if not resp["success"]:
            return resp

        positions = resp.get("result", [])
        if isinstance(positions, dict):
            positions = [positions]

        for pos in positions:
            if str(pos.get("product_symbol", "")).upper() == symbol.upper():
                size = _safe_float(pos.get("size", 0))
                if size == 0:
                    continue
                side_raw = str(pos.get("side", "")).lower()
                side = "long" if side_raw == "buy" else "short"
                return {
                    "success":       True,
                    "result":        {
                        "side":            side,
                        "size":            abs(size),
                        "entry_price":     _safe_float(pos.get("entry_price")),
                        "unrealized_pnl":  _safe_float(pos.get("unrealized_pnl")),
                        "leverage":        _safe_float(pos.get("leverage")),
                        "margin":          _safe_float(pos.get("margin")),
                        "symbol":          symbol,
                        "liquidation_price": _safe_float(pos.get("liquidation_price")),
                        "product_id":      pos.get("product_id"),
                        "_raw":            pos,
                    },
                    "error":         None,
                    "status_code":   200,
                }

        # No open position found
        return {
            "success": True,
            "result":  {"size": 0.0, "side": None, "symbol": symbol},
            "error":   None,
            "status_code": 200,
        }

    def close_position(self, product_id: int, size: Optional[float] = None) -> Dict:
        """
        POST /v2/positions/close — close an open position (whole or partial).
        """
        body: Dict[str, Any] = {"product_id": product_id}
        if size is not None:
            body["size"] = size
        return self._post("/v2/positions/close", body=body)

    def add_margin(self, product_id: int, delta_margin: float) -> Dict:
        """
        POST /v2/positions/change_margin — add or reduce position margin.

        delta_margin: positive = add, negative = reduce
        """
        body = {"product_id": product_id, "delta_margin": str(delta_margin)}
        return self._post("/v2/positions/change_margin", body=body)

    def set_auto_topup(self, product_id: int, enabled: bool) -> Dict:
        """
        PUT /v2/positions/auto_topup — enable/disable auto margin top-up.
        """
        body = {"product_id": product_id, "auto_topup": enabled}
        return self._put("/v2/positions/auto_topup", body=body)

    # =========================================================================
    # LEVERAGE
    # =========================================================================

    def set_leverage(
        self,
        product_id:     Optional[int] = None,
        symbol:         Optional[str] = None,
        leverage:       float         = 1.0,
        exchange:       str           = "",   # kept for interface compat, unused
    ) -> Dict:
        """
        POST /v2/products/{product_id}/orders/leverage
        product_id is a PATH parameter per official docs — NOT a body field.
        """
        _pid = product_id
        if _pid is None and symbol:
            _pid = self._symbol_to_product_id(symbol)
        if _pid is None:
            return {"success": False, "result": None,
                    "error": f"product_id not found for symbol={symbol}", "status_code": 0}
        # API doc requires integer, NOT a string — "10" causes bad_schema error
        return self._post(f"/v2/products/{_pid}/orders/leverage", body={"leverage": int(leverage)})

    def get_leverage(self, product_id: int) -> Dict:
        """GET /v2/products/{id}/orders/leverage — current leverage for a product."""
        return self._get(f"/v2/products/{product_id}/orders/leverage")

    # =========================================================================
    # MARGIN MODE
    # =========================================================================

    def set_margin_mode(
        self,
        product_id: int,
        margin_mode: str,   # "isolated" | "cross"
    ) -> Dict:
        """
        POST /v2/products/margin_mode — switch between isolated and cross margin.
        """
        # Official docs: PUT /v2/users/margin_mode — no product_id in body
        return self._put("/v2/users/margin_mode", body={"margin_mode": margin_mode})

    # =========================================================================
    # ORDERS — PLACE
    # =========================================================================

    def place_order(
        self,
        symbol:         str,
        side:           str,           # "buy" | "sell"
        order_type:     str,           # "limit_order" | "market_order" | "stop_market_order" |
                                       # "stop_limit_order" | "take_profit_market_order" | "take_profit_limit_order"
        size:           float,
        limit_price:    Optional[float]   = None,
        stop_price:     Optional[float]   = None,
        reduce_only:    bool              = False,
        post_only:      bool              = False,
        time_in_force:  str               = "gtc",  # gtc | ioc | fok | good_till_cancelled
        client_order_id: Optional[str]    = None,
        bracket_stop_loss_price:    Optional[float] = None,
        bracket_take_profit_price:  Optional[float] = None,
        trailing_stop_delta:        Optional[float] = None,
        mmp:            bool              = False,   # market maker protection
        exchange:       str               = "",      # kept for CoinSwitch compat
        product_id:     Optional[int]     = None,
        trigger_price:  Optional[float]   = None,   # alias for stop_price
        quantity:       Optional[float]   = None,   # alias for size
        price:          Optional[float]   = None,   # alias for limit_price
        # ── Delta stop/TP conditional order fields ────────────────────────────
        # stop_order_type: "stop_loss_order" | "take_profit_order"
        # When present this field is included directly in the request body.
        # Delta Exchange uses it to distinguish SL vs TP conditional orders
        # that share the same base order_type (market_order + stop_price).
        stop_order_type: Optional[str]    = None,
        # isomorphic_slippage_check is passed through if provided (advanced)
        isomorphic_slippage_check: Optional[bool] = None,
    ) -> Dict:
        """
        POST /v2/orders — place a new order.

        Normalised interface compatible with existing bot code.
        Returns {"success": True, "result": {"order_id": ..., ...}} on success.
        """
        # Alias resolution (compat with CoinSwitch interface)
        _size  = size or quantity or 0.0
        _price = limit_price or price
        _stop  = stop_price or trigger_price

        # Resolve product_id if not provided
        _product_id = product_id
        if _product_id is None:
            _product_id = self._symbol_to_product_id(symbol)
            if _product_id is None:
                return {"success": False, "result": None,
                        "error": f"Unknown symbol {symbol}", "status_code": 0}

        # Normalise side
        _side = "buy" if str(side).lower() in ("buy", "long") else "sell"

        # Normalise order type
        _otype_map = {
            "market":              "market_order",
            "limit":               "limit_order",
            "stop_market":         "stop_market_order",
            "stop_loss_market":    "stop_market_order",
            "stop_market_order":   "stop_market_order",
            "stop_limit":          "stop_limit_order",
            "take_profit_market":  "take_profit_market_order",
            "take_profit_limit":   "take_profit_limit_order",
            "take_profit_market_order": "take_profit_market_order",
        }
        _otype = _otype_map.get(str(order_type).lower().replace(" ", "_"), order_type)

        body: Dict[str, Any] = {
            "product_id":  _product_id,
            "side":        _side,
            "order_type":  _otype,
            "size":        int(_size),       # Delta uses integer contract sizes
            "reduce_only": reduce_only,
        }

        # post_only and time_in_force are ONLY valid on limit-type orders.
        # Delta returns bad_schema if either field is present on stop_market_order
        # or take_profit_market_order — even with value False/"gtc".
        # stop_limit_order and take_profit_limit_order still accept both fields.
        _CONDITIONAL_MARKET_TYPES = {"stop_market_order", "take_profit_market_order"}
        if _otype not in _CONDITIONAL_MARKET_TYPES:
            body["post_only"]     = post_only
            body["time_in_force"] = time_in_force

        if _price is not None:
            body["limit_price"] = str(_price)
        if _stop is not None:
            body["stop_price"] = str(_stop)
        if client_order_id:
            body["client_order_id"] = client_order_id
        if bracket_stop_loss_price is not None:
            body["bracket_stop_loss_price"] = str(bracket_stop_loss_price)
        if bracket_take_profit_price is not None:
            body["bracket_take_profit_price"] = str(bracket_take_profit_price)
        if trailing_stop_delta is not None:
            body["trailing_stop_delta"] = str(trailing_stop_delta)
        if mmp:
            body["mmp"] = True
        # ── Conditional order type (stop-loss / take-profit classifier) ──────
        # Delta Exchange uses stop_order_type to differentiate SL from TP
        # when both share order_type=market_order + stop_price.
        # Values: "stop_loss_order" | "take_profit_order"
        if stop_order_type is not None:
            body["stop_order_type"] = stop_order_type
        if isomorphic_slippage_check is not None:
            body["isomorphic_slippage_check"] = isomorphic_slippage_check

        resp = self._post("/v2/orders", body=body)

        # Normalise response to match existing bot's expected keys
        if resp["success"] and resp.get("result"):
            raw = resp["result"]
            resp["result"] = {
                "order_id":   str(raw.get("id", raw.get("order_id", ""))),
                "symbol":     symbol,
                "side":       _side,
                "order_type": _otype,
                "size":       raw.get("size"),
                "limit_price": raw.get("limit_price"),
                "stop_price":  raw.get("stop_price"),
                "status":     raw.get("state", raw.get("status", "")),
                "_raw":       raw,
            }
        return resp

    def place_market_order(
        self,
        symbol:      str,
        side:        str,
        size:        float,
        reduce_only: bool = False,
        exchange:    str  = "",
        product_id:  Optional[int] = None,
    ) -> Dict:
        """Convenience: place a market order."""
        return self.place_order(
            symbol      = symbol,
            side        = side,
            order_type  = "market",
            size        = size,
            reduce_only = reduce_only,
            product_id  = product_id,
        )

    def place_limit_order(
        self,
        symbol:        str,
        side:          str,
        size:          float,
        limit_price:   float,
        post_only:     bool  = True,
        reduce_only:   bool  = False,
        time_in_force: str   = "gtc",
        exchange:      str   = "",
        product_id:    Optional[int] = None,
    ) -> Dict:
        """Convenience: place a passive limit order (post-only by default)."""
        return self.place_order(
            symbol        = symbol,
            side          = side,
            order_type    = "limit",
            size          = size,
            limit_price   = limit_price,
            post_only     = post_only,
            reduce_only   = reduce_only,
            time_in_force = time_in_force,
            product_id    = product_id,
        )

    def place_stop_market_order(
        self,
        symbol:      str,
        side:        str,
        size:        float,
        stop_price:  float,
        reduce_only: bool = True,
        product_id:  Optional[int] = None,
    ) -> Dict:
        """Convenience: place a stop-market order (for SL)."""
        return self.place_order(
            symbol     = symbol,
            side       = side,
            order_type = "stop_market",
            size       = size,
            stop_price = stop_price,
            reduce_only = reduce_only,
            product_id = product_id,
        )

    def place_take_profit_market_order(
        self,
        symbol:      str,
        side:        str,
        size:        float,
        stop_price:  float,
        reduce_only: bool = True,
        product_id:  Optional[int] = None,
    ) -> Dict:
        """Convenience: place a take-profit market order."""
        return self.place_order(
            symbol     = symbol,
            side       = side,
            order_type = "take_profit_market",
            size       = size,
            stop_price = stop_price,
            reduce_only = reduce_only,
            product_id = product_id,
        )

    def place_bracket_order(
        self,
        symbol:                    str,
        side:                      str,
        size:                      float,
        order_type:                str   = "market",
        limit_price:               Optional[float] = None,
        bracket_stop_loss_price:   Optional[float] = None,
        bracket_take_profit_price: Optional[float] = None,
        trailing_stop_delta:       Optional[float] = None,
        product_id:                Optional[int]   = None,
    ) -> Dict:
        """
        Bracket order: entry + automatic SL + TP in one request.

        Delta supports native bracket orders — more efficient than
        placing three separate orders.
        """
        return self.place_order(
            symbol                    = symbol,
            side                      = side,
            order_type                = order_type,
            size                      = size,
            limit_price               = limit_price,
            bracket_stop_loss_price   = bracket_stop_loss_price,
            bracket_take_profit_price = bracket_take_profit_price,
            trailing_stop_delta       = trailing_stop_delta,
            product_id                = product_id,
        )

    def place_trailing_stop_order(
        self,
        symbol:               str,
        side:                 str,
        size:                 float,
        trailing_stop_delta:  float,
        reduce_only:          bool = True,
        product_id:           Optional[int] = None,
    ) -> Dict:
        """
        Trailing stop order — stop price trails by trailing_stop_delta USD.
        """
        return self.place_order(
            symbol               = symbol,
            side                 = side,
            order_type           = "market",
            size                 = size,
            trailing_stop_delta  = trailing_stop_delta,
            reduce_only          = reduce_only,
            product_id           = product_id,
        )

    # =========================================================================
    # ORDERS — BATCH
    # =========================================================================

    def batch_create_orders(self, orders: List[Dict]) -> Dict:
        """
        POST /v2/orders/batch — create up to 10 orders in a single request.
        Each element in `orders` is a dict matching place_order() body params.
        """
        return self._post("/v2/orders/batch", body={"orders": orders})

    def batch_cancel_orders(self, order_ids: List[str]) -> Dict:
        """DELETE /v2/orders/batch — cancel multiple orders."""
        return self._delete("/v2/orders/batch", body={"ids": order_ids})

    def batch_edit_orders(self, edits: List[Dict]) -> Dict:
        """
        PUT /v2/orders/batch — edit multiple orders.
        Each element: {"id": <order_id>, "limit_price": ..., "size": ...}
        """
        return self._put("/v2/orders/batch", body={"orders": edits})

    # =========================================================================
    # ORDERS — QUERY & CANCEL
    # =========================================================================

    def get_order(self, order_id: str, exchange: str = "") -> Dict:
        """
        GET /v2/orders/{id} — fetch a single order.

        Compatible with existing bot interface:
            api.get_order(order_id)
        """
        resp = self._get(f"/v2/orders/{order_id}")
        if resp["success"] and resp.get("result"):
            raw = resp["result"]
            resp["result"] = {
                "order_id":   str(raw.get("id", order_id)),
                "status":     raw.get("state", raw.get("status", "")),
                "filled_qty": _safe_float(raw.get("size_filled", 0)),
                "fill_price": _safe_float(raw.get("average_fill_price", 0)),
                "size":       _safe_float(raw.get("size", 0)),
                "_raw":       raw,
            }
        return resp

    def get_open_orders(
        self,
        symbol:     Optional[str] = None,
        product_id: Optional[int] = None,
        order_type: Optional[str] = None,
        exchange:   str           = "",
    ) -> Dict:
        """
        GET /v2/orders — all open orders, optionally filtered.

        Returns normalised list compatible with order_manager.py:
          [{"order_id": str, "type": str, "trigger_price": float, ...}, ...]
        """
        params: Dict[str, Any] = {"state": "open"}
        if product_id:
            params["product_id"] = product_id
        elif symbol:
            pid = self._symbol_to_product_id(symbol)
            if pid:
                params["product_id"] = pid

        resp = self._get("/v2/orders", params=params)
        if not resp["success"]:
            return resp

        raw_orders = resp.get("result", []) or []
        if isinstance(raw_orders, dict):
            raw_orders = raw_orders.get("result", []) or []

        normalised = []
        for o in raw_orders:
            otype_raw  = str(o.get("order_type", "")).lower()
            stop_otype = str(o.get("stop_order_type", "")).lower()
            # Bracket child orders arrive with order_type="market_order" for BOTH
            # the SL and TP legs — stop_order_type is the only distinguishing field.
            # Check it first so bracket children are correctly labelled instead of
            # both being mapped to "MARKET".
            _stop_otype_map = {
                "stop_loss_order":   "STOP_MARKET",
                "take_profit_order": "TAKE_PROFIT_MARKET",
            }
            otype_map = {
                "stop_market_order":        "STOP_MARKET",
                "take_profit_market_order": "TAKE_PROFIT_MARKET",
                "stop_limit_order":         "STOP_LIMIT",
                "take_profit_limit_order":  "TAKE_PROFIT_LIMIT",
                "limit_order":              "LIMIT",
                "market_order":             "MARKET",
            }
            mapped_type = (
                _stop_otype_map.get(stop_otype)
                or otype_map.get(otype_raw, otype_raw.upper())
            )
            normalised.append({
                "order_id":      str(o.get("id", "")),
                "type":          mapped_type,
                "trigger_price": _safe_float(o.get("stop_price", 0)),
                "limit_price":   _safe_float(o.get("limit_price", 0)),
                "size":          _safe_float(o.get("size", 0)),
                "side":          str(o.get("side", "")),
                "status":        str(o.get("state", "")),
                "_raw":          o,
            })
        resp["result"] = normalised
        return resp

    def cancel_order(self, order_id: str, product_id: Optional[int] = None,
                     exchange: str = "") -> Dict:
        """
        DELETE /v2/orders — cancel a single order.

        API doc: id goes in the REQUEST BODY as "id", NOT in the URL path.
        DELETE /v2/orders/{id} returns 404 Not Found — correct endpoint has no id suffix.
        product_id is required in the body per the official API schema.

        Body: {"id": <int>, "product_id": <int>}
        """
        # Resolve product_id from cache (populated at startup via ticker call)
        _pid = product_id
        if _pid is None:
            for _sym in ("BTCUSD", "BTCUSDT", "ETHUSD"):
                _pid = self._product_id_cache.get(_sym)
                if _pid:
                    break
        if _pid is None:
            _pid = self._symbol_to_product_id("BTCUSD")

        body: Dict[str, Any] = {"id": int(order_id)}
        if _pid:
            body["product_id"] = int(_pid)

        # Endpoint: DELETE /v2/orders (NO id suffix — id is in the body)
        return self._delete("/v2/orders", body=body)

    def cancel_all_orders(
        self,
        symbol:     Optional[str] = None,
        product_id: Optional[int] = None,
        exchange:   str           = "",
    ) -> Dict:
        """
        DELETE /v2/orders/all — cancel all open orders (optionally filtered).

        Compatible with existing bot interface:
            api.cancel_all_orders(exchange=..., symbol=...)
        """
        body: Dict[str, Any] = {}
        if product_id:
            body["product_id"] = product_id
        elif symbol:
            pid = self._symbol_to_product_id(symbol)
            if pid:
                body["product_id"] = pid
        return self._delete("/v2/orders/all", body=body if body else None)

    def edit_order(
        self,
        order_id:    str,
        size:        Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price:  Optional[float] = None,
        product_id:  Optional[int]   = None,
    ) -> Dict:
        """
        PUT /v2/orders — atomically edit stop_price (and limit_price) of a stop order.

        API doc (EditOrderRequest schema):
          {"id": <int>, "product_id": <int>, "stop_price": "56000",
           "limit_price": "55000", "size": <int>, ...}

        Key facts confirmed from API doc:
          - Endpoint: PUT /v2/orders (NO id in URL — id goes in BODY)
          - product_id REQUIRED in body (without it: bad_schema)
          - stop_price editable for stop-market AND stop-limit orders
          - limit_price editable for stop-limit orders (used together with stop_price)
          - Atomic: no cancel+replace needed — no unprotected window

        Used by order_manager.replace_stop_loss for trailing SL updates.
        For stop-limit trailing SLs, both stop_price and limit_price are sent together
        to keep the limit offset in sync with the new trigger level.
        """
        # Resolve product_id from cache (populated at startup via ticker call)
        _pid = product_id
        if _pid is None:
            for _sym in ("BTCUSD", "BTCUSDT", "ETHUSD"):
                _pid = self._product_id_cache.get(_sym)
                if _pid:
                    break
        if _pid is None:
            _pid = self._symbol_to_product_id("BTCUSD")

        body: Dict[str, Any] = {"id": int(order_id)}   # id in BODY, not URL path
        if _pid:
            body["product_id"] = int(_pid)              # required — without it: bad_schema
        if size is not None:
            body["size"] = int(size)
        if limit_price is not None:
            body["limit_price"] = str(limit_price)      # for stop-limit: limit exec price
        if stop_price is not None:
            body["stop_price"] = str(stop_price)        # trigger price

        resp = self._put("/v2/orders", body=body)       # endpoint has NO id suffix
        if resp["success"] and resp.get("result"):
            raw = resp["result"]
            resp["result"] = {
                "order_id":   str(raw.get("id", order_id)),
                "status":     raw.get("state", ""),
                "_raw":       raw,
            }
        return resp

    # =========================================================================
    # ORDER HISTORY
    # =========================================================================

    def get_order_history(
        self,
        symbol:     Optional[str] = None,
        product_id: Optional[int] = None,
        page_size:  int           = 50,
        after:      Optional[str] = None,
        before:     Optional[str] = None,
    ) -> Dict:
        """GET /v2/orders/history — paginated order history."""
        params: Dict[str, Any] = {"page_size": page_size}
        if product_id:
            params["product_id"] = product_id
        elif symbol:
            pid = self._symbol_to_product_id(symbol)
            if pid:
                params["product_id"] = pid
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return self._get("/v2/orders/history", params=params)

    def get_fills(
        self,
        symbol:     Optional[str] = None,
        product_id: Optional[int] = None,
        page_size:  int           = 50,
    ) -> Dict:
        """GET /v2/fills — trade fill history."""
        params: Dict[str, Any] = {"page_size": page_size}
        if product_id:
            # API docs: parameter is 'product_ids' (plural, comma-separated)
            params["product_ids"] = str(product_id)
        elif symbol:
            pid = self._symbol_to_product_id(symbol)
            if pid:
                params["product_ids"] = str(pid)
        return self._get("/v2/fills", params=params)

    # =========================================================================
    # UTILITY — product ID cache
    # =========================================================================


    def _symbol_to_product_id(self, symbol: str) -> Optional[int]:
        """
        Resolve a symbol string to Delta's internal integer product_id.

        Strategy (fastest-first):
          1. Cache hit — return immediately
          2. Alias normalisation — BTCUSDT → BTCUSD (Delta India perpetual naming)
          3. Ticker shortcut — GET /v2/tickers/{symbol} returns product_id directly
          4. Full product scan — GET /v2/products fallback

        Results are cached in-memory for the lifetime of the instance.
        """
        # ── Normalise common aliases ──────────────────────────────────────────
        _ALIASES = {
            "BTCUSDT":  "BTCUSD",
            "ETHUSDT":  "ETHUSD",
            "SOLUSDT":  "SOLUSD",
            "BNBUSDT":  "BNBUSD",
            "XRPUSDT":  "XRPUSD",
            "ADAUSDT":  "ADAUSD",
            "DOGEUSDT": "DOGEUSD",
        }
        sym_upper = _ALIASES.get(symbol.upper(), symbol.upper())

        # ── Cache hit ─────────────────────────────────────────────────────────
        if sym_upper in self._product_id_cache:
            return self._product_id_cache[sym_upper]
        # Also check original in case it was cached under original name
        if symbol.upper() in self._product_id_cache:
            return self._product_id_cache[symbol.upper()]

        # ── Ticker shortcut (fastest — single REST call, returns product_id) ─
        ticker_resp = self.get_ticker(sym_upper)
        if ticker_resp.get("success"):
            raw = ticker_resp.get("result", {})
            pid = raw.get("product_id") if isinstance(raw, dict) else None
            if pid:
                self._product_id_cache[sym_upper] = int(pid)
                logger.info(f"product_id resolved via ticker: {sym_upper} → {pid}")
                return int(pid)

        # ── Full product scan fallback ────────────────────────────────────────
        resp = self.get_products()
        if not resp["success"]:
            logger.warning(f"Cannot resolve product_id for {symbol}: {resp.get('error')}")
            return None

        products = resp.get("result", [])
        if isinstance(products, dict):
            products = products.get("result", [])

        for p in products:
            if not isinstance(p, dict):
                continue
            p_sym  = str(p.get("symbol", "")).upper()
            p_name = str(p.get("name",   "")).upper()
            pid    = p.get("id")
            if not pid:
                continue
            # Cache every product we see so subsequent lookups are instant
            self._product_id_cache[p_sym]  = int(pid)
            self._product_id_cache[p_name] = int(pid)
            if p_sym == sym_upper or p_name == sym_upper:
                logger.info(f"product_id resolved via product scan: {sym_upper} → {pid}")
                return int(pid)

        logger.warning(f"product_id not found for symbol '{symbol}' (tried as '{sym_upper}')")
        return None

    def get_product_id(self, symbol: str) -> Optional[int]:
        """Public access to product_id resolver."""
        return self._symbol_to_product_id(symbol)

    def prefetch_product_ids(self, symbols: List[str]) -> Dict[str, int]:
        """
        Pre-populate the product_id cache for a list of symbols.
        Call once at startup to avoid repeated product-list fetches.
        Returns {symbol: product_id} for all resolved symbols.
        """
        resp = self.get_products()
        if not resp["success"]:
            logger.warning(f"prefetch_product_ids failed: {resp.get('error')}")
            return {}

        products = resp.get("result", [])
        if isinstance(products, dict):
            products = products.get("result", [])

        syms_upper = {s.upper() for s in symbols}
        resolved: Dict[str, int] = {}
        for p in products:
            sym = str(p.get("symbol", "")).upper()
            pid = p.get("id")
            if pid and sym in syms_upper:
                self._product_id_cache[sym] = int(pid)
                resolved[sym] = int(pid)

        logger.info(f"Prefetched product IDs: {resolved}")
        return resolved

    # =========================================================================
    # SELF-TEST
    # =========================================================================

    def self_test(self, symbol: str = "BTCUSD") -> bool:
        """
        Quick connectivity and auth check.
        Returns True if all checks pass, logs each failure.
        """
        ok = True

        # ── Server time ────────────────────────────────────────────────────────
        t = self.get_server_time()
        if t["success"] and t.get("result"):
            r    = t["result"]
            ts_s = r.get("server_time_s", 0)
            ts_u = r.get("server_time_us", 0)
            logger.info(f"✅ Server time: {ts_s}s (unix) — {ts_u}µs raw")
        else:
            logger.warning(f"⚠️  Server time unavailable: {t.get('error')}")

        # ── Ticker: public, also caches product_id ─────────────────────────────
        ticker = self.get_ticker(symbol)
        if ticker["success"]:
            r    = ticker.get("result") or {}   # guard against None
            pid  = r.get("product_id")
            last = r.get("close", r.get("mark_price", "?"))
            cval = r.get("contract_value", "?")
            tick = r.get("tick_size", "?")
            logger.info(
                f"✅ Ticker {symbol}: last={last}  product_id={pid}  "
                f"contract_value={cval}  tick_size={tick}"
            )
            if pid and symbol.upper() not in self._product_id_cache:
                self._product_id_cache[symbol.upper()] = int(pid)
                logger.info(f"✅ Cached product_id {pid} for {symbol}")
        else:
            logger.error(f"❌ Ticker failed: {ticker.get('error')}")
            ok = False

        # ── Balance: private, confirms auth ────────────────────────────────────
        bal = self.get_balance("USD")
        if "error" not in bal:
            logger.info(
                f"✅ Balance: available=${bal.get('available', 0):.2f}  "
                f"locked=${bal.get('locked', 0):.2f} USDT"
            )
        else:
            logger.error(f"❌ Balance failed: {bal.get('error')}")
            ok = False

        # ── Product ID ─────────────────────────────────────────────────────────
        pid_cached = self._product_id_cache.get(symbol.upper())
        if pid_cached:
            logger.info(f"✅ Product ID for {symbol}: {pid_cached}")
        else:
            pid_r = self._symbol_to_product_id(symbol)
            if pid_r:
                logger.info(f"✅ Product ID for {symbol}: {pid_r} (from /v2/products)")
            else:
                logger.warning(f"⚠️  Product ID not resolved for {symbol}")

        # ── Clock summary ──────────────────────────────────────────────────────
        if self._clock_calibrated:
            logger.info(
                f"⏱️  Clock calibration: {self._time_offset_s:+d}s offset applied."
            )
        else:
            logger.info("⏱️  Clock: no skew detected.")

        return ok


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    logging.basicConfig(level=logging.INFO)
    try:
        api = DeltaAPI()
        api.self_test("BTCUSD")
    except Exception as e:
        print(f"\n❌ CRASH: {e}")
        traceback.print_exc()
