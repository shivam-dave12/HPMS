"""
delta_websocket.py — Delta Exchange WebSocket Plugin
=====================================================
Production-grade WebSocket client for Delta Exchange.

Features:
  - Native WebSocket (websocket-client library) — Delta uses raw WS, not Socket.IO
  - Automatic reconnection with exponential backoff (max 30s)
  - Full auto-resubscription after reconnect (callbacks survive)
  - Thread-safe callback management
  - Authenticated private channels (orders, positions, fills, account)
  - Public channels (orderbook L1/L2, trades, tickers, candlesticks, markprice)
  - Connection health monitoring
  - Same public interface as CoinSwitch futures_websocket.py for drop-in use

Channel reference:
  Public (no auth):
    v2/ticker:{symbol}              — best bid/ask, last price, 24h stats
    v2/orderbook:{symbol}:{depth}   — L2 orderbook (depth = 5|10|20|50|200)
    all_trades:{symbol}             — public trade feed
    candlestick_1m:{symbol}         — 1-minute OHLCV
    candlestick_{N}m:{symbol}       — Nm OHLCV (N = 3,5,15,30,60,120,240,1d)
    mark_price:{symbol}             — mark price updates
    funding_rate:{symbol}           — funding rate changes

  Private (requires auth):
    v2/user/orders:{symbol}         — own order updates (create/fill/cancel)
    v2/user/fills:{symbol}          — own fill events
    v2/user/positions:{symbol}      — own position changes
    v2/user/account                 — account balance / margin changes
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional

import websocket          # pip install websocket-client
from dotenv import load_dotenv
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DELTA_WS_LIVE    = "wss://socket.india.delta.exchange"    # Delta India production
DELTA_WS_TESTNET = "wss://socket-ind.testnet.deltaex.org"  # Delta India testnet

# NOTE: wss://socket.delta.exchange is Delta GLOBAL — different product, different keys.

# Candle channel name map: minutes → channel name prefix
_CANDLE_CHANNEL: Dict[int, str] = {
    1:     "candlestick_1m",
    3:     "candlestick_3m",
    5:     "candlestick_5m",
    15:    "candlestick_15m",
    30:    "candlestick_30m",
    60:    "candlestick_1h",
    120:   "candlestick_2h",
    240:   "candlestick_4h",
    360:   "candlestick_6h",
    720:   "candlestick_12h",
    1440:  "candlestick_1d",
    10080: "candlestick_1w",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class _Subscription:
    """
    Internal record for a single channel subscription.

    delta_channel_spec is the dict passed to Delta's subscribe payload:
      {"name": "l2_orderbook", "symbols": ["BTCUSDT"]}
      {"name": "candlestick_1m", "symbols": ["BTCUSDT"]}
      {"name": "v2/user/orders"}   (private, no symbols)
    """
    __slots__ = ("channel", "callback", "is_private", "delta_channel_spec")

    def __init__(
        self,
        channel:            str,
        callback:           Optional[Callable],
        is_private:         bool = False,
        delta_channel_spec: Optional[Dict] = None,
    ):
        self.channel            = channel
        self.callback           = callback
        self.is_private         = is_private
        self.delta_channel_spec = delta_channel_spec or {"name": channel}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DeltaWebSocket:
    """
    Production WebSocket client for Delta Exchange.

    Usage
    -----
    ws = DeltaWebSocket()
    ws.subscribe_orderbook("BTCUSDT", callback=my_ob_handler, depth=20)
    ws.subscribe_candlestick("BTCUSDT", interval=5, callback=my_candle_handler)
    ws.subscribe_trades("BTCUSDT", callback=my_trade_handler)
    ws.connect()    # non-blocking; starts background thread

    # Private (authenticated) channels:
    ws.subscribe_orders("BTCUSDT", callback=my_order_handler)
    ws.subscribe_fills("BTCUSDT",  callback=my_fill_handler)

    # Health check:
    ws.is_healthy(timeout_seconds=30)
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        testnet:    bool          = False,
    ):
        self.api_key    = api_key    or os.getenv("DELTA_API_KEY",    "")
        self.secret_key = secret_key or os.getenv("DELTA_SECRET_KEY", "")
        self.ws_url     = DELTA_WS_TESTNET if testnet else DELTA_WS_LIVE

        # Connection state
        self._ws:              Optional[websocket.WebSocketApp] = None
        self._ws_thread:       Optional[threading.Thread]       = None
        self.is_connected:     bool      = False
        self._stop_event:      threading.Event = threading.Event()
        self._reconnect_delay: float     = 1.0   # seconds, doubles on failure
        self._max_delay:       float     = 30.0
        self._connection_failures: int   = 0

        # Health monitoring
        self._last_message_time: Optional[datetime] = None
        self._msg_count:         int = 0

        # Subscriptions: channel_name → _Subscription
        self._subs_lock = threading.RLock()
        self._subs:     Dict[str, _Subscription] = {}

        # Callbacks indexed by channel (public)
        # Candle callbacks keyed by interval string ("1", "5", "15", ...)
        self._cb_lock              = threading.RLock()
        self.orderbook_callbacks:  Dict[str, List[Callable]] = {}   # {symbol: [cb]}
        self.trades_callbacks:     Dict[str, List[Callable]] = {}   # {symbol: [cb]}
        self.ticker_callbacks:     Dict[str, List[Callable]] = {}   # {symbol: [cb]}
        self.candlestick_callbacks: Dict[str, List[Callable]] = {}  # {interval: [cb]}
        self.mark_price_callbacks: Dict[str, List[Callable]] = {}   # {symbol: [cb]}
        self.funding_callbacks:    Dict[str, List[Callable]] = {}   # {symbol: [cb]}

        # Private callbacks
        self.order_callbacks:      Dict[str, List[Callable]] = {}   # {symbol: [cb]}
        self.fill_callbacks:       Dict[str, List[Callable]] = {}
        self.position_callbacks:   Dict[str, List[Callable]] = {}
        self.account_callbacks:    List[Callable]             = []

        logger.info(
            f"DeltaWebSocket initialized — endpoint: {self.ws_url} "
            f"(testnet={testnet})"
        )

    # =========================================================================
    # AUTH
    # =========================================================================

    def _auth_payload(self) -> Dict:
        """
        Build authentication payload for private channel subscription.
        Delta uses HMAC-SHA256: signature = HMAC(secret, method+timestamp+path)
        For WS auth: method="GET", path="/live"
        """
        if not self.api_key or not self.secret_key:
            return {}
        ts      = str(int(time.time()))          # seconds, NOT milliseconds
        sig_str = "GET" + ts + "/live"
        sig     = hmac.new(
            self.secret_key.encode("utf-8"),
            sig_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "type":    "key-auth",               # "auth" deprecated Dec 2025
            "payload": {
                "api-key":   self.api_key,
                "signature": sig,
                "timestamp": ts,
            },
        }

    def _is_authenticated(self) -> bool:
        return bool(self.api_key and self.secret_key)

    # =========================================================================
    # CONNECTION LIFECYCLE
    # =========================================================================

    def connect(self, timeout: int = 30) -> bool:
        """
        Connect to Delta Exchange WebSocket in a background thread.

        The connection is non-blocking. Returns True once the socket is open
        (or False on timeout).
        """
        self._stop_event.clear()
        self._reconnect_delay = 1.0

        connected_event = threading.Event()

        def _on_open(ws):
            self.is_connected    = True
            self._connection_failures = 0
            self._reconnect_delay     = 1.0
            self._last_message_time   = datetime.now()
            logger.info("✅ DeltaWebSocket connected")
            connected_event.set()
            # Authenticate if credentials available
            if self._is_authenticated():
                self._send_json(self._auth_payload())
                time.sleep(0.3)
            # Resubscribe all channels
            self._resubscribe_all()

        def _on_message(ws, message: str):
            self._last_message_time = datetime.now()
            self._msg_count += 1
            try:
                data = json.loads(message)
                self._dispatch(data)
            except Exception as e:
                logger.debug(f"WS message parse error: {e}")

        def _on_error(ws, error):
            self._connection_failures += 1
            logger.error(f"DeltaWebSocket error (failure #{self._connection_failures}): {error}")

        def _on_close(ws, close_status_code, close_msg):
            self.is_connected = False
            logger.warning(
                f"DeltaWebSocket closed (code={close_status_code} msg={close_msg})"
            )
            # Auto-reconnect unless explicitly stopped
            if not self._stop_event.is_set():
                self._schedule_reconnect()

        websocket.enableTrace(False)
        self._ws = websocket.WebSocketApp(
            self.ws_url,
            on_open    = _on_open,
            on_message = _on_message,
            on_error   = _on_error,
            on_close   = _on_close,
        )

        self._ws_thread = threading.Thread(
            target = self._ws.run_forever,
            kwargs = {"ping_interval": 30, "ping_timeout": 10},
            daemon = True,
            name   = "DeltaWSThread",
        )
        self._ws_thread.start()

        connected = connected_event.wait(timeout=timeout)
        if not connected:
            logger.error(f"DeltaWebSocket connection timeout after {timeout}s")
        return connected

    def disconnect(self):
        """Gracefully disconnect and stop reconnection loop."""
        logger.info("DeltaWebSocket disconnecting...")
        self._stop_event.set()
        self.is_connected = False
        try:
            if self._ws:
                self._ws.close()
        except Exception as e:
            logger.debug(f"Disconnect error: {e}")

    def _schedule_reconnect(self):
        """Exponential back-off reconnect."""
        delay = min(self._reconnect_delay, self._max_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._max_delay)
        logger.warning(f"DeltaWebSocket reconnecting in {delay:.1f}s...")
        threading.Timer(delay, self._do_reconnect).start()

    def _do_reconnect(self):
        if self._stop_event.is_set():
            return
        logger.info("DeltaWebSocket reconnecting now...")
        self.connect(timeout=30)

    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """True if connected and received data within timeout_seconds."""
        if not self.is_connected:
            return False
        if self._last_message_time is None:
            return True   # just connected, no messages yet
        elapsed = (datetime.now() - self._last_message_time).total_seconds()
        return elapsed < timeout_seconds

    # =========================================================================
    # SEND
    # =========================================================================

    def _send_json(self, payload: Dict) -> bool:
        try:
            if self._ws and self.is_connected:
                self._ws.send(json.dumps(payload))
                return True
        except Exception as e:
            logger.error(f"WS send error: {e}")
        return False

    def _subscribe_channels(self, channels: List[Dict]) -> bool:
        """
        Send a batch subscription request.
        Delta format:
          {"type": "subscribe",
           "payload": {"channels": [{"name": "l2_orderbook", "symbols": ["BTCUSDT"]}, ...]}}
        """
        return self._send_json({
            "type":    "subscribe",
            "payload": {"channels": channels},
        })

    def _unsubscribe_channels(self, channels: List[Dict]) -> bool:
        return self._send_json({
            "type":    "unsubscribe",
            "payload": {"channels": channels},
        })

    # =========================================================================
    # RESUBSCRIPTION AFTER RECONNECT
    # =========================================================================

    def _resubscribe_all(self):
        """
        Resubscribe to all stored channels after reconnect.

        Public channels are sent immediately. Private channels are NOT sent here
        because auth hasn't been confirmed yet at call time (_on_open fires this,
        then sends the auth frame). Private channels are subscribed exclusively in
        _dispatch when the key-auth response arrives with success=True.
        Sending privates before auth confirmation causes Delta to reject them
        silently, requiring a second subscription anyway — causing a race condition
        and duplicate subscriptions on every reconnect.
        """
        with self._subs_lock:
            subs = list(self._subs.values())

        if not subs:
            logger.debug("No subscriptions to restore")
            return

        public_channels = [s.delta_channel_spec for s in subs if not s.is_private]

        if public_channels:
            self._subscribe_channels(public_channels)
            logger.info(f"🔄 Resubscribed {len(public_channels)} public channels")

    # =========================================================================
    # MESSAGE DISPATCH
    # =========================================================================

    def _dispatch(self, data: Dict):
        """
        Route an incoming Delta WS JSON message to the correct callbacks.

        Delta message format:
          {"type": "l2_orderbook",  "symbol": "BTCUSDT", "buy": [...], "sell": [...]}
          {"type": "all_trades",    "symbol": "BTCUSDT", "data": [...]}
          {"type": "candlestick_1m","symbol": "BTCUSDT", "open": ..., "close": ...}
          {"type": "v2/ticker",     "symbol": "BTCUSDT", ...}
          {"type": "subscriptions", "payload": {...}}   ← subscription confirmation
          {"type": "heartbeat"}

        The "type" field IS the channel name — no separate "channel" field.
        """
        msg_type = str(data.get("type", "")).lower()
        symbol   = str(data.get("symbol", "")).upper()

        # ── System messages — ignore silently ─────────────────────────────────
        if msg_type in ("subscriptions", "heartbeat", "info"):
            logger.debug(f"WS system message: {msg_type}")
            return

        # ── Auth response ─────────────────────────────────────────────────────
        if msg_type in ("key-auth", "auth"):
            if data.get("success"):
                logger.info("✅ DeltaWebSocket authenticated (private channels active)")
                with self._subs_lock:
                    priv = [s.delta_channel_spec for s in self._subs.values() if s.is_private]
                if priv:
                    self._subscribe_channels(priv)
                    logger.info(f"🔄 Resubscribed {len(priv)} private channels")
            else:
                logger.error(f"❌ DeltaWebSocket auth failed: status={data.get('status','')} {data.get('message','')}")
            return

        # ── Orderbook: type = "l2_orderbook" ─────────────────────────────────
        if msg_type == "l2_orderbook":
            # Log the first raw orderbook message so we can verify field names
            if not getattr(self, '_ob_raw_logged', False):
                self._ob_raw_logged = True
                buy_sample  = data.get("buy",  [])[:1]
                sell_sample = data.get("sell", [])[:1]
                logger.info(
                    f"🔍 Orderbook raw sample — "
                    f"buy[0]={buy_sample}  sell[0]={sell_sample}"
                )
            fmt = self._fmt_orderbook(data)
            self._fire(self.orderbook_callbacks.get(symbol, []), fmt)
            return

        # ── Trades: type = "all_trades" ───────────────────────────────────────
        if msg_type == "all_trades":
            # Delta sends a list of trades under "data" key
            trade_list = data.get("data", data.get("trades", [data]))
            if not isinstance(trade_list, list):
                trade_list = [trade_list]
            for t in trade_list:
                if isinstance(t, dict):
                    t["symbol"] = symbol   # ensure symbol present
                    self._fire(self.trades_callbacks.get(symbol, []), self._fmt_trade(t))
            return

        # ── Ticker: type = "v2/ticker" or "ticker" ────────────────────────────
        if "ticker" in msg_type:
            self._fire(self.ticker_callbacks.get(symbol, []), data)
            return

        # ── Candlestick: type = "candlestick_1m", "candlestick_5m", etc. ─────
        if msg_type.startswith("candlestick_"):
            interval_key = self._parse_candle_interval_from_type(msg_type)
            if interval_key is None:
                return   # unrecognised format — warning already logged, skip
            self._fire(
                self.candlestick_callbacks.get(interval_key, []),
                self._fmt_candle(data, interval_key),
            )
            return

        # ── Mark price ────────────────────────────────────────────────────────
        if "mark_price" in msg_type:
            self._fire(self.mark_price_callbacks.get(symbol, []), data)
            return

        # ── Funding rate ──────────────────────────────────────────────────────
        if "funding_rate" in msg_type:
            self._fire(self.funding_callbacks.get(symbol, []), data)
            return

        # ── Private: own orders ───────────────────────────────────────────────
        if msg_type == "orders" or "user/orders" in msg_type or msg_type == "user_orders":
            payload = data.get("payload", data)
            sym = str(payload.get("product_symbol", symbol)).upper()
            cbs = self.order_callbacks.get(sym, self.order_callbacks.get("*", []))
            self._fire(cbs, payload)
            return

        # ── Private: fills ────────────────────────────────────────────────────
        if msg_type == "fills" or "user/fills" in msg_type:
            payload = data.get("payload", data)
            sym = str(payload.get("product_symbol", symbol)).upper()
            cbs = self.fill_callbacks.get(sym, self.fill_callbacks.get("*", []))
            self._fire(cbs, payload)
            return

        # ── Private: positions ────────────────────────────────────────────────
        if msg_type == "positions" or "user/positions" in msg_type or "position" in msg_type:
            payload = data.get("payload", data)
            sym = str(payload.get("symbol", symbol)).upper()
            cbs = self.position_callbacks.get(sym, self.position_callbacks.get("*", []))
            self._fire(cbs, payload)
            return

        # ── Private: account ──────────────────────────────────────────────────
        if msg_type in ("trading_notifications","account_updates") or "user/account" in msg_type or "account" in msg_type:
            self._fire(self.account_callbacks, data.get("payload", data))
            return

        logger.debug(f"WS unhandled message type '{msg_type}' symbol={symbol}")

    @staticmethod
    def _parse_candle_interval_from_type(msg_type: str) -> Optional[str]:
        """
        Extract interval key from Delta message type string.
        Examples:
          "candlestick_1m"  → "1"
          "candlestick_5m"  → "5"
          "candlestick_1h"  → "60"
          "candlestick_1d"  → "1440"

        Returns None on any parse failure — callers must guard against None
        to avoid misdirecting candles to the wrong timeframe bucket.
        Previously returned "1" as fallback, which silently routed 4h/1d
        candles to the 1m callback if Delta sent an unexpected type string.
        """
        try:
            # Remove "candlestick_" prefix
            suffix = msg_type.replace("candlestick_", "")  # e.g. "1m", "5m", "1h"
            if suffix.endswith("m"):
                return suffix[:-1]                          # "5m" → "5"
            if suffix.endswith("h"):
                return str(int(suffix[:-1]) * 60)           # "1h" → "60"
            if suffix.endswith("d"):
                return str(int(suffix[:-1]) * 1440)
            if suffix.endswith("w"):
                return str(int(suffix[:-1]) * 10080)
        except Exception:
            pass
        logger.warning(f"WS: unrecognised candlestick type '{msg_type}' — skipping")
        return None

    @staticmethod
    def _fire(callbacks: List[Callable], data: Any) -> None:
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                logger.error(f"WS callback error: {e}", exc_info=True)

    # =========================================================================
    # DATA FORMATTERS — convert Delta WS format to bot-internal format
    # =========================================================================

    @staticmethod
    def _fmt_orderbook(data: Dict) -> Dict:
        """
        Convert Delta L2 orderbook WS message to internal format.

        Delta WS l2_orderbook message:
          {"type": "l2_orderbook", "symbol": "BTCUSDT",
           "buy":  [{"price": "74878.0", "size": 248}, ...],
           "sell": [{"price": "74879.5", "size": 500}, ...]}

        Note: some Delta WS versions may send price as int/float, others as string.
        Some versions include a "depth" field alongside price/size.

        Output: {"bids": [[price_str, size_str], ...], "asks": [...], "symbol": str}
        """
        bids_raw = data.get("buy",  [])
        asks_raw = data.get("sell", [])

        def _extract_level(lvl) -> Optional[List]:
            if isinstance(lvl, dict):
                # Try all known Delta price field names in priority order
                p = (lvl.get("price") or lvl.get("limit_price") or
                     lvl.get("rate")  or lvl.get("p") or 0)
                s = (lvl.get("size")  or lvl.get("quantity") or
                     lvl.get("qty")   or lvl.get("q") or 0)
                p_f = float(p) if p else 0.0
                s_f = float(s) if s else 0.0
                if p_f > 0:
                    # BUG-DWS-1 FIX: return native floats, not strings.
                    # Returning str(p_f) forced every consumer (market_aggregator,
                    # fee_engine, orderbook engine) to silently re-cast on every
                    # orderbook update — adding per-tick overhead and masking type
                    # errors. All other exchanges already produce float pairs.
                    return [p_f, s_f]
                # price == 0 → level deletion message; no useful data
                return None
            elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                p_f = float(lvl[0]) if lvl[0] else 0.0
                s_f = float(lvl[1]) if lvl[1] else 0.0
                if p_f > 0:
                    return [p_f, s_f]  # BUG-DWS-1 FIX: native floats
                return None
            return None

        bids = [r for lvl in bids_raw if (r := _extract_level(lvl)) is not None]
        asks = [r for lvl in asks_raw if (r := _extract_level(lvl)) is not None]

        return {
            "bids":      bids,
            "asks":      asks,
            "symbol":    data.get("symbol", ""),
            "timestamp": data.get("timestamp", int(time.time() * 1000)),
        }

    @staticmethod
    def _fmt_trade(data: Dict) -> Dict:
        """
        Convert one Delta trade entry to internal format.

        Delta sends trade data as:
          {"type": "all_trades", "symbol": "BTCUSDT",
           "data": [{"price": "75112", "size": 1, "side": "buy",
                     "timestamp": <unix_us>, "trade_id": 123}]}

        This method is called per-trade (dispatch loops over data["data"]).

        Output: {"p": price, "q": qty, "m": is_buyer_maker, "s": symbol, "T": ts_ms}
        """
        price = float(data.get("price", data.get("p", 0)) or 0)
        size  = float(data.get("size",  data.get("q", 0)) or 0)
        side  = str(data.get("side", "buy")).lower()

        # Delta: side = "buy"/"sell" from the aggressor's perspective
        # "buy" = buyer aggressed (bought at ask) = seller was maker
        # is_buyer_maker=True means buyer is the passive side (placed the bid, maker filled)
        is_buyer_maker = (side == "sell")  # sell aggressor = buyer was maker

        # timestamp is in microseconds on Delta
        ts_raw = data.get("timestamp", data.get("T", 0))
        ts_ms  = int(ts_raw) // 1000 if ts_raw and int(ts_raw) > 1e12 else int(ts_raw or 0)

        return {
            "p": price,
            "q": size,
            "m": is_buyer_maker,
            "s": data.get("symbol", ""),
            "T": ts_ms,
        }

    @staticmethod
    def _fmt_candle(data: Dict, interval_key: str) -> Dict:
        """
        Convert Delta candlestick message to internal OHLCV format.

        Delta candlestick message (top-level fields, no "payload" wrapper):
          {"type": "candlestick_5m", "symbol": "BTCUSDT",
           "time": <unix_seconds>, "open": "72502.5", "high": "75965.0",
           "low": "72455.5", "close": "75112.0", "volume": "211.228"}

        Output:
          {"t": ms, "o": f, "h": f, "l": f, "c": f, "v": f, "i": str, "x": bool}
        """
        # Delta puts candle data at top level, not in a "payload" sub-dict
        ts_raw = data.get("time", data.get("t", 0))
        ts_sec = int(ts_raw) if ts_raw else 0
        return {
            "t": ts_sec * 1000,                            # s → ms
            "o": float(data.get("open",   0) or 0),
            "h": float(data.get("high",   0) or 0),
            "l": float(data.get("low",    0) or 0),
            "c": float(data.get("close",  0) or 0),
            "v": float(data.get("volume", 0) or 0),
            "i": interval_key,
            # Delta doesn't send an "is_closed" flag — treat every message as
            # a forming update; data_manager handles closed logic via next candle arrival
            "x": False,
        }

    @staticmethod
    def _parse_candle_interval(channel: str) -> Optional[str]:
        """Extract interval string from channel name, e.g. 'candlestick_5m:BTCUSDT' → '5'.
        Returns None on any parse failure — callers must guard against None."""
        # Channel format: candlestick_Xm:SYMBOL or candlestick_1h:SYMBOL
        try:
            part = channel.split(":")[0]          # e.g. 'candlestick_5m'
            suffix = part.split("_", 1)[1]        # e.g. '5m', '1h', '1d'
            if suffix.endswith("m"):
                return suffix[:-1]                 # '5m' → '5'
            if suffix.endswith("h"):
                return str(int(suffix[:-1]) * 60)  # '1h' → '60'
            if suffix.endswith("d"):
                return str(int(suffix[:-1]) * 1440)
            if suffix.endswith("w"):
                return str(int(suffix[:-1]) * 10080)
        except Exception:
            pass
        logger.warning(f"WS: unrecognised candle channel '{channel}' — skipping")
        return None

    # =========================================================================
    # PUBLIC CHANNEL SUBSCRIPTIONS
    # =========================================================================

    def subscribe_orderbook(
        self,
        symbol:   str,
        callback: Optional[Callable] = None,
        depth:    int                = 20,
    ):
        """
        Subscribe to L2 orderbook snapshots.

        Delta subscription:
          {"name": "l2_orderbook", "symbols": ["BTCUSDT"]}

        Messages arrive as:
          {"type": "l2_orderbook", "symbol": "BTCUSDT",
           "buy": [{"price": ..., "size": ...}, ...],
           "sell": [...]}

        callback(data) receives normalised:
          {"bids": [[price_str, size_str], ...], "asks": [...], "symbol": str}
        """
        sym_upper = symbol.upper()
        channel   = f"l2_orderbook.{sym_upper}"   # internal key only

        spec = {"name": "l2_orderbook", "symbols": [sym_upper]}

        with self._cb_lock:
            if sym_upper not in self.orderbook_callbacks:
                self.orderbook_callbacks[sym_upper] = []
            if callback and callback not in self.orderbook_callbacks[sym_upper]:
                self.orderbook_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])
        logger.info(f"📊 Subscribed orderbook: {sym_upper}")

    def subscribe_trades(
        self,
        symbol:   str,
        callback: Optional[Callable] = None,
    ):
        """
        Subscribe to real-time trade feed.

        Delta subscription:
          {"name": "all_trades", "symbols": ["BTCUSDT"]}

        Messages: {"type": "all_trades", "symbol": "BTCUSDT",
                   "data": [{"price": ..., "size": ..., "side": ...}, ...]}

        callback(data) receives normalised per-trade:
          {"p": price, "q": qty, "m": is_buyer_maker, "s": symbol, "T": ts_ms}
        """
        sym_upper = symbol.upper()
        channel   = f"all_trades.{sym_upper}"
        spec      = {"name": "all_trades", "symbols": [sym_upper]}

        with self._cb_lock:
            if sym_upper not in self.trades_callbacks:
                self.trades_callbacks[sym_upper] = []
            if callback and callback not in self.trades_callbacks[sym_upper]:
                self.trades_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])
        logger.info(f"💹 Subscribed trades: {sym_upper}")

    def subscribe_ticker(
        self,
        symbol:   str,
        callback: Optional[Callable] = None,
    ):
        """
        Subscribe to ticker updates (best bid/ask, last price, OI, 24h stats).

        Delta subscription: {"name": "v2/ticker", "symbols": ["BTCUSDT"]}
        """
        sym_upper = symbol.upper()
        channel   = f"ticker.{sym_upper}"
        spec      = {"name": "v2/ticker", "symbols": [sym_upper]}

        with self._cb_lock:
            if sym_upper not in self.ticker_callbacks:
                self.ticker_callbacks[sym_upper] = []
            if callback and callback not in self.ticker_callbacks[sym_upper]:
                self.ticker_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])
        logger.info(f"📈 Subscribed ticker: {sym_upper}")

    def subscribe_candlestick(
        self,
        symbol:   str,
        interval: int                = 1,
        callback: Optional[Callable] = None,
    ):
        """
        Subscribe to OHLCV candlestick updates.

        interval: candle size in minutes.
          Delta channel names: candlestick_1m, candlestick_5m, candlestick_15m,
                               candlestick_30m, candlestick_1h, candlestick_2h,
                               candlestick_4h, candlestick_6h, candlestick_12h,
                               candlestick_1d, candlestick_1w

        Delta subscription:
          {"name": "candlestick_1m", "symbols": ["BTCUSDT"]}

        Messages: {"type": "candlestick_1m", "symbol": "BTCUSDT",
                   "time": <unix_s>, "open": f, "high": f, "low": f,
                   "close": f, "volume": f}

        callback(data) receives normalised:
          {"t": ms, "o": f, "h": f, "l": f, "c": f, "v": f, "i": interval_str, "x": bool}
        """
        sym_upper    = symbol.upper()
        channel_name = _CANDLE_CHANNEL.get(interval, f"candlestick_{interval}m")
        channel_key  = f"{channel_name}.{sym_upper}"
        interval_key = str(interval)
        spec         = {"name": channel_name, "symbols": [sym_upper]}

        with self._cb_lock:
            if interval_key not in self.candlestick_callbacks:
                self.candlestick_callbacks[interval_key] = []
            if callback and callback not in self.candlestick_callbacks[interval_key]:
                self.candlestick_callbacks[interval_key].append(callback)

        with self._subs_lock:
            self._subs[channel_key] = _Subscription(channel_key, callback, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])
        logger.info(f"🕯️  Subscribed {channel_name}: {sym_upper} (key={interval_key})")

    def subscribe_mark_price(
        self,
        symbol:   str,
        callback: Optional[Callable] = None,
    ):
        """Subscribe to mark price updates."""
        sym_upper = symbol.upper()
        channel   = f"mark_price.{sym_upper}"
        spec      = {"name": "mark_price", "symbols": [sym_upper]}

        with self._cb_lock:
            if sym_upper not in self.mark_price_callbacks:
                self.mark_price_callbacks[sym_upper] = []
            if callback and callback not in self.mark_price_callbacks[sym_upper]:
                self.mark_price_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])

    def subscribe_funding_rate(
        self,
        symbol:   str,
        callback: Optional[Callable] = None,
    ):
        """Subscribe to funding rate changes."""
        sym_upper = symbol.upper()
        channel   = f"funding_rate.{sym_upper}"
        spec      = {"name": "funding_rate", "symbols": [sym_upper]}

        with self._cb_lock:
            if sym_upper not in self.funding_callbacks:
                self.funding_callbacks[sym_upper] = []
            if callback and callback not in self.funding_callbacks[sym_upper]:
                self.funding_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])

    # =========================================================================
    # PRIVATE CHANNEL SUBSCRIPTIONS (require authentication)
    # =========================================================================

    def subscribe_orders(
        self,
        symbol:   str           = "*",
        callback: Optional[Callable] = None,
    ):
        """
        Subscribe to own order updates (create/partial-fill/fill/cancel).

        Delta subscription: {"name": "v2/user/orders", "symbols": ["BTCUSDT"]}
        or without symbols for all products.
        """
        if not self._is_authenticated():
            logger.warning("subscribe_orders: no credentials — private channel unavailable")
            return
        sym_upper = symbol.upper()
        channel   = f"user_orders.{sym_upper}"
        spec: Dict = {"name": "orders", "symbols": [sym_upper if sym_upper != "*" else "all"]}

        with self._cb_lock:
            if sym_upper not in self.order_callbacks:
                self.order_callbacks[sym_upper] = []
            if callback and callback not in self.order_callbacks[sym_upper]:
                self.order_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, is_private=True, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])
        logger.info(f"🔐 Subscribed orders: {sym_upper}")

    def subscribe_fills(
        self,
        symbol:   str           = "*",
        callback: Optional[Callable] = None,
    ):
        """Subscribe to own fill events."""
        if not self._is_authenticated():
            logger.warning("subscribe_fills: no credentials")
            return
        sym_upper = symbol.upper()
        channel   = f"user_fills.{sym_upper}"
        spec: Dict = {"name": "fills", "symbols": [sym_upper if sym_upper != "*" else "all"]}

        with self._cb_lock:
            if sym_upper not in self.fill_callbacks:
                self.fill_callbacks[sym_upper] = []
            if callback and callback not in self.fill_callbacks[sym_upper]:
                self.fill_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, is_private=True, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])
        logger.info(f"🔐 Subscribed fills: {sym_upper}")

    def subscribe_positions(
        self,
        symbol:   str           = "*",
        callback: Optional[Callable] = None,
    ):
        """Subscribe to own position changes."""
        if not self._is_authenticated():
            logger.warning("subscribe_positions: no credentials")
            return
        sym_upper = symbol.upper()
        channel   = f"user_positions.{sym_upper}"
        spec: Dict = {"name": "positions", "symbols": [sym_upper if sym_upper != "*" else "all"]}

        with self._cb_lock:
            if sym_upper not in self.position_callbacks:
                self.position_callbacks[sym_upper] = []
            if callback and callback not in self.position_callbacks[sym_upper]:
                self.position_callbacks[sym_upper].append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, is_private=True, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])

    def subscribe_account(self, callback: Optional[Callable] = None):
        """Subscribe to account-level events (balance changes, margin)."""
        if not self._is_authenticated():
            logger.warning("subscribe_account: no credentials")
            return
        channel = "user_account"
        spec    = {"name": "trading_notifications"}

        with self._cb_lock:
            if callback and callback not in self.account_callbacks:
                self.account_callbacks.append(callback)

        with self._subs_lock:
            self._subs[channel] = _Subscription(channel, callback, is_private=True, delta_channel_spec=spec)

        if self.is_connected:
            self._subscribe_channels([spec])

    # =========================================================================
    # UNSUBSCRIBE
    # =========================================================================

    def unsubscribe(self, channel: str) -> bool:
        """Unsubscribe from a channel by its internal key."""
        with self._subs_lock:
            sub = self._subs.pop(channel, None)
        if sub:
            return self._unsubscribe_channels([sub.delta_channel_spec])
        return False

    def unsubscribe_orderbook(self, symbol: str):
        self.unsubscribe(f"l2_orderbook.{symbol.upper()}")

    def unsubscribe_trades(self, symbol: str):
        self.unsubscribe(f"all_trades.{symbol.upper()}")

    def unsubscribe_candlestick(self, symbol: str, interval: int):
        name = _CANDLE_CHANNEL.get(interval, f"candlestick_{interval}m")
        self.unsubscribe(f"{name}.{symbol.upper()}")

    # =========================================================================
    # PING / KEEP-ALIVE
    # =========================================================================

    def ping(self) -> bool:
        """Send a heartbeat to keep the connection alive."""
        return self._send_json({"type": "heartbeat"})

    # =========================================================================
    # STATS
    # =========================================================================

    @property
    def subscription_count(self) -> int:
        with self._subs_lock:
            return len(self._subs)

    @property
    def message_count(self) -> int:
        return self._msg_count


# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIAS — unused but avoids NameError in _fire
# ─────────────────────────────────────────────────────────────────────────────
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ob_count = [0]
    trade_count = [0]
    candle_count = [0]

    def on_ob(data):
        ob_count[0] += 1
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if ob_count[0] <= 3:
            if bids and asks:
                bb = bids[0][0]; ba = asks[0][0]
                warn = " ⚠️ ZERO PRICE - check raw log above" if float(bb) == 0 else ""
                print(f"  📊 OB  best_bid={bb} best_ask={ba}  "
                      f"(depth: {len(bids)}×{len(asks)}){warn}")
            else:
                print(f"  📊 OB received but bids/asks empty — raw keys: {list(data.keys())}")

    def on_trade(data):
        trade_count[0] += 1
        if trade_count[0] <= 5:
            side = "SELL" if data.get("m") else "BUY"
            print(f"  💹 Trade  {side}  price={data.get('p')}  qty={data.get('q')}")

    def on_candle(data):
        candle_count[0] += 1
        if candle_count[0] <= 3:
            print(f"  🕯️  Candle[{data.get('i')}m]  "
                  f"o={data.get('o')}  h={data.get('h')}  "
                  f"l={data.get('l')}  c={data.get('c')}  v={data.get('v')}")

    symbol = "BTCUSD"
    ws = DeltaWebSocket()
    ws.subscribe_orderbook(symbol, callback=on_ob, depth=5)
    ws.subscribe_trades(symbol, callback=on_trade)
    ws.subscribe_candlestick(symbol, interval=1, callback=on_candle)

    print(f"Connecting to {ws.ws_url}...")
    connected = ws.connect(timeout=15)
    if connected:
        print("✅ Connected! Listening for 60s...")
        for i in range(60):
            time.sleep(1)
            if i % 10 == 9:
                print(f"  [{i+1}s] OB updates: {ob_count[0]}  Trades: {trade_count[0]}  Candles: {candle_count[0]}")
        print(f"\n=== FINAL COUNTS ===")
        print(f"  Orderbook updates: {ob_count[0]}")
        print(f"  Trades:            {trade_count[0]}")
        print(f"  Candle updates:    {candle_count[0]}")
        print(f"  Total WS messages: {ws.message_count}")
    else:
        print("❌ Connection failed")

    ws.disconnect()
