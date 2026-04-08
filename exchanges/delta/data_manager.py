"""
exchanges/delta/data_manager.py — Delta Exchange Data Manager
=============================================================
Implements the same public interface as CoinSwitchDataManager.
Uses DeltaAPI for REST warmup (0.25s sleep) and DeltaWebSocket for streams.
Product ID prefetched at startup.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from core.candle import Candle
from exchanges.delta.api    import DeltaAPI
from exchanges.delta.websocket import DeltaWebSocket

logger = logging.getLogger(__name__)


class StreamStats:
    def __init__(self):
        self._last_update: Optional[datetime] = None
        self._ob = self._tr = self._can = 0
        self._lock = threading.RLock()

    def record_orderbook(self):
        with self._lock: self._ob += 1; self._last_update = datetime.now(timezone.utc)

    def record_trade(self):
        with self._lock: self._tr += 1; self._last_update = datetime.now(timezone.utc)

    def record_candle(self):
        with self._lock: self._can += 1; self._last_update = datetime.now(timezone.utc)

    def get_last_update(self) -> Optional[datetime]:
        with self._lock: return self._last_update


class DeltaDataManager:
    """
    Delta Exchange data manager.
    Same public interface as CoinSwitchDataManager.
    """

    _WARMUP_CONFIG = {
        "1m":  (1,    200, "_candles_1m"),
        "5m":  (5,    200, "_candles_5m"),
        "15m": (15,   200, "_candles_15m"),
        "1h":  (60,   100, "_candles_1h"),
        "4h":  (240,   50, "_candles_4h"),
        "1d":  (1440,  30, "_candles_1d"),
    }

    _WARMUP_SLEEP = float(getattr(config, "DELTA_API_MIN_INTERVAL", 0.25))

    def __init__(self) -> None:
        self.api = DeltaAPI(
            api_key    = config.DELTA_API_KEY,
            secret_key = config.DELTA_SECRET_KEY,
            testnet    = getattr(config, "DELTA_TESTNET", False),
        )
        self.ws:    Optional[DeltaWebSocket] = None
        self.stats  = StreamStats()

        self._candles_1m:  deque = deque(maxlen=2000)
        self._candles_5m:  deque = deque(maxlen=1200)
        self._candles_15m: deque = deque(maxlen=800)
        self._candles_1h:  deque = deque(maxlen=500)
        self._candles_4h:  deque = deque(maxlen=400)
        self._candles_1d:  deque = deque(maxlen=100)

        self._last_price:             float = 0.0
        self._last_price_update_time: float = 0.0
        self._orderbook:              Dict  = {"bids": [], "asks": []}
        self._recent_trades:          deque = deque(maxlen=500)

        self._lock            = threading.RLock()
        self._forming_ts:     Dict[str, int] = {}
        self._warmup_complete = False

        self._strategy_ref = None
        self.is_ready      = False
        self.is_streaming  = False
        self._product_id:  Optional[int] = None

        logger.info("DeltaDataManager initialised")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            self.is_ready = self.is_streaming = False
            symbol = getattr(config, "DELTA_SYMBOL", "BTCUSD")

            # Prefetch product ID
            self._product_id = self.api.get_product_id(symbol)
            if self._product_id:
                logger.info(f"Delta product_id for {symbol}: {self._product_id}")
            else:
                logger.warning(f"Delta product_id not resolved for {symbol}")

            logger.info("Delta DM: starting WebSocket...")
            self.ws = DeltaWebSocket(
                api_key    = config.DELTA_API_KEY,
                secret_key = config.DELTA_SECRET_KEY,
                testnet    = getattr(config, "DELTA_TESTNET", False),
            )

            # Subscribe before connect (DeltaWebSocket queues until connected)
            self.ws.subscribe_orderbook(symbol, callback=self._on_orderbook, depth=20)
            self.ws.subscribe_trades(symbol, callback=self._on_trade)
            for tf, (iv_min, _, _) in self._WARMUP_CONFIG.items():
                self.ws.subscribe_candlestick(
                    symbol, interval=iv_min, callback=self._make_candle_cb(tf))

            # Private channels
            if config.DELTA_API_KEY:
                self.ws.subscribe_orders(symbol,    callback=self._on_order_update)
                self.ws.subscribe_positions(symbol, callback=self._on_position_update)
                self.ws.subscribe_account(          callback=self._on_account_update)

            if not self.ws.connect(timeout=30):
                logger.error("❌ Delta WS connection failed")
                return False

            self.is_streaming = True
            logger.info("✅ Delta WS streams started")

            # REST warmup
            logger.info("Delta DM: starting REST warmup...")
            for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
                self._warmup_klines(tf)
                time.sleep(self._WARMUP_SLEEP)

            # BUG-DM FIX: After REST warmup, the last candle in each deque
            # is the current partially-formed bar (Delta REST returns it).
            # When the first WS candle arrives for that same bar, forming_ts
            # is None (never set), so the comparison forming_ts == start_ts
            # is False → WS appends a duplicate instead of replacing in-place.
            # Fix: seed _forming_ts with each deque's last candle timestamp
            # so the first WS tick correctly updates candles[-1] in-place.
            _TF_KEY_MAP = {
                "1m": ("1",    self._candles_1m),
                "5m": ("5",    self._candles_5m),
                "15m": ("15",  self._candles_15m),
                "1h": ("60",   self._candles_1h),
                "4h": ("240",  self._candles_4h),
                "1d": ("1440", self._candles_1d),
            }
            with self._lock:
                for tf_label, (tf_key, deq) in _TF_KEY_MAP.items():
                    if deq:
                        last_c = deq[-1]
                        # Candle.timestamp is in seconds; forming_ts dict stores ms
                        self._forming_ts[tf_key] = int(last_c.timestamp * 1000)

            self._warmup_complete = True
            logger.info("✅ Delta REST warmup complete")

            self.is_ready = self._check_minimum_data()
            logger.info(
                f"Delta DM ready={self.is_ready} "
                f"(1m={len(self._candles_1m)} 5m={len(self._candles_5m)} "
                f"15m={len(self._candles_15m)} 4h={len(self._candles_4h)})"
            )
            return True

        except Exception as e:
            logger.error(f"Delta DM start error: {e}", exc_info=True)
            self.is_ready = self.is_streaming = False
            return False

    def stop(self) -> None:
        try:
            self.is_ready = self.is_streaming = False
            if self.ws:
                self.ws.disconnect()
            logger.info("Delta DM stopped")
        except Exception as e:
            logger.error(f"Delta DM stop error: {e}")

    def restart_streams(self) -> bool:
        try:
            logger.warning("Delta DM: restarting streams")
            self._warmup_complete = False
            self._forming_ts.clear()
            self.stop()
            time.sleep(1.0)
            # Clear candle deques before warmup — without this, warmup appends to
            # existing data producing duplicate candles (same timestamps) that cause
            # the ICT engine to create duplicate OBs and distort structure detection.
            with self._lock:
                self._candles_1m.clear()
                self._candles_5m.clear()
                self._candles_15m.clear()
                self._candles_1h.clear()
                self._candles_4h.clear()
                self._candles_1d.clear()
                self._recent_trades.clear()
            success = self.start()
            if success and self._strategy_ref is not None:
                try:
                    self._strategy_ref.on_stream_restart()
                except Exception:
                    pass
            return success
        except Exception as e:
            logger.error(f"Delta DM restart error: {e}", exc_info=True)
            return False

    def register_strategy(self, strategy) -> None:
        self._strategy_ref = strategy

    def wait_until_ready(self, timeout_sec: float = 120.0) -> bool:
        start = time.time()
        while not self.is_ready and (time.time() - start) < timeout_sec:
            time.sleep(1.0)
            if not self.is_ready:
                self.is_ready = self._check_minimum_data()
        return self.is_ready

    # ── REST warmup ───────────────────────────────────────────────────────────

    def _warmup_klines(self, label: str, limit: int = 0, retries: int = 2) -> None:
        cfg = self._WARMUP_CONFIG.get(label)
        if not cfg:
            return
        interval_min, default_limit, deque_attr = cfg
        limit  = limit or default_limit
        target: deque = getattr(self, deque_attr)
        symbol = getattr(config, "DELTA_SYMBOL", "BTCUSD")

        for attempt in range(1, retries + 2):
            try:
                end_ms   = int(time.time() * 1000)
                start_ms = end_ms - limit * interval_min * 60 * 1000

                resp = self.api.get_candles(
                    symbol     = symbol,
                    resolution = interval_min,
                    start_time = start_ms,
                    end_time   = end_ms,
                    limit      = limit,
                )

                if not resp.get("success"):
                    logger.warning(f"Delta warmup {label} attempt {attempt}: "
                                   f"{resp.get('error')}")
                    if attempt <= retries:
                        time.sleep(2.0)
                    continue

                raw = sorted(
                    [c for c in (resp.get("result") or []) if c.get("t") and c.get("c")],
                    key=lambda c: c["t"],
                )

                seeded = 0
                for c in raw:
                    try:
                        candle = Candle(
                            timestamp = c["t"] / 1000.0,
                            open      = float(c["o"]),
                            high      = float(c["h"]),
                            low       = float(c["l"]),
                            close     = float(c["c"]),
                            volume    = float(c["v"]),
                        )
                        if candle.close > 0:
                            target.append(candle)
                            if label == "1m":
                                self._last_price = candle.close
                            seeded += 1
                    except Exception:
                        continue

                if seeded > 0:
                    logger.info(f"Delta warmup {label}: {seeded} candles")
                    return
                else:
                    if attempt <= retries:
                        time.sleep(2.0)

            except Exception as e:
                logger.error(f"Delta warmup {label} attempt {attempt}: {e}")
                if attempt <= retries:
                    time.sleep(2.0)

    # ── Candle deque helper ───────────────────────────────────────────────────

    def _process_ws_candle(self, data: Dict, candle: Candle,
                           target: deque, tf_key: str, tf_label: str) -> None:
        if not self._warmup_complete:
            self._last_price = candle.close
            self._last_price_update_time = time.time()
            return

        is_closed  = bool(data.get("x", False))
        start_ts   = int(data.get("t", 0))
        forming_ts = self._forming_ts.get(tf_key)

        self._last_price = candle.close
        self._last_price_update_time = time.time()

        if is_closed:
            if forming_ts == start_ts and target:
                target[-1] = candle
            else:
                target.append(candle)
            self._forming_ts.pop(tf_key, None)
            if tf_label != "1m":
                logger.info(f"✅ Delta {tf_label} CLOSED @ ${candle.close:.2f}")
            else:
                logger.debug(f"✅ Delta 1m CLOSED @ ${candle.close:.2f}")
        else:
            if forming_ts == start_ts and target:
                target[-1] = candle
            else:
                target.append(candle)
                self._forming_ts[tf_key] = start_ts

        self.stats.record_candle()

    def _make_candle_cb(self, label: str):
        _TF_MAP = {
            "1m":  ("1",    self._candles_1m),
            "5m":  ("5",    self._candles_5m),
            "15m": ("15",   self._candles_15m),
            "1h":  ("60",   self._candles_1h),
            "4h":  ("240",  self._candles_4h),
            "1d":  ("1440", self._candles_1d),
        }
        tf_key, target = _TF_MAP[label]

        def cb(data: Dict):
            try:
                with self._lock:
                    try:
                        c = Candle(
                            timestamp = data["t"] / 1000.0,
                            open      = float(data["o"]),
                            high      = float(data["h"]),
                            low       = float(data["l"]),
                            close     = float(data["c"]),
                            volume    = float(data["v"]),
                        )
                    except Exception:
                        return
                    if c.close > 0:
                        self._process_ws_candle(data, c, target, tf_key, label)
            except Exception as e:
                logger.error(f"Delta {label} candle callback error: {e}")
        return cb

    # ── WS callbacks ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_ob_side(raw: list) -> list:
        """
        Convert Delta orderbook levels to canonical [[price, qty], ...] format.
        Delta WS delivers dicts: {'limit_price': '74041.0', 'size': 477, 'depth': '477'}
        All consumers downstream expect [price, qty] lists.
        """
        result = []
        for lvl in (raw or []):
            try:
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    result.append([float(lvl[0]), float(lvl[1])])
                elif isinstance(lvl, dict):
                    px  = float(lvl.get("limit_price") or lvl.get("price") or 0)
                    qty = float(lvl.get("size") or lvl.get("quantity") or
                                lvl.get("depth") or 0)
                    if px > 0:
                        result.append([px, qty])
            except Exception:
                continue
        return result

    def _on_orderbook(self, data: Dict) -> None:
        try:
            with self._lock:
                # Delta WS uses "buy"/"sell" keys, NOT "bids"/"asks"
                raw_bids = data.get("buy") or data.get("bids", [])
                raw_asks = data.get("sell") or data.get("asks", [])
                self._orderbook = {
                    "bids": self._normalise_ob_side(raw_bids),
                    "asks": self._normalise_ob_side(raw_asks),
                }
                bids, asks = self._orderbook["bids"], self._orderbook["asks"]
                if bids and asks:
                    try:
                        self._last_price = (bids[0][0] + asks[0][0]) / 2.0
                        self._last_price_update_time = time.time()
                    except Exception:
                        pass
                self.stats.record_orderbook()
        except Exception as e:
            logger.debug(f"Delta OB callback: {e}")

    def _on_trade(self, data: Dict) -> None:
        try:
            with self._lock:
                # Delta public trades channel uses "price"/"size"/"side" fields.
                # "p"/"q"/"m" is the aggregated ticker format — different channel.
                # Support both formats defensively.
                price = float(data.get("price") or data.get("p") or 0)
                qty   = float(data.get("size")  or data.get("q") or 0)
                side_raw = data.get("side", "")
                if side_raw:
                    side = "buy" if str(side_raw).lower() == "buy" else "sell"
                else:
                    # Fallback: "m" = True means buyer was maker = sell aggressor
                    side = "sell" if bool(data.get("m")) else "buy"
                if price > 0:
                    self._last_price = price
                    self._last_price_update_time = time.time()
                    self._recent_trades.append({
                        "price":     price,
                        "quantity":  qty,
                        "side":      side,
                        "timestamp": time.time(),
                    })
                    if self._strategy_ref is not None:
                        try:
                            on_rt = getattr(self._strategy_ref, "_on_realtime_trade", None)
                            if on_rt:
                                on_rt(price, qty, side)
                        except Exception:
                            pass
                self.stats.record_trade()
        except Exception as e:
            logger.debug(f"Delta trade callback: {e}")

    def _on_order_update(self, data: Dict) -> None:
        logger.debug(f"Delta order update: state={data.get('state')} id={data.get('id')}")

    def _on_position_update(self, data: Dict) -> None:
        logger.debug(f"Delta position update: size={data.get('size')}")

    def _on_account_update(self, data: Dict) -> None:
        logger.debug(f"Delta account update: {data}")

    # ── Readiness ─────────────────────────────────────────────────────────────

    def _check_minimum_data(self) -> bool:
        counts = {
            "1m": len(self._candles_1m), "5m": len(self._candles_5m),
            "15m": len(self._candles_15m), "1h": len(self._candles_1h),
            "4h": len(self._candles_4h),  "1d": len(self._candles_1d),
        }
        mins = {
            "1m":  getattr(config, "MIN_CANDLES_1M",   100),
            "5m":  getattr(config, "MIN_CANDLES_5M",   100),
            "15m": getattr(config, "MIN_CANDLES_15M",  100),
            "1h":  getattr(config, "MIN_CANDLES_1H",    20),
            "4h":  max(getattr(config, "MIN_CANDLES_4H", 40), 29),
            "1d":  getattr(config, "MIN_CANDLES_1D",     7),
        }
        missing = [f"{tf}({counts[tf]}<{mins[tf]})" for tf in mins if counts[tf] < mins[tf]]
        if missing:
            logger.debug(f"Delta DM not ready: {', '.join(missing)}")
            return False
        return True

    # ── Public interface ──────────────────────────────────────────────────────

    def get_last_price(self) -> float:
        with self._lock: return self._last_price

    def get_orderbook(self) -> Dict:
        with self._lock:
            return {
                "bids": list(self._orderbook.get("bids", [])),
                "asks": list(self._orderbook.get("asks", [])),
                "timestamp": time.time(),
            }

    def get_recent_trades_raw(self) -> List[Dict]:
        with self._lock: return list(self._recent_trades)[-200:]

    def is_price_fresh(self, max_stale_seconds: float = 90.0) -> bool:
        if self._last_price_update_time <= 0:
            return False
        return (time.time() - self._last_price_update_time) < max_stale_seconds

    def get_candles(self, timeframe: str = "5m", limit: int = 100) -> List[Dict]:
        tf_map = {
            "1m": self._candles_1m, "5m": self._candles_5m,
            "15m": self._candles_15m, "1h": self._candles_1h,
            "4h": self._candles_4h,  "1d": self._candles_1d,
        }
        src = tf_map.get(timeframe, self._candles_5m)
        with self._lock:
            candles = list(src)
        return [
            {"t": int(c.timestamp * 1000), "o": c.open, "h": c.high,
             "l": c.low, "c": c.close, "v": c.volume}
            for c in candles[-limit:]
        ]

    def get_volume_delta(self, lookback_seconds: float = 60.0) -> Dict:
        with self._lock:
            cutoff   = time.time() - lookback_seconds
            buy_vol  = sum(t["quantity"] for t in self._recent_trades
                          if t["timestamp"] >= cutoff and t["side"] == "buy")
            sell_vol = sum(t["quantity"] for t in self._recent_trades
                          if t["timestamp"] >= cutoff and t["side"] == "sell")
        total = buy_vol + sell_vol
        return {
            "buy_volume":  buy_vol,
            "sell_volume": sell_vol,
            "delta":       buy_vol - sell_vol,
            "delta_pct":   (buy_vol - sell_vol) / total if total > 0 else 0.0,
        }