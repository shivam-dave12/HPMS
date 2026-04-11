"""
hl_data_manager.py — Hyperliquid Data Manager
===============================================
REST-based data manager for Hyperliquid. Provides the same public interface
as the old DeltaDataManager so strategy.py and main.py need minimal changes.

Uses the HyperliquidClient (async) via a background asyncio event loop.

Data sources:
  - REST /info candleSnapshot   — candle warmup + polling
  - REST /info allMids          — mid prices
  - REST /info l2Book           — orderbook snapshots
  - WS   userFills (own wallet) — SL/TP fill detection
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import config
from core.candle import Candle
from core.api_client import HyperliquidClient

logger = logging.getLogger(__name__)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


class HLDataManager:
    """
    Hyperliquid data manager.
    Same public interface as DeltaDataManager.
    """

    _WARMUP_INTERVALS = {
        "1m":  ("1m",  200),
        "5m":  ("5m",  200),
        "15m": ("15m", 200),
        "1h":  ("1h",  100),
        "4h":  ("4h",   50),
        "1d":  ("1d",   30),
    }

    def __init__(self) -> None:
        # Async infrastructure
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Optional[HyperliquidClient] = None

        self._symbol = getattr(config, "HL_SYMBOL", "BTC")

        # Candle storage
        self._candles_1m:  deque = deque(maxlen=2000)
        self._candles_5m:  deque = deque(maxlen=1200)
        self._candles_15m: deque = deque(maxlen=800)
        self._candles_1h:  deque = deque(maxlen=500)
        self._candles_4h:  deque = deque(maxlen=400)
        self._candles_1d:  deque = deque(maxlen=100)

        self._last_price: float = 0.0
        self._last_price_update_time: float = 0.0
        self._orderbook: Dict = {"bids": [], "asks": []}
        self._recent_trades: deque = deque(maxlen=500)

        self._lock = threading.RLock()
        self.is_ready = False
        self.is_streaming = False

        # External callbacks
        self._external_fill_callbacks: list = []
        self._external_order_callbacks: list = []

        logger.info("HLDataManager initialised")

    # ── Async bridge ──────────────────────────────────────────────────────────

    def _start_loop(self):
        """Start a background asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout=30):
        """Run an async coroutine from sync code. Returns the result."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("Event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            self.is_ready = self.is_streaming = False

            # Start background event loop
            self._loop_thread = threading.Thread(
                target=self._start_loop, daemon=True, name="HL-AsyncLoop"
            )
            self._loop_thread.start()
            time.sleep(0.3)  # let loop start

            # Create async client
            self._client = self._run_async(self._create_client())

            # REST warmup
            logger.info("HL DM: starting REST warmup...")
            for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
                self._warmup_candles(tf)
                time.sleep(0.3)

            logger.info("HL DM: REST warmup complete")

            # Fetch initial price
            self._refresh_price()

            self.is_streaming = True
            self.is_ready = self._check_minimum_data()
            logger.info(
                f"HL DM ready={self.is_ready} "
                f"(1m={len(self._candles_1m)} 5m={len(self._candles_5m)} "
                f"15m={len(self._candles_15m)} 1h={len(self._candles_1h)})"
            )
            return True

        except Exception as e:
            logger.error(f"HL DM start error: {e}", exc_info=True)
            self.is_ready = self.is_streaming = False
            return False

    async def _create_client(self) -> HyperliquidClient:
        return HyperliquidClient()

    def stop(self) -> None:
        try:
            self.is_ready = self.is_streaming = False
            if self._client and self._loop:
                try:
                    self._run_async(self._client.close(), timeout=5)
                except Exception:
                    pass
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("HL DM stopped")
        except Exception as e:
            logger.error(f"HL DM stop error: {e}")

    def restart_streams(self) -> bool:
        logger.warning("HL DM: restarting...")
        with self._lock:
            self._candles_1m.clear()
            self._candles_5m.clear()
            self._candles_15m.clear()
            self._candles_1h.clear()
            self._candles_4h.clear()
            self._candles_1d.clear()
        self.stop()
        time.sleep(1.0)
        return self.start()

    # ── REST warmup ───────────────────────────────────────────────────────────

    def _warmup_candles(self, label: str) -> None:
        cfg = self._WARMUP_INTERVALS.get(label)
        if not cfg:
            return
        interval_str, limit = cfg
        tf_map = {
            "1m": self._candles_1m, "5m": self._candles_5m,
            "15m": self._candles_15m, "1h": self._candles_1h,
            "4h": self._candles_4h, "1d": self._candles_1d,
        }
        target = tf_map[label]

        # Compute time range
        _INTERVAL_SECONDS = {
            "1m": 60, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        now_ms = int(time.time() * 1000)
        iv_s = _INTERVAL_SECONDS[label]
        start_ms = now_ms - limit * iv_s * 1000

        for attempt in range(3):
            try:
                raw = self._run_async(
                    self._client._post_info({
                        "type": "candleSnapshot",
                        "req": {
                            "coin": self._symbol,
                            "interval": interval_str,
                            "startTime": start_ms,
                            "endTime": now_ms,
                        },
                    }),
                    timeout=15,
                )

                if not isinstance(raw, list):
                    logger.warning(f"HL warmup {label} attempt {attempt+1}: unexpected response type")
                    time.sleep(1)
                    continue

                # Sort by open time
                raw = sorted(raw, key=lambda c: int(c.get("t", 0)))

                seeded = 0
                for c in raw:
                    try:
                        close_ts = int(c.get("T", 0))
                        # Skip the current forming candle (incomplete volume)
                        if close_ts >= now_ms:
                            continue
                        candle = Candle(
                            timestamp=int(c["t"]) / 1000.0,
                            open=float(c["o"]),
                            high=float(c["h"]),
                            low=float(c["l"]),
                            close=float(c["c"]),
                            volume=float(c["v"]),
                        )
                        if candle.close > 0:
                            target.append(candle)
                            if label == "1m":
                                self._last_price = candle.close
                                self._last_price_update_time = time.time()
                            seeded += 1
                    except Exception:
                        continue

                if seeded > 0:
                    logger.info(f"HL warmup {label}: {seeded} candles")
                    return
                time.sleep(1)

            except Exception as e:
                logger.error(f"HL warmup {label} attempt {attempt+1}: {e}")
                time.sleep(1)

    # ── Price refresh (REST polling) ──────────────────────────────────────────

    def _refresh_price(self):
        """Fetch current mid price via REST."""
        try:
            mids = self._run_async(self._client.get_all_mids(), timeout=10)
            mid = mids.get(self._symbol)
            if mid and mid > 0:
                with self._lock:
                    self._last_price = mid
                    self._last_price_update_time = time.time()
        except Exception as e:
            logger.debug(f"Price refresh error: {e}")

    def _refresh_orderbook(self):
        """Fetch orderbook snapshot via REST."""
        try:
            raw = self._run_async(
                self._client._post_info({
                    "type": "l2Book",
                    "coin": self._symbol,
                }),
                timeout=10,
            )
            if isinstance(raw, dict):
                levels = raw.get("levels", [])
                if len(levels) >= 2:
                    bids = [[float(e["px"]), float(e["sz"])] for e in levels[0]]
                    asks = [[float(e["px"]), float(e["sz"])] for e in levels[1]]
                    with self._lock:
                        self._orderbook = {"bids": bids, "asks": asks}
                        if bids and asks:
                            self._last_price = (bids[0][0] + asks[0][0]) / 2.0
                            self._last_price_update_time = time.time()
        except Exception as e:
            logger.debug(f"Orderbook refresh error: {e}")

    # ── Readiness ─────────────────────────────────────────────────────────────

    def _check_minimum_data(self) -> bool:
        counts = {
            "1m": len(self._candles_1m), "5m": len(self._candles_5m),
            "15m": len(self._candles_15m), "1h": len(self._candles_1h),
            "4h": len(self._candles_4h), "1d": len(self._candles_1d),
        }
        mins = {
            "1m": getattr(config, "MIN_CANDLES_1M", 100),
            "5m": getattr(config, "MIN_CANDLES_5M", 50),
            "15m": getattr(config, "MIN_CANDLES_15M", 20),
            "1h": getattr(config, "MIN_CANDLES_1H", 10),
            "4h": getattr(config, "MIN_CANDLES_4H", 5),
            "1d": getattr(config, "MIN_CANDLES_1D", 3),
        }
        missing = [f"{tf}({counts[tf]}<{mins[tf]})" for tf in mins if counts[tf] < mins[tf]]
        if missing:
            logger.debug(f"HL DM not ready: {', '.join(missing)}")
            return False
        return True

    # ── Candle refresh (called from main loop) ───────────────────────────────

    def refresh_latest_candles(self):
        """
        Fetch latest 1m candles (last 5 bars) and update the deque.
        Also refreshes price and orderbook.
        Called periodically from the main loop.

        IMPORTANT: HL candleSnapshot returns the current FORMING candle as
        the last entry. This candle has volume ≈ 0 because it just started.
        We must NOT let it overwrite the completed candle in the deque,
        otherwise the volume filter sees v=0.0 and blocks all entries.

        HL candle fields:
          t  = open time (ms)     — start of the bar
          T  = close time (ms)    — end of the bar
          v  = volume (base asset, e.g. BTC)
        If T >= now, the candle is still forming.
        """
        try:
            now_ms = int(time.time() * 1000)

            # Fetch 8 minutes instead of 5 to avoid boundary misses:
            # with exactly 5 min the candle at start_ms boundary may be
            # excluded when its open time is 1–2 ms before start_ms.
            start_ms = now_ms - 8 * 60 * 1000

            # Extend endTime 2 minutes past now so the request always
            # covers the current forming candle regardless of API latency.
            end_ms = now_ms + 2 * 60 * 1000

            raw = self._run_async(
                self._client._post_info({
                    "type": "candleSnapshot",
                    "req": {
                        "coin": self._symbol,
                        "interval": "1m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                }),
                timeout=10,
            )

            if isinstance(raw, list) and raw:
                raw = sorted(raw, key=lambda c: int(c.get("t", 0)))
                with self._lock:
                    for c in raw:
                        try:
                            ts = int(c["t"]) / 1000.0
                            close_ts = int(c.get("T", 0))
                            candle = Candle(
                                timestamp=ts,
                                open=float(c["o"]),
                                high=float(c["h"]),
                                low=float(c["l"]),
                                close=float(c["c"]),
                                volume=float(c["v"]),
                            )
                            if candle.close <= 0:
                                continue

                            # Always update price from latest data (even forming)
                            self._last_price = candle.close
                            self._last_price_update_time = time.time()

                            # Forming candle detection with 1-second clock-skew buffer.
                            #
                            # Bug in original code: `close_ts >= now_ms` fails when the
                            # client clock is even slightly behind HL server time — a
                            # completed candle (T = minute_boundary) would read as forming
                            # if client now_ms < T by even 1 ms, causing it to be silently
                            # dropped.  The buffer means: only treat a candle as forming
                            # if its close time is MORE than 1 s in the future.  This
                            # tolerates up to 1 s of client-behind-server clock skew
                            # without ever dropping a completed bar.
                            is_forming = close_ts > (now_ms + 1000)

                            if is_forming:
                                # Don't write forming candle into the deque —
                                # it has incomplete volume and would block filters.
                                # Price update above is sufficient.
                                continue

                            # Completed candle — update or append
                            if self._candles_1m and abs(self._candles_1m[-1].timestamp - ts) < 30:
                                self._candles_1m[-1] = candle
                            elif not self._candles_1m or ts > self._candles_1m[-1].timestamp:
                                self._candles_1m.append(candle)
                        except Exception:
                            continue

        except Exception as e:
            logger.debug(f"Candle refresh error: {e}")

        # Also refresh orderbook (for spread filter)
        self._refresh_orderbook()

    # ── Public interface (same as DeltaDataManager) ───────────────────────────

    def register_fill_callback(self, fn) -> None:
        if fn not in self._external_fill_callbacks:
            self._external_fill_callbacks.append(fn)

    def register_order_callback(self, fn) -> None:
        if fn not in self._external_order_callbacks:
            self._external_order_callbacks.append(fn)

    def get_last_price(self) -> float:
        with self._lock:
            return self._last_price

    def get_orderbook(self) -> Dict:
        with self._lock:
            return {
                "bids": list(self._orderbook.get("bids", [])),
                "asks": list(self._orderbook.get("asks", [])),
                "timestamp": time.time(),
            }

    def get_recent_trades_raw(self) -> List[Dict]:
        with self._lock:
            return list(self._recent_trades)[-200:]

    def is_price_fresh(self, max_stale_seconds: float = 90.0) -> bool:
        if self._last_price_update_time <= 0:
            return False
        return (time.time() - self._last_price_update_time) < max_stale_seconds

    def get_candles(self, timeframe: str = "5m", limit: int = 100) -> List[Dict]:
        tf_map = {
            "1m": self._candles_1m, "5m": self._candles_5m,
            "15m": self._candles_15m, "1h": self._candles_1h,
            "4h": self._candles_4h, "1d": self._candles_1d,
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
            cutoff = time.time() - lookback_seconds
            buy_vol = sum(t["quantity"] for t in self._recent_trades
                          if t["timestamp"] >= cutoff and t["side"] == "buy")
            sell_vol = sum(t["quantity"] for t in self._recent_trades
                          if t["timestamp"] >= cutoff and t["side"] == "sell")
        total = buy_vol + sell_vol
        return {
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "delta": buy_vol - sell_vol,
            "delta_pct": (buy_vol - sell_vol) / total if total > 0 else 0.0,
        }

    def wait_until_ready(self, timeout_sec: float = 120.0) -> bool:
        start = time.time()
        while not self.is_ready and (time.time() - start) < timeout_sec:
            time.sleep(1.0)
            if not self.is_ready:
                self.is_ready = self._check_minimum_data()
        return self.is_ready

    def register_strategy(self, strategy) -> None:
        pass  # not needed for HL
