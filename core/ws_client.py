"""
core/ws_client.py — Persistent WebSocket connection to Hyperliquid.

Two fill queues
---------------
fill_queue      — fills from watched source traders (copy signals)
own_fill_queue  — fills from the bot's own wallet (SL/TP / close events)

The own-wallet subscription is added via subscribe_own() so the bot can
detect bracket orders firing without confusing them with trader copy signals.

Connection lifecycle
--------------------
• Exponential backoff on reconnect (2 s → 60 s cap).
• Re-subscribes all known wallets on every reconnect.
• Application-level {"method":"ping"} every 30 s to prevent the HL server
  from closing idle connections (server timeout ≈ 60 s).
• Staleness pre-filter drops fills older than max_fill_age_seconds to prevent
  the 30-day replay that fires on every (re)connect from flooding the queue.

closedPnl parsing
-----------------
The userFills WebSocket channel includes a closedPnl field on each fill:
  "0" or 0.0  →  opening fill (new position or scale-in)
  non-zero    →  closing fill (reduction or full close)

This field is parsed into FillEvent.closed_pnl and used by WatchedPositionTracker
to classify fills without relying on side-comparison alone.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from config import get_settings
from models.trader import FillEvent, Side
from utils.logger import get_logger

log = get_logger(__name__)
cfg = get_settings()

_RECONNECT_BACKOFF_BASE = 2
_RECONNECT_BACKOFF_MAX  = 60
_APP_PING_INTERVAL      = 30.0


def _safe_float(value: object) -> float:
    """Safely parse a value to float, returning 0.0 on failure."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return 0.0
        try:
            return float(v)
        except ValueError:
            return 0.0
    return 0.0


class HyperliquidWSClient:
    def __init__(
        self,
        fill_queue:     asyncio.Queue[FillEvent],
        own_fill_queue: asyncio.Queue[FillEvent],
    ) -> None:
        self._fill_queue     = fill_queue
        self._own_fill_queue = own_fill_queue
        self._own_wallet:    str | None = None

        self._subscribed:    set[str] = set()
        self._pending_sub:   set[str] = set()
        self._pending_unsub: set[str] = set()
        self._ws:   ClientConnection | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def subscribe(self, wallet: str) -> None:
        wallet = wallet.lower()
        if wallet not in self._subscribed:
            self._pending_sub.add(wallet)

    def subscribe_own(self, wallet: str) -> None:
        """
        Subscribe to the bot's own wallet. Fills from this wallet are routed
        to own_fill_queue so SL/TP bracket hits are detected separately from
        copy signals.
        """
        wallet = wallet.lower()
        self._own_wallet = wallet
        if wallet not in self._subscribed:
            self._pending_sub.add(wallet)

    def unsubscribe(self, wallet: str) -> None:
        wallet = wallet.lower()
        self._pending_unsub.add(wallet)
        self._pending_sub.discard(wallet)

    async def start(self) -> None:
        self._running = True
        self._task    = asyncio.create_task(self._run_forever(), name="ws-client")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Connection loop ───────────────────────────────────────────────────────

    async def _run_forever(self) -> None:
        backoff = _RECONNECT_BACKOFF_BASE
        while self._running:
            try:
                log.info("ws_connecting", url=cfg.ws_url)
                async with connect(
                    cfg.ws_url,
                    ping_interval = 20,
                    ping_timeout  = 10,
                    close_timeout = 5,
                ) as ws:
                    self._ws = ws
                    backoff  = _RECONNECT_BACKOFF_BASE
                    log.info("ws_connected")

                    for wallet in self._subscribed:
                        await self._send_subscribe(ws, wallet)
                    if self._subscribed:
                        trader_wallets = [w for w in self._subscribed if w != self._own_wallet]
                        log.info(
                            "ws_resubscribed",
                            total   = len(self._subscribed),
                            traders = len(trader_wallets),
                            own     = self._own_wallet[:12] if self._own_wallet else "none",
                            sample  = [w[:10] for w in trader_wallets[:4]],
                        )

                    await self._message_loop(ws)

            except ConnectionClosed as e:
                log.warning("ws_disconnected", code=e.rcvd and e.rcvd.code)
            except OSError as e:
                log.error("ws_os_error", err=str(e))
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("ws_unexpected_error", err=str(e))

            if self._running:
                log.info("ws_reconnecting", backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

        self._ws = None
        log.info("ws_stopped")

    async def _message_loop(self, ws: ClientConnection) -> None:
        import time as _time
        last_app_ping = _time.monotonic()

        while self._running:
            await self._flush_pending(ws)

            now = _time.monotonic()
            if now - last_app_ping >= _APP_PING_INTERVAL:
                try:
                    await ws.send(json.dumps({"method": "ping"}))
                    last_app_ping = now
                    log.debug("ws_app_ping_sent")
                except Exception as e:
                    log.debug("ws_app_ping_failed", err=str(e))
                    break

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("ws_bad_json", raw=str(raw)[:200])
                continue

            if msg.get("channel") == "pong":
                log.debug("ws_app_pong_received")
                continue

            await self._dispatch(msg)

    async def _flush_pending(self, ws: ClientConnection) -> None:
        while self._pending_sub:
            w = self._pending_sub.pop()
            await self._send_subscribe(ws, w)
            self._subscribed.add(w)
            label = "own" if w == self._own_wallet else "trader"
            log.info("ws_subscribed", wallet=w[:12], type=label)

        while self._pending_unsub:
            w = self._pending_unsub.pop()
            if w in self._subscribed:
                await self._send_unsubscribe(ws, w)
                self._subscribed.discard(w)
                log.info("ws_unsubscribed", wallet=w[:12])

    # ── Wire messages ─────────────────────────────────────────────────────────

    @staticmethod
    async def _send_subscribe(ws: ClientConnection, wallet: str) -> None:
        await ws.send(json.dumps({
            "method":       "subscribe",
            "subscription": {"type": "userFills", "user": wallet},
        }))

    @staticmethod
    async def _send_unsubscribe(ws: ClientConnection, wallet: str) -> None:
        await ws.send(json.dumps({
            "method":       "unsubscribe",
            "subscription": {"type": "userFills", "user": wallet},
        }))

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, msg: dict) -> None:
        if msg.get("channel") != "userFills":
            return

        data   = msg.get("data", {})
        wallet = data.get("user", "").lower()
        fills  = data.get("fills", [])

        is_own_wallet = self._own_wallet is not None and wallet == self._own_wallet
        target_queue  = self._own_fill_queue if is_own_wallet else self._fill_queue
        now_utc       = datetime.now(tz=timezone.utc)

        for raw_fill in fills:
            event = self._parse_fill(raw_fill, wallet)
            if event is None:
                continue

            if cfg.max_fill_age_seconds > 0:
                age_s = (now_utc - event.time).total_seconds()
                if age_s > cfg.max_fill_age_seconds:
                    log.debug(
                        "ws_fill_stale_dropped",
                        coin   = event.coin,
                        age_s  = round(age_s),
                        wallet = wallet[:10],
                        queue  = "own" if is_own_wallet else "trader",
                    )
                    continue

            try:
                target_queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning(
                    "ws_fill_queue_full",
                    coin  = event.coin,
                    queue = "own" if is_own_wallet else "trader",
                )
                continue

            log.debug(
                "ws_fill",
                wallet     = wallet[:10],
                coin       = event.coin,
                side       = event.side,
                px         = event.px,
                sz         = event.sz,
                closed_pnl = event.closed_pnl,
                queue      = "own" if is_own_wallet else "trader",
            )

    @staticmethod
    def _parse_fill(raw: dict, wallet: str) -> FillEvent | None:
        try:
            return FillEvent(
                coin       = raw["coin"],
                px         = float(raw["px"]),
                sz         = float(raw["sz"]),
                side       = Side(raw["side"]),
                time       = datetime.fromtimestamp(raw.get("time", 0) / 1000, tz=timezone.utc),
                crossed    = bool(raw.get("crossed", True)),
                fee        = float(raw.get("fee", 0)),
                oid        = int(raw.get("oid", 0)),
                tid        = int(raw.get("tid", 0)),
                wallet     = wallet,
                closed_pnl = _safe_float(raw.get("closedPnl")),
            )
        except (KeyError, ValueError) as e:
            log.warning("ws_fill_parse_error", err=str(e))
            return None
