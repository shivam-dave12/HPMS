"""
telegram_bot.py — HPMS Telegram Notification & Control Bot
===========================================================
Provides real-time push notifications and remote control via Telegram.
Uses python-telegram-bot v20+ (async) running in a dedicated daemon thread.
All send_message() calls from external threads are thread-safe.

All timestamps are in Indian Standard Time (IST, UTC+5:30).

Commands registered in bot menu
────────────────────────────────
/status     Live system dashboard
/position   Current open position details
/price      Current market price
/market     Market snapshot (OB, spread, depth)
/pnl        Session P&L breakdown
/risk       Risk manager state
/thinking   Last engine decision trace
/engine     Engine parameters
/halt       Halt new entries (admin)
/resume     Resume after halt (admin)
/help       Full command reference
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter
from telegram.ext import Application, CommandHandler, ContextTypes

from logger_core import elog

logger = logging.getLogger("telegram_bot")

# ── Indian Standard Time ─────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def _now_ist() -> str:
    """HH:MM:SS IST — used consistently in every bot response."""
    return datetime.now(_IST).strftime("%H:%M:%S IST")


def _date_ist() -> str:
    """DD Mon YYYY HH:MM IST."""
    return datetime.now(_IST).strftime("%d %b %Y  %H:%M IST")


# ── Bot menu commands ─────────────────────────────────────────────────────────
_BOT_COMMANDS: List[BotCommand] = [
    BotCommand("status",    "Live system dashboard"),
    BotCommand("position",  "Current open position"),
    BotCommand("price",     "Current market price"),
    BotCommand("market",    "Market snapshot: OB, spread, depth"),
    BotCommand("pnl",       "Session P&L breakdown"),
    BotCommand("risk",      "Risk manager state and limits"),
    BotCommand("thinking",  "Last engine decision trace"),
    BotCommand("engine",    "Engine parameters"),
    BotCommand("halt",      "Halt new entries — admin only"),
    BotCommand("resume",    "Resume after halt — admin only"),
    BotCommand("help",      "Full command reference"),
]


# ═════════════════════════════════════════════════════════════════════════════
# TelegramBot
# ═════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    """
    Thread-safe Telegram notification and control bot.

    Lifecycle
    ─────────
    1. Constructed with component references in HPMSRunner.start().
    2. start()  — launches a daemon thread that owns an asyncio event loop,
                  initialises the Application, registers command handlers,
                  pushes the command list to Telegram, then starts polling.
    3. send_message(text) — schedules a coroutine on the bot thread via
                  asyncio.run_coroutine_threadsafe; non-blocking for callers.
    4. stop()   — signals the async stop-event, waits for polling teardown.
    """

    def __init__(
        self,
        token:     str,
        chat_id:   int | str,
        admin_ids: List[int],
        strategy,
        engine,
        risk_mgr,
        order_mgr,
        data_mgr,
        api,
        config,
    ) -> None:
        self._token     = token
        self._chat_id   = int(chat_id) if chat_id else 0
        self._admin_ids = set(int(x) for x in (admin_ids or []))

        # Component references (strategy may be None at construction time and
        # injected via attribute assignment before start() is called)
        self._strategy = strategy
        self._engine   = engine
        self._risk     = risk_mgr
        self._orders   = order_mgr
        self._data     = data_mgr
        self._api      = api
        self._config   = config

        # Threading / event loop
        self._loop:       Optional[asyncio.AbstractEventLoop] = None
        self._thread:     Optional[threading.Thread]          = None
        self._app:        Optional[Application]               = None
        self._stop_async: Optional[asyncio.Event]             = None
        self._ready       = threading.Event()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Launch bot polling in a background daemon thread."""
        if not self._token:
            logger.warning("TelegramBot: no token configured — notifications disabled")
            return
        self._thread = threading.Thread(
            target=self._thread_main, daemon=True, name="TelegramBot"
        )
        self._thread.start()
        if not self._ready.wait(timeout=20):
            logger.error("TelegramBot: failed to become ready within 20 s")

    def stop(self) -> None:
        """Signal the async stop-event and wait for the polling thread."""
        if self._loop and self._stop_async and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_async.set)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=12)

    def send_message(self, text: str) -> None:
        """
        Thread-safe fire-and-forget: schedule a Markdown message on the bot thread.
        Returns immediately.  Delivery errors are logged at DEBUG level only.
        """
        if not self._loop or not self._loop.is_running() or not self._chat_id:
            return
        asyncio.run_coroutine_threadsafe(self._send(text), self._loop)

    # ══════════════════════════════════════════════════════════════════════════
    # BOT THREAD
    # ══════════════════════════════════════════════════════════════════════════

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            logger.error("TelegramBot thread crashed: %s", exc, exc_info=True)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_main(self) -> None:
        self._stop_async = asyncio.Event()
        self._app = Application.builder().token(self._token).build()

        self._register_handlers()

        async with self._app:
            try:
                await self._app.bot.set_my_commands(_BOT_COMMANDS)
                logger.info("TelegramBot: %d commands registered in bot menu", len(_BOT_COMMANDS))
            except Exception as exc:
                logger.warning("TelegramBot: set_my_commands failed: %s", exc)

            await self._app.start()
            await self._app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=["message"],
            )
            self._ready.set()
            elog.log("SYSTEM_START", component="TelegramBot", status="polling_active")

            await self._stop_async.wait()

            await self._app.updater.stop()
            await self._app.stop()

        elog.log("SYSTEM_SHUTDOWN", component="TelegramBot", status="polling_stopped")

    async def _send(self, text: str) -> None:
        """
        Internal delivery coroutine.

        Falls back to plain text if Markdown parsing fails (e.g. unclosed
        backtick from a runtime value containing a backtick character).
        """
        if not self._app:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except BadRequest as exc:
            # Markdown parse error — retry as plain text
            logger.debug("TelegramBot: Markdown send failed (%s), retrying plain", exc)
            try:
                plain = text.replace("*", "").replace("`", "").replace("_", "")
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=plain,
                    disable_web_page_preview=True,
                )
            except Exception as inner:
                logger.debug("TelegramBot: plain-text fallback also failed: %s", inner)
        except RetryAfter as exc:
            logger.warning("TelegramBot: rate-limited, retry after %.0fs", exc.retry_after)
            await asyncio.sleep(exc.retry_after + 0.5)
            await self._send(text)
        except NetworkError as exc:
            logger.debug("TelegramBot: network error: %s", exc)
        except Exception as exc:
            logger.debug("TelegramBot: send failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════════
    # HANDLER REGISTRATION
    # ══════════════════════════════════════════════════════════════════════════

    def _register_handlers(self) -> None:
        add = self._app.add_handler
        add(CommandHandler("start",      self._cmd_start))
        add(CommandHandler("help",       self._cmd_help))
        add(CommandHandler("status",     self._cmd_status))
        add(CommandHandler("position",   self._cmd_position))
        add(CommandHandler("price",      self._cmd_price))
        add(CommandHandler("market",     self._cmd_market))
        add(CommandHandler("pnl",        self._cmd_pnl))
        add(CommandHandler("overallpnl", self._cmd_pnl))      # legacy alias
        add(CommandHandler("risk",       self._cmd_risk))
        add(CommandHandler("thinking",   self._cmd_thinking))
        add(CommandHandler("engine",     self._cmd_engine))
        add(CommandHandler("halt",       self._cmd_halt))
        add(CommandHandler("resume",     self._cmd_resume))

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _is_admin(self, update: Update) -> bool:
        """Return True when admin_ids is empty (open) or the user is listed."""
        uid = update.effective_user.id if update.effective_user else None
        return (not self._admin_ids) or (uid in self._admin_ids)

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        """Reply to a command message with Markdown formatting."""
        await update.message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )

    @staticmethod
    def _safe_get(d: dict, *keys, default=None):
        """Nested dict get with a safe default."""
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d

    # ══════════════════════════════════════════════════════════════════════════
    # COMMAND HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    # ── /start ────────────────────────────────────────────────────────────────
    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        symbol = getattr(self._config, "DELTA_SYMBOL", "BTCUSD")
        await self._reply(update, (
            "🤖 *HPMS Trading Bot*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Hamiltonian Phase-Space Micro-Scalping\n"
            f"Instrument: `{symbol}`\n\n"
            "Quick access:\n"
            "  /status    live dashboard\n"
            "  /position  current trade\n"
            "  /pnl       session P&L\n"
            "  /thinking  last engine trace\n\n"
            "Use /help for the full command list."
        ))

    # ── /help ─────────────────────────────────────────────────────────────────
    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self._reply(update, (
            "📖 *Command Reference*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "*Market Data*\n"
            "  /price    last-trade price\n"
            "  /market   OB snapshot, spread, depth\n\n"
            "*Strategy*\n"
            "  /status   full system dashboard\n"
            "  /position current open position\n"
            "  /thinking last engine decision trace\n"
            "  /engine   engine parameters\n\n"
            "*P&L and Risk*\n"
            "  /pnl      session P&L breakdown\n"
            "  /risk     risk manager state\n\n"
            "*Admin*\n"
            "  /halt     halt new entries\n"
            "  /resume   resume after halt\n"
        ))

    # ── /status ───────────────────────────────────────────────────────────────
    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            st  = self._strategy.get_status() if self._strategy else {}
            rs  = st.get("risk",     {})
            pos = st.get("position", {})
            sig = st.get("last_signal", {})

            from state import STATE
            mode_icon = "🧪" if STATE.dry_run  else "⚡"
            net_icon  = "🔬" if STATE.testnet  else "🌐"
            strat_str = "✅ Active" if st.get("enabled") else "⏸ Stopped"

            # Halt banner
            halt_line = ""
            if rs.get("halted"):
                reason    = rs.get("halt_reason", "unknown")
                halt_line = f"\n🚨 *HALTED* — `{reason}`"

            # Position summary
            if pos.get("in_position"):
                side    = pos.get("side", "?").upper()
                entry   = pos.get("entry_price", 0.0)
                bars    = pos.get("bars_held", 0)
                size    = pos.get("size", 0)
                pos_line = (
                    f"\n📍 `{side}` {size}c @ `${entry:,.1f}`"
                    f"  ·  {bars} bars held"
                )
            else:
                pos_line = "\n📭 Flat — no open position"

            # Last signal summary
            sig_line = ""
            if sig.get("type") and sig["type"] != "FLAT":
                sig_icon  = "🟢" if sig["type"] == "LONG" else "🔴"
                sig_conf  = sig.get("confidence") or 0.0
                sig_dq    = sig.get("delta_q")    or 0.0
                sig_line  = (
                    f"\n{sig_icon} Last signal: `{sig['type']}`"
                    f"  conf `{sig_conf:.1%}`"
                    f"  Δq `{sig_dq:+.5f}`"
                )

            # Daily P&L
            daily_pnl  = rs.get("daily_pnl",       0.0)
            daily_fees = rs.get("daily_fees",       0.0)
            n_trades   = rs.get("trades_today",     0)
            peak       = rs.get("session_high_pnl", 0.0)
            c_loss     = rs.get("consecutive_losses", 0)
            pnl_icon   = "💰" if daily_pnl >= 0 else "🔻"

            await self._reply(update, (
                f"{mode_icon} *System Status*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Strategy: {strat_str}{halt_line}\n"
                f"{net_icon} Mode: `{STATE.mode}`"
                f"  ·  Bars: `{st.get('bar_count', 0):,}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{pnl_icon} *Session P&L*\n"
                f"  Trades:      `{n_trades}`\n"
                f"  Net P&L:     `${daily_pnl:+.4f}`\n"
                f"  Fees:        `$-{daily_fees:.4f}`\n"
                f"  Session high: `${peak:+.4f}`\n"
                f"  Consec loss:  `{c_loss}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                f"{pos_line}"
                f"{sig_line}"
            ))
        except Exception as exc:
            logger.error("/status error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /position ─────────────────────────────────────────────────────────────
    async def _cmd_position(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            pos = self._orders.get_status() if self._orders else {}

            if not pos.get("in_position"):
                await self._reply(update, (
                    f"📭 *No Open Position*  `{_now_ist()}`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "Strategy is flat — awaiting the next signal."
                ))
                return

            side   = pos.get("side", "?").upper()
            size   = pos.get("size", 0)
            entry  = pos.get("entry_price", 0.0)
            tp     = pos.get("tp_price",    0.0)
            sl     = pos.get("sl_price",    0.0)
            bars   = pos.get("bars_held",   0)

            last_px = self._data.get_last_price() if self._data else 0.0

            # Unrealised P&L
            cv     = getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001)
            lev    = getattr(self._config, "RISK_LEVERAGE", 10)
            unreal = 0.0
            if last_px and entry:
                move   = (last_px - entry) if side == "LONG" else (entry - last_px)
                unreal = move * cv * size

            # Geometry
            tp_dist = abs(tp - entry) if tp and entry else 0.0
            sl_dist = abs(sl - entry) if sl and entry else 0.0
            rr      = tp_dist / sl_dist if sl_dist > 0 else 0.0

            # TP progress
            progress_line = ""
            if tp and sl and entry and last_px:
                if side == "LONG" and (tp - entry) > 0:
                    prog = (last_px - entry) / (tp - entry)
                elif side == "SHORT" and (entry - sl) > 0:
                    prog = (entry - last_px) / (entry - sl)
                else:
                    prog = 0.0
                prog = max(0.0, min(1.0, prog))
                progress_line = f"\nProgress to TP: `{prog:.0%}`"

            unreal_icon = "📈" if unreal >= 0 else "📉"
            notional    = entry * cv * size
            margin      = notional / max(lev, 1)

            await self._reply(update, (
                f"📍 *Open {side}*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Size:      `{size}c` @ `${entry:,.2f}`\n"
                f"Notional:  `${notional:,.2f}`"
                f"  ·  Margin: `${margin:.2f}` @ `{lev}x`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Current:   `${last_px:,.2f}`\n"
                f"{unreal_icon} Unrealised: `${unreal:+.4f}`\n"
                f"Held:      `{bars}` bars\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"TP: `${tp:,.2f}` (+`${tp_dist:.2f}`)  "
                f"SL: `${sl:,.2f}` (-`${sl_dist:.2f}`)\n"
                f"R:R: `{rr:.2f}:1`"
                f"{progress_line}"
            ))
        except Exception as exc:
            logger.error("/position error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /price ────────────────────────────────────────────────────────────────
    async def _cmd_price(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            price  = self._data.get_last_price() if self._data else 0.0
            symbol = getattr(self._config, "DELTA_SYMBOL", "BTCUSD")
            fresh  = self._data.is_price_fresh(max_stale_seconds=10) if self._data else False
            feed   = "🟢" if fresh else "🔴"

            if not price:
                await self._reply(update, (
                    f"⚠️ *{symbol}*  `{_now_ist()}`\n"
                    "Price not available — feed may be initialising."
                ))
                return

            await self._reply(update, (
                f"💹 *{symbol}*  `{_now_ist()}`\n"
                f"Last price: `${price:,.2f}`  Feed: {feed}"
            ))
        except Exception as exc:
            logger.error("/price error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /market ───────────────────────────────────────────────────────────────
    async def _cmd_market(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            symbol = getattr(self._config, "DELTA_SYMBOL", "BTCUSD")
            price  = self._data.get_last_price() if self._data else 0.0
            ob     = self._data.get_orderbook()  if self._data else {}
            bids   = ob.get("bids", [])
            asks   = ob.get("asks", [])
            fresh  = self._data.is_price_fresh(max_stale_seconds=30) if self._data else False
            feed   = "🟢" if fresh else "🔴"

            ob_section = ""
            if bids and asks:
                try:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spread   = best_ask - best_bid
                    sp_pct   = spread / best_bid * 100 if best_bid > 0 else 0.0
                    bid5_vol = sum(float(b[1]) for b in bids[:5])
                    ask5_vol = sum(float(a[1]) for a in asks[:5])
                    total    = bid5_vol + ask5_vol
                    imb      = bid5_vol / total if total > 0 else 0.5
                    imb_icon = "🟢" if imb > 0.55 else ("🔴" if imb < 0.45 else "⚪")

                    ob_section = (
                        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "*Order Book*\n"
                        f"  Best bid: `${best_bid:,.2f}`"
                        f"  Best ask: `${best_ask:,.2f}`\n"
                        f"  Spread:   `${spread:.2f}` (`{sp_pct:.4f}%`)\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "*Depth (top 5 levels)*\n"
                        f"  Bid vol: `{bid5_vol:.4f}`"
                        f"  Ask vol: `{ask5_vol:.4f}`\n"
                        f"  {imb_icon} Imbalance: `{imb:.1%}` bid-side"
                    )
                except (IndexError, ValueError, TypeError):
                    ob_section = "\n⚠️ Order book data unavailable"

            await self._reply(update, (
                f"📊 *Market Snapshot — {symbol}*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Last price: `${price:,.2f}`  Feed: {feed}"
                f"{ob_section}"
            ))
        except Exception as exc:
            logger.error("/market error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /pnl  (also handles /overallpnl) ─────────────────────────────────────
    async def _cmd_pnl(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            rs        = self._risk.get_status()         if self._risk else {}
            trade_log = self._risk.get_trade_log(10)    if self._risk else []

            net_pnl    = rs.get("daily_pnl",           0.0)
            total_fees = rs.get("daily_fees",           0.0)
            n_trades   = rs.get("trades_today",         0)
            peak_pnl   = rs.get("session_high_pnl",    0.0)
            c_loss     = rs.get("consecutive_losses",  0)
            gross_pnl  = net_pnl + total_fees

            result_icon = "💰" if net_pnl >= 0 else "🔻"
            win_count   = sum(1 for t in trade_log if t.get("net_pnl", 0) >= 0)
            loss_count  = len(trade_log) - win_count
            win_rate    = win_count / len(trade_log) if trade_log else 0.0

            # Per-trade breakdown
            trade_rows = ""
            if trade_log:
                rows = []
                for t in trade_log:
                    net    = t.get("net_pnl",     0.0)
                    side   = (t.get("side",       "?")[:1]).upper()
                    entry  = t.get("entry_price", 0.0)
                    exit_  = t.get("exit_price",  0.0)
                    reason = (t.get("reason",     "?"))[:10]
                    icon   = "✅" if net >= 0 else "❌"
                    rows.append(
                        f"  {icon} `{side}`"
                        f" `${entry:,.0f}`→`${exit_:,.0f}`"
                        f" `${net:+.4f}`"
                        f" _{reason}_"
                    )
                trade_rows = (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "*Recent Trades*\n"
                    + "\n".join(rows)
                )

            await self._reply(update, (
                f"{result_icon} *Session P&L*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Trades:       `{n_trades}`\n"
                f"Gross P&L:    `${gross_pnl:+.4f}`\n"
                f"Total fees:   `$-{total_fees:.4f}`\n"
                f"Net P&L:      `${net_pnl:+.4f}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Session peak: `${peak_pnl:+.4f}`\n"
                f"Consec loss:  `{c_loss}`\n"
                f"Win rate:     `{win_rate:.0%}`"
                f" ({win_count}W / {loss_count}L)"
                f"{trade_rows}"
            ))
        except Exception as exc:
            logger.error("/pnl error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /risk ─────────────────────────────────────────────────────────────────
    async def _cmd_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            rs = self._risk.get_status() if self._risk else {}

            halted    = rs.get("halted", False)
            state_str = "🚨 HALTED" if halted else "✅ Active"
            halt_line = ""
            if halted:
                reason    = rs.get("halt_reason",        "unknown")
                remaining = rs.get("cooldown_remaining", 0.0)
                halt_line = (
                    f"\nReason:        `{reason}`"
                    + (f"\nCooldown left: `{remaining:.0f}s`" if remaining > 0 else "")
                )

            await self._reply(update, (
                f"🛡 *Risk Manager*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"State: {state_str}"
                f"{halt_line}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "*Configured Limits*\n"
                f"  Max loss/day:     `${rs.get('max_daily_loss', 0):.2f}`\n"
                f"  Max trades/day:   `{rs.get('max_daily_trades', 0)}`\n"
                f"  Max consec loss:  `{rs.get('max_consecutive_losses', 0)}`\n"
                f"  Cooldown:         `{rs.get('cooldown_seconds', 0)}s`\n"
                f"  Max drawdown:     `{rs.get('max_drawdown_pct', 0):.1f}%`\n"
                f"  Equity per trade: `{rs.get('equity_pct_per_trade', 0):.1f}%`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "*Session Counters*\n"
                f"  Trades today:     `{rs.get('trades_today', 0)}`\n"
                f"  Net P&L:          `${rs.get('daily_pnl', 0.0):+.4f}`\n"
                f"  Fees paid:        `$-{rs.get('daily_fees', 0.0):.4f}`\n"
                f"  Consec losses:    `{rs.get('consecutive_losses', 0)}`"
            ))
        except Exception as exc:
            logger.error("/risk error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /thinking ─────────────────────────────────────────────────────────────
    async def _cmd_thinking(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            st  = self._strategy.get_status() if self._strategy else {}
            sig = st.get("last_signal", {})

            if not sig.get("type") or sig["type"] == "FLAT":
                await self._reply(update, (
                    f"🧠 *Engine Decision Trace*  `{_now_ist()}`\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "No actionable signal on record.\n"
                    f"Bars processed: `{st.get('bar_count', 0):,}`\n\n"
                    "_Engine may still be priming — "
                    "typically ready after 50+ bars._"
                ))
                return

            sig_type = sig.get("type",       "FLAT")
            conf     = sig.get("confidence") or 0.0
            delta_q  = sig.get("delta_q")    or 0.0
            reason   = sig.get("reason")     or "—"
            compute  = sig.get("compute_us") or 0.0
            ep       = self._engine.get_params() if self._engine else {}

            sig_icon = {"LONG": "🟢", "SHORT": "🔴"}.get(sig_type, "⬜")
            conf_bar = _confidence_bar(conf)

            await self._reply(update, (
                f"🧠 *Engine Decision Trace*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Signal:     {sig_icon} `{sig_type}`\n"
                f"Confidence: `{conf:.1%}`  {conf_bar}\n"
                f"Delta-q:    `{delta_q:+.6f}`\n"
                f"Reason:     _{reason}_\n"
                f"Compute:    `{compute:.0f} µs`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "*Engine State*\n"
                f"  Lookback:    `{ep.get('lookback', '?')}` bars\n"
                f"  Horizon:     `{ep.get('prediction_horizon', '?')}` bars\n"
                f"  KDE bw:      `{ep.get('kde_bandwidth', '?')}`\n"
                f"  Integrator:  `{ep.get('integrator', '?')}`\n"
                f"  Bars in:     `{st.get('bar_count', 0):,}`"
            ))
        except Exception as exc:
            logger.error("/thinking error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /engine ───────────────────────────────────────────────────────────────
    async def _cmd_engine(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            ep = self._engine.get_params() if self._engine else {}
            await self._reply(update, (
                f"⚙️ *Engine Parameters*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "*Phase Space*\n"
                f"  τ (tau):        `{ep.get('tau', '?')}`\n"
                f"  Lookback:       `{ep.get('lookback', '?')}` bars\n"
                f"  Horizon:        `{ep.get('prediction_horizon', '?')}` bars\n"
                f"  Mass:           `{ep.get('mass', '?')}`\n"
                "*KDE*\n"
                f"  Bandwidth:      `{ep.get('kde_bandwidth', '?')}`\n"
                f"  Grid points:    `{ep.get('kde_grid_points', '?')}`\n"
                f"  Rebuild every:  `{ep.get('kde_rebuild_interval', '?')}` bars\n"
                "*Integration*\n"
                f"  Integrator:     `{ep.get('integrator', '?')}`\n"
                f"  dt:             `{ep.get('integration_dt', '?')}`\n"
                "*Signal Thresholds*\n"
                f"  ΔQ threshold:   `{ep.get('delta_q_threshold', '?')}`\n"
                f"  dH/dt max:      `{ep.get('dH_dt_max', '?')}`\n"
                f"  Min momentum:   `{ep.get('min_momentum', '?')}`\n"
                f"  H percentile:   `{ep.get('H_percentile', '?')}`"
            ))
        except Exception as exc:
            logger.error("/engine error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /halt (admin) ─────────────────────────────────────────────────────────
    async def _cmd_halt(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_admin(update):
            await self._reply(update, "⛔ Unauthorized — admin access required.")
            return
        try:
            halted_ok = False
            if self._risk and hasattr(self._risk, "manual_halt"):
                self._risk.manual_halt("MANUAL_TELEGRAM_HALT")
                halted_ok = True

            user = (
                f"@{update.effective_user.username}"
                if update.effective_user and update.effective_user.username
                else f"uid:{update.effective_user.id if update.effective_user else '?'}"
            )
            extra = "" if halted_ok else "\n⚠️ `manual_halt` not available on this risk build."

            await self._reply(update, (
                f"🚨 *Trading HALTED*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Issued by: `{user}`\n"
                "No new entries will be opened.\n"
                "Open positions continue to be managed.\n"
                f"{extra}\n"
                "Use /resume to re-enable."
            ))
        except Exception as exc:
            logger.error("/halt error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")

    # ── /resume (admin) ───────────────────────────────────────────────────────
    async def _cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_admin(update):
            await self._reply(update, "⛔ Unauthorized — admin access required.")
            return
        try:
            resumed_ok = False
            if self._risk and hasattr(self._risk, "manual_resume"):
                self._risk.manual_resume()
                resumed_ok = True

            user = (
                f"@{update.effective_user.username}"
                if update.effective_user and update.effective_user.username
                else f"uid:{update.effective_user.id if update.effective_user else '?'}"
            )
            extra = "" if resumed_ok else "\n⚠️ `manual_resume` not available on this risk build."

            await self._reply(update, (
                f"✅ *Trading RESUMED*  `{_now_ist()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Issued by: `{user}`\n"
                "Strategy is active — scanning for signals."
                f"{extra}"
            ))
        except Exception as exc:
            logger.error("/resume error: %s", exc, exc_info=True)
            await self._reply(update, f"⚠️ Internal error: `{exc}`")


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _confidence_bar(conf: float, width: int = 10) -> str:
    """
    Convert a 0–1 confidence value to a compact visual bar.

    0.87 → [████████░░]  (8 filled out of 10)
    """
    filled = round(conf * width)
    filled = max(0, min(width, filled))
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"
