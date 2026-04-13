"""
telegram_bot.py — HPMS Telegram Control Interface
===================================================
Full remote control and visibility into the HPMS trading system.

INFO COMMANDS
  /status        — Full dashboard (strategy, risk, position, last signal)
  /thinking      — HPMS decision stack: why did/didn't we trade last bar?
  /phase         — Phase-space state (q, p, H, K, V, dH/dt)
  /signal        — Last signal detail (type, conf, Δq, p_final, TP/SL)
  /diag          — Engine diagnostics (KDE, trajectory, H-smoothing)
  /engine        — All engine parameters at a glance
  /market        — Price, spread, orderbook depth, volatility, ATR
  /filter        — Current bar vs filter thresholds (pass/fail each gate)
  /position      — Open position detail + unrealised PnL
  /pnl           — Daily P&L summary
  /trades        — Recent trade log (last 10)
  /balance       — Exchange wallet balance
  /risk          — Risk gate status: what's blocking / what's open
  /orderbook     — Top-5 orderbook snapshot
  /price         — Current price + spread
  /ping          — Latency check

TRADING CONTROLS
  /start_trading — Enable strategy (new entries allowed)
  /stop_trading  — Disable strategy (no new entries; open positions remain)
  /halt          — Emergency: stop strategy + cancel all orders + flatten
  /resume        — Clear risk halt + reset consecutive-loss counter
  /resetrisk     — Alias for /resume (clear lockout only, no flatten)
  /flatten       — Cancel all orders + close all positions (no strategy stop)
  /close         — Close current position at market

PARAMETER CONTROLS
  /params             — List all tunable parameters with current values
  /set <p> <v>        — Set any parameter (engine, risk, or config)
  /get <p>            — Get current value of any parameter
  /set_engine <p> <v> — Set engine parameter
  /set_risk   <p> <v> — Set risk parameter
  /set_trade  <p> <v> — Set TRADE_* config value
  /set_filter <p> <v> — Set FILTER_* config value
  /leverage <N>       — Set exchange leverage (or view current if no arg)
  /cooldown <sec>     — Set inter-trade cooldown
  /maxloss <usd>      — Set daily loss limit
  /maxsize <n>        — Set max position contracts

  /help               — Command list
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    from telegram import Update, BotCommand
    from telegram.ext import (
        Application, CommandHandler, ContextTypes, MessageHandler, filters,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    logger.warning("python-telegram-bot not installed — Telegram controls disabled")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _pct_bar(value: float, lo: float, hi: float, width: int = 10) -> str:
    """Return a Unicode fill-bar for value in [lo, hi]."""
    if hi <= lo:
        return "░" * width
    frac  = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def _gate(ok: bool) -> str:
    return "✅" if ok else "❌"


def _mk(text: str) -> str:
    """Escape Markdown special chars in dynamic text fragments."""
    for ch in r"\_*[]()~`>#+=|{}.!-":
        text = text.replace(ch, f"\\{ch}")
    return text


def _md_safe(text: str) -> str:
    """Strip characters that break Telegram Markdown v1 code-spans (backticks, asterisks)."""
    return str(text).replace("`", "'").replace("*", "x").replace("_", "-")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT
# ══════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    """
    Thread-safe Telegram command interface for the HPMS trading system.

    Lifecycle:
      • start()       — called from main thread; spawns bot in daemon thread,
                        blocks up to 15 s until the event loop is live.
      • send_message()— thread-safe push; buffers until loop is ready.
      • stop()        — graceful shutdown.
    """

    def __init__(
        self,
        token:     str,
        chat_id:   str,
        admin_ids: list,
        strategy=None, engine=None, risk_mgr=None,
        order_mgr=None, data_mgr=None, api=None, config=None,
    ):
        self._token    = token
        self._chat_id  = str(chat_id) if chat_id else ""
        self._admin_ids: Set[int] = set(admin_ids or [])
        self._strategy = strategy
        self._engine   = engine
        self._risk     = risk_mgr
        self._orders   = order_mgr
        self._data     = data_mgr
        self._api      = api
        self._config   = config

        self._app:    Optional[Any]                         = None
        self._thread: Optional[threading.Thread]           = None
        self._loop:   Optional[asyncio.AbstractEventLoop]  = None
        self._running = False
        self._ready   = threading.Event()   # set once the bot loop is live

        # Periodic-report state
        self._periodic_thread: Optional[threading.Thread] = None
        self._report_interval  = 900         # 15 min default; overridable via config
        self._last_report_ts   = 0.0

    # ─── AUTH ─────────────────────────────────────────────────────────────────

    def _is_admin(self, update: Update) -> bool:
        uid = update.effective_user.id if update.effective_user else 0
        return (not self._admin_ids) or (uid in self._admin_ids)

    async def _guard(self, update: Update) -> bool:
        if not self._is_admin(update):
            await update.message.reply_text("⛔ Unauthorized")
            return False
        return True

    # ─── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self):
        if not HAS_TELEGRAM or not self._token:
            logger.warning("Telegram bot not started (no token or library missing)")
            return

        if self._config:
            self._report_interval = int(
                getattr(self._config, "TELEGRAM_REPORT_INTERVAL_SEC", 900)
            )

        self._thread = threading.Thread(
            target=self._run_bot, daemon=True, name="TelegramBot"
        )
        self._thread.start()
        if not self._ready.wait(timeout=15):
            logger.warning("Telegram bot did not become ready in 15s — continuing anyway")

        # Start periodic report thread
        self._periodic_thread = threading.Thread(
            target=self._periodic_report_loop, daemon=True, name="TelegramPeriodicReport"
        )
        self._periodic_thread.start()

    def _run_bot(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_start())

    async def _async_start(self):
        self._app = Application.builder().token(self._token).build()

        # ── Register command handlers ─────────────────────────────────────────
        cmds = {
            # Info
            "start":         self._cmd_start,
            "help":          self._cmd_start,
            "ping":          self._cmd_ping,
            "status":        self._cmd_status,
            "thinking":      self._cmd_thinking,
            "signal":        self._cmd_signal,
            "phase":         self._cmd_phase,
            "diag":          self._cmd_diag,
            "engine":        self._cmd_engine,
            "market":        self._cmd_market,
            "filter":        self._cmd_filter,
            "trades":        self._cmd_trades,
            "pnl":           self._cmd_pnl,
            "balance":       self._cmd_balance,
            "position":      self._cmd_position,
            "risk":          self._cmd_risk,
            "orderbook":     self._cmd_orderbook,
            "price":         self._cmd_price,
            # Controls
            "start_trading": self._cmd_start_trading,
            "stop_trading":  self._cmd_stop_trading,
            "halt":          self._cmd_halt,
            "resume":        self._cmd_resume,
            "resetrisk":     self._cmd_resetrisk,
            "flatten":       self._cmd_flatten,
            "close":         self._cmd_close,
            # Parameters
            "set":           self._cmd_set,
            "get":           self._cmd_get,
            "params":        self._cmd_params,
            "set_engine":    self._cmd_set_engine,
            "set_risk":      self._cmd_set_risk,
            "set_trade":     self._cmd_set_trade,
            "set_filter":    self._cmd_set_filter,
            "leverage":      self._cmd_leverage,
            "cooldown":      self._cmd_cooldown,
            "maxloss":       self._cmd_maxloss,
            "maxsize":       self._cmd_maxsize,
        }
        for name, handler in cmds.items():
            self._app.add_handler(CommandHandler(name, handler))

        # ── Global error handler — surfaces exceptions instead of silent failure ──
        async def _error_handler(update, context):
            logger.error(f"Telegram handler error: {context.error}", exc_info=context.error)
            if update and update.effective_message:
                try:
                    await update.effective_message.reply_text(
                        f"❌ Command error: {context.error}"
                    )
                except Exception:
                    pass

        self._app.add_error_handler(_error_handler)

        # ── Catch-all for non-command messages ────────────────────────────────────
        async def _fallback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
            if update.message and update.message.text:
                await update.message.reply_text(
                    "Use /help to see available commands."
                )

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _fallback))

        # ── Initialize and start FIRST (required before set_my_commands) ────────
        await self._app.initialize()
        await self._app.start()

        # ── Bot command menu (AFTER initialize — bot must be set up first) ────
        try:
            await asyncio.wait_for(
                self._app.bot.set_my_commands([
                    BotCommand("status",        "Full system dashboard"),
                    BotCommand("thinking",      "HPMS decision stack — why we trade or not"),
                    BotCommand("phase",         "Phase-space state (q, p, H, dH/dt)"),
                    BotCommand("signal",        "Last signal detail"),
                    BotCommand("market",        "Price, spread, ATR, data readiness"),
                    BotCommand("filter",        "Filter gate pass/fail vs market"),
                    BotCommand("pnl",           "Daily P&L summary"),
                    BotCommand("trades",        "Recent trade log"),
                    BotCommand("position",      "Open position + unrealised PnL"),
                    BotCommand("risk",          "Risk gate status"),
                    BotCommand("balance",       "Exchange wallet balance"),
                    BotCommand("price",         "Current price + spread"),
                    BotCommand("start_trading", "Enable trading"),
                    BotCommand("stop_trading",  "Disable trading (positions remain)"),
                    BotCommand("halt",          "Emergency halt + flatten"),
                    BotCommand("resume",        "Resume from halt"),
                    BotCommand("resetrisk",     "Clear lockout (no flatten)"),
                    BotCommand("flatten",       "Close all positions + cancel orders"),
                    BotCommand("params",        "All tunable parameters"),
                    BotCommand("set",           "Set a parameter live"),
                    BotCommand("leverage",      "View / set exchange leverage"),
                    BotCommand("help",          "Command list"),
                ]),
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"set_my_commands failed (non-fatal, commands still work): {e}")

        # ── Start polling (v20+: updater.start_polling is non-blocking) ───────
        await self._app.updater.start_polling(drop_pending_updates=True)

        self._running = True
        logger.info("Telegram bot online")
        self._ready.set()  # unblock TelegramBot.start()

        # Keep alive until stop() is called
        while self._running:
            await asyncio.sleep(1)

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def stop(self):
        self._running = False

    # ─── PUSH NOTIFICATION ────────────────────────────────────────────────────

    def send_message(self, text: str):
        """
        Thread-safe push to the configured chat.
        Blocks until the bot loop is ready (up to 20 s on first call at startup).
        Silently drops if the loop never started.
        Falls back to plain text if Markdown parsing fails.
        """
        if not self._chat_id:
            return
        if not self._ready.wait(timeout=20):
            return
        if not self._app or not self._loop:
            return

        async def _send_with_fallback():
            try:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="Markdown",
                    disable_web_page_preview=True,
                )
            except Exception:
                # Markdown parse failed — retry as plain text
                try:
                    plain = text.replace("`", "").replace("*", "").replace("_", "")
                    await self._app.bot.send_message(
                        chat_id=self._chat_id,
                        text=plain,
                        disable_web_page_preview=True,
                    )
                except Exception as e2:
                    logger.debug(f"send_message fallback error: {e2}")

        try:
            asyncio.run_coroutine_threadsafe(_send_with_fallback(), self._loop)
        except Exception as e:
            logger.debug(f"send_message error: {e}")

    # ─── PERIODIC REPORT ──────────────────────────────────────────────────────

    def _periodic_report_loop(self):
        """Background thread — sends a status report every N seconds."""
        # Wait until bot is ready before starting
        if not self._ready.wait(timeout=30):
            return
        time.sleep(60)   # let the system warm up first
        while self._running:
            try:
                interval = int(getattr(self._config, "TELEGRAM_REPORT_INTERVAL_SEC",
                                       self._report_interval))
                now = time.time()
                if now - self._last_report_ts >= interval:
                    self._last_report_ts = now
                    report = self._build_periodic_report()
                    if report:
                        self.send_message(report)
            except Exception as e:
                logger.debug(f"Periodic report error: {e}")
            time.sleep(60)

    def _build_periodic_report(self) -> str:
        """Build a rich periodic status report with clear system intent."""
        try:
            s     = self._strategy.get_status() if self._strategy else {}
            risk  = s.get("risk", {})
            pos   = s.get("position", {})
            price = self._data.get_last_price() if self._data else 0
            p     = risk.get("params", {})

            # ── Status icons ──────────────────────────────────────────────────
            strat_on  = s.get("enabled", False)
            is_halted = risk.get("is_halted", False)
            net_pnl   = risk.get("daily_pnl", 0)

            strat_icon = "🟢" if strat_on else "⏸"
            risk_icon  = "🚨" if is_halted else "🛡"
            pnl_icon   = "💰" if net_pnl >= 0 else "🔻"

            # ── Position block ────────────────────────────────────────────────
            if pos.get("in_position"):
                side = pos["side"].upper()
                side_icon = "📈" if side == "LONG" else "📉"
                pos_txt = (
                    f"{side_icon} *{side}*  {pos['size']}c @ `${pos['entry_price']:,.1f}`\n"
                    f"   Held `{pos['bars_held']}` bars"
                )
            else:
                pos_txt = "⬜ Flat — watching for setup"

            # ── What is the engine thinking right now? ────────────────────────
            last_sig = s.get("last_signal", {})
            sig_type = last_sig.get("type") or "—"
            sig_conf = last_sig.get("confidence") or 0
            if sig_type in ("LONG", "SHORT"):
                thinking = f"🎯 Signal: *{sig_type}* @ `{sig_conf:.0%}` conf"
            elif sig_type == "FLAT":
                thinking = f"👀 Engine output: *FLAT* — no momentum edge detected"
            else:
                thinking = "⏳ Engine still warming up — building phase-space history"

            # ── Risk limit gauges ─────────────────────────────────────────────
            loss_limit = p.get("max_daily_loss", 200)
            loss_used  = max(0.0, -net_pnl)
            pnl_bar    = _pct_bar(loss_used, 0, loss_limit, width=12)

            trades_today = risk.get("trades_today", 0)
            max_trades   = p.get("max_daily_trades", "?")
            trd_bar      = _pct_bar(trades_today, 0, (max_trades or 50), width=12)

            halt_note = ""
            if is_halted:
                halt_note = f"\n🚨 *HALT:* `{_md_safe(risk.get('halt_reason', ''))}`  — /resume to clear"

            return (
                f"📡 *HPMS Pulse*  ·  `{price:,.1f}`\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{strat_icon} Strategy: `{'ACTIVE' if strat_on else 'STOPPED'}`    "
                f"{risk_icon} Risk: `{'HALTED' if is_halted else 'OK'}`\n\n"
                f"*Position:*\n{pos_txt}\n\n"
                f"*Engine:*\n{thinking}\n\n"
                f"{pnl_icon} *P&L Today:*  `${net_pnl:+.2f}` "
                f"(peak `${risk.get('session_high_pnl', 0):+.2f}`)\n"
                f"  Loss used:  `[{pnl_bar}]`  `${loss_used:.2f}` / `${loss_limit}`\n"
                f"  Trades:     `[{trd_bar}]`  `{trades_today}` / `{max_trades}`\n"
                f"  Consec ↓:   `{risk.get('consecutive_losses', 0)}` in a row\n"
                f"  Bars seen:  `{s.get('bar_count', 0)}`"
                f"{halt_note}\n\n"
                f"_/thinking for full decision stack · /status for dashboard_"
            )
        except Exception as e:
            return f"⚠️ Periodic report error: {e}"

    # ──────────────────────────────────────────────────────────────────────────
    # COMMANDS: /help
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text(
                "🔬 *HPMS — Hamiltonian Phase-Space Micro-Scalping*\n\n"
                "📊 *Info*\n"
                "/status — Full dashboard\n"
                "/thinking — Decision stack (why we trade)\n"
                "/phase — Phase-space (q, p, H, dH/dt)\n"
                "/signal — Last signal detail\n"
                "/market — Price, spread, ATR\n"
                "/filter — Filter gate status\n"
                "/risk — Risk gate status\n"
                "/pnl — Daily P&L\n"
                "/trades — Trade log\n"
                "/position — Open position\n"
                "/balance — Balance\n"
                "/diag — Engine diagnostics\n"
                "/engine — Engine parameters\n\n"
                "⚡ *Controls*\n"
                "/start_trading — Enable strategy\n"
                "/stop_trading — Disable (positions remain)\n"
                "/halt — Emergency stop + flatten\n"
                "/resume — Resume from halt\n"
                "/resetrisk — Clear lockout only (no flatten)\n"
                "/flatten — Close all positions\n"
                "/close — Close current position\n\n"
                "⚙️ *Config*\n"
                "/params — All parameters\n"
                "/set <param> <value>\n"
                "/get <param>\n"
                "/leverage <N>\n"
                "/cooldown <sec>\n"
                "/maxloss <usd>\n"
                "/maxsize <contracts>",
                parse_mode="Markdown",
            )
        except Exception as e:
            # Fallback to plain text if Markdown fails
            logger.warning(f"Help command Markdown failed: {e}")
            await update.message.reply_text(
                "HPMS — Hamiltonian Phase-Space Micro-Scalping\n\n"
                "Info: /status /thinking /phase /signal /market /filter "
                "/risk /pnl /trades /position /balance /diag /engine\n\n"
                "Controls: /start_trading /stop_trading /halt /resume "
                "/resetrisk /flatten /close\n\n"
                "Config: /params /set /get /leverage /cooldown /maxloss /maxsize"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # COMMANDS: INFO
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_ping(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        t0  = time.time()
        msg = await update.message.reply_text("⏱ …")
        rtt = (time.time() - t0) * 1000
        await msg.edit_text(
            f"🏓 Pong — {rtt:.0f}ms RTT — "
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
        )

    async def _cmd_price(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        price = self._data.get_last_price() if self._data else 0
        ob    = self._data.get_orderbook() if self._data else {}
        bids  = ob.get("bids", [])
        asks  = ob.get("asks", [])
        spread_txt = ""
        if bids and asks:
            try:
                bb = float(bids[0][0])
                ba = float(asks[0][0])
                spread_txt = f"  Spread: `${ba - bb:.1f}` ({(ba-bb)/bb*100:.3f}%)"
            except Exception:
                pass
        await update.message.reply_text(
            f"💹 `${price:,.1f}`{spread_txt}",
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        try:
            s     = self._strategy.get_status() if self._strategy else {}
            risk  = s.get("risk", {})
            pos   = s.get("position", {})
            sig   = s.get("last_signal", {})
            price = self._data.get_last_price() if self._data else 0
            dm_ok = self._data.is_ready if self._data else False

            # ── Position ──────────────────────────────────────────────────────
            if pos.get("in_position"):
                side_icon = "📈" if pos["side"] == "long" else "📉"
                pos_txt = (
                    f"{side_icon} *{pos['side'].upper()}*  "
                    f"`{pos['size']}c` @ `${pos['entry_price']:,.1f}`  "
                    f"({pos['bars_held']} bars held)"
                )
            else:
                pos_txt = "⬜ Flat — no open position"

            # ── Risk status ───────────────────────────────────────────────────
            if risk.get("is_halted"):
                halt_txt = f"🚨 *HALTED*  reason: `{risk.get('halt_reason', '')}`"
            else:
                halt_txt = "🛡 OK — gates clear"

            # ── What the engine last decided ──────────────────────────────────
            sig_type = sig.get("type") or "—"
            sig_conf = sig.get("confidence") or 0
            sig_dq   = sig.get("delta_q") or 0
            sig_why  = sig.get("reason") or "no signal yet"

            if sig_type in ("LONG", "SHORT"):
                sig_summary = f"🎯 *{sig_type}*  conf `{sig_conf:.1%}`  Δq `{sig_dq:+.5f}`"
            elif sig_type == "FLAT":
                sig_summary = f"👀 FLAT — engine sees no edge this bar"
            else:
                sig_summary = "⏳ Warming up — not enough bars yet"

            sig_summary += f"\n  Reason: `{sig_why[:90]}`"

            # ── P&L gauges ────────────────────────────────────────────────────
            net_pnl    = risk.get("daily_pnl", 0)
            loss_limit = risk.get("params", {}).get("max_daily_loss", 200)
            pnl_bar    = _pct_bar(max(0.0, -net_pnl), 0, loss_limit)
            pnl_icon   = "💰" if net_pnl >= 0 else "🔻"

            await update.message.reply_text(
                f"📊 *HPMS Dashboard*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Strategy:  `{'▶ ACTIVE' if s.get('enabled') else '⏸ STOPPED'}`\n"
                f"Data feed: `{'🟢 Live' if dm_ok else '🔴 Not Ready'}`\n"
                f"Risk:      {halt_txt}\n"
                f"Price:     `${price:,.1f}`\n\n"
                f"*Position:*\n{pos_txt}\n\n"
                f"*Engine decision (last bar):*\n{sig_summary}\n"
                f"Compute: `{sig.get('compute_us', 0):.0f}µs`    "
                f"Bars seen: `{s.get('bar_count', 0)}`\n\n"
                f"{pnl_icon} *Today's P&L:*\n"
                f"  Gross: `${risk.get('daily_gross_pnl', 0):+.4f}`\n"
                f"  Fees:  `$-{risk.get('daily_fees', 0):.4f}`\n"
                f"  *Net:  `${net_pnl:+.4f}`*  "
                f"(peak `${risk.get('session_high_pnl', 0):+.4f}`)\n"
                f"  Loss vs limit:  [{pnl_bar}]\n"
                f"  Trades: `{risk.get('trades_today', 0)}`"
                f"/{risk.get('params', {}).get('max_daily_trades', '?')}"
                f"   Consec ↓: `{risk.get('consecutive_losses', 0)}`\n"
                f"  Cooldown: `{risk.get('cooldown_remaining', 0):.0f}s`\n\n"
                f"_/thinking for decision stack · /risk for gate details_",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Status error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # /thinking — HPMS DECISION STACK
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_thinking(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        HPMS decision stack — shows exactly why the engine did or did not
        generate a tradeable signal on the last bar, and whether risk/filters
        would have allowed it through.

        LAYER 1 — Phase-space state   (q, p, H, dH/dt)
        LAYER 2 — KDE landscape       (built? bandwidth, grid, rebuild age)
        LAYER 3 — Signal gates        (Δq, energy, momentum, acceleration)
        LAYER 4 — Risk gate           (halt, cooldown, daily limits)
        LAYER 5 — Filter gate         (spread, volume, volatility)
        VERDICT — What's green / what's blocking
        """
        if not await self._guard(update):
            return
        try:
            lines = [f"🧠 *HPMS — What I'm Thinking*"]
            price = self._data.get_last_price() if self._data else 0
            if price:
                lines.append(f"Current price: `${price:,.1f}`")
            lines.append("")

            # ── LAYER 1: Phase-space ─────────────────────────────────────────
            lines.append("*── LAYER 1: PHASE-SPACE STATE ──*")
            lines.append("_This tells us where the market is in Hamiltonian space._")
            ps = self._engine.get_phase_state() if self._engine else None
            if ps:
                p_dir   = "rising ⬆️" if ps.p > 0.0001 else ("falling ⬇️" if ps.p < -0.0001 else "flat ➡️")
                dH_max  = getattr(self._config, "SIGNAL_DH_DT_MAX", 0.05) if self._config else 0.05
                dH_ok   = abs(ps.dH_dt) <= dH_max
                dH_note = "✅ energy conserved — system is stable" if dH_ok \
                    else "❌ energy leaking — chaotic regime, no trade"
                lines += [
                    f"  q (phase position): `{ps.q:+.6f}`",
                    f"  p (momentum):       `{ps.p:+.6f}`  → market is {p_dir}",
                    f"  H (total energy):   `{ps.H:.6f}`"
                    f"   K=`{ps.kinetic:.4f}` V=`{ps.potential:.4f}`",
                    f"  dH/dt (stability):  `{ps.dH_dt:.6f}`  {dH_note}",
                    f"  _(threshold |dH/dt| ≤ {dH_max} for a trade to qualify)_",
                ]
            else:
                lines.append("  ⏳ *Not ready* — engine is still warming up (need more bars)")

            # ── LAYER 2: KDE landscape ───────────────────────────────────────
            lines.append("\n*── LAYER 2: POTENTIAL LANDSCAPE V(q) ──*")
            lines.append("_KDE turns price history into a probability landscape the engine navigates._")
            d = self._engine.get_diagnostics() if self._engine else {}
            ep = self._engine.get_params() if self._engine else {}

            built       = d.get("landscape_built", False)
            bars_since  = d.get("bars_since_kde_build", 0)
            rebuild_int = d.get("kde_rebuild_interval", "?")
            hist_len    = d.get("history_len", 0)
            bw          = ep.get("kde_bandwidth", "?")
            grid        = ep.get("kde_grid_points", "?")
            needed      = ep.get("lookback", "?")

            kde_status  = "✅ Built and ready" if built else "❌ Not yet built — need more history"
            lines += [
                f"  Status:    {kde_status}",
                f"  Bandwidth: `{bw}`   Grid: `{grid}` points",
                f"  Rebuild:   every `{rebuild_int}` bars  (last built `{bars_since}` bars ago)",
                f"  History:   `{hist_len}` bars available  (need ≥ `{needed}` to build)",
            ]

            # ── LAYER 3: Signal gates ────────────────────────────────────────
            lines.append("\n*── LAYER 3: SIGNAL QUALITY GATES ──*")
            lines.append("_Engine checks three criteria before declaring a valid signal._")
            sig = self._strategy._last_signal if self._strategy else None
            if sig:
                cfg = self._config
                dq_thresh  = getattr(cfg, "SIGNAL_DELTA_Q_THRESHOLD", 0.0022) if cfg else 0.0022
                dH_thresh  = getattr(cfg, "SIGNAL_DH_DT_MAX",          0.05)  if cfg else 0.05
                mom_thresh = getattr(cfg, "SIGNAL_MIN_MOMENTUM",       0.0001) if cfg else 0.0001

                dq_ok  = abs(sig.predicted_delta_q) >= dq_thresh
                dH_ok  = abs(sig.dH_dt) <= dH_thresh
                mom_ok = abs(sig.predicted_p_final) >= mom_thresh

                dq_bar  = _pct_bar(abs(sig.predicted_delta_q), 0, dq_thresh * 3)
                dH_bar  = _pct_bar(abs(sig.dH_dt), 0, dH_thresh * 2)

                gate_all_ok = dq_ok and dH_ok and mom_ok

                lines += [
                    f"  {_gate(dq_ok)} *Displacement Δq* — did price move far enough?",
                    f"       `{sig.predicted_delta_q:+.6f}`  (need |Δq| ≥ {dq_thresh})",
                    f"       [{dq_bar}]  {'✔ sufficient move' if dq_ok else '✘ move too small — signal suppressed'}",
                    f"",
                    f"  {_gate(dH_ok)} *Energy stability dH/dt* — is the system well-behaved?",
                    f"       `{abs(sig.dH_dt):.6f}`  (must be ≤ {dH_thresh})",
                    f"       [{dH_bar}]  {'✔ stable energy' if dH_ok else '✘ energy diverging — too chaotic'}",
                    f"",
                    f"  {_gate(mom_ok)} *Momentum p_final* — is there directional conviction?",
                    f"       `{abs(sig.predicted_p_final):.6f}`  (need |p| ≥ {mom_thresh})",
                    f"       {'✔ momentum confirmed' if mom_ok else '✘ momentum too weak — no edge'}",
                    f"",
                    f"  → Signal: *{sig.signal_type.name}*  "
                    f"confidence `{sig.confidence:.1%}`  "
                    f"({'all gates passed 🟢' if gate_all_ok else 'one or more gates failed 🔴'})",
                    f"  → Reason: `{_md_safe(sig.reason[:100])}`",
                    f"  → Computed in `{sig.compute_time_us:.0f}µs`",
                ]
            else:
                lines.append("  ⏳ *No signal yet* — waiting for enough bars to run the engine")

            # ── LAYER 4: Risk gate ───────────────────────────────────────────
            lines.append("\n*── LAYER 4: RISK GATE ──*")
            lines.append("_Even a valid signal is blocked if risk limits are breached._")
            if self._risk:
                can_trade, reason = self._risk.can_trade()
                rs = self._risk.get_status()
                p  = rs.get("params", {})
                cd = rs.get("cooldown_remaining", 0)
                auto_rem = rs.get("auto_resume_remaining", 0)
                eff_cd = p.get("effective_cooldown", p.get("cooldown", "?"))

                halt_line = "No — trading is permitted"
                if rs.get("is_halted"):
                    halt_line = "🚨 YES — " + _md_safe(rs.get("halt_reason", ""))
                    if auto_rem > 0:
                        halt_line += f"  (auto-resume in {auto_rem:.0f}s)"

                lines += [
                    f"  {_gate(can_trade)} *Can trade right now?*  → `{reason}`",
                    f"  Halt active:  `{halt_line}`",
                    f"  Cooldown:     `{cd:.0f}s` remaining  (effective cooldown: `{eff_cd}s`)",
                    f"  Daily PnL:    `${rs.get('daily_pnl', 0):+.2f}`  / loss limit `${p.get('max_daily_loss', '?')}`",
                    f"  Trades today: `{rs.get('trades_today', 0)}` / max `{p.get('max_daily_trades', '?')}`",
                    f"  Consec loss:  `{rs.get('consecutive_losses', 0)}` in a row "
                    f"/ max `{p.get('max_consec_losses', '?')}`",
                ]
            else:
                lines.append("  ⚠️ Risk manager unavailable")

            # ── LAYER 5: Filter gate ─────────────────────────────────────────
            lines.append("\n*── LAYER 5: MARKET CONDITION FILTERS ──*")
            lines.append("_Microstructure checks — spread, volume, volatility must be suitable._")
            lines.append(self._build_filter_inline())

            # ── VERDICT ─────────────────────────────────────────────────────
            lines.append("\n*── VERDICT ──*")
            if self._orders and self._orders.is_in_position:
                pos = self._orders.get_status()
                side_icon = "📈" if pos["side"] == "long" else "📉"
                lines.append(
                    f"  {side_icon} *Currently in a trade* — monitoring for TP/SL/exit conditions\n"
                    f"  {pos['side'].upper()} `{pos['size']}c` @ `${pos['entry_price']:,.1f}`  "
                    f"({pos['bars_held']} bars held)"
                )
            elif sig and sig.signal_type.name != "FLAT":
                can_trade = self._risk.can_trade()[0] if self._risk else False
                if can_trade:
                    lines.append(
                        "  🎯 *Signal present + risk gate open*\n"
                        "  → Waiting for market filters to clear before entry"
                    )
                else:
                    lines.append(
                        "  ⏳ *Signal present but risk gate is blocking*\n"
                        "  → Use /risk to see what's holding us back"
                    )
            else:
                lines.append(
                    "  👀 *Watching — no actionable signal this bar*\n"
                    "  → Engine sees the market as directionless or too risky right now"
                )

            try:
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            except Exception:
                plain = "\n".join(lines).replace("`", "").replace("*", "").replace("_", "")
                await update.message.reply_text(plain)

        except Exception as e:
            logger.error(f"Thinking error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Thinking error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # /market
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_market(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        try:
            price = self._data.get_last_price() if self._data else 0
            ob    = self._data.get_orderbook() if self._data else {}
            bids  = ob.get("bids", [])
            asks  = ob.get("asks", [])
            dm_ok = self._data.is_ready if self._data else False

            spread_txt = "N/A"
            spread_pct = 0.0
            if bids and asks:
                try:
                    bb = float(bids[0][0])
                    ba = float(asks[0][0])
                    spread_pts = ba - bb
                    spread_pct = spread_pts / bb * 100 if bb else 0
                    max_sp = getattr(self._config, "FILTER_SPREAD_MAX_PCT", 0.05) \
                        if self._config else 0.05
                    sp_ok = spread_pct <= max_sp
                    spread_txt = (f"`${spread_pts:.1f}` ({spread_pct:.3f}%) "
                                  f"{_gate(sp_ok)} max {max_sp}%")
                except Exception:
                    pass

            depth_lines = []
            for a in list(reversed(asks[:3])):
                try:
                    depth_lines.append(f"  ASK `${float(a[0]):>10,.1f}`  {int(float(a[1])):>6,}")
                except Exception:
                    pass
            depth_lines.append("  ─────────────────────────")
            for b in bids[:3]:
                try:
                    depth_lines.append(f"  BID `${float(b[0]):>10,.1f}`  {int(float(b[1])):>6,}")
                except Exception:
                    pass

            candles = self._data.get_candles("1m", limit=15) if self._data else []
            atr_txt = "N/A"
            vol_txt = "N/A"
            if len(candles) >= 10:
                try:
                    recent  = candles[-10:]
                    atr     = sum(c["h"] - c["l"] for c in recent) / len(recent)
                    atr_pct = atr / price * 100 if price else 0
                    vol_min = getattr(self._config, "FILTER_VOLATILITY_MIN_PCT", 0.01) \
                        if self._config else 0.01
                    vol_max = getattr(self._config, "FILTER_VOLATILITY_MAX_PCT", 2.0) \
                        if self._config else 2.0
                    vol_ok  = vol_min <= atr_pct <= vol_max
                    atr_txt = f"`${atr:.1f}` ({atr_pct:.3f}%) {_gate(vol_ok)}"
                    last_vol = candles[-1].get("v", 0)
                    min_vol  = getattr(self._config, "FILTER_MIN_VOLUME_1M", 10.0) \
                        if self._config else 10.0
                    vol_ok2  = last_vol >= min_vol
                    vol_txt  = f"`{last_vol:.1f}` {_gate(vol_ok2)} (min {min_vol})"
                except Exception:
                    pass

            fresh_ok = False
            try:
                fresh_ok = self._data.is_price_fresh(max_stale_seconds=30)
            except Exception:
                pass

            lines = [
                f"📈 *Market Snapshot*\n",
                f"Price:   `${price:,.1f}`",
                f"Data:    {_gate(dm_ok)} ready  {_gate(fresh_ok)} fresh\n",
                f"Spread:  {spread_txt}",
                f"ATR 10b: {atr_txt}",
                f"Vol 1m:  {vol_txt}\n",
                "*Orderbook (Top 3):*",
            ] + depth_lines

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Market error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # /filter
    # ──────────────────────────────────────────────────────────────────────────

    def _build_filter_inline(self) -> str:
        """Build filter gate lines (used by /filter and /thinking)."""
        if not self._data or not self._config:
            return "  ⚠️ Data/config unavailable"

        lines = []
        cfg   = self._config

        ob   = self._data.get_orderbook()
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if bids and asks:
            try:
                bb = float(bids[0][0])
                ba = float(asks[0][0])
                spread_pct = (ba - bb) / bb * 100 if bb else 0
                max_sp     = getattr(cfg, "FILTER_SPREAD_MAX_PCT", 0.05)
                sp_ok      = spread_pct <= max_sp
                lines.append(f"  {_gate(sp_ok)} Spread:   `{spread_pct:.3f}%` (max {max_sp}%)")
            except Exception:
                lines.append("  ⚠️ Spread: n/a")

        candles = self._data.get_candles("1m", limit=15) or []
        price   = self._data.get_last_price() or 0

        if candles:
            last_vol = candles[-1].get("v", 0)
            min_vol  = getattr(cfg, "FILTER_MIN_VOLUME_1M", 10.0)
            vol_ok   = last_vol >= min_vol
            lines.append(f"  {_gate(vol_ok)} Volume:   `{last_vol:.1f}` (min {min_vol})")

        if len(candles) >= 10 and price:
            try:
                atr     = sum(c["h"] - c["l"] for c in candles[-10:]) / 10
                atr_pct = atr / price * 100
                vol_min = getattr(cfg, "FILTER_VOLATILITY_MIN_PCT", 0.01)
                vol_max = getattr(cfg, "FILTER_VOLATILITY_MAX_PCT", 2.0)
                vol_ok  = vol_min <= atr_pct <= vol_max
                lines.append(
                    f"  {_gate(vol_ok)} ATR:      `{atr_pct:.3f}%` "
                    f"(range {vol_min}%–{vol_max}%)"
                )
            except Exception:
                pass

        return "\n".join(lines) if lines else "  ⏳ Not enough data"

    async def _cmd_filter(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        try:
            price = self._data.get_last_price() if self._data else 0
            lines = [f"🔍 *Filter Gate @ `${price:,.1f}`*\n"]
            lines.append(self._build_filter_inline())
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Filter error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # /risk
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._risk:
            await update.message.reply_text("No risk manager")
            return
        try:
            rs = self._risk.get_status()
            p  = rs.get("params", {})
            cd = rs.get("cooldown_remaining", 0)

            can_trade, reason = self._risk.can_trade()
            gate_icon = "🟢" if can_trade else "🔴"

            # ── Loss gauge: how much of the daily limit have we burned? ───────
            net_pnl   = rs.get("daily_pnl", 0)
            loss_limit = p.get("max_daily_loss", 200)
            loss_used  = max(0.0, -net_pnl)
            pnl_bar = _pct_bar(loss_used, 0, loss_limit)
            loss_pct = (loss_used / loss_limit * 100) if loss_limit else 0

            # ── Trade count gauge ─────────────────────────────────────────────
            trades_today = rs.get("trades_today", 0)
            max_trades   = p.get("max_daily_trades", 50)
            trd_bar = _pct_bar(trades_today, 0, (max_trades or 50))

            auto_resume = rs.get("auto_resume_remaining", 0)
            halt_info = "No — trading permitted"
            if rs.get("is_halted"):
                halt_info = "🚨 YES — " + rs.get("halt_reason", "")
                if auto_resume > 0:
                    halt_info += f"\n           (auto-resume in {auto_resume:.0f}s)"

            eff_cd = p.get("effective_cooldown", p.get("cooldown", "?"))

            # ── Blocking summary ──────────────────────────────────────────────
            if can_trade:
                gate_summary = "🟢 *All gates clear — new entries allowed*"
            else:
                gate_summary = f"🔴 *Blocked:* `{reason}`"

            await update.message.reply_text(
                f"🛡 *Risk Gate Status*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{gate_summary}\n\n"
                f"*Halt:*      `{halt_info}`\n"
                f"*Cooldown:* `{cd:.0f}s` remaining  (effective `{eff_cd}s`)\n\n"
                f"*Daily Loss Used:*\n"
                f"  `${loss_used:.2f}` of `${loss_limit}` limit  ({loss_pct:.1f}%)\n"
                f"  `[{pnl_bar}]`\n"
                f"  Net PnL today: `${net_pnl:+.2f}`\n\n"
                f"*Trades Used:*\n"
                f"  `{trades_today}` of `{max_trades}` allowed today\n"
                f"  `[{trd_bar}]`\n\n"
                f"*Other counters:*\n"
                f"  Consec losses: `{rs.get('consecutive_losses', 0)}` / max `{p.get('max_consec_losses', '?')}`\n"
                f"  Session high:  `${rs.get('session_high_pnl', 0):+.2f}`\n\n"
                f"*Position limits:*\n"
                f"  Max pos USD:   `${p.get('max_pos_usd','?')}`\n"
                f"  Max contracts: `{p.get('max_pos_contracts','?')}c`\n"
                f"  Leverage:      `{p.get('leverage','?')}x`\n"
                f"  Max drawdown:  `{p.get('max_dd_pct','?')}%`\n"
                f"  Equity/trade:  `{p.get('equity_pct','?')}%`\n"
                f"  Auto-resume:   `{p.get('auto_resume_sec','?')}s`\n"
                f"  Soft loss wt:  `{p.get('soft_loss_weight','?')}`\n\n"
                f"_/resetrisk to clear lockout  ·  /resume to resume from halt_",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Risk error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # /signal, /phase, /diag, /engine
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        s = self._strategy._last_signal if self._strategy else None
        if not s:
            await update.message.reply_text(
                "⏳ *No signal yet*\n\n"
                "The engine is still warming up — it needs enough bars to build\n"
                "the KDE landscape and compute phase-space trajectories.\n\n"
                "_Use /diag to check how many bars have been processed._"
            )
            return

        sig_name = s.signal_type.name
        if sig_name == "LONG":
            dir_line = "📈 *LONG* — engine predicts upward displacement"
        elif sig_name == "SHORT":
            dir_line = "📉 *SHORT* — engine predicts downward displacement"
        else:
            dir_line = "⬜ *FLAT* — no directional edge, engine is neutral"

        dq_abs = abs(s.predicted_delta_q)
        cfg = self._config
        dq_thresh = getattr(cfg, "SIGNAL_DELTA_Q_THRESHOLD", 0.0022) if cfg else 0.0022
        dq_bar = _pct_bar(dq_abs, 0, dq_thresh * 3)

        await update.message.reply_text(
            f"🔬 *Last Engine Signal*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{dir_line}\n"
            f"Confidence:     `{s.confidence:.1%}`\n"
            f"Reason:         `{s.reason}`\n\n"
            f"*Phase-space metrics:*\n"
            f"  Predicted Δq:  `{s.predicted_delta_q:+.6f}`  (need |Δq| ≥ {dq_thresh})\n"
            f"                 [{dq_bar}]\n"
            f"  p_final:       `{s.predicted_p_final:.6f}`  (residual momentum)\n"
            f"  H (energy):    `{s.current_H:.6f}`\n"
            f"  dH/dt:         `{s.dH_dt:.6f}`  (stability indicator)\n\n"
            f"*Execution levels:*\n"
            f"  Entry: `${s.entry_price:,.1f}`\n"
            f"  TP:    `${s.tp_price:,.1f}`  (+`${abs(s.tp_price - s.entry_price):.1f}`)\n"
            f"  SL:    `${s.sl_price:,.1f}`  (-`${abs(s.sl_price - s.entry_price):.1f}`)\n\n"
            f"  Computed in `{s.compute_time_us:.0f}µs`",
            parse_mode="Markdown",
        )

    async def _cmd_phase(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ps = self._engine.get_phase_state() if self._engine else None
        if not ps:
            await update.message.reply_text(
                "⏳ *Engine not ready yet*\n\n"
                "Phase-space coordinates are computed after the KDE landscape\n"
                "is built\\. This requires enough price bars to fill the lookback window\\.\n\n"
                "_Use /diag to check progress\\._"
            )
            return
        dH_max  = getattr(self._config, "SIGNAL_DH_DT_MAX", 0.05) if self._config else 0.05
        dH_ok   = abs(ps.dH_dt) <= dH_max

        p_desc = "rising — bullish momentum" if ps.p > 0.0001 \
            else ("falling — bearish momentum" if ps.p < -0.0001 else "flat — no net momentum")
        h_desc = "conserved ✅ — market in a predictable regime" if dH_ok \
            else "diverging ❌ — chaotic regime, signal quality low"

        await update.message.reply_text(
            f"🌀 *Phase-Space State*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"*q (phase position):*  `{ps.q:+.6f}`\n"
            f"  _How far price has displaced in normalised space_\n\n"
            f"*p (momentum):*        `{ps.p:+.6f}`\n"
            f"  _Direction & strength: {p_desc}_\n\n"
            f"*H (total energy):*    `{ps.H:.6f}`\n"
            f"  _Kinetic:   `{ps.kinetic:.6f}`  (motion energy)_\n"
            f"  _Potential: `{ps.potential:.6f}`  (landscape energy)_\n\n"
            f"*dH/dt (EMA):*         `{ps.dH_dt:.6f}`\n"
            f"  _Stability check — {h_desc}_\n"
            f"  _Threshold: |dH/dt| must be ≤ `{dH_max}` for a valid signal_",
            parse_mode="Markdown",
        )

    async def _cmd_diag(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        d  = self._engine.get_diagnostics() if self._engine else {}
        ps = d.get("phase_state", {})
        hs = d.get("H_smoothing", {})
        text = (
            f"🔧 *Engine Diagnostics*\n\n"
            f"Landscape built: `{d.get('landscape_built')}`\n"
            f"KDE rebuild:  `{d.get('bars_since_kde_build')}`/`{d.get('kde_rebuild_interval')}` bars\n"
            f"History len:  `{d.get('history_len')}` bars\n"
            f"Signals gen:  `{d.get('signal_count')}`\n"
            f"Traj. logged: `{d.get('trajectory_log_depth')}`\n\n"
            f"*Phase State:*\n"
            f"q=`{ps.get('q')}` p=`{ps.get('p')}`\n"
            f"H=`{ps.get('H')}` dH/dt_ema=`{ps.get('dH_dt_ema')}`\n\n"
            f"*H Smoothing (span={hs.get('ema_span')}):*\n"
            f"H_raw=`{hs.get('H_raw')}` H_ema=`{hs.get('H_ema')}`\n"
            f"dH_raw=`{hs.get('dH_raw')}` dH_ema=`{hs.get('dH_ema')}`"
        )
        traj = d.get("last_trajectory")
        if traj:
            text += f"\n\n*Last Trajectory ({len(traj)} pts):*\n```\n"
            for tp in traj:
                text += f"  t={tp['t']:.0f} q={tp['q']:+.5f} p={tp['p']:+.5f} H={tp['H']:.5f}\n"
            text += "```"
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_engine(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ep = self._engine.get_params() if self._engine else {}
        if not ep:
            await update.message.reply_text("Engine not available")
            return

        cfg = self._config
        lines = ["⚙️ *Engine Parameters*\n"]

        sections = {
            "Phase Space": [
                ("tau",                  "Takens embedding delay (bars)"),
                ("lookback",             "KDE window (bars)"),
                ("prediction_horizon",   "Integration horizon (bars)"),
                ("normalization_window", "Z-score window (bars)"),
            ],
            "KDE / Landscape": [
                ("kde_bandwidth",        "KDE bandwidth"),
                ("kde_rebuild_interval", "Rebuild every N bars"),
            ],
            "Integrator": [
                ("integrator",           "Algorithm (rk4/euler/leapfrog)"),
                ("integration_dt",       "Sub-step size"),
                ("mass",                 "Effective mass m"),
                ("H_ema_span",           "dH/dt EMA span"),
            ],
            "Signal Gates": [
                ("delta_q_threshold",    "Min |Δq| for entry"),
                ("dH_dt_max",            "Max |dH/dt| (energy check)"),
                ("H_percentile",         "H percentile cap (chaos filter)"),
                ("min_momentum",         "Min |p_final|"),
                ("acceleration_check",   "Require downhill accel"),
            ],
            "TP / SL": [
                ("tp_pct",               "Take-profit %"),
                ("sl_pct",               "Stop-loss %"),
            ],
        }

        for section, params in sections.items():
            lines.append(f"*{section}:*")
            for key, desc in params:
                val = ep.get(key, "—")
                lines.append(f"  `{key}` = `{val}`  _{desc}_")
            lines.append("")

        if cfg:
            lines.append("*Execution:*")
            for k in ["TRADE_MAX_HOLD_BARS", "TRADE_DH_DT_EXIT_SPIKE",
                      "TRADE_USE_BRACKET_ORDERS", "TRADE_ORDER_TYPE"]:
                if hasattr(cfg, k):
                    lines.append(f"  `{k}` = `{getattr(cfg, k)}`")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ──────────────────────────────────────────────────────────────────────────
    # /trades, /pnl, /balance, /position, /orderbook
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        trades = self._risk.get_trade_log(10) if self._risk else []
        if not trades:
            await update.message.reply_text("📋 No trades today yet")
            return
        lines = ["📋 *Recent Trades*\n"]
        total_gross = total_fees = total_net = 0.0
        for t in trades:
            emoji = "💰" if t["net_pnl"] >= 0 else "🔻"
            lines.append(
                f"{emoji} `{t['time']}` {t['side'].upper()} {t.get('size', 1)}c "
                f"gross `${t['gross_pnl']:+.4f}` "
                f"fee `$-{t['fees']:.4f}` "
                f"*net `${t['net_pnl']:+.4f}`* "
                f"ROE `{t.get('roe_pct', 0):+.1f}%` "
                f"({t['bars']}b) _{t['reason']}_"
            )
            total_gross += t["gross_pnl"]
            total_fees += t["fees"]
            total_net += t["net_pnl"]
        lines.append(
            f"Gross: `${total_gross:+.4f}` | Fees: `$-{total_fees:.4f}`"
            f"*Net: `${total_net:+.4f}`*"
        )
        try:
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception:
            plain = "\n".join(lines).replace("`", "").replace("*", "").replace("_", "")
            await update.message.reply_text(plain)

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        r = self._risk.get_status() if self._risk else {}
        p = r.get("params", {})
        net_pnl   = r.get("daily_pnl", 0)
        gross_pnl = r.get("daily_gross_pnl", 0)
        fees      = r.get("daily_fees", 0)
        pnl_max   = p.get("max_daily_loss", 200)
        loss_used = max(0.0, -net_pnl)
        pnl_bar   = _pct_bar(loss_used, 0, pnl_max)
        loss_pct  = (loss_used / pnl_max * 100) if pnl_max else 0
        pnl_icon  = "💰" if net_pnl >= 0 else "🔻"
        sess_high = r.get("session_high_pnl", 0)

        status_note = ""
        if r.get("is_halted"):
            status_note = f"\n\n🚨 *Risk halted:* `{r.get('halt_reason', '')}`"
        elif loss_pct > 75:
            status_note = "\n\n⚠️ *Close to daily loss limit — trade carefully*"
        elif loss_pct > 50:
            status_note = "\n\n⚠️ Over 50% of daily loss limit used"

        await update.message.reply_text(
            f"{pnl_icon} *Daily P&L Summary*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Gross P&L:    `${gross_pnl:+.4f}`\n"
            f"Total fees:   `$-{fees:.4f}`\n"
            f"*Net P&L:     `${net_pnl:+.4f}`*\n\n"
            f"Session high: `${sess_high:+.4f}`\n"
            f"Drawdown from peak: `${min(0.0, net_pnl - sess_high):+.4f}`\n\n"
            f"*Loss limit used:*\n"
            f"  `${loss_used:.2f}` of `${pnl_max}` ({loss_pct:.1f}%)\n"
            f"  `[{pnl_bar}]`\n\n"
            f"Trades today:   `{r.get('trades_today', 0)}` / `{p.get('max_daily_trades', '?')}`\n"
            f"Consec losses:  `{r.get('consecutive_losses', 0)}` / `{p.get('max_consec_losses', '?')}`"
            f"{status_note}",
            parse_mode="Markdown",
        )

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        b = self._api.get_balance("USD") if self._api else {}
        await update.message.reply_text(
            f"💳 *Balance*\n\n"
            f"Available: `${b.get('available', 0):.2f}`\n"
            f"Locked:    `${b.get('locked', 0):.2f}`\n"
            f"Total:     `${(b.get('available', 0) + b.get('locked', 0)):.2f}`",
            parse_mode="Markdown",
        )

    async def _cmd_position(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        p = self._orders.get_status() if self._orders else {}
        if not p.get("in_position"):
            await update.message.reply_text("⬜ No open position")
            return
        price = self._data.get_last_price() if self._data else 0
        cfg   = self._config

        pnl_block = ""
        if price and p.get("entry_price"):
            diff = price - p["entry_price"]
            if p["side"] == "short":
                diff = -diff
            cv   = getattr(cfg, "TRADE_CONTRACT_VALUE", 0.001) if cfg else 0.001
            lev  = getattr(cfg, "RISK_LEVERAGE", 10) if cfg else 10
            size = p["size"]

            notional  = p["entry_price"] * cv * size
            margin    = notional / max(lev, 1)
            gross_pnl = diff * cv * size

            # Use the ACTUAL entry fee already paid (recorded from API at entry)
            entry_fee_paid = getattr(self._orders, "_entry_fee_usd", 0.0)
            # Exit fee is not yet known — it will be settled by the exchange on close
            net_pnl_so_far = gross_pnl - entry_fee_paid
            roe_pct = (net_pnl_so_far / margin * 100) if margin > 0 else 0.0

            pnl_block = (
                f"\n*Unrealised P&L:*\n"
                f"  Gross: `${gross_pnl:+.4f}`\n"
                f"  Entry fee paid: `$-{entry_fee_paid:.4f}` _(actual, from API)_\n"
                f"  Exit fee: _(charged by exchange on close)_\n"
                f"  *Net (excl. exit fee): `${net_pnl_so_far:+.4f}`*\n"
                f"  ROE: `{roe_pct:+.2f}%` on `${margin:.2f}` margin"
            )

        side_icon = "🟢" if p.get("side") == "long" else "🔴"
        await update.message.reply_text(
            f"{side_icon} *Open Position*\n\n"
            f"Side:  `{p['side'].upper()}`\n"
            f"Size:  `{p['size']}` contracts\n"
            f"Entry: `${p['entry_price']:,.1f}`\n"
            f"Price: `${price:,.1f}`"
            f"{pnl_block}\n"
            f"Bars:  `{p['bars_held']}`\n"
            f"SL order: `{p.get('sl_order') or 'N/A'}`\n"
            f"TP order: `{p.get('tp_order') or 'N/A'}`",
            parse_mode="Markdown",
        )

    async def _cmd_orderbook(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ob   = self._data.get_orderbook() if self._data else {}
        bids = ob.get("bids", [])[:5]
        asks = ob.get("asks", [])[:5]
        lines = ["📗 *Orderbook (Top 5)*\n```"]
        for a in reversed(asks):
            try:
                lines.append(f"  ASK ${float(a[0]):>10,.1f}  {int(float(a[1])):>8,}")
            except Exception:
                pass
        lines.append("  ─────────────────────────")
        for b in bids:
            try:
                lines.append(f"  BID ${float(b[0]):>10,.1f}  {int(float(b[1])):>8,}")
            except Exception:
                pass
        lines.append("```")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ──────────────────────────────────────────────────────────────────────────
    # TRADING CONTROLS
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_start_trading(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._strategy:
            await update.message.reply_text("No strategy available")
            return
        if self._strategy.is_enabled:
            await update.message.reply_text("⚡ Strategy is already running")
            return
        if self._risk:
            rs = self._risk.get_status()
            if rs.get("is_halted"):
                await update.message.reply_text(
                    f"⚠️ Risk manager is HALTED (`{rs['halt_reason']}`)\n"
                    f"Use /resume first to clear the halt, then /start\\_trading",
                    parse_mode="Markdown",
                )
                return
        self._strategy.start()
        price = self._data.get_last_price() if self._data else 0
        await update.message.reply_text(
            f"⚡ *Strategy STARTED*\nCurrent price: `${price:,.1f}`",
            parse_mode="Markdown",
        )

    async def _cmd_stop_trading(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._strategy:
            await update.message.reply_text("No strategy available")
            return
        if not self._strategy.is_enabled:
            await update.message.reply_text("⏸ Strategy is already stopped")
            return
        self._strategy.stop()
        pos = self._orders.get_status() if self._orders else {}
        pos_note = (
            f"\n⚠️ Open position remains: {pos['side'].upper()} "
            f"{pos['size']}c @ `${pos['entry_price']:,.1f}`"
            if pos.get("in_position") else "\nNo open positions."
        )
        await update.message.reply_text(
            f"⏸ *Strategy STOPPED* — no new entries.{pos_note}",
            parse_mode="Markdown",
        )

    async def _cmd_halt(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        actions = []
        if self._strategy:
            self._strategy.stop()
            actions.append("strategy stopped")
        if self._risk:
            self._risk.force_halt("TELEGRAM_HALT")
            actions.append("risk halted")
        if self._orders:
            result = self._orders.emergency_flatten()
            actions += result.get("actions", [])
        await update.message.reply_text(
            f"⛔ *EMERGENCY HALT*\n\n"
            f"Actions taken:\n" +
            "\n".join(f"• {a}" for a in actions) +
            f"\n\n_Use /resume then /start\\_trading to re-enable._",
            parse_mode="Markdown",
        )

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._risk:
            await update.message.reply_text("No risk manager")
            return
        was = self._risk.resume()
        await update.message.reply_text(
            f"✅ *Risk resumed* (was: `{was}`)\n"
            f"Consecutive losses reset to 0.\n\n"
            f"_Use /start\\_trading to re-enable the strategy._",
            parse_mode="Markdown",
        )

    async def _cmd_resetrisk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Clear lockout only — does NOT flatten positions."""
        if not await self._guard(update):
            return
        if not self._risk:
            await update.message.reply_text("No risk manager")
            return
        was = self._risk.resume()
        rs  = self._risk.get_status()
        pos = self._orders.get_status() if self._orders else {}
        pos_note = (
            f"\n⚠️ Open position: {pos['side'].upper()} {pos['size']}c "
            f"@ `${pos['entry_price']:,.1f}` ({pos['bars_held']} bars)"
            if pos.get("in_position") else "\nNo open position."
        )
        await update.message.reply_text(
            f"🔄 *Risk lockout cleared* (was: `{was}`)\n"
            f"Consecutive losses → `0`\n"
            f"Daily trades: `{rs.get('trades_today', 0)}`  "
            f"Daily PnL: `${rs.get('daily_pnl', 0):+.2f}`{pos_note}\n\n"
            f"_Positions NOT closed — use /flatten to close all._",
            parse_mode="Markdown",
        )

    async def _cmd_flatten(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._orders:
            await update.message.reply_text("No order manager")
            return
        result  = self._orders.emergency_flatten()
        actions = result.get("actions", [])
        await update.message.reply_text(
            f"🔨 *Flatten complete*\n" +
            ("\n".join(f"• {a}" for a in actions)
             if actions else "• No open orders/positions"),
            parse_mode="Markdown",
        )

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._orders:
            await update.message.reply_text("No order manager")
            return
        if not self._orders.is_in_position:
            await update.message.reply_text("⬜ No open position to close")
            return
        price  = self._data.get_last_price() if self._data else 0
        result = self._orders.close_position(reason="TELEGRAM_CLOSE", current_price=price)
        if result.get("success"):
            await update.message.reply_text(
                f"✅ *Position closed*\nPnL: `${result.get('pnl_usd', 0):+.2f}`",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"❌ Close failed: `{result.get('error')}`", parse_mode="Markdown"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PARAMETER COMMANDS
    # ──────────────────────────────────────────────────────────────────────────

    async def _cmd_set(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text(
                "Usage: `/set <param> <value>`", parse_mode="Markdown"
            )
            return
        key = ctx.args[0]
        val = " ".join(ctx.args[1:])

        if key.lower() == "leverage":
            try:
                await self._apply_leverage(update, int(val))
            except ValueError:
                await update.message.reply_text("❌ Leverage must be an integer")
            return

        updated = False
        if self._engine and self._engine.update_param(key, val):
            updated = True
        elif self._risk and self._risk.update_param(key, val):
            updated = True
        elif self._config and hasattr(self._config, key.upper()):
            try:
                old     = getattr(self._config, key.upper())
                new_val = type(old)(val) if not isinstance(old, bool) \
                    else val.lower() in ("true", "1")
                setattr(self._config, key.upper(), new_val)
                updated = True
            except Exception:
                pass

        if updated:
            await update.message.reply_text(
                f"✅ `{key}` = `{val}`", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"❌ Unknown param: `{key}`\nUse /params to list all", parse_mode="Markdown"
            )

    async def _cmd_get(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            await update.message.reply_text("Usage: `/get <param>`", parse_mode="Markdown")
            return
        key = ctx.args[0]
        val = None
        if self._engine:
            val = self._engine.get_params().get(key)
        if val is None and self._risk:
            val = self._risk.get_status().get("params", {}).get(key)
        if val is None and self._config and hasattr(self._config, key.upper()):
            val = getattr(self._config, key.upper())
        if val is not None:
            await update.message.reply_text(f"`{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(
                f"Unknown param: `{key}`\nUse /params to list all", parse_mode="Markdown"
            )

    async def _cmd_params(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        lines = ["⚙️ *All Tunable Parameters*\n"]
        if self._engine:
            lines.append("*Engine:*")
            for k, v in self._engine.get_params().items():
                lines.append(f"  `{k}` = `{v}`")
        if self._risk:
            lines.append("\n*Risk:*")
            for k, v in self._risk.get_status().get("params", {}).items():
                lines.append(f"  `{k}` = `{v}`")
        trade_keys = [
            "TRADE_TP_PCT", "TRADE_SL_PCT", "TRADE_MAX_HOLD_BARS",
            "TRADE_DH_DT_EXIT_SPIKE", "TRADE_USE_BRACKET_ORDERS",
            "TRADE_ORDER_TYPE", "TRADE_LIMIT_OFFSET_TICKS",
        ]
        filter_keys = [
            "FILTER_SPREAD_MAX_PCT", "FILTER_MIN_VOLUME_1M",
            "FILTER_VOLATILITY_MIN_PCT", "FILTER_VOLATILITY_MAX_PCT",
        ]
        signal_keys = [
            "SIGNAL_DELTA_Q_THRESHOLD", "SIGNAL_DH_DT_MAX",
            "SIGNAL_H_PERCENTILE", "SIGNAL_MIN_MOMENTUM",
            "SIGNAL_ACCELERATION_CHECK",
        ]
        if self._config:
            lines.append("\n*Trade:*")
            for k in trade_keys:
                if hasattr(self._config, k):
                    lines.append(f"  `{k}` = `{getattr(self._config, k)}`")
            lines.append("\n*Signal:*")
            for k in signal_keys:
                if hasattr(self._config, k):
                    lines.append(f"  `{k}` = `{getattr(self._config, k)}`")
            lines.append("\n*Filter:*")
            for k in filter_keys:
                if hasattr(self._config, k):
                    lines.append(f"  `{k}` = `{getattr(self._config, k)}`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_set_engine(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text(
                "Usage: `/set_engine <param> <value>`", parse_mode="Markdown"
            )
            return
        key, val = ctx.args[0], ctx.args[1]
        if self._engine and self._engine.update_param(key, val):
            await update.message.reply_text(
                f"✅ Engine `{key}` = `{val}`", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"❌ Unknown engine param: `{key}`", parse_mode="Markdown"
            )

    async def _cmd_set_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text(
                "Usage: `/set_risk <param> <value>`", parse_mode="Markdown"
            )
            return
        key, val = ctx.args[0], ctx.args[1]
        if key.lower() == "leverage":
            try:
                await self._apply_leverage(update, int(val))
            except ValueError:
                await update.message.reply_text("❌ Leverage must be an integer")
            return
        if self._risk and self._risk.update_param(key, val):
            await update.message.reply_text(
                f"✅ Risk `{key}` = `{val}`", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"❌ Unknown risk param: `{key}`", parse_mode="Markdown"
            )

    async def _cmd_set_trade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text(
                "Usage: `/set_trade <param> <value>`", parse_mode="Markdown"
            )
            return
        key     = ctx.args[0]
        val     = ctx.args[1]
        cfg_key = f"TRADE_{key.upper()}"
        if self._config and hasattr(self._config, cfg_key):
            try:
                old = getattr(self._config, cfg_key)
                new = type(old)(val) if not isinstance(old, bool) \
                    else val.lower() in ("true", "1")
                setattr(self._config, cfg_key, new)
                await update.message.reply_text(
                    f"✅ `{cfg_key}` = `{new}`", parse_mode="Markdown"
                )
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text(
                f"❌ Unknown: `{cfg_key}`", parse_mode="Markdown"
            )

    async def _cmd_set_filter(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text(
                "Usage: `/set_filter <param> <value>`", parse_mode="Markdown"
            )
            return
        key     = ctx.args[0]
        val     = ctx.args[1]
        cfg_key = f"FILTER_{key.upper()}"
        if self._config and hasattr(self._config, cfg_key):
            try:
                old = getattr(self._config, cfg_key)
                new = type(old)(val) if not isinstance(old, bool) \
                    else val.lower() in ("true", "1")
                setattr(self._config, cfg_key, new)
                await update.message.reply_text(
                    f"✅ `{cfg_key}` = `{new}`", parse_mode="Markdown"
                )
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {e}")
        else:
            await update.message.reply_text(
                f"❌ Unknown: `{cfg_key}`", parse_mode="Markdown"
            )

    # ─── SHORTCUT COMMANDS ────────────────────────────────────────────────────

    async def _cmd_cooldown(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            cur = self._risk.get_status().get("params", {}).get("cooldown", "?") \
                if self._risk else "?"
            await update.message.reply_text(
                f"Current cooldown: `{cur}s`  Usage: `/cooldown <seconds>`",
                parse_mode="Markdown",
            )
            return
        if self._risk and self._risk.update_param("cooldown_seconds", ctx.args[0]):
            await update.message.reply_text(
                f"✅ Cooldown: `{ctx.args[0]}s`", parse_mode="Markdown"
            )

    async def _cmd_maxloss(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            cur = self._risk.get_status().get("params", {}).get("max_daily_loss", "?") \
                if self._risk else "?"
            await update.message.reply_text(
                f"Current max daily loss: `${cur}`  Usage: `/maxloss <usd>`",
                parse_mode="Markdown",
            )
            return
        if self._risk and self._risk.update_param("max_daily_loss_usd", ctx.args[0]):
            await update.message.reply_text(
                f"✅ Max daily loss: `${ctx.args[0]}`", parse_mode="Markdown"
            )

    async def _cmd_maxsize(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            cur = self._risk.get_status().get("params", {}).get("max_pos_contracts", "?") \
                if self._risk else "?"
            await update.message.reply_text(
                f"Current max contracts: `{cur}`  Usage: `/maxsize <n>`",
                parse_mode="Markdown",
            )
            return
        if self._risk and self._risk.update_param("max_position_contracts", ctx.args[0]):
            await update.message.reply_text(
                f"✅ Max contracts: `{ctx.args[0]}`", parse_mode="Markdown"
            )

    # ─── LEVERAGE ─────────────────────────────────────────────────────────────

    async def _apply_leverage(self, update: Update, val: int) -> bool:
        if val <= 0 or val > 200:
            await update.message.reply_text(
                f"❌ Invalid leverage `{val}` — must be 1–200", parse_mode="Markdown"
            )
            return False

        lines = [f"⚙️ Setting leverage to *{val}x*…"]

        if self._api:
            symbol = getattr(self._config, "DELTA_SYMBOL", "BTCUSD") if self._config else "BTCUSD"
            try:
                result = self._api.set_leverage(symbol=symbol, leverage=val)
                if result.get("success"):
                    lines.append(f"✅ Exchange confirmed `{val}x`")
                else:
                    lines.append(f"❌ Exchange rejected: `{result.get('error', 'unknown')}`")
                    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
                    return False
            except Exception as e:
                lines.append(f"❌ Exchange call failed: `{e}`")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
                return False
        else:
            lines.append("⚠️ No API — exchange NOT updated")

        if self._risk:
            self._risk.update_param("leverage", val)
            lines.append(f"✅ RiskManager `_leverage` = {val}")
        if self._config:
            self._config.RISK_LEVERAGE = val
            lines.append(f"✅ Config `RISK_LEVERAGE` = {val}")

        lines.append(f"\n🔢 *Leverage is now {val}x*")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return True

    async def _cmd_leverage(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            risk_lev = (self._risk.get_status().get("params", {}).get("leverage", "?")
                        if self._risk else "?")
            cfg_lev  = getattr(self._config, "RISK_LEVERAGE", "?") if self._config else "?"
            exch_lev = "?"
            if self._api and self._orders and self._orders._product_id:
                try:
                    r = self._api.get_leverage(self._orders._product_id)
                    if r.get("success"):
                        exch_lev = r.get("result", {}).get("leverage", "?")
                except Exception:
                    pass
            await update.message.reply_text(
                f"📊 *Current Leverage*\n\n"
                f"Exchange:    `{exch_lev}x`\n"
                f"RiskManager: `{risk_lev}x`\n"
                f"Config:      `{cfg_lev}x`\n\n"
                f"To change: `/leverage <N>`",
                parse_mode="Markdown",
            )
            return
        try:
            val = int(ctx.args[0])
        except ValueError:
            await update.message.reply_text(
                "❌ Must be an integer, e.g. `/leverage 50`", parse_mode="Markdown"
            )
            return
        await self._apply_leverage(update, val)
