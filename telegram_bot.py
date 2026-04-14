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
  /position      — Open position detail + unrealised PnL + TP/SL levels
  /pnl           — Daily P&L summary with fee breakdown
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
    frac   = max(0.0, min(1.0, (value - lo) / (hi - lo)))
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
    """Strip characters that break Telegram Markdown v1 code-spans."""
    return str(text).replace("`", "'").replace("*", "x").replace("_", "-")


def _utc_now() -> str:
    """Current UTC time as HH:MM:SS UTC."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def _streak_icons(trade_log: List[Dict], n: int = 5) -> str:
    """Return emoji strip for the last n trades: ✅ / ❌."""
    recent = trade_log[-n:]
    if not recent:
        return "—"
    return " ".join("✅" if t["net_pnl"] >= 0 else "❌" for t in recent)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT
# ══════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    """
    Thread-safe Telegram command interface for the HPMS trading system.

    Lifecycle:
      • start()        — called from main thread; spawns bot in daemon thread,
                         blocks up to 15 s until the event loop is live.
      • send_message() — thread-safe push; buffers until loop is ready.
      • stop()         — graceful shutdown.
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

        # ── Global error handler ──────────────────────────────────────────────
        async def _error_handler(update, context):
            logger.error("Telegram handler error: %s", context.error, exc_info=context.error)
            if update and update.effective_message:
                try:
                    await update.effective_message.reply_text(
                        f"❌ Command error: {context.error}"
                    )
                except Exception:
                    pass

        self._app.add_error_handler(_error_handler)

        # ── Catch-all for non-command text ────────────────────────────────────
        async def _fallback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
            if update.message and update.message.text:
                await update.message.reply_text(
                    "Use /help to see available commands."
                )

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _fallback)
        )

        # ── Initialize and start FIRST (required before set_my_commands) ─────
        await self._app.initialize()
        await self._app.start()

        # ── Bot command menu ──────────────────────────────────────────────────
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
            logger.warning("set_my_commands failed (non-fatal): %s", e)

        await self._app.updater.start_polling(drop_pending_updates=True)

        self._running = True
        logger.info("Telegram bot online — polling for commands, push notifications active")
        self._ready.set()

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
                try:
                    plain = text.replace("`", "").replace("*", "").replace("_", "")
                    await self._app.bot.send_message(
                        chat_id=self._chat_id,
                        text=plain,
                        disable_web_page_preview=True,
                    )
                except Exception as e2:
                    logger.debug("send_message fallback error: %s", e2)

        try:
            asyncio.run_coroutine_threadsafe(_send_with_fallback(), self._loop)
        except Exception as e:
            logger.debug("send_message error: %s", e)

    # ─── PERIODIC REPORT ──────────────────────────────────────────────────────

    def _periodic_report_loop(self):
        """Background thread — sends a status report every N seconds."""
        if not self._ready.wait(timeout=30):
            return
        time.sleep(60)   # let the system warm up first
        while self._running:
            try:
                interval = int(getattr(
                    self._config, "TELEGRAM_REPORT_INTERVAL_SEC",
                    self._report_interval,
                ))
                now = time.time()
                if now - self._last_report_ts >= interval:
                    self._last_report_ts = now
                    report = self._build_periodic_report()
                    if report:
                        self.send_message(report)
            except Exception as e:
                logger.debug("Periodic report error: %s", e)
            time.sleep(60)

    def _build_periodic_report(self) -> str:
        """Build a compact but complete periodic status report."""
        try:
            s     = self._strategy.get_status() if self._strategy else {}
            risk  = s.get("risk", {})
            pos   = s.get("position", {})
            sig   = s.get("last_signal", {})
            price = self._data.get_last_price() if self._data else 0
            p     = risk.get("params", {})

            # ── Strategy / risk status ────────────────────────────────────────
            strat_on  = s.get("enabled", False)
            is_halted = risk.get("is_halted", False)

            strat_icon = "▶️" if strat_on  else "⏸"
            halt_icon  = "🟢" if not is_halted else "🔴"

            # ── Position summary ──────────────────────────────────────────────
            if pos.get("in_position"):
                side_icon = "🟢" if pos["side"] == "long" else "🔴"
                # Unrealised PnL estimate
                if price and pos.get("entry_price"):
                    cv        = getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001) \
                                if self._config else 0.001
                    lev       = getattr(self._config, "RISK_LEVERAGE", 10) \
                                if self._config else 10
                    diff      = price - pos["entry_price"]
                    if pos["side"] == "short":
                        diff = -diff
                    gross_unr = diff * cv * pos["size"]
                    margin    = pos["entry_price"] * cv * pos["size"] / max(lev, 1)
                    roe_unr   = gross_unr / margin * 100 if margin > 0 else 0.0
                    unr_txt   = f"  unr `${gross_unr:+.4f}` / ROE `{roe_unr:+.1f}%`"
                else:
                    unr_txt = ""
                pos_txt = (
                    f"{side_icon} {pos['side'].upper()} `{pos['size']}c` "
                    f"@ `${pos['entry_price']:,.1f}`  ({pos['bars_held']} bars)"
                    f"{unr_txt}"
                )
            else:
                pos_txt = "⬜ Flat — awaiting signal"

            # ── P&L summary ───────────────────────────────────────────────────
            net_pnl   = risk.get("daily_pnl", 0)
            gross_pnl = risk.get("daily_gross_pnl", 0)
            fees      = risk.get("daily_fees", 0)
            pnl_icon  = "💰" if net_pnl >= 0 else "🔻"
            pnl_limit = p.get("max_daily_loss", 0)
            pnl_used  = f"{abs(net_pnl) / pnl_limit * 100:.0f}%" if pnl_limit else "—"

            # ── Last signal ───────────────────────────────────────────────────
            sig_type = sig.get("type") or "—"
            sig_conf = sig.get("confidence") or 0
            if sig_type not in ("—", "FLAT", None):
                sig_txt = f"`{sig_type}`  conf=`{sig_conf:.0%}`"
            elif sig_type == "FLAT":
                sig_txt = "`FLAT` — no edge this bar"
            else:
                sig_txt = "`—` warming up"

            # ── Risk gate ─────────────────────────────────────────────────────
            if is_halted:
                risk_txt = "🚨 HALTED — `" + _md_safe(risk.get("halt_reason", "?")) + "`"
                auto_rem = risk.get("auto_resume_remaining", 0)
                if auto_rem > 0:
                    risk_txt += f" (auto-resume in `{auto_rem:.0f}s`)"
            else:
                cd = risk.get("cooldown_remaining", 0)
                risk_txt = "✅ Clear" if cd == 0 else f"⏱ Cooldown `{cd:.0f}s`"

            # ── Data health ───────────────────────────────────────────────────
            dm_ok = self._data.is_ready if self._data else False
            try:
                price_fresh = self._data.is_price_fresh(max_stale_seconds=30) if self._data else False
            except Exception:
                price_fresh = False
            data_txt = (
                "🟢 Ready & fresh" if (dm_ok and price_fresh) else
                ("🟡 Ready, price stale" if dm_ok else "🔴 Not ready")
            )

            # ── Trade streak ──────────────────────────────────────────────────
            streak = _streak_icons(self._risk.get_trade_log(5) if self._risk else [])

            return (
                f"📡 *HPMS Periodic Report*  ⏱ `{_utc_now()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{strat_icon} Strategy: `{'ON' if strat_on else 'OFF'}`   "
                f"{halt_icon} Risk: `{'HALTED' if is_halted else 'OK'}`\n"
                f"💹 Price:    `${price:,.1f}`   Data: {data_txt}\n"
                f"📍 Position: {pos_txt}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{pnl_icon} *P&L Today*\n"
                f"  Gross: `${gross_pnl:+.4f}`  │  Fees: `$-{fees:.4f}`\n"
                f"  Net:   `${net_pnl:+.4f}`  │  High: `${risk.get('session_high_pnl', 0):+.4f}`\n"
                f"  Limit used: `{pnl_used}`  │  "
                f"Trades: `{risk.get('trades_today', 0)}`/`{p.get('max_daily_trades', '?')}`\n"
                f"  Consec losses: `{risk.get('consecutive_losses', 0)}`  │  "
                f"Streak: {streak}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔬 Last signal: {sig_txt}\n"
                f"🛡 Risk gate:   {risk_txt}\n"
                f"📊 Bars seen:  `{s.get('bar_count', 0)}`\n"
                "_/thinking for full decision stack  ·  /status for dashboard_"
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
            logger.warning("Help command Markdown failed: %s", e)
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
            f"🏓 Pong — `{rtt:.0f}ms` RTT — `{_utc_now()}`"
        )

    async def _cmd_price(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # Bug fix: /price had no auth guard — any Telegram user could poll price.
        if not await self._guard(update):
            return
        price = self._data.get_last_price() if self._data else 0
        ob    = self._data.get_orderbook() if self._data else {}
        bids  = ob.get("bids", [])
        asks  = ob.get("asks", [])
        spread_txt = ""
        if bids and asks:
            try:
                bb = float(bids[0][0])
                ba = float(asks[0][0])
                spread_pts = ba - bb
                spread_pct = spread_pts / bb * 100 if bb else 0
                spread_txt = (
                    f"\nSpread: `${spread_pts:.1f}` (`{spread_pct:.3f}%`)"
                )
            except Exception:
                pass

        fresh_ok = False
        try:
            fresh_ok = self._data.is_price_fresh(max_stale_seconds=30) if self._data else False
        except Exception:
            pass

        await update.message.reply_text(
            f"💹 *Price*  ⏱ `{_utc_now()}`\n\n"
            f"`${price:,.1f}`{spread_txt}\n"
            f"Feed: {'🟢 fresh' if fresh_ok else '🟡 stale'}",
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
            p     = risk.get("params", {})

            # Position line
            if pos.get("in_position"):
                side_icon = "🟢" if pos["side"] == "long" else "🔴"
                # Unrealised PnL
                cv  = getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001) \
                      if self._config else 0.001
                lev = getattr(self._config, "RISK_LEVERAGE", 10) \
                      if self._config else 10
                diff = price - pos["entry_price"]
                if pos["side"] == "short":
                    diff = -diff
                gross_unr = diff * cv * pos["size"]
                margin    = pos["entry_price"] * cv * pos["size"] / max(lev, 1)
                roe_unr   = gross_unr / margin * 100 if margin > 0 else 0.0
                pos_txt = (
                    f"{side_icon} {pos['side'].upper()} `{pos['size']}c` "
                    f"@ `${pos['entry_price']:,.1f}` ({pos['bars_held']} bars)\n"
                    f"  Unr: `${gross_unr:+.4f}` / ROE `{roe_unr:+.1f}%`"
                )
            else:
                pos_txt = "⬜ Flat — no open position"

            # Halt / gate line
            halt_txt = (
                f"🔴 HALTED: `{_md_safe(risk.get('halt_reason', ''))}`"
                if risk.get("is_halted") else "🟢 Clear — trades allowed"
            )
            auto_rem = risk.get("auto_resume_remaining", 0)
            if risk.get("is_halted") and auto_rem > 0:
                halt_txt += f" (auto-resume in `{auto_rem:.0f}s`)"

            # Signal line
            sig_type = sig.get("type") or "—"
            sig_conf = sig.get("confidence") or 0
            sig_dq   = sig.get("delta_q") or 0
            sig_comp = sig.get("compute_us") or 0
            sig_why  = sig.get("reason") or "—"

            # Trade streak
            streak = _streak_icons(self._risk.get_trade_log(5) if self._risk else [])

            await update.message.reply_text(
                f"📊 *HPMS Dashboard*  ⏱ `{_utc_now()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Strategy:  `{'▶ ON — entering new trades' if s.get('enabled') else '⏸ OFF — no new entries'}`\n"
                f"Data feed: `{'🟢 Ready' if dm_ok else '🔴 Not Ready'}`\n"
                f"Risk gate: {halt_txt}\n"
                f"Price:     `${price:,.1f}`\n"
                f"Position:  {pos_txt}\n"
                f"Bars seen: `{s.get('bar_count', 0)}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💰 *Today's P&L*\n"
                f"Gross:       `${risk.get('daily_gross_pnl', 0):+.4f}`\n"
                f"Fees paid:   `$-{risk.get('daily_fees', 0):.4f}`\n"
                f"*Net:        `${risk.get('daily_pnl', 0):+.4f}`*  "
                f"(high: `${risk.get('session_high_pnl', 0):+.4f}`)\n"
                f"Trades:      `{risk.get('trades_today', 0)}`"
                f"/`{p.get('max_daily_trades', '?')}`  │  "
                f"Streak: {streak}\n"
                f"Consec ↓:   `{risk.get('consecutive_losses', 0)}`  │  "
                f"Cooldown: `{risk.get('cooldown_remaining', 0):.0f}s`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔬 *Last Signal*\n"
                f"Type: `{sig_type}`   Conf: `{sig_conf:.1%}`\n"
                f"Δq: `{sig_dq:+.5f}`   Compute: `{sig_comp:.0f}µs`\n"
                f"Reason: `{_md_safe(sig_why[:80])}`",
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
        generate a tradeable signal on the last bar.

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
            lines = [f"🧠 *HPMS Decision Stack*  ⏱ `{_utc_now()}`"]
            price = self._data.get_last_price() if self._data else 0
            if price:
                lines.append(f"Price at query: `${price:,.1f}`")
            lines.append("")

            # ── LAYER 1: Phase-space ─────────────────────────────────────────
            lines.append(
                "*━━ LAYER 1: PHASE-SPACE STATE*\n"
                "_Where is the market in phase space right now?_"
            )
            ps = self._engine.get_phase_state() if self._engine else None
            if ps:
                p_arrow = "⬆️" if ps.p > 0 else ("⬇️" if ps.p < 0 else "➡️")
                dH_max  = getattr(self._config, "SIGNAL_DH_DT_MAX", 0.05) if self._config else 0.05
                dH_ok   = abs(ps.dH_dt) <= dH_max
                lines += [
                    f"  q (norm price position): `{ps.q:+.6f}`",
                    f"  p (momentum):            `{ps.p:+.6f}` {p_arrow}",
                    f"  H (total energy):        `{ps.H:.6f}`",
                    f"    ├ K kinetic   `{ps.kinetic:.4f}`  _(speed of price move)_",
                    f"    └ V potential `{ps.potential:.4f}`  _(landscape curvature)_",
                    f"  dH/dt (EMA):             `{ps.dH_dt:.6f}` "
                    f"{'🟢 conserved' if dH_ok else '🔴 non-conserved'}"
                    f"  (limit ±`{dH_max}`)",
                ]
            else:
                lines.append("  ⏳ Engine not yet ready — still warming up")

            # ── LAYER 2: KDE landscape ───────────────────────────────────────
            lines.append(
                "\n*━━ LAYER 2: POTENTIAL LANDSCAPE V(q)*\n"
                "_Has the engine built a probability density map?_"
            )
            d  = self._engine.get_diagnostics() if self._engine else {}
            ep = self._engine.get_params()       if self._engine else {}

            built       = d.get("landscape_built", False)
            bars_since  = d.get("bars_since_kde_build", 0)
            rebuild_int = d.get("kde_rebuild_interval", "?")
            hist_len    = d.get("history_len", 0)
            bw          = ep.get("kde_bandwidth", "?")
            grid        = ep.get("kde_grid_points", "?")

            lines += [
                f"  Built:     {_gate(built)}  "
                f"{'(landscape ready)' if built else '(waiting for enough history)'}",
                f"  Bandwidth: `{bw}`  Grid: `{grid}` pts",
                f"  Rebuild:   every `{rebuild_int}` bars  (last: `{bars_since}` bars ago)",
                f"  History:   `{hist_len}` bars  (need ≥ `{ep.get('lookback', '?')}`)",
            ]

            # ── LAYER 3: Signal gates ────────────────────────────────────────
            lines.append(
                "\n*━━ LAYER 3: SIGNAL GATES*\n"
                "_Do the Hamiltonian indicators meet the entry thresholds?_"
            )
            sig = self._strategy._last_signal if self._strategy else None
            if sig:
                cfg = self._config
                dq_thresh  = getattr(cfg, "SIGNAL_DELTA_Q_THRESHOLD", 0.0022) if cfg else 0.0022
                dH_thresh  = getattr(cfg, "SIGNAL_DH_DT_MAX",         0.05)   if cfg else 0.05
                mom_thresh = getattr(cfg, "SIGNAL_MIN_MOMENTUM",      0.0001)  if cfg else 0.0001

                dq_ok  = abs(sig.predicted_delta_q) >= dq_thresh
                dH_ok  = abs(sig.dH_dt) <= dH_thresh
                mom_ok = abs(sig.predicted_p_final) >= mom_thresh

                dq_bar = _pct_bar(abs(sig.predicted_delta_q), 0, dq_thresh * 3)
                dH_bar = _pct_bar(abs(sig.dH_dt), 0, dH_thresh * 2)

                lines += [
                    f"  {_gate(dq_ok)} Δq (predicted move):    `{sig.predicted_delta_q:+.6f}`  "
                    f"(need ±`{dq_thresh}`)\n             [{dq_bar}]",
                    f"  {_gate(dH_ok)} |dH/dt| (energy drift):  `{abs(sig.dH_dt):.6f}`  "
                    f"(max `{dH_thresh}`)\n             [{dH_bar}]",
                    f"  {_gate(mom_ok)} |p_final| (end momentum): `{abs(sig.predicted_p_final):.6f}`  "
                    f"(min `{mom_thresh}`)",
                    f"\n  ➤ Signal:  *{sig.signal_type.name}*  conf=`{sig.confidence:.1%}`",
                    f"  ➤ TP/SL:   `${sig.tp_price:,.1f}` / `${sig.sl_price:,.1f}`",
                    f"  ➤ Reason:  `{_md_safe(sig.reason[:100])}`",
                    f"  ➤ Compute: `{sig.compute_time_us:.0f}µs`",
                ]
            else:
                lines.append("  ⏳ No signal computed yet — engine warming up")

            # ── LAYER 4: Risk gate ───────────────────────────────────────────
            lines.append(
                "\n*━━ LAYER 4: RISK GATE*\n"
                "_Would the risk manager allow a trade right now?_"
            )
            if self._risk:
                can_trade, reason = self._risk.can_trade()
                rs       = self._risk.get_status()
                p_dict   = rs.get("params", {})
                cd       = rs.get("cooldown_remaining", 0)
                auto_rem = rs.get("auto_resume_remaining", 0)
                eff_cd   = p_dict.get("effective_cooldown", p_dict.get("cooldown", "?"))

                halt_line = "No"
                if rs.get("is_halted"):
                    halt_line = "Yes — " + _md_safe(rs.get("halt_reason", ""))
                    if auto_rem > 0:
                        halt_line += f" (auto-resume in `{auto_rem:.0f}s`)"

                pnl_bar = _pct_bar(
                    max(0.0, -rs.get("daily_pnl", 0)), 0,
                    p_dict.get("max_daily_loss", 200),
                )
                trd_bar = _pct_bar(
                    rs.get("trades_today", 0), 0,
                    p_dict.get("max_daily_trades", 50),
                )

                lines += [
                    f"  {_gate(can_trade)} Can trade: `{reason}`",
                    f"  Halted:    `{halt_line}`",
                    f"  Cooldown:  `{cd:.0f}s` remaining  (effective `{eff_cd}s`)",
                    f"  Daily PnL: `${rs.get('daily_pnl', 0):+.2f}` / limit `$-{p_dict.get('max_daily_loss', '?')}`\n"
                    f"             [{pnl_bar}]",
                    f"  Trades:    `{rs.get('trades_today', 0)}` / max `{p_dict.get('max_daily_trades', '?')}`\n"
                    f"             [{trd_bar}]",
                    f"  Consec ↓:  `{rs.get('consecutive_losses', 0)}` / max `{p_dict.get('max_consec_losses', '?')}`",
                ]
            else:
                lines.append("  ⚠️ Risk manager unavailable")

            # ── LAYER 5: Filter gate ─────────────────────────────────────────
            lines.append(
                "\n*━━ LAYER 5: MARKET FILTER GATE*\n"
                "_Are current market conditions suitable for entry?_"
            )
            lines.append(self._build_filter_inline())

            # ── VERDICT ─────────────────────────────────────────────────────
            lines.append("\n*━━ VERDICT*")
            if self._orders and self._orders.is_in_position:
                p_status = self._orders.get_status()
                lines.append(
                    f"  📍 *Already in position* — {p_status['side'].upper()} "
                    f"`{p_status['size']}c` @ `${p_status['entry_price']:,.1f}`  "
                    f"({p_status['bars_held']} bars)\n"
                    f"  _Watching for TP/SL or max-hold exit conditions._"
                )
            elif sig and sig.signal_type.name != "FLAT":
                ct = self._risk.can_trade()[0] if self._risk else False
                if ct:
                    lines.append(
                        "  🎯 *Signal present + risk OK — awaiting filter confirmation to fire*"
                    )
                else:
                    lines.append(
                        "  ⏳ *Signal present but risk gate is blocking entry*"
                    )
            else:
                lines.append("  👀 *No trade this bar* — engine sees no significant edge")

            try:
                await update.message.reply_text(
                    "\n".join(lines), parse_mode="Markdown"
                )
            except Exception:
                plain = "\n".join(lines).replace("`", "").replace("*", "").replace("_", "")
                await update.message.reply_text(plain)

        except Exception as e:
            logger.error("Thinking error: %s", e, exc_info=True)
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
            if bids and asks:
                try:
                    bb = float(bids[0][0])
                    ba = float(asks[0][0])
                    spread_pts = ba - bb
                    spread_pct = spread_pts / bb * 100 if bb else 0
                    max_sp = getattr(self._config, "FILTER_SPREAD_MAX_PCT", 0.05) \
                             if self._config else 0.05
                    sp_ok = spread_pct <= max_sp
                    spread_txt = (
                        f"`${spread_pts:.1f}` (`{spread_pct:.3f}%`) "
                        f"{_gate(sp_ok)} max `{max_sp}%`"
                    )
                except Exception:
                    pass

            depth_lines = []
            for a in list(reversed(asks[:3])):
                try:
                    depth_lines.append(
                        f"  ASK `${float(a[0]):>10,.1f}`  {int(float(a[1])):>6,}"
                    )
                except Exception:
                    pass
            depth_lines.append("  ─────────────────────────")
            for b in bids[:3]:
                try:
                    depth_lines.append(
                        f"  BID `${float(b[0]):>10,.1f}`  {int(float(b[1])):>6,}"
                    )
                except Exception:
                    pass

            candles = self._data.get_candles("1m", limit=15) if self._data else []
            atr_txt = "N/A"
            vol_txt = "N/A"
            if len(candles) >= 10:
                try:
                    recent  = candles[-10:]
                    atr     = sum(c.get("h", c.get("c", 0)) - c.get("l", c.get("c", 0))
                                  for c in recent) / len(recent)
                    atr_pct = atr / price * 100 if price else 0
                    vol_min = getattr(self._config, "FILTER_VOLATILITY_MIN_PCT", 0.01) \
                              if self._config else 0.01
                    vol_max = getattr(self._config, "FILTER_VOLATILITY_MAX_PCT", 2.0) \
                              if self._config else 2.0
                    vol_ok  = vol_min <= atr_pct <= vol_max
                    atr_txt = (
                        f"`${atr:.1f}` (`{atr_pct:.3f}%`) {_gate(vol_ok)}  "
                        f"range `{vol_min}%`–`{vol_max}%`"
                    )
                    last_vol = candles[-1].get("v", 0)
                    min_vol  = getattr(self._config, "FILTER_MIN_VOLUME_1M", 10.0) \
                               if self._config else 10.0
                    vol_ok2  = last_vol >= min_vol
                    vol_txt  = f"`{last_vol:.1f}` {_gate(vol_ok2)}  (min `{min_vol}`)"
                except Exception:
                    pass

            fresh_ok = False
            try:
                fresh_ok = self._data.is_price_fresh(max_stale_seconds=30)
            except Exception:
                pass

            lines = [
                f"📈 *Market Snapshot*  ⏱ `{_utc_now()}`\n",
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
    # /filter helper + command
    # ──────────────────────────────────────────────────────────────────────────

    def _build_filter_inline(self) -> str:
        """Build filter gate lines with pass/fail + threshold context."""
        if not self._data or not self._config:
            return "  ⚠️ Data or config unavailable — cannot evaluate filters"

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
                note = "✓ tight enough" if sp_ok else "✗ too wide for safe entry"
                lines.append(
                    f"  {_gate(sp_ok)} Spread:  `{spread_pct:.3f}%` / max `{max_sp}%`  _{note}_"
                )
            except Exception:
                lines.append("  ⚠️ Spread: n/a")

        candles = self._data.get_candles("1m", limit=15) or []
        price   = self._data.get_last_price() or 0

        if candles:
            last_vol = candles[-1].get("v", 0)
            min_vol  = getattr(cfg, "FILTER_MIN_VOLUME_1M", 10.0)
            vol_ok   = last_vol >= min_vol
            note = "✓ sufficient" if vol_ok else "✗ insufficient liquidity"
            lines.append(
                f"  {_gate(vol_ok)} Volume:  `{last_vol:.1f}` / min `{min_vol}`  _{note}_"
            )

        if len(candles) >= 10 and price:
            try:
                atr     = sum(
                    c.get("h", c.get("c", 0)) - c.get("l", c.get("c", 0))
                    for c in candles[-10:]
                ) / 10
                atr_pct = atr / price * 100
                vol_min = getattr(cfg, "FILTER_VOLATILITY_MIN_PCT", 0.01)
                vol_max = getattr(cfg, "FILTER_VOLATILITY_MAX_PCT", 2.0)
                vol_ok  = vol_min <= atr_pct <= vol_max
                if atr_pct < vol_min:
                    note = "✗ too quiet — no edge"
                elif atr_pct > vol_max:
                    note = "✗ too volatile — risk too high"
                else:
                    note = "✓ good volatility regime"
                lines.append(
                    f"  {_gate(vol_ok)} ATR 10b: `{atr_pct:.3f}%`  "
                    f"(range `{vol_min}%`–`{vol_max}%`)  _{note}_"
                )
            except Exception:
                pass

        return "\n".join(lines) if lines else "  ⏳ Not enough data to evaluate filters yet"

    async def _cmd_filter(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        try:
            price = self._data.get_last_price() if self._data else 0
            lines = [f"🔍 *Filter Gate @ `${price:,.1f}`*  ⏱ `{_utc_now()}`\n"]
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

            pnl_bar = _pct_bar(
                max(0.0, -rs.get("daily_pnl", 0)), 0, p.get("max_daily_loss", 200)
            )
            trd_bar = _pct_bar(
                rs.get("trades_today", 0), 0, p.get("max_daily_trades", 50)
            )
            cl_bar = _pct_bar(
                rs.get("consecutive_losses", 0), 0, p.get("max_consec_losses", 5)
            )

            auto_resume = rs.get("auto_resume_remaining", 0)
            halt_info   = "No"
            if rs.get("is_halted"):
                halt_info = "Yes — " + _md_safe(rs.get("halt_reason", ""))
                if auto_resume > 0:
                    halt_info += f" (auto-resume in `{auto_resume:.0f}s`)"

            eff_cd = p.get("effective_cooldown", p.get("cooldown", "?"))

            # Streak
            streak = _streak_icons(self._risk.get_trade_log(5))

            await update.message.reply_text(
                f"🛡 *Risk Gate Status*  ⏱ `{_utc_now()}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{gate_icon} *Can trade: `{reason}`*\n\n"
                f"Halted:     `{halt_info}`\n"
                f"Cooldown:   `{cd:.0f}s` remaining  (effective `{eff_cd}s`)\n\n"
                f"Daily PnL:   `${rs.get('daily_pnl', 0):+.4f}` / limit `$-{p.get('max_daily_loss', '?')}`\n"
                f"             [{pnl_bar}]\n"
                f"Trades:      `{rs.get('trades_today', 0)}` / max `{p.get('max_daily_trades', '?')}`\n"
                f"             [{trd_bar}]\n"
                f"Consec ↓:   `{rs.get('consecutive_losses', 0)}` / max `{p.get('max_consec_losses', '?')}`\n"
                f"             [{cl_bar}]\n"
                f"Streak:      {streak}\n"
                f"Sess. high: `${rs.get('session_high_pnl', 0):+.4f}`\n\n"
                f"*Limits:*\n"
                f"  Max pos:     `${p.get('max_pos_usd','?')}` / `{p.get('max_pos_contracts','?')}c`\n"
                f"  Leverage:    `{p.get('leverage','?')}x`\n"
                f"  Max DD:      `{p.get('max_dd_pct','?')}%`\n"
                f"  Equity%:     `{p.get('equity_pct','?')}%` per trade\n"
                f"  Auto-resume: `{p.get('auto_resume_sec','?')}s`\n"
                f"  Soft loss wt:`{p.get('soft_loss_weight','?')}`\n\n"
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
            await update.message.reply_text("No signal computed yet — warming up")
            return

        tp_dist = abs(s.tp_price - s.entry_price)
        sl_dist = abs(s.sl_price - s.entry_price)
        rr      = tp_dist / sl_dist if sl_dist > 0 else 0.0
        tp_pct  = tp_dist / s.entry_price * 100
        sl_pct  = sl_dist / s.entry_price * 100

        await update.message.reply_text(
            f"🔬 *Last Signal*  ⏱ `{_utc_now()}`\n\n"
            f"Type:       *{s.signal_type.name}*\n"
            f"Confidence: `{s.confidence:.1%}`\n"
            f"Predicted Δq: `{s.predicted_delta_q:+.6f}`\n"
            f"p_final:    `{s.predicted_p_final:.6f}`\n"
            f"H (energy): `{s.current_H:.6f}`\n"
            f"dH/dt:      `{s.dH_dt:.6f}`\n\n"
            f"Entry: `${s.entry_price:,.1f}`\n"
            f"TP:    `${s.tp_price:,.1f}` (`+${tp_dist:.1f}` / `+{tp_pct:.2f}%`)\n"
            f"SL:    `${s.sl_price:,.1f}` (`-${sl_dist:.1f}` / `-{sl_pct:.2f}%`)\n"
            f"R:R:   `{rr:.2f}:1`\n\n"
            f"Compute: `{s.compute_time_us:.0f}µs`\n"
            f"Reason: `{_md_safe(s.reason[:120])}`",
            parse_mode="Markdown",
        )

    async def _cmd_phase(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ps = self._engine.get_phase_state() if self._engine else None
        if not ps:
            await update.message.reply_text("Engine not ready yet — need more bars")
            return
        dH_max = getattr(self._config, "SIGNAL_DH_DT_MAX", 0.05) if self._config else 0.05
        dH_ok  = abs(ps.dH_dt) <= dH_max
        await update.message.reply_text(
            f"🌀 *Phase-Space State*  ⏱ `{_utc_now()}`\n\n"
            f"q (position):   `{ps.q:+.6f}`\n"
            f"p (momentum):   `{ps.p:+.6f}`\n"
            f"H (energy):     `{ps.H:.6f}`\n"
            f"  K (kinetic):  `{ps.kinetic:.6f}`\n"
            f"  V (potential):`{ps.potential:.6f}`\n"
            f"dH/dt (EMA):    `{ps.dH_dt:.6f}` "
            f"{'🟢 conserved' if dH_ok else '🔴 non-conserved'}\n"
            f"  threshold:    max `±{dH_max}`",
            parse_mode="Markdown",
        )

    async def _cmd_diag(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        d  = self._engine.get_diagnostics() if self._engine else {}
        ps = d.get("phase_state", {})
        hs = d.get("H_smoothing", {})
        text = (
            f"🔧 *Engine Diagnostics*  ⏱ `{_utc_now()}`\n\n"
            f"Landscape built: `{d.get('landscape_built')}`\n"
            f"KDE rebuild: `{d.get('bars_since_kde_build')}`/`{d.get('kde_rebuild_interval')}` bars\n"
            f"History len: `{d.get('history_len')}` bars\n"
            f"Signals gen: `{d.get('signal_count')}`\n"
            f"Traj logged: `{d.get('trajectory_log_depth')}`\n\n"
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

        cfg   = self._config
        lines = [f"⚙️ *Engine Parameters*  ⏱ `{_utc_now()}`\n"]

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
                ("kde_grid_points",      "Grid resolution"),
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
            "TP / SL (Fibonacci)": [
                ("fib_min_rr",           "Minimum R:R ratio"),
                ("fib_tp_cap_pct",       "TP cap (% of price)"),
                ("fib_sl_cap_pct",       "SL cap (% of price)"),
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
            for k in [
                "TRADE_DH_DT_EXIT_SPIKE", "TRADE_USE_BRACKET_ORDERS",
                "TRADE_ORDER_TYPE", "TRAILING_ENABLED",
                "TRAILING_WARMUP_BARS", "TRAILING_BE_ACTIVATION_PCT",
            ]:
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

        lines = [f"📋 *Recent Trades*  ⏱ `{_utc_now()}`\n"]
        total_gross = total_fees = total_net = 0.0

        for t in trades:
            emoji    = "💰" if t["net_pnl"] >= 0 else "🔻"
            move_usd = t["exit"] - t["entry"]
            move_pct = move_usd / t["entry"] * 100 if t["entry"] else 0.0
            lines.append(
                f"{emoji} `{t['time']}` *{t['side'].upper()}* `{t.get('size', 1)}c`\n"
                f"   `${t['entry']:,.1f}` → `${t['exit']:,.1f}` "
                f"(`{move_usd:+.1f}` / `{move_pct:+.2f}%`)\n"
                f"   Net `${t['net_pnl']:+.4f}` │ ROE `{t.get('roe_pct', 0):+.1f}%` "
                f"│ Fee `$-{t['fees']:.4f}` │ `{t['bars']}b` │ _{t['reason']}_"
            )
            total_gross += t["gross_pnl"]
            total_fees  += t["fees"]
            total_net   += t["net_pnl"]

        # Bug fix: was a bare string concat without \n — total line rendered garbled.
        pnl_icon = "💰" if total_net >= 0 else "🔻"
        lines.append(
            f"\n{pnl_icon} *Totals ({len(trades)} trades)*\n"
            f"Gross: `${total_gross:+.4f}` │ Fees: `$-{total_fees:.4f}`\n"
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
        pnl_bar   = _pct_bar(max(0.0, -net_pnl), 0, pnl_max)
        pnl_icon  = "💰" if net_pnl >= 0 else "🔻"

        # Per-trade average
        n_trades  = r.get("trades_today", 0)
        avg_net   = net_pnl / n_trades if n_trades > 0 else 0.0
        avg_fee   = fees    / n_trades if n_trades > 0 else 0.0

        # Streak
        streak = _streak_icons(self._risk.get_trade_log(5) if self._risk else [])

        await update.message.reply_text(
            f"{pnl_icon} *Daily P&L*  ⏱ `{_utc_now()}`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Gross:        `${gross_pnl:+.4f}`\n"
            f"Total fees:   `$-{fees:.4f}`\n"
            f"*Net P&L:     `${net_pnl:+.4f}`*  / limit `$-{pnl_max}`\n"
            f"              [{pnl_bar}]\n"
            f"Session high: `${r.get('session_high_pnl', 0):+.4f}`\n\n"
            f"Trades:       `{n_trades}` / `{p.get('max_daily_trades', '?')}`\n"
            f"Avg net/trade:`${avg_net:+.4f}`  │  Avg fee: `$-{avg_fee:.4f}`\n"
            f"Consec losses:`{r.get('consecutive_losses', 0)}` / `{p.get('max_consec_losses', '?')}`\n"
            f"Streak:       {streak}\n"
            f"Halted:       `{'Yes — ' + _md_safe(r.get('halt_reason', '')) if r.get('is_halted') else 'No'}`",
            parse_mode="Markdown",
        )

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        b     = self._api.get_balance("USD") if self._api else {}
        avail = b.get("available", 0)
        locked = b.get("locked", 0)
        total  = avail + locked
        await update.message.reply_text(
            f"💳 *Balance*  ⏱ `{_utc_now()}`\n\n"
            f"Available: `${avail:.2f}`\n"
            f"Locked:    `${locked:.2f}`\n"
            f"Total:     `${total:.2f}`",
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

        cv  = getattr(cfg, "TRADE_CONTRACT_VALUE", 0.001) if cfg else 0.001
        lev = getattr(cfg, "RISK_LEVERAGE", 10) if cfg else 10
        size = p["size"]

        notional  = p["entry_price"] * cv * size
        margin    = notional / max(lev, 1)

        # Unrealised PnL
        pnl_block = ""
        if price and p.get("entry_price"):
            diff = price - p["entry_price"]
            if p["side"] == "short":
                diff = -diff
            move_pct      = diff / p["entry_price"] * 100
            gross_unr     = diff * cv * size
            entry_fee_paid = getattr(self._orders, "_entry_fee_usd", 0.0)
            net_unr       = gross_unr - entry_fee_paid
            roe_pct       = net_unr / margin * 100 if margin > 0 else 0.0

            # Distance to TP and SL
            tp_price = getattr(self._orders, "_tp_price", None)
            sl_price = getattr(self._orders, "_sl_price", None)
            tp_dist_txt = (
                f"`${tp_price:,.1f}` (`{(abs(tp_price - price) / price * 100):.2f}%` away)"
                if tp_price else "N/A"
            )
            sl_dist_txt = (
                f"`${sl_price:,.1f}` (`{(abs(sl_price - price) / price * 100):.2f}%` away)"
                if sl_price else "N/A"
            )

            pnl_block = (
                f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"*Unrealised P&L (live):*\n"
                f"  Move:  `${diff:+.1f}` (`{move_pct:+.2f}%`)\n"
                f"  Gross: `${gross_unr:+.4f}`\n"
                f"  Entry fee paid: `$-{entry_fee_paid:.4f}`\n"
                f"  *Net (excl. exit fee): `${net_unr:+.4f}`*\n"
                f"  ROE: `{roe_pct:+.2f}%` on `${margin:.2f}` margin\n"
                f"\n🎯 TP:  {tp_dist_txt}\n"
                f"🛑 SL:  {sl_dist_txt}"
            )

        side_icon = "🟢" if p.get("side") == "long" else "🔴"
        await update.message.reply_text(
            f"{side_icon} *Open Position*  ⏱ `{_utc_now()}`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Side:     `{p['side'].upper()}`\n"
            f"Size:     `{size}` contracts\n"
            f"Entry:    `${p['entry_price']:,.1f}`\n"
            f"Price:    `${price:,.1f}`\n"
            f"Notional: `${notional:,.2f}` │ Margin: `${margin:.2f}` @ `{lev}x`\n"
            f"Bars held:`{p['bars_held']}`"
            f"{pnl_block}\n\n"
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
        lines = [f"📗 *Orderbook (Top 5)*  ⏱ `{_utc_now()}`\n```"]
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
            f"⚡ *Strategy STARTED*  ⏱ `{_utc_now()}`\n"
            f"Price: `${price:,.1f}`\n"
            f"_Scanning for signals on next bar close._",
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
            f"`{pos['size']}c` @ `${pos['entry_price']:,.1f}`"
            if pos.get("in_position") else "\nNo open positions."
        )
        await update.message.reply_text(
            f"⏸ *Strategy STOPPED*  ⏱ `{_utc_now()}`\n"
            f"No new entries will be placed.{pos_note}",
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
            actions.append("risk halted (TELEGRAM_HALT)")
        if self._orders:
            result = self._orders.emergency_flatten()
            actions += result.get("actions", [])
        await update.message.reply_text(
            f"⛔ *EMERGENCY HALT*  ⏱ `{_utc_now()}`\n\n"
            "Actions taken:\n" +
            "\n".join(f"• {a}" for a in actions) +
            "\n\n_Use /resume then /start\\_trading to re-enable._",
            parse_mode="Markdown",
        )

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._risk:
            await update.message.reply_text("No risk manager")
            return
        was = self._risk.resume()
        rs  = self._risk.get_status()
        await update.message.reply_text(
            f"✅ *Risk resumed*  ⏱ `{_utc_now()}`\n"
            f"Was halted: `{was or 'none'}`\n"
            f"Consecutive losses → `0`\n"
            f"Daily PnL: `${rs.get('daily_pnl', 0):+.4f}`  │  "
            f"Trades today: `{rs.get('trades_today', 0)}`\n\n"
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
            f"\n⚠️ Open position: {pos['side'].upper()} `{pos['size']}c` "
            f"@ `${pos['entry_price']:,.1f}` ({pos['bars_held']} bars)"
            if pos.get("in_position") else "\nNo open position."
        )
        await update.message.reply_text(
            f"🔄 *Risk lockout cleared*  ⏱ `{_utc_now()}`\n"
            f"Was: `{was or 'none'}`  →  Consecutive losses: `0`\n"
            f"Daily trades: `{rs.get('trades_today', 0)}`  │  "
            f"Daily PnL: `${rs.get('daily_pnl', 0):+.4f}`"
            f"{pos_note}\n\n"
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
            f"🔨 *Flatten complete*  ⏱ `{_utc_now()}`\n" +
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
        result = self._orders.close_position(
            reason="TELEGRAM_CLOSE", current_price=price
        )
        if result.get("success"):
            await update.message.reply_text(
                f"✅ *Position closed*  ⏱ `{_utc_now()}`\n"
                f"PnL: `${result.get('pnl_usd', 0):+.4f}`",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"❌ Close failed: `{result.get('error')}`",
                parse_mode="Markdown",
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
                f"❌ Unknown param: `{key}`\nUse /params to list all",
                parse_mode="Markdown",
            )

    async def _cmd_get(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/get <param>`", parse_mode="Markdown"
            )
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
            await update.message.reply_text(
                f"`{key}` = `{val}`", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"Unknown param: `{key}`\nUse /params to list all",
                parse_mode="Markdown",
            )

    async def _cmd_params(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        lines = [f"⚙️ *All Tunable Parameters*  ⏱ `{_utc_now()}`\n"]
        if self._engine:
            lines.append("*Engine:*")
            for k, v in self._engine.get_params().items():
                lines.append(f"  `{k}` = `{v}`")
        if self._risk:
            lines.append("\n*Risk:*")
            for k, v in self._risk.get_status().get("params", {}).items():
                lines.append(f"  `{k}` = `{v}`")

        trade_keys = [
            "TRADE_DH_DT_EXIT_SPIKE", "TRADE_USE_BRACKET_ORDERS",
            "TRADE_ORDER_TYPE", "TRADE_LIMIT_OFFSET_TICKS",
            "TRADE_CONTRACT_VALUE", "TRAILING_ENABLED",
            "TRAILING_WARMUP_BARS", "TRAILING_BE_ACTIVATION_PCT",
            "TRAILING_BE_FEE_MARGIN", "TRAILING_MIN_STEP_TICKS",
            "TRAILING_ABSOLUTE_MAX_BARS",
        ]
        filter_keys = [
            "FILTER_SPREAD_MAX_PCT", "FILTER_MIN_VOLUME_1M",
            "FILTER_VOLATILITY_MIN_PCT", "FILTER_VOLATILITY_MAX_PCT",
            "FILTER_NEWS_BLACKOUT_SECONDS",
        ]
        signal_keys = [
            "SIGNAL_DELTA_Q_THRESHOLD", "SIGNAL_DH_DT_MAX",
            "SIGNAL_H_PERCENTILE", "SIGNAL_MIN_MOMENTUM",
            "SIGNAL_ACCELERATION_CHECK",
        ]
        fib_keys = [
            "FIB_MIN_RR", "FIB_TP_CAP_PCT", "FIB_SL_CAP_PCT",
            "FIB_SL_ATR_BUFFER_MULT", "FIB_SWING_MIN_ORDER",
            "FIB_SWING_MAX_ORDER", "FIB_SWING_ATR_NOISE",
        ]
        if self._config:
            for section, keys in [
                ("Trade / Trailing", trade_keys),
                ("Signal", signal_keys),
                ("Filter", filter_keys),
                ("Fibonacci", fib_keys),
            ]:
                visible = [k for k in keys if hasattr(self._config, k)]
                if visible:
                    lines.append(f"\n*{section}:*")
                    for k in visible:
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
                new = (type(old)(val) if not isinstance(old, bool)
                       else val.lower() in ("true", "1"))
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
                new = (type(old)(val) if not isinstance(old, bool)
                       else val.lower() in ("true", "1"))
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
            cur = (self._risk.get_status().get("params", {}).get("cooldown", "?")
                   if self._risk else "?")
            await update.message.reply_text(
                f"Current cooldown: `{cur}s`\nUsage: `/cooldown <seconds>`",
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
            cur = (self._risk.get_status().get("params", {}).get("max_daily_loss", "?")
                   if self._risk else "?")
            await update.message.reply_text(
                f"Current max daily loss: `${cur}`\nUsage: `/maxloss <usd>`",
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
            cur = (self._risk.get_status().get("params", {}).get("max_pos_contracts", "?")
                   if self._risk else "?")
            await update.message.reply_text(
                f"Current max contracts: `{cur}`\nUsage: `/maxsize <n>`",
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
                f"❌ Invalid leverage `{val}` — must be 1–200",
                parse_mode="Markdown",
            )
            return False

        lines = [f"⚙️ Setting leverage to *{val}x*…"]

        if self._api:
            symbol = (
                getattr(self._config, "DELTA_SYMBOL", "BTCUSD")
                if self._config else "BTCUSD"
            )
            try:
                result = self._api.set_leverage(symbol=symbol, leverage=val)
                if result.get("success"):
                    lines.append(f"✅ Exchange confirmed `{val}x`")
                else:
                    lines.append(f"❌ Exchange rejected: `{result.get('error', 'unknown')}`")
                    await update.message.reply_text(
                        "\n".join(lines), parse_mode="Markdown"
                    )
                    return False
            except Exception as e:
                lines.append(f"❌ Exchange call failed: `{e}`")
                await update.message.reply_text(
                    "\n".join(lines), parse_mode="Markdown"
                )
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
            if self._api and self._orders:
                prod_id = getattr(self._orders, "_product_id", None)
                if prod_id:
                    try:
                        r = self._api.get_leverage(prod_id)
                        if r.get("success"):
                            exch_lev = r.get("result", {}).get("leverage", "?")
                    except Exception:
                        pass
            await update.message.reply_text(
                f"📊 *Current Leverage*  ⏱ `{_utc_now()}`\n\n"
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
                "❌ Must be an integer, e.g. `/leverage 50`",
                parse_mode="Markdown",
            )
            return
        await self._apply_leverage(update, val)
