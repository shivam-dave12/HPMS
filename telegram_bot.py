"""
telegram_bot.py — HPMS Telegram Control Interface
===================================================
Full remote control of the HPMS system via Telegram.

Commands:
  /start          — Welcome + command list
  /status         — Full system status
  /signal         — Last signal details
  /phase          — Current phase-space state (q, p, H, dH/dt)
  /diag           — Engine diagnostics
  /trades         — Recent trade log
  /pnl            — Daily P&L summary
  /balance        — Exchange balance
  /position       — Current position details
  /orderbook      — Top-of-book snapshot
  /price          — Current price

  /start_trading  — Enable strategy execution
  /stop_trading   — Disable strategy (no new entries; open positions remain)
  /halt           — Emergency: stop + cancel all orders + flatten
  /resume         — Resume from halt
  /flatten        — Force-close all positions + cancel orders
  /close          — Close current position at market

  /set <p> <v>    — Set any parameter (engine, risk, or config)
  /get <p>        — Get current parameter value
  /params         — List all tunable parameters

  /set_engine <p> <v>
  /set_risk   <p> <v>
  /set_trade  <p> <v>
  /set_filter <p> <v>

  /leverage <N>     — Set exchange leverage (or view current)
  /cooldown <sec>   — Set inter-trade cooldown
  /maxloss <usd>    — Set daily loss limit
  /maxsize <n>      — Set max position contracts

  /help  — Command list
  /ping  — Latency check
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

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


class TelegramBot:
    """
    Thread-safe Telegram command interface for the HPMS trading system.

    Race-condition fix: send_message buffers outgoing messages until the
    async event loop is confirmed ready, then flushes. This ensures startup
    notifications aren't silently dropped.
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
        self._chat_id  = chat_id
        self._admin_ids: Set[int] = set(admin_ids)
        self._strategy = strategy
        self._engine   = engine
        self._risk     = risk_mgr
        self._orders   = order_mgr
        self._data     = data_mgr
        self._api      = api
        self._config   = config

        self._app:    Optional[Any]                          = None
        self._thread: Optional[threading.Thread]            = None
        self._loop:   Optional[asyncio.AbstractEventLoop]   = None
        self._running = False
        self._ready   = threading.Event()   # set once the bot loop is live

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
        self._thread = threading.Thread(target=self._run_bot, daemon=True, name="TelegramBot")
        self._thread.start()
        # Wait up to 15 s for the bot to come online before returning
        if not self._ready.wait(timeout=15):
            logger.warning("Telegram bot did not become ready in 15s — continuing anyway")

    def _run_bot(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_start())

    async def _async_start(self):
        self._app = Application.builder().token(self._token).build()

        # Register all command handlers
        cmds = {
            "start":         self._cmd_start,
            "help":          self._cmd_start,
            "ping":          self._cmd_ping,
            "status":        self._cmd_status,
            "signal":        self._cmd_signal,
            "phase":         self._cmd_phase,
            "diag":          self._cmd_diag,
            "trades":        self._cmd_trades,
            "pnl":           self._cmd_pnl,
            "balance":       self._cmd_balance,
            "position":      self._cmd_position,
            "orderbook":     self._cmd_orderbook,
            "price":         self._cmd_price,
            "start_trading": self._cmd_start_trading,
            "stop_trading":  self._cmd_stop_trading,
            "halt":          self._cmd_halt,
            "resume":        self._cmd_resume,
            "flatten":       self._cmd_flatten,
            "close":         self._cmd_close,
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

        # Set bot menu commands (visible in Telegram)
        await self._app.bot.set_my_commands([
            BotCommand("status",        "System status"),
            BotCommand("price",         "Current price"),
            BotCommand("signal",        "Last signal"),
            BotCommand("phase",         "Phase-space state"),
            BotCommand("pnl",           "Daily P&L"),
            BotCommand("trades",        "Trade log"),
            BotCommand("position",      "Open position"),
            BotCommand("balance",       "Exchange balance"),
            BotCommand("start_trading", "▶ Enable trading"),
            BotCommand("stop_trading",  "⏸ Disable trading"),
            BotCommand("halt",          "⛔ Emergency halt"),
            BotCommand("resume",        "✅ Resume from halt"),
            BotCommand("flatten",       "🔨 Close all"),
            BotCommand("params",        "All parameters"),
            BotCommand("set",           "Set parameter"),
            BotCommand("help",          "Command list"),
        ])

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        self._running = True
        logger.info("Telegram bot online")
        self._ready.set()  # unblock TelegramBot.start()

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

        Blocks until the bot loop is ready (up to 20 s on startup).
        Silently drops the message if the loop never started.
        """
        if not self._chat_id:
            return
        # Wait for readiness before trying to fire into the loop.
        # After startup this returns immediately (event already set).
        if not self._ready.wait(timeout=20):
            return
        if not self._app or not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="Markdown",
                ),
                self._loop,
            )
        except Exception as e:
            logger.debug(f"send_message error: {e}")

    # ─── COMMANDS: INFO ───────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🔬 *HPMS — Hamiltonian Phase-Space Micro-Scalping*\n\n"
            "📊 *Info*\n"
            "/status — Full system status\n"
            "/price — Current price\n"
            "/signal — Last signal\n"
            "/phase — Phase-space (q,p,H)\n"
            "/diag — Engine diagnostics\n"
            "/pnl — Daily P&L\n"
            "/trades — Trade log\n"
            "/balance — Exchange balance\n"
            "/position — Open position\n"
            "/orderbook — Top of book\n\n"
            "⚡ *Controls*\n"
            "/start\\_trading — ▶ Enable\n"
            "/stop\\_trading — ⏸ Disable (positions remain)\n"
            "/halt — ⛔ Emergency stop + flatten\n"
            "/resume — ✅ Resume from halt\n"
            "/flatten — 🔨 Close all positions\n"
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

    async def _cmd_ping(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        t0 = time.time()
        msg = await update.message.reply_text("⏱ …")
        rtt = (time.time() - t0) * 1000
        await msg.edit_text(f"🏓 Pong — {rtt:.0f}ms RTT — "
                            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

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
                spread_txt = f" | Spread: `${ba - bb:.1f}` ({(ba-bb)/bb*100:.3f}%)"
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
            s    = self._strategy.get_status() if self._strategy else {}
            risk = s.get("risk", {})
            pos  = s.get("position", {})
            sig  = s.get("last_signal", {})
            price  = self._data.get_last_price() if self._data else 0
            dm_ok  = self._data.is_ready if self._data else False

            pos_txt = (f"📈 {pos['side'].upper()} {pos['size']}c @ ${pos['entry_price']:,.1f} "
                       f"({pos['bars_held']}bars)"
                       if pos.get("in_position") else "⬜ Flat")

            halt_txt = f"🔴 HALTED: {risk.get('halt_reason', '')}" if risk.get("is_halted") else "🟢 OK"

            await update.message.reply_text(
                f"📊 *HPMS Status*\n\n"
                f"Strategy: `{'▶ ON' if s.get('enabled') else '⏸ OFF'}`\n"
                f"Data: `{'🟢 Ready' if dm_ok else '🔴 Not Ready'}`\n"
                f"Risk: `{halt_txt}`\n"
                f"Position: {pos_txt}\n"
                f"Price: `${price:,.1f}`\n"
                f"Bars processed: `{s.get('bar_count', 0)}`\n\n"
                f"💰 *Today*\n"
                f"PnL: `${risk.get('daily_pnl', 0):+.2f}` "
                f"(high `${risk.get('session_high_pnl', 0):+.2f}`)\n"
                f"Trades: `{risk.get('trades_today', 0)}`/"
                f"`{risk.get('params', {}).get('max_daily_trades', '?')}`\n"
                f"Consec losses: `{risk.get('consecutive_losses', 0)}`\n"
                f"Cooldown: `{risk.get('cooldown_remaining', 0):.0f}s`\n\n"
                f"🔬 *Last Signal*\n"
                f"Type: `{sig.get('type', 'N/A')}`\n"
                f"Δq: `{sig.get('delta_q', 0):.5f}` | "
                f"Conf: `{(sig.get('confidence') or 0):.1%}`\n"
                f"Compute: `{(sig.get('compute_us') or 0):.0f}µs`",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        s = self._strategy._last_signal if self._strategy else None
        if not s:
            await update.message.reply_text("No signal computed yet")
            return
        await update.message.reply_text(
            f"🔬 *Last Signal*\n\n"
            f"Type: *{s.signal_type.name}*\n"
            f"Confidence: `{s.confidence:.1%}`\n"
            f"Predicted Δq: `{s.predicted_delta_q:+.6f}`\n"
            f"p\\_final: `{s.predicted_p_final:.6f}`\n"
            f"H: `{s.current_H:.6f}`\n"
            f"dH/dt: `{s.dH_dt:.6f}`\n"
            f"Entry: `${s.entry_price:,.1f}`\n"
            f"TP: `${s.tp_price:,.1f}` | SL: `${s.sl_price:,.1f}`\n"
            f"Compute: `{s.compute_time_us:.0f}µs`\n"
            f"Reason: `{s.reason}`",
            parse_mode="Markdown",
        )

    async def _cmd_phase(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ps = self._engine.get_phase_state() if self._engine else None
        if not ps:
            await update.message.reply_text("Engine not ready yet")
            return
        await update.message.reply_text(
            f"🌀 *Phase Space*\n\n"
            f"q (position):  `{ps.q:.6f}`\n"
            f"p (momentum):  `{ps.p:.6f}`\n"
            f"H (energy):    `{ps.H:.6f}`\n"
            f"K (kinetic):   `{ps.kinetic:.6f}`\n"
            f"V (potential): `{ps.potential:.6f}`\n"
            f"dH/dt (EMA):   `{ps.dH_dt:.6f}`",
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
            f"KDE rebuild: `{d.get('bars_since_kde_build')}/{d.get('kde_rebuild_interval')}` bars\n"
            f"History: `{d.get('history_len')}` bars\n"
            f"Signals generated: `{d.get('signal_count')}`\n\n"
            f"*Phase State:*\n"
            f"q=`{ps.get('q')}` p=`{ps.get('p')}`\n"
            f"H=`{ps.get('H')}` dH/dt\\_ema=`{ps.get('dH_dt_ema')}`\n\n"
            f"*H Smoothing (span={hs.get('ema_span')}):*\n"
            f"H\\_raw=`{hs.get('H_raw')}` H\\_ema=`{hs.get('H_ema')}`\n"
            f"dH\\_raw=`{hs.get('dH_raw')}` dH\\_ema=`{hs.get('dH_ema')}`"
        )
        traj = d.get("last_trajectory")
        if traj:
            text += f"\n\n*Last Trajectory ({len(traj)} pts):*\n```\n"
            for tp in traj:
                text += f"  t={tp['t']:.0f} q={tp['q']:+.5f} p={tp['p']:+.5f} H={tp['H']:.5f}\n"
            text += "```"
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        trades = self._risk.get_trade_log(10) if self._risk else []
        if not trades:
            await update.message.reply_text("No trades today yet")
            return
        lines = ["📋 *Recent Trades*\n"]
        total_pnl = 0.0
        for t in trades:
            emoji = "💰" if t["pnl"] >= 0 else "🔻"
            lines.append(
                f"{emoji} `{t['time']}` {t['side'].upper()} "
                f"`${t['pnl']:+.2f}` ({t['bars']}b) _{t['reason']}_"
            )
            total_pnl += t["pnl"]
        lines.append(f"\nTotal shown: `${total_pnl:+.2f}`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        r = self._risk.get_status() if self._risk else {}
        p = r.get("params", {})
        await update.message.reply_text(
            f"💰 *Daily P&L*\n\n"
            f"PnL: `${r.get('daily_pnl', 0):+.2f}` / "
            f"limit `${p.get('max_daily_loss', '?')}`\n"
            f"Session high: `${r.get('session_high_pnl', 0):+.2f}`\n"
            f"Trades: `{r.get('trades_today', 0)}` / `{p.get('max_daily_trades', '?')}`\n"
            f"Consec losses: `{r.get('consecutive_losses', 0)}` / "
            f"`{p.get('max_consec_losses', '?')}`\n"
            f"Halted: `{'Yes — ' + r.get('halt_reason', '') if r.get('is_halted') else 'No'}`",
            parse_mode="Markdown",
        )

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        b = self._api.get_balance("USD") if self._api else {}
        await update.message.reply_text(
            f"💳 *Balance*\n\n"
            f"Available: `${b.get('available', 0):.2f}`\n"
            f"Locked:    `${b.get('locked', 0):.2f}`",
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
        unrealised = ""
        if price and p.get("entry_price"):
            diff = price - p["entry_price"]
            if p["side"] == "short":
                diff = -diff
            cv = getattr(self._config, "TRADE_CONTRACT_VALUE", 0.001) if self._config else 0.001
            pnl = diff * cv * p["size"]
            unrealised = f"\nUnrealised PnL: `${pnl:+.2f}`"
        await update.message.reply_text(
            f"📈 *Open Position*\n\n"
            f"Side: `{p['side'].upper()}`\n"
            f"Size: `{p['size']}` contracts\n"
            f"Entry: `${p['entry_price']:,.1f}`\n"
            f"Current: `${price:,.1f}`{unrealised}\n"
            f"Bars held: `{p['bars_held']}`\n"
            f"SL order: `{p.get('sl_order', 'N/A')}`\n"
            f"TP order: `{p.get('tp_order', 'N/A')}`",
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
            lines.append(f"  ASK {float(a[0]):>10,.1f}  {float(a[1]):>8,.0f}")
        lines.append("  ─────────────────────────")
        for b in bids:
            lines.append(f"  BID {float(b[0]):>10,.1f}  {float(b[1]):>8,.0f}")
        lines.append("```")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ─── COMMANDS: TRADING CONTROLS ───────────────────────────────────────────

    async def _cmd_start_trading(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._strategy:
            await update.message.reply_text("No strategy available")
            return
        if self._strategy.is_enabled:
            await update.message.reply_text("⚡ Strategy is already running")
            return
        # Check risk is not halted
        if self._risk:
            risk_st = self._risk.get_status()
            if risk_st.get("is_halted"):
                await update.message.reply_text(
                    f"⚠️ Risk manager is HALTED (`{risk_st['halt_reason']}`)\n"
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
        pos_note = (f"\n⚠️ Open position remains: {pos['side'].upper()} "
                    f"{pos['size']}c @ ${pos['entry_price']:,.1f}"
                    if pos.get("in_position") else "\nNo open positions.")
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
            f"Actions taken:\n" + "\n".join(f"• {a}" for a in actions) + "\n\n"
            f"Use /resume then /start\\_trading to re-enable.",
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
            f"Consecutive losses reset to 0.\n"
            f"Use /start\\_trading to re-enable the strategy.",
            parse_mode="Markdown",
        )

    async def _cmd_flatten(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not self._orders:
            await update.message.reply_text("No order manager")
            return
        result = self._orders.emergency_flatten()
        actions = result.get("actions", [])
        await update.message.reply_text(
            f"🔨 *Flatten complete*\n" +
            ("\n".join(f"• {a}" for a in actions) if actions else "• No open orders/positions"),
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
                f"✅ Position closed\nPnL: `${result.get('pnl_usd', 0):+.2f}`",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(f"❌ Close failed: `{result.get('error')}`",
                                            parse_mode="Markdown")

    # ─── COMMANDS: PARAMETERS ────────────────────────────────────────────────

    async def _cmd_set(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: `/set <param> <value>`", parse_mode="Markdown")
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
                new_val = type(old)(val) if not isinstance(old, bool) else val.lower() in ("true", "1")
                setattr(self._config, key.upper(), new_val)
                updated = True
            except Exception:
                pass

        if updated:
            await update.message.reply_text(f"✅ `{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Unknown param: `{key}`", parse_mode="Markdown")

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
            await update.message.reply_text(f"Unknown param: `{key}`", parse_mode="Markdown")

    async def _cmd_params(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        lines = ["⚙️ *All Parameters*\n"]
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
            "TRADE_ORDER_TYPE", "FILTER_SPREAD_MAX_PCT",
            "FILTER_MIN_VOLUME_1M", "FILTER_VOLATILITY_MIN_PCT",
            "FILTER_VOLATILITY_MAX_PCT", "SIGNAL_DELTA_Q_THRESHOLD",
        ]
        if self._config:
            lines.append("\n*Trade / Filter:*")
            for k in trade_keys:
                if hasattr(self._config, k):
                    lines.append(f"  `{k}` = `{getattr(self._config, k)}`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_set_engine(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: `/set_engine <param> <value>`",
                                            parse_mode="Markdown")
            return
        key, val = ctx.args[0], ctx.args[1]
        if self._engine and self._engine.update_param(key, val):
            await update.message.reply_text(f"✅ Engine `{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Unknown engine param: `{key}`",
                                            parse_mode="Markdown")

    async def _cmd_set_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: `/set_risk <param> <value>`",
                                            parse_mode="Markdown")
            return
        key, val = ctx.args[0], ctx.args[1]
        if key.lower() == "leverage":
            try:
                await self._apply_leverage(update, int(val))
            except ValueError:
                await update.message.reply_text("❌ Leverage must be an integer")
            return
        if self._risk and self._risk.update_param(key, val):
            await update.message.reply_text(f"✅ Risk `{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Unknown risk param: `{key}`",
                                            parse_mode="Markdown")

    async def _cmd_set_trade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: `/set_trade <param> <value>`",
                                            parse_mode="Markdown")
            return
        key     = ctx.args[0]
        val     = ctx.args[1]
        cfg_key = f"TRADE_{key.upper()}"
        if self._config and hasattr(self._config, cfg_key):
            try:
                old = getattr(self._config, cfg_key)
                new = type(old)(val) if not isinstance(old, bool) else val.lower() in ("true","1")
                setattr(self._config, cfg_key, new)
                await update.message.reply_text(f"✅ `{cfg_key}` = `{new}`", parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        else:
            await update.message.reply_text(f"Unknown: `{cfg_key}`", parse_mode="Markdown")

    async def _cmd_set_filter(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: `/set_filter <param> <value>`",
                                            parse_mode="Markdown")
            return
        key     = ctx.args[0]
        val     = ctx.args[1]
        cfg_key = f"FILTER_{key.upper()}"
        if self._config and hasattr(self._config, cfg_key):
            try:
                old = getattr(self._config, cfg_key)
                new = type(old)(val) if not isinstance(old, bool) else val.lower() in ("true","1")
                setattr(self._config, cfg_key, new)
                await update.message.reply_text(f"✅ `{cfg_key}` = `{new}`", parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        else:
            await update.message.reply_text(f"Unknown: `{cfg_key}`", parse_mode="Markdown")

    # ─── SHORTCUTS ────────────────────────────────────────────────────────────

    async def _cmd_cooldown(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            cur = self._risk.get_status().get("params", {}).get("cooldown", "?") if self._risk else "?"
            await update.message.reply_text(f"Current cooldown: `{cur}s`  Usage: `/cooldown <seconds>`",
                                            parse_mode="Markdown")
            return
        if self._risk and self._risk.update_param("cooldown_seconds", ctx.args[0]):
            await update.message.reply_text(f"✅ Cooldown: `{ctx.args[0]}s`", parse_mode="Markdown")

    async def _cmd_maxloss(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            cur = self._risk.get_status().get("params", {}).get("max_daily_loss", "?") if self._risk else "?"
            await update.message.reply_text(f"Current max daily loss: `${cur}`  Usage: `/maxloss <usd>`",
                                            parse_mode="Markdown")
            return
        if self._risk and self._risk.update_param("max_daily_loss_usd", ctx.args[0]):
            await update.message.reply_text(f"✅ Max daily loss: `${ctx.args[0]}`",
                                            parse_mode="Markdown")

    async def _cmd_maxsize(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            cur = self._risk.get_status().get("params", {}).get("max_pos_contracts", "?") if self._risk else "?"
            await update.message.reply_text(f"Current max contracts: `{cur}`  Usage: `/maxsize <n>`",
                                            parse_mode="Markdown")
            return
        if self._risk and self._risk.update_param("max_position_contracts", ctx.args[0]):
            await update.message.reply_text(f"✅ Max contracts: `{ctx.args[0]}`",
                                            parse_mode="Markdown")

    # ─── LEVERAGE HELPER ──────────────────────────────────────────────────────

    async def _apply_leverage(self, update: Update, val: int) -> bool:
        if val <= 0 or val > 200:
            await update.message.reply_text(
                f"❌ Invalid leverage `{val}` — must be 1–200",
                parse_mode="Markdown",
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
            await update.message.reply_text("❌ Must be an integer, e.g. `/leverage 50`",
                                            parse_mode="Markdown")
            return
        await self._apply_leverage(update, val)
