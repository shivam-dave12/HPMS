"""
telegram_bot.py — HPMS Telegram Control Interface
===================================================
Full remote control of the HPMS system via Telegram commands.

Commands:
  /start         — Show welcome + command list
  /status        — Full system status (engine, risk, position, data)
  /signal        — Last signal details
  /phase         — Current phase-space state (q, p, H, dH/dt)
  /diag          — Engine diagnostics
  /trades        — Recent trade log
  /pnl           — Daily P&L summary
  /balance       — Exchange balance
  /position      — Current position details
  /orderbook     — Top-of-book snapshot

  /start_trading — Enable strategy execution
  /stop_trading  — Disable strategy execution (no new entries)
  /halt          — Emergency halt: stop + cancel all orders + flatten
  /resume        — Resume from halt state
  /flatten       — Force-close position + cancel orders
  /close         — Close current position at market

  /set <param> <value> — Update any parameter at runtime
  /get <param>         — Get current parameter value
  /params              — List all tunable parameters + current values

  /set_engine <param> <value>  — Update HPMS engine parameter
  /set_risk <param> <value>    — Update risk parameter
  /set_trade <param> <value>   — Update trade execution parameter
  /set_filter <param> <value>  — Update filter parameter

  /leverage <value>    — Set exchange leverage
  /cooldown <seconds>  — Set inter-trade cooldown
  /maxloss <usd>       — Set daily loss limit
  /maxsize <contracts> — Set max position contracts

  /help          — Show this command list
  /ping          — Health check
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Lazy import — telegram may not be installed in all environments
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
    Telegram command interface for the HPMS trading system.

    Designed for institutional-grade remote control:
      - All parameters tunable without restart
      - Emergency flatten/halt accessible in <1 second
      - Real-time push notifications for entries, exits, risk events
      - Admin whitelist for security
    """

    def __init__(
        self,
        token:      str,
        chat_id:    str,
        admin_ids:  list,
        strategy    = None,
        engine      = None,
        risk_mgr    = None,
        order_mgr   = None,
        data_mgr    = None,
        api         = None,
        config      = None,
    ):
        self._token     = token
        self._chat_id   = chat_id
        self._admin_ids: Set[int] = set(admin_ids)
        self._strategy  = strategy
        self._engine    = engine
        self._risk      = risk_mgr
        self._orders    = order_mgr
        self._data      = data_mgr
        self._api       = api
        self._config    = config

        self._app: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    # ─── AUTH CHECK ───────────────────────────────────────────────────────────

    def _is_admin(self, update: Update) -> bool:
        uid = update.effective_user.id if update.effective_user else 0
        if not self._admin_ids:
            return True  # no whitelist = allow all
        return uid in self._admin_ids

    async def _guard(self, update: Update) -> bool:
        if not self._is_admin(update):
            await update.message.reply_text("⛔ Unauthorized")
            return False
        return True

    # ─── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self):
        if not HAS_TELEGRAM or not self._token:
            logger.warning("Telegram bot not started (no token or lib missing)")
            return

        self._thread = threading.Thread(target=self._run_bot, daemon=True, name="TelegramBot")
        self._thread.start()
        logger.info("Telegram bot thread started")

    def _run_bot(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_start())

    async def _async_start(self):
        self._app = Application.builder().token(self._token).build()

        # Register commands
        handlers = {
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
        for name, handler in handlers.items():
            self._app.add_handler(CommandHandler(name, handler))

        # Set bot commands for menu
        await self._app.bot.set_my_commands([
            BotCommand("status",        "System status"),
            BotCommand("signal",        "Last signal"),
            BotCommand("phase",         "Phase-space state"),
            BotCommand("pnl",           "Daily P&L"),
            BotCommand("trades",        "Trade log"),
            BotCommand("position",      "Position info"),
            BotCommand("start_trading", "Enable trading"),
            BotCommand("stop_trading",  "Disable trading"),
            BotCommand("halt",          "Emergency halt"),
            BotCommand("resume",        "Resume from halt"),
            BotCommand("flatten",       "Flatten all"),
            BotCommand("params",        "All parameters"),
            BotCommand("set",           "Set parameter"),
            BotCommand("help",          "Command list"),
        ])

        self._running = True
        logger.info("✅ Telegram bot online")

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        # Keep alive
        while self._running:
            await asyncio.sleep(1)

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def stop(self):
        self._running = False

    # ─── PUSH NOTIFICATION ────────────────────────────────────────────────────

    def send_message(self, text: str):
        """Thread-safe push notification to the configured chat."""
        if not self._app or not self._chat_id or not self._loop:
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
            logger.debug(f"Send message error: {e}")

    # ─── COMMAND HANDLERS ─────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = (
            "🔬 *HPMS — Hamiltonian Phase-Space Micro-Scalping*\n\n"
            "📊 *Info*\n"
            "/status — System status\n"
            "/signal — Last signal\n"
            "/phase — Phase-space (q,p,H)\n"
            "/diag — Diagnostics\n"
            "/pnl — Daily P&L\n"
            "/trades — Trade log\n"
            "/balance — Balance\n"
            "/position — Position\n"
            "/orderbook — Top of book\n\n"
            "⚡ *Controls*\n"
            "/start\\_trading — Enable\n"
            "/stop\\_trading — Disable\n"
            "/halt — Emergency stop\n"
            "/resume — Resume\n"
            "/flatten — Close all\n"
            "/close — Close position\n\n"
            "⚙️ *Config*\n"
            "/params — All parameters\n"
            "/set <param> <value>\n"
            "/get <param>\n"
            "/set\\_engine <p> <v>\n"
            "/set\\_risk <p> <v>\n"
            "/leverage <N>\n"
            "/cooldown <sec>\n"
            "/maxloss <usd>\n"
            "/maxsize <contracts>\n"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_ping(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f"🏓 Pong — {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        try:
            s = self._strategy.get_status() if self._strategy else {}
            risk = s.get("risk", {})
            pos  = s.get("position", {})
            eng  = s.get("engine", {})
            sig  = s.get("last_signal", {})

            price = self._data.get_last_price() if self._data else 0
            dm_ready = self._data.is_ready if self._data else False

            text = (
                f"📊 *HPMS Status*\n\n"
                f"Strategy: {'🟢 ON' if s.get('enabled') else '🔴 OFF'}\n"
                f"Data: {'🟢 Ready' if dm_ready else '🔴 Not Ready'}\n"
                f"Risk: {'🔴 HALTED' if risk.get('is_halted') else '🟢 OK'}\n"
                f"Position: {'📈 ' + pos.get('side', '').upper() if pos.get('in_position') else '⬜ Flat'}\n"
                f"Price: ${price:,.1f}\n"
                f"Bars: {s.get('bar_count', 0)}\n\n"
                f"💰 *Risk*\n"
                f"Daily PnL: ${risk.get('daily_pnl', 0):+.2f}\n"
                f"Trades: {risk.get('trades_today', 0)}/{risk.get('params', {}).get('max_daily_trades', '-')}\n"
                f"Consec losses: {risk.get('consecutive_losses', 0)}\n"
                f"Cooldown: {risk.get('cooldown_remaining', 0):.0f}s\n\n"
                f"🔬 *Last Signal*\n"
                f"Type: {sig.get('type', 'N/A')}\n"
                f"Δq: {sig.get('delta_q', 0):.5f}\n"
                f"Confidence: {(sig.get('confidence') or 0):.1%}\n"
                f"Compute: {(sig.get('compute_us') or 0):.0f}µs"
            )
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        s = self._strategy._last_signal if self._strategy else None
        if not s:
            await update.message.reply_text("No signal yet")
            return
        text = (
            f"🔬 *Last Signal*\n\n"
            f"Type: *{s.signal_type.name}*\n"
            f"Confidence: {s.confidence:.1%}\n"
            f"Predicted Δq: {s.predicted_delta_q:.6f}\n"
            f"p\\_final: {s.predicted_p_final:.6f}\n"
            f"H: {s.current_H:.6f}\n"
            f"dH/dt: {s.dH_dt:.6f}\n"
            f"Entry: ${s.entry_price:,.1f}\n"
            f"TP: ${s.tp_price:,.1f}\n"
            f"SL: ${s.sl_price:,.1f}\n"
            f"Compute: {s.compute_time_us:.0f}µs\n"
            f"Reason: {s.reason}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_phase(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ps = self._engine.get_phase_state() if self._engine else None
        if not ps:
            await update.message.reply_text("Engine not ready")
            return
        text = (
            f"🌀 *Phase Space*\n\n"
            f"q (position): {ps.q:.6f}\n"
            f"p (momentum): {ps.p:.6f}\n"
            f"H (energy):   {ps.H:.6f}\n"
            f"K (kinetic):  {ps.kinetic:.6f}\n"
            f"V (potential): {ps.potential:.6f}\n"
            f"dH/dt:        {ps.dH_dt:.6f}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_diag(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        d = self._engine.get_diagnostics() if self._engine else {}
        ps = d.get("phase_state", {})
        hs = d.get("H_smoothing", {})

        text = (
            f"🔧 *Diagnostics*\n\n"
            f"Landscape built: {d.get('landscape_built')}\n"
            f"KDE rebuild: {d.get('bars_since_kde_build')}/{d.get('kde_rebuild_interval')} bars\n"
            f"History length: {d.get('history_len')}\n"
            f"Signal count: {d.get('signal_count')}\n\n"
            f"*Phase State:*\n"
            f"q={ps.get('q')}, p={ps.get('p')}\n"
            f"H={ps.get('H')}\n"
            f"dH/dt\\_ema={ps.get('dH_dt_ema')}\n\n"
            f"*H Smoothing (EMA span={hs.get('ema_span')}):*\n"
            f"H\\_raw={hs.get('H_raw')} H\\_ema={hs.get('H_ema')}\n"
            f"dH\\_raw={hs.get('dH_raw')} dH\\_ema={hs.get('dH_ema')}\n"
        )

        # Append last trajectory summary
        traj = d.get("last_trajectory")
        if traj:
            text += f"\n*Last Trajectory ({len(traj)} pts):*\n```\n"
            for tp in traj:
                text += f"  t={tp['t']:.0f} q={tp['q']:+.5f} p={tp['p']:+.5f} H={tp['H']:.5f}\n"
            text += "```"

        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        trades = self._risk.get_trade_log(10) if self._risk else []
        if not trades:
            await update.message.reply_text("No trades today")
            return
        lines = ["📋 *Recent Trades*\n"]
        for t in trades:
            emoji = "💰" if t["pnl"] >= 0 else "🔻"
            lines.append(
                f"{emoji} {t['time']} {t['side'].upper()} "
                f"${t['pnl']:+.2f} ({t['bars']}bars) {t['reason']}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        r = self._risk.get_status() if self._risk else {}
        text = (
            f"💰 *Daily P&L*\n\n"
            f"PnL: ${r.get('daily_pnl', 0):+.2f}\n"
            f"Session high: ${r.get('session_high_pnl', 0):+.2f}\n"
            f"Trades: {r.get('trades_today', 0)}\n"
            f"Consec losses: {r.get('consecutive_losses', 0)}\n"
            f"Halted: {'Yes — ' + r.get('halt_reason', '') if r.get('is_halted') else 'No'}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        b = self._api.get_balance("USD") if self._api else {}
        text = (
            f"💳 *Balance*\n\n"
            f"Available: ${b.get('available', 0):.2f}\n"
            f"Locked: ${b.get('locked', 0):.2f}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_position(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        p = self._orders.get_status() if self._orders else {}
        if not p.get("in_position"):
            await update.message.reply_text("⬜ No open position")
            return
        text = (
            f"📈 *Position*\n\n"
            f"Side: {p.get('side', '').upper()}\n"
            f"Size: {p.get('size')} contracts\n"
            f"Entry: ${p.get('entry_price', 0):,.1f}\n"
            f"Bars held: {p.get('bars_held')}\n"
            f"SL order: {p.get('sl_order', 'N/A')}\n"
            f"TP order: {p.get('tp_order', 'N/A')}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_orderbook(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        ob = self._data.get_orderbook() if self._data else {}
        bids = ob.get("bids", [])[:5]
        asks = ob.get("asks", [])[:5]
        lines = ["📗 *Orderbook (Top 5)*\n", "```"]
        for a in reversed(asks):
            lines.append(f"  ASK {float(a[0]):>10,.1f}  {float(a[1]):>8,.0f}")
        lines.append("  ─────────────────────────")
        for b in bids:
            lines.append(f"  BID {float(b[0]):>10,.1f}  {float(b[1]):>8,.0f}")
        lines.append("```")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ─── TRADING CONTROLS ────────────────────────────────────────────────────

    async def _cmd_start_trading(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if self._strategy:
            self._strategy.start()
        await update.message.reply_text("⚡ Strategy *STARTED*", parse_mode="Markdown")

    async def _cmd_stop_trading(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if self._strategy:
            self._strategy.stop()
        await update.message.reply_text("⏹️ Strategy *STOPPED* (positions remain open)", parse_mode="Markdown")

    async def _cmd_halt(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if self._strategy:
            self._strategy.stop()
        if self._risk:
            self._risk.force_halt("TELEGRAM_HALT")
        if self._orders:
            result = self._orders.emergency_flatten()
            await update.message.reply_text(
                f"⛔ *EMERGENCY HALT*\n\n"
                f"Strategy stopped\n"
                f"Risk halted\n"
                f"Actions: {result.get('actions', [])}",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("⛔ *HALTED*", parse_mode="Markdown")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if self._risk:
            was = self._risk.resume()
            await update.message.reply_text(f"✅ Resumed (was: {was})")
        else:
            await update.message.reply_text("No risk manager")

    async def _cmd_flatten(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if self._orders:
            result = self._orders.emergency_flatten()
            await update.message.reply_text(f"🔨 Flattened: {result.get('actions', [])}")
        else:
            await update.message.reply_text("No order manager")

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if self._orders:
            price = self._data.get_last_price() if self._data else 0
            result = self._orders.close_position(reason="TELEGRAM_CLOSE", current_price=price)
            await update.message.reply_text(f"Close: {result}")
        else:
            await update.message.reply_text("No order manager")

    # ─── PARAMETER COMMANDS ───────────────────────────────────────────────────

    async def _cmd_set(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Universal /set <param> <value> — tries engine, risk, then config.
        Leverage is intercepted and routed through _apply_leverage so the
        exchange API is always called alongside internal state updates.
        """
        if not await self._guard(update):
            return
        args = ctx.args
        if not args or len(args) < 2:
            await update.message.reply_text("Usage: /set <param> <value>")
            return

        key, val = args[0], " ".join(args[1:])

        # Intercept leverage — must go through _apply_leverage (hits exchange API)
        if key.lower() == "leverage":
            try:
                await self._apply_leverage(update, int(val))
            except ValueError:
                await update.message.reply_text("❌ Leverage must be an integer", parse_mode="Markdown")
            return

        updated = False

        if self._engine and self._engine.update_param(key, val):
            updated = True
        elif self._risk and self._risk.update_param(key, val):
            updated = True
        elif self._config and hasattr(self._config, key.upper()):
            try:
                old = getattr(self._config, key.upper())
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
        args = ctx.args
        if not args:
            await update.message.reply_text("Usage: /get <param>")
            return

        key = args[0]
        val = None

        if self._engine:
            params = self._engine.get_params()
            if key in params:
                val = params[key]
        if val is None and self._risk:
            rp = self._risk.get_status().get("params", {})
            if key in rp:
                val = rp[key]
        if val is None and self._config and hasattr(self._config, key.upper()):
            val = getattr(self._config, key.upper())

        if val is not None:
            await update.message.reply_text(f"`{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"Unknown: {key}")

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

        # Trade/filter config params
        trade_keys = [
            "TRADE_TP_PCT", "TRADE_SL_PCT", "TRADE_MAX_HOLD_BARS",
            "TRADE_DH_DT_EXIT_SPIKE", "TRADE_USE_BRACKET_ORDERS",
            "TRADE_ORDER_TYPE", "FILTER_SPREAD_MAX_PCT",
            "FILTER_MIN_VOLUME_1M", "FILTER_VOLATILITY_MIN_PCT",
            "FILTER_VOLATILITY_MAX_PCT", "SIGNAL_DELTA_Q_THRESHOLD",
        ]
        if self._config:
            lines.append("\n*Trade/Filter:*")
            for k in trade_keys:
                if hasattr(self._config, k):
                    lines.append(f"  `{k}` = `{getattr(self._config, k)}`")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_set_engine(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: /set_engine <param> <value>")
            return
        key, val = ctx.args[0], ctx.args[1]
        if self._engine and self._engine.update_param(key, val):
            await update.message.reply_text(f"✅ Engine `{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Unknown engine param: {key}")

    async def _cmd_set_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: /set_risk <param> <value>")
            return
        key, val = ctx.args[0], ctx.args[1]

        # Intercept leverage — must go through _apply_leverage (hits exchange API)
        if key.lower() == "leverage":
            try:
                await self._apply_leverage(update, int(val))
            except ValueError:
                await update.message.reply_text("❌ Leverage must be an integer", parse_mode="Markdown")
            return

        if self._risk and self._risk.update_param(key, val):
            await update.message.reply_text(f"✅ Risk `{key}` = `{val}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Unknown risk param: {key}")

    async def _cmd_set_trade(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: /set_trade <param> <value>")
            return
        key, val = ctx.args[0], ctx.args[1]
        cfg_key = f"TRADE_{key.upper()}"
        if self._config and hasattr(self._config, cfg_key):
            try:
                old = getattr(self._config, cfg_key)
                new = type(old)(val) if not isinstance(old, bool) else val.lower() in ("true", "1")
                setattr(self._config, cfg_key, new)
                await update.message.reply_text(f"✅ `{cfg_key}` = `{new}`", parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        else:
            await update.message.reply_text(f"Unknown: {cfg_key}")

    async def _cmd_set_filter(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: /set_filter <param> <value>")
            return
        key, val = ctx.args[0], ctx.args[1]
        cfg_key = f"FILTER_{key.upper()}"
        if self._config and hasattr(self._config, cfg_key):
            try:
                old = getattr(self._config, cfg_key)
                new = type(old)(val) if not isinstance(old, bool) else val.lower() in ("true", "1")
                setattr(self._config, cfg_key, new)
                await update.message.reply_text(f"✅ `{cfg_key}` = `{new}`", parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        else:
            await update.message.reply_text(f"Unknown: {cfg_key}")

    # ─── SHORTCUT COMMANDS ────────────────────────────────────────────────────

    # ─── LEVERAGE HELPER ──────────────────────────────────────────────────────

    async def _apply_leverage(self, update: Update, val: int) -> bool:
        """
        Single authoritative path for changing leverage.
          1. Validate range
          2. Push to exchange API
          3. Update RiskManager._leverage
          4. Update config.RISK_LEVERAGE
        Returns True on success.
        """
        if val <= 0:
            await update.message.reply_text(
                "❌ Leverage must be a positive integer (e.g. `/leverage 50`)",
                parse_mode="Markdown",
            )
            return False
        if val > 200:
            await update.message.reply_text(
                f"❌ Leverage {val}x seems dangerously high. Max accepted: 200x"
            )
            return False

        lines = [f"⚙️ Setting leverage to *{val}x*…"]

        # ── 2. Exchange API ───────────────────────────────────────────────────
        if self._api:
            symbol = getattr(self._config, "DELTA_SYMBOL", "BTCUSD") if self._config else "BTCUSD"
            try:
                result = self._api.set_leverage(symbol=symbol, leverage=val)
                if result.get("success"):
                    lines.append(f"✅ Exchange: leverage confirmed at {val}x")
                else:
                    err = result.get("error", "unknown")
                    lines.append(f"❌ Exchange rejected: `{err}`")
                    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
                    return False
            except Exception as e:
                lines.append(f"❌ Exchange call failed: `{e}`")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
                return False
        else:
            lines.append("⚠️ No API — exchange leverage NOT updated (internal state only)")

        # ── 3. RiskManager ───────────────────────────────────────────────────
        if self._risk:
            self._risk.update_param("leverage", val)
            lines.append(f"✅ RiskManager: `_leverage` = {val}")
        else:
            lines.append("⚠️ No RiskManager — position sizing won't reflect new leverage until restart")

        # ── 4. Config module ─────────────────────────────────────────────────
        if self._config:
            self._config.RISK_LEVERAGE = val
            lines.append(f"✅ Config: `RISK_LEVERAGE` = {val}")
        else:
            lines.append("⚠️ No config reference — config not updated")

        lines.append(f"\n🔢 *Leverage is now {val}x*")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return True

    async def _cmd_leverage(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        /leverage        — show current leverage across exchange, risk, config
        /leverage <N>    — set leverage to N atomically (exchange + risk + config)
        """
        if not await self._guard(update):
            return

        # No args → show current value from all three sources
        if not ctx.args:
            risk_lev = self._risk.get_status().get("params", {}).get("leverage", "?") if self._risk else "?"
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
                "❌ Leverage must be an integer, e.g. `/leverage 50`",
                parse_mode="Markdown",
            )
            return

        await self._apply_leverage(update, val)

    async def _cmd_cooldown(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            await update.message.reply_text("Usage: /cooldown <seconds>")
            return
        if self._risk:
            self._risk.update_param("cooldown_seconds", ctx.args[0])
            await update.message.reply_text(f"✅ Cooldown: {ctx.args[0]}s")

    async def _cmd_maxloss(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            await update.message.reply_text("Usage: /maxloss <usd>")
            return
        if self._risk:
            self._risk.update_param("max_daily_loss_usd", ctx.args[0])
            await update.message.reply_text(f"✅ Max daily loss: ${ctx.args[0]}")

    async def _cmd_maxsize(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not await self._guard(update):
            return
        if not ctx.args:
            await update.message.reply_text("Usage: /maxsize <contracts>")
            return
        if self._risk:
            self._risk.update_param("max_position_contracts", ctx.args[0])
            await update.message.reply_text(f"✅ Max contracts: {ctx.args[0]}")
