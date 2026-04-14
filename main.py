"""
main.py — HPMS System Orchestrator
=====================================
Initializes all components and runs the main loop:
  DataManager -> HPMSEngine -> Strategy -> OrderManager -> TelegramBot

Usage:
  python main.py              # live mode
  python main.py --testnet    # testnet mode
  python main.py --dry-run    # signal-only mode (no orders)

NOTE: All conditional strings are pre-computed OUTSIDE f-strings to maintain
compatibility with Python 3.6 - 3.11 (f-string expressions cannot include
backslashes in those versions).
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from hpms_engine import HPMSEngine
from risk_manager import RiskManager
from order_manager import OrderManager
from strategy import HPMSStrategy
from telegram_bot import TelegramBot
from logger_core import elog
from state import STATE

from exchanges.delta.api import DeltaAPI
from exchanges.delta.data_manager import DeltaDataManager

logger = logging.getLogger("hpms")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

class _ColoredFormatter(logging.Formatter):
    """
    Rich, color-coded terminal formatter.

    Layout:
      HH:MM:SS.mmm  ▸  LVL  MODULE    message…

    Level colors:  DEBUG=cyan  INFO=green  WARNING=yellow  ERROR=red  CRITICAL=magenta
    Module column: always 8 chars wide, dimmed for readability.
    Continuation lines are indented to align under the message.
    """

    _RESET   = "\033[0m"
    _BOLD    = "\033[1m"
    _DIM     = "\033[2m"

    # (color, bold-label, dim-bullet)
    _LEVELS = {
        logging.DEBUG:    ("\033[36m",  "DBG", "·"),   # cyan
        logging.INFO:     ("\033[32m",  "INF", "▸"),   # green
        logging.WARNING:  ("\033[33m",  "WRN", "▲"),   # yellow
        logging.ERROR:    ("\033[31m",  "ERR", "✖"),   # red
        logging.CRITICAL: ("\033[35m",  "CRT", "★"),   # magenta
    }

    # Short display names for known loggers
    _NAME_MAP = {
        "hpms":                         "HPMS",
        "strategy":                     "STRAT",
        "hpms_engine":                  "ENGINE",
        "risk_manager":                 "RISK",
        "order_manager":                "ORDER",
        "telegram_bot":                 "TG",
        "logger_core":                  "ELOG",
        "hpms.elog":                    "ELOG",
        "exchanges.delta.api":          "DAPI",
        "exchanges.delta.data_manager": "DATA",
        "exchanges.delta.websocket":    "WS",
    }

    # Indent width = timestamp(12) + space(1) + bullet(1) + space(2) + lvl(3) + space(2) + name(8) + space(2)
    _INDENT = " " * 31

    def format(self, record: logging.LogRecord) -> str:
        color, lvl, bullet = self._LEVELS.get(record.levelno, ("", "???", "·"))
        short_name = self._NAME_MAP.get(record.name, record.name.split(".")[-1][:8])

        ts = self.formatTime(record, "%H:%M:%S")
        ms = f"{record.msecs:03.0f}"

        # Highlight WARNING / ERROR lines with a leading accent bar
        accent = ""
        if record.levelno >= logging.WARNING:
            accent = f"{color}▌{self._RESET} "

        header = (
            f"{self._DIM}{ts}.{ms}{self._RESET} "
            f"{color}{bullet}{self._RESET}  "
            f"{color}{self._BOLD}{lvl}{self._RESET}  "
            f"{self._DIM}{short_name:<8}{self._RESET}  "
            f"{accent}"
        )

        msg = record.getMessage()
        msg_indented = msg.replace("\n", "\n" + self._INDENT)

        if record.exc_info:
            msg_indented += "\n" + self._INDENT + self.formatException(record.exc_info)

        return header + msg_indented


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output — no ANSI codes, full timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        ts  = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]  # trim to ms
        return (
            f"{ts} UTC  {record.levelname:<8}  "
            f"{record.name:<35}  {record.getMessage()}"
        )


def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.DEBUG))

    # ── Console handler (colored) ─────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setFormatter(_ColoredFormatter())
    ch.setLevel(logging.DEBUG)
    root.addHandler(ch)

    # ── Rotating file handler (plain text, 10 MB × 5 files) ──────────────────
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "hpms.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(_PlainFormatter())
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    # ── hpms.elog: ALWAYS at DEBUG regardless of LOG_LEVEL ───────────────────
    # elog emits structured one-liners for every engine calculation.
    # Without an explicit DEBUG level on this logger the isEnabledFor(DEBUG)
    # guard silently drops all ENGINE_/RISK_/ORDER_ events at LOG_LEVEL=INFO.
    logging.getLogger("hpms.elog").setLevel(logging.DEBUG)

    # ── Silence noisy third-party loggers ────────────────────────────────────
    for name in (
        "httpx", "httpcore", "telegram", "telegram.ext",
        "apscheduler", "websocket", "websockets",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging initialised — console + rotating file (%s)", log_dir / "hpms.log")


# ═══════════════════════════════════════════════════════════════════════════════
# DRY-RUN API STUB
# ═══════════════════════════════════════════════════════════════════════════════

class DryRunAPI:
    """
    Wraps the real DeltaAPI, passing through all read-only calls and
    simulating write calls (order placement, cancellation, leverage set) so
    the full code path can be exercised without touching the exchange.
    """

    # Class-level constant — built once, not rebuilt on every __getattr__ call.
    # Bug fix: the old code reconstructed this set inside __getattr__ on every
    # attribute access (every API call), which is wasteful and error-prone.
    _READ_METHODS: frozenset[str] = frozenset({
        "get_ticker", "get_tickers", "get_orderbook", "get_candles",
        "get_balance", "get_positions", "get_position", "get_open_orders",
        "get_product_id", "prefetch_product_ids", "get_server_time",
        "get_wallet_balances", "get_products", "get_product",
        "get_recent_trades", "get_funding_rate", "get_mark_price",
        "self_test", "_symbol_to_product_id",
    })

    def __init__(self, real_api):
        self._real = real_api

    def __getattr__(self, name: str):
        if name in self._READ_METHODS:
            return getattr(self._real, name)

        def _simulated(*args, **kwargs):
            elog.log(
                "SYSTEM_START", component="DryRunAPI",
                call=name, note="simulated_no_exchange_hit",
            )
            return {
                "success": True,
                "result":  {
                    "order_id": f"dry_{int(time.time() * 1000)}",
                    "status":   "simulated",
                },
                "error": None,
            }
        return _simulated


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSRunner:
    """Main application runner — wires all components together."""

    def __init__(self, dry_run: bool = False, testnet: bool = False):
        self._dry_run  = dry_run
        self._testnet  = testnet or config.DELTA_TESTNET
        self._shutdown = threading.Event()

        # Update global state singleton
        STATE.dry_run = dry_run
        STATE.testnet = self._testnet
        STATE.mode    = "DRY-RUN" if dry_run else "LIVE"

        self._api      = None
        self._data_mgr = None
        self._engine   = None
        self._risk     = None
        self._orders   = None
        self._strategy = None
        self._telegram = None

    def start(self) -> bool:
        mode_str = "DRY-RUN" if self._dry_run else "LIVE"
        net_str  = "TESTNET" if self._testnet  else "MAINNET"

        elog.log("SYSTEM_START", component="HPMSRunner",
                 mode=mode_str, network=net_str, symbol=config.DELTA_SYMBOL)

        # ── Startup banner ────────────────────────────────────────────────────
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║   HPMS — Hamiltonian Phase-Space Micro-Scalping          ║")
        logger.info("╠══════════════════════════════════════════════════════════╣")
        logger.info("║   Mode    : " + mode_str.ljust(46) + "║")
        logger.info("║   Network : " + net_str.ljust(46)  + "║")
        logger.info("║   Symbol  : " + config.DELTA_SYMBOL.ljust(46) + "║")
        logger.info("║   Leverage: " + (str(config.RISK_LEVERAGE) + "x").ljust(46) + "║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

        # ── 0. Validate critical config ───────────────────────────────────────
        if not config.TELEGRAM_BOT_TOKEN:
            elog.error("SYSTEM_START", error="TELEGRAM_BOT_TOKEN missing",
                       note="push_notifications_disabled")
            logger.error("TELEGRAM NOT CONFIGURED — TELEGRAM_BOT_TOKEN is empty")
        if not config.TELEGRAM_CHAT_ID:
            elog.error("SYSTEM_START", error="TELEGRAM_CHAT_ID missing",
                       note="push_notifications_disabled")
            logger.error("TELEGRAM NOT CONFIGURED — TELEGRAM_CHAT_ID is empty")

        # ── 1. Exchange API ───────────────────────────────────────────────────
        if self._testnet:
            config.DELTA_TESTNET = True

        real_api = DeltaAPI(
            api_key=config.DELTA_API_KEY,
            secret_key=config.DELTA_SECRET_KEY,
            testnet=config.DELTA_TESTNET,
        )
        self._api = DryRunAPI(real_api) if self._dry_run else real_api
        elog.log("SYSTEM_START", component="DeltaAPI",
                 testnet=config.DELTA_TESTNET, dry_run=self._dry_run,
                 note="orders_simulated" if self._dry_run else "live_orders_enabled")

        # ── 2. Data Manager ──────────────────────────────────────────────────
        self._data_mgr = DeltaDataManager()
        if not self._data_mgr.start():
            elog.error("SYSTEM_START", error="DataManager_failed_to_start",
                       note="check_exchange_connectivity_and_credentials")
            return False
        elog.log("SYSTEM_START", component="DeltaDataManager",
                 status="ready", note="websocket_streams_active")

        # ── 3. HPMS Engine ────────────────────────────────────────────────────
        self._engine = HPMSEngine(
            tau=config.HPMS_TAU,
            lookback=config.HPMS_LOOKBACK,
            prediction_horizon=config.HPMS_PREDICTION_HORIZON,
            kde_bandwidth=config.HPMS_KDE_BANDWIDTH,
            kde_grid_points=config.HPMS_KDE_GRID_POINTS,
            integrator=config.HPMS_INTEGRATOR,
            integration_dt=config.HPMS_INTEGRATION_DT,
            mass=config.HPMS_MASS,
            normalization_window=config.HPMS_NORMALIZATION_WINDOW,
            delta_q_threshold=config.SIGNAL_DELTA_Q_THRESHOLD,
            dH_dt_max=config.SIGNAL_DH_DT_MAX,
            H_percentile=config.SIGNAL_H_PERCENTILE,
            min_momentum=config.SIGNAL_MIN_MOMENTUM,
            acceleration_check=config.SIGNAL_ACCELERATION_CHECK,
            H_ema_span=config.HPMS_H_EMA_SPAN,
            kde_rebuild_interval=config.HPMS_KDE_REBUILD_INTERVAL,
            trajectory_log_depth=config.HPMS_TRAJECTORY_LOG_DEPTH,
            fib_tp_cap_pct=config.FIB_TP_CAP_PCT,
            fib_sl_cap_pct=config.FIB_SL_CAP_PCT,
            fib_sl_atr_buffer=config.FIB_SL_ATR_BUFFER_MULT,
            fib_min_rr=config.FIB_MIN_RR,
            fib_swing_min_order=config.FIB_SWING_MIN_ORDER,
            fib_swing_max_order=config.FIB_SWING_MAX_ORDER,
            fib_swing_atr_noise=config.FIB_SWING_ATR_NOISE,
            fib_max_swing_pairs=config.FIB_MAX_SWING_PAIRS,
            fib_confluence_tol=config.FIB_CONFLUENCE_ATR_TOL,
        )

        # ── 4. Risk Manager ──────────────────────────────────────────────────
        self._risk = RiskManager(
            max_position_usd=config.RISK_MAX_POSITION_USD,
            max_position_contracts=config.RISK_MAX_POSITION_CONTRACTS,
            leverage=config.RISK_LEVERAGE,
            max_daily_loss_usd=config.RISK_MAX_DAILY_LOSS_USD,
            max_daily_trades=config.RISK_MAX_DAILY_TRADES,
            max_consecutive_losses=config.RISK_MAX_CONSECUTIVE_LOSSES,
            cooldown_seconds=config.RISK_COOLDOWN_SECONDS,
            max_drawdown_pct=config.RISK_MAX_DRAWDOWN_PCT,
            equity_pct_per_trade=config.RISK_EQUITY_PCT_PER_TRADE,
            auto_resume_seconds=getattr(config, "RISK_AUTO_RESUME_SECONDS", 600.0),
            soft_loss_weight=getattr(config, "RISK_SOFT_LOSS_WEIGHT", 0.5),
        )

        # ── 5. Order Manager ─────────────────────────────────────────────────
        self._orders = OrderManager(
            api=self._api,
            symbol=config.DELTA_SYMBOL,
            contract_value=getattr(config, "TRADE_CONTRACT_VALUE", 0.001),
        )
        self._orders.register_with_data_manager(self._data_mgr)

        # ── 6. Telegram Bot ──────────────────────────────────────────────────
        self._telegram = TelegramBot(
            token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            admin_ids=config.TELEGRAM_ADMIN_IDS,
            strategy=None,
            engine=self._engine,
            risk_mgr=self._risk,
            order_mgr=self._orders,
            data_mgr=self._data_mgr,
            api=self._api,
            config=config,
        )

        # ── 7. Strategy ──────────────────────────────────────────────────────
        self._strategy = HPMSStrategy(
            engine=self._engine,
            risk_mgr=self._risk,
            order_mgr=self._orders,
            data_mgr=self._data_mgr,
            api=self._api,
            config=config,
            notify_fn=self._telegram.send_message,
        )
        self._telegram._strategy = self._strategy

        # ── 8. Wire RiskManager → Telegram for halt push notifications ────────
        self._risk.set_notify_fn(self._telegram.send_message)

        # ── 9. Set leverage on exchange ───────────────────────────────────────
        try:
            self._api.set_leverage(symbol=config.DELTA_SYMBOL, leverage=config.RISK_LEVERAGE)
            elog.log("RISK_PARAM_UPDATE", key="leverage",
                     value=config.RISK_LEVERAGE, source="startup",
                     note="confirmed_on_exchange")
        except Exception as e:
            elog.error("SYSTEM_START", error=str(e), stage="set_leverage",
                       note="leverage_may_differ_from_config")

        # ── 10. Start Telegram ─────────────────────────────────────────────────
        self._telegram.start()

        # ── 11. Start strategy (sets STATE.trading_enabled = True) ────────────
        self._strategy.start()

        elog.log("SYSTEM_START", component="HPMSRunner",
                 status="fully_operational", mode=mode_str, network=net_str)

        # ── Startup notification ───────────────────────────────────────────────
        now_utc    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        mode_icon  = "🧪" if self._dry_run else "⚡"
        net_icon   = "🔬" if self._testnet  else "🌐"

        self._telegram.send_message(
            "🚀 *HPMS System Online*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + mode_icon + " Mode:          `" + mode_str                                    + "`\n"
            + net_icon  + " Network:       `" + net_str                                     + "`\n"
            "⏱ Started:       `" + now_utc                                                  + "`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📊 *Instrument*\n"
            "  Symbol:       `" + config.DELTA_SYMBOL                                       + "`\n"
            "  Leverage:     `" + str(config.RISK_LEVERAGE)                                 + "x`\n"
            "  Contract:     `" + str(getattr(config, "TRADE_CONTRACT_VALUE", 0.001))       + "` BTC\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔬 *Engine*\n"
            "  τ=`"         + str(config.HPMS_TAU)                 + "`  "
            "lookback=`"    + str(config.HPMS_LOOKBACK)            + "`  "
            "horizon=`"     + str(config.HPMS_PREDICTION_HORIZON)  + "`\n"
            "  KDE bw=`"    + str(config.HPMS_KDE_BANDWIDTH)       + "`  "
            "rebuild=`"     + str(config.HPMS_KDE_REBUILD_INTERVAL) + "` bars  "
            "integrator=`"  + str(config.HPMS_INTEGRATOR)          + "`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🛡 *Risk Limits*\n"
            "  Max loss/day:  `$" + str(config.RISK_MAX_DAILY_LOSS_USD) + "`\n"
            "  Max trades:    `"  + str(config.RISK_MAX_DAILY_TRADES)   + "` / day\n"
            "  Cooldown:      `"  + str(config.RISK_COOLDOWN_SECONDS)   + "s`\n"
            "  Equity/trade:  `"  + str(config.RISK_EQUITY_PCT_PER_TRADE) + "%`  "
            "Min R:R: `"          + str(getattr(config, "FIB_MIN_RR", 2.5))  + ":1`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Warming up — engine priming from REST candle history…_\n"
            "_Use /status for dashboard  /thinking for decision trace_"
        )

        self._main_loop()
        return True

    # ─── MAIN LOOP ────────────────────────────────────────────────────────────

    def _warm_start(self):
        """
        Feed REST warmup candles into the engine in a rolling window so the
        engine builds a full H-EMA history before going live.

        Auto-detects the timestamp key used by the DataManager (some
        implementations use "t", others "timestamp", "close_time", etc.)
        so the main loop's bar-detection works correctly.
        """
        try:
            candles = self._data_mgr.get_candles("1m", limit=300)
            if not candles:
                logger.warning(
                    "[WARM_START] DataManager returned no candles — "
                    "engine will start cold and wait for the first live bar"
                )
                return

            # ── Auto-detect timestamp key ─────────────────────────────────────
            ts_candidates = ["t", "timestamp", "close_time", "time",
                             "open_time", "ts", "closeTime"]
            last_c  = candles[-1]
            ts_key  = "t"
            ts_best = 0
            for k in ts_candidates:
                v = last_c.get(k, 0)
                try:
                    v = float(v)
                except Exception:
                    continue
                if v > ts_best:
                    ts_best = v
                    ts_key  = k
            logger.info(
                "[WARM_START] Timestamp key auto-detected: field='%s'  value=%s",
                ts_key, ts_best,
            )

            # ── Rolling warm-up: feed candles one window at a time ────────────
            n     = len(candles)
            start = max(1, n - 30)
            logger.info(
                "[WARM_START] Replaying bars %d–%d of %d REST candles "
                "(ts_key='%s'  last_close=%.1f) — building H-EMA history…",
                start, n, n, ts_key, last_c.get("c", 0),
            )
            for i in range(start, n + 1):
                self._strategy.on_bar_close(candles[:i])

            last_ts = float(last_c.get(ts_key, 0))
            logger.info(
                "[WARM_START] ✓ Engine primed — replayed %d bars.  "
                "last_ts=%.0f  Main loop will fire on next ts > %.0f",
                n - start + 1, last_ts, last_ts,
            )
            return

        except Exception as e:
            logger.error("[WARM_START] Warmup failed — engine will start cold: %s",
                         e, exc_info=True)

    def _main_loop(self):
        """
        Polls DataManager for new 1m candles and feeds them to the strategy.
        Uses wall-clock minute-boundary detection — immune to deque maxlen
        rollover and timestamp format differences across exchange APIs.
        """
        # Prime the engine with REST warmup data immediately
        self._strategy.set_warming_up(True)
        self._warm_start()
        self._strategy.set_warming_up(False)

        health_check_interval = 60
        last_health_check     = time.time()
        poll_n                = 0

        last_bar_minute = int(time.time() / 60)
        logger.info(
            "[LOOP] Live poll started — wall-clock minute-boundary bar detection.  "
            "current_minute=%d  (fires once per 60-second bar close)",
            last_bar_minute,
        )

        while not self._shutdown.is_set():
            try:
                poll_n += 1
                candles = self._data_mgr.get_candles("1m", limit=300)

                if not candles:
                    if poll_n % 20 == 1:
                        logger.debug(
                            "[LOOP] poll #%d — no candles yet "
                            "(WebSocket buffer still filling; clears on first live bar)",
                            poll_n,
                        )
                else:
                    current_minute = int(time.time() / 60)
                    last_close     = candles[-1].get("c", 0)

                    if current_minute > last_bar_minute:
                        # Brief pause to let the WebSocket is_closed=True event
                        # arrive and finalize candles[-1] before we snapshot.
                        # WS latency is <100ms; 200ms leaves >99% of the minute
                        # available for signal computation and order placement.
                        time.sleep(0.20)
                        candles = self._data_mgr.get_candles("1m", limit=300)
                        logger.info(
                            "[LOOP] ▶ Bar closed  minute=%d  close=%.1f  "
                            "candles=%d  poll=#%d — invoking strategy…",
                            current_minute, last_close, len(candles), poll_n,
                        )
                        last_bar_minute = current_minute
                        self._strategy.on_bar_close(candles)
                    else:
                        if poll_n % 20 == 1:
                            secs_to_next = 60 - (time.time() % 60)
                            logger.debug(
                                "[LOOP] poll #%d — waiting for bar close  "
                                "%.0fs remaining  last_close=%.1f",
                                poll_n, secs_to_next, last_close,
                            )

                if time.time() - last_health_check > health_check_interval:
                    last_health_check = time.time()
                    self._health_check()

                self._shutdown.wait(0.5)

            except Exception as e:
                logger.error(
                    "[LOOP] Unhandled exception in main loop (retrying in 5s): %s",
                    e, exc_info=True,
                )
                elog.error("SYSTEM_MAIN_LOOP", error=str(e),
                           stage="main_loop", note="retrying_after_5s")
                time.sleep(5)

    def _health_check(self):
        if not self._data_mgr.is_ready:
            elog.warn("SYSTEM_HEALTH", issue="DataManager_not_ready",
                      action="restarting_websocket_streams")
            self._telegram.send_message(
                "⚠️ *Health Alert: Data Feed Down*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "DataManager is reporting *not ready* — WebSocket streams appear "
                "to have dropped.\n\n"
                "🔄 *Action:* Automatically restarting streams now.\n"
                "_No new signals until the feed recovers.  Use /market to verify._"
            )
            self._data_mgr.restart_streams()

        if not self._data_mgr.is_price_fresh(max_stale_seconds=120):
            elog.warn("SYSTEM_HEALTH", issue="price_stale_gt_120s",
                      action="alerting_operator")
            self._telegram.send_message(
                "⚠️ *Health Alert: Stale Price Data*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Last price tick is *>120 seconds old* — the exchange feed "
                "may be degraded or the WebSocket has silently disconnected.\n\n"
                "🔎 Signal generation is safe-halted until fresh data arrives.\n"
                "_Check /market for data status, or /price for the last known price._"
            )

    def shutdown(self):
        elog.log("SYSTEM_SHUTDOWN", component="HPMSRunner",
                 note="graceful_shutdown_initiated")
        self._shutdown.set()

        if self._strategy:
            self._strategy.stop()

        if self._orders and self._orders.is_in_position:
            elog.warn("SYSTEM_SHUTDOWN", note="closing_open_position_before_exit",
                      action="market_close_at_current_price")
            price = self._data_mgr.get_last_price() if self._data_mgr else 0
            self._orders.close_position(reason="SHUTDOWN", current_price=price)

        if self._data_mgr:
            self._data_mgr.stop()

        if self._telegram:
            net_str  = "TESTNET" if self._testnet  else "MAINNET"
            now_utc  = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

            # Pull final session stats for shutdown message
            shutdown_pnl  = ""
            if self._risk:
                rs = self._risk.get_status()
                shutdown_pnl = (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📊 *Final Session Stats*\n"
                    "  Trades: `" + str(rs.get("trades_today", 0)) + "`  "
                    "Net P&L: `$" + f"{rs.get('daily_pnl', 0):+.4f}" + "`\n"
                    "  Fees:   `$-" + f"{rs.get('daily_fees', 0):.4f}" + "`"
                )

            self._telegram.send_message(
                "🛑 *HPMS System Shutdown*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Mode:    `" + STATE.mode + "`\n"
                "Network: `" + net_str    + "`\n"
                "Time:    `" + now_utc    + "`"
                + shutdown_pnl + "\n\n"
                "_All positions handled.  System offline._\n"
                "_Restart: `python main.py`_"
            )
            self._telegram.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPMS Trading System")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate orders — no exchange interaction")
    parser.add_argument("--testnet", action="store_true",
                        help="Connect to Delta testnet instead of mainnet")
    args = parser.parse_args()

    setup_logging()

    runner = HPMSRunner(dry_run=args.dry_run, testnet=args.testnet)

    def _handle_signal(signum, frame):
        runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        runner.start()
    except KeyboardInterrupt:
        runner.shutdown()
    except Exception as e:
        elog.error("SYSTEM_SHUTDOWN", error=str(e), stage="fatal",
                   note="unhandled_exception_forced_exit")
        logger.critical("Fatal unhandled exception — forcing shutdown: %s", e,
                        exc_info=True)
        runner.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
