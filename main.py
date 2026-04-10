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
import signal
import sys
import threading
import time
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
    Pretty, color-coded terminal formatter.
    Strips the package prefix from logger names so the column stays narrow.
    """
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _LEVELS = {
        logging.DEBUG:    ("\033[36m",  "DBG"),   # cyan
        logging.INFO:     ("\033[32m",  "INF"),   # green
        logging.WARNING:  ("\033[33m",  "WRN"),   # yellow
        logging.ERROR:    ("\033[31m",  "ERR"),   # red
        logging.CRITICAL: ("\033[35m",  "CRT"),   # magenta
    }
    # Short display names for known loggers
    _NAME_MAP = {
        "hpms":                       "HPMS",
        "strategy":                   "STRAT",
        "hpms_engine":                "ENGINE",
        "risk_manager":               "RISK",
        "order_manager":              "ORDER",
        "telegram_bot":               "TG",
        "logger_core":                "ELOG",
        "exchanges.delta.api":        "DAPI",
        "exchanges.delta.data_manager": "DATA",
        "exchanges.delta.websocket":  "WS",
    }

    def format(self, record: logging.LogRecord) -> str:
        color, lvl = self._LEVELS.get(record.levelno, ("", "???"))
        short_name = self._NAME_MAP.get(record.name, record.name.split(".")[-1][:8])
        ts = self.formatTime(record, "%H:%M:%S")
        ms = f"{record.msecs:03.0f}"
        header = (
            f"{self._DIM}{ts}.{ms}{self._RESET} "
            f"{color}{self._BOLD}{lvl}{self._RESET} "
            f"{self._DIM}{short_name:<8}{self._RESET} "
        )
        msg = record.getMessage()
        # Indent continuation lines
        msg_indented = msg.replace("\n", "\n" + " " * 24)
        if record.exc_info:
            msg_indented += "\n" + self.formatException(record.exc_info)
        return header + msg_indented


def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.DEBUG))

    ch = logging.StreamHandler()
    ch.setFormatter(_ColoredFormatter())
    ch.setLevel(logging.DEBUG)          # handler must not re-filter below root
    root.addHandler(ch)

    # ── hpms.elog: ALWAYS at DEBUG regardless of LOG_LEVEL ───────────────────
    # elog emits structured JSON for every engine calculation (ENGINE_PHASE_STATE,
    # ENGINE_CRITERIA, ENGINE_TRAJECTORY, ENGINE_KDE_REBUILD, ENGINE_SIGNAL,
    # ENGINE_SKIP).  Without an explicit DEBUG level on this logger the
    # isEnabledFor(DEBUG) guard inside _ELog.log() evaluates False at INFO,
    # silently dropping all per-bar diagnostics.
    logging.getLogger("hpms.elog").setLevel(logging.DEBUG)

    # ── Silence noisy third-party loggers ────────────────────────────────────
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# DRY-RUN API STUB
# ═══════════════════════════════════════════════════════════════════════════════

class DryRunAPI:
    """Simulates API calls without touching the exchange."""

    def __init__(self, real_api):
        self._real = real_api

    def __getattr__(self, name):
        _READ_METHODS = {
            "get_ticker", "get_tickers", "get_orderbook", "get_candles",
            "get_balance", "get_positions", "get_position", "get_open_orders",
            "get_product_id", "prefetch_product_ids", "get_server_time",
            "get_wallet_balances", "get_products", "get_product",
            "get_recent_trades", "get_funding_rate", "get_mark_price",
            "self_test", "_symbol_to_product_id",
        }
        if name in _READ_METHODS:
            return getattr(self._real, name)

        def fake(*args, **kwargs):
            elog.log("SYSTEM_START", component="DryRunAPI", call=name)
            return {
                "success": True,
                "result":  {"order_id": "dry_" + str(int(time.time() * 1000)), "status": "simulated"},
                "error":   None,
            }
        return fake


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSRunner:
    """Main application runner."""

    def __init__(self, dry_run=False, testnet=False):
        self._dry_run  = dry_run
        self._testnet  = testnet or config.DELTA_TESTNET
        self._shutdown = threading.Event()

        # Update global state singleton
        STATE.dry_run = dry_run
        STATE.testnet = self._testnet
        STATE.mode = "DRY-RUN" if dry_run else "LIVE"

        self._api      = None
        self._data_mgr = None
        self._engine   = None
        self._risk     = None
        self._orders   = None
        self._strategy = None
        self._telegram = None

    def start(self):
        # Pre-compute all display strings OUTSIDE f-strings.
        # This eliminates the "f-string expression part cannot include a backslash"
        # SyntaxError that occurs on Python 3.6 - 3.11 when ternary operators
        # or string literals containing backslashes appear inside {}.
        mode_str = "DRY-RUN" if self._dry_run else "LIVE"
        net_str  = "TESTNET" if self._testnet  else "MAINNET"

        elog.log("SYSTEM_START", component="HPMSRunner",
                 mode=mode_str, network=net_str, symbol=config.DELTA_SYMBOL)

        logger.info("=" * 60)
        logger.info("  HPMS - Hamiltonian Phase-Space Micro-Scalping")
        logger.info("  Mode:    " + mode_str)
        logger.info("  Network: " + net_str)
        logger.info("  Symbol:  " + config.DELTA_SYMBOL)
        logger.info("=" * 60)

        # ── 0. Validate critical config ───────────────────────────────────────
        if not config.TELEGRAM_BOT_TOKEN:
            elog.error("SYSTEM_START", error="TELEGRAM_BOT_TOKEN missing", component="config")
            logger.error("TELEGRAM NOT CONFIGURED: TELEGRAM_BOT_TOKEN is empty")
        if not config.TELEGRAM_CHAT_ID:
            elog.error("SYSTEM_START", error="TELEGRAM_CHAT_ID missing", component="config")
            logger.error("TELEGRAM NOT CONFIGURED: TELEGRAM_CHAT_ID is empty")

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
                 testnet=config.DELTA_TESTNET, dry_run=self._dry_run)

        # ── 2. Data Manager ──────────────────────────────────────────────────
        self._data_mgr = DeltaDataManager()
        if not self._data_mgr.start():
            elog.error("SYSTEM_START", error="DataManager failed to start")
            return False
        elog.log("SYSTEM_START", component="DeltaDataManager", status="ready")

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
            tp_pct=config.TRADE_TP_PCT,
            sl_pct=config.TRADE_SL_PCT,
            H_ema_span=config.HPMS_H_EMA_SPAN,
            kde_rebuild_interval=config.HPMS_KDE_REBUILD_INTERVAL,
            trajectory_log_depth=config.HPMS_TRAJECTORY_LOG_DEPTH,
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
            auto_resume_seconds=getattr(config, "RISK_AUTO_RESUME_SECONDS", 300.0),
            soft_loss_weight=getattr(config, "RISK_SOFT_LOSS_WEIGHT", 0.5),
        )

        # ── 5. Order Manager ─────────────────────────────────────────────────
        self._orders = OrderManager(
            api=self._api,
            symbol=config.DELTA_SYMBOL,
            contract_value=getattr(config, "TRADE_CONTRACT_VALUE", 0.001),
            taker_fee_pct=getattr(config, "TRADE_TAKER_FEE_PCT", 0.05),
            maker_fee_pct=getattr(config, "TRADE_MAKER_FEE_PCT", 0.02),
        )
        # Wire real-time fill/order events from WebSocket into OrderManager
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

        # ── 8. Set leverage on exchange ───────────────────────────────────────
        try:
            self._api.set_leverage(symbol=config.DELTA_SYMBOL, leverage=config.RISK_LEVERAGE)
            elog.log("RISK_PARAM_UPDATE", key="leverage",
                     value=config.RISK_LEVERAGE, source="startup")
        except Exception as e:
            elog.error("SYSTEM_START", error=str(e), stage="set_leverage")

        # ── 9. Start Telegram ─────────────────────────────────────────────────
        self._telegram.start()

        # ── 10. Start strategy (sets STATE.trading_enabled = True) ────────────
        self._strategy.start()

        elog.log("SYSTEM_START", component="HPMSRunner", status="fully_operational")

        # Startup notification — built with concatenation, no f-string ternaries
        self._telegram.send_message(
            "*HPMS System Online*\n\n"
            "Mode: " + mode_str + "\n"
            "Network: " + net_str + "\n"
            "Symbol: " + config.DELTA_SYMBOL + "\n"
            "Leverage: " + str(config.RISK_LEVERAGE) + "x\n"
            "Integrator: " + str(config.HPMS_INTEGRATOR) + "\n"
            "tau=" + str(config.HPMS_TAU) +
            " lookback=" + str(config.HPMS_LOOKBACK) +
            " horizon=" + str(config.HPMS_PREDICTION_HORIZON)
        )

        self._main_loop()
        return True

    # ─── MAIN LOOP ────────────────────────────────────────────────────────────

    def _warm_start(self):
        """
        Feed REST warmup candles into the engine in a rolling window so the
        engine builds a full H-EMA history (needs ≥2 bars) before going live.

        Also discovers the real timestamp key used by the DataManager
        (some implementations use "t", others use "timestamp", "close_time",
        etc.) so the main loop's bar-detection works correctly.

        Returns (last_ts_value, ts_key) so the main loop can detect new bars.
        """
        try:
            candles = self._data_mgr.get_candles("1m", limit=300)
            if not candles:
                logger.warning("[WARM_START] DataManager returned no candles — "
                               "engine will wait for first live bar")
                return

            # ── Auto-detect timestamp key ─────────────────────────────────────
            # Try common field names; pick the one with the largest non-zero value.
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
            logger.info("[WARM_START] timestamp key detected: '%s' → %s", ts_key, ts_best)

            # ── Rolling warm-up: feed candles one window at a time ────────────
            # The engine needs on_bar_close called with the FULL slice ending at
            # bar N, not just the last bar.  We call it once per bar in the
            # second half of the warmup so the H-EMA history fills up properly.
            n       = len(candles)
            # Process the last 30 bars as rolling windows so dH/dt history builds
            start   = max(1, n - 30)
            logger.info(
                "[WARM_START] replaying bars %d–%d of %d warmup candles "
                "(ts_key='%s' last_price=%.1f)",
                start, n, n, ts_key, last_c.get("c", 0),
            )
            for i in range(start, n + 1):
                self._strategy.on_bar_close(candles[:i])

            last_ts = float(last_c.get(ts_key, 0))
            logger.info(
                "[WARM_START] done — engine primed over %d bar replay. "
                "last_ts=%.0f  main loop now watching for ts > %.0f",
                n - start + 1, last_ts, last_ts,
            )
            return

        except Exception as e:
            logger.error("[WARM_START] exception: %s", e, exc_info=True)
            return

    def _main_loop(self):
        """
        Polls DataManager for new 1m candles and feeds them to the strategy.
        Uses TIMESTAMP-based bar detection — immune to deque maxlen rollover.

        IMPORTANT: get_candles() may return an empty list during the first
        ~60 s if the DataManager's live buffer is separate from the REST
        warmup buffer and no WebSocket bar has closed yet.  The warm_start()
        call above pre-populates the engine from REST data so the system is
        active from second 0.
        """
        # Prime the engine with REST warmup data immediately
        self._strategy.set_warming_up(True)
        self._warm_start()
        self._strategy.set_warming_up(False)

        health_check_interval = 60
        last_health_check     = time.time()
        poll_n                = 0

        # Candle timestamps from this DataManager are always 0 (timestamp field
        # absent or zero in the normalised candle dict).  Use minute-boundary
        # wall-clock detection instead: fire on_bar_close once per calendar minute.
        # This is drift-free and works regardless of the exchange API format.
        last_bar_minute = int(time.time() / 60)
        logger.info(
            "[LOOP] starting live poll — minute-boundary bar detection "
            "(current minute=%d)", last_bar_minute,
        )

        while not self._shutdown.is_set():
            try:
                poll_n += 1
                candles = self._data_mgr.get_candles("1m", limit=300)

                if not candles:
                    if poll_n % 20 == 1:
                        logger.debug(
                            "[LOOP] poll #%d — DataManager returned no candles "
                            "(waiting for WS data)", poll_n
                        )
                else:
                    current_minute = int(time.time() / 60)
                    last_close     = candles[-1].get("c", 0)

                    if current_minute > last_bar_minute:
                        logger.info(
                            "[LOOP] ▶ new 1m bar | minute=%d close=%.1f "
                            "candles=%d poll=#%d",
                            current_minute, last_close, len(candles), poll_n,
                        )
                        last_bar_minute = current_minute
                        self._strategy.on_bar_close(candles)
                    else:
                        if poll_n % 20 == 1:
                            secs_to_next = 60 - (time.time() % 60)
                            logger.debug(
                                "[LOOP] poll #%d — waiting for bar close "
                                "(%.0fs remaining) close=%.1f",
                                poll_n, secs_to_next, last_close,
                            )

                if time.time() - last_health_check > health_check_interval:
                    last_health_check = time.time()
                    self._health_check()

                self._shutdown.wait(0.5)

            except Exception as e:
                logger.error("[LOOP] unhandled exception (will retry in 5s): %s",
                             e, exc_info=True)
                elog.error("SYSTEM_MAIN_LOOP", error=str(e), stage="main_loop")
                time.sleep(5)

    def _health_check(self):
        if not self._data_mgr.is_ready:
            elog.warn("SYSTEM_HEALTH", issue="DataManager_not_ready")
            self._telegram.send_message("DataManager not ready - restarting streams")
            self._data_mgr.restart_streams()

        if not self._data_mgr.is_price_fresh(max_stale_seconds=120):
            elog.warn("SYSTEM_HEALTH", issue="price_stale_gt_120s")
            self._telegram.send_message("Price data stale - check exchange connection")

    def shutdown(self):
        elog.log("SYSTEM_SHUTDOWN", component="HPMSRunner")
        self._shutdown.set()

        if self._strategy:
            self._strategy.stop()
        if self._orders and self._orders.is_in_position:
            elog.warn("SYSTEM_SHUTDOWN", note="closing_open_position")
            price = self._data_mgr.get_last_price() if self._data_mgr else 0
            self._orders.close_position(reason="SHUTDOWN", current_price=price)
        if self._data_mgr:
            self._data_mgr.stop()
        if self._telegram:
            self._telegram.send_message("*HPMS System Shutdown*")
            self._telegram.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPMS Trading System")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--testnet", action="store_true")
    args = parser.parse_args()

    setup_logging()

    runner = HPMSRunner(dry_run=args.dry_run, testnet=args.testnet)

    def handle_signal(signum, frame):
        runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        runner.start()
    except KeyboardInterrupt:
        runner.shutdown()
    except Exception as e:
        elog.error("SYSTEM_SHUTDOWN", error=str(e), stage="fatal")
        runner.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
