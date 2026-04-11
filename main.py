"""
main.py — HPMS System Orchestrator (Hyperliquid)
===================================================
Initializes all components and runs the main loop:
  HLDataManager -> HPMSEngine -> Strategy -> OrderManager -> TelegramBot

Usage:
  python main.py              # live mode (mainnet)
  python main.py --testnet    # testnet mode
  python main.py --dry-run    # signal-only mode (no orders)
"""

from __future__ import annotations

import argparse
import asyncio
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

from core.api_client import HyperliquidClient
from hl_data_manager import HLDataManager

logger = logging.getLogger("hpms")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

class _ColoredFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _BOLD  = "\033[1m"
    _DIM   = "\033[2m"
    _LEVELS = {
        logging.DEBUG:    ("\033[36m",  "DBG"),
        logging.INFO:     ("\033[32m",  "INF"),
        logging.WARNING:  ("\033[33m",  "WRN"),
        logging.ERROR:    ("\033[31m",  "ERR"),
        logging.CRITICAL: ("\033[35m",  "CRT"),
    }
    _NAME_MAP = {
        "hpms":            "HPMS",
        "strategy":        "STRAT",
        "hpms_engine":     "ENGINE",
        "risk_manager":    "RISK",
        "order_manager":   "ORDER",
        "telegram_bot":    "TG",
        "logger_core":     "ELOG",
        "hl_data_manager": "DATA",
        "core.api_client": "HLAPI",
        "core.ws_client":  "HLWS",
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
        msg_indented = msg.replace("\n", "\n" + " " * 24)
        if record.exc_info:
            msg_indented += "\n" + self.formatException(record.exc_info)
        return header + msg_indented


def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.DEBUG))

    ch = logging.StreamHandler()
    ch.setFormatter(_ColoredFormatter())
    ch.setLevel(logging.DEBUG)
    root.addHandler(ch)

    logging.getLogger("hpms.elog").setLevel(logging.DEBUG)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "telegram", "telegram.ext",
                 "apscheduler", "websocket", "websockets"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC API WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class SyncHLAPI:
    """
    Thin sync wrapper around the async HyperliquidClient.
    Runs an asyncio event loop in a background thread.
    Exposes sync methods that strategy.py and telegram_bot.py can call.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="HL-API-Loop"
        )
        self._thread.start()
        time.sleep(0.2)
        self._client: HyperliquidClient = self._run(self._create())

    async def _create(self):
        return HyperliquidClient()

    def _run(self, coro, timeout=30):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def client(self) -> HyperliquidClient:
        return self._client

    # ── Convenience sync methods (used by telegram_bot, strategy) ─────────

    def get_clearinghouse_state(self, wallet: str) -> dict:
        return self._run(self._client.get_clearinghouse_state(wallet))

    def get_all_mids(self) -> dict:
        return self._run(self._client.get_all_mids())

    def get_open_orders(self, wallet: str) -> list:
        return self._run(self._client.get_open_orders(wallet))

    def set_leverage(self, coin: str, leverage: int, **kwargs) -> None:
        self._run(self._client.set_leverage(coin, leverage))

    def close(self):
        self._run(self._client.close(), timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)


# ═══════════════════════════════════════════════════════════════════════════════
# DRY-RUN API STUB
# ═══════════════════════════════════════════════════════════════════════════════

class DryRunAPI:
    """Simulates write calls without touching the exchange. Reads pass through."""

    def __init__(self, real_api: SyncHLAPI):
        self._real = real_api

    def __getattr__(self, name):
        _READ_METHODS = {
            "get_clearinghouse_state", "get_all_mids", "get_open_orders",
            "get_meta", "loop", "client",
        }
        if name in _READ_METHODS:
            return getattr(self._real, name)

        def fake(*args, **kwargs):
            elog.log("SYSTEM_START", component="DryRunAPI", call=name)
            return {"status": "ok", "response": {"type": "order", "data": {
                "statuses": [{"filled": {"avgPx": "0", "oid": 0, "totalFee": "0"}}]
            }}}
        return fake


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSRunner:
    def __init__(self, dry_run=False, testnet=False):
        self._dry_run  = dry_run
        self._testnet  = testnet or not config.HL_MAINNET
        self._shutdown = threading.Event()

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
        mode_str = "DRY-RUN" if self._dry_run else "LIVE"
        net_str  = "TESTNET" if self._testnet  else "MAINNET"
        symbol   = config.HL_SYMBOL

        elog.log("SYSTEM_START", component="HPMSRunner",
                 mode=mode_str, network=net_str, symbol=symbol)

        logger.info("=" * 60)
        logger.info("  HPMS - Hamiltonian Phase-Space Micro-Scalping")
        logger.info("  Exchange: Hyperliquid")
        logger.info("  Mode:    " + mode_str)
        logger.info("  Network: " + net_str)
        logger.info("  Symbol:  " + symbol)
        logger.info("=" * 60)

        # ── 0. Validate config ────────────────────────────────────────────────
        if not config.HL_PRIVATE_KEY:
            logger.error("HL_PRIVATE_KEY is empty — cannot sign orders")
        if not config.HL_WALLET_ADDRESS:
            logger.error("HL_WALLET_ADDRESS is empty — cannot query balance")
        if not config.TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN is empty")

        # ── 1. Exchange API ───────────────────────────────────────────────────
        real_api = SyncHLAPI()
        self._api = DryRunAPI(real_api) if self._dry_run else real_api
        elog.log("SYSTEM_START", component="HyperliquidAPI",
                 testnet=self._testnet, dry_run=self._dry_run)

        # ── 2. Data Manager ──────────────────────────────────────────────────
        self._data_mgr = HLDataManager()
        if not self._data_mgr.start():
            elog.error("SYSTEM_START", error="DataManager failed to start")
            return False
        elog.log("SYSTEM_START", component="HLDataManager", status="ready")

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
            leverage=config.RISK_LEVERAGE,
            max_daily_loss_usd=config.RISK_MAX_DAILY_LOSS_USD,
            max_daily_trades=config.RISK_MAX_DAILY_TRADES,
            max_consecutive_losses=config.RISK_MAX_CONSECUTIVE_LOSSES,
            cooldown_seconds=config.RISK_COOLDOWN_SECONDS,
            max_drawdown_pct=config.RISK_MAX_DRAWDOWN_PCT,
            equity_pct_per_trade=config.RISK_EQUITY_PCT_PER_TRADE,
            auto_resume_seconds=config.RISK_AUTO_RESUME_SECONDS,
            soft_loss_weight=config.RISK_SOFT_LOSS_WEIGHT,
        )

        # ── 5. Order Manager ─────────────────────────────────────────────────
        # Get the underlying async client for order execution
        api_client = self._api.client if not self._dry_run else real_api.client
        self._orders = OrderManager(
            api=api_client,
            symbol=config.HL_SYMBOL,
        )
        # Wire the async event loop so OrderManager can call async methods
        api_loop = real_api.loop
        self._orders.set_event_loop(api_loop)

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
            self._api.set_leverage(coin=config.HL_SYMBOL, leverage=config.RISK_LEVERAGE)
            elog.log("RISK_PARAM_UPDATE", key="leverage",
                     value=config.RISK_LEVERAGE, source="startup")
        except Exception as e:
            elog.error("SYSTEM_START", error=str(e), stage="set_leverage")

        # ── 9. Start Telegram ─────────────────────────────────────────────────
        self._telegram.start()

        # ── 10. Start strategy ────────────────────────────────────────────────
        self._strategy.start()

        elog.log("SYSTEM_START", component="HPMSRunner", status="fully_operational")

        self._telegram.send_message(
            "*HPMS System Online (Hyperliquid)*\n\n"
            "Mode: " + mode_str + "\n"
            "Network: " + net_str + "\n"
            "Symbol: " + config.HL_SYMBOL + "\n"
            "Leverage: " + str(config.RISK_LEVERAGE) + "x\n"
            "Integrator: " + str(config.HPMS_INTEGRATOR) + "\n"
            "Taker fee: " + str(config.FEE_TAKER_RATE * 100) + "%\n"
            "tau=" + str(config.HPMS_TAU) +
            " lookback=" + str(config.HPMS_LOOKBACK) +
            " horizon=" + str(config.HPMS_PREDICTION_HORIZON)
        )

        self._main_loop()
        return True

    # ─── WARM START ───────────────────────────────────────────────────────────

    def _warm_start(self):
        try:
            candles = self._data_mgr.get_candles("1m", limit=300)
            if not candles:
                logger.warning("[WARM_START] No candles — engine will wait for first live bar")
                return

            n = len(candles)
            start = max(1, n - 30)
            logger.info(
                f"[WARM_START] replaying bars {start}–{n} of {n} candles "
                f"(last_price={candles[-1].get('c', 0):.1f})"
            )
            for i in range(start, n + 1):
                self._strategy.on_bar_close(candles[:i])

            logger.info(f"[WARM_START] done — engine primed over {n - start + 1} bar replay")

        except Exception as e:
            logger.error(f"[WARM_START] exception: {e}", exc_info=True)

    # ─── MAIN LOOP ────────────────────────────────────────────────────────────

    def _main_loop(self):
        self._strategy.set_warming_up(True)
        self._warm_start()
        self._strategy.set_warming_up(False)

        health_check_interval = 60
        last_health_check     = time.time()
        poll_n                = 0

        last_bar_minute = int(time.time() / 60)
        logger.info(
            f"[LOOP] starting live poll — minute-boundary bar detection "
            f"(current minute={last_bar_minute})"
        )

        while not self._shutdown.is_set():
            try:
                poll_n += 1

                # Refresh latest candle data from HL REST
                self._data_mgr.refresh_latest_candles()

                candles = self._data_mgr.get_candles("1m", limit=300)

                if not candles:
                    if poll_n % 20 == 1:
                        logger.debug(f"[LOOP] poll #{poll_n} — no candles yet")
                else:
                    current_minute = int(time.time() / 60)
                    last_close = candles[-1].get("c", 0)

                    if current_minute > last_bar_minute:
                        time.sleep(0.20)
                        self._data_mgr.refresh_latest_candles()
                        candles = self._data_mgr.get_candles("1m", limit=300)
                        logger.info(
                            f"[LOOP] ▶ new 1m bar | minute={current_minute} "
                            f"close={last_close:.1f} candles={len(candles)} "
                            f"poll=#{poll_n}"
                        )
                        last_bar_minute = current_minute
                        self._strategy.on_bar_close(candles)
                    else:
                        if poll_n % 20 == 1:
                            secs_to_next = 60 - (time.time() % 60)
                            logger.debug(
                                f"[LOOP] poll #{poll_n} — waiting for bar close "
                                f"({secs_to_next:.0f}s remaining) close={last_close:.1f}"
                            )

                if time.time() - last_health_check > health_check_interval:
                    last_health_check = time.time()
                    self._health_check()

                self._shutdown.wait(0.5)

            except Exception as e:
                logger.error(f"[LOOP] unhandled exception (retry in 5s): {e}", exc_info=True)
                elog.error("SYSTEM_MAIN_LOOP", error=str(e), stage="main_loop")
                time.sleep(5)

    def _health_check(self):
        if not self._data_mgr.is_ready:
            elog.warn("SYSTEM_HEALTH", issue="DataManager_not_ready")
            self._telegram.send_message("DataManager not ready - restarting")
            self._data_mgr.restart_streams()

        if not self._data_mgr.is_price_fresh(max_stale_seconds=120):
            elog.warn("SYSTEM_HEALTH", issue="price_stale_gt_120s")
            self._telegram.send_message("Price data stale - check connection")

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
        if self._api and hasattr(self._api, "close"):
            try:
                self._api.close()
            except Exception:
                pass
        if self._telegram:
            self._telegram.send_message("*HPMS System Shutdown*")
            self._telegram.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPMS Trading System (Hyperliquid)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--testnet", action="store_true")
    args = parser.parse_args()

    setup_logging()

    runner = HPMSRunner(dry_run=args.dry_run, testnet=args.testnet)

    def handle_signal(signum, frame):
        runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
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
