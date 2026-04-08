"""
main.py — HPMS System Orchestrator
=====================================
Initializes all components and runs the main loop:
  DataManager → HPMSEngine → Strategy → OrderManager → TelegramBot

Usage:
  python main.py              # live mode
  python main.py --testnet    # testnet mode
  python main.py --dry-run    # signal-only mode (no orders)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from hpms_engine import HPMSEngine
from risk_manager import RiskManager
from order_manager import OrderManager
from strategy import HPMSStrategy
from telegram_bot import TelegramBot

# ── Delta Exchange plugins ────────────────────────────────────────────────────
# These are the user's uploaded plugins — placed in exchanges/delta/
from exchanges.delta.api import DeltaAPI
from exchanges.delta.data_manager import DeltaDataManager

logger = logging.getLogger("hpms")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # Console only — no file handlers.
    # AWS captures stdout/stderr automatically via CloudWatch / journald.
    # Writing log files burns EBS IOPS and fills disk on long runs.
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)


# ═══════════════════════════════════════════════════════════════════════════════
# DRY-RUN API STUB
# ═══════════════════════════════════════════════════════════════════════════════

class DryRunAPI:
    """Simulates API calls without touching the exchange."""

    def __init__(self, real_api: DeltaAPI):
        self._real = real_api

    def __getattr__(self, name):
        # Read-only calls pass through to real API
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

        # Write calls return fake success
        def fake(*args, **kwargs):
            logger.info(f"[DRY-RUN] {name}({args}, {kwargs})")
            return {
                "success": True,
                "result": {"order_id": f"dry_{int(time.time()*1000)}", "status": "simulated"},
                "error": None,
            }
        return fake


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSRunner:
    """Main application runner."""

    def __init__(self, dry_run: bool = False, testnet: bool = False):
        self._dry_run = dry_run
        self._testnet = testnet or config.DELTA_TESTNET
        self._shutdown = threading.Event()

        # Components (initialized in start())
        self._api:       DeltaAPI       = None
        self._data_mgr:  DeltaDataManager = None
        self._engine:    HPMSEngine     = None
        self._risk:      RiskManager    = None
        self._orders:    OrderManager   = None
        self._strategy:  HPMSStrategy   = None
        self._telegram:  TelegramBot    = None

    def start(self):
        logger.info("=" * 60)
        logger.info("  HPMS — Hamiltonian Phase-Space Micro-Scalping")
        logger.info(f"  Mode: {'DRY-RUN' if self._dry_run else 'LIVE'}")
        logger.info(f"  Network: {'TESTNET' if self._testnet else 'MAINNET'}")
        logger.info(f"  Symbol: {config.DELTA_SYMBOL}")
        logger.info("=" * 60)

        # ── 1. Exchange API ───────────────────────────────────────────────────
        if self._testnet:
            config.DELTA_TESTNET = True

        real_api = DeltaAPI(
            api_key=config.DELTA_API_KEY,
            secret_key=config.DELTA_SECRET_KEY,
            testnet=config.DELTA_TESTNET,
        )

        if self._dry_run:
            self._api = DryRunAPI(real_api)
        else:
            self._api = real_api

        logger.info("✅ API initialized")

        # ── 2. Data Manager ──────────────────────────────────────────────────
        self._data_mgr = DeltaDataManager()
        if not self._data_mgr.start():
            logger.error("❌ DataManager failed to start")
            return False

        logger.info("✅ DataManager ready")

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
        logger.info("✅ HPMS Engine initialized")

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
        )
        logger.info("✅ Risk Manager initialized")

        # ── 5. Order Manager ─────────────────────────────────────────────────
        self._orders = OrderManager(
            api=self._api,
            symbol=config.DELTA_SYMBOL,
            contract_value=getattr(config, "TRADE_CONTRACT_VALUE", 0.001),
        )
        logger.info("✅ Order Manager initialized")

        # ── 6. Telegram Bot ──────────────────────────────────────────────────
        self._telegram = TelegramBot(
            token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            admin_ids=config.TELEGRAM_ADMIN_IDS,
            strategy=None,  # set after strategy init
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
        logger.info("✅ Strategy initialized")

        # ── 8. Set leverage on exchange ───────────────────────────────────────
        try:
            self._api.set_leverage(
                symbol=config.DELTA_SYMBOL,
                leverage=config.RISK_LEVERAGE,
            )
            logger.info(f"✅ Leverage set to {config.RISK_LEVERAGE}x")
        except Exception as e:
            logger.warning(f"Leverage set failed: {e}")

        # ── 9. Start Telegram ─────────────────────────────────────────────────
        self._telegram.start()

        # ── 10. Start strategy ────────────────────────────────────────────────
        self._strategy.start()

        logger.info("🚀 HPMS system fully operational")
        self._telegram.send_message(
            "🚀 *HPMS System Online*\n\n"
            f"Mode: {'DRY-RUN' if self._dry_run else 'LIVE'}\n"
            f"Network: {'TESTNET' if self._testnet else 'MAINNET'}\n"
            f"Symbol: {config.DELTA_SYMBOL}\n"
            f"Leverage: {config.RISK_LEVERAGE}x\n"
            f"Integrator: {config.HPMS_INTEGRATOR}\n"
            f"τ={config.HPMS_TAU} lookback={config.HPMS_LOOKBACK} "
            f"horizon={config.HPMS_PREDICTION_HORIZON}"
        )

        # ── Main loop ────────────────────────────────────────────────────────
        self._main_loop()
        return True

    def _main_loop(self):
        """
        Polls DataManager for new 1m candles and feeds them to the strategy.

        Uses TIMESTAMP-based detection: track the last bar's timestamp.
        When a new bar with a newer timestamp appears, trigger the strategy.

        NOTE: count-based detection (old code) breaks after the deque hits
        maxlen — adding a new bar pops the oldest so len() stays constant.
        Timestamp comparison is the only reliable method.
        """
        last_bar_ts    = 0
        health_check_interval = 60  # seconds
        last_health_check = time.time()

        while not self._shutdown.is_set():
            try:
                candles = self._data_mgr.get_candles("1m", limit=300)

                if candles:
                    newest_ts = candles[-1].get("t", 0)
                    if newest_ts > last_bar_ts:
                        last_bar_ts = newest_ts
                        self._strategy.on_bar_close(candles)

                # Periodic health check
                if time.time() - last_health_check > health_check_interval:
                    last_health_check = time.time()
                    self._health_check()

                self._shutdown.wait(0.5)

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(5)

    def _health_check(self):
        """Periodic system health verification."""
        if not self._data_mgr.is_ready:
            logger.warning("⚠️ DataManager not ready — attempting restart")
            self._telegram.send_message("⚠️ DataManager not ready — restarting streams")
            self._data_mgr.restart_streams()

        if not self._data_mgr.is_price_fresh(max_stale_seconds=120):
            logger.warning("⚠️ Price data stale (>120s)")
            self._telegram.send_message("⚠️ Price data stale — check exchange connection")

    def shutdown(self):
        logger.info("Shutting down HPMS...")
        self._shutdown.set()

        if self._strategy:
            self._strategy.stop()
        if self._orders and self._orders.is_in_position:
            logger.warning("⚠️ Closing open position on shutdown")
            price = self._data_mgr.get_last_price() if self._data_mgr else 0
            self._orders.close_position(reason="SHUTDOWN", current_price=price)
        if self._data_mgr:
            self._data_mgr.stop()
        if self._telegram:
            self._telegram.send_message("🔌 *HPMS System Shutdown*")
            self._telegram.stop()

        logger.info("HPMS shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPMS Trading System")
    parser.add_argument("--dry-run", action="store_true", help="Signal-only mode, no orders")
    parser.add_argument("--testnet", action="store_true", help="Use testnet endpoint")
    args = parser.parse_args()

    setup_logging()

    runner = HPMSRunner(dry_run=args.dry_run, testnet=args.testnet)

    # Graceful shutdown on SIGINT/SIGTERM
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
        logger.critical(f"Fatal error: {e}", exc_info=True)
        runner.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
