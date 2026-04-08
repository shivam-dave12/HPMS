"""
main.py — HPMS System Orchestrator
=====================================
Usage:
  python main.py              # live mode
  python main.py --testnet    # testnet mode
  python main.py --dry-run    # signal-only, no orders
  python main.py --no-auto-start  # start paused; enable via /start_trading
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
from exchanges.delta.api import DeltaAPI
from exchanges.delta.data_manager import DeltaDataManager

logger = logging.getLogger("hpms")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # ── Silence noisy third-party loggers ─────────────────────────────────────
    # httpx fires INFO on every Telegram getUpdates poll (~every 10s).
    # None of that is useful — suppress to WARNING so real errors still show.
    for noisy in ("httpx", "telegram", "telegram.ext",
                  "telegram.ext.Application", "telegram.ext.Updater",
                  "websocket"):          # low-level websocket-client library
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Exchange layers: WS subscription spam → WARNING; DM warmup → INFO; API → WARNING
    logging.getLogger("exchanges.delta.websocket").setLevel(logging.WARNING)
    logging.getLogger("exchanges.delta.data_manager").setLevel(logging.INFO)
    logging.getLogger("exchanges.delta.api").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# DRY-RUN API STUB
# ═══════════════════════════════════════════════════════════════════════════════

class DryRunAPI:
    """Passes read-only calls to real API; returns fake success for all writes."""

    _READ = {
        "get_ticker", "get_tickers", "get_orderbook", "get_candles",
        "get_balance", "get_positions", "get_position", "get_open_orders",
        "get_product_id", "prefetch_product_ids", "get_server_time",
        "get_wallet_balances", "get_products", "get_product",
        "get_recent_trades", "get_funding_rate", "get_mark_price",
        "self_test", "_symbol_to_product_id", "get_leverage",
    }

    def __init__(self, real_api):
        self._real = real_api

    def __getattr__(self, name):
        if name in self._READ:
            return getattr(self._real, name)
        def fake(*args, **kwargs):
            logger.debug(f"[DRY-RUN] {name}({args}, {kwargs})")
            return {"success": True,
                    "result": {"order_id": f"dry_{int(time.time()*1000)}", "status": "simulated"},
                    "error": None}
        return fake


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSRunner:
    HEARTBEAT_INTERVAL          = 300   # console heartbeat every 5 min
    TELEGRAM_HEARTBEAT_INTERVAL = 3600  # Telegram hourly summary

    def __init__(self, dry_run=False, testnet=False, auto_start=True):
        self._dry_run    = dry_run
        self._testnet    = testnet or config.DELTA_TESTNET
        self._auto_start = auto_start
        self._shutdown   = threading.Event()

        self._api      = None
        self._data_mgr = None
        self._engine   = None
        self._risk     = None
        self._orders   = None
        self._strategy = None
        self._telegram = None

    def start(self):
        mode = "DRY-RUN" if self._dry_run else "LIVE"
        net  = "TESTNET" if self._testnet else "MAINNET"

        logger.info("=" * 60)
        logger.info(f"  HPMS  |  {mode}  |  {net}  |  {config.DELTA_SYMBOL}")
        logger.info("=" * 60)

        # 1 — API
        if self._testnet:
            config.DELTA_TESTNET = True
        real_api = DeltaAPI(api_key=config.DELTA_API_KEY,
                            secret_key=config.DELTA_SECRET_KEY,
                            testnet=config.DELTA_TESTNET)
        self._api = DryRunAPI(real_api) if self._dry_run else real_api
        logger.info("API ready")

        # 2 — Data Manager
        self._data_mgr = DeltaDataManager()
        if not self._data_mgr.start():
            logger.error("DataManager failed to start — aborting")
            return False
        logger.info("DataManager ready")

        # 3 — HPMS Engine
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
        logger.info(
            f"HPMS Engine ready — τ={config.HPMS_TAU} lookback={config.HPMS_LOOKBACK} "
            f"horizon={config.HPMS_PREDICTION_HORIZON} integrator={config.HPMS_INTEGRATOR} "
            f"TP={config.TRADE_TP_PCT:.2%} SL={config.TRADE_SL_PCT:.2%}"
        )

        # 4 — Risk Manager
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
        logger.info(
            f"Risk Manager ready — max_loss=${config.RISK_MAX_DAILY_LOSS_USD} "
            f"max_trades={config.RISK_MAX_DAILY_TRADES} cooldown={config.RISK_COOLDOWN_SECONDS}s "
            f"max_consec_loss={config.RISK_MAX_CONSECUTIVE_LOSSES}"
        )

        # 5 — Order Manager
        self._orders = OrderManager(
            api=self._api,
            symbol=config.DELTA_SYMBOL,
            contract_value=getattr(config, "TRADE_CONTRACT_VALUE", 0.001),
        )
        logger.info("Order Manager ready")

        # 6 — Telegram Bot (start early so it can receive commands during setup)
        self._telegram = TelegramBot(
            token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            admin_ids=config.TELEGRAM_ADMIN_IDS,
            strategy=None,   # filled after strategy init
            engine=self._engine,
            risk_mgr=self._risk,
            order_mgr=self._orders,
            data_mgr=self._data_mgr,
            api=self._api,
            config=config,
        )
        self._telegram.start()

        # 7 — Strategy
        self._strategy = HPMSStrategy(
            engine=self._engine,
            risk_mgr=self._risk,
            order_mgr=self._orders,
            data_mgr=self._data_mgr,
            api=self._api,
            config=config,
            notify_fn=self._telegram.send_message,
        )
        # Back-fill the strategy reference into the bot
        self._telegram._strategy = self._strategy

        # Wire Telegram notifier into RiskManager so circuit-breaker halts
        # push to Telegram instantly without going through the strategy.
        self._risk.set_notify_fn(self._telegram.send_message)

        logger.info("Strategy ready")

        # 8 — Leverage
        try:
            self._api.set_leverage(symbol=config.DELTA_SYMBOL, leverage=config.RISK_LEVERAGE)
            logger.info(f"Leverage confirmed: {config.RISK_LEVERAGE}x")
        except Exception as e:
            logger.warning(f"Leverage set failed (non-fatal): {e}")

        # 9 — Auto-start
        if self._auto_start:
            self._strategy.start()
        else:
            logger.info("Strategy PAUSED — send /start_trading in Telegram to begin")

        # 10 — Startup notification (small delay ensures bot loop is ready)
        time.sleep(2)
        self._telegram.send_message(
            f"🚀 *HPMS Online*\n"
            f"Mode: `{mode}` | Net: `{net}`\n"
            f"Symbol: `{config.DELTA_SYMBOL}` | Lev: `{config.RISK_LEVERAGE}x`\n"
            f"Strategy: `{'▶ RUNNING' if self._auto_start else '⏸ PAUSED — /start\\_trading to begin'}`\n"
            f"TP `{config.TRADE_TP_PCT:.2%}` | SL `{config.TRADE_SL_PCT:.2%}` | "
            f"MaxLoss `${config.RISK_MAX_DAILY_LOSS_USD}`\n"
            f"τ=`{config.HPMS_TAU}` lb=`{config.HPMS_LOOKBACK}` "
            f"hz=`{config.HPMS_PREDICTION_HORIZON}` ({config.HPMS_INTEGRATOR})"
        )

        logger.info("🚀 HPMS fully operational")
        self._main_loop()
        return True

    # ─── MAIN LOOP ────────────────────────────────────────────────────────────

    def _main_loop(self):
        last_bar_ts             = 0
        last_heartbeat          = time.time()
        last_tg_heartbeat       = time.time()
        last_health_check       = time.time()
        health_interval         = 60

        while not self._shutdown.is_set():
            try:
                candles = self._data_mgr.get_candles("1m", limit=300)
                if candles:
                    newest_ts = candles[-1].get("t", 0)
                    if newest_ts > last_bar_ts:
                        last_bar_ts = newest_ts
                        self._strategy.on_bar_close(candles)

                now = time.time()

                if now - last_heartbeat > self.HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    self._console_heartbeat()

                if now - last_tg_heartbeat > self.TELEGRAM_HEARTBEAT_INTERVAL:
                    last_tg_heartbeat = now
                    self._telegram_heartbeat()

                if now - last_health_check > health_interval:
                    last_health_check = now
                    self._health_check()

                self._shutdown.wait(0.5)

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                self._telegram.send_message(f"⚠️ Main loop error: `{e}`")
                time.sleep(5)

    def _console_heartbeat(self):
        price   = self._data_mgr.get_last_price() if self._data_mgr else 0
        risk    = self._risk.get_status()
        pos     = self._orders.get_status()
        in_pos  = f"IN {pos['side'].upper()} {pos['size']}c" if pos.get("in_position") else "flat"
        logger.info(
            f"♥ price=${price:,.1f} | {in_pos} "
            f"| pnl=${risk['daily_pnl']:+.2f} | trades={risk['trades_today']} "
            f"| strategy={'ON' if self._strategy.is_enabled else 'OFF'} "
            f"| halted={risk['is_halted']}"
        )

    def _telegram_heartbeat(self):
        price  = self._data_mgr.get_last_price() if self._data_mgr else 0
        risk   = self._risk.get_status()
        pos    = self._orders.get_status()
        pos_tx = (f"📈 {pos['side'].upper()} {pos['size']}c @ ${pos['entry_price']:,.1f}"
                  if pos.get("in_position") else "⬜ Flat")
        self._telegram.send_message(
            f"📊 *Hourly Summary*\n"
            f"Price: `${price:,.1f}`\n"
            f"Position: {pos_tx}\n"
            f"Daily PnL: `${risk['daily_pnl']:+.2f}`\n"
            f"Trades: `{risk['trades_today']}` | Consec losses: `{risk['consecutive_losses']}`\n"
            f"Strategy: `{'ON ▶' if self._strategy.is_enabled else 'OFF ⏸'}`\n"
            f"Risk: `{'🔴 HALTED — ' + risk['halt_reason'] if risk['is_halted'] else '🟢 OK'}`"
        )

    def _health_check(self):
        if not self._data_mgr.is_ready:
            logger.warning("DataManager not ready — restarting streams")
            self._telegram.send_message("⚠️ *DataManager not ready* — restarting WS streams")
            self._data_mgr.restart_streams()
        elif not self._data_mgr.is_price_fresh(max_stale_seconds=120):
            logger.warning("Price data stale (>120s)")
            self._telegram.send_message("⚠️ *Price data stale* (>2 min) — check connection")

    # ─── SHUTDOWN ─────────────────────────────────────────────────────────────

    def shutdown(self):
        logger.info("Shutting down HPMS...")
        self._shutdown.set()
        if self._strategy:
            self._strategy.stop()
        if self._orders and self._orders.is_in_position:
            logger.warning("Closing open position on shutdown")
            price = self._data_mgr.get_last_price() if self._data_mgr else 0
            self._orders.close_position(reason="SHUTDOWN", current_price=price)
        if self._data_mgr:
            self._data_mgr.stop()
        if self._telegram:
            self._telegram.send_message("🔌 *HPMS Shutdown*")
            time.sleep(1)
            self._telegram.stop()
        logger.info("Shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPMS Trading System")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--testnet",       action="store_true")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="Start paused; use /start_trading in Telegram")
    args = parser.parse_args()

    setup_logging()
    runner = HPMSRunner(dry_run=args.dry_run, testnet=args.testnet,
                        auto_start=not args.no_auto_start)

    def handle_signal(sig, frame):
        runner.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
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
