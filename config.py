"""
config.py — HPMS System Configuration (Hyperliquid)
=====================================================
All tunable parameters for Hamiltonian Phase-Space Micro-Scalping.
Every value here is overridable via Telegram /set commands at runtime.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE — HYPERLIQUID
# ═══════════════════════════════════════════════════════════════════════════════
HL_PRIVATE_KEY      = os.getenv("HL_PRIVATE_KEY", "")
HL_WALLET_ADDRESS   = os.getenv("HL_WALLET_ADDRESS", "")       # master wallet (for balance queries)
HL_MAINNET          = os.getenv("HL_MAINNET", "true").lower() == "true"
HL_SYMBOL           = os.getenv("HL_SYMBOL", "BTC")             # coin name on HL (e.g. "BTC", "ETH")

# Hyperliquid REST + WS endpoints
HL_REST_URL_MAINNET  = "https://api.hyperliquid.xyz"
HL_REST_URL_TESTNET  = "https://api.hyperliquid-testnet.xyz"
HL_WS_URL_MAINNET    = "wss://api.hyperliquid.xyz/ws"
HL_WS_URL_TESTNET    = "wss://api.hyperliquid-testnet.xyz/ws"

# HTTP settings
HL_HTTP_TIMEOUT_S    = 15.0
HL_RETRY_ATTEMPTS    = 3
HL_RETRY_DELAY_S     = 1.0
HL_MAX_FILL_AGE_S    = 120        # drop WS fills older than this (seconds)

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERLIQUID FEES (base tier — actual fees come from fill data)
# These are used ONLY for pre-flight estimation; real P&L uses exchange data.
#
# Current Hyperliquid perp fee schedule (base tier, no staking discount):
#   Taker:  0.045%  (0.00045)
#   Maker:  0.015%  (0.00015)
#
# The bot uses market orders → always taker.
# If you have staking discounts or volume tiers, the actual fee from
# fills will be lower than this estimate — that's fine, estimates are
# only for pre-flight margin checks.
# ═══════════════════════════════════════════════════════════════════════════════
FEE_TAKER_RATE       = 0.00045    # 0.045% — used for estimation only
FEE_MAKER_RATE       = 0.00015    # 0.015% — informational

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ADMIN_IDS  = [int(x) for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()]

# ═══════════════════════════════════════════════════════════════════════════════
# HPMS ENGINE — PHASE SPACE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
HPMS_TAU                  = 5
HPMS_LOOKBACK             = 60
HPMS_PREDICTION_HORIZON   = 5
HPMS_KDE_BANDWIDTH        = 0.3
HPMS_KDE_GRID_POINTS      = 256
HPMS_INTEGRATOR           = "rk4"
HPMS_INTEGRATION_DT       = 0.2
HPMS_MASS                 = 1.0
HPMS_NORMALIZATION_WINDOW = 120
HPMS_H_EMA_SPAN          = 5
HPMS_KDE_REBUILD_INTERVAL = 3
HPMS_TRAJECTORY_LOG_DEPTH = 20

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS (v2 — Institutional)
# ═══════════════════════════════════════════════════════════════════════════════
SIGNAL_DELTA_Q_THRESHOLD  = 0.0006
SIGNAL_DH_DT_MAX          = 0.08
SIGNAL_H_PERCENTILE       = 99.5
SIGNAL_MIN_MOMENTUM       = 0.00005
SIGNAL_ACCELERATION_CHECK = True

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTION (v2 — Dynamic TP/SL)
# ═══════════════════════════════════════════════════════════════════════════════
TRADE_TP_PCT              = 0.0035
TRADE_SL_PCT              = 0.0018
TRADE_MAX_HOLD_BARS       = 8
TRADE_DH_DT_EXIT_SPIKE    = 0.25
TRADE_USE_BRACKET_ORDERS  = True    # place SL/TP as positionTpsl on HL
TRADE_ORDER_TYPE           = "market"
TRADE_LIMIT_OFFSET_TICKS  = 1

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING & RISK (v2 — Confidence-weighted, Vol-normalized)
# ═══════════════════════════════════════════════════════════════════════════════
RISK_MAX_POSITION_USD     = 500.0
RISK_LEVERAGE             = 40
RISK_MAX_DAILY_LOSS_USD   = 200.0
RISK_MAX_DAILY_TRADES     = 100
RISK_MAX_CONSECUTIVE_LOSSES = 5
RISK_COOLDOWN_SECONDS     = 10.0
RISK_MAX_DRAWDOWN_PCT     = 5.0
RISK_EQUITY_PCT_PER_TRADE = 2.0
RISK_AUTO_RESUME_SECONDS  = 300.0
RISK_SOFT_LOSS_WEIGHT     = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGER — MINIMUM CANDLES FOR READINESS
# ═══════════════════════════════════════════════════════════════════════════════
MIN_CANDLES_1M   = 100
MIN_CANDLES_5M   = 50
MIN_CANDLES_15M  = 20
MIN_CANDLES_1H   = 10
MIN_CANDLES_4H   = 5
MIN_CANDLES_1D   = 3

# ═══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════════════════
FILTER_NEWS_BLACKOUT_SECONDS = 120
FILTER_SPREAD_MAX_PCT        = 0.05
FILTER_MIN_VOLUME_1M         = 0.1     # HL volume is in BASE ASSET (e.g. BTC, not USD)
                                        # 0.1 BTC ≈ $7k–$10k notional — filters dead markets
FILTER_VOLATILITY_MIN_PCT    = 0.01
FILTER_VOLATILITY_MAX_PCT    = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS DATACLASS — used by api_client.py / ws_client.py
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Settings:
    """Runtime settings consumed by the HL SDK modules."""
    private_key:          str   = ""
    wallet_address:       str   = ""
    hl_mainnet:           bool  = True
    rest_url:             str   = ""
    ws_url:               str   = ""
    http_timeout_s:       float = 15.0
    retry_attempts:       int   = 3
    retry_delay_s:        float = 1.0
    max_fill_age_seconds: int   = 120

_settings_instance: Optional[_Settings] = None

def get_settings() -> _Settings:
    """Singleton settings for HL SDK modules."""
    global _settings_instance
    if _settings_instance is None:
        is_mainnet = HL_MAINNET
        _settings_instance = _Settings(
            private_key          = HL_PRIVATE_KEY,
            wallet_address       = HL_WALLET_ADDRESS.lower() if HL_WALLET_ADDRESS else "",
            hl_mainnet           = is_mainnet,
            rest_url             = HL_REST_URL_MAINNET if is_mainnet else HL_REST_URL_TESTNET,
            ws_url               = HL_WS_URL_MAINNET  if is_mainnet else HL_WS_URL_TESTNET,
            http_timeout_s       = HL_HTTP_TIMEOUT_S,
            retry_attempts       = HL_RETRY_ATTEMPTS,
            retry_delay_s        = HL_RETRY_DELAY_S,
            max_fill_age_seconds = HL_MAX_FILL_AGE_S,
        )
    return _settings_instance
