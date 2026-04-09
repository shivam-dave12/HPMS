"""
config.py — HPMS System Configuration
=======================================
All tunable parameters for Hamiltonian Phase-Space Micro-Scalping.
Every value here is overridable via Telegram /set commands at runtime.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE — DELTA
# ═══════════════════════════════════════════════════════════════════════════════
DELTA_API_KEY       = os.getenv("DELTA_API_KEY", "")
DELTA_SECRET_KEY    = os.getenv("DELTA_SECRET_KEY", "")
DELTA_TESTNET       = os.getenv("DELTA_TESTNET", "false").lower() == "true"
DELTA_SYMBOL        = os.getenv("DELTA_SYMBOL", "BTCUSD")
DELTA_API_MIN_INTERVAL = 0.25  # seconds between REST calls

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ADMIN_IDS  = [int(x) for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()]

# ═══════════════════════════════════════════════════════════════════════════════
# HPMS ENGINE — PHASE SPACE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
HPMS_TAU                  = 5       # Takens embedding delay (bars)
HPMS_LOOKBACK             = 60      # bars for V(q) kernel density estimation
HPMS_PREDICTION_HORIZON   = 5       # bars to integrate forward
HPMS_KDE_BANDWIDTH        = 0.3     # KDE bandwidth for potential landscape
HPMS_KDE_GRID_POINTS      = 256     # resolution of V(q) mesh
HPMS_INTEGRATOR           = "rk4"   # "rk4" | "euler" | "leapfrog"
HPMS_INTEGRATION_DT       = 0.2     # sub-step size for integrator (fraction of 1 bar)
HPMS_MASS                 = 1.0     # effective mass m in H = p²/2m + V(q)
HPMS_NORMALIZATION_WINDOW = 120     # bars for z-score normalisation of q
HPMS_H_EMA_SPAN          = 5       # EMA span for dH/dt smoothing (reduces false exits)
HPMS_KDE_REBUILD_INTERVAL = 3      # rebuild V(q) every N bars (1 = every bar)
HPMS_TRAJECTORY_LOG_DEPTH = 20     # keep last N trajectories for diagnostics

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
# delta_q_threshold: predicted log-price move required for entry.
#   predicted_pct_move = delta_q_zscored * std(log_prices_120bar)
#   On 1m BTC, std_log ≈ 0.001–0.003, so 0.0022 required delta_q>0.73–2.2σ
#   — almost never fires.  0.0006 = ~0.06% predicted move, ~$43 at $71k. 
SIGNAL_DELTA_Q_THRESHOLD  = 0.0006  # was 0.0022 — lowered to fire on 1m BTC
SIGNAL_DH_DT_MAX          = 0.08    # was 0.05 — slightly relaxed
SIGNAL_H_PERCENTILE       = 99.0    # skip if |H| above this percentile (chaotic)
SIGNAL_MIN_MOMENTUM       = 0.00005 # was 0.0001 — lowered to match scaled units
SIGNAL_ACCELERATION_CHECK = False   # was True — disable until signal rate confirmed

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
TRADE_TP_PCT              = 0.0035  # 0.35% fixed TP
TRADE_SL_PCT              = 0.0018  # 0.18% fixed SL
TRADE_MAX_HOLD_BARS       = 5       # forced flat after N bars (~N minutes)
TRADE_DH_DT_EXIT_SPIKE    = 0.15    # exit if |dH/dt| spikes above this
TRADE_USE_BRACKET_ORDERS  = True    # use Delta bracket orders for atomic SL/TP
TRADE_ORDER_TYPE           = "market"  # "market" | "limit"
TRADE_LIMIT_OFFSET_TICKS  = 1       # ticks away from mid for limit entries
TRADE_CONTRACT_VALUE      = 0.001   # base asset units per contract (Delta BTCUSD = 0.001 BTC)

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING & RISK
# ═══════════════════════════════════════════════════════════════════════════════
RISK_MAX_POSITION_USD     = 500.0   # max notional per trade
RISK_MAX_POSITION_CONTRACTS = 100   # max contracts per trade
RISK_LEVERAGE             = 50      # exchange leverage setting  ← raised from 10 so 1 contract (~$72 notional) needs only ~$1.44 margin
RISK_MAX_DAILY_LOSS_USD   = 200.0   # daily loss circuit breaker
RISK_MAX_DAILY_TRADES     = 50      # max trades per day
RISK_MAX_CONSECUTIVE_LOSSES = 5     # pause after N consecutive losses
RISK_COOLDOWN_SECONDS     = 30.0    # cooldown between trades
RISK_MAX_DRAWDOWN_PCT     = 5.0     # max drawdown % from session high
RISK_EQUITY_PCT_PER_TRADE = 2.0     # % of equity risked per trade

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
FILTER_NEWS_BLACKOUT_SECONDS = 120  # skip signals ±2min around news
FILTER_SPREAD_MAX_PCT        = 0.05 # max bid-ask spread % to trade
FILTER_MIN_VOLUME_1M         = 10.0 # min 1m bar volume to trade
FILTER_VOLATILITY_MIN_PCT    = 0.01 # min 1m ATR% (avoid dead market)
FILTER_VOLATILITY_MAX_PCT    = 2.0  # max 1m ATR% (avoid insane vol)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING — console only, no disk writes
# ═══════════════════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
