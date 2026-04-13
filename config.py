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
# SIGNAL THRESHOLDS (v2 — Institutional)
# ═══════════════════════════════════════════════════════════════════════════════
# delta_q_threshold: predicted log-price move required for entry.
#   Adaptive: actual threshold = max(config * 0.5, std_log * 0.3)
#   This value is the floor — volatility scaling handles the rest.
SIGNAL_DELTA_Q_THRESHOLD  = 0.0006  # floor for adaptive threshold
SIGNAL_DH_DT_MAX          = 0.08    # soft confidence factor reference (NOT a hard gate)
SIGNAL_H_PERCENTILE       = 99.5    # relaxed from 99.0 — allows breakout regimes
SIGNAL_MIN_MOMENTUM       = 0.00005 # minimum |p_pred| at horizon
SIGNAL_ACCELERATION_CHECK = True    # ALWAYS ON — second derivative confirmation required

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTION — FIBONACCI TP/SL & TRAILING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture: Fibonacci-level TP/SL derived from structural swing analysis
# ──────────────────────────────────────────────────────────────────────────
# SL is placed at a Fibonacci retracement level below (LONG) or above (SHORT)
# the entry, identified from volume-weighted swing highs/lows.  An ATR buffer
# pushes the SL just beyond the Fibonacci level (institutional practice: SL
# sits past the structure, not on it).
#
# TP is placed at a Fibonacci extension level (1.272 or 1.618 projection of
# the originating swing), capped by FIB_TP_CAP_PCT.
#
# Trailing uses Fibonacci retracement of the developing move (entry → high
# watermark), tightening through the golden ratio schedule as TP progress
# increases.  The breakeven floor guarantees net-zero after fees at all times
# once trailing activates.

# ── Fibonacci Swing Detection ─────────────────────────────────────────────
FIB_SWING_MIN_ORDER       = 3       # minimum fractal order (7-bar pattern)
FIB_SWING_MAX_ORDER       = 10      # maximum fractal order (21-bar pattern)
FIB_SWING_ATR_NOISE       = 0.5     # min swing range as fraction of ATR (noise filter)
FIB_MAX_SWING_PAIRS       = 6       # max swing pairs for multi-scale Fib computation
FIB_CONFLUENCE_ATR_TOL    = 0.3     # confluence clustering tolerance (fraction of ATR)

# ── Fibonacci TP/SL Caps ──────────────────────────────────────────────────
FIB_TP_CAP_PCT            = 0.014   # max TP distance as fraction of price (1.4% — was 0.8%)
FIB_SL_CAP_PCT            = 0.007   # max SL distance as fraction of price (0.7% — was 0.4%)
FIB_SL_ATR_BUFFER_MULT    = 0.2     # ATR multiplier for SL buffer beyond Fib level (was 0.3)
FIB_MIN_RR                = 2.0     # minimum gross R:R ratio

# ── Fibonacci Trailing Stop ──────────────────────────────────────────────
# Phase lifecycle (SL only ever ratchets in the favorable direction):
#   WARMUP      bars_held < warmup_bars         SL unchanged
#   INITIAL     favorable_move < activation      SL unchanged
#   BREAKEVEN   trailing active, fee floor binds  SL = entry + fees × margin
#   FIB_TRAIL   trailing active, Fib SL binds     SL at Fib retracement of move
#   FIB_LOCK    tp_progress ≥ 75%                 SL at 0.236 retracement (tight)
#
# Fibonacci trail schedule (tp_progress → retracement ratio):
#   20% → 0.786 (wide, let trade breathe — keep 21% of move)
#   40% → 0.618 (golden ratio — keep 38% of move)
#   60% → 0.500 (half — keep 50% of move)
#   75% → 0.382 (tight — keep 62% of move)
#   90% → 0.236 (lock — keep 76% of move)
TRAILING_ENABLED              = True
TRAILING_WARMUP_BARS          = 2       # don't modify SL for first N bars
TRAILING_BE_ACTIVATION_PCT    = 0.20    # activate trailing after 20% of TP distance
TRAILING_BE_FEE_MARGIN        = 1.1     # breakeven SL = entry + (round_trip_fees × this)
TRAILING_MIN_STEP_TICKS       = 0.5     # minimum SL move in price (avoid API spam)
TRAILING_ABSOLUTE_MAX_BARS    = 120     # hard safety ceiling (2 hours) — exits ONLY if profitable

TRADE_DH_DT_EXIT_SPIKE    = 0.25    # exit if |dH/dt| spikes above this (adaptive)
TRADE_USE_BRACKET_ORDERS  = True    # use Delta bracket orders for atomic SL/TP
TRADE_ORDER_TYPE           = "market"  # "market" | "limit"
TRADE_LIMIT_OFFSET_TICKS  = 1       # ticks away from mid for limit entries
TRADE_CONTRACT_VALUE      = 0.001   # base asset units per contract (Delta BTCUSD = 0.001 BTC)

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING & RISK (v2 — Confidence-weighted, Vol-normalized)
# ═══════════════════════════════════════════════════════════════════════════════
RISK_MAX_POSITION_USD     = 500.0   # max notional per trade
RISK_MAX_POSITION_CONTRACTS = 100   # max contracts per trade
RISK_LEVERAGE             = 50      # exchange leverage setting
RISK_MAX_DAILY_LOSS_USD   = 200.0   # daily loss circuit breaker
RISK_MAX_DAILY_TRADES     = 100      # max trades per day
RISK_MAX_CONSECUTIVE_LOSSES = 5     # pause after N consecutive losses
RISK_COOLDOWN_SECONDS     = 10.0    # base cooldown (scales with consec losses)
RISK_MAX_DRAWDOWN_PCT     = 5.0     # max drawdown % from session high
RISK_EQUITY_PCT_PER_TRADE = 2.0     # base % of equity risked (scaled by confidence × vol)
RISK_AUTO_RESUME_SECONDS  = 300.0   # auto-resume from CONSECUTIVE_LOSSES after 5 min
RISK_SOFT_LOSS_WEIGHT     = 0.5     # forced exits count as 0.5 toward consecutive losses

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
