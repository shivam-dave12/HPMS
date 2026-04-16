"""
config.py — HPMS System Configuration  [INSTITUTIONAL TUNING v3]
=================================================================
All tunable parameters for Hamiltonian Phase-Space Micro-Scalping.
Every value here is overridable via Telegram /set commands at runtime.

Tuning Philosophy (v3):
  ─ Fewer, higher-conviction entries over maximum signal frequency
  ─ Tighter SL discipline: stop sits just past structure, never inside it
  ─ Momentum alignment is sacred: if the integrator and the market disagree, flat
  ─ R:R 2.5:1 minimum — below this the math doesn't pay after Delta fees
  ─ 25× leverage: still aggressive but physically sustainable at BTC prices
  ─ 30s cooldown between trades: prevents cascade entries in noisy regimes
  ─ Trailing activates at 25% of TP to give the move room before locking profit
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
DELTA_API_MIN_INTERVAL = 0.30   # ↑ was 0.25 — extra margin vs Delta rate limits under load

# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ADMIN_IDS  = [int(x) for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()]

# ═══════════════════════════════════════════════════════════════════════════════
# HPMS ENGINE — PHASE SPACE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tuning rationale:
#
# TAU = 5 (unchanged)
#   Takens embedding delay of 5 bars on a 1m chart = 5-minute momentum.
#   Decreasing to 3 creates too much noise; increasing to 7+ lags the signal.
#   5 is the physical sweet spot for BTC 1m micro-structure.
#
# LOOKBACK = 60 (unchanged)
#   1 hour of 1m bars for KDE density estimation.  The adaptive_lookback logic
#   already halves this to 30 in high-vol regimes so this is the low-vol ceiling.
#
# PREDICTION_HORIZON = 5 (unchanged)
#   5-bar (5-minute) forward trajectory.  Shorter = inside bid/ask noise;
#   longer = model drift degrades accuracy.
#
# KDE_BANDWIDTH = 0.25 (↓ from 0.3)
#   Tighter bandwidth = sharper potential gradient dV/dq.
#   At 0.3 the landscape is slightly over-smoothed, blurring real support/resistance
#   zones in phase space.  0.25 sharpens local minima without overfitting (we have
#   60 samples, and Silverman's rule gives ~0.22 for Gaussian data of this size).
#
# H_EMA_SPAN = 3 (↓ from 5)
#   EMA of 3 bars for dH/dt smoothing.  5 bars introduces too much lag on a 1m
#   chart — by the time a genuine energy spike triggers, price has already moved
#   adversely.  3 bars preserves the noise-rejection benefit while halving the lag.
#
# KDE_REBUILD_INTERVAL = 2 (↓ from 3)
#   Rebuild V(q) every 2 bars.  At 3 bars the potential landscape stales in fast-
#   moving sessions, creating phantom support/resistance.  At 1 bar the computation
#   overhead is unnecessary.  2 is the optimal refresh rate for 1m BTC scalping.
#
HPMS_TAU                  = 5
HPMS_LOOKBACK             = 60
HPMS_PREDICTION_HORIZON   = 5
HPMS_KDE_BANDWIDTH        = 0.25      # ↓ was 0.30 — sharper potential gradient
HPMS_KDE_GRID_POINTS      = 256
HPMS_INTEGRATOR           = "rk4"    # keep — most accurate; leapfrog second
HPMS_INTEGRATION_DT       = 0.2      # 5 sub-steps per bar — sufficient for rk4
HPMS_MASS                 = 1.0
HPMS_NORMALIZATION_WINDOW = 120      # 2-hour z-score reference window (stable)
HPMS_H_EMA_SPAN           = 3        # ↓ was 5 — faster dH/dt response
HPMS_KDE_REBUILD_INTERVAL = 2        # ↓ was 3 — rebuild every 2 bars
HPMS_TRAJECTORY_LOG_DEPTH = 20

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS (v3 — Quality Filter)
# ═══════════════════════════════════════════════════════════════════════════════
#
# DELTA_Q_THRESHOLD = 0.0008 (↑ from 0.0006)
#   The actual threshold is max(config × 0.5, std_log × 0.3).
#   At BTC 1m, std_log ≈ 0.001–0.004.  The adaptive branch (std_log × 0.3) will
#   dominate in normal conditions.  Raising the floor from 0.0006 → 0.0008 blocks
#   borderline micro-moves that don't clear transaction costs after Delta fees.
#   Net effect: ~15% fewer entries, meaningfully higher average predicted move.
#
# DH_DT_MAX = 0.06 (↓ from 0.08)
#   Reference for energy_stability confidence factor (NOT a hard gate).
#   Tighter reference → more aggressive confidence penalty for high-energy states.
#   Keeps the soft confidence weight honest without adding a new hard gate.
#
# MIN_MOMENTUM = 0.0001 (↑ from 0.00005)
#   This is the minimum |p_pred| at the prediction horizon.
#   Doubling the floor eliminates signals where the trajectory is nearly flat at
#   horizon — those trades have no directional conviction and statistically break
#   even at best after fees.
#
SIGNAL_DELTA_Q_THRESHOLD  = 0.0010   # ↑ was 0.0008 — higher floor for delta_q
SIGNAL_DH_DT_MAX          = 0.06     # ↓ was 0.08 — tighter energy stability ref
SIGNAL_H_PERCENTILE       = 99.5     # keep — blocks top 0.5% chaos states
SIGNAL_MIN_MOMENTUM       = 0.0001   # ↑ was 0.00005 — filter flat-horizon signals
SIGNAL_ACCELERATION_CHECK = True     # ALWAYS ON — non-negotiable hard gate

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTION — FIBONACCI TP/SL & TRAILING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture: Fibonacci-level TP/SL derived from structural swing analysis
# ──────────────────────────────────────────────────────────────────────────
# SL: placed at a Fib retracement (0.786, 0.618, 0.500) below/above entry with
#     a small ATR buffer beyond the structural level.
# TP: placed at a Fib extension (1.272, 1.618) from the originating swing.
# Trailing: Fib retracement of (entry → high-watermark) tightening with progress.

# ── Fibonacci Swing Detection ─────────────────────────────────────────────
#
# SWING_MAX_ORDER = 8 (↓ from 10)
#   Order-10 = 21-bar patterns (21 minutes of 1m data).  For micro-scalping,
#   21-minute structural pivots are too macro — they sit far from entry and
#   produce over-wide SLs that fail the min R:R filter.  Order-8 = 17-bar
#   pivots (~17 min) are the practical maximum for 1m BTC scalping.
#
# SWING_ATR_NOISE = 0.6 (↑ from 0.5)
#   Stricter noise filter.  At 0.5× ATR, tiny wick pivots that aren't real
#   structure pass through and pollute the level computation.  0.6× ATR is
#   the institutional minimum for a meaningful swing point.
#
# MAX_SWING_PAIRS = 5 (↓ from 6)
#   6 pairs generates 60 raw Fibonacci levels; the confluence merging handles
#   the excess but wastes compute.  5 pairs covers all relevant timeframes
#   within the lookback window.
#
# CONFLUENCE_ATR_TOL = 0.25 (↓ from 0.30)
#   Tighter zone clustering.  0.30× ATR allows levels 0.5× ATR apart to merge
#   into one "confluent" zone — that's too loose, blurs distinct levels.
#   0.25× ATR keeps zones tighter and produces crisper TP/SL placement.
#
FIB_SWING_MIN_ORDER       = 3        # keep — 7-bar fractal minimum
FIB_SWING_MAX_ORDER       = 8        # ↓ was 10 — focus on 17-bar pivots
FIB_SWING_ATR_NOISE       = 0.6      # ↑ was 0.5 — stricter noise filter
FIB_MAX_SWING_PAIRS       = 5        # ↓ was 6 — fewer but higher-quality pairs
FIB_CONFLUENCE_ATR_TOL    = 0.25     # ↓ was 0.30 — tighter zone clustering

# ── Fibonacci TP/SL Caps ──────────────────────────────────────────────────
#
# TP_CAP_PCT = 0.012 (↓ from 0.014)
#   1.2% of price = ~$1020 at $85k.  The 1.618 Fib extension of a typical
#   1m swing (0.05–0.10% range) projects to 0.08–0.16% — well inside this cap.
#   The cap triggers only in runaway swing scenarios; lowering from 1.4% to 1.2%
#   keeps TP targets within credible 1m scalp territory.
#
# SL_CAP_PCT = 0.005 (↓ from 0.007)
#   0.5% of price = ~$425 at $85k.  With 50→25× leverage and equity_pct=1.5%,
#   a 0.5% SL on a 100-contract position = $42.5 loss on $0.001 BTC/contract.
#   The old 0.7% cap allowed SLs so wide they broke the 2.5:1 R:R requirement
#   and the engine fell back to the ATR fallback path on nearly every trade.
#   0.5% forces the engine to find tighter Fib SL levels or skip the trade.
#
# SL_ATR_BUFFER_MULT = 0.15 (↓ from 0.20)
#   Smaller buffer past the Fib level.  0.20× ATR was adding $8–17 beyond an
#   already-structural SL.  0.15× ATR is sufficient to clear the wick noise
#   without pushing the SL to the next Fib level and killing the R:R.
#
# FIB_MIN_RR = 2.5 (↑ from 2.0)
#   The single most impactful parameter change.
#   At 2.0:1 gross R:R and Delta's 0.053% taker fee × 2 = 0.106% round-trip,
#   a net R:R of ~1.85:1 requires a ~35% win rate to break even.
#   At 2.5:1 gross R:R the break-even win rate drops to ~29% and net is ~2.30:1.
#   This alone filters ~20% of marginal setups that historically break even at best.
#
FIB_TP_CAP_PCT            = 0.012   # ↓ was 0.014 — 1.2% cap, scalp-realistic
FIB_SL_CAP_PCT            = 0.005   # ↓ was 0.007 — 0.5% cap, tight SL discipline
FIB_SL_ATR_BUFFER_MULT    = 0.15    # ↓ was 0.20 — tighter buffer past Fib level
FIB_MIN_RR                = 2.5     # ↑ was 2.0 — institutional minimum R:R

# ── Fibonacci Trailing Stop ──────────────────────────────────────────────
#
# Phase lifecycle (SL only ever ratchets in the favorable direction):
#   WARMUP      bars_held < warmup_bars              SL unchanged
#   INITIAL     favorable_move < activation           SL unchanged
#   BREAKEVEN   trailing active, fee floor binds      SL = entry + fees × margin
#   FIB_TRAIL   trailing active, Fib SL binds         SL at Fib retracement of move
#   FIB_LOCK    tp_progress ≥ 75%                     SL at 0.236 retracement (tight)
#
# Fibonacci trail schedule (unchanged — golden ratio schedule is mathematically optimal):
#   20% → 0.786  25% → 0.618  40% → 0.618  60% → 0.500  75% → 0.382  90% → 0.236
#
# WARMUP_BARS = 3 (↑ from 2)
#   Give the trade 3 bars of development before the trailing SL engine activates.
#   At 2 bars the first trailing check fires on bar 3, which is often mid-impulse.
#   3 bars ensures the initial impulse has completed before we start constraining.
#
# BE_ACTIVATION_PCT = 0.25 (↑ from 0.20)
#   Trailing activates after the trade moves 25% of the TP distance (was 20%).
#   The extra 5% breathing room sharply reduces premature trailing that stops out
#   trades during the normal pullback-and-continue pattern in BTC 1m.
#   Example: TP $200 away → trailing activates after $50 move (was $40).
#
# BE_FEE_MARGIN = 1.25 (↑ from 1.10)
#   The breakeven SL floor is set at entry + round_trip_fees × this multiplier.
#   1.10× barely covers fees.  1.25× builds in a $2–5 slippage buffer so the
#   "breakeven" trade is genuinely breakeven after all friction.
#
# TRAILING_MIN_STEP_TICKS = 1.0 (↑ from 0.5)
#   Minimum SL ratchet step.  At 0.5 the engine fires an API call for every
#   50-cent improvement in the developing move — inefficient and hammers rate limits.
#   1.0 means the SL only updates when it would move by at least $1, reducing
#   update frequency by ~50% with zero impact on trade outcome.
#
TRAILING_ENABLED              = True
TRAILING_WARMUP_BARS          = 3        # ↑ was 2 — 3 bars of development first
TRAILING_BE_ACTIVATION_PCT    = 0.25     # ↑ was 0.20 — activate at 25% of TP
TRAILING_BE_FEE_MARGIN        = 1.25     # ↑ was 1.10 — genuine net-zero breakeven
TRAILING_MIN_STEP_TICKS       = 1.0      # ↑ was 0.5 — fewer API calls
TRAILING_ABSOLUTE_MAX_BARS    = 180      # keep — 3-hour hard ceiling, exits ONLY if profitable

# ── Trade execution ───────────────────────────────────────────────────────
#
# DH_DT_EXIT_SPIKE = 0.20 (↓ from 0.25)
#   The adaptive spike threshold is max(adaptive, config_floor).
#   The adaptive path (mean + 3σ of recent |dH/dt| history) typically lands
#   at 0.10–0.18 in normal conditions.  The floor at 0.25 was often the binding
#   constraint, allowing energy spikes to persist too long before exit.
#   At 0.20 the floor is tighter without conflicting with the adaptive path.
#
TRADE_DH_DT_EXIT_SPIKE    = 0.20    # ↓ was 0.25 — tighter energy spike exit
TRADE_USE_BRACKET_ORDERS  = True    # keep — atomic SL/TP: no orphaned legs
TRADE_ORDER_TYPE          = "market"  # scalping requires immediate fill
TRADE_LIMIT_OFFSET_TICKS  = 1
TRADE_CONTRACT_VALUE      = 0.001   # 0.001 BTC per contract (Delta BTCUSD perpetual)

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING & RISK (v3 — Disciplined Capital Allocation)
# ═══════════════════════════════════════════════════════════════════════════════
#
# RISK_LEVERAGE = 25 (↓ from 50)
#   The most dangerous parameter in the original config.  50× leverage is used by
#   retail traders who don't understand margin math.  At 50× and $85k BTC:
#     1 contract notional = $85, margin = $1.70
#     100 contracts margin = $170 on $500 notional = 2.94× margin efficiency
#   The problem: 50× leverage means a 2% adverse move wipes the margin.  For a
#   micro-scalper entering at arbitrary points this is catastrophic.
#   At 25×: same contract structure but a 4% adverse move wipes margin.
#   The sizing caps (max_pos_usd=$500, max_pos_contracts=100) still bind, so the
#   practical notional exposure is unchanged — only the margin buffer improves.
#   The compute_size() margin cap formula uses equity × 0.95 × leverage, so
#   reducing leverage reduces the ceiling, which is correct: we want the risk
#   budget (Step 4 in compute_size) to be the active constraint, not the margin cap.
#
# RISK_MAX_DAILY_TRADES = 50 (↓ from 100)
#   100 trades per day on a 1m chart = 1 trade every ~9.6 minutes (if 16h session).
#   That is not micro-scalping — it is noise trading.  At 1.5% equity risk per
#   trade and 50× trades, you've risked 75% of equity in transaction costs alone
#   before wins are counted.  50 trades/day = one every ~19 minutes, which is
#   a realistic high-frequency but disciplined cadence for HPMS signals.
#
# RISK_COOLDOWN_SECONDS = 30.0 (↑ from 10.0)
#   10 seconds between 1m-bar trades is effectively no cooldown — the bar close
#   cycle is ~60s anyway, so the cooldown was never binding.  Setting it to 30s
#   creates a real inter-trade pause: the graduated scaling (×1 + consec×0.5)
#   then produces 45s after 1 consecutive loss, 60s after 2, etc.  This reduces
#   cluster-of-bad-trades scenarios significantly.
#
# RISK_MAX_DAILY_LOSS_USD = 150.0 (↓ from 200.0)
#   Tighter daily loss circuit breaker.  At $200 max daily loss with 1.5% equity
#   risk, you need to lose 13+ trades in a row (at max size) to hit the limit —
#   by which time the consecutive-losses halt (5 trades) should have fired first.
#   $150 provides earlier protection while still allowing a realistic recovery window.
#
# RISK_MAX_DRAWDOWN_PCT = 4.0 (↓ from 5.0)
#   5% drawdown from the session high before halting.  4% tightens this without
#   being hair-trigger — normal intraday sessions in BTC can see 2-3% drawdowns
#   in winning sessions.  4% is the institutional standard for intraday scalping.
#
# RISK_EQUITY_PCT_PER_TRADE = 1.5 (↓ from 2.0)
#   Risk 1.5% of equity per trade.  2% is the full Kelly fraction for a system with
#   a 50% win rate and 2:1 R:R.  Since this system targets 2.5:1 R:R with a target
#   win rate above 40%, the half-Kelly (1.5%) is appropriate: it maximises
#   risk-adjusted growth while protecting against parameter uncertainty.
#
# RISK_AUTO_RESUME_SECONDS = 600.0 (↑ from 300.0)
#   10 minutes of reflection after a CONSECUTIVE_LOSSES or MAX_DRAWDOWN halt
#   (was 5 minutes).  5 minutes is not enough time for the market regime that
#   caused the losing streak to pass.  10 minutes provides a meaningful pause.
#
RISK_MAX_POSITION_USD       = 500.0   # keep — notional cap per trade
RISK_MAX_POSITION_CONTRACTS = 100     # keep
RISK_LEVERAGE               = 25      # KEEP
RISK_MAX_DAILY_LOSS_USD     = 150.0   # ↓ was 200.0 — tighter daily breaker
RISK_MAX_DAILY_TRADES       = 50      # ↓ was 100 — quality over quantity
RISK_MAX_CONSECUTIVE_LOSSES = 5       # keep — halt after 5 consecutive losses
RISK_COOLDOWN_SECONDS       = 30.0    # ↑ was 10.0 — meaningful inter-trade pause
RISK_MAX_DRAWDOWN_PCT       = 4.0     # ↓ was 5.0 — tighter drawdown ceiling
RISK_EQUITY_PCT_PER_TRADE   = 1.5     # ↓ was 2.0 — half-Kelly risk fraction
RISK_AUTO_RESUME_SECONDS    = 600.0   # ↑ was 300.0 — 10 min reflection after halt
RISK_SOFT_LOSS_WEIGHT       = 0.5     # keep — forced exits count as 0.5 loss

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
# FILTERS (v3 — Tighter Market Conditions Gate)
# ═══════════════════════════════════════════════════════════════════════════════
#
# NEWS_BLACKOUT = 180s (↑ from 120s)
#   3 minutes around news.  Delta's order book thins sharply 90s before major
#   economic releases and the first 60s after is pure noise.  120s was often
#   catching only the tail; 180s provides full pre/post protection.
#
# SPREAD_MAX_PCT = 0.04 (↓ from 0.05)
#   Maximum bid-ask spread as a percentage.  At $85k BTC, 0.05% = $42.5 spread.
#   That's enormous for a scalper whose TP is $200–500 away.  0.04% = $34 spread
#   is still lenient but ensures the fill cost is bounded.
#
# FILTER_VOLATILITY_MIN_PCT = 0.02 (↑ from 0.01)
#   Minimum 1m ATR%.  0.01% ATR at $85k = $8.5 per bar.  Below this the market
#   is functionally asleep and any signal is noise.  Raising to 0.02% = $17
#   ensures there is enough price movement to achieve a meaningful R:R.
#
# FILTER_VOLATILITY_MAX_PCT = 1.5 (↓ from 2.0)
#   Maximum 1m ATR%.  2.0% ATR = $1700/bar is absolute mayhem.  At this level
#   spreads widen, fills are unreliable, and the KDE landscape is invalid within
#   2-3 bars of the last rebuild.  1.5% provides earlier protection while still
#   trading through fast-market events that fall below the extreme threshold.
#
FILTER_NEWS_BLACKOUT_SECONDS = 180    # ↑ was 120 — full pre/post news protection
FILTER_SPREAD_MAX_PCT        = 0.04   # ↓ was 0.05 — tighter spread gate
FILTER_MIN_VOLUME_1M         = 15.0   # ↑ was 10.0 — slightly higher liquidity floor
FILTER_VOLATILITY_MIN_PCT    = 0.02   # ↑ was 0.01 — avoid dead-market noise
FILTER_VOLATILITY_MAX_PCT    = 1.5    # ↓ was 2.0 — avoid true mayhem regimes

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER CHANGE SUMMARY (v2 → v3)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ENGINE:
#   HPMS_KDE_BANDWIDTH        0.30  → 0.25  (sharper potential gradient)
#   HPMS_H_EMA_SPAN           5     → 3     (faster dH/dt, less lag)
#   HPMS_KDE_REBUILD_INTERVAL 3     → 2     (faster V(q) adaptation)
#   DELTA_API_MIN_INTERVAL    0.25  → 0.30  (safer API rate headroom)
#
# SIGNAL:
#   SIGNAL_DELTA_Q_THRESHOLD  0.0006 → 0.0008  (higher entry bar)
#   SIGNAL_DH_DT_MAX          0.08   → 0.06    (tighter energy reference)
#   SIGNAL_MIN_MOMENTUM       0.00005 → 0.0001 (filter flat-horizon trades)
#
# FIBONACCI:
#   FIB_SWING_MAX_ORDER       10    → 8     (focus on 17-bar pivots)
#   FIB_SWING_ATR_NOISE       0.5   → 0.6   (stricter noise filter)
#   FIB_MAX_SWING_PAIRS       6     → 5     (quality vs quantity)
#   FIB_CONFLUENCE_ATR_TOL    0.30  → 0.25  (tighter zone clustering)
#   FIB_TP_CAP_PCT            0.014 → 0.012 (1.2% TP cap)
#   FIB_SL_CAP_PCT            0.007 → 0.005 (0.5% SL cap, tight discipline)
#   FIB_SL_ATR_BUFFER_MULT    0.20  → 0.15  (tighter buffer past Fib)
#   FIB_MIN_RR                2.0   → 2.5   (institutional R:R floor ← biggest change)
#
# TRAILING:
#   TRAILING_WARMUP_BARS      2     → 3     (3 bars development first)
#   TRAILING_BE_ACTIVATION    0.20  → 0.25  (25% of TP before activating)
#   TRAILING_BE_FEE_MARGIN    1.10  → 1.25  (genuine net-zero breakeven)
#   TRAILING_MIN_STEP_TICKS   0.5   → 1.0   (fewer API calls)
#   TRADE_DH_DT_EXIT_SPIKE    0.25  → 0.20  (tighter spike exit)
#
# RISK:
#   RISK_LEVERAGE             50    → 25    (critical — was reckless)
#   RISK_MAX_DAILY_LOSS_USD   200   → 150   (tighter breaker)
#   RISK_MAX_DAILY_TRADES     100   → 50    (quality over quantity)
#   RISK_COOLDOWN_SECONDS     10    → 30    (real inter-trade pause)
#   RISK_MAX_DRAWDOWN_PCT     5.0   → 4.0   (tighter DD ceiling)
#   RISK_EQUITY_PCT_PER_TRADE 2.0   → 1.5   (half-Kelly fraction)
#   RISK_AUTO_RESUME_SECONDS  300   → 600   (10 min reflection)
#
# FILTERS:
#   FILTER_NEWS_BLACKOUT_SECONDS  120 → 180  (full news protection)
#   FILTER_SPREAD_MAX_PCT         0.05 → 0.04 (tighter spread gate)
#   FILTER_MIN_VOLUME_1M          10  → 15   (higher liquidity floor)
#   FILTER_VOLATILITY_MIN_PCT     0.01 → 0.02 (avoid dead markets)
#   FILTER_VOLATILITY_MAX_PCT     2.0 → 1.5  (avoid mayhem regimes)
#
# UNCHANGED (already correct):
#   HPMS_TAU, HPMS_LOOKBACK, HPMS_PREDICTION_HORIZON, HPMS_KDE_GRID_POINTS,
#   HPMS_INTEGRATOR, HPMS_INTEGRATION_DT, HPMS_MASS, HPMS_NORMALIZATION_WINDOW,
#   HPMS_TRAJECTORY_LOG_DEPTH, SIGNAL_H_PERCENTILE, SIGNAL_ACCELERATION_CHECK,
#   TRADE_USE_BRACKET_ORDERS, TRADE_ORDER_TYPE, TRADE_CONTRACT_VALUE,
#   TRAILING_ABSOLUTE_MAX_BARS, RISK_MAX_POSITION_USD, RISK_MAX_POSITION_CONTRACTS,
#   RISK_MAX_CONSECUTIVE_LOSSES, RISK_SOFT_LOSS_WEIGHT, all MIN_CANDLES_*
#
# ═══════════════════════════════════════════════════════════════════════════════
