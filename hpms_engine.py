"""
hpms_engine.py — Hamiltonian Phase-Space Micro-Scalping Engine (v2)
====================================================================
Institutional-grade rewrite addressing:
  1. Energy conservation removed as hard gate (markets are driven, not conservative)
  2. Chaos filter relaxed (breakout regimes are profitable, not chaotic)
  3. Momentum direction is now a HARD gate (no conflicting signals)
  4. Acceleration check re-enabled (second derivative confirmation)
  5. Adaptive KDE lookback (shorter in high-vol, longer in low-vol)
  6. TP/SL proportional to predicted move (not fixed %)
  7. Regime detection embedded in confidence scoring

All computations are pure NumPy — sub-millisecond per tick.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde

from fibonacci import compute_fib_tp_sl, compute_fib_trailing_stop, FibTPSL, FibTrailResult
from logger_core import elog

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SignalType(Enum):
    LONG  = auto()
    SHORT = auto()
    FLAT  = auto()


class RegimeType(Enum):
    TRENDING   = auto()
    CHOPPY     = auto()
    VOLATILE   = auto()
    UNKNOWN    = auto()


@dataclass(slots=True)
class PhaseState:
    q: float
    p: float
    H: float
    kinetic: float
    potential: float
    dH_dt: float
    timestamp: float


@dataclass(slots=True)
class TrajectoryPoint:
    t: float
    q: float
    p: float
    H: float


@dataclass(slots=True)
class HPMSSignal:
    signal_type:       SignalType
    confidence:        float
    predicted_delta_q: float
    predicted_p_final: float
    current_H:         float
    dH_dt:             float
    trajectory:        List[TrajectoryPoint]
    entry_price:       float
    tp_price:          float
    sl_price:          float
    reason:            str
    compute_time_us:   float
    bar_timestamp:     float
    bars_since_kde:    int   = 0
    regime:            RegimeType = RegimeType.UNKNOWN
    trend_strength:    float = 0.0


@dataclass
class EngineState:
    q_history:      List[float] = field(default_factory=list)
    p_history:      List[float] = field(default_factory=list)
    H_history:      List[float] = field(default_factory=list)
    H_ema_history:  List[float] = field(default_factory=list)
    close_history:  List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    timestamps:     List[float] = field(default_factory=list)
    trajectory_log: List[List[TrajectoryPoint]] = field(default_factory=list)
    signal_count:   int = 0
    last_signal_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# POTENTIAL LANDSCAPE V(q)
# ═══════════════════════════════════════════════════════════════════════════════

class PotentialLandscape:
    def __init__(self, bandwidth: float = 0.3, grid_points: int = 256):
        self._bandwidth   = bandwidth
        self._grid_points = grid_points
        self._q_grid:  Optional[np.ndarray] = None
        self._V_grid:  Optional[np.ndarray] = None
        self._dV_grid: Optional[np.ndarray] = None
        self._q_min:   float = 0.0
        self._q_max:   float = 0.0
        self._is_built: bool = False

    def build(self, q_samples: np.ndarray, weights: Optional[np.ndarray] = None) -> bool:
        if len(q_samples) < 10:
            return False
        try:
            self._q_min = float(np.min(q_samples)) - 1.0
            self._q_max = float(np.max(q_samples)) + 1.0
            self._q_grid = np.linspace(self._q_min, self._q_max, self._grid_points)
            kde = gaussian_kde(q_samples, bw_method=self._bandwidth, weights=weights)
            density = kde(self._q_grid)
            density = np.maximum(density, 1e-12)
            self._V_grid = -np.log(density)
            self._V_grid -= np.min(self._V_grid)
            dq = self._q_grid[1] - self._q_grid[0]
            self._dV_grid = np.gradient(self._V_grid, dq)
            self._is_built = True
            return True
        except Exception as e:
            elog.error("ENGINE_KDE_REBUILD", error=str(e), stage="build_failed")
            self._is_built = False
            return False

    def V(self, q: float) -> float:
        if not self._is_built:
            return 0.0
        return float(np.interp(q, self._q_grid, self._V_grid))

    def dV_dq(self, q: float) -> float:
        if not self._is_built:
            return 0.0
        return float(np.interp(q, self._q_grid, self._dV_grid))

    @property
    def is_built(self) -> bool:
        return self._is_built


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _rk4_step(q, p, dt, m, dV_dq_func):
    dq1 = p / m;         dp1 = -dV_dq_func(q)
    q2 = q + .5*dt*dq1;  p2 = p + .5*dt*dp1
    dq2 = p2 / m;        dp2 = -dV_dq_func(q2)
    q3 = q + .5*dt*dq2;  p3 = p + .5*dt*dp2
    dq3 = p3 / m;        dp3 = -dV_dq_func(q3)
    q4 = q + dt*dq3;     p4 = p + dt*dp3
    dq4 = p4 / m;        dp4 = -dV_dq_func(q4)
    return (q + (dt/6.0)*(dq1+2*dq2+2*dq3+dq4),
            p + (dt/6.0)*(dp1+2*dp2+2*dp3+dp4))

def _leapfrog_step(q, p, dt, m, dV_dq_func):
    p_half = p - .5*dt*dV_dq_func(q)
    q_new  = q + dt*p_half/m
    return q_new, p_half - .5*dt*dV_dq_func(q_new)

def _euler_step(q, p, dt, m, dV_dq_func):
    return q + dt*p/m, p - dt*dV_dq_func(q)

_INTEGRATORS = {"rk4": _rk4_step, "leapfrog": _leapfrog_step, "euler": _euler_step}


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_regime(closes: np.ndarray, lookback: int = 30) -> Tuple[RegimeType, float]:
    """
    Detect market regime from recent closes.
    Returns (regime, trend_strength) where trend_strength is -1..+1.

    Uses:
      - Efficiency ratio (directional move / total path) for trend detection
      - Normalized ATR for volatility classification
    """
    if len(closes) < lookback:
        return RegimeType.UNKNOWN, 0.0

    window = closes[-lookback:]
    log_returns = np.diff(np.log(window))

    # Efficiency ratio: |net move| / sum(|bar moves|)
    net_move = abs(window[-1] - window[0])
    total_path = np.sum(np.abs(np.diff(window)))
    efficiency = net_move / total_path if total_path > 0 else 0.0

    # Trend direction: slope of linear regression
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]
    trend_strength = np.sign(slope) * efficiency

    # Volatility: normalized ATR
    atr = np.mean(np.abs(np.diff(window)))
    mid_price = np.mean(window)
    norm_vol = (atr / mid_price * 100) if mid_price > 0 else 0.0

    if norm_vol > 0.15:  # high volatility regime
        return RegimeType.VOLATILE, trend_strength
    elif efficiency > 0.35:  # trending
        return RegimeType.TRENDING, trend_strength
    else:  # choppy
        return RegimeType.CHOPPY, trend_strength


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSEngine:
    """
    Hamiltonian Phase-Space Micro-Scalping Engine (v2 — Institutional).

    Key changes from v1:
      - Energy conservation is a SOFT confidence factor, not a hard gate
      - Chaos filter uses 99.5th percentile (was 99th) — allows breakouts
      - Momentum direction at horizon is a HARD gate — no conflicting signals
      - Acceleration check is always ON — second derivative confirmation
      - KDE lookback adapts to volatility regime
      - TP/SL proportional to predicted move, bounded by ATR
    """

    def __init__(
        self,
        tau:                  int   = 5,
        lookback:             int   = 60,
        prediction_horizon:   int   = 5,
        kde_bandwidth:        float = 0.3,
        kde_grid_points:      int   = 256,
        integrator:           str   = "rk4",
        integration_dt:       float = 0.2,
        mass:                 float = 1.0,
        normalization_window: int   = 120,
        delta_q_threshold:    float = 0.0006,
        dH_dt_max:            float = 0.08,
        H_percentile:         float = 99.0,
        min_momentum:         float = 0.00005,
        acceleration_check:   bool  = True,
        H_ema_span:           int   = 5,
        kde_rebuild_interval: int   = 3,
        trajectory_log_depth: int   = 20,
        fib_tp_cap_pct:       float = 0.008,
        fib_sl_cap_pct:       float = 0.004,
        fib_sl_atr_buffer:    float = 0.3,
        fib_min_rr:           float = 2.0,
        fib_swing_min_order:  int   = 3,
        fib_swing_max_order:  int   = 10,
        fib_swing_atr_noise:  float = 0.5,
        fib_max_swing_pairs:  int   = 6,
        fib_confluence_tol:   float = 0.3,
    ):
        self._tau                  = tau
        self._lookback             = lookback
        self._horizon              = prediction_horizon
        self._integrator_name      = integrator
        self._dt                   = integration_dt
        self._mass                 = mass
        self._norm_window          = normalization_window
        self._delta_q_threshold    = delta_q_threshold
        self._dH_dt_max            = dH_dt_max
        self._H_percentile         = H_percentile
        self._min_momentum         = min_momentum
        self._acceleration_check   = acceleration_check  # always True now
        self._H_ema_span           = H_ema_span
        self._kde_rebuild_interval = kde_rebuild_interval
        self._trajectory_log_depth = trajectory_log_depth
        self._bars_since_kde_build = 0

        # Fibonacci TP/SL parameters
        self._fib_tp_cap_pct       = fib_tp_cap_pct
        self._fib_sl_cap_pct       = fib_sl_cap_pct
        self._fib_sl_atr_buffer    = fib_sl_atr_buffer
        self._fib_min_rr           = fib_min_rr
        self._fib_swing_min_order  = fib_swing_min_order
        self._fib_swing_max_order  = fib_swing_max_order
        self._fib_swing_atr_noise  = fib_swing_atr_noise
        self._fib_max_swing_pairs  = fib_max_swing_pairs
        self._fib_confluence_tol   = fib_confluence_tol

        # Fee rate for R:R computation (Delta taker fee ≈ 0.053%)
        self._fee_rate = 0.00053

        # Trailing stop high watermark (reset on each new trade by strategy)
        self._trail_high_watermark: float = 0.0

        self._landscape  = PotentialLandscape(bandwidth=kde_bandwidth, grid_points=kde_grid_points)
        self._integrator = _INTEGRATORS.get(integrator, _rk4_step)
        self._state      = EngineState()

        elog.log("SYSTEM_START", component="HPMSEngine",
                 tau=tau, lookback=lookback, horizon=prediction_horizon,
                 integrator=integrator, mass=mass,
                 delta_q_threshold=delta_q_threshold, dH_dt_max=dH_dt_max,
                 fib_tp_cap=fib_tp_cap_pct, fib_sl_cap=fib_sl_cap_pct,
                 fib_min_rr=fib_min_rr)

    # ─── CONFIGURATION HOT-RELOAD ─────────────────────────────────────────────

    def update_param(self, key: str, value) -> bool:
        _MAP = {
            "tau":                  ("_tau",               int),
            "lookback":             ("_lookback",          int),
            "prediction_horizon":   ("_horizon",           int),
            "integration_dt":       ("_dt",                float),
            "mass":                 ("_mass",              float),
            "normalization_window": ("_norm_window",       int),
            "delta_q_threshold":    ("_delta_q_threshold", float),
            "dH_dt_max":            ("_dH_dt_max",         float),
            "H_percentile":         ("_H_percentile",      float),
            "min_momentum":         ("_min_momentum",      float),
            "acceleration_check":   ("_acceleration_check", bool),
            "integrator":           ("_integrator_name",   str),
            "H_ema_span":           ("_H_ema_span",        int),
            "kde_rebuild_interval": ("_kde_rebuild_interval", int),
            "fee_rate":             ("_fee_rate",            float),
            "fib_tp_cap_pct":       ("_fib_tp_cap_pct",    float),
            "fib_sl_cap_pct":       ("_fib_sl_cap_pct",    float),
            "fib_sl_atr_buffer":    ("_fib_sl_atr_buffer", float),
            "fib_min_rr":           ("_fib_min_rr",        float),
            "fib_swing_min_order":  ("_fib_swing_min_order", int),
            "fib_swing_max_order":  ("_fib_swing_max_order", int),
            "fib_swing_atr_noise":  ("_fib_swing_atr_noise", float),
            "fib_max_swing_pairs":  ("_fib_max_swing_pairs", int),
            "fib_confluence_tol":   ("_fib_confluence_tol", float),
        }
        if key not in _MAP:
            return False
        attr, typ = _MAP[key]
        try:
            casted = typ(value) if typ != bool else str(value).lower() in ("true", "1", "yes")
            setattr(self, attr, casted)
            if key == "integrator":
                self._integrator = _INTEGRATORS.get(str(casted), _rk4_step)
            elog.log("RISK_PARAM_UPDATE", component="HPMSEngine", key=key, value=casted)
            return True
        except Exception as e:
            elog.error("RISK_PARAM_UPDATE", error=str(e), key=key, value=value)
            return False

    def get_params(self) -> Dict:
        return {
            "tau":                  self._tau,
            "lookback":             self._lookback,
            "prediction_horizon":   self._horizon,
            "integrator":           self._integrator_name,
            "integration_dt":       self._dt,
            "mass":                 self._mass,
            "normalization_window": self._norm_window,
            "delta_q_threshold":    self._delta_q_threshold,
            "dH_dt_max":            self._dH_dt_max,
            "H_percentile":         self._H_percentile,
            "min_momentum":         self._min_momentum,
            "acceleration_check":   self._acceleration_check,
            "H_ema_span":           self._H_ema_span,
            "kde_rebuild_interval": self._kde_rebuild_interval,
            "fee_rate":             self._fee_rate,
            "fib_tp_cap_pct":       self._fib_tp_cap_pct,
            "fib_sl_cap_pct":       self._fib_sl_cap_pct,
            "fib_sl_atr_buffer":    self._fib_sl_atr_buffer,
            "fib_min_rr":           self._fib_min_rr,
            "fib_swing_min_order":  self._fib_swing_min_order,
            "fib_swing_max_order":  self._fib_swing_max_order,
            "fib_swing_atr_noise":  self._fib_swing_atr_noise,
            "fib_max_swing_pairs":  self._fib_max_swing_pairs,
            "fib_confluence_tol":   self._fib_confluence_tol,
        }

    # ─── PHASE SPACE RECONSTRUCTION ───────────────────────────────────────────

    def _compute_q(self, closes: np.ndarray) -> Tuple[np.ndarray, float]:
        need = self._norm_window + self._lookback + self._tau + 10
        if len(closes) > need:
            closes = closes[-need:]
        log_c = np.log(closes)
        n = min(self._norm_window, len(log_c))
        window = log_c[-n:]

        half_life = max(n / 4.0, 1.0)
        decay = np.exp(-np.log(2.0) / half_life)
        exp_weights = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
        exp_weights /= exp_weights.sum()

        mu  = float(np.dot(exp_weights, window))
        var = float(np.dot(exp_weights, (window - mu) ** 2))
        std = math.sqrt(var) if var > 1e-24 else 1e-12

        return (log_c - mu) / std, std

    def _compute_p(self, q_series: np.ndarray) -> np.ndarray:
        """
        Compute phase-space momentum as smoothed rate of change of q.

        Original: p = (q[t] - q[t-tau]) / tau — pure finite difference.
        Problem: extremely noisy, single-bar spikes dominate.

        Fix: exponentially-weighted moving average of the tau-lagged difference.
        This preserves the directional signal while rejecting single-bar noise.
        The EWM span matches tau, so the smoothing window is physically consistent
        with the embedding delay.
        """
        tau = self._tau
        p = np.full_like(q_series, np.nan)
        if len(q_series) <= tau:
            return p

        # Raw tau-lagged difference (same as before)
        raw_diff = (q_series[tau:] - q_series[:-tau]) / tau

        # EWM smoothing: span = tau gives half-life ≈ tau/2.7 bars
        # This rejects single-bar spikes while preserving multi-bar trends
        if len(raw_diff) >= 3:
            alpha = 2.0 / (tau + 1.0)
            smoothed = np.empty_like(raw_diff)
            smoothed[0] = raw_diff[0]
            for i in range(1, len(raw_diff)):
                smoothed[i] = alpha * raw_diff[i] + (1.0 - alpha) * smoothed[i - 1]
            p[tau:] = smoothed
        else:
            p[tau:] = raw_diff

        return p

    def _compute_H(self, q: float, p: float) -> Tuple[float, float, float]:
        kinetic   = (p ** 2) / (2.0 * self._mass)
        potential = self._landscape.V(q)
        return kinetic + potential, kinetic, potential

    # ─── ADAPTIVE KDE LOOKBACK ────────────────────────────────────────────────

    def _get_adaptive_lookback(self, closes: np.ndarray) -> int:
        """
        Shorter lookback in high-vol (V(q) stales faster), longer in low-vol.
        Range: [30, lookback_config].
        """
        if len(closes) < 30:
            return self._lookback
        recent = closes[-30:]
        atr = np.mean(np.abs(np.diff(recent)))
        mid = np.mean(recent)
        norm_vol = (atr / mid) if mid > 0 else 0.0

        # High vol → shorter lookback (min 30), low vol → full lookback
        if norm_vol > 0.002:  # ~0.2% per bar = high vol for 1m BTC
            return max(30, self._lookback // 2)
        elif norm_vol > 0.001:
            return max(40, int(self._lookback * 0.75))
        else:
            return self._lookback

    # ─── FORWARD INTEGRATION ──────────────────────────────────────────────────

    def _integrate_forward(self, q0: float, p0: float) -> List[TrajectoryPoint]:
        """
        Forward-integrate the trajectory in phase space.

        Core improvement: drift-corrected Hamiltonian.

        The original H = p²/2m + V(q) is purely conservative with V(q) derived
        from KDE of historical q values. This creates a mean-reversion bias:
        trajectories always get pulled back toward the density center (where
        prices WERE), even in trending markets.

        Fix: Add a regime-dependent drift force F:
            H_eff = p²/2m + V(q) - F·q

        In trending markets, F biases trajectories forward with the trend.
        In choppy markets, F ≈ 0 (pure mean-reversion is correct).

        The drift is derived from the efficiency ratio of recent price action,
        NOT from a separate indicator — it's an intrinsic property of the
        phase-space dynamics.
        """
        trajectory = [TrajectoryPoint(t=0.0, q=q0, p=p0, H=self._compute_H(q0, p0)[0])]
        q, p = q0, p0
        steps_per_bar = max(1, int(1.0 / self._dt))

        # ── Compute drift force from recent phase-space dynamics ─────
        # Drift = weighted average of recent momentum, scaled by trend efficiency
        drift_force = 0.0
        if len(self._state.p_history) >= 10 and len(self._state.close_history) >= 20:
            p_arr = np.array(self._state.p_history[-20:])
            closes_recent = np.array(self._state.close_history[-20:])

            # Efficiency ratio: |net move| / total path
            net_move = abs(closes_recent[-1] - closes_recent[0])
            total_path = np.sum(np.abs(np.diff(closes_recent)))
            # efficiency² forces drift to vanish in choppy markets (efficiency≈0)
            # and grow in trending ones.  Cast to Python float before arithmetic
            # to prevent numpy.float64 from contaminating the integrator chain:
            # once dp1 = -dV_func(q) carries a numpy scalar, every subsequent
            # p += dt*dp expression becomes numpy.float64, causing all downstream
            # boolean comparisons (p_pred > 0 etc.) to produce numpy.bool_ instead
            # of Python bool.  numpy.bool_ is not handled natively by json.dumps
            # and falls through to default=str → serialised as the string "True".
            efficiency = float(net_move / total_path) if total_path > 0 else 0.0

            # Drift force = mean momentum × efficiency² (only strong in clear trends)
            # efficiency² makes it vanish quickly in choppy markets
            mean_p = float(np.mean(p_arr[-10:]))
            drift_force = mean_p * efficiency * efficiency * 0.5

        def dV_dq_with_drift(q_val):
            """Effective force: -dV/dq + F (drift opposes potential gradient in trends)"""
            return self._landscape.dV_dq(q_val) - drift_force

        for bar in range(1, self._horizon + 1):
            for _ in range(steps_per_bar):
                q, p = self._integrator(q, p, self._dt, self._mass, dV_dq_with_drift)
            H, _, _ = self._compute_H(q, p)
            trajectory.append(TrajectoryPoint(t=float(bar), q=q, p=p, H=H))

        return trajectory

    # ─── MAIN TICK PROCESSOR ──────────────────────────────────────────────────

    def on_bar_close(
        self,
        closes:    List[float],
        volumes:   List[float],
        timestamp: float,
    ) -> Optional[HPMSSignal]:
        t_start = time.perf_counter_ns()

        closes_arr  = np.array(closes,  dtype=np.float64)
        volumes_arr = np.array(volumes, dtype=np.float64) if volumes else np.ones(len(closes_arr))
        if len(volumes_arr) != len(closes_arr):
            volumes_arr = np.ones(len(closes_arr))

        min_required = max(self._norm_window, self._lookback) + self._tau + 5
        if len(closes_arr) < min_required:
            elog.log("ENGINE_SKIP",
                     reason="INSUFFICIENT_DATA",
                     have=len(closes_arr), need=min_required)
            return None

        # ── Step 1: Phase-space coordinates ───────────────────────────────────
        q_series, ewma_std = self._compute_q(closes_arr)
        p_series = self._compute_p(q_series)

        q_now = float(q_series[-1])
        p_now = float(p_series[-1])

        if np.isnan(p_now):
            elog.log("ENGINE_SKIP", reason="NAN_MOMENTUM", q=q_now)
            return None

        # ── Step 1b: Regime detection ─────────────────────────────────────────
        regime, trend_strength = _detect_regime(closes_arr)

        # ── Step 2: Build / refresh V(q) with adaptive lookback ───────────────
        self._bars_since_kde_build += 1
        kde_rebuilt = False

        if self._bars_since_kde_build >= self._kde_rebuild_interval or not self._landscape.is_built:
            adaptive_lb = self._get_adaptive_lookback(closes_arr)
            q_lookback = q_series[-adaptive_lb:]
            v_lookback = volumes_arr[-adaptive_lb:]

            valid_mask = ~np.isnan(q_lookback)
            q_lookback = q_lookback[valid_mask]
            v_lookback = v_lookback[valid_mask] if len(v_lookback) == len(valid_mask) else None

            v_weights = None
            if v_lookback is not None and len(v_lookback) == len(q_lookback):
                v_sum = np.sum(v_lookback)
                if v_sum > 0:
                    v_weights = v_lookback / v_sum

            nk = len(q_lookback)
            recency = np.exp(np.linspace(-2.0, 0.0, nk))
            recency /= recency.sum()
            if v_weights is not None:
                combined = recency * v_weights
                combined /= combined.sum()
                v_weights = combined
            else:
                v_weights = recency

            if not self._landscape.build(q_lookback, weights=v_weights):
                elog.log("ENGINE_SKIP", reason="KDE_BUILD_FAILED", samples=len(q_lookback))
                return None

            self._bars_since_kde_build = 0
            kde_rebuilt = True
            elog.log("ENGINE_KDE_REBUILD",
                     samples=len(q_lookback),
                     adaptive_lookback=adaptive_lb,
                     volume_weighted=(v_weights is not None),
                     regime=regime.name,
                     q_min=round(float(np.min(q_lookback)), 4),
                     q_max=round(float(np.max(q_lookback)), 4),
                     q_range=round(float(np.max(q_lookback) - np.min(q_lookback)), 4))

        # ── Step 3: Current Hamiltonian ───────────────────────────────────────
        H_now, K_now, V_now = self._compute_H(q_now, p_now)
        dV_at_q = self._landscape.dV_dq(q_now)

        # ── EMA-smoothed H ───────────────────────────────────────────────────
        alpha = 2.0 / (self._H_ema_span + 1.0)
        if self._state.H_ema_history:
            if kde_rebuilt:
                self._state.H_ema_history[-1] = H_now
            H_ema_now = alpha * H_now + (1.0 - alpha) * self._state.H_ema_history[-1]
        else:
            H_ema_now = H_now

        self._state.H_history.append(H_now)
        self._state.H_ema_history.append(H_ema_now)
        self._state.q_history.append(q_now)
        self._state.p_history.append(p_now)
        self._state.close_history.append(float(closes_arr[-1]))
        self._state.volume_history.append(float(volumes_arr[-1]))
        self._state.timestamps.append(timestamp)

        max_hist = self._norm_window + 100
        for lst in (self._state.H_history, self._state.H_ema_history,
                    self._state.q_history, self._state.p_history,
                    self._state.close_history, self._state.volume_history,
                    self._state.timestamps):
            if len(lst) > max_hist:
                del lst[:-max_hist]

        if len(self._state.H_ema_history) < 2:
            elog.log("ENGINE_SKIP", reason="WARMING_UP",
                     history_len=len(self._state.H_ema_history))
            return None

        dH_dt     = H_ema_now - self._state.H_ema_history[-2]
        dH_dt_raw = H_now     - self._state.H_history[-2]

        elog.log("ENGINE_PHASE_STATE",
                 bar=self._state.signal_count + 1,
                 price=round(float(closes_arr[-1]), 2),
                 q=round(q_now, 6), p=round(p_now, 6),
                 H=round(H_now, 6), H_ema=round(H_ema_now, 6),
                 K=round(K_now, 6), V=round(V_now, 6),
                 dV_dq=round(dV_at_q, 6),
                 dH_dt_ema=round(dH_dt, 6), dH_dt_raw=round(dH_dt_raw, 6),
                 kde_rebuilt=kde_rebuilt, regime=regime.name)

        # ── Step 4: Chaos filter (RELAXED — 99.5th percentile) ────────────────
        # Only blocks truly extreme energy states. Breakouts are allowed through.
        H_arr       = np.array(self._state.H_history)
        H_threshold = float(np.percentile(np.abs(H_arr), 99.5))
        chaos_ok    = abs(H_now) <= H_threshold

        elog.log("ENGINE_CRITERIA",
                 check="chaos_filter",
                 H_abs=round(abs(H_now), 6),
                 H_threshold=round(H_threshold, 6),
                 percentile=99.5,
                 pass_=chaos_ok)

        if not chaos_ok:
            reason = f"CHAOTIC: |H|={abs(H_now):.4f} > {H_threshold:.4f}"
            elog.log("ENGINE_SKIP", reason=reason, H_abs=round(abs(H_now), 6))
            t_end = time.perf_counter_ns()
            return HPMSSignal(
                signal_type=SignalType.FLAT, confidence=0.0,
                predicted_delta_q=0.0, predicted_p_final=0.0,
                current_H=H_now, dH_dt=dH_dt, trajectory=[],
                entry_price=float(closes_arr[-1]),
                tp_price=0.0, sl_price=0.0, reason=reason,
                compute_time_us=(t_end - t_start) / 1000.0,
                bar_timestamp=timestamp,
                bars_since_kde=self._bars_since_kde_build,
                regime=regime,
            )

        # ── Step 5: Forward integration ───────────────────────────────────────
        trajectory = self._integrate_forward(q_now, p_now)

        self._state.trajectory_log.append(trajectory)
        if len(self._state.trajectory_log) > self._trajectory_log_depth:
            del self._state.trajectory_log[:-self._trajectory_log_depth]

        q_pred  = trajectory[-1].q
        p_pred  = trajectory[-1].p
        delta_q = q_pred - q_now

        std_log = max(ewma_std, 1e-12)

        # ── Proper inverse z-score: q_pred back to log-price space ───────
        # q = (log_price - mu) / std, so log_price_pred = q_pred * std + mu
        # But mu is the EWMA mean of the normalization window, which we need.
        # We recover mu from: q_now * std + mu = log(current_price)
        # → mu = log(current_price) - q_now * std
        log_current = math.log(float(closes_arr[-1]))
        mu_recovered = log_current - q_now * std_log
        log_price_pred = q_pred * std_log + mu_recovered
        predicted_pct_move = math.exp(log_price_pred - log_current) - 1.0

        elog.log("ENGINE_TRAJECTORY",
                 q_start=round(q_now, 6), q_pred=round(q_pred, 6),
                 p_start=round(p_now, 6), p_pred=round(p_pred, 6),
                 delta_q_zscale=round(delta_q, 6),
                 predicted_pct_move=round(predicted_pct_move, 6),
                 std_log=round(std_log, 8), horizon_bars=self._horizon,
                 drift_corrected=True)

        # ── Step 6: Signal criteria ───────────────────────────────────────────

        # (a) Adaptive delta_q threshold — scale with volatility
        effective_delta_q_threshold = max(
            self._delta_q_threshold * 0.5,
            std_log * 0.3,
        )

        delta_q_ok   = abs(predicted_pct_move) > effective_delta_q_threshold
        delta_q_dir  = "LONG" if predicted_pct_move > 0 else "SHORT"
        elog.log("ENGINE_CRITERIA",
                 check="delta_q_magnitude",
                 predicted_pct=round(predicted_pct_move, 6),
                 threshold=effective_delta_q_threshold,
                 config_threshold=self._delta_q_threshold,
                 std_log=round(std_log, 8),
                 direction=delta_q_dir, pass_=delta_q_ok)

        # (b) Energy conservation — SOFT FACTOR (not hard gate)
        # Markets are driven systems. High dH/dt means external forces acting.
        # Instead of blocking, reduce confidence proportionally.
        energy_stability = max(0.0, 1.0 - abs(dH_dt) / (self._dH_dt_max * 2.0))
        elog.log("ENGINE_CRITERIA",
                 check="energy_stability",
                 dH_dt_ema=round(abs(dH_dt), 6),
                 dH_dt_raw=round(abs(dH_dt_raw), 6),
                 stability_factor=round(energy_stability, 4),
                 note="soft_confidence_factor_not_hard_gate")

        # (c) Momentum magnitude at horizon
        momentum_ok = abs(p_pred) > self._min_momentum
        elog.log("ENGINE_CRITERIA",
                 check="min_momentum",
                 p_pred=round(abs(p_pred), 6),
                 threshold=self._min_momentum, pass_=momentum_ok)

        # (d) Momentum DIRECTION at horizon — HARD GATE
        # If predicted move is LONG but final momentum is negative → conflicting signal → reject.
        # This was the #1 flaw in v1: allowing trades where momentum opposes direction.
        if predicted_pct_move > 0:
            momentum_direction_ok = p_pred > 0
        elif predicted_pct_move < 0:
            momentum_direction_ok = p_pred < 0
        else:
            momentum_direction_ok = True

        elog.log("ENGINE_CRITERIA",
                 check="momentum_direction",
                 p_pred=round(p_pred, 6),
                 predicted_direction=delta_q_dir,
                 direction_aligned=momentum_direction_ok,
                 note="HARD_GATE")

        # (e) Acceleration check — ALWAYS ON
        # LONG: potential slopes down ahead (dV/dq < 0 = downhill = price pushed up)
        # SHORT: potential slopes up ahead (dV/dq > 0 = uphill = price pushed down)
        accel_long  = dV_at_q < 0 and p_pred > 0
        accel_short = dV_at_q > 0 and p_pred < 0
        elog.log("ENGINE_CRITERIA",
                 check="acceleration",
                 dV_dq=round(dV_at_q, 6),
                 p_pred=round(p_pred, 6),
                 accel_long_ok=accel_long,
                 accel_short_ok=accel_short,
                 note="ALWAYS_ENABLED")

        # (f) Current momentum direction — not strongly opposing signal
        if len(self._state.p_history) >= 10:
            p_arr = np.array(self._state.p_history[-min(60, len(self._state.p_history)):]
)
            p_std_recent = float(np.std(p_arr)) if len(p_arr) > 1 else 0.1
            p_now_tol = max(p_std_recent * 1.5, abs(p_now) * 0.5)
        else:
            p_now_tol = 0.30

        p_now_long_ok  = p_now > -p_now_tol
        p_now_short_ok = p_now <  p_now_tol

        elog.log("ENGINE_CRITERIA",
                 check="p_now_direction",
                 p_now=round(p_now, 6),
                 p_now_tol=round(p_now_tol, 6),
                 long_ok=p_now_long_ok, short_ok=p_now_short_ok)

        # (g) Trajectory consistency — soft confidence factor
        traj_consistent = True
        if len(trajectory) >= 3:
            mid = len(trajectory) // 2
            second_half_dq = trajectory[-1].q - trajectory[mid].q
            if predicted_pct_move > 0 and second_half_dq < 0:
                traj_consistent = False
            elif predicted_pct_move < 0 and second_half_dq > 0:
                traj_consistent = False
        traj_factor = 1.0 if traj_consistent else 0.70

        elog.log("ENGINE_CRITERIA",
                 check="trajectory_consistency",
                 traj_consistent=traj_consistent, traj_factor=traj_factor,
                 note="soft_confidence_factor")

        # ── Step 7: Signal classification ─────────────────────────────────────
        current_price = float(closes_arr[-1])
        signal_type   = SignalType.FLAT
        reason        = "NO_SIGNAL"

        # HARD GATES (all must pass):
        #   1. delta_q magnitude — real predicted move, not noise
        #   2. momentum magnitude — residual momentum at horizon
        #   3. momentum direction — must AGREE with predicted direction
        #   4. acceleration — potential gradient supports the move
        #   5. p_now direction — current bar not strongly opposing
        long_criteria  = (predicted_pct_move > effective_delta_q_threshold
                          and momentum_ok
                          and momentum_direction_ok
                          and accel_long
                          and p_now_long_ok)

        short_criteria = (predicted_pct_move < -effective_delta_q_threshold
                          and momentum_ok
                          and momentum_direction_ok
                          and accel_short
                          and p_now_short_ok)

        if long_criteria:
            signal_type = SignalType.LONG
            reason = (f"LONG: dq={predicted_pct_move:.5f} dH_ema={abs(dH_dt):.5f} "
                      f"p_f={p_pred:.5f} dVdq={dV_at_q:.5f} regime={regime.name}")

        elif short_criteria:
            signal_type = SignalType.SHORT
            reason = (f"SHORT: dq={predicted_pct_move:.5f} dH_ema={abs(dH_dt):.5f} "
                      f"p_f={p_pred:.5f} dVdq={dV_at_q:.5f} regime={regime.name}")

        else:
            blockers = []
            if not delta_q_ok:
                blockers.append(f"delta_q={abs(predicted_pct_move):.5f}<{effective_delta_q_threshold:.6f}")
            if not momentum_ok:
                blockers.append(f"|p_pred|={abs(p_pred):.5f}<min={self._min_momentum}")
            if not momentum_direction_ok:
                blockers.append(f"p_pred={p_pred:+.5f} opposes {delta_q_dir}")
            if predicted_pct_move > 0 and not accel_long:
                blockers.append(f"accel_LONG_fail(dVdq={dV_at_q:+.5f})")
            elif predicted_pct_move < 0 and not accel_short:
                blockers.append(f"accel_SHORT_fail(dVdq={dV_at_q:+.5f})")
            if predicted_pct_move > 0 and not p_now_long_ok:
                blockers.append(f"p_now={p_now:+.5f} opposes LONG(tol={p_now_tol:.4f})")
            elif predicted_pct_move < 0 and not p_now_short_ok:
                blockers.append(f"p_now={p_now:+.5f} opposes SHORT(tol={p_now_tol:.4f})")
            soft_notes = []
            if not traj_consistent:
                soft_notes.append(f"traj_reversal(conf×{traj_factor:.2f})")
            if energy_stability < 0.5:
                soft_notes.append(f"energy_unstable({energy_stability:.2f})")
            reason = "FLAT: " + (", ".join(blockers) if blockers else "no criteria met")
            if soft_notes:
                reason += " | soft: " + ", ".join(soft_notes)

        # ── LOG ───────────────────────────────────────────────────────────────
        if signal_type != SignalType.FLAT:
            elog.log("ENGINE_SIGNAL",
                     signal=signal_type.name, reason=reason,
                     predicted_pct=round(predicted_pct_move, 6),
                     price=current_price,
                     regime=regime.name,
                     trend_strength=round(trend_strength, 4),
                     long_criteria_met=long_criteria,
                     short_criteria_met=short_criteria)
        else:
            elog.log("ENGINE_SKIP",
                     reason=reason,
                     predicted_pct=round(predicted_pct_move, 6),
                     delta_q_ok=delta_q_ok,
                     momentum_mag_ok=momentum_ok,
                     momentum_dir_ok=momentum_direction_ok,
                     accel_long_ok=accel_long,
                     accel_short_ok=accel_short,
                     p_now_long_ok=p_now_long_ok,
                     p_now_short_ok=p_now_short_ok,
                     energy_stability=round(energy_stability, 4),
                     traj_factor=round(traj_factor, 3),
                     regime=regime.name)

        # ── Step 8: Fibonacci TP/SL ───────────────────────────────────────────
        #
        # Architecture: structural Fibonacci levels from multi-scale swing analysis
        # ─────────────────────────────────────────────────────────────────────────
        # SL is placed at a Fibonacci retracement level (0.618, 0.786, etc.)
        # derived from detected swing highs/lows, with an ATR buffer to sit
        # just beyond the structural level.
        #
        # TP is placed at a Fibonacci extension level (1.272 or 1.618 projection
        # of the originating swing), capped by fib_tp_cap_pct.
        #
        # Trailing uses Fibonacci retracement of the developing move (entry →
        # high watermark), tightening through the golden ratio schedule as
        # TP progress increases. Breakeven floor ensures net-zero after fees.
        #
        if signal_type != SignalType.FLAT and len(closes_arr) >= 30:
            side = "long" if signal_type == SignalType.LONG else "short"

            fib_result = compute_fib_tp_sl(
                side=side,
                current_price=current_price,
                closes=closes_arr,
                volumes=volumes_arr,
                fee_rate=self._fee_rate,
                min_rr=self._fib_min_rr,
                sl_atr_buffer_mult=self._fib_sl_atr_buffer,
                tp_cap_pct=self._fib_tp_cap_pct,
                sl_cap_pct=self._fib_sl_cap_pct,
                swing_min_order=self._fib_swing_min_order,
                swing_max_order=self._fib_swing_max_order,
                swing_atr_noise=self._fib_swing_atr_noise,
                max_swing_pairs=self._fib_max_swing_pairs,
                confluence_tolerance_atr=self._fib_confluence_tol,
            )

            tp_price    = fib_result.tp_price
            sl_price    = fib_result.sl_price
            tp_distance = fib_result.tp_distance
            sl_distance = fib_result.sl_distance
            gross_rr    = fib_result.gross_rr

            fee_rt_price  = current_price * self._fee_rate * 2
            net_reward    = tp_distance - fee_rt_price
            net_risk      = sl_distance + fee_rt_price
            final_net_rr  = net_reward / net_risk if net_risk > 0 else 0.0

            if gross_rr < self._fib_min_rr:
                signal_type = SignalType.FLAT
                reason = (
                    f"FLAT: fib_rr={gross_rr:.2f}<{self._fib_min_rr} "
                    f"tp=${tp_distance:.0f} sl=${sl_distance:.0f} "
                    f"tp_fib={fib_result.tp_fib_ratio:.3f} "
                    f"sl_fib={fib_result.sl_fib_ratio:.3f}"
                )
                tp_price = 0.0
                sl_price = 0.0
                elog.log("ENGINE_SKIP", reason=reason,
                         gross_rr=round(gross_rr, 3),
                         tp_dist=round(tp_distance, 1),
                         sl_dist=round(sl_distance, 1),
                         min_rr=self._fib_min_rr)
            else:
                elog.log(
                    "ENGINE_FIB_RR",
                    tp_price=round(tp_price, 1),
                    sl_price=round(sl_price, 1),
                    tp_dist=round(tp_distance, 1),
                    sl_dist=round(sl_distance, 1),
                    gross_rr=round(gross_rr, 2),
                    net_rr=round(final_net_rr, 2),
                    tp_fib_ratio=fib_result.tp_fib_ratio,
                    sl_fib_ratio=fib_result.sl_fib_ratio,
                    tp_confluence=fib_result.tp_confluence,
                    sl_confluence=fib_result.sl_confluence,
                    fee_rt_price=round(fee_rt_price, 1),
                    predicted_move=round(abs(predicted_pct_move) * current_price, 1),
                    levels_above=len(fib_result.levels_above),
                    levels_below=len(fib_result.levels_below),
                    be_wr_gross=round(1.0 / (1.0 + gross_rr) * 100.0, 1),
                    be_wr_net=(round(1.0 / (1.0 + final_net_rr) * 100.0, 1)
                               if final_net_rr > 0 else 100.0),
                )

        elif signal_type != SignalType.FLAT:
            # Insufficient data for swing detection — cap-based placement
            if signal_type == SignalType.LONG:
                tp_price = current_price * (1.0 + self._fib_tp_cap_pct)
                sl_price = current_price * (1.0 - self._fib_sl_cap_pct)
            else:
                tp_price = current_price * (1.0 - self._fib_tp_cap_pct)
                sl_price = current_price * (1.0 + self._fib_sl_cap_pct)
            elog.log("ENGINE_FIB_RR", note="insufficient_data_cap_based",
                     tp_price=round(tp_price, 1), sl_price=round(sl_price, 1))
        else:
            tp_price = 0.0
            sl_price = 0.0

        # ── Confidence score (regime-aware) ───────────────────────────────────
        if signal_type != SignalType.FLAT:
            # c1: predicted move magnitude vs threshold (30%)
            c1 = min(1.0, abs(predicted_pct_move) / (effective_delta_q_threshold * 3))
            # c2: energy stability — soft, not binary (20%)
            c2 = energy_stability
            # c3: momentum magnitude (15%)
            c3 = min(1.0, abs(p_pred) / (self._min_momentum * 10))
            # c4: trajectory consistency (15%)
            c4 = traj_factor
            # c5: regime alignment (20%)
            # Trending regime + signal aligns with trend = high confidence
            if regime == RegimeType.TRENDING:
                trend_aligns = ((signal_type == SignalType.LONG and trend_strength > 0) or
                                (signal_type == SignalType.SHORT and trend_strength < 0))
                c5 = 0.9 if trend_aligns else 0.3
            elif regime == RegimeType.CHOPPY:
                c5 = 0.4  # choppy regime — lower confidence
            elif regime == RegimeType.VOLATILE:
                c5 = 0.6  # volatile — medium, could go either way
            else:
                c5 = 0.5

            confidence = c1 * 0.30 + c2 * 0.20 + c3 * 0.15 + c4 * 0.15 + c5 * 0.20
        else:
            confidence = 0.0

        t_end = time.perf_counter_ns()
        compute_us = (t_end - t_start) / 1000.0
        self._state.signal_count += 1

        # ── INFO-level bar summary ────────────────────────────────────────────
        bar_n = self._state.signal_count
        if signal_type != SignalType.FLAT:
            logger.info(
                f"▶ bar#{bar_n:4d}  ${current_price:,.1f}  "
                f"Δq={predicted_pct_move*100:+.4f}%  "
                f"|dH/dt|={abs(dH_dt):.5f}  |p|={abs(p_pred):.6f}  "
                f"→ {signal_type.name}  conf={confidence:.1%}  "
                f"TP=${tp_price:,.1f}  SL=${sl_price:,.1f}  "
                f"regime={regime.name}  [{compute_us:.0f}µs]"
            )
        else:
            blockers_str = reason.replace("FLAT: ", "")
            logger.info(
                f"  bar#{bar_n:4d}  ${current_price:,.1f}  "
                f"q={q_now:+.4f} p={p_now:+.5f} H={H_now:.4f} "
                f"dH/dt={dH_dt:+.5f} regime={regime.name}  "
                f"FLAT {blockers_str}"
            )

        return HPMSSignal(
            signal_type=signal_type, confidence=confidence,
            predicted_delta_q=predicted_pct_move, predicted_p_final=p_pred,
            current_H=H_now, dH_dt=dH_dt, trajectory=trajectory,
            entry_price=current_price, tp_price=tp_price, sl_price=sl_price,
            reason=reason, compute_time_us=compute_us,
            bar_timestamp=timestamp,
            bars_since_kde=self._bars_since_kde_build,
            regime=regime,
            trend_strength=trend_strength,
        )

    # ─── DIAGNOSTICS ──────────────────────────────────────────────────────────

    def get_phase_state(self) -> Optional[PhaseState]:
        if not self._state.q_history:
            return None
        q = self._state.q_history[-1]
        p = self._state.p_history[-1]
        H, K, V = self._compute_H(q, p)
        dH = (self._state.H_ema_history[-1] - self._state.H_ema_history[-2]
              if len(self._state.H_ema_history) >= 2 else 0.0)
        return PhaseState(q=q, p=p, H=H, kinetic=K, potential=V, dH_dt=dH,
                          timestamp=self._state.timestamps[-1] if self._state.timestamps else 0.0)

    def get_diagnostics(self) -> Dict:
        ps = self.get_phase_state()
        traj_summary = None
        if self._state.trajectory_log:
            last_traj = self._state.trajectory_log[-1]
            traj_summary = [
                {"t": tp.t, "q": round(tp.q, 6), "p": round(tp.p, 6), "H": round(tp.H, 6)}
                for tp in last_traj
            ]
        H_raw  = self._state.H_history[-1]     if self._state.H_history     else None
        H_ema  = self._state.H_ema_history[-1] if self._state.H_ema_history else None
        dH_raw = (self._state.H_history[-1]     - self._state.H_history[-2]
                  if len(self._state.H_history) >= 2 else None)
        dH_ema = (self._state.H_ema_history[-1] - self._state.H_ema_history[-2]
                  if len(self._state.H_ema_history) >= 2 else None)
        return {
            "phase_state": {
                "q": ps.q if ps else None, "p": ps.p if ps else None,
                "H": ps.H if ps else None, "K": ps.kinetic if ps else None,
                "V": ps.potential if ps else None, "dH_dt_ema": ps.dH_dt if ps else None,
            },
            "H_smoothing": {
                "H_raw": round(H_raw, 6) if H_raw is not None else None,
                "H_ema": round(H_ema, 6) if H_ema is not None else None,
                "dH_raw": round(dH_raw, 6) if dH_raw is not None else None,
                "dH_ema": round(dH_ema, 6) if dH_ema is not None else None,
                "ema_span": self._H_ema_span,
            },
            "landscape_built": self._landscape.is_built,
            "bars_since_kde_build": self._bars_since_kde_build,
            "kde_rebuild_interval": self._kde_rebuild_interval,
            "history_len": len(self._state.close_history),
            "signal_count": self._state.signal_count,
            "trajectory_log_depth": len(self._state.trajectory_log),
            "last_trajectory": traj_summary,
            "params": self.get_params(),
        }

    def get_adaptive_dH_spike_threshold(self, multiplier: float = 3.0) -> float:
        if len(self._state.H_ema_history) < 10:
            return 0.15
        dH_series = []
        ema_hist = self._state.H_ema_history
        for i in range(1, min(len(ema_hist), self._norm_window)):
            dH_series.append(abs(ema_hist[-i] - ema_hist[-i - 1]))
        if not dH_series:
            return 0.15
        arr = np.array(dH_series)
        threshold = float(np.mean(arr) + multiplier * np.std(arr))
        return max(threshold, 0.10)

    # ─── FIBONACCI TRAILING STOP ────────────────────────────────────────────

    def compute_trailing_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        tp_price: float,
        entry_fee_usd: float,
        position_size: int,
        contract_value: float,
        bars_held: int,
        config,
    ) -> dict:
        """
        Compute Fibonacci-level trailing stop with breakeven floor.

        Delegates to fibonacci.compute_fib_trailing_stop(). The trailing SL
        is placed at the Fibonacci retracement of the developing move (entry →
        high watermark), with the specific ratio tightening as tp_progress
        increases through the golden ratio schedule:

          20% progress → 0.786 retracement (wide, keep 21% of move)
          40% progress → 0.618 retracement (golden ratio, keep 38%)
          60% progress → 0.500 retracement (half, keep 50%)
          75% progress → 0.382 retracement (tight, keep 62%)
          90% progress → 0.236 retracement (lock, keep 76%)

        The breakeven floor (entry + round_trip_fees × margin) ensures the
        SL never sits where a fill would result in a net loss after fees.

        Returns dict matching the interface expected by strategy.py:
            {
                "new_sl":            float | None,
                "phase":             str,
                "tp_progress":       float,
                "fee_breakeven_move": float,
                "fib_ratio":         float,
                "high_watermark":    float,
            }
        """
        warmup = getattr(config, "TRAILING_WARMUP_BARS", 2)
        be_activation = getattr(config, "TRAILING_BE_ACTIVATION_PCT", 0.20)
        be_fee_margin = getattr(config, "TRAILING_BE_FEE_MARGIN", 1.1)
        min_step = getattr(config, "TRAILING_MIN_STEP_TICKS", 0.5)
        abs_max = getattr(config, "TRAILING_ABSOLUTE_MAX_BARS", 120)

        trail_result = compute_fib_trailing_stop(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            current_sl=current_sl,
            tp_price=tp_price,
            entry_fee_usd=entry_fee_usd,
            position_size=position_size,
            contract_value=contract_value,
            bars_held=bars_held,
            warmup_bars=warmup,
            be_activation_pct=be_activation,
            be_fee_margin=be_fee_margin,
            min_step_ticks=min_step,
            absolute_max_bars=abs_max,
            high_watermark=self._trail_high_watermark,
        )

        # Persist high watermark for next bar
        self._trail_high_watermark = trail_result.high_watermark

        result = {
            "new_sl":             trail_result.new_sl,
            "phase":              trail_result.phase,
            "tp_progress":        trail_result.tp_progress,
            "fee_breakeven_move": trail_result.fee_breakeven_move,
            "fib_ratio":          trail_result.current_fib_ratio,
            "high_watermark":     trail_result.high_watermark,
        }

        elog.log(
            "ENGINE_FIB_TRAIL",
            phase=trail_result.phase,
            bars=bars_held,
            progress=trail_result.tp_progress,
            fib_ratio=trail_result.current_fib_ratio,
            fib_sl=round(trail_result.fib_sl_price, 1),
            current_sl=round(current_sl, 1),
            new_sl=trail_result.new_sl,
            fee_be=trail_result.fee_breakeven_move,
            hwm=round(trail_result.high_watermark, 1),
        )

        return result

    def reset_trail_watermark(self):
        """Reset the trailing high watermark. Called when a new trade opens."""
        self._trail_high_watermark = 0.0

    def reset(self):
        self._state = EngineState()
        self._bars_since_kde_build = 0
        self._trail_high_watermark = 0.0
        elog.log("SYSTEM_START", component="HPMSEngine", event="state_reset")
