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
    bars_since_kde:    int = 0
    regime:            RegimeType = RegimeType.UNKNOWN


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
        tp_pct:               float = 0.0035,
        sl_pct:               float = 0.0018,
        H_ema_span:           int   = 5,
        kde_rebuild_interval: int   = 3,
        trajectory_log_depth: int   = 20,
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
        self._tp_pct               = tp_pct
        self._sl_pct               = sl_pct
        self._H_ema_span           = H_ema_span
        self._kde_rebuild_interval = kde_rebuild_interval
        self._trajectory_log_depth = trajectory_log_depth
        self._bars_since_kde_build = 0

        self._landscape  = PotentialLandscape(bandwidth=kde_bandwidth, grid_points=kde_grid_points)
        self._integrator = _INTEGRATORS.get(integrator, _rk4_step)
        self._state      = EngineState()

        elog.log("SYSTEM_START", component="HPMSEngine",
                 tau=tau, lookback=lookback, horizon=prediction_horizon,
                 integrator=integrator, mass=mass,
                 delta_q_threshold=delta_q_threshold, dH_dt_max=dH_dt_max)

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
            "tp_pct":               ("_tp_pct",            float),
            "sl_pct":               ("_sl_pct",            float),
            "integrator":           ("_integrator_name",   str),
            "H_ema_span":           ("_H_ema_span",        int),
            "kde_rebuild_interval": ("_kde_rebuild_interval", int),
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
            "tp_pct":               self._tp_pct,
            "sl_pct":               self._sl_pct,
            "H_ema_span":           self._H_ema_span,
            "kde_rebuild_interval": self._kde_rebuild_interval,
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
        tau = self._tau
        p = np.full_like(q_series, np.nan)
        if len(q_series) > tau:
            p[tau:] = (q_series[tau:] - q_series[:-tau]) / tau
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
        trajectory = [TrajectoryPoint(t=0.0, q=q0, p=p0, H=self._compute_H(q0, p0)[0])]
        q, p = q0, p0
        steps_per_bar = max(1, int(1.0 / self._dt))

        for bar in range(1, self._horizon + 1):
            for _ in range(steps_per_bar):
                q, p = self._integrator(q, p, self._dt, self._mass, self._landscape.dV_dq)
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
        predicted_pct_move = delta_q * std_log

        elog.log("ENGINE_TRAJECTORY",
                 q_start=round(q_now, 6), q_pred=round(q_pred, 6),
                 p_start=round(p_now, 6), p_pred=round(p_pred, 6),
                 delta_q_zscale=round(delta_q, 6),
                 predicted_pct_move=round(predicted_pct_move, 6),
                 std_log=round(std_log, 8), horizon_bars=self._horizon)

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

        # ── Step 8: TP/SL — PROPORTIONAL TO PREDICTED MOVE ───────────────────
        # TP = f(predicted_move), bounded by ATR reality
        # SL = volatility-based (ATR), NOT fixed %
        if signal_type != SignalType.FLAT and len(closes_arr) >= 15:
            recent = closes_arr[-15:]
            bar_ranges = np.abs(np.diff(recent))
            atr_1bar = float(np.mean(bar_ranges)) if len(bar_ranges) > 0 else current_price * 0.0005

            # TP proportional to predicted move: use 70% of predicted move as TP
            # (leaving room — don't need the full predicted move to be profitable)
            predicted_dollar_move = abs(predicted_pct_move) * current_price
            tp_distance = max(predicted_dollar_move * 0.70, atr_1bar * 1.5)

            # SL based on ATR: 1.5 × ATR × sqrt(horizon)
            sl_distance = atr_1bar * 1.5 * math.sqrt(self._horizon)

            # Ensure minimum R:R of 1.5:1
            if tp_distance < sl_distance * 1.5:
                tp_distance = sl_distance * 1.5

            # Hard caps from config
            tp_distance = min(tp_distance, current_price * self._tp_pct)
            sl_distance = min(sl_distance, current_price * self._sl_pct)

            # Floors
            tp_distance = max(tp_distance, current_price * 0.0001)
            sl_distance = max(sl_distance, current_price * 0.0001)

            if signal_type == SignalType.LONG:
                tp_price = current_price + tp_distance
                sl_price = current_price - sl_distance
            else:
                tp_price = current_price - tp_distance
                sl_price = current_price + sl_distance
        elif signal_type != SignalType.FLAT:
            if signal_type == SignalType.LONG:
                tp_price = current_price * (1.0 + self._tp_pct)
                sl_price = current_price * (1.0 - self._sl_pct)
            else:
                tp_price = current_price * (1.0 - self._tp_pct)
                sl_price = current_price * (1.0 + self._sl_pct)
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

    def get_dynamic_max_hold(self, tp_distance: float, current_price: float) -> int:
        """Legacy ATR-based max hold estimate (retained for fallback use)."""
        if not self._state.close_history or len(self._state.close_history) < 5:
            return 8
        closes = np.array(self._state.close_history[-20:])
        atr_per_bar = float(np.mean(np.abs(np.diff(closes))))
        if atr_per_bar <= 0:
            return 8
        bars_to_tp = tp_distance / atr_per_bar
        max_hold = int(math.ceil(bars_to_tp * 1.5))
        return max(3, min(max_hold, 30))

    # ─── TRAILING STOP COMPUTATION ──────────────────────────────────────────

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
        Compute the new trailing stop price.

        Design principles:
          1. SL can ONLY improve (ratchet) — never moves backward.
          2. Breakeven is computed from ACTUAL exchange fees (entry + estimated
             exit), so the breakeven SL guarantees zero loss after all fees.
          3. Activation requires the trade to move past BOTH the fee breakeven
             point AND a % of the TP distance — whichever is larger.
          4. Trail distance is wide enough (3× ATR) to survive normal
             1-minute pullbacks.
          5. A warmup period prevents SL modification in the first N bars.

        Fee math (example):
          entry_fee_usd = $0.039 (actual from Delta API)
          exit_fee_est  = $0.039 (same rate)
          total_fees    = $0.078
          For 1 contract (0.001 BTC): need $78 price move to cover fees
          breakeven SL  = entry ± ($78 × 1.1 safety margin)

        Returns:
            {
                "new_sl":            float | None,
                "phase":             str,
                "atr":               float,
                "trail_dist":        float,
                "tp_progress":       float,
                "fee_breakeven_move": float,   # price move needed to cover fees
            }
        """
        is_long = side == "long"
        result = {
            "new_sl": None,
            "phase": "INITIAL",
            "atr": 0.0,
            "trail_dist": 0.0,
            "tp_progress": 0.0,
            "fee_breakeven_move": 0.0,
        }

        # ── Warmup: don't touch SL for first N bars ─────────────────────
        warmup = getattr(config, "TRAILING_WARMUP_BARS", 2)
        if bars_held < warmup:
            result["phase"] = "WARMUP"
            return result

        # ── Compute ATR ──────────────────────────────────────────────────
        closes = list(self._state.close_history) if self._state.close_history else []
        atr_lookback = getattr(config, "TRAILING_ATR_LOOKBACK", 14)
        if len(closes) < 5:
            return result

        closes_arr = np.array(closes[-max(atr_lookback + 1, 20):])
        if len(closes_arr) < 3:
            return result

        atr = float(np.mean(np.abs(np.diff(closes_arr[-atr_lookback:]))))
        if atr <= 0:
            atr = abs(tp_price - entry_price) * 0.05
        result["atr"] = round(atr, 2)

        # ── Fee breakeven computation (from actual exchange fees) ────────
        # entry_fee_usd: exact fee paid on entry (from Delta API paid_commission)
        # exit_fee_est:  estimate exit fee at same rate
        # total: what the trade must earn in gross PnL to break even
        exit_fee_est = entry_fee_usd  # same fee rate on exit
        total_round_trip_fees = entry_fee_usd + exit_fee_est

        # Convert fee $ to price-move needed:
        #   gross_pnl = price_move × position_size × contract_value
        #   price_move = total_fees / (position_size × contract_value)
        if position_size > 0 and contract_value > 0:
            fee_breakeven_move = total_round_trip_fees / (position_size * contract_value)
        else:
            fee_breakeven_move = atr * 2.0  # conservative fallback

        # Safety margin on top of fees (covers slippage on SL trigger)
        fee_margin = getattr(config, "TRAILING_BE_FEE_MARGIN", 1.1)
        fee_breakeven_with_margin = fee_breakeven_move * fee_margin
        result["fee_breakeven_move"] = round(fee_breakeven_move, 2)

        # ── TP distance & progress ───────────────────────────────────────
        entry_to_tp = abs(tp_price - entry_price)
        if entry_to_tp <= 0:
            return result

        if is_long:
            favorable_move = current_price - entry_price
        else:
            favorable_move = entry_price - current_price

        tp_progress = favorable_move / entry_to_tp  # <0 = losing, 0 = flat, 1 = at TP
        result["tp_progress"] = round(tp_progress, 4)

        # ── Activation threshold ─────────────────────────────────────────
        # Must exceed BOTH:
        #   a) configured % of TP distance (default 40%)
        #   b) actual fee breakeven + margin
        # This ensures we never move to "breakeven" at a price that would
        # still result in a net loss after fees.
        be_activation_pct = getattr(config, "TRAILING_BE_ACTIVATION_PCT", 0.40)
        activation_move = max(
            entry_to_tp * be_activation_pct,
            fee_breakeven_with_margin,
        )
        activation_progress = activation_move / entry_to_tp

        lock_tp_pct = getattr(config, "TRAILING_LOCK_TP_PCT", 0.75)
        atr_mult = getattr(config, "TRAILING_ATR_MULTIPLIER", 3.0)
        lock_atr_mult = getattr(config, "TRAILING_LOCK_ATR_MULTIPLIER", 1.5)
        min_step = getattr(config, "TRAILING_MIN_STEP_TICKS", 0.5)

        if favorable_move < activation_move:
            # ── INITIAL: haven't moved past fee breakeven yet ────────────
            result["phase"] = "INITIAL"
            elog.log(
                "ENGINE_TRAIL",
                phase="INITIAL",
                bars=bars_held,
                progress=round(tp_progress, 4),
                favorable_move=round(favorable_move, 2),
                activation_move=round(activation_move, 2),
                fee_be=round(fee_breakeven_move, 2),
                fee_be_margin=round(fee_breakeven_with_margin, 2),
                atr=round(atr, 2),
            )
            return result

        # ── Past activation — compute breakeven floor from actual fees ───
        # The SL floor guarantees: if SL is hit, gross PnL >= total fees.
        # floor = entry + fee_breakeven_with_margin (long)
        # floor = entry - fee_breakeven_with_margin (short)
        be_floor = fee_breakeven_with_margin

        if tp_progress >= lock_tp_pct:
            # Phase LOCK: close to TP, tighten trail
            trail_distance = atr * lock_atr_mult
            result["phase"] = "LOCK"
        elif tp_progress >= activation_progress:
            # Phase TRAILING: standard wide trail
            trail_distance = atr * atr_mult
            result["phase"] = "TRAILING"
        else:
            # Phase BREAKEVEN: just entered the zone
            trail_distance = favorable_move
            result["phase"] = "BREAKEVEN"

        result["trail_dist"] = round(trail_distance, 2)

        # ── Compute candidate SL ─────────────────────────────────────────
        if is_long:
            # Trail below current price, floor at entry + fee breakeven
            trail_sl = current_price - trail_distance
            floor_sl = entry_price + be_floor
            candidate_sl = max(trail_sl, floor_sl)

            # Ratchet: only move UP
            if candidate_sl > current_sl + min_step:
                result["new_sl"] = round(candidate_sl, 1)
        else:
            # Trail above current price, ceiling at entry - fee breakeven
            trail_sl = current_price + trail_distance
            ceiling_sl = entry_price - be_floor
            candidate_sl = min(trail_sl, ceiling_sl)

            # Ratchet: only move DOWN
            if candidate_sl < current_sl - min_step:
                result["new_sl"] = round(candidate_sl, 1)

        elog.log(
            "ENGINE_TRAIL",
            phase=result["phase"],
            bars=bars_held,
            progress=round(tp_progress, 4),
            atr=round(atr, 2),
            trail_dist=round(trail_distance, 2),
            current_sl=round(current_sl, 1),
            new_sl=result["new_sl"],
            candidate=round(candidate_sl, 1) if candidate_sl else None,
            fee_be=round(fee_breakeven_move, 2),
            fee_be_margin=round(fee_breakeven_with_margin, 2),
            total_fees=round(total_round_trip_fees, 4),
        )

        return result

    def reset(self):
        self._state = EngineState()
        self._bars_since_kde_build = 0
        elog.log("SYSTEM_START", component="HPMSEngine", event="state_reset")
