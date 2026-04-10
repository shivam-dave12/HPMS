"""
hpms_engine.py — Hamiltonian Phase-Space Micro-Scalping Engine
================================================================
Core physics engine implementing:
  1. Takens embedding theorem for phase-space reconstruction
  2. Kernel density estimation for potential energy landscape V(q)
  3. Hamilton's equations of motion: dq/dt = p/m, dp/dt = -dV/dq
  4. 4th-order Runge-Kutta / Symplectic Leapfrog / Euler integration
  5. Energy conservation monitoring (dH/dt)
  6. Trajectory prediction and signal classification

All computations are pure NumPy — no heavy ML deps, sub-millisecond per tick.

DECISION TRACING
----------------
Every bar emits a sequence of structured JSON events so you can replay
exactly why the engine accepted or rejected a trade:

  ENGINE_PHASE_STATE   — q, p, H, K, V, dH/dt for this bar
  ENGINE_KDE_REBUILD   — when the potential landscape is rebuilt
  ENGINE_TRAJECTORY    — start/end of predicted trajectory
  ENGINE_CRITERIA      — each signal gate: value, threshold, pass/fail
  ENGINE_SIGNAL        — final decision with reason
  ENGINE_SKIP          — when signal is FLAT, with the blocking reason
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


@dataclass(slots=True)
class PhaseState:
    """Instantaneous state in reconstructed phase space."""
    q: float
    p: float
    H: float
    kinetic: float
    potential: float
    dH_dt: float
    timestamp: float


@dataclass(slots=True)
class TrajectoryPoint:
    """Single point on predicted micro-trajectory."""
    t: float
    q: float
    p: float
    H: float


@dataclass(slots=True)
class HPMSSignal:
    """Complete signal output from the engine."""
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


@dataclass
class EngineState:
    """Persistent state across ticks."""
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
# POTENTIAL LANDSCAPE V(q) — KERNEL DENSITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

class PotentialLandscape:
    """
    Builds V(q) from recent price histogram via kernel density estimation.

    Low V(q)  = high-density region = "valley" (price dwells here = liquidity).
    High V(q) = low-density region  = "hill"   (price rolls away from here).

    V(q) = -log(KDE(q)) + const
    dV/dq computed via central finite differences on the KDE grid.
    """

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
# INTEGRATORS — HAMILTON'S EQUATIONS
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
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSEngine:
    """
    Hamiltonian Phase-Space Micro-Scalping Engine.

    On each new 1m bar close:
      1. Update phase-space coordinates (q, p) via Takens embedding
      2. Rebuild potential landscape V(q) from recent q history
      3. Compute current Hamiltonian H = p^2/2m + V(q)
      4. Integrate Hamilton's equations forward N bars
      5. Evaluate signal criteria and emit HPMSSignal

    Every step is logged via elog so you can see exactly what the engine
    is computing and why it accepts or rejects a trade.
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
        delta_q_threshold:    float = 0.0022,
        dH_dt_max:            float = 0.05,
        H_percentile:         float = 99.0,
        min_momentum:         float = 0.0001,
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
        self._acceleration_check   = acceleration_check
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

    def _compute_q(self, closes: np.ndarray) -> np.ndarray:
        need = self._norm_window + self._lookback + self._tau + 10
        if len(closes) > need:
            closes = closes[-need:]
        log_c = np.log(closes)
        n = min(self._norm_window, len(log_c))
        window = log_c[-n:]

        # Exponentially-weighted mean/std so the baseline tracks current price
        # in trends.  Flat 120-bar mean lags by ~60 bars; during a $674 rally
        # that pushes q to +5.5σ, making every bar look "extreme" and triggering
        # a mean-reversion SHORT.  Half-life = window/4 gives ~4× weight to the
        # newest bar vs the oldest, keeping μ close to current price.
        half_life = max(n / 4.0, 1.0)
        decay = np.exp(-np.log(2.0) / half_life)
        # Geometric weights oldest→newest: decay^(n-1), …, decay^0
        exp_weights = decay ** np.arange(n - 1, -1, -1, dtype=np.float64)
        exp_weights /= exp_weights.sum()

        mu  = float(np.dot(exp_weights, window))
        var = float(np.dot(exp_weights, (window - mu) ** 2))
        std = math.sqrt(var) if var > 1e-24 else 1e-12

        return (log_c - mu) / std

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
        """
        Process a new 1-minute bar close.

        Every intermediate calculation is logged via elog so you can trace
        exactly what the engine is "thinking" on every bar.
        """
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
        q_series = self._compute_q(closes_arr)
        p_series = self._compute_p(q_series)

        q_now = float(q_series[-1])
        p_now = float(p_series[-1])

        if np.isnan(p_now):
            elog.log("ENGINE_SKIP", reason="NAN_MOMENTUM", q=q_now)
            return None

        # ── Step 2: Build / refresh potential landscape V(q) ──────────────────
        self._bars_since_kde_build += 1
        kde_rebuilt = False

        if self._bars_since_kde_build >= self._kde_rebuild_interval or not self._landscape.is_built:
            q_lookback = q_series[-self._lookback:]
            v_lookback = volumes_arr[-self._lookback:]

            valid_mask = ~np.isnan(q_lookback)
            q_lookback = q_lookback[valid_mask]
            v_lookback = v_lookback[valid_mask] if len(v_lookback) == len(valid_mask) else None

            v_weights = None
            if v_lookback is not None and len(v_lookback) == len(q_lookback):
                v_sum = np.sum(v_lookback)
                if v_sum > 0:
                    v_weights = v_lookback / v_sum

            # ── Recency weights for KDE (Fix 2) ──────────────────────────────
            # Without this, V(q) is centred on old price levels: the "valley"
            # of high density sits far below current price, so every trajectory
            # rolls back toward past prices → chronic SHORT bias in uptrends.
            # Exponential decay (oldest weight ≈ exp(-2) ≈ 0.135, newest = 1.0)
            # shifts the density peak to recent bars, centering V(q) on current
            # price so the trajectory can predict continuation as well as reversal.
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
                     volume_weighted=(v_weights is not None),
                     q_min=round(float(np.min(q_lookback)), 4),
                     q_max=round(float(np.max(q_lookback)), 4),
                     q_range=round(float(np.max(q_lookback) - np.min(q_lookback)), 4))

        # ── Step 3: Current Hamiltonian ───────────────────────────────────────
        H_now, K_now, V_now = self._compute_H(q_now, p_now)
        dV_at_q = self._landscape.dV_dq(q_now)

        # ── EMA-smoothed H for stable dH/dt ──────────────────────────────────
        #
        # KDE REBASE (Bug fix):
        # Every kde_rebuild_interval bars, V(q) is reconstructed from scratch.
        # H = p²/2m + V(q) therefore jumps discontinuously.  If we feed that
        # jump straight into the EMA, the smoothed dH/dt = H_ema[t] - H_ema[t-1]
        # spikes far above SIGNAL_DH_DT_MAX for several bars, setting
        # energy_conserved=False and blocking every signal on rebuild bars.
        #
        # Fix: when a rebuild just happened, overwrite the previously stored EMA
        # with H_now before computing the new value.  This makes the baseline
        # match the new landscape so dH/dt on the rebuild bar measures only the
        # within-bar market-driven energy change, not the cross-landscape jump.
        alpha = 2.0 / (self._H_ema_span + 1.0)
        if self._state.H_ema_history:
            if kde_rebuilt:
                # Rebase: align the stored EMA baseline to the new landscape.
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

        # H_history and H_ema_history are always appended together, so their
        # lengths are identical.  A single guard suffices for both.
        if len(self._state.H_ema_history) < 2:
            elog.log("ENGINE_SKIP", reason="WARMING_UP",
                     history_len=len(self._state.H_ema_history))
            return None

        # len >= 2 is guaranteed by the guard above — [-2] access is safe.
        dH_dt     = H_ema_now - self._state.H_ema_history[-2]
        dH_dt_raw = H_now     - self._state.H_history[-2]

        # ── LOG: Phase state snapshot ─────────────────────────────────────────
        # This is the "what is the engine currently seeing" log.
        elog.log("ENGINE_PHASE_STATE",
                 bar=self._state.signal_count + 1,
                 price=round(float(closes_arr[-1]), 2),
                 q=round(q_now, 6),
                 p=round(p_now, 6),
                 H=round(H_now, 6),
                 H_ema=round(H_ema_now, 6),
                 K=round(K_now, 6),
                 V=round(V_now, 6),
                 dV_dq=round(dV_at_q, 6),
                 dH_dt_ema=round(dH_dt, 6),
                 dH_dt_raw=round(dH_dt_raw, 6),
                 kde_rebuilt=kde_rebuilt)

        # ── Step 4: Chaos regime filter ───────────────────────────────────────
        H_arr       = np.array(self._state.H_history)
        H_threshold = float(np.percentile(np.abs(H_arr), self._H_percentile))
        chaos_ok    = abs(H_now) <= H_threshold

        elog.log("ENGINE_CRITERIA",
                 check="chaos_filter",
                 H_abs=round(abs(H_now), 6),
                 H_threshold=round(H_threshold, 6),
                 percentile=self._H_percentile,
                 pass_=chaos_ok)

        if not chaos_ok:
            reason = "CHAOTIC: |H|=" + str(round(abs(H_now), 4)) + " > " + str(round(H_threshold, 4))
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
            )

        # ── Step 5: Forward integration ───────────────────────────────────────
        trajectory = self._integrate_forward(q_now, p_now)

        self._state.trajectory_log.append(trajectory)
        if len(self._state.trajectory_log) > self._trajectory_log_depth:
            del self._state.trajectory_log[:-self._trajectory_log_depth]

        q_pred  = trajectory[-1].q
        p_pred  = trajectory[-1].p
        delta_q = q_pred - q_now

        # Convert z-score delta_q back to approximate % price move
        log_closes = np.log(closes_arr[-self._norm_window:])
        std_log    = float(np.std(log_closes))
        if std_log < 1e-12:
            std_log = 1e-12
        predicted_pct_move = delta_q * std_log

        elog.log("ENGINE_TRAJECTORY",
                 q_start=round(q_now, 6),
                 q_pred=round(q_pred, 6),
                 p_start=round(p_now, 6),
                 p_pred=round(p_pred, 6),
                 delta_q_zscale=round(delta_q, 6),
                 predicted_pct_move=round(predicted_pct_move, 6),
                 std_log=round(std_log, 8),
                 horizon_bars=self._horizon)

        # ── Step 6: Evaluate each signal criterion ────────────────────────────
        # Every gate is logged individually so you can see exactly which one
        # is blocking trades when the system stays flat.

        # ── Adaptive delta_q threshold (Fix 3a) ──────────────────────────────
        # A fixed 0.0006 threshold means ~0.06% move; on a quiet 1m bar that's
        # fine, but on high-volatility bars it's trivially met by noise.  Scale
        # with current realized volatility: require at least 0.3σ of log-return
        # std, floored at half the config value so it never goes below the
        # operator's intent.
        effective_delta_q_threshold = max(
            self._delta_q_threshold * 0.5,
            std_log * 0.3,
        )

        # (a) Magnitude of predicted move
        delta_q_ok   = abs(predicted_pct_move) > effective_delta_q_threshold
        delta_q_dir  = "LONG" if predicted_pct_move > 0 else "SHORT"
        elog.log("ENGINE_CRITERIA",
                 check="delta_q_magnitude",
                 predicted_pct=round(predicted_pct_move, 6),
                 threshold=effective_delta_q_threshold,
                 config_threshold=self._delta_q_threshold,
                 std_log=round(std_log, 8),
                 direction=delta_q_dir,
                 pass_=delta_q_ok)

        # (b) Energy conservation (EMA-smoothed dH/dt)
        energy_conserved = abs(dH_dt) < self._dH_dt_max
        elog.log("ENGINE_CRITERIA",
                 check="energy_conservation",
                 dH_dt_ema=round(abs(dH_dt), 6),
                 dH_dt_raw=round(abs(dH_dt_raw), 6),
                 threshold=self._dH_dt_max,
                 pass_=energy_conserved)

        # (c) Residual momentum at prediction horizon
        momentum_ok = abs(p_pred) > self._min_momentum
        elog.log("ENGINE_CRITERIA",
                 check="min_momentum",
                 p_pred=round(abs(p_pred), 6),
                 threshold=self._min_momentum,
                 pass_=momentum_ok)

        # (d) Downhill acceleration in V(q)
        # LONG: V slopes down ahead (dV/dq < 0) AND final momentum positive
        # SHORT: V slopes up ahead (dV/dq > 0) AND final momentum negative
        accel_long  = (dV_at_q < 0 and p_pred > 0) if self._acceleration_check else True
        accel_short = (dV_at_q > 0 and p_pred < 0) if self._acceleration_check else True
        elog.log("ENGINE_CRITERIA",
                 check="acceleration",
                 dV_dq=round(dV_at_q, 6),
                 p_pred=round(p_pred, 6),
                 accel_long_ok=accel_long,
                 accel_short_ok=accel_short,
                 check_enabled=self._acceleration_check)

        # ── Step 7: Signal classification ─────────────────────────────────────
        current_price = float(closes_arr[-1])
        signal_type   = SignalType.FLAT
        reason        = "NO_SIGNAL"

        # CRITICAL FIX: Momentum SIGN must agree with predicted direction.
        # Previously only abs(p_pred) was checked, ignoring sign.
        # This caused 56% of signals to have contradictory momentum:
        #   e.g. SHORT signal with p_pred=+2.55 (momentum heading UP).
        # Such signals trade INTO a reversal — the trajectory overshoots
        # through a potential well and bounces back. Net displacement is
        # negative but the particle is heading back up at horizon end.
        # That's the worst possible entry for a short.
        momentum_direction_long  = p_pred > self._min_momentum
        momentum_direction_short = p_pred < -self._min_momentum

        # Trajectory quality: check if the trajectory is monotonic
        # in the predicted direction over the second half of the horizon.
        # If the trajectory reverses in the final bars, the signal is unreliable.
        traj_consistent = True
        if len(trajectory) >= 3:
            mid = len(trajectory) // 2
            second_half_dq = trajectory[-1].q - trajectory[mid].q
            # For a LONG, second half should move UP (dq > 0)
            # For a SHORT, second half should move DOWN (dq < 0)
            if predicted_pct_move > 0 and second_half_dq < 0:
                traj_consistent = False
            elif predicted_pct_move < 0 and second_half_dq > 0:
                traj_consistent = False

        elog.log("ENGINE_CRITERIA",
                 check="momentum_direction",
                 p_pred=round(p_pred, 6),
                 momentum_long_ok=momentum_direction_long,
                 momentum_short_ok=momentum_direction_short,
                 traj_consistent=traj_consistent)

        # ── p_now trend-alignment gate (Fix 3b) ──────────────────────────────
        # The trajectory can predict a LONG even when current momentum p_now is
        # negative (i.e. price is actively falling NOW).  This happens when V(q)
        # has a valley ahead: the particle will decelerate, reverse, and end up
        # higher — but only after passing through a losing region first.
        # Gate: current momentum must agree with signal direction.
        p_now_long_ok  = p_now > 0
        p_now_short_ok = p_now < 0
        elog.log("ENGINE_CRITERIA",
                 check="p_now_direction",
                 p_now=round(p_now, 6),
                 long_ok=p_now_long_ok,
                 short_ok=p_now_short_ok)

        long_criteria  = (predicted_pct_move > effective_delta_q_threshold
                          and energy_conserved
                          and momentum_direction_long   # SIGN check, not just magnitude
                          and p_now_long_ok             # current bar momentum agrees
                          and traj_consistent
                          and accel_long)
        short_criteria = (predicted_pct_move < -effective_delta_q_threshold
                          and energy_conserved
                          and momentum_direction_short  # SIGN check, not just magnitude
                          and p_now_short_ok            # current bar momentum agrees
                          and traj_consistent
                          and accel_short)

        if long_criteria:
            signal_type = SignalType.LONG
            reason = (
                "LONG:"
                " dq=" + str(round(predicted_pct_move, 5)) +
                " dH_ema=" + str(round(abs(dH_dt), 5)) +
                " p_f=" + str(round(p_pred, 5)) +
                " dVdq=" + str(round(dV_at_q, 5))
            )

        elif short_criteria:
            signal_type = SignalType.SHORT
            reason = (
                "SHORT:"
                " dq=" + str(round(predicted_pct_move, 5)) +
                " dH_ema=" + str(round(abs(dH_dt), 5)) +
                " p_f=" + str(round(p_pred, 5)) +
                " dVdq=" + str(round(dV_at_q, 5))
            )

        else:
            # Build a complete list of blocking reasons
            blockers = []
            if not delta_q_ok:
                blockers.append(
                    "delta_q=" + str(round(abs(predicted_pct_move), 5)) +
                    "<" + str(round(effective_delta_q_threshold, 6))
                )
            if not energy_conserved:
                blockers.append(
                    "dH_dt=" + str(round(abs(dH_dt), 5)) +
                    ">" + str(self._dH_dt_max)
                )
            # p_now direction gate
            if predicted_pct_move > 0 and not p_now_long_ok:
                blockers.append(
                    "p_now=" + str(round(p_now, 5)) +
                    " contradicts LONG (current momentum is DOWN)"
                )
            elif predicted_pct_move < 0 and not p_now_short_ok:
                blockers.append(
                    "p_now=" + str(round(p_now, 5)) +
                    " contradicts SHORT (current momentum is UP)"
                )
            # Momentum direction check — the most common blocker for bad signals
            if predicted_pct_move > 0 and not momentum_direction_long:
                blockers.append(
                    "p_pred=" + str(round(p_pred, 5)) +
                    " contradicts LONG (need >0)"
                )
            elif predicted_pct_move < 0 and not momentum_direction_short:
                blockers.append(
                    "p_pred=" + str(round(p_pred, 5)) +
                    " contradicts SHORT (need <0)"
                )
            if not traj_consistent:
                blockers.append(
                    "trajectory_reversal (2nd half contradicts direction)"
                )
            if self._acceleration_check:
                if predicted_pct_move > 0 and not accel_long:
                    blockers.append(
                        "accel_LONG_fail: dV/dq=" + str(round(dV_at_q, 5)) +
                        " p_f=" + str(round(p_pred, 5))
                    )
                elif predicted_pct_move < 0 and not accel_short:
                    blockers.append(
                        "accel_SHORT_fail: dV/dq=" + str(round(dV_at_q, 5)) +
                        " p_f=" + str(round(p_pred, 5))
                    )
            reason = "FLAT: " + (", ".join(blockers) if blockers else "no criteria met")

        # ── LOG: Final decision ───────────────────────────────────────────────
        if signal_type != SignalType.FLAT:
            elog.log("ENGINE_SIGNAL",
                     signal=signal_type.name,
                     reason=reason,
                     predicted_pct=round(predicted_pct_move, 6),
                     price=current_price,
                     long_criteria_met=long_criteria,
                     short_criteria_met=short_criteria)
        else:
            elog.log("ENGINE_SKIP",
                     reason=reason,
                     predicted_pct=round(predicted_pct_move, 6),
                     delta_q_ok=delta_q_ok,
                     energy_ok=energy_conserved,
                     momentum_dir_long=momentum_direction_long,
                     momentum_dir_short=momentum_direction_short,
                     traj_consistent=traj_consistent,
                     accel_long_ok=accel_long,
                     accel_short_ok=accel_short)

        # ── Step 8: TP / SL prices (ATR-based) ────────────────────────────────
        # Fixed % TP/SL was unreachable: 0.35% = $252 at $72k, but 8-bar
        # range is only ~$46. Use actual recent volatility instead.
        if signal_type != SignalType.FLAT and len(closes_arr) >= 15:
            # Compute ATR from last 14 bars (actual high-low would be better
            # but we only have closes — use close-to-close range as proxy)
            recent = closes_arr[-15:]
            bar_ranges = np.abs(np.diff(recent))
            atr_1bar = float(np.mean(bar_ranges)) if len(bar_ranges) > 0 else current_price * 0.0005

            # TP = 2.0 × ATR × prediction_horizon bars — achievable target
            # SL = 1.2 × ATR × sqrt(prediction_horizon) — tighter risk
            tp_atr_mult = 2.0
            sl_atr_mult = 1.2
            tp_distance = atr_1bar * tp_atr_mult * self._horizon
            sl_distance = atr_1bar * sl_atr_mult * math.sqrt(self._horizon)

            # Floor: at least 0.01% of price
            tp_distance = max(tp_distance, current_price * 0.0001)
            sl_distance = max(sl_distance, current_price * 0.0001)

            # Cap: respect config max TP/SL %
            tp_distance = min(tp_distance, current_price * self._tp_pct)
            sl_distance = min(sl_distance, current_price * self._sl_pct)

            if signal_type == SignalType.LONG:
                tp_price = current_price + tp_distance
                sl_price = current_price - sl_distance
            else:
                tp_price = current_price - tp_distance
                sl_price = current_price + sl_distance
        elif signal_type != SignalType.FLAT:
            # Fallback if not enough data for ATR
            if signal_type == SignalType.LONG:
                tp_price = current_price * (1.0 + self._tp_pct)
                sl_price = current_price * (1.0 - self._sl_pct)
            else:
                tp_price = current_price * (1.0 - self._tp_pct)
                sl_price = current_price * (1.0 + self._sl_pct)
        else:
            tp_price = 0.0
            sl_price = 0.0

        # ── Confidence score ──────────────────────────────────────────────────
        if signal_type != SignalType.FLAT:
            c1 = min(1.0, abs(predicted_pct_move) / (self._delta_q_threshold * 3))
            c2 = max(0.0, 1.0 - abs(dH_dt) / self._dH_dt_max)
            c3 = min(1.0, abs(p_pred) / (self._min_momentum * 10))
            confidence = c1 * 0.4 + c2 * 0.35 + c3 * 0.25
        else:
            confidence = 0.0

        t_end = time.perf_counter_ns()
        compute_us = (t_end - t_start) / 1000.0

        self._state.signal_count += 1

        # ── INFO-level bar summary — always visible, no DEBUG needed ─────────
        bar_n = self._state.signal_count
        pct   = lambda v: f"{v*100:+.4f}%"  # noqa: E731

        if signal_type != SignalType.FLAT:
            logger.info(
                f"▶ bar#{bar_n:4d}  ${current_price:,.1f}  "
                f"Δq={pct(predicted_pct_move)}  "
                f"|dH/dt|={abs(dH_dt):.5f}  |p|={abs(p_pred):.6f}  "
                f"→ {signal_type.name}  conf={confidence:.1%}  "
                f"TP=${tp_price:,.1f}  SL=${sl_price:,.1f}  [{compute_us:.0f}µs]"
            )
        else:
            dq_pct  = abs(predicted_pct_move) / effective_delta_q_threshold * 100 if effective_delta_q_threshold else 0
            dh_pct  = abs(dH_dt) / self._dH_dt_max * 100 if self._dH_dt_max else 0
            mom_pct = abs(p_pred) / self._min_momentum * 100 if self._min_momentum else 0
            g = lambda ok: "✓" if ok else "✗"  # noqa: E731
            blockers = []
            if not delta_q_ok:
                blockers.append(f"{g(False)}Δq={pct(predicted_pct_move)}({dq_pct:.0f}%/thr)")
            if not energy_conserved:
                blockers.append(f"{g(False)}dH={abs(dH_dt):.5f}({dh_pct:.0f}%/max)")
            # p_now direction
            if predicted_pct_move > 0 and not p_now_long_ok:
                blockers.append(f"{g(False)}p_now={p_now:+.5f}(↓,need↑)")
            elif predicted_pct_move < 0 and not p_now_short_ok:
                blockers.append(f"{g(False)}p_now={p_now:+.5f}(↑,need↓)")
            # Momentum direction check
            if predicted_pct_move > 0 and not momentum_direction_long:
                blockers.append(f"{g(False)}p_dir={p_pred:+.5f}(need>0)")
            elif predicted_pct_move < 0 and not momentum_direction_short:
                blockers.append(f"{g(False)}p_dir={p_pred:+.5f}(need<0)")
            if not traj_consistent:
                blockers.append(f"{g(False)}traj_reversal")
            if self._acceleration_check:
                if predicted_pct_move > 0 and not accel_long:
                    blockers.append(f"✗accel_L(dVdq={dV_at_q:+.5f},p={p_pred:+.5f})")
                elif predicted_pct_move < 0 and not accel_short:
                    blockers.append(f"✗accel_S(dVdq={dV_at_q:+.5f},p={p_pred:+.5f})")
            blocking_str = "  ".join(blockers) if blockers else "no criteria met"
            logger.info(
                f"  bar#{bar_n:4d}  ${current_price:,.1f}  "
                f"q={q_now:+.4f} p={p_now:+.5f} H={H_now:.4f} "
                f"dH/dt={dH_dt:+.5f} KDE={'OK' if self._landscape.is_built else 'NO'}  "
                f"FLAT {blocking_str}"
            )

        return HPMSSignal(
            signal_type=signal_type,
            confidence=confidence,
            predicted_delta_q=predicted_pct_move,
            predicted_p_final=p_pred,
            current_H=H_now,
            dH_dt=dH_dt,
            trajectory=trajectory,
            entry_price=current_price,
            tp_price=tp_price,
            sl_price=sl_price,
            reason=reason,
            compute_time_us=compute_us,
            bar_timestamp=timestamp,
            bars_since_kde=self._bars_since_kde_build,
        )

    # ─── DIAGNOSTICS ──────────────────────────────────────────────────────────

    def get_phase_state(self) -> Optional[PhaseState]:
        if not self._state.q_history:
            return None
        q = self._state.q_history[-1]
        p = self._state.p_history[-1]
        H, K, V = self._compute_H(q, p)
        if len(self._state.H_ema_history) >= 2:
            dH = self._state.H_ema_history[-1] - self._state.H_ema_history[-2]
        else:
            dH = 0.0
        return PhaseState(
            q=q, p=p, H=H, kinetic=K, potential=V, dH_dt=dH,
            timestamp=self._state.timestamps[-1] if self._state.timestamps else 0.0,
        )

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
                "q":         ps.q       if ps else None,
                "p":         ps.p       if ps else None,
                "H":         ps.H       if ps else None,
                "K":         ps.kinetic if ps else None,
                "V":         ps.potential if ps else None,
                "dH_dt_ema": ps.dH_dt   if ps else None,
            },
            "H_smoothing": {
                "H_raw":   round(H_raw,  6) if H_raw  is not None else None,
                "H_ema":   round(H_ema,  6) if H_ema  is not None else None,
                "dH_raw":  round(dH_raw, 6) if dH_raw is not None else None,
                "dH_ema":  round(dH_ema, 6) if dH_ema is not None else None,
                "ema_span": self._H_ema_span,
            },
            "landscape_built":      self._landscape.is_built,
            "bars_since_kde_build": self._bars_since_kde_build,
            "kde_rebuild_interval": self._kde_rebuild_interval,
            "history_len":          len(self._state.close_history),
            "signal_count":         self._state.signal_count,
            "trajectory_log_depth": len(self._state.trajectory_log),
            "last_trajectory":      traj_summary,
            "params":               self.get_params(),
        }

    def get_adaptive_dH_spike_threshold(self, multiplier: float = 3.0) -> float:
        """
        Compute an adaptive exit spike threshold based on recent dH/dt history.

        Returns multiplier * std(dH/dt) over the last normalization_window bars,
        floored at a minimum of 0.10 to avoid exiting on noise.
        """
        if len(self._state.H_ema_history) < 10:
            return 0.15  # fallback default
        dH_series = []
        ema_hist = self._state.H_ema_history
        for i in range(1, min(len(ema_hist), self._norm_window)):
            dH_series.append(abs(ema_hist[-i] - ema_hist[-i - 1]))
        if not dH_series:
            return 0.15
        arr = np.array(dH_series)
        threshold = float(np.mean(arr) + multiplier * np.std(arr))
        return max(threshold, 0.10)  # floor at 0.10

    def get_dynamic_max_hold(self, tp_distance: float, current_price: float) -> int:
        """
        Compute how many bars the trade should be allowed to live based
        on recent volatility and the TP distance.

        Logic: estimate bars needed to reach TP given recent 1-bar ATR.
        max_hold = ceil(tp_distance / atr_per_bar) + buffer
        Clamped to [3, 30] bars.
        """
        if not self._state.close_history or len(self._state.close_history) < 5:
            return 8  # default
        closes = np.array(self._state.close_history[-20:])
        atr_per_bar = float(np.mean(np.abs(np.diff(closes))))
        if atr_per_bar <= 0:
            return 8
        bars_to_tp = tp_distance / atr_per_bar
        # Add 50% buffer for non-directional drift
        max_hold = int(math.ceil(bars_to_tp * 1.5))
        return max(3, min(max_hold, 30))

    def reset(self):
        self._state = EngineState()
        self._bars_since_kde_build = 0
        elog.log("SYSTEM_START", component="HPMSEngine", event="state_reset")
