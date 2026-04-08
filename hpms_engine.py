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
    q: float          # normalised log-close ("position")
    p: float          # time-delay derivative ("momentum")
    H: float          # Hamiltonian (total energy)
    kinetic: float    # p²/2m
    potential: float  # V(q)
    dH_dt: float      # energy dissipation rate
    timestamp: float  # bar timestamp


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
    confidence:        float        # 0..1
    predicted_delta_q: float        # predicted % move
    predicted_p_final: float        # momentum at horizon
    current_H:         float        # current energy
    dH_dt:             float        # energy conservation metric
    trajectory:        List[TrajectoryPoint]
    entry_price:       float        # current close
    tp_price:          float        # predicted target
    sl_price:          float        # stop loss
    reason:            str          # human-readable reason
    compute_time_us:   float        # microseconds to compute
    bar_timestamp:     float


@dataclass
class EngineState:
    """Persistent state across ticks."""
    q_history:      List[float] = field(default_factory=list)
    p_history:      List[float] = field(default_factory=list)
    H_history:      List[float] = field(default_factory=list)
    H_ema_history:  List[float] = field(default_factory=list)  # EMA-smoothed H
    close_history:  List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)  # raw volumes
    timestamps:     List[float] = field(default_factory=list)
    trajectory_log: List[List[TrajectoryPoint]] = field(default_factory=list)  # last N trajectories
    signal_count:   int = 0
    last_signal_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# POTENTIAL LANDSCAPE V(q) — KERNEL DENSITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

class PotentialLandscape:
    """
    Builds V(q) from recent price histogram via kernel density estimation.

    Low V(q) = high-density region = "valley" where price dwells (liquidity).
    High V(q) = low-density region = "hill" that price rolls away from.

    V(q) = -log(KDE(q)) + const
    dV/dq computed via central finite differences on the KDE grid.
    """

    def __init__(
        self,
        bandwidth:   float = 0.3,
        grid_points: int   = 256,
    ):
        self._bandwidth   = bandwidth
        self._grid_points = grid_points
        self._q_grid:     Optional[np.ndarray] = None
        self._V_grid:     Optional[np.ndarray] = None
        self._dV_grid:    Optional[np.ndarray] = None
        self._q_min:      float = 0.0
        self._q_max:      float = 0.0
        self._is_built:   bool  = False

    def build(self, q_samples: np.ndarray, weights: Optional[np.ndarray] = None) -> bool:
        """
        Construct V(q) from recent q values, optionally volume-weighted.

        When weights (normalised bar volumes) are supplied the KDE treats
        high-volume bars as repeated observations — their q-values pull the
        density estimate harder, so "valleys" form where *traded* liquidity
        actually concentrates rather than where price merely passed through.

        Parameters
        ----------
        q_samples : array of q values (lookback window)
        weights   : array of same length, typically normalised volumes.
                    Passed directly to scipy.stats.gaussian_kde as `weights`.

        Returns True if landscape was built successfully.
        """
        if len(q_samples) < 10:
            return False

        try:
            self._q_min = float(np.min(q_samples)) - 1.0
            self._q_max = float(np.max(q_samples)) + 1.0
            self._q_grid = np.linspace(self._q_min, self._q_max, self._grid_points)

            # Volume-weighted KDE: scipy gaussian_kde accepts a `weights`
            # kwarg (since scipy 1.2). Weights must be non-negative and
            # are normalised internally so they sum to 1.
            kde = gaussian_kde(q_samples, bw_method=self._bandwidth,
                               weights=weights)
            density = kde(self._q_grid)

            # V(q) = -log(density) — potential is negative log-likelihood
            # Clamp density to avoid log(0)
            density = np.maximum(density, 1e-12)
            self._V_grid = -np.log(density)

            # Normalise V so min = 0
            self._V_grid -= np.min(self._V_grid)

            # dV/dq via central differences
            dq = self._q_grid[1] - self._q_grid[0]
            self._dV_grid = np.gradient(self._V_grid, dq)

            self._is_built = True
            return True

        except Exception as e:
            logger.error(f"PotentialLandscape.build failed: {e}")
            self._is_built = False
            return False

    def V(self, q: float) -> float:
        """Evaluate V at arbitrary q via linear interpolation."""
        if not self._is_built:
            return 0.0
        return float(np.interp(q, self._q_grid, self._V_grid))

    def dV_dq(self, q: float) -> float:
        """Evaluate dV/dq at arbitrary q via linear interpolation."""
        if not self._is_built:
            return 0.0
        return float(np.interp(q, self._q_grid, self._dV_grid))

    @property
    def is_built(self) -> bool:
        return self._is_built


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATORS — HAMILTON'S EQUATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# dq/dt = ∂H/∂p = p / m
# dp/dt = -∂H/∂q = -dV/dq

def _rk4_step(
    q: float, p: float, dt: float, m: float,
    dV_dq_func,
) -> Tuple[float, float]:
    """4th-order Runge-Kutta integrator for Hamilton's equations."""
    # k1
    dq1 = p / m
    dp1 = -dV_dq_func(q)

    # k2
    q2 = q + 0.5 * dt * dq1
    p2 = p + 0.5 * dt * dp1
    dq2 = p2 / m
    dp2 = -dV_dq_func(q2)

    # k3
    q3 = q + 0.5 * dt * dq2
    p3 = p + 0.5 * dt * dp2
    dq3 = p3 / m
    dp3 = -dV_dq_func(q3)

    # k4
    q4 = q + dt * dq3
    p4 = p + dt * dp3
    dq4 = p4 / m
    dp4 = -dV_dq_func(q4)

    q_new = q + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
    p_new = p + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
    return q_new, p_new


def _leapfrog_step(
    q: float, p: float, dt: float, m: float,
    dV_dq_func,
) -> Tuple[float, float]:
    """Symplectic Leapfrog (Störmer-Verlet) — exactly preserves phase-space volume."""
    p_half = p - 0.5 * dt * dV_dq_func(q)
    q_new  = q + dt * p_half / m
    p_new  = p_half - 0.5 * dt * dV_dq_func(q_new)
    return q_new, p_new


def _euler_step(
    q: float, p: float, dt: float, m: float,
    dV_dq_func,
) -> Tuple[float, float]:
    """Simple Euler (fast but non-symplectic — use for latency-critical paths only)."""
    q_new = q + dt * p / m
    p_new = p - dt * dV_dq_func(q)
    return q_new, p_new


_INTEGRATORS = {
    "rk4":      _rk4_step,
    "leapfrog": _leapfrog_step,
    "euler":    _euler_step,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HPMSEngine:
    """
    Hamiltonian Phase-Space Micro-Scalping Engine.

    On each new 1m bar close:
      1. Update phase-space coordinates (q, p) via Takens embedding
      2. Rebuild potential landscape V(q) from recent q history
      3. Compute current Hamiltonian H = p²/2m + V(q)
      4. Integrate Hamilton's equations forward N bars
      5. Evaluate signal criteria and emit HPMSSignal
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
        # Signal thresholds
        delta_q_threshold:    float = 0.0022,
        dH_dt_max:            float = 0.05,
        H_percentile:         float = 99.0,
        min_momentum:         float = 0.0001,
        acceleration_check:   bool  = True,
        # Trade params
        tp_pct:               float = 0.0035,
        sl_pct:               float = 0.0018,
        # Smoothing
        H_ema_span:           int   = 5,       # EMA span for dH/dt smoothing
        kde_rebuild_interval: int   = 3,       # rebuild V(q) every N bars (1 = every bar)
        trajectory_log_depth: int   = 20,      # keep last N trajectories for /diag
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

        self._landscape = PotentialLandscape(
            bandwidth=kde_bandwidth,
            grid_points=kde_grid_points,
        )
        self._integrator = _INTEGRATORS.get(integrator, _rk4_step)
        self._state = EngineState()

        logger.info(
            f"HPMSEngine initialized: tau={tau} lookback={lookback} "
            f"horizon={prediction_horizon} integrator={integrator} mass={mass}"
        )

    # ─── CONFIGURATION HOT-RELOAD ─────────────────────────────────────────────

    def update_param(self, key: str, value) -> bool:
        """Update a parameter at runtime (called from Telegram /set)."""
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
            logger.info(f"HPMS param updated: {key} = {casted}")
            return True
        except Exception as e:
            logger.error(f"Failed to set {key}={value}: {e}")
            return False

    def get_params(self) -> Dict:
        """Return all current engine parameters."""
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
        """
        q_t = z-score of log(close) over normalization window.
        This maps raw price into a stationary, centred coordinate.

        Only the trailing (norm_window + lookback + tau + margin) bars are
        used — earlier bars are irrelevant and discarding them avoids
        wasting cycles on z-scoring stale data.
        """
        # Trim to the minimum tail we actually need downstream
        need = self._norm_window + self._lookback + self._tau + 10
        if len(closes) > need:
            closes = closes[-need:]

        log_c = np.log(closes)
        n = min(self._norm_window, len(log_c))
        window = log_c[-n:]
        mu  = np.mean(window)
        std = np.std(window)
        if std < 1e-12:
            std = 1e-12
        return (log_c - mu) / std

    def _compute_p(self, q_series: np.ndarray) -> np.ndarray:
        """
        p_t = (q_t - q_{t-tau}) / tau — Takens time-delay derivative.
        Represents "momentum" in phase space.
        """
        tau = self._tau
        p = np.full_like(q_series, np.nan)
        if len(q_series) > tau:
            p[tau:] = (q_series[tau:] - q_series[:-tau]) / tau
        return p

    def _compute_H(self, q: float, p: float) -> Tuple[float, float, float]:
        """
        H(q,p) = p²/(2m) + V(q)
        Returns (H, kinetic, potential).
        """
        kinetic   = (p ** 2) / (2.0 * self._mass)
        potential = self._landscape.V(q)
        return kinetic + potential, kinetic, potential

    # ─── FORWARD INTEGRATION ──────────────────────────────────────────────────

    def _integrate_forward(self, q0: float, p0: float) -> List[TrajectoryPoint]:
        """
        Integrate Hamilton's equations forward `horizon` bars.
        Each bar is subdivided into steps of size dt.
        Returns the predicted trajectory.
        """
        trajectory = [TrajectoryPoint(t=0.0, q=q0, p=p0, H=self._compute_H(q0, p0)[0])]

        q, p = q0, p0
        steps_per_bar = max(1, int(1.0 / self._dt))

        for bar in range(1, self._horizon + 1):
            for _ in range(steps_per_bar):
                q, p = self._integrator(
                    q, p, self._dt, self._mass,
                    self._landscape.dV_dq,
                )
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

        Parameters
        ----------
        closes   : list of recent close prices (at least lookback + norm_window)
        volumes  : list of recent volumes (same length as closes) — used as
                   KDE weights so the potential landscape V(q) reflects where
                   *traded* liquidity concentrates, not merely where price
                   passed through.
        timestamp: bar close timestamp

        Returns
        -------
        HPMSSignal or None if no signal / insufficient data.
        """
        t_start = time.perf_counter_ns()

        closes_arr  = np.array(closes,  dtype=np.float64)
        volumes_arr = np.array(volumes, dtype=np.float64) if volumes else np.ones(len(closes_arr))
        # Ensure volumes_arr matches closes_arr length
        if len(volumes_arr) != len(closes_arr):
            volumes_arr = np.ones(len(closes_arr))

        min_required = max(self._norm_window, self._lookback) + self._tau + 5
        if len(closes_arr) < min_required:
            return None

        # ── Step 1: Phase-space coordinates ───────────────────────────────────
        q_series = self._compute_q(closes_arr)
        p_series = self._compute_p(q_series)

        # Current state
        q_now = float(q_series[-1])
        p_now = float(p_series[-1])
        if np.isnan(p_now):
            return None

        # ── Step 2: Build potential landscape V(q) — volume-weighted ─────────
        # Conditional rebuild: skip expensive KDE if interval not reached
        self._bars_since_kde_build += 1
        if self._bars_since_kde_build >= self._kde_rebuild_interval or not self._landscape.is_built:
            q_lookback = q_series[-self._lookback:]
            v_lookback = volumes_arr[-self._lookback:]

            # Align after NaN removal
            valid_mask = ~np.isnan(q_lookback)
            q_lookback = q_lookback[valid_mask]
            v_lookback = v_lookback[valid_mask] if len(v_lookback) == len(valid_mask) else None

            # Normalise volume weights: prevent zero-sum (scipy requirement)
            if v_lookback is not None and len(v_lookback) == len(q_lookback):
                v_sum = np.sum(v_lookback)
                if v_sum > 0:
                    v_weights = v_lookback / v_sum
                else:
                    v_weights = None
            else:
                v_weights = None

            if not self._landscape.build(q_lookback, weights=v_weights):
                return None
            self._bars_since_kde_build = 0

        # ── Step 3: Current Hamiltonian ───────────────────────────────────────
        H_now, K_now, V_now = self._compute_H(q_now, p_now)

        # ── EMA-smoothed H for stable dH/dt ──────────────────────────────────
        # Raw bar-to-bar dH/dt = H[t] - H[t-1] is extremely noisy on 1m bars.
        # An exponential moving average with span = H_ema_span filters
        # transient spikes while still reacting fast to genuine regime breaks.
        alpha = 2.0 / (self._H_ema_span + 1.0)
        if self._state.H_ema_history:
            H_ema_prev = self._state.H_ema_history[-1]
            H_ema_now  = alpha * H_now + (1.0 - alpha) * H_ema_prev
        else:
            H_ema_now = H_now

        self._state.H_history.append(H_now)
        self._state.H_ema_history.append(H_ema_now)
        self._state.q_history.append(q_now)
        self._state.p_history.append(p_now)
        self._state.close_history.append(float(closes_arr[-1]))
        self._state.volume_history.append(float(volumes_arr[-1]))
        self._state.timestamps.append(timestamp)

        # Trim histories
        max_hist = self._norm_window + 100
        for lst in (self._state.H_history, self._state.H_ema_history,
                    self._state.q_history, self._state.p_history,
                    self._state.close_history, self._state.volume_history,
                    self._state.timestamps):
            if len(lst) > max_hist:
                del lst[:-max_hist]

        if len(self._state.H_ema_history) < 2:
            return None

        # dH/dt from EMA-smoothed H — far fewer false energy-spike exits
        dH_dt = H_ema_now - self._state.H_ema_history[-2]

        # Also keep raw dH/dt for diagnostics
        dH_dt_raw = H_now - self._state.H_history[-2] if len(self._state.H_history) >= 2 else 0.0

        # ── Step 4: Chaos regime filter ───────────────────────────────────────
        H_arr = np.array(self._state.H_history)
        H_threshold = float(np.percentile(np.abs(H_arr), self._H_percentile))
        if abs(H_now) > H_threshold:
            return HPMSSignal(
                signal_type=SignalType.FLAT,
                confidence=0.0,
                predicted_delta_q=0.0,
                predicted_p_final=0.0,
                current_H=H_now,
                dH_dt=dH_dt,
                trajectory=[],
                entry_price=float(closes_arr[-1]),
                tp_price=0.0, sl_price=0.0,
                reason=f"CHAOTIC: |H|={abs(H_now):.4f} > {H_threshold:.4f} (p{self._H_percentile})",
                compute_time_us=0.0,
                bar_timestamp=timestamp,
            )

        # ── Step 5: Forward integration ───────────────────────────────────────
        trajectory = self._integrate_forward(q_now, p_now)

        # Log trajectory for /diag
        self._state.trajectory_log.append(trajectory)
        if len(self._state.trajectory_log) > self._trajectory_log_depth:
            del self._state.trajectory_log[:-self._trajectory_log_depth]

        q_pred = trajectory[-1].q
        p_pred = trajectory[-1].p
        delta_q = q_pred - q_now

        # Convert delta_q (z-score space) back to approximate % price move
        # delta_q * std(log_close) ≈ log-return
        log_closes = np.log(closes_arr[-self._norm_window:])
        std_log = float(np.std(log_closes))
        if std_log < 1e-12:
            std_log = 1e-12
        predicted_pct_move = delta_q * std_log

        # ── Step 6: Energy conservation check (uses EMA-smoothed dH/dt) ──────
        energy_conserved = abs(dH_dt) < self._dH_dt_max

        # ── Step 7: Acceleration check (rolling downhill in V) ────────────────
        dV_at_q = self._landscape.dV_dq(q_now)
        # For long: need dV/dq < 0 (potential slopes down ahead) AND p > 0
        # For short: need dV/dq > 0 AND p < 0
        accel_long  = (dV_at_q < 0 and p_pred > 0) if self._acceleration_check else True
        accel_short = (dV_at_q > 0 and p_pred < 0) if self._acceleration_check else True

        # ── Step 8: Signal classification ─────────────────────────────────────
        current_price = float(closes_arr[-1])
        signal_type = SignalType.FLAT
        reason = "NO_SIGNAL"

        if (predicted_pct_move > self._delta_q_threshold
                and energy_conserved
                and abs(p_pred) > self._min_momentum
                and accel_long):
            signal_type = SignalType.LONG
            reason = (f"LONG: Δq={predicted_pct_move:.5f} > {self._delta_q_threshold:.4f}, "
                      f"|dH/dt|={abs(dH_dt):.5f}(raw={abs(dH_dt_raw):.5f}), "
                      f"p_f={p_pred:.5f}, dV/dq={dV_at_q:.5f}")

        elif (predicted_pct_move < -self._delta_q_threshold
                and energy_conserved
                and abs(p_pred) > self._min_momentum
                and accel_short):
            signal_type = SignalType.SHORT
            reason = (f"SHORT: Δq={predicted_pct_move:.5f} < -{self._delta_q_threshold:.4f}, "
                      f"|dH/dt|={abs(dH_dt):.5f}(raw={abs(dH_dt_raw):.5f}), "
                      f"p_f={p_pred:.5f}, dV/dq={dV_at_q:.5f}")

        else:
            reasons = []
            if abs(predicted_pct_move) <= self._delta_q_threshold:
                reasons.append(f"|Δq|={abs(predicted_pct_move):.5f}≤thresh")
            if not energy_conserved:
                reasons.append(f"|dH/dt_ema|={abs(dH_dt):.5f}>max(raw={abs(dH_dt_raw):.5f})")
            if abs(p_pred) <= self._min_momentum:
                reasons.append(f"|p_f|={abs(p_pred):.5f}≤min")
            reason = "FLAT: " + ", ".join(reasons) if reasons else "FLAT: no criteria met"

        # ── Compute TP/SL ─────────────────────────────────────────────────────
        if signal_type == SignalType.LONG:
            tp_price = current_price * (1.0 + self._tp_pct)
            sl_price = current_price * (1.0 - self._sl_pct)
            # Use predicted target if tighter than fixed TP
            pred_target = current_price * math.exp(predicted_pct_move)
            if pred_target < tp_price:
                tp_price = pred_target
        elif signal_type == SignalType.SHORT:
            tp_price = current_price * (1.0 - self._tp_pct)
            sl_price = current_price * (1.0 + self._sl_pct)
            pred_target = current_price * math.exp(predicted_pct_move)
            if pred_target > tp_price:
                tp_price = pred_target
        else:
            tp_price = 0.0
            sl_price = 0.0

        # Confidence = normalised composite score
        if signal_type != SignalType.FLAT:
            c1 = min(1.0, abs(predicted_pct_move) / (self._delta_q_threshold * 3))
            c2 = max(0.0, 1.0 - abs(dH_dt) / self._dH_dt_max)
            c3 = min(1.0, abs(p_pred) / (self._min_momentum * 10))
            confidence = (c1 * 0.4 + c2 * 0.35 + c3 * 0.25)
        else:
            confidence = 0.0

        t_end = time.perf_counter_ns()
        compute_us = (t_end - t_start) / 1000.0

        self._state.signal_count += 1

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
        )

    # ─── DIAGNOSTICS ──────────────────────────────────────────────────────────

    def get_phase_state(self) -> Optional[PhaseState]:
        """Return the most recent phase-space state (dH_dt uses EMA-smoothed H)."""
        if not self._state.q_history:
            return None
        q = self._state.q_history[-1]
        p = self._state.p_history[-1]
        H, K, V = self._compute_H(q, p)
        # Use EMA-smoothed dH/dt for consistency with signal logic
        if len(self._state.H_ema_history) >= 2:
            dH = self._state.H_ema_history[-1] - self._state.H_ema_history[-2]
        else:
            dH = 0.0
        return PhaseState(
            q=q, p=p, H=H, kinetic=K, potential=V, dH_dt=dH,
            timestamp=self._state.timestamps[-1] if self._state.timestamps else 0.0,
        )

    def get_diagnostics(self) -> Dict:
        """Return engine diagnostics for Telegram /diag."""
        ps = self.get_phase_state()

        # Last trajectory summary
        traj_summary = None
        if self._state.trajectory_log:
            last_traj = self._state.trajectory_log[-1]
            traj_summary = [
                {"t": tp.t, "q": round(tp.q, 6), "p": round(tp.p, 6), "H": round(tp.H, 6)}
                for tp in last_traj
            ]

        # H raw vs EMA comparison
        H_raw = self._state.H_history[-1] if self._state.H_history else None
        H_ema = self._state.H_ema_history[-1] if self._state.H_ema_history else None
        dH_raw = (self._state.H_history[-1] - self._state.H_history[-2]
                  if len(self._state.H_history) >= 2 else None)
        dH_ema = (self._state.H_ema_history[-1] - self._state.H_ema_history[-2]
                  if len(self._state.H_ema_history) >= 2 else None)

        return {
            "phase_state": {
                "q": ps.q if ps else None,
                "p": ps.p if ps else None,
                "H": ps.H if ps else None,
                "K": ps.kinetic if ps else None,
                "V": ps.potential if ps else None,
                "dH_dt_ema": ps.dH_dt if ps else None,
            },
            "H_smoothing": {
                "H_raw":  round(H_raw, 6)  if H_raw  is not None else None,
                "H_ema":  round(H_ema, 6)  if H_ema  is not None else None,
                "dH_raw": round(dH_raw, 6) if dH_raw is not None else None,
                "dH_ema": round(dH_ema, 6) if dH_ema is not None else None,
                "ema_span": self._H_ema_span,
            },
            "landscape_built":       self._landscape.is_built,
            "bars_since_kde_build":  self._bars_since_kde_build,
            "kde_rebuild_interval":  self._kde_rebuild_interval,
            "history_len":           len(self._state.close_history),
            "signal_count":          self._state.signal_count,
            "trajectory_log_depth":  len(self._state.trajectory_log),
            "last_trajectory":       traj_summary,
            "params":                self.get_params(),
        }

    def reset(self):
        """Clear all state (after parameter change or restart)."""
        self._state = EngineState()
        self._bars_since_kde_build = 0
        logger.info("HPMSEngine state reset")
