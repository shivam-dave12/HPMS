"""
fibonacci.py — Advanced Fibonacci Level Engine
================================================
Institutional-grade Fibonacci TP/SL and trailing stop computation.

Architecture:
  1. Volume-weighted swing detection via fractal pivot identification
  2. Multi-scale Fibonacci retracement/extension from overlapping swing pairs
  3. Confluence zone scoring — clustered levels from different swings amplify
  4. ATR-adaptive validation — levels must exceed noise floor to be tradeable
  5. Fibonacci-level trailing — SL steps through Fib retracement of the
     developing move (entry → high watermark), with breakeven floor

Fibonacci ratios used (derived from the golden ratio φ = 1.618...):
  Retracement: 0.236, 0.382, 0.500, 0.618, 0.786
  Extension:   1.000, 1.272, 1.618, 2.000, 2.618

All computations are pure NumPy — sub-millisecond.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Golden ratio derived levels
FIB_RETRACEMENT_LEVELS = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
FIB_EXTENSION_LEVELS   = np.array([1.000, 1.272, 1.618, 2.000, 2.618])

# All levels combined for confluence analysis
_ALL_FIB_RATIOS = np.concatenate([FIB_RETRACEMENT_LEVELS, FIB_EXTENSION_LEVELS])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SwingPoint:
    """A detected swing high or swing low in price history."""
    index:  int          # bar index in the closes array
    price:  float        # price at this swing
    kind:   str          # "high" or "low"
    volume: float        # volume at this bar (for weighting)
    strength: float      # fractional strength: how many bars on each side confirm it


@dataclass(slots=True)
class FibLevel:
    """A single Fibonacci-derived price level with confluence metadata."""
    price:       float   # the computed price level
    ratio:       float   # Fibonacci ratio (e.g. 0.618, 1.272)
    kind:        str     # "retracement" or "extension"
    swing_span:  float   # price range of the originating swing (larger = more significant)
    confluence:  float   # confluence score: how many other levels cluster here


@dataclass(slots=True)
class FibTPSL:
    """Computed TP and SL with full Fibonacci context."""
    tp_price:         float
    sl_price:         float
    tp_fib_ratio:     float         # which Fib ratio determined TP
    sl_fib_ratio:     float         # which Fib ratio determined SL
    tp_confluence:    float         # confluence score at TP level
    sl_confluence:    float         # confluence score at SL level
    tp_distance:      float         # |tp - entry| in price
    sl_distance:      float         # |sl - entry| in price
    gross_rr:         float         # tp_distance / sl_distance
    levels_above:     List[FibLevel]  # all computed levels above entry
    levels_below:     List[FibLevel]  # all computed levels below entry
    # ── Source / audit metadata (defaults → backward-compatible) ──────────
    swings_detected:  int   = 0     # number of swing points found
    sl_source:        str   = "FIB" # "FIB" | "FIB_RELAXED" | "ATR_FALLBACK" | "PERCENTAGE_FALLBACK"
    tp_source:        str   = "FIB" # same
    used_actual_hlv:  bool  = False  # True when caller supplied real OHLCV H/L
    swing_range_low:  float = 0.0   # lowest swing low used in level generation
    swing_range_high: float = 0.0   # highest swing high used in level generation
    atr:              float = 0.0   # ATR used for sizing (debug)


@dataclass(slots=True)
class FibTrailResult:
    """Result of Fibonacci-based trailing stop computation."""
    new_sl:              Optional[float]
    phase:               str              # WARMUP | INITIAL | BREAKEVEN | FIB_TRAIL | FIB_LOCK
    current_fib_ratio:   float            # which Fib retracement is active
    fib_sl_price:        float            # raw Fibonacci SL before breakeven floor
    fee_breakeven_move:  float
    tp_progress:         float
    high_watermark:      float            # best price reached during trade


# ═══════════════════════════════════════════════════════════════════════════════
# SWING DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_swings(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    min_order: int = 3,
    max_order: int = 10,
    atr_noise_filter: float = 0.5,
) -> List[SwingPoint]:
    """
    Detect swing highs and lows using adaptive fractal pivot identification.

    A swing high of order N means the high is the highest of 2N+1 bars
    (N bars on each side). Higher order = more significant swing.

    The algorithm tests orders from max_order down to min_order, assigning
    each bar the highest order at which it qualifies. This naturally ranks
    pivots by structural significance.

    ATR noise filter: swings whose range from the adjacent swing is less than
    atr_noise_filter × ATR are rejected (avoids micro-noise pivots).

    Parameters:
        closes:   1D array of close prices
        highs:    1D array of high prices
        lows:     1D array of low prices
        volumes:  1D array of volumes
        min_order: minimum fractal order (default 3 = 7-bar pattern)
        max_order: maximum fractal order (default 10 = 21-bar pattern)
        atr_noise_filter: minimum swing range as fraction of ATR

    Returns:
        List of SwingPoint, sorted by index (chronological order).
    """
    n = len(closes)
    if n < 2 * max_order + 1:
        max_order = max(min_order, (n - 1) // 2)
    if n < 2 * min_order + 1:
        return []

    # Compute ATR for noise filtering
    atr = float(np.mean(np.abs(np.diff(closes[-min(60, n):]))))
    noise_threshold = atr * atr_noise_filter

    swing_highs: dict[int, float] = {}  # index → strength
    swing_lows:  dict[int, float] = {}

    for order in range(max_order, min_order - 1, -1):
        for i in range(order, n - order):
            # Swing high: high[i] is the max of the window
            if i not in swing_highs:
                window_highs = highs[i - order: i + order + 1]
                if highs[i] == np.max(window_highs) and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                    swing_highs[i] = order / max_order

            # Swing low: low[i] is the min of the window
            if i not in swing_lows:
                window_lows = lows[i - order: i + order + 1]
                if lows[i] == np.min(window_lows) and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    swing_lows[i] = order / max_order

    # Build SwingPoint list and apply noise filter
    raw_swings: List[SwingPoint] = []
    for idx, strength in swing_highs.items():
        raw_swings.append(SwingPoint(
            index=idx, price=float(highs[idx]), kind="high",
            volume=float(volumes[idx]) if idx < len(volumes) else 1.0,
            strength=strength,
        ))
    for idx, strength in swing_lows.items():
        raw_swings.append(SwingPoint(
            index=idx, price=float(lows[idx]), kind="low",
            volume=float(volumes[idx]) if idx < len(volumes) else 1.0,
            strength=strength,
        ))

    raw_swings.sort(key=lambda s: s.index)

    # Noise filter: remove swings where the range to the nearest opposite swing
    # is less than the noise threshold
    if len(raw_swings) < 2 or noise_threshold <= 0:
        return raw_swings

    filtered: List[SwingPoint] = [raw_swings[0]]
    for i in range(1, len(raw_swings)):
        prev = filtered[-1]
        curr = raw_swings[i]
        swing_range = abs(curr.price - prev.price)
        if swing_range >= noise_threshold or curr.kind != prev.kind:
            filtered.append(curr)
        elif curr.strength > prev.strength:
            # Replace weaker same-type swing with stronger one
            filtered[-1] = curr

    return filtered


def _extract_hlv(closes: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize high/low arrays from close prices.
    In a real system these would come from OHLCV data. Here we approximate:
      high ≈ close + half the absolute bar-to-bar change (captures the wick)
      low  ≈ close - half the absolute bar-to-bar change
    """
    diffs = np.zeros_like(closes)
    diffs[1:] = np.abs(np.diff(closes))
    half_range = diffs * 0.5
    # Ensure minimum range from ATR-like measure
    min_range = np.mean(diffs[diffs > 0]) * 0.3 if np.any(diffs > 0) else 1.0
    half_range = np.maximum(half_range, min_range)
    highs = closes + half_range
    lows = closes - half_range
    return highs, lows, volumes


# ═══════════════════════════════════════════════════════════════════════════════
# FIBONACCI LEVEL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fib_levels(
    swings: List[SwingPoint],
    current_price: float,
    max_swing_pairs: int = 6,
    confluence_tolerance_atr: float = 0.3,
    atr: float = 1.0,
) -> Tuple[List[FibLevel], List[FibLevel]]:
    """
    Compute Fibonacci retracement and extension levels from multiple swing pairs.

    Multi-scale approach: uses overlapping swing pairs (not just the most recent)
    to find levels where Fibonacci ratios from different structural moves converge.

    Confluence scoring: when levels from different swing pairs fall within
    confluence_tolerance_atr × ATR of each other, they receive a higher score.

    Parameters:
        swings: detected swing points (chronologically ordered)
        current_price: current market price
        max_swing_pairs: maximum number of swing pairs to analyze
        confluence_tolerance_atr: clustering tolerance as fraction of ATR
        atr: current ATR for tolerance computation

    Returns:
        (levels_above, levels_below): FibLevels above and below current_price,
        sorted by distance from current_price (nearest first).
    """
    if len(swings) < 2:
        return [], []

    # Select the most recent and significant swing pairs
    # A swing pair is (swing_low, swing_high) or vice versa
    pairs: List[Tuple[SwingPoint, SwingPoint]] = []

    for i in range(len(swings) - 1):
        for j in range(i + 1, min(i + 5, len(swings))):
            s1, s2 = swings[i], swings[j]
            if s1.kind != s2.kind:  # need high-low or low-high pair
                pairs.append((s1, s2))

    if not pairs:
        return [], []

    # Score pairs by: recency × strength × swing_range
    def pair_score(pair: Tuple[SwingPoint, SwingPoint]) -> float:
        s1, s2 = pair
        recency = (s1.index + s2.index) / 2.0  # higher index = more recent
        strength = (s1.strength + s2.strength) / 2.0
        swing_range = abs(s1.price - s2.price)
        return recency * strength * swing_range

    pairs.sort(key=pair_score, reverse=True)
    pairs = pairs[:max_swing_pairs]

    # Generate all Fibonacci levels from all pairs
    all_levels: List[FibLevel] = []
    tolerance = confluence_tolerance_atr * atr

    for s1, s2 in pairs:
        swing_low  = min(s1.price, s2.price)
        swing_high = max(s1.price, s2.price)
        swing_range = swing_high - swing_low

        if swing_range < atr * 0.5:
            continue  # skip trivially small swings

        # Determine swing direction for retracement/extension
        # If the move was UP (low → high), retracements are from the high
        is_upswing = (s1.kind == "low" and s2.kind == "high") or \
                     (s1.index < s2.index and s1.price < s2.price)

        if is_upswing:
            # Retracements pull back from the high toward the low
            for ratio in FIB_RETRACEMENT_LEVELS:
                level_price = swing_high - ratio * swing_range
                all_levels.append(FibLevel(
                    price=level_price, ratio=ratio, kind="retracement",
                    swing_span=swing_range, confluence=1.0,
                ))
            # Extensions project above the high
            for ratio in FIB_EXTENSION_LEVELS:
                level_price = swing_low + ratio * swing_range
                all_levels.append(FibLevel(
                    price=level_price, ratio=ratio, kind="extension",
                    swing_span=swing_range, confluence=1.0,
                ))
        else:
            # Downswing: retracements bounce from the low toward the high
            for ratio in FIB_RETRACEMENT_LEVELS:
                level_price = swing_low + ratio * swing_range
                all_levels.append(FibLevel(
                    price=level_price, ratio=ratio, kind="retracement",
                    swing_span=swing_range, confluence=1.0,
                ))
            # Extensions project below the low
            for ratio in FIB_EXTENSION_LEVELS:
                level_price = swing_high - ratio * swing_range
                all_levels.append(FibLevel(
                    price=level_price, ratio=ratio, kind="extension",
                    swing_span=swing_range, confluence=1.0,
                ))

    if not all_levels:
        return [], []

    # ── Confluence scoring ────────────────────────────────────────────────
    # For each level, count how many other levels cluster within tolerance
    prices = np.array([lv.price for lv in all_levels])
    for i, level in enumerate(all_levels):
        distances = np.abs(prices - level.price)
        # Count levels within tolerance (excluding self)
        cluster_count = int(np.sum(distances < tolerance)) - 1
        # Weight by swing_span: levels from larger swings get bonus
        span_weight = level.swing_span / (np.mean([lv.swing_span for lv in all_levels]) + 1e-12)
        level.confluence = 1.0 + cluster_count * 0.5 + span_weight * 0.3

    # ── Merge confluent levels ────────────────────────────────────────────
    # When multiple levels cluster, keep the one with highest confluence
    # and absorb the others' confluence scores
    all_levels.sort(key=lambda lv: lv.price)
    merged: List[FibLevel] = []
    i = 0
    while i < len(all_levels):
        cluster = [all_levels[i]]
        j = i + 1
        while j < len(all_levels) and abs(all_levels[j].price - all_levels[i].price) < tolerance:
            cluster.append(all_levels[j])
            j += 1

        # Keep the level with highest individual confluence, sum all confluences
        best = max(cluster, key=lambda lv: lv.confluence)
        total_confluence = sum(lv.confluence for lv in cluster)
        best.confluence = total_confluence
        # Use volume-weighted average price for the merged level
        best.price = sum(lv.price * lv.confluence for lv in cluster) / total_confluence
        merged.append(best)
        i = j

    # ── Split into above/below current price ──────────────────────────────
    levels_above = [lv for lv in merged if lv.price > current_price + tolerance * 0.5]
    levels_below = [lv for lv in merged if lv.price < current_price - tolerance * 0.5]

    # Sort by distance from current price (nearest first)
    levels_above.sort(key=lambda lv: lv.price - current_price)
    levels_below.sort(key=lambda lv: current_price - lv.price)

    return levels_above, levels_below


# ═══════════════════════════════════════════════════════════════════════════════
# TP/SL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fib_tp_sl(
    side: str,
    current_price: float,
    closes: np.ndarray,
    volumes: np.ndarray,
    fee_rate: float,
    min_rr: float = 2.0,
    sl_atr_buffer_mult: float = 0.3,
    tp_cap_pct: float = 0.008,
    sl_cap_pct: float = 0.004,
    swing_min_order: int = 3,
    swing_max_order: int = 10,
    swing_atr_noise: float = 0.5,
    max_swing_pairs: int = 6,
    confluence_tolerance_atr: float = 0.3,
    preferred_sl_ratios: Tuple[float, ...] = (0.786, 0.618, 0.500, 0.382),
    preferred_tp_ratios: Tuple[float, ...] = (1.272, 1.618, 1.000, 2.000),
    highs: Optional[np.ndarray] = None,
    lows:  Optional[np.ndarray] = None,
) -> FibTPSL:
    """
    Compute TP and SL from Fibonacci levels derived from structural swing analysis.

    SL placement logic (LONG example):
      1. Detect swing lows/highs from recent price history
      2. Compute Fibonacci retracement levels from all qualifying swing pairs
      3. Select the nearest Fibonacci support level below entry that:
         a) Is beyond the ATR noise buffer (avoids wick stops)
         b) Has meaningful confluence score (structural significance)
         c) Maintains minimum R:R after fees
      4. Add a small ATR buffer below the Fib level (institutional practice:
         SL sits just beyond the level, not on it)

    TP placement logic (LONG example):
      1. Compute Fibonacci extension levels from swing pairs
      2. Select the nearest extension level above entry that:
         a) Provides adequate net R:R after round-trip fees
         b) Is achievable (within tp_cap_pct of current price)
         c) Has confluence from multiple swing measurements

    Parameters:
        side:          "long" or "short"
        current_price: current market price (entry candidate)
        closes:        recent 1m close prices (at least 60 bars)
        volumes:       matching volume array
        fee_rate:      one-way fee rate (e.g. 0.00053 for 0.053%)
        min_rr:        minimum gross risk:reward ratio
        sl_atr_buffer_mult: ATR multiplier for SL buffer beyond Fib level
        tp_cap_pct:    maximum TP distance as fraction of price
        sl_cap_pct:    maximum SL distance as fraction of price
        swing_min_order: min fractal order for swing detection
        swing_max_order: max fractal order for swing detection
        swing_atr_noise: ATR fraction for swing noise filter
        max_swing_pairs: max swing pairs for Fib computation
        confluence_tolerance_atr: confluence clustering tolerance as ATR fraction
        preferred_sl_ratios: Fib ratios to prefer for SL (tried in order)
        preferred_tp_ratios: Fib ratios to prefer for TP (tried in order)

    Returns:
        FibTPSL with computed TP, SL, and full level context.
    """
    is_long = side == "long"
    n = len(closes)

    # ── ATR computation ───────────────────────────────────────────────────
    atr_window = min(20, n - 1)
    if atr_window < 3:
        # Insufficient data — use percentage-based fallback
        return _percentage_fallback(side, current_price, tp_cap_pct, sl_cap_pct)

    atr = float(np.mean(np.abs(np.diff(closes[-atr_window - 1:]))))
    if atr <= 0:
        atr = current_price * 0.0005

    # ── Fee computation ───────────────────────────────────────────────────
    fee_rt_price = current_price * fee_rate * 2.0

    # ── Swing detection ───────────────────────────────────────────────────
    # Use real OHLCV highs/lows when provided by the caller (MUCH better
    # swing quality than synthesized H/L from close differences).
    used_actual_hlv = (
        highs is not None and lows is not None
        and len(highs) == len(closes) and len(lows) == len(closes)
    )
    if used_actual_hlv:
        h_arr = highs
        l_arr = lows
        vols  = volumes
    else:
        h_arr, l_arr, vols = _extract_hlv(closes, volumes)
        logger.debug("compute_fib_tp_sl: synthesising H/L from closes (no OHLCV supplied)")

    swings = detect_swings(
        closes, h_arr, l_arr, vols,
        min_order=swing_min_order,
        max_order=swing_max_order,
        atr_noise_filter=swing_atr_noise,
    )

    # ── Fibonacci levels ──────────────────────────────────────────────────
    levels_above, levels_below = compute_fib_levels(
        swings, current_price,
        max_swing_pairs=max_swing_pairs,
        confluence_tolerance_atr=confluence_tolerance_atr,
        atr=atr,
    )

    # ── SL selection ──────────────────────────────────────────────────────
    sl_buffer = atr * sl_atr_buffer_mult
    sl_max_dist = current_price * sl_cap_pct
    # Minimum: 0.5 × ATR (was 1.0×) — allows more Fib levels to qualify;
    # structural buffer is handled by the ATR buffer added beyond the Fib level.
    sl_min_dist = atr * 0.5

    # Swing range for diagnostics
    swing_lows  = [s.price for s in swings if s.kind == "low"]
    swing_highs = [s.price for s in swings if s.kind == "high"]
    swing_range_low  = float(min(swing_lows))  if swing_lows  else current_price
    swing_range_high = float(max(swing_highs)) if swing_highs else current_price

    sl_candidates = levels_below if is_long else levels_above
    sl_price, sl_ratio, sl_confluence, sl_relaxed = _select_sl_level(
        candidates=sl_candidates,
        current_price=current_price,
        is_long=is_long,
        buffer=sl_buffer,
        min_dist=sl_min_dist,
        max_dist=sl_max_dist,
        preferred_ratios=preferred_sl_ratios,
    )

    sl_source = "FIB"
    if sl_price <= 0:
        sl_source = "ATR_FALLBACK"
        # Wider fallback: 3.5× ATR ensures the SL is well beyond noise
        sl_dist = min(atr * 3.5, sl_max_dist)
        sl_dist = max(sl_dist, sl_min_dist)
        sl_price = (current_price - sl_dist) if is_long else (current_price + sl_dist)
        sl_ratio = 0.0
        sl_confluence = 0.0
        logger.debug(
            "compute_fib_tp_sl: no Fib SL found in [%.1f, %.1f] — "
            "using ATR fallback sl_dist=%.1f",
            sl_min_dist, sl_max_dist, sl_dist,
        )
    elif sl_relaxed:
        sl_source = "FIB_RELAXED"

    sl_distance = abs(current_price - sl_price)

    # ── TP selection ──────────────────────────────────────────────────────
    tp_max_dist = current_price * tp_cap_pct
    tp_min_dist = sl_distance * min_rr  # ensure minimum R:R
    tp_min_dist = max(tp_min_dist, fee_rt_price * 3.0)  # must exceed 3× fees

    tp_candidates = levels_above if is_long else levels_below
    tp_price, tp_ratio, tp_confluence = _select_tp_level(
        candidates=tp_candidates,
        current_price=current_price,
        is_long=is_long,
        min_dist=tp_min_dist,
        max_dist=tp_max_dist,
        preferred_ratios=preferred_tp_ratios,
    )

    tp_source = "FIB"
    # If no Fibonacci level qualified, use ATR-scaled TP
    if tp_price <= 0:
        tp_source = "ATR_FALLBACK"
        tp_dist = min(atr * 5.0, tp_max_dist)
        tp_dist = max(tp_dist, tp_min_dist)
        tp_price = (current_price + tp_dist) if is_long else (current_price - tp_dist)
        tp_ratio = 0.0
        tp_confluence = 0.0
        logger.debug(
            "compute_fib_tp_sl: no Fib TP found — using ATR fallback tp_dist=%.1f", tp_dist
        )

    tp_distance = abs(tp_price - current_price)

    # ── R:R validation ────────────────────────────────────────────────────
    gross_rr = tp_distance / sl_distance if sl_distance > 0 else 0.0

    # If R:R is below minimum, widen TP or tighten SL (prefer widening TP)
    if gross_rr < min_rr and sl_distance > 0:
        needed_tp_dist = sl_distance * min_rr
        if needed_tp_dist <= tp_max_dist:
            tp_distance = needed_tp_dist
            tp_price = (current_price + tp_distance) if is_long else (current_price - tp_distance)
            gross_rr = min_rr

    return FibTPSL(
        tp_price=round(tp_price, 1),
        sl_price=round(sl_price, 1),
        tp_fib_ratio=tp_ratio,
        sl_fib_ratio=sl_ratio,
        tp_confluence=round(tp_confluence, 2),
        sl_confluence=round(sl_confluence, 2),
        tp_distance=round(tp_distance, 1),
        sl_distance=round(sl_distance, 1),
        gross_rr=round(gross_rr, 2),
        levels_above=levels_above[:5],
        levels_below=levels_below[:5],
        swings_detected=len(swings),
        sl_source=sl_source,
        tp_source=tp_source,
        used_actual_hlv=used_actual_hlv,
        swing_range_low=round(swing_range_low, 1),
        swing_range_high=round(swing_range_high, 1),
        atr=round(atr, 2),
    )


def _select_sl_level(
    candidates: List[FibLevel],
    current_price: float,
    is_long: bool,
    buffer: float,
    min_dist: float,
    max_dist: float,
    preferred_ratios: Tuple[float, ...],
) -> Tuple[float, float, float, bool]:
    """
    Select the best SL Fibonacci level from candidates.

    Priority:
      1. Preferred ratios (0.786, 0.618, 0.500 are structurally strongest)
      2. Within distance bounds (min_dist to max_dist)
      3. Highest confluence score among qualifying levels

    Pass 1 — strict [min_dist, max_dist].
    Pass 2 — relax max_dist to 2× cap (captures wider structural levels).
    Pass 3 — take any level beyond 0.5× min_dist (last resort before ATR fallback).

    Returns (price, ratio, confluence, relaxed) — relaxed=True when pass 2 or 3 fired.
    """
    if not candidates:
        return 0.0, 0.0, 0.0, False

    def _score(lv: FibLevel) -> float:
        ratio_bonus = 0.0
        for rank, pref_ratio in enumerate(preferred_ratios):
            if abs(lv.ratio - pref_ratio) < 0.01:
                ratio_bonus = (len(preferred_ratios) - rank) * 2.0
                break
        return ratio_bonus + lv.confluence

    def _apply_buffer(lv: FibLevel) -> float:
        return (lv.price - buffer) if is_long else (lv.price + buffer)

    def _filter(lo: float, hi: float) -> List[FibLevel]:
        return [lv for lv in candidates if lo <= abs(lv.price - current_price) <= hi]

    # Pass 1: strict window
    valid = _filter(min_dist, max_dist)
    if valid:
        best = max(valid, key=_score)
        return _apply_buffer(best), best.ratio, best.confluence, False

    # Pass 2: relax max to 2× cap — allows structurally wider Fib levels
    valid = _filter(min_dist, max_dist * 2.0)
    if valid:
        best = max(valid, key=_score)
        logger.debug(
            "_select_sl_level pass2 (relaxed max=%.1f): found ratio=%.3f dist=%.1f",
            max_dist * 2.0, best.ratio, abs(best.price - current_price),
        )
        return _apply_buffer(best), best.ratio, best.confluence, True

    # Pass 3: any level beyond 0.5× min_dist (avoids placing SL in the noise floor)
    valid = [lv for lv in candidates if abs(lv.price - current_price) >= min_dist * 0.5]
    if valid:
        # Prefer the nearest qualifying level so SL isn't absurdly far
        valid.sort(key=lambda lv: abs(lv.price - current_price))
        best = max(valid[:3], key=_score)  # score the 3 nearest, pick best-scored
        logger.debug(
            "_select_sl_level pass3 (any>=0.5×min): found ratio=%.3f dist=%.1f",
            best.ratio, abs(best.price - current_price),
        )
        return _apply_buffer(best), best.ratio, best.confluence, True

    return 0.0, 0.0, 0.0, False


def _select_tp_level(
    candidates: List[FibLevel],
    current_price: float,
    is_long: bool,
    min_dist: float,
    max_dist: float,
    preferred_ratios: Tuple[float, ...],
) -> Tuple[float, float, float]:
    """
    Select the best TP Fibonacci level from candidates.

    Priority:
      1. Within distance bounds
      2. Preferred extension ratios (1.272, 1.618 are the primary institutional targets)
      3. Highest confluence

    Returns (price, ratio, confluence) or (0, 0, 0) if none qualify.
    """
    if not candidates:
        return 0.0, 0.0, 0.0

    valid = []
    for lv in candidates:
        dist = abs(lv.price - current_price)
        if min_dist <= dist <= max_dist:
            valid.append(lv)

    if not valid:
        # Take the nearest extension level that's at least min_dist away
        for lv in candidates:
            dist = abs(lv.price - current_price)
            if dist >= min_dist and lv.kind == "extension":
                valid.append(lv)
                break

    if not valid:
        return 0.0, 0.0, 0.0

    def score(lv: FibLevel) -> float:
        ratio_bonus = 0.0
        for rank, pref_ratio in enumerate(preferred_ratios):
            if abs(lv.ratio - pref_ratio) < 0.05:
                ratio_bonus = (len(preferred_ratios) - rank) * 2.0
                break
        return ratio_bonus + lv.confluence

    best = max(valid, key=score)
    return best.price, best.ratio, best.confluence


def _percentage_fallback(
    side: str,
    current_price: float,
    tp_pct: float,
    sl_pct: float,
) -> FibTPSL:
    """Minimal TP/SL when insufficient data for swing detection."""
    is_long = side == "long"
    tp_dist = current_price * tp_pct
    sl_dist = current_price * sl_pct
    tp = (current_price + tp_dist) if is_long else (current_price - tp_dist)
    sl = (current_price - sl_dist) if is_long else (current_price + sl_dist)
    return FibTPSL(
        tp_price=round(tp, 1), sl_price=round(sl, 1),
        tp_fib_ratio=0.0, sl_fib_ratio=0.0,
        tp_confluence=0.0, sl_confluence=0.0,
        tp_distance=round(tp_dist, 1), sl_distance=round(sl_dist, 1),
        gross_rr=round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0,
        levels_above=[], levels_below=[],
        swings_detected=0,
        sl_source="PERCENTAGE_FALLBACK",
        tp_source="PERCENTAGE_FALLBACK",
        used_actual_hlv=False,
        swing_range_low=current_price,
        swing_range_high=current_price,
        atr=0.0,
    )


def format_fib_telegram_section(result: FibTPSL, side: str) -> str:
    """
    Return a compact Telegram-ready Markdown string summarising all Fibonacci
    level decisions for a trade entry.  Append this directly to the entry
    notification so the user can verify every level without looking at logs.

    Format:
        📐 *Fibonacci Analysis*
        SL: $84,123.0 | Fib 0.786 | conf 3.2 | [FIB]
        TP: $85,400.0 | Fib 1.272 | conf 4.1 | [FIB]
        Gross R:R  2.34:1 | ATR $42.1 | Swings 8
        H/L source: REAL OHLCV
        Swing range: $83,800 – $85,600
        SL levels ↓  0.786@83980  0.618@84090  0.500@84180
        TP levels ↑  1.272@85420  1.618@85900  1.000@85060
    """
    is_long = side == "long"
    sl_ratio_str = f"{result.sl_fib_ratio:.3f}" if result.sl_fib_ratio else "n/a"
    tp_ratio_str = f"{result.tp_fib_ratio:.3f}" if result.tp_fib_ratio else "n/a"

    sl_levels_str = "  ".join(
        f"{lv.ratio:.3f}@{lv.price:,.0f}"
        for lv in (result.levels_below if is_long else result.levels_above)[:4]
    ) or "none"
    tp_levels_str = "  ".join(
        f"{lv.ratio:.3f}@{lv.price:,.0f}"
        for lv in (result.levels_above if is_long else result.levels_below)[:4]
    ) or "none"

    hlv_label = "✅ REAL OHLCV" if result.used_actual_hlv else "⚠️ synthesised (no H/L)"
    sl_src_emoji = "✅" if result.sl_source == "FIB" else ("⚠️" if result.sl_source == "FIB_RELAXED" else "❌")
    tp_src_emoji = "✅" if result.tp_source == "FIB" else ("⚠️" if result.tp_source == "FIB_RELAXED" else "❌")

    lines = [
        "📐 *Fibonacci Analysis*",
        f"SL: `${result.sl_price:,.1f}` | Fib `{sl_ratio_str}` | conf `{result.sl_confluence:.1f}` | {sl_src_emoji} `{result.sl_source}`",
        f"TP: `${result.tp_price:,.1f}` | Fib `{tp_ratio_str}` | conf `{result.tp_confluence:.1f}` | {tp_src_emoji} `{result.tp_source}`",
        f"R:R `{result.gross_rr:.2f}:1` | ATR `${result.atr:.1f}` | Swings `{result.swings_detected}`",
        f"H/L data: {hlv_label}",
        f"Swing range: `${result.swing_range_low:,.0f}` – `${result.swing_range_high:,.0f}`",
        f"{'SL' if is_long else 'SL'} levels ({'↓' if is_long else '↑'}): `{sl_levels_str}`",
        f"{'TP' if is_long else 'TP'} levels ({'↑' if is_long else '↓'}): `{tp_levels_str}`",
    ]
    return "\n".join(lines)




# Fibonacci retracement ratios for trailing (from tight to wide)
# As the trade progresses toward TP, we step from wider to tighter retracements
_TRAIL_FIB_SCHEDULE = [
    # (tp_progress_threshold, fib_retracement_ratio)
    # At early profit: wide trail (0.786 retracement of move = keep 21.4%)
    # At mid profit:   medium trail (0.618 = keep 38.2%)
    # Near TP:         tight trail (0.382 = keep 61.8%)
    (0.20, 0.786),   # 20-40% of TP: trail at 0.786 retracement (wide, let it breathe)
    (0.40, 0.618),   # 40-60% of TP: trail at 0.618 retracement (golden ratio)
    (0.60, 0.500),   # 60-75% of TP: trail at 0.500 retracement (half)
    (0.75, 0.382),   # 75-90% of TP: trail at 0.382 retracement (tight)
    (0.90, 0.236),   # 90%+ of TP:   trail at 0.236 retracement (lock profits)
]


def compute_fib_trailing_stop(
    side: str,
    entry_price: float,
    current_price: float,
    current_sl: float,
    tp_price: float,
    entry_fee_usd: float,
    position_size: int,
    contract_value: float,
    bars_held: int,
    warmup_bars: int = 2,
    be_activation_pct: float = 0.20,
    be_fee_margin: float = 1.1,
    min_step_ticks: float = 0.5,
    absolute_max_bars: int = 120,
    high_watermark: float = 0.0,
) -> FibTrailResult:
    """
    Fibonacci-level trailing stop with breakeven floor.

    Architecture:
      1. Warmup phase: no SL modification for first N bars
      2. Activation: trailing begins after price moves be_activation_pct of TP distance
      3. Breakeven floor: SL never placed where a fill would result in net loss
         after round-trip fees (entry + estimated exit)
      4. Fibonacci trail: SL placed at the Fib retracement of (entry → high_watermark),
         where the specific Fib ratio tightens as tp_progress increases
      5. Ratchet: SL can only improve (move in favorable direction)

    The key innovation over ATR-based trailing:
      - ATR trailing uses a fixed multiplier that doesn't adapt to the trade's progress
      - Fibonacci trailing uses the golden ratio to naturally tighten the stop as
        the trade develops, matching the mathematical structure of price retracements

    Parameters:
        side:            "long" or "short"
        entry_price:     actual fill price
        current_price:   latest bar close
        current_sl:      current SL price on the exchange
        tp_price:        take-profit price
        entry_fee_usd:   exact entry fee from API
        position_size:   contracts
        contract_value:  BTC per contract
        bars_held:       bars since entry
        warmup_bars:     bars to wait before trailing begins
        be_activation_pct: fraction of TP distance to activate trailing
        be_fee_margin:   safety multiplier on fee breakeven
        min_step_ticks:  minimum SL move to avoid API spam
        absolute_max_bars: hard safety ceiling
        high_watermark:  best price reached during this trade

    Returns:
        FibTrailResult with new_sl (None if no move), phase, and diagnostics.
    """
    is_long = side == "long"

    result = FibTrailResult(
        new_sl=None, phase="INITIAL",
        current_fib_ratio=0.0, fib_sl_price=0.0,
        fee_breakeven_move=0.0, tp_progress=0.0,
        high_watermark=high_watermark,
    )

    # ── Warmup ────────────────────────────────────────────────────────────
    if bars_held < warmup_bars:
        result.phase = "WARMUP"
        return result

    # ── Track high watermark ──────────────────────────────────────────────
    if is_long:
        hwm = max(high_watermark, current_price) if high_watermark > 0 else current_price
    else:
        hwm = min(high_watermark, current_price) if high_watermark > 0 else current_price
    result.high_watermark = hwm

    # ── TP distance & progress ────────────────────────────────────────────
    entry_to_tp = abs(tp_price - entry_price)
    if entry_to_tp <= 0:
        return result

    if is_long:
        favorable_move = hwm - entry_price
    else:
        favorable_move = entry_price - hwm

    tp_progress = favorable_move / entry_to_tp
    result.tp_progress = round(tp_progress, 4)

    # ── Fee breakeven computation ─────────────────────────────────────────
    exit_fee_est = entry_fee_usd
    total_round_trip_fees = entry_fee_usd + exit_fee_est

    if position_size > 0 and contract_value > 0:
        fee_breakeven_move = total_round_trip_fees / (position_size * contract_value)
    else:
        fee_breakeven_move = entry_to_tp * 0.25

    fee_breakeven_with_margin = fee_breakeven_move * be_fee_margin
    result.fee_breakeven_move = round(fee_breakeven_move, 2)

    # ── Activation check ──────────────────────────────────────────────────
    activation_move = entry_to_tp * be_activation_pct
    if favorable_move < activation_move:
        result.phase = "INITIAL"
        return result

    # ── Determine Fibonacci retracement ratio for current progress ─────────
    active_fib_ratio = _TRAIL_FIB_SCHEDULE[-1][1]  # tightest by default
    for threshold, ratio in _TRAIL_FIB_SCHEDULE:
        if tp_progress < threshold + 0.001:  # small epsilon for floating point
            # Use the previous (wider) ratio if we haven't reached this threshold
            break
        active_fib_ratio = ratio

    # Walk the schedule properly
    active_fib_ratio = _TRAIL_FIB_SCHEDULE[0][1]  # start with widest
    for threshold, ratio in _TRAIL_FIB_SCHEDULE:
        if tp_progress >= threshold:
            active_fib_ratio = ratio

    result.current_fib_ratio = active_fib_ratio

    # ── Compute Fibonacci trailing SL ─────────────────────────────────────
    # The SL is placed at the Fibonacci retracement of the move from entry
    # to the high watermark. As the move extends, the SL naturally follows.
    #
    # For LONG: SL = hwm - (hwm - entry) × fib_ratio
    #   At 0.618 retracement: keep 38.2% of the move
    #   At 0.382 retracement: keep 61.8% of the move
    #
    # For SHORT: SL = hwm + (entry - hwm) × fib_ratio
    #   Same logic, mirrored
    move_range = abs(hwm - entry_price)

    if is_long:
        fib_sl = hwm - move_range * active_fib_ratio
    else:
        fib_sl = hwm + move_range * active_fib_ratio

    result.fib_sl_price = round(fib_sl, 1)

    # ── Breakeven floor ───────────────────────────────────────────────────
    if is_long:
        floor_sl = entry_price + fee_breakeven_with_margin
        candidate_sl = max(fib_sl, floor_sl)
        phase = "FIB_TRAIL" if fib_sl >= floor_sl else "BREAKEVEN"
    else:
        ceiling_sl = entry_price - fee_breakeven_with_margin
        candidate_sl = min(fib_sl, ceiling_sl)
        phase = "FIB_TRAIL" if fib_sl <= ceiling_sl else "BREAKEVEN"

    # ── Detect lock phase ─────────────────────────────────────────────────
    if tp_progress >= 0.75 and phase == "FIB_TRAIL":
        phase = "FIB_LOCK"

    result.phase = phase

    # ── Ratchet: SL can only improve ──────────────────────────────────────
    if is_long:
        if candidate_sl > current_sl + min_step_ticks:
            result.new_sl = round(candidate_sl, 1)
    else:
        if candidate_sl < current_sl - min_step_ticks:
            result.new_sl = round(candidate_sl, 1)

    return result
