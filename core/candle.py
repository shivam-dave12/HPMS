"""core/candle.py — Lightweight candle dataclass."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class Candle:
    timestamp: float
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
