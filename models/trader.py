"""models/trader.py — Data models for fill events."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    A = "A"   # Ask / Sell
    B = "B"   # Bid / Buy


@dataclass(slots=True)
class FillEvent:
    coin:       str
    px:         float
    sz:         float
    side:       Side
    time:       datetime
    crossed:    bool
    fee:        float
    oid:        int
    tid:        int
    wallet:     str
    closed_pnl: float = 0.0
