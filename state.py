"""
state.py — Global Shared State
================================
Provides the STATE singleton used across HPMS components for
cross-module coordination (e.g. trading_enabled flag, dry-run mode).

All attributes are plain Python values — no locks needed because
they are only written during startup/shutdown on the main thread
and read by the strategy/telegram threads.
"""

from __future__ import annotations


class _State:
    """Lightweight mutable global state bag."""

    def __init__(self):
        # Set True by HPMSStrategy.start(), False by HPMSStrategy.stop()
        self.trading_enabled: bool = False

        # Set True when running with --dry-run flag
        self.dry_run: bool = False

        # Set True when running against testnet
        self.testnet: bool = False

        # Human-readable run mode label ("LIVE" / "DRY-RUN")
        self.mode: str = "LIVE"

    def __repr__(self) -> str:
        return (
            f"STATE(trading_enabled={self.trading_enabled}, "
            f"dry_run={self.dry_run}, testnet={self.testnet}, "
            f"mode={self.mode!r})"
        )


# Singleton — imported as `from state import STATE`
STATE = _State()
