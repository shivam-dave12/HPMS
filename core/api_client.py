"""
core/api_client.py — Async REST + signing client for Hyperliquid.

Signing uses the official hyperliquid-python-sdk's functions (sign_l1_action,
order_request_to_order_wire, etc.) to guarantee byte-for-byte compatibility
with the HL server's EIP-712 verification.

Nonces are monotonically increasing to prevent duplicate-nonce rejection
when market + SL + TP orders are placed within the same millisecond.

Price rounding enforces Hyperliquid's two-part rule:
  • ≤ 5 significant figures
  • ≤ (MAX_DECIMALS − szDecimals) decimal places
Both constraints are applied; the stricter one wins.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Optional

import httpx
from eth_account import Account
from eth_account.signers.local import LocalAccount

from hyperliquid.utils.signing import (
    order_request_to_order_wire,
    order_wires_to_order_action,
    sign_l1_action,
)

from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
cfg = get_settings()


# ── Equity parsing ────────────────────────────────────────────────────────────

def _sf(value: object) -> float:
    """Safe float conversion. Returns 0.0 on None, empty string, or invalid input."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        return float(v) if v else 0.0
    return 0.0


def parse_equity_from_state(state: dict) -> float:
    """
    Robust equity extraction from a clearinghouseState API response.

    Hyperliquid clearinghouseState fields (all returned as strings):
      withdrawable              — free USDC available to withdraw (cross free cash).
                                  Zero when all cash is tied up in isolated margin.
      marginSummary.accountValue  — total NAV = cross equity + isolated equity + unrealizedPnl.
                                    This is the most complete picture of account worth.
      marginSummary.totalRawUsd   — raw deposited USDC excluding unrealized PnL.
                                    Reliable even when positions are deeply underwater.
      crossMarginSummary.accountValue — cross-margin NAV only (excludes isolated positions).

    Priority waterfall — stops at the first positive value:
      1. marginSummary.accountValue   — total equity across ALL margin types (best single number)
      2. crossMarginSummary.accountValue — cross-margin equity (isolated accounts may be 0 here)
      3. withdrawable                 — free cash only; 0 when margin is locked in positions
      4. marginSummary.totalRawUsd    — raw deposited USDC (last resort; ignores PnL)

    WHY marginSummary.accountValue is preferred over withdrawable
    ------------------------------------------------------------
    `withdrawable` is 0 whenever funds are locked in an open position or isolated
    margin. Using it as the primary source caused the bot to report 0 equity while
    a trade was live.  marginSummary.accountValue correctly includes mark-to-market
    value of all positions.

    WHY we don't just use `or` on the containing dict
    --------------------------------------------------
    The naive pattern:
        margin = state.get("crossMarginSummary") or state.get("marginSummary", {})
    is BROKEN when crossMarginSummary is present but its accountValue is "0".
    A non-empty dict evaluates as truthy regardless of its contents.

    Returns 0.0 when all fields are absent or zero. Never returns negative.
    """
    ms  = state.get("marginSummary")      or {}
    cms = state.get("crossMarginSummary") or {}

    # 1. marginSummary.accountValue — best: total NAV across ALL margin types
    v = _sf(ms.get("accountValue"))
    if v > 0:
        return v

    # 2. crossMarginSummary.accountValue — cross-margin NAV only
    v = _sf(cms.get("accountValue"))
    if v > 0:
        return v

    # 3. withdrawable — free cash (0 when funds locked in positions)
    v = _sf(state.get("withdrawable"))
    if v > 0:
        return v

    # 4. marginSummary.totalRawUsd — raw deposited USDC (ignores unrealized PnL)
    v = _sf(ms.get("totalRawUsd"))
    return max(v, 0.0)


# ── Price / size helpers ──────────────────────────────────────────────────────

def _round_price(price: float, sz_dec: int, is_spot: bool = False) -> float:
    """
    Round price to satisfy Hyperliquid's two-part validity rule.

    Constraint 1: ≤ 5 significant figures
    Constraint 2: ≤ (MAX_DECIMALS − szDecimals) decimal places
                  MAX_DECIMALS = 6 for perps, 8 for spot

    The stricter (smaller dp value) wins. Prevents rejections on coins
    with szDecimals ≥ 1 where constraint 2 is tighter than constraint 1.
    """
    if price <= 0:
        return price
    import math
    magnitude    = math.floor(math.log10(abs(price)))
    sig_fig_dp   = 4 - magnitude
    max_decimals = max((8 if is_spot else 6) - sz_dec, 0)
    dp           = max(min(sig_fig_dp, max_decimals), 0)
    return round(price, dp)


def _round_sz(size: float, sz_dec: int) -> float:
    """
    Round size to the coin's allowed decimal places.
    Raises ValueError if rounding produces zero (prevents silent zero-size orders).
    """
    rounded = round(size, sz_dec)
    if rounded <= 0:
        raise ValueError(
            f"Size {size} rounds to zero at szDecimals={sz_dec} — order skipped"
        )
    return rounded


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class _RateLimiter:
    """
    Steady-state: 0.5 req/s (1 token every 2 s).
    On HTTP 429: 30-second server cooldown imposed on all queued requests.
    """
    _RATE_PER_SEC:   float = 0.5
    _COOLDOWN_429_S: float = 30.0

    def __init__(self) -> None:
        self._interval        = 1.0 / self._RATE_PER_SEC
        self._last: float     = 0.0
        self._throttle_until: float = 0.0
        self._lock            = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now  = time.monotonic()
            cool = self._throttle_until - now
            if cool > 0:
                await asyncio.sleep(cool)
                now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    def notify_throttle(self, extra_s: float | None = None) -> None:
        pause = extra_s if extra_s is not None else self._COOLDOWN_429_S
        self._throttle_until = max(
            self._throttle_until,
            time.monotonic() + pause,
        )
        log.warning(
            "rate_limiter_cooldown",
            pause_s    = pause,
            resumes_in = round(self._throttle_until - time.monotonic(), 1),
        )


# ── Signing ───────────────────────────────────────────────────────────────────

def _sign_and_build_payload(
    wallet:        LocalAccount,
    action:        dict,
    vault_address: str | None,
    nonce:         int,
    is_mainnet:    bool = True,
) -> dict:
    signature = sign_l1_action(wallet, action, vault_address, nonce, None, is_mainnet)
    return {
        "action":       action,
        "nonce":        nonce,
        "signature":    signature,
        "vaultAddress": vault_address,
    }


def _build_order_action(
    coin_index:   int,
    is_buy:       bool,
    sz:           float,
    limit_px:     float,
    order_type:   dict,
    reduce_only:  bool = False,
    grouping:     str  = "na",
) -> dict:
    order_wire = order_request_to_order_wire(
        {
            "coin":        "",
            "is_buy":      is_buy,
            "sz":          sz,
            "limit_px":    limit_px,
            "order_type":  order_type,
            "reduce_only": reduce_only,
        },
        asset=coin_index,
    )
    return order_wires_to_order_action([order_wire], grouping=grouping)


# ── Client ────────────────────────────────────────────────────────────────────

class HyperliquidClient:
    def __init__(self) -> None:
        self._wallet: LocalAccount       = Account.from_key(cfg.private_key)
        self._api_wallet_address_str     = self._wallet.address.lower()
        self._vault_address: str | None  = None  # None = API wallet signing (not vault)

        self._http = httpx.AsyncClient(
            base_url = cfg.rest_url,
            timeout  = httpx.Timeout(cfg.http_timeout_s, connect=5.0),
            headers  = {"Content-Type": "application/json"},
        )
        self._stats_http = httpx.AsyncClient(
            timeout = httpx.Timeout(15.0, connect=5.0),
        )
        self._meta_cache: dict | None = None
        self._rl = _RateLimiter()

        # Cache which (coin, leverage) pairs have already had updateLeverage
        # called this session — avoids a redundant API call on every order.
        self._leverage_set: set[tuple[str, int]] = set()

        self._nonce_lock  = threading.Lock()
        self._last_nonce: int = 0

    def _next_nonce(self) -> int:
        with self._nonce_lock:
            n = max(int(time.time() * 1000), self._last_nonce + 1)
            self._last_nonce = n
            return n

    @property
    def api_wallet_address(self) -> str:
        return self._api_wallet_address_str

    # ── Wallet validation ─────────────────────────────────────────────────────

    async def validate_api_wallet(self) -> tuple[bool, str]:
        # ── Critical misconfiguration check ───────────────────────────────────
        # HL_WALLET_ADDRESS must be the MASTER account address.
        # HL_PRIVATE_KEY derives the API sub-wallet used only for signing.
        # If both resolve to the same address, funds will never be found.
        if cfg.wallet_address.lower() == self._api_wallet_address_str.lower():
            return False, (
                f"MISCONFIGURATION: HL_WALLET_ADDRESS ({cfg.wallet_address[:20]}) "
                f"is the same as the address derived from HL_PRIVATE_KEY. "
                "These must be different accounts. "
                "HL_WALLET_ADDRESS = your MASTER account (login address at app.hyperliquid.xyz). "
                "HL_PRIVATE_KEY = the API sub-wallet private key added under Settings → API."
            )

        try:
            state = await self.get_clearinghouse_state(cfg.wallet_address)
            _ = state.get("marginSummary", {})
        except Exception as e:
            return False, (
                f"Main wallet not found on Hyperliquid ({cfg.wallet_address}). "
                f"Have you deposited USDC? Error: {e}"
            )

        try:
            nonce   = self._next_nonce()
            action  = {"type": "cancel", "cancels": [{"a": 0, "o": 0}]}
            payload = _sign_and_build_payload(
                self._wallet, action, self._vault_address, nonce, cfg.hl_mainnet
            )
            resp     = await self._request_with_retry("POST", "/exchange", payload)
            resp_str = json.dumps(resp).lower()
            if "user or api wallet does not exist" in resp_str:
                return False, (
                    f"Signing wallet {self._api_wallet_address_str} is NOT authorised. "
                    "Go to app.hyperliquid.xyz → Settings → API → Add API Wallet, "
                    "paste that address, approve, then restart."
                )
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (400, 422):
                return False, f"Unexpected HTTP {e.response.status_code} during wallet check"
        except Exception as e:
            log.warning("api_wallet_probe_error", err=str(e))

        return True, "wallet_ok"

    # ── Info endpoints ────────────────────────────────────────────────────────

    async def get_leaderboard(self) -> list[dict]:
        url = (
            "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
            if cfg.hl_mainnet
            else "https://stats-data.hyperliquid-testnet.xyz/Testnet/leaderboard"
        )
        last_exc: Exception | None = None
        for attempt in range(1, cfg.retry_attempts + 1):
            try:
                r = await self._stats_http.get(url)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("leaderboardRows", [])
                return []
            except httpx.HTTPStatusError as e:
                last_exc = e
                log.warning("leaderboard_http_error", status=e.response.status_code, attempt=attempt)
            except httpx.RequestError as e:
                last_exc = e
                log.warning("leaderboard_network_error", err=str(e), attempt=attempt)
            if attempt < cfg.retry_attempts:
                await asyncio.sleep(cfg.retry_delay_s * (2 ** (attempt - 1)))
        raise RuntimeError("Leaderboard fetch failed") from last_exc

    async def get_user_fills_by_time(
        self,
        wallet:        str,
        start_time_ms: int,
        end_time_ms:   Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch fills for a wallet within a time window.

        Uses the 'userFillsByTime' endpoint which correctly filters by
        startTime (and optionally endTime). This is NOT the same as
        'userFills', which returns ALL historical fills and ignores startTime.
        """
        payload: dict = {
            "type":      "userFillsByTime",
            "user":      wallet,
            "startTime": start_time_ms,
        }
        if end_time_ms is not None:
            payload["endTime"] = end_time_ms

        r = await self._post_info(payload)

        if not isinstance(r, list):
            log.warning(
                "api_fills_unexpected_response",
                wallet   = wallet[:12],
                type_got = type(r).__name__,
                snippet  = str(r)[:120],
            )
            return []
        return r

    async def get_user_historical_orders(self, wallet: str) -> list[dict]:
        r = await self._post_info({"type": "historicalOrders", "user": wallet})
        return r if isinstance(r, list) else []

    async def get_clearinghouse_state(self, wallet: str) -> dict:
        # ── Misconfiguration guard ────────────────────────────────────────────
        # The most common cause of zero balance is querying the API sub-wallet
        # (derived from HL_PRIVATE_KEY) instead of the master account wallet.
        # HL returns all-zero clearinghouseState for agent/API wallets because
        # they hold no margin — funds live on the master account only.
        if wallet.lower() == self._api_wallet_address_str.lower():
            log.warning(
                "clearinghouse_wallet_misconfiguration",
                queried_wallet  = wallet[:20],
                api_wallet      = self._api_wallet_address_str[:20],
                note            = (
                    "HL_WALLET_ADDRESS matches the API signing wallet derived from "
                    "HL_PRIVATE_KEY. This will always return zero balance. "
                    "Set HL_WALLET_ADDRESS to your MASTER Hyperliquid account address "
                    "(the one you use to log into app.hyperliquid.xyz)."
                ),
            )

        result = await self._post_info({"type": "clearinghouseState", "user": wallet})
        if not result or not isinstance(result, dict):
            log.warning(
                "api_clearinghouse_empty",
                wallet = wallet,  # log full address — truncation hides misconfiguration
                note   = (
                    "Response missing or not a dict. Ensure HL_WALLET_ADDRESS is the "
                    "MASTER account address, not the API sub-wallet address."
                ),
            )
            return {}

        # Log ALL equity fields at DEBUG (full address, not truncated — truncation
        # previously hid whether the master or API wallet was being queried).
        ms  = result.get("marginSummary")      or {}
        cms = result.get("crossMarginSummary") or {}
        parsed = parse_equity_from_state(result)
        log.debug(
            "clearinghouse_state_raw",
            wallet            = wallet,           # full address
            withdrawable      = result.get("withdrawable"),
            ms_acct_val       = ms.get("accountValue"),
            ms_total_raw_usd  = ms.get("totalRawUsd"),
            cms_acct_val      = cms.get("accountValue"),
            open_positions    = len(result.get("assetPositions", [])),
            parsed_equity     = parsed,
        )

        if parsed == 0.0 and wallet.lower() != self._api_wallet_address_str.lower():
            log.warning(
                "clearinghouse_zero_equity",
                wallet = wallet,
                note   = (
                    "All balance fields are zero. Verify: "
                    "(1) HL_WALLET_ADDRESS is the master account (not API sub-wallet), "
                    "(2) USDC has been deposited to the perp account at app.hyperliquid.xyz."
                ),
            )

        return result

    async def get_spot_clearinghouse_state(self, wallet: str) -> dict:
        result = await self._post_info({"type": "spotClearinghouseState", "user": wallet})
        return result or {}

    async def get_equity(self, wallet: str, state: dict | None = None) -> float:
        """
        Return account equity using a full multi-field waterfall with fallbacks.

        Parameters
        ----------
        wallet : master account address (HL_WALLET_ADDRESS from .env)
        state  : optional pre-fetched clearinghouseState dict — when supplied,
                 perp fields are read from it directly, avoiding a redundant
                 API call. A spot call is still made if all perp fields are zero.

        Waterfall
        ---------
        1. marginSummary.accountValue     (total NAV — cross + isolated + unrealizedPnl)
        2. crossMarginSummary.accountValue (cross-margin NAV only)
        3. withdrawable                   (free cash; 0 when locked in positions)
        4. marginSummary.totalRawUsd      (raw deposited USDC; ignores unrealizedPnl)
        5. Spot USDC balance              (funds not yet transferred to perp)
        6. Re-fetch with API wallet addr  (last resort — catches HL_WALLET_ADDRESS misconfiguration)

        Never raises. Returns 0.0 on complete failure.
        """
        try:
            if state is None:
                state = await self.get_clearinghouse_state(wallet)

            equity = parse_equity_from_state(state)
            if equity > 0:
                return equity

            # Spot USDC fallback — triggered when the account has USDC in the
            # spot wallet but has not yet transferred it to the perp margin account.
            spot = await self.get_spot_clearinghouse_state(wallet)
            for b in spot.get("balances", []):
                if b.get("coin") == "USDC":
                    v = _sf(b.get("total"))
                    if v > 0:
                        log.info(
                            "equity_from_spot_usdc",
                            wallet = wallet,
                            usdc   = round(v, 4),
                            note   = (
                                "All perp equity fields are 0; reading spot USDC. "
                                "Transfer funds at app.hyperliquid.xyz → Transfer "
                                "to enable perp trading."
                            ),
                        )
                        return v

            # Last-resort: if HL_WALLET_ADDRESS was misconfigured as the API sub-wallet,
            # try the API wallet address to detect the problem and guide the user.
            if wallet.lower() != self._api_wallet_address_str.lower():
                log.warning(
                    "get_equity_retrying_api_wallet",
                    master_wallet = wallet,
                    api_wallet    = self._api_wallet_address_str,
                    note          = (
                        "Master wallet returned 0. Checking if HL_WALLET_ADDRESS "
                        "and HL_PRIVATE_KEY belong to separate accounts (correct) "
                        "or the same account (misconfigured)."
                    ),
                )
                api_state = await self._post_info({
                    "type": "clearinghouseState",
                    "user": self._api_wallet_address_str,
                })
                if isinstance(api_state, dict):
                    api_equity = parse_equity_from_state(api_state)
                    if api_equity > 0:
                        log.warning(
                            "get_equity_misconfiguration_detected",
                            api_wallet_equity = api_equity,
                            master_wallet     = wallet,
                            action_required   = (
                                "FUNDS FOUND ON API WALLET. "
                                "Set HL_WALLET_ADDRESS in .env to your master account address "
                                "(the address you log in with at app.hyperliquid.xyz). "
                                "The API wallet derived from HL_PRIVATE_KEY should NOT be "
                                "used as HL_WALLET_ADDRESS."
                            ),
                        )
                        # Return the equity but log it as misconfigured
                        return api_equity

            return 0.0

        except Exception as e:
            log.warning("get_equity_failed", wallet=wallet, err=str(e))
            return 0.0

    async def get_meta(self) -> dict:
        return await self._post_info({"type": "meta"})

    async def get_all_mids(self) -> dict[str, float]:
        r = await self._post_info({"type": "allMids"})
        return {k: float(v) for k, v in r.items()} if isinstance(r, dict) else {}

    async def get_open_orders(self, wallet: str) -> list[dict]:
        r = await self._post_info({"type": "openOrders", "user": wallet})
        return r if isinstance(r, list) else []

    # ── Exchange endpoints ────────────────────────────────────────────────────

    async def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> None:
        """
        Set leverage for a coin. Cached per (coin, leverage) so the API call
        is only made once per coin per session.
        """
        key = (coin, leverage)
        if key in self._leverage_set:
            return

        coin_index, _ = await self._get_coin_info(coin)
        nonce  = self._next_nonce()
        action = {
            "type":     "updateLeverage",
            "asset":    coin_index,
            "isCross":  is_cross,
            "leverage": leverage,
        }
        payload = _sign_and_build_payload(
            self._wallet, action, self._vault_address, nonce, cfg.hl_mainnet
        )
        resp     = await self._post_exchange(payload)
        resp_str = json.dumps(resp).lower() if isinstance(resp, dict) else str(resp)
        if "err" in resp_str or "error" in resp_str:
            log.warning(
                "set_leverage_failed",
                coin     = coin,
                leverage = leverage,
                resp     = str(resp)[:120],
                note     = "Order will proceed at account's existing leverage setting",
            )
        else:
            self._leverage_set.add(key)
            log.debug("set_leverage_ok", coin=coin, leverage=leverage, is_cross=is_cross)

    async def place_market_order(
        self,
        coin:        str,
        is_buy:      bool,
        size:        float,
        slippage:    float = 0.005,
        reduce_only: bool  = False,
    ) -> dict:
        mids = await self.get_all_mids()
        mid  = mids.get(coin)
        if mid is None:
            raise ValueError(f"No mid-price for {coin!r} — coin not in allMids")
        if mid <= 0:
            raise ValueError(
                f"Mid-price for {coin!r} is {mid} (zero or negative) — market may be halted"
            )

        coin_index, sz_dec = await self._get_coin_info(coin)
        limit_px = _round_price(mid * (1 + slippage if is_buy else 1 - slippage), sz_dec)
        if limit_px <= 0:
            raise ValueError(f"Computed limit_px={limit_px} for {coin!r} is non-positive")
        size     = _round_sz(size, sz_dec)
        nonce    = self._next_nonce()

        action = _build_order_action(
            coin_index  = coin_index,
            is_buy      = is_buy,
            sz          = size,
            limit_px    = limit_px,
            order_type  = {"limit": {"tif": "Ioc"}},
            reduce_only = reduce_only,
            grouping    = "na",
        )
        payload = _sign_and_build_payload(
            self._wallet, action, self._vault_address, nonce, cfg.hl_mainnet
        )
        return await self._post_exchange(payload)

    async def place_stop_loss(
        self,
        coin:          str,
        is_buy:        bool,
        size:          float,
        trigger_price: float,
    ) -> dict:
        """
        Place a stop-loss trigger order.
        limit_px uses a 20% band on the loss side so the order fills through gaps.
        grouping="positionTpsl" is correct for attaching a trigger to an existing
        position; "normalTpsl" requires an atomic entry+SL+TP batch.
        """
        coin_index, sz_dec = await self._get_coin_info(coin)
        nonce         = self._next_nonce()
        size          = _round_sz(size, sz_dec)
        trigger_price = _round_price(trigger_price, sz_dec)
        limit_px      = _round_price(trigger_price * (1.20 if is_buy else 0.80), sz_dec)

        action = _build_order_action(
            coin_index  = coin_index,
            is_buy      = is_buy,
            sz          = size,
            limit_px    = limit_px,
            order_type  = {"trigger": {"triggerPx": trigger_price, "isMarket": True, "tpsl": "sl"}},
            reduce_only = True,
            grouping    = "positionTpsl",
        )
        payload = _sign_and_build_payload(
            self._wallet, action, self._vault_address, nonce, cfg.hl_mainnet
        )
        return await self._post_exchange(payload)

    async def place_take_profit(
        self,
        coin:          str,
        is_buy:        bool,
        size:          float,
        trigger_price: float,
    ) -> dict:
        """
        Place a take-profit trigger order.
        limit_px uses the same 20% band logic as place_stop_loss, applied on the
        profitable side so the TP fills even if price momentarily moves past trigger.
        """
        coin_index, sz_dec = await self._get_coin_info(coin)
        nonce         = self._next_nonce()
        size          = _round_sz(size, sz_dec)
        trigger_price = _round_price(trigger_price, sz_dec)
        limit_px      = _round_price(trigger_price * (1.20 if is_buy else 0.80), sz_dec)

        action = _build_order_action(
            coin_index  = coin_index,
            is_buy      = is_buy,
            sz          = size,
            limit_px    = limit_px,
            order_type  = {"trigger": {"triggerPx": trigger_price, "isMarket": True, "tpsl": "tp"}},
            reduce_only = True,
            grouping    = "positionTpsl",
        )
        payload = _sign_and_build_payload(
            self._wallet, action, self._vault_address, nonce, cfg.hl_mainnet
        )
        return await self._post_exchange(payload)

    async def cancel_order(self, coin: str, order_id: int) -> dict:
        coin_index, _ = await self._get_coin_info(coin)
        nonce  = self._next_nonce()
        action = {"type": "cancel", "cancels": [{"a": coin_index, "o": order_id}]}
        payload = _sign_and_build_payload(
            self._wallet, action, self._vault_address, nonce, cfg.hl_mainnet
        )
        return await self._post_exchange(payload)

    async def close_position(self, coin: str, is_long: bool, size: float) -> dict:
        """Market close with reduceOnly=True. Safe even if position is partially closed."""
        return await self.place_market_order(
            coin        = coin,
            is_buy      = not is_long,
            size        = size,
            slippage    = 0.01,
            reduce_only = True,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _post_info(self, payload: dict) -> Any:
        return await self._request_with_retry("POST", "/info", payload)

    async def _post_exchange(self, payload: dict) -> Any:
        result = await self._request_with_retry("POST", "/exchange", payload)
        log.debug("exchange_response", snippet=str(result)[:120])
        return result

    async def _request_with_retry(self, method: str, path: str, payload: dict) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, cfg.retry_attempts + 1):
            await self._rl.acquire()
            try:
                r = await self._http.request(method, path, content=json.dumps(payload))
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status < 500 and status != 429:
                    try:
                        body = e.response.json()
                    except Exception:
                        body = e.response.text[:200]
                    log.error("hl_client_error", status=status, path=path, body=body)
                    raise
                if status == 429:
                    self._rl.notify_throttle()
                last_exc = e
                log.warning("hl_server_error_retry", status=status, attempt=attempt, max=cfg.retry_attempts)
            except httpx.RequestError as e:
                last_exc = e
                log.warning("hl_network_error_retry", err=str(e), attempt=attempt, max=cfg.retry_attempts)
            if attempt < cfg.retry_attempts:
                await asyncio.sleep(cfg.retry_delay_s * (2 ** (attempt - 1)))
        raise RuntimeError(f"HL API failed after {cfg.retry_attempts} attempts") from last_exc

    async def _get_coin_info(self, coin: str) -> tuple[int, int]:
        """
        Returns (coin_index, sz_decimals).
        Re-fetches meta if the cache is absent or contains a non-dict
        (can happen when a prior call returned a transient error string).
        """
        if not isinstance(self._meta_cache, dict):
            self._meta_cache = await self.get_meta()
        if not isinstance(self._meta_cache, dict):
            raise RuntimeError(
                f"get_meta() returned {type(self._meta_cache).__name__} — "
                "HL API may be returning an error body; cannot resolve coin info"
            )
        for i, asset in enumerate(self._meta_cache.get("universe", [])):
            if isinstance(asset, dict) and asset.get("name") == coin:
                return i, int(asset.get("szDecimals", 6))
        raise ValueError(f"Unknown coin: {coin!r}")

    async def invalidate_meta_cache(self) -> None:
        self._meta_cache = None

    async def close(self) -> None:
        await self._http.aclose()
        await self._stats_http.aclose()

    async def __aenter__(self) -> "HyperliquidClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
