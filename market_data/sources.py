"""Normalized OHLCV source adapters for CCXT exchanges and Pionex."""

from __future__ import annotations

import time

import ccxt

from . import pionex


def find_market_symbol(
    markets: dict,
    base: str,
    prefer_spot: bool = True,
    allow_swap_fallback: bool = False,
):
    """Find an active BASE/USDT spot or optional linear-swap market."""
    base_u = str(base).upper()
    spot_symbol = None
    swap_symbol = None
    for market in markets.values():
        if (
            market.get("base") == base_u
            and market.get("quote") == "USDT"
            and market.get("active", True)
        ):
            if market.get("spot") and spot_symbol is None:
                spot_symbol = market["symbol"]
            if market.get("swap") and market.get("linear") and swap_symbol is None:
                swap_symbol = market["symbol"]
    if prefer_spot and spot_symbol:
        return spot_symbol, "spot"
    if not prefer_spot and swap_symbol:
        return swap_symbol, "swap"
    if spot_symbol:
        return spot_symbol, "spot"
    if allow_swap_fallback and swap_symbol:
        return swap_symbol, "swap"
    return None, None


def fetch_ccxt_ohlcv_segmented(
    exchange,
    symbol,
    timeframe="1h",
    since_ms=None,
    end_ms=None,
    limit=1000,
    pause=0.12,
    max_retries=5,
):
    """Fetch CCXT OHLCV pages with retry, progress and dedup guards."""
    output = []
    cursor = since_ms
    last_seen_ms = -1
    calls = 0
    consecutive_errors = 0
    while calls < 100_000:
        calls += 1
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=limit
            )
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            consecutive_errors += 1
            if consecutive_errors > int(max_retries):
                raise RuntimeError(
                    f"OHLCV 網路錯誤，已重試 {max_retries} 次"
                ) from exc
            delay = min(2 ** (consecutive_errors - 1), 16)
            print(f"[Info] Network error: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
            continue

        consecutive_errors = 0
        if not batch:
            break
        output.extend(batch)
        last_ms = batch[-1][0]
        if last_ms <= last_seen_ms:
            break
        last_seen_ms = last_ms
        cursor = last_ms + 1
        if end_ms is not None and cursor > end_ms:
            break
        if pause > 0:
            time.sleep(float(pause))

    if end_ms is not None and output:
        output = [row for row in output if row[0] <= end_ms]
    dedup = {row[0]: row for row in output}
    return [dedup[timestamp] for timestamp in sorted(dedup)]


def fetch_source_ohlcv(
    name,
    base,
    interval,
    since_ms,
    end_ms,
    pause,
    *,
    prefer_spot=True,
    allow_swap_fallback=False,
):
    """Fetch one provider and return ``(market_symbol, market_type, rows)``."""
    source_name = str(name)
    if source_name.lower() == "pionex":
        rows = pionex.fetch_ohlcv_segmented(
            base,
            interval,
            since_ms=int(since_ms),
            end_ms=int(end_ms),
            pause=pause,
        )
        return f"{str(base).upper()}/USDT", "spot", rows

    exchange_class = getattr(ccxt, source_name, None)
    if exchange_class is None:
        return None, None, []
    exchange = exchange_class({"enableRateLimit": True})
    try:
        market_symbol, market_type = find_market_symbol(
            exchange.load_markets(),
            base,
            prefer_spot=prefer_spot,
            allow_swap_fallback=allow_swap_fallback,
        )
        if not market_symbol:
            return None, None, []
        rows = fetch_ccxt_ohlcv_segmented(
            exchange,
            market_symbol,
            timeframe=interval,
            since_ms=since_ms,
            end_ms=end_ms,
            pause=pause,
        )
        return market_symbol, market_type, rows
    finally:
        close = getattr(exchange, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
