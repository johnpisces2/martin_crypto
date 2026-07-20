"""Normalized OHLCV source adapters for Alpaca and CCXT exchanges."""

from __future__ import annotations

import time
from types import SimpleNamespace

import ccxt

from . import alpaca, rate_limit, universe


def _ccxt_retryable_types():
    names = (
        "NetworkError",
        "RequestTimeout",
        "RateLimitExceeded",
        "DDoSProtection",
        "ExchangeNotAvailable",
    )
    return tuple(
        error_type
        for name in names
        if isinstance((error_type := getattr(ccxt, name, None)), type)
    )


def _is_ccxt_rate_limit_error(exc) -> bool:
    rate_types = tuple(
        error_type
        for name in ("RateLimitExceeded", "DDoSProtection")
        if isinstance((error_type := getattr(ccxt, name, None)), type)
    )
    text = str(exc).lower()
    return (
        bool(rate_types and isinstance(exc, rate_types))
        or "429" in text
        or "rate limit" in text
    )


def call_ccxt(
    exchange,
    method,
    *args,
    provider=None,
    weight: float = 1.0,
    max_retries: int = 5,
    **kwargs,
):
    """Call one CCXT endpoint through the shared provider limiter."""
    provider_name = str(
        provider or getattr(exchange, "id", None) or exchange.__class__.__name__
    ).lower()
    retryable_types = _ccxt_retryable_types()
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            rate_limit.acquire(provider_name, weight)
            return method(*args, **kwargs)
        except Exception as exc:
            if not retryable_types or not isinstance(exc, retryable_types):
                raise
            if attempt >= max(0, int(max_retries)):
                raise RuntimeError(
                    f"{provider_name} API failed after {max_retries} retries: {exc}"
                ) from exc
            if _is_ccxt_rate_limit_error(exc):
                response = SimpleNamespace(
                    headers=getattr(exchange, "last_response_headers", {}) or {}
                )
                delay = rate_limit.retry_delay(
                    provider_name, response=response, attempt=attempt
                )
                rate_limit.defer(provider_name, delay)
            else:
                delay = min(2 ** attempt, 16)
                time.sleep(delay)


def find_market_symbol(
    markets: dict,
    base: str,
):
    """Find an active Binance BASE/USDT crypto spot market."""
    base_u = str(base).upper()
    for market in markets.values():
        if (
            market.get("base") == base_u
            and universe.is_binance_crypto_spot_market(market, quote="USDT")
        ):
            return market["symbol"], "spot"
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
    while calls < 100_000:
        calls += 1
        batch = call_ccxt(
            exchange,
            exchange.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            since=cursor,
            limit=limit,
            weight=2,
            max_retries=max_retries,
        )
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
):
    """Fetch one provider and return ``(market_symbol, market_type, rows)``."""
    source_name = str(name)
    if source_name.lower() == "alpaca":
        rows = alpaca.fetch_ohlcv_segmented(
            base,
            interval,
            since_ms=int(since_ms),
            end_ms=int(end_ms),
            pause=pause,
        )
        return str(base).upper(), "stock", rows

    if source_name.lower() != "binance":
        raise ValueError(f"Unsupported market-data source: {source_name}")
    exchange_class = ccxt.binance
    exchange = exchange_class({"enableRateLimit": True})
    try:
        markets = call_ccxt(
            exchange,
            exchange.load_markets,
            provider=source_name,
            weight=20,
        )
        market_symbol, market_type = find_market_symbol(
            markets,
            base,
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
