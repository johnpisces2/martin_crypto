# -*- coding: utf-8 -*-
"""Read-only Pionex public market-data adapter.

Only public SPOT endpoints are used.  No API key, account endpoint, or order
endpoint is implemented here.
"""

from __future__ import annotations

import math
import threading
import time

import requests


PIONEX_API_BASE = "https://api.pionex.com"
PIONEX_MAX_KLINES = 10_000
PIONEX_KLINE_LIMIT = 500
# Pionex documents a shared IP limit of 10 request-weight units/second.
# Keep roughly 17% headroom instead of running directly on the boundary.
PIONEX_MIN_REQUEST_INTERVAL = 0.12

PIONEX_INTERVALS = {
    "1m": "1M",
    "5m": "5M",
    "15m": "15M",
    "30m": "30M",
    "1h": "60M",
    "4h": "4H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}

class PionexPublicError(RuntimeError):
    pass


class _SharedRateLimiter:
    """Process-wide request scheduler shared by all scanner threads."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def acquire(self, weight: float = 1.0):
        request_weight = max(1.0, float(weight))
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_request_at)
            self._next_request_at = scheduled + self.min_interval * request_weight
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)

    def defer(self, seconds: float):
        delay = max(0.0, float(seconds))
        with self._lock:
            self._next_request_at = max(
                self._next_request_at, time.monotonic() + delay
            )


_GLOBAL_RATE_LIMITER = _SharedRateLimiter(PIONEX_MIN_REQUEST_INTERVAL)
_THREAD_LOCAL = threading.local()


def _thread_session():
    """Reuse one requests session per scanner worker for HTTP keep-alive."""
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _retry_after_seconds(response, fallback: float) -> float:
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After")
    try:
        parsed = float(value)
        if math.isfinite(parsed) and parsed >= 0:
            return max(parsed, float(fallback))
    except (TypeError, ValueError):
        pass
    return float(fallback)


def normalize_pionex_symbol(symbol: str, quote: str = "USDT") -> str:
    """Return Pionex's ``BASE_QUOTE`` public-API symbol."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        raise ValueError("Pionex symbol 不可為空")
    raw = raw.split(":", 1)[0].replace("/", "_").replace("-", "_")
    suffix = f"_{quote.upper()}"
    if raw.endswith(suffix):
        return raw
    if "_" in raw:
        raise ValueError(f"不支援的 Pionex symbol 格式：{symbol}")
    return f"{raw}{suffix}"


def _request_json(
    session,
    path: str,
    *,
    params=None,
    timeout: float = 15.0,
    max_retries: int = 6,
    weight: float = 1.0,
):
    url = f"{PIONEX_API_BASE}{path}"
    last_error = None
    for attempt in range(max(1, int(max_retries))):
        try:
            _GLOBAL_RATE_LIMITER.acquire(weight)
            response = session.get(url, params=params or {}, timeout=timeout)
            if int(getattr(response, "status_code", 200)) == 429:
                fallback = min(2 ** attempt, 30)
                delay = _retry_after_seconds(response, fallback)
                _GLOBAL_RATE_LIMITER.defer(delay)
                last_error = requests.HTTPError(
                    f"429 Too Many Requests; retry after {delay:.1f}s",
                    response=response,
                )
                if attempt + 1 >= max(1, int(max_retries)):
                    break
                continue
            response.raise_for_status()
            payload = response.json()
            if not payload.get("result", False):
                code = payload.get("code", "UNKNOWN")
                message = payload.get("message", "Pionex API error")
                raise PionexPublicError(f"{code}: {message}")
            return payload
        except PionexPublicError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(max_retries)):
                break
            _GLOBAL_RATE_LIMITER.defer(min(2 ** attempt, 8))
    raise PionexPublicError(f"Pionex 公開 API 連線失敗：{last_error}") from last_error


class PionexPublicClient:
    """Small CCXT-shaped adapter used by the volatility scanner."""

    def __init__(self, *, session=None, timeout: float = 15.0):
        self._owns_session = session is None
        self.session = session or requests.Session()
        self.timeout = float(timeout)
        self.markets = {}

    def load_markets(self):
        payload = _request_json(
            self.session,
            "/api/v1/common/symbols",
            params={"type": "SPOT"},
            timeout=self.timeout,
            weight=5,
        )
        markets = {}
        for item in payload.get("data", {}).get("symbols", []):
            if str(item.get("type", "")).upper() != "SPOT":
                continue
            base = str(item.get("baseCurrency") or "").upper()
            quote = str(item.get("quoteCurrency") or "").upper()
            if not base or not quote:
                continue
            symbol = f"{base}/{quote}"
            markets[symbol] = {
                "id": str(item.get("symbol") or f"{base}_{quote}"),
                "symbol": symbol,
                "base": base,
                "quote": quote,
                "active": bool(item.get("enable", True)),
                "spot": True,
                "swap": False,
                "linear": False,
                "info": item,
            }
        self.markets = markets
        return markets

    def fetch_tickers(self):
        payload = _request_json(
            self.session,
            "/api/v1/market/tickers",
            params={"type": "SPOT"},
            timeout=self.timeout,
        )
        tickers = {}
        for item in payload.get("data", {}).get("tickers", []):
            api_symbol = str(item.get("symbol") or "").upper()
            if "_" not in api_symbol:
                continue
            base, quote = api_symbol.rsplit("_", 1)
            symbol = f"{base}/{quote}"

            def number(name):
                value = item.get(name)
                try:
                    return float(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            tickers[symbol] = {
                "symbol": symbol,
                "timestamp": int(item.get("time") or 0),
                "open": number("open"),
                "high": number("high"),
                "low": number("low"),
                "close": number("close"),
                "baseVolume": number("volume"),
                "quoteVolume": number("amount"),
                "info": item,
            }
        return tickers

    def close(self):
        if self._owns_session:
            self.session.close()


def fetch_ohlcv_segmented(
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    end_ms: int,
    limit: int = PIONEX_KLINE_LIMIT,
    pause: float = 0.05,
    max_records: int = PIONEX_MAX_KLINES,
    session=None,
):
    """Fetch Pionex OHLCV backwards and return rows sorted oldest-first.

    Pionex returns newest-first and exposes ``endTime`` rather than ``since``.
    Its public endpoint documents a maximum retrievable history of 10,000
    records, so long requests return the newest available 10,000; callers that
    require full coverage can reject that using their normal coverage checks.
    """
    if timeframe not in PIONEX_INTERVALS:
        supported = ", ".join(PIONEX_INTERVALS)
        raise ValueError(f"Pionex 不支援 interval={timeframe}；支援：{supported}")
    since_ms = int(since_ms)
    end_ms = int(end_ms)
    if end_ms < since_ms:
        return []
    page_limit = min(PIONEX_KLINE_LIMIT, max(1, int(limit)))
    record_cap = min(PIONEX_MAX_KLINES, max(1, int(max_records)))
    api_symbol = normalize_pionex_symbol(symbol)
    # A session is local to one worker thread, so requests' connection pool is
    # reused without sharing a Session concurrently across threads.
    sess = session or _thread_session()
    out = []
    cursor_end = end_ms
    previous_oldest = None
    max_calls = int(math.ceil(record_cap / page_limit)) + 1
    for _ in range(max_calls):
        remaining = record_cap - len(out)
        if remaining <= 0:
            break
        request_limit = min(page_limit, remaining)
        payload = _request_json(
            sess,
            "/api/v1/market/klines",
            params={
                "symbol": api_symbol,
                "interval": PIONEX_INTERVALS[timeframe],
                "endTime": int(cursor_end),
                "limit": int(request_limit),
            },
        )
        batch = payload.get("data", {}).get("klines", [])
        if not batch:
            break
        rows = []
        for item in batch:
            try:
                row = [
                    int(item["time"]),
                    float(item["open"]),
                    float(item["high"]),
                    float(item["low"]),
                    float(item["close"]),
                    float(item.get("volume", 0.0)),
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise PionexPublicError("Pionex K 線欄位格式錯誤") from exc
            rows.append(row)

        oldest = min(row[0] for row in rows)
        out.extend(row for row in rows if since_ms <= row[0] <= end_ms)
        if oldest <= since_ms or len(batch) < request_limit:
            break
        if previous_oldest is not None and oldest >= previous_oldest:
            break
        previous_oldest = oldest
        cursor_end = oldest - 1
        if pause > 0:
            time.sleep(float(pause))
    dedup = {int(row[0]): row for row in out}
    return [dedup[timestamp] for timestamp in sorted(dedup)]
