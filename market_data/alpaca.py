"""Alpaca historical US stock/ETF OHLCV adapter."""

from __future__ import annotations

import os
import math
import shlex
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from . import rate_limit


DEFAULT_DATA_URL = "https://data.alpaca.markets"
DEFAULT_FEED = "iex"
DEFAULT_ENV_PATH = Path("~/.config/workspace/alpaca.env")
ENV_FILE_KEYS = frozenset(
    {
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "ALPACA_DATA_FEED",
        "ALPACA_DATA_URL",
    }
)
TIMEFRAME_MAP = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "4h": "4Hour",
    "1d": "1Day",
}
DEFAULT_BARS_PER_TRADING_DAY = {
    "15m": 26.0,
    "1h": 7.0,
    "4h": 2.0,
    "1d": 1.0,
}
_CREDENTIAL_ALIASES = {
    "APCA_API_KEY_ID": ("APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    "ALPACA_API_KEY_ID": ("APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    "APCA_API_SECRET_KEY": (
        "APCA_API_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
    ),
    "ALPACA_API_SECRET_KEY": (
        "APCA_API_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
    ),
}


def load_env_file(path=None, environ=None) -> Path:
    """Load supported Alpaca settings without overriding exported values."""
    env = os.environ if environ is None else environ
    env_path = Path(path or DEFAULT_ENV_PATH).expanduser()
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Alpaca credential file was not found:\n"
            f"{env_path}\n\n"
            "Create the file with APCA_API_KEY_ID, APCA_API_SECRET_KEY, "
            "and optionally ALPACA_DATA_FEED=iex."
        ) from exc
    except OSError as exc:
        raise OSError(f"Could not read Alpaca credential file {env_path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(
                f"Invalid Alpaca credential file syntax at {env_path}:{line_number}: {exc}"
            ) from exc
        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        if len(parts) != 1 or "=" not in parts[0]:
            raise ValueError(
                f"Invalid Alpaca credential assignment at {env_path}:{line_number}. "
                "Use NAME=value or export NAME=value."
            )
        key, value = parts[0].split("=", 1)
        if key not in ENV_FILE_KEYS:
            continue
        aliases = _CREDENTIAL_ALIASES.get(key, (key,))
        if not any(str(env.get(alias) or "").strip() for alias in aliases):
            env[key] = value
    return env_path


def credentials_from_env(environ=None) -> tuple[str, str]:
    """Read Alpaca credentials without persisting secrets in the application."""
    env = os.environ if environ is None else environ
    key_id = str(
        env.get("APCA_API_KEY_ID") or env.get("ALPACA_API_KEY_ID") or ""
    ).strip()
    secret_key = str(
        env.get("APCA_API_SECRET_KEY") or env.get("ALPACA_API_SECRET_KEY") or ""
    ).strip()
    if not key_id or not secret_key:
        raise ValueError(
            "Alpaca API credentials are missing. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY before requesting market data."
        )
    return key_id, secret_key


def is_configured(environ=None) -> bool:
    try:
        credentials_from_env(environ)
    except ValueError:
        return False
    return True


def alpaca_timeframe(interval: str) -> str:
    try:
        return TIMEFRAME_MAP[str(interval).strip()]
    except KeyError as exc:
        supported = ", ".join(TIMEFRAME_MAP)
        raise ValueError(
            f"Alpaca does not support interval {interval!r}; supported: {supported}"
        ) from exc


def full_new_york_date_range(start, end) -> tuple[str, str]:
    """Return inclusive RFC-3339 bounds for New York calendar dates."""

    def as_new_york_midnight(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("America/New_York")
        else:
            timestamp = timestamp.tz_convert("America/New_York")
        return timestamp.normalize()

    start_ts = as_new_york_midnight(start)
    end_ts = as_new_york_midnight(end)
    next_date = end_ts.date() + timedelta(days=1)
    next_midnight = pd.Timestamp(next_date).tz_localize("America/New_York")
    inclusive_end = next_midnight - pd.Timedelta(milliseconds=1)
    return start_ts.isoformat(), inclusive_end.isoformat()


def bars_per_trading_day(df: pd.DataFrame | None, interval: str) -> float:
    """Estimate observed bars/session, falling back to regular US hours."""
    fallback = DEFAULT_BARS_PER_TRADING_DAY.get(str(interval), 1.0)
    if df is None or df.empty or "time" not in df:
        return fallback
    index = pd.DatetimeIndex(df["time"])
    if index.tz is None:
        index = index.tz_localize("UTC")
    local_dates = pd.Series(index.tz_convert("America/New_York").date)
    counts = local_dates.value_counts()
    if len(counts) >= 3:
        counts = counts.drop(
            labels=[local_dates.iloc[0], local_dates.iloc[-1]], errors="ignore"
        )
    if counts.empty:
        return fallback
    observed = float(counts.median())
    return observed if math.isfinite(observed) and observed > 0 else fallback


def _rfc3339_from_ms(timestamp_ms: int) -> str:
    value = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def fetch_ohlcv_segmented(
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int,
    pause: float = 0.0,
    *,
    api_key: str | None = None,
    secret_key: str | None = None,
    feed: str | None = None,
    data_url: str | None = None,
    max_retries: int = 5,
):
    """Fetch all Alpaca bar pages and normalize them to CCXT-style rows."""
    if not api_key or not secret_key:
        api_key, secret_key = credentials_from_env()

    ticker = str(symbol or "").strip().upper()
    if not ticker:
        raise ValueError("Alpaca stock symbol is empty")

    selected_feed = str(feed or os.getenv("ALPACA_DATA_FEED") or DEFAULT_FEED).strip()
    base_url = str(data_url or os.getenv("ALPACA_DATA_URL") or DEFAULT_DATA_URL).rstrip("/")
    url = f"{base_url}/v2/stocks/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": str(api_key),
        "APCA-API-SECRET-KEY": str(secret_key),
        "Accept": "application/json",
    }
    params = {
        "timeframe": alpaca_timeframe(timeframe),
        "start": _rfc3339_from_ms(since_ms),
        "end": _rfc3339_from_ms(end_ms),
        "adjustment": "split",
        "feed": selected_feed,
        "sort": "asc",
        "limit": 10_000,
    }

    output = []
    page_token = None
    seen_tokens = set()
    while True:
        request_params = dict(params)
        if page_token:
            request_params["page_token"] = page_token

        response = None
        for attempt in range(int(max_retries) + 1):
            try:
                rate_limit.acquire("alpaca")
                response = requests.get(
                    url, headers=headers, params=request_params, timeout=30
                )
            except requests.RequestException as exc:
                if attempt >= int(max_retries):
                    raise RuntimeError(f"Alpaca network request failed: {exc}") from exc
                rate_limit.defer("alpaca", min(2 ** attempt, 16))
                continue

            rate_limit.observe_response("alpaca", response)
            if response.status_code == 429 and attempt < int(max_retries):
                rate_limit.defer_from_response("alpaca", response, attempt)
                continue
            break

        if response is None:
            raise RuntimeError("Alpaca request did not return a response")
        if response.status_code in (401, 403):
            raise RuntimeError(
                "Alpaca authentication/feed access failed. Check the API keys and "
                f"ALPACA_DATA_FEED={selected_feed!r}."
            )
        if response.status_code == 429:
            raise RuntimeError("Alpaca HTTP 429 Too Many Requests")
        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            body = str(getattr(response, "text", ""))[:300]
            raise RuntimeError(f"Alpaca bars request failed: {body or exc}") from exc

        for bar in payload.get("bars") or []:
            try:
                timestamp_ms = int(pd.Timestamp(bar["t"]).timestamp() * 1000)
                row = [
                    timestamp_ms,
                    float(bar["o"]),
                    float(bar["h"]),
                    float(bar["l"]),
                    float(bar["c"]),
                    float(bar.get("v", 0.0)),
                ]
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"Alpaca returned an invalid bar: {bar!r}") from exc
            if int(since_ms) <= timestamp_ms <= int(end_ms):
                output.append(row)

        next_token = payload.get("next_page_token")
        if not next_token:
            break
        if next_token in seen_tokens:
            raise RuntimeError("Alpaca pagination returned a repeated page token")
        seen_tokens.add(next_token)
        page_token = next_token
        if pause > 0:
            time.sleep(float(pause))

    dedup = {row[0]: row for row in output}
    return [dedup[timestamp] for timestamp in sorted(dedup)]
