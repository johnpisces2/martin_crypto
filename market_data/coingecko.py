"""Read-only CoinGecko market-cap data client."""

from __future__ import annotations

import math
import time

import requests


COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_PAGE_SIZE = 250


def fetch_market_cap_ranks(
    limit: int = 250,
    *,
    session=None,
    timeout: float = 10.0,
    page_pause: float = 1.0,
) -> dict[str, int]:
    """Return ``{symbol_lowercase: market_cap_rank}`` for the top N coins.

    CoinGecko may contain duplicate symbols. The first occurrence is the one
    with the better market-cap position and is therefore retained.
    """
    requested = max(0, int(limit))
    if requested == 0:
        return {}

    client = session or requests
    mapping: dict[str, int] = {}
    try:
        total_pages = math.ceil(requested / COINGECKO_PAGE_SIZE)
        for page in range(1, total_pages + 1):
            remaining = requested - ((page - 1) * COINGECKO_PAGE_SIZE)
            per_page = min(remaining, COINGECKO_PAGE_SIZE)
            response = client.get(
                COINGECKO_MARKETS_URL,
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": per_page,
                    "page": page,
                    "sparkline": "false",
                },
                timeout=float(timeout),
            )
            if int(getattr(response, "status_code", 200)) != 200:
                return {}
            data = response.json()
            if not data:
                return {}

            for item in data:
                symbol = str(item.get("symbol") or "").lower()
                rank = item.get("market_cap_rank")
                if symbol and rank and symbol not in mapping:
                    mapping[symbol] = int(rank)

            if len(data) < per_page:
                break
            if page < total_pages and page_pause > 0:
                time.sleep(float(page_pause))
    except (requests.RequestException, TypeError, ValueError):
        return {}
    return mapping
