"""Asset classification and scanner-universe policies."""

from __future__ import annotations

import re


# Real crypto bases whose ticker happens to end in X. Pionex public symbol
# metadata has no asset category, so these protect the xStock suffix heuristic.
KNOWN_CRYPTO_X_BASES = frozenset({
    "AVAX", "BEAMX", "CFX", "CVX", "DYDX", "FLUX", "GMX", "HTX",
    "ICX", "IDEX", "IMX", "IOTX", "MBOX", "PIVX", "POLYX", "PUNDIX",
    "SNX", "STRAX", "STX", "TRX", "ZRX",
})

STABLECOIN_BASES = frozenset({
    "USDT", "USDC", "USDD", "TUSD", "BUSD", "DAI", "FRAX", "FDUSD",
    "USDP", "GUSD", "PYUSD", "USDS", "USDE", "USD1", "USDI", "USDJ",
    "USDK", "USDX", "USTC", "EUR", "EURT", "EURS", "GBP", "GBPT",
    "USDR", "USDN", "USDB",
})


def is_probable_stock_token(base: str, known_crypto_symbols=None) -> bool:
    """Best-effort xStock/Ondo-style classification from public metadata.

    CoinGecko symbols are intentionally not used as exclusions because it also
    lists xStocks such as TSLAX, SPYX and NVDAX. ``known_crypto_symbols`` is
    retained only for compatibility with earlier callers.
    """
    base_u = str(base or "").strip().upper()
    return base_u.endswith("X") and base_u not in KNOWN_CRYPTO_X_BASES


def is_stablecoin_base(base: str) -> bool:
    base_u = str(base or "").strip().upper()
    if not base_u:
        return False
    return (
        base_u in STABLECOIN_BASES
        or bool(re.match(r"^USD[A-Z0-9]{0,4}$", base_u))
        or base_u.endswith("USD")
    )


def listed_stock_token_bases(markets: dict, quote: str = "USDT") -> list[str]:
    """Return active spot stock-token bases from normalized market metadata."""
    quote_u = str(quote or "").strip().upper()
    bases = {
        str(market.get("base") or "").strip().upper()
        for market in (markets or {}).values()
        if market
        and market.get("active", True)
        and market.get("spot")
        and str(market.get("quote") or "").strip().upper() == quote_u
        and is_probable_stock_token(market.get("base"))
    }
    bases.discard("")
    return sorted(bases)


def bypass_scanner_volume_filters(exchange_name: str) -> bool:
    """Pionex volume is not reliable enough for scanner minimum thresholds."""
    return str(exchange_name or "").strip().lower() == "pionex"


def scanner_rank_filter_allows(
    rank: int,
    max_rank: int,
    *,
    enabled: bool,
    mapping_available: bool,
    is_stock_token: bool,
) -> bool:
    """Apply CoinGecko rank only to Crypto when rank data is available."""
    if not enabled or not mapping_available or is_stock_token:
        return True
    return 0 < int(rank) <= int(max_rank)
