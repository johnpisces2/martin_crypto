"""Asset classification and scanner-universe policies."""

from __future__ import annotations

import re


# Exclusion-only denylist. Binance tokenized equities are never exposed as
# supported symbols; keeping their exact bases here prevents manual entry or
# market discovery from treating them as cryptocurrencies. Explicit symbols
# avoid false positives for real crypto assets ending in ``B`` such as BNB.
BINANCE_TOKENIZED_EQUITY_BASES = frozenset({
    "AAOIB", "AMDB", "ARMB", "AVGOB", "BABAB", "CBRSB", "COINB",
    "CRCLB", "DRAMB", "EWYB", "GLWB", "GOOGLB", "HOODB", "IBMB",
    "INTCB", "LITEB", "METAB", "MRVLB", "MSFTB", "MSTRB", "MUB",
    "NBISB", "NOKB", "NVDAB", "PLTRB", "QCOMB", "QQQB", "RKLBB",
    "SKHYB", "SNDKB", "SOXLB", "SPCXB", "SPYB", "TSLAB", "TSMB",
    "WDCB",
})

# Ordered by broad US-market relevance so the scanner's default limit selects
# liquid, widely followed names first.
ALPACA_MAINSTREAM_STOCK_SYMBOLS = tuple(dict.fromkeys((
    # Default top 50: liquid US stocks and ETFs relevant to this workflow.
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "TSM", "ARM", "QCOM", "MU", "INTC", "SNDK", "WDC", "JPM",
    "BAC", "GS", "MS", "V", "MA", "LLY", "UNH", "JNJ", "ABBV", "MRK",
    "COST", "WMT", "HD", "MCD", "XOM", "CVX", "ORCL", "NFLX", "CRM",
    "ADBE", "PLTR", "COIN", "HOOD", "MSTR", "STRC", "CRCL", "RKLB", "SPY",
    "QQQ", "SOXX", "SMH", "SOXL", "GLD",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK.B",
    "AVGO", "JPM", "LLY", "V", "XOM", "UNH", "MA", "COST", "WMT",
    "ORCL", "NFLX", "HD", "PG", "JNJ", "ABBV", "BAC", "CRM", "KO",
    "AMD", "MRK", "CVX", "PLTR", "CSCO", "PEP", "ADBE", "TMO", "MCD",
    "GE", "IBM", "WFC", "QCOM", "AMGN", "TXN", "INTU", "CAT", "NOW",
    "ISRG", "GS", "DIS", "UBER", "RTX", "BKNG", "SPGI", "LOW", "AXP",
    "BLK", "SCHW", "C", "MS", "VZ", "T", "PM", "MO", "NKE", "SBUX",
    "TGT", "CMCSA", "PFE", "GILD", "MDT", "SYK", "BA", "LMT", "DE",
    "UPS", "FDX", "COP", "SLB", "NEE", "DUK",
    "ARM", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MU", "INTC", "MRVL",
    "DELL", "ANET", "PANW", "CRWD", "SNOW", "APP", "SHOP", "ABNB",
    "DASH", "PYPL", "XYZ", "COIN", "HOOD", "MSTR", "STRC", "RDDT", "RBLX",
    "ROKU", "SOFI", "CRCL", "SNDK", "WDC", "RKLB", "ASTS", "NBIS",
    "AAOI", "LITE", "GLW", "BABA", "NOK",
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "SOXX", "SMH", "XLK",
    "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "TLT", "IEF",
    "GLD", "SLV", "ARKK", "SOXL", "TQQQ", "SQQQ", "EWY",
)))

STABLECOIN_BASES = frozenset({
    "USDT", "USDC", "USDD", "TUSD", "BUSD", "DAI", "FRAX", "FDUSD",
    "USDP", "GUSD", "PYUSD", "USDS", "USDE", "USD1", "USDI", "USDJ",
    "USDK", "USDX", "USTC", "EUR", "EURT", "EURS", "GBP", "GBPT",
    "USDR", "USDN", "USDB",
})


def is_stablecoin_base(base: str) -> bool:
    base_u = str(base or "").strip().upper()
    if not base_u:
        return False
    return (
        base_u in STABLECOIN_BASES
        or bool(re.match(r"^USD[A-Z0-9]{0,4}$", base_u))
        or base_u.endswith("USD")
    )


def is_binance_crypto_spot_market(market: dict, quote: str = "USDT") -> bool:
    """Allow active Binance crypto spot markets and reject tokenized equities."""
    quote_u = str(quote or "").strip().upper()
    if not market:
        return False
    base = str(market.get("base") or "").strip().upper()
    return bool(
        base
        and base not in BINANCE_TOKENIZED_EQUITY_BASES
        and market.get("active", True)
        and market.get("spot")
        and str(market.get("quote") or "").strip().upper() == quote_u
    )


def scanner_rank_filter_allows(
    rank: int,
    max_rank: int,
    *,
    enabled: bool,
    mapping_available: bool,
) -> bool:
    """Apply CoinGecko rank to Binance crypto when rank data is available."""
    if not enabled or not mapping_available:
        return True
    return 0 < int(rank) <= int(max_rank)
