"""Market-data provider clients and source adapters.

Provider-specific HTTP behavior belongs in this package. Strategy, caching and
GUI modules consume the normalized interfaces without knowing endpoint details.
"""

__all__ = ["coingecko", "pionex", "sources", "universe"]
