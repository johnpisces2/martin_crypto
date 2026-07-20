import pandas as pd
import pytest

import martin
from market_data import alpaca, sources


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None, text=""):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise alpaca.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _ms(value):
    return int(pd.Timestamp(value).timestamp() * 1000)


def test_alpaca_credentials_accept_official_and_alias_names():
    assert alpaca.credentials_from_env(
        {"APCA_API_KEY_ID": "key", "APCA_API_SECRET_KEY": "secret"}
    ) == ("key", "secret")
    assert alpaca.credentials_from_env(
        {"ALPACA_API_KEY_ID": "key2", "ALPACA_API_SECRET_KEY": "secret2"}
    ) == ("key2", "secret2")
    with pytest.raises(ValueError, match="APCA_API_KEY_ID"):
        alpaca.credentials_from_env({})


def test_exported_credential_aliases_override_env_file_aliases(tmp_path):
    env_path = tmp_path / "alpaca.env"
    env_path.write_text(
        "APCA_API_KEY_ID=file-key\nAPCA_API_SECRET_KEY=file-secret\n",
        encoding="utf-8",
    )
    environ = {
        "ALPACA_API_KEY_ID": "exported-key",
        "ALPACA_API_SECRET_KEY": "exported-secret",
    }

    alpaca.load_env_file(env_path, environ)

    assert alpaca.credentials_from_env(environ) == (
        "exported-key",
        "exported-secret",
    )
    assert "APCA_API_KEY_ID" not in environ
    assert "APCA_API_SECRET_KEY" not in environ


@pytest.mark.parametrize(
    ("date_value", "expected_end"),
    [
        ("2026-03-08", "2026-03-08T23:59:59.999000-04:00"),
        ("2026-11-01", "2026-11-01T23:59:59.999000-05:00"),
    ],
)
def test_new_york_date_range_preserves_calendar_day_across_dst(
    date_value, expected_end
):
    start, end = alpaca.full_new_york_date_range(date_value, date_value)

    assert start.startswith(date_value)
    assert end == expected_end


def test_alpaca_fetches_and_deduplicates_paginated_bars(monkeypatch):
    calls = []
    pages = [
        FakeResponse({
            "bars": [{
                "t": "2026-01-05T14:30:00Z", "o": 100, "h": 102,
                "l": 99, "c": 101, "v": 1000,
            }],
            "next_page_token": "page-2",
        }),
        FakeResponse({
            "bars": [
                {
                    "t": "2026-01-05T14:30:00Z", "o": 100, "h": 102,
                    "l": 99, "c": 101, "v": 1000,
                },
                {
                    "t": "2026-01-05T14:45:00Z", "o": 101, "h": 103,
                    "l": 100, "c": 102, "v": 1200,
                },
            ],
            "next_page_token": None,
        }),
    ]

    def fake_get(url, headers, params, timeout):
        calls.append((url, headers, dict(params), timeout))
        return pages.pop(0)

    monkeypatch.setattr(alpaca.requests, "get", fake_get)
    rows = alpaca.fetch_ohlcv_segmented(
        "AAPL",
        "15m",
        _ms("2026-01-05T14:00:00Z"),
        _ms("2026-01-05T20:00:00Z"),
        api_key="key",
        secret_key="secret",
        feed="iex",
    )

    assert len(rows) == 2
    assert rows[-1][4] == 102.0
    assert calls[0][0].endswith("/v2/stocks/AAPL/bars")
    assert calls[0][2]["timeframe"] == "15Min"
    assert calls[0][2]["adjustment"] == "split"
    assert calls[0][2]["feed"] == "iex"
    assert calls[1][2]["page_token"] == "page-2"


def test_normalized_source_routes_alpaca_as_stock(monkeypatch):
    expected = [[1, 10, 11, 9, 10.5, 100]]
    monkeypatch.setattr(alpaca, "fetch_ohlcv_segmented", lambda *args, **kwargs: expected)

    symbol, market_type, rows = sources.fetch_source_ohlcv(
        "alpaca", "aapl", "4h", 0, 1000, 0
    )

    assert symbol == "AAPL"
    assert market_type == "stock"
    assert rows == expected


def test_martin_accepts_normal_us_market_gaps_for_alpaca(monkeypatch):
    rows = [
        [_ms("2025-01-03T14:30:00Z"), 100, 102, 99, 101, 1000],
        [_ms("2025-01-06T14:30:00Z"), 101, 103, 100, 102, 1200],
    ]
    monkeypatch.setattr(
        martin,
        "_fetch_source_ohlcv",
        lambda *args, **kwargs: ("AAPL", "stock", rows),
    )

    frame = martin.get_klines(
        symbol="AAPL",
        interval="4h",
        start="2025-01-01T00:00:00Z",
        end="2025-01-10T23:59:59Z",
        exch_list=["alpaca"],
        use_cache=False,
        allow_gaps=True,
        require_full_coverage=False,
    )

    assert len(frame) == 2
    assert frame.attrs["exchange"] == "alpaca"
    assert frame.attrs["market_type"] == "stock"
    assert frame.attrs["market"] == "stock:alpaca"
    assert frame.attrs["symbol"] == "AAPL"
    assert martin._cache_key("AAPL", "4h", "alpaca", "stock").startswith(
        "AAPL_USD_alpaca_stock_4h"
    )


def test_martin_distinguishes_empty_market_data_from_provider_failures(monkeypatch):
    monkeypatch.setattr(
        martin,
        "_fetch_source_ohlcv",
        lambda *args, **kwargs: ("AAPL", "stock", []),
    )

    with pytest.raises(martin.KlineDataUnavailableError, match="沒有 AAPL/USD"):
        martin.get_klines(
            symbol="AAPL",
            interval="15m",
            start="2026-01-05T00:00:00-05:00",
            end="2026-01-06T23:59:59.999000-05:00",
            exch_list=["alpaca"],
            use_cache=False,
            allow_gaps=True,
            require_full_coverage=False,
        )
