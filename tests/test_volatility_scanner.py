import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

import martin
from market_data import alpaca, coingecko, rate_limit, sources, universe
from volatility_scanner_gui import (
    DATA_FILTERED_FLAG,
    DATA_FILTERED_VERDICT,
    VolatilityScannerGUI,
    add_martin_fit_columns,
    initialize_alpaca_environment,
    martin_fit_from_row,
    martin_recovery_metrics,
    make_data_filtered_result,
    optional_millions,
    source_date_range,
    stock_bars_per_trading_day,
)


class _CapturedSignal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values[0] if len(values) == 1 else values)


class _CapturedSignals:
    def __init__(self):
        self.warning = _CapturedSignal()
        self.log = _CapturedSignal()
        self.progress = _CapturedSignal()
        self.stopped = _CapturedSignal()


class _ProcessCoinHarness:
    def __init__(self):
        self.stop_event = threading.Event()
        self.cached_df = None

    def _get_cached_chart_df(self, *_args):
        return self.cached_df

    def _store_chart_df(self, *_args):
        return None


@pytest.fixture(scope="module")
def scanner_window(keep_qapplication_alive):
    window = VolatilityScannerGUI()
    yield window
    window.close()
    window.deleteLater()
    keep_qapplication_alive.processEvents()


def _fit_row(**overrides):
    row = {
        "ATR(M)%": 6.0,
        "RV(A)%": 150.0,
        "ER": 0.05,
        "Chg%": 0.0,
        "MaxDD%": -35.0,
        "MaxRed": 5,
        "MaxDown": 5,
        "Recovery%": 90.0,
        "Recovery Events": 10,
        "Cycles/30D": 3.0,
        "Recent90%": 0.0,
        "CurrentDD%": -5.0,
    }
    row.update(overrides)
    return row


def test_blank_volume_threshold_means_no_limit():
    assert optional_millions("") == 0.0
    assert optional_millions("   ") == 0.0
    assert optional_millions("12.5") == 12_500_000.0


def test_recovery_metrics_reward_repeated_returns_to_origin():
    cycle = np.array([100.0, 97.0, 94.0, 97.0, 100.0, 103.0, 100.0])
    result = martin_recovery_metrics(
        pd.Series(np.tile(cycle, 12)),
        bars_per_day=6,
        drop_pct=0.03,
        max_recovery_days=30,
    )

    assert result["events"] >= 10
    assert result["success_rate"] == 1.0
    assert result["cycles_per_30d"] > 3.0


def test_recovery_metrics_reject_persistent_slow_decline():
    closes = pd.Series(100.0 * (0.997 ** np.arange(240)))
    result = martin_recovery_metrics(
        closes,
        bars_per_day=6,
        drop_pct=0.03,
        max_recovery_days=30,
    )

    assert result["events"] >= 1
    assert result["successes"] == 0
    assert result["success_rate"] == 0.0


def test_martin_fit_gives_clear_verdicts_for_good_and_bad_behavior():
    good = martin_fit_from_row(_fit_row())
    bad = martin_fit_from_row(
        _fit_row(
            **{
                "ER": 0.40,
                "Chg%": -70.0,
                "Recovery%": 20.0,
                "Recent90%": -35.0,
                "CurrentDD%": -50.0,
                "MaxDown": 25,
            }
        )
    )

    assert good["Verdict"] == "Suitable"
    assert good["Martin Score"] >= 75
    assert good["Downtrend Risk"] == "Low"
    assert bad["Verdict"] == "Unsuitable"
    assert bad["Martin Score"] < 50
    assert bad["Downtrend Risk"] == "High"


def test_low_volatility_is_unsuitable_without_being_called_a_downtrend():
    result = martin_fit_from_row(_fit_row(**{"ATR(M)%": 0.5, "RV(A)%": 15.0}))

    assert result["Verdict"] == "Unsuitable"
    assert result["Downtrend Risk"] == "Low"
    assert "Insufficient volatility" in result["Reason"]


def test_add_martin_fit_columns_keeps_raw_metrics_for_advanced_view():
    frame = pd.DataFrame([_fit_row(Symbol="TEST/USDT")])
    result = add_martin_fit_columns(frame)

    assert result.loc[0, "Symbol"] == "TEST/USDT"
    assert result.loc[0, "Verdict"] == "Suitable"
    assert "Martin Score" in result.columns
    assert "Reason" in result.columns


def test_data_filtered_rows_keep_reason_instead_of_receiving_a_fit_score():
    row = make_data_filtered_result(
        {"symbol": "SHORT", "is_stock": True, "asset_type": "US Stock/ETF"},
        "Only 12 K-lines available; minimum required is 50.",
    )

    result = add_martin_fit_columns(pd.DataFrame([row]))

    assert result.loc[0, DATA_FILTERED_FLAG]
    assert result.loc[0, "Verdict"] == DATA_FILTERED_VERDICT
    assert pd.isna(result.loc[0, "Martin Score"])
    assert result.loc[0, "Reason"].startswith("Only 12 K-lines")


def _scanner_test_frame(periods: int, freq: str) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(periods) / 4.0)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC"),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.ones(periods),
    })


def _scanner_process_args(interval: str, requested_days: int, min_avg_vol=0.0):
    return {
        "interval": interval,
        "start_str": "2025-01-01",
        "end_str": "2025-12-31",
        "exch_name": "binance",
        "quote_asset": "USDT",
        "requested_days": requested_days,
        "min_avg_vol": min_avg_vol,
    }


def test_process_coin_returns_diagnostic_rows_for_each_data_filter():
    scanner = _ProcessCoinHarness()
    candidate = {
        "symbol": "TEST/USDT", "is_stock": False, "is_manual": False,
        "asset_type": "Crypto Spot", "mc_rank": 999,
    }

    short_df = _scanner_test_frame(12, "1h")
    scanner.cached_df = short_df
    short = VolatilityScannerGUI.process_coin(
        scanner, candidate, _scanner_process_args("1h", 1)
    )
    assert short[DATA_FILTERED_FLAG]
    assert "Only 12 K-lines" in short["Reason"]

    short_history_df = _scanner_test_frame(100, "1h")
    scanner.cached_df = short_history_df
    history = VolatilityScannerGUI.process_coin(
        scanner, candidate, _scanner_process_args("1h", 365)
    )
    assert history[DATA_FILTERED_FLAG]
    assert "History coverage" in history["Reason"]
    assert "80%" in history["Reason"]

    low_volume_df = _scanner_test_frame(100, "1D")
    scanner.cached_df = low_volume_df
    volume = VolatilityScannerGUI.process_coin(
        scanner,
        candidate,
        _scanner_process_args("1d", 100, min_avg_vol=1_000_000.0),
    )
    assert volume[DATA_FILTERED_FLAG]
    assert "Average daily volume" in volume["Reason"]
    assert "configured minimum 1.00M" in volume["Reason"]


def test_process_coin_lists_empty_provider_history_as_data_filtered(monkeypatch):
    scanner = _ProcessCoinHarness()
    candidate = {
        "symbol": "EMPTY",
        "is_stock": True,
        "is_manual": False,
        "asset_type": "US Stock/ETF",
        "mc_rank": -1,
    }
    args = {
        "interval": "15m",
        "start_str": "2026-01-01",
        "end_str": "2026-02-01",
        "exch_name": "alpaca",
        "quote_asset": "USD",
        "requested_days": 32,
        "min_avg_vol": 0.0,
    }
    monkeypatch.setattr(
        martin,
        "get_klines",
        lambda **kwargs: (_ for _ in ()).throw(
            martin.KlineDataUnavailableError("no bars for requested period")
        ),
    )

    result = VolatilityScannerGUI.process_coin(scanner, candidate, args)

    assert result[DATA_FILTERED_FLAG]
    assert "No K-line data" in result["Reason"]
    assert "no bars for requested period" in result["Reason"]


def test_candidate_scan_lists_filtered_rows_and_counts_them_separately(
    monkeypatch, scanner_window
):
    window = scanner_window
    signals = _CapturedSignals()
    candidates = [
        {"symbol": "GOOD", "is_stock": True},
        {"symbol": "SHORT", "is_stock": True},
        {"symbol": "BROKEN", "is_stock": True},
    ]

    def fake_process(candidate, _args):
        if candidate["symbol"] == "GOOD":
            return _fit_row(Symbol="GOOD", Asset="US Stock/ETF")
        if candidate["symbol"] == "SHORT":
            return make_data_filtered_result(candidate, "Only 8 K-lines available.")
        raise RuntimeError("provider failure")

    monkeypatch.setattr(window, "process_coin", fake_process)
    config = {
        "exch_name": "alpaca", "interval": "1h",
        "start_str": "2025-01-01", "end_str": "2025-12-31",
        "requested_days": 365, "min_avg_vol": 0.0,
    }
    result = window._execute_candidate_scan(signals, config, candidates, "USD")

    assert len(result) == 2
    assert set(result["Verdict"]) == {"Suitable", DATA_FILTERED_VERDICT}
    assert result.attrs["scan_stats"] == {
        "selected": 3,
        "results": 1,
        "listed_rows": 2,
        "data_filtered": 1,
        "errors": 1,
        "rate_limited": 0,
        "workers": 8,
    }

    window.scan_results = result
    window.update_table()
    window.cb_result_filter.setCurrentText("Data Filtered")
    assert len(window.display_results) == 1
    assert window.display_results.iloc[0]["Symbol"] == "SHORT"
    assert "Only 8 K-lines" in window.display_results.iloc[0]["Reason"]
    assert "Data Filtered 1" in window.lbl_result_summary.text()

def test_binance_tokenized_equities_are_excluded_without_crypto_false_positives():
    for base in ("SNDKB", "NVDAB", "MUB", "BABAB", "RKLBB"):
        assert base in universe.BINANCE_TOKENIZED_EQUITY_BASES
    for base in (
        "BNB", "ARB", "SHIB", "DGB", "TRB", "BB", "YB",
        "ACX", "ADX", "FRAX", "SNDKX",
    ):
        assert base not in universe.BINANCE_TOKENIZED_EQUITY_BASES

    assert universe.is_binance_crypto_spot_market(
        {"base": "BTC", "quote": "USDT", "active": True, "spot": True}
    )
    assert not universe.is_binance_crypto_spot_market(
        {"base": "SNDKB", "quote": "USDT", "active": True, "spot": True}
    )


def test_binance_source_resolves_crypto_spot_only():
    markets = {
        "BTC/USDT": {
            "symbol": "BTC/USDT", "base": "BTC", "quote": "USDT",
            "active": True, "spot": True, "swap": False,
        },
        "ETH/USDT:USDT": {
            "symbol": "ETH/USDT:USDT", "base": "ETH", "quote": "USDT",
            "active": True, "spot": False, "swap": True, "linear": True,
        },
        "SNDKB/USDT": {
            "symbol": "SNDKB/USDT", "base": "SNDKB", "quote": "USDT",
            "active": True, "spot": True, "swap": False,
        },
    }

    assert sources.find_market_symbol(markets, "BTC") == ("BTC/USDT", "spot")
    assert sources.find_market_symbol(markets, "ETH") == (None, None)
    assert sources.find_market_symbol(markets, "SNDKB") == (None, None)

    with pytest.raises(ValueError, match="Unsupported market-data source"):
        sources.fetch_source_ohlcv("pionex", "BTC", "1h", 0, 1_000, 0)


def test_scanner_exposes_binance_and_alpaca_modes(scanner_window):
    window = scanner_window

    assert window.cb_exchange.count() == 2
    assert window.cb_exchange.currentText() == "binance"
    assert window.cb_asset_universe.currentText() == "Crypto Spot"
    assert not window.cb_asset_universe.isEnabled()
    assert window.e_min_vol.isEnabled()
    assert window.e_min_avg_vol.isEnabled()

    window.cb_exchange.setCurrentText("alpaca")
    assert window.cb_asset_universe.currentText() == "US Stocks & ETFs"
    assert not window.cb_asset_universe.isEnabled()
    assert not window.e_min_vol.isEnabled()
    assert window.e_min_avg_vol.isEnabled()
    assert not window.chk_mc_filter.isEnabled()
    assert str(len(universe.ALPACA_MAINSTREAM_STOCK_SYMBOLS)) in window.stock_hint.text()

def test_alpaca_mainstream_universe_is_broad_and_unique():
    symbols = universe.ALPACA_MAINSTREAM_STOCK_SYMBOLS

    assert len(symbols) >= 100
    assert len(symbols) == len(set(symbols))
    for symbol in (
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "LLY",
        "SNDK", "CRCL", "RKLB", "STRC", "SPY", "QQQ", "SOXL",
    ):
        assert symbol in symbols
    assert symbols.index("STRC") < 50


def test_alpaca_candidate_limit_prioritizes_manual_then_mainstream_symbols(
    scanner_window,
):
    window = scanner_window
    signals = _CapturedSignals()

    candidates = window._build_alpaca_candidates(
        signals,
        {"top_n": 4, "manual_input": "SNDK, ZZZZ"},
    )

    assert [candidate["symbol"] for candidate in candidates] == [
        "SNDK", "ZZZZ", "AAPL", "MSFT",
    ]
    assert all(candidate["is_stock"] for candidate in candidates)
    assert candidates[0]["is_manual"]
    assert not candidates[2]["is_manual"]

def test_alpaca_dates_use_new_york_calendar_and_stock_sessions_are_inferred():
    start, end = source_date_range("alpaca", "2026-01-05", "2026-01-09")
    assert start == "2026-01-05T00:00:00-05:00"
    assert end == "2026-01-09T23:59:59.999000-05:00"
    assert martin._to_timestamp(end).isoformat() == "2026-01-10T12:59:59.999000+08:00"

    times = pd.to_datetime(
        [
            "2026-01-05T14:30:00Z", "2026-01-05T18:30:00Z",
            "2026-01-06T14:30:00Z", "2026-01-06T18:30:00Z",
            "2026-01-07T14:30:00Z", "2026-01-07T18:30:00Z",
            "2026-01-08T14:30:00Z", "2026-01-08T18:30:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame({"time": times})
    assert stock_bars_per_trading_day(frame, "4h") == 2.0

    _, spring_end = source_date_range("alpaca", "2026-03-08", "2026-03-08")
    _, fall_end = source_date_range("alpaca", "2026-11-01", "2026-11-01")
    assert spring_end == "2026-03-08T23:59:59.999000-04:00"
    assert fall_end == "2026-11-01T23:59:59.999000-05:00"


def test_load_alpaca_env_file_accepts_shell_style_assignments(tmp_path):
    env_path = tmp_path / "alpaca.env"
    env_path.write_text(
        "# Alpaca market-data credentials\n"
        'export APCA_API_KEY_ID="key-id"\n'
        "APCA_API_SECRET_KEY='secret-key'\n"
        "ALPACA_DATA_FEED=iex # free feed\n",
        encoding="utf-8",
    )
    environ = {}

    result = alpaca.load_env_file(env_path, environ)

    assert result == env_path
    assert environ == {
        "APCA_API_KEY_ID": "key-id",
        "APCA_API_SECRET_KEY": "secret-key",
        "ALPACA_DATA_FEED": "iex",
    }
    assert initialize_alpaca_environment(env_path, environ) is None


def test_load_alpaca_env_file_does_not_override_exported_values(tmp_path):
    env_path = tmp_path / "alpaca.env"
    env_path.write_text(
        "APCA_API_KEY_ID=file-key\nAPCA_API_SECRET_KEY=file-secret\n",
        encoding="utf-8",
    )
    environ = {
        "APCA_API_KEY_ID": "exported-key",
        "APCA_API_SECRET_KEY": "exported-secret",
    }

    alpaca.load_env_file(env_path, environ)

    assert environ["APCA_API_KEY_ID"] == "exported-key"
    assert environ["APCA_API_SECRET_KEY"] == "exported-secret"


def test_missing_alpaca_env_file_returns_actionable_message(tmp_path):
    env_path = tmp_path / "missing.env"

    message = initialize_alpaca_environment(env_path, {})

    assert str(env_path) in message
    assert "was not found" in message
    assert "APCA_API_KEY_ID" in message


def test_invalid_alpaca_env_file_reports_line_number(tmp_path):
    env_path = tmp_path / "alpaca.env"
    env_path.write_text('APCA_API_KEY_ID="unterminated\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"alpaca\.env:1"):
        alpaca.load_env_file(env_path, {})


class _FakeRateClock:
    def __init__(self, wall_time=1_000.0):
        self.now = 0.0
        self.wall_time = float(wall_time)

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall_time + self.now

    def sleep(self, seconds):
        self.now += float(seconds)


def test_global_rate_limiter_keeps_provider_budgets_independent():
    clock = _FakeRateClock()
    registry = rate_limit.RateLimiterRegistry(
        policies={
            "alpha": rate_limit.RateLimitPolicy(1, 10),
            "beta": rate_limit.RateLimitPolicy(1, 30),
        },
        clock=clock.monotonic,
        wall_clock=clock.time,
        sleeper=clock.sleep,
        environ={},
    )

    registry.acquire("alpha")
    registry.acquire("beta")
    assert clock.now == 0.0

    registry.acquire("alpha")
    assert clock.now == 10.0


def test_global_rate_limiter_applies_weight_spacing_and_shared_deferral():
    clock = _FakeRateClock()
    registry = rate_limit.RateLimiterRegistry(
        policies={"weighted": rate_limit.RateLimitPolicy(100, 60, 0.5)},
        clock=clock.monotonic,
        wall_clock=clock.time,
        sleeper=clock.sleep,
        environ={},
    )

    registry.acquire("weighted", weight=2)
    registry.acquire("weighted")
    assert clock.now == 1.0

    registry.defer("weighted", 5.0)
    registry.acquire("weighted")
    assert clock.now == 6.0


def test_global_rate_limiter_serializes_concurrent_workers():
    registry = rate_limit.RateLimiterRegistry(
        policies={"shared": rate_limit.RateLimitPolicy(1, 0.04)},
        environ={},
    )
    barrier = threading.Barrier(3)
    acquired_at = []

    def worker():
        barrier.wait()
        registry.acquire("shared")
        acquired_at.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert len(acquired_at) == 2
    assert max(acquired_at) - min(acquired_at) >= 0.03


def test_rate_limiter_honors_reset_header_and_environment_overrides():
    clock = _FakeRateClock(wall_time=1_800_000_000.0)
    registry = rate_limit.RateLimiterRegistry(
        policies={"alpaca": rate_limit.RateLimitPolicy(180, 60)},
        clock=clock.monotonic,
        wall_clock=clock.time,
        sleeper=clock.sleep,
        environ={
            "MARTIN_RATE_LIMIT_ALPACA_LIMIT": "150",
            "MARTIN_RATE_LIMIT_ALPACA_MIN_INTERVAL": "0.2",
        },
    )
    response = type(
        "Response",
        (),
        {"headers": {"X-RateLimit-Reset": "1800000005"}},
    )()

    assert registry.policy("alpaca").limit == 150
    assert registry.policy("alpaca").min_interval == 0.2
    delay = registry.defer_from_response("alpaca", response, attempt=0)
    assert delay == pytest.approx(5.1)
    registry.acquire("alpaca")
    assert clock.now == pytest.approx(5.1)

    no_headers = type("Response", (), {"headers": {}})()
    assert registry.retry_delay("alpaca", no_headers, attempt=0) == 60.0


def test_all_api_clients_route_through_the_shared_provider_registry(monkeypatch):
    acquisitions = []
    monkeypatch.setattr(
        rate_limit,
        "acquire",
        lambda provider, weight=1.0: acquisitions.append((provider, weight)),
    )
    monkeypatch.setattr(rate_limit, "observe_response", lambda *args: None)

    class Response:
        status_code = 200
        headers = {}
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr(
        alpaca.requests,
        "get",
        lambda *args, **kwargs: Response({"bars": [], "next_page_token": None}),
    )
    alpaca.fetch_ohlcv_segmented(
        "AAPL", "15m", 0, 1_000, api_key="key", secret_key="secret"
    )

    class CoinGeckoSession:
        def get(self, *args, **kwargs):
            return Response([{"symbol": "btc", "market_cap_rank": 1}])

    assert coingecko.fetch_market_cap_ranks(limit=1, session=CoinGeckoSession())

    exchange = type("Exchange", (), {"id": "binance"})()
    assert sources.call_ccxt(exchange, lambda: "ok", weight=2) == "ok"

    assert acquisitions == [
        ("alpaca", 1.0),
        ("coingecko", 1.0),
        ("binance", 2),
    ]
