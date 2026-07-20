import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, Qt

import martin
from market_data import alpaca
from martin_gui import (
    ALPACA_DEFAULT_SYMBOLS,
    CRYPTO_SYMBOLS,
    DATA_SOURCES,
    MartinGUI,
    backtest_accounting_summary,
    configure_datetime_axis,
)


@pytest.fixture(scope="module")
def martin_window(keep_qapplication_alive):
    window = MartinGUI()
    yield window
    window.close()
    window.deleteLater()
    keep_qapplication_alive.processEvents()


def test_backtest_chart_uses_a_datetime_formatter():
    times = pd.date_range("2025-07-19", "2026-07-19", periods=100, tz="Asia/Taipei")
    figure = Figure()
    axis = figure.add_subplot(111)
    line, = axis.plot([], [])
    line.set_data(times, np.linspace(1_000.0, 2_000.0, len(times)))

    configure_datetime_axis(axis, times)
    axis.relim()
    axis.autoscale_view()

    assert isinstance(axis.xaxis.get_major_locator(), mdates.AutoDateLocator)
    assert isinstance(axis.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)


def test_backtest_accounting_reconciles_realized_and_open_pnl():
    result = {
        "trades_log": [{"pnl": 35.0}, {"pnl": 15.0}],
        "open_trade": {"pnl": -20.0},
        "equity_curve": [1_000.0, 1_030.0],
    }

    summary = backtest_accounting_summary(result, capital=1_000.0)

    assert summary["closed_pnl"] == 50.0
    assert summary["open_pnl"] == -20.0
    assert summary["net_pnl"] == 30.0
    assert summary["final_equity"] == 1_030.0


def test_backtest_accounting_rejects_inconsistent_performance():
    result = {
        "trades_log": [{"pnl": 50.0}],
        "open_trade": None,
        "equity_curve": [1_000.0, 1_500.0],
    }

    with pytest.raises(ValueError, match="accounting mismatch"):
        backtest_accounting_summary(result, capital=1_000.0)


def test_performance_always_uses_original_capital_for_tiny_first_bar_fee_loss():
    capital = 1_000.0
    final_equity = 14_541.378841581906
    times = pd.to_datetime(["2025-01-01", "2026-01-01"], utc=True).tz_convert("Asia/Taipei")

    performance = martin.compute_performance_metrics(
        equity_curve=[999.9995, final_equity],
        time_index=times,
        trades_log=[],
        capital=capital,
    )

    assert performance["total_return"] == pytest.approx(
        final_equity / capital - 1.0,
        rel=1e-12,
    )


def test_gui_date_inputs_cannot_select_future_dates(martin_window):
    window = martin_window
    today = QDate.currentDate()

    for date_edit in (
        window.e_start,
        window.e_end,
        window.m_start,
        window.m_end,
        window.s_start,
        window.s_end,
    ):
        assert date_edit.maximumDate() == today
        date_edit.setDate(today.addYears(1))
        assert date_edit.date() == today


def test_gui_source_switches_between_crypto_and_50_scrollable_stocks(
    monkeypatch, martin_window
):
    monkeypatch.setattr(alpaca, "load_env_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        alpaca, "credentials_from_env", lambda *args, **kwargs: ("key", "secret")
    )
    window = martin_window

    assert DATA_SOURCES == ["binance", "alpaca"]
    assert martin._EXCH_LIST == ["binance"]
    assert "BTC" in CRYPTO_SYMBOLS
    assert "SNDKB" not in CRYPTO_SYMBOLS
    assert len(ALPACA_DEFAULT_SYMBOLS) == 50
    assert len(set(ALPACA_DEFAULT_SYMBOLS)) == 50

    pairs = (
        (window.e_source, window.e_symbol),
        (window.m_source, window.m_symbol),
        (window.s_source, window.s_symbol),
    )
    for source_combo, symbol_combo in pairs:
        assert source_combo.count() == 2
        assert source_combo.currentText() == "binance"
        assert symbol_combo.findText("BTC") >= 0
        assert symbol_combo.findText("SNDKB") < 0
        assert symbol_combo.findText("SNDKX") < 0
        assert symbol_combo.maxVisibleItems() == 12

        source_combo.setCurrentText("alpaca")
        assert symbol_combo.count() == 50
        assert symbol_combo.currentText() == "AAPL"
        assert symbol_combo.findText("STRC") >= 0
        assert symbol_combo.findText("BTC") < 0
        assert symbol_combo.view().verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded

        source_combo.setCurrentText("binance")
        assert symbol_combo.count() == len(CRYPTO_SYMBOLS)
        assert symbol_combo.currentText() == "XRP"
        assert symbol_combo.findText("BTC") >= 0
        assert symbol_combo.findText("AAPL") < 0


def test_gui_routes_binance_and_requires_full_crypto_history(
    monkeypatch, martin_window
):
    window = martin_window
    calls = []

    def fake_get_klines(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-06-12", periods=2, freq="15min", tz="UTC"),
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1.0, 1.0],
            }
        )

    monkeypatch.setattr(martin, "get_klines", fake_get_klines)
    window._fetch_klines_if_needed(
        "BTC", "15m", "2026-01-01", "2026-07-01", "never", source="binance"
    )

    assert calls[0]["exch_list"] == ["binance"]
    assert calls[0]["allow_gaps"] is False
    assert calls[0]["require_full_coverage"] is True


def test_gui_routes_alpaca_with_credentials_and_stock_market_gaps(
    monkeypatch, martin_window
):
    window = martin_window
    calls = []
    credential_calls = []

    monkeypatch.setattr(
        alpaca, "load_env_file", lambda *args, **kwargs: credential_calls.append("file")
    )
    monkeypatch.setattr(
        alpaca,
        "credentials_from_env",
        lambda *args, **kwargs: credential_calls.append("credentials") or ("key", "secret"),
    )

    def fake_get_klines(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "time": pd.date_range("2026-06-12", periods=2, freq="15min", tz="UTC"),
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1.0, 1.0],
            }
        )

    monkeypatch.setattr(martin, "get_klines", fake_get_klines)
    window._fetch_klines_if_needed(
        "AAPL", "15m", "2026-01-01", "2026-07-01", "never", source="alpaca"
    )

    assert credential_calls == ["file", "credentials"]
    assert calls[0]["exch_list"] == ["alpaca"]
    assert calls[0]["allow_gaps"] is True
    assert calls[0]["require_full_coverage"] is False
    assert pd.Timestamp(calls[0]["start"]).tz_convert("America/New_York").hour == 0
    assert pd.Timestamp(calls[0]["end"]).tz_convert("America/New_York").hour == 23
    assert window._days_to_bars(365, "15m", "alpaca") == 26 * 252


def test_single_backtest_metrics_show_closed_trades_and_open_position(
    monkeypatch, martin_window
):
    window = martin_window
    closes = np.array([100.0, 106.0, 100.0, 106.0, 100.0])
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-05", periods=len(closes), freq="1h", tz="UTC"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.ones(len(closes)),
        }
    )
    df.attrs.update(
        exchange="alpaca", market="stock:alpaca", market_type="stock",
        symbol="AAPL", interval="1h",
    )
    monkeypatch.setattr(window, "_fetch_klines_if_needed", lambda *args, **kwargs: df)

    result_df, result, performance = window._compute_backtest(
        "AAPL", "1h", "2026-01-05", "2026-01-06", "never",
        0.0, 1000.0, 0.10, 2.0, 3, 0.05, source="alpaca",
    )
    window._render_plot_and_metrics(
        result_df, result, performance, 0.10, 2.0, 3, 0.05,
        window.figure_single, window.canvas_single, window.metrics_text,
    )
    metrics = window.metrics_text.toPlainText()

    assert result["trades"] == 2
    assert performance["closed_trades"] == 2
    assert "Closed Trades: 2 | Open Position at End: Yes" in metrics
    assert window.figure_single.axes[0].get_ylabel() == "Equity (USD)"


def test_alpaca_backtest_uses_stock_session_annualization(
    monkeypatch, martin_window
):
    window = martin_window
    days = pd.bdate_range("2026-01-05", periods=4, tz="America/New_York")
    times = []
    for day in days:
        session_start = day.normalize() + pd.Timedelta(hours=9, minutes=30)
        times.extend(
            session_start + pd.to_timedelta(np.arange(26) * 15, unit="min")
        )
    close = 100.0 + np.sin(np.arange(len(times)) / 8.0)
    frame = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(len(times)),
        }
    )
    frame.attrs.update(
        exchange="alpaca",
        market="stock:alpaca",
        market_type="stock",
        symbol="AAPL",
        interval="15m",
    )
    monkeypatch.setattr(window, "_fetch_klines_if_needed", lambda *args, **kwargs: frame)
    original_compute = martin.compute_performance_metrics
    captured = {}

    def capture_annualization(*args, **kwargs):
        captured["factor"] = kwargs.get("annualization_factor")
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(martin, "compute_performance_metrics", capture_annualization)

    window._compute_backtest(
        "AAPL", "15m", "2026-01-05", "2026-01-08", "never",
        0.0, 1_000.0, 0.03, 1.5, 5, 0.02, source="alpaca",
    )

    assert captured["factor"] == pytest.approx(26 * 252)


def test_scan_detail_uses_the_successful_scan_config_snapshot(
    monkeypatch, martin_window
):
    window = martin_window
    scan_result = pd.DataFrame(
        [{
            "add_drop": 0.03,
            "tp": 0.02,
            "multiplier": 1.5,
            "max_orders": 5,
            "min_buy_ratio": 0.88,
            "final_equity": 1_234.0,
            "max_dd_overall": 12.0,
            "trades": 8,
            "trapped_time_ratio": 0.1,
        }],
        index=[42],
    )
    scan_config = {
        "symbol": "AAPL",
        "source": "alpaca",
        "interval": "1h",
        "start": "2025-01-01",
        "end": "2026-01-01",
        "capital": 2_500.0,
    }
    window._scan_update_ui((scan_result, scan_config))
    window.table.selectRow(0)
    captured_workers = []
    monkeypatch.setattr(
        window.thread_pool, "start", lambda worker: captured_workers.append(worker)
    )

    # Current controls intentionally remain Binance/XRP. The old AAPL result
    # must still open with the config that produced that row.
    assert window.e_source.currentText() == "binance"
    assert window.e_symbol.currentText() == "XRP"
    window.plot_selected_from_table()

    assert len(captured_workers) == 1
    args = captured_workers[0].args
    assert args[0] == "AAPL"
    assert args[1:4] == ("1h", "2025-01-01", "2026-01-01")
    assert args[6] == 2_500.0
    assert args[-1] == "alpaca"

    mc_result = scan_result.reset_index(drop=True).assign(
        feasible=True,
        mc_early_rejected=False,
        mc_paths_evaluated=100,
        mc_terminal_median=1_200.0,
        mc_terminal_p5=900.0,
        mc_p_loss=0.2,
        mc_p_severe=0.01,
        mc_p_dd50=0.03,
        mc_mdd_mean=18.0,
        mc_seed_median_std=10.0,
    )
    window._mc_scan_update_ui((mc_result, 1, 1, scan_config))
    window.mc_table.selectRow(0)
    captured_workers.clear()
    window.plot_selected_from_mc_table()

    assert len(captured_workers) == 1
    mc_args = captured_workers[0].args
    assert mc_args[0] == "AAPL"
    assert mc_args[1:4] == ("1h", "2025-01-01", "2026-01-01")
    assert mc_args[6] == 2_500.0
    assert mc_args[-1] == "alpaca"
