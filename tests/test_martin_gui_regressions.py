import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, Qt

import martin
from diy_strategy import (
    build_diy_templates,
    compute_tp_or_horizon_mae_grid,
    idle_aware_capital_weights,
    integer_shares,
)
from market_data import alpaca
from martin_gui import (
    ALPACA_DEFAULT_SYMBOLS,
    CRYPTO_SYMBOLS,
    DATA_SOURCES,
    MartinGUI,
    backtest_accounting_summary,
    configure_datetime_axis,
    parse_range,
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


def test_historical_scan_switches_between_fixed_and_diy_fields(martin_window):
    window = martin_window
    window.e_scan_mode.setCurrentText("Fixed Mode")

    assert window.e_scan_mode.count() == 2
    assert not window.e_add_drop.isHidden()
    assert not window.e_multiplier.isHidden()
    assert window.e_mae_horizon.isHidden()

    window.e_scan_mode.setCurrentText("DIY Mode (TP/Horizon MAE)")

    assert window.e_add_drop.isHidden()
    assert window.e_multiplier.isHidden()
    assert not window.e_mae_horizon.isHidden()
    config = window._capture_scan_config()
    assert config["scan_mode"] == "DIY Mode (TP/Horizon MAE)"
    assert "diy_idle_bias_range" not in config
    assert "mae_q_end_range" not in config
    assert "diy_gamma_range" not in config

    window.e_scan_mode.setCurrentText("Fixed Mode")


def test_historical_scan_defaults_to_top_20_without_performance_filters(
    martin_window,
):
    window = martin_window

    assert window.e_min_trades.text() == ""
    assert window.e_max_dd.text() == ""
    assert window.e_max_trap.text() == ""
    assert window.e_topn.text() == "20"


def test_tp_or_horizon_mae_stops_at_first_unambiguous_tp_and_keeps_non_hits():
    closes = np.full(6, 100.0)
    highs = np.array([100.0, 102.0, 106.0, 103.0, 100.0, 110.0])
    lows = np.array([100.0, 98.0, 100.0, 95.0, 90.0, 100.0])

    mae, hit, bars = compute_tp_or_horizon_mae_grid(
        highs,
        lows,
        closes,
        horizon_bars=3,
        tp_values=np.array([0.04, 0.20]),
        fee_rate=0.0,
    )

    assert mae.shape == hit.shape == bars.shape == (2, 3)
    assert hit[0].tolist() == [1, 1, 1]
    assert bars[0].tolist() == [2, 1, 3]
    assert mae[0] == pytest.approx([0.02, 0.00, 0.10])
    assert hit[1].tolist() == [0, 0, 0]
    assert bars[1].tolist() == [3, 3, 3]
    assert mae[1] == pytest.approx([0.05, 0.10, 0.10])


def test_tp_or_horizon_mae_skips_ambiguous_add_and_tp_bar():
    closes = np.array([100.0, 100.0, 75.0, 100.0])
    highs = np.array([100.0, 106.0, 80.0, 106.0])
    lows = np.array([100.0, 90.0, 70.0, 100.0])

    mae, hit, bars = compute_tp_or_horizon_mae_grid(
        highs,
        lows,
        closes,
        horizon_bars=3,
        tp_values=np.array([0.05]),
        fee_rate=0.0,
    )
    detailed = martin.martingale_backtest_diy(
        closes,
        level_ratios=[1.0, 0.95, 0.90],
        order_shares=[10, 20, 70],
        tp=0.05,
        capital=1_000.0,
        fee_rate=0.0,
        opens=np.array([100.0, 100.0, 80.0, 100.0]),
        highs=highs,
        lows=lows,
    )

    assert hit[0, 0] == 1
    assert bars[0, 0] == 3
    assert mae[0, 0] == pytest.approx(0.30)
    assert detailed["max_dd_overall"] == pytest.approx(30.0)


def test_tp_or_horizon_mae_rejects_oversized_output_before_allocation():
    values = np.full(20, 100.0)

    with pytest.raises(ValueError, match="輸出矩陣預估需要"):
        compute_tp_or_horizon_mae_grid(
            values,
            values,
            values,
            horizon_bars=5,
            tp_values=np.linspace(0.01, 0.20, 20),
            max_output_bytes=1,
        )
    with pytest.raises(ValueError, match="超過 10 個限制"):
        parse_range("0.001:1.000:0.001", max_items=10)


def test_idle_bias_preserves_initial_share_and_raises_expected_deployment():
    base = np.array([0.05, 0.10, 0.20, 0.25, 0.40])
    fill = np.array([1.0, 0.80, 0.50, 0.20, 0.05])
    adjusted = idle_aware_capital_weights(base, fill, idle_bias=1.0)

    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[0] == pytest.approx(base[0])
    assert adjusted[-1] < base[-1]
    assert np.dot(adjusted, fill) > np.dot(base, fill)


def test_diy_templates_produce_strict_ladders_and_risk_columns():
    mae = np.linspace(0.005, 0.45, 2_000)
    templates, level_matrix, share_matrix, stress_mae = build_diy_templates(
        mae_samples=mae,
        max_orders_values=np.array([5, 7]),
        q_start=0.50,
        q_end_values=np.array([0.90, 0.95]),
        initial_fraction_values=np.array([0.03, 0.05]),
        gamma_values=np.array([0.8, 1.2]),
        stress_quantile=0.99,
        total_shares=100,
        fee_rate=0.0005,
        max_last_order_pct=60.0,
        max_stress_loss_pct=80.0,
    )

    assert not templates.empty
    assert level_matrix.shape == share_matrix.shape
    assert stress_mae == pytest.approx(np.quantile(mae, 0.99))
    assert integer_shares([0.03, 0.07, 0.15, 0.25, 0.50], 100).sum() == 100
    for row in templates.itertuples():
        levels = np.asarray(row.level_ratios)
        shares = np.asarray(row.order_shares)
        assert levels[0] == 1.0
        assert np.all(np.diff(levels) < 0.0)
        assert shares.sum() == 100
        assert row.last_order_pct <= 60.0
        assert row.stress_loss_pct <= 80.0


def test_diy_templates_validate_empty_samples_and_deduplicate_flat_quantiles():
    kwargs = dict(
        max_orders_values=np.array([5]),
        q_start=0.50,
        q_end_values=np.array([0.90, 0.92, 0.94, 0.96, 0.98]),
        initial_fraction_values=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
        gamma_values=np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8]),
        stress_quantile=0.99,
        total_shares=100,
        fee_rate=0.0005,
        max_last_order_pct=30.0,
        max_stress_loss_pct=50.0,
    )

    with pytest.raises(ValueError, match="MAE samples"):
        build_diy_templates(mae_samples=[], **kwargs)

    templates, level_matrix, share_matrix, _ = build_diy_templates(
        mae_samples=np.full(1_000, 0.10),
        **kwargs,
    )
    executable = {
        (row.max_orders, row.level_ratios, row.order_shares)
        for row in templates.itertuples()
    }
    assert len(templates) == len(executable)
    assert level_matrix.shape[0] == share_matrix.shape[0] == len(templates)


def test_numba_diy_core_matches_detailed_backtest():
    closes = np.array(
        [100.0, 96.0, 91.0, 86.0, 90.0, 95.0, 101.0, 100.0, 103.0]
    )
    opens = closes.copy()
    highs = closes + 2.0
    lows = closes - 2.0
    levels = np.array([1.0, 0.95, 0.90, 0.85])
    shares = np.array([10.0, 20.0, 30.0, 40.0])
    tp = 0.04
    capital = 1_000.0
    fee = 0.0005

    detailed = martin.martingale_backtest_diy(
        closes,
        levels,
        shares,
        tp=tp,
        capital=capital,
        fee_rate=fee,
        opens=opens,
        highs=highs,
        lows=lows,
        return_curve=True,
    )
    fast = martin._backtest_core_diy_ohlc(
        opens,
        highs,
        lows,
        closes,
        levels,
        shares,
        len(levels),
        tp,
        capital,
        fee,
    )

    assert fast[0] == pytest.approx(detailed["equity_curve"][-1], rel=1e-10)
    assert fast[1] == pytest.approx(detailed["max_dd_overall"], abs=0.01)
    assert fast[2] == detailed["trades"]
    assert fast[3] == pytest.approx(detailed["trapped_time_ratio"])
    assert fast[4] == pytest.approx(detailed["avg_capital_utilization"])
    extended = martin._backtest_core_diy_ohlc_extended(
        opens,
        highs,
        lows,
        closes,
        levels,
        shares,
        len(levels),
        tp,
        capital,
        fee,
    )
    assert extended[5] == pytest.approx(detailed["underwater_position_ratio"])
    assert extended[6] == pytest.approx(detailed["full_capital_time_ratio"])


def test_diy_scan_uses_tp_horizon_mae_bias_zero_and_internal_search_defaults(
    martin_window,
):
    window = martin_window
    rng = np.random.default_rng(21)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 500)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.008
    low = np.minimum(open_, close) * 0.992
    frame = pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01", periods=len(close), freq="1h", tz="UTC"
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    ctx = {
        "scan_mode": "DIY Mode (TP/Horizon MAE)",
        "tp_range": "0.02:0.03:0.01",
        "max_orders_range": "5:6:1",
        "mae_horizon_days": "2",
        "min_trades": "",
        "max_dd": "",
        "max_trap": "",
        "interval": "1h",
        "source": "binance",
        "capital": 1_000.0,
        "fee_rate": 0.0005,
        "df": frame,
        "prices_np": close,
        "opens_np": open_,
        "highs_np": high,
        "lows_np": low,
    }

    result = window._compute_filtered_diy_scan_results(ctx)

    assert not result.empty
    assert result["diy_variant"].eq("tp_horizon_mae").all()
    assert result["mae_model"].eq("tp_or_horizon").all()
    assert result["idle_bias"].eq(0.0).all()
    assert result["mae_q_start"].eq(0.50).all()
    assert set(result["mae_q_end"]).issubset({0.90, 0.92, 0.94, 0.96, 0.98})
    assert result["diy_total_shares"].eq(100).all()
    assert result["last_order_pct"].le(30.0).all()
    assert result["stress_loss_pct"].le(50.0).all()
    assert result["tp_hit_rate_horizon"].between(0.0, 1.0).all()
    assert result["underwater_position_ratio"].between(0.0, 1.0).all()
    assert result["full_capital_time_ratio"].between(0.0, 1.0).all()
    assert result["actual_idle_ratio"].between(0.0, 1.0).all()
    assert result["fill_probabilities"].map(
        lambda values: values[0] == pytest.approx(1.0)
    ).all()
    display, columns = window._format_hist_scan_display(result.head(2))
    expected_columns = [
        "Mode", "TP", "Orders", "Final Equity", "Return", "Max DD",
        "Trades", "Trapped", "Lowest Buy", "Capital Use", "Setup",
    ]
    assert columns == expected_columns
    assert display["Mode"].eq("DIY TP/H-MAE").all()
    assert display["Setup"].str.contains("Drops").all()
    assert display["Setup"].str.contains("Shares").all()
    empty_display, empty_columns = window._format_hist_scan_display(
        result.head(0)
    )
    assert empty_display.empty
    assert empty_columns == expected_columns

    fixed = pd.DataFrame(
        {
            "add_drop": [0.05],
            "tp": [0.04],
            "multiplier": [2.0],
            "max_orders": [9],
            "capital": [1_000.0],
            "final_equity": [1_500.0],
            "max_dd_overall": [35.0],
            "trades": [20],
            "trapped_time_ratio": [0.12],
            "min_buy_ratio": [0.6634],
        }
    )
    fixed_display, fixed_columns = window._format_hist_scan_display(fixed)
    assert fixed_columns == expected_columns
    assert fixed_display.loc[0, "Mode"] == "Fixed"
    assert fixed_display.loc[0, "Capital Use"] == "—"
    assert fixed_display.loc[0, "Return"] == "50.0%"
    assert fixed_display.loc[0, "Setup"] == "Drop 5.0% | Multiplier 2.0×"

    csv = window._format_hist_scan_csv(result.head(2))
    assert {"buy_drops_pct", "shares_plan", "stress_loss_pct"}.issubset(
        csv.columns
    )


def test_diy_scan_detail_rebuilds_ladder_and_renders_mae_analysis(
    monkeypatch, martin_window
):
    window = martin_window
    close = np.array([100.0, 95.0, 90.0, 86.0, 91.0, 97.0, 103.0])
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(close), freq="1h", tz="UTC"),
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.ones(len(close)),
        }
    )
    frame.attrs.update(
        exchange="binance", market="spot:binance", market_type="spot",
        symbol="BTC", interval="1h",
    )
    monkeypatch.setattr(window, "_fetch_klines_if_needed", lambda *args, **kwargs: frame)
    metadata = {
        "mae_horizon_days": 7.0,
        "mae_horizon_bars": 168,
        "mae_sample_count": 1_000,
        "mae_q_start": 0.50,
        "mae_q_end": 0.95,
        "mae_quantiles": (0.0, 0.50, 0.725, 0.95),
        "stress_mae_q": 0.99,
        "stress_mae": 0.30,
        "stress_loss_pct": 18.0,
        "initial_capital_pct": 10.0,
        "capital_gamma": 1.2,
        "idle_bias": 0.0,
        "last_order_pct": 40.0,
        "diy_total_shares": 100,
        "diy_variant": "tp_horizon_mae",
        "mae_model": "tp_or_horizon",
        "fill_probabilities": (1.0, 0.5, 0.275, 0.05),
        "expected_capital_utilization": 0.29,
        "expected_idle_ratio": 0.71,
        "tp_hit_rate_horizon": 0.72,
        "tp_non_hit_rate_horizon": 0.28,
        "median_bars_to_tp": 18.0,
        "p90_bars_to_tp": 90.0,
    }

    result_df, result, performance = window._compute_diy_backtest(
        "BTC", "1h", "2026-01-01", "2026-01-02", "never",
        0.0005, 1_000.0, (1.0, 0.95, 0.90, 0.85),
        (10, 20, 30, 40), 0.04, "binance", metadata,
    )
    window._render_plot_and_metrics(
        result_df, result, performance, np.nan, np.nan, 4, 0.04,
        window.figure_scan, window.canvas_scan, window.scan_metrics_text,
    )
    metrics = window.scan_metrics_text.toPlainText()

    assert "=== Strategy Setup ===" in metrics
    assert "=== Key Performance ===" in metrics
    assert "=== DIY Order Plan ===" in metrics
    assert metrics.index("=== Key Performance ===") < metrics.index(
        "=== DIY Order Plan ==="
    )
    assert "Order | MAE q | Fill P | Cum.Drop | Shares" in metrics
    assert "Idle Bias" not in metrics
    assert "Capital Use:" in metrics
    assert "DIY TP/Horizon MAE q_end=95%" in window.figure_scan.axes[0].get_title()


def test_diy_detail_cache_uses_metadata_from_current_result_row(
    monkeypatch,
    martin_window,
):
    window = martin_window
    close = np.array([100.0, 96.0, 92.0, 98.0, 104.0])
    frame = pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-02-01",
                periods=len(close),
                freq="1h",
                tz="UTC",
            ),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(len(close)),
        }
    )
    frame.attrs.update(
        exchange="binance",
        market="spot:binance",
        market_type="spot",
        symbol="CACHEMETA",
        interval="1h",
    )
    monkeypatch.setattr(
        window,
        "_fetch_klines_if_needed",
        lambda *args, **kwargs: frame,
    )
    args = (
        "CACHEMETA",
        "1h",
        "2026-02-01",
        "2026-02-02",
        "never",
        0.0005,
        1_000.0,
        (1.0, 0.95, 0.90),
        (10, 30, 60),
        0.04,
        "binance",
    )

    _, first, _ = window._compute_diy_backtest(
        *args,
        {"mae_q_end": 0.90, "capital_gamma": 0.8},
    )
    _, second, _ = window._compute_diy_backtest(
        *args,
        {"mae_q_end": 0.98, "capital_gamma": 1.8},
    )

    assert first["_diy_metadata"]["mae_q_end"] == 0.90
    assert second["_diy_metadata"]["mae_q_end"] == 0.98
    assert second["_diy_metadata"]["capital_gamma"] == 1.8


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
    assert "Closed Trades: 2" in metrics
    assert "Open Position: Yes" in metrics
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
