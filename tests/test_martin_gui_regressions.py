import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QDate
from PySide6.QtWidgets import QApplication

import martin
from martin_gui import (
    MartinGUI,
    backtest_accounting_summary,
    configure_datetime_axis,
)


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


def test_gui_date_inputs_cannot_select_future_dates():
    app = QApplication.instance() or QApplication([])
    window = MartinGUI()
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

    window.close()
    app.processEvents()
