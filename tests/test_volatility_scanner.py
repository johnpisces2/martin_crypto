import numpy as np
import pandas as pd

from volatility_scanner_gui import (
    add_martin_fit_columns,
    martin_fit_from_row,
    martin_recovery_metrics,
    optional_millions,
)


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
