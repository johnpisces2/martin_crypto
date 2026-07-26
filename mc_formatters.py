# -*- coding: utf-8 -*-
"""Display and export formatting helpers."""

from __future__ import annotations

import pandas as pd


def format_hist_scan_display(df: pd.DataFrame, pct_str_fn):
    disp = df.copy()
    is_diy = (
        "strategy_mode" in disp.columns
        and (
            "add_drop" not in disp.columns
            or disp.empty
            or disp["strategy_mode"].astype(str).str.lower().eq("diy").all()
        )
    )

    def percent(value, digits=1):
        if pd.isna(value):
            return "—"
        return f"{pct_str_fn(float(value), digits)}%"

    def money(value):
        return "—" if pd.isna(value) else f"{float(value):.2f}"

    def mdd(value):
        return "—" if pd.isna(value) else f"{float(value):.2f}%"

    if is_diy:
        mode = pd.Series("DIY TP/H-MAE", index=disp.index, dtype=object)
        capital_use = disp["avg_capital_utilization"].map(percent)

        def diy_setup(row):
            drops = "/".join(
                f"{100.0 * (1.0 - float(value)):.1f}"
                for value in tuple(row["level_ratios"])[1:]
            )
            shares = "/".join(
                str(int(value)) for value in tuple(row["order_shares"])
            )
            return f"Drops {drops}% | Shares {shares}"

        setup = (
            pd.Series(index=disp.index, dtype=object)
            if disp.empty else disp.apply(diy_setup, axis=1)
        )
    else:
        mode = pd.Series("Fixed", index=disp.index, dtype=object)
        capital_use = pd.Series("—", index=disp.index, dtype=object)
        setup = (
            pd.Series(index=disp.index, dtype=object)
            if disp.empty
            else disp.apply(
                lambda row: (
                    f"Drop {100.0 * float(row['add_drop']):.1f}% | "
                    f"Multiplier {float(row['multiplier']):.1f}×"
                ),
                axis=1,
            )
        )

    capital = (
        pd.to_numeric(disp["capital"], errors="coerce")
        if "capital" in disp.columns
        else pd.Series(float("nan"), index=disp.index, dtype=float)
    )
    final_equity = pd.to_numeric(
        disp.get("final_equity"), errors="coerce"
    )
    total_return = final_equity / capital - 1.0
    total_return = total_return.where(capital > 0.0)

    compact = pd.DataFrame(index=disp.index)
    compact["Mode"] = mode
    compact["TP"] = disp["tp"].map(percent)
    compact["Orders"] = disp["max_orders"].map(
        lambda value: "—" if pd.isna(value) else str(int(value))
    )
    compact["Final Equity"] = disp["final_equity"].map(money)
    compact["Return"] = total_return.map(percent)
    compact["Max DD"] = disp["max_dd_overall"].map(mdd)
    compact["Trades"] = disp["trades"].map(
        lambda value: "—" if pd.isna(value) else str(int(value))
    )
    compact["Trapped"] = disp["trapped_time_ratio"].map(
        lambda value: percent(value, 2)
    )
    compact["Lowest Buy"] = disp["min_buy_ratio"].map(percent)
    compact["Capital Use"] = capital_use
    compact["Setup"] = setup
    columns = list(compact.columns)
    return compact, columns


def format_hist_scan_csv(df: pd.DataFrame):
    out = df.copy()
    is_diy = (
        "strategy_mode" in out.columns
        and (
            "add_drop" not in out.columns
            or out.empty
            or out["strategy_mode"].astype(str).str.lower().eq("diy").all()
        )
    )
    if is_diy:
        out["buy_drops_pct"] = out["level_ratios"].map(
            lambda values: "/".join(
                f"{100.0 * (1.0 - float(v)):.4f}" for v in tuple(values)[1:]
            )
        )
        out["shares_plan"] = out["order_shares"].map(
            lambda values: "/".join(str(int(v)) for v in tuple(values))
        )
        out["tp"] = (out["tp"] * 100.0).map(lambda v: f"{float(v):.2f}")
        out["mae_q_start"] = (out["mae_q_start"] * 100.0).map(
            lambda v: f"{float(v):.2f}"
        )
        out["mae_q_end"] = (out["mae_q_end"] * 100.0).map(
            lambda v: f"{float(v):.2f}"
        )
        out["stress_mae_q"] = (out["stress_mae_q"] * 100.0).map(
            lambda v: f"{float(v):.2f}"
        )
        out["stress_mae"] = (out["stress_mae"] * 100.0).map(
            lambda v: f"{float(v):.4f}"
        )
        out["avg_capital_utilization"] = (
            out["avg_capital_utilization"] * 100.0
        ).map(lambda v: f"{float(v):.2f}")
        for col in (
            "expected_capital_utilization",
            "expected_idle_ratio",
            "actual_idle_ratio",
            "underwater_position_ratio",
            "full_capital_time_ratio",
            "tp_hit_rate_horizon",
            "tp_non_hit_rate_horizon",
        ):
            if col in out.columns:
                out[col] = (out[col] * 100.0).map(
                    lambda v: f"{float(v):.2f}"
                )
        out["trapped_time_ratio"] = (
            out["trapped_time_ratio"] * 100.0
        ).map(lambda v: f"{float(v):.2f}")
        cols = [
            "strategy_mode", "diy_variant", "mae_model", "tp", "max_orders",
            "mae_horizon_days",
            "mae_horizon_bars", "mae_sample_count", "mae_q_start", "mae_q_end",
            "stress_mae_q", "stress_mae", "capital_gamma",
            "idle_bias", "initial_capital_pct", "diy_total_shares",
            "last_order_pct", "stress_loss_pct", "tp_hit_rate_horizon",
            "tp_non_hit_rate_horizon", "median_bars_to_tp",
            "p90_bars_to_tp", "expected_capital_utilization",
            "expected_idle_ratio", "avg_capital_utilization",
            "actual_idle_ratio", "underwater_position_ratio",
            "full_capital_time_ratio", "min_buy_ratio", "fill_probabilities",
            "buy_drops_pct", "shares_plan", "final_equity",
            "max_dd_overall", "trades", "trapped_time_ratio",
        ]
        cols = [col for col in cols if col in out.columns]
        return out[cols].copy()

    cols = [
        "add_drop", "tp", "multiplier", "max_orders", "min_buy_ratio",
        "final_equity", "max_dd_overall", "trades", "trapped_time_ratio",
    ]
    out["add_drop"] = (out["add_drop"] * 100.0).map(lambda v: f"{float(v):.1f}")
    out["tp"] = (out["tp"] * 100.0).map(lambda v: f"{float(v):.1f}")
    out["trapped_time_ratio"] = (out["trapped_time_ratio"] * 100.0).map(lambda v: f"{float(v):.2f}")
    out["multiplier"] = out["multiplier"].map(lambda v: f"{float(v):.1f}")
    out["max_orders"] = out["max_orders"].astype(int).astype(str)
    out["trades"] = out["trades"].astype(int).astype(str)
    out["min_buy_ratio"] = out["min_buy_ratio"].map(lambda v: f"{float(v):.2f}")
    out["final_equity"] = out["final_equity"].map(lambda v: f"{float(v):.2f}")
    out["max_dd_overall"] = out["max_dd_overall"].map(lambda v: f"{float(v):.2f}")
    return out[cols].copy()


def format_mc_scan_display(df: pd.DataFrame, pct_str_fn, human_pct_fn):
    disp = df.copy()
    disp["add_drop"] = disp["add_drop"].apply(lambda v: pct_str_fn(v, 1))
    disp["tp"] = disp["tp"].apply(lambda v: pct_str_fn(v, 1))
    disp["multiplier"] = disp["multiplier"].apply(lambda v: f"{v:.1f}")
    disp["trapped_time_ratio"] = disp["trapped_time_ratio"].apply(lambda v: pct_str_fn(v, 2))
    for col in (
        "final_equity", "max_dd_overall", "mc_terminal_median", "mc_terminal_p5",
        "mc_mdd_mean", "mc_trapped_mean", "mc_seed_median_std",
    ):
        if col in disp.columns:
            disp[col] = disp[col].map(lambda v: "N/A" if pd.isna(v) else f"{float(v):.2f}")
    early = disp.get("mc_early_rejected", pd.Series(False, index=disp.index)).astype(bool)
    for col in ("mc_p_loss", "mc_p_severe", "mc_p_dd50"):
        disp[col] = [
            ("N/A" if pd.isna(v) else (("≥" if is_early else "") + human_pct_fn(v, 2)))
            for v, is_early in zip(disp[col], early)
        ]
    disp["feasible"] = disp["feasible"].apply(lambda v: "Y" if bool(v) else "N")
    if "mc_early_rejected" in disp.columns:
        disp["mc_early_rejected"] = disp["mc_early_rejected"].apply(lambda v: "Y" if bool(v) else "N")

    cols = [
        "feasible", "mc_early_rejected", "mc_paths_evaluated",
        "add_drop", "tp", "multiplier", "max_orders",
        "final_equity", "max_dd_overall", "trades", "trapped_time_ratio",
        "mc_terminal_median", "mc_terminal_p5",
        "mc_p_loss", "mc_p_severe", "mc_p_dd50", "mc_mdd_mean", "mc_seed_median_std",
    ]
    cols = [c for c in cols if c in disp.columns]
    return disp, cols
