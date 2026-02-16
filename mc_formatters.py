# -*- coding: utf-8 -*-
"""Display and export formatting helpers."""

from __future__ import annotations

import pandas as pd


def format_hist_scan_display(df: pd.DataFrame, pct_str_fn):
    disp = df.copy()
    disp["add_drop"] = disp["add_drop"].apply(lambda v: pct_str_fn(v, 1))
    disp["tp"] = disp["tp"].apply(lambda v: pct_str_fn(v, 1))
    disp["multiplier"] = disp["multiplier"].apply(lambda v: f"{v:.1f}")
    disp["trapped_time_ratio"] = disp["trapped_time_ratio"].apply(lambda v: pct_str_fn(v, 2))
    disp["min_buy_ratio"] = disp["min_buy_ratio"].round(2)
    cols = [
        "add_drop", "tp", "multiplier", "max_orders", "min_buy_ratio",
        "final_equity", "max_dd_overall", "trades", "trapped_time_ratio",
    ]
    return disp, cols


def format_hist_scan_csv(df: pd.DataFrame):
    out = df.copy()
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
    for col in ("final_equity", "max_dd_overall", "mc_terminal_median", "mc_terminal_p5", "mc_mdd_mean", "mc_trapped_mean"):
        if col in disp.columns:
            disp[col] = disp[col].map(lambda v: f"{float(v):.2f}")
    disp["mc_p_loss"] = disp["mc_p_loss"].apply(lambda v: human_pct_fn(v, 2))
    disp["mc_p_severe"] = disp["mc_p_severe"].apply(lambda v: human_pct_fn(v, 2))
    disp["mc_p_dd50"] = disp["mc_p_dd50"].apply(lambda v: human_pct_fn(v, 2))
    disp["feasible"] = disp["feasible"].apply(lambda v: "Y" if bool(v) else "N")

    cols = [
        "feasible", "add_drop", "tp", "multiplier", "max_orders",
        "final_equity", "max_dd_overall", "trades", "trapped_time_ratio",
        "mc_terminal_median", "mc_terminal_p5",
        "mc_p_loss", "mc_p_severe", "mc_p_dd50", "mc_mdd_mean",
    ]
    cols = [c for c in cols if c in disp.columns]
    return disp, cols
