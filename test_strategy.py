# -*- coding: utf-8 -*-
"""CLI smoke test for martin.py (grid scan + best-params backtest)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import martin
from martin import (
    _grid_search_parallel,
    apply_filters,
    compute_performance_metrics,
    get_klines,
    martingale_backtest,
    pct_str,
)


def _pct(x):
    return f"{x * 100:.2f}%" if pd.notna(x) else "NaN"


def _fmt2(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{x:.2f}"


def main():
    fee_rate = 0.0005
    capital = 1000.0
    days = 60
    symbol = "SUI"
    interval = "15m"

    min_trades = days
    max_dd_overall = 15
    max_buy_ratio = None
    max_trapped_ratio = 0.15

    print("Fetching data...")
    bars = days * 24 * 60 // 15
    df = get_klines(
        symbol=symbol,
        interval=interval,
        bars=bars,
        cache_dir=martin.DEFAULT_CACHE_DIR,
        use_cache=True,
        refresh_policy="auto",
    )
    prices_np = df["close"].to_numpy(dtype=np.float64)
    if prices_np.size < 2:
        raise ValueError("K 線資料不足（<2 根），無法回測。")

    print(
        f"[Source] {df.attrs.get('market','?')} | exch={df.attrs.get('exchange','?')} | "
        f"symbol={df.attrs.get('symbol','?')} | interval={df.attrs.get('interval','?')}"
    )
    print(f"[Time frame] {df['time'].iloc[0]}  →  {df['time'].iloc[-1]}")

    print("Running grid search...")
    ad = np.array([i / 1000 for i in range(10, 81)], dtype=np.float64)
    tp = np.array([i / 1000 for i in range(10, 81)], dtype=np.float64)
    mul = np.array([i / 10 for i in range(15, 21)], dtype=np.float64)
    mo = np.array(list(range(5, 10)), dtype=np.int32)

    AD, MUL, MO, TP = np.meshgrid(ad, mul, mo, tp, indexing="ij")
    ADf = AD.ravel()
    MULf = MUL.ravel()
    MOf = MO.ravel()
    TPf = TP.ravel()

    min_buy_ratio_theory = np.maximum(0.0, (1.0 - ADf) ** (MOf.astype(np.float64) - 1.0))
    if max_buy_ratio is not None:
        mask = min_buy_ratio_theory <= float(max_buy_ratio)
        ADf = ADf[mask]
        MULf = MULf[mask]
        MOf = MOf[mask]
        TPf = TPf[mask]
        min_buy_ratio_theory = min_buy_ratio_theory[mask]

    fe, mdd, tr, trap = _grid_search_parallel(
        prices_np, ADf, MULf, MOf, TPf, capital=capital, fee_rate=fee_rate
    )
    results_df = pd.DataFrame(
        {
            "add_drop": ADf,
            "multiplier": MULf,
            "max_orders": MOf.astype(int),
            "tp": TPf,
            "capital": capital,
            "final_equity": np.round(fe, 2),
            "max_dd_overall": np.round(mdd, 2),
            "trades": tr.astype(int),
            "trapped_time_ratio": np.round(trap, 6),
            "min_buy_ratio": min_buy_ratio_theory,
        }
    )

    filtered = apply_filters(results_df, min_trades, max_dd_overall, max_trapped_ratio)
    if filtered.empty:
        print("[Filter] 過濾條件過於嚴格，回退使用全部結果。")
        filtered = results_df

    top = filtered.nlargest(20, "final_equity").copy()
    if top.empty:
        print("[Result] 無符合過濾條件的結果。")
        return

    top["min_buy_ratio"] = top["min_buy_ratio"].round(2)
    top_disp = top.copy()
    top_disp["add_drop"] = top_disp["add_drop"].apply(lambda v: pct_str(v, 1))
    top_disp["tp"] = top_disp["tp"].apply(lambda v: pct_str(v, 1))
    top_disp["trapped_time_ratio"] = top_disp["trapped_time_ratio"].apply(lambda v: pct_str(v, 1))
    cols = [
        "add_drop",
        "tp",
        "multiplier",
        "max_orders",
        "min_buy_ratio",
        "final_equity",
        "max_dd_overall",
        "trades",
        "trapped_time_ratio",
    ]
    print(top_disp[cols].to_string(index=False))

    best = top.iloc[0]
    res = martingale_backtest(
        prices_np,
        add_drop=float(best["add_drop"]),
        multiplier=float(best["multiplier"]),
        max_orders=int(best["max_orders"]),
        tp=float(best["tp"]),
        capital=float(best["capital"]),
        return_curve=True,
        times=df["time"].tolist(),
        fee_rate=fee_rate,
    )

    first_price = float(prices_np[0])
    bh_qty = capital / first_price
    bh_curve = bh_qty * prices_np
    perf = compute_performance_metrics(
        res["equity_curve"], res["time_index"], res["trades_log"], capital=capital, bh_curve=bh_curve
    )

    print("\n=== Performance (Strategy) ===")
    print(
        f"Total Return: {_pct(perf.get('total_return'))} | CAGR: {_pct(perf.get('cagr'))} | "
        f"Ann Vol: {_pct(perf.get('ann_vol'))}"
    )
    print(
        f"Sharpe: {_fmt2(perf.get('sharpe'))} | Sortino: {_fmt2(perf.get('sortino'))} | "
        f"Calmar: {_fmt2(perf.get('calmar'))}"
    )
    print(
        f"Max DD: {_fmt2(perf.get('max_dd_pct'))}% | "
        f"DD Duration (days): {perf.get('max_dd_days','NaN')} | Recovery: {perf.get('recovery_days','NaN')}"
    )
    print(
        f"Underwater (avg/max): {_fmt2(perf.get('avg_underwater_days'))}/"
        f"{_fmt2(perf.get('max_underwater_days'))} days"
    )

    pf = perf.get("profit_factor")
    pf_str = "∞" if (isinstance(pf, (int, float)) and np.isinf(pf)) else ("NaN" if pd.isna(pf) else f"{pf:.2f}")
    print(f"Win Rate: {_pct(perf.get('win_rate'))} | Profit Factor: {pf_str}")
    print(f"Avg Win: {_fmt2(perf.get('avg_win'))} | Avg Loss: {_fmt2(perf.get('avg_loss'))}")
    print(
        f"Max Consec Wins: {perf.get('max_consec_wins','NaN')} | "
        f"Max Consec Losses: {perf.get('max_consec_losses','NaN')}"
    )
    print(
        f"Avg Trade Return: {_pct(perf.get('avg_trade_return'))} | "
        f"Median Trade Return: {_pct(perf.get('median_trade_return'))}"
    )

    if "bh_total_return" in perf:
        print("\n=== Buy & Hold (Benchmark) ===")
        print(
            f"Total: {_pct(perf.get('bh_total_return'))} | CAGR: {_pct(perf.get('bh_cagr'))} | "
            f"Ann Vol: {_pct(perf.get('bh_ann_vol'))}"
        )
        print(
            f"Sharpe: {_fmt2(perf.get('bh_sharpe'))} | "
            f"Max DD: {_fmt2(perf.get('bh_max_dd_pct'))}%"
        )

    sym = df.attrs.get("symbol", "UNKNOWN")
    market_code = df.attrs.get("market", "spot")
    market_label = (
        "Spot"
        if str(market_code).startswith("spot")
        else ("USDT Perp" if "usdt_perp" in str(market_code) else str(market_code))
    )
    interval_str = df.attrs.get("interval", "N/A")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(res["time_index"], res["equity_curve"], label="Strategy")
    ax.plot(df["time"], bh_curve, label="Buy & Hold")
    handles, labels = ax.get_legend_handles_labels()
    for start, end in res.get("trapped_intervals", []):
        label = "Trapped Period" if "Trapped Period" not in labels else ""
        ax.axvspan(start, end, alpha=0.1, color="red", label=label)
        handles, labels = ax.get_legend_handles_labels()
    ax.set_title(
        f"{sym} | {market_label} | exch={df.attrs.get('exchange','?')} | interval={interval_str}\n"
        f"[{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}] "
        f"add_drop={best['add_drop']:.3f}, tp={best['tp']:.3f}, mul={best['multiplier']:.1f}, "
        f"max_orders={int(best['max_orders'])}"
    )
    ax.set_xlabel("Time (Taipei)")
    ax.set_ylabel("Equity (USDT)")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
