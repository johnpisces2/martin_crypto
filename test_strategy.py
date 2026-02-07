# -*- coding: utf-8 -*-
"""
馬丁策略測試腳本 (從 martin.py 分離)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import martin
from martin import get_klines, martingale_backtest, compute_performance_metrics, pct_str, apply_filters, _grid_search_parallel

# 設定快取目錄
DEFAULT_CACHE_DIR = "cache"

if __name__ == "__main__":
    #----------- Parameter ---------------------------
    FEE_RATE = 0.0005  # 0.05%
    day = 60

    # 過濾（可關閉）
    MIN_TRADES = day
    MAX_DD_OVERALL = 15
    MAX_BUY_RATIO =  None
    MAX_TRAPPED_RATIO = 0.15

    # 抓取資料
    print("Fetching data...")
    df = get_klines(symbol="MNT", interval="15m", bars=day*24*60//15,
                    cache_dir=DEFAULT_CACHE_DIR, use_cache=True, refresh_policy="auto")

    add_drop_list   = [i / 1000 for i in range(10,  81)]  # 0.010 ~ 0.080
    tp_list         = [i / 1000 for i in range(10,  81)]  # 0.010 ~ 0.080
    multiplier_list = [i / 10   for i in range(15, 21)]   # 1.5 ~ 2.0
    max_orders_list = list(range(5, 10))                  # 5~9（含首筆）
    #-------------------------------------------------

    prices_np = df["close"].to_numpy(dtype=np.float64)
    if prices_np.size < 2:
        raise ValueError("K 線資料不足（<2 根），無法回測。")

    first_price = float(prices_np[0])
    print(f"[Source] {df.attrs.get('market','?')} | exch={df.attrs.get('exchange','?')} | symbol={df.attrs.get('symbol','?')} | interval={df.attrs.get('interval','?')}")
    print(f"[Time frame] {df['time'].iloc[0]}  →  {df['time'].iloc[-1]}")

    # ======== 平行網格（numba） ========
    print("Running grid search...")
    ad  = np.array(add_drop_list, dtype=np.float64)
    tp  = np.array(tp_list, dtype=np.float64)
    mul = np.array(multiplier_list, dtype=np.float64)
    mo  = np.array(max_orders_list, dtype=np.int32)

    AD, MUL, MO, TP = np.meshgrid(ad, mul, mo, tp, indexing="ij")
    ADf = AD.ravel(); MULf = MUL.ravel(); MOf = MO.ravel(); TPf = TP.ravel()

    # ======== 先遮罩（min_buy_ratio 理論下限）========
    # min_buy_ratio_theory = (1 - add_drop) ** (max_orders - 1)
    min_buy_ratio_theory = np.maximum(0.0, (1.0 - ADf) ** (MOf.astype(np.float64) - 1.0))
    mask = np.ones_like(ADf, dtype=bool)
    if MAX_BUY_RATIO is not None:
        mask = min_buy_ratio_theory <= float(MAX_BUY_RATIO)
    ADf = ADf[mask]; MULf = MULf[mask]; MOf = MOf[mask]; TPf = TPf[mask]
    min_buy_ratio_theory = min_buy_ratio_theory[mask]

    # ======== 平行運算（含手續費）========
    # 注意：這裡需要呼叫 martin 內部的 _grid_search_parallel，但因為它被隱藏了，
    # 我們可以直接用 martin._grid_search_parallel (如果沒被限制) 
    # 或者我們應該在 martin.py 裡把這個函數公開，或者直接在這裡 import
    # 為了方便，我們假設 martin.py 裡面的 _grid_search_parallel 是可以被 import 的 (Python 預設可以)
    
    fe, mdd, tr, trap = _grid_search_parallel(prices_np, ADf, MULf, MOf, TPf, capital=1000.0, fee_rate=FEE_RATE)

    results_df = pd.DataFrame({
        "add_drop": ADf,
        "multiplier": MULf,
        "max_orders": MOf.astype(int),
        "tp": TPf,
        "capital": 1000.0,
        "final_equity": np.round(fe, 2),
        "max_dd_overall": np.round(mdd, 2),
        "trades": tr.astype(int),
        "trapped_time_ratio": np.round(trap, 6)
    })

    # 記錄理論 min_buy_ratio（展示用）
    results_df["min_buy_ratio"] = min_buy_ratio_theory

    # 一次過濾（不再重複針對 MAX_BUY_RATIO）
    filtered = apply_filters(results_df, MIN_TRADES, MAX_DD_OVERALL, MAX_TRAPPED_RATIO)
    if filtered.empty:
        print("[Filter] 過濾條件過於嚴格，將回退使用全部結果。")
        filtered = results_df

    # 前 20 名
    top20 = filtered.nlargest(20, "final_equity")

    # 顯示用的 DataFrame（保持內部數值，但額外產出百分比欄位）
    topN = top20.copy()
    topN["min_buy_ratio"] = topN["min_buy_ratio"].round(2)

    # 百分比文字欄位（展示友好）
    topN_disp = topN.copy()
    topN_disp["add_drop"] = topN_disp["add_drop"].apply(lambda v: pct_str(v, 1))
    topN_disp["tp"]       = topN_disp["tp"].apply(lambda v: pct_str(v, 1))
    topN_disp["trapped_time_ratio"] = topN_disp["trapped_time_ratio"].apply(lambda v: pct_str(v, 1))

    cols = ["add_drop","tp","multiplier","max_orders",
            "min_buy_ratio","final_equity","max_dd_overall","trades","trapped_time_ratio"]

    print(topN_disp[cols].to_string(index=False))

    # 用第一名跑詳細版拿曲線 + 交易日誌（單次模擬，含手續費）
    if not topN.empty:
        best = topN.iloc[0]
        res_curve = martingale_backtest(
            prices_np,
            add_drop=float(best["add_drop"]),
            multiplier=float(best["multiplier"]),
            max_orders=int(best["max_orders"]),
            tp=float(best["tp"]),
            capital=float(best["capital"]),
            return_curve=True,
            times=df["time"].tolist(),
            fee_rate=FEE_RATE,
        )

        # Buy & Hold（不考慮手續費）
        initial_capital = float(best["capital"])
        bh_qty = initial_capital / first_price
        bh_curve = bh_qty * prices_np

        # 完整績效
        perf = compute_performance_metrics(
            res_curve["equity_curve"],
            res_curve["time_index"],
            res_curve["trades_log"],
            capital=initial_capital,
            bh_curve=bh_curve
        )

        def pct(x): return f"{x*100:.2f}%" if pd.notna(x) else "NaN"

        print("\n=== Performance (Strategy) ===")
        print(f"Total Return: {pct(perf['total_return'])} | CAGR: {pct(perf['cagr'])} | Ann Vol: {pct(perf['ann_vol'])}")
        print(f"Sharpe: {perf['sharpe']:.2f} | Sortino: {perf['sortino']:.2f} | Calmar: {perf['calmar']:.2f}")
        print(f"Max DD: {perf['max_dd_pct']:.2f}% | DD Duration (days): {perf['max_dd_days'] if perf['max_dd_days'] is not None else 'NaN'} | Recovery: {perf['recovery_days'] if perf['recovery_days'] is not None else 'NaN'}")
        print(f"Underwater (avg/max): {perf['avg_underwater_days']:.1f}/{perf['max_underwater_days']:.1f} days")
        pf = perf['profit_factor']
        pf_str = "∞" if (isinstance(pf, (int, float)) and np.isinf(pf)) else ("NaN" if pd.isna(pf) else f"{pf:.2f}")
        print(f"Win Rate: {pct(perf['win_rate'])} | Profit Factor: {pf_str}")
        print(f"Avg Win: {perf['avg_win']:.2f} | Avg Loss: {perf['avg_loss']:.2f}")
        print(f"Max Consec Wins: {perf['max_consec_wins']} | Max Consec Losses: {perf['max_consec_losses']}")
        print(f"Avg Trade Return: {pct(perf['avg_trade_return'])} | Median Trade Return: {pct(perf['median_trade_return'])}")

        if "bh_total_return" in perf:
            print("\n=== Buy & Hold (Benchmark) ===")
            print(f"Total: {pct(perf['bh_total_return'])} | CAGR: {pct(perf['bh_cagr'])} | Ann Vol: {pct(perf['bh_ann_vol'])}")
            print(f"Sharpe: {perf['bh_sharpe']:.2f} | Max DD: {perf['bh_max_dd_pct']:.2f}%")

        # 畫圖：策略 vs. 買入持有
        sym = df.attrs.get("symbol", "UNKNOWN")
        market_code = df.attrs.get("market", "spot")
        market_label = "Spot" if str(market_code).startswith("spot") else ("USDT Perp" if "usdt_perp" in str(market_code) else str(market_code))
        interval_str = df.attrs.get("interval", "N/A")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(res_curve["time_index"], res_curve["equity_curve"], label="Strategy")
        ax.plot(df["time"], bh_curve, label="Buy & Hold")
        # 套牢區間（垂直帶狀）
        handles, labels = ax.get_legend_handles_labels()
        for start, end in res_curve.get("trapped_intervals", []):
            ax.axvspan(start, end, alpha=0.1, color='red', label='Trapped Period' if 'Trapped Period' not in labels else "")
            handles, labels = ax.get_legend_handles_labels()
        ax.set_title(
            f'{sym} | {market_label} | exch={df.attrs.get("exchange","?")} | interval={interval_str}\n'
            f'[{df["time"].iloc[0].date()} → {df["time"].iloc[-1].date()}] '
            f'add_drop={best["add_drop"]:.3f}, tp={best["tp"]:.3f}, mul={best["multiplier"]:.1f}, '
            f'max_orders={int(best["max_orders"])}'
        )
        ax.set_xlabel("Time (Taipei)")
        ax.set_ylabel("Equity (USDT)")
        ax.legend()
        fig.tight_layout()
        plt.show()
    else:
        print("[Result] 無符合過濾條件的結果。")
