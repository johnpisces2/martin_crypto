# -*- coding: utf-8 -*-
"""
Volatility Scanner GUI for Martin Strategy
用來掃描高波動幣種，輔助馬丁策略選幣。
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import ccxt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import traceback
import concurrent.futures
import requests
import time
import json
import re


# 嘗試匯入 martin.py 以重用 get_klines
try:
    import martin
except ImportError:
    messagebox.showerror("Import Error", "無法匯入 martin.py，請確保檔案在同一目錄下。")
    raise

import math

# ... (imports remain the same)

# ===================== 指標計算 (From scan_vol_rank.py) =====================
def realized_vol_annual_from_closes(closes: pd.Series, bars_per_year: float) -> float:
    c = closes.to_numpy(dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < 2:
        return float("nan")
    rets = np.diff(np.log(c))
    if rets.size == 0:
        return float("nan")
    return float(np.std(rets, ddof=1) * math.sqrt(bars_per_year))

def true_atr_pct(df: pd.DataFrame, n: int = 14) -> float:
    req = {"high","low","close"}
    if not req.issubset(df.columns):
        return float("nan")
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    if c.size < n or c[-1] <= 0:
        return float("nan")
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr_last = tr[-n:].mean()
    return float(atr_last / c[-1])

def approx_atr_pct_from_close(closes: pd.Series, n: int = 14) -> float:
    c = closes.to_numpy(dtype=float)
    if c.size < n + 1 or c[-1] <= 0:
        return float("nan")
    tr = np.abs(np.diff(c))
    atr_last = tr[-n:].mean()
    return float(atr_last / c[-1])

def bb_width_pct(closes: pd.Series, n: int = 20, k: float = 2.0) -> float:
    c = closes.to_numpy(dtype=float)
    if c.size < n:
        return float("nan")
    tail = c[-n:]
    ma = tail.mean()
    sd = tail.std(ddof=1)
    if not np.isfinite(ma) or not np.isfinite(sd) or ma == 0:
        return float("nan")
    return float((2.0 * k * sd) / ma)

def fetch_top_mc_coins(limit=250):
    """
    Fetch top N coins by market cap from CoinGecko.
    Returns a dict: { symbol_lowercase: rank }
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": min(limit, 250), # CoinGecko max per page is 250
        "page": 1,
        "sparkline": "false"
    }
    
    mapping = {}
    try:
        # Page 1
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for i, item in enumerate(data):
                sym = item['symbol'].lower()
                if sym not in mapping:
                    mapping[sym] = item['market_cap_rank']
        
        # If limit > 250, fetch page 2
        if limit > 250:
            params["page"] = 2
            params["per_page"] = limit - 250
            time.sleep(1.0) # Rate limit politeness
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for i, item in enumerate(data):
                    sym = item['symbol'].lower()
                    if sym not in mapping:
                        mapping[sym] = item['market_cap_rank']
        
        return mapping
    except Exception as e:
        print(f"CoinGecko API Error: {e}")
        return {}

def is_stablecoin_base(base: str) -> bool:
    b = (base or "").upper()
    if not b:
        return False
    # Common stablecoins (expand as needed)
    stable_set = {
        "USDT","USDC","USDD","TUSD","BUSD","DAI","FRAX","FDUSD","USDP","GUSD",
        "PYUSD","USDS","USDE","USD1","USDI","USDJ","USDK","USDX","USTC","EUR",
        "EURT","EURS","GBP","GBPT","USDR","USDN","USDB",
    }
    if b in stable_set:
        return True
    # Catch USD* tokens like USD1, USDC, USDT, USDP, etc.
    if re.match(r"^USD[A-Z0-9]{0,4}$", b):
        return True
    # Catch *USD (rare, but defensive)
    if b.endswith("USD"):
        return True
    return False

class VolatilityScannerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Volatility Scanner")
        self.geometry("1300x800") # Slightly wider for new columns

        # 資料變數
        self.scan_results = None  # DataFrame

        self.is_scanning = False
        self.stop_event = threading.Event()

        # --- 上方控制區 ---
        ctrl_frame = ttk.LabelFrame(self, text="Scanner Settings")
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 第一排：交易所、Quote、Interval、天數
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(row1, text="Exchange:").pack(side=tk.LEFT)
        self.cb_exchange = ttk.Combobox(row1, values=["binance", "bybit", "okx", "gateio"], width=10)
        self.cb_exchange.set("binance")
        self.cb_exchange.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Quote Asset:").pack(side=tk.LEFT)
        self.e_quote = ttk.Entry(row1, width=8)
        self.e_quote.insert(0, "USDT")
        self.e_quote.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Interval:").pack(side=tk.LEFT)
        self.cb_interval = ttk.Combobox(row1, values=["5m", "15m", "1h", "4h", "1d"], width=6)
        self.cb_interval.set("4h")
        self.cb_interval.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="Scan Days:").pack(side=tk.LEFT)
        self.e_days = ttk.Entry(row1, width=6)
        self.e_days.insert(0, "730")
        self.e_days.pack(side=tk.LEFT, padx=5)

        # 第二排：過濾條件 (Volume, Top N)
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(row2, text="Pre-filter 24h Vol (M):").pack(side=tk.LEFT)
        self.e_min_vol = ttk.Entry(row2, width=8)
        self.e_min_vol.insert(0, "10") # 10M (Pre-filter)
        self.e_min_vol.pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Min Avg Daily Vol (M):").pack(side=tk.LEFT)
        self.e_min_avg_vol = ttk.Entry(row2, width=8)
        self.e_min_avg_vol.insert(0, "40") # 40M (Avg Vol)
        self.e_min_avg_vol.pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Scan Top N (by Vol):").pack(side=tk.LEFT)
        self.e_top_n = ttk.Entry(row2, width=6)
        self.e_top_n.insert(0, "50")
        self.e_top_n.pack(side=tk.LEFT, padx=5)

        self.btn_scan = ttk.Button(row2, text="Start Scan", command=self.start_scan_thread)
        self.btn_scan.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(row2, text="Stop", command=self.stop_scan, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.lbl_status = ttk.Label(row2, text="Ready.", foreground="#00FF00")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # 進度條
        self.progress = ttk.Progressbar(row2, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=10)

        # 第三排：Market Cap Filter
        row3 = ttk.Frame(ctrl_frame)
        row3.pack(fill=tk.X, padx=5, pady=2)
        
        self.chk_mc_filter_var = tk.BooleanVar(value=True)
        self.chk_mc_filter = ttk.Checkbutton(row3, text="Filter Top Rank (CoinGecko)", variable=self.chk_mc_filter_var)
        self.chk_mc_filter.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row3, text="Max Rank:").pack(side=tk.LEFT)
        self.e_max_rank = ttk.Entry(row3, width=6)
        self.e_max_rank.insert(0, "100")
        self.e_max_rank.pack(side=tk.LEFT, padx=5)

        # 第四排：手動輸入 Symbol
        row4 = ttk.Frame(ctrl_frame)
        row4.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(row4, text="Manual Add (Base):").pack(side=tk.LEFT)
        self.e_manual = ttk.Entry(row4, width=40)
        self.e_manual.pack(side=tk.LEFT, padx=5)
        ttk.Label(row4, text="(e.g. SUI, BTC. Separated by space/comma)").pack(side=tk.LEFT)

        # --- 主畫面分割 (左: 表格, 右: 圖表) ---
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左側表格
        frame_table = ttk.Frame(paned)
        paned.add(frame_table, weight=1)

        # Updated columns: Removed ATR(Day)%, Removed BBW(%), Added ATR(Month)%, Added MC Rank
        cols = ("Symbol", "Price", "Avg_Vol(M)", "RV(Ann)%", "ATR(Month)%", "Max_DD(%)", "Change(%)", "MC Rank")
        self.tree = ttk.Treeview(frame_table, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c, command=lambda _c=c: self.sort_tree(_c, False))
            self.tree.column(c, width=75, anchor="center")
        
        # Scrollbar
        sb = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # 右側圖表
        frame_chart = ttk.Frame(paned)
        paned.add(frame_chart, weight=1)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_chart)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.lbl_status.config(text=msg)
        self.update_idletasks()

    def start_scan_thread(self):
        if self.is_scanning:
            return
        self.is_scanning = True
        self.stop_event.clear()
        self.btn_scan.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress['value'] = 0
        threading.Thread(target=self.run_scan, daemon=True).start()

    def stop_scan(self):
        if self.is_scanning:
            self.stop_event.set()
            self.log("Stopping scan...")
            self.btn_stop.config(state=tk.DISABLED)

    def run_scan(self):
        try:
            exch_name = self.cb_exchange.get()
            quote_asset = self.e_quote.get().upper()
            interval = self.cb_interval.get()
            days = int(self.e_days.get())
            min_vol_pre = float(self.e_min_vol.get()) * 1_000_000
            min_avg_vol = float(self.e_min_avg_vol.get()) * 1_000_000
            top_n = int(self.e_top_n.get())

            use_mc_filter = self.chk_mc_filter_var.get()
            max_rank = int(self.e_max_rank.get())
            
            mc_mapping = {}
            if use_mc_filter:
                self.log("Fetching Market Cap Rank from CoinGecko...")
                mc_mapping = fetch_top_mc_coins(limit=max(max_rank, 250))
                if not mc_mapping:
                    self.log("Warning: CoinGecko fetch failed. MC filter ignored.")

            self.log(f"Fetching tickers from {exch_name}...")
            
            # 1. 初始化交易所
            ex_cls = getattr(ccxt, exch_name)()
            markets = ex_cls.load_markets()
            
            # 2. 篩選 Tickers (Spot Only for simplicity, or allow perps if user wants)
            # 這裡我們主要找 Spot 或 Swap，優先找 Quote 符合的
            tickers = ex_cls.fetch_tickers()
            
            # Prepare manual symbols
            manual_input = self.e_manual.get().strip()
            manual_bases = set()
            if manual_input:
                # Split by comma or space
                parts = manual_input.replace(",", " ").split()
                for p in parts:
                    p = p.strip().upper()
                    # Remove quote if user typed SUI/USDT
                    if '/' in p:
                        p = p.split('/')[0]
                    manual_bases.add(p)
            
            # Construct target manual symbols (e.g. SUI/USDT) to check existence
            manual_pairs_needed = {f"{b}/{quote_asset}" for b in manual_bases}
            manual_pairs_found = set()

            candidates = []
            for symbol, ticker in tickers.items():
                if not ticker: continue
                # 簡單過濾：必須包含 Quote Asset (e.g. BTC/USDT)
                if f"/{quote_asset}" not in symbol and not symbol.endswith(quote_asset):
                    continue
                
                base = symbol.split('/')[0] if '/' in symbol else symbol.replace(quote_asset, "")

                # Check if this is a manual override symbol
                is_manual = (base in manual_bases)
                if is_manual:
                    # Check partial match or exact match depending on exchange format
                    # Most CCXT tickers are BASE/QUOTE
                    if symbol == f"{base}/{quote_asset}":
                         manual_pairs_found.add(symbol)
                
                if not is_manual:
                    # 過濾 Stablecoins
                    if is_stablecoin_base(base):
                        continue
                    
                    # Market Cap Filter
                    rank = -1
                    if mc_mapping:
                        base_lower = base.lower()
                        # Mapping usually has 'btc', 'eth'. Tickers are 'BTC/USDT'.
                        if base_lower in mc_mapping:
                            rank = mc_mapping[base_lower]
                            if rank > max_rank:
                                continue # Rank too low (number too high)
                        else:
                            continue # Not in top N list
                else:
                    # For manual, we still try to get rank if available, but don't filter
                    rank = -1
                    if mc_mapping:
                        base_lower = base.lower()
                        if base_lower in mc_mapping:
                            rank = mc_mapping[base_lower]

                # 過濾掉槓桿代幣等 (簡單判斷: 包含 DOWN, UP, BEAR, BULL 且長度怪怪的，這裡先不嚴格過濾)
                
                vol = ticker.get('quoteVolume') or 0
                if not is_manual and vol < min_vol_pre:
                    continue
                
                candidates.append({
                    'symbol': symbol,
                    'volume': vol,
                    'close': ticker.get('close'),
                    'mc_rank': rank,
                    'is_manual': is_manual
                })

            
            # Check for missing manual symbols
            if manual_bases:
                # manual_pairs_needed vs manual_pairs_found
                # We simply check if we found a valid ticker for the base
                # Actually, our manual_pairs_found logic above only adds if exact match BASE/QUOTE
                # Let's check which BASES were not found
                found_bases = {p.split('/')[0] for p in manual_pairs_found}
                missing_bases = manual_bases - found_bases
                if missing_bases:
                    msg = f"Warning: The following manual symbols were not found on {exch_name} (Quote: {quote_asset}):\n" + ", ".join(missing_bases)
                    self.after(0, lambda: messagebox.showwarning("Symbol Not Found", msg))
            # Separate candidates
            manual_candidates = [c for c in candidates if c['symbol'] in manual_pairs_found]
            auto_candidates = [c for c in candidates if c['symbol'] not in manual_pairs_found]
            
            # Sort Auto candidates by volume and take Top N
            auto_candidates.sort(key=lambda x: x['volume'], reverse=True)
            auto_candidates = auto_candidates[:top_n]
            
            # Combine: Manual + Auto (ensure no duplicates, though manual_pairs_found check should handle it)
            # Priorities manual first
            final_candidates = manual_candidates + auto_candidates
            candidates = final_candidates

            total_cands = len(candidates)
            self.log(f"Found {total_cands} candidates. Fetching K-lines (Parallel)...")
            
            # 設定進度條最大值
            self.after(0, lambda: self.progress.configure(maximum=total_cands))

            results = []
            
            # 計算需要的 bars 數 / 時間區間
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            # 準備參數供平行處理
            scan_args = {
                "interval": interval,
                "start_str": start_str,
                "end_str": end_str,
                "exch_name": exch_name,
                "quote_asset": quote_asset,
                "days": days,
                "min_avg_vol": min_avg_vol
            }

            completed_count = 0
            
            # 使用 ThreadPoolExecutor 進行平行掃描
            # 建議 worker 數不要太多，以免觸發 API Rate Limit (雖然 martin.py 有 retry)
            # 一般 5~10 左右
            max_workers = 8 
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任務
                future_to_cand = {executor.submit(self.process_coin, cand, scan_args): cand for cand in candidates}
                
                for future in concurrent.futures.as_completed(future_to_cand):
                    if self.stop_event.is_set():
                        break
                    
                    cand = future_to_cand[future]
                    sym = cand['symbol']
                    
                    try:
                        res = future.result()
                        if res:
                            results.append(res)
                    except Exception as e:
                        print(f"Error processing {sym}: {e}")
                    
                    completed_count += 1
                    # 更新進度條與狀態
                    self.after(0, lambda v=completed_count: self.progress.configure(value=v))
                    self.log(f"Scanning {completed_count}/{total_cands}...")

            if self.stop_event.is_set():
                self.log("Scan stopped by user.")
            else:
                self.log("Scan completed.")

            self.scan_results = pd.DataFrame(results)
            self.after(0, self.update_table)

        except Exception as e:
            traceback.print_exc()
            err_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", err_msg))
            self.after(0, lambda: self.log("Scan failed."))
        finally:
            self.is_scanning = False
            self.after(0, lambda: self.btn_scan.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

    def process_coin(self, cand, args):
        """單一幣種處理邏輯 (在 Worker Thread 執行)"""
        if self.stop_event.is_set():
            return None
            
        sym = cand['symbol']
        interval = args['interval']
        start_str = args['start_str']
        end_str = args['end_str']
        exch_name = args['exch_name']
        quote_asset = args['quote_asset']
        days = args['days']
        days = args['days']
        min_avg_vol = args['min_avg_vol']
        is_manual = cand.get('is_manual', False)

        try:
            base = sym.split('/')[0] if '/' in sym else sym.replace(quote_asset, "")
            
            # 呼叫 martin.get_klines
            df = martin.get_klines(
                symbol=base,
                interval=interval,
                start=start_str,
                end=end_str,
                exch_list=[exch_name],
                refresh_policy="auto"
            )
            
            if df is None or df.empty or len(df) < 50:
                return None

            # --- 計算指標 ---
            closes = df['close'].astype(float)
            
            # 1. 漲跌幅
            change = (closes.iloc[-1] / closes.iloc[0]) - 1.0

            # 2. 年化波動率 (RV Annual)
            # 計算 bars_per_year
            step_ms = martin._interval_ms(interval)
            bars_per_year = (365.25 * 24 * 3600) / (step_ms / 1000.0)
            rv_annual = realized_vol_annual_from_closes(closes, bars_per_year)

            # 3. ATR %
            atr_p = true_atr_pct(df, n=14)
            if not np.isfinite(atr_p):
                atr_p = approx_atr_pct_from_close(closes, n=14)
            
            # 3.1 ATR Daily Estimate (Square Root of Time Rule)
            # bars_per_day = 24h / interval_hours
            bars_per_day = 24 * 3600 * 1000 / step_ms
            atr_daily_est = atr_p * np.sqrt(bars_per_day)

            # 3.2 ATR Monthly Estimate (Approx 30 days)
            atr_monthly_est = atr_daily_est * np.sqrt(30)

            # 4. BB Width %
            bbw_p = bb_width_pct(closes, n=20, k=2.0)

            # 5. 最大回撤
            roll_max = closes.cummax()
            dd = (closes - roll_max) / roll_max
            max_dd = dd.min()

            # 6. 平均日成交量
            quote_vols = df['volume'] * df['close']
            total_quote_vol = quote_vols.sum()
            
            time_span_days = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 86400.0
            
            # Relaxed check: Allow coins with > 30 days history even if requested 'days' is larger
            # This ensures we don't miss new volatile listings.
            if time_span_days < (days - 1) and time_span_days < 30 and not is_manual:
                return None

            if time_span_days < 0.5: time_span_days = 0.5
            
            avg_daily_vol = total_quote_vol / time_span_days
            
            if avg_daily_vol < min_avg_vol and not is_manual:
                return None

            return {
                "Symbol": sym,
                "Price": closes.iloc[-1],
                "Avg_Vol(M)": avg_daily_vol / 1_000_000,
                "RV(Ann)%": rv_annual * 100,
                "ATR(Day)%": atr_daily_est * 100,
                "ATR(Month)%": atr_monthly_est * 100,
                "BBW(%)": bbw_p * 100,
                "Max_DD(%)": max_dd * 100,
                "Change(%)": change * 100,
                "MC Rank": cand.get('mc_rank', -1),
                "df": df
            }

        except Exception as e:
            # print(f"Error in process_coin {sym}: {e}")
            return None

    def update_table(self):
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.scan_results is None or self.scan_results.empty:
            return

        # 預設依 RV(Ann)% 排序
        if "RV(Ann)%" in self.scan_results.columns:
            self.scan_results.sort_values("RV(Ann)%", ascending=False, inplace=True)

        for _, row in self.scan_results.iterrows():
            p_val = float(row['Price'])
            p_str = f"{p_val:.8f}" if p_val < 0.01 else f"{p_val:.4f}"
            vals = (
                row["Symbol"],
                p_str,
                f"{row['Avg_Vol(M)']:.2f}",
                f"{row['RV(Ann)%']:.2f}",
                f"{row['ATR(Month)%']:.2f}",
                f"{row['Max_DD(%)']:.2f}",
                f"{row['Change(%)']:.2f}",
                f"{int(row['MC Rank'])}" if row['MC Rank'] > 0 else "-"
            )
            self.tree.insert("", tk.END, values=vals)

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        item = self.tree.item(sel[0])
        sym = item['values'][0]
        
        # 找回對應的 df
        record = self.scan_results[self.scan_results["Symbol"] == sym].iloc[0]
        df = record["df"]
        
        self.plot_chart(df, sym)

    def plot_chart(self, df, title):
        self.ax.clear()
        self.ax.plot(df['time'], df['close'], label='Close')
        self.ax.set_title(f"{title} - {self.cb_interval.get()}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def sort_tree(self, col, reverse):
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        # 嘗試轉成 float 排序
        try:
            l.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            l.sort(reverse=reverse)

        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)

        self.tree.heading(col, command=lambda: self.sort_tree(col, not reverse))

if __name__ == "__main__":
    app = VolatilityScannerGUI()
    app.mainloop()
