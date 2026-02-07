# martin_gui.py — Thread-safe GUI (compute in threads, UI on main), vertical split with "sash when ready"
# Adds: Trapped Time Ratio in metrics log — 2025-10-11
# -*- coding: utf-8 -*-

import threading as _thr
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pathlib import Path

# ---- 匯入 martin.py ----
try:
    import martin
except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("Import Error", f"無法匯入 martin.py：\n{e}")
    raise


def parse_range_or_list(s: str, is_int=False):
    s = (s or "").strip()
    if not s:
        return np.array([], dtype=int if is_int else float)
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError("範圍格式應為 start:end:step")
        a, b, c = parts
        a = (int(a) if is_int else float(a))
        b = (int(b) if is_int else float(b))
        c = (int(c) if is_int else float(c))
        if c == 0:
            raise ValueError("step 不可為 0")
        if is_int:
            return np.arange(a, b + (1 if c > 0 else -1), c, dtype=int)
        vals = []
        x = a
        forward = c > 0
        if forward:
            while x <= b + 1e-12:
                vals.append(x); x += c
        else:
            while x >= b - 1e-12:
                vals.append(x); x += c
        return np.array(vals, dtype=float)
    items = [t.strip() for t in s.split(",") if t.strip()]
    return np.array([int(x) for x in items], dtype=int) if is_int else np.array([float(x) for x in items], dtype=float)


def safe_float(s, default=None):
    try: return float(s)
    except: return default


def safe_int(s, default=None):
    try: return int(s)
    except: return default


def human_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "NaN"
    if isinstance(x, float) and np.isinf(x): return "∞"
    return f"{x*100:.{digits}f}%"


class MartinGUI(tk.Tk):
    # --- tolerant float input helpers ---
    def _validate_float_tolerant(self, proposed: str) -> bool:
        # allow empty and in-progress states
        if proposed in ("", "+", "-", ".", "+.", "-."):
            return True
        try:
            float(proposed)
            return True
        except Exception:
            return False

    def _normalize_float_field(self, var, ndigits=1):
        s = (var.get() or "").strip()
        try:
            var.set(f"{float(s):.{ndigits}f}")
        except Exception:
            pass

    INIT_SPLIT = 0.70  # 初次顯示時設定一次比例；之後可自由拖曳

    def __init__(self):
        super().__init__()
        self.title("Spot Martingale GUI")
        self.geometry("1220x960")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.tab_scan = ttk.Frame(self.nb)
        self.tab_single = ttk.Frame(self.nb)
        self.nb.add(self.tab_scan, text="Parameters Scan")
        self.nb.add(self.tab_single, text="Single Backtest")

        # handles
        self.figure_scan = None; self.canvas_scan = None; self.scan_metrics_text = None
        self.figure_single = None; self.canvas_single = None; self.metrics_text = None
        self.scan_paned = None; self.single_paned = None

        # shared cache（僅作為資料快取）
        self.last_df = None; self.last_df_key = None
        self.scan_df = None  # 最新表格對應的非格式化 DataFrame

        self.build_scan_tab()
        self.build_single_tab()

        # 啟動後排程一次初始化（等待高度就緒）
        self.after(0, lambda: self._set_sash_when_ready(self.scan_paned, self.INIT_SPLIT))
        self.after(0, lambda: self._set_sash_when_ready(self.single_paned, self.INIT_SPLIT))

        # 切換掃描子分頁時，如果是進到「Backtest Chart / Performance」，補一次
        self.scan_subnb.bind(
            "<<NotebookTabChanged>>",
            lambda e: (self.scan_subnb.select() == str(self.scan_tab_detail)) and
                      self._set_sash_when_ready(self.scan_paned, self.INIT_SPLIT)
        )
        # 切換主分頁時也補一次（避免第一次顯示高度 0）
        self.nb.bind("<<NotebookTabChanged>>",
                     lambda e: self._set_sash_when_ready(self.single_paned, self.INIT_SPLIT))

    # ---------- Parameters Scan（子分頁） ----------
    def build_scan_tab(self):
        top = ttk.Frame(self.tab_scan)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        lf_data = ttk.LabelFrame(top, text="Data Settings")
        lf_data.pack(side=tk.LEFT, padx=4, pady=4, fill=tk.X)
        self.e_symbol   = self._add_entry(lf_data, "Symbol:", "SUI", 10)
        self.e_interval = self._add_entry(lf_data, "Interval：", "15m", 10)
        self.e_scan_days = self._add_entry(lf_data, "Scan Days:", "730", 10)
        self.e_start    = self._add_entry(lf_data, "Start (e.g.2025-07-01):", "", 12)
        self.e_end      = self._add_entry(lf_data, "End (e.g.2025-08-30):", "", 12)
        self.e_refresh  = self._add_entry(lf_data, "Cache Policy:", "auto", 10)
        self.e_fee      = self._add_entry(lf_data, "Fee rate:", "0.0005", 10)
        self.e_capital  = self._add_entry(lf_data, "Capital:", "1000", 10)

        lf_grid = ttk.LabelFrame(top, text="Scan Parameters (range 'start:end:step' or list 'a,b,c')")
        lf_grid.pack(side=tk.LEFT, padx=4, pady=4, fill=tk.BOTH, expand=True)
        self.e_add_drop   = self._add_entry(lf_grid, "add_drop：", "0.010:0.080:0.001", 22)
        self.e_tp         = self._add_entry(lf_grid, "tp：", "0.010:0.080:0.001", 22)
        self.e_multiplier = self._add_entry(lf_grid, "multiplier：", "1.5:2.0:0.1", 22)
        self.e_max_orders = self._add_entry(lf_grid, "max_orders：", "5:9:1", 22)

        lf_filter = ttk.LabelFrame(top, text="Filter Conditions (optional)")
        lf_filter.pack(side=tk.LEFT, padx=4, pady=4, fill=tk.Y)
        self.e_min_trades = self._add_entry(lf_filter, "min_trades：", "104", 10)
        self.e_max_dd     = self._add_entry(lf_filter, "max_dd_overall(%)：", "", 10)
        self.e_max_trap   = self._add_entry(lf_filter, "max_trapped_ratio(%)：", "20", 10)
        self.e_topn       = self._add_entry(lf_filter, "Show Top N:", "20", 10)

        btns = ttk.Frame(self.tab_scan); btns.pack(fill=tk.X, padx=8)
        ttk.Button(btns, text="Run Scan", command=self.run_scan_thread).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save Result as CSV", command=self.save_scan_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save Result as Markdown", command=self.save_scan_markdown).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear Results", command=self.clear_scan_results).pack(side=tk.LEFT, padx=4)

        # 子分頁：Results Table / 回測圖＋績效（上下）
        self.scan_subnb = ttk.Notebook(self.tab_scan)
        self.scan_subnb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.scan_tab_table = ttk.Frame(self.scan_subnb)
        self.scan_tab_detail = ttk.Frame(self.scan_subnb)
        self.scan_subnb.add(self.scan_tab_table, text="Results Table")
        self.scan_subnb.add(self.scan_tab_detail, text="Backtest Chart / Performance")

        # 表格
        cols = ("add_drop","tp","multiplier","max_orders","min_buy_ratio","final_equity","max_dd_overall","trades","trapped_time_ratio")
        self.tree = ttk.Treeview(self.scan_tab_table, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c); self.tree.column(c, width=120, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        ctrl = ttk.Frame(self.scan_tab_table); ctrl.pack(pady=6)
        self.tree.bind("<Double-1>", lambda e: self.plot_selected_from_table_thread())
        ttk.Label(self.tab_scan, text="提示：雙擊表格繪圖，會自動切換到「Backtest Chart / Performance」。").pack(anchor="w", padx=10, pady=(0,6))

        # 回測圖 + 績效（上下，可拖曳）
        self.scan_paned = ttk.Panedwindow(self.scan_tab_detail, orient=tk.VERTICAL)
        self.scan_paned.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(self.scan_paned)
        metrics_frame = ttk.Frame(self.scan_paned)

        self.scan_paned.add(plot_frame, weight=7)
        self.scan_paned.add(metrics_frame, weight=3)

        self.figure_scan = Figure(figsize=(6,4), dpi=100)
        self.canvas_scan = FigureCanvasTkAgg(self.figure_scan, master=plot_frame)
        self.canvas_scan.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.scan_metrics_text = tk.Text(metrics_frame, height=8)
        self.scan_metrics_text.pack(fill=tk.BOTH, expand=True)

    # ---------- Single Backtest（上下，可拖曳） ----------
    def build_single_tab(self):
        top = ttk.Frame(self.tab_single)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        lf_data = ttk.LabelFrame(top, text="Data Settings")
        lf_data.pack(side=tk.LEFT, padx=4, pady=4, fill=tk.X)
        self.s_symbol   = self._add_entry(lf_data, "Symbol:", "SUI", 10)
        self.s_interval = self._add_entry(lf_data, "Interval：", "15m", 10)
        self.s_scan_days = self._add_entry(lf_data, "Scan Days:", "365", 10)
        self.s_start    = self._add_entry(lf_data, "Start (e.g.2025-07-01):", "", 12)
        self.s_end      = self._add_entry(lf_data, "End (e.g.2025-08-30):", "", 12)
        self.s_refresh  = self._add_entry(lf_data, "Cache Policy:", "auto", 10)
        self.s_fee      = self._add_entry(lf_data, "Fee rate:", "0.0005", 10)
        self.s_capital  = self._add_entry(lf_data, "Capital:", "1000", 10)

        lf_params = ttk.LabelFrame(top, text="Strategy Parameters")
        lf_params.pack(side=tk.LEFT, padx=4, pady=4, fill=tk.X)
        self.add_drop_var = tk.StringVar(value="0.015")
        self.s_add_drop = self._add_entry(lf_params, "add_drop：", "0.015", 10, textvariable=self.add_drop_var)
        self.tp_var = tk.StringVar(value="0.018")
        self.s_tp = self._add_entry(lf_params, "tp：", "0.018", 10, textvariable=self.tp_var)
        self.multiplier_var = tk.StringVar(value="1.5")
        self.s_multiplier = self._add_entry(lf_params, "multiplier：", "1.5", 10, textvariable=self.multiplier_var)

        self.s_max_orders = self._add_entry(lf_params, "max_orders：", "7", 10)

        btns = ttk.Frame(self.tab_single); btns.pack(fill=tk.X, padx=8)
        ttk.Button(btns, text="Execute Single Backtest and Plot", command=self.run_single_thread).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear Results", command=self.clear_single_results).pack(side=tk.LEFT, padx=4)

        # 垂直分割（可拖曳）
        self.single_paned = ttk.Panedwindow(self.tab_single, orient=tk.VERTICAL)
        self.single_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        plot_frame = ttk.Frame(self.single_paned)
        metrics_frame = ttk.Frame(self.single_paned)

        self.single_paned.add(plot_frame, weight=7)
        self.single_paned.add(metrics_frame, weight=3)

        self.figure_single = Figure(figsize=(6,4), dpi=100)
        self.canvas_single = FigureCanvasTkAgg(self.figure_single, master=plot_frame)
        self.canvas_single.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(metrics_frame, height=8)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

    # ---------- 等高度就緒才設定分隔條，避免初始高度 = 0 ----------
    def _set_sash_when_ready(self, paned, ratio=0.70, tries=20, delay=60):
        if paned is None or not str(paned):
            return
        try:
            paned.update_idletasks()
            h = paned.winfo_height()
        except Exception:
            h = 0
        if h and h > 50:
            try:
                paned.sashpos(0, int(h * float(ratio)))
            except Exception:
                pass
            return
        if tries <= 0:
            return
        self.after(delay, lambda: self._set_sash_when_ready(paned, ratio, tries-1, delay))

    # ---------- 小工具 ----------
    def _add_entry(self, parent, label, default="", width=12, **entry_kwargs):
        row = ttk.Frame(parent); row.pack(anchor="w", pady=2, fill=tk.X)
        ttk.Label(row, text=label, width=24, anchor="w").pack(side=tk.LEFT)
        e = ttk.Entry(row, width=width, **entry_kwargs)
        tv = entry_kwargs.get('textvariable')
        if tv is not None:
            tv.set(default)
        else:
            e.insert(0, default)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return e

    # ---------- 數據擷取（帶快取） ----------
    def _fetch_klines_if_needed(self, symbol, interval, bars, start, end, refresh_policy):
        key = (symbol, interval, bars, start, end, refresh_policy)
        if self.last_df is not None and self.last_df_key == key:
            return self.last_df
        if bars:
            df = martin.get_klines(symbol=symbol, interval=interval, bars=bars,
                                   cache_dir=martin.DEFAULT_CACHE_DIR, use_cache=True,
                                   refresh_policy=refresh_policy)
        else:
            df = martin.get_klines(symbol=symbol, interval=interval, start=start, end=end,
                                   cache_dir=martin.DEFAULT_CACHE_DIR, use_cache=True,
                                   refresh_policy=refresh_policy)
        self.last_df = df; self.last_df_key = key
        return df

    # =================== Parameters Scan（Thread → Main）===================
    def run_scan_thread(self):
        self.status_var.set("Scanning…（首次可能較慢，numba 正在編譯）")
        _thr.Thread(target=self._scan_compute_then_update, daemon=True).start()

    def _scan_compute_then_update(self):
        try:
            symbol = self.e_symbol.get().strip(); interval = self.e_interval.get().strip()
            scan_days = safe_float(self.e_scan_days.get().strip(), 0.0); start = self.e_start.get().strip(); end = self.e_end.get().strip()
            bars = 0
            if scan_days > 0:
                step_ms = martin._interval_ms(interval)
                bars = int(scan_days * 86400 * 1000 / step_ms)

            refresh_policy = (self.e_refresh.get().strip() or "auto")
            fee_rate = safe_float(self.e_fee.get().strip(), 0.0); capital = safe_float(self.e_capital.get().strip(), 1000.0)
            if not symbol or not interval: raise ValueError("請填入 symbol 與 interval")
            if (not bars) and (not start or not end): raise ValueError("Scan Days 與 Start/End 需擇一填寫")

            df = self._fetch_klines_if_needed(symbol, interval, bars, start, end, refresh_policy)
            prices_np = df["close"].to_numpy(dtype=np.float64)
            if prices_np.size < 2: raise ValueError("K 線資料不足（<2 根），無法回測/掃描。")

            add_drop_arr = parse_range_or_list(self.e_add_drop.get())
            tp_arr       = parse_range_or_list(self.e_tp.get())
            mul_arr      = parse_range_or_list(self.e_multiplier.get())
            mo_arr       = parse_range_or_list(self.e_max_orders.get(), is_int=True)
            if any(x.size == 0 for x in (add_drop_arr, tp_arr, mul_arr, mo_arr)):
                raise ValueError("掃描參數不得為空（add_drop/tp/multiplier/max_orders）")

            AD, MUL, MO, TP = np.meshgrid(add_drop_arr, mul_arr, mo_arr, tp_arr, indexing="ij")
            ADf = AD.ravel().astype(np.float64); MULf = MUL.ravel().astype(np.float64)
            MOf = MO.ravel().astype(np.int32);   TPf  = TP.ravel().astype(np.float64)
            min_buy_ratio_theory = np.maximum(0.0, (1.0 - ADf) ** (MOf.astype(np.float64) - 1.0))

            fe, mdd, tr, trap = martin._grid_search_parallel(
                prices_np, ADf, MULf, MOf, TPf, capital=float(capital), fee_rate=float(fee_rate)
            )
            results_df = pd.DataFrame({
                "add_drop": ADf, "multiplier": MULf, "max_orders": MOf.astype(int), "tp": TPf,
                "capital": float(capital), "final_equity": np.round(fe, 2),
                "max_dd_overall": np.round(mdd, 2), "trades": tr.astype(int),
                "trapped_time_ratio": np.round(trap, 6), "min_buy_ratio": np.round(min_buy_ratio_theory, 6)
            })

            min_trades = safe_int(self.e_min_trades.get().strip()) if self.e_min_trades.get().strip() else None
            max_dd     = safe_float(self.e_max_dd.get().strip())   if self.e_max_dd.get().strip() else None
            max_trap   = safe_float(self.e_max_trap.get().strip()) if self.e_max_trap.get().strip() else None
            if max_trap is not None:
                max_trap /= 100.0  # 使用者以百分比輸入（例：15 => 0.15）
            filtered = martin.apply_filters(results_df, min_trades, max_dd, max_trap)
            if filtered.empty:
                filtered = results_df

            topn = safe_int(self.e_topn.get().strip(), 20)
            top_df = filtered.nlargest(topn, "final_equity").copy()

            self.after(0, lambda df=top_df: self._scan_update_ui(df))
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, lambda: (self.status_var.set("掃描失敗。"),
                                   messagebox.showerror("掃描錯誤", f"{e}\n\n{tb}")))

    def _scan_update_ui(self, top_df: pd.DataFrame):
        for item in self.tree.get_children(): self.tree.delete(item)
        disp = top_df.copy()
        disp["add_drop"] = disp["add_drop"].apply(lambda v: martin.pct_str(v, 1))
        disp["tp"]       = disp["tp"].apply(lambda v: martin.pct_str(v, 1))
        disp["multiplier"] = disp["multiplier"].apply(lambda v: f"{v:.1f}")
        disp["trapped_time_ratio"] = disp["trapped_time_ratio"].apply(lambda v: martin.pct_str(v, 2))
        disp["min_buy_ratio"] = disp["min_buy_ratio"].round(2)
        cols = ["add_drop","tp","multiplier","max_orders","min_buy_ratio","final_equity","max_dd_overall","trades","trapped_time_ratio"]
        for _, row in disp[cols].iterrows():
            self.tree.insert("", tk.END, values=[row[c] for c in cols])

        self.scan_df = top_df.reset_index(drop=True)
        self.status_var.set("Scan completed。")
        self.scan_subnb.select(self.scan_tab_table)

    def save_scan_markdown(self):
        """Save scan results as Markdown with GUI-aligned columns and per-column formatting (aligned table)"""
        if self.scan_df is None or self.scan_df.empty:
            messagebox.showinfo("Info", "No scan results to save. Please run a scan first.")
            return
        fpath = filedialog.asksaveasfilename(
            title="Save Scan Results as Markdown", defaultextension=".md",
            filetypes=[("Markdown", "*.md")]
        )
        if not fpath:
            return
        try:
            out = self.scan_df.copy()

            # GUI-aligned columns
            cols = ["add_drop","tp","multiplier","max_orders","min_buy_ratio",
                    "final_equity","max_dd_overall","trades","trapped_time_ratio"]

            # Apply same display formatting as CSV
            out["add_drop"] = (out["add_drop"] * 100.0).map(lambda v: f"{float(v):.1f}")
            out["tp"] = (out["tp"] * 100.0).map(lambda v: f"{float(v):.1f}")
            out["trapped_time_ratio"] = (out["trapped_time_ratio"] * 100.0).map(lambda v: f"{float(v):.2f}")
            out["multiplier"] = out["multiplier"].map(lambda v: f"{float(v):.1f}")
            out["max_orders"] = out["max_orders"].astype(int).astype(str)
            out["trades"] = out["trades"].astype(int).astype(str)
            out["min_buy_ratio"] = out["min_buy_ratio"].map(lambda v: f"{float(v):.2f}")
            out["final_equity"] = out["final_equity"].map(lambda v: f"{float(v):.2f}")
            out["max_dd_overall"] = out["max_dd_overall"].map(lambda v: f"{float(v):.2f}")

            df_md = out[cols].copy()

            # Convert all to strings for width calculation
            for c in cols:
                df_md[c] = df_md[c].astype(str)

            # Compute column widths for alignment
            col_widths = [max(len(str(col)), df_md[col].map(len).max()) for col in cols]
            header = "| " + " | ".join(f"{col.ljust(col_widths[i])}" for i, col in enumerate(cols)) + " |"
            sep = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(cols))) + "-|"

            rows = []
            for _, row in df_md.iterrows():
                row_str = "| " + " | ".join(f"{row[col].ljust(col_widths[i])}" for i, col in enumerate(cols)) + " |"
                rows.append(row_str)

            md_table = "\n".join([header, sep] + rows)
            with open(fpath, "w", encoding="utf-8-sig") as _fp:
                _fp.write(md_table)
            messagebox.showinfo("Done", f"Saved: {fpath}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Save Failed", f"{e}")


            def fmt_pct_num(x):
                try:
                    v = float(x) * 100.0
                    s = f"{v:.3f}".rstrip("0").rstrip(".")
                    return s
                except Exception:
                    return str(x)

            out["add_drop"] = out["add_drop"].map(fmt_pct_num)
            out["tp"] = out["tp"].map(fmt_pct_num)
            out["trapped_time_ratio"] = out["trapped_time_ratio"].map(fmt_pct_num)

            # Integers for these
            out["max_orders"] = out["max_orders"].astype(int).astype(str)
            out["trades"] = out["trades"].astype(int).astype(str)

            # Keep other columns as strings
            out["final_equity"] = out["final_equity"].astype(str)
            out["max_dd_overall"] = out["max_dd_overall"].astype(str)
            out["multiplier"] = out["multiplier"].astype(str)

            # Select columns identical to CSV export order
            cols = ["add_drop", "tp", "multiplier", "max_orders",
                    "final_equity", "max_dd_overall", "trades", "trapped_time_ratio"]
            df_md = out[cols].copy()

            # Compute widths
            col_widths = [max(len(str(col)), df_md[col].astype(str).map(len).max()) for col in cols]
            header = "| " + " | ".join(f"{col.ljust(col_widths[i])}" for i, col in enumerate(cols)) + " |"
            sep = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(cols))) + "-|"

            rows = []
            for _, row in df_md.iterrows():
                row_str = "| " + " | ".join(f"{str(row[col]).ljust(col_widths[i])}" for i, col in enumerate(cols)) + " |"
                rows.append(row_str)

            md_table = "\n".join([header, sep] + rows)
            with open(fpath, "w", encoding="utf-8-sig") as _fp:
                _fp.write(md_table)
            messagebox.showinfo("Done", f"Saved: {fpath}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Save Failed", f"{e}")

    def save_scan_csv(self):
        """Save scan results to CSV with GUI-aligned columns and per-column formatting"""
        if self.scan_df is None or self.scan_df.empty:
            messagebox.showinfo("提示", "沒有可儲存的掃描結果，請先Run Scan。")
            return
        fpath = filedialog.asksaveasfilename(
            title="另存掃描結果為 CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not fpath:
            return
        try:
            out = self.scan_df.copy()

            # GUI-aligned columns
            cols = ["add_drop","tp","multiplier","max_orders","min_buy_ratio",
                    "final_equity","max_dd_overall","trades","trapped_time_ratio"]

            # Convert to display formats:
            # - add_drop, tp, trapped_time_ratio: percentage numeric (ratio*100) with 1 decimal (e.g., 0.015 -> "1.5")
            out["add_drop"] = (out["add_drop"] * 100.0).map(lambda v: f"{float(v):.1f}")
            out["tp"] = (out["tp"] * 100.0).map(lambda v: f"{float(v):.1f}")
            out["trapped_time_ratio"] = (out["trapped_time_ratio"] * 100.0).map(lambda v: f"{float(v):.2f}")

            # - multiplier: one decimal (e.g., 2 -> "2.0")
            out["multiplier"] = out["multiplier"].map(lambda v: f"{float(v):.1f}")

            # - max_orders, trades: integers
            out["max_orders"] = out["max_orders"].astype(int).astype(str)
            out["trades"] = out["trades"].astype(int).astype(str)

            # - min_buy_ratio: 2 decimals
            out["min_buy_ratio"] = out["min_buy_ratio"].map(lambda v: f"{float(v):.2f}")

            # - final_equity, max_dd_overall: 2 decimals
            out["final_equity"] = out["final_equity"].map(lambda v: f"{float(v):.2f}")
            out["max_dd_overall"] = out["max_dd_overall"].map(lambda v: f"{float(v):.2f}")

            # Write CSV with exact ordering & formatted strings
            out = out[cols].copy()
            out.to_csv(fpath, index=False, encoding="utf-8-sig")
            messagebox.showinfo("完成", f"已儲存：{fpath}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("儲存失敗", f"{e}")


    def clear_scan_results(self):
        """Clear all scan results from the UI."""
        # Clear Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Clear Metrics Text
        if self.scan_metrics_text:
            self.scan_metrics_text.delete("1.0", tk.END)
        
        # Clear Plot
        if self.figure_scan:
            self.figure_scan.clf()
            self.canvas_scan.draw()
            
        # Reset internal data
        self.scan_df = None
        self.status_var.set("Scan results cleared.")

    def clear_single_results(self):
        """Clear single backtest results from the UI."""
        # Clear Metrics Text
        if self.metrics_text:
            self.metrics_text.delete("1.0", tk.END)
        
        # Clear Plot
        if self.figure_single:
            self.figure_single.clf()
            self.canvas_single.draw()
            
        self.status_var.set("Single backtest results cleared.")



    def plot_selected_from_table_thread(self):
        if self.scan_df is None or self.scan_df.empty:
            messagebox.showinfo("提示", "請先Run Scan，並在表格選擇一列。"); return
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "請在表格中選擇一列。"); return
        idx = self.tree.index(sel[0])
        if idx >= len(self.scan_df):
            messagebox.showerror("錯誤", "選擇索引超出範圍。"); return
        params = self.scan_df.iloc[idx]
        self.status_var.set("載入詳細回測…")
        bars = 0
        scan_days = safe_float(self.e_scan_days.get().strip(), 0.0)
        interval_str = self.e_interval.get().strip()
        if scan_days > 0 and interval_str:
            step_ms = martin._interval_ms(interval_str)
            bars = int(scan_days * 86400 * 1000 / step_ms)

        _thr.Thread(target=self._compute_then_plot,
                    kwargs=dict(
                        symbol=self.e_symbol.get().strip(), interval=interval_str,
                        bars=bars, start=self.e_start.get().strip(), end=self.e_end.get().strip(),
                        refresh_policy=(self.e_refresh.get().strip() or "auto"),
                        fee_rate=safe_float(self.e_fee.get().strip(), 0.0), capital=safe_float(self.e_capital.get().strip(), 1000.0),
                        add_drop=float(params["add_drop"]), multiplier=float(params["multiplier"]), max_orders=int(params["max_orders"]), tp=float(params["tp"]),
                        figure=self.figure_scan, canvas=self.canvas_scan,
                        metrics_widget=self.scan_metrics_text
                    ), daemon=True).start()

    # =================== Single Backtest（Thread → Main）===================
    def run_single_thread(self):
        self.status_var.set("Single Backtest中…")
        _thr.Thread(target=self._run_single_compute_then_update, daemon=True).start()

    def _run_single_compute_then_update(self):
        try:
            symbol   = self.s_symbol.get().strip()
            interval = self.s_interval.get().strip()
            scan_days = safe_float(self.s_scan_days.get().strip(), 0.0)
            bars = 0
            if scan_days > 0:
                step_ms = martin._interval_ms(interval)
                bars = int(scan_days * 86400 * 1000 / step_ms)
            start    = self.s_start.get().strip()
            end      = self.s_end.get().strip()
            refresh  = (self.s_refresh.get().strip() or "auto")
            fee_rate = safe_float(self.s_fee.get().strip(), 0.0)
            capital  = safe_float(self.s_capital.get().strip(), 1000.0)

            add_drop   = safe_float(self.s_add_drop.get().strip(), None)
            tp         = safe_float(self.s_tp.get().strip(), None)
            multiplier = safe_float(self.s_multiplier.get().strip(), None)
            max_orders = safe_int(self.s_max_orders.get().strip(), None)

            if None in (add_drop, tp, multiplier, max_orders):
                raise ValueError("請完整填入策略參數（add_drop, tp, multiplier, max_orders）")
            if not symbol or not interval:
                raise ValueError("請填入 symbol 與 interval")
            if (not bars) and (not start or not end):
                raise ValueError("Scan Days 與 Start/End 需擇一填寫")

            df, res, perf = self._compute_backtest(symbol, interval, bars, start, end, refresh, fee_rate, capital,
                                                   add_drop, multiplier, max_orders, tp)
            self.after(0, lambda: self._render_plot_and_metrics(
                df, res, perf, add_drop, multiplier, max_orders, tp,
                self.figure_single, self.canvas_single, self.metrics_text
            ))
            self.after(0, lambda: self.status_var.set("回測完成。"))
            self.after(0, lambda: self._set_sash_when_ready(self.single_paned, self.INIT_SPLIT))
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = str(e)
            self.after(0, lambda: (self.status_var.set("回測失敗。"),
                                   messagebox.showerror("回測錯誤", f"{err_msg}\n\n{tb}")))

    # ---------- 計算核心（無任何 UI 操作） ----------
    def _compute_backtest(self, symbol, interval, bars, start, end, refresh_policy,
                          fee_rate, capital, add_drop, multiplier, max_orders, tp):
        df = self._fetch_klines_if_needed(symbol, interval, bars, start, end, refresh_policy)
        prices_np = df["close"].to_numpy(dtype=np.float64)
        if prices_np.size < 2: raise ValueError("K 線資料不足（<2 根）。")

        res = martin.martingale_backtest(
            prices_np, add_drop=float(add_drop), multiplier=float(multiplier),
            max_orders=int(max_orders), tp=float(tp), capital=float(capital),
            return_curve=True, times=df["time"].tolist(), fee_rate=float(fee_rate),
        )
        # === Buy & Hold baseline（用同樣資金）===
        first_price = float(prices_np[0])
        bh_qty = float(capital) / first_price
        bh_curve = bh_qty * prices_np

        perf = martin.compute_performance_metrics(
            res["equity_curve"], res["time_index"], res["trades_log"],
            capital=float(capital), bh_curve=bh_curve
        )

        # 把 bh_curve 與 capital 附帶回去，render 時直接使用
        res["bh_curve"] = bh_curve
        res["capital"] = float(capital)
        return df, res, perf

    # For scan-table "plot detail" use the same compute + main-thread render
    def _compute_then_plot(self, *, symbol, interval, bars, start, end, refresh_policy,
                           fee_rate, capital, add_drop, multiplier, max_orders, tp,
                           figure, canvas, metrics_widget):
        try:
            df, res, perf = self._compute_backtest(symbol, interval, bars, start, end, refresh_policy,
                                                   fee_rate, capital, add_drop, multiplier, max_orders, tp)
            self.after(0, lambda: (
                self._render_plot_and_metrics(df, res, perf, add_drop, multiplier, max_orders, tp,
                                              figure, canvas, metrics_widget),
                self.scan_subnb.select(self.scan_tab_detail),
                self._set_sash_when_ready(self.scan_paned, self.INIT_SPLIT),
                self.status_var.set("詳細回測完成。")
            ))
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = str(e)
            self.after(0, lambda: (self.status_var.set("詳細回測失敗。"),
                                   messagebox.showerror("繪圖錯誤", f"{err_msg}\n\n{tb}")))
    
        # ---------- 主執行緒：繪圖 + 寫績效（含 Trapped 比例） ----------
    def _render_plot_and_metrics(self, df, res, perf, add_drop, multiplier, max_orders, tp,
                                     figure, canvas, metrics_widget):
            equity_curve = res["equity_curve"]; time_index = res["time_index"]
            trapped_intervals = res.get("trapped_intervals", [])
            bh_curve = res.get("bh_curve", None)
    
            figure.clf(); ax = figure.add_subplot(111)
            ax.plot(time_index, equity_curve, label="Strategy")
            if bh_curve is not None:
                ax.plot(df["time"], bh_curve, label="Buy & Hold")
            for s_i, e_i in trapped_intervals:
                if s_i is None or e_i is None: continue
                ax.axvspan(s_i, e_i, alpha=0.1, color="red")
            sym = df.attrs.get("symbol", "UNKNOWN"); market_code = df.attrs.get("market", "spot")
            market_label = "Spot" if str(market_code).startswith("spot") else ("USDT Perp" if "usdt_perp" in str(market_code) else str(market_code))
            interval_str = df.attrs.get("interval", "N/A")
            ax.set_title(
                f'{sym} | {market_label} | exch={df.attrs.get("exchange","?")} | interval={interval_str}\n'
                f'[{df["time"].iloc[0].date()} → {df["time"].iloc[-1].date()}] '
                f'add_drop={add_drop:.3f}, tp={tp:.3f}, mul={multiplier:.1f}, max_orders={int(max_orders)}'
            )
            ax.set_xlabel("Time (Taipei)"); ax.set_ylabel("Equity (USDT)")
            ax.legend(); figure.tight_layout(); canvas.draw()
    
            # ====== 計算 Trapped Time Ratio（GUI端計算即可） ======
            trap_ratio = None
            total_secs = None
            trapped_secs = 0.0
            try:
                if time_index is not None and len(time_index) >= 2:
                    start_t = time_index[0]
                    end_t   = time_index[-1]
                    total_secs = (end_t - start_t).total_seconds()
                    if total_secs > 0 and trapped_intervals:
                        for s_i, e_i in trapped_intervals:
                            if s_i is None or e_i is None:
                                continue
                            s = max(s_i, start_t)
                            e = min(e_i, end_t)
                            if e > s:
                                trapped_secs += (e - s).total_seconds()
                        trap_ratio = trapped_secs / total_secs
                    else:
                        trap_ratio = 0.0
            except Exception:
                trap_ratio = None
    
            # 績效文字
            if metrics_widget is not None and isinstance(perf, dict):
                t = []
                t.append("=== Performance (Strategy) ===")
                t.append(f"Total Return: {human_pct(perf.get('total_return'))} | CAGR: {human_pct(perf.get('cagr'))} | Ann Vol: {human_pct(perf.get('ann_vol'))}")
                sharpe = perf.get('sharpe'); sortino = perf.get('sortino'); calmar = perf.get('calmar')
                t.append(f"Sharpe: {('%.2f' % sharpe) if sharpe is not None else 'NaN'} | "
                         f"Sortino: {('%.2f' % sortino) if sortino is not None else 'NaN'} | "
                         f"Calmar: {('%.2f' % calmar) if calmar is not None else 'NaN'}")
                t.append(f"Max DD: {perf.get('max_dd_pct', float('nan')):.2f}% | DD Duration (days): {perf.get('max_dd_days','NaN')} | Recovery: {perf.get('recovery_days','NaN')}")
                t.append(f"Underwater (avg/max): {perf.get('avg_underwater_days', float('nan')):.1f}/{perf.get('max_underwater_days', float('nan')):.1f} days")
                pf = perf.get('profit_factor'); pf_str = "∞" if (isinstance(pf, (int, float)) and np.isinf(pf)) else ("NaN" if pf is None or (isinstance(pf, float) and np.isnan(pf)) else f"{pf:.2f}")
                t.append(f"Win Rate: {human_pct(perf.get('win_rate'))} | Profit Factor: {pf_str}")
                # ✨ 新增：Trapped 比例（含天數）
                if trap_ratio is not None and total_secs is not None:
                    trapped_days = trapped_secs / 86400.0
                    total_days = total_secs / 86400.0
                    t.append(f"Trapped Time Ratio: {human_pct(trap_ratio, 2)} | Trapped/Total: {trapped_days:.1f}/{total_days:.1f} days")
                t.append(f"Avg Win: {perf.get('avg_win', float('nan')):.2f} | Avg Loss: {perf.get('avg_loss', float('nan')):.2f}")
                t.append(f"Max Consec Wins: {perf.get('max_consec_wins','NaN')} | Max Consec Losses: {perf.get('max_consec_losses','NaN')}")
                t.append(f"Avg Trade Return: {human_pct(perf.get('avg_trade_return'))} | Median Trade Return: {human_pct(perf.get('median_trade_return'))}")
                if "bh_total_return" in perf:
                    t.append("\n=== Buy & Hold (Benchmark) ===")
                    t.append(f"Total: {human_pct(perf.get('bh_total_return'))} | CAGR: {human_pct(perf.get('bh_cagr'))} | Ann Vol: {human_pct(perf.get('bh_ann_vol'))}")
                    t.append(f"Sharpe: {('%.2f' % perf.get('bh_sharpe')) if perf.get('bh_sharpe') is not None else 'NaN'} | Max DD: {perf.get('bh_max_dd_pct', float('nan')):.2f}%")
                metrics_widget.delete("1.0", tk.END); metrics_widget.insert(tk.END, "\n".join(t))
    
    
if __name__ == "__main__":
    app = MartinGUI()
    app.mainloop()
