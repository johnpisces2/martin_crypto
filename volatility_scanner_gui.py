# -*- coding: utf-8 -*-
"""
Volatility Scanner GUI for Martin Strategy (PySide6)
用來掃描高波動幣種，輔助馬丁策略選幣。
"""

import traceback
import threading
import concurrent.futures
import time
import re
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ccxt
import requests

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThreadPool, QRunnable, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QLabel, QSplitter, QTableWidget,
    QTableWidgetItem, QMessageBox, QGroupBox, QStatusBar, QProgressBar,
    QAbstractItemView, QHeaderView, QCheckBox,
)

# 嘗試匯入 martin.py 以重用 get_klines
try:
    import martin
    _import_error = None
except Exception as e:
    martin = None
    _import_error = e


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
    req = {"high", "low", "close"}
    if not req.issubset(df.columns):
        return float("nan")
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    if c.size < n or c[-1] <= 0:
        return float("nan")
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
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
        "per_page": min(limit, 250),
        "page": 1,
        "sparkline": "false",
    }

    mapping = {}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                sym = item['symbol'].lower()
                if sym not in mapping:
                    mapping[sym] = item['market_cap_rank']

        if limit > 250:
            params["page"] = 2
            params["per_page"] = limit - 250
            time.sleep(1.0)
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data:
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
    stable_set = {
        "USDT", "USDC", "USDD", "TUSD", "BUSD", "DAI", "FRAX", "FDUSD", "USDP", "GUSD",
        "PYUSD", "USDS", "USDE", "USD1", "USDI", "USDJ", "USDK", "USDX", "USTC", "EUR",
        "EURT", "EURS", "GBP", "GBPT", "USDR", "USDN", "USDB",
    }
    if b in stable_set:
        return True
    if re.match(r"^USD[A-Z0-9]{0,4}$", b):
        return True
    if b.endswith("USD"):
        return True
    return False


class ScanSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)
    progress = Signal(int, int)
    warning = Signal(str)
    stopped = Signal(str)


class ScanWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = ScanSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(self.signals, *self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class MarqueeLabel(QLabel):
    def __init__(self, text="", parent=None, interval_ms=120):
        super().__init__(text, parent)
        self._full_text = text
        self._offset = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._interval_ms = interval_ms

    def setFullText(self, text: str):
        self._full_text = text or ""
        self._offset = 0
        self._refresh()

    def _needs_marquee(self) -> bool:
        fm = self.fontMetrics()
        return fm.horizontalAdvance(self._full_text) > self.contentsRect().width()

    def _refresh(self):
        if not self._full_text:
            self.setText("")
            self._timer.stop()
            return
        if self._needs_marquee():
            if not self._timer.isActive():
                self._timer.start(self._interval_ms)
        else:
            self._timer.stop()
            self.setText(self._full_text)

    def _tick(self):
        if not self._needs_marquee():
            self._refresh()
            return
        s = self._full_text + "   "
        if self._offset >= len(s):
            self._offset = 0
        view = s[self._offset:] + s[:self._offset]
        self.setText(view)
        self._offset += 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class VolatilityScannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("volatility_scanner_gui")
        self.resize(900, 820)

        self.thread_pool = QThreadPool.globalInstance()
        self.stop_event = threading.Event()

        self.scan_results = None
        self.cols = [
            "Symbol", "Price", "Avg_Vol(M)", "RV(Ann)%", "ATR(Month)%",
            "Max_DD(%)", "Change(%)", "MC Rank",
        ]

        self._build_ui()
        self._apply_style()
        QTimer.singleShot(0, self._autosize_table_columns)

    def _apply_style(self):
        app = QApplication.instance()
        if app:
            app.setStyle("Fusion")

        base_font = QFont("Avenir Next", 11)
        self.setFont(base_font)

        self.setStyleSheet(
            """
            QMainWindow { background: #2f2f2f; }
            QWidget { color: #f2f2f2; }
            QGroupBox { font-weight: 600; border: 1px solid #7c7c7c; border-radius: 10px; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #ffffff; }
            QLabel { color: #e8e8e8; }
            QLineEdit, QComboBox {
                background: #ffffff; color: #1d1d1d; border: 1px solid #b9b9b9; border-radius: 6px; padding: 4px 6px;
            }
            QComboBox QAbstractItemView { background: #ffffff; color: #1d1d1d; }
            QTableWidget { background: #ffffff; color: #1d1d1d; border: 1px solid #7c7c7c; border-radius: 8px; gridline-color: #d0d0d0; }
            QTableWidget::item:selected { font-weight: 400; }
            QHeaderView::section { background: #e9e9e9; color: #1d1d1d; padding: 6px; border: 0px; }
            QHeaderView::section:selected { font-weight: 400; }
            QHeaderView::section:checked { font-weight: 400; }
            QTabWidget::pane { border: 1px solid #6e6e6e; border-radius: 8px; }
            QPushButton { background: #59626a; color: #ffffff; border: 0px; border-radius: 8px; padding: 6px 12px; }
            QPushButton:hover { background: #4f575e; }
            QPushButton:pressed { background: #454c52; }
            QPushButton#btnPrimary { background: #2d6a7a; font-weight: 700; }
            QPushButton#btnPrimary:hover { background: #295f6d; }
            QPushButton#btnPrimary:pressed { background: #23535f; }
            QPushButton#btnDanger { background: #8a2d3a; }
            QPushButton#btnDanger:hover { background: #7a2833; }
            QPushButton#btnDanger:pressed { background: #6b232c; }
            QSplitter::handle { background: #7c7c7c; }
            """
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Top controls
        ctrl_group = QGroupBox("Scanner Settings")
        main_layout.addWidget(ctrl_group)
        ctrl_layout = QVBoxLayout(ctrl_group)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        row3 = QHBoxLayout()
        row4 = QHBoxLayout()

        ctrl_layout.addLayout(row1)
        ctrl_layout.addLayout(row2)
        ctrl_layout.addLayout(row3)
        ctrl_layout.addLayout(row4)

        # Row 1
        row1.addWidget(QLabel("Exchange:"))
        self.cb_exchange = QComboBox()
        self.cb_exchange.addItems(["binance", "bybit", "okx", "gateio"])
        self.cb_exchange.setCurrentText("binance")
        self.cb_exchange.setMinimumWidth(100)
        row1.addWidget(self.cb_exchange)

        row1.addWidget(QLabel("Quote Asset:"))
        self.e_quote = QLineEdit("USDT")
        self.e_quote.setMaximumWidth(80)
        row1.addWidget(self.e_quote)

        row1.addWidget(QLabel("Interval:"))
        self.cb_interval = QComboBox()
        self.cb_interval.addItems(["15m", "1h", "4h", "1d"])
        self.cb_interval.setCurrentText("4h")
        self.cb_interval.setMaximumWidth(80)
        row1.addWidget(self.cb_interval)

        row1.addWidget(QLabel("Scan Days:"))
        self.e_days = QLineEdit("730")
        self.e_days.setMaximumWidth(80)
        row1.addWidget(self.e_days)

        row1.addStretch(1)

        # Row 2
        row2.addWidget(QLabel("Pre-filter 24h Vol (M):"))
        self.e_min_vol = QLineEdit("10")
        self.e_min_vol.setMaximumWidth(90)
        row2.addWidget(self.e_min_vol)

        row2.addWidget(QLabel("Min Avg Daily Vol (M):"))
        self.e_min_avg_vol = QLineEdit("40")
        self.e_min_avg_vol.setMaximumWidth(90)
        row2.addWidget(self.e_min_avg_vol)

        row2.addWidget(QLabel("Scan Top N (by Vol):"))
        self.e_top_n = QLineEdit("50")
        self.e_top_n.setMaximumWidth(80)
        row2.addWidget(self.e_top_n)

        row2.addStretch(1)

        # Row 3
        self.chk_mc_filter = QCheckBox("Filter Top Rank (CoinGecko)")
        self.chk_mc_filter.setChecked(True)
        row3.addWidget(self.chk_mc_filter)

        row3.addWidget(QLabel("Max Rank:"))
        self.e_max_rank = QLineEdit("100")
        self.e_max_rank.setMaximumWidth(80)
        row3.addWidget(self.e_max_rank)

        row3.addStretch(1)

        self.btn_scan = QPushButton("Start Scan")
        self.btn_scan.setObjectName("btnPrimary")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setEnabled(False)

        row3.addWidget(self.btn_scan)
        row3.addWidget(self.btn_stop)

        self.lbl_status = MarqueeLabel("Ready.")
        self.lbl_status.setStyleSheet("color: #9ddc91;")
        self.lbl_status.setFixedWidth(170)
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row3.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        row3.addWidget(self.progress)
        row3.addStretch(0)

        # Row 4
        row4.addWidget(QLabel("Manual Add (Base):"))
        self.e_manual = QLineEdit()
        self.e_manual.setMinimumWidth(320)
        row4.addWidget(self.e_manual)
        row4.addWidget(QLabel("(e.g. SUI, BTC. Separated by space/comma)"))
        row4.addStretch(1)

        self.btn_scan.clicked.connect(self.start_scan)
        self.btn_stop.clicked.connect(self.stop_scan)

        # Splitter (vertical: table on top, chart at bottom)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)

        # Table
        table_wrap = QWidget()
        table_layout = QVBoxLayout(table_wrap)
        self.table = QTableWidget(0, len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setWordWrap(False)
        self._autosize_table_columns()
        table_layout.addWidget(self.table)
        splitter.addWidget(table_wrap)

        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.table.itemSelectionChanged.connect(self.on_table_select)

        # Chart
        chart_wrap = QWidget()
        chart_layout = QVBoxLayout(chart_wrap)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        chart_layout.addWidget(self.canvas)
        splitter.addWidget(chart_wrap)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _set_status(self, msg, ok=True):
        self.lbl_status.setFullText(msg)
        self.lbl_status.setStyleSheet("color: #9ddc91;" if ok else "color: #f3b6b6;")
        self.status_bar.showMessage(msg)

    def _autosize_table_columns(self, max_col_width=280):
        if not hasattr(self, "table") or self.table is None:
            return
        header = self.table.horizontalHeader()
        metrics = QFontMetrics(header.font())
        pad_px = 24
        min_col_width = 80
        for c in range(self.table.columnCount()):
            h_item = self.table.horizontalHeaderItem(c)
            text = h_item.text() if h_item is not None else ""
            best = metrics.horizontalAdvance(text) + pad_px
            for r in range(self.table.rowCount()):
                item = self.table.item(r, c)
                if item is None:
                    continue
                best = max(best, metrics.horizontalAdvance(item.text()) + pad_px)
            self.table.setColumnWidth(c, min(max_col_width, max(min_col_width, best)))
        header.setStretchLastSection(True)

    def start_scan(self):
        self.stop_event.clear()
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self._set_status("Starting scan...")

        worker = ScanWorker(self._run_scan_logic)
        worker.signals.log.connect(lambda m: self._set_status(m, ok=True))
        worker.signals.progress.connect(self._on_progress)
        worker.signals.warning.connect(self._show_warning)
        worker.signals.stopped.connect(lambda m: self._set_status(m, ok=False))
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(self._on_scan_finished)

        self.thread_pool.start(worker)

    def stop_scan(self):
        self.stop_event.set()
        self.btn_stop.setEnabled(False)
        self._set_status("Stopping scan...", ok=False)

    @Slot(int, int)
    def _on_progress(self, value, total):
        if total > 0:
            self.progress.setMaximum(total)
        self.progress.setValue(value)

    @Slot(str)
    def _show_warning(self, msg):
        QMessageBox.warning(self, "Warning", msg)

    @Slot(str)
    def _show_error(self, tb):
        self._set_status("Scan failed.", ok=False)
        QMessageBox.critical(self, "Error", tb)
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)

    @Slot(object)
    def _on_scan_finished(self, df):
        self.scan_results = df
        self.update_table()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.stop_event.is_set():
            self._set_status("Scan stopped by user.", ok=False)
        else:
            self._set_status("Scan completed.")

    # --------- scan logic (background) ---------
    def _run_scan_logic(self, signals: ScanSignals):
        exch_name = self.cb_exchange.currentText()
        quote_asset = self.e_quote.text().upper().strip()
        interval = self.cb_interval.currentText()
        days = int(self.e_days.text())
        min_vol_pre = float(self.e_min_vol.text()) * 1_000_000
        min_avg_vol = float(self.e_min_avg_vol.text()) * 1_000_000
        top_n = int(self.e_top_n.text())

        use_mc_filter = self.chk_mc_filter.isChecked()
        max_rank = int(self.e_max_rank.text())

        mc_mapping = {}
        if use_mc_filter:
            signals.log.emit("Fetching Market Cap Rank from CoinGecko...")
            mc_mapping = fetch_top_mc_coins(limit=max(max_rank, 250))
            if not mc_mapping:
                signals.log.emit("Warning: CoinGecko fetch failed. MC filter ignored.")

        signals.log.emit(f"Fetching tickers from {exch_name}...")

        ex_cls = getattr(ccxt, exch_name)()
        ex_cls.load_markets()
        tickers = ex_cls.fetch_tickers()

        manual_input = self.e_manual.text().strip()
        manual_bases = set()
        if manual_input:
            parts = manual_input.replace(",", " ").split()
            for p in parts:
                p = p.strip().upper()
                if '/' in p:
                    p = p.split('/')[0]
                manual_bases.add(p)

        manual_pairs_needed = {f"{b}/{quote_asset}" for b in manual_bases}
        manual_pairs_found = set()

        candidates = []
        for symbol, ticker in tickers.items():
            if not ticker:
                continue
            if f"/{quote_asset}" not in symbol and not symbol.endswith(quote_asset):
                continue

            base = symbol.split('/')[0] if '/' in symbol else symbol.replace(quote_asset, "")
            is_manual = (base in manual_bases)
            if is_manual:
                if symbol == f"{base}/{quote_asset}":
                    manual_pairs_found.add(symbol)

            if not is_manual:
                if is_stablecoin_base(base):
                    continue

                rank = -1
                if mc_mapping:
                    base_lower = base.lower()
                    if base_lower in mc_mapping:
                        rank = mc_mapping[base_lower]
                        if rank > max_rank:
                            continue
                    else:
                        continue
            else:
                rank = -1
                if mc_mapping:
                    base_lower = base.lower()
                    if base_lower in mc_mapping:
                        rank = mc_mapping[base_lower]

            vol = ticker.get('quoteVolume') or 0
            if not is_manual and vol < min_vol_pre:
                continue

            candidates.append({
                'symbol': symbol,
                'volume': vol,
                'close': ticker.get('close'),
                'mc_rank': rank,
                'is_manual': is_manual,
            })

        if manual_bases:
            found_bases = {p.split('/')[0] for p in manual_pairs_found}
            missing_bases = manual_bases - found_bases
            if missing_bases:
                msg = (
                    f"Warning: The following manual symbols were not found on {exch_name} "
                    f"(Quote: {quote_asset}):\n" + ", ".join(missing_bases)
                )
                signals.warning.emit(msg)

        manual_candidates = [c for c in candidates if c['symbol'] in manual_pairs_found]
        auto_candidates = [c for c in candidates if c['symbol'] not in manual_pairs_found]
        auto_candidates.sort(key=lambda x: x['volume'], reverse=True)
        auto_candidates = auto_candidates[:top_n]

        candidates = manual_candidates + auto_candidates
        total_cands = len(candidates)
        signals.log.emit(f"Found {total_cands} candidates. Fetching K-lines (Parallel)...")

        results = []

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        scan_args = {
            "interval": interval,
            "start_str": start_str,
            "end_str": end_str,
            "exch_name": exch_name,
            "quote_asset": quote_asset,
            "days": days,
            "min_avg_vol": min_avg_vol,
        }

        completed_count = 0
        max_workers = 8

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cand = {executor.submit(self.process_coin, cand, scan_args): cand for cand in candidates}
            for future in concurrent.futures.as_completed(future_to_cand):
                if self.stop_event.is_set():
                    signals.stopped.emit("Scan stopped by user.")
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
                signals.progress.emit(completed_count, total_cands)
                signals.log.emit(f"Scanning {completed_count}/{total_cands}...")

        if self.stop_event.is_set():
            signals.log.emit("Scan stopped by user.")
        else:
            signals.log.emit("Scan completed.")

        return pd.DataFrame(results)

    def process_coin(self, cand, args):
        if self.stop_event.is_set():
            return None

        sym = cand['symbol']
        interval = args['interval']
        start_str = args['start_str']
        end_str = args['end_str']
        exch_name = args['exch_name']
        quote_asset = args['quote_asset']
        days = args['days']
        min_avg_vol = args['min_avg_vol']
        is_manual = cand.get('is_manual', False)

        try:
            base = sym.split('/')[0] if '/' in sym else sym.replace(quote_asset, "")

            df = martin.get_klines(
                symbol=base,
                interval=interval,
                start=start_str,
                end=end_str,
                exch_list=[exch_name],
                refresh_policy="auto",
            )

            if df is None or df.empty or len(df) < 50:
                return None

            closes = df['close'].astype(float)

            change = (closes.iloc[-1] / closes.iloc[0]) - 1.0

            step_ms = martin._interval_ms(interval)
            bars_per_year = (365.25 * 24 * 3600) / (step_ms / 1000.0)
            rv_annual = realized_vol_annual_from_closes(closes, bars_per_year)

            atr_p = true_atr_pct(df, n=14)
            if not np.isfinite(atr_p):
                atr_p = approx_atr_pct_from_close(closes, n=14)

            bars_per_day = 24 * 3600 * 1000 / step_ms
            atr_daily_est = atr_p * np.sqrt(bars_per_day)
            atr_monthly_est = atr_daily_est * np.sqrt(30)

            bbw_p = bb_width_pct(closes, n=20, k=2.0)

            roll_max = closes.cummax()
            dd = (closes - roll_max) / roll_max
            max_dd = dd.min()

            quote_vols = df['volume'] * df['close']
            total_quote_vol = quote_vols.sum()

            time_span_days = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 86400.0

            if time_span_days < (days - 1) and time_span_days < 30 and not is_manual:
                return None

            if time_span_days < 0.5:
                time_span_days = 0.5

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
                "df": df,
            }
        except Exception:
            return None

    # --------- table / chart ---------
    def update_table(self):
        self.table.setRowCount(0)
        if self.scan_results is None or self.scan_results.empty:
            return

        if "RV(Ann)%" in self.scan_results.columns:
            self.scan_results.sort_values("RV(Ann)%", ascending=False, inplace=True)

        quote_asset = (self.e_quote.text() or "").upper()
        raw_symbols = self.scan_results["Symbol"].tolist()
        norm_symbols = []
        for sym in raw_symbols:
            if "/" in sym:
                norm_symbols.append(sym.split("/")[0])
            elif quote_asset and sym.endswith(quote_asset):
                norm_symbols.append(sym[:-len(quote_asset)])
            else:
                norm_symbols.append(sym)
        print("[Scan Results] Symbols:", ", ".join(norm_symbols))

        self._render_table(self.scan_results)

    def _render_table(self, df: pd.DataFrame):
        self.table.setRowCount(0)
        for _, row in df.iterrows():
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
                f"{int(row['MC Rank'])}" if row['MC Rank'] > 0 else "-",
            )
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        self._autosize_table_columns()

    @Slot(int)
    def on_header_clicked(self, idx: int):
        if self.scan_results is None or self.scan_results.empty:
            return
        col = self.cols[idx]
        ascending = True
        if getattr(self, "_sort_col", None) == col:
            ascending = not getattr(self, "_sort_asc", True)
        self._sort_col = col
        self._sort_asc = ascending

        if col in ["Symbol"]:
            self.scan_results.sort_values(col, ascending=ascending, inplace=True)
        else:
            self.scan_results.sort_values(col, ascending=ascending, inplace=True, key=lambda s: pd.to_numeric(s, errors="coerce"))
        self._render_table(self.scan_results)

    @Slot()
    def on_table_select(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        idx = sel[0].row()
        if self.scan_results is None or idx >= len(self.scan_results):
            return
        sym = self.scan_results.iloc[idx]["Symbol"]
        record = self.scan_results[self.scan_results["Symbol"] == sym]
        if record.empty:
            return
        df = record.iloc[0]["df"]
        self.plot_chart(df, sym)

    def plot_chart(self, df, title):
        self.ax.clear()
        self.ax.plot(df['time'], df['close'], label='Close')
        self.ax.set_title(f"{title} - {self.cb_interval.currentText()}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._autosize_table_columns()


if __name__ == "__main__":
    app = QApplication([])
    if _import_error is not None:
        QMessageBox.critical(None, "Import Error", f"無法匯入 martin.py：\n{_import_error}")
        raise SystemExit(1)
    w = VolatilityScannerGUI()
    w.show()
    app.exec()
