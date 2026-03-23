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
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import ccxt
import requests

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import Qt, QDate, QThreadPool, QRunnable, QObject, Signal, Slot, QTimer, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLineEdit, QComboBox, QPushButton, QLabel, QSplitter, QTableView, QDateEdit,
    QMessageBox, QGroupBox, QStatusBar, QProgressBar,
    QAbstractItemView, QHeaderView, QCheckBox, QSizePolicy,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 嘗試匯入 martin.py 以重用 get_klines
try:
    import martin
    _import_error = None
except Exception as e:
    martin = None
    _import_error = e

DEFAULT_QUOTE_ASSET = "USDT"
CHART_CACHE_LIMIT = 12
COINGECKO_CACHE_TTL = 600
TICKER_CACHE_TTL = 30


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


class ScanTableModel(QAbstractTableModel):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self._columns = list(columns)
        self._df = pd.DataFrame(columns=self._columns)

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        if df is None or df.empty:
            self._df = pd.DataFrame(columns=self._columns)
        else:
            self._df = df.reset_index(drop=True).copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._df):
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return "" if pd.isna(value) else str(value)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self._columns):
            return self._columns[section]
        if orientation == Qt.Vertical:
            return str(section + 1)
        return None


class VolatilityScannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("volatility_scanner_gui")
        self.resize(900, 820)

        self.thread_pool = QThreadPool.globalInstance()
        self.stop_event = threading.Event()
        self._cache_lock = threading.Lock()

        self.scan_results = None
        self._scan_context = None
        self._chart_cache = OrderedDict()
        self._chart_request_token = 0
        self._exchange_clients = {}
        self._ticker_cache = {}
        self._mc_rank_cache = {"ts": 0.0, "limit": 0, "mapping": {}}
        self.cols = [
            "Symbol", "Price", "Avg_Vol(M)", "RV(Ann)%", "ATR(Month)%",
            "Max_DD(%)", "Change(%)", "MC Rank",
        ]

        self._build_ui()
        self._apply_style()
        QTimer.singleShot(0, self._autosize_table_columns)
        QTimer.singleShot(0, self._clear_startup_focus)

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
            QGroupBox { font-weight: 600; border: 1px solid #7c7c7c; border-radius: 10px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #ffffff; }
            QLabel { color: #e8e8e8; }
            QLabel#fieldLabel { color: #cfd2d4; font-size: 10pt; font-weight: 600; }
            QLabel#hintLabel { color: #aeb3b8; font-size: 9pt; }
            QLineEdit, QComboBox {
                background: #ffffff; color: #1d1d1d; border: 1px solid #b9b9b9; border-radius: 6px; padding: 3px 6px;
            }
            QComboBox QAbstractItemView { background: #ffffff; color: #1d1d1d; }
            QTableView { background: #ffffff; color: #1d1d1d; border: 1px solid #7c7c7c; border-radius: 8px; gridline-color: #d0d0d0; }
            QTableView::item:selected { font-weight: 400; }
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
            QProgressBar {
                background: #262626; color: #ffffff; border: 1px solid #6e6e6e;
                border-radius: 8px; text-align: center; min-height: 16px;
            }
            QProgressBar::chunk { background: #4d99e6; border-radius: 7px; }
            QSplitter::handle { background: #7c7c7c; }
            """
        )

    def _build_field(self, label_text: str, widget: QWidget, min_width: int | None = None):
        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        layout.addWidget(label)
        if min_width is not None:
            widget.setFixedWidth(min_width)
        layout.addWidget(widget, alignment=Qt.AlignLeft)
        return wrap

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 10)
        main_layout.setSpacing(12)

        # Top controls
        ctrl_group = QGroupBox("Scanner Settings")
        main_layout.addWidget(ctrl_group)
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setContentsMargins(12, 14, 12, 12)
        ctrl_layout.setSpacing(10)

        self.cb_exchange = QComboBox()
        self.cb_exchange.addItems(["binance", "bybit", "okx", "gateio"])
        self.cb_exchange.setCurrentText("binance")
        self.cb_interval = QComboBox()
        self.cb_interval.addItems(["15m", "1h", "4h", "1d"])
        self.cb_interval.setCurrentText("4h")
        today = QDate.currentDate()
        default_start = today.addYears(-2)
        self.d_start = QDateEdit(default_start)
        self.d_start.setCalendarPopup(True)
        self.d_start.setDisplayFormat("yyyy/MM/dd")
        self.d_end = QDateEdit(today)
        self.d_end.setCalendarPopup(True)
        self.d_end.setDisplayFormat("yyyy/MM/dd")
        self.e_min_vol = QLineEdit("10")
        self.e_min_avg_vol = QLineEdit("40")
        self.e_top_n = QLineEdit("50")
        self.chk_mc_filter = QCheckBox("Filter Top Rank (CoinGecko)")
        self.chk_mc_filter.setChecked(True)
        self.e_max_rank = QLineEdit("100")
        self.btn_scan = QPushButton("Start Scan")
        self.btn_scan.setObjectName("btnPrimary")
        self.btn_scan.setMinimumHeight(38)
        self.btn_scan.setFixedWidth(132)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setFixedWidth(88)

        self.lbl_status = MarqueeLabel("Ready.")
        self.lbl_status.setStyleSheet("color: #9ddc91;")
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_status.setMinimumHeight(22)
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self.progress = QProgressBar()
        self.e_manual = QLineEdit()
        self.e_manual.setPlaceholderText("e.g. SUI, BTC")

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)
        ctrl_layout.addLayout(top_row)

        filters_group = QGroupBox("Universe & Filters")
        filters_layout = QGridLayout(filters_group)
        filters_layout.setContentsMargins(12, 14, 12, 10)
        filters_layout.setHorizontalSpacing(8)
        filters_layout.setVerticalSpacing(8)
        filters_layout.addWidget(self._build_field("Exchange", self.cb_exchange, 110), 0, 0, alignment=Qt.AlignLeft)
        filters_layout.addWidget(self._build_field("Interval", self.cb_interval, 110), 0, 1, alignment=Qt.AlignLeft)
        filters_layout.addWidget(self._build_field("Start", self.d_start, 124), 0, 2, alignment=Qt.AlignLeft)
        filters_layout.addWidget(self._build_field("End", self.d_end, 124), 0, 3, alignment=Qt.AlignLeft)
        filters_layout.addWidget(self._build_field("Pre-filter 24h Vol (M)", self.e_min_vol, 96), 1, 0, alignment=Qt.AlignLeft)
        filters_layout.addWidget(self._build_field("Min Avg Daily Vol (M)", self.e_min_avg_vol, 96), 1, 1, alignment=Qt.AlignLeft)
        filters_layout.addWidget(self._build_field("Scan Top N (by Vol)", self.e_top_n, 96), 1, 2, alignment=Qt.AlignLeft)

        rank_wrap = QWidget()
        rank_layout = QHBoxLayout(rank_wrap)
        rank_layout.setContentsMargins(0, 0, 0, 0)
        rank_layout.setSpacing(6)
        rank_layout.addWidget(self.chk_mc_filter)
        self.e_max_rank.setFixedWidth(72)
        rank_layout.addWidget(self.e_max_rank)
        rank_layout.addStretch(1)
        filters_layout.addWidget(self._build_field("CoinGecko Max Rank", rank_wrap, 176), 1, 3, alignment=Qt.AlignLeft)

        for col in range(4):
            filters_layout.setColumnStretch(col, 1)

        top_row.addWidget(filters_group, 3)

        status_group = QGroupBox("Run Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(12, 14, 12, 12)
        status_layout.setSpacing(8)
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.btn_scan)
        action_row.addWidget(self.btn_stop)
        status_layout.addLayout(action_row)
        status_layout.addWidget(self.lbl_status)
        self.progress.setTextVisible(True)
        status_layout.addWidget(self.progress)
        top_row.addWidget(status_group, 1)

        manual_group = QGroupBox("Manual Include")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setContentsMargins(12, 14, 12, 10)
        manual_layout.setSpacing(4)
        manual_layout.addWidget(self.e_manual)
        manual_hint = QLabel("Add base symbols separated by space or comma. Example: SUI, BTC")
        manual_hint.setObjectName("hintLabel")
        manual_layout.addWidget(manual_hint)
        ctrl_layout.addWidget(manual_group)

        self.btn_scan.clicked.connect(self.start_scan)
        self.btn_stop.clicked.connect(self.stop_scan)

        # Splitter (vertical: table on top, chart at bottom)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)

        # Table
        table_wrap = QWidget()
        table_layout = QVBoxLayout(table_wrap)
        self.table = QTableView()
        self.table_model = ScanTableModel(self.cols, self)
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setWordWrap(False)
        self._autosize_table_columns()
        table_layout.addWidget(self.table)
        splitter.addWidget(table_wrap)

        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        self.table.selectionModel().selectionChanged.connect(self.on_table_select)

        # Chart
        chart_wrap = QWidget()
        chart_layout = QVBoxLayout(chart_wrap)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._chart_line, = self.ax.plot([], [], color="#2d6a7a", linewidth=1.7)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Select a symbol")
        self.canvas = FigureCanvas(self.fig)
        chart_layout.addWidget(self.canvas)
        splitter.addWidget(chart_wrap)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _clear_startup_focus(self):
        if self.focusWidget() is not None:
            self.focusWidget().clearFocus()
        if self.centralWidget() is not None:
            self.centralWidget().setFocus()

    def _set_status(self, msg, ok=True):
        self.lbl_status.setFullText(msg)
        self.lbl_status.setStyleSheet("color: #9ddc91;" if ok else "color: #f3b6b6;")
        self.status_bar.showMessage(msg)

    def _autosize_table_columns(self, max_col_width=280):
        if not hasattr(self, "table") or self.table is None:
            return
        model = self.table.model()
        if model is None:
            return
        header = self.table.horizontalHeader()
        metrics = QFontMetrics(header.font())
        pad_px = 24
        min_col_width = 80
        sample_rows = min(model.rowCount(), 24)
        for c in range(model.columnCount()):
            text = model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or ""
            best = metrics.horizontalAdvance(text) + pad_px
            for r in range(sample_rows):
                cell_text = model.data(model.index(r, c), Qt.DisplayRole) or ""
                best = max(best, metrics.horizontalAdvance(str(cell_text)) + pad_px)
            self.table.setColumnWidth(c, min(max_col_width, max(min_col_width, best)))
        header.setStretchLastSection(True)

    def _chart_cache_key(self, symbol: str):
        if not self._scan_context:
            return None
        return (
            symbol,
            self._scan_context["exch_name"],
            self._scan_context["interval"],
            self._scan_context["start_str"],
            self._scan_context["end_str"],
        )

    def _get_exchange_client(self, exch_name: str):
        with self._cache_lock:
            client = self._exchange_clients.get(exch_name)
            if client is not None:
                return client
        client = getattr(ccxt, exch_name)()
        client.load_markets()
        with self._cache_lock:
            self._exchange_clients[exch_name] = client
        return client

    def _get_cached_tickers(self, exch_name: str):
        now = time.time()
        with self._cache_lock:
            cached = self._ticker_cache.get(exch_name)
            if cached and (now - cached["ts"] <= TICKER_CACHE_TTL):
                return cached["tickers"]
        client = self._get_exchange_client(exch_name)
        tickers = client.fetch_tickers()
        with self._cache_lock:
            self._ticker_cache[exch_name] = {"ts": now, "tickers": tickers}
        return tickers

    def _get_cached_mc_mapping(self, limit: int):
        now = time.time()
        with self._cache_lock:
            cached = self._mc_rank_cache
            if cached["mapping"] and cached["limit"] >= limit and (now - cached["ts"] <= COINGECKO_CACHE_TTL):
                return cached["mapping"]
        mapping = fetch_top_mc_coins(limit=limit)
        if mapping:
            with self._cache_lock:
                self._mc_rank_cache = {"ts": now, "limit": limit, "mapping": mapping}
        return mapping

    def _chart_cache_key_for_context(self, symbol: str, context: dict):
        return (
            symbol,
            context["exch_name"],
            context["interval"],
            context["start_str"],
            context["end_str"],
        )

    def _get_cached_chart_df(self, symbol: str, context: dict | None = None):
        context = context or self._scan_context
        if not context:
            return None
        cache_key = self._chart_cache_key_for_context(symbol, context)
        if cache_key is None:
            return None
        with self._cache_lock:
            if cache_key in self._chart_cache:
                df = self._chart_cache.pop(cache_key)
                self._chart_cache[cache_key] = df
                return df
        return None

    def _store_chart_df(self, symbol: str, context: dict, df: pd.DataFrame):
        cache_key = self._chart_cache_key_for_context(symbol, context)
        with self._cache_lock:
            self._chart_cache[cache_key] = df
            while len(self._chart_cache) > CHART_CACHE_LIMIT:
                self._chart_cache.popitem(last=False)

    def _fetch_chart_df(self, symbol: str, context: dict):
        cached_df = self._get_cached_chart_df(symbol, context)
        if cached_df is not None:
            return cached_df

        quote_asset = context["quote_asset"]
        base = symbol.split('/')[0] if '/' in symbol else symbol.replace(quote_asset, "")
        df = martin.get_klines(
            symbol=base,
            interval=context["interval"],
            start=context["start_str"],
            end=context["end_str"],
            exch_list=[context["exch_name"]],
            refresh_policy="auto",
        )
        if df is None or df.empty:
            return None
        self._store_chart_df(symbol, context, df)
        return df

    def _load_chart_data(self, _signals: ScanSignals, request_token: int, symbol: str, context: dict):
        df = self._fetch_chart_df(symbol, context)
        return {
            "request_token": request_token,
            "symbol": symbol,
            "context": context,
            "df": df,
        }

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

    @Slot(object)
    def _on_chart_data_loaded(self, payload):
        if not payload:
            return
        if payload["request_token"] != self._chart_request_token:
            return
        df = payload.get("df")
        if df is None or df.empty:
            return
        self.plot_chart(df, payload["symbol"])

    @Slot(str)
    def _on_chart_load_error(self, tb):
        self._set_status("Chart load failed.", ok=False)
        QMessageBox.critical(self, "Chart Load Error", tb)

    # --------- scan logic (background) ---------
    def _run_scan_logic(self, signals: ScanSignals):
        exch_name = self.cb_exchange.currentText()
        quote_asset = DEFAULT_QUOTE_ASSET
        interval = self.cb_interval.currentText()
        start_qdate = self.d_start.date()
        end_qdate = self.d_end.date()
        if start_qdate > end_qdate:
            raise ValueError("Start date cannot be later than end date.")
        requested_days = max(1, start_qdate.daysTo(end_qdate) + 1)
        start_str = start_qdate.toString("yyyy-MM-dd")
        end_str = end_qdate.toString("yyyy-MM-dd")
        min_vol_pre = float(self.e_min_vol.text()) * 1_000_000
        min_avg_vol = float(self.e_min_avg_vol.text()) * 1_000_000
        top_n = int(self.e_top_n.text())

        use_mc_filter = self.chk_mc_filter.isChecked()
        max_rank = int(self.e_max_rank.text())

        mc_mapping = {}
        if use_mc_filter:
            signals.log.emit("Fetching Market Cap Rank from CoinGecko...")
            mc_mapping = self._get_cached_mc_mapping(limit=max(max_rank, 250))
            if not mc_mapping:
                signals.log.emit("Warning: CoinGecko fetch failed. MC filter ignored.")

        signals.log.emit(f"Fetching tickers from {exch_name}...")

        tickers = self._get_cached_tickers(exch_name)

        manual_input = self.e_manual.text().strip()
        manual_bases = set()
        if manual_input:
            parts = manual_input.replace(",", " ").split()
            for p in parts:
                p = p.strip().upper()
                if '/' in p:
                    p = p.split('/')[0]
                manual_bases.add(p)

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

        scan_args = {
            "interval": interval,
            "start_str": start_str,
            "end_str": end_str,
            "exch_name": exch_name,
            "quote_asset": quote_asset,
            "requested_days": requested_days,
            "min_avg_vol": min_avg_vol,
        }
        self._scan_context = {
            "interval": interval,
            "start_str": start_str,
            "end_str": end_str,
            "exch_name": exch_name,
            "quote_asset": quote_asset,
        }
        self._chart_cache.clear()

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
        requested_days = args['requested_days']
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

            if time_span_days < (requested_days - 1) and time_span_days < 30 and not is_manual:
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
            }
        except Exception:
            return None

    # --------- table / chart ---------
    def update_table(self):
        self.table.clearSelection()
        if self.scan_results is None or self.scan_results.empty:
            self.table_model.set_dataframe(pd.DataFrame(columns=self.cols))
            return

        if "MC Rank" in self.scan_results.columns:
            work_df = self.scan_results.copy()
            mc_rank = pd.to_numeric(work_df["MC Rank"], errors="coerce")
            work_df["_mc_sort"] = np.where(mc_rank > 0, mc_rank, np.inf)
            sort_cols = ["_mc_sort"]
            sort_asc = [True]
            if "RV(Ann)%" in work_df.columns:
                sort_cols.append("RV(Ann)%")
                sort_asc.append(False)
            work_df.sort_values(sort_cols, ascending=sort_asc, inplace=True)
            self.scan_results = work_df.drop(columns=["_mc_sort"])
            self._sort_col = "MC Rank"
            self._sort_asc = True
        elif "RV(Ann)%" in self.scan_results.columns:
            self.scan_results.sort_values("RV(Ann)%", ascending=False, inplace=True)
            self._sort_col = "RV(Ann)%"
            self._sort_asc = False

        quote_asset = DEFAULT_QUOTE_ASSET
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
        if df is None or df.empty:
            self.table_model.set_dataframe(pd.DataFrame(columns=self.cols))
            return
        disp = pd.DataFrame({
            "Symbol": df["Symbol"],
            "Price": df["Price"].map(lambda v: f"{float(v):.8f}" if float(v) < 0.01 else f"{float(v):.4f}"),
            "Avg_Vol(M)": df["Avg_Vol(M)"].map(lambda v: f"{float(v):.2f}"),
            "RV(Ann)%": df["RV(Ann)%"].map(lambda v: f"{float(v):.2f}"),
            "ATR(Month)%": df["ATR(Month)%"].map(lambda v: f"{float(v):.2f}"),
            "Max_DD(%)": df["Max_DD(%)"].map(lambda v: f"{float(v):.2f}"),
            "Change(%)": df["Change(%)"].map(lambda v: f"{float(v):.2f}"),
            "MC Rank": df["MC Rank"].map(lambda v: f"{int(v)}" if pd.notna(v) and float(v) > 0 else "-"),
        })
        self.table_model.set_dataframe(disp)
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

    @Slot(object, object)
    def on_table_select(self, *_args):
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return
        sel = selection_model.selectedRows()
        if not sel:
            return
        idx = sel[0].row()
        if self.scan_results is None or idx >= len(self.scan_results):
            return
        record = self.scan_results.iloc[idx]
        sym = record["Symbol"]
        if not self._scan_context:
            return
        context = dict(self._scan_context)
        self._chart_request_token += 1
        request_token = self._chart_request_token
        df = self._get_cached_chart_df(sym, context)
        if df is not None:
            self.plot_chart(df, sym)
            return
        self._set_status(f"Loading chart: {sym}")
        worker = ScanWorker(self._load_chart_data, request_token, sym, context)
        worker.signals.finished.connect(self._on_chart_data_loaded)
        worker.signals.error.connect(self._on_chart_load_error)
        self.thread_pool.start(worker)

    def plot_chart(self, df, title):
        self._chart_line.set_data(df['time'], df['close'])
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(f"{title} - {self.cb_interval.currentText()}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

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
