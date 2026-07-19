# -*- coding: utf-8 -*-
"""
Volatility Scanner GUI for Martin Strategy (PySide6)
用來掃描高波動幣種，輔助馬丁策略選幣。
"""

import traceback
import threading
import concurrent.futures
import time
import math
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import ccxt
from market_data import coingecko, pionex, universe

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import Qt, QDate, QThreadPool, QRunnable, QObject, Signal, Slot, QTimer, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLineEdit, QComboBox, QPushButton, QLabel, QSplitter, QTableView, QDateEdit,
    QMessageBox, QGroupBox, QStatusBar, QProgressBar,
    QAbstractItemView, QHeaderView, QCheckBox, QSizePolicy,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
from matplotlib.figure import Figure

# 嘗試匯入 martin.py 以重用 get_klines
try:
    import martin
    _import_error = None
except Exception as e:
    martin = None
    _import_error = e

DEFAULT_QUOTE_ASSET = "USDT"
CHART_CACHE_LIMIT = 96
KLINE_MEMORY_CACHE_TTL = 300
COINGECKO_CACHE_TTL = 600
COINGECKO_RANK_LOOKUP_LIMIT = 1000
TICKER_CACHE_TTL = 30
MIN_HISTORY_COVERAGE_RATIO = 0.80
STOCK_TOKEN_MAX_REQUIRED_HISTORY_DAYS = 90.0


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
    ohlc = df[["high", "low", "close"]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    h = ohlc["high"].to_numpy(dtype=float)
    l = ohlc["low"].to_numpy(dtype=float)
    c = ohlc["close"].to_numpy(dtype=float)
    if c.size < n or c[-1] <= 0:
        return float("nan")
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr_last = tr[-n:].mean()
    return float(atr_last / c[-1])


def approx_atr_pct_from_close(closes: pd.Series, n: int = 14) -> float:
    c = closes.to_numpy(dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < n + 1 or c[-1] <= 0:
        return float("nan")
    tr = np.abs(np.diff(c))
    atr_last = tr[-n:].mean()
    return float(atr_last / c[-1])


def max_drawdown_pct_from_ohlc(df: pd.DataFrame, closes: pd.Series) -> float:
    if {"high", "low"}.issubset(df.columns):
        hl = df[["high", "low"]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        hl = hl[(hl["high"] > 0) & (hl["low"] > 0)]
        if not hl.empty:
            roll_peak = hl["high"].cummax()
            dd = (hl["low"] - roll_peak) / roll_peak
            return float(dd.min())
    roll_max = closes.cummax()
    dd = (closes - roll_max) / roll_max
    return float(dd.min())


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
        except Exception:
            try:
                self.signals.error.emit(traceback.format_exc())
            except RuntimeError:
                pass
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass


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
        self.resize(1280, 820)

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
            "Symbol", "Asset", "Price", "Vol(M)", "Active%", "MaxGap%",
            "RV(A)%", "ATR(M)%", "MaxDD%", "Chg%", "ER", "MaxRed", "MC Rank",
        ]

        self._build_ui()
        self._apply_style()
        QTimer.singleShot(0, self._fit_initial_window_to_screen)
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
            QLineEdit, QComboBox, QDateEdit {
                background: #ffffff; color: #1d1d1d; border: 1px solid #b9b9b9; border-radius: 6px; padding: 3px 6px;
                selection-background-color: #2d6a7a; selection-color: #ffffff;
            }
            QLineEdit:disabled, QComboBox:disabled, QDateEdit:disabled {
                background: #dedede; color: #555555; border-color: #a8a8a8;
            }
            QDateEdit::drop-down { background: #f2f2f2; border-left: 1px solid #c5c5c5; width: 24px; }
            QComboBox QAbstractItemView, QDateEdit QAbstractItemView { background: #ffffff; color: #1d1d1d; }
            QCalendarWidget QWidget { background: #ffffff; color: #1d1d1d; }
            QCalendarWidget QAbstractItemView:enabled {
                background: #ffffff; color: #1d1d1d;
                selection-background-color: #2d6a7a; selection-color: #ffffff;
            }
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

    @staticmethod
    def _apply_date_edit_palette(date_edit: QDateEdit):
        """Keep date text readable under WSL/Linux dark Qt themes."""
        palette = date_edit.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#1d1d1d"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#f2f2f2"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1d1d1d"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#2d6a7a"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#555555")
        )
        date_edit.setPalette(palette)

    def _build_field(self, label_text: str, widget: QWidget, min_width: int | None = None):
        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        layout.addWidget(label)
        if min_width is not None:
            widget.setMinimumWidth(min_width)
        widget_policy = widget.sizePolicy()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, widget_policy.verticalPolicy())
        layout.addWidget(widget)
        return wrap

    def _fit_initial_window_to_screen(self):
        """Choose a readable initial size without exceeding the WSL display."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        min_width = min(1180, max(800, available.width() - 40))
        min_height = min(700, max(560, available.height() - 60))
        target_width = min(1320, max(min_width, int(available.width() * 0.82)))
        target_height = min(960, max(min_height, int(available.height() * 0.90)))
        self.setMinimumSize(min_width, min_height)
        self.resize(target_width, target_height)

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
        self.cb_exchange.addItems(["pionex", "binance"])
        self.cb_exchange.setCurrentText("pionex")
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
        self._apply_date_edit_palette(self.d_start)
        self._apply_date_edit_palette(self.d_end)
        self.e_min_vol = QLineEdit("10")
        self.e_min_avg_vol = QLineEdit("40")
        self.cb_asset_universe = QComboBox()
        self.cb_asset_universe.addItems(["All spot", "Stock/RWA tokens only", "Crypto only"])
        self.e_top_n = QLineEdit("50")
        self.chk_mc_filter = QCheckBox("Enable rank filter")
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

        filters_group = QGroupBox("Universe && Filters")
        filters_layout = QGridLayout(filters_group)
        filters_layout.setContentsMargins(12, 14, 12, 10)
        filters_layout.setHorizontalSpacing(8)
        filters_layout.setVerticalSpacing(8)
        filters_layout.addWidget(self._build_field("Exchange", self.cb_exchange, 110), 0, 0)
        filters_layout.addWidget(self._build_field("Interval", self.cb_interval, 110), 0, 1)
        filters_layout.addWidget(self._build_field("Start", self.d_start, 168), 0, 2)
        filters_layout.addWidget(self._build_field("End", self.d_end, 168), 0, 3)
        filters_layout.addWidget(self._build_field("Pre-filter 24h Vol (M)", self.e_min_vol, 96), 1, 0)
        filters_layout.addWidget(self._build_field("Min Avg Daily Vol (M)", self.e_min_avg_vol, 96), 1, 1)
        filters_layout.addWidget(self._build_field("Scan Top N (Total)", self.e_top_n, 96), 1, 2)

        rank_wrap = QWidget()
        rank_layout = QHBoxLayout(rank_wrap)
        rank_layout.setContentsMargins(0, 0, 0, 0)
        rank_layout.setSpacing(6)
        rank_layout.addWidget(self.chk_mc_filter)
        self.e_max_rank.setFixedWidth(72)
        rank_layout.addWidget(self.e_max_rank)
        rank_layout.addStretch(1)
        filters_layout.addWidget(self._build_field("CoinGecko Max Rank", rank_wrap, 260), 1, 3)
        filters_layout.addWidget(self._build_field("Asset Universe", self.cb_asset_universe, 140), 2, 0)
        stock_hint = QLabel(
            "Pionex 全部現貨不套用 24h 與歷史日均成交量門檻；"
            "Rank 只篩 Crypto；Top N 是全部候選的合計上限。"
        )
        stock_hint.setObjectName("hintLabel")
        stock_hint.setWordWrap(True)
        filters_layout.addWidget(stock_hint, 2, 1, 1, 3)

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
        self.cb_exchange.currentTextChanged.connect(self._on_exchange_changed)
        self._on_exchange_changed(self.cb_exchange.currentText())

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

    def _on_exchange_changed(self, exchange_name):
        """Show that Pionex ignores unreliable exchange volume thresholds."""
        bypass_volume = str(exchange_name or "").strip().lower() == "pionex"
        tooltip = "Pionex 掃描不套用最低成交量門檻。" if bypass_volume else ""
        for field in (self.e_min_vol, self.e_min_avg_vol):
            field.setEnabled(not bypass_volume)
            field.setToolTip(tooltip)

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
        pad_px = 12
        min_col_width = 60
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
        if str(exch_name).lower() == "pionex":
            client = pionex.PionexPublicClient()
        else:
            client_cls = getattr(ccxt, exch_name, None)
            if client_cls is None:
                raise ValueError(f"不支援的交易所：{exch_name}")
            client = client_cls()
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
        mapping = coingecko.fetch_market_cap_ranks(limit=limit)
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
                entry = self._chart_cache.pop(cache_key)
                if time.time() - entry["ts"] <= KLINE_MEMORY_CACHE_TTL:
                    self._chart_cache[cache_key] = entry
                    return entry["df"]
        return None

    def _store_chart_df(self, symbol: str, context: dict, df: pd.DataFrame):
        cache_key = self._chart_cache_key_for_context(symbol, context)
        with self._cache_lock:
            self._chart_cache[cache_key] = {"ts": time.time(), "df": df}
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
            pause=(0.0 if str(context["exch_name"]).lower() == "pionex" else 0.12),
            refresh_policy="auto",
            require_full_coverage=False,
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
        try:
            start_qdate = self.d_start.date()
            end_qdate = self.d_end.date()
            if start_qdate > end_qdate:
                raise ValueError("Start date cannot be later than end date.")
            config = {
                "exch_name": self.cb_exchange.currentText(),
                "interval": self.cb_interval.currentText(),
                "start_str": start_qdate.toString("yyyy-MM-dd"),
                "end_str": end_qdate.toString("yyyy-MM-dd"),
                "requested_days": max(1, start_qdate.daysTo(end_qdate) + 1),
                "min_vol_pre": float(self.e_min_vol.text()) * 1_000_000,
                "min_avg_vol": float(self.e_min_avg_vol.text()) * 1_000_000,
                "asset_universe": self.cb_asset_universe.currentText(),
                "top_n": int(self.e_top_n.text()),
                "use_mc_filter": self.chk_mc_filter.isChecked(),
                "max_rank": int(self.e_max_rank.text()),
                "manual_input": self.e_manual.text().strip(),
            }
            volume_values = (config["min_vol_pre"], config["min_avg_vol"])
            if any(not np.isfinite(v) or v < 0 for v in volume_values):
                raise ValueError("成交量門檻必須為有限非負數")
            if config["top_n"] <= 0 or config["max_rank"] <= 0:
                raise ValueError("Top N 與 Max rank 必須 > 0")
        except Exception:
            self._show_error(traceback.format_exc())
            return
        self.stop_event.clear()
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self._set_status("Starting scan...")

        worker = ScanWorker(self._run_scan_logic, config)
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
        stats = dict(getattr(df, "attrs", {}).get("scan_stats", {}))
        self.scan_results = df
        self.update_table()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.stop_event.is_set():
            self._set_status("Scan stopped by user.", ok=False)
        elif stats:
            self._set_status(
                "Scan completed: "
                f"results {stats.get('results', 0)}/{stats.get('selected', 0)}, "
                f"data-filtered {stats.get('data_filtered', 0)}, "
                f"errors {stats.get('errors', 0)}, "
                f"429 {stats.get('rate_limited', 0)}."
            )
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
        self.plot_chart(df, payload["symbol"], payload.get("context", {}).get("interval"))

    @Slot(str)
    def _on_chart_load_error(self, tb):
        self._set_status("Chart load failed.", ok=False)
        QMessageBox.critical(self, "Chart Load Error", tb)

    # --------- scan logic (background) ---------
    def _run_scan_logic(self, signals: ScanSignals, config):
        exch_name = config["exch_name"]
        quote_asset = DEFAULT_QUOTE_ASSET
        interval = config["interval"]
        requested_days = config["requested_days"]
        start_str = config["start_str"]
        end_str = config["end_str"]
        min_vol_pre = config["min_vol_pre"]
        min_avg_vol = config["min_avg_vol"]
        asset_universe = config["asset_universe"]
        top_n = config["top_n"]

        use_mc_filter = config["use_mc_filter"]
        max_rank = config["max_rank"]

        mc_mapping = {}
        # When filtering, ranks beyond max_rank cannot be selected. Avoid the
        # extra CoinGecko pages and one-second inter-page waits in that case.
        rank_lookup_limit = (
            max_rank if use_mc_filter
            else max(max_rank, COINGECKO_RANK_LOOKUP_LIMIT)
        )
        signals.log.emit("Fetching Market Cap Rank from CoinGecko...")
        mc_mapping = self._get_cached_mc_mapping(limit=rank_lookup_limit)
        if not mc_mapping:
            if use_mc_filter:
                signals.log.emit("Warning: CoinGecko fetch failed. MC filter ignored.")
            else:
                signals.log.emit("Warning: CoinGecko fetch failed. MC Rank will show '-'.")

        signals.log.emit(f"Fetching tickers from {exch_name}...")

        tickers = self._get_cached_tickers(exch_name)

        manual_input = config["manual_input"]
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
        client = self._get_exchange_client(exch_name)
        markets = getattr(client, "markets", {}) or {}
        for symbol, ticker in tickers.items():
            if not ticker:
                continue
            market = markets.get(symbol)
            if (
                not market
                or not market.get("active", True)
                or not market.get("spot")
                or market.get("quote") != quote_asset
            ):
                continue
            base = str(market.get("base") or "").upper()
            is_manual = (base in manual_bases)
            is_stock_token = universe.is_probable_stock_token(base)
            if not is_manual:
                if asset_universe == "Stock/RWA tokens only" and not is_stock_token:
                    continue
                if asset_universe == "Crypto only" and is_stock_token:
                    continue
            if is_manual:
                if symbol == f"{base}/{quote_asset}":
                    manual_pairs_found.add(symbol)

            rank = -1
            if mc_mapping:
                base_lower = base.lower()
                if base_lower in mc_mapping:
                    rank = mc_mapping[base_lower]

            if not is_manual:
                if universe.is_stablecoin_base(base):
                    continue

                if not universe.scanner_rank_filter_allows(
                    rank,
                    max_rank,
                    enabled=use_mc_filter,
                    mapping_available=bool(mc_mapping),
                    is_stock_token=is_stock_token,
                ):
                    continue

            vol = ticker.get('quoteVolume') or 0
            # Pionex public volume is not reliable enough as a universe filter,
            # especially for its smaller market and tokenized equities.
            bypass_volume = universe.bypass_scanner_volume_filters(exch_name)
            if not is_manual and not bypass_volume and vol < min_vol_pre:
                continue

            candidates.append({
                'symbol': symbol,
                'volume': vol,
                'close': ticker.get('close'),
                'mc_rank': rank,
                'is_manual': is_manual,
                'is_stock_token': is_stock_token,
                'asset_type': "Stock/RWA" if is_stock_token else "Crypto",
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
        manual_candidates.sort(key=lambda x: x['symbol'])
        if len(manual_candidates) > top_n:
            signals.warning.emit(
                f"Manual Include 找到 {len(manual_candidates)} 個標的，"
                f"但 Top N 為 {top_n}；只掃描前 {top_n} 個。"
            )
            manual_candidates = manual_candidates[:top_n]
        remaining_slots = max(0, top_n - len(manual_candidates))

        if asset_universe == "All spot":
            auto_stock = [c for c in auto_candidates if c['is_stock_token']]
            auto_crypto = [c for c in auto_candidates if not c['is_stock_token']]
            # Pionex stock-token ticker volume is unreliable, so keep a stable
            # symbol order and reserve their slots before filling with Crypto.
            auto_stock.sort(key=lambda x: x['symbol'])
            auto_crypto.sort(key=lambda x: x['volume'], reverse=True)
            selected_stock = auto_stock[:remaining_slots]
            crypto_slots = max(0, remaining_slots - len(selected_stock))
            auto_candidates = selected_stock + auto_crypto[:crypto_slots]
        elif asset_universe == "Stock/RWA tokens only":
            auto_candidates.sort(key=lambda x: x['symbol'])
            auto_candidates = auto_candidates[:remaining_slots]
        else:
            auto_candidates.sort(key=lambda x: x['volume'], reverse=True)
            auto_candidates = auto_candidates[:remaining_slots]

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
        completed_count = 0
        failed_count = 0
        no_result_count = 0
        rate_limited_count = 0
        # Pionex's K-line route has weight 1 and the adapter enforces a
        # weight-aware IP limiter. Four workers keep HTTP/network latency from
        # leaving permitted request slots idle without exceeding that limiter.
        max_workers = 4 if str(exch_name).lower() == "pionex" else 8
        signals.log.emit(
            f"Fetching K-lines with {max_workers} worker(s) for {exch_name}..."
        )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        stopped = False
        try:
            future_to_cand = {executor.submit(self.process_coin, cand, scan_args): cand for cand in candidates}
            for future in concurrent.futures.as_completed(future_to_cand):
                if self.stop_event.is_set():
                    signals.stopped.emit("Scan stopped by user.")
                    stopped = True
                    for pending in future_to_cand:
                        pending.cancel()
                    break

                cand = future_to_cand[future]
                sym = cand['symbol']
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                    else:
                        no_result_count += 1
                except Exception as e:
                    failed_count += 1
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        rate_limited_count += 1
                    signals.log.emit(f"Skipped {sym}: {e}")

                completed_count += 1
                signals.progress.emit(completed_count, total_cands)
                signals.log.emit(f"Scanning {completed_count}/{total_cands}...")
        finally:
            executor.shutdown(wait=not stopped, cancel_futures=True)

        if self.stop_event.is_set():
            signals.log.emit("Scan stopped by user.")
        else:
            signals.log.emit(
                f"Scan completed: selected={total_cands}, results={len(results)}, "
                f"data-filtered={no_result_count}, errors={failed_count}, "
                f"rate-limited={rate_limited_count}."
            )

        out = pd.DataFrame(results)
        out.attrs["scan_stats"] = {
            "selected": int(total_cands),
            "results": int(len(results)),
            "data_filtered": int(no_result_count),
            "errors": int(failed_count),
            "rate_limited": int(rate_limited_count),
            "workers": int(max_workers),
        }
        return out

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
        is_stock_token = bool(cand.get('is_stock_token', False))
        kline_pause = 0.0 if str(exch_name).lower() == "pionex" else 0.12

        try:
            base = sym.split('/')[0] if '/' in sym else sym.replace(quote_asset, "")
            kline_context = {
                "interval": interval,
                "start_str": start_str,
                "end_str": end_str,
                "exch_name": exch_name,
                "quote_asset": quote_asset,
            }
            df = self._get_cached_chart_df(sym, kline_context)
            if df is None:
                df = martin.get_klines(
                    symbol=base,
                    interval=interval,
                    start=start_str,
                    end=end_str,
                    exch_list=[exch_name],
                    pause=kline_pause,
                    refresh_policy="auto",
                    require_full_coverage=False,
                )
            has_complete_ohlc = (
                df is not None
                and not df.empty
                and {"high", "low", "close"}.issubset(df.columns)
                and not df[["high", "low", "close"]].isna().any().any()
            )
            if df is not None and not df.empty and not has_complete_ohlc:
                df = martin.get_klines(
                    symbol=base,
                    interval=interval,
                    start=start_str,
                    end=end_str,
                    exch_list=[exch_name],
                    pause=kline_pause,
                    refresh_policy="force",
                    require_full_coverage=False,
                )

            if df is not None and not df.empty:
                self._store_chart_df(sym, kline_context, df)

            if df is None or df.empty or len(df) < 50:
                return None

            closes = df['close'].astype(float)

            change = (closes.iloc[-1] / closes.iloc[0]) - 1.0

            step_ms = martin._interval_ms(interval)
            bars_per_year = (365.25 * 24 * 3600) / (step_ms / 1000.0)
            rv_annual = realized_vol_annual_from_closes(closes, bars_per_year)

            bars_per_day = 24 * 3600 * 1000 / step_ms
            atr_day_window = max(1, min(len(df), int(round(bars_per_day))))
            atr_month_window = max(1, min(len(df), int(round(bars_per_day * 30))))
            atr_daily_est = true_atr_pct(df, n=atr_day_window)
            if not np.isfinite(atr_daily_est):
                atr_daily_est = approx_atr_pct_from_close(closes, n=atr_day_window)
            atr_monthly_est = true_atr_pct(df, n=atr_month_window)
            if not np.isfinite(atr_monthly_est):
                atr_monthly_est = approx_atr_pct_from_close(closes, n=atr_month_window)

            bbw_p = bb_width_pct(closes, n=20, k=2.0)

            max_dd = max_drawdown_pct_from_ohlc(df, closes)

            # Efficiency Ratio (ER)
            abs_net_change = abs(closes.iloc[-1] - closes.iloc[0])
            sum_abs_changes = np.sum(np.abs(np.diff(closes)))
            er = abs_net_change / sum_abs_changes if sum_abs_changes > 0 else 1.0

            # Max Consecutive Red Bars: candle close below candle open.
            is_red = (df["close"].to_numpy(float) < df["open"].to_numpy(float))
            max_consec_red = 0
            if len(is_red) > 0:
                consec_red = 0
                for v in is_red:
                    if v:
                        consec_red += 1
                        if consec_red > max_consec_red:
                            max_consec_red = consec_red
                    else:
                        consec_red = 0

            quote_vols = df['volume'] * df['close']
            total_quote_vol = quote_vols.sum()
            volumes = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0).to_numpy(float)
            active_ratio = float(np.mean(volumes > 0.0)) if volumes.size else float("nan")
            opens = df['open'].to_numpy(dtype=float)
            prev_closes = closes.to_numpy(dtype=float)[:-1]
            if len(opens) >= 2 and np.all(prev_closes > 0):
                max_gap = float(np.max(np.abs(opens[1:] / prev_closes - 1.0)))
            else:
                max_gap = float("nan")

            time_span_days = (
                (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
                + step_ms / 1000.0
            ) / 86400.0

            required_days = max(1.0, (requested_days - 1) * MIN_HISTORY_COVERAGE_RATIO)
            if requested_days > 30:
                required_days = max(30.0, required_days)
            if is_stock_token:
                # Many tokenized equities are newly listed, and Pionex exposes
                # at most 10,000 public K-lines.  Still require a meaningful
                # recent sample without demanding nonexistent multi-year data.
                required_days = min(required_days, STOCK_TOKEN_MAX_REQUIRED_HISTORY_DAYS)
            if time_span_days < required_days and not is_manual:
                return None

            if time_span_days < 0.5:
                time_span_days = 0.5

            avg_daily_vol = total_quote_vol / time_span_days
            bypass_volume = universe.bypass_scanner_volume_filters(exch_name)
            if avg_daily_vol < min_avg_vol and not is_manual and not bypass_volume:
                return None

            return {
                "Symbol": sym,
                "Asset": cand.get('asset_type', "Stock/RWA" if is_stock_token else "Crypto"),
                "Price": closes.iloc[-1],
                "Vol(M)": avg_daily_vol / 1_000_000,
                "Active%": active_ratio * 100,
                "MaxGap%": max_gap * 100,
                "RV(A)%": rv_annual * 100,
                "ATR(Day)%": atr_daily_est * 100,
                "ATR(M)%": atr_monthly_est * 100,
                "BBW(%)": bbw_p * 100,
                "MaxDD%": max_dd * 100,
                "Chg%": change * 100,
                "ER": er,
                "MaxRed": int(max_consec_red),
                "MC Rank": cand.get('mc_rank', -1),
            }
        except Exception as e:
            raise RuntimeError(f"{sym}: {e}") from e

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
            if "RV(A)%" in work_df.columns:
                sort_cols.append("RV(A)%")
                sort_asc.append(False)
            work_df.sort_values(sort_cols, ascending=sort_asc, inplace=True)
            self.scan_results = work_df.drop(columns=["_mc_sort"])
            self._sort_col = "MC Rank"
            self._sort_asc = True
        elif "RV(A)%" in self.scan_results.columns:
            self.scan_results.sort_values("RV(A)%", ascending=False, inplace=True)
            self._sort_col = "RV(A)%"
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
            "Asset": df["Asset"],
            "Price": df["Price"].map(lambda v: f"{float(v):.8f}" if float(v) < 0.01 else f"{float(v):.4f}"),
            "Vol(M)": df["Vol(M)"].map(lambda v: f"{float(v):.2f}"),
            "Active%": df["Active%"].map(lambda v: f"{float(v):.2f}"),
            "MaxGap%": df["MaxGap%"].map(lambda v: f"{float(v):.2f}"),
            "RV(A)%": df["RV(A)%"].map(lambda v: f"{float(v):.2f}"),
            "ATR(M)%": df["ATR(M)%"].map(lambda v: f"{float(v):.2f}"),
            "MaxDD%": df["MaxDD%"].map(lambda v: f"{float(v):.2f}"),
            "Chg%": df["Chg%"].map(lambda v: f"{float(v):.2f}"),
            "ER": df["ER"].map(lambda v: f"{float(v):.3f}"),
            "MaxRed": df["MaxRed"].map(lambda v: f"{int(v)}"),
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

        if col in ["Symbol", "Asset"]:
            self.scan_results.sort_values(col, ascending=ascending, inplace=True)
        elif col == "MC Rank":
            def sort_mc_rank(s):
                n = pd.to_numeric(s, errors="coerce")
                return np.where(n > 0, n, np.inf if ascending else -np.inf)
            self.scan_results.sort_values(col, ascending=ascending, inplace=True, key=sort_mc_rank)
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
            self.plot_chart(df, sym, context.get("interval"))
            return
        self._set_status(f"Loading chart: {sym}")
        worker = ScanWorker(self._load_chart_data, request_token, sym, context)
        worker.signals.finished.connect(self._on_chart_data_loaded)
        worker.signals.error.connect(self._on_chart_load_error)
        self.thread_pool.start(worker)

    def plot_chart(self, df, title, interval=None):
        times = pd.to_datetime(df["time"])
        if getattr(times.dt, "tz", None) is not None:
            times = times.dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
        prices = df["close"].astype(float)

        self._chart_line.set_data(times, prices)
        self.ax.relim()
        self.ax.autoscale_view()
        if len(times) > 0:
            self.ax.set_xlim(times.iloc[0], times.iloc[-1])

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        if len(times) >= 2:
            span_days = (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400.0
        else:
            span_days = 0.0
        if span_days <= 3:
            date_fmt = "%m-%d %H:%M"
        elif span_days <= 180:
            date_fmt = "%Y-%m-%d"
        else:
            date_fmt = "%Y-%m"
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        self.ax.set_title(f"{title} - {interval or '?'}")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate(rotation=30, ha="right")
        self.canvas.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._autosize_table_columns()

    def closeEvent(self, event):
        self.stop_event.set()
        with self._cache_lock:
            clients = list(self._exchange_clients.values())
            self._exchange_clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication([])
    if _import_error is not None:
        QMessageBox.critical(None, "Import Error", f"無法匯入 martin.py：\n{_import_error}")
        raise SystemExit(1)
    w = VolatilityScannerGUI()
    w.show()
    app.exec()
