# martin_gui.py — PySide6 GUI
# Adds: Trapped Time Ratio in metrics log — 2025-10-11
# -*- coding: utf-8 -*-

import traceback
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThreadPool, QRunnable, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QComboBox, QPushButton, QLabel, QSplitter, QTextEdit,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QGroupBox, QStatusBar,
    QAbstractItemView,
    QHeaderView,
)

try:
    import martin
    _import_error = None
except Exception as e:
    martin = None
    _import_error = e

from mc_sampling import sample_parameter_grid, refine_neighbors
from mc_eval import eval_candidates_parallel, bootstrap_path_from_returns
from mc_formatters import format_hist_scan_display, format_hist_scan_csv, format_mc_scan_display


CRYPTO_SYMBOLS = ["PUMP", "ASTER", "BONK", "ENA", "PEPE", "WLD", "WLFI", "ZEC", "TRUMP", "TAO", "SUI", "HBAR", "UNI",
                  "NEAR", "FIL", "APT", "ARB", "DOGE", "SHIB", "ADA", "AVAX", "LINK", "XRP", "SOL", "LTC", "ETH",
                  "BNB", "TRX", "BCH", "BTC"]
CRYPTO_INTERVALS = ["15m", "1h", "4h", "1d"]


def parse_range(s: str, is_int=False):
    s = (s or "").strip()
    if not s:
        return np.array([], dtype=int if is_int else float)
    if ":" not in s:
        raise ValueError("掃描參數僅支援範圍格式 start:end:step")
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


def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default


def safe_int(s, default=None):
    try:
        return int(s)
    except Exception:
        return default


def human_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    return f"{x * 100:.{digits}f}%"



class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class MartinGUI(QMainWindow):
    INIT_SPLIT = 0.70

    def __init__(self):
        super().__init__()
        self.setWindowTitle("martin_gui")
        self.resize(1030, 920)

        self.thread_pool = QThreadPool.globalInstance()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

        self.nb = QTabWidget()
        self.setCentralWidget(self.nb)

        self.tab_hist_scan = QWidget()
        self.tab_mc_scan = QWidget()
        self.tab_single = QWidget()
        self.nb.addTab(self.tab_hist_scan, "Historical Scan")
        self.nb.addTab(self.tab_mc_scan, "MC Scan")
        self.nb.addTab(self.tab_single, "Single Backtest")

        # shared cache
        self.last_df = None
        self.last_df_key = None
        self.scan_df = None
        self.mc_scan_df = None
        self._mc_executor = None
        self._mc_executor_workers = 0

        self._build_scan_tab()
        self._build_mc_scan_tab()
        self._build_single_tab()

        self.nb.currentChanged.connect(self._on_main_tab_changed)

        self._apply_style()
        QTimer.singleShot(0, self._autosize_scan_table_columns)
        QTimer.singleShot(0, self._autosize_mc_scan_table_columns)

        QTimer.singleShot(0, lambda: self._set_splitter_when_ready(self.scan_splitter, self.INIT_SPLIT))
        QTimer.singleShot(0, lambda: self._set_splitter_when_ready(self.single_splitter, self.INIT_SPLIT))

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
            QLabel#formLabel { font-size: 11pt; font-weight: 500; }
            QLineEdit, QComboBox, QTextEdit {
                background: #ffffff; color: #1d1d1d; border: 1px solid #b9b9b9; border-radius: 6px; padding: 4px 6px;
            }
            QComboBox QAbstractItemView { background: #ffffff; color: #1d1d1d; }
            QTableWidget { background: #ffffff; color: #1d1d1d; border: 1px solid #7c7c7c; border-radius: 8px; gridline-color: #d0d0d0; }
            QTableWidget::item:selected { font-weight: 400; }
            QHeaderView::section { background: #e9e9e9; color: #1d1d1d; padding: 6px; border: 0px; }
            QHeaderView::section:selected { font-weight: 400; }
            QHeaderView::section:checked { font-weight: 400; }
            QTabWidget::pane { border: 1px solid #6e6e6e; border-radius: 8px; }
            QTabBar::tab { background: #4b4b4b; color: #f2f2f2; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #35507a; }

            QPushButton { background: #59626a; color: #ffffff; border: 0px; border-radius: 8px; padding: 6px 12px; }
            QPushButton:hover { background: #4f575e; }
            QPushButton:pressed { background: #454c52; }

            QPushButton#btnPrimary { background: #2d6a7a; font-weight: 700; }
            QPushButton#btnPrimary:hover { background: #295f6d; }
            QPushButton#btnPrimary:pressed { background: #23535f; }

            QPushButton#btnInfo { background: #b26a00; }
            QPushButton#btnInfo:hover { background: #9f5f00; }
            QPushButton#btnInfo:pressed { background: #8c5400; }

            QPushButton#btnInfo2 { background: #6b7d2a; }
            QPushButton#btnInfo2:hover { background: #5f6f24; }
            QPushButton#btnInfo2:pressed { background: #54621f; }

            QPushButton#btnDanger { background: #8a2d3a; }
            QPushButton#btnDanger:hover { background: #7a2833; }
            QPushButton#btnDanger:pressed { background: #6b232c; }

            QSplitter::handle { background: #7c7c7c; }
            """
        )

    def _set_status(self, text):
        self.status_bar.showMessage(text)

    def _set_splitter_when_ready(self, splitter, ratio=0.70, tries=20, delay=60):
        if splitter is None:
            return
        sizes = splitter.sizes()
        h = sum(sizes)
        if h and h > 50:
            top = int(h * float(ratio))
            splitter.setSizes([top, max(10, h - top)])
            return
        if tries <= 0:
            return
        QTimer.singleShot(delay, lambda: self._set_splitter_when_ready(splitter, ratio, tries - 1, delay))

    # ---------- widgets ----------
    def _add_form_row(self, form: QFormLayout, label: str, widget: QWidget):
        lbl = QLabel(label)
        lbl.setMinimumWidth(140)
        lbl.setObjectName("formLabel")
        form.addRow(lbl, widget)

    def _add_entry(self, form: QFormLayout, label: str, default="", width=140):
        e = QLineEdit()
        e.setText(default)
        e.setMinimumWidth(width)
        self._add_form_row(form, label, e)
        return e

    def _add_combobox(self, form: QFormLayout, label: str, values, default=""):
        cb = QComboBox()
        cb.addItems(list(values))
        if default:
            idx = cb.findText(default)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        self._add_form_row(form, label, cb)
        return cb

    # ---------- scan tab ----------
    def _build_scan_tab(self):
        layout = QVBoxLayout(self.tab_hist_scan)

        top = QHBoxLayout()
        layout.addLayout(top)

        gb_data = QGroupBox("Data Settings")
        gb_grid = QGroupBox("Scan Parameters ('start:end:step')")
        gb_filter = QGroupBox("Filter Conditions (optional)")

        top.addWidget(gb_data)
        top.addWidget(gb_grid, 1)
        top.addWidget(gb_filter)

        form_data = QFormLayout(gb_data)
        self.e_symbol = self._add_combobox(form_data, "Symbol:", CRYPTO_SYMBOLS, default="SUI")
        self.e_interval = self._add_combobox(form_data, "Interval:", CRYPTO_INTERVALS, default="15m")
        self.e_scan_days = self._add_entry(form_data, "Scan Days:", "730")
        self.e_start = self._add_entry(form_data, "Start (e.g.2025-07-01):", "")
        self.e_end = self._add_entry(form_data, "End (e.g.2025-08-30):", "")
        self.e_refresh = self._add_entry(form_data, "Cache Policy:", "auto")
        self.e_fee = self._add_entry(form_data, "Fee rate:", "0.0005")
        self.e_capital = self._add_entry(form_data, "Capital:", "1000")

        form_grid = QFormLayout(gb_grid)
        self.e_add_drop = self._add_entry(form_grid, "add_drop:", "0.010:0.080:0.001")
        self.e_tp = self._add_entry(form_grid, "tp:", "0.010:0.080:0.001")
        self.e_multiplier = self._add_entry(form_grid, "multiplier:", "1.5:2.0:0.1")
        self.e_max_orders = self._add_entry(form_grid, "max_orders:", "5:9:1")

        form_filter = QFormLayout(gb_filter)
        self.e_min_trades = self._add_entry(form_filter, "min_trades:", "104")
        self.e_max_dd = self._add_entry(form_filter, "max_dd_overall(%):", "")
        self.e_max_trap = self._add_entry(form_filter, "max_trapped_ratio(%):", "20")
        self.e_topn = self._add_entry(form_filter, "Show Top N:", "20")

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_run = QPushButton("Run Scan")
        btn_run.setObjectName("btnPrimary")
        btn_csv = QPushButton("Save Result as CSV")
        btn_csv.setObjectName("btnInfo")
        btn_clear = QPushButton("Clear Results")
        btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)

        btn_run.clicked.connect(self.run_scan)
        btn_csv.clicked.connect(self.save_scan_csv)
        btn_clear.clicked.connect(self.clear_scan_results)

        self.scan_tabs = QTabWidget()
        layout.addWidget(self.scan_tabs, 1)

        # Table tab
        self.scan_tab_table = QWidget()
        table_layout = QVBoxLayout(self.scan_tab_table)
        self.table = QTableWidget(0, 9)
        cols = ["add_drop", "tp", "multiplier", "max_orders", "min_buy_ratio",
                "final_equity", "max_dd_overall", "trades", "trapped_time_ratio"]
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setWordWrap(False)
        self._autosize_scan_table_columns()
        table_layout.addWidget(self.table)

        hint = QLabel("提示：雙擊表格繪圖，會自動切換到「Backtest Chart / Performance」。")
        hint.setStyleSheet("color: #6b6b6b; padding: 6px 2px;")
        table_layout.addWidget(hint)

        self.table.cellDoubleClicked.connect(self.plot_selected_from_table)

        # Detail tab
        self.scan_tab_detail = QWidget()
        detail_layout = QVBoxLayout(self.scan_tab_detail)
        self.scan_splitter = QSplitter(Qt.Vertical)
        detail_layout.addWidget(self.scan_splitter, 1)

        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        self.figure_scan = Figure(figsize=(6, 4), dpi=100)
        self.canvas_scan = FigureCanvas(self.figure_scan)
        plot_layout.addWidget(self.canvas_scan)

        metrics_frame = QWidget()
        metrics_layout = QVBoxLayout(metrics_frame)
        self.scan_metrics_text = QTextEdit()
        self.scan_metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.scan_metrics_text)

        self.scan_splitter.addWidget(plot_frame)
        self.scan_splitter.addWidget(metrics_frame)

        self.scan_tabs.addTab(self.scan_tab_table, "Results Table")
        self.scan_tabs.addTab(self.scan_tab_detail, "Backtest Chart / Performance")

        self.scan_tabs.currentChanged.connect(self._on_scan_tab_changed)

    def _autosize_scan_table_columns(self, max_col_width=280):
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

    # ---------- MC scan tab ----------
    def _build_mc_scan_tab(self):
        layout = QVBoxLayout(self.tab_mc_scan)

        top = QHBoxLayout()
        layout.addLayout(top)

        gb_data = QGroupBox("Data Settings")
        gb_grid = QGroupBox("Global Parameter Grid")
        gb_mc = QGroupBox("Monte Carlo Settings")
        gb_risk = QGroupBox("Risk Constraints")

        top.addWidget(gb_data)
        top.addWidget(gb_grid, 1)
        top.addWidget(gb_mc)
        top.addWidget(gb_risk)

        form_data = QFormLayout(gb_data)
        self.m_symbol = self._add_combobox(form_data, "Symbol:", CRYPTO_SYMBOLS, default="SUI")
        self.m_interval = self._add_combobox(form_data, "Interval:", CRYPTO_INTERVALS, default="15m")
        self.m_scan_days = self._add_entry(form_data, "Scan Days:", "730")
        self.m_start = self._add_entry(form_data, "Start (e.g.2025-07-01):", "")
        self.m_end = self._add_entry(form_data, "End (e.g.2025-08-30):", "")
        self.m_refresh = self._add_entry(form_data, "Cache Policy:", "auto")
        self.m_fee = self._add_entry(form_data, "Fee rate:", "0.0005")
        self.m_capital = self._add_entry(form_data, "Capital:", "1000")

        form_grid = QFormLayout(gb_grid)
        self.m_add_drop = self._add_entry(form_grid, "add_drop:", "0.010:0.080:0.001")
        self.m_tp = self._add_entry(form_grid, "tp:", "0.010:0.080:0.001")
        self.m_multiplier = self._add_entry(form_grid, "multiplier:", "1.5:2.0:0.1")
        self.m_max_orders = self._add_entry(form_grid, "max_orders:", "5:9:1")
        self.m_sampling_mode = self._add_combobox(
            form_grid,
            "Sampling:",
            ["LHS (分層抽樣)", "Random (隨機抽樣)", "Full Grid (全排列)"],
            default="LHS (分層抽樣)",
        )
        self.m_sample_size = self._add_entry(form_grid, "Sample size:", "5000")
        self.m_refine_pct = self._add_entry(form_grid, "Refine top(%):", "5")
        self.m_max_combos = self._add_entry(form_grid, "Full-grid cap (0=all):", "0")
        self.m_hist_min_trades = self._add_entry(form_grid, "Hist min_trades:", "104")
        self.m_hist_max_trap = self._add_entry(form_grid, "Hist max_trap(%):", "20")
        self.m_show_topn = self._add_entry(form_grid, "Show Top N:", "50")
        self._mc_grid_form = form_grid
        self.m_sampling_mode.currentTextChanged.connect(self._on_mc_sampling_mode_changed)
        self._on_mc_sampling_mode_changed(self.m_sampling_mode.currentText())

        form_mc = QFormLayout(gb_mc)
        self.m_mc_paths = self._add_entry(form_mc, "MC paths:", "300")
        self.m_mc_days = self._add_entry(form_mc, "Days/path:", "730")
        self.m_mc_block = self._add_entry(form_mc, "Block size:", "96")
        self.m_mc_seed = self._add_entry(form_mc, "Seed:", "42")
        self.m_mc_seed_runs = self._add_entry(form_mc, "Seed runs:", "1")
        self.m_mc_workers = self._add_entry(form_mc, "Workers (0=auto):", "0")
        self.m_rank_by = self._add_combobox(
            form_mc,
            "Rank by:",
            ["Median terminal", "P5 terminal", "Lowest DD mean"],
            default="Median terminal",
        )

        form_risk = QFormLayout(gb_risk)
        self.m_max_loss = self._add_entry(form_risk, "Max P(loss)%:", "100")
        self.m_max_severe = self._add_entry(form_risk, "Max P(severe)%:", "100")
        self.m_max_dd50 = self._add_entry(form_risk, "Max P(DD>50)%:", "100")

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_run = QPushButton("Run Scan")
        btn_run.setObjectName("btnPrimary")
        btn_csv = QPushButton("Save Result as CSV")
        btn_csv.setObjectName("btnInfo")
        btn_clear = QPushButton("Clear Results")
        btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)

        btn_run.clicked.connect(self.run_mc_scan)
        btn_csv.clicked.connect(self.save_mc_scan_csv)
        btn_clear.clicked.connect(self.clear_mc_scan_results)

        self.mc_tabs = QTabWidget()
        layout.addWidget(self.mc_tabs, 1)

        self.mc_tab_table = QWidget()
        mc_table_layout = QVBoxLayout(self.mc_tab_table)
        self.mc_table = QTableWidget(0, 0)
        self.mc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.mc_table.horizontalHeader().setHighlightSections(False)
        self.mc_table.setWordWrap(False)
        mc_table_layout.addWidget(self.mc_table)

        mc_hint = QLabel("提示：雙擊表格繪圖，會自動切換到「Backtest Chart / Performance」。")
        mc_hint.setStyleSheet("color: #6b6b6b; padding: 6px 2px;")
        mc_table_layout.addWidget(mc_hint)
        self.mc_table.cellDoubleClicked.connect(self.plot_selected_from_mc_table)

        self.mc_tab_detail = QWidget()
        mc_detail_layout = QVBoxLayout(self.mc_tab_detail)
        self.mc_splitter = QSplitter(Qt.Vertical)
        mc_detail_layout.addWidget(self.mc_splitter, 1)

        mc_plot_frame = QWidget()
        mc_plot_layout = QVBoxLayout(mc_plot_frame)
        self.figure_mc_scan = Figure(figsize=(6, 4), dpi=100)
        self.canvas_mc_scan = FigureCanvas(self.figure_mc_scan)
        mc_plot_layout.addWidget(self.canvas_mc_scan)

        mc_metrics_frame = QWidget()
        mc_metrics_layout = QVBoxLayout(mc_metrics_frame)
        self.mc_metrics_text = QTextEdit()
        self.mc_metrics_text.setReadOnly(True)
        mc_metrics_layout.addWidget(self.mc_metrics_text)

        self.mc_splitter.addWidget(mc_plot_frame)
        self.mc_splitter.addWidget(mc_metrics_frame)

        self.mc_tabs.addTab(self.mc_tab_table, "Results Table")
        self.mc_tabs.addTab(self.mc_tab_detail, "Backtest Chart / Performance")
        self.mc_tabs.currentChanged.connect(self._on_mc_tab_changed)

    def _autosize_mc_scan_table_columns(self, max_col_width=280):
        if not hasattr(self, "mc_table") or self.mc_table is None:
            return
        header = self.mc_table.horizontalHeader()
        metrics = QFontMetrics(header.font())
        pad_px = 24
        min_col_width = 80

        for c in range(self.mc_table.columnCount()):
            h_item = self.mc_table.horizontalHeaderItem(c)
            text = h_item.text() if h_item is not None else ""
            best = metrics.horizontalAdvance(text) + pad_px
            for r in range(self.mc_table.rowCount()):
                item = self.mc_table.item(r, c)
                if item is None:
                    continue
                best = max(best, metrics.horizontalAdvance(item.text()) + pad_px)
            self.mc_table.setColumnWidth(c, min(max_col_width, max(min_col_width, best)))

        header.setStretchLastSection(True)

    def _set_form_row_visible(self, form: QFormLayout, field: QWidget, visible: bool):
        lbl = form.labelForField(field)
        if lbl is not None:
            lbl.setVisible(bool(visible))
        field.setVisible(bool(visible))

    def _on_mc_sampling_mode_changed(self, text: str):
        mode = (text or "").strip().lower()
        is_full = mode.startswith("full grid")
        self._set_form_row_visible(self._mc_grid_form, self.m_sample_size, not is_full)
        self._set_form_row_visible(self._mc_grid_form, self.m_refine_pct, not is_full)
        self._set_form_row_visible(self._mc_grid_form, self.m_max_combos, is_full)

    # ---------- single tab ----------
    def _build_single_tab(self):
        layout = QVBoxLayout(self.tab_single)

        top = QHBoxLayout()
        layout.addLayout(top)

        gb_data = QGroupBox("Data Settings")
        gb_params = QGroupBox("Strategy Parameters")
        gb_mc = QGroupBox("Monte Carlo")

        top.addWidget(gb_data)
        top.addWidget(gb_params)
        top.addWidget(gb_mc)
        top.addStretch(1)

        form_data = QFormLayout(gb_data)
        self.s_symbol = self._add_combobox(form_data, "Symbol:", CRYPTO_SYMBOLS, default="SUI")
        self.s_interval = self._add_combobox(form_data, "Interval:", CRYPTO_INTERVALS, default="15m")
        self.s_scan_days = self._add_entry(form_data, "Scan Days:", "730")
        self.s_start = self._add_entry(form_data, "Start (e.g.2025-07-01):", "")
        self.s_end = self._add_entry(form_data, "End (e.g.2025-08-30):", "")
        self.s_refresh = self._add_entry(form_data, "Cache Policy:", "auto")
        self.s_fee = self._add_entry(form_data, "Fee rate:", "0.0005")
        self.s_capital = self._add_entry(form_data, "Capital:", "1000")

        form_params = QFormLayout(gb_params)
        self.s_add_drop = self._add_entry(form_params, "add_drop:", "0.05")
        self.s_tp = self._add_entry(form_params, "tp:", "0.05")
        self.s_multiplier = self._add_entry(form_params, "multiplier:", "2.0")
        self.s_max_orders = self._add_entry(form_params, "max_orders:", "7")

        form_mc = QFormLayout(gb_mc)
        self.s_mc_paths = self._add_entry(form_mc, "MC paths:", "1000")
        self.s_mc_days = self._add_entry(form_mc, "Days/path:", "730")
        self.s_mc_block = self._add_entry(form_mc, "Block size:", "96")
        self.s_mc_seed = self._add_entry(form_mc, "Random seed:", "42")

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_run = QPushButton("Execute Single Backtest and Plot")
        btn_run.setObjectName("btnPrimary")
        btn_mc = QPushButton("Run Monte Carlo")
        btn_mc.setObjectName("btnInfo")
        btn_clear = QPushButton("Clear Results")
        btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_mc)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)

        btn_run.clicked.connect(self.run_single)
        btn_mc.clicked.connect(self.run_single_mc)
        btn_clear.clicked.connect(self.clear_single_results)

        self.single_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.single_splitter, 1)

        plot_frame = QWidget()
        plot_layout = QVBoxLayout(plot_frame)
        self.figure_single = Figure(figsize=(6, 4), dpi=100)
        self.canvas_single = FigureCanvas(self.figure_single)
        plot_layout.addWidget(self.canvas_single)

        metrics_frame = QWidget()
        metrics_layout = QVBoxLayout(metrics_frame)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)

        self.single_splitter.addWidget(plot_frame)
        self.single_splitter.addWidget(metrics_frame)

    # ---------- events ----------
    def _on_scan_tab_changed(self, idx):
        if self.scan_tabs.widget(idx) == self.scan_tab_detail:
            self._set_splitter_when_ready(self.scan_splitter, self.INIT_SPLIT)
        elif self.scan_tabs.widget(idx) == self.scan_tab_table:
            self._autosize_scan_table_columns()

    def _on_mc_tab_changed(self, idx):
        if self.mc_tabs.widget(idx) == self.mc_tab_detail:
            self._set_splitter_when_ready(self.mc_splitter, self.INIT_SPLIT)
        elif self.mc_tabs.widget(idx) == self.mc_tab_table:
            self._autosize_mc_scan_table_columns()

    def _on_main_tab_changed(self, _idx):
        self._set_splitter_when_ready(self.single_splitter, self.INIT_SPLIT)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._autosize_scan_table_columns()
        self._autosize_mc_scan_table_columns()

    def closeEvent(self, event):
        try:
            if self._mc_executor is not None:
                self._mc_executor.shutdown(wait=False, cancel_futures=False)
                self._mc_executor = None
                self._mc_executor_workers = 0
        finally:
            super().closeEvent(event)

    # ---------- data ----------
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
        self.last_df = df
        self.last_df_key = key
        return df

    # ---------- scan ----------
    def run_scan(self):
        self._set_status("Scanning…（首次可能較慢，numba 正在編譯）")
        worker = Worker(self._scan_compute)
        worker.signals.finished.connect(self._scan_update_ui)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _collect_context_from_inputs(
        self, symbol_cb, interval_cb, scan_days_edit, start_edit, end_edit, refresh_edit, fee_edit, capital_edit
    ):
        symbol = symbol_cb.currentText().strip()
        interval = interval_cb.currentText().strip()
        scan_days = safe_float(scan_days_edit.text().strip(), 0.0)
        start = start_edit.text().strip()
        end = end_edit.text().strip()
        bars = 0
        if scan_days > 0:
            step_ms = martin._interval_ms(interval)
            bars = int(scan_days * 86400 * 1000 / step_ms)

        refresh_policy = (refresh_edit.text().strip().lower() or "auto")
        fee_rate = safe_float(fee_edit.text().strip(), 0.0)
        capital = safe_float(capital_edit.text().strip(), 1000.0)
        if not symbol or not interval:
            raise ValueError("請填入 symbol 與 interval")
        if (not bars) and (not start or not end):
            raise ValueError("Scan Days 與 Start/End 需擇一填寫")

        df = self._fetch_klines_if_needed(symbol, interval, bars, start, end, refresh_policy)
        prices_np = df["close"].to_numpy(dtype=np.float64)
        if prices_np.size < 2:
            raise ValueError("K 線資料不足（<2 根），無法回測/掃描。")

        return {
            "symbol": symbol,
            "interval": interval,
            "bars": bars,
            "start": start,
            "end": end,
            "refresh_policy": refresh_policy,
            "fee_rate": float(fee_rate),
            "capital": float(capital),
            "df": df,
            "prices_np": prices_np,
        }

    def _collect_scan_context(self):
        return self._collect_context_from_inputs(
            self.e_symbol,
            self.e_interval,
            self.e_scan_days,
            self.e_start,
            self.e_end,
            self.e_refresh,
            self.e_fee,
            self.e_capital,
        )

    def _collect_mc_scan_context(self):
        return self._collect_context_from_inputs(
            self.m_symbol,
            self.m_interval,
            self.m_scan_days,
            self.m_start,
            self.m_end,
            self.m_refresh,
            self.m_fee,
            self.m_capital,
        )

    def _compute_filtered_scan_results(self, prices_np, capital, fee_rate):
        add_drop_arr = parse_range(self.e_add_drop.text())
        tp_arr = parse_range(self.e_tp.text())
        mul_arr = parse_range(self.e_multiplier.text())
        mo_arr = parse_range(self.e_max_orders.text(), is_int=True)
        if any(x.size == 0 for x in (add_drop_arr, tp_arr, mul_arr, mo_arr)):
            raise ValueError("掃描參數不得為空（add_drop/tp/multiplier/max_orders）")

        AD, MUL, MO, TP = np.meshgrid(add_drop_arr, mul_arr, mo_arr, tp_arr, indexing="ij")
        ADf = AD.ravel().astype(np.float64)
        MULf = MUL.ravel().astype(np.float64)
        MOf = MO.ravel().astype(np.int32)
        TPf = TP.ravel().astype(np.float64)
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

        min_trades = safe_int(self.e_min_trades.text().strip()) if self.e_min_trades.text().strip() else None
        max_dd = safe_float(self.e_max_dd.text().strip()) if self.e_max_dd.text().strip() else None
        max_trap = safe_float(self.e_max_trap.text().strip()) if self.e_max_trap.text().strip() else None
        if max_trap is not None:
            max_trap /= 100.0
        return martin.apply_filters(results_df, min_trades, max_dd, max_trap)

    def _compute_filtered_mc_candidates(self, prices_np, capital, fee_rate):
        add_drop_arr = parse_range(self.m_add_drop.text())
        tp_arr = parse_range(self.m_tp.text())
        mul_arr = parse_range(self.m_multiplier.text())
        mo_arr = parse_range(self.m_max_orders.text(), is_int=True)
        if any(x.size == 0 for x in (add_drop_arr, tp_arr, mul_arr, mo_arr)):
            raise ValueError("掃描參數不得為空（add_drop/tp/multiplier/max_orders）")

        mode_text = self.m_sampling_mode.currentText().strip().lower()
        mode = "lhs"
        if mode_text.startswith("random"):
            mode = "random"
        elif mode_text.startswith("full grid"):
            mode = "full grid"

        seed = safe_int(self.m_mc_seed.text().strip(), 42)
        sample_size = max(1, safe_int(self.m_sample_size.text().strip(), 5000))
        max_combos = safe_int(self.m_max_combos.text().strip(), 0)
        refine_pct = max(0.0, safe_float(self.m_refine_pct.text().strip(), 5.0))
        refine_radius = 1
        refine_max_add = max(0, min(3000, sample_size))
        if mode == "full grid":
            refine_pct = 0.0

        params_df = sample_parameter_grid(
            add_drop_arr=add_drop_arr,
            tp_arr=tp_arr,
            mul_arr=mul_arr,
            mo_arr=mo_arr,
            mode=mode,
            sample_size=sample_size,
            max_combos=max_combos if max_combos is not None else 0,
            seed=seed,
        )
        if params_df.empty:
            return params_df

        ADf = params_df["add_drop"].to_numpy(dtype=np.float64)
        MULf = params_df["multiplier"].to_numpy(dtype=np.float64)
        MOf = params_df["max_orders"].to_numpy(dtype=np.int32)
        TPf = params_df["tp"].to_numpy(dtype=np.float64)
        min_buy_ratio_theory = np.maximum(0.0, (1.0 - ADf) ** (MOf.astype(np.float64) - 1.0))

        fe, mdd, tr, trap = martin._grid_search_parallel(
            prices_np, ADf, MULf, MOf, TPf, capital=float(capital), fee_rate=float(fee_rate)
        )
        results_df = pd.DataFrame({
            "add_drop": ADf,
            "multiplier": MULf,
            "max_orders": MOf.astype(int),
            "tp": TPf,
            "capital": float(capital),
            "final_equity": np.round(fe, 2),
            "max_dd_overall": np.round(mdd, 2),
            "trades": tr.astype(int),
            "trapped_time_ratio": np.round(trap, 6),
            "min_buy_ratio": np.round(min_buy_ratio_theory, 6),
        }).drop_duplicates(subset=["add_drop", "multiplier", "max_orders", "tp"], ignore_index=True)

        min_trades = safe_int(self.m_hist_min_trades.text().strip()) if self.m_hist_min_trades.text().strip() else None
        max_trap = safe_float(self.m_hist_max_trap.text().strip()) if self.m_hist_max_trap.text().strip() else None
        if max_trap is not None:
            max_trap /= 100.0
        filtered = martin.apply_filters(results_df, min_trades, None, max_trap)

        if refine_radius > 0 and refine_max_add > 0 and refine_pct > 0 and not filtered.empty:
            top_k = max(1, int(np.ceil(len(filtered) * (refine_pct / 100.0))))
            seeds = filtered.nlargest(top_k, "final_equity")
            refine_df = refine_neighbors(
                base_df=results_df,
                seeds_df=seeds,
                add_drop_arr=add_drop_arr,
                mul_arr=mul_arr,
                mo_arr=mo_arr,
                tp_arr=tp_arr,
                radius=refine_radius,
                max_add=refine_max_add,
            )
            if not refine_df.empty:
                AD2 = refine_df["add_drop"].to_numpy(dtype=np.float64)
                MUL2 = refine_df["multiplier"].to_numpy(dtype=np.float64)
                MO2 = refine_df["max_orders"].to_numpy(dtype=np.int32)
                TP2 = refine_df["tp"].to_numpy(dtype=np.float64)
                min_buy2 = np.maximum(0.0, (1.0 - AD2) ** (MO2.astype(np.float64) - 1.0))
                fe2, mdd2, tr2, trap2 = martin._grid_search_parallel(
                    prices_np, AD2, MUL2, MO2, TP2, capital=float(capital), fee_rate=float(fee_rate)
                )
                extra = pd.DataFrame({
                    "add_drop": AD2,
                    "multiplier": MUL2,
                    "max_orders": MO2.astype(int),
                    "tp": TP2,
                    "capital": float(capital),
                    "final_equity": np.round(fe2, 2),
                    "max_dd_overall": np.round(mdd2, 2),
                    "trades": tr2.astype(int),
                    "trapped_time_ratio": np.round(trap2, 6),
                    "min_buy_ratio": np.round(min_buy2, 6),
                })
                results_df = pd.concat([results_df, extra], ignore_index=True).drop_duplicates(
                    subset=["add_drop", "multiplier", "max_orders", "tp"],
                    ignore_index=True,
                )
                filtered = martin.apply_filters(results_df, min_trades, None, max_trap)

        return filtered.copy()

    def _format_hist_scan_display(self, top_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        return format_hist_scan_display(top_df, martin.pct_str)

    def _format_hist_scan_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        return format_hist_scan_csv(df)

    def _format_mc_scan_display(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        return format_mc_scan_display(df, martin.pct_str, human_pct)

    def _get_mc_executor(self, workers: int):
        if workers <= 0:
            return None
        if self._mc_executor is not None and self._mc_executor_workers == workers:
            return self._mc_executor
        if self._mc_executor is not None:
            self._mc_executor.shutdown(wait=True, cancel_futures=False)
        from concurrent.futures import ProcessPoolExecutor
        self._mc_executor = ProcessPoolExecutor(max_workers=workers)
        self._mc_executor_workers = workers
        return self._mc_executor

    def _eval_mc_candidates_parallel(
        self,
        candidates: pd.DataFrame,
        hist_rets: np.ndarray,
        start_price: float,
        capital: float,
        fee_rate: float,
        mc_bars: int,
        block_size: int,
        total_paths: int,
        base_seed: int,
        max_loss: float,
        max_severe: float,
        max_dd50: float,
        workers: int,
    ) -> pd.DataFrame:
        ex = self._get_mc_executor(int(workers))
        return eval_candidates_parallel(
            candidates=candidates,
            hist_rets=hist_rets,
            start_price=float(start_price),
            capital=float(capital),
            fee_rate=float(fee_rate),
            mc_bars=int(mc_bars),
            block_size=int(block_size),
            total_paths=int(total_paths),
            base_seed=int(base_seed),
            max_loss=float(max_loss),
            max_severe=float(max_severe),
            max_dd50=float(max_dd50),
            workers=int(workers),
            executor=ex,
        )

    def _scan_compute(self):
        ctx = self._collect_scan_context()
        filtered = self._compute_filtered_scan_results(ctx["prices_np"], ctx["capital"], ctx["fee_rate"])
        topn = safe_int(self.e_topn.text().strip(), 20)
        top_df = filtered.nlargest(topn, "final_equity").copy()
        return top_df

    def _populate_scan_table(self, disp: pd.DataFrame, cols):
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(list(cols))
        self.table.setRowCount(0)
        for _, row in disp[cols].iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, col in enumerate(cols):
                item = QTableWidgetItem(str(row[col]))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
        self._autosize_scan_table_columns()

    def _populate_mc_scan_table(self, disp: pd.DataFrame, cols):
        self.mc_table.setColumnCount(len(cols))
        self.mc_table.setHorizontalHeaderLabels(list(cols))
        self.mc_table.setRowCount(0)
        for _, row in disp[cols].iterrows():
            r = self.mc_table.rowCount()
            self.mc_table.insertRow(r)
            for c, col in enumerate(cols):
                item = QTableWidgetItem(str(row[col]))
                item.setTextAlignment(Qt.AlignCenter)
                self.mc_table.setItem(r, c, item)
        self._autosize_mc_scan_table_columns()

    @Slot(object)
    def _scan_update_ui(self, top_df: pd.DataFrame):
        disp, cols = self._format_hist_scan_display(top_df)
        self._populate_scan_table(disp, cols)

        self.scan_df = top_df.reset_index(drop=True)
        if self.scan_df.empty:
            self._set_status("Scan completed. No rows matched current filters.")
            QMessageBox.information(self, "Info", "篩選條件過嚴，請放寬。")
        else:
            self._set_status("Scan completed。")
        self.scan_tabs.setCurrentWidget(self.scan_tab_table)

    # ---------- MC scan ----------
    def run_mc_scan(self):
        sampling = (self.m_sampling_mode.currentText() or "").strip()
        sampling_short = sampling.split("(", 1)[0].strip() if sampling else "Unknown"
        self._set_status(f"Scan中…（{sampling_short} + 風險約束，可能較久）")
        worker = Worker(self._mc_scan_compute)
        worker.signals.finished.connect(self._mc_scan_update_ui)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _mc_scan_compute(self):
        ctx = self._collect_mc_scan_context()
        candidates = self._compute_filtered_mc_candidates(ctx["prices_np"], ctx["capital"], ctx["fee_rate"])
        n_cand = len(candidates)
        if candidates.empty:
            return pd.DataFrame(), 0, 0

        mc_paths = safe_int(self.m_mc_paths.text().strip(), None)
        mc_days = safe_float(self.m_mc_days.text().strip(), None)
        mc_block = safe_int(self.m_mc_block.text().strip(), None)
        mc_seed = safe_int(self.m_mc_seed.text().strip(), 42)
        seed_runs = safe_int(self.m_mc_seed_runs.text().strip(), 1)
        workers = safe_int(self.m_mc_workers.text().strip(), 0)
        if None in (mc_paths, mc_days, mc_block):
            raise ValueError("請完整填入參數（paths/days/block）")
        if mc_paths <= 0 or mc_days <= 0 or mc_block <= 0 or seed_runs <= 0:
            raise ValueError("參數需滿足：paths>0、days>0、block>0、seed runs>0")
        if workers <= 0:
            workers = max(1, min(os.cpu_count() or 1, 8))
        workers = min(int(workers), max(1, int(n_cand)))

        max_loss = safe_float(self.m_max_loss.text().strip(), 30.0) / 100.0
        max_severe = safe_float(self.m_max_severe.text().strip(), 10.0) / 100.0
        max_dd50 = safe_float(self.m_max_dd50.text().strip(), 20.0) / 100.0

        prices_np = ctx["prices_np"]
        hist_rets = prices_np[1:] / prices_np[:-1] - 1.0
        start_price = float(prices_np[-1])
        mc_bars = self._days_to_bars(mc_days, ctx["interval"])
        total_paths = int(mc_paths * seed_runs)
        out = self._eval_mc_candidates_parallel(
            candidates=candidates,
            hist_rets=hist_rets,
            start_price=start_price,
            capital=float(ctx["capital"]),
            fee_rate=float(ctx["fee_rate"]),
            mc_bars=int(mc_bars),
            block_size=int(mc_block),
            total_paths=int(total_paths),
            base_seed=int(mc_seed),
            max_loss=float(max_loss),
            max_severe=float(max_severe),
            max_dd50=float(max_dd50),
            workers=int(workers),
        )
        out["mc_paths"] = int(mc_paths)
        out["mc_seed_runs"] = int(seed_runs)
        out["mc_days"] = float(mc_days)
        out["mc_bars"] = int(mc_bars)
        out["mc_block"] = int(mc_block)
        out["mc_seed"] = int(mc_seed)

        rank_mode = self.m_rank_by.currentText().strip()
        if rank_mode == "P5 terminal":
            sort_by = ["feasible", "mc_terminal_p5", "mc_terminal_median", "mc_p_dd50", "mc_p_loss"]
            asc = [False, False, False, True, True]
        elif rank_mode == "Lowest DD mean":
            sort_by = ["feasible", "mc_mdd_mean", "mc_p_dd50", "mc_p_loss", "mc_terminal_median"]
            asc = [False, True, True, True, False]
        else:
            sort_by = ["feasible", "mc_terminal_median", "mc_terminal_p5", "mc_p_dd50", "mc_p_loss"]
            asc = [False, False, False, True, True]

        out = out.sort_values(by=sort_by, ascending=asc).reset_index(drop=True)
        feasible_count = int(out["feasible"].sum())
        show_topn = safe_int(self.m_show_topn.text().strip(), 50)
        if show_topn > 0:
            out = out.head(show_topn).copy()
        return out, feasible_count, int(n_cand)

    @Slot(object)
    def _mc_scan_update_ui(self, payload):
        df, feasible_count, total = payload
        self.mc_scan_df = df.reset_index(drop=True)
        if self.mc_scan_df.empty:
            self._populate_mc_scan_table(pd.DataFrame(columns=[]), [])
            self._set_status("Scan completed. No candidate after filters.")
            QMessageBox.information(self, "Info", "沒有候選參數。請放寬條件。")
            return

        disp, cols = self._format_mc_scan_display(self.mc_scan_df)
        self._populate_mc_scan_table(disp, cols)
        self._set_status(f"Scan completed. Feasible {feasible_count}/{total}.")

    def save_mc_scan_csv(self):
        if self.mc_scan_df is None or self.mc_scan_df.empty:
            QMessageBox.information(self, "Info", "No scan results to save. Please run Scan first.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Save Scan Results as CSV", "", "CSV (*.csv)")
        if not fpath:
            return
        try:
            self.mc_scan_df.to_csv(fpath, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Done", f"Saved: {fpath}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Save Failed", f"{e}")

    def clear_mc_scan_results(self):
        self.mc_table.setRowCount(0)
        self.mc_table.setColumnCount(0)
        if hasattr(self, "mc_metrics_text") and self.mc_metrics_text:
            self.mc_metrics_text.clear()
        if hasattr(self, "figure_mc_scan") and self.figure_mc_scan:
            self.figure_mc_scan.clf()
            self.canvas_mc_scan.draw()
        self.mc_scan_df = None
        self._set_status("Scan results cleared.")

    def save_scan_csv(self):
        if self.scan_df is None or self.scan_df.empty:
            QMessageBox.information(self, "提示", "沒有可儲存的掃描結果，請先 Run Scan。")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "另存掃描結果為 CSV", "", "CSV (*.csv)")
        if not fpath:
            return
        try:
            out = self._format_hist_scan_csv(self.scan_df)
            out.to_csv(fpath, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "完成", f"已儲存：{fpath}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "儲存失敗", f"{e}")

    def clear_scan_results(self):
        self.table.setRowCount(0)
        if self.scan_metrics_text:
            self.scan_metrics_text.clear()
        if self.figure_scan:
            self.figure_scan.clf()
            self.canvas_scan.draw()
        self.scan_df = None
        self._set_status("Scan results cleared.")

    def plot_selected_from_table(self, _row=None, _col=None):
        if self.scan_df is None or self.scan_df.empty:
            QMessageBox.information(self, "提示", "請先 Run Scan，並在表格選擇一列。")
            return
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "提示", "請在表格中選擇一列。")
            return
        idx = sel[0].row()
        if idx >= len(self.scan_df):
            QMessageBox.critical(self, "錯誤", "選擇索引超出範圍。")
            return
        params = self.scan_df.iloc[idx]
        self._set_status("載入詳細回測…")

        scan_days = safe_float(self.e_scan_days.text().strip(), 0.0)
        interval_str = self.e_interval.currentText().strip()
        bars = 0
        if scan_days > 0 and interval_str:
            step_ms = martin._interval_ms(interval_str)
            bars = int(scan_days * 86400 * 1000 / step_ms)

        worker = Worker(
            self._compute_backtest,
            self.e_symbol.currentText().strip(),
            interval_str,
            bars,
            self.e_start.text().strip(),
            self.e_end.text().strip(),
            (self.e_refresh.text().strip() or "auto"),
            safe_float(self.e_fee.text().strip(), 0.0),
            safe_float(self.e_capital.text().strip(), 1000.0),
            float(params["add_drop"]),
            float(params["multiplier"]),
            int(params["max_orders"]),
            float(params["tp"]),
        )
        worker.signals.finished.connect(self._render_scan_detail)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def plot_selected_from_mc_table(self, _row=None, _col=None):
        if self.mc_scan_df is None or self.mc_scan_df.empty:
            QMessageBox.information(self, "提示", "請先 Run Scan，並在表格選擇一列。")
            return
        sel = self.mc_table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "提示", "請在表格中選擇一列。")
            return
        idx = sel[0].row()
        if idx >= len(self.mc_scan_df):
            QMessageBox.critical(self, "錯誤", "選擇索引超出範圍。")
            return
        params = self.mc_scan_df.iloc[idx]
        self._set_status("載入候選參數詳細回測…")

        scan_days = safe_float(self.m_scan_days.text().strip(), 0.0)
        interval_str = self.m_interval.currentText().strip()
        bars = 0
        if scan_days > 0 and interval_str:
            step_ms = martin._interval_ms(interval_str)
            bars = int(scan_days * 86400 * 1000 / step_ms)

        worker = Worker(
            self._compute_backtest,
            self.m_symbol.currentText().strip(),
            interval_str,
            bars,
            self.m_start.text().strip(),
            self.m_end.text().strip(),
            (self.m_refresh.text().strip() or "auto"),
            safe_float(self.m_fee.text().strip(), 0.0),
            safe_float(self.m_capital.text().strip(), 1000.0),
            float(params["add_drop"]),
            float(params["multiplier"]),
            int(params["max_orders"]),
            float(params["tp"]),
        )
        worker.signals.finished.connect(self._render_mc_scan_detail)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    @Slot(object)
    def _render_scan_detail(self, payload):
        df, res, perf = payload
        self._render_plot_and_metrics(
            df, res, perf,
            float(res["_add_drop"]), float(res["_multiplier"]), int(res["_max_orders"]), float(res["_tp"]),
            self.figure_scan, self.canvas_scan, self.scan_metrics_text
        )
        self.scan_tabs.setCurrentWidget(self.scan_tab_detail)
        self._set_splitter_when_ready(self.scan_splitter, self.INIT_SPLIT)
        self._set_status("詳細回測完成。")

    @Slot(object)
    def _render_mc_scan_detail(self, payload):
        df, res, perf = payload
        self._render_plot_and_metrics(
            df, res, perf,
            float(res["_add_drop"]), float(res["_multiplier"]), int(res["_max_orders"]), float(res["_tp"]),
            self.figure_mc_scan, self.canvas_mc_scan, self.mc_metrics_text
        )
        self.mc_tabs.setCurrentWidget(self.mc_tab_detail)
        self._set_splitter_when_ready(self.mc_splitter, self.INIT_SPLIT)
        self._set_status("候選參數詳細回測完成。")

    # ---------- single ----------
    def run_single(self):
        self._set_status("Single Backtest中…")
        worker = Worker(self._run_single_compute)
        worker.signals.finished.connect(self._run_single_update)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _run_single_compute(self):
        symbol = self.s_symbol.currentText().strip()
        interval = self.s_interval.currentText().strip()
        scan_days = safe_float(self.s_scan_days.text().strip(), 0.0)
        bars = 0
        if scan_days > 0:
            step_ms = martin._interval_ms(interval)
            bars = int(scan_days * 86400 * 1000 / step_ms)
        start = self.s_start.text().strip()
        end = self.s_end.text().strip()
        refresh = (self.s_refresh.text().strip().lower() or "auto")
        fee_rate = safe_float(self.s_fee.text().strip(), 0.0)
        capital = safe_float(self.s_capital.text().strip(), 1000.0)

        add_drop = safe_float(self.s_add_drop.text().strip(), None)
        tp = safe_float(self.s_tp.text().strip(), None)
        multiplier = safe_float(self.s_multiplier.text().strip(), None)
        max_orders = safe_int(self.s_max_orders.text().strip(), None)

        if None in (add_drop, tp, multiplier, max_orders):
            raise ValueError("請完整填入策略參數（add_drop, tp, multiplier, max_orders）")
        if not symbol or not interval:
            raise ValueError("請填入 symbol 與 interval")
        if (not bars) and (not start or not end):
            raise ValueError("Scan Days 與 Start/End 需擇一填寫")

        df, res, perf = self._compute_backtest(symbol, interval, bars, start, end, refresh, fee_rate, capital,
                                               add_drop, multiplier, max_orders, tp)
        return df, res, perf

    @Slot(object)
    def _run_single_update(self, payload):
        df, res, perf = payload
        self._render_plot_and_metrics(
            df, res, perf,
            float(res["_add_drop"]), float(res["_multiplier"]), int(res["_max_orders"]), float(res["_tp"]),
            self.figure_single, self.canvas_single, self.metrics_text
        )
        self._set_status("回測完成。")
        self._set_splitter_when_ready(self.single_splitter, self.INIT_SPLIT)

    def run_single_mc(self):
        self._set_status("Monte Carlo模擬中…")
        worker = Worker(self._run_single_mc_compute)
        worker.signals.finished.connect(self._run_single_mc_update)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def clear_single_results(self):
        if self.metrics_text:
            self.metrics_text.clear()
        if self.figure_single:
            self.figure_single.clf()
            self.canvas_single.draw()
        self._set_status("Single backtest results cleared.")

    def _interval_bars_per_year(self, interval: str) -> float:
        step_ms = martin._interval_ms(interval)
        return (365.0 * 24.0 * 3600.0 * 1000.0) / float(step_ms)

    def _days_to_bars(self, days: float, interval: str) -> int:
        step_ms = martin._interval_ms(interval)
        bars = int(round(float(days) * 86400.0 * 1000.0 / float(step_ms)))
        return max(2, bars)

    def _run_single_mc_compute(self):
        df, res, perf = self._run_single_compute()

        mc_paths = safe_int(self.s_mc_paths.text().strip(), None)
        mc_days = safe_float(self.s_mc_days.text().strip(), None)
        mc_block = safe_int(self.s_mc_block.text().strip(), None)
        mc_seed = safe_int(self.s_mc_seed.text().strip(), 42)
        if None in (mc_paths, mc_days, mc_block):
            raise ValueError("請完整填入 Monte Carlo 參數（paths/days/block size）")
        if mc_paths <= 0 or mc_days <= 0 or mc_block <= 0:
            raise ValueError("Monte Carlo 參數需滿足：paths>0、days>0、block size>0")

        prices_np = df["close"].to_numpy(dtype=np.float64)
        hist_rets = prices_np[1:] / prices_np[:-1] - 1.0
        interval = self.s_interval.currentText().strip()
        mc_bars = self._days_to_bars(mc_days, interval)
        rng = np.random.default_rng(int(mc_seed))

        add_drop = float(res["_add_drop"])
        multiplier = float(res["_multiplier"])
        max_orders = int(res["_max_orders"])
        tp = float(res["_tp"])
        capital = float(res["capital"])
        fee_rate = safe_float(self.s_fee.text().strip(), 0.0)
        bars_per_year = self._interval_bars_per_year(interval)
        years_per_path = mc_bars / bars_per_year if bars_per_year > 0 else np.nan

        start_price = float(prices_np[-1]) if len(prices_np) else 1.0
        terminal = np.empty(mc_paths, dtype=np.float64)
        mdd_pct = np.empty(mc_paths, dtype=np.float64)
        trades = np.empty(mc_paths, dtype=np.int32)
        trapped_ratio = np.empty(mc_paths, dtype=np.float64)

        for i in range(mc_paths):
            one_ret = bootstrap_path_from_returns(hist_rets, int(mc_bars), int(mc_block), rng)
            one_path_prices = start_price * np.cumprod(np.concatenate(([1.0], 1.0 + one_ret)))
            fe_i, mdd_i, tr_i, trap_i = martin._backtest_core(
                one_path_prices.astype(np.float64),
                add_drop,
                multiplier,
                max_orders,
                tp,
                capital,
                float(fee_rate),
            )
            terminal[i] = float(fe_i)
            mdd_pct[i] = float(mdd_i)
            trades[i] = int(tr_i)
            trapped_ratio[i] = float(trap_i)

        cagr = np.full(mc_paths, np.nan, dtype=np.float64)
        if years_per_path and years_per_path > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                cagr = np.power(np.maximum(terminal, 1e-12) / capital, 1.0 / years_per_path) - 1.0

        summary_lines = [
            "=== Monte Carlo ===",
            (
                f"Paths: {mc_paths} | Days/path: {mc_days:.2f} | Bars/path: {mc_bars} | Block: {mc_block} | Seed: {mc_seed} | "
                f"Horizon: {years_per_path:.2f} years/path"
            ),
            (
                f"Terminal Equity Mean/Median: {np.mean(terminal):.2f}/{np.median(terminal):.2f} | "
                f"P5/P1: {np.percentile(terminal, 5):.2f}/{np.percentile(terminal, 1):.2f}"
            ),
            (
                f"Loss Prob (terminal < capital): {human_pct(np.mean(terminal < capital), 2)} | "
                f"Severe Loss (terminal < 50% capital): {human_pct(np.mean(terminal < 0.5 * capital), 2)}"
            ),
            (
                f"Max DD Mean/P95/P99: {np.mean(mdd_pct):.2f}%/{np.percentile(mdd_pct, 95):.2f}%/"
                f"{np.percentile(mdd_pct, 99):.2f}%"
            ),
            (
                f"P(MaxDD > 30%): {human_pct(np.mean(mdd_pct > 30.0), 2)} | "
                f"P(MaxDD > 50%): {human_pct(np.mean(mdd_pct > 50.0), 2)}"
            ),
            (
                f"Trades Mean/Median: {np.mean(trades):.1f}/{np.median(trades):.1f} | "
                f"Trapped Ratio Mean/P95: {human_pct(np.mean(trapped_ratio), 2)}/"
                f"{human_pct(np.percentile(trapped_ratio, 95), 2)}"
            ),
            (
                f"CAGR Mean/P5: {human_pct(np.nanmean(cagr), 2)}/"
                f"{human_pct(np.nanpercentile(cagr, 5), 2)}"
            ),
        ]

        mc = {
            "terminal": terminal,
            "summary_text": "\n".join(summary_lines),
        }
        return df, res, perf, mc

    @Slot(object)
    def _run_single_mc_update(self, payload):
        df, res, perf, mc = payload
        self._render_plot_and_metrics(
            df, res, perf,
            float(res["_add_drop"]), float(res["_multiplier"]), int(res["_max_orders"]), float(res["_tp"]),
            self.figure_single, self.canvas_single, self.metrics_text,
            mc_terminal=mc.get("terminal"),
            mc_summary_text=mc.get("summary_text"),
        )
        self._set_status("Monte Carlo 模擬完成。")
        self._set_splitter_when_ready(self.single_splitter, self.INIT_SPLIT)

    # ---------- compute ----------
    def _compute_backtest(self, symbol, interval, bars, start, end, refresh_policy,
                          fee_rate, capital, add_drop, multiplier, max_orders, tp):
        df = self._fetch_klines_if_needed(symbol, interval, bars, start, end, refresh_policy)
        prices_np = df["close"].to_numpy(dtype=np.float64)
        if prices_np.size < 2:
            raise ValueError("K 線資料不足（<2 根）。")

        res = martin.martingale_backtest(
            prices_np, add_drop=float(add_drop), multiplier=float(multiplier),
            max_orders=int(max_orders), tp=float(tp), capital=float(capital),
            return_curve=True, times=df["time"].tolist(), fee_rate=float(fee_rate),
        )

        first_price = float(prices_np[0])
        bh_qty = float(capital) / first_price
        bh_curve = bh_qty * prices_np

        perf = martin.compute_performance_metrics(
            res["equity_curve"], res["time_index"], res["trades_log"],
            capital=float(capital), bh_curve=bh_curve
        )

        res["bh_curve"] = bh_curve
        res["capital"] = float(capital)
        res["_add_drop"] = float(add_drop)
        res["_multiplier"] = float(multiplier)
        res["_max_orders"] = int(max_orders)
        res["_tp"] = float(tp)
        return df, res, perf

    def _render_plot_and_metrics(self, df, res, perf, add_drop, multiplier, max_orders, tp,
                                 figure, canvas, metrics_widget,
                                 mc_terminal=None, mc_summary_text=None):
        equity_curve = res["equity_curve"]
        time_index = res["time_index"]
        trapped_intervals = res.get("trapped_intervals", [])
        bh_curve = res.get("bh_curve", None)

        figure.clf()
        use_constrained = False
        try:
            figure.set_layout_engine("constrained")
            use_constrained = True
        except Exception:
            use_constrained = False

        if mc_terminal is not None:
            gs = figure.add_gridspec(2, 1, height_ratios=[2.5, 1.2])
            ax = figure.add_subplot(gs[0, 0])
            ax_mc = figure.add_subplot(gs[1, 0])
        else:
            ax = figure.add_subplot(111)
            ax_mc = None

        ax.plot(time_index, equity_curve, label="Strategy")
        if bh_curve is not None:
            ax.plot(df["time"], bh_curve, label="Buy & Hold")
        for s_i, e_i in trapped_intervals:
            if s_i is None or e_i is None:
                continue
            ax.axvspan(s_i, e_i, alpha=0.1, color="red")
        sym = df.attrs.get("symbol", "UNKNOWN")
        market_code = df.attrs.get("market", "spot")
        market_label = "Spot" if str(market_code).startswith("spot") else ("USDT Perp" if "usdt_perp" in str(market_code) else str(market_code))
        interval_str = df.attrs.get("interval", "N/A")
        ax.set_title(
            f"{sym} | {market_label} | exch={df.attrs.get('exchange','?')} | interval={interval_str}\n"
            f"[{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}] "
            f"add_drop={add_drop:.3f}, tp={tp:.3f}, mul={multiplier:.1f}, max_orders={int(max_orders)}",
            fontsize=14,
            pad=12,
        )
        ax.set_xlabel("" if ax_mc is not None else "Time (Taipei)")
        ax.set_ylabel("Equity (USDT)")
        ax.legend(loc="upper left")

        if ax_mc is not None:
            safe_terminal = np.maximum(np.asarray(mc_terminal, dtype=np.float64), 0.0)
            ax_mc.hist(safe_terminal, bins=45, color="#35507a", alpha=0.75)
            ax_mc.axvline(np.median(safe_terminal), color="#8a2d3a", linestyle="--", linewidth=1.5, label="Median")
            ax_mc.set_title("Monte Carlo terminal equity distribution", fontsize=12, pad=10)
            ax_mc.set_xlabel("Terminal Equity (USDT)")
            ax_mc.set_ylabel("Count")
            ax_mc.legend(loc="best")
        if not use_constrained:
            if ax_mc is not None:
                figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10, hspace=0.38)
            else:
                figure.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12)
        canvas.draw()

        trap_ratio = None
        total_secs = None
        trapped_secs = 0.0
        try:
            if time_index is not None and len(time_index) >= 2:
                start_t = time_index[0]
                end_t = time_index[-1]
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

        if metrics_widget is not None and isinstance(perf, dict):
            t = []
            t.append("=== Performance (Strategy) ===")
            t.append(
                f"Total Return: {human_pct(perf.get('total_return'))} | CAGR: {human_pct(perf.get('cagr'))} | Ann Vol: {human_pct(perf.get('ann_vol'))}"
            )
            sharpe = perf.get('sharpe')
            sortino = perf.get('sortino')
            calmar = perf.get('calmar')
            t.append(
                f"Sharpe: {('%.2f' % sharpe) if sharpe is not None else 'NaN'} | "
                f"Sortino: {('%.2f' % sortino) if sortino is not None else 'NaN'} | "
                f"Calmar: {('%.2f' % calmar) if calmar is not None else 'NaN'}"
            )
            t.append(
                f"Max DD: {perf.get('max_dd_pct', float('nan')):.2f}% | "
                f"DD Duration (days): {perf.get('max_dd_days','NaN')} | Recovery: {perf.get('recovery_days','NaN')}"
            )
            t.append(
                f"Underwater (avg/max): {perf.get('avg_underwater_days', float('nan')):.1f}/"
                f"{perf.get('max_underwater_days', float('nan')):.1f} days"
            )
            pf = perf.get('profit_factor')
            pf_str = "∞" if (isinstance(pf, (int, float)) and np.isinf(pf)) else (
                "NaN" if pf is None or (isinstance(pf, float) and np.isnan(pf)) else f"{pf:.2f}"
            )
            t.append(f"Win Rate: {human_pct(perf.get('win_rate'))} | Profit Factor: {pf_str}")
            if trap_ratio is not None and total_secs is not None:
                trapped_days = trapped_secs / 86400.0
                total_days = total_secs / 86400.0
                t.append(
                    f"Trapped Time Ratio: {human_pct(trap_ratio, 2)} | Trapped/Total: {trapped_days:.1f}/{total_days:.1f} days"
                )
            t.append(f"Avg Win: {perf.get('avg_win', float('nan')):.2f} | Avg Loss: {perf.get('avg_loss', float('nan')):.2f}")
            t.append(
                f"Max Consec Wins: {perf.get('max_consec_wins','NaN')} | "
                f"Max Consec Losses: {perf.get('max_consec_losses','NaN')}"
            )
            t.append(
                f"Avg Trade Return: {human_pct(perf.get('avg_trade_return'))} | "
                f"Median Trade Return: {human_pct(perf.get('median_trade_return'))}"
            )
            if "bh_total_return" in perf:
                t.append("\n=== Buy & Hold (Benchmark) ===")
                t.append(
                    f"Total: {human_pct(perf.get('bh_total_return'))} | "
                    f"CAGR: {human_pct(perf.get('bh_cagr'))} | Ann Vol: {human_pct(perf.get('bh_ann_vol'))}"
                )
                t.append(
                    f"Sharpe: {('%.2f' % perf.get('bh_sharpe')) if perf.get('bh_sharpe') is not None else 'NaN'} | "
                    f"Max DD: {perf.get('bh_max_dd_pct', float('nan')):.2f}%"
                )
            if mc_summary_text:
                t.append("")
                t.append(mc_summary_text)
            metrics_widget.setPlainText("\n".join(t))

    @Slot(str)
    def _show_error(self, tb):
        self._set_status("操作失敗。")
        QMessageBox.critical(self, "錯誤", tb)


if __name__ == "__main__":
    app = QApplication([])
    if _import_error is not None:
        QMessageBox.critical(None, "Import Error", f"無法匯入 martin.py：\n{_import_error}")
        raise SystemExit(1)
    w = MartinGUI()
    w.show()
    app.exec()
