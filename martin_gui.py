# martin_gui.py — PySide6 GUI
# Adds: Trapped Time Ratio in metrics log — 2025-10-11
# -*- coding: utf-8 -*-

import traceback
import os
import threading
import time
import math
from collections import OrderedDict

import numpy as np
import pandas as pd

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import Qt, QDate, QAbstractTableModel, QModelIndex, QThreadPool, QRunnable, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QComboBox, QPushButton, QLabel, QSplitter, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QStatusBar,
    QAbstractItemView,
    QHeaderView,
    QDateEdit,
    QTableView,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
from matplotlib.figure import Figure

try:
    import martin
    _import_error = None
except Exception as e:
    martin = None
    _import_error = e

from market_data import alpaca, universe

from mc_sampling import sample_parameter_grid, refine_neighbors
from mc_eval import (
    eval_candidates_parallel,
    bootstrap_ohlc_path_from_ratios,
    ohlc_ratios_from_history,
)
from mc_formatters import format_hist_scan_display, format_hist_scan_csv, format_mc_scan_display
from diy_strategy import (
    DEFAULT_MAX_MAE_GRID_BYTES,
    build_diy_templates,
    compute_tp_or_horizon_mae_grid,
    estimate_tp_or_horizon_mae_grid_bytes,
)
CRYPTO_SYMBOLS = [
    "ASTER", "BONK", "ENA", "PEPE", "WLD", "ZEC", "TAO", "SUI", "HBAR",
    "UNI", "NEAR", "FIL", "APT", "ARB",
    "DOGE", "SHIB", "ADA", "AVAX", "LINK", "XRP", "SOL", "LTC", "ETH",
    "BNB", "TRX", "BCH", "BTC",
]
DEFAULT_SYMBOLS = CRYPTO_SYMBOLS
ALPACA_DEFAULT_SYMBOLS = list(universe.ALPACA_MAINSTREAM_STOCK_SYMBOLS[:50])
CRYPTO_INTERVALS = ["15m", "1h", "4h", "1d"]
DATA_SOURCES = ["binance", "alpaca"]
DEFAULT_REFRESH_POLICY = "auto"
DEFAULT_FEE_RATE = 0.001
DEFAULT_FEE_LABEL = "0.1%"
ALPACA_TRADING_DAYS_PER_YEAR = 252.0
ALPACA_BARS_PER_TRADING_DAY = dict(alpaca.DEFAULT_BARS_PER_TRADING_DAY)

# DIY Mode deliberately exposes only the parameters that describe the trading
# target.  The MAE-ladder search space stays fixed here so scans are simple and
# reproducible; TP/Horizon MAE capital reweighting is disabled (bias = 0).
DIY_MAE_Q_START = 0.50
DIY_MAE_Q_END_VALUES = (0.90, 0.92, 0.94, 0.96, 0.98)
DIY_INITIAL_FRACTION_VALUES = (0.01, 0.02, 0.03, 0.04, 0.05)
DIY_CAPITAL_GAMMA_VALUES = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
DIY_IDLE_BIAS_VALUES = (0.0,)
DIY_TOTAL_SHARES = 100
DIY_STRESS_MAE_Q = 0.99
DIY_MAX_LAST_ORDER_PCT = 30.0
DIY_MAX_STRESS_LOSS_PCT = 50.0
DIY_MAX_CANDIDATES = 2_000_000
DIY_MAX_MAE_GRID_BYTES = DEFAULT_MAX_MAE_GRID_BYTES
DIY_TEMPLATE_AXES_PER_MAX_ORDER = (
    len(DIY_MAE_Q_END_VALUES)
    * len(DIY_INITIAL_FRACTION_VALUES)
    * len(DIY_CAPITAL_GAMMA_VALUES)
    * len(DIY_IDLE_BIAS_VALUES)
)
DIY_MAX_TP_VALUES = max(
    1,
    DIY_MAX_CANDIDATES // DIY_TEMPLATE_AXES_PER_MAX_ORDER,
)


def parse_range(s: str, is_int=False, max_items=None):
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
    if not is_int and not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError("範圍 start/end/step 必須為有限數值")
    if c == 0:
        raise ValueError("step 不可為 0")
    if (b - a) * c < 0:
        raise ValueError("step 方向必須能由 start 走到 end")
    if is_int:
        count = (
            (b - a) // c + 1
            if c > 0
            else (a - b) // (-c) + 1
        )
    else:
        span = (b - a) / c
        count = int(math.floor(span + 1e-12)) + 1
    hard_limit = 2_000_000 if max_items is None else min(
        2_000_000,
        int(max_items),
    )
    if hard_limit <= 0:
        raise ValueError("max_items 必須 > 0")
    if count > hard_limit:
        raise ValueError(
            f"單一參數範圍共有 {count:,} 個值，超過 {hard_limit:,} 個限制"
        )
    if is_int:
        return a + np.arange(count, dtype=int) * c
    return a + np.arange(count, dtype=np.float64) * c


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


def validate_strategy_param_arrays(add_drop, tp, multiplier, max_orders):
    add_drop = np.asarray(add_drop, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    max_orders = np.asarray(max_orders, dtype=np.int64)
    if (
        np.any(~np.isfinite(add_drop))
        or np.any(~np.isfinite(tp))
        or np.any(~np.isfinite(multiplier))
        or np.any(add_drop <= 0.0)
        or np.any(add_drop >= 1.0)
        or np.any(tp <= 0.0)
        or np.any(multiplier <= 0.0)
        or np.any(max_orders < 1)
    ):
        raise ValueError("策略參數需滿足：0 < add_drop < 1、tp > 0、multiplier > 0、max_orders >= 1")
    for mul in np.unique(multiplier):
        for mo in np.unique(max_orders):
            martin._order_factor_sum(float(mul), int(mo))


def human_pct(x, digits=2):
    if x is None or pd.isna(x):
        return "NaN"
    if np.isinf(float(x)):
        return "∞"
    return f"{x * 100:.{digits}f}%"


def configure_datetime_axis(ax, time_values, timezone_name="Asia/Taipei"):
    """Install an explicit date converter and readable formatter on an axis."""
    time_index = pd.DatetimeIndex(time_values)
    if time_index.empty:
        return None
    if time_index.tz is None:
        time_index = time_index.tz_localize(timezone_name)
    else:
        time_index = time_index.tz_convert(timezone_name)
    locator = mdates.AutoDateLocator(
        tz=time_index.tz,
        minticks=4,
        maxticks=9,
        interval_multiples=True,
    )
    formatter = mdates.ConciseDateFormatter(locator, tz=time_index.tz)
    ax.xaxis_date(tz=time_index.tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=20)
    return formatter


def backtest_accounting_summary(result, capital):
    """Reconcile final equity against realized and open-position PnL."""
    capital = float(capital)
    closed_pnl = float(sum(float(trade.get("pnl", 0.0)) for trade in result.get("trades_log", [])))
    open_trade = result.get("open_trade")
    open_pnl = (
        float(open_trade.get("pnl", 0.0))
        if isinstance(open_trade, dict) else 0.0
    )
    equity_curve = np.asarray(result.get("equity_curve", []), dtype=np.float64)
    if equity_curve.size == 0 or not np.isfinite(equity_curve[-1]):
        raise ValueError("Backtest equity curve has no finite terminal value")
    final_equity = float(equity_curve[-1])
    reconciled_equity = capital + closed_pnl + open_pnl
    tolerance = max(0.01, abs(final_equity) * 1e-9)
    if not np.isclose(final_equity, reconciled_equity, rtol=1e-9, atol=tolerance):
        raise ValueError(
            "Backtest accounting mismatch: "
            f"curve={final_equity:.8f}, reconciled={reconciled_equity:.8f}"
        )
    return {
        "closed_pnl": closed_pnl,
        "open_pnl": open_pnl,
        "net_pnl": final_equity - capital,
        "final_equity": final_equity,
    }



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
        except Exception:
            try:
                self.signals.error.emit(traceback.format_exc())
            except RuntimeError:
                pass  # Window/signals may have been deleted during shutdown.
        else:
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass  # Window/signals may have been deleted during shutdown.


class DataFrameTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._headers = []

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def set_frame(self, frame: pd.DataFrame, headers: list[str]):
        self.beginResetModel()
        self._headers = list(headers)
        if frame is None or frame.empty or not self._headers:
            self._frame = pd.DataFrame(columns=self._headers)
        else:
            self._frame = frame.loc[:, self._headers].reset_index(drop=True).copy()
        self.endResetModel()

    def clear(self):
        self.set_frame(pd.DataFrame(), [])

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._frame)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return "" if pd.isna(value) else str(value)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
            return None
        return str(section + 1)

    def sort(self, column, order):
        if not self._headers or self._frame.empty:
            return
        
        self.layoutAboutToBeChanged.emit()
        col_name = self._headers[column]
        ascending = order == Qt.AscendingOrder
        
        # Create a temporary sorting series by cleaning strings
        series = self._frame[col_name].copy()
        if series.dtype == object or series.dtype == str:
            # Clean common formatted strings (e.g., '10.5%', 'N/A', 'Y', 'N', commas)
            cleaned = series.astype(str).str.replace(r'[≥%$ ,]', '', regex=True)
            # Map 'Y' to 1 and 'N' to 0 for feasible columns
            cleaned = cleaned.replace({'Y': '1', 'N': '0', 'N/A': '-inf'})
            numeric_series = pd.to_numeric(cleaned, errors='coerce')
            
            # If at least some values were successfully converted to numbers
            if not numeric_series.isna().all():
                series = numeric_series
                
        # Sort frame using the temporary series
        sorted_indices = series.sort_values(ascending=ascending, kind='mergesort', na_position='first' if ascending else 'last').index
        self._frame = self._frame.loc[sorted_indices]
        self.layoutChanged.emit()


class MartinGUI(QMainWindow):
    INIT_SPLIT = 0.70
    BACKTEST_CACHE_LIMIT = 24
    DATA_CACHE_LIMIT = 8

    def __init__(self):
        super().__init__()
        self.setWindowTitle("martin_gui")
        self.resize(1280, 920)

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
        self._cache_lock = threading.RLock()
        self._data_cache = OrderedDict()
        self.scan_df = None
        self.mc_scan_df = None
        self._scan_run_config = None
        self._mc_scan_run_config = None
        self._mc_executor = None
        self._mc_executor_workers = 0
        self._backtest_cache = OrderedDict()
        self._plot_states = {}
        self._build_scan_tab()
        self._build_mc_scan_tab()
        self._build_single_tab()
        self._configure_symbol_inputs()

        self.nb.currentChanged.connect(self._on_main_tab_changed)

        self._apply_style()
        QTimer.singleShot(0, self._fit_initial_window_to_screen)
        QTimer.singleShot(0, self._autosize_scan_table_columns)
        QTimer.singleShot(0, self._autosize_mc_scan_table_columns)

        QTimer.singleShot(0, lambda: self._set_splitter_when_ready(self.scan_splitter, self.INIT_SPLIT))
        QTimer.singleShot(0, lambda: self._set_splitter_when_ready(self.single_splitter, self.INIT_SPLIT))
        QTimer.singleShot(0, self._restore_startup_focus)

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
            QLabel#feeHintLabel { color: #d6d6d6; font-size: 10.5pt; padding: 0 6px; }
            QLineEdit, QComboBox, QTextEdit, QDateEdit {
                background: #ffffff; color: #1d1d1d; border: 1px solid #b9b9b9; border-radius: 6px; padding: 4px 6px;
                selection-background-color: #2d6a7a; selection-color: #ffffff;
            }
            QDateEdit:disabled { background: #e8e8e8; color: #555555; }
            QDateEdit::drop-down { background: #f2f2f2; border-left: 1px solid #c5c5c5; width: 24px; }
            QComboBox QAbstractItemView, QDateEdit QAbstractItemView { background: #ffffff; color: #1d1d1d; }
            QCalendarWidget QWidget { background: #ffffff; color: #1d1d1d; }
            QCalendarWidget QAbstractItemView:enabled {
                background: #ffffff; color: #1d1d1d;
                selection-background-color: #2d6a7a; selection-color: #ffffff;
            }
            QTableWidget, QTableView { background: #ffffff; color: #1d1d1d; border: 1px solid #7c7c7c; border-radius: 8px; gridline-color: #d0d0d0; }
            QTableWidget::item:selected, QTableView::item:selected { font-weight: 400; }
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

    def _restore_startup_focus(self):
        for date_edit in (self.e_start, self.e_end, self.m_start, self.m_end, self.s_start, self.s_end):
            line_edit = date_edit.lineEdit()
            if line_edit is not None:
                line_edit.deselect()
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        self.nb.setFocus(Qt.OtherFocusReason)

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

    def _build_fee_hint_label(self):
        value = QLabel(f"Fee {DEFAULT_FEE_LABEL}")
        value.setObjectName("feeHintLabel")
        return value

    def _default_date_range(self):
        end_date = QDate.currentDate()
        start_date = end_date.addYears(-2)
        return start_date, end_date

    def _fit_initial_window_to_screen(self):
        """Use available WSL display space while keeping forms readable."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        min_width = min(1180, max(800, available.width() - 40))
        min_height = min(720, max(580, available.height() - 60))
        target_width = min(1550, max(min_width, int(available.width() * 0.92)))
        target_height = min(980, max(min_height, int(available.height() * 0.92)))
        self.setMinimumSize(min_width, min_height)
        self.resize(target_width, target_height)

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

    def _add_date_edit(self, form: QFormLayout, label: str, default_date: QDate, width=140):
        e = QDateEdit()
        e.setCalendarPopup(True)
        e.setDisplayFormat("yyyy/MM/dd")
        e.setMaximumDate(QDate.currentDate())
        e.setDate(default_date)
        date_text_width = e.fontMetrics().horizontalAdvance("0000/00/00")
        e.setMinimumWidth(max(width, date_text_width + 52))
        self._apply_date_edit_palette(e)
        self._add_form_row(form, label, e)
        return e

    def _get_date_range_strings(self, start_edit: QDateEdit, end_edit: QDateEdit):
        start_date = start_edit.date()
        end_date = end_edit.date()
        if start_date > end_date:
            raise ValueError("Start 不可晚於 End")
        return (
            start_date.toString("yyyy-MM-dd"),
            end_date.toString("yyyy-MM-dd"),
        )

    def _add_combobox(self, form: QFormLayout, label: str, values, default=""):
        cb = QComboBox()
        cb.addItems(list(values))
        if default:
            idx = cb.findText(default)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        self._add_form_row(form, label, cb)
        return cb

    def _configure_symbol_inputs(self):
        pairs = (
            (self.e_source, self.e_symbol),
            (self.m_source, self.m_symbol),
            (self.s_source, self.s_symbol),
        )
        for source_combo, symbol_combo in pairs:
            symbol_combo.setMaxVisibleItems(12)
            symbol_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            symbol_combo.view().setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
            source_combo.currentTextChanged.connect(
                lambda source, combo=symbol_combo: self._update_symbol_input(combo, source)
            )
            self._update_symbol_input(symbol_combo, source_combo.currentText())

    def _update_symbol_input(self, symbol_combo: QComboBox, source: str):
        source_name = str(source or "").strip().lower()
        if source_name == "alpaca":
            symbols = ALPACA_DEFAULT_SYMBOLS
            default = "AAPL"
            tooltip = (
                "Alpaca US stock/ETF ticker，例如 AAPL、NVDA、STRC。"
                "下拉清單預設提供 50 個主流標的，也可手動輸入其他有效 ticker。"
            )
        else:
            symbols = CRYPTO_SYMBOLS
            default = "XRP"
            tooltip = (
                "Binance crypto spot base symbol，例如 BTC、ETH、XRP。"
                "可從清單選擇，亦可手動輸入有效的 USDT 現貨幣種。"
            )

        previous = symbol_combo.blockSignals(True)
        try:
            symbol_combo.clear()
            symbol_combo.addItems(symbols)
            symbol_combo.setCurrentText(default)
            symbol_combo.setToolTip(tooltip)
        finally:
            symbol_combo.blockSignals(previous)

        if source_name == "alpaca":
            try:
                alpaca.load_env_file()
                alpaca.credentials_from_env()
            except (OSError, ValueError) as exc:
                self._set_status(f"Alpaca 尚未就緒：{str(exc).splitlines()[0]}")
            else:
                self._set_status("Alpaca 已就緒；Symbol 清單載入 50 個美股／ETF。")
        else:
            self._set_status("Binance crypto spot symbols loaded.")

    @staticmethod
    def _source_date_range(source: str, start: str, end: str):
        """Use full New York trading dates for Alpaca stock requests."""
        if str(source or "").strip().lower() != "alpaca":
            return start, end
        return alpaca.full_new_york_date_range(start, end)

    # ---------- scan tab ----------
    def _build_scan_tab(self):
        layout = QVBoxLayout(self.tab_hist_scan)
        default_start, default_end = self._default_date_range()

        top = QHBoxLayout()
        layout.addLayout(top)

        gb_data = QGroupBox("Data Settings")
        gb_grid = QGroupBox("Scan Parameters ('start:end:step')")
        gb_filter = QGroupBox("Filter Conditions (optional)")

        top.addWidget(gb_data)
        top.addWidget(gb_grid, 1)
        top.addWidget(gb_filter)

        form_data = QFormLayout(gb_data)
        self.e_source = self._add_combobox(form_data, "Source:", DATA_SOURCES, default="binance")
        self.e_symbol = self._add_combobox(form_data, "Symbol:", DEFAULT_SYMBOLS, default="XRP")
        self.e_symbol.setEditable(True)
        self.e_interval = self._add_combobox(form_data, "Interval:", CRYPTO_INTERVALS, default="15m")
        self.e_start = self._add_date_edit(form_data, "Start:", default_start)
        self.e_end = self._add_date_edit(form_data, "End:", default_end)
        self.e_capital = self._add_entry(form_data, "Capital:", "1000")

        form_grid = QFormLayout(gb_grid)
        self.e_scan_mode = self._add_combobox(
            form_grid,
            "Mode:",
            [
                "Fixed Mode",
                "DIY Mode (TP/Horizon MAE)",
            ],
            default="Fixed Mode",
        )
        self.e_add_drop = self._add_entry(form_grid, "add_drop:", "0.010:0.080:0.001")
        self.e_tp = self._add_entry(form_grid, "tp:", "0.010:0.080:0.001")
        self.e_multiplier = self._add_entry(form_grid, "multiplier:", "1.5:2.0:0.1")
        self.e_max_orders = self._add_entry(form_grid, "max_orders:", "5:12:1")
        self.e_mae_horizon = self._add_entry(form_grid, "MAE horizon(days):", "30")
        self._hist_grid_form = form_grid
        self._hist_diy_fields = (self.e_mae_horizon,)
        self.e_scan_mode.currentTextChanged.connect(self._on_hist_scan_mode_changed)
        self._on_hist_scan_mode_changed(self.e_scan_mode.currentText())

        form_filter = QFormLayout(gb_filter)
        self.e_min_trades = self._add_entry(form_filter, "min_trades:", "")
        self.e_max_dd = self._add_entry(form_filter, "max_dd_overall(%):", "")
        self.e_max_trap = self._add_entry(form_filter, "max_trapped_ratio(%):", "")
        self.e_topn = self._add_entry(form_filter, "Show Top N:", "20")

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_run = QPushButton("Run Scan")
        self.btn_hist_run = btn_run
        btn_run.setObjectName("btnPrimary")
        btn_csv = QPushButton("Save Result as CSV")
        btn_csv.setObjectName("btnInfo")
        btn_clear = QPushButton("Clear Results")
        btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(self._build_fee_hint_label())

        btn_run.clicked.connect(self.run_scan)
        btn_csv.clicked.connect(self.save_scan_csv)
        btn_clear.clicked.connect(self.clear_scan_results)

        self.scan_tabs = QTabWidget()
        layout.addWidget(self.scan_tabs, 1)

        # Table tab
        self.scan_tab_table = QWidget()
        table_layout = QVBoxLayout(self.scan_tab_table)
        self.table = QTableView()
        self.scan_table_model = DataFrameTableModel(self.table)
        self.table.setModel(self.scan_table_model)
        cols = [
            "Mode", "TP", "Orders", "Final Equity", "Return", "Max DD",
            "Trades", "Trapped", "Lowest Buy", "Capital Use", "Setup",
        ]
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.scan_table_model.set_frame(pd.DataFrame(columns=cols), cols)
        self.table.setWordWrap(False)
        self.table.setSortingEnabled(True)
        self._autosize_scan_table_columns()
        table_layout.addWidget(self.table)

        hint = QLabel("提示：雙擊表格繪圖，會自動切換到「Backtest Chart / Performance」。")
        hint.setStyleSheet("color: #6b6b6b; padding: 6px 2px;")
        table_layout.addWidget(hint)

        self.table.doubleClicked.connect(self.plot_selected_from_table)

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
        self._autosize_table_columns(self.table, max_col_width=max_col_width)

    # ---------- MC scan tab ----------
    def _build_mc_scan_tab(self):
        layout = QVBoxLayout(self.tab_mc_scan)
        default_start, default_end = self._default_date_range()

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
        self.m_source = self._add_combobox(form_data, "Source:", DATA_SOURCES, default="binance")
        self.m_symbol = self._add_combobox(form_data, "Symbol:", DEFAULT_SYMBOLS, default="XRP")
        self.m_symbol.setEditable(True)
        self.m_interval = self._add_combobox(form_data, "Interval:", CRYPTO_INTERVALS, default="15m")
        self.m_start = self._add_date_edit(form_data, "Start:", default_start)
        self.m_end = self._add_date_edit(form_data, "End:", default_end)
        self.m_capital = self._add_entry(form_data, "Capital:", "1000")

        form_grid = QFormLayout(gb_grid)
        self.m_add_drop = self._add_entry(form_grid, "add_drop:", "0.010:0.080:0.001")
        self.m_tp = self._add_entry(form_grid, "tp:", "0.010:0.080:0.001")
        self.m_multiplier = self._add_entry(form_grid, "multiplier:", "1.5:2.0:0.1")
        self.m_max_orders = self._add_entry(form_grid, "max_orders:", "5:12:1")
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
        self.m_mc_block = self._add_entry(form_mc, "Block size:", "672")
        self.m_mc_seed = self._add_entry(form_mc, "Seed:", "42")
        self.m_mc_seed_runs = self._add_entry(form_mc, "Seed runs:", "1")
        self.m_mc_holdout = self._add_entry(form_mc, "MC holdout(%):", "30")
        self.m_mc_workers = self._add_entry(form_mc, "Workers (0=auto):", "0")
        self.m_rank_by = self._add_combobox(
            form_mc,
            "Rank by:",
            ["Median terminal", "P5 terminal", "Lowest DD mean"],
            default="Median terminal",
        )

        form_risk = QFormLayout(gb_risk)
        self.m_max_loss = self._add_entry(form_risk, "Max P(loss)%:", "30")
        self.m_max_severe = self._add_entry(form_risk, "Max P(severe)%:", "10")
        self.m_max_dd50 = self._add_entry(form_risk, "Max P(DD>50)%:", "20")

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_run = QPushButton("Run Scan")
        self.btn_mc_run = btn_run
        btn_run.setObjectName("btnPrimary")
        btn_csv = QPushButton("Save Result as CSV")
        btn_csv.setObjectName("btnInfo")
        btn_clear = QPushButton("Clear Results")
        btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(self._build_fee_hint_label())

        btn_run.clicked.connect(self.run_mc_scan)
        btn_csv.clicked.connect(self.save_mc_scan_csv)
        btn_clear.clicked.connect(self.clear_mc_scan_results)

        self.mc_tabs = QTabWidget()
        layout.addWidget(self.mc_tabs, 1)

        self.mc_tab_table = QWidget()
        mc_table_layout = QVBoxLayout(self.mc_tab_table)
        self.mc_table = QTableView()
        self.mc_table_model = DataFrameTableModel(self.mc_table)
        self.mc_table.setModel(self.mc_table_model)
        self.mc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.mc_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.mc_table.horizontalHeader().setHighlightSections(False)
        self.mc_table.verticalHeader().setVisible(False)
        self.mc_table.setWordWrap(False)
        self.mc_table.setSortingEnabled(True)
        mc_table_layout.addWidget(self.mc_table)

        mc_hint = QLabel("提示：雙擊表格繪圖，會自動切換到「Backtest Chart / Performance」。")
        mc_hint.setStyleSheet("color: #6b6b6b; padding: 6px 2px;")
        mc_table_layout.addWidget(mc_hint)
        self.mc_table.doubleClicked.connect(self.plot_selected_from_mc_table)

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
        self._autosize_table_columns(self.mc_table, max_col_width=max_col_width)

    def _autosize_table_columns(self, table_view, max_col_width=280):
        model = table_view.model()
        if model is None:
            return
        header = table_view.horizontalHeader()
        metrics = QFontMetrics(header.font())
        pad_px = 24
        min_col_width = 80

        for c in range(model.columnCount()):
            text = model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or ""
            best = metrics.horizontalAdvance(text) + pad_px
            for r in range(model.rowCount()):
                idx = model.index(r, c)
                cell_text = model.data(idx, Qt.DisplayRole) or ""
                best = max(best, metrics.horizontalAdvance(cell_text) + pad_px)
            table_view.setColumnWidth(c, min(max_col_width, max(min_col_width, best)))

        header.setStretchLastSection(True)

    def _set_form_row_visible(self, form: QFormLayout, field: QWidget, visible: bool):
        lbl = form.labelForField(field)
        if lbl is not None:
            lbl.setVisible(bool(visible))
        field.setVisible(bool(visible))

    def _on_hist_scan_mode_changed(self, text: str):
        mode = str(text or "").strip().lower()
        is_diy = mode.startswith("diy")
        self._set_form_row_visible(self._hist_grid_form, self.e_add_drop, not is_diy)
        self._set_form_row_visible(self._hist_grid_form, self.e_multiplier, not is_diy)
        for field in self._hist_diy_fields:
            self._set_form_row_visible(self._hist_grid_form, field, is_diy)

    def _on_mc_sampling_mode_changed(self, text: str):
        mode = (text or "").strip().lower()
        is_full = mode.startswith("full grid")
        self._set_form_row_visible(self._mc_grid_form, self.m_sample_size, not is_full)
        self._set_form_row_visible(self._mc_grid_form, self.m_refine_pct, not is_full)
        self._set_form_row_visible(self._mc_grid_form, self.m_max_combos, is_full)

    # ---------- single tab ----------
    def _build_single_tab(self):
        layout = QVBoxLayout(self.tab_single)
        default_start, default_end = self._default_date_range()

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
        self.s_source = self._add_combobox(form_data, "Source:", DATA_SOURCES, default="binance")
        self.s_symbol = self._add_combobox(form_data, "Symbol:", DEFAULT_SYMBOLS, default="XRP")
        self.s_symbol.setEditable(True)
        self.s_interval = self._add_combobox(form_data, "Interval:", CRYPTO_INTERVALS, default="15m")
        self.s_start = self._add_date_edit(form_data, "Start:", default_start)
        self.s_end = self._add_date_edit(form_data, "End:", default_end)
        self.s_capital = self._add_entry(form_data, "Capital:", "1000")

        form_params = QFormLayout(gb_params)
        self.s_add_drop = self._add_entry(form_params, "add_drop:", "0.05")
        self.s_tp = self._add_entry(form_params, "tp:", "0.05")
        self.s_multiplier = self._add_entry(form_params, "multiplier:", "2.0")
        self.s_max_orders = self._add_entry(form_params, "max_orders:", "7")

        form_mc = QFormLayout(gb_mc)
        self.s_mc_paths = self._add_entry(form_mc, "MC paths:", "1000")
        self.s_mc_days = self._add_entry(form_mc, "Days/path:", "730")
        self.s_mc_block = self._add_entry(form_mc, "Block size:", "672")
        self.s_mc_seed = self._add_entry(form_mc, "Random seed:", "42")

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_run = QPushButton("Execute Single Backtest and Plot")
        self.btn_single_run = btn_run
        btn_run.setObjectName("btnPrimary")
        btn_mc = QPushButton("Run Monte Carlo")
        self.btn_single_mc = btn_mc
        btn_mc.setObjectName("btnInfo")
        btn_clear = QPushButton("Clear Results")
        btn_clear.setObjectName("btnDanger")
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_mc)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(self._build_fee_hint_label())

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
                self._mc_executor.shutdown(wait=False, cancel_futures=True)
                self._mc_executor = None
                self._mc_executor_workers = 0
        finally:
            super().closeEvent(event)

    # ---------- data ----------
    def _refresh_generation(self, interval, refresh_policy):
        if str(refresh_policy).lower() != "auto":
            return None
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        return martin._align_to_interval_end(now_ms, martin._interval_ms(interval))

    def _fetch_klines_if_needed(self, symbol, interval, start, end, refresh_policy, source="binance"):
        source = str(source or "binance").strip().lower()
        if source not in DATA_SOURCES:
            raise ValueError(f"不支援的資料來源：{source}")
        key = (
            symbol, interval, start, end, refresh_policy, source,
            self._refresh_generation(interval, refresh_policy),
        )
        with self._cache_lock:
            cached = self._data_cache.get(key)
            if cached is not None:
                self._data_cache.move_to_end(key)
                return cached
            # Serialize cache misses so two tabs cannot concurrently overwrite
            # the same parquet file or pair one request key with another frame.
            if source == "alpaca":
                alpaca.load_env_file()
                alpaca.credentials_from_env()
            request_start, request_end = self._source_date_range(source, start, end)
            df = martin.get_klines(
                symbol=symbol,
                interval=interval,
                start=request_start,
                end=request_end,
                cache_dir=martin.DEFAULT_CACHE_DIR,
                use_cache=True,
                refresh_policy=refresh_policy,
                exch_list=[source],
                allow_gaps=(source == "alpaca"),
                require_full_coverage=(source != "alpaca"),
            )
            self._data_cache[key] = df
            self._data_cache.move_to_end(key)
            while len(self._data_cache) > self.DATA_CACHE_LIMIT:
                self._data_cache.popitem(last=False)
        return df

    def _make_backtest_cache_key(
        self, symbol, interval, start, end, refresh_policy, fee_rate, capital,
        add_drop, multiplier, max_orders, tp, source="binance",
    ):
        return (
            symbol,
            interval,
            start,
            end,
            refresh_policy,
            float(fee_rate),
            float(capital),
            float(add_drop),
            float(multiplier),
            int(max_orders),
            float(tp),
            str(source or "binance").lower(),
            self._refresh_generation(interval, refresh_policy),
        )

    def _get_cached_backtest(self, key):
        with self._cache_lock:
            cached = self._backtest_cache.get(key)
            if cached is None:
                return None
            self._backtest_cache.move_to_end(key)
            return cached

    def _store_cached_backtest(self, key, value):
        with self._cache_lock:
            self._backtest_cache[key] = value
            self._backtest_cache.move_to_end(key)
            while len(self._backtest_cache) > self.BACKTEST_CACHE_LIMIT:
                self._backtest_cache.popitem(last=False)

    def _get_plot_state(self, figure, canvas, with_mc: bool):
        state = self._plot_states.get(canvas)
        if state is not None and state.get("with_mc") == with_mc:
            return state

        figure.clear()
        use_constrained = False
        try:
            figure.set_layout_engine("constrained")
            use_constrained = True
        except Exception:
            use_constrained = False

        if with_mc:
            gs = figure.add_gridspec(2, 1, height_ratios=[2.5, 1.2])
            ax = figure.add_subplot(gs[0, 0])
            ax_mc = figure.add_subplot(gs[1, 0])
        else:
            ax = figure.add_subplot(111)
            ax_mc = None

        (strategy_line,) = ax.plot([], [], label="Strategy")
        (bh_line,) = ax.plot([], [], label="Buy & Hold")
        bh_line.set_visible(False)

        state = {
            "with_mc": with_mc,
            "use_constrained": use_constrained,
            "ax": ax,
            "ax_mc": ax_mc,
            "strategy_line": strategy_line,
            "bh_line": bh_line,
            "trap_patches": [],
        }
        self._plot_states[canvas] = state
        return state

    def _clear_plot(self, figure, canvas):
        self._plot_states.pop(canvas, None)
        figure.clear()
        canvas.draw_idle()

    # ---------- scan ----------
    def run_scan(self):
        try:
            config = self._capture_scan_config()
        except Exception:
            self._show_error(traceback.format_exc())
            return
        self._set_status("Scanning…（首次可能較慢，numba 正在編譯）")
        self.btn_hist_run.setEnabled(False)
        worker = Worker(self._scan_compute, config)
        worker.signals.finished.connect(self._scan_update_ui)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _capture_data_config(
        self, symbol_cb, source_cb, interval_cb, start_edit, end_edit, capital_edit
    ):
        start, end = self._get_date_range_strings(start_edit, end_edit)
        return {
            "symbol": symbol_cb.currentText().strip(),
            "source": source_cb.currentText().strip().lower(),
            "interval": interval_cb.currentText().strip(),
            "start": start,
            "end": end,
            "refresh_policy": DEFAULT_REFRESH_POLICY,
            "fee_rate": DEFAULT_FEE_RATE,
            "capital": safe_float(capital_edit.text().strip(), None),
        }

    def _capture_scan_config(self):
        config = self._capture_data_config(
            self.e_symbol, self.e_source, self.e_interval, self.e_start, self.e_end, self.e_capital
        )
        config.update({
            "scan_mode": self.e_scan_mode.currentText(),
            "add_drop_range": self.e_add_drop.text(),
            "tp_range": self.e_tp.text(),
            "multiplier_range": self.e_multiplier.text(),
            "max_orders_range": self.e_max_orders.text(),
            "mae_horizon_days": self.e_mae_horizon.text(),
            "min_trades": self.e_min_trades.text(),
            "max_dd": self.e_max_dd.text(),
            "max_trap": self.e_max_trap.text(),
            "topn": self.e_topn.text(),
        })
        return config

    def _capture_mc_scan_config(self):
        config = self._capture_data_config(
            self.m_symbol, self.m_source, self.m_interval, self.m_start, self.m_end, self.m_capital
        )
        config.update({
            "add_drop_range": self.m_add_drop.text(),
            "tp_range": self.m_tp.text(),
            "multiplier_range": self.m_multiplier.text(),
            "max_orders_range": self.m_max_orders.text(),
            "sampling_mode": self.m_sampling_mode.currentText(),
            "sample_size": self.m_sample_size.text(),
            "refine_pct": self.m_refine_pct.text(),
            "max_combos": self.m_max_combos.text(),
            "hist_min_trades": self.m_hist_min_trades.text(),
            "hist_max_trap": self.m_hist_max_trap.text(),
            "show_topn": self.m_show_topn.text(),
            "mc_paths": self.m_mc_paths.text(),
            "mc_days": self.m_mc_days.text(),
            "mc_block": self.m_mc_block.text(),
            "mc_seed": self.m_mc_seed.text(),
            "seed_runs": self.m_mc_seed_runs.text(),
            "holdout_pct": self.m_mc_holdout.text(),
            "workers": self.m_mc_workers.text(),
            "max_loss": self.m_max_loss.text(),
            "max_severe": self.m_max_severe.text(),
            "max_dd50": self.m_max_dd50.text(),
            "rank_mode": self.m_rank_by.currentText(),
        })
        return config

    def _collect_context_from_config(self, config):
        symbol = str(config["symbol"]).strip()
        interval = str(config["interval"]).strip()
        capital = config.get("capital")
        if not symbol or not interval:
            raise ValueError("請填入 symbol 與 interval")
        if capital is None or not np.isfinite(capital) or capital <= 0:
            raise ValueError("capital 必須為正數")
        df = self._fetch_klines_if_needed(
            symbol, interval, config["start"], config["end"], config["refresh_policy"],
            config.get("source", "binance"),
        )
        prices_np = df["close"].to_numpy(dtype=np.float64)
        required = {"open", "high", "low", "close"}
        if prices_np.size < 2 or not required.issubset(df.columns):
            raise ValueError("需要至少 2 根完整 OHLC K 線")
        return {
            **config,
            "capital": float(capital),
            "fee_rate": float(config["fee_rate"]),
            "df": df,
            "prices_np": prices_np,
            "opens_np": df["open"].to_numpy(dtype=np.float64),
            "highs_np": df["high"].to_numpy(dtype=np.float64),
            "lows_np": df["low"].to_numpy(dtype=np.float64),
        }

    def _collect_context_from_inputs(
        self, symbol_cb, source_cb, interval_cb, start_edit, end_edit, capital_edit
    ):
        symbol = symbol_cb.currentText().strip()
        source = source_cb.currentText().strip().lower()
        interval = interval_cb.currentText().strip()
        start, end = self._get_date_range_strings(start_edit, end_edit)

        refresh_policy = DEFAULT_REFRESH_POLICY
        fee_rate = DEFAULT_FEE_RATE
        capital = safe_float(capital_edit.text().strip(), 1000.0)
        if not symbol or not interval:
            raise ValueError("請填入 symbol 與 interval")
        if capital is None or not np.isfinite(capital) or capital <= 0:
            raise ValueError("capital 必須為正數")

        df = self._fetch_klines_if_needed(
            symbol, interval, start, end, refresh_policy, source
        )
        prices_np = df["close"].to_numpy(dtype=np.float64)
        if prices_np.size < 2:
            raise ValueError("K 線資料不足（<2 根），無法回測/掃描。")
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError("歷史回測需要完整 OHLC，請重新抓取資料。")

        return {
            "symbol": symbol,
            "source": source,
            "interval": interval,
            "start": start,
            "end": end,
            "refresh_policy": refresh_policy,
            "fee_rate": float(fee_rate),
            "capital": float(capital),
            "df": df,
            "prices_np": prices_np,
            "opens_np": df["open"].to_numpy(dtype=np.float64),
            "highs_np": df["high"].to_numpy(dtype=np.float64),
            "lows_np": df["low"].to_numpy(dtype=np.float64),
        }

    def _collect_scan_context(self):
        return self._collect_context_from_inputs(
            self.e_symbol,
            self.e_source,
            self.e_interval,
            self.e_start,
            self.e_end,
            self.e_capital,
        )

    def _collect_mc_scan_context(self):
        return self._collect_context_from_inputs(
            self.m_symbol,
            self.m_source,
            self.m_interval,
            self.m_start,
            self.m_end,
            self.m_capital,
        )

    def _compute_filtered_scan_results(self, ctx):
        if str(ctx.get("scan_mode", "Fixed Mode")).strip().lower().startswith("diy"):
            return self._compute_filtered_diy_scan_results(ctx)

        add_drop_arr = parse_range(ctx["add_drop_range"])
        tp_arr = parse_range(ctx["tp_range"])
        mul_arr = parse_range(ctx["multiplier_range"])
        mo_arr = parse_range(ctx["max_orders_range"], is_int=True)
        if any(x.size == 0 for x in (add_drop_arr, tp_arr, mul_arr, mo_arr)):
            raise ValueError("掃描參數不得為空（add_drop/tp/multiplier/max_orders）")
        validate_strategy_param_arrays(add_drop_arr, tp_arr, mul_arr, mo_arr)

        params_df = sample_parameter_grid(
            add_drop_arr=add_drop_arr,
            tp_arr=tp_arr,
            mul_arr=mul_arr,
            mo_arr=mo_arr,
            mode="full grid",
            sample_size=1,
            max_combos=0,
            seed=0,
        )
        results_df = self._evaluate_param_candidates(
            params_df,
            ctx["prices_np"],
            ctx["capital"],
            ctx["fee_rate"],
            opens=ctx["opens_np"],
            highs=ctx["highs_np"],
            lows=ctx["lows_np"],
        )
        ctx["_scan_candidate_count"] = int(len(results_df))

        min_trades = safe_int(ctx["min_trades"].strip()) if ctx["min_trades"].strip() else None
        max_dd = safe_float(ctx["max_dd"].strip()) if ctx["max_dd"].strip() else None
        max_trap = safe_float(ctx["max_trap"].strip()) if ctx["max_trap"].strip() else None
        if max_trap is not None:
            max_trap /= 100.0
        return martin.apply_filters(results_df, min_trades, max_dd, max_trap)

    def _compute_filtered_diy_scan_results(self, ctx):
        return self._compute_filtered_tp_horizon_mae_diy_scan_results(ctx)

    def _compute_filtered_tp_horizon_mae_diy_scan_results(self, ctx):
        """Build the fixed bias=0 TP-or-horizon MAE family for every TP."""
        tp_arr = parse_range(
            ctx["tp_range"],
            max_items=DIY_MAX_TP_VALUES,
        )
        mo_arr = parse_range(
            ctx["max_orders_range"],
            is_int=True,
            max_items=DIY_TOTAL_SHARES,
        )
        q_end_arr = np.asarray(DIY_MAE_Q_END_VALUES, dtype=np.float64)
        initial_fraction_arr = np.asarray(
            DIY_INITIAL_FRACTION_VALUES, dtype=np.float64
        )
        gamma_arr = np.asarray(DIY_CAPITAL_GAMMA_VALUES, dtype=np.float64)
        idle_bias_arr = np.asarray(DIY_IDLE_BIAS_VALUES, dtype=np.float64)
        if tp_arr.size == 0 or mo_arr.size == 0:
            raise ValueError("TP/Horizon MAE DIY 掃描參數不得為空")
        if np.any(tp_arr <= 0.0) or np.any(mo_arr < 2):
            raise ValueError("TP/Horizon MAE DIY 需滿足 tp > 0 且 max_orders >= 2")

        horizon_days = safe_float(ctx["mae_horizon_days"].strip(), None)
        if horizon_days is None or horizon_days <= 0.0:
            raise ValueError("MAE horizon(days) 必須 > 0")
        if DIY_TOTAL_SHARES < int(np.max(mo_arr)):
            raise ValueError(
                f"max_orders 不可大於內建 Total shares ({DIY_TOTAL_SHARES})"
            )

        horizon_bars = self._days_to_bars(
            horizon_days,
            ctx["interval"],
            ctx.get("source", "binance"),
            ctx.get("df"),
        )
        raw_candidate_count = int(
            tp_arr.size
            * mo_arr.size
            * q_end_arr.size
            * initial_fraction_arr.size
            * gamma_arr.size
            * idle_bias_arr.size
        )
        if raw_candidate_count > DIY_MAX_CANDIDATES:
            raise ValueError(
                f"TP/Horizon MAE DIY 原始候選共 "
                f"{raw_candidate_count:,} 組，超過 "
                f"{DIY_MAX_CANDIDATES:,}；請放大 step 或縮小參數範圍"
            )
        mae_grid_bytes = estimate_tp_or_horizon_mae_grid_bytes(
            len(ctx["prices_np"]),
            horizon_bars,
            tp_arr.size,
        )
        if mae_grid_bytes > DIY_MAX_MAE_GRID_BYTES:
            raise ValueError(
                "TP/Horizon MAE 輸出矩陣預估需要 "
                f"{mae_grid_bytes / (1024 ** 2):,.1f} MiB，超過 "
                f"{DIY_MAX_MAE_GRID_BYTES / (1024 ** 2):,.1f} MiB 限制；"
                "請放大 TP step、縮小 TP 範圍或縮短資料期間"
            )
        mae_grid, hit_grid, bars_grid = compute_tp_or_horizon_mae_grid(
            ctx["highs_np"],
            ctx["lows_np"],
            ctx["prices_np"],
            horizon_bars,
            tp_arr,
            fee_rate=float(ctx["fee_rate"]),
            max_output_bytes=DIY_MAX_MAE_GRID_BYTES,
        )

        template_frames = []
        level_parts = []
        share_parts = []
        next_template_index = 0
        for tp_index, tp_value in enumerate(tp_arr):
            mae_samples = mae_grid[tp_index]
            hit_mask = hit_grid[tp_index].astype(bool)
            hit_bars = bars_grid[tp_index][hit_mask]
            templates, levels, shares, _ = build_diy_templates(
                mae_samples=mae_samples,
                max_orders_values=mo_arr,
                q_start=DIY_MAE_Q_START,
                q_end_values=q_end_arr,
                initial_fraction_values=initial_fraction_arr,
                gamma_values=gamma_arr,
                stress_quantile=DIY_STRESS_MAE_Q,
                total_shares=DIY_TOTAL_SHARES,
                fee_rate=float(ctx["fee_rate"]),
                max_last_order_pct=DIY_MAX_LAST_ORDER_PCT,
                max_stress_loss_pct=DIY_MAX_STRESS_LOSS_PCT,
                idle_bias_values=idle_bias_arr,
            )
            if templates.empty:
                continue
            templates = templates.copy()
            template_count = len(templates)
            templates["template_index"] = np.arange(
                next_template_index,
                next_template_index + template_count,
                dtype=np.int64,
            )
            templates["tp"] = float(tp_value)
            templates["diy_variant"] = "tp_horizon_mae"
            templates["mae_model"] = "tp_or_horizon"
            templates["tp_hit_rate_horizon"] = float(np.mean(hit_mask))
            templates["tp_non_hit_rate_horizon"] = float(1.0 - np.mean(hit_mask))
            templates["median_bars_to_tp"] = (
                float(np.median(hit_bars)) if hit_bars.size else np.nan
            )
            templates["p90_bars_to_tp"] = (
                float(np.percentile(hit_bars, 90.0))
                if hit_bars.size else np.nan
            )
            templates["mae_sample_count"] = int(mae_samples.size)
            template_frames.append(templates)
            level_parts.append(levels)
            share_parts.append(shares)
            next_template_index += template_count

        if not template_frames:
            raise ValueError(
                "TP/Horizon MAE DIY 在內建風險限制下沒有可用的 Shares 模板；"
                "請調整 TP、max_orders 或 MAE horizon"
            )

        results_df = pd.concat(template_frames, ignore_index=True)
        level_matrix = np.ascontiguousarray(
            np.vstack(level_parts), dtype=np.float64
        )
        share_matrix = np.ascontiguousarray(
            np.vstack(share_parts), dtype=np.float64
        )
        candidate_count = int(len(results_df))
        if candidate_count > DIY_MAX_CANDIDATES:
            raise ValueError(
                f"TP/Horizon MAE DIY 候選共 {candidate_count:,} 組，超過 "
                f"{DIY_MAX_CANDIDATES:,}；"
                "請放大 step 或縮小參數範圍"
            )

        template_indices = np.arange(candidate_count, dtype=np.int32)
        max_orders_full = results_df["max_orders"].to_numpy(dtype=np.int32)
        tp_full = results_df["tp"].to_numpy(dtype=np.float64)
        (
            fe,
            mdd,
            tr,
            trap,
            utilization,
            underwater,
            full_capital,
        ) = martin._grid_search_parallel_diy_idle_ohlc(
            np.asarray(ctx["opens_np"], dtype=np.float64),
            np.asarray(ctx["highs_np"], dtype=np.float64),
            np.asarray(ctx["lows_np"], dtype=np.float64),
            np.asarray(ctx["prices_np"], dtype=np.float64),
            level_matrix,
            share_matrix,
            template_indices,
            max_orders_full,
            tp_full,
            capital=float(ctx["capital"]),
            fee_rate=float(ctx["fee_rate"]),
        )

        results_df["capital"] = float(ctx["capital"])
        results_df["final_equity"] = np.round(fe, 2)
        results_df["max_dd_overall"] = np.round(mdd, 2)
        results_df["trades"] = tr.astype(int)
        results_df["trapped_time_ratio"] = np.round(trap, 6)
        results_df["avg_capital_utilization"] = np.round(utilization, 6)
        results_df["actual_idle_ratio"] = np.round(1.0 - utilization, 6)
        results_df["underwater_position_ratio"] = np.round(underwater, 6)
        results_df["full_capital_time_ratio"] = np.round(full_capital, 6)
        results_df["mae_horizon_days"] = float(horizon_days)
        results_df["mae_horizon_bars"] = int(horizon_bars)
        results_df["stress_mae_q"] = DIY_STRESS_MAE_Q
        results_df["diy_total_shares"] = DIY_TOTAL_SHARES

        ctx["_scan_candidate_count"] = candidate_count
        ctx["_diy_template_count"] = candidate_count
        ctx["_mae_sample_count"] = int(mae_grid.shape[1])
        ctx["_mae_horizon_bars"] = int(horizon_bars)

        min_trades = (
            safe_int(ctx["min_trades"].strip())
            if ctx["min_trades"].strip() else None
        )
        max_dd = (
            safe_float(ctx["max_dd"].strip())
            if ctx["max_dd"].strip() else None
        )
        max_trap = (
            safe_float(ctx["max_trap"].strip())
            if ctx["max_trap"].strip() else None
        )
        if max_trap is not None:
            max_trap /= 100.0
        return martin.apply_filters(results_df, min_trades, max_dd, max_trap)

    def _evaluate_param_candidates(
        self, params_df: pd.DataFrame, prices_np, capital, fee_rate, *, opens=None, highs=None, lows=None
    ) -> pd.DataFrame:
        if params_df.empty:
            return pd.DataFrame(
                columns=[
                    "add_drop",
                    "multiplier",
                    "max_orders",
                    "tp",
                    "capital",
                    "final_equity",
                    "max_dd_overall",
                    "trades",
                    "trapped_time_ratio",
                    "min_buy_ratio",
                ]
            )

        unique_params = params_df.drop_duplicates(
            subset=["add_drop", "multiplier", "max_orders", "tp"], ignore_index=True
        )
        add_drop = unique_params["add_drop"].to_numpy(dtype=np.float64)
        multiplier = unique_params["multiplier"].to_numpy(dtype=np.float64)
        max_orders = unique_params["max_orders"].to_numpy(dtype=np.int32)
        tp = unique_params["tp"].to_numpy(dtype=np.float64)
        validate_strategy_param_arrays(add_drop, tp, multiplier, max_orders)
        min_buy_ratio = np.maximum(0.0, (1.0 - add_drop) ** (max_orders.astype(np.float64) - 1.0))

        if opens is not None and highs is not None and lows is not None:
            fe, mdd, tr, trap = martin._grid_search_parallel_ohlc(
                np.asarray(opens, dtype=np.float64),
                np.asarray(highs, dtype=np.float64),
                np.asarray(lows, dtype=np.float64),
                np.asarray(prices_np, dtype=np.float64),
                add_drop,
                multiplier,
                max_orders,
                tp,
                capital=float(capital),
                fee_rate=float(fee_rate),
            )
        else:
            fe, mdd, tr, trap = martin._grid_search_parallel(
                prices_np,
                add_drop,
                multiplier,
                max_orders,
                tp,
                capital=float(capital),
                fee_rate=float(fee_rate),
            )
        return pd.DataFrame(
            {
                "add_drop": add_drop,
                "multiplier": multiplier,
                "max_orders": max_orders.astype(int),
                "tp": tp,
                "capital": float(capital),
                "final_equity": np.round(fe, 2),
                "max_dd_overall": np.round(mdd, 2),
                "trades": tr.astype(int),
                "trapped_time_ratio": np.round(trap, 6),
                "min_buy_ratio": np.round(min_buy_ratio, 6),
            }
        )

    def _get_mc_hist_filter_values(self, config):
        min_trades = safe_int(config["hist_min_trades"].strip()) if config["hist_min_trades"].strip() else None
        max_trap = safe_float(config["hist_max_trap"].strip()) if config["hist_max_trap"].strip() else None
        if max_trap is not None:
            max_trap /= 100.0
        return min_trades, max_trap

    def _compute_filtered_mc_candidates(self, ctx):
        add_drop_arr = parse_range(ctx["add_drop_range"])
        tp_arr = parse_range(ctx["tp_range"])
        mul_arr = parse_range(ctx["multiplier_range"])
        mo_arr = parse_range(ctx["max_orders_range"], is_int=True)
        if any(x.size == 0 for x in (add_drop_arr, tp_arr, mul_arr, mo_arr)):
            raise ValueError("掃描參數不得為空（add_drop/tp/multiplier/max_orders）")
        validate_strategy_param_arrays(add_drop_arr, tp_arr, mul_arr, mo_arr)

        mode_text = ctx["sampling_mode"].strip().lower()
        mode = "lhs"
        if mode_text.startswith("random"):
            mode = "random"
        elif mode_text.startswith("full grid"):
            mode = "full grid"

        seed = safe_int(ctx["mc_seed"].strip(), 42)
        sample_size = max(1, safe_int(ctx["sample_size"].strip(), 5000))
        max_combos = safe_int(ctx["max_combos"].strip(), 0)
        refine_pct = max(0.0, safe_float(ctx["refine_pct"].strip(), 5.0))
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

        results_df = self._evaluate_param_candidates(
            params_df,
            ctx["prices_np"],
            ctx["capital"],
            ctx["fee_rate"],
            opens=ctx["opens_np"],
            highs=ctx["highs_np"],
            lows=ctx["lows_np"],
        )

        min_trades, max_trap = self._get_mc_hist_filter_values(ctx)
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
                extra = self._evaluate_param_candidates(
                    refine_df,
                    ctx["prices_np"],
                    ctx["capital"],
                    ctx["fee_rate"],
                    opens=ctx["opens_np"],
                    highs=ctx["highs_np"],
                    lows=ctx["lows_np"],
                )
                filtered_extra = martin.apply_filters(extra, min_trades, None, max_trap)
                if not filtered_extra.empty:
                    filtered = pd.concat([filtered, filtered_extra], ignore_index=True)

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
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self._mc_executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        )
        self._mc_executor_workers = workers
        return self._mc_executor

    def _eval_mc_candidates_parallel(
        self,
        candidates: pd.DataFrame,
        hist_rets: np.ndarray,
        hist_ohlc_ratios: np.ndarray,
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
            hist_ohlc_ratios=hist_ohlc_ratios,
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

    def _scan_compute(self, config):
        started = time.perf_counter()
        ctx = self._collect_context_from_config(config)
        filtered = self._compute_filtered_scan_results(ctx)
        topn = safe_int(ctx["topn"].strip(), 20)
        if topn <= 0:
            raise ValueError("Show Top N 必須 > 0")
        top_df = filtered.nlargest(topn, "final_equity").copy()
        run_config = dict(config)
        for key in (
            "_scan_candidate_count",
            "_diy_template_count",
            "_mae_sample_count",
            "_mae_horizon_bars",
            "_stress_mae",
        ):
            if key in ctx:
                run_config[key] = ctx[key]
        run_config["_scan_elapsed_seconds"] = float(time.perf_counter() - started)
        return top_df, run_config

    def _populate_scan_table(self, disp: pd.DataFrame, cols):
        self.table.setUpdatesEnabled(False)
        self.scan_table_model.set_frame(disp, cols)
        self._autosize_scan_table_columns()
        self.table.setUpdatesEnabled(True)
        self.table.viewport().update()

    def _populate_mc_scan_table(self, disp: pd.DataFrame, cols):
        self.mc_table.setUpdatesEnabled(False)
        self.mc_table_model.set_frame(disp, cols)
        self._autosize_mc_scan_table_columns()
        self.mc_table.setUpdatesEnabled(True)
        self.mc_table.viewport().update()

    @Slot(object)
    def _scan_update_ui(self, payload):
        self.btn_hist_run.setEnabled(True)
        if isinstance(payload, tuple) and len(payload) == 2:
            top_df, run_config = payload
        else:
            top_df, run_config = payload, None
        self.scan_df = top_df.reset_index(drop=True)
        self._scan_run_config = dict(run_config) if run_config is not None else None
        disp, cols = self._format_hist_scan_display(self.scan_df)
        self._populate_scan_table(disp, cols)
        if self.scan_df.empty:
            self._set_status("Scan completed. No rows matched current filters.")
            QMessageBox.information(self, "Info", "篩選條件過嚴，請放寬。")
        else:
            elapsed = (
                float(run_config.get("_scan_elapsed_seconds", 0.0))
                if run_config else 0.0
            )
            candidate_count = (
                int(run_config.get("_scan_candidate_count", len(self.scan_df)))
                if run_config else len(self.scan_df)
            )
            mode = (
                str(run_config.get("scan_mode", "Fixed Mode")).split("(", 1)[0].strip()
                if run_config else "Fixed Mode"
            )
            self._set_status(
                f"{mode} scan completed: {candidate_count:,} candidates / {elapsed:.2f}s."
            )
        self.scan_tabs.setCurrentWidget(self.scan_tab_table)

    # ---------- MC scan ----------
    def run_mc_scan(self):
        try:
            config = self._capture_mc_scan_config()
        except Exception:
            self._show_error(traceback.format_exc())
            return
        sampling = (self.m_sampling_mode.currentText() or "").strip()
        sampling_short = sampling.split("(", 1)[0].strip() if sampling else "Unknown"
        self._set_status(f"Scan中…（{sampling_short} + 風險約束，可能較久）")
        self.btn_mc_run.setEnabled(False)
        worker = Worker(self._mc_scan_compute, config)
        worker.signals.finished.connect(self._mc_scan_update_ui)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _mc_scan_compute(self, config):
        ctx = self._collect_context_from_config(config)
        mc_paths = safe_int(ctx["mc_paths"].strip(), None)
        mc_days = safe_float(ctx["mc_days"].strip(), None)
        mc_block = safe_int(ctx["mc_block"].strip(), None)
        mc_seed = safe_int(ctx["mc_seed"].strip(), 42)
        seed_runs = safe_int(ctx["seed_runs"].strip(), 1)
        holdout_pct = safe_float(ctx["holdout_pct"].strip(), 30.0)
        workers = safe_int(ctx["workers"].strip(), 0)
        if None in (mc_paths, mc_days, mc_block):
            raise ValueError("請完整填入參數（paths/days/block）")
        if mc_paths <= 0 or mc_days <= 0 or mc_block <= 0 or seed_runs <= 0:
            raise ValueError("參數需滿足：paths>0、days>0、block>0、seed runs>0")
        if seed_runs > 100:
            raise ValueError("Seed runs 不可超過 100")
        if not np.isfinite(holdout_pct) or not (5.0 <= holdout_pct <= 50.0):
            raise ValueError("MC holdout 必須介於 5% 到 50%")
        if len(ctx["prices_np"]) < 10:
            raise ValueError("MC holdout 至少需要 10 根歷史 K 線")

        split_idx = int(np.floor(len(ctx["prices_np"]) * (1.0 - holdout_pct / 100.0)))
        split_idx = max(2, min(split_idx, len(ctx["prices_np"]) - 3))
        train_ctx = dict(ctx)
        for key in ("prices_np", "opens_np", "highs_np", "lows_np"):
            train_ctx[key] = ctx[key][:split_idx]
        train_ctx["df"] = ctx["df"].iloc[:split_idx].copy()

        candidates = self._compute_filtered_mc_candidates(train_ctx)
        n_cand = len(candidates)
        if candidates.empty:
            return pd.DataFrame(), 0, 0, dict(config)
        if workers <= 0:
            workers = max(1, min(os.cpu_count() or 1, 8))
        workers = min(int(workers), max(1, int(n_cand)))

        max_loss = safe_float(ctx["max_loss"].strip(), 30.0) / 100.0
        max_severe = safe_float(ctx["max_severe"].strip(), 10.0) / 100.0
        max_dd50 = safe_float(ctx["max_dd50"].strip(), 20.0) / 100.0
        if any((not np.isfinite(x)) or x < 0.0 or x > 1.0 for x in (max_loss, max_severe, max_dd50)):
            raise ValueError("MC 風險門檻必須介於 0% 到 100%")

        holdout_prices = ctx["prices_np"][split_idx - 1:]
        hist_rets = holdout_prices[1:] / holdout_prices[:-1] - 1.0
        hist_ohlc_ratios = ohlc_ratios_from_history(
            ctx["opens_np"][split_idx - 1:],
            ctx["highs_np"][split_idx - 1:],
            ctx["lows_np"][split_idx - 1:],
            holdout_prices,
        )
        if mc_block > len(hist_rets):
            raise ValueError(
                f"MC block size ({mc_block}) 大於 holdout 報酬樣本數 ({len(hist_rets)})；"
                "請縮小 block 或 holdout。"
            )
        start_price = float(ctx["prices_np"][-1])
        mc_bars = self._days_to_bars(
            mc_days,
            ctx["interval"],
            ctx.get("source", "binance"),
            ctx.get("df"),
        )
        seed_children = np.random.SeedSequence(int(mc_seed)).spawn(int(seed_runs))
        run_outputs = []
        for child in seed_children:
            run_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            run_outputs.append(self._eval_mc_candidates_parallel(
                candidates=candidates,
                hist_rets=hist_rets,
                hist_ohlc_ratios=hist_ohlc_ratios,
                start_price=start_price,
                capital=float(ctx["capital"]),
                fee_rate=float(ctx["fee_rate"]),
                mc_bars=int(mc_bars),
                block_size=int(mc_block),
                total_paths=int(mc_paths),
                base_seed=run_seed,
                max_loss=float(max_loss),
                max_severe=float(max_severe),
                max_dd50=float(max_dd50),
                workers=int(workers),
            ))

        out = run_outputs[0].copy()
        def _nan_reduce(arrays, mode):
            stack = np.vstack(arrays)
            valid = np.isfinite(stack)
            count = valid.sum(axis=0)
            if mode == "mean":
                values = np.divide(
                    np.where(valid, stack, 0.0).sum(axis=0), count,
                    out=np.full(stack.shape[1], np.nan), where=count > 0,
                )
            elif mode == "min":
                values = np.where(valid, stack, np.inf).min(axis=0)
                values[count == 0] = np.nan
            else:
                values = np.where(valid, stack, -np.inf).max(axis=0)
                values[count == 0] = np.nan
            return values

        mean_cols = ("mc_terminal_mean", "mc_terminal_median")
        conservative_min_cols = ("mc_terminal_p5",)
        conservative_max_cols = (
            "mc_p_loss", "mc_p_severe", "mc_p_dd50", "mc_mdd_mean", "mc_trapped_mean",
        )
        for col in mean_cols:
            out[col] = _nan_reduce([r[col].to_numpy(float) for r in run_outputs], "mean")
        for col in conservative_min_cols:
            out[col] = _nan_reduce([r[col].to_numpy(float) for r in run_outputs], "min")
        for col in conservative_max_cols:
            out[col] = _nan_reduce([r[col].to_numpy(float) for r in run_outputs], "max")
        median_stack = np.vstack([r["mc_terminal_median"].to_numpy(float) for r in run_outputs])
        median_mean = _nan_reduce([r["mc_terminal_median"].to_numpy(float) for r in run_outputs], "mean")
        median_valid = np.isfinite(median_stack)
        median_count = median_valid.sum(axis=0)
        median_var = np.divide(
            np.where(median_valid, (median_stack - median_mean) ** 2, 0.0).sum(axis=0),
            median_count,
            out=np.full(median_stack.shape[1], np.nan),
            where=median_count > 0,
        )
        out["mc_seed_median_std"] = np.sqrt(median_var)
        out["feasible"] = np.logical_and.reduce([r["feasible"].to_numpy(bool) for r in run_outputs])
        out["mc_early_rejected"] = np.logical_or.reduce([
            r["mc_early_rejected"].to_numpy(bool) for r in run_outputs
        ])
        out["mc_paths_evaluated"] = np.sum(np.vstack([
            r["mc_paths_evaluated"].to_numpy(np.int64) for r in run_outputs
        ]), axis=0)
        rejected = out["mc_early_rejected"].to_numpy(bool)
        for col in (*mean_cols, *conservative_min_cols, "mc_mdd_mean", "mc_trapped_mean"):
            out.loc[rejected, col] = np.nan
        out["mc_paths"] = int(mc_paths)
        out["mc_seed_runs"] = int(seed_runs)
        out["mc_days"] = float(mc_days)
        out["mc_bars"] = int(mc_bars)
        out["mc_block"] = int(mc_block)
        out["mc_seed"] = int(mc_seed)
        out["mc_holdout_pct"] = float(holdout_pct)
        out["mc_train_bars"] = int(split_idx)
        out["mc_holdout_bars"] = int(len(ctx["prices_np"]) - split_idx + 1)

        rank_mode = ctx["rank_mode"].strip()
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
        show_topn = safe_int(ctx["show_topn"].strip(), 50)
        if show_topn > 0:
            out = out.head(show_topn).copy()
        return out, feasible_count, int(n_cand), dict(config)

    @Slot(object)
    def _mc_scan_update_ui(self, payload):
        self.btn_mc_run.setEnabled(True)
        if len(payload) == 4:
            df, feasible_count, total, run_config = payload
        else:
            df, feasible_count, total = payload
            run_config = None
        self.mc_scan_df = df.reset_index(drop=True)
        self._mc_scan_run_config = (
            dict(run_config) if run_config is not None else None
        )
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
        self.mc_table_model.clear()
        if hasattr(self, "mc_metrics_text") and self.mc_metrics_text:
            self.mc_metrics_text.clear()
        if hasattr(self, "figure_mc_scan") and self.figure_mc_scan:
            self._clear_plot(self.figure_mc_scan, self.canvas_mc_scan)
        self.mc_scan_df = None
        self._mc_scan_run_config = None
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
        self.scan_table_model.clear()
        if self.scan_metrics_text:
            self.scan_metrics_text.clear()
        if self.figure_scan:
            self._clear_plot(self.figure_scan, self.canvas_scan)
        self.scan_df = None
        self._scan_run_config = None
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
        original_idx = self.scan_table_model.frame.index[idx]
        params = self.scan_df.loc[original_idx]
        config = self._scan_run_config
        if not config:
            QMessageBox.information(self, "提示", "掃描設定已不存在，請重新 Run Scan。")
            return
        self._set_status("載入詳細回測…")

        interval_str = str(config["interval"]).strip()
        start, end = str(config["start"]), str(config["end"])

        is_diy = str(params.get("strategy_mode", "fixed")).strip().lower() == "diy"
        if is_diy:
            metadata_keys = (
                "mae_horizon_days", "mae_horizon_bars", "mae_sample_count",
                "mae_q_start", "mae_q_end", "mae_quantiles", "stress_mae_q",
                "stress_mae", "stress_loss_pct", "initial_capital_pct",
                "capital_gamma", "idle_bias", "last_order_pct",
                "diy_total_shares", "diy_variant", "mae_model",
                "fill_probabilities", "expected_capital_utilization",
                "expected_idle_ratio", "tp_hit_rate_horizon",
                "tp_non_hit_rate_horizon", "median_bars_to_tp",
                "p90_bars_to_tp",
            )
            metadata = {
                key: params[key] for key in metadata_keys if key in params.index
            }
            worker = Worker(
                self._compute_diy_backtest,
                str(config["symbol"]).strip(),
                interval_str,
                start,
                end,
                str(config.get("refresh_policy", DEFAULT_REFRESH_POLICY)),
                float(config.get("fee_rate", DEFAULT_FEE_RATE)),
                float(config["capital"]),
                tuple(params["level_ratios"]),
                tuple(params["order_shares"]),
                float(params["tp"]),
                str(config.get("source", "binance")).strip().lower(),
                metadata,
            )
        else:
            worker = Worker(
                self._compute_backtest,
                str(config["symbol"]).strip(),
                interval_str,
                start,
                end,
                str(config.get("refresh_policy", DEFAULT_REFRESH_POLICY)),
                float(config.get("fee_rate", DEFAULT_FEE_RATE)),
                float(config["capital"]),
                float(params["add_drop"]),
                float(params["multiplier"]),
                int(params["max_orders"]),
                float(params["tp"]),
                str(config.get("source", "binance")).strip().lower(),
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
        original_idx = self.mc_table_model.frame.index[idx]
        params = self.mc_scan_df.loc[original_idx]
        config = self._mc_scan_run_config
        if not config:
            QMessageBox.information(self, "提示", "掃描設定已不存在，請重新 Run Scan。")
            return
        self._set_status("載入候選參數詳細回測…")

        interval_str = str(config["interval"]).strip()
        start, end = str(config["start"]), str(config["end"])

        worker = Worker(
            self._compute_backtest,
            str(config["symbol"]).strip(),
            interval_str,
            start,
            end,
            str(config.get("refresh_policy", DEFAULT_REFRESH_POLICY)),
            float(config.get("fee_rate", DEFAULT_FEE_RATE)),
            float(config["capital"]),
            float(params["add_drop"]),
            float(params["multiplier"]),
            int(params["max_orders"]),
            float(params["tp"]),
            str(config.get("source", "binance")).strip().lower(),
        )
        worker.signals.finished.connect(self._render_mc_scan_detail)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    @Slot(object)
    def _render_scan_detail(self, payload):
        df, res, perf = payload
        self._render_plot_and_metrics(
            df, res, perf,
            float(res.get("_add_drop", np.nan)),
            float(res.get("_multiplier", np.nan)),
            int(res["_max_orders"]),
            float(res["_tp"]),
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
        try:
            config = self._capture_single_config(include_mc=False)
        except Exception:
            self._show_error(traceback.format_exc())
            return
        self._set_status("Single Backtest中…")
        self.btn_single_run.setEnabled(False)
        worker = Worker(self._run_single_compute, config)
        worker.signals.finished.connect(self._run_single_update)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def _run_single_compute(self, config):
        symbol = config["symbol"]
        interval = config["interval"]
        start, end = config["start"], config["end"]
        refresh = config["refresh_policy"]
        source = config.get("source", "binance")
        fee_rate = config["fee_rate"]
        capital = config["capital"]

        add_drop = config.get("add_drop")
        tp = config.get("tp")
        multiplier = config.get("multiplier")
        max_orders = config.get("max_orders")

        if None in (add_drop, tp, multiplier, max_orders):
            raise ValueError("請完整填入策略參數（add_drop, tp, multiplier, max_orders）")
        if not symbol or not interval:
            raise ValueError("請填入 symbol 與 interval")
        if capital is None or not np.isfinite(capital) or capital <= 0:
            raise ValueError("capital 必須為有限正數")
        validate_strategy_param_arrays([add_drop], [tp], [multiplier], [max_orders])

        df, res, perf = self._compute_backtest(symbol, interval, start, end, refresh, fee_rate, capital,
                                               add_drop, multiplier, max_orders, tp, source)
        return df, res, perf

    @Slot(object)
    def _run_single_update(self, payload):
        self.btn_single_run.setEnabled(True)
        df, res, perf = payload
        self._render_plot_and_metrics(
            df, res, perf,
            float(res["_add_drop"]), float(res["_multiplier"]), int(res["_max_orders"]), float(res["_tp"]),
            self.figure_single, self.canvas_single, self.metrics_text
        )
        self._set_status("回測完成。")
        self._set_splitter_when_ready(self.single_splitter, self.INIT_SPLIT)

    def run_single_mc(self):
        try:
            config = self._capture_single_config(include_mc=True)
        except Exception:
            self._show_error(traceback.format_exc())
            return
        self._set_status("Monte Carlo模擬中…")
        self.btn_single_mc.setEnabled(False)
        worker = Worker(self._run_single_mc_compute, config)
        worker.signals.finished.connect(self._run_single_mc_update)
        worker.signals.error.connect(self._show_error)
        self.thread_pool.start(worker)

    def clear_single_results(self):
        if self.metrics_text:
            self.metrics_text.clear()
        if self.figure_single:
            self._clear_plot(self.figure_single, self.canvas_single)
        self._set_status("Single backtest results cleared.")

    def _interval_bars_per_year(
        self, interval: str, source="binance", df: pd.DataFrame | None = None
    ) -> float:
        if str(source or "").strip().lower() == "alpaca":
            return alpaca.bars_per_trading_day(
                df, interval
            ) * ALPACA_TRADING_DAYS_PER_YEAR
        step_ms = martin._interval_ms(interval)
        return (365.0 * 24.0 * 3600.0 * 1000.0) / float(step_ms)

    def _days_to_bars(
        self,
        days: float,
        interval: str,
        source="binance",
        df: pd.DataFrame | None = None,
    ) -> int:
        if str(source or "").strip().lower() == "alpaca":
            bars_per_calendar_day = (
                alpaca.bars_per_trading_day(df, interval)
                * ALPACA_TRADING_DAYS_PER_YEAR
                / 365.0
            )
            return max(2, int(round(float(days) * bars_per_calendar_day)))
        step_ms = martin._interval_ms(interval)
        bars = int(round(float(days) * 86400.0 * 1000.0 / float(step_ms)))
        return max(2, bars)

    def _capture_single_config(self, *, include_mc):
        config = self._capture_data_config(
            self.s_symbol, self.s_source, self.s_interval, self.s_start, self.s_end, self.s_capital
        )
        config.update({
            "add_drop": safe_float(self.s_add_drop.text().strip(), None),
            "tp": safe_float(self.s_tp.text().strip(), None),
            "multiplier": safe_float(self.s_multiplier.text().strip(), None),
            "max_orders": safe_int(self.s_max_orders.text().strip(), None),
        })
        if include_mc:
            config.update({
                "mc_paths": safe_int(self.s_mc_paths.text().strip(), None),
                "mc_days": safe_float(self.s_mc_days.text().strip(), None),
                "mc_block": safe_int(self.s_mc_block.text().strip(), None),
                "mc_seed": safe_int(self.s_mc_seed.text().strip(), 42),
            })
        return config

    def _run_single_mc_compute(self, config):
        df, res, perf = self._run_single_compute(config)

        mc_paths = config.get("mc_paths")
        mc_days = config.get("mc_days")
        mc_block = config.get("mc_block")
        mc_seed = config.get("mc_seed", 42)
        if None in (mc_paths, mc_days, mc_block):
            raise ValueError("請完整填入 Monte Carlo 參數（paths/days/block size）")
        if mc_paths <= 0 or mc_days <= 0 or mc_block <= 0:
            raise ValueError("Monte Carlo 參數需滿足：paths>0、days>0、block size>0")

        prices_np = df["close"].to_numpy(dtype=np.float64)
        hist_rets = prices_np[1:] / prices_np[:-1] - 1.0
        hist_ohlc_ratios = ohlc_ratios_from_history(
            df["open"].to_numpy(dtype=np.float64),
            df["high"].to_numpy(dtype=np.float64),
            df["low"].to_numpy(dtype=np.float64),
            prices_np,
        )
        interval = config["interval"]
        source = config.get("source", "binance")
        mc_bars = self._days_to_bars(mc_days, interval, source, df)
        rng = np.random.default_rng(int(mc_seed))

        add_drop = float(res["_add_drop"])
        multiplier = float(res["_multiplier"])
        max_orders = int(res["_max_orders"])
        tp = float(res["_tp"])
        capital = float(res["capital"])
        fee_rate = DEFAULT_FEE_RATE
        bars_per_year = self._interval_bars_per_year(interval, source, df)
        years_per_path = mc_bars / bars_per_year if bars_per_year > 0 else np.nan

        start_price = float(prices_np[-1]) if len(prices_np) else 1.0
        terminal = np.empty(mc_paths, dtype=np.float64)
        mdd_pct = np.empty(mc_paths, dtype=np.float64)
        trades = np.empty(mc_paths, dtype=np.int32)
        trapped_ratio = np.empty(mc_paths, dtype=np.float64)

        for i in range(mc_paths):
            path_open, path_high, path_low, path_close = bootstrap_ohlc_path_from_ratios(
                hist_ohlc_ratios, start_price, int(mc_bars), int(mc_block), rng
            )
            fe_i, mdd_i, tr_i, trap_i = martin._backtest_core_ohlc(
                path_open,
                path_high,
                path_low,
                path_close,
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
        self.btn_single_mc.setEnabled(True)
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
    def _compute_diy_backtest(
        self,
        symbol,
        interval,
        start,
        end,
        refresh_policy,
        fee_rate,
        capital,
        level_ratios,
        order_shares,
        tp,
        source="binance",
        metadata=None,
    ):
        levels = tuple(float(v) for v in level_ratios)
        shares = tuple(float(v) for v in order_shares)
        cache_key = (
            "diy",
            symbol,
            interval,
            start,
            end,
            refresh_policy,
            float(fee_rate),
            float(capital),
            levels,
            shares,
            float(tp),
            str(source or "binance").lower(),
            self._refresh_generation(interval, refresh_policy),
        )
        cached = self._get_cached_backtest(cache_key)
        if cached is not None:
            cached_df, cached_result, cached_performance = cached
            result_for_row = dict(cached_result)
            result_for_row["_diy_metadata"] = dict(metadata or {})
            return cached_df, result_for_row, cached_performance

        df = self._fetch_klines_if_needed(
            symbol, interval, start, end, refresh_policy, source
        )
        prices_np = df["close"].to_numpy(dtype=np.float64)
        if prices_np.size < 2:
            raise ValueError("K 線資料不足（<2 根）。")
        res = martin.martingale_backtest_diy(
            prices_np,
            level_ratios=levels,
            order_shares=shares,
            tp=float(tp),
            capital=float(capital),
            return_curve=True,
            times=df["time"].tolist(),
            fee_rate=float(fee_rate),
            opens=df["open"].to_numpy(dtype=np.float64),
            highs=df["high"].to_numpy(dtype=np.float64),
            lows=df["low"].to_numpy(dtype=np.float64),
        )

        accounting = backtest_accounting_summary(res, capital)
        res["_closed_pnl"] = accounting["closed_pnl"]
        res["_open_pnl"] = accounting["open_pnl"]
        res["_net_pnl"] = accounting["net_pnl"]
        res["_accounting_final_equity"] = accounting["final_equity"]
        res["_data_rows"] = int(len(df))
        res["_price_start"] = float(prices_np[0])
        res["_price_end"] = float(prices_np[-1])
        res["_price_return"] = float(prices_np[-1] / prices_np[0] - 1.0)
        res["_strategy_mode"] = "diy"
        res["_max_orders"] = int(len(levels))
        res["_tp"] = float(tp)
        res["_diy_metadata"] = dict(metadata or {})

        first_price = float(prices_np[0])
        bh_qty = float(capital) / (first_price * (1.0 + float(fee_rate)))
        bh_curve = bh_qty * prices_np * (1.0 - float(fee_rate))
        perf = martin.compute_performance_metrics(
            res["equity_curve"],
            res["time_index"],
            res["trades_log"],
            capital=float(capital),
            bh_curve=bh_curve,
            position_curve=res.get("position_curve"),
            open_trade=res.get("open_trade"),
            max_dd_override=res.get("max_dd_overall"),
            annualization_factor=self._interval_bars_per_year(
                interval, source, df
            ),
        )
        expected_total_return = accounting["final_equity"] / float(capital) - 1.0
        if not np.isclose(
            float(perf.get("total_return", np.nan)),
            expected_total_return,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(
                "Performance total-return mismatch: "
                f"metrics={perf.get('total_return')}, expected={expected_total_return}"
            )
        res["bh_curve"] = bh_curve
        res["capital"] = float(capital)
        if df.attrs.get("coverage_complete") is False:
            actual_start_ms = df.attrs.get("actual_start_ms")
            actual_end_ms = df.attrs.get("actual_end_ms")
            if actual_start_ms is not None and actual_end_ms is not None:
                actual_start = pd.Timestamp(
                    int(actual_start_ms), unit="ms", tz="UTC"
                ).tz_convert("Asia/Taipei").strftime("%Y-%m-%d")
                actual_end = pd.Timestamp(
                    int(actual_end_ms), unit="ms", tz="UTC"
                ).tz_convert("Asia/Taipei").strftime("%Y-%m-%d")
                res["_coverage_note"] = (
                    f"Partial market history used: {actual_start} → {actual_end}"
                )
        payload = (df, res, perf)
        self._store_cached_backtest(cache_key, payload)
        return payload

    def _compute_backtest(self, symbol, interval, start, end, refresh_policy,
                          fee_rate, capital, add_drop, multiplier, max_orders, tp,
                          source="binance"):
        cache_key = self._make_backtest_cache_key(
            symbol, interval, start, end, refresh_policy, fee_rate, capital,
            add_drop, multiplier, max_orders, tp, source,
        )
        cached = self._get_cached_backtest(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_klines_if_needed(
            symbol, interval, start, end, refresh_policy, source
        )
        prices_np = df["close"].to_numpy(dtype=np.float64)
        if prices_np.size < 2:
            raise ValueError("K 線資料不足（<2 根）。")

        res = martin.martingale_backtest(
            prices_np, add_drop=float(add_drop), multiplier=float(multiplier),
            max_orders=int(max_orders), tp=float(tp), capital=float(capital),
            return_curve=True, times=df["time"].tolist(), fee_rate=float(fee_rate),
            opens=df["open"].to_numpy(dtype=np.float64),
            highs=df["high"].to_numpy(dtype=np.float64),
            lows=df["low"].to_numpy(dtype=np.float64),
        )

        accounting = backtest_accounting_summary(res, capital)
        res["_closed_pnl"] = accounting["closed_pnl"]
        res["_open_pnl"] = accounting["open_pnl"]
        res["_net_pnl"] = accounting["net_pnl"]
        res["_accounting_final_equity"] = accounting["final_equity"]
        res["_data_rows"] = int(len(df))
        res["_price_start"] = float(prices_np[0])
        res["_price_end"] = float(prices_np[-1])
        res["_price_return"] = float(prices_np[-1] / prices_np[0] - 1.0)

        first_price = float(prices_np[0])
        bh_qty = float(capital) / (first_price * (1.0 + float(fee_rate)))
        bh_curve = bh_qty * prices_np * (1.0 - float(fee_rate))

        perf = martin.compute_performance_metrics(
            res["equity_curve"], res["time_index"], res["trades_log"],
            capital=float(capital), bh_curve=bh_curve,
            position_curve=res.get("position_curve"),
            open_trade=res.get("open_trade"),
            max_dd_override=res.get("max_dd_overall"),
            annualization_factor=self._interval_bars_per_year(
                interval, source, df
            ),
        )
        expected_total_return = accounting["final_equity"] / float(capital) - 1.0
        if not np.isclose(
            float(perf.get("total_return", np.nan)),
            expected_total_return,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(
                "Performance total-return mismatch: "
                f"metrics={perf.get('total_return')}, expected={expected_total_return}"
            )

        res["bh_curve"] = bh_curve
        res["capital"] = float(capital)
        res["_add_drop"] = float(add_drop)
        res["_multiplier"] = float(multiplier)
        res["_max_orders"] = int(max_orders)
        res["_tp"] = float(tp)
        if df.attrs.get("coverage_complete") is False:
            actual_start_ms = df.attrs.get("actual_start_ms")
            actual_end_ms = df.attrs.get("actual_end_ms")
            if actual_start_ms is not None and actual_end_ms is not None:
                actual_start = pd.Timestamp(
                    int(actual_start_ms), unit="ms", tz="UTC"
                ).tz_convert("Asia/Taipei").strftime("%Y-%m-%d")
                actual_end = pd.Timestamp(
                    int(actual_end_ms), unit="ms", tz="UTC"
                ).tz_convert("Asia/Taipei").strftime("%Y-%m-%d")
                res["_coverage_note"] = (
                    f"Partial market history used: {actual_start} → {actual_end}"
                )
        payload = (df, res, perf)
        self._store_cached_backtest(cache_key, payload)
        return payload

    def _render_plot_and_metrics(self, df, res, perf, add_drop, multiplier, max_orders, tp,
                                 figure, canvas, metrics_widget,
                                 mc_terminal=None, mc_summary_text=None):
        equity_curve = res["equity_curve"]
        time_index = res["time_index"]
        trapped_intervals = res.get("trapped_intervals", [])
        bh_curve = res.get("bh_curve", None)
        state = self._get_plot_state(figure, canvas, with_mc=(mc_terminal is not None))
        ax = state["ax"]
        ax_mc = state["ax_mc"]
        strategy_line = state["strategy_line"]
        bh_line = state["bh_line"]

        strategy_line.set_data(time_index, equity_curve)
        if bh_curve is not None:
            bh_line.set_data(df["time"], bh_curve)
            bh_line.set_visible(True)
        else:
            bh_line.set_visible(False)
        configure_datetime_axis(ax, time_index)

        for patch in state["trap_patches"]:
            try:
                patch.remove()
            except Exception:
                pass
        state["trap_patches"] = []
        for s_i, e_i in trapped_intervals:
            if s_i is None or e_i is None:
                continue
            state["trap_patches"].append(ax.axvspan(s_i, e_i, alpha=0.1, color="red"))

        sym = df.attrs.get("symbol", "UNKNOWN")
        interval_str = df.attrs.get("interval", "N/A")
        is_stock = str(df.attrs.get("market_type", "")).lower() == "stock"
        is_diy = str(res.get("_strategy_mode", res.get("strategy_mode", "fixed"))).lower() == "diy"
        market_label = "Stock" if is_stock else "Spot"
        quote_currency = "USD" if is_stock else "USDT"
        if is_diy:
            metadata = res.get("_diy_metadata", {})
            q_end = metadata.get("mae_q_end")
            q_label = (
                f"{100.0 * float(q_end):.0f}%"
                if q_end is not None else "N/A"
            )
            strategy_label = (
                f"DIY TP/Horizon MAE q_end={q_label}, "
                f"gamma={float(metadata.get('capital_gamma', np.nan)):.2f}, "
                f"tp={tp:.3f}, max_orders={int(max_orders)}"
            )
        else:
            strategy_label = (
                f"add_drop={add_drop:.3f}, tp={tp:.3f}, "
                f"mul={multiplier:.1f}, max_orders={int(max_orders)}"
            )
        ax.set_title(
            f"{sym} | {market_label} | exch={df.attrs.get('exchange','?')} | interval={interval_str}\n"
            f"[{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}] "
            f"{strategy_label}",
            fontsize=14,
            pad=12,
        )
        ax.set_xlabel("" if ax_mc is not None else "Time (Taipei)")
        ax.set_ylabel(f"Equity ({quote_currency})")
        ax.relim()
        ax.autoscale_view()
        ax.legend(loc="upper left")

        if ax_mc is not None:
            ax_mc.clear()
            safe_terminal = np.maximum(np.asarray(mc_terminal, dtype=np.float64), 0.0)
            ax_mc.hist(safe_terminal, bins=45, color="#35507a", alpha=0.75)
            ax_mc.axvline(np.median(safe_terminal), color="#8a2d3a", linestyle="--", linewidth=1.5, label="Median")
            ax_mc.set_title("Monte Carlo terminal equity distribution", fontsize=12, pad=10)
            ax_mc.set_xlabel(f"Terminal Equity ({quote_currency})")
            ax_mc.set_ylabel("Count")
            ax_mc.legend(loc="best")
        if not state["use_constrained"]:
            if ax_mc is not None:
                figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10, hspace=0.38)
            else:
                figure.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12)
        canvas.draw_idle()

        trap_ratio = None
        total_secs = None
        trapped_secs = 0.0
        try:
            trapped_mask = np.asarray(res.get("trapped_mask", []), dtype=bool)
            if time_index is not None and len(time_index) >= 2 and trapped_mask.size == len(time_index):
                diffs = pd.Series(pd.DatetimeIndex(time_index)).diff().dropna().dt.total_seconds()
                bar_sec = float(diffs.median()) if len(diffs) else 0.0
                total_secs = bar_sec * len(time_index)
                trapped_secs = bar_sec * int(trapped_mask.sum())
                trap_ratio = float(trapped_mask.mean())
        except Exception:
            trap_ratio = None

        if metrics_widget is not None and isinstance(perf, dict):
            t = []
            if res.get("_coverage_note"):
                t.append(f"=== Data Coverage ===\n{res['_coverage_note']}\n")
            price_start = float(res.get("_price_start", df["close"].iloc[0]))
            price_end = float(res.get("_price_end", df["close"].iloc[-1]))
            price_return = float(res.get("_price_return", price_end / price_start - 1.0))
            final_equity = float(res.get("_accounting_final_equity", equity_curve[-1]))
            closed_pnl = float(res.get("_closed_pnl", 0.0))
            open_pnl = float(res.get("_open_pnl", 0.0))
            net_pnl = float(res.get("_net_pnl", final_equity - float(res.get("capital", 0.0))))
            t.append("=== Data ===")
            t.append(
                f"Period: {df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')} → "
                f"{df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')} | "
                f"Candles: {int(res.get('_data_rows', len(df))):,}"
            )
            t.append(
                f"Price: {price_start:.4f} → {price_end:.4f} | "
                f"Change: {human_pct(price_return)}"
            )

            metadata = {}
            levels = ()
            shares = ()
            quantiles = ()
            fill_probabilities = ()
            share_total = 0.0
            t.append("")
            t.append("=== Strategy Setup ===")
            if is_diy:
                metadata = res.get("_diy_metadata", {})
                levels = tuple(float(v) for v in res.get("level_ratios", ()))
                shares = tuple(float(v) for v in res.get("order_shares", ()))
                quantiles = tuple(
                    float(v) for v in metadata.get("mae_quantiles", ())
                )
                fill_probabilities = tuple(
                    float(v) for v in metadata.get("fill_probabilities", ())
                )
                share_total = float(sum(shares)) if shares else 0.0
                t.append(
                    f"Mode: DIY TP/Horizon MAE | TP: {human_pct(tp)} | "
                    f"Orders: {int(max_orders)} | "
                    f"MAE Horizon: {float(metadata.get('mae_horizon_days', np.nan)):.1f} days"
                )
                t.append(
                    f"MAE q end: "
                    f"{100.0 * float(metadata.get('mae_q_end', np.nan)):.0f}% | "
                    f"Capital Gamma: "
                    f"{float(metadata.get('capital_gamma', np.nan)):.1f}"
                )
            else:
                t.append(
                    f"Mode: Fixed | TP: {human_pct(tp)} | "
                    f"Orders: {int(max_orders)} | "
                    f"Add Drop: {human_pct(add_drop)} | "
                    f"Multiplier: {float(multiplier):.1f}×"
                )

            t.append("")
            t.append("=== Key Performance ===")
            t.append(
                f"Final Equity: {final_equity:.2f} | "
                f"Net PnL: {net_pnl:+.2f} | "
                f"Total Return: {human_pct(perf.get('total_return'))}"
            )
            closed_trades = int(
                perf.get("closed_trades", res.get("trades", len(res.get("trades_log", []))))
            )
            has_open_position = bool(
                perf.get("has_open_position", isinstance(res.get("open_trade"), dict))
            )
            t.append(
                f"CAGR: {human_pct(perf.get('cagr'))} | "
                f"Max DD: {perf.get('max_dd_pct', float('nan')):.2f}% | "
                f"Closed Trades: {closed_trades:,}"
            )
            sharpe = perf.get('sharpe')
            pf = perf.get('profit_factor')
            pf_str = "∞" if (isinstance(pf, (int, float)) and np.isinf(pf)) else (
                "NaN" if pf is None or (isinstance(pf, float) and np.isnan(pf)) else f"{pf:.2f}"
            )
            t.append(
                f"Sharpe: {('%.2f' % sharpe) if sharpe is not None else 'NaN'} | "
                f"Win Rate: {human_pct(perf.get('win_rate'))} | "
                f"Profit Factor: {pf_str}"
            )
            effective_trap_ratio = (
                trap_ratio
                if trap_ratio is not None
                else res.get("trapped_time_ratio")
            )
            t.append(
                f"Trapped: {human_pct(effective_trap_ratio, 2)} | "
                f"Exposure: {human_pct(perf.get('exposure'))} | "
                f"Open Position: {'Yes' if has_open_position else 'No'}"
            )
            if has_open_position:
                t.append(
                    f"Closed/Open PnL: {closed_pnl:+.2f}/{open_pnl:+.2f}"
                )
            if is_diy:
                t.append(
                    f"Capital Use: "
                    f"{human_pct(res.get('avg_capital_utilization'), 2)} | "
                    f"Position Underwater: "
                    f"{human_pct(res.get('underwater_position_ratio'), 2)} | "
                    f"Full Capital: "
                    f"{human_pct(res.get('full_capital_time_ratio'), 2)}"
                )

                t.append("")
                t.append("=== DIY Order Plan ===")
                t.append(
                    f"TP Hit Rate: "
                    f"{human_pct(metadata.get('tp_hit_rate_horizon'), 2)} | "
                    f"Median Bars to TP: "
                    f"{float(metadata.get('median_bars_to_tp', np.nan)):.1f} | "
                    f"Stress Loss: "
                    f"{float(metadata.get('stress_loss_pct', np.nan)):.2f}%"
                )
                t.append(
                    "Order | MAE q | Fill P | Cum.Drop | Shares | Allocation"
                )
                for i, (level, share) in enumerate(zip(levels, shares)):
                    q_text = (
                        "Initial"
                        if i == 0
                        else (
                            f"{100.0 * quantiles[i]:.1f}%"
                            if i < len(quantiles) else "N/A"
                        )
                    )
                    fill_probability = (
                        fill_probabilities[i]
                        if i < len(fill_probabilities) else np.nan
                    )
                    allocation = (
                        100.0 * share / share_total
                        if share_total > 0 else np.nan
                    )
                    t.append(
                        f"{i + 1:>5} | {q_text:>7} | "
                        f"{100.0 * fill_probability:>6.1f}% | "
                        f"{100.0 * (1.0 - level):>7.2f}% | "
                        f"{int(round(share)):>6} | {allocation:>9.2f}%"
                    )
            if "bh_total_return" in perf:
                t.append("\n=== Buy & Hold ===")
                t.append(
                    f"Return: {human_pct(perf.get('bh_total_return'))} | "
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
        for name in ("btn_hist_run", "btn_mc_run", "btn_single_run", "btn_single_mc"):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(True)
        QMessageBox.critical(self, "錯誤", tb)


if __name__ == "__main__":
    app = QApplication([])
    if _import_error is not None:
        QMessageBox.critical(None, "Import Error", f"無法匯入 martin.py：\n{_import_error}")
        raise SystemExit(1)
    w = MartinGUI()
    w.show()
    app.exec()
