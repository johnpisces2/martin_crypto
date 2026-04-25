# Martin Crypto Strategy Toolkit

這個專案是一套加密貨幣 `martingale / grid averaging` 策略研究工具，包含歷史回測、參數掃描、`Monte Carlo` 風險評估、GUI 操作介面，以及波動率篩選器。

目前 GUI 預設標的為 `XRP`，可在介面中切換其他幣種。

> 注意：本工具用於策略研究與風險分析，不構成投資建議。回測與 `Monte Carlo` 結果取決於資料品質、模型假設與交易成本設定。

---

## 主要功能

### `martin_gui.py`

主 GUI，包含三個主要工作流：

- `Historical Scan`
- `MC Scan`
- `Single Backtest`

#### Historical Scan

對歷史 K 線進行參數掃描，核心參數包含：

- `add_drop`
- `tp`
- `multiplier`
- `max_orders`

掃描結果會計算：

- `final_equity`
- `max_dd_overall`
- `trades`
- `trapped_time_ratio`
- `min_buy_ratio`

可用篩選條件：

- `min_trades`
- `max_dd_overall`
- `max_trapped_ratio`

雙擊掃描結果可切換到 `Backtest Chart / Performance`，查看策略 equity curve、`Buy & Hold` benchmark、`trapped intervals` 與績效指標。

#### MC Scan

`MC Scan` 用於對候選參數做 `Monte Carlo` 壓力測試。流程分成兩段：

1. 先用歷史資料進行參數候選篩選。
2. 再使用 `block bootstrap` 產生多條模擬價格路徑，評估候選參數的 tail risk。

支援的參數抽樣方式：

- `LHS (Latin Hypercube Sampling)`
- `Random`
- `Full Grid`

風險限制包含：

- `Max P(loss)%`
- `Max P(severe)%`
- `Max P(DD>50)%`

MC 結果欄位包含：

- `mc_terminal_mean`
- `mc_terminal_median`
- `mc_terminal_p5`
- `mc_p_loss`
- `mc_p_severe`
- `mc_p_dd50`
- `mc_mdd_mean`
- `mc_trapped_mean`
- `mc_paths_evaluated`
- `mc_early_rejected`
- `feasible`

`mc_early_rejected=True` 表示該候選參數在尚未跑完整個 `total_paths` 前，風險次數已經足以判定超過限制。這類 row 的 `terminal / mdd / trapped` 類統計會顯示為 `N/A`，避免把未完整估計的數值誤認為完整 MC 結果。

#### Single Backtest

針對單一參數組合執行完整回測，並顯示：

- Strategy equity curve
- `Buy & Hold` benchmark
- `trapped intervals`
- `Sharpe`
- `Sortino`
- `Calmar`
- `Max Drawdown`
- `Drawdown Duration`
- `Win Rate`
- `Profit Factor`
- trade-level statistics

也可對單一參數組合執行 `Monte Carlo` 分析。

---

## 策略核心

### 回測邏輯

策略邏輯位於 `martin.py`。

核心函式：

- `martingale_backtest(...)`：Python 版完整回測，支援 equity curve、trade log、trapped intervals。
- `_backtest_core(...)`：`Numba` 編譯版核心，用於 grid scan 與 MC scan。
- `_grid_search_parallel(...)`：使用 `@njit(parallel=True)` 與 `prange` 平行掃描候選參數。

目前 `martin.py` 是策略 mechanics 的 single source of truth。`mc_eval.py` 不再維護另一份複製的 Numba backtest core，而是直接呼叫 `martin._backtest_core`。

### 交易成本

目前模型包含固定 `fee_rate`：

- Buy：扣除 `alloc * (1 + fee_rate)`
- Sell：以 `qty * price * (1 - fee_rate)` 計算 proceeds
- TP 判斷使用含費 PnL

注意：未平倉部位的 `final_equity` 與 bar-by-bar equity 目前採用 mark-to-market，未額外扣除假設性賣出費。這是目前模型假設，解讀 terminal equity 時需留意。

### 加碼邏輯

每一輪倉位以第一筆成交價格作為 `base_price`，後續補倉層級為：

```text
base_price * (1 - add_drop) ** k
```

當價格跨過多個層級時，核心會用 `O(1)` level calculation 一次計算應觸發到的最高層級，避免逐層檢查。

加碼金額由 `multiplier` 控制：

- 第 1、2 筆使用初始 order size
- 第 3 筆起依 `multiplier` 遞增
- `max_orders` 控制單輪最大下單次數

---

## Monte Carlo 實作

MC 相關檔案：

- `mc_sampling.py`：候選參數抽樣與 refine neighbors
- `mc_eval.py`：MC path 產生與候選評估
- `mc_formatters.py`：GUI 顯示格式化

### Bootstrap path

MC 使用歷史 return 做 `block bootstrap`：

- `block_size <= 1` 時使用 independent return sampling
- `block_size > 1` 時保留局部時間序列結構
- `mc_days` 會依 K 線 interval 轉成 `mc_bars`

### Two-pass survivor quantile

目前 MC 評估採用 hybrid 架構：

1. 若風險限制為預設全開，例如 `Max P(loss)% = 100`、`Max P(severe)% = 100`、`Max P(DD>50)% = 100`，使用 `one-pass streaming`，避免不必要的第二輪計算。
2. 若有實際風險限制，使用 `two-pass survivor quantile`：
   - 第一輪：每條 MC path 只建立一次，對所有仍 active 的候選參數做風險篩選。
   - 若某候選已確定超過風險限制，立即 `early rejection`。
   - 第二輪：只對 survivors 計算精確 `median / p5` terminal quantile。

此設計降低了大量 rejected candidates 時的計算成本，同時保留 survivors 的精確分位數。

---

## 計算優化

### Grid scan

`_backtest_core` 已做過 hot path 微優化：

- 預先計算 `1 / (1 + fee_rate)`
- 預先計算 `1 - fee_rate`
- 預先計算 `log(1 - add_drop)`
- 將 `calc_init_order` 的 total factor 移出 bar loop
- 將補倉倍率由 `multiplier ** n` 改為遞推 `next_order_factor`

`_grid_search_parallel` 使用 `Numba parallel=True` 對候選參數平行計算。

### MC scan

MC scan 已做以下優化：

- `early rejection`
- `two-pass survivor quantile`
- rejected rows 的非完整統計以 `NaN / N/A` 表示
- 共用 `martin._backtest_core`，避免策略邏輯雙份維護

---

## Volatility Scanner

`volatility_scanner_gui.py` 是獨立的波動率篩選工具，可協助尋找適合 grid / martingale 策略研究的幣種。

目前支援的指標包含：

- realized volatility
- ATR%
- Bollinger Band width
- volume / market filters

掃描結果可搭配主 GUI 的 `Historical Scan` 與 `MC Scan` 使用。

---

## 安裝

```bash
pip install -r requirements.txt
```

若要使用 Parquet cache，建議安裝：

```bash
pip install pyarrow
```

---

## 使用方式

啟動主 GUI：

```bash
python martin_gui.py
```

啟動 volatility scanner：

```bash
python volatility_scanner_gui.py
```

執行 CLI smoke test：

```bash
python test_strategy.py
```

---

## 檔案結構

- `martin.py`：資料抓取、cache、策略回測、Numba grid core、績效指標
- `martin_gui.py`：主 GUI，包含 `Historical Scan`、`MC Scan`、`Single Backtest`
- `mc_sampling.py`：MC 參數抽樣與 refine neighbors
- `mc_eval.py`：MC path generation、early rejection、two-pass survivor quantile
- `mc_formatters.py`：MC / historical scan 顯示格式化
- `volatility_scanner_gui.py`：波動率篩選 GUI
- `test_strategy.py`：CLI smoke test
- `requirements.txt`：依賴套件
- `cache/`：K 線 cache 目錄

---

## 已知模型限制

- 尚未建模 `slippage`
- 尚未建模 `funding fee`
- 尚未建模 `liquidation / margin requirement`
- 未平倉 equity 未額外扣除假設性 sell fee
- `trapped_time_ratio` 定義為「已達 `max_orders` 且價格低於 average cost」的時間比例，不等同完整 capital lock ratio
- MC 使用歷史 return bootstrap，不保證涵蓋未來 regime shift

---

## Screenshots

### `martin_gui.py`

![martin_gui](screenshot%20martin_gui.png)

### `volatility_scanner_gui.py`

![volatility_scanner_gui](screenshot%20volatility_scanner_gui.png)
