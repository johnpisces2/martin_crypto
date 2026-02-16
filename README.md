# Martin Crypto Strategy Toolkit

馬丁現貨策略研究工具，包含歷史回測、歷史參數掃描、Monte Carlo 風險掃描與可視化 GUI。

> 注意：本專案僅供研究與教學用途，非投資建議。

## 目前功能

### 1) `martin_gui.py`（主 GUI）
三個主分頁：
- `Historical Scan`
- `MC Scan`
- `Single Backtest`

#### Historical Scan
- 以歷史 K 棒資料做全網格參數掃描：`add_drop / tp / multiplier / max_orders`
- 可用條件過濾：`min_trades / max_dd_overall / max_trapped_ratio`
- 顯示 Top N 結果
- 雙擊表格列可進入 `Backtest Chart / Performance` 詳細圖表
- 支援匯出 CSV

#### MC Scan
- 從全域參數空間取樣候選（`LHS(分層抽樣) / Random / Full Grid`）
- 先做歷史條件初篩（`Hist min_trades / Hist max_trap`）
- 對候選進行 block bootstrap Monte Carlo 評估
- 風險約束：`Max P(loss)% / Max P(severe)% / Max P(DD>50)%`
- 支援 `Seed runs`（總路徑數 = `paths * seed_runs`）
- 支援 `Workers` 平行計算
- 結果表格雙擊可進入 `Backtest Chart / Performance` 詳細圖表
- 支援匯出 CSV

#### Single Backtest
- 單一參數歷史回測（策略淨值 vs Buy & Hold）
- 可直接執行單一參數 Monte Carlo
- 顯示績效指標（含 Sharpe/Sortino/Calmar、DD、trapped time ratio 等）

### 2) `martin.py`（策略與資料核心）
- 交易所資料抓取與快取（`ccxt`）
- 馬丁策略回測核心
- 網格掃描加速（numba）
- 績效指標計算

### 3) MC 模組化檔案
- `mc_sampling.py`：候選參數取樣與鄰域 refine
- `mc_eval.py`：平行 Monte Carlo 評估
- `mc_formatters.py`：GUI 顯示格式化

### 4) `volatility_scanner_gui.py`
- 高波動標的掃描 GUI（獨立工具）

---

## 安裝

```bash
pip install -r requirements.txt
```

若要啟用 Parquet 快取（選配）：
```bash
pip install pyarrow
```

---

## 使用方式

### 啟動主 GUI
```bash
python martin_gui.py
```

### 啟動波動掃描 GUI
```bash
python volatility_scanner_gui.py
```

### 命令列範例
```bash
python test_strategy.py
```

---

## Screenshots

### `martin_gui.py`
![martin_gui](screenshot%20martin_gui.png)

### `volatility_scanner_gui.py`
![volatility_scanner_gui](screenshot%20volatility_scanner_gui.png)

---

## 專案結構
- `martin.py`：資料抓取、回測、指標、掃描核心
- `martin_gui.py`：Historical Scan / MC Scan / Single Backtest 主 GUI
- `mc_sampling.py`：MC 候選抽樣
- `mc_eval.py`：MC 平行評估
- `mc_formatters.py`：表格格式化
- `volatility_scanner_gui.py`：波動掃描 GUI
- `test_strategy.py`：命令列範例
- `cache/`：K 線資料快取

---

## 備註
- `refresh_policy`：`never / auto / force`
- 交易所資料可用性、速率限制與品種支援受 `ccxt` 與交易所端影響
