# Martin Crypto Strategy Toolkit

以馬丁現貨策略為核心的回測與掃描工具，包含：
- 策略回測核心（`martin.py`）
- 參數掃描與單次回測 GUI（`martin_gui.py`）
- 高波動幣種掃描 GUI（`volatility_scanner_gui.py`）
- 命令列測試腳本（`test_strategy.py`）

> 注意：本專案僅供研究與教學用途，非投資建議。實盤風險自負。

## Features
- 固定樓梯加倉（單輪以首單基準價計算層級）
- 手續費與多種風險績效指標（Sharpe/Sortino/Calmar 等）
- Numba 平行網格加速
- Parquet 快取（需 `pyarrow`，未安裝則自動略過）
- GUI 支援：參數掃描、回測圖表、績效統計、結果匯出
- 波動度掃描：依交易量/市值篩選高波動標的

## Requirements
- Python 3.9+（建議）
- 依賴套件：
  - `requests`
  - `ccxt`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `numba`
  - `pyarrow`（選用，用於 Parquet 快取）

安裝：
```bash
pip install -r requirements.txt
```
若要啟用 Parquet 快取：
```bash
pip install pyarrow
```

## Quick Start

### 1) GUI：參數掃描 + 單次回測
```bash
python martin_gui.py
```
可設定：
- 交易所/幣種/週期/時間範圍
- 網格參數（`add_drop`/`tp`/`multiplier`/`max_orders`）
- 過濾條件（交易次數、最大回撤、套牢時間比例）

### 2) GUI：高波動幣種掃描
```bash
python volatility_scanner_gui.py
```
支援：
- 交易所與週期選擇
- 交易量與市值篩選
- 波動度/ATR/布林寬度等指標排名

### 3) 命令列測試腳本
```bash
python test_strategy.py
```
`test_strategy.py` 會：
- 下載指定幣種 K 線
- 平行網格掃描參數
- 顯示 Top 結果
- 以最佳參數繪製策略曲線與 Buy & Hold 對比

## Project Structure
- `martin.py`：策略核心、資料抓取、快取、回測、績效統計
- `martin_gui.py`：參數掃描與單次回測 GUI
- `volatility_scanner_gui.py`：高波動掃描 GUI
- `test_strategy.py`：命令列回測與示例
- `cache/`：資料快取目錄（若啟用）

## Notes
- `refresh_policy` 支援：`never` / `auto` / `force`
- 快取檔案位於 `cache/`，預設為 Parquet 格式（需 `pyarrow`）
- 交易所資料來自 `ccxt`，受交易所限流與可用性影響

## License
如需授權資訊，請自行補上 License 檔案或告知我新增。
