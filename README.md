# Martin 策略研究工具

這個專案提供兩個 PySide6 圖形介面，用來研究 Martin／網格策略：

- `martin_gui.py`：Fixed 等比價格樓梯、TP/Horizon MAE DIY 樓梯、歷史參數掃描、Monte Carlo 與詳細績效圖表。
- `volatility_scanner_gui.py`：先從市場中篩選較可能適合 Martin 策略的標的。

目前支援的行情來源只有：

- Binance：加密貨幣 USDT 現貨。
- Alpaca：美股與 ETF 歷史行情。

Binance 股票代幣、swap、futures 與 Pionex 都不再是支援的資料來源。美股請使用 Alpaca 的原始 ticker，例如 `AAPL`、`NVDA`、`SNDK`、`STRC`，不要使用 `SNDKX`、`NVDAB` 等代幣名稱。

DIY 結果會輸出可手動填入 Pionex DIY Martin Bot 的累積跌幅與 Shares，但專案沒有串接 Pionex API。

> 本工具只讀取歷史行情，不會查詢交易帳戶或送出訂單。所有結果僅供策略研究，不構成投資建議。

## 資料來源

| 項目 | Binance | Alpaca |
| --- | --- | --- |
| 資產 | 加密貨幣現貨 | 美國股票與 ETF |
| Symbol 格式 | base，例如 `BTC`、`XRP` | 原始 ticker，例如 `AAPL`、`BRK.B` |
| 報價幣 | USDT | USD |
| API 憑證 | 公開行情不需要 | 需要 Market Data API Key |
| GUI 週期 | `15m`、`1h`、`4h`、`1d` | `15m`、`1h`、`4h`、`1d` |
| 市場限制 | active USDT crypto spot | Historical Stock Bars |
| 缺棒處理 | 主回測要求連續、完整的 K 線 | 接受週末、休市及不同交易時段造成的正常間隔 |

Alpaca 請求使用 split-adjusted bars，預設 feed 為 `iex`。日期欄位會按完整的紐約日曆日送出請求，包含夏令時間處理；資料回傳後統一轉為 `Asia/Taipei` 時區顯示。

`martin.get_klines()` 未指定來源時仍只會使用 Binance，不會自動切換到 Alpaca。兩個 GUI 則會把使用者選取的 `Source` 明確傳給資料層。

## 安裝與啟動

安裝必要套件：

```bash
pip install -r requirements.txt
```

如需使用跨次啟動的 Parquet K 線快取，再安裝：

```bash
pip install pyarrow
```

啟動主 GUI：

```bash
python martin_gui.py
```

啟動標的篩選器：

```bash
python volatility_scanner_gui.py
```

第一次執行 Numba 回測核心時會進行編譯，因此通常會比後續執行久。

## Alpaca 憑證設定

兩個 GUI 在切換到 Alpaca 或開始 Alpaca 掃描時，會自動讀取：

```text
~/.config/workspace/alpaca.env
```

檔案內容：

```bash
APCA_API_KEY_ID="your-key-id"
APCA_API_SECRET_KEY="your-secret-key"

# 選填；未設定時使用 iex
ALPACA_DATA_FEED="iex"

# 選填；未設定時使用 https://data.alpaca.markets
# ALPACA_DATA_URL="https://data.alpaca.markets"
```

建議限制檔案權限：

```bash
chmod 600 ~/.config/workspace/alpaca.env
```

程式也接受 `ALPACA_API_KEY_ID` 與 `ALPACA_API_SECRET_KEY` 這組別名。若 shell 已經匯出任一組有效憑證，已匯出的值優先，檔案不會覆蓋它。設定檔可使用 `NAME=value`、`export NAME=value`、引號及註解。

GUI 會自行載入這個檔案，不需要先執行 `source`。在某個 terminal 執行 `source` 只會改變該 shell 及其子程序；新開 terminal 看不到先前匯出的變數是正常行為。若直接從其他 Python 程式呼叫 Alpaca adapter，則需自行先呼叫 `alpaca.load_env_file()`，或在啟動程序前匯出環境變數。

憑證只載入目前 GUI 程序，不會寫入 K 線快取。請勿把 API Key 放進 repository 或 commit。

## Symbol 清單

### Binance

主 GUI 目前的建議清單為：

```text
ASTER, BONK, ENA, PEPE, WLD, ZEC, TAO, SUI, HBAR, UNI, NEAR, FIL,
APT, ARB, DOGE, SHIB, ADA, AVAX, LINK, XRP, SOL, LTC, ETH, BNB,
TRX, BCH, BTC
```

這只是方便選取的清單，Symbol 欄位仍可手動輸入其他 Binance active USDT crypto spot base。程式使用精確 denylist 排除股票代幣，不會因名稱以 `B` 結尾就誤刪 `BNB` 等正常加密貨幣。

### Alpaca

`ALPACA_MAINSTREAM_STOCK_SYMBOLS` 目前有 `143` 個去重後、按優先順序排列的股票與 ETF。它是 Scanner 的內建候選 universe，不是 Alpaca API 支援標的的完整名單；可在可編輯的 Symbol 欄位或 Scanner 的 `Manual Include` 輸入清單外 ticker。

主 GUI 下拉選單與 Scanner 預設 `Scan Limit = 50` 使用前 50 個：

```text
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, AMD, TSM,
ARM, QCOM, MU, INTC, SNDK, WDC, JPM, BAC, GS, MS,
V, MA, LLY, UNH, JNJ, ABBV, MRK, COST, WMT, HD,
MCD, XOM, CVX, ORCL, NFLX, CRM, ADBE, PLTR, COIN, HOOD,
MSTR, STRC, CRCL, RKLB, SPY, QQQ, SOXX, SMH, SOXL, GLD
```

主 GUI 的 Alpaca 下拉選單可捲動，也可直接輸入 ticker。Scanner 會先放入 `Manual Include` 的 ticker，再依內建 universe 順序補足 `Scan Limit`；提高 Scan Limit 即可掃描後續標的。

## 主 GUI：`martin_gui.py`

主 GUI 的三個頁籤各自都有 `Source`、`Symbol`、`Interval`、日期與本金欄位。預設為 Binance、`XRP`、`15m`、最近兩年至今天、本金 `1000`，固定回測手續費為單邊 `0.1%`（買入與賣出各計一次）。日期不可選到未來。

### Historical Scan

Historical Scan 可在兩種模式間切換：

- `Fixed Mode`：既有等比價格樓梯與 multiplier 資金倍率。
- `DIY Mode (TP/Horizon MAE)`：依每個 TP 分別建立「無歧義 TP 或觀察期限前 MAE」樓梯及 Pionex Shares。

Fixed Mode 以 `start:end:step` 格式掃描四個既有策略參數：

- `add_drop`：每層價格跌幅。
- `tp`：含費止盈門檻。
- `multiplier`：後續加碼倍率。
- `max_orders`：單輪最大下單次數，包含首單。

掃描欄位的 `add_drop` 與 `tp` 使用小數而非百分數，例如 `0.010:0.080:0.001` 代表 `1.0%` 到 `8.0%`、每次增加 `0.1%`。`max_orders` 必須使用整數範圍。Historical Scan 不接受單一值，仍需寫成 `start:end:step`。

目前預設值：

| 項目 | Fixed Mode | DIY Mode |
| --- | --- | --- |
| `add_drop` | `0.010:0.080:0.001` | 由 TP/Horizon MAE 產生 |
| `tp` | `0.010:0.080:0.001` | `0.010:0.080:0.001` |
| `multiplier` | `1.5:2.0:0.1` | 由 Shares 曲線取代 |
| `max_orders` | `5:12:1` | `5:12:1` |
| `MAE horizon(days)` | 不使用 | `30` |

`min_trades`、`max_dd_overall(%)`、`max_trapped_ratio(%)` 預設全部留空，不限制候選；只套用 `Show Top N = 20`。若手動填入篩選條件，會在依 `final_equity` 排名前先過濾。

DIY Mode 保留 `tp` 與 `max_orders` 掃描，另外只需設定：

- `MAE horizon(days)`：每個歷史進場點向後觀察的完整期間。

TP/Horizon MAE 模式會為每個 TP 尋找含買賣費目標價的第一個「無加倉歧義」觸及點：該棒 `high` 必須觸及目標，而且 `low` 不得低於首單基準價。若 `low` 低於首單價，代表任何一個較近的 DIY Safety Order 都可能先成交；完整回測在補單棒禁止同棒止盈，因此這個 high passage 會被略過並繼續向後搜尋。若 horizon 內沒有無歧義 TP，樣本就使用完整 horizon MAE。這些資料是明確的 TP-or-horizon capped MAE，不是「最終一定到達 TP 前」的未截尾 MAE。搜尋仍使用 sparse-table range query，避免對每個 entry/TP 全區間逐根暴力掃描。

為保持操作簡潔且可重現，下列研究維度已內建，不由 GUI 輸入：

| 內建項目 | 候選／限制 |
| --- | --- |
| MAE 起始分位數 | `50%` |
| MAE 終點分位數 | `90/92/94/96/98%` |
| 首單資金比例 | `1/2/3/4/5%` |
| Capital gamma | `0.8/1.0/1.2/1.4/1.6/1.8` |
| 總 Shares | `100` |
| Stress MAE 分位數 | `99%` |
| 最後一單上限 | `30%` |
| Stress loss 上限 | `50%` |
| TP/Horizon MAE bias | 固定 `0` |

Bias 不再掃描或依觸發率重加權。舊 forward-MAE 模式與 GUI 分支已移除。程式會在配置大型陣列前檢查原始 DIY 候選上限 `2,000,000`，並限制 MAE 輸出矩陣預估值為 `512 MiB`；超過時會要求放大 TP step、縮小範圍或縮短資料期間。Drops、Shares 與 max_orders 完全相同的模板會先去重，不重複執行 Numba 回測。

Fixed 與 DIY 的畫面結果表共用相同順序：

```text
Mode | TP | Orders | Final Equity | Return | Max DD | Trades |
Trapped | Lowest Buy | Capital Use | Setup
```

- `Lowest Buy`：最後一層買入價相對首單基準價的比例。
- `Capital Use`：DIY 在所有回測 K 線上的實際平均本金部署比例；沒有持倉的 K 線以 `0` 計入。Fixed 掃描核心尚未計算此值，因此顯示 `—`。
- `Setup`：Fixed 顯示 `Drop + Multiplier`；DIY 顯示可手動設定到 Pionex 的累積 `Drops + Shares`。

完整的 MAE 分位數、TP hit rate、逐層觸發率、壓力損失與其他診斷仍保留在 DIY CSV。

雙擊結果後，Fixed 與 DIY 都會先顯示相同順序的 `Strategy Setup` 與 `Key Performance`；DIY 的逐層 MAE、觸發率與 Shares 另放在後面的 `DIY Order Plan`。

雙擊結果會切到 `Backtest Chart / Performance`，使用該次掃描保存的資料來源、日期、本金與手續費設定重建詳細回測，避免之後修改畫面欄位導致圖表與表格不一致。

### MC Scan

MC Scan 目前只掃描 Fixed Mode 的 `add_drop/tp/multiplier/max_orders`，不支援 DIY TP/Horizon MAE。計算分成兩階段：

1. 在前段歷史資料以 LHS、Random 或 Full Grid 產生候選參數，套用歷史交易次數與 trapped ratio 篩選，並可在較佳候選附近 refine。
2. 預設保留最後 `30%` 歷史資料建立 block-bootstrap 路徑，評估候選參數的尾端風險。

預設使用：

| 類別 | 預設 |
| --- | --- |
| Parameter grid | 與 Historical Fixed Mode 相同 |
| Sampling | LHS、`Sample size = 5000` |
| Refine | 歷史績效前 `5%` 的相鄰 grid，最多增加 `3000` 組 |
| 歷史篩選 | `min_trades = 104`、`max_trap = 20%` |
| MC | `300` paths、每條 `730` 天、block `672`、seed `42` |
| Robustness | `Seed runs = 1`、holdout `30%`、workers `0`（自動） |
| 風險限制 | `P(loss) ≤ 30%`、`P(severe) ≤ 10%`、`P(DD>50%) ≤ 20%` |
| 排名／顯示 | Median terminal、Top `50` |

`P(loss)` 是期末資產低於初始本金的路徑比例；`P(severe)` 是期末資產低於初始本金 `50%` 的比例。主要輸出包含 terminal median／P5、三個風險機率、平均最大回撤、seed median 標準差、已評估路徑數與是否通過限制。

若候選在尚未跑完全部 paths 前就已確定超過限制，會標示 `mc_early_rejected=True`。這類 row 的不完整 terminal、回撤與 trapped 統計顯示為 `N/A`。風險限制全部設為 `100%` 時使用 one-pass；有實際限制時，survivor 才進入第二輪精確計算 median 與 P5。

`Workers = 0` 會自動使用最多 8 個 process，且不會超過候選數。`block_size` 不可大於 holdout 報酬樣本數，程式不會靜默退化成 IID。

### Single Backtest

Single Backtest 目前只接受一組 Fixed Mode 參數；預設為 `add_drop=5%`、`tp=5%`、`multiplier=2.0`、`max_orders=7`。可執行完整歷史回測，或以同一組參數執行 Monte Carlo；Single Monte Carlo 預設為 `1000` paths、每條 `730` 天、block `672`、seed `42`，並使用整段所選歷史 OHLC 作為 bootstrap 樣本。

詳細績效區目前精簡為：

- 實際資料期間、K 線數、價格變化。
- Strategy Setup。
- Final Equity、Net PnL、Total Return、CAGR、最大回撤與 Closed Trades。
- Sharpe、terminal-adjusted Win Rate、Profit Factor。
- Trapped ratio、Exposure、期末持倉狀態；有未平倉時另外顯示 Closed／Open PnL。
- DIY 詳細回測另外顯示 Capital Use、Position Underwater、Full Capital 與逐層 Order Plan。
- Buy & Hold 的報酬、Sharpe 與最大回撤。

圖表包含 Strategy equity curve、含買賣費的 Buy & Hold benchmark，以及已達最大加碼層數且尚未損益兩平的 trapped intervals。若 Final Equity 無法與 `Capital + Closed PnL + Open PnL` 對帳，或績效 Total Return 不一致，詳細回測會直接報錯，不顯示可能誤導的結果。

## 策略與計算方式

### 固定樓梯加倉

每輪從第一筆成交價 `base_price` 建立固定樓梯，第 `k` 層觸發價為：

```text
base_price * (1 - add_drop) ** k
```

第 1、2 筆使用相同初始 order size，第 3 筆起才依 `multiplier` 遞增。首單金額會預留整個樓梯所需本金與買入手續費。

歷史回測使用完整 OHLC。若同一根 K 的 low 穿越多個樓梯，每一層按自己的 trigger price 成交；若同一根 K 同時碰到補單與 TP，採保守順序：先補單，該根 K 不再止盈。只有觸及 TP 且沒有補單時，才按 TP 價成交。

買入成本為 `alloc * (1 + fee_rate)`，賣出 proceeds 為 `qty * price * (1 - fee_rate)`。每次加倉後會依新的平均成本重算含費目標價：

```text
(deployed_capital + buy_fees + deployed_capital * tp)
/ (position_qty * (1 - sell_fee))
```

因此回測中的 `tp` 是扣除買賣費後、相對本輪已部署本金的目標淨利。期末未平倉部位以扣除假設性賣出費後的保守清算價值計入 Final Equity。

### TP/Horizon MAE DIY 樓梯

首單含費 TP 價格為：

```text
entry_price * (1 + buy_fee + tp) / (1 - sell_fee)
```

對每個 entry 找到 horizon 內第一個「`high` 觸及目標且同棒 `low` 不低於 entry」的位置，並計算在該位置以前的 MAE；可能同棒先加倉的 high passage 不視為可退出。若沒有無歧義 TP，就計算到 horizon。第 `i` 個 Safety Order 由這個 TP-or-horizon MAE 分布的分位數取得累積跌幅 `D_i`，買入價格為 `base_price * (1 - D_i)`。

若首單資金比例為 `u0`、最深層累積跌幅為 `D_max`，累計資金部署曲線為：

```text
U_i = u0 + (1 - u0) * (D_i / D_max) ** gamma
```

每一單的連續權重為相鄰 `U_i` 之差，再以 largest-remainder 修正方式取整為正整數 Shares，總和保持等於 `Total shares`。第一個 Share 是首單，後續才是 Safety Orders；`max_orders` 同樣包含首單。`gamma > 1` 會偏後段配置，`gamma < 1` 會偏前段配置。

壓力損失會在 `Stress MAE q` 對應的價格，依序成交已被穿越的訂單、保留尚未觸發的現金，並以包含假設性賣出費的清算價值計算。這是單一歷史 MAE 壓力點，不等同於完整的未來尾端保證。

若某層累積跌幅為 `D_i`，其經驗觸發率為：

```text
p_i = P(MAE_TP_or_horizon >= D_i)
```

觸發率用於診斷預期資金使用率，不再改寫資金權重；bias 固定為 `0`。所有候選仍依完整 OHLC 回測的 Final Equity 排名。

### 年化與交易日

Binance 依 24/7 市場與實際 interval 年化。Alpaca 依實際觀察到的每個紐約交易日 bars 中位數乘以 `252` 個交易日年化；資料太短時才使用 regular-session fallback：15m=`26`、1h=`7`、4h=`2`、1d=`1` bars/day。Monte Carlo 的天數轉 bars 也使用相同邏輯。

### Monte Carlo 路徑

Bootstrap 不只抽 close-to-close return，也會一起抽樣 open／high／low／close 相對前一根 close 的比例，因此 MC 與歷史回測共用相同的 intrabar 補單、止盈與 drawdown mechanics。

- `block_size <= 1`：獨立抽樣。
- `block_size > 1`：保留局部時間序列結構。
- `Seed runs > 1`：使用獨立 seed batches；風險採較差值，terminal mean／median 採各 batch 平均。

## Volatility Scanner：`volatility_scanner_gui.py`

Scanner 用來先找出較值得進一步做 Historical Scan 或 MC Scan 的標的。快速設定預設為 Binance、`4h`、最近一年、`Scan Limit = 50`。可選最近 180 天、1 年、2 年或自訂日期。

預設畫面只顯示快速設定；勾選 `Show Advanced Settings` 才會展開自訂日期、成交量門檻、Scan Limit、CoinGecko rank 與 `Manual Include`。兩個成交量門檻預設不設限；Binance 的 CoinGecko filter 預設啟用且 Max Rank 為 `100`，切到 Alpaca 時會停用。

### 候選選取

Binance 模式：

- 只掃 active USDT crypto spot。
- 排除穩定幣、股票代幣、swap 與 futures。
- 自動候選依 24 小時 quote volume 排序。
- CoinGecko 市值排名篩選只作用於 Binance；取得排名失敗時會略過該篩選並顯示警告。

Alpaca 模式：

- 依 `ALPACA_MAINSTREAM_STOCK_SYMBOLS` 優先順序選取股票／ETF。
- 不使用 CoinGecko，也不套用 Binance 的 24 小時 volume prefilter。
- `Min Avg Daily Volume` 使用實際歷史 K 線的每日美元成交額。
- 預設 `iex` feed 的成交量口徑不等於全市場 consolidated volume；只有在帳戶具備權限時才應設定 `ALPACA_DATA_FEED=sip`。

兩種來源都可用 `Manual Include` 優先加入標的。手動標的仍必須有資料且至少有 50 根 K 線，但不受 80% 歷史覆蓋率與成交量門檻排除。

### 為什麼結果數少於 Scan Limit

完成狀態會分開顯示：

```text
selected = results + data-filtered + errors
listed = results + data-filtered
```

例如 `results 45/50, data-filtered 3, errors 2` 表示選了 50 個候選，其中 45 個完成指標計算、3 個通過 API 但資料不符合分析條件、2 個在認證／網路／provider／執行階段失敗。

`Data Filtered` 不會再直接消失，而會保留為灰色 row。從 `Show Results` 選 `Data Filtered` 或 `All Results` 可查看 `Reason`，目前包含：

- 指定期間沒有 K 線。
- 少於 50 根 K 線。
- 非手動標的在超過 30 天的請求期間未達約 80% 歷史覆蓋率。
- 非手動標的低於設定的歷史平均每日成交額。

`errors` 與 `Data Filtered` 不同：前者代表資料無法可靠取得或程式執行失敗，不會偽裝成品質篩選結果。表格預設只顯示 `Suitable Only`，因此要查看所有已完成但判定為 Watch／Unsuitable 的標的，需切到 `Suitable + Watch` 或 `All Results`。

### Martin Fit

結果依 `Martin Fit` 由高到低排序，簡化表格顯示：

- `Martin Fit`：0–100 分。
- `Verdict`：Suitable、Watch、Unsuitable 或 Data Filtered。
- `Recovery Rate`：符合跌幅條件後，30 天內回到先前局部高點的比例。
- `Cycles/30D`：每 30 天完成的跌落與回復循環數。
- `Downtrend Risk`：綜合最近 90 天報酬、距高點跌幅、連續下跌與歷史最大回撤。
- `Reason`：判定或資料過濾原因。

參考跌幅會依最近月度 ATR 調整並限制在 2%–5%。分數權重為波動 `25%`、回復行為 `35%`、均值回歸 `20%`、下跌風險 `20%`；嚴重跌勢、深度回撤、低回復率或長時間單向走勢會限制最終分數。

勾選 `Show Advanced Metrics` 可查看年度 RV、月度 ATR、Efficiency Ratio、OHLC 最大回撤、全期與最近 90 天變化、Current DD、連續紅 K、Active%、MaxGap%、歷史平均每日成交額與 CoinGecko rank。選取表格 row 後，下方會繪製該標的價格圖。

Scanner 固定用 8 個 thread 同時取得 K 線，按 `Stop` 可停止尚未完成的工作。

## API 節流、重試與快取

所有行情請求在同一個程序內共用 provider-aware 全域節流器；同一 provider 的 workers 共用 request budget 與 429 暫停狀態，不同 provider 彼此隔離。預設政策：

| Provider | 預算 | 最小請求間隔 | 最大 backoff |
| --- | ---: | ---: | ---: |
| Alpaca | 180 requests／60 秒 | 0.05 秒 | 65 秒 |
| Binance | 1000 weight／60 秒 | 0.01 秒 | 120 秒 |
| CoinGecko | 20 requests／60 秒 | 1 秒 | 65 秒 |

可在啟動程序前覆寫任一 provider：

```bash
export MARTIN_RATE_LIMIT_ALPACA_LIMIT=150
export MARTIN_RATE_LIMIT_ALPACA_WINDOW=60
export MARTIN_RATE_LIMIT_ALPACA_MIN_INTERVAL=0.2
export MARTIN_RATE_LIMIT_ALPACA_MAX_BACKOFF=65
```

命名格式為 `MARTIN_RATE_LIMIT_<PROVIDER>_LIMIT`、`_WINDOW`、`_MIN_INTERVAL`、`_MAX_BACKOFF`。

Alpaca 與 Binance 預設最多重試 5 次，CoinGecko 最多重試 4 次。HTTP 429 會優先遵守 `Retry-After` 或 rate-limit reset header，否則依 provider policy backoff；暫時性網路錯誤也會退避後重試。認證失敗、無 feed 權限及不可重試的資料錯誤會直接列為 error。

安裝 `pyarrow` 後，K 線會存到 `cache/`，檔名依 source、market type、symbol、quote 與 interval 隔離，避免 Binance 與 Alpaca 資料互相誤用。`refresh_policy=auto` 會在可行時增量補齊尾端資料；`force` 會略過快取重抓；`never` 優先使用現有快取。

GUI 另外有程序內記憶體快取。主 GUI 最多保留 8 組行情與 24 組詳細回測；Scanner 的 K 線快取有效 5 分鐘、最多 96 組，ticker cache 為 30 秒，CoinGecko rank cache 為 10 分鐘。

## 測試

執行完整測試：

```bash
pytest -q
```

目前共有 `56` 個測試，涵蓋 Fixed／DIY OHLC 核心一致性、TP/Horizon MAE 同棒順序、矩陣記憶體上限、模板去重、詳細頁 metadata 快取、bias=`0` 內建設定、統一結果表、績效對帳、Monte Carlo、Alpaca 憑證與分頁、紐約日期及夏令時間、全域節流與 retry、Binance 股票代幣排除、Scanner Data Filtered、source routing、掃描設定快照、交易次數與美股年化等回歸案例。

## 主要檔案

- `martin.py`：K 線整合與 Parquet cache、策略回測、Numba 平行 grid core、績效指標。
- `martin_gui.py`：Historical Scan、MC Scan、Single Backtest 主介面。
- `volatility_scanner_gui.py`：Martin Fit 標的篩選、資料品質診斷與圖表。
- `market_data/alpaca.py`：Alpaca 憑證檔、Historical Stock Bars 分頁與重試。
- `market_data/sources.py`：Alpaca 與 Binance／CCXT 的統一 OHLCV adapter。
- `market_data/rate_limit.py`：各 provider 共用但彼此隔離的全域節流器。
- `market_data/coingecko.py`：Binance 候選的 CoinGecko 市值排名 client。
- `market_data/universe.py`：Alpaca 股票／ETF universe、Binance 股票代幣 denylist 與穩定幣規則。
- `mc_sampling.py`：LHS、Random、Full Grid 與 refine neighbors。
- `mc_eval.py`：OHLC block bootstrap、early rejection 與 survivor quantile。
- `mc_formatters.py`：Historical／MC 掃描表格與 CSV 格式。
- `diy_strategy.py`：TP/Horizon MAE sparse-table queries、分位數樓梯、Pionex Shares、觸發率診斷與壓力損失。
- `tests/`：資料層、GUI 與 Scanner 回歸測試。

## 已知限制

- 未建模 slippage、market impact、funding fee、liquidation 或 margin requirement。
- 未套用交易所實際 amount／price precision、最小單量或最小名目金額；本專案不是下單引擎。
- GUI 目前使用固定 `0.1%` 單邊手續費，買入與賣出各計一次；不能代表所有帳戶、標的或市場的實際成本。
- MC Scan 與 Single Backtest 目前只支援 Fixed Mode；DIY TP/Horizon MAE 只存在於 Historical Scan 與其詳細回測。
- Fixed Historical Scan 尚未輸出實際平均 `Capital Use`，因此統一結果表中顯示 `—`。
- Alpaca 可用歷史深度、ticker 與 feed 內容取決於帳戶權限和 provider 回傳；內建 143 檔 universe 不代表完整支援清單。
- `trapped_time_ratio` 是「已達 `max_orders` 且扣除買賣費後仍未損益兩平」的 bar 比例。
- `underwater_position_ratio` 是任何尚未損益兩平持倉的 bar 比例，與只有滿單才計入的 `trapped_time_ratio` 不同。
- TP/Horizon MAE 的 first-passage 仍以首單成本推導 TP；可能同棒加倉的 passage 會被保守略過，加倉後平均成本與實際動態 TP 則由完整 OHLC 回測驗證，尚未做逐候選反覆迭代校準。
- TP/Horizon MAE 是使用者指定 horizon 內的 capped MAE；未在期限內到達無歧義 TP 的路徑不代表其更長期尾端風險已被完整觀察。
- Monte Carlo 只重抽歷史 OHLC 結構，無法保證涵蓋未來 regime shift、停牌、跳空或流動性危機。
- DIY MAE 樓梯若直接使用同一段資料選參數與排名，仍有樣本內偏誤；正式使用前仍應做 walk-forward 與未參與選擇的 OOS 測試。
- Martin／martingale 策略可能在長期單向下跌時持續占用資金並造成重大損失，應另外設定總資金風險與停損規則。
