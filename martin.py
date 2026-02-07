# -*- coding: utf-8 -*-
"""
馬丁現貨策略（固定樓梯加倉版 + 加速 + 快取 + 手續費）

【與原版差異重點】
- 加倉規則改為【固定樓梯】：
  - 一輪（從開倉到止盈）僅在首單成交時設定一次基準價 base_price。
  - 第 k 層（k=1..max_orders-1）觸發價為 base_price * (1-add_drop)**k。
  - 同一根 K 棒若一次跌破多個樓梯，會連續補多層；補完後不重置基準價。
  - 保留 anchor_price 欄位（相容原結構），但不再用來判斷加倉門檻。
- 其餘：
  - 支援 Parquet 快取與 refresh_policy={"never","auto","force"}
  - 手續費：買入扣 alloc*(1+fee)，賣出收 qty*price*(1-fee)，TP 以含費 PnL 判斷
  - Numba 平行網格加速（_grid_search_parallel）
  - 風險指標與績效統計（Sharpe/Sortino/Calmar 等）

注意：期末未平倉的 final_equity 仍採【不扣賣出費】（與原版一致）。若需更保守，可自行在期末再扣一次賣出費。
"""
import os
import time
import math
import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import ccxt
from numba import njit, prange
import traceback

# ===================== 交易所清單 =====================
_EXCH_LIST = [
    "binance",       # Binance Spot
    "binanceusdm",   # Binance USDT-M Futures (linear swap)
    "okx",
    "bybit",
    "kucoin",
    "kraken",
    "gateio",
    "mexc",
    "huobi",
]

# ===================== Parquet 快取設定 =====================
DEFAULT_CACHE_DIR = "cache"
try:
    import pyarrow  # noqa: F401
    PARQUET_OK = True
except Exception:
    PARQUET_OK = False


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _cache_key(base: str, interval: str) -> str:
    base = base.upper()
    return f"{base}_USDT_{interval}.parquet"


def _load_cached_klines(cache_dir: str, base: str, interval: str):
    """讀取快取並妥善處理 tz、排序/去重、以及 attrs 還原與保留。"""
    if not PARQUET_OK:
        return None
    fpath = os.path.join(cache_dir, _cache_key(base, interval))
    if not os.path.isfile(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)

        # tz-aware / naive 判斷，統一轉 Asia/Taipei
        t = df["time"]
        if isinstance(t.dtype, DatetimeTZDtype):
            df["time"] = pd.DatetimeIndex(t).tz_convert("Asia/Taipei")
        else:
            df["time"] = pd.to_datetime(t, utc=True).tz_convert("Asia/Taipei")

        # 去重 / 排序
        df = df.drop_duplicates(subset="time").sort_values("time")

        # 復原 attrs（若 parquet 有保留）
        attrs = {}
        for k in ("exchange", "market", "symbol", "interval"):
            if k in df.columns and pd.notna(df[k]).any():
                attrs[k] = str(df[k].dropna().iloc[0])

        ret = df[["time", "close", "volume"]].copy()
        if attrs:
            ret.attrs = attrs.copy()
        return ret
    except Exception:
        return None


def _save_cached_klines(cache_dir: str, base: str, interval: str, df: pd.DataFrame, attrs: dict):
    if not PARQUET_OK:
        return
    _ensure_dir(cache_dir)
    fpath = os.path.join(cache_dir, _cache_key(base, interval))
    out = df.copy()
    for k in ("exchange", "market", "symbol", "interval"):
        out[k] = attrs.get(k, None)
    out.attrs = attrs.copy()
    out.to_parquet(fpath, index=False)

# ===================== 工具 =====================

def _interval_ms(interval: str) -> int:
    """把 timeframe 轉成毫秒（m/h/d/w/M；M 以 30 天近似）"""
    unit = interval[-1]
    val = int(interval[:-1])
    seconds_map = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    sec = seconds_map.get(unit)
    if sec is None:
        raise ValueError(f"不支援的 interval: {interval}")
    return val * sec * 1000


def _to_ms(dt_str: str) -> int:
    """把 'YYYY-MM-DD' 或 'YYYY.MM.DD' 轉為 UTC 毫秒"""
    if "." in dt_str:
        dt_str = dt_str.replace(".", "-")
    dt = pd.to_datetime(dt_str, utc=True)
    return int(dt.timestamp() * 1000)


def _align_to_interval_end(now_ms: int, step_ms: int) -> int:
    """對齊到 timeframe 的『最後已收』棒結束時間"""
    return now_ms - (now_ms % step_ms)


def _ohlcv_to_taipei_df(ohlcv):
    """把 ccxt OHLCV 轉成 df[['time','close']] 並轉到 Asia/Taipei；同時去重/排序"""
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["close"] = df["close"].astype(float)
    df["time"]  = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df = df.drop_duplicates(subset="time").sort_values("time")
    df = df.drop_duplicates(subset="time").sort_values("time")
    return df[["time","close","volume"]].copy()


def _find_market_symbol(markets: dict, base: str, prefer_spot=True):
    """
    在 ccxt markets 裡，找 base/USDT：
      1) 優先 spot
      2) 找不到再找 USDT 線性永續 swap（linear=True）
    回傳 (symbol_str, market_type)；market_type ∈ {"spot","swap"} 或 (None, None)
    """
    base = base.upper()
    spot_sym = None
    swap_sym = None
    for m in markets.values():
        if m.get("base") == base and m.get("quote") == "USDT" and m.get("active", True):
            if m.get("spot") and spot_sym is None:
                spot_sym = m["symbol"]
            if m.get("swap") and m.get("linear") and swap_sym is None:
                swap_sym = m["symbol"]
    if prefer_spot and spot_sym:
        return spot_sym, "spot"
    if swap_sym:
        return swap_sym, "swap"
    if spot_sym:
        return spot_sym, "spot"
    return None, None


def _ccxt_fetch_ohlcv_segmented(ex, symbol, timeframe="1h", since_ms=None, end_ms=None, limit=1000, pause=0.12):
    """
    分段抓取 OHLCV：避免一次取太多。回傳 list[[ms,o,h,l,c,v], ...]
    - 若時間沒有前進，中止（避免無窮迴圈）
    - 末尾裁切到 end_ms（含）
    - 尾端以 timestamp 去重（保留最後一次）
    """
    out = []
    cursor = since_ms
    last_seen_ms = -1
    max_calls = 100000

    calls = 0
    while True:
        calls += 1
        if calls > max_calls:
            break
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print(f"[Info] Network error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except Exception:
            raise

        if not batch:
            break
        out.extend(batch)
        last_ms = batch[-1][0]
        if last_ms <= last_seen_ms:
            break
        last_seen_ms = last_ms
        cursor = last_ms + 1  # +1ms 避免邊界跳根
        if end_ms is not None and cursor > end_ms:
            break
        time.sleep(pause)

    if end_ms is not None and out:
        out = [row for row in out if row[0] <= end_ms]
    if out:
        dedup = {row[0]: row for row in out}
        out = [dedup[k] for k in sorted(dedup.keys())]
    return out

# ===================== get_klines（含 Parquet 快取 + refresh_policy） =====================

def get_klines(symbol="ETH", interval="1h", bars=None, start=None, end=None, pause=0.12,
               exch_list=None, prefer_spot=True,
               cache_dir=DEFAULT_CACHE_DIR, use_cache=True, refresh_policy: str = "auto"):
    """
    多交易所抓 'BASE/USDT' 的 K 線，回傳 df[['time','close']]，
    並在 attrs 記錄 market/symbol/interval/exchange。

    refresh_policy:
      - "never": 優先用快取，不做自動補資料；不足再抓
      - "auto" : 自動偵測尾端過期/區間覆蓋，做『增量補到最新』
      - "force": 忽略快取，重抓並覆蓋快取
    """
    if (bars is None) and (start is None and end is None):
        raise ValueError("請提供 bars 或 start/end 其中一種方式。")

    base = symbol.upper()
    step_ms = _interval_ms(interval)
    exnames = exch_list or _EXCH_LIST

    # 讀快取（除非 force）
    cached_df = None
    if use_cache and refresh_policy != "force":
        cached_df = _load_cached_klines(cache_dir, base, interval)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    expected_end_ms = _align_to_interval_end(now_ms, step_ms)

    # ====== 用快取（never/auto）======
    if cached_df is not None and not cached_df.empty and refresh_policy in ("never", "auto"):
        if refresh_policy == "auto":
            ts_utc = pd.DatetimeIndex(cached_df["time"]).tz_convert("UTC")
            cached_min_ms = int(ts_utc[0].timestamp() * 1000)
            cached_max_ms = int(ts_utc[-1].timestamp() * 1000)

            need_refresh = False
            refresh_since = None
            refresh_end = None

            if bars is not None:
                if cached_max_ms + step_ms <= expected_end_ms:
                    need_refresh = True
                    refresh_since = cached_max_ms + step_ms
                    refresh_end = expected_end_ms
            else:
                s_ms = _to_ms(start); e_ms = _to_ms(end)
                if (cached_min_ms > s_ms + step_ms) or (cached_max_ms < e_ms - step_ms):
                    need_refresh = True
                    refresh_since = s_ms
                    refresh_end = e_ms

            if need_refresh:
                try:
                    for name in exnames:
                        ex_cls = getattr(ccxt, name, None)
                        if ex_cls is None:
                            continue
                        ex = ex_cls({"enableRateLimit": True})
                        try:
                            markets = ex.load_markets()
                            m_symbol, m_type = _find_market_symbol(markets, base, prefer_spot=prefer_spot)
                            if not m_symbol:
                                continue
                            ohlcv_new = _ccxt_fetch_ohlcv_segmented(
                                ex, m_symbol, timeframe=interval,
                                since_ms=refresh_since, end_ms=refresh_end, pause=pause
                            )
                            if ohlcv_new:
                                df_new = _ohlcv_to_taipei_df(ohlcv_new)
                                tmp = pd.concat([cached_df, df_new], ignore_index=True)
                                tmp = tmp.drop_duplicates(subset="time").sort_values("time")
                                cached_df = tmp[["time","close","volume"]].copy()

                                attrs = {
                                    "exchange": name,
                                    "market": (f"spot:{name}" if m_type=="spot" else f"usdt_perp:{name}"),
                                    "symbol": f"{base}USDT",
                                    "interval": interval
                                }
                                if use_cache and PARQUET_OK:
                                    _save_cached_klines(cache_dir, base, interval, cached_df, attrs)
                                cached_df.attrs = attrs.copy()
                                break
                        finally:
                            if hasattr(ex, "close"):
                                try: ex.close()
                                except Exception: pass
                except Exception as e:
                    print(f"[Warning] Auto-refresh failed: {e}")
                    # traceback.print_exc()
                    pass  # 補失敗就用原快取

        # 依需求回傳（bars 或 range）
        if bars is not None:
            n = int(bars)
            out = cached_df
            # 再以 expected_end_ms 做一次對齊切齊
            utc_idx = pd.DatetimeIndex(out["time"]).tz_convert("UTC").asi8  # ns
            mask_end = utc_idx <= (expected_end_ms * 1_000_000)  # ns
            out = out.loc[mask_end]
            if len(out) >= n:
                out = out.iloc[-n:].copy()
                base_attrs = getattr(cached_df, "attrs", {})
                out.attrs["symbol"]   = base_attrs.get("symbol",   f"{base}USDT")
                out.attrs["interval"] = base_attrs.get("interval", interval)
                out.attrs["exchange"] = base_attrs.get("exchange", base_attrs.get("exch", None))
                out.attrs["market"]   = base_attrs.get("market",   None)
                return out
            # 快取太短 → 走抓取流程
        else:
            s_utc = pd.to_datetime(start, utc=True)
            e_utc = pd.to_datetime(end,   utc=True)
            utc_idx = pd.DatetimeIndex(cached_df["time"]).tz_convert("UTC")
            mask = (utc_idx >= s_utc) & (utc_idx <= e_utc)
            sliced = cached_df.loc[mask].copy()
            if not sliced.empty:
                base_attrs = getattr(cached_df, "attrs", {})
                sliced.attrs["symbol"]   = base_attrs.get("symbol",   f"{base}USDT")
                sliced.attrs["interval"] = base_attrs.get("interval", interval)
                sliced.attrs["exchange"] = base_attrs.get("exchange", base_attrs.get("exch", None))
                sliced.attrs["market"]   = base_attrs.get("market",   None)
                return sliced
            # 覆蓋不到 → 走抓取流程

    # ====== 走抓取（force 或無快取或快取不足）======
    if bars is not None:
        end_ms = expected_end_ms
        since_ms = end_ms - int(bars) * step_ms
    else:
        since_ms = _to_ms(start)
        end_ms   = _to_ms(end)

    last_error = None
    for name in exnames:
        try:
            ex_cls = getattr(ccxt, name, None)
            if ex_cls is None:
                continue
            ex = ex_cls({"enableRateLimit": True})
            try:
                markets = ex.load_markets()
                m_symbol, m_type = _find_market_symbol(markets, base, prefer_spot=prefer_spot)
                if not m_symbol:
                    continue

                ohlcv = _ccxt_fetch_ohlcv_segmented(
                    ex, m_symbol, timeframe=interval,
                    since_ms=since_ms, end_ms=end_ms, pause=pause
                )
                df = _ohlcv_to_taipei_df(ohlcv)
                if df is None or df.empty:
                    continue

                df.attrs["exchange"] = name
                df.attrs["market"] = f"spot:{name}" if m_type == "spot" else f"usdt_perp:{name}"
                df.attrs["symbol"] = f"{base}USDT"
                df.attrs["interval"] = interval

                if use_cache and PARQUET_OK:
                    _save_cached_klines(cache_dir, base, interval, df, df.attrs)

                if bars is not None and len(df) > int(bars):
                    df = df.iloc[-int(bars):].copy()
                    df.attrs = df.attrs.copy()
                return df
            finally:
                if hasattr(ex, "close"):
                    try: ex.close()
                    except Exception: pass

        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"CCXT 無法在以下交易所找到 {base}/USDT 的 K 線。最後錯誤：{last_error}")


# --- 向下相容薄封裝 ---

def get_klines_lookback(symbol="ETH", interval="1h", bars=4320, pause=0.12,
                        cache_dir=DEFAULT_CACHE_DIR, use_cache=True, refresh_policy="auto"):
    return get_klines(symbol=symbol, interval=interval, bars=bars, start=None, end=None,
                      pause=pause, cache_dir=cache_dir, use_cache=use_cache, refresh_policy=refresh_policy)


def get_klines_range(symbol="ETH", interval="1h", start="2025-01-01", end="2025-08-01", pause=0.12,
                     cache_dir=DEFAULT_CACHE_DIR, use_cache=True, refresh_policy="auto"):
    return get_klines(symbol=symbol, interval=interval, bars=None, start=start, end=end,
                      pause=pause, cache_dir=cache_dir, use_cache=use_cache, refresh_policy=refresh_policy)

# ===================== 策略與績效（單次模擬） =====================

def calc_init_order(capital, multiplier, max_orders):
    """計算首單等比序列的單筆本金（避免 multiplier ≈ 1 的數值不穩）。"""
    if max_orders <= 0:
        raise ValueError("max_orders 必須 > 0")
    if multiplier <= 0:
        raise ValueError("multiplier 必須 > 0")
    if (abs(multiplier - 1.0) < 1e-9) or (max_orders <= 2):
        total_factor = float(max_orders)
    else:
        k = max_orders - 2
        tail_sum = (multiplier**(k + 1) - multiplier) / (multiplier - 1.0)
        total_factor = 2.0 + tail_sum
    return capital / total_factor


def _annualization_factor_from_times(times: list[pd.Timestamp]) -> float:
    if len(times) < 2:
        return 365.25
    diffs = pd.Series(times).diff().dropna().dt.total_seconds()
    bar_sec = float(diffs.median())
    year_sec = 365.25 * 24 * 3600
    return year_sec / bar_sec if bar_sec > 0 else 365.25


def _max_drawdown_with_timing(equity: pd.Series):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min()
    trough_idx = dd.idxmin()
    peak_mask = (equity.loc[:trough_idx] == roll_max.loc[:trough_idx])
    peak_time = peak_mask[peak_mask].index.max()
    trough_time = trough_idx
    recovery_time = None
    post = equity.loc[trough_time:]
    rec = post[post >= roll_max.loc[peak_time]]
    if len(rec) > 0:
        recovery_time = rec.index[0]
    underwater = equity < roll_max
    durations = []
    start = None
    for t, under in underwater.items():
        if under and start is None:
            start = t
        if (not under) and (start is not None):
            durations.append((t - start).total_seconds() / 86400.0)
            start = None
    if start is not None:
        durations.append((equity.index[-1] - start).total_seconds() / 86400.0)
    max_underwater_days = max(durations) if durations else 0.0
    avg_underwater_days = float(np.mean(durations)) if durations else 0.0
    return abs(max_dd*100.0), peak_time, trough_time, recovery_time, max_underwater_days, avg_underwater_days


def _years_between(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    sec = (t1 - t0).total_seconds()
    return sec / (365.25 * 24 * 3600)


def _trade_streaks(pnls: list[float]):
    max_w = max_l = cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            max_w = max(max_w, cur_w)
            cur_l = 0
        elif p < 0:
            cur_l += 1
            max_l = max(max_l, cur_l)
            cur_w = 0
        else:
            cur_w = cur_l = 0
    return max_w, max_l


def compute_performance_metrics(equity_curve, time_index, trades_log, capital, bh_curve=None):
    ec = pd.Series(equity_curve, index=pd.to_datetime(time_index))
    af = _annualization_factor_from_times(ec.index.tolist())
    rets = ec.pct_change().dropna()
    total_return = ec.iloc[-1] / ec.iloc[0] - 1.0
    years = _years_between(ec.index[0], ec.index[-1]) if len(ec) >= 2 else np.nan
    cagr = (ec.iloc[-1] / ec.iloc[0]) ** (1.0 / years) - 1.0 if (pd.notna(years) and years > 0) else np.nan
    mu = rets.mean() * af if len(rets) > 0 else np.nan
    ann_vol = rets.std() * math.sqrt(af) if len(rets) > 1 else np.nan
    sharpe = (mu / ann_vol) if (pd.notna(mu) and ann_vol and ann_vol > 0) else np.nan
    downside = rets[rets < 0]
    ddv = downside.std() * math.sqrt(af) if len(downside) > 0 else np.nan
    sortino = (mu / ddv) if (pd.notna(mu) and ddv and ddv > 0) else np.nan
    max_dd_pct, peak_t, trough_t, recover_t, max_uw_days, avg_uw_days = _max_drawdown_with_timing(ec)
    calmar = (cagr / (max_dd_pct/100.0)) if (pd.notna(cagr) and max_dd_pct > 0) else np.nan
    max_dd_days = ((trough_t - peak_t).total_seconds() / 86400.0) if (pd.notna(peak_t) and pd.notna(trough_t)) else None
    recov_days  = ((recover_t - trough_t).total_seconds() / 86400.0) if (pd.notna(recover_t) and pd.notna(trough_t)) else None
    total_bars = len(ec)
    bars_held_sum = sum(t["bars_held"] for t in trades_log) if trades_log else 0
    exposure = bars_held_sum / total_bars if total_bars > 0 else 0.0
    closed = trades_log
    pnl_list = [t["pnl"] for t in closed]
    win_pnls = [p for p in pnl_list if p > 0]
    loss_pnls = [p for p in pnl_list if p < 0]
    wins = len(win_pnls)
    losses = len(loss_pnls)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan
    avg_win = np.mean(win_pnls) if wins > 0 else np.nan
    avg_loss = np.mean(loss_pnls) if losses > 0 else np.nan
    profit_factor = (sum(win_pnls) / abs(sum(loss_pnls))) if losses > 0 else np.inf
    max_consec_w, max_consec_l = _trade_streaks(pnls=pnl_list)
    rtn_list = [t["rtn"] for t in closed]
    avg_trade_rtn = np.mean(rtn_list) if rtn_list else np.nan
    med_trade_rtn = np.median(rtn_list) if rtn_list else np.nan
    diffs = pd.Series(ec.index).diff().dropna().dt.total_seconds()
    bar_sec = float(diffs.median()) if len(diffs) else 3600.0
    dur_bars = [t["bars_held"] for t in closed]
    avg_bars = np.mean(dur_bars) if dur_bars else np.nan
    med_bars = np.median(dur_bars) if dur_bars else np.nan
    avg_days = (avg_bars * bar_sec) / 86400.0 if not np.isnan(avg_bars) else np.nan
    med_days = (med_bars * bar_sec) / 86400.0 if not np.isnan(med_bars) else np.nan
    bh_stats = {}
    if bh_curve is not None:
        bh = pd.Series(bh_curve, index=ec.index)
        bh_rets = bh.pct_change().dropna()
        bh_mu = bh_rets.mean() * af if len(bh_rets) > 0 else np.nan
        bh_ann_vol = bh_rets.std() * math.sqrt(af) if len(bh_rets) > 1 else np.nan
        bh_sharpe = (bh_mu / bh_ann_vol) if (pd.notna(bh_mu) and bh_ann_vol and bh_ann_vol > 0) else np.nan
        bh_total = bh.iloc[-1]/bh.iloc[0] - 1.0
        bh_years = _years_between(bh.index[0], bh.index[-1]) if len(bh) >= 2 else np.nan
        bh_cagr = (bh.iloc[-1]/bh.iloc[0])**(1.0/bh_years) - 1.0 if (pd.notna(bh_years) and bh_years > 0) else np.nan
        bh_mdd_pct, *_ = _max_drawdown_with_timing(bh)
        bh_stats = {
            "bh_total_return": bh_total, "bh_cagr": bh_cagr,
            "bh_ann_vol": bh_ann_vol, "bh_sharpe": bh_sharpe,
            "bh_max_dd_pct": bh_mdd_pct
        }
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd_pct": max_dd_pct,
        "max_dd_days": None if max_dd_days is None else round(max_dd_days, 1),
        "recovery_days": None if recov_days is None else round(recov_days, 1),
        "max_underwater_days": max_uw_days,
        "avg_underwater_days": avg_uw_days,
        "exposure": exposure,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_consec_wins": max_consec_w,
        "max_consec_losses": max_consec_l,
        "avg_trade_return": avg_trade_rtn,
        "median_trade_return": med_trade_rtn,
        "avg_hold_days": avg_days,
        "median_hold_days": med_days,
        **bh_stats
    }

# ===== 百分比字串工具（0.014 -> "1.4"）=====

def pct_str(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    return f"{x * 100:.{digits}f}"


def martingale_backtest(
    prices,
    add_drop=0.02,
    multiplier=2,
    max_orders=5,  # 『總下單次數（含首筆）』
    tp=0.01,
    capital=1000,
    return_curve=False,
    times=None,
    fee_rate=0.0,   # 由呼叫端傳入（預設不計費用）
):
    """單次模擬（Python 版）— 固定樓梯加倉。
    - base_price：一輪的固定基準價（首單成交價）。
    - next_level_idx：下一層要觀察的台階序號（1..max_orders-1）。
    """
    if add_drop <= 0 or tp <= 0 or max_orders < 1 or multiplier <= 0:
        raise ValueError("參數異常：add_drop,tp需>0，max_orders>=1，multiplier>0")

    cash = capital
    qty = 0.0
    cost = 0.0
    order_count = 0
    init_order_round = None
    entry_time = None
    round_cost_sum = 0.0
    round_fee_sum = 0.0    # 本輪累計買入手續費
    bars_held_this_round = 0
    anchor_price = None    # 相容欄位（不作為門檻）

    # 固定樓梯狀態
    base_price = None
    next_level_idx = 1

    peak_equity_overall = capital
    max_drawdown_overall = 0.0

    trades = 0
    equity_curve, time_curve = [], []
    trades_log = []

    trapped_intervals = []
    start_trapped = None
    is_trapped_prev = False

    for idx, price in enumerate(prices):
        equity = cash + qty * price  # 未平倉不扣賣出費，買入費已在 cash 反映
        if equity > peak_equity_overall:
            peak_equity_overall = equity
        if peak_equity_overall > 0:
            dd_overall = (equity - peak_equity_overall) / peak_equity_overall
            if dd_overall < max_drawdown_overall:
                max_drawdown_overall = dd_overall

        if return_curve:
            equity_curve.append(equity)
            time_curve.append(times[idx] if times is not None else idx)

        # 套牢判定（同原邏輯）
        avg_cost = cost / qty if qty > 0 else float('inf')
        is_trapped = (order_count == max_orders and qty > 0 and price < avg_cost)
        current_time = times[idx] if times is not None else idx
        if is_trapped and not is_trapped_prev:
            start_trapped = current_time
        elif not is_trapped and is_trapped_prev and start_trapped is not None:
            trapped_intervals.append((start_trapped, current_time))
            start_trapped = None
        is_trapped_prev = is_trapped

        # 開倉（設定固定樓梯基準）
        if qty == 0.0 and cash > 0:
            init_order_round = calc_init_order(cash, multiplier, max_orders)
            alloc = min(init_order_round, cash / (1.0 + fee_rate))
            if alloc > 0:
                buy_qty = alloc / price
                qty += buy_qty
                cost += alloc
                fee = alloc * fee_rate
                cash -= (alloc + fee)
                round_fee_sum += fee
                entry_time = times[idx] if times is not None else idx
                anchor_price = price     # 保留但不作觸發
                base_price = price       # 固定樓梯基準價
                next_level_idx = 1
                order_count = 1
                round_cost_sum = alloc
                bars_held_this_round = 0
            continue

        if qty == 0.0:
            continue

        bars_held_this_round += 1

        # 止盈（以含費 PnL 達標為準）
        prospective_proceeds = qty * price * (1.0 - fee_rate)
        prospective_pnl = prospective_proceeds - round_cost_sum - round_fee_sum
        target_pnl = round_cost_sum * tp
        if prospective_pnl >= target_pnl:
            cash += prospective_proceeds
            pnl = prospective_pnl
            rtn = pnl / round_cost_sum if round_cost_sum > 0 else np.nan
            exit_time = times[idx] if times is not None else idx
            trades_log.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "pnl": float(pnl),
                "rtn": float(rtn),
                "bars_held": int(bars_held_this_round)
            })
            qty = 0.0
            cost = 0.0
            anchor_price = None
            trades += 1
            order_count = 0
            init_order_round = None
            round_cost_sum = 0.0
            round_fee_sum = 0.0
            bars_held_this_round = 0
            base_price = None
            next_level_idx = 1
            continue


        # 固定樓梯加倉（O(1)層級計算）：
        # 直接用當前價位相對 base_price 的比值，推回理論應觸發的最高層級 k*
        if qty > 0.0 and order_count < max_orders and cash > 0.0 and base_price is not None:
            r = (1.0 - add_drop)
            if r > 0.0 and price <= base_price:
                ratio = price / base_price if base_price > 0 else 0.0
                if ratio <= 0.0:
                    k_star = max_orders - 1
                else:
                    # 注意 log(r)<0，因此 floor(log(ratio)/log(r)) 為非負整數
                    k_star = int(math.floor(math.log(ratio) / math.log(r)))
                    if k_star < 0:
                        k_star = 0
                    elif k_star > (max_orders - 1):
                        k_star = max_orders - 1
                # 需要新增的層數
                if k_star >= next_level_idx:
                    to_add = min(k_star - next_level_idx + 1, max_orders - order_count)
                    for _ in range(to_add):
                        next_idx = order_count + 1  # 第幾筆訂單（含首單）
                        if next_idx <= 2 or multiplier == 1:
                            factor = 1.0
                        else:
                            factor = multiplier ** (next_idx - 2)
                        target_alloc = init_order_round * factor
                        max_afford = cash / (1.0 + fee_rate)
                        alloc = target_alloc if target_alloc < max_afford else max_afford
                        if alloc <= 0.0:
                            break
                        add_qty = alloc / price
                        qty += add_qty
                        cost += alloc
                        fee = alloc * fee_rate
                        cash -= (alloc + fee)
                        round_cost_sum += alloc
                        round_fee_sum += fee
                        order_count += 1
                        next_level_idx += 1
            # 固定樓梯：不重設 anchor/base

    # 期末：若最後仍在套牢，記錄到結束
    if start_trapped is not None:
        trapped_intervals.append((start_trapped, current_time))

    final_equity = cash + (qty * prices[-1])  # 未平倉不收賣出費（維持原版）
    result = {
        "add_drop": add_drop,
        "multiplier": multiplier,
        "max_orders": max_orders,
        "tp": tp,
        "capital": capital,
        "final_equity": round(final_equity, 2),
        "max_dd_overall": round(abs(max_drawdown_overall * 100), 2),
        "trades": trades
    }
    if return_curve:
        result["equity_curve"] = equity_curve
        result["time_index"] = time_curve
        result["trades_log"] = trades_log
        result["trapped_intervals"] = trapped_intervals
    return result

# ===================== 快速版（平行網格 + 先遮罩） =====================

@njit(cache=True)
def _calc_init_order_numba(capital, multiplier, max_orders):
    if max_orders <= 0 or multiplier <= 0.0:
        return 0.0
    if (abs(multiplier - 1.0) < 1e-12) or (max_orders <= 2):
        total_factor = float(max_orders)
    else:
        k = max_orders - 2
        tail_sum = (multiplier**(k + 1) - multiplier) / (multiplier - 1.0)
        total_factor = 2.0 + tail_sum
    return capital / total_factor


@njit(cache=True)
def _backtest_core(prices, add_drop, multiplier, max_orders, tp, capital, fee_rate):
    """Numba 版單次模擬（固定樓梯加倉）。"""
    trapped_bars = 0
    total_bars = prices.shape[0]

    cash = capital
    qty = 0.0
    cost = 0.0
    order_count = 0
    anchor_price = 0.0  # 相容欄位
    init_order_round = 0.0

    # 固定樓梯狀態
    base_price = 0.0
    next_level_idx = 1

    peak_equity_overall = capital
    max_drawdown_overall = 0.0
    trades = 0
    round_cost_sum = 0.0   # 累計買入本金
    round_fee_sum = 0.0    # 累計買入費

    for i in range(prices.shape[0]):
        price = prices[i]
        equity = cash + qty * price
        if equity > peak_equity_overall:
            peak_equity_overall = equity
        dd = (equity - peak_equity_overall) / peak_equity_overall
        # trapped 條件：
        avg_cost = (cost / qty) if qty > 0.0 else 1e18
        is_trapped = (order_count == max_orders) and (qty > 0.0) and (price < avg_cost)
        if is_trapped:
            trapped_bars += 1
        if dd < max_drawdown_overall:
            max_drawdown_overall = dd

        # 開倉
        if (qty == 0.0) and (cash > 0.0):
            init_order_round = _calc_init_order_numba(cash, multiplier, max_orders)
            alloc = init_order_round
            max_afford = cash / (1.0 + fee_rate)
            if alloc > max_afford:
                alloc = max_afford
            if alloc > 0.0:
                qty += alloc / price
                cost += alloc
                fee = alloc * fee_rate
                cash -= (alloc + fee)
                anchor_price = price
                base_price = price
                next_level_idx = 1
                order_count = 1
                round_cost_sum = alloc
                round_fee_sum = fee
            continue

        if qty == 0.0:
            continue

        # 止盈（以含費 PnL 達標）
        prospective_proceeds = qty * price * (1.0 - fee_rate)
        prospective_pnl = prospective_proceeds - round_cost_sum - round_fee_sum
        target_pnl = round_cost_sum * tp
        if prospective_pnl >= target_pnl:
            cash += prospective_proceeds
            qty = 0.0
            cost = 0.0
            anchor_price = 0.0
            trades += 1
            order_count = 0
            init_order_round = 0.0
            round_cost_sum = 0.0
            round_fee_sum = 0.0
            base_price = 0.0
            next_level_idx = 1
            continue

        # 固定樓梯加倉
        if (qty > 0.0) and (order_count < max_orders) and (cash > 0.0) and (base_price > 0.0):

            # O(1) 計層：由當前價位計算應觸發到的最高層級 k*
            if (order_count < max_orders) and (cash > 0.0) and (base_price > 0.0) and (price <= base_price):
                r = (1.0 - add_drop)
                if r > 0.0:
                    ratio = price / base_price
                    if ratio <= 0.0:
                        k_star = max_orders - 1
                    else:
                        k_star = int(math.floor(math.log(ratio) / math.log(r)))
                        if k_star < 0:
                            k_star = 0
                        elif k_star > (max_orders - 1):
                            k_star = max_orders - 1
                    if k_star >= next_level_idx:
                        to_add = min(k_star - next_level_idx + 1, max_orders - order_count)
                        for _ in range(to_add):
                            next_idx = order_count + 1
                            if next_idx <= 2 or multiplier == 1:
                                factor = 1.0
                            else:
                                factor = multiplier ** (next_idx - 2)
                            target_alloc = init_order_round * factor
                            max_afford = cash / (1.0 + fee_rate)
                            alloc = target_alloc if target_alloc < max_afford else max_afford
                            if alloc <= 0.0:
                                break
                            qty += alloc / price
                            cost += alloc
                            fee = alloc * fee_rate
                            cash -= (alloc + fee)
                            order_count += 1
                            round_cost_sum += alloc
                            round_fee_sum += fee
                            next_level_idx += 1
            # 固定樓梯：不重設 anchor/base


    final_equity = cash + qty * prices[-1]
    mdd_overall_pct = -max_drawdown_overall * 100.0
    trapped_ratio = (trapped_bars / total_bars) if total_bars > 0 else 0.0
    return final_equity, mdd_overall_pct, trades, trapped_ratio


@njit(parallel=True, cache=True)
def _grid_search_parallel(prices_np, add_drop_arr, mul_arr, max_orders_arr, tp_arr, capital, fee_rate):
    n = add_drop_arr.shape[0]
    fe = np.empty(n)
    mdd = np.empty(n)
    tr = np.empty(n)
    trap = np.empty(n)
    for i in prange(n):
        fe_i, mdd_i, tr_i, trap_i = _backtest_core(
            prices_np,
            float(add_drop_arr[i]),
            float(mul_arr[i]),
            int(max_orders_arr[i]),
            float(tp_arr[i]),
            float(capital),
            float(fee_rate)
        )
        fe[i] = fe_i
        mdd[i] = mdd_i
        tr[i] = tr_i
        trap[i] = trap_i
    return fe, mdd, tr, trap

# ===================== 結果過濾工具（只保留必要條件） =====================

def apply_filters(df, min_trades=None, max_dd_overall=None, max_trapped_ratio=None):
    out = df.copy()
    if min_trades is not None:
        out = out[out["trades"] >= int(min_trades)]
    if max_dd_overall is not None:
        out = out[out["max_dd_overall"] <= float(max_dd_overall)]
    if max_trapped_ratio is not None and "trapped_time_ratio" in out.columns:
        out = out[out["trapped_time_ratio"] <= float(max_trapped_ratio)]
    return out
