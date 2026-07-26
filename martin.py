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
- 歷史回測使用 OHLC；補單按樓梯 trigger 成交，同棒補單與 TP 採保守順序
  - Numba 平行網格加速（_grid_search_parallel）
  - 風險指標與績效統計（Sharpe/Sortino/Calmar 等）

期末未平倉的 final_equity 採保守清算價值，包含假設性賣出費。
"""
import os
import math
import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype
from datetime import datetime, timezone
from numba import njit, prange
from market_data import sources as market_sources

# ===================== 交易所清單 =====================
_EXCH_LIST = [
    "binance",  # Binance crypto spot only.
]

# ===================== Parquet 快取設定 =====================
DEFAULT_CACHE_DIR = "cache"
ALLOWED_REFRESH_POLICIES = {"never", "auto", "force"}


class KlineDataUnavailableError(ValueError):
    """The provider request succeeded but no usable bars were available."""


try:
    import pyarrow  # noqa: F401
    PARQUET_OK = True
except Exception:
    PARQUET_OK = False

LOCAL_TIMEZONE = "Asia/Taipei"


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _cache_key(base: str, interval: str, exchange: str, market_type: str) -> str:
    """A cache is valid for exactly one exchange and one market type."""
    base = base.upper()
    safe_exchange = str(exchange).lower().replace("/", "_").replace(":", "_")
    safe_market = str(market_type).lower().replace("/", "_").replace(":", "_")
    quote = "USD" if str(market_type).lower() == "stock" else "USDT"
    return f"{base}_{quote}_{safe_exchange}_{safe_market}_{interval}.parquet"


def _load_cached_klines(cache_dir: str, base: str, interval: str, exchange: str, market_type: str):
    """讀取快取並妥善處理 tz、排序/去重、以及 attrs 還原與保留。"""
    if not PARQUET_OK:
        return None
    fpath = os.path.join(cache_dir, _cache_key(base, interval, exchange, market_type))
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
        for k in ("exchange", "market", "market_type", "symbol", "interval"):
            if k in df.columns and pd.notna(df[k]).any():
                attrs[k] = str(df[k].dropna().iloc[0])
        loaded_market_type = attrs.get("market_type")
        if loaded_market_type is None:
            loaded_market_type = str(market_type)
            attrs["market_type"] = loaded_market_type
        if attrs.get("exchange") != str(exchange) or loaded_market_type != str(market_type):
            return None

        price_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        if "close" not in price_cols:
            return None
        ret = df[["time", *price_cols]].copy()
        if attrs:
            ret.attrs = attrs.copy()
        return ret
    except Exception:
        return None


def _save_cached_klines(
    cache_dir: str, base: str, interval: str, exchange: str, market_type: str,
    df: pd.DataFrame, attrs: dict,
):
    if not PARQUET_OK:
        return
    _ensure_dir(cache_dir)
    fpath = os.path.join(cache_dir, _cache_key(base, interval, exchange, market_type))
    out = df.copy()
    for k in ("exchange", "market", "market_type", "symbol", "interval"):
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


def _to_timestamp(dt_str: str, *, end_of_day: bool = False) -> pd.Timestamp:
    """把使用者輸入的日期/時間解析成 Asia/Taipei 時區 Timestamp。"""
    dt_str = dt_str.strip()
    # Accept the legacy YYYY.MM.DD date separator without corrupting the
    # fractional seconds in an ISO-8601 timestamp such as 23:59:59.999000.
    if len(dt_str) >= 10 and dt_str[4] == "." and dt_str[7] == ".":
        dt_str = dt_str[:10].replace(".", "-") + dt_str[10:]
    date_only = (":" not in dt_str) and ("T" not in dt_str)
    if end_of_day and date_only:
        dt_str = f"{dt_str} 23:59:59.999"
    ts = pd.Timestamp(pd.to_datetime(dt_str))
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TIMEZONE)
    else:
        ts = ts.tz_convert(LOCAL_TIMEZONE)
    return ts


def _to_ms(dt_str: str, *, end_of_day: bool = False) -> int:
    """把使用者輸入的日期/時間轉為 UTC 毫秒。日期格式預設以 Asia/Taipei 解讀。"""
    dt = _to_timestamp(dt_str, end_of_day=end_of_day).tz_convert("UTC")
    return int(dt.timestamp() * 1000)


def _datetime_index_ms(values) -> np.ndarray:
    """Convert timestamps to UTC milliseconds across pandas 2/3 resolutions."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return np.fromiter(
        (int(ts.timestamp() * 1000) for ts in idx), dtype=np.int64, count=len(idx)
    )


def _range_is_covered(df: pd.DataFrame, start_ms: int, end_ms: int, step_ms: int) -> bool:
    if df is None or df.empty:
        return False
    utc_ms = _datetime_index_ms(df["time"])
    first_ms = int(utc_ms[0])
    last_ms = int(utc_ms[-1])
    return (first_ms <= start_ms + step_ms) and (last_ms >= end_ms - step_ms)


def _validate_kline_frame(df: pd.DataFrame, step_ms: int, *, allow_gaps: bool = False):
    """Validate OHLC values and continuous candle-open timestamps."""
    required = {"time", "open", "high", "low", "close"}
    if df is None or df.empty or not required.issubset(df.columns):
        raise ValueError("K 線缺少必要的 time/open/high/low/close 欄位")
    ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    if not np.isfinite(ohlc).all() or np.any(ohlc <= 0.0):
        raise ValueError("K 線包含非有限值或非正價格")
    if np.any(ohlc[:, 1] < np.maximum(ohlc[:, 0], ohlc[:, 3])):
        raise ValueError("K 線 high 小於 open/close")
    if np.any(ohlc[:, 2] > np.minimum(ohlc[:, 0], ohlc[:, 3])):
        raise ValueError("K 線 low 大於 open/close")
    if not allow_gaps and len(df) >= 2:
        utc_ms = _datetime_index_ms(df["time"])
        diffs_ms = np.diff(utc_ms)
        bad = np.flatnonzero(diffs_ms != int(step_ms))
        if bad.size:
            first_gap = int(bad[0])
            raise ValueError(
                f"K 線存在 {bad.size} 個缺棒/不規則間隔；第一處位於 "
                f"{df['time'].iloc[first_gap]} → {df['time'].iloc[first_gap + 1]}"
            )
    return df


def _align_to_interval_end(now_ms: int, step_ms: int) -> int:
    """Return the opening timestamp of the last fully closed candle."""
    return now_ms - (now_ms % step_ms) - step_ms


def _source_market_preferences(source_name):
    if str(source_name or "").strip().lower() == "alpaca":
        return ["stock"]
    return ["spot"]


def _source_market_label(source_name, market_type):
    if str(market_type).lower() == "stock":
        return f"stock:{source_name}"
    return f"spot:{source_name}"


def _source_symbol_label(base, market_type):
    return str(base).upper() if str(market_type).lower() == "stock" else f"{base}USDT"


def _ohlcv_to_taipei_df(ohlcv):
    """把 ccxt OHLCV 轉成 Taipei 時區 DataFrame；同時去重/排序。"""
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["time"]  = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df = df.drop_duplicates(subset="time").sort_values("time")
    return df[["time","open","high","low","close","volume"]].copy()


# Backward-compatible private aliases. Provider details live in market_data.
_find_market_symbol = market_sources.find_market_symbol
_ccxt_fetch_ohlcv_segmented = market_sources.fetch_ccxt_ohlcv_segmented
_fetch_source_ohlcv = market_sources.fetch_source_ohlcv

# ===================== get_klines（含 Parquet 快取 + refresh_policy） =====================

def get_klines(symbol="ETH", interval="1h", bars=None, start=None, end=None, pause=0.12,
               exch_list=None,
               cache_dir=DEFAULT_CACHE_DIR, use_cache=True, refresh_policy: str = "auto",
               allow_gaps=False, require_full_coverage=True,
               allow_partial_sources=()):
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

    refresh_policy = (refresh_policy or "auto").strip().lower()
    if refresh_policy not in ALLOWED_REFRESH_POLICIES:
        allowed = ", ".join(sorted(ALLOWED_REFRESH_POLICIES))
        raise ValueError(f"refresh_policy must be one of: {allowed}")

    partial_sources = {
        str(name).strip().lower() for name in (allow_partial_sources or ())
    }

    # Resolve auto one provider at a time.  This keeps the declared source
    # priority authoritative even when only a fallback provider has a cache.
    if exch_list is None:
        source_errors = []
        for source_name in _EXCH_LIST:
            try:
                return get_klines(
                    symbol=symbol,
                    interval=interval,
                    bars=bars,
                    start=start,
                    end=end,
                    pause=pause,
                    exch_list=[source_name],
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    refresh_policy=refresh_policy,
                    allow_gaps=allow_gaps,
                    require_full_coverage=require_full_coverage,
                    allow_partial_sources=partial_sources,
                )
            except Exception as exc:
                source_errors.append(f"{source_name}: {exc}")
        details = " | ".join(source_errors) if source_errors else "no configured source"
        raise ValueError(f"資料來源無法取得 {str(symbol).upper()}/USDT 的 K 線。{details}")

    base = symbol.upper()
    step_ms = _interval_ms(interval)
    exnames = list(exch_list)

    # 讀取與請求來源完全相同的快取（除非 force）。
    cached_df = None
    cached_exchange = None
    cached_market_type = None
    if use_cache and refresh_policy != "force":
        for name in exnames:
            market_preferences = _source_market_preferences(name)
            for market_type in market_preferences:
                cached_df = _load_cached_klines(
                    cache_dir, base, interval, str(name), market_type
                )
                if cached_df is not None and not cached_df.empty:
                    cached_exchange = str(name)
                    cached_market_type = market_type
                    break
            if cached_df is not None and not cached_df.empty:
                break

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
                s_ms = _to_ms(start)
                e_ms = min(_to_ms(end, end_of_day=True), expected_end_ms)
                if e_ms < s_ms:
                    raise ValueError("指定區間尚無已收盤 K 線")
                cached_requires_full = (
                    require_full_coverage
                    and str(cached_exchange or "").lower() not in partial_sources
                )
                if (not cached_requires_full) and cached_max_ms + step_ms <= e_ms:
                    need_refresh = True
                    refresh_since = max(s_ms, cached_max_ms + step_ms)
                    refresh_end = e_ms
                elif cached_requires_full and not _range_is_covered(cached_df, s_ms, e_ms, step_ms):
                    need_refresh = True
                    refresh_since = s_ms
                    refresh_end = e_ms

            if need_refresh:
                try:
                    refresh_names = [cached_exchange] if cached_exchange else list(exnames)
                    for name in refresh_names:
                        m_symbol, m_type, ohlcv_new = _fetch_source_ohlcv(
                            name,
                            base,
                            interval,
                            refresh_since,
                            refresh_end,
                            pause,
                        )
                        if not m_symbol or (cached_market_type and m_type != cached_market_type):
                            continue
                        if ohlcv_new:
                            df_new = _ohlcv_to_taipei_df(ohlcv_new)
                            tmp = pd.concat([cached_df, df_new], ignore_index=True)
                            tmp = tmp.drop_duplicates(subset="time").sort_values("time")
                            price_cols = [
                                c for c in ("open", "high", "low", "close", "volume")
                                if c in tmp.columns
                            ]
                            cached_df = tmp[["time", *price_cols]].copy()

                            attrs = {
                                "exchange": name,
                                "market": _source_market_label(name, m_type),
                                "market_type": m_type,
                                "symbol": _source_symbol_label(base, m_type),
                                "interval": interval
                            }
                            if use_cache and PARQUET_OK:
                                _save_cached_klines(
                                    cache_dir, base, interval, name, m_type, cached_df, attrs
                                )
                            cached_df.attrs = attrs.copy()
                            break
                except Exception as e:
                    print(f"[Warning] Auto-refresh failed: {e}")
                    # traceback.print_exc()
                    pass  # 補失敗就用原快取

        # 依需求回傳（bars 或 range）
        if bars is not None:
            n = int(bars)
            out = cached_df
            # 再以 expected_end_ms 做一次對齊切齊
            utc_idx_ms = _datetime_index_ms(out["time"])
            mask_end = utc_idx_ms <= expected_end_ms
            out = out.loc[mask_end]
            if len(out) >= n:
                out = out.iloc[-n:].copy()
                base_attrs = getattr(cached_df, "attrs", {})
                out.attrs["symbol"]   = base_attrs.get(
                    "symbol", _source_symbol_label(base, cached_market_type)
                )
                out.attrs["interval"] = base_attrs.get("interval", interval)
                out.attrs["exchange"] = base_attrs.get("exchange", base_attrs.get("exch", None))
                out.attrs["market"]   = base_attrs.get("market",   None)
                out.attrs["market_type"] = base_attrs.get("market_type", cached_market_type)
                try:
                    _validate_kline_frame(out, step_ms, allow_gaps=allow_gaps)
                    return out
                except ValueError as e:
                    print(f"[Warning] Cached K-line validation failed: {e}")
            # 快取太短 → 走抓取流程
        else:
            s_ms = _to_ms(start)
            e_ms = min(_to_ms(end, end_of_day=True), expected_end_ms)
            if e_ms < s_ms:
                raise ValueError("指定區間尚無已收盤 K 線")
            s_utc = _to_timestamp(start).tz_convert("UTC")
            e_utc = pd.Timestamp(e_ms, unit="ms", tz="UTC")
            utc_idx = pd.DatetimeIndex(cached_df["time"]).tz_convert("UTC")
            mask = (utc_idx >= s_utc) & (utc_idx <= e_utc)
            sliced = cached_df.loc[mask].copy()
            cached_requires_full = (
                require_full_coverage
                and str(cached_exchange or "").lower() not in partial_sources
            )
            coverage_complete = _range_is_covered(sliced, s_ms, e_ms, step_ms)
            if (not cached_requires_full and not sliced.empty) or coverage_complete:
                base_attrs = getattr(cached_df, "attrs", {})
                sliced.attrs["symbol"]   = base_attrs.get(
                    "symbol", _source_symbol_label(base, cached_market_type)
                )
                sliced.attrs["interval"] = base_attrs.get("interval", interval)
                sliced.attrs["exchange"] = base_attrs.get("exchange", base_attrs.get("exch", None))
                sliced.attrs["market"]   = base_attrs.get("market",   None)
                sliced.attrs["market_type"] = base_attrs.get("market_type", cached_market_type)
                sliced.attrs["coverage_complete"] = bool(coverage_complete)
                sliced.attrs["requested_start_ms"] = int(s_ms)
                sliced.attrs["requested_end_ms"] = int(e_ms)
                if not sliced.empty:
                    actual_ms = _datetime_index_ms(sliced["time"])
                    sliced.attrs["actual_start_ms"] = int(actual_ms[0])
                    sliced.attrs["actual_end_ms"] = int(actual_ms[-1])
                try:
                    _validate_kline_frame(sliced, step_ms, allow_gaps=allow_gaps)
                    return sliced
                except ValueError as e:
                    print(f"[Warning] Cached K-line validation failed: {e}")
            # 覆蓋不到 → 走抓取流程

    # ====== 走抓取（force 或無快取或快取不足）======
    if bars is not None:
        end_ms = expected_end_ms
        n_bars = int(bars)
        if n_bars <= 0:
            raise ValueError("bars 必須 > 0")
        since_ms = end_ms - (n_bars - 1) * step_ms
    else:
        since_ms = _to_ms(start)
        end_ms = min(_to_ms(end, end_of_day=True), expected_end_ms)
        if end_ms < since_ms:
            raise ValueError("指定區間尚無已收盤 K 線")

    last_error = None
    saw_source_without_data = False
    for name in exnames:
        try:
            m_symbol, m_type, ohlcv = _fetch_source_ohlcv(
                name,
                base,
                interval,
                since_ms,
                end_ms,
                pause,
            )
            if not m_symbol:
                saw_source_without_data = True
                continue
            df = _ohlcv_to_taipei_df(ohlcv)
            if df is None or df.empty:
                saw_source_without_data = True
                continue

            df.attrs["exchange"] = name
            df.attrs["market"] = _source_market_label(name, m_type)
            df.attrs["market_type"] = m_type
            df.attrs["symbol"] = _source_symbol_label(base, m_type)
            df.attrs["interval"] = interval

            if bars is not None and len(df) > int(bars):
                df = df.iloc[-int(bars):].copy()
                df.attrs = df.attrs.copy()
            _validate_kline_frame(df, step_ms, allow_gaps=allow_gaps)
            source_requires_full = (
                require_full_coverage and str(name).lower() not in partial_sources
            )
            if source_requires_full:
                if bars is not None and len(df) < int(bars):
                    raise ValueError(f"K 線僅取得 {len(df)}/{int(bars)} 根")
                if bars is None and not _range_is_covered(df, since_ms, end_ms, step_ms):
                    raise ValueError("K 線未完整覆蓋指定起訖區間")
            if bars is not None:
                coverage_complete = len(df) >= int(bars)
            else:
                coverage_complete = _range_is_covered(df, since_ms, end_ms, step_ms)
            df.attrs["coverage_complete"] = bool(coverage_complete)
            df.attrs["requested_start_ms"] = int(since_ms)
            df.attrs["requested_end_ms"] = int(end_ms)
            actual_ms = _datetime_index_ms(df["time"])
            df.attrs["actual_start_ms"] = int(actual_ms[0])
            df.attrs["actual_end_ms"] = int(actual_ms[-1])
            if use_cache and PARQUET_OK:
                _save_cached_klines(
                    cache_dir, base, interval, name, m_type, df, df.attrs
                )
            return df

        except Exception as e:
            last_error = e
            continue

    if last_error is None and (saw_source_without_data or not exnames):
        quote = "USD" if exnames and all(
            str(name).strip().lower() == "alpaca" for name in exnames
        ) else "USDT"
        raise KlineDataUnavailableError(
            f"資料來源沒有 {base}/{quote} 在指定區間的 K 線資料"
        )
    raise ValueError(f"資料來源無法取得 {base}/USDT 的 K 線。最後錯誤：{last_error}")


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

def _order_factor_sum(multiplier: float, max_orders: int) -> float:
    """Return 1 + 1 + m + m^2 ... without the cancellation near m == 1."""
    if max_orders <= 0:
        raise ValueError("max_orders 必須 > 0")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier 必須為有限正數")

    total_factor = 0.0
    factor = 1.0
    for order_no in range(1, int(max_orders) + 1):
        if order_no >= 3:
            factor *= multiplier
        total_factor += factor
        if not math.isfinite(total_factor):
            raise ValueError("multiplier/max_orders 組合過大，資金倍率已溢位")
    return total_factor


def calc_init_order(capital, multiplier, max_orders, fee_rate=0.0):
    """計算首單本金，並預留整個樓梯所有買入手續費。"""
    if not math.isfinite(float(capital)) or capital <= 0:
        raise ValueError("capital 必須為有限正數")
    if not math.isfinite(float(fee_rate)) or not (0.0 <= fee_rate < 1.0):
        raise ValueError("fee_rate 必須滿足 0 <= fee_rate < 1")
    total_factor = _order_factor_sum(float(multiplier), int(max_orders))
    return float(capital) / ((1.0 + float(fee_rate)) * total_factor)


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


def _matches_capital_baseline(value: float, capital: float) -> bool:
    """Use a strict comparison so fees never erase the true capital baseline."""
    return math.isclose(
        float(value),
        float(capital),
        rel_tol=1e-12,
        abs_tol=1e-9,
    )


def compute_performance_metrics(
    equity_curve,
    time_index,
    trades_log,
    capital,
    bh_curve=None,
    position_curve=None,
    open_trade=None,
    max_dd_override=None,
    annualization_factor=None,
):
    if not math.isfinite(float(capital)) or capital <= 0:
        raise ValueError("capital 必須為有限正數")
    if len(equity_curve) != len(time_index):
        raise ValueError("equity_curve 與 time_index 長度必須一致")
    ec = pd.Series(equity_curve, index=pd.to_datetime(time_index))
    if ec.empty:
        raise ValueError("equity_curve must contain at least 1 point")
    if not np.isfinite(ec.to_numpy(dtype=np.float64)).all() or np.any(ec.to_numpy(dtype=np.float64) <= 0):
        raise ValueError("equity_curve 必須為有限正數")
    if not ec.index.is_monotonic_increasing or ec.index.has_duplicates:
        raise ValueError("time_index 必須嚴格遞增且不可重複")

    ec_stats = ec
    if capital > 0 and not _matches_capital_baseline(ec.iloc[0], capital):
        first_ts = ec.index[0]
        if len(ec.index) >= 2:
            inferred_step = ec.index[1] - ec.index[0]
            if inferred_step <= pd.Timedelta(0):
                inferred_step = pd.Timedelta(seconds=1)
        else:
            inferred_step = pd.Timedelta(seconds=1)
        baseline_ts = first_ts - inferred_step
        ec_stats = pd.concat([pd.Series([float(capital)], index=[baseline_ts]), ec])

    if annualization_factor is None:
        af = _annualization_factor_from_times(ec_stats.index.tolist())
    else:
        af = float(annualization_factor)
        if not math.isfinite(af) or af <= 0:
            raise ValueError("annualization_factor 必須為有限正數")
    rets = ec_stats.pct_change().dropna()
    total_return = ec_stats.iloc[-1] / ec_stats.iloc[0] - 1.0
    years = _years_between(ec_stats.index[0], ec_stats.index[-1]) if len(ec_stats) >= 2 else np.nan
    cagr = (ec_stats.iloc[-1] / ec_stats.iloc[0]) ** (1.0 / years) - 1.0 if (pd.notna(years) and years > 0) else np.nan
    mu = rets.mean() * af if len(rets) > 0 else np.nan
    ann_vol = rets.std() * math.sqrt(af) if len(rets) > 1 else np.nan
    sharpe = (mu / ann_vol) if (pd.notna(mu) and ann_vol and ann_vol > 0) else np.nan
    downside = rets[rets < 0]
    ddv = downside.std() * math.sqrt(af) if len(downside) > 0 else np.nan
    sortino = (mu / ddv) if (pd.notna(mu) and ddv and ddv > 0) else np.nan
    max_dd_pct, peak_t, trough_t, recover_t, max_uw_days, avg_uw_days = _max_drawdown_with_timing(ec_stats)
    if max_dd_override is not None and np.isfinite(float(max_dd_override)):
        # The execution engine can observe an intrabar low that is not present in
        # the close-only equity curve. Keep timing close-based but report the
        # more conservative magnitude.
        max_dd_pct = max(float(max_dd_pct), float(max_dd_override))
    calmar = (cagr / (max_dd_pct/100.0)) if (pd.notna(cagr) and max_dd_pct > 0) else np.nan
    max_dd_days = ((trough_t - peak_t).total_seconds() / 86400.0) if (pd.notna(peak_t) and pd.notna(trough_t)) else None
    recov_days  = ((recover_t - trough_t).total_seconds() / 86400.0) if (pd.notna(recover_t) and pd.notna(trough_t)) else None
    total_bars = len(ec)
    if position_curve is not None:
        pos = np.asarray(position_curve, dtype=bool)
        if pos.shape[0] != total_bars:
            raise ValueError("position_curve 長度必須與 equity_curve 相同")
        exposure = float(np.mean(pos)) if total_bars > 0 else 0.0
    else:
        bars_held_sum = sum(t["bars_held"] for t in trades_log) if trades_log else 0
        exposure = bars_held_sum / total_bars if total_bars > 0 else 0.0

    closed = list(trades_log or [])
    outcomes = list(closed)
    if isinstance(open_trade, dict) and np.isfinite(float(open_trade.get("pnl", np.nan))):
        outcomes.append(open_trade)
    pnl_list = [float(t["pnl"]) for t in outcomes]
    win_pnls = [p for p in pnl_list if p > 0]
    loss_pnls = [p for p in pnl_list if p < 0]
    wins = len(win_pnls)
    losses = len(loss_pnls)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan
    avg_win = np.mean(win_pnls) if wins > 0 else np.nan
    avg_loss = np.mean(loss_pnls) if losses > 0 else np.nan
    if wins == 0 and losses == 0:
        profit_factor = np.nan
    elif losses == 0:
        profit_factor = np.inf
    else:
        profit_factor = sum(win_pnls) / abs(sum(loss_pnls))
    max_consec_w, max_consec_l = _trade_streaks(pnls=pnl_list)
    rtn_list = [t["rtn"] for t in outcomes]
    avg_trade_rtn = np.mean(rtn_list) if rtn_list else np.nan
    med_trade_rtn = np.median(rtn_list) if rtn_list else np.nan
    diffs = pd.Series(ec.index).diff().dropna().dt.total_seconds()
    bar_sec = float(diffs.median()) if len(diffs) else 3600.0
    dur_bars = [t["bars_held"] for t in outcomes]
    avg_bars = np.mean(dur_bars) if dur_bars else np.nan
    med_bars = np.median(dur_bars) if dur_bars else np.nan
    avg_days = (avg_bars * bar_sec) / 86400.0 if not np.isnan(avg_bars) else np.nan
    med_days = (med_bars * bar_sec) / 86400.0 if not np.isnan(med_bars) else np.nan
    bh_stats = {}
    if bh_curve is not None:
        if len(bh_curve) != len(ec):
            raise ValueError("bh_curve 長度必須與 equity_curve 相同")
        bh = pd.Series(np.asarray(bh_curve, dtype=np.float64), index=ec.index)
        if not np.isfinite(bh.to_numpy()).all() or np.any(bh.to_numpy() <= 0):
            raise ValueError("bh_curve 必須為有限正數")
        bh_stats_curve = bh
        if not _matches_capital_baseline(bh.iloc[0], capital):
            if len(bh.index) >= 2:
                inferred_step = bh.index[1] - bh.index[0]
                if inferred_step <= pd.Timedelta(0):
                    inferred_step = pd.Timedelta(seconds=1)
            else:
                inferred_step = pd.Timedelta(seconds=1)
            baseline_ts = bh.index[0] - inferred_step
            bh_stats_curve = pd.concat([
                pd.Series([float(capital)], index=[baseline_ts]), bh
            ])
        bh_rets = bh_stats_curve.pct_change().dropna()
        bh_mu = bh_rets.mean() * af if len(bh_rets) > 0 else np.nan
        bh_ann_vol = bh_rets.std() * math.sqrt(af) if len(bh_rets) > 1 else np.nan
        bh_sharpe = (bh_mu / bh_ann_vol) if (pd.notna(bh_mu) and bh_ann_vol and bh_ann_vol > 0) else np.nan
        bh_total = bh_stats_curve.iloc[-1]/bh_stats_curve.iloc[0] - 1.0
        bh_years = _years_between(
            bh_stats_curve.index[0], bh_stats_curve.index[-1]
        ) if len(bh_stats_curve) >= 2 else np.nan
        bh_cagr = (
            (bh_stats_curve.iloc[-1]/bh_stats_curve.iloc[0])**(1.0/bh_years) - 1.0
            if (pd.notna(bh_years) and bh_years > 0) else np.nan
        )
        bh_mdd_pct, *_ = _max_drawdown_with_timing(bh_stats_curve)
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
        "closed_trades": len(closed),
        "has_open_position": bool(open_trade),
        "open_trade_pnl": float(open_trade["pnl"]) if isinstance(open_trade, dict) else np.nan,
        **bh_stats
    }

# ===== 百分比字串工具（0.014 -> "1.4"）=====

def pct_str(x: float, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "NaN"
    if np.isinf(float(x)):
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
    opens=None,
    highs=None,
    lows=None,
):
    """固定樓梯加倉回測。

    歷史模式可傳入 OHLC。補單按各樓梯價成交；若同一根 K 同時觸發
    補單與 TP，採保守順序：先補單，且該根不再止盈。只提供 prices
    時會使用 O=H=L=C，僅供 close-path 向下相容使用；目前 GUI 的 MC
    會另外重抽樣歷史 OHLC 形狀並直接呼叫 OHLC 核心。
    """
    closes = np.asarray(prices, dtype=np.float64)
    if closes.ndim != 1 or closes.size == 0:
        raise ValueError("prices must be a non-empty 1-D array")
    opens_np = closes if opens is None else np.asarray(opens, dtype=np.float64)
    highs_np = closes if highs is None else np.asarray(highs, dtype=np.float64)
    lows_np = closes if lows is None else np.asarray(lows, dtype=np.float64)
    if any(x.ndim != 1 or x.size != closes.size for x in (opens_np, highs_np, lows_np)):
        raise ValueError("open/high/low/close 長度必須一致")
    if not all(np.isfinite(x).all() for x in (opens_np, highs_np, lows_np, closes)):
        raise ValueError("OHLC 不可包含 NaN/Inf")
    if not all((x > 0.0).all() for x in (opens_np, highs_np, lows_np, closes)):
        raise ValueError("OHLC 價格必須全部 > 0")
    if np.any(highs_np < np.maximum(opens_np, closes)) or np.any(lows_np > np.minimum(opens_np, closes)):
        raise ValueError("OHLC 關係異常：high/low 未包住 open/close")
    if np.any(highs_np < lows_np):
        raise ValueError("OHLC 關係異常：high < low")
    if not all(math.isfinite(float(x)) for x in (add_drop, tp, multiplier, capital, fee_rate)):
        raise ValueError("策略參數必須為有限數")
    if not (0.0 < add_drop < 1.0) or tp <= 0 or max_orders < 1 or multiplier <= 0:
        raise ValueError("參數需滿足：0 < add_drop < 1、tp > 0、max_orders >= 1、multiplier > 0")
    if capital <= 0:
        raise ValueError("capital 必須 > 0")
    if not (0.0 <= fee_rate < 1.0):
        raise ValueError("fee_rate 必須滿足 0 <= fee_rate < 1")
    time_values = None
    if times is not None:
        if len(times) != closes.size:
            raise ValueError("times 長度必須與 prices 相同")
        time_idx = pd.DatetimeIndex(times)
        if len(times) >= 2:
            if not time_idx.is_monotonic_increasing or time_idx.has_duplicates:
                raise ValueError("times 必須嚴格遞增且不可重複")
        time_values = time_idx.tolist()

    cash = capital
    qty = 0.0
    order_count = 0
    init_order_round = None
    entry_time = None
    round_cost_sum = 0.0
    round_fee_sum = 0.0    # 本輪累計買入手續費
    bars_held_this_round = 0
    base_price = None
    next_level_price = None
    next_order_factor = 1.0
    target_exit_price = None

    peak_equity_overall = capital
    max_drawdown_overall = 0.0

    trades = 0
    equity_curve, time_curve = [], []
    trades_log = []
    position_curve = []
    trapped_mask = []

    trapped_intervals = []
    start_trapped = None
    is_trapped_prev = False

    r = 1.0 - float(add_drop)
    sell_fee_mult = 1.0 - float(fee_rate)

    for idx, price in enumerate(closes):
        current_time = time_values[idx] if time_values is not None else idx
        high = float(highs_np[idx])
        low = float(lows_np[idx])
        low_equity = None

        # 開倉（設定固定樓梯基準）
        if qty == 0.0 and cash > 0:
            init_order_round = calc_init_order(cash, multiplier, max_orders, fee_rate=fee_rate)
            alloc = min(init_order_round, cash / (1.0 + fee_rate))
            if alloc > 0:
                buy_qty = alloc / price
                qty += buy_qty
                fee = alloc * fee_rate
                cash -= (alloc + fee)
                round_fee_sum += fee
                entry_time = current_time
                base_price = float(price)
                next_level_price = base_price * r
                next_order_factor = 1.0
                order_count = 1
                round_cost_sum = alloc
                bars_held_this_round = 0
                target_exit_price = (
                    round_cost_sum + round_fee_sum + round_cost_sum * tp
                ) / (qty * sell_fee_mult)
        elif qty > 0.0:
            bars_held_this_round += 1
            added_this_bar = False

            # 所有已掛樓梯按各自 trigger price 成交。用極小 tolerance
            # 僅吸收浮點乘法誤差，不改變實際門檻。
            while (
                order_count < max_orders
                and cash > 0.0
                and next_level_price is not None
                and low <= next_level_price * (1.0 + 1e-12)
            ):
                target_alloc = init_order_round * next_order_factor
                max_afford = cash / (1.0 + fee_rate)
                alloc = min(target_alloc, max_afford)
                if alloc <= 0.0:
                    break
                fill_price = next_level_price
                qty += alloc / fill_price
                fee = alloc * fee_rate
                cash -= alloc + fee
                round_cost_sum += alloc
                round_fee_sum += fee
                order_count += 1
                added_this_bar = True
                next_level_price *= r
                if multiplier != 1.0 and order_count >= 2:
                    next_order_factor *= multiplier

            if qty > 0.0:
                target_exit_price = (
                    round_cost_sum + round_fee_sum + round_cost_sum * tp
                ) / (qty * sell_fee_mult)
                low_equity = cash + qty * low * sell_fee_mult

            # 同棒有補單時不允許立即止盈，避免未知 OHLC 路徑造成樂觀偏誤。
            if (not added_this_bar) and target_exit_price is not None and high >= target_exit_price:
                prospective_proceeds = qty * target_exit_price * sell_fee_mult
                prospective_pnl = prospective_proceeds - round_cost_sum - round_fee_sum
                cash += prospective_proceeds
                pnl = prospective_pnl
                rtn = pnl / round_cost_sum if round_cost_sum > 0 else np.nan
                exit_time = current_time
                trades_log.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "pnl": float(pnl),
                    "rtn": float(rtn),
                    "bars_held": int(bars_held_this_round)
                })
                qty = 0.0
                trades += 1
                order_count = 0
                init_order_round = None
                round_cost_sum = 0.0
                round_fee_sum = 0.0
                bars_held_this_round = 0
                base_price = None
                next_level_price = None
                next_order_factor = 1.0
                target_exit_price = None

        if low_equity is not None and peak_equity_overall > 0:
            dd_low = (low_equity - peak_equity_overall) / peak_equity_overall
            if dd_low < max_drawdown_overall:
                max_drawdown_overall = dd_low

        # 用可清算淨值計價，包含假設性賣出費。
        equity = cash + qty * price * sell_fee_mult
        if equity > peak_equity_overall:
            peak_equity_overall = equity
        if peak_equity_overall > 0:
            dd_overall = (equity - peak_equity_overall) / peak_equity_overall
            if dd_overall < max_drawdown_overall:
                max_drawdown_overall = dd_overall

        if return_curve:
            equity_curve.append(equity)
            time_curve.append(current_time)
            position_curve.append(qty > 0.0)

        is_trapped = (
            order_count == max_orders
            and qty > 0.0
            and qty * price * sell_fee_mult < (round_cost_sum + round_fee_sum)
        )
        trapped_mask.append(bool(is_trapped))
        if is_trapped and not is_trapped_prev:
            start_trapped = current_time
        elif not is_trapped and is_trapped_prev and start_trapped is not None:
            trapped_intervals.append((start_trapped, current_time))
            start_trapped = None
        is_trapped_prev = is_trapped

    # 期末：若最後仍在套牢，記錄到結束
    if start_trapped is not None:
        if time_values is not None and len(time_values) >= 2:
            terminal_interval_end = current_time + (time_values[-1] - time_values[-2])
        else:
            terminal_interval_end = current_time + 1
        trapped_intervals.append((start_trapped, terminal_interval_end))

    final_equity = cash + qty * closes[-1] * sell_fee_mult
    open_trade = None
    if qty > 0.0:
        open_proceeds = qty * closes[-1] * sell_fee_mult
        open_pnl = open_proceeds - round_cost_sum - round_fee_sum
        open_trade = {
            "entry_time": entry_time,
            "exit_time": time_values[-1] if time_values is not None else int(closes.size - 1),
            "pnl": float(open_pnl),
            "rtn": float(open_pnl / round_cost_sum) if round_cost_sum > 0 else np.nan,
            "bars_held": int(bars_held_this_round),
            "is_open": True,
        }
    result = {
        "add_drop": add_drop,
        "multiplier": multiplier,
        "max_orders": max_orders,
        "tp": tp,
        "capital": capital,
        "final_equity": round(final_equity, 2),
        "max_dd_overall": round(abs(max_drawdown_overall * 100), 2),
        "trades": trades,
        "trapped_time_ratio": float(np.mean(trapped_mask)) if trapped_mask else 0.0,
        "open_trade": open_trade,
    }
    if return_curve:
        result["equity_curve"] = equity_curve
        result["time_index"] = time_curve
        result["trades_log"] = trades_log
        result["trapped_intervals"] = trapped_intervals
        result["position_curve"] = position_curve
        result["trapped_mask"] = trapped_mask
    return result


def martingale_backtest_diy(
    prices,
    level_ratios,
    order_shares,
    tp=0.01,
    capital=1000,
    return_curve=False,
    times=None,
    fee_rate=0.0,
    opens=None,
    highs=None,
    lows=None,
):
    """Detailed Pionex DIY backtest with explicit price ratios and shares."""
    closes = np.asarray(prices, dtype=np.float64)
    opens_np = closes if opens is None else np.asarray(opens, dtype=np.float64)
    highs_np = closes if highs is None else np.asarray(highs, dtype=np.float64)
    lows_np = closes if lows is None else np.asarray(lows, dtype=np.float64)
    levels = np.asarray(level_ratios, dtype=np.float64)
    shares = np.asarray(order_shares, dtype=np.float64)
    if closes.ndim != 1 or closes.size == 0:
        raise ValueError("prices must be a non-empty 1-D array")
    if any(x.ndim != 1 or x.size != closes.size for x in (opens_np, highs_np, lows_np)):
        raise ValueError("open/high/low/close 長度必須一致")
    if not all(np.isfinite(x).all() for x in (opens_np, highs_np, lows_np, closes)):
        raise ValueError("OHLC 不可包含 NaN/Inf")
    if not all((x > 0.0).all() for x in (opens_np, highs_np, lows_np, closes)):
        raise ValueError("OHLC 價格必須全部 > 0")
    if np.any(highs_np < np.maximum(opens_np, closes)) or np.any(lows_np > np.minimum(opens_np, closes)):
        raise ValueError("OHLC 關係異常：high/low 未包住 open/close")
    if np.any(highs_np < lows_np):
        raise ValueError("OHLC 關係異常：high < low")
    if (
        levels.ndim != 1
        or shares.ndim != 1
        or levels.size < 2
        or levels.size != shares.size
        or not np.isfinite(levels).all()
        or not np.isfinite(shares).all()
        or not np.isclose(levels[0], 1.0)
        or np.any(levels <= 0.0)
        or np.any(np.diff(levels) >= 0.0)
        or np.any(shares <= 0.0)
    ):
        raise ValueError("DIY levels/shares 必須等長；levels 由 1.0 嚴格遞減且 shares > 0")
    if not math.isfinite(float(tp)) or tp <= 0.0:
        raise ValueError("tp 必須為有限正數")
    if not math.isfinite(float(capital)) or capital <= 0.0:
        raise ValueError("capital 必須為有限正數")
    if not math.isfinite(float(fee_rate)) or not (0.0 <= fee_rate < 1.0):
        raise ValueError("fee_rate 必須滿足 0 <= fee_rate < 1")

    max_orders = int(levels.size)
    time_values = None
    if times is not None:
        if len(times) != closes.size:
            raise ValueError("times 長度必須與 prices 相同")
        time_idx = pd.DatetimeIndex(times)
        if len(times) >= 2 and (
            not time_idx.is_monotonic_increasing or time_idx.has_duplicates
        ):
            raise ValueError("times 必須嚴格遞增且不可重複")
        time_values = time_idx.tolist()

    cash = float(capital)
    qty = 0.0
    order_count = 0
    round_unit = 0.0
    round_start_cash = 0.0
    entry_time = None
    round_cost_sum = 0.0
    round_fee_sum = 0.0
    bars_held_this_round = 0
    base_price = None
    target_exit_price = None
    max_order_count_round = 0

    peak_equity_overall = float(capital)
    max_drawdown_overall = 0.0
    sell_fee_mult = 1.0 - float(fee_rate)
    total_shares = float(shares.sum())

    trades = 0
    equity_curve, time_curve = [], []
    trades_log = []
    position_curve = []
    utilization_curve = []
    underwater_position_mask = []
    full_capital_mask = []
    trapped_mask = []
    trapped_intervals = []
    start_trapped = None
    is_trapped_prev = False

    for idx, price in enumerate(closes):
        current_time = time_values[idx] if time_values is not None else idx
        high = float(highs_np[idx])
        low = float(lows_np[idx])
        low_equity = None

        if qty == 0.0 and cash > 0.0:
            round_start_cash = cash
            round_unit = cash / ((1.0 + fee_rate) * total_shares)
            alloc = min(round_unit * shares[0], cash / (1.0 + fee_rate))
            if alloc > 0.0:
                qty = alloc / price
                fee = alloc * fee_rate
                cash -= alloc + fee
                round_cost_sum = alloc
                round_fee_sum = fee
                entry_time = current_time
                base_price = float(price)
                order_count = 1
                max_order_count_round = 1
                bars_held_this_round = 0
                target_exit_price = (
                    round_cost_sum + round_fee_sum + round_cost_sum * tp
                ) / (qty * sell_fee_mult)
        elif qty > 0.0:
            bars_held_this_round += 1
            added_this_bar = False
            while (
                order_count < max_orders
                and cash > 0.0
                and low <= base_price * levels[order_count] * (1.0 + 1e-12)
            ):
                alloc = min(
                    round_unit * shares[order_count],
                    cash / (1.0 + fee_rate),
                )
                if alloc <= 0.0:
                    break
                fill_price = base_price * levels[order_count]
                qty += alloc / fill_price
                fee = alloc * fee_rate
                cash -= alloc + fee
                round_cost_sum += alloc
                round_fee_sum += fee
                order_count += 1
                max_order_count_round = max(max_order_count_round, order_count)
                added_this_bar = True

            target_exit_price = (
                round_cost_sum + round_fee_sum + round_cost_sum * tp
            ) / (qty * sell_fee_mult)
            low_equity = cash + qty * low * sell_fee_mult

            if (not added_this_bar) and high >= target_exit_price:
                prospective_proceeds = qty * target_exit_price * sell_fee_mult
                pnl = prospective_proceeds - round_cost_sum - round_fee_sum
                cash += prospective_proceeds
                trades_log.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "pnl": float(pnl),
                    "rtn": float(pnl / round_cost_sum) if round_cost_sum > 0 else np.nan,
                    "bars_held": int(bars_held_this_round),
                    "max_orders_reached": int(max_order_count_round),
                })
                qty = 0.0
                trades += 1
                order_count = 0
                round_unit = 0.0
                round_start_cash = 0.0
                round_cost_sum = 0.0
                round_fee_sum = 0.0
                bars_held_this_round = 0
                base_price = None
                target_exit_price = None
                max_order_count_round = 0

        if low_equity is not None and peak_equity_overall > 0.0:
            dd_low = low_equity / peak_equity_overall - 1.0
            if dd_low < max_drawdown_overall:
                max_drawdown_overall = dd_low

        equity = cash + qty * price * sell_fee_mult
        if equity > peak_equity_overall:
            peak_equity_overall = equity
        if peak_equity_overall > 0.0:
            dd_overall = equity / peak_equity_overall - 1.0
            if dd_overall < max_drawdown_overall:
                max_drawdown_overall = dd_overall

        utilization = (
            round_cost_sum * (1.0 + fee_rate) / round_start_cash
            if qty > 0.0 and round_start_cash > 0.0 else 0.0
        )
        utilization_curve.append(float(utilization))
        if return_curve:
            equity_curve.append(float(equity))
            time_curve.append(current_time)
            position_curve.append(qty > 0.0)

        is_underwater_position = (
            qty > 0.0
            and qty * price * sell_fee_mult < (round_cost_sum + round_fee_sum)
        )
        is_full_capital = qty > 0.0 and order_count == max_orders
        is_trapped = is_full_capital and is_underwater_position
        underwater_position_mask.append(bool(is_underwater_position))
        full_capital_mask.append(bool(is_full_capital))
        trapped_mask.append(bool(is_trapped))
        if is_trapped and not is_trapped_prev:
            start_trapped = current_time
        elif not is_trapped and is_trapped_prev and start_trapped is not None:
            trapped_intervals.append((start_trapped, current_time))
            start_trapped = None
        is_trapped_prev = is_trapped

    if start_trapped is not None:
        if time_values is not None and len(time_values) >= 2:
            terminal_interval_end = current_time + (time_values[-1] - time_values[-2])
        else:
            terminal_interval_end = current_time + 1
        trapped_intervals.append((start_trapped, terminal_interval_end))

    final_equity = cash + qty * closes[-1] * sell_fee_mult
    open_trade = None
    if qty > 0.0:
        open_proceeds = qty * closes[-1] * sell_fee_mult
        open_pnl = open_proceeds - round_cost_sum - round_fee_sum
        open_trade = {
            "entry_time": entry_time,
            "exit_time": time_values[-1] if time_values is not None else int(closes.size - 1),
            "pnl": float(open_pnl),
            "rtn": float(open_pnl / round_cost_sum) if round_cost_sum > 0 else np.nan,
            "bars_held": int(bars_held_this_round),
            "max_orders_reached": int(max_order_count_round),
            "is_open": True,
        }

    result = {
        "strategy_mode": "diy",
        "level_ratios": tuple(float(v) for v in levels),
        "order_shares": tuple(float(v) for v in shares),
        "max_orders": max_orders,
        "tp": float(tp),
        "capital": float(capital),
        "final_equity": round(float(final_equity), 2),
        "max_dd_overall": round(abs(max_drawdown_overall * 100.0), 2),
        "trades": int(trades),
        "trapped_time_ratio": float(np.mean(trapped_mask)) if trapped_mask else 0.0,
        "avg_capital_utilization": float(np.mean(utilization_curve)) if utilization_curve else 0.0,
        "underwater_position_ratio": (
            float(np.mean(underwater_position_mask))
            if underwater_position_mask else 0.0
        ),
        "full_capital_time_ratio": (
            float(np.mean(full_capital_mask)) if full_capital_mask else 0.0
        ),
        "open_trade": open_trade,
    }
    if return_curve:
        result["equity_curve"] = equity_curve
        result["time_index"] = time_curve
        result["trades_log"] = trades_log
        result["trapped_intervals"] = trapped_intervals
        result["position_curve"] = position_curve
        result["trapped_mask"] = trapped_mask
        result["capital_utilization_curve"] = utilization_curve
        result["underwater_position_mask"] = underwater_position_mask
        result["full_capital_mask"] = full_capital_mask
    return result

# ===================== 快速版（平行網格 + 先遮罩） =====================

@njit(cache=True)
def _order_factor_sum_numba(multiplier, max_orders):
    if max_orders <= 0 or multiplier <= 0.0 or not math.isfinite(multiplier):
        return np.nan
    total_factor = 0.0
    factor = 1.0
    for order_no in range(1, max_orders + 1):
        if order_no >= 3:
            factor *= multiplier
        total_factor += factor
        if not math.isfinite(total_factor):
            return np.nan
    return total_factor


@njit(cache=True)
def _calc_init_order_numba(capital, multiplier, max_orders, fee_rate=0.0):
    total_factor = _order_factor_sum_numba(multiplier, max_orders)
    if not math.isfinite(total_factor) or capital <= 0.0 or fee_rate < 0.0 or fee_rate >= 1.0:
        return np.nan
    return capital / ((1.0 + fee_rate) * total_factor)


@njit(cache=True)
def _backtest_core_ohlc(opens, highs, lows, closes, add_drop, multiplier, max_orders, tp, capital, fee_rate):
    """Numba OHLC core; mechanics mirror martingale_backtest()."""
    trapped_bars = 0
    total_bars = closes.shape[0]
    if total_bars == 0 or opens.shape[0] != total_bars or highs.shape[0] != total_bars or lows.shape[0] != total_bars:
        return np.nan, np.nan, 0, np.nan
    if (
        not math.isfinite(add_drop) or not math.isfinite(multiplier)
        or not math.isfinite(tp) or not math.isfinite(capital) or not math.isfinite(fee_rate)
        or add_drop <= 0.0 or add_drop >= 1.0 or multiplier <= 0.0
        or max_orders < 1 or tp <= 0.0 or capital <= 0.0
        or fee_rate < 0.0 or fee_rate >= 1.0
    ):
        return np.nan, np.nan, 0, 0.0

    cash = capital
    qty = 0.0
    order_count = 0
    init_order_round = 0.0

    base_price = 0.0
    next_level_price = 0.0
    next_order_factor = 1.0

    peak_equity_overall = capital
    inv_peak_equity = 1.0 / capital if capital > 0.0 else 1.0
    max_drawdown_overall = 0.0
    trades = 0
    round_cost_sum = 0.0   # 累計買入本金
    round_fee_sum = 0.0    # 累計買入費
    target_exit_price = 0.0

    inv_one_plus_fee = 1.0 / (1.0 + fee_rate)
    sell_fee_mult = 1.0 - fee_rate
    r = 1.0 - add_drop
    multiplier_is_one = multiplier == 1.0
    total_factor = _order_factor_sum_numba(multiplier, max_orders)
    if not math.isfinite(total_factor):
        return np.nan, np.nan, 0, np.nan

    for i in range(total_bars):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        open_price = opens[i]
        if (
            not math.isfinite(price) or not math.isfinite(high) or not math.isfinite(low)
            or not math.isfinite(open_price) or price <= 0.0 or high <= 0.0 or low <= 0.0
            or open_price <= 0.0 or high < low or high < price or high < open_price
            or low > price or low > open_price
        ):
            return np.nan, np.nan, 0, np.nan
        low_equity = 0.0
        has_low_equity = False

        # 開倉
        if (qty == 0.0) and (cash > 0.0):
            init_order_round = cash * inv_one_plus_fee / total_factor
            alloc = init_order_round
            max_afford = cash * inv_one_plus_fee
            if alloc > max_afford:
                alloc = max_afford
            if alloc > 0.0:
                qty += alloc / price
                fee = alloc * fee_rate
                cash -= (alloc + fee)
                base_price = price
                next_level_price = base_price * r
                next_order_factor = 1.0
                order_count = 1
                round_cost_sum = alloc
                round_fee_sum = fee
                target_exit_price = (round_cost_sum + round_fee_sum + round_cost_sum * tp) / (qty * sell_fee_mult)

        elif qty > 0.0:
            added_this_bar = False
            while (
                order_count < max_orders and cash > 0.0
                and low <= next_level_price * (1.0 + 1e-12)
            ):
                target_alloc = init_order_round * next_order_factor
                max_afford = cash * inv_one_plus_fee
                alloc = target_alloc if target_alloc < max_afford else max_afford
                if alloc <= 0.0:
                    break
                qty += alloc / next_level_price
                fee = alloc * fee_rate
                cash -= alloc + fee
                order_count += 1
                round_cost_sum += alloc
                round_fee_sum += fee
                added_this_bar = True
                next_level_price *= r
                if (not multiplier_is_one) and order_count >= 2:
                    next_order_factor *= multiplier

            target_exit_price = (round_cost_sum + round_fee_sum + round_cost_sum * tp) / (qty * sell_fee_mult)
            low_equity = cash + qty * low * sell_fee_mult
            has_low_equity = True

            if (not added_this_bar) and high >= target_exit_price:
                prospective_proceeds = qty * target_exit_price * sell_fee_mult
                cash += prospective_proceeds
                qty = 0.0
                trades += 1
                order_count = 0
                init_order_round = 0.0
                round_cost_sum = 0.0
                round_fee_sum = 0.0
                base_price = 0.0
                next_level_price = 0.0
                next_order_factor = 1.0
                target_exit_price = 0.0

        if has_low_equity and peak_equity_overall > 0.0:
            dd_low = (low_equity * inv_peak_equity) - 1.0
            if dd_low < max_drawdown_overall:
                max_drawdown_overall = dd_low

        equity = cash + qty * price * sell_fee_mult
        if equity > peak_equity_overall:
            peak_equity_overall = equity
            inv_peak_equity = 1.0 / equity
        else:
            dd = (equity * inv_peak_equity) - 1.0
            if dd < max_drawdown_overall:
                max_drawdown_overall = dd

        if (
            order_count == max_orders and qty > 0.0
            and qty * price * sell_fee_mult < (round_cost_sum + round_fee_sum)
        ):
            trapped_bars += 1

    final_equity = cash + qty * closes[-1] * sell_fee_mult
    mdd_overall_pct = -max_drawdown_overall * 100.0
    trapped_ratio = (trapped_bars / total_bars) if total_bars > 0 else 0.0
    return final_equity, mdd_overall_pct, trades, trapped_ratio


@njit(cache=True)
def _backtest_core_diy_ohlc_extended(
    opens,
    highs,
    lows,
    closes,
    level_ratios,
    order_shares,
    max_orders,
    tp,
    capital,
    fee_rate,
):
    """DIY OHLC core with utilization and position-state diagnostics."""
    total_bars = closes.shape[0]
    if (
        total_bars == 0
        or opens.shape[0] != total_bars
        or highs.shape[0] != total_bars
        or lows.shape[0] != total_bars
        or max_orders < 2
        or max_orders > level_ratios.shape[0]
        or max_orders > order_shares.shape[0]
        or tp <= 0.0
        or capital <= 0.0
        or fee_rate < 0.0
        or fee_rate >= 1.0
    ):
        return np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan
    if (
        not math.isfinite(tp)
        or not math.isfinite(capital)
        or not math.isfinite(fee_rate)
        or not math.isfinite(level_ratios[0])
        or abs(level_ratios[0] - 1.0) > 1e-12
    ):
        return np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan

    total_weight = 0.0
    previous_level = 2.0
    for j in range(max_orders):
        level = level_ratios[j]
        weight = order_shares[j]
        if (
            not math.isfinite(level)
            or not math.isfinite(weight)
            or level <= 0.0
            or weight <= 0.0
            or (j > 0 and level >= previous_level)
        ):
            return np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan
        previous_level = level
        total_weight += weight
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan

    cash = capital
    qty = 0.0
    order_count = 0
    round_unit = 0.0
    round_start_cash = 0.0
    base_price = 0.0
    round_cost_sum = 0.0
    round_fee_sum = 0.0
    target_exit_price = 0.0

    peak_equity_overall = capital
    inv_peak_equity = 1.0 / capital
    max_drawdown_overall = 0.0
    trades = 0
    trapped_bars = 0
    underwater_bars = 0
    full_capital_bars = 0
    utilization_sum = 0.0
    inv_one_plus_fee = 1.0 / (1.0 + fee_rate)
    sell_fee_mult = 1.0 - fee_rate

    for i in range(total_bars):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        open_price = opens[i]
        if (
            not math.isfinite(price)
            or not math.isfinite(high)
            or not math.isfinite(low)
            or not math.isfinite(open_price)
            or price <= 0.0
            or high <= 0.0
            or low <= 0.0
            or open_price <= 0.0
            or high < low
            or high < price
            or high < open_price
            or low > price
            or low > open_price
        ):
            return np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan

        low_equity = 0.0
        has_low_equity = False
        if qty == 0.0 and cash > 0.0:
            round_start_cash = cash
            round_unit = cash * inv_one_plus_fee / total_weight
            alloc = round_unit * order_shares[0]
            max_afford = cash * inv_one_plus_fee
            if alloc > max_afford:
                alloc = max_afford
            if alloc > 0.0:
                qty = alloc / price
                fee = alloc * fee_rate
                cash -= alloc + fee
                base_price = price
                order_count = 1
                round_cost_sum = alloc
                round_fee_sum = fee
                target_exit_price = (
                    round_cost_sum + round_fee_sum + round_cost_sum * tp
                ) / (qty * sell_fee_mult)
        elif qty > 0.0:
            added_this_bar = False
            while (
                order_count < max_orders
                and cash > 0.0
                and low <= base_price * level_ratios[order_count] * (1.0 + 1e-12)
            ):
                alloc = round_unit * order_shares[order_count]
                max_afford = cash * inv_one_plus_fee
                if alloc > max_afford:
                    alloc = max_afford
                if alloc <= 0.0:
                    break
                fill_price = base_price * level_ratios[order_count]
                qty += alloc / fill_price
                fee = alloc * fee_rate
                cash -= alloc + fee
                round_cost_sum += alloc
                round_fee_sum += fee
                order_count += 1
                added_this_bar = True

            target_exit_price = (
                round_cost_sum + round_fee_sum + round_cost_sum * tp
            ) / (qty * sell_fee_mult)
            low_equity = cash + qty * low * sell_fee_mult
            has_low_equity = True

            if (not added_this_bar) and high >= target_exit_price:
                cash += qty * target_exit_price * sell_fee_mult
                qty = 0.0
                trades += 1
                order_count = 0
                round_unit = 0.0
                round_start_cash = 0.0
                base_price = 0.0
                round_cost_sum = 0.0
                round_fee_sum = 0.0
                target_exit_price = 0.0

        if has_low_equity and peak_equity_overall > 0.0:
            dd_low = low_equity * inv_peak_equity - 1.0
            if dd_low < max_drawdown_overall:
                max_drawdown_overall = dd_low

        equity = cash + qty * price * sell_fee_mult
        if equity > peak_equity_overall:
            peak_equity_overall = equity
            inv_peak_equity = 1.0 / equity
        else:
            dd = equity * inv_peak_equity - 1.0
            if dd < max_drawdown_overall:
                max_drawdown_overall = dd

        if qty > 0.0 and round_start_cash > 0.0:
            utilization_sum += (
                round_cost_sum * (1.0 + fee_rate) / round_start_cash
            )
            is_underwater = (
                qty * price * sell_fee_mult < (round_cost_sum + round_fee_sum)
            )
            is_full_capital = order_count == max_orders
            if is_underwater:
                underwater_bars += 1
            if is_full_capital:
                full_capital_bars += 1
            if is_underwater and is_full_capital:
                trapped_bars += 1

    final_equity = cash + qty * closes[-1] * sell_fee_mult
    mdd_overall_pct = -max_drawdown_overall * 100.0
    trapped_ratio = trapped_bars / total_bars
    avg_utilization = utilization_sum / total_bars
    underwater_ratio = underwater_bars / total_bars
    full_capital_ratio = full_capital_bars / total_bars
    return (
        final_equity,
        mdd_overall_pct,
        trades,
        trapped_ratio,
        avg_utilization,
        underwater_ratio,
        full_capital_ratio,
    )


@njit(cache=True)
def _backtest_core_diy_ohlc(
    opens,
    highs,
    lows,
    closes,
    level_ratios,
    order_shares,
    max_orders,
    tp,
    capital,
    fee_rate,
):
    """Backward-compatible five-metric DIY OHLC core."""
    (
        final_equity,
        mdd,
        trades,
        trapped_ratio,
        avg_utilization,
        _,
        _,
    ) = _backtest_core_diy_ohlc_extended(
        opens,
        highs,
        lows,
        closes,
        level_ratios,
        order_shares,
        max_orders,
        tp,
        capital,
        fee_rate,
    )
    return final_equity, mdd, trades, trapped_ratio, avg_utilization


@njit(cache=True)
def _backtest_core(prices, add_drop, multiplier, max_orders, tp, capital, fee_rate):
    """Close-path compatibility wrapper used by Monte Carlo."""
    return _backtest_core_ohlc(
        prices, prices, prices, prices,
        add_drop, multiplier, max_orders, tp, capital, fee_rate,
    )


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


@njit(parallel=True, cache=True)
def _grid_search_parallel_ohlc(opens, highs, lows, closes, add_drop_arr, mul_arr, max_orders_arr, tp_arr, capital, fee_rate):
    n = add_drop_arr.shape[0]
    fe = np.empty(n)
    mdd = np.empty(n)
    tr = np.empty(n)
    trap = np.empty(n)
    for i in prange(n):
        fe_i, mdd_i, tr_i, trap_i = _backtest_core_ohlc(
            opens, highs, lows, closes,
            float(add_drop_arr[i]),
            float(mul_arr[i]),
            int(max_orders_arr[i]),
            float(tp_arr[i]),
            float(capital),
            float(fee_rate),
        )
        fe[i] = fe_i
        mdd[i] = mdd_i
        tr[i] = tr_i
        trap[i] = trap_i
    return fe, mdd, tr, trap


@njit(parallel=True, cache=True)
def _grid_search_parallel_diy_ohlc(
    opens,
    highs,
    lows,
    closes,
    level_matrix,
    share_matrix,
    template_index_arr,
    max_orders_arr,
    tp_arr,
    capital,
    fee_rate,
):
    """Evaluate DIY candidates while reusing compact ladder/share templates."""
    n = template_index_arr.shape[0]
    fe = np.empty(n)
    mdd = np.empty(n)
    tr = np.empty(n)
    trap = np.empty(n)
    utilization = np.empty(n)
    for i in prange(n):
        template_index = int(template_index_arr[i])
        fe_i, mdd_i, tr_i, trap_i, util_i = _backtest_core_diy_ohlc(
            opens,
            highs,
            lows,
            closes,
            level_matrix[template_index],
            share_matrix[template_index],
            int(max_orders_arr[i]),
            float(tp_arr[i]),
            float(capital),
            float(fee_rate),
        )
        fe[i] = fe_i
        mdd[i] = mdd_i
        tr[i] = tr_i
        trap[i] = trap_i
        utilization[i] = util_i
    return fe, mdd, tr, trap, utilization


@njit(parallel=True, cache=True)
def _grid_search_parallel_diy_idle_ohlc(
    opens,
    highs,
    lows,
    closes,
    level_matrix,
    share_matrix,
    template_index_arr,
    max_orders_arr,
    tp_arr,
    capital,
    fee_rate,
):
    """Evaluate DIY candidates with idle/underwater diagnostics."""
    n = template_index_arr.shape[0]
    fe = np.empty(n)
    mdd = np.empty(n)
    tr = np.empty(n)
    trap = np.empty(n)
    utilization = np.empty(n)
    underwater = np.empty(n)
    full_capital = np.empty(n)
    for i in prange(n):
        template_index = int(template_index_arr[i])
        (
            fe_i,
            mdd_i,
            tr_i,
            trap_i,
            util_i,
            underwater_i,
            full_capital_i,
        ) = _backtest_core_diy_ohlc_extended(
            opens,
            highs,
            lows,
            closes,
            level_matrix[template_index],
            share_matrix[template_index],
            int(max_orders_arr[i]),
            float(tp_arr[i]),
            float(capital),
            float(fee_rate),
        )
        fe[i] = fe_i
        mdd[i] = mdd_i
        tr[i] = tr_i
        trap[i] = trap_i
        utilization[i] = util_i
        underwater[i] = underwater_i
        full_capital[i] = full_capital_i
    return fe, mdd, tr, trap, utilization, underwater, full_capital

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
