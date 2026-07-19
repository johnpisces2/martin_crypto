# -*- coding: utf-8 -*-
"""Parallel MC evaluation utilities."""

from __future__ import annotations

import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import martin

try:
    from numba import njit
    _NUMBA_LOCAL = True
except Exception:
    _NUMBA_LOCAL = False


def bootstrap_path_from_returns(hist_rets: np.ndarray, n_bars: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_hist = hist_rets.shape[0]
    if n_hist < 2:
        raise ValueError("歷史報酬樣本不足，至少需要 2 個報酬點。")
    if n_bars <= 0 or block_size <= 0:
        raise ValueError("n_bars 與 block_size 必須 > 0")
    if block_size > n_hist:
        raise ValueError(f"block_size ({block_size}) 不可大於歷史報酬樣本數 ({n_hist})")

    out = np.empty(n_bars, dtype=np.float64)
    if block_size <= 1:
        idx = rng.integers(0, n_hist, size=n_bars)
        out[:] = hist_rets[idx]
        return out

    max_start = n_hist - block_size + 1

    n_blocks = int(math.ceil(n_bars / block_size))
    write_pos = 0
    for _ in range(n_blocks):
        s = int(rng.integers(0, max_start))
        blk = hist_rets[s:s + block_size]
        m = min(blk.shape[0], n_bars - write_pos)
        out[write_pos:write_pos + m] = blk[:m]
        write_pos += m
        if write_pos >= n_bars:
            break
    return out


def ohlc_ratios_from_history(opens, highs, lows, closes) -> np.ndarray:
    """Normalize each historical OHLC bar by its previous close.

    Row ``i`` represents historical bar ``i + 1`` and can therefore be scaled
    onto any synthetic previous close while preserving gaps and intrabar range.
    """
    arrays = [np.asarray(values, dtype=np.float64) for values in (opens, highs, lows, closes)]
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("歷史 OHLC 必須是一維")
    if len({values.size for values in arrays}) != 1 or arrays[0].size < 3:
        raise ValueError("歷史 OHLC 長度必須一致且至少包含 3 根 K 線")
    open_np, high_np, low_np, close_np = arrays
    if not all(np.isfinite(values).all() and np.all(values > 0.0) for values in arrays):
        raise ValueError("歷史 OHLC 必須為有限正數")
    if np.any(high_np < np.maximum(open_np, close_np)) or np.any(low_np > np.minimum(open_np, close_np)):
        raise ValueError("歷史 OHLC high/low 未包住 open/close")
    prev_close = close_np[:-1]
    ratios = np.column_stack((
        open_np[1:] / prev_close,
        high_np[1:] / prev_close,
        low_np[1:] / prev_close,
        close_np[1:] / prev_close,
    ))
    return np.ascontiguousarray(ratios, dtype=np.float64)


def _validate_ohlc_ratios(hist_ohlc_ratios, expected_rows: int) -> np.ndarray:
    ratios = np.asarray(hist_ohlc_ratios, dtype=np.float64)
    if ratios.shape != (int(expected_rows), 4):
        raise ValueError("hist_ohlc_ratios 必須為 (len(hist_rets), 4)")
    if not np.isfinite(ratios).all() or np.any(ratios <= 0.0):
        raise ValueError("hist_ohlc_ratios 必須為有限正數")
    if (
        np.any(ratios[:, 1] < np.maximum(ratios[:, 0], ratios[:, 3]))
        or np.any(ratios[:, 2] > np.minimum(ratios[:, 0], ratios[:, 3]))
    ):
        raise ValueError("hist_ohlc_ratios high/low 關係異常")
    return np.ascontiguousarray(ratios)


def bootstrap_ohlc_path_from_ratios(
    hist_ohlc_ratios: np.ndarray,
    start_price: float,
    n_bars: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic OHLC path using IID or moving-block resampling."""
    ratios = np.asarray(hist_ohlc_ratios, dtype=np.float64)
    if ratios.ndim != 2 or ratios.shape[1] != 4 or ratios.shape[0] < 2:
        raise ValueError("hist_ohlc_ratios 至少需要 2 列、4 欄")
    ratios = _validate_ohlc_ratios(ratios, ratios.shape[0])
    if not np.isfinite(start_price) or start_price <= 0.0:
        raise ValueError("start_price 必須為有限正數")
    if n_bars <= 0 or block_size <= 0 or block_size > ratios.shape[0]:
        raise ValueError("n_bars/block_size 不合法或 block_size 大於歷史樣本")

    if block_size <= 1:
        sampled = rng.integers(0, ratios.shape[0], size=int(n_bars))
    else:
        sampled_parts = []
        max_start = ratios.shape[0] - int(block_size) + 1
        remaining = int(n_bars)
        while remaining > 0:
            start = int(rng.integers(0, max_start))
            take = min(int(block_size), remaining)
            sampled_parts.append(np.arange(start, start + take, dtype=np.int64))
            remaining -= take
        sampled = np.concatenate(sampled_parts)

    opens = np.empty(int(n_bars) + 1, dtype=np.float64)
    highs = np.empty_like(opens)
    lows = np.empty_like(opens)
    closes = np.empty_like(opens)
    opens[0] = highs[0] = lows[0] = closes[0] = float(start_price)
    for pos, hist_idx in enumerate(sampled, start=1):
        scale = closes[pos - 1]
        opens[pos] = scale * ratios[hist_idx, 0]
        highs[pos] = scale * ratios[hist_idx, 1]
        lows[pos] = scale * ratios[hist_idx, 2]
        closes[pos] = scale * ratios[hist_idx, 3]
    return opens, highs, lows, closes


def build_bootstrap_plan(
    hist_rets: np.ndarray,
    n_bars: int,
    block_size: int,
    total_paths: int,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    """Build compact per-path seeds shared by every candidate/worker.

    Workers deterministically expand each seed into IID indices or block starts.
    Memory is O(paths), not O(paths * bars).
    """
    n_hist = hist_rets.shape[0]
    if n_hist < 2:
        raise ValueError("歷史報酬樣本不足，至少需要 2 個報酬點。")
    if total_paths <= 0:
        raise ValueError("total_paths 必須 > 0")
    if n_bars <= 0 or block_size <= 0:
        raise ValueError("n_bars 與 block_size 必須 > 0")
    if block_size > n_hist:
        raise ValueError(f"block_size ({block_size}) 不可大於歷史報酬樣本數 ({n_hist})")

    plan = rng.integers(
        0, np.iinfo(np.uint64).max, size=int(total_paths), dtype=np.uint64
    )
    return "seeds", plan


def _plan_kind_code(plan_kind: str) -> int:
    if plan_kind == "indices":
        return 0
    if plan_kind == "blocks":
        return 1
    if plan_kind == "seeds":
        return 2
    raise ValueError(f"未知 bootstrap plan kind: {plan_kind}")


if _NUMBA_LOCAL:
    # Single source of truth for strategy mechanics; MC shares martin.py's grid core.
    _backtest_core_numba_local = martin._backtest_core
    _backtest_core_ohlc_numba_local = martin._backtest_core_ohlc


    @njit(cache=True)
    def _fill_returns_from_indices(hist_rets, plan_row, out):
        for i in range(out.shape[0]):
            out[i] = hist_rets[plan_row[i]]


    @njit(cache=True)
    def _fill_returns_from_blocks(hist_rets, plan_row, block_size, out):
        write_pos = 0
        for b in range(plan_row.shape[0]):
            start = plan_row[b]
            remaining = out.shape[0] - write_pos
            if remaining <= 0:
                break
            copy_n = block_size if block_size < remaining else remaining
            for k in range(copy_n):
                out[write_pos + k] = hist_rets[start + k]
            write_pos += copy_n


    @njit(cache=True)
    def _splitmix64_next(state):
        state = state + np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return state, z


    @njit(cache=True)
    def _fill_returns_from_seed(hist_rets, seed, block_size, out):
        state = np.uint64(seed)
        n_hist = hist_rets.shape[0]
        if block_size <= 1:
            for i in range(out.shape[0]):
                state, z = _splitmix64_next(state)
                out[i] = hist_rets[int(z % np.uint64(n_hist))]
            return

        max_start = n_hist - block_size + 1
        write_pos = 0
        while write_pos < out.shape[0]:
            state, z = _splitmix64_next(state)
            start = int(z % np.uint64(max_start))
            copy_n = block_size
            if copy_n > out.shape[0] - write_pos:
                copy_n = out.shape[0] - write_pos
            for k in range(copy_n):
                out[write_pos + k] = hist_rets[start + k]
            write_pos += copy_n


    @njit(cache=True)
    def _fill_ohlc_from_seed(
        hist_ohlc_ratios, seed, block_size, start_price,
        out_open, out_high, out_low, out_close,
    ):
        state = np.uint64(seed)
        n_hist = hist_ohlc_ratios.shape[0]
        out_open[0] = start_price
        out_high[0] = start_price
        out_low[0] = start_price
        out_close[0] = start_price
        write_pos = 0
        while write_pos < out_close.shape[0] - 1:
            state, z = _splitmix64_next(state)
            if block_size <= 1:
                start = int(z % np.uint64(n_hist))
                copy_n = 1
            else:
                max_start = n_hist - block_size + 1
                start = int(z % np.uint64(max_start))
                copy_n = block_size
                remaining = (out_close.shape[0] - 1) - write_pos
                if copy_n > remaining:
                    copy_n = remaining
            for k in range(copy_n):
                out_idx = write_pos + k + 1
                hist_idx = start + k
                scale = out_close[out_idx - 1]
                out_open[out_idx] = scale * hist_ohlc_ratios[hist_idx, 0]
                out_high[out_idx] = scale * hist_ohlc_ratios[hist_idx, 1]
                out_low[out_idx] = scale * hist_ohlc_ratios[hist_idx, 2]
                out_close[out_idx] = scale * hist_ohlc_ratios[hist_idx, 3]
            write_pos += copy_n


    @njit(cache=True)
    def _returns_to_prices(start_price, one_ret, out_prices):
        px = start_price
        out_prices[0] = px
        for i in range(one_ret.shape[0]):
            px = px * (1.0 + one_ret[i])
            out_prices[i + 1] = px


    @njit(cache=True)
    def _quantile_from_prefix(values, count, q):
        if count <= 0:
            return np.nan
        tmp = np.empty(count, dtype=np.float64)
        for i in range(count):
            tmp[i] = values[i]
        tmp.sort()
        if count == 1:
            return tmp[0]
        pos = q * float(count - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return tmp[lo]
        weight = pos - float(lo)
        return tmp[lo] * (1.0 - weight) + tmp[hi] * weight


    @njit(cache=True)
    def _mc_eval_metrics_onepass_numba(
        hist_rets,
        hist_ohlc_ratios,
        start_price,
        capital,
        fee_rate,
        max_loss,
        max_severe,
        max_dd50,
        mc_bars,
        block_size,
        total_paths,
        add_drop,
        multiplier,
        max_orders,
        tp,
        plan_code,
        path_plan,
    ):
        n = add_drop.shape[0]
        terminal_mean = np.empty(n, dtype=np.float64)
        terminal_median = np.empty(n, dtype=np.float64)
        terminal_p5 = np.empty(n, dtype=np.float64)
        p_loss = np.empty(n, dtype=np.float64)
        p_severe = np.empty(n, dtype=np.float64)
        p_dd50 = np.empty(n, dtype=np.float64)
        mdd_mean = np.empty(n, dtype=np.float64)
        trapped_mean = np.empty(n, dtype=np.float64)
        feasible = np.empty(n, dtype=np.bool_)
        paths_evaluated = np.empty(n, dtype=np.int64)
        early_rejected = np.zeros(n, dtype=np.bool_)

        one_path_open = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_high = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_low = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_close = np.empty(mc_bars + 1, dtype=np.float64)
        terminal_samples = np.empty(total_paths, dtype=np.float64)

        for j in range(n):
            terminal_sum = 0.0
            mdd_sum = 0.0
            trap_sum = 0.0
            loss_count = 0
            severe_count = 0
            dd50_count = 0
            done = 0

            for p in range(total_paths):
                plan_row = path_plan[p]
                _fill_ohlc_from_seed(
                    hist_ohlc_ratios, plan_row, block_size, start_price,
                    one_path_open, one_path_high, one_path_low, one_path_close,
                )

                fe_i, mdd_i, _, trap_i = _backtest_core_ohlc_numba_local(
                    one_path_open,
                    one_path_high,
                    one_path_low,
                    one_path_close,
                    float(add_drop[j]),
                    float(multiplier[j]),
                    int(max_orders[j]),
                    float(tp[j]),
                    capital,
                    fee_rate,
                )
                terminal = float(fe_i)
                mdd = float(mdd_i)
                trap = float(trap_i)

                terminal_samples[p] = terminal
                terminal_sum += terminal
                mdd_sum += mdd
                trap_sum += trap
                if terminal < capital:
                    loss_count += 1
                if terminal < (0.5 * capital):
                    severe_count += 1
                if mdd > 50.0:
                    dd50_count += 1

                done = p + 1
                if (
                    (float(loss_count) / float(total_paths) > max_loss)
                    or (float(severe_count) / float(total_paths) > max_severe)
                    or (float(dd50_count) / float(total_paths) > max_dd50)
                ):
                    early_rejected[j] = True
                    break

            paths_evaluated[j] = done
            terminal_mean[j] = terminal_sum / float(done)
            terminal_median[j] = _quantile_from_prefix(terminal_samples, done, 0.5)
            terminal_p5[j] = _quantile_from_prefix(terminal_samples, done, 0.05)
            p_loss[j] = float(loss_count) / float(total_paths)
            p_severe[j] = float(severe_count) / float(total_paths)
            p_dd50[j] = float(dd50_count) / float(total_paths)
            mdd_mean[j] = mdd_sum / float(done)
            trapped_mean[j] = trap_sum / float(done)
            feasible[j] = (
                (not early_rejected[j])
                and (p_loss[j] <= max_loss)
                and (p_severe[j] <= max_severe)
                and (p_dd50[j] <= max_dd50)
            )

        return (
            terminal_mean,
            terminal_median,
            terminal_p5,
            p_loss,
            p_severe,
            p_dd50,
            mdd_mean,
            trapped_mean,
            feasible,
            paths_evaluated,
            early_rejected,
        )


    @njit(cache=True)
    def _mc_eval_metrics_numba(
        hist_rets,
        hist_ohlc_ratios,
        start_price,
        capital,
        fee_rate,
        max_loss,
        max_severe,
        max_dd50,
        mc_bars,
        block_size,
        total_paths,
        add_drop,
        multiplier,
        max_orders,
        tp,
        plan_code,
        path_plan,
    ):
        n = add_drop.shape[0]
        terminal_mean = np.empty(n, dtype=np.float64)
        terminal_median = np.empty(n, dtype=np.float64)
        terminal_p5 = np.empty(n, dtype=np.float64)
        p_loss = np.empty(n, dtype=np.float64)
        p_severe = np.empty(n, dtype=np.float64)
        p_dd50 = np.empty(n, dtype=np.float64)
        mdd_mean = np.empty(n, dtype=np.float64)
        trapped_mean = np.empty(n, dtype=np.float64)
        feasible = np.empty(n, dtype=np.bool_)
        paths_evaluated = np.empty(n, dtype=np.int64)
        early_rejected = np.zeros(n, dtype=np.bool_)

        terminal_sum = np.zeros(n, dtype=np.float64)
        mdd_sum = np.zeros(n, dtype=np.float64)
        trap_sum = np.zeros(n, dtype=np.float64)
        loss_count = np.zeros(n, dtype=np.int64)
        severe_count = np.zeros(n, dtype=np.int64)
        dd50_count = np.zeros(n, dtype=np.int64)
        active = np.ones(n, dtype=np.bool_)
        active_count = n

        one_path_open = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_high = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_low = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_close = np.empty(mc_bars + 1, dtype=np.float64)

        # First pass: build each MC path once and screen all active candidates.
        for p in range(total_paths):
            if active_count <= 0:
                break

            plan_row = path_plan[p]
            _fill_ohlc_from_seed(
                hist_ohlc_ratios, plan_row, block_size, start_price,
                one_path_open, one_path_high, one_path_low, one_path_close,
            )

            done = p + 1
            for j in range(n):
                if not active[j]:
                    continue

                fe_i, mdd_i, _, trap_i = _backtest_core_ohlc_numba_local(
                    one_path_open,
                    one_path_high,
                    one_path_low,
                    one_path_close,
                    float(add_drop[j]),
                    float(multiplier[j]),
                    int(max_orders[j]),
                    float(tp[j]),
                    capital,
                    fee_rate,
                )
                terminal = float(fe_i)
                mdd = float(mdd_i)
                trap = float(trap_i)

                terminal_sum[j] += terminal
                mdd_sum[j] += mdd
                trap_sum[j] += trap
                if terminal < capital:
                    loss_count[j] += 1
                if terminal < (0.5 * capital):
                    severe_count[j] += 1
                if mdd > 50.0:
                    dd50_count[j] += 1

                paths_evaluated[j] = done
                if (
                    (float(loss_count[j]) / float(total_paths) > max_loss)
                    or (float(severe_count[j]) / float(total_paths) > max_severe)
                    or (float(dd50_count[j]) / float(total_paths) > max_dd50)
                ):
                    active[j] = False
                    early_rejected[j] = True
                    active_count -= 1

        survivor_count = 0
        for j in range(n):
            done = paths_evaluated[j]
            if done <= 0:
                done = total_paths
                paths_evaluated[j] = done
            denom_done = float(done)
            terminal_mean[j] = terminal_sum[j] / denom_done
            mdd_mean[j] = mdd_sum[j] / denom_done
            trapped_mean[j] = trap_sum[j] / denom_done
            p_loss[j] = float(loss_count[j]) / float(total_paths)
            p_severe[j] = float(severe_count[j]) / float(total_paths)
            p_dd50[j] = float(dd50_count[j]) / float(total_paths)
            feasible[j] = (
                active[j]
                and (p_loss[j] <= max_loss)
                and (p_severe[j] <= max_severe)
                and (p_dd50[j] <= max_dd50)
            )
            if feasible[j]:
                survivor_count += 1
            elif early_rejected[j]:
                terminal_mean[j] = np.nan
                terminal_median[j] = np.nan
                terminal_p5[j] = np.nan
                mdd_mean[j] = np.nan
                trapped_mean[j] = np.nan
            else:
                terminal_median[j] = terminal_mean[j]
                terminal_p5[j] = terminal_mean[j]

        # Second pass: compute exact terminal quantiles only for survivors.
        if survivor_count > 0:
            survivor_idx = np.empty(survivor_count, dtype=np.int64)
            s = 0
            for j in range(n):
                if feasible[j]:
                    survivor_idx[s] = j
                    s += 1

            survivor_terminals = np.empty((survivor_count, total_paths), dtype=np.float64)
            for p in range(total_paths):
                plan_row = path_plan[p]
                _fill_ohlc_from_seed(
                    hist_ohlc_ratios, plan_row, block_size, start_price,
                    one_path_open, one_path_high, one_path_low, one_path_close,
                )

                for s in range(survivor_count):
                    j = survivor_idx[s]
                    fe_i, _, _, _ = _backtest_core_ohlc_numba_local(
                        one_path_open,
                        one_path_high,
                        one_path_low,
                        one_path_close,
                        float(add_drop[j]),
                        float(multiplier[j]),
                        int(max_orders[j]),
                        float(tp[j]),
                        capital,
                        fee_rate,
                    )
                    survivor_terminals[s, p] = float(fe_i)

            for s in range(survivor_count):
                j = survivor_idx[s]
                terminal_median[j] = _quantile_from_prefix(survivor_terminals[s], total_paths, 0.5)
                terminal_p5[j] = _quantile_from_prefix(survivor_terminals[s], total_paths, 0.05)

        return (
            terminal_mean,
            terminal_median,
            terminal_p5,
            p_loss,
            p_severe,
            p_dd50,
            mdd_mean,
            trapped_mean,
            feasible,
            paths_evaluated,
            early_rejected,
        )
else:
    def _backtest_core_numba_local(prices, add_drop, multiplier, max_orders, tp, capital, fee_rate):
        return martin._backtest_core(prices, add_drop, multiplier, max_orders, tp, capital, fee_rate)

    def _backtest_core_ohlc_numba_local(
        opens, highs, lows, closes,
        add_drop, multiplier, max_orders, tp, capital, fee_rate,
    ):
        return martin._backtest_core_ohlc(
            opens, highs, lows, closes,
            add_drop, multiplier, max_orders, tp, capital, fee_rate,
        )

    def _fill_returns_from_indices(hist_rets, plan_row, out):
        out[:] = hist_rets[plan_row]

    def _fill_returns_from_blocks(hist_rets, plan_row, block_size, out):
        write_pos = 0
        for start in plan_row:
            if write_pos >= out.shape[0]:
                break
            remaining = out.shape[0] - write_pos
            copy_n = min(int(block_size), remaining)
            out[write_pos:write_pos + copy_n] = hist_rets[start:start + copy_n]
            write_pos += copy_n

    def _fill_returns_from_seed(hist_rets, seed, block_size, out):
        mask = (1 << 64) - 1
        state = int(seed) & mask

        def next_value():
            nonlocal state
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            return (z ^ (z >> 31)) & mask

        n_hist = len(hist_rets)
        if block_size <= 1:
            for i in range(len(out)):
                out[i] = hist_rets[next_value() % n_hist]
            return
        max_start = n_hist - block_size + 1
        write_pos = 0
        while write_pos < len(out):
            start = next_value() % max_start
            copy_n = min(block_size, len(out) - write_pos)
            out[write_pos:write_pos + copy_n] = hist_rets[start:start + copy_n]
            write_pos += copy_n

    def _returns_to_prices(start_price, one_ret, out_prices):
        out_prices[0] = start_price
        np.multiply.accumulate(1.0 + one_ret, out=out_prices[1:])
        out_prices[1:] *= start_price

    def _fill_ohlc_from_seed(
        hist_ohlc_ratios, seed, block_size, start_price,
        out_open, out_high, out_low, out_close,
    ):
        mask = (1 << 64) - 1
        state = int(seed) & mask

        def next_value():
            nonlocal state
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            return (z ^ (z >> 31)) & mask

        out_open[0] = out_high[0] = out_low[0] = out_close[0] = start_price
        n_hist = len(hist_ohlc_ratios)
        write_pos = 0
        while write_pos < len(out_close) - 1:
            if block_size <= 1:
                start = next_value() % n_hist
                copy_n = 1
            else:
                start = next_value() % (n_hist - block_size + 1)
                copy_n = min(block_size, (len(out_close) - 1) - write_pos)
            for k in range(copy_n):
                out_idx = write_pos + k + 1
                scale = out_close[out_idx - 1]
                ratio = hist_ohlc_ratios[start + k]
                out_open[out_idx] = scale * ratio[0]
                out_high[out_idx] = scale * ratio[1]
                out_low[out_idx] = scale * ratio[2]
                out_close[out_idx] = scale * ratio[3]
            write_pos += copy_n

    def _mc_eval_metrics_numba(
        hist_rets,
        hist_ohlc_ratios,
        start_price,
        capital,
        fee_rate,
        max_loss,
        max_severe,
        max_dd50,
        mc_bars,
        block_size,
        total_paths,
        add_drop,
        multiplier,
        max_orders,
        tp,
        plan_code,
        path_plan,
    ):
        n = add_drop.shape[0]
        terminal_mean = np.empty(n, dtype=np.float64)
        terminal_median = np.empty(n, dtype=np.float64)
        terminal_p5 = np.empty(n, dtype=np.float64)
        p_loss = np.empty(n, dtype=np.float64)
        p_severe = np.empty(n, dtype=np.float64)
        p_dd50 = np.empty(n, dtype=np.float64)
        mdd_mean = np.empty(n, dtype=np.float64)
        trapped_mean = np.empty(n, dtype=np.float64)
        feasible = np.empty(n, dtype=bool)
        paths_evaluated = np.empty(n, dtype=np.int64)
        early_rejected = np.zeros(n, dtype=bool)

        one_path_open = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_high = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_low = np.empty(mc_bars + 1, dtype=np.float64)
        one_path_close = np.empty(mc_bars + 1, dtype=np.float64)
        # Reused per candidate; avoids holding candidates x paths terminal values.
        terminal_samples = np.empty(total_paths, dtype=np.float64)

        for j in range(n):
            terminal_sum = 0.0
            mdd_sum = 0.0
            trap_sum = 0.0
            loss_count = 0
            severe_count = 0
            dd50_count = 0
            done = 0

            for p in range(total_paths):
                plan_row = path_plan[p]
                _fill_ohlc_from_seed(
                    hist_ohlc_ratios, plan_row, block_size, start_price,
                    one_path_open, one_path_high, one_path_low, one_path_close,
                )

                fe_i, mdd_i, _, trap_i = _backtest_core_ohlc_numba_local(
                    one_path_open,
                    one_path_high,
                    one_path_low,
                    one_path_close,
                    float(add_drop[j]),
                    float(multiplier[j]),
                    int(max_orders[j]),
                    float(tp[j]),
                    capital,
                    fee_rate,
                )
                terminal = float(fe_i)
                mdd = float(mdd_i)
                trap = float(trap_i)

                terminal_samples[p] = terminal
                terminal_sum += terminal
                mdd_sum += mdd
                trap_sum += trap
                if terminal < capital:
                    loss_count += 1
                if terminal < (0.5 * capital):
                    severe_count += 1
                if mdd > 50.0:
                    dd50_count += 1

                done = p + 1
                if (
                    (loss_count / float(total_paths) > max_loss)
                    or (severe_count / float(total_paths) > max_severe)
                    or (dd50_count / float(total_paths) > max_dd50)
                ):
                    early_rejected[j] = True
                    break

            paths_evaluated[j] = done
            terminal_mean[j] = terminal_sum / float(done)
            terminal_median[j] = float(np.percentile(terminal_samples[:done], 50))
            terminal_p5[j] = float(np.percentile(terminal_samples[:done], 5))
            p_loss[j] = loss_count / float(total_paths)
            p_severe[j] = severe_count / float(total_paths)
            p_dd50[j] = dd50_count / float(total_paths)
            mdd_mean[j] = mdd_sum / float(done)
            trapped_mean[j] = trap_sum / float(done)
            feasible[j] = (
                (not early_rejected[j])
                and (p_loss[j] <= max_loss)
                and (p_severe[j] <= max_severe)
                and (p_dd50[j] <= max_dd50)
            )
            if early_rejected[j]:
                terminal_mean[j] = np.nan
                terminal_median[j] = np.nan
                terminal_p5[j] = np.nan
                mdd_mean[j] = np.nan
                trapped_mean[j] = np.nan

        return (
            terminal_mean,
            terminal_median,
            terminal_p5,
            p_loss,
            p_severe,
            p_dd50,
            mdd_mean,
            trapped_mean,
            feasible,
            paths_evaluated,
            early_rejected,
        )


def _mc_eval_chunk_worker(payload: dict) -> dict:
    hist_rets = payload["hist_rets"]
    hist_ohlc_ratios = payload["hist_ohlc_ratios"]
    start_price = float(payload["start_price"])
    capital = float(payload["capital"])
    fee_rate = float(payload["fee_rate"])
    mc_bars = int(payload["mc_bars"])
    block_size = int(payload["block_size"])
    total_paths = int(payload["total_paths"])
    base_seed = int(payload["base_seed"])
    max_loss = float(payload["max_loss"])
    max_severe = float(payload["max_severe"])
    max_dd50 = float(payload["max_dd50"])
    add_drop = payload["add_drop"]
    multiplier = payload["multiplier"]
    max_orders = payload["max_orders"]
    tp = payload["tp"]
    plan_code = int(payload["plan_code"])
    path_plan = payload["path_plan"]
    metrics_fn = _mc_eval_metrics_numba
    if _NUMBA_LOCAL and max_loss >= 1.0 and max_severe >= 1.0 and max_dd50 >= 1.0:
        metrics_fn = _mc_eval_metrics_onepass_numba

    (
        terminal_mean,
        terminal_median,
        terminal_p5,
        p_loss,
        p_severe,
        p_dd50,
        mdd_mean,
        trapped_mean,
        feasible,
        paths_evaluated,
        early_rejected,
    ) = metrics_fn(
        np.asarray(hist_rets, dtype=np.float64),
        np.asarray(hist_ohlc_ratios, dtype=np.float64),
        start_price,
        capital,
        fee_rate,
        max_loss,
        max_severe,
        max_dd50,
        mc_bars,
        block_size,
        total_paths,
        np.asarray(add_drop, dtype=np.float64),
        np.asarray(multiplier, dtype=np.float64),
        np.asarray(max_orders, dtype=np.int32),
        np.asarray(tp, dtype=np.float64),
        plan_code,
        np.asarray(path_plan, dtype=np.uint64 if plan_code == 2 else np.int32),
    )

    return {
        "terminal_mean": terminal_mean,
        "terminal_median": terminal_median,
        "terminal_p5": terminal_p5,
        "p_loss": p_loss,
        "p_severe": p_severe,
        "p_dd50": p_dd50,
        "mdd_mean": mdd_mean,
        "trapped_mean": trapped_mean,
        "feasible": feasible,
        "paths_evaluated": paths_evaluated,
        "early_rejected": early_rejected,
    }


def eval_candidates_parallel(
    *,
    candidates,
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
    hist_ohlc_ratios: np.ndarray | None = None,
    workers: int = 0,
    executor: ProcessPoolExecutor | None = None,
):
    n = len(candidates)
    if n == 0:
        return candidates.copy()

    hist_rets = np.asarray(hist_rets, dtype=np.float64)
    if hist_rets.ndim != 1 or hist_rets.size < 2 or not np.isfinite(hist_rets).all():
        raise ValueError("hist_rets 必須是一維、有限且至少包含 2 筆資料")
    if np.any(hist_rets <= -1.0):
        raise ValueError("歷史報酬不可 <= -100%")
    if hist_ohlc_ratios is None:
        close_rel = 1.0 + hist_rets
        hist_ohlc_ratios = np.column_stack(
            (close_rel, close_rel, close_rel, close_rel)
        )
    hist_ohlc_ratios = _validate_ohlc_ratios(
        hist_ohlc_ratios, hist_rets.shape[0]
    )
    if not np.isfinite(start_price) or start_price <= 0:
        raise ValueError("start_price 必須為有限正數")
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital 必須為有限正數")
    if not np.isfinite(fee_rate) or not (0.0 <= fee_rate < 1.0):
        raise ValueError("fee_rate 必須滿足 0 <= fee_rate < 1")
    if any((not np.isfinite(v)) or v < 0.0 or v > 1.0 for v in (max_loss, max_severe, max_dd50)):
        raise ValueError("MC 風險門檻必須介於 0% 到 100%")

    if workers <= 0:
        workers = max(1, min(os.cpu_count() or 1, 8))
    workers = min(workers, n)

    rng = np.random.default_rng(int(base_seed))
    plan_kind, path_plan = build_bootstrap_plan(
        hist_rets=hist_rets,
        n_bars=int(mc_bars),
        block_size=int(block_size),
        total_paths=int(total_paths),
        rng=rng,
    )
    plan_code = _plan_kind_code(plan_kind)

    add_drop = candidates["add_drop"].to_numpy(dtype=np.float64)
    multiplier = candidates["multiplier"].to_numpy(dtype=np.float64)
    max_orders = candidates["max_orders"].to_numpy(dtype=np.int32)
    tp = candidates["tp"].to_numpy(dtype=np.float64)
    idx_chunks = np.array_split(np.arange(n), workers)

    out = candidates.reset_index(drop=True).copy()
    for col in (
        "mc_terminal_mean", "mc_terminal_median", "mc_terminal_p5",
        "mc_p_loss", "mc_p_severe", "mc_p_dd50",
        "mc_mdd_mean", "mc_trapped_mean",
    ):
        out[col] = np.nan
    out["feasible"] = False
    out["mc_paths_evaluated"] = 0
    out["mc_early_rejected"] = False

    own_executor = False
    ex = executor
    if ex is None:
        own_executor = True
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))

    try:
        futures = []
        for idxs in idx_chunks:
            if idxs.size == 0:
                continue
            payload = {
                "hist_rets": hist_rets,
                "hist_ohlc_ratios": hist_ohlc_ratios,
                "start_price": float(start_price),
                "capital": float(capital),
                "fee_rate": float(fee_rate),
                "mc_bars": int(mc_bars),
                "block_size": int(block_size),
                "total_paths": int(total_paths),
                "base_seed": int(base_seed),
                "max_loss": float(max_loss),
                "max_severe": float(max_severe),
                "max_dd50": float(max_dd50),
                "add_drop": add_drop[idxs],
                "multiplier": multiplier[idxs],
                "max_orders": max_orders[idxs],
                "tp": tp[idxs],
                "plan_code": int(plan_code),
                "path_plan": path_plan,
            }
            futures.append((idxs, ex.submit(_mc_eval_chunk_worker, payload)))

        for idxs, fut in futures:
            r = fut.result()
            out.loc[idxs, "mc_terminal_mean"] = r["terminal_mean"]
            out.loc[idxs, "mc_terminal_median"] = r["terminal_median"]
            out.loc[idxs, "mc_terminal_p5"] = r["terminal_p5"]
            out.loc[idxs, "mc_p_loss"] = r["p_loss"]
            out.loc[idxs, "mc_p_severe"] = r["p_severe"]
            out.loc[idxs, "mc_p_dd50"] = r["p_dd50"]
            out.loc[idxs, "mc_mdd_mean"] = r["mdd_mean"]
            out.loc[idxs, "mc_trapped_mean"] = r["trapped_mean"]
            out.loc[idxs, "feasible"] = r["feasible"]
            out.loc[idxs, "mc_paths_evaluated"] = r["paths_evaluated"]
            out.loc[idxs, "mc_early_rejected"] = r["early_rejected"]
    finally:
        if own_executor and ex is not None:
            ex.shutdown(wait=True, cancel_futures=False)

    return out
