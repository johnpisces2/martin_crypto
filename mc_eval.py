# -*- coding: utf-8 -*-
"""Parallel MC evaluation utilities."""

from __future__ import annotations

import math
import os
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

    out = np.empty(n_bars, dtype=np.float64)
    if block_size <= 1:
        idx = rng.integers(0, n_hist, size=n_bars)
        out[:] = hist_rets[idx]
        return out

    max_start = n_hist - block_size + 1
    if max_start <= 1:
        idx = rng.integers(0, n_hist, size=n_bars)
        out[:] = hist_rets[idx]
        return out

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


def build_bootstrap_plan(
    hist_rets: np.ndarray,
    n_bars: int,
    block_size: int,
    total_paths: int,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    """Precompute a shared bootstrap sampling plan for all workers."""
    n_hist = hist_rets.shape[0]
    if n_hist < 2:
        raise ValueError("歷史報酬樣本不足，至少需要 2 個報酬點。")
    if total_paths <= 0:
        raise ValueError("total_paths 必須 > 0")

    direct_sample = (block_size <= 1) or (n_hist - block_size + 1 <= 1)
    if direct_sample:
        plan = np.empty((total_paths, n_bars), dtype=np.int32)
        for p in range(total_paths):
            plan[p, :] = rng.integers(0, n_hist, size=n_bars, dtype=np.int32)
        return "indices", plan

    n_blocks = int(math.ceil(n_bars / block_size))
    max_start = n_hist - block_size + 1
    plan = np.empty((total_paths, n_blocks), dtype=np.int32)
    for p in range(total_paths):
        plan[p, :] = rng.integers(0, max_start, size=n_blocks, dtype=np.int32)
    return "blocks", plan


def _plan_kind_code(plan_kind: str) -> int:
    return 0 if plan_kind == "indices" else 1


if _NUMBA_LOCAL:
    # Single source of truth for strategy mechanics; MC shares martin.py's grid core.
    _backtest_core_numba_local = martin._backtest_core


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

        one_ret = np.empty(mc_bars, dtype=np.float64)
        one_path_prices = np.empty(mc_bars + 1, dtype=np.float64)
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
                if plan_code == 0:
                    _fill_returns_from_indices(hist_rets, plan_row, one_ret)
                else:
                    _fill_returns_from_blocks(hist_rets, plan_row, block_size, one_ret)
                _returns_to_prices(start_price, one_ret, one_path_prices)

                fe_i, mdd_i, _, trap_i = _backtest_core_numba_local(
                    one_path_prices,
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

        one_ret = np.empty(mc_bars, dtype=np.float64)
        one_path_prices = np.empty(mc_bars + 1, dtype=np.float64)

        # First pass: build each MC path once and screen all active candidates.
        for p in range(total_paths):
            if active_count <= 0:
                break

            plan_row = path_plan[p]
            if plan_code == 0:
                _fill_returns_from_indices(hist_rets, plan_row, one_ret)
            else:
                _fill_returns_from_blocks(hist_rets, plan_row, block_size, one_ret)
            _returns_to_prices(start_price, one_ret, one_path_prices)

            done = p + 1
            for j in range(n):
                if not active[j]:
                    continue

                fe_i, mdd_i, _, trap_i = _backtest_core_numba_local(
                    one_path_prices,
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
                if plan_code == 0:
                    _fill_returns_from_indices(hist_rets, plan_row, one_ret)
                else:
                    _fill_returns_from_blocks(hist_rets, plan_row, block_size, one_ret)
                _returns_to_prices(start_price, one_ret, one_path_prices)

                for s in range(survivor_count):
                    j = survivor_idx[s]
                    fe_i, _, _, _ = _backtest_core_numba_local(
                        one_path_prices,
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

    def _returns_to_prices(start_price, one_ret, out_prices):
        out_prices[0] = start_price
        np.multiply.accumulate(1.0 + one_ret, out=out_prices[1:])
        out_prices[1:] *= start_price

    def _mc_eval_metrics_numba(
        hist_rets,
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

        one_ret = np.empty(mc_bars, dtype=np.float64)
        one_path_prices = np.empty(mc_bars + 1, dtype=np.float64)
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
                if plan_code == 0:
                    _fill_returns_from_indices(hist_rets, plan_row, one_ret)
                else:
                    _fill_returns_from_blocks(hist_rets, plan_row, block_size, one_ret)
                _returns_to_prices(start_price, one_ret, one_path_prices)

                fe_i, mdd_i, _, trap_i = _backtest_core_numba_local(
                    one_path_prices,
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
        np.asarray(path_plan, dtype=np.int32),
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
    workers: int = 0,
    executor: ProcessPoolExecutor | None = None,
):
    n = len(candidates)
    if n == 0:
        return candidates.copy()

    if workers <= 0:
        workers = max(1, min(os.cpu_count() or 1, 8))
    workers = min(workers, n)

    rng = np.random.default_rng(int(base_seed))
    plan_kind, path_plan = build_bootstrap_plan(
        hist_rets=np.asarray(hist_rets, dtype=np.float64),
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
        ex = ProcessPoolExecutor(max_workers=workers)

    try:
        futures = []
        for idxs in idx_chunks:
            if idxs.size == 0:
                continue
            payload = {
                "hist_rets": hist_rets,
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
