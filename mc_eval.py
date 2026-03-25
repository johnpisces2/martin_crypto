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
    @njit(cache=True)
    def _calc_init_order_local(capital, multiplier, max_orders):
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
    def _backtest_core_numba_local(prices, add_drop, multiplier, max_orders, tp, capital, fee_rate):
        trapped_bars = 0
        total_bars = prices.shape[0]
        if total_bars == 0:
            return np.nan, np.nan, 0, 0.0

        cash = capital
        qty = 0.0
        cost = 0.0
        order_count = 0
        init_order_round = 0.0
        base_price = 0.0
        next_level_idx = 1

        peak_equity_overall = capital
        max_drawdown_overall = 0.0
        trades = 0
        round_cost_sum = 0.0
        round_fee_sum = 0.0

        for i in range(prices.shape[0]):
            price = prices[i]
            if (qty == 0.0) and (cash > 0.0):
                init_order_round = _calc_init_order_local(cash, multiplier, max_orders)
                alloc = init_order_round
                max_afford = cash / (1.0 + fee_rate)
                if alloc > max_afford:
                    alloc = max_afford
                if alloc > 0.0:
                    qty += alloc / price
                    cost += alloc
                    fee = alloc * fee_rate
                    cash -= (alloc + fee)
                    base_price = price
                    next_level_idx = 1
                    order_count = 1
                    round_cost_sum = alloc
                    round_fee_sum = fee
            elif qty > 0.0:
                prospective_proceeds = qty * price * (1.0 - fee_rate)
                prospective_pnl = prospective_proceeds - round_cost_sum - round_fee_sum
                target_pnl = round_cost_sum * tp
                if prospective_pnl >= target_pnl:
                    cash += prospective_proceeds
                    qty = 0.0
                    cost = 0.0
                    trades += 1
                    order_count = 0
                    init_order_round = 0.0
                    round_cost_sum = 0.0
                    round_fee_sum = 0.0
                    base_price = 0.0
                    next_level_idx = 1
                elif order_count < max_orders and cash > 0.0 and base_price > 0.0:
                    r = (1.0 - add_drop)
                    if r > 0.0 and price <= base_price:
                        ratio = price / base_price if base_price > 0 else 0.0
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
                                round_cost_sum += alloc
                                round_fee_sum += fee
                                order_count += 1
                                next_level_idx += 1

            equity = cash + qty * price
            if equity > peak_equity_overall:
                peak_equity_overall = equity
            dd = (equity - peak_equity_overall) / peak_equity_overall
            avg_cost = (cost / qty) if qty > 0.0 else 1e18
            is_trapped = (order_count == max_orders) and (qty > 0.0) and (price < avg_cost)
            if is_trapped:
                trapped_bars += 1
            if dd < max_drawdown_overall:
                max_drawdown_overall = dd

        final_equity = cash + qty * prices[-1]
        trapped_ratio = (trapped_bars / total_bars) if total_bars > 0 else 0.0
        return final_equity, abs(max_drawdown_overall * 100.0), trades, trapped_ratio


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
    def _mc_eval_metrics_numba(
        hist_rets,
        start_price,
        capital,
        fee_rate,
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
        terminals = np.empty((n, total_paths), dtype=np.float64)
        terminal_sum = np.zeros(n, dtype=np.float64)
        mdd_sum = np.zeros(n, dtype=np.float64)
        trap_sum = np.zeros(n, dtype=np.float64)
        loss_count = np.zeros(n, dtype=np.int64)
        severe_count = np.zeros(n, dtype=np.int64)
        dd50_count = np.zeros(n, dtype=np.int64)

        one_ret = np.empty(mc_bars, dtype=np.float64)
        one_path_prices = np.empty(mc_bars + 1, dtype=np.float64)

        for p in range(total_paths):
            plan_row = path_plan[p]
            if plan_code == 0:
                _fill_returns_from_indices(hist_rets, plan_row, one_ret)
            else:
                _fill_returns_from_blocks(hist_rets, plan_row, block_size, one_ret)
            _returns_to_prices(start_price, one_ret, one_path_prices)

            for j in range(n):
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

                terminals[j, p] = terminal
                terminal_sum[j] += terminal
                mdd_sum[j] += mdd
                trap_sum[j] += trap
                if terminal < capital:
                    loss_count[j] += 1
                if terminal < (0.5 * capital):
                    severe_count[j] += 1
                if mdd > 50.0:
                    dd50_count[j] += 1

        return terminals, terminal_sum, mdd_sum, trap_sum, loss_count, severe_count, dd50_count
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
        terminals = np.empty((n, total_paths), dtype=np.float64)
        terminal_sum = np.zeros(n, dtype=np.float64)
        mdd_sum = np.zeros(n, dtype=np.float64)
        trap_sum = np.zeros(n, dtype=np.float64)
        loss_count = np.zeros(n, dtype=np.int64)
        severe_count = np.zeros(n, dtype=np.int64)
        dd50_count = np.zeros(n, dtype=np.int64)

        one_ret = np.empty(mc_bars, dtype=np.float64)
        one_path_prices = np.empty(mc_bars + 1, dtype=np.float64)

        for p in range(total_paths):
            plan_row = path_plan[p]
            if plan_code == 0:
                _fill_returns_from_indices(hist_rets, plan_row, one_ret)
            else:
                _fill_returns_from_blocks(hist_rets, plan_row, block_size, one_ret)
            _returns_to_prices(start_price, one_ret, one_path_prices)

            for j in range(n):
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

                terminals[j, p] = terminal
                terminal_sum[j] += terminal
                mdd_sum[j] += mdd
                trap_sum[j] += trap
                if terminal < capital:
                    loss_count[j] += 1
                if terminal < (0.5 * capital):
                    severe_count[j] += 1
                if mdd > 50.0:
                    dd50_count[j] += 1

        return terminals, terminal_sum, mdd_sum, trap_sum, loss_count, severe_count, dd50_count


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

    terminals, terminal_sum, mdd_sum, trap_sum, loss_count, severe_count, dd50_count = _mc_eval_metrics_numba(
        np.asarray(hist_rets, dtype=np.float64),
        start_price,
        capital,
        fee_rate,
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

    p_loss = loss_count / float(total_paths)
    p_severe = severe_count / float(total_paths)
    p_dd50 = dd50_count / float(total_paths)
    feasible = (p_loss <= max_loss) & (p_severe <= max_severe) & (p_dd50 <= max_dd50)

    return {
        "terminal_mean": terminal_sum / float(total_paths),
        "terminal_median": np.median(terminals, axis=1),
        "terminal_p5": np.percentile(terminals, 5, axis=1),
        "p_loss": p_loss,
        "p_severe": p_severe,
        "p_dd50": p_dd50,
        "mdd_mean": mdd_sum / float(total_paths),
        "trapped_mean": trap_sum / float(total_paths),
        "feasible": feasible,
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
    finally:
        if own_executor and ex is not None:
            ex.shutdown(wait=True, cancel_futures=False)

    return out
