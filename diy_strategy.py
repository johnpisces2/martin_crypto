# -*- coding: utf-8 -*-
"""MAE-derived price ladders and risk-budgeted Pionex DIY share templates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit, prange


DEFAULT_MAX_MAE_GRID_BYTES = 512 * 1024 * 1024
MAE_GRID_BYTES_PER_SAMPLE = (
    np.dtype(np.float64).itemsize
    + np.dtype(np.uint8).itemsize
    + np.dtype(np.int32).itemsize
)


def estimate_tp_or_horizon_mae_grid_bytes(
    price_count: int,
    horizon_bars: int,
    tp_count: int,
) -> int:
    """Return the exact output-array bytes required by the MAE grid."""
    price_count = int(price_count)
    horizon_bars = int(horizon_bars)
    tp_count = int(tp_count)
    sample_count = price_count - horizon_bars
    if sample_count <= 0 or tp_count <= 0:
        return 0
    return int(sample_count * tp_count * MAE_GRID_BYTES_PER_SAMPLE)


@njit(cache=True)
def _build_sparse_extrema(highs, lows):
    """Build O(1) inclusive range max/min tables."""
    n = highs.shape[0]
    logs = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        logs[i] = logs[i // 2] + 1
    level_count = int(logs[n]) + 1
    max_table = np.empty((level_count, n), dtype=np.float64)
    min_table = np.empty((level_count, n), dtype=np.float64)
    max_table[0, :] = highs
    min_table[0, :] = lows
    span = 2
    for level in range(1, level_count):
        half = span // 2
        valid = n - span + 1
        for i in range(valid):
            left_max = max_table[level - 1, i]
            right_max = max_table[level - 1, i + half]
            max_table[level, i] = (
                left_max if left_max >= right_max else right_max
            )
            left_min = min_table[level - 1, i]
            right_min = min_table[level - 1, i + half]
            min_table[level, i] = (
                left_min if left_min <= right_min else right_min
            )
        span *= 2
    return logs, max_table, min_table


@njit(cache=True, inline="always")
def _sparse_range_max(table, logs, left, right):
    length = right - left + 1
    level = logs[length]
    span = 1 << level
    a = table[level, left]
    b = table[level, right - span + 1]
    return a if a >= b else b


@njit(cache=True, inline="always")
def _sparse_range_min(table, logs, left, right):
    length = right - left + 1
    level = logs[length]
    span = 1 << level
    a = table[level, left]
    b = table[level, right - span + 1]
    return a if a <= b else b


@njit(parallel=True, cache=True)
def _tp_or_horizon_mae_grid_from_sparse(
    closes,
    tp_values,
    horizon_bars,
    fee_rate,
    logs,
    max_table,
    min_table,
):
    """Find the first unambiguous TP passage, otherwise stop at horizon."""
    sample_count = closes.shape[0] - horizon_bars
    tp_count = tp_values.shape[0]
    mae = np.empty((tp_count, sample_count), dtype=np.float64)
    hit = np.empty((tp_count, sample_count), dtype=np.uint8)
    bars_to_event = np.empty((tp_count, sample_count), dtype=np.int32)
    sell_fee_mult = 1.0 - fee_rate

    for tp_index in prange(tp_count):
        tp = tp_values[tp_index]
        target_ratio = (1.0 + fee_rate + tp) / sell_fee_mult
        for entry_index in range(sample_count):
            left = entry_index + 1
            horizon_end = entry_index + horizon_bars
            target = closes[entry_index] * target_ratio
            event_end = horizon_end
            did_hit = 0

            # A bar whose low is below the entry price could have triggered a
            # candidate safety order before its high touched TP. The real
            # backtest deliberately forbids TP on a bar that adds an order.
            # Because ladder levels do not exist yet at this stage, accept only
            # a passage whose low cannot have reached any strictly-lower DIY
            # level. Ambiguous high passages are skipped conservatively.
            search_left = left
            while (
                search_left <= horizon_end
                and _sparse_range_max(
                    max_table, logs, search_left, horizon_end
                ) >= target
            ):
                lo = search_left
                hi = horizon_end
                while lo < hi:
                    mid = (lo + hi) // 2
                    if _sparse_range_max(
                        max_table, logs, search_left, mid
                    ) >= target:
                        hi = mid
                    else:
                        lo = mid + 1
                candidate_end = lo
                if (
                    min_table[0, candidate_end]
                    >= closes[entry_index] * (1.0 - 1e-12)
                ):
                    event_end = candidate_end
                    did_hit = 1
                    break
                search_left = candidate_end + 1

            # Include the accepted TP bar's low. Non-hit rows are exact
            # TP-or-horizon MAE observations, not uncensored eventual-TP MAE.
            future_low = _sparse_range_min(
                min_table, logs, left, event_end
            )
            adverse = 1.0 - future_low / closes[entry_index]
            mae[tp_index, entry_index] = adverse if adverse > 0.0 else 0.0
            hit[tp_index, entry_index] = did_hit
            bars_to_event[tp_index, entry_index] = event_end - entry_index
    return mae, hit, bars_to_event


def compute_tp_or_horizon_mae_grid(
    highs,
    lows,
    closes,
    horizon_bars: int,
    tp_values,
    fee_rate: float = 0.0,
    max_output_bytes: int | None = DEFAULT_MAX_MAE_GRID_BYTES,
):
    """Compute MAE until an unambiguous TP hit or the configured horizon.

    A high passage is accepted only if that bar's low stayed at or above the
    entry price, so it cannot conflict with the backtest's add-before-TP rule
    for any strictly lower DIY level. Samples without such a passage keep their
    exact full-horizon MAE and have ``hit=False``. The resulting distribution is
    intentionally horizon-capped; it is not an estimate of eventual-TP MAE.
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    tp_values = np.asarray(tp_values, dtype=np.float64)
    horizon_bars = int(horizon_bars)
    fee_rate = float(fee_rate)
    if (
        highs.ndim != 1
        or lows.ndim != 1
        or closes.ndim != 1
        or highs.size != closes.size
        or lows.size != closes.size
    ):
        raise ValueError("TP/Horizon MAE high/low/close 必須是一維且長度一致")
    if (
        highs.size == 0
        or not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or not np.isfinite(closes).all()
        or np.any(highs <= 0.0)
        or np.any(lows <= 0.0)
        or np.any(closes <= 0.0)
    ):
        raise ValueError("TP/Horizon MAE OHLC 不可為空、非正數或包含 NaN/Inf")
    if np.any(highs < closes) or np.any(lows > closes) or np.any(highs < lows):
        raise ValueError("TP/Horizon MAE OHLC 關係異常")
    if horizon_bars <= 0 or closes.size <= horizon_bars:
        raise ValueError(
            f"TP/Horizon MAE horizon ({horizon_bars} bars) "
            f"必須小於歷史資料 ({closes.size} bars)"
        )
    if (
        tp_values.ndim != 1
        or tp_values.size == 0
        or not np.isfinite(tp_values).all()
        or np.any(tp_values <= 0.0)
    ):
        raise ValueError("TP/Horizon MAE tp_values 必須是非空有限正數")
    if not math.isfinite(fee_rate) or not (0.0 <= fee_rate < 1.0):
        raise ValueError("TP/Horizon MAE fee_rate 必須介於 0 與 1")
    output_bytes = estimate_tp_or_horizon_mae_grid_bytes(
        closes.size,
        horizon_bars,
        tp_values.size,
    )
    if max_output_bytes is not None:
        max_output_bytes = int(max_output_bytes)
        if max_output_bytes <= 0:
            raise ValueError("TP/Horizon MAE max_output_bytes 必須 > 0 或為 None")
        if output_bytes > max_output_bytes:
            raise ValueError(
                "TP/Horizon MAE 輸出矩陣預估需要 "
                f"{output_bytes / (1024 ** 2):,.1f} MiB，超過 "
                f"{max_output_bytes / (1024 ** 2):,.1f} MiB 限制；"
                "請放大 TP step、縮小 TP 範圍或縮短資料期間"
            )

    logs, max_table, min_table = _build_sparse_extrema(highs, lows)
    return _tp_or_horizon_mae_grid_from_sparse(
        closes,
        tp_values,
        horizon_bars,
        fee_rate,
        logs,
        max_table,
        min_table,
    )


def _strict_deviations(values, *, min_gap=1e-6, max_deviation=0.99):
    out = np.asarray(values, dtype=np.float64).copy()
    previous = 0.0
    for i in range(out.size):
        value = max(float(out[i]), previous + float(min_gap))
        if not math.isfinite(value) or value >= max_deviation:
            raise ValueError("MAE 分位數產生無效或過深的 DIY 買入層級")
        out[i] = value
        previous = value
    return out


def mae_quantile_ladder(
    mae_samples,
    max_orders: int,
    q_start: float,
    q_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [initial + safety] price ratios from evenly spaced MAE quantiles."""
    samples = np.asarray(mae_samples, dtype=np.float64)
    max_orders = int(max_orders)
    if max_orders < 2:
        raise ValueError("DIY max_orders 必須 >= 2")
    if (
        samples.ndim != 1
        or samples.size == 0
        or not np.isfinite(samples).all()
        or np.any(samples < 0.0)
        or np.any(samples >= 1.0)
    ):
        raise ValueError("MAE samples 必須為非空、有限且介於 0（含）與 1 之間")
    if not (0.0 < q_start < q_end < 1.0):
        raise ValueError("MAE quantile 必須滿足 0 < q_start < q_end < 1")

    quantiles = np.linspace(float(q_start), float(q_end), max_orders - 1)
    deviations = _strict_deviations(np.quantile(samples, quantiles))
    level_ratios = np.empty(max_orders, dtype=np.float64)
    level_ratios[0] = 1.0
    level_ratios[1:] = 1.0 - deviations
    return level_ratios, np.concatenate(([0.0], quantiles))


def capital_curve_weights(
    level_ratios,
    initial_fraction: float,
    gamma: float,
) -> np.ndarray:
    """Create normalized order weights from a cumulative deployment curve."""
    levels = np.asarray(level_ratios, dtype=np.float64)
    initial_fraction = float(initial_fraction)
    gamma = float(gamma)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("DIY level ratios 至少需要首單與一筆加倉")
    if (
        not np.isfinite(levels).all()
        or levels[0] != 1.0
        or np.any(levels <= 0.0)
        or np.any(np.diff(levels) >= 0.0)
    ):
        raise ValueError("DIY level ratios 必須由 1.0 開始並嚴格遞減")
    if not (0.0 < initial_fraction < 1.0):
        raise ValueError("DIY initial capital fraction 必須介於 0 與 1")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("DIY capital gamma 必須為有限正數")

    deviations = 1.0 - levels
    max_deviation = float(deviations[-1])
    cumulative = np.empty(levels.size, dtype=np.float64)
    cumulative[0] = initial_fraction
    cumulative[1:] = initial_fraction + (1.0 - initial_fraction) * np.power(
        deviations[1:] / max_deviation, gamma
    )
    weights = np.diff(np.concatenate(([0.0], cumulative)))
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("DIY 資金曲線產生無效權重")
    return weights


def idle_aware_capital_weights(
    base_weights,
    fill_probabilities,
    idle_bias: float,
) -> np.ndarray:
    """Shift safety-order capital toward levels that are reached more often.

    The initial allocation is preserved. ``idle_bias=0`` is exactly the base
    risk curve; larger values progressively discount rare tail orders.
    """
    weights = np.asarray(base_weights, dtype=np.float64)
    fill_probabilities = np.asarray(fill_probabilities, dtype=np.float64)
    idle_bias = float(idle_bias)
    if (
        weights.ndim != 1
        or fill_probabilities.shape != weights.shape
        or weights.size < 2
        or not np.isfinite(weights).all()
        or not np.isfinite(fill_probabilities).all()
        or np.any(weights <= 0.0)
        or np.any(fill_probabilities < 0.0)
        or np.any(fill_probabilities > 1.0)
    ):
        raise ValueError("Idle-aware weights/fill probabilities 無效")
    if not math.isfinite(idle_bias) or idle_bias < 0.0:
        raise ValueError("Idle bias 必須為有限非負數")
    if idle_bias == 0.0:
        return weights.copy()

    initial = float(weights[0])
    safety_budget = 1.0 - initial
    adjusted = weights[1:] * np.power(
        np.maximum(fill_probabilities[1:], 1e-6),
        idle_bias,
    )
    adjusted_sum = float(adjusted.sum())
    if adjusted_sum <= 0.0 or not math.isfinite(adjusted_sum):
        raise ValueError("Idle-aware safety order 權重無效")
    out = np.empty_like(weights)
    out[0] = initial
    out[1:] = adjusted * (safety_budget / adjusted_sum)
    return out


def integer_shares(weights, total_shares: int) -> np.ndarray:
    """Round normalized weights to positive integer Pionex shares."""
    weights = np.asarray(weights, dtype=np.float64)
    total_shares = int(total_shares)
    if weights.ndim != 1 or weights.size == 0 or np.any(weights <= 0.0):
        raise ValueError("DIY weights 必須為非空正數")
    if total_shares < weights.size:
        raise ValueError("DIY total shares 不得小於 max_orders")

    normalized = weights / weights.sum()
    raw = normalized * total_shares
    shares = np.floor(raw).astype(np.int64)
    shares[shares < 1] = 1

    while int(shares.sum()) < total_shares:
        idx = int(np.argmax(raw - shares))
        shares[idx] += 1
    while int(shares.sum()) > total_shares:
        removable = shares > 1
        if not np.any(removable):
            raise ValueError("DIY shares 無法在保持每層至少 1 share 下完成取整")
        excess = np.where(removable, shares - raw, -np.inf)
        idx = int(np.argmax(excess))
        shares[idx] -= 1
    return shares


def stress_loss_pct(level_ratios, order_shares, stress_deviation, fee_rate) -> float:
    """Mark the filled ladder at a stress price, leaving untriggered cash idle."""
    levels = np.asarray(level_ratios, dtype=np.float64)
    shares = np.asarray(order_shares, dtype=np.float64)
    stress_deviation = float(stress_deviation)
    fee_rate = float(fee_rate)
    if levels.shape != shares.shape or levels.ndim != 1:
        raise ValueError("DIY levels/shares 長度必須一致")
    if not (0.0 <= stress_deviation < 1.0):
        raise ValueError("stress deviation 必須介於 0 與 1")
    if not (0.0 <= fee_rate < 1.0):
        raise ValueError("fee_rate 必須介於 0 與 1")

    stress_price = 1.0 - stress_deviation
    unit = 1.0 / ((1.0 + fee_rate) * float(shares.sum()))
    cash = 1.0
    qty = 0.0
    for i in range(levels.size):
        if i > 0 and stress_price > levels[i] * (1.0 + 1e-12):
            break
        allocation = unit * shares[i]
        cash -= allocation * (1.0 + fee_rate)
        qty += allocation / levels[i]
    equity = cash + qty * stress_price * (1.0 - fee_rate)
    return max(0.0, (1.0 - equity) * 100.0)


@njit(cache=True)
def _expand_template_allocations(
    ladder_levels,
    ladder_fill_probabilities,
    ladder_max_orders,
    initial_fraction_values,
    gamma_values,
    idle_bias_values,
    stress_deviation,
    total_shares,
    fee_rate,
    use_max_last_order,
    max_last_order_pct,
    use_max_stress_loss,
    max_stress_loss_pct,
):
    """Expand all capital curves for precomputed ladders in compiled code."""
    ladder_count = ladder_levels.shape[0]
    max_width = ladder_levels.shape[1]
    candidate_count = (
        ladder_count
        * initial_fraction_values.shape[0]
        * gamma_values.shape[0]
        * idle_bias_values.shape[0]
    )
    valid = np.zeros(candidate_count, dtype=np.uint8)
    ladder_index_out = np.empty(candidate_count, dtype=np.int32)
    initial_out = np.empty(candidate_count, dtype=np.float64)
    gamma_out = np.empty(candidate_count, dtype=np.float64)
    idle_out = np.empty(candidate_count, dtype=np.float64)
    share_out = np.zeros((candidate_count, max_width), dtype=np.int64)
    initial_pct_out = np.empty(candidate_count, dtype=np.float64)
    last_pct_out = np.empty(candidate_count, dtype=np.float64)
    stress_loss_out = np.empty(candidate_count, dtype=np.float64)
    expected_util_out = np.empty(candidate_count, dtype=np.float64)
    stress_price = 1.0 - stress_deviation
    unit = 1.0 / ((1.0 + fee_rate) * total_shares)

    candidate_index = 0
    for ladder_index in range(ladder_count):
        max_orders = int(ladder_max_orders[ladder_index])
        max_deviation = 1.0 - ladder_levels[ladder_index, max_orders - 1]
        for initial_index in range(initial_fraction_values.shape[0]):
            initial_fraction = initial_fraction_values[initial_index]
            for gamma_index in range(gamma_values.shape[0]):
                gamma = gamma_values[gamma_index]
                base_weights = np.empty(max_orders, dtype=np.float64)
                previous_cumulative = 0.0
                for order_index in range(max_orders):
                    if order_index == 0:
                        cumulative = initial_fraction
                    else:
                        deviation = 1.0 - ladder_levels[
                            ladder_index, order_index
                        ]
                        cumulative = initial_fraction + (
                            (1.0 - initial_fraction)
                            * (deviation / max_deviation) ** gamma
                        )
                    base_weights[order_index] = (
                        cumulative - previous_cumulative
                    )
                    previous_cumulative = cumulative

                for idle_index in range(idle_bias_values.shape[0]):
                    idle_bias = idle_bias_values[idle_index]
                    weights = np.empty(max_orders, dtype=np.float64)
                    weights[0] = initial_fraction
                    if idle_bias == 0.0:
                        for order_index in range(1, max_orders):
                            weights[order_index] = base_weights[order_index]
                    else:
                        adjusted_sum = 0.0
                        for order_index in range(1, max_orders):
                            probability = ladder_fill_probabilities[
                                ladder_index, order_index
                            ]
                            if probability < 1e-6:
                                probability = 1e-6
                            adjusted = (
                                base_weights[order_index]
                                * probability ** idle_bias
                            )
                            weights[order_index] = adjusted
                            adjusted_sum += adjusted
                        scale = (1.0 - initial_fraction) / adjusted_sum
                        for order_index in range(1, max_orders):
                            weights[order_index] *= scale

                    raw = np.empty(max_orders, dtype=np.float64)
                    share_sum = 0
                    for order_index in range(max_orders):
                        raw_value = weights[order_index] * total_shares
                        raw[order_index] = raw_value
                        share_value = int(math.floor(raw_value))
                        if share_value < 1:
                            share_value = 1
                        share_out[candidate_index, order_index] = share_value
                        share_sum += share_value

                    while share_sum < total_shares:
                        best_index = 0
                        best_value = -np.inf
                        for order_index in range(max_orders):
                            remainder = (
                                raw[order_index]
                                - share_out[candidate_index, order_index]
                            )
                            if remainder > best_value:
                                best_value = remainder
                                best_index = order_index
                        share_out[candidate_index, best_index] += 1
                        share_sum += 1

                    while share_sum > total_shares:
                        best_index = -1
                        best_value = -np.inf
                        for order_index in range(max_orders):
                            share_value = share_out[
                                candidate_index, order_index
                            ]
                            if share_value > 1:
                                excess = share_value - raw[order_index]
                                if excess > best_value:
                                    best_value = excess
                                    best_index = order_index
                        if best_index < 0:
                            break
                        share_out[candidate_index, best_index] -= 1
                        share_sum -= 1

                    last_pct = (
                        100.0
                        * share_out[candidate_index, max_orders - 1]
                        / total_shares
                    )
                    cash = 1.0
                    qty = 0.0
                    for order_index in range(max_orders):
                        level = ladder_levels[ladder_index, order_index]
                        if (
                            order_index > 0
                            and stress_price > level * (1.0 + 1e-12)
                        ):
                            break
                        allocation = (
                            unit
                            * share_out[candidate_index, order_index]
                        )
                        cash -= allocation * (1.0 + fee_rate)
                        qty += allocation / level
                    equity = cash + qty * stress_price * (1.0 - fee_rate)
                    stress_loss = max(0.0, (1.0 - equity) * 100.0)

                    expected_utilization = 0.0
                    for order_index in range(max_orders):
                        expected_utilization += (
                            share_out[candidate_index, order_index]
                            / total_shares
                            * ladder_fill_probabilities[
                                ladder_index, order_index
                            ]
                        )

                    is_valid = True
                    if (
                        use_max_last_order
                        and last_pct > max_last_order_pct + 1e-12
                    ):
                        is_valid = False
                    if (
                        use_max_stress_loss
                        and stress_loss > max_stress_loss_pct + 1e-12
                    ):
                        is_valid = False

                    valid[candidate_index] = 1 if is_valid else 0
                    ladder_index_out[candidate_index] = ladder_index
                    initial_out[candidate_index] = initial_fraction
                    gamma_out[candidate_index] = gamma
                    idle_out[candidate_index] = idle_bias
                    initial_pct_out[candidate_index] = (
                        100.0
                        * share_out[candidate_index, 0]
                        / total_shares
                    )
                    last_pct_out[candidate_index] = last_pct
                    stress_loss_out[candidate_index] = stress_loss
                    expected_util_out[candidate_index] = expected_utilization
                    candidate_index += 1

    return (
        valid,
        ladder_index_out,
        initial_out,
        gamma_out,
        idle_out,
        share_out,
        initial_pct_out,
        last_pct_out,
        stress_loss_out,
        expected_util_out,
    )


def build_diy_templates(
    *,
    mae_samples,
    max_orders_values,
    q_start: float,
    q_end_values,
    initial_fraction_values,
    gamma_values,
    stress_quantile: float,
    total_shares: int,
    fee_rate: float,
    max_last_order_pct: float | None,
    max_stress_loss_pct: float | None,
    idle_bias_values=None,
):
    """Return risk-filtered templates and compact padded matrices for Numba."""
    samples = np.asarray(mae_samples, dtype=np.float64)
    if (
        samples.ndim != 1
        or samples.size == 0
        or not np.isfinite(samples).all()
        or np.any(samples < 0.0)
        or np.any(samples >= 1.0)
    ):
        raise ValueError("MAE samples 必須為非空、有限且介於 0（含）與 1 之間")
    stress_quantile = float(stress_quantile)
    if not (0.0 < stress_quantile < 1.0):
        raise ValueError("Stress MAE quantile 必須介於 0 與 1")
    stress_deviation = float(np.quantile(samples, stress_quantile))

    max_orders_array = np.asarray(max_orders_values, dtype=np.int64)
    if (
        max_orders_array.ndim != 1
        or max_orders_array.size == 0
        or np.any(max_orders_array < 2)
    ):
        raise ValueError("DIY max_orders 掃描值必須為非空且全部 >= 2")
    max_width = int(np.max(max_orders_array))
    if int(total_shares) < max_width:
        raise ValueError("DIY total shares 不得小於最大 max_orders")
    if not math.isfinite(float(fee_rate)) or not (0.0 <= fee_rate < 1.0):
        raise ValueError("DIY fee_rate 必須介於 0 與 1")
    idle_values = (
        np.array([0.0], dtype=np.float64)
        if idle_bias_values is None
        else np.asarray(idle_bias_values, dtype=np.float64)
    )
    if (
        idle_values.ndim != 1
        or idle_values.size == 0
        or not np.isfinite(idle_values).all()
        or np.any(idle_values < 0.0)
    ):
        raise ValueError("Idle bias 掃描值必須為非空有限非負數")
    initial_values = np.asarray(initial_fraction_values, dtype=np.float64)
    gamma_array = np.asarray(gamma_values, dtype=np.float64)
    if (
        initial_values.ndim != 1
        or initial_values.size == 0
        or not np.isfinite(initial_values).all()
        or np.any(initial_values <= 0.0)
        or np.any(initial_values >= 1.0)
    ):
        raise ValueError("Initial capital fraction 掃描值必須介於 0 與 1")
    if (
        gamma_array.ndim != 1
        or gamma_array.size == 0
        or not np.isfinite(gamma_array).all()
        or np.any(gamma_array <= 0.0)
    ):
        raise ValueError("Capital gamma 掃描值必須為有限正數")

    ladder_levels = []
    ladder_quantiles = []
    ladder_fill_probabilities = []
    ladder_max_orders = []
    ladder_q_end = []
    for max_orders_value in max_orders_array:
        max_orders = int(max_orders_value)
        for q_end_value in np.asarray(q_end_values, dtype=np.float64):
            q_end = float(q_end_value)
            if not (q_start < q_end <= stress_quantile):
                continue
            levels, quantiles = mae_quantile_ladder(
                samples, max_orders, float(q_start), q_end
            )
            deviations = 1.0 - levels
            fill_probabilities = np.ones(max_orders, dtype=np.float64)
            for level_index in range(1, max_orders):
                fill_probabilities[level_index] = float(
                    np.mean(samples >= deviations[level_index] - 1e-12)
                )
            level_row = np.zeros(max_width, dtype=np.float64)
            quantile_row = np.zeros(max_width, dtype=np.float64)
            probability_row = np.zeros(max_width, dtype=np.float64)
            level_row[:max_orders] = levels
            quantile_row[:max_orders] = quantiles
            probability_row[:max_orders] = fill_probabilities
            ladder_levels.append(level_row)
            ladder_quantiles.append(quantile_row)
            ladder_fill_probabilities.append(probability_row)
            ladder_max_orders.append(max_orders)
            ladder_q_end.append(q_end)

    if not ladder_levels:
        return (
            pd.DataFrame(),
            np.empty((0, max_width), dtype=np.float64),
            np.empty((0, max_width), dtype=np.float64),
            stress_deviation,
        )

    ladder_level_matrix = np.ascontiguousarray(
        np.vstack(ladder_levels), dtype=np.float64
    )
    ladder_quantile_matrix = np.ascontiguousarray(
        np.vstack(ladder_quantiles), dtype=np.float64
    )
    ladder_probability_matrix = np.ascontiguousarray(
        np.vstack(ladder_fill_probabilities), dtype=np.float64
    )
    ladder_max_orders_array = np.asarray(ladder_max_orders, dtype=np.int32)
    (
        valid,
        ladder_indices,
        initial_out,
        gamma_out,
        idle_out,
        shares_out,
        initial_pct_out,
        last_pct_out,
        stress_loss_out,
        expected_util_out,
    ) = _expand_template_allocations(
        ladder_level_matrix,
        ladder_probability_matrix,
        ladder_max_orders_array,
        initial_values,
        gamma_array,
        idle_values,
        stress_deviation,
        int(total_shares),
        float(fee_rate),
        max_last_order_pct is not None,
        float(max_last_order_pct or 0.0),
        max_stress_loss_pct is not None,
        float(max_stress_loss_pct or 0.0),
    )

    rows = []
    level_rows = []
    share_rows = []
    seen_executable_templates = set()
    for candidate_index in np.flatnonzero(valid):
        ladder_index = int(ladder_indices[candidate_index])
        max_orders = int(ladder_max_orders_array[ladder_index])
        levels = ladder_level_matrix[ladder_index, :max_orders]
        quantiles = ladder_quantile_matrix[ladder_index, :max_orders]
        fill_probabilities = ladder_probability_matrix[
            ladder_index, :max_orders
        ]
        shares = shares_out[candidate_index, :max_orders]
        executable_key = (
            max_orders,
            tuple(float(value) for value in levels),
            tuple(int(value) for value in shares),
        )
        if executable_key in seen_executable_templates:
            continue
        seen_executable_templates.add(executable_key)
        level_row = ladder_level_matrix[ladder_index].copy()
        share_row = np.zeros(max_width, dtype=np.float64)
        share_row[:max_orders] = shares
        template_index = len(rows)
        level_rows.append(level_row)
        share_rows.append(share_row)
        expected_utilization = float(expected_util_out[candidate_index])
        rows.append(
            {
                "template_index": template_index,
                "strategy_mode": "diy",
                "mae_q_start": float(q_start),
                "mae_q_end": float(ladder_q_end[ladder_index]),
                "initial_capital_pct": float(
                    initial_pct_out[candidate_index]
                ),
                "capital_gamma": float(gamma_out[candidate_index]),
                "idle_bias": float(idle_out[candidate_index]),
                "expected_capital_utilization": expected_utilization,
                "expected_idle_ratio": 1.0 - expected_utilization,
                "max_orders": max_orders,
                "min_buy_ratio": float(levels[-1]),
                "last_order_pct": float(last_pct_out[candidate_index]),
                "stress_mae": stress_deviation,
                "stress_loss_pct": float(stress_loss_out[candidate_index]),
                "level_ratios": tuple(float(v) for v in levels),
                "order_shares": tuple(int(v) for v in shares),
                "mae_quantiles": tuple(float(v) for v in quantiles),
                "fill_probabilities": tuple(
                    float(v) for v in fill_probabilities
                ),
            }
        )

    if not rows:
        return (
            pd.DataFrame(),
            np.empty((0, max_width), dtype=np.float64),
            np.empty((0, max_width), dtype=np.float64),
            stress_deviation,
        )
    return (
        pd.DataFrame(rows),
        np.ascontiguousarray(np.vstack(level_rows), dtype=np.float64),
        np.ascontiguousarray(np.vstack(share_rows), dtype=np.float64),
        stress_deviation,
    )
