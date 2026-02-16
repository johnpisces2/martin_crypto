# -*- coding: utf-8 -*-
"""Sampling utilities for MC parameter candidate generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _from_flat_idx(flat_idx: np.ndarray, add_drop_arr, mul_arr, mo_arr, tp_arr):
    na, nm, no, nt = len(add_drop_arr), len(mul_arr), len(mo_arr), len(tp_arr)
    i_ad, i_mul, i_mo, i_tp = np.unravel_index(flat_idx, (na, nm, no, nt))
    ad = add_drop_arr[i_ad].astype(np.float64)
    mul = mul_arr[i_mul].astype(np.float64)
    mo = mo_arr[i_mo].astype(np.int32)
    tp = tp_arr[i_tp].astype(np.float64)
    return ad, mul, mo, tp


def sample_parameter_grid(
    *,
    add_drop_arr,
    tp_arr,
    mul_arr,
    mo_arr,
    mode: str,
    sample_size: int,
    max_combos: int,
    seed: int,
):
    """Return sampled parameter dataframe.

    mode: 'lhs' | 'random' | 'full grid'
    """
    rng = np.random.default_rng(int(seed))
    na, nm, no, nt = len(add_drop_arr), len(mul_arr), len(mo_arr), len(tp_arr)
    total_space = int(na * nm * no * nt)

    mode = (mode or "lhs").strip().lower()

    if mode == "full grid":
        if max_combos and max_combos > 0 and total_space > max_combos:
            pick = rng.choice(total_space, size=max_combos, replace=False)
            ad, mul, mo, tp = _from_flat_idx(pick, add_drop_arr, mul_arr, mo_arr, tp_arr)
            return pd.DataFrame(
                {
                    "add_drop": ad,
                    "multiplier": mul,
                    "max_orders": mo.astype(int),
                    "tp": tp,
                }
            ).drop_duplicates(ignore_index=True)

        # Stream full grid to avoid allocating a giant index vector.
        rows = []
        for i_ad in range(na):
            ad = float(add_drop_arr[i_ad])
            for i_mul in range(nm):
                mul = float(mul_arr[i_mul])
                for i_mo in range(no):
                    mo = int(mo_arr[i_mo])
                    for i_tp in range(nt):
                        rows.append((ad, mul, mo, float(tp_arr[i_tp])))
        return pd.DataFrame(rows, columns=["add_drop", "multiplier", "max_orders", "tp"])

    if mode == "random":
        n = min(max(1, int(sample_size)), total_space)
        pick = rng.choice(total_space, size=n, replace=False)
        ad, mul, mo, tp = _from_flat_idx(pick, add_drop_arr, mul_arr, mo_arr, tp_arr)
        return pd.DataFrame(
            {
                "add_drop": ad,
                "multiplier": mul,
                "max_orders": mo.astype(int),
                "tp": tp,
            }
        ).drop_duplicates(ignore_index=True)

    # LHS
    n = min(max(1, int(sample_size)), total_space)
    ad_idx = np.floor(((rng.permutation(n) + rng.random(n)) / n) * na).astype(np.int32)
    mul_idx = np.floor(((rng.permutation(n) + rng.random(n)) / n) * nm).astype(np.int32)
    mo_idx = np.floor(((rng.permutation(n) + rng.random(n)) / n) * no).astype(np.int32)
    tp_idx = np.floor(((rng.permutation(n) + rng.random(n)) / n) * nt).astype(np.int32)

    ad_idx = np.clip(ad_idx, 0, na - 1)
    mul_idx = np.clip(mul_idx, 0, nm - 1)
    mo_idx = np.clip(mo_idx, 0, no - 1)
    tp_idx = np.clip(tp_idx, 0, nt - 1)

    df = pd.DataFrame(
        {
            "add_drop": add_drop_arr[ad_idx].astype(np.float64),
            "multiplier": mul_arr[mul_idx].astype(np.float64),
            "max_orders": mo_arr[mo_idx].astype(np.int32).astype(int),
            "tp": tp_arr[tp_idx].astype(np.float64),
        }
    ).drop_duplicates(ignore_index=True)

    while len(df) < n:
        fill_n = min(n - len(df), total_space)
        fill_pick = rng.choice(total_space, size=fill_n, replace=False)
        ad, mul, mo, tp = _from_flat_idx(fill_pick, add_drop_arr, mul_arr, mo_arr, tp_arr)
        extra = pd.DataFrame(
            {
                "add_drop": ad,
                "multiplier": mul,
                "max_orders": mo.astype(int),
                "tp": tp,
            }
        )
        df = pd.concat([df, extra], ignore_index=True).drop_duplicates(ignore_index=True)
        if len(df) >= n:
            break

    return df.head(n).copy()


def refine_neighbors(
    *,
    base_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    add_drop_arr,
    mul_arr,
    mo_arr,
    tp_arr,
    radius: int = 1,
    max_add: int = 3000,
):
    """Generate neighborhood combinations around seed parameter rows."""
    if radius <= 0 or max_add <= 0 or seeds_df.empty:
        return pd.DataFrame(columns=["add_drop", "multiplier", "max_orders", "tp"])

    na, nm, no, nt = len(add_drop_arr), len(mul_arr), len(mo_arr), len(tp_arr)
    ad_to_i = {float(v): i for i, v in enumerate(add_drop_arr)}
    mul_to_i = {float(v): i for i, v in enumerate(mul_arr)}
    mo_to_i = {int(v): i for i, v in enumerate(mo_arr)}
    tp_to_i = {float(v): i for i, v in enumerate(tp_arr)}

    existing = {
        (float(r.add_drop), float(r.multiplier), int(r.max_orders), float(r.tp))
        for r in base_df.itertuples()
    }

    out = []
    offsets = [d for d in range(-radius, radius + 1) if d != 0]
    for r in seeds_df.itertuples():
        i_ad = ad_to_i.get(float(r.add_drop), None)
        i_mul = mul_to_i.get(float(r.multiplier), None)
        i_mo = mo_to_i.get(int(r.max_orders), None)
        i_tp = tp_to_i.get(float(r.tp), None)
        if None in (i_ad, i_mul, i_mo, i_tp):
            continue
        for d in offsets:
            cand = [
                (i_ad + d, i_mul, i_mo, i_tp),
                (i_ad, i_mul + d, i_mo, i_tp),
                (i_ad, i_mul, i_mo + d, i_tp),
                (i_ad, i_mul, i_mo, i_tp + d),
            ]
            for ia, im, io, it in cand:
                if ia < 0 or ia >= na or im < 0 or im >= nm or io < 0 or io >= no or it < 0 or it >= nt:
                    continue
                key = (float(add_drop_arr[ia]), float(mul_arr[im]), int(mo_arr[io]), float(tp_arr[it]))
                if key in existing:
                    continue
                existing.add(key)
                out.append(key)
                if len(out) >= max_add:
                    break
            if len(out) >= max_add:
                break
        if len(out) >= max_add:
            break

    if not out:
        return pd.DataFrame(columns=["add_drop", "multiplier", "max_orders", "tp"])
    return pd.DataFrame(out, columns=["add_drop", "multiplier", "max_orders", "tp"])
