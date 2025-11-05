"""Analytical helper functions for SMF simulations."""

from __future__ import annotations

import math
from typing import Union

import numpy as np

Number = Union[float, int, np.ndarray]

# Small constant to avoid division by zero with vanishing seed amplitudes.
_MIN_SEED_MAGNITUDE = 1e-30


def pump_threshold(gamma_bar: float, b0: float, reflectivity: float) -> float:
    """Return the pump strength threshold p_th."""
    if b0 == 0 or reflectivity == 0:
        return math.inf
    return (2.0 * gamma_bar) / (b0 * reflectivity)


def analytic_delay_time(
    p0: Number, p_th: float, gamma_bar: float, seed: float
) -> Number:
    """
    Compute the analytic modulation delay time t0.

    Parameters
    ----------
    p0:
        Pump strength(s). Can be a scalar or numpy array.
    p_th:
        Threshold pump strength.
    gamma_bar:
        Effective optical growth-rate parameter (commonly written as Gamma-bar).
    seed:
        Initial seed amplitude (cosine coefficient).

    Returns
    -------
    Number
        Analytic delay time(s). Returns ``math.inf`` when no real-valued
        solution exists (e.g., p0 <= p_th, vanishing seed, or ``gamma_bar == 0``).
    """
    p0_arr = np.asarray(p0, dtype=float)
    results = np.full_like(p0_arr, math.inf, dtype=float)

    if not np.isfinite(p_th) or gamma_bar == 0 or abs(seed) < _MIN_SEED_MAGNITUDE:
        return results if results.ndim else float(results)

    mask = np.isfinite(p0_arr) & (p0_arr > p_th)
    if not np.any(mask):
        return results if results.ndim else float(results)

    ratio = p0_arr[mask] / p_th
    excess = ratio - 1.0
    if np.any(excess <= 0):
        excess = np.where(excess > 0, excess, 0.0)

    numerator = (
        np.sqrt(2.0) * (p_th / p0_arr[mask]) * np.sqrt(excess) / abs(seed)
    )

    valid = numerator > 1.0
    if np.any(valid):
        acos_args = numerator[valid]
        denom = np.sqrt(excess[valid]) * abs(gamma_bar)
        results_subset = np.arccosh(acos_args) / denom
        results_subset = np.where(np.isfinite(results_subset), results_subset, math.inf)
        sub_results = np.full_like(numerator, math.inf, dtype=float)
        sub_results[valid] = results_subset
        results[mask] = sub_results
    else:
        results[mask] = math.inf

    return results if results.ndim else float(results)
