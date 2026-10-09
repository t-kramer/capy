"""Shared EPW file utilities.

All modules that read outdoor weather data should import from here
rather than defining their own parsers.
"""
from __future__ import annotations

import numpy as np


def load_dry_bulb(epw_path: str) -> np.ndarray:
    """Load 8760 hourly dry-bulb temperatures from an EPW file.

    Uses ladybug-core if installed; falls back to a minimal CSV parser
    so the function works without the full LBT stack.

    Parameters
    ----------
    epw_path:
        Path to an EnergyPlus Weather (.epw) file.

    Returns
    -------
    np.ndarray of shape ``(8760,)`` in °C.

    Raises
    ------
    ValueError
        If fewer than 8760 data rows are found.
    """
    try:
        from ladybug.epw import EPW
        return np.array(list(EPW(epw_path).dry_bulb_temperature.values), dtype=float)
    except ImportError:
        pass

    values: list[float] = []
    with open(epw_path) as f:
        for i, line in enumerate(f):
            if i < 8:
                continue
            parts = line.split(",")
            if len(parts) >= 7:
                try:
                    values.append(float(parts[6]))
                except ValueError:
                    pass

    arr = np.array(values, dtype=float)
    if arr.shape[0] < 8760:
        raise ValueError(
            f"EPW file yielded {arr.shape[0]} dry-bulb values (expected 8760): {epw_path}"
        )
    return arr[:8760]


def running_mean(t_outdoor: np.ndarray, alpha: float = 0.8, n_days: int = 7) -> np.ndarray:
    """Exponentially weighted running mean of daily outdoor temperature.

    Returns an (8760,) hourly array where each hour has the running mean
    of the preceding *n_days* days, weighted by ``alpha``.

    This is the prevailing mean outdoor temperature used by ASHRAE 55 and
    EN 16798-1 adaptive comfort models.

    Parameters
    ----------
    t_outdoor:
        (8760,) hourly dry-bulb temperatures [°C].
    alpha:
        Exponential weighting factor (ASHRAE 55 default: 0.8).
    n_days:
        Number of preceding days to include (default: 7).

    Returns
    -------
    np.ndarray of shape ``(8760,)``.
    """
    daily_mean = t_outdoor.reshape(365, 24).mean(axis=1)  # (365,)
    rm_daily = np.empty(365)
    rm_daily[0] = daily_mean[0]
    for d in range(1, 365):
        start = max(0, d - n_days)
        raw_w = np.array([(1 - alpha) * alpha ** (d - i - 1) for i in range(start, d)])
        w = raw_w / raw_w.sum()
        rm_daily[d] = float(np.dot(daily_mean[start:d], w))
    return np.repeat(rm_daily, 24)[:8760]
