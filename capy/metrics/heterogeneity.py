"""Spatial temperature variability metrics.

Measures how much the temperature field varies *across sensor points* at any
given hour, and how often that variability falls in the evidence-based
beneficial range (2–4°C spatial spread).

Results are only meaningful when sensor points cover the accessible occupied
area — not building envelope, remote corners, or structural elements.

Reference: Kramer, T., Schiavon, S., & Garcia-Hansen, V. (2024). CATE 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from capy.core.environment import ThermalEnvironment
from capy.core.schedule import OccupancySchedule


@dataclass
class SpatialVariabilityResult:
    """Result of spatial_variability().

    Attributes
    ----------
    hourly_range:
        (n_occ_hours,) spatial temperature range (max − min across points)
        for each occupied hour. Exposed for downstream use (e.g. alliesthesia).
    median_spatial_range:
        Median of hourly_range [°C]. Summary scalar.
    fraction_variable:
        % of occupied hours where spatial range ∈ [delta_t_min, delta_t_max].
        The evidence-based 'beneficial variability' band.
    fraction_uniform:
        % of occupied hours where spatial range < delta_t_min (too uniform).
    fraction_excessive:
        % of occupied hours where spatial range > delta_t_max (excessive spread).
    delta_t_min:
        Lower bound of the beneficial band [°C] (default 2.0).
    delta_t_max:
        Upper bound of the beneficial band [°C] (default 4.0).
    """

    hourly_range: np.ndarray
    median_spatial_range: float
    fraction_variable: float
    fraction_uniform: float
    fraction_excessive: float
    delta_t_min: float
    delta_t_max: float

    def __repr__(self) -> str:
        return (
            f"SpatialVariabilityResult("
            f"variable={self.fraction_variable:.1f}%, "
            f"uniform={self.fraction_uniform:.1f}%, "
            f"excessive={self.fraction_excessive:.1f}%, "
            f"median_range={self.median_spatial_range:.2f}°C)"
        )


def spatial_variability(
    env: ThermalEnvironment,
    variable: str = "operative_temperature",
    schedule: Optional[OccupancySchedule] = None,
    delta_t_min: float = 2.0,
    delta_t_max: float = 4.0,
) -> SpatialVariabilityResult:
    """Spatial temperature variability across sensor points.

    Per occupied hour, computes the spatial temperature range (max − min across
    all sensor points), then classifies how often that range falls in the
    evidence-based beneficial band [delta_t_min, delta_t_max].

    A spatial spread of 2–4°C is associated with occupant agency, thermal
    variety, and positive alliesthesia (Kramer et al., 2024).

    Parameters
    ----------
    env:
        ThermalEnvironment to analyse.
    variable:
        Variable to use (default: ``'operative_temperature'``).
    schedule:
        Occupancy schedule; all hours used if ``None``.
    delta_t_min:
        Lower bound of the beneficial spatial range in °C (default 2.0).
    delta_t_max:
        Upper bound of the beneficial spatial range in °C (default 4.0).

    Returns
    -------
    SpatialVariabilityResult
    """
    T = _get_variable(env, variable)
    occ_mask = _occ_mask(schedule, T.shape[1])
    hourly_range = _hourly_spatial_range(T, occ_mask)

    fraction_variable = float(
        ((hourly_range >= delta_t_min) & (hourly_range <= delta_t_max)).mean() * 100
    )
    fraction_uniform = float((hourly_range < delta_t_min).mean() * 100)
    fraction_excessive = float((hourly_range > delta_t_max).mean() * 100)

    return SpatialVariabilityResult(
        hourly_range=hourly_range,
        median_spatial_range=float(np.median(hourly_range)),
        fraction_variable=fraction_variable,
        fraction_uniform=fraction_uniform,
        fraction_excessive=fraction_excessive,
        delta_t_min=delta_t_min,
        delta_t_max=delta_t_max,
    )


# ------------------------------------------------------------------ #
# Private helpers (also used by alliesthesia.thermal_niches)          #
# ------------------------------------------------------------------ #

def _hourly_spatial_range(T: np.ndarray, occ_mask: np.ndarray) -> np.ndarray:
    """Return (n_occ_hours,) spatial range from a (n_points, n_hours) array."""
    return T[:, occ_mask].max(axis=0) - T[:, occ_mask].min(axis=0)


def _get_variable(env: ThermalEnvironment, variable: str) -> np.ndarray:
    if variable == "operative_temperature":
        return env.operative_temperature()
    return env.get(variable)


def _occ_mask(schedule: Optional[OccupancySchedule], n_hours: int) -> np.ndarray:
    if schedule is not None:
        return schedule.as_boolean()
    return np.ones(n_hours, dtype=bool)
