"""Spatial Thermal Autonomy (sTA) metric.

Reference: Kramer, T., Schiavon, S., & Garcia-Hansen, V. (2024).
Spatial Thermal Autonomy (sTA): A New Metric for Enhancing Building Design
Towards Comfort, Heat Resilience and Energy Autonomy. CATE 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from capy.comfort.base import ComfortResult
from capy.core.schedule import OccupancySchedule


@dataclass
class sTAResult:
    """Result of spatial_thermal_autonomy()."""

    annual_pct: float               # annual sTA in % (0–100)
    hourly_array: np.ndarray        # (n_hours,) fraction of area comfortable each hour
    comfort_hours_matrix: np.ndarray  # (n_points, n_hours) bool — comfortable?
    tau: float                      # area threshold used
    schedule_applied: bool

    def __repr__(self) -> str:
        return (
            f"sTAResult(annual_pct={self.annual_pct:.1f}%, tau={self.tau:.0%}, "
            f"schedule_applied={self.schedule_applied})"
        )


def spatial_thermal_autonomy(
    comfort_result: ComfortResult,
    tau: float = 0.8,
    schedule: Optional[OccupancySchedule] = None,
    area_weighted: bool = True,
) -> sTAResult:
    """Calculate spatial Thermal Autonomy (sTA).

    sTA_annual is the fraction of occupied hours in which at least tau (default
    80%) of the floor area is thermally comfortable.

    Args:
        comfort_result: Output from any comfort model in capy.comfort.
        tau: Spatial comfort threshold — minimum fraction of floor area that
             must be comfortable for the hour to count (default 0.8 = 80%).
        schedule: Occupancy schedule for filtering (default: all hours).
        area_weighted: If True, use SpatialGrid.areas to weight each point's
                       contribution. If False, each point counts equally.

    Returns:
        sTAResult with annual_pct, hourly_array, comfort_hours_matrix.
    """
    comfort_mask = comfort_result.is_comfortable()  # (n_points, n_hours) bool

    if area_weighted:
        weights = comfort_result.env.grid.area_weights  # (n_points,) sum=1
        hourly_sta = (comfort_mask * weights[:, None]).sum(axis=0)  # (n_hours,)
    else:
        hourly_sta = comfort_mask.mean(axis=0)  # (n_hours,)

    occ_mask = (
        schedule.as_boolean()
        if schedule is not None
        else np.ones(len(comfort_result.env.time_index), dtype=bool)
    )

    occupied_sta = hourly_sta[occ_mask]
    annual_pct = float((occupied_sta >= tau).mean() * 100.0)

    return sTAResult(
        annual_pct=annual_pct,
        hourly_array=hourly_sta,
        comfort_hours_matrix=comfort_mask,
        tau=tau,
        schedule_applied=schedule is not None,
    )
