"""Natural air movement assessment for thermal pleasure.

Two modes:
  1. Velocity data — uniform scalar or spatially resolved (n_points, n_hours) array.
  2. Geometry proxy — simplified NV potential from window geometry.

Evidence base (from "Dynamic Thermal Environments as Biophilic Design"):
  - Mean air velocity 0.2–0.8 m/s is associated with thermal pleasure
    at operative temperatures above ~24°C (Parkinson & de Dear 2017).
  - Turbulence intensity 30–60% is preferred over steady flow.
  - EnergyPlus / Honeybee standard simulations report a single zone-mean
    air velocity (typically 0.1 m/s) — not spatially resolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from capy.core.environment import ThermalEnvironment
from capy.core.schedule import OccupancySchedule

# Evidence-based defaults (configurable in all public functions)
T_PLEASURE_THRESHOLD = 24.0   # °C — above this, air movement becomes pleasant
V_PLEASANT_MIN = 0.2          # m/s
V_PLEASANT_MAX = 0.8          # m/s
WFR_TARGET = 0.04             # window-to-floor ratio target (ASHRAE 62.1: 4%)


@dataclass
class Window:
    """Simplified window geometry for NV potential calculation.

    Only openable area and orientation are required. Heights are optional
    refinements for future stack-effect calculations.
    """

    area: float               # m² of openable window area
    orientation: float        # degrees clockwise from North (0=N, 90=E, 180=S, 270=W)
    sill_height: float = 0.9  # m above floor
    head_height: float = 2.1  # m above floor
    zone_id: Union[str, int] = 0


@dataclass
class AirMovementResult:
    """Output of assess_air_movement()."""

    mean_velocity: Optional[np.ndarray]  # (n_points, n_hours) m/s, or None
    pleasure_zone_fraction: float        # fraction of occupied hours in pleasure zone
    nv_potential: Optional[float]        # 0–1 from geometry proxy, or None
    data_mode: str                       # 'uniform' | 'point_data' | 'geometry_proxy' | 'unavailable'
    T_threshold: float
    v_target_range: tuple

    def __repr__(self) -> str:
        pzf = f"{self.pleasure_zone_fraction * 100:.1f}%" if not np.isnan(
            self.pleasure_zone_fraction
        ) else "N/A"
        return (
            f"AirMovementResult(mode='{self.data_mode}', "
            f"pleasure_zone={pzf}, "
            f"nv_potential={self.nv_potential})"
        )


def assess_air_movement(
    env: ThermalEnvironment,
    velocity: Union[float, np.ndarray, None] = None,
    windows: Optional[List[Window]] = None,
    T_threshold: float = T_PLEASURE_THRESHOLD,
    v_min: float = V_PLEASANT_MIN,
    v_max: float = V_PLEASANT_MAX,
    schedule: Optional[OccupancySchedule] = None,
) -> AirMovementResult:
    """Assess natural air movement potential for thermal pleasure.

    Three operating modes depending on available data:

    **Mode 1 — uniform velocity** (``velocity=float``):
        Apply a single air speed to the whole grid. Typical for EnergyPlus
        outputs where a zone-mean value (e.g. 0.1 m/s) is available.

    **Mode 2 — point-based velocity** (``velocity=np.ndarray``):
        Spatially resolved (n_points, n_hours) array from CFD or an
        EnergyPlus airflow network simulation.

    **Mode 3 — geometry proxy** (``velocity=None``, ``windows=[...]``):
        Estimate NV potential from window geometry (area, orientation).
        Returns a 0–1 score and an estimated pleasure-zone fraction.
        Use when no velocity data is available from the simulation.

    **Mode 4 — unavailable** (``velocity=None``, ``windows=None``):
        Returns NaN with a warning flag.

    Args:
        env: ThermalEnvironment with operative_temperature.
        velocity: Uniform float (m/s), (n_points, n_hours) array, or None.
        windows: List of Window objects for geometry-proxy mode.
        T_threshold: Operative temperature above which air movement is pleasant.
        v_min: Lower bound of pleasant velocity range (m/s).
        v_max: Upper bound of pleasant velocity range (m/s).
        schedule: Occupancy schedule for filtering (default: all hours).

    Returns:
        AirMovementResult.
    """
    n_hours = len(env.time_index)
    occ_mask = schedule.as_boolean() if schedule is not None else np.ones(n_hours, dtype=bool)
    T_op = env.operative_temperature()        # (n_points, n_hours)
    above_threshold = T_op > T_threshold      # (n_points, n_hours)

    if velocity is not None:
        if np.isscalar(velocity):
            v_arr = np.full_like(T_op, float(velocity))
            mode = "uniform"
        else:
            v_arr = np.asarray(velocity, dtype=float)
            if v_arr.shape != T_op.shape:
                raise ValueError(
                    f"velocity array shape {v_arr.shape} must match "
                    f"(n_points, n_hours) = {T_op.shape}"
                )
            mode = "point_data"

        in_range = (v_arr >= v_min) & (v_arr <= v_max)
        pleasure = above_threshold & in_range        # (n_points, n_hours)
        pzf = float(pleasure[:, occ_mask].mean())

        return AirMovementResult(
            mean_velocity=v_arr,
            pleasure_zone_fraction=pzf,
            nv_potential=None,
            data_mode=mode,
            T_threshold=T_threshold,
            v_target_range=(v_min, v_max),
        )

    if windows is not None:
        nv_score = _nv_potential_from_geometry(windows, env.grid.total_area)
        above_frac = float(above_threshold[:, occ_mask].mean())
        pzf = above_frac * nv_score

        return AirMovementResult(
            mean_velocity=None,
            pleasure_zone_fraction=pzf,
            nv_potential=nv_score,
            data_mode="geometry_proxy",
            T_threshold=T_threshold,
            v_target_range=(v_min, v_max),
        )

    # Mode 4: no data
    return AirMovementResult(
        mean_velocity=None,
        pleasure_zone_fraction=float("nan"),
        nv_potential=None,
        data_mode="unavailable",
        T_threshold=T_threshold,
        v_target_range=(v_min, v_max),
    )


def _nv_potential_from_geometry(windows: List[Window], floor_area: float) -> float:
    """Simplified NV potential score (0–1) from window geometry.

    Combines two checks:
    1. Window-to-floor ratio (WFR): target > 4% (ASHRAE 62.1 guideline).
       WFR = total openable window area / floor area.
       Score: linear from 0 at WFR=0 to 1.0 at WFR=8% (2× target), capped.

    2. Cross-ventilation: are there openings on substantially different
       orientations (> 90° apart)? Cross-vent doubles the effective driving
       pressure vs. single-sided ventilation.
       Multiplier: 1.0 if cross-vent present, 0.5 if single-sided only.

    Returns:
        float in [0, 1].
    """
    if not windows or floor_area <= 0:
        return 0.0

    total_area = sum(w.area for w in windows)
    wfr = total_area / floor_area
    wfr_score = min(wfr / (2 * WFR_TARGET), 1.0)

    # Cross-ventilation check: any pair of windows more than 90° apart
    orientations = [w.orientation % 360 for w in windows]
    cross_vent = any(
        min(abs(o1 - o2), 360 - abs(o1 - o2)) > 90
        for i, o1 in enumerate(orientations)
        for o2 in orientations[i + 1:]
    )
    cross_vent_multiplier = 1.0 if cross_vent else 0.5

    return float(wfr_score * cross_vent_multiplier)
