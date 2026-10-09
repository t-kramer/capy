"""Thermal journey mapping along circulation paths.

A 'thermal journey' is the sequence of thermal conditions experienced by an
occupant walking along a defined path through a space. It captures:
  - The temperature profile at each step along the path
  - The gradient (ΔT per step) — direction and magnitude of thermal transitions
  - The dominant transition type per step (warming, cooling, neutral)

This metric operationalises the concept of spatial thermal variety as a
sequential experience rather than just a spatial snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from capy.core.environment import ThermalEnvironment
from capy.core.schedule import OccupancySchedule

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JourneyResult:
    """Output of thermal_journey().

    Attributes
    ----------
    profile:
        (n_steps, n_hours) — variable value at each path point over time.
    gradient:
        (n_steps-1, n_hours) — signed ΔT between consecutive path steps
        (positive = warming as you move along the path).
    transitions:
        (n_steps-1,) int8 — annual median transition per step:
        ``+1`` warming, ``0`` neutral (<= neutral_threshold), ``-1`` cooling.
    journey_comfort_score:
        Mean comfort fraction along the path during occupied hours, taken from
        ``comfort_result`` if provided (else ``nan``).
    path_points:
        (n_steps, 3) path waypoints as supplied (model coordinates).
    sensor_indices:
        (n_steps,) index of the nearest / IDW sensor for each path point.
    variable:
        Name of the mapped variable.
    """

    profile: np.ndarray
    gradient: np.ndarray
    transitions: np.ndarray
    journey_comfort_score: float
    path_points: np.ndarray
    sensor_indices: np.ndarray
    variable: str
    neutral_threshold: float

    def __repr__(self) -> str:
        n_steps = self.profile.shape[0]
        return (
            f"JourneyResult(n_steps={n_steps}, variable='{self.variable}', "
            f"comfort_score={self.journey_comfort_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def thermal_journey(
    env: ThermalEnvironment,
    path_points: np.ndarray,
    variable: str = "operative_temperature",
    method: Literal["nearest", "idw"] = "nearest",
    schedule: Optional[OccupancySchedule] = None,
    neutral_threshold: float = 0.5,
    comfort_result=None,
    idw_k: int = 4,
    idw_power: float = 2.0,
) -> JourneyResult:
    """Map thermal conditions along a circulation path.

    Interpolates *variable* at each path waypoint using the sensor grid,
    building a ``(n_steps, n_hours)`` profile of the thermal experience along
    the journey.

    Parameters
    ----------
    env:
        ThermalEnvironment with sensor grid coordinates in ``env.grid.points``.
    path_points:
        ``(m, 3)`` ordered array of path waypoints in model coordinates (same
        units as ``env.grid.points``). In Grasshopper, convert via
        ``capy.grasshopper.convert.points_from_rhino()``.
    variable:
        Environmental variable to map (default: ``'operative_temperature'``).
    method:
        Interpolation method:

        * ``'nearest'`` — assign each path point the value of its nearest
          sensor. Fast; works well for dense grids.
        * ``'idw'`` — inverse distance weighting over the *idw_k* nearest
          sensors. Smoother profiles; better for sparse grids.

    schedule:
        Occupancy schedule for the journey comfort score (all hours used if
        ``None``).
    neutral_threshold:
        ΔT in °C below which a step transition is classified as neutral
        (default 0.5°C).
    comfort_result:
        Optional ``ComfortResult``; if provided, the journey comfort score is
        the mean comfort fraction along path points during occupied hours.
    idw_k:
        Number of neighbours for IDW interpolation (default 4, ignored when
        *method* is ``'nearest'``).
    idw_power:
        Distance decay power for IDW (default 2.0).

    Returns
    -------
    JourneyResult

    Raises
    ------
    ValueError
        If ``path_points`` is not ``(m, 3)``, if the grid has fewer points
        than requested neighbours, or if the variable is not available.
    """
    from scipy.spatial import KDTree

    path_points = np.asarray(path_points, dtype=float)
    if path_points.ndim != 2 or path_points.shape[1] != 3:
        raise ValueError(
            f"path_points must be (m, 3), got {path_points.shape}"
        )

    n_steps = path_points.shape[0]
    grid_pts = env.grid.points                        # (n_grid, 3)
    n_grid = grid_pts.shape[0]
    data = env.get(variable)                          # (n_grid, n_hours)

    if n_steps == 0:
        raise ValueError("path_points must contain at least one point.")

    tree = KDTree(grid_pts)

    if method == "nearest":
        _, sensor_indices = tree.query(path_points)   # (n_steps,)
        profile = data[sensor_indices, :]             # (n_steps, n_hours)

    elif method == "idw":
        k = min(idw_k, n_grid)
        distances, indices = tree.query(path_points, k=k)  # (n_steps, k)

        # Handle exact matches (distance = 0) to avoid division by zero
        exact = distances[:, 0] == 0.0
        profile = np.empty((n_steps, data.shape[1]))
        sensor_indices = indices[:, 0].copy()         # nearest for index tracking

        for i in range(n_steps):
            if exact[i]:
                profile[i] = data[indices[i, 0], :]
            else:
                w = 1.0 / distances[i] ** idw_power  # (k,)
                w /= w.sum()
                profile[i] = (w[:, None] * data[indices[i], :]).sum(axis=0)

    else:
        raise ValueError(f"method must be 'nearest' or 'idw', got '{method}'")

    # Gradient between consecutive steps
    gradient = np.diff(profile, axis=0)              # (n_steps-1, n_hours)

    # Median annual transition per step
    median_grad = np.median(gradient, axis=1)        # (n_steps-1,)
    transitions = np.zeros(len(median_grad), dtype=np.int8)
    transitions[median_grad > neutral_threshold] = 1
    transitions[median_grad < -neutral_threshold] = -1

    # Journey comfort score
    comfort_score = float("nan")
    if comfort_result is not None:
        is_comfortable = comfort_result.is_comfortable()   # (n_grid, n_hours)
        path_comfort = is_comfortable[sensor_indices, :]   # (n_steps, n_hours)
        if schedule is not None:
            mask = schedule.mask
            comfort_score = float(path_comfort[:, mask].mean())
        else:
            comfort_score = float(path_comfort.mean())

    return JourneyResult(
        profile=profile,
        gradient=gradient,
        transitions=transitions,
        journey_comfort_score=comfort_score,
        path_points=path_points,
        sensor_indices=sensor_indices,
        variable=variable,
        neutral_threshold=neutral_threshold,
    )
