from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from capy.core.grid import SpatialGrid
from capy.core.schedule import OccupancySchedule

HOURS_PER_YEAR = 8760

# Recognised aliases for standard variable names
_ALIASES: Dict[str, List[str]] = {
    "air_temperature": ["air_temperature", "ta", "tdb", "t_air"],
    "mean_radiant_temperature": ["mean_radiant_temperature", "mrt", "tr", "t_mrt"],
    "relative_humidity": ["relative_humidity", "rh", "hr"],
    "air_velocity": ["air_velocity", "v", "vr", "vel"],
    "operative_temperature": ["operative_temperature", "t_op", "top", "to"],
}

# Reverse map: alias → canonical name
_ALIAS_MAP: Dict[str, str] = {
    alias: canonical
    for canonical, aliases in _ALIASES.items()
    for alias in aliases
}


@dataclass
class ThermalEnvironment:
    """Central data type for capy.

    Stores spatially and temporally resolved environmental variables on a
    SpatialGrid. Each variable is a numpy array of shape (n_points, n_hours).

    The standard case is annual hourly data: n_hours = 8760. The class does
    not enforce this but metric functions expect it.
    """

    grid: SpatialGrid
    time_index: pd.DatetimeIndex
    variables: Dict[str, np.ndarray] = field(default_factory=dict)
    epw_path: Optional[str] = None
    info: dict = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        grid: SpatialGrid,
        variables: Dict[str, np.ndarray],
        n_hours: int = HOURS_PER_YEAR,
        year: int = 2001,
        epw_path: Optional[str] = None,
        info: Optional[dict] = None,
    ) -> ThermalEnvironment:
        """Primary entry point for pure-Python usage.

        Constructs a ThermalEnvironment from raw numpy arrays without
        requiring a DatetimeIndex. A standard annual hourly index is
        generated automatically.

        Parameters
        ----------
        grid:
            Sensor point geometry and area weights.
        variables:
            Dict of variable name → (n_points, n_hours) array.
            Accepts canonical names and aliases (e.g. ``"ta"``).
        n_hours:
            Number of time steps. Defaults to 8760 (annual hourly).
        year:
            Calendar year used to generate the time index. Affects
            weekday-aware operations (e.g. OccupancySchedule.work_hours).
        epw_path:
            Optional path to an EPW file for climate-dependent metrics.
        info:
            Optional metadata dict.
        """
        time_index = pd.date_range(f"{year}-01-01", periods=n_hours, freq="h")
        return cls(
            grid=grid,
            time_index=time_index,
            variables=variables,
            epw_path=epw_path,
            info=info or {},
        )

    def __post_init__(self) -> None:
        for name, arr in list(self.variables.items()):
            canonical = _ALIAS_MAP.get(name, name)
            if canonical != name:
                self.variables[canonical] = self.variables.pop(name)
            self._validate(canonical, self.variables[canonical])

    # ------------------------------------------------------------------ #
    # Variable management                                                  #
    # ------------------------------------------------------------------ #

    def _validate(self, name: str, arr: np.ndarray) -> None:
        expected_shape = (self.grid.n_points, len(self.time_index))
        if arr.shape != expected_shape:
            raise ValueError(
                f"Variable '{name}' has shape {arr.shape}, "
                f"expected {expected_shape} (n_points, n_hours)"
            )

    def add(self, name: str, data: np.ndarray) -> None:
        """Add or overwrite a variable. Accepts canonical names and aliases."""
        canonical = _ALIAS_MAP.get(name, name)
        data = np.asarray(data, dtype=float)
        self._validate(canonical, data)
        self.variables[canonical] = data

    def get(self, name: str) -> np.ndarray:
        """Retrieve a variable by canonical name or alias.

        ``operative_temperature`` is returned as a computed fallback
        (``(ta + mrt) / 2``) when not stored as an explicit variable.
        """
        canonical = _ALIAS_MAP.get(name, name)
        if canonical in self.variables:
            return self.variables[canonical]
        if canonical == "operative_temperature":
            return self.operative_temperature()
        raise KeyError(
            f"Variable '{name}' not found. "
            f"Available: {list(self.variables)}"
        )

    def has(self, name: str) -> bool:
        canonical = _ALIAS_MAP.get(name, name)
        return canonical in self.variables

    # ------------------------------------------------------------------ #
    # Named accessors                                                      #
    # ------------------------------------------------------------------ #

    def air_temperature(self) -> np.ndarray:
        return self.get("air_temperature")

    def mrt(self) -> np.ndarray:
        return self.get("mean_radiant_temperature")

    def relative_humidity(self) -> np.ndarray:
        return self.get("relative_humidity")

    def air_velocity(self) -> np.ndarray:
        """Air velocity in m/s. Raises KeyError if not present.

        Use capy.metrics.air_movement.assess_air_movement() to handle the
        common case where velocity is not available from the simulation.
        """
        return self.get("air_velocity")

    def operative_temperature(self) -> np.ndarray:
        """Operative temperature in °C.

        Returns stored values if available, otherwise computes as
        (air_temperature + mrt) / 2 — a reasonable approximation for
        typical indoor environments with low air speed.
        """
        if self.has("operative_temperature"):
            return self.get("operative_temperature")
        return (self.air_temperature() + self.mrt()) / 2.0

    # ------------------------------------------------------------------ #
    # Filtering                                                            #
    # ------------------------------------------------------------------ #

    def with_variables(self, **updates: np.ndarray) -> ThermalEnvironment:
        """Return a copy with one or more variable arrays replaced or added.

        Accepts canonical names and aliases. The original environment is
        never modified. Intended as the primary building block for
        post-processing steps such as PCS device application.

        Parameters
        ----------
        **updates:
            Keyword arguments mapping variable name → ``(n_points, n_hours)``
            array. Any recognised alias is resolved to its canonical form.

        Returns
        -------
        ThermalEnvironment
            A new instance sharing the same grid, time index, and all
            unmodified variables, with ``info["pcs_applied"] = True``.
        """
        new_vars = {**self.variables}
        for name, arr in updates.items():
            canonical = _ALIAS_MAP.get(name, name)
            arr = np.asarray(arr, dtype=float)
            self._validate(canonical, arr)
            new_vars[canonical] = arr
        return ThermalEnvironment(
            grid=self.grid,
            time_index=self.time_index,
            variables=new_vars,
            epw_path=self.epw_path,
            info={**self.info, "pcs_applied": True},
        )

    def with_schedule(self, schedule: OccupancySchedule) -> ThermalEnvironment:
        """Return a copy where unoccupied hours are set to NaN.

        Metric functions that accept a schedule parameter are preferred
        over this method, but this is useful for exploratory analysis.
        """
        if len(schedule.mask) != len(self.time_index):
            raise ValueError(
                f"Schedule length {len(schedule.mask)} does not match "
                f"time_index length {len(self.time_index)}"
            )
        mask = schedule.as_boolean()
        new_vars: Dict[str, np.ndarray] = {}
        for k, v in self.variables.items():
            arr = v.astype(float).copy()
            arr[:, ~mask] = np.nan
            new_vars[k] = arr
        return ThermalEnvironment(
            grid=self.grid,
            time_index=self.time_index,
            variables=new_vars,
            epw_path=self.epw_path,
            info={**self.info, "schedule_applied": True},
        )

    # ------------------------------------------------------------------ #
    # Dunder                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n_hours = len(self.time_index)
        vars_str = ", ".join(self.variables)
        return (
            f"ThermalEnvironment("
            f"n_points={self.grid.n_points}, "
            f"n_hours={n_hours}, "
            f"variables=[{vars_str}])"
        )
