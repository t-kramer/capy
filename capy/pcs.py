"""Personal Comfort Systems (PCS) — post-processing device library.

Devices modify a subset of grid points in a ThermalEnvironment and return a
new environment that can be fed back through any comfort model or spatial
metric to quantify the corrective benefit.

Each device declares ``_variable_deltas``, a ``{variable_name: scalar_delta}``
dict.  The base-class ``apply()`` handles point selection, time-window masking,
and applying every delta, so a multi-parameter device simply lists more entries::

    @dataclass
    class CooledChairWithFan(PersonalComfortDevice):
        \"\"\"Chair with cooling surface and integrated fan.\"\"\"
        delta_mrt: float = -4.0
        delta_v: float = 0.3

        @property
        def _variable_deltas(self) -> dict[str, float]:
            return {
                "mean_radiant_temperature": self.delta_mrt,
                "air_velocity": self.delta_v,
            }

Typical workflow::

    from capy.pcs import Fan, FootWarmer, PointSelector

    result_base = calculate_pmv(env)

    desk = PointSelector.from_bbox(env.grid, x=(0, 6), y=(0, 4))
    fan  = Fan(delta_v=0.5)

    # always-on
    env_fans = fan.apply(env, points=desk)

    # comfort-triggered: only active on discomfort hours
    env_fans = fan.apply(env, points=desk, when=~result_base.is_comfortable())

    result_fans = calculate_pmv(env_fans)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np

from capy.core.environment import ThermalEnvironment
from capy.core.grid import SpatialGrid
from capy.core.schedule import OccupancySchedule


# ---------------------------------------------------------------------------
# Point selection
# ---------------------------------------------------------------------------

class PointSelector:
    """Specifies which grid points a PCS device is assigned to.

    Build one via the factory class-methods and pass to ``device.apply()``.
    The resolved mask is a ``(n_points,)`` boolean array.
    """

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = np.asarray(mask, dtype=bool)

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_indices(cls, indices: list[int], n_points: int) -> PointSelector:
        """Select points by explicit index list."""
        mask = np.zeros(n_points, dtype=bool)
        mask[list(indices)] = True
        return cls(mask)

    @classmethod
    def from_fraction(
        cls,
        grid: SpatialGrid,
        fraction: float,
        seed: int = 0,
    ) -> PointSelector:
        """Randomly select a fraction of points, area-weighted.

        Useful for scenarios like "50 % of desks have a fan".
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        rng = np.random.default_rng(seed)
        n_select = max(1, round(grid.n_points * fraction))
        weights = grid.areas / grid.areas.sum()
        chosen = rng.choice(grid.n_points, size=n_select, replace=False, p=weights)
        mask = np.zeros(grid.n_points, dtype=bool)
        mask[chosen] = True
        return cls(mask)

    @classmethod
    def from_bbox(
        cls,
        grid: SpatialGrid,
        x: tuple[float, float],
        y: tuple[float, float],
    ) -> PointSelector:
        """Select points whose (x, y) coordinates fall within a bounding box."""
        px, py = grid.points[:, 0], grid.points[:, 1]
        mask = (px >= x[0]) & (px <= x[1]) & (py >= y[0]) & (py <= y[1])
        return cls(mask)

    # ------------------------------------------------------------------ #
    # Resolution                                                           #
    # ------------------------------------------------------------------ #

    def resolve(self) -> np.ndarray:
        """Return the ``(n_points,)`` boolean mask."""
        return self._mask.copy()

    def __repr__(self) -> str:
        return f"PointSelector(n_selected={self._mask.sum()}/{len(self._mask)})"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PersonalComfortDevice(ABC):
    """Abstract base class for all PCS devices.

    Subclasses declare :attr:`_variable_deltas` — a ``{variable_name: delta}``
    dict listing every environmental parameter the device modifies and by how
    much (scalar, in the variable's native unit).  The base ``apply()``
    implementation uses this dict to apply all corrections at once, so a device
    that affects multiple parameters (e.g. both MRT and air temperature) only
    needs to list them — no additional ``apply()`` logic is required.

    A fallback baseline is provided for ``air_velocity`` (ASHRAE 55 still-air
    default, 0.1 m/s) so that fan devices work even when the environment was
    simulated without explicit air-speed data.  For all other variables the
    variable must already be present in the environment.
    """

    # Baseline values used when a variable is absent from the environment.
    _VARIABLE_DEFAULTS: dict[str, float] = {"air_velocity": 0.1}

    @property
    @abstractmethod
    def _variable_deltas(self) -> dict[str, float]:
        """Map of ``{variable_name: scalar_delta}`` for every parameter this
        device modifies.  Positive values increase, negative values decrease."""

    def apply(
        self,
        env: ThermalEnvironment,
        points: Union[list[int], np.ndarray, PointSelector],
        when: Union[OccupancySchedule, np.ndarray, None] = None,
    ) -> ThermalEnvironment:
        """Apply the device correction to selected points and hours.

        Parameters
        ----------
        env:
            Base environment to modify (not mutated).
        points:
            Which grid points the device covers. Accepts:

            - ``list[int]`` of point indices
            - ``(n_points,)`` boolean numpy array
            - :class:`PointSelector` instance
        when:
            Which hours the device is active. Accepts:

            - ``None`` → always active (default)
            - :class:`~capy.core.schedule.OccupancySchedule` → active during occupied hours
            - ``(n_hours,)`` bool array → same window for all selected points
            - ``(n_points, n_hours)`` bool array → per-point per-hour, e.g.
              ``~result.is_comfortable()`` to activate only on discomfort hours

        Returns
        -------
        ThermalEnvironment
            New environment with every variable in :attr:`_variable_deltas`
            updated at the active (point, hour) positions.
        """
        n_pts = env.grid.n_points
        n_hrs = len(env.time_index)
        point_mask = self._point_mask(points, n_pts)
        active = self._active_mask(point_mask, when, n_pts, n_hrs)

        updates: dict[str, np.ndarray] = {}
        for var, delta in self._variable_deltas.items():
            if env.has(var):
                arr = env.get(var).copy()
            elif var in self._VARIABLE_DEFAULTS:
                arr = np.full((n_pts, n_hrs), self._VARIABLE_DEFAULTS[var])
            else:
                raise KeyError(
                    f"Variable '{var}' not found in environment and has no default. "
                    f"Available: {list(env.variables)}"
                )
            arr[active] += delta
            updates[var] = arr
        return env.with_variables(**updates)

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _point_mask(
        points: Union[list[int], np.ndarray, PointSelector],
        n_points: int,
    ) -> np.ndarray:
        """Resolve *points* to a ``(n_points,)`` boolean mask."""
        if isinstance(points, PointSelector):
            return points.resolve()
        arr = np.asarray(points)
        if arr.dtype == bool:
            if arr.shape != (n_points,):
                raise ValueError(
                    f"Boolean mask shape {arr.shape} does not match n_points={n_points}."
                )
            return arr.copy()
        # Integer index list
        mask = np.zeros(n_points, dtype=bool)
        mask[arr] = True
        return mask

    @staticmethod
    def _active_mask(
        point_mask: np.ndarray,
        when: Union[OccupancySchedule, np.ndarray, None],
        n_points: int,
        n_hours: int,
    ) -> np.ndarray:
        """Return ``(n_points, n_hours)`` bool activity matrix.

        Combines *point_mask* (which rows) with *when* (which columns).
        """
        if when is None:
            hour_mask = np.ones(n_hours, dtype=bool)
        elif isinstance(when, OccupancySchedule):
            hour_mask = when.as_boolean()
        else:
            hour_mask = np.asarray(when, dtype=bool)

        if hour_mask.ndim == 1:
            active = point_mask[:, np.newaxis] & hour_mask[np.newaxis, :]
        elif hour_mask.ndim == 2:
            if hour_mask.shape != (n_points, n_hours):
                raise ValueError(
                    f"when array shape {hour_mask.shape} does not match "
                    f"(n_points={n_points}, n_hours={n_hours})."
                )
            active = point_mask[:, np.newaxis] & hour_mask
        else:
            raise ValueError(f"when must be 1-D or 2-D, got shape {hour_mask.shape}.")

        return active


# ---------------------------------------------------------------------------
# Concrete devices
# ---------------------------------------------------------------------------

@dataclass
class Fan(PersonalComfortDevice):
    """Desktop or pedestal fan — increases local air velocity.

    Raises air velocity at selected points, which reduces the PMV thermal
    sensation via increased convective cooling.  If the environment does not
    contain an ``air_velocity`` variable, the ASHRAE 55 still-air default of
    0.1 m/s is assumed as the baseline.

    Parameters
    ----------
    delta_v:
        Air velocity increment in m/s (default 0.5 m/s — typical desk fan).
    """

    delta_v: float = 0.5

    @property
    def _variable_deltas(self) -> dict[str, float]:
        return {"air_velocity": self.delta_v}


@dataclass
class FootWarmer(PersonalComfortDevice):
    """Under-desk foot warmer — raises local mean radiant temperature.

    Heated floor panels or foot-warming mats add a radiant heat contribution
    at seated level, increasing MRT for affected occupants.

    Parameters
    ----------
    delta_mrt:
        MRT increment in °C (default +3.0 °C).
    """

    delta_mrt: float = 3.0

    @property
    def _variable_deltas(self) -> dict[str, float]:
        return {"mean_radiant_temperature": self.delta_mrt}


@dataclass
class HeatedChair(PersonalComfortDevice):
    """Heated task chair — raises local mean radiant temperature.

    Radiant heat from the chair surface increases MRT for the seated
    occupant, effective for cold-season heating.

    Parameters
    ----------
    delta_mrt:
        MRT increment in °C (default +4.0 °C).
    """

    delta_mrt: float = 4.0

    @property
    def _variable_deltas(self) -> dict[str, float]:
        return {"mean_radiant_temperature": self.delta_mrt}


@dataclass
class CooledChair(PersonalComfortDevice):
    """Cooled task chair — lowers local mean radiant temperature.

    A cooling surface (e.g. phase-change or thermoelectric back/seat pad)
    reduces the effective MRT experienced by the occupant.

    Parameters
    ----------
    delta_mrt:
        MRT decrement in °C (default −4.0 °C, stored as a negative value).
    """

    delta_mrt: float = -4.0

    @property
    def _variable_deltas(self) -> dict[str, float]:
        return {"mean_radiant_temperature": self.delta_mrt}


@dataclass
class PersonalHeater(PersonalComfortDevice):
    """Personal space heater — raises local air temperature.

    A small convective heater directed at an individual workstation raises
    the local air temperature without affecting the rest of the room.

    Parameters
    ----------
    delta_ta:
        Air temperature increment in °C (default +2.0 °C).
    """

    delta_ta: float = 2.0

    @property
    def _variable_deltas(self) -> dict[str, float]:
        return {"air_temperature": self.delta_ta}
