"""Honeybee / Ladybug Tools I/O adapter.

Converts Ladybug Tools data structures (SensorGrid, HourlyContinuousCollection)
into capy's ThermalEnvironment.

Requires: ladybug-core (pip install ladybug-core) or the full LBT stack
when running inside Rhino/Grasshopper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from capy.core.environment import ThermalEnvironment
from capy.core.grid import SpatialGrid

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_honeybee_grid(
    sensor_grid,
    data_collections: Dict[str, Union[object, List[object], np.ndarray]],
    epw_path: Optional[str] = None,
    unit_area: float = 1.0,
) -> ThermalEnvironment:
    """Build a ThermalEnvironment from Honeybee simulation outputs.

    Parameters
    ----------
    sensor_grid:
        A ``ladybug_radiance.sensorgrid.SensorGrid`` providing sensor
        coordinates, **or** a (n, 3) numpy array of xyz positions.
    data_collections:
        Dict mapping variable names to data. Each value may be:

        * A list of ``HourlyContinuousCollection`` objects — one per sensor
          point (shape inferred as ``[n_points][8760]``).
        * A single ``HourlyContinuousCollection`` — broadcast to all points.
        * A (n_points, 8760) numpy array — used directly.
        * A (8760,) numpy array — broadcast to all points.

        Canonical variable names (or any recognised alias)::

            'air_temperature', 'mean_radiant_temperature',
            'relative_humidity', 'air_velocity', 'operative_temperature'

    epw_path:
        Path to the EPW used for the simulation (stored for downstream use
        by adaptive comfort models).
    unit_area:
        Floor area (m²) assigned uniformly to each sensor when the grid
        carries no area data.

    Returns
    -------
    ThermalEnvironment
        Ready for capy metric functions.

    Raises
    ------
    ImportError
        If ``ladybug-core`` is not installed when a SensorGrid object is passed.
    ValueError
        If point counts are inconsistent across collections.

    Examples
    --------
    Inside a GHPython component::

        from capy.io.honeybee import from_honeybee_grid

        env = from_honeybee_grid(
            sensor_grid=SensorGrid,
            data_collections={
                'air_temperature': air_temp_data,
                'mean_radiant_temperature': mrt_data,
                'relative_humidity': rh_data,
            },
            epw_path=epw._file_path,
        )
    """
    pts = _extract_sensor_points(sensor_grid)
    n_points = pts.shape[0]

    zone_id, zone_name = _extract_zone_identity(sensor_grid)

    grid = SpatialGrid(
        points=pts,
        areas=np.full(n_points, unit_area),
        zone_id=zone_id,
        zone_name=zone_name,
        metadata={"source": "honeybee", "sensor_grid_id": _grid_id(sensor_grid)},
    )

    # Determine year from the first collection that carries header info
    year = _infer_year(data_collections)
    time_index = pd.date_range(f"{year}-01-01", periods=8760, freq="h")

    variables: Dict[str, np.ndarray] = {}
    for var_name, collection in data_collections.items():
        variables[var_name] = _collection_to_array(collection, n_points, var_name)

    return ThermalEnvironment(
        grid=grid,
        time_index=time_index,
        variables=variables,
        epw_path=epw_path,
        info={"source": "honeybee", "sensor_grid_id": _grid_id(sensor_grid)},
    )


def from_lbt_data_collection(
    data_collection,
    variable_name: str,
    n_points: Optional[int] = None,
) -> np.ndarray:
    """Extract a (n_points, 8760) array from a LBT data collection.

    Parameters
    ----------
    data_collection:
        Either a list of ``HourlyContinuousCollection`` objects (one per
        sensor point) or a single collection that will be broadcast.
    variable_name:
        Name used in error messages.
    n_points:
        Expected number of sensor points. Validated when provided.

    Returns
    -------
    np.ndarray of shape ``(n_points, 8760)``.
    """
    if n_points is None:
        if isinstance(data_collection, (list, tuple)):
            n_points = len(data_collection)
        else:
            n_points = 1
    return _collection_to_array(data_collection, n_points, variable_name)


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _extract_sensor_points(sensor_grid) -> np.ndarray:
    """Return (n, 3) float64 array of sensor xyz coordinates."""
    if isinstance(sensor_grid, np.ndarray):
        arr = np.asarray(sensor_grid, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        raise ValueError(
            f"sensor_grid numpy array must be (n, 3), got {arr.shape}"
        )

    # ladybug_radiance.sensorgrid.SensorGrid
    if hasattr(sensor_grid, "sensors"):
        try:
            sensors = sensor_grid.sensors
            return np.array([[s.pos.x, s.pos.y, s.pos.z] for s in sensors], dtype=float)
        except AttributeError:
            pass

    # ladybug.sensorgrid.SensorGrid (older API uses .sensor_positions)
    if hasattr(sensor_grid, "sensor_positions"):
        pos = sensor_grid.sensor_positions
        return np.array([[p.x, p.y, p.z] for p in pos], dtype=float)

    # Fallback: try to iterate and pull xyz
    try:
        pts = list(sensor_grid)
        return np.array([[p.x, p.y, p.z] for p in pts], dtype=float)
    except (TypeError, AttributeError):
        pass

    raise ValueError(
        f"Cannot extract sensor points from {type(sensor_grid).__name__}. "
        "Pass a ladybug SensorGrid or a (n, 3) numpy array."
    )


def _extract_zone_identity(sensor_grid):
    """Return (zone_id, zone_name) from a SensorGrid object, or sensible defaults."""
    zone_id = getattr(sensor_grid, "identifier", 0) or 0
    zone_name = getattr(sensor_grid, "display_name", "") or ""
    if not zone_name:
        zone_name = getattr(sensor_grid, "room_identifier", "") or ""
    return zone_id, zone_name


def _grid_id(sensor_grid) -> str:
    for attr in ("identifier", "id", "name"):
        val = getattr(sensor_grid, attr, None)
        if val is not None:
            return str(val)
    return "unknown"


def _infer_year(
    data_collections: Dict[str, object],
    default: int = 2001,
) -> int:
    """Try to read the analysis-period year from LBT collection headers."""
    for collection in data_collections.values():
        items = collection if isinstance(collection, (list, tuple)) else [collection]
        for item in items:
            try:
                year = item.header.analysis_period.VALIDYEAR
                return int(year)
            except AttributeError:
                pass
    return default


def _collection_to_array(
    collection,
    n_points: int,
    var_name: str = "variable",
) -> np.ndarray:
    """Convert a data collection (or list/array) to (n_points, 8760) float64."""

    # --- numpy array ---
    if isinstance(collection, np.ndarray):
        arr = collection.astype(float)
        if arr.ndim == 1:
            if arr.shape[0] != 8760:
                raise ValueError(
                    f"'{var_name}': 1-D array must have 8760 values, got {arr.shape[0]}"
                )
            return np.tile(arr, (n_points, 1))
        if arr.ndim == 2:
            if arr.shape != (n_points, 8760):
                raise ValueError(
                    f"'{var_name}': 2-D array must be ({n_points}, 8760), got {arr.shape}"
                )
            return arr
        raise ValueError(f"'{var_name}': array must be 1-D or 2-D, got {arr.ndim}-D")

    # --- list/tuple of collections (one per sensor) ---
    if isinstance(collection, (list, tuple)):
        if len(collection) != n_points:
            raise ValueError(
                f"'{var_name}': expected {n_points} collections (one per sensor), "
                f"got {len(collection)}"
            )
        rows = []
        for i, col in enumerate(collection):
            vals = _single_collection_values(col, var_name, i)
            rows.append(vals)
        arr = np.array(rows, dtype=float)
        if arr.shape != (n_points, 8760):
            raise ValueError(
                f"'{var_name}': assembled array is {arr.shape}, expected ({n_points}, 8760)"
            )
        return arr

    # --- single collection (broadcast to all points) ---
    vals = _single_collection_values(collection, var_name)
    return np.tile(vals, (n_points, 1))


def _single_collection_values(collection, var_name: str, idx: Optional[int] = None) -> np.ndarray:
    """Extract the (8760,) float64 array from one LBT collection."""
    loc = f"'{var_name}'" if idx is None else f"'{var_name}' (sensor {idx})"

    if hasattr(collection, "values"):
        vals = np.asarray(list(collection.values), dtype=float)
    elif hasattr(collection, "to_array"):
        vals = np.asarray(collection.to_array(), dtype=float)
    else:
        try:
            vals = np.asarray(list(collection), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{loc}: cannot extract values from {type(collection).__name__}"
            ) from exc

    if vals.shape[0] != 8760:
        raise ValueError(
            f"{loc}: expected 8760 hourly values, got {vals.shape[0]}"
        )
    return vals
