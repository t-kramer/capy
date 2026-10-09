"""CSV loader — backward-compatible with comfortSIM export format.

comfortSIM exported each environmental variable as a CSV file with shape
(n_points, 8760): rows = sensor points, columns = hourly timesteps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from capy.core.environment import ThermalEnvironment
from capy.core.grid import SpatialGrid

HOURS_PER_YEAR = 8760

# Recognised filenames for each canonical variable
_FILE_MAP: Dict[str, list] = {
    "air_temperature": ["air_temperature.csv", "air_temp.csv", "ta.csv", "tdb.csv"],
    "mean_radiant_temperature": ["mean_radiant_temperature.csv", "mrt.csv", "tr.csv"],
    "relative_humidity": ["relative_humidity.csv", "rh.csv"],
    "air_velocity": ["air_velocity.csv", "v.csv", "vel.csv"],
    "operative_temperature": ["operative_temperature.csv", "t_op.csv", "top.csv"],
}


def from_csv_folder(
    folder: str,
    grid_pts: Optional[np.ndarray] = None,
    pts_file: Optional[str] = None,
    unit_area: float = 1.0,
    year: int = 2001,
) -> ThermalEnvironment:
    """Load a ThermalEnvironment from a folder of comfortSIM-format CSV files.

    Each CSV must have shape (n_points, 8760) — rows are sensor points,
    columns are hourly timesteps.

    Args:
        folder: Path to the folder containing CSV files.
        grid_pts: Optional (n, 3) array of sensor point coordinates. If not
            provided and no pts_file is given, a dummy geometry is created.
        pts_file: Optional path to a Radiance/Honeybee .pts file for grid geometry.
        unit_area: Area in m² assigned to each point when no area data is available.
        year: Reference year for building the time index (non-leap, default 2001).

    Returns:
        ThermalEnvironment with all discovered variables loaded.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    variables: Dict[str, np.ndarray] = {}
    n_points: Optional[int] = None

    for canonical, filenames in _FILE_MAP.items():
        for fname in filenames:
            fp = folder / fname
            if fp.exists():
                arr = np.loadtxt(fp, delimiter=",")
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[1] != HOURS_PER_YEAR:
                    raise ValueError(
                        f"{fp.name}: expected {HOURS_PER_YEAR} columns, "
                        f"got {arr.shape[1]}. Rows=points, columns=hours."
                    )
                if n_points is None:
                    n_points = arr.shape[0]
                elif arr.shape[0] != n_points:
                    raise ValueError(
                        f"{fp.name}: {arr.shape[0]} points but previous files "
                        f"had {n_points} points."
                    )
                variables[canonical] = arr
                break

    if not variables:
        raise FileNotFoundError(
            f"No recognised CSV files found in {folder}. "
            f"Expected one of: {[f for flist in _FILE_MAP.values() for f in flist]}"
        )

    # Build spatial grid
    if grid_pts is not None:
        grid = SpatialGrid.from_points(np.asarray(grid_pts), unit_area=unit_area)
    elif pts_file is not None:
        grid = SpatialGrid.from_pts_file(pts_file, unit_area=unit_area)
    else:
        # Dummy grid: points along x-axis, no real geometry
        pts = np.column_stack([
            np.arange(n_points, dtype=float),
            np.zeros(n_points),
            np.zeros(n_points),
        ])
        grid = SpatialGrid.from_points(pts, unit_area=unit_area, source="csv_dummy")

    time_index = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h")

    return ThermalEnvironment(
        grid=grid,
        time_index=time_index,
        variables=variables,
        info={"source": "csv", "folder": str(folder)},
    )
