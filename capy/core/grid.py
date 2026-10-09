from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np


@dataclass
class SpatialGrid:
    """Spatial measurement grid for a single zone.

    Each instance represents one zone (room / space). Multi-zone analysis
    is done by working with a collection of SpatialGrid objects, one per zone.

    All arrays must share the same length n (number of sensor points).
    """

    points: np.ndarray          # (n, 3)  x/y/z in model units (metres)
    areas: np.ndarray           # (n,)    floor area in m² represented by each point
    zone_id: Union[str, int] = 0    # opaque zone identifier (e.g. "98juht34jk")
    zone_name: str = ""             # human-readable label (e.g. "Office, South")
    sensor_height: float = 0.8     # m above floor (seated occupant default)
    north_angle: float = 0.0       # degrees clockwise from +Y axis to geographic North
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.areas = np.asarray(self.areas, dtype=float)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"points must be shape (n, 3), got {self.points.shape}"
            )
        n = self.points.shape[0]
        if self.areas.shape != (n,):
            raise ValueError(f"areas must be shape ({n},), got {self.areas.shape}")

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_area(self) -> float:
        return float(self.areas.sum())

    @property
    def area_weights(self) -> np.ndarray:
        """Normalised area weights that sum to 1.

        Used for area-weighted spatial aggregation — e.g. computing what
        fraction of floor *area* (not just points) meets a condition.
        Uniform grids produce equal weights; non-uniform grids (denser near
        walls, complex geometry) require per-point areas for correct results.
        """
        total = self.areas.sum()
        if total == 0.0:
            raise ValueError("Cannot compute area weights: all areas are zero.")
        return self.areas / total

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        unit_area: float = 1.0,
        zone_id: Union[str, int] = 0,
        zone_name: str = "",
        sensor_height: float = 0.8,
        **metadata,
    ) -> SpatialGrid:
        """Create a uniform grid from an (n, 3) points array.

        Each point is assigned equal area. Use ``zone_id`` and ``zone_name``
        to identify the zone this grid belongs to.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1 and points.shape[0] == 3:
            points = points[np.newaxis, :]
        n = points.shape[0]
        areas = np.full(n, unit_area)
        return cls(
            points=points,
            areas=areas,
            zone_id=zone_id,
            zone_name=zone_name,
            sensor_height=sensor_height,
            metadata=metadata,
        )

    @classmethod
    def from_pts_file(
        cls,
        path: str,
        unit_area: float = 1.0,
        zone_id: Union[str, int] = 0,
        zone_name: str = "",
    ) -> SpatialGrid:
        """Load grid points from a Radiance/Honeybee .pts file.

        Each row: x y z dx dy dz  (direction columns are ignored).
        """
        pts = np.loadtxt(path)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        coords = pts[:, :3]
        return cls.from_points(
            coords,
            unit_area=unit_area,
            zone_id=zone_id,
            zone_name=zone_name,
            source="pts_file",
            path=path,
        )

    def __repr__(self) -> str:
        label = self.zone_name if self.zone_name else str(self.zone_id)
        return (
            f"SpatialGrid(zone='{label}', "
            f"n_points={self.n_points}, "
            f"total_area={self.total_area:.1f} m²)"
        )
