from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from capy.core.environment import ThermalEnvironment
    from capy.core.schedule import OccupancySchedule


@dataclass
class ComfortResult:
    """Output of any comfort model.

    values:   (n_points, n_hours) continuous comfort index (e.g. PMV)
    discrete: (n_points, n_hours) integer classification {-1, 0, +1}
              -1 = too cold, 0 = comfortable, +1 = too warm
    """

    values: np.ndarray
    discrete: np.ndarray
    model: str
    comfort_band: tuple          # (lower, upper) thresholds for 'comfortable'
    env: ThermalEnvironment

    def __post_init__(self) -> None:
        expected = (self.env.grid.n_points, len(self.env.time_index))
        if self.values.shape != expected:
            raise ValueError(
                f"values shape {self.values.shape} != expected {expected}"
            )
        if self.discrete.shape != expected:
            raise ValueError(
                f"discrete shape {self.discrete.shape} != expected {expected}"
            )

    def is_comfortable(self) -> np.ndarray:
        """Boolean mask — True where discrete == 0 (comfortable)."""
        return self.discrete == 0

    def comfort_fraction(
        self, schedule: Optional[OccupancySchedule] = None
    ) -> np.ndarray:
        """Per-point fraction of (occupied) hours that are comfortable.

        Returns:
            (n_points,) array of fractions in [0, 1].
        """
        mask = schedule.as_boolean() if schedule is not None else np.ones(
            len(self.env.time_index), dtype=bool
        )
        return self.is_comfortable()[:, mask].mean(axis=1)

    def __repr__(self) -> str:
        n_pts = self.env.grid.n_points
        n_h = len(self.env.time_index)
        return (
            f"ComfortResult(model='{self.model}', "
            f"band={self.comfort_band}, "
            f"shape=({n_pts}, {n_h}))"
        )
