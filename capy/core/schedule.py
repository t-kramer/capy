from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760


@dataclass
class OccupancySchedule:
    """Binary occupancy mask over 8760 annual hours.

    True = occupied, False = unoccupied.
    """

    mask: np.ndarray  # (8760,) bool

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.shape != (HOURS_PER_YEAR,):
            raise ValueError(
                f"mask must be shape ({HOURS_PER_YEAR},), got {self.mask.shape}"
            )

    @property
    def n_occupied(self) -> int:
        return int(self.mask.sum())

    @property
    def occupied_fraction(self) -> float:
        return float(self.mask.mean())

    def as_boolean(self) -> np.ndarray:
        return self.mask

    def as_indices(self) -> np.ndarray:
        return np.where(self.mask)[0]

    @classmethod
    def work_hours(
        cls,
        start_hour: int = 8,
        end_hour: int = 18,
        include_weekends: bool = False,
        year: int = 2001,
    ) -> OccupancySchedule:
        """Weekday working hours schedule (8 am – 6 pm by default)."""
        idx = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h")
        mask = (idx.hour >= start_hour) & (idx.hour < end_hour)
        if not include_weekends:
            mask &= idx.dayofweek < 5
        return cls(mask=np.asarray(mask, dtype=bool))

    @classmethod
    def always(cls) -> OccupancySchedule:
        """All 8760 hours occupied."""
        return cls(mask=np.ones(HOURS_PER_YEAR, dtype=bool))

    @classmethod
    def from_array(cls, array) -> OccupancySchedule:
        return cls(mask=np.asarray(array, dtype=bool))

    @classmethod
    def from_hours(cls, hours: List[int], year: int = 2001) -> OccupancySchedule:
        """Occupied during specific hours-of-day (0–23) every day."""
        idx = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h")
        mask = np.isin(idx.hour, hours)
        return cls(mask=mask)

    def __repr__(self) -> str:
        return (
            f"OccupancySchedule(occupied={self.n_occupied}h / {HOURS_PER_YEAR}h, "
            f"{self.occupied_fraction * 100:.1f}%)"
        )
