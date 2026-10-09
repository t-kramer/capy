"""Shared pytest fixtures for capy tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capy.core.environment import ThermalEnvironment
from capy.core.grid import SpatialGrid
from capy.core.schedule import OccupancySchedule

DATA_DIR = Path(__file__).parent / "data"
CSV_DIR = DATA_DIR / "csv"

N_POINTS = 60
N_HOURS = 8760
YEAR = 2001


@pytest.fixture
def sample_grid() -> SpatialGrid:
    """60-point uniform grid on a 6×10 rectangular floor plan (1 m spacing)."""
    xs = np.tile(np.arange(6), 10).astype(float)
    ys = np.repeat(np.arange(10), 6).astype(float)
    zs = np.zeros(N_POINTS)
    points = np.column_stack([xs, ys, zs])
    areas = np.full(N_POINTS, 1.0)  # 1 m² each → 60 m² total
    return SpatialGrid(points=points, areas=areas)


@pytest.fixture
def sample_env(sample_grid) -> ThermalEnvironment:
    """ThermalEnvironment loaded from comfortSIM sample CSV data."""
    time_index = pd.date_range(f"{YEAR}-01-01", periods=N_HOURS, freq="h")

    ta = np.loadtxt(CSV_DIR / "air_temperature.csv", delimiter=",")
    mrt = np.loadtxt(CSV_DIR / "mean_radiant_temperature.csv", delimiter=",")
    rh = np.loadtxt(CSV_DIR / "relative_humidity.csv", delimiter=",")

    return ThermalEnvironment(
        grid=sample_grid,
        time_index=time_index,
        variables={
            "air_temperature": ta,
            "mean_radiant_temperature": mrt,
            "relative_humidity": rh,
        },
        info={"source": "comfortSIM_sample"},
    )


@pytest.fixture
def work_schedule() -> OccupancySchedule:
    return OccupancySchedule.work_hours()


@pytest.fixture
def synthetic_env(sample_grid) -> ThermalEnvironment:
    """Fully synthetic environment with controlled temperature patterns.

    All points have air_temperature = 22°C + seasonal variation,
    plus spatial gradient (warmer near window on one side).
    """
    time_index = pd.date_range(f"{YEAR}-01-01", periods=N_HOURS, freq="h")
    hours = np.arange(N_HOURS)

    # Seasonal variation: ±5°C over the year
    seasonal = 5 * np.sin(2 * np.pi * hours / N_HOURS - np.pi / 2)
    # Diurnal variation: ±2°C within each day
    diurnal = 2 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi / 2)
    base_temp = 22.0 + seasonal + diurnal  # (8760,)

    # Spatial gradient: 0–4°C across 60 points (simulates window proximity)
    spatial_gradient = np.linspace(0, 4, N_POINTS)  # (60,)
    ta = base_temp[None, :] + spatial_gradient[:, None]  # (60, 8760)

    return ThermalEnvironment(
        grid=sample_grid,
        time_index=time_index,
        variables={
            "air_temperature": ta,
            "mean_radiant_temperature": ta + 2,
            "relative_humidity": np.full((N_POINTS, N_HOURS), 50.0),
        },
        info={"source": "synthetic"},
    )
