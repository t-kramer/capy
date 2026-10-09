"""Diurnal temperature flux metrics.

Evidence base (from "Dynamic Thermal Environments as Biophilic Design"):
  - Daily operative temperature amplitude of 2–4°C is associated with
    circadian rhythm alignment and occupant wellbeing.
  - Slow warming rates (0.5–1.0°C/hr) are preferred; abrupt changes
    (>2°C/hr) are generally dispreferred.
  - Phase alignment with the outdoor cycle supports the diurnal zeitgeber
    function of the thermal environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from capy.core.environment import ThermalEnvironment
from capy.core.schedule import OccupancySchedule
from capy.io.epw import load_dry_bulb as _load_dry_bulb

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DiurnalAmplitudeResult:
    amplitude_matrix: np.ndarray   # (n_points, 365)  daily ΔT per point
    median_amplitude: np.ndarray   # (n_points,)       annual median per point
    median_amplitude_grid: float   # median of per-point medians — single grid summary
    mean_amplitude: float          # mean over all points and days
    fraction_in_target: float      # % of point-days within [target_min, target_max]
    target_min: float
    target_max: float
    variable: str

    def __repr__(self) -> str:
        return (
            f"DiurnalAmplitudeResult(median_grid={self.median_amplitude_grid:.2f}°C, "
            f"in_target={self.fraction_in_target * 100:.1f}%)"
        )


@dataclass
class DriftRateResult:
    drift_array: np.ndarray        # (n_points, 8759)  |ΔT| per hour per point
    median_rate: np.ndarray        # (n_points,)        median per point
    abrupt_fraction: float         # % of all point-hours exceeding threshold
    abrupt_threshold: float
    variable: str

    def __repr__(self) -> str:
        return (
            f"DriftRateResult(median={float(np.median(self.median_rate)):.2f}°C/hr, "
            f"abrupt={self.abrupt_fraction * 100:.1f}%)"
        )


@dataclass
class PhaseAlignmentResult:
    scores: np.ndarray             # (n_points,)  Pearson r per point
    mean_score: float
    n_days_included: int           # days used after outdoor_temp_range filter
    variable: str

    def __repr__(self) -> str:
        return (
            f"PhaseAlignmentResult(mean_r={self.mean_score:.3f}, "
            f"n_days={self.n_days_included}, n_points={len(self.scores)})"
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def diurnal_amplitude(
    env: ThermalEnvironment,
    variable: str = "operative_temperature",
    schedule: Optional[OccupancySchedule] = None,
    target_min: float = 2.0,
    target_max: float = 4.0,
) -> DiurnalAmplitudeResult:
    """Daily temperature amplitude per sensor point.

    Computes ``max(T) − min(T)`` over each 24-hour day for every point, then
    summarises as the per-point annual median and the fraction of point-days
    within the evidence-based 2–4°C target band.

    Parameters
    ----------
    env:
        ThermalEnvironment to analyse.
    variable:
        Variable name to retrieve (default: ``'operative_temperature'``).
    schedule:
        If provided, only days that contain at least one occupied hour are
        included in the summary statistics.
    target_min:
        Lower bound of the beneficial daily amplitude in °C (default 2.0).
    target_max:
        Upper bound of the beneficial daily amplitude in °C (default 4.0).

    Returns
    -------
    DiurnalAmplitudeResult
    """
    data = env.get(variable)                     # (n_points, 8760)
    n_points = data.shape[0]

    # Reshape to (n_points, 365, 24) for per-day aggregation
    data_daily = data.reshape(n_points, 365, 24)      # (n, 365, 24)
    amplitude_matrix = data_daily.max(axis=2) - data_daily.min(axis=2)  # (n, 365)

    # Day mask from schedule (which days have ≥1 occupied hour)
    if schedule is not None:
        hour_mask = schedule.mask.reshape(365, 24)     # (365, 24)
        day_mask = hour_mask.any(axis=1)               # (365,) bool
        included = amplitude_matrix[:, day_mask]
    else:
        included = amplitude_matrix

    median_amplitude = np.median(included, axis=1)    # (n_points,)
    mean_amplitude = float(included.mean())

    in_band = (included >= target_min) & (included <= target_max)
    fraction_in_target = float(in_band.mean())

    return DiurnalAmplitudeResult(
        amplitude_matrix=amplitude_matrix,
        median_amplitude=median_amplitude,
        median_amplitude_grid=float(np.median(median_amplitude)),
        mean_amplitude=mean_amplitude,
        fraction_in_target=fraction_in_target,
        target_min=target_min,
        target_max=target_max,
        variable=variable,
    )


def temperature_drift_rate(
    env: ThermalEnvironment,
    variable: str = "operative_temperature",
    abrupt_threshold: float = 2.0,
) -> DriftRateResult:
    """Hourly temperature drift rate per sensor point.

    Computes ``|T[t+1] − T[t]|`` for each point, returning the full
    ``(n_points, 8759)`` rate array plus summary statistics.

    Parameters
    ----------
    env:
        ThermalEnvironment to analyse.
    variable:
        Variable name (default: ``'operative_temperature'``).
    abrupt_threshold:
        Rate in °C/hr above which change is flagged as abrupt (default 2.0).

    Returns
    -------
    DriftRateResult
    """
    data = env.get(variable)                          # (n_points, 8760)
    drift_array = np.abs(np.diff(data, axis=1))       # (n_points, 8759)

    median_rate = np.median(drift_array, axis=1)      # (n_points,)
    abrupt_fraction = float((drift_array > abrupt_threshold).mean())

    return DriftRateResult(
        drift_array=drift_array,
        median_rate=median_rate,
        abrupt_fraction=abrupt_fraction,
        abrupt_threshold=abrupt_threshold,
        variable=variable,
    )


def diurnal_phase_alignment(
    env: ThermalEnvironment,
    epw_path: Optional[str] = None,
    variable: str = "operative_temperature",
    outdoor_temp_range: tuple = (15.0, 28.0),
) -> PhaseAlignmentResult:
    """Pearson correlation of indoor vs outdoor daily temperature cycle.

    Computes the correlation between the mean indoor 24-hour profile and the
    mean outdoor profile from the EPW, restricted to days where the outdoor
    daily mean falls within *outdoor_temp_range*. This focuses the metric on
    the intermediate season when thermal coupling is most relevant — excluding
    cold winter days (where decoupling is intentional) and peak summer days.

    Scores per point:

    * ``+1.0`` — indoor peaks align with outdoor peaks (naturally coupled).
    *  ``0.0`` — no relationship.
    * ``-1.0`` — inverted cycle (e.g., overcooled building peaking at night).

    Parameters
    ----------
    env:
        ThermalEnvironment. Uses ``env.epw_path`` if *epw_path* is not given.
    epw_path:
        Path to the EPW file that provides the outdoor dry-bulb temperature.
        If neither *epw_path* nor ``env.epw_path`` is available, raises
        ``ValueError``.
    variable:
        Indoor variable to correlate (default: ``'operative_temperature'``).
    outdoor_temp_range:
        ``(low, high)`` range of daily mean outdoor temperature (°C) used to
        select qualifying days (default ``(15.0, 28.0)``).

    Returns
    -------
    PhaseAlignmentResult
    """
    resolved_epw = epw_path or env.epw_path
    if resolved_epw is None:
        raise ValueError(
            "diurnal_phase_alignment() requires an EPW file. Pass epw_path= or "
            "set env.epw_path when building the ThermalEnvironment."
        )

    t_outdoor = _load_dry_bulb(resolved_epw)               # (8760,)
    daily_outdoor = t_outdoor.reshape(365, 24)             # (365, 24)
    daily_mean = daily_outdoor.mean(axis=1)                # (365,)

    t_lo, t_hi = outdoor_temp_range
    day_mask = (daily_mean >= t_lo) & (daily_mean <= t_hi) # (365,) bool
    n_days = int(day_mask.sum())
    if n_days == 0:
        raise ValueError(
            f"No days found with outdoor daily mean in {outdoor_temp_range}°C. "
            "Adjust outdoor_temp_range or check the EPW file."
        )

    outdoor_profile = daily_outdoor[day_mask].mean(axis=0)  # (24,)

    data = env.get(variable)                                 # (n_points, 8760)
    n_points = data.shape[0]
    indoor_daily = data.reshape(n_points, 365, 24)[:, day_mask, :].mean(axis=1)  # (n, 24)

    scores = np.array([
        float(np.corrcoef(indoor_daily[i], outdoor_profile)[0, 1])
        for i in range(n_points)
    ])

    return PhaseAlignmentResult(
        scores=scores,
        mean_score=float(np.mean(scores)),
        n_days_included=n_days,
        variable=variable,
    )
