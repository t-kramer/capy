"""Alliesthesia-based experiential metrics.

Grounded in Cabanac (1971), Parkinson & de Dear (2015, 2016, 2017), and the
review paper "Dynamic Thermal Environments as Biophilic Design".

These metrics distinguish beneficial thermal heterogeneity (which activates
positive alliesthesia) from problematic heterogeneity (discomfort).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from capy.comfort.base import ComfortResult
from capy.core.environment import ThermalEnvironment
from capy.core.schedule import OccupancySchedule
from capy.io.epw import load_dry_bulb as _load_dry_bulb
from capy.io.epw import running_mean as _running_mean_epw

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ThermalNicheResult:
    """Output of thermal_niches().

    All fractions are computed over the set of occupied hours (or all hours
    if no schedule is provided).
    """
    niche_fraction: float       # fraction of hours with ≥2 distinct niches
    discomfort_fraction: float  # fraction of hours where spatial range > delta_t_max
    uniform_fraction: float     # fraction of hours where spatial range < delta_t_min

    # Per-hour classification arrays (length = n_occupied_hours)
    hourly_n_niches: np.ndarray     # (n_hours,) number of niches per hour
    hourly_spatial_range: np.ndarray  # (n_hours,) max−min T_op per hour

    delta_t_min: float
    delta_t_max: float
    niche_deviation: float

    def __repr__(self) -> str:
        return (
            f"ThermalNicheResult("
            f"niche={self.niche_fraction * 100:.1f}%, "
            f"discomfort={self.discomfort_fraction * 100:.1f}%, "
            f"uniform={self.uniform_fraction * 100:.1f}%)"
        )


@dataclass
class ContextScoreResult:
    """Output of context_score().

    ``scores[i, t]`` is one of {-1, 0, +1}.
    """
    scores: np.ndarray       # (n_points, n_hours) int8
    beneficial_fraction: float
    neutral_fraction: float
    problematic_fraction: float

    def __repr__(self) -> str:
        return (
            f"ContextScoreResult("
            f"beneficial={self.beneficial_fraction * 100:.1f}%, "
            f"neutral={self.neutral_fraction * 100:.1f}%, "
            f"problematic={self.problematic_fraction * 100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def thermal_niches(
    env: ThermalEnvironment,
    schedule: Optional[OccupancySchedule] = None,
    delta_t_min: float = 2.0,
    delta_t_max: float = 4.0,
    niche_deviation: float = 1.0,
    variable: str = "operative_temperature",
) -> ThermalNicheResult:
    """Assess the spatial thermal 'menu' — co-existing warm, neutral, and cool zones.

    Per occupied hour, classifies sensor points as:

    * **warm niche** — temperature > spatial mean + *niche_deviation*
    * **cool niche** — temperature < spatial mean − *niche_deviation*
    * **neutral** — within *niche_deviation* of the spatial mean

    Returns the fraction of occupied hours that offer ≥2 distinct niches
    (the *thermal menu*), as well as hours where the spatial range is too
    uniform (<delta_t_min) or too extreme (>delta_t_max).

    Parameters
    ----------
    env:
        ThermalEnvironment to analyse.
    schedule:
        Occupancy schedule; all hours used if ``None``.
    delta_t_min:
        Minimum spatial temperature range (°C) to consider the environment
        as thermally varied (default 2.0).
    delta_t_max:
        Spatial range above which the environment is likely to include
        uncomfortable extremes (default 4.0).
    niche_deviation:
        Deviation from the spatial mean (°C) needed for a point to be
        classified as a warm or cool niche (default 1.0).
    variable:
        Variable to use for spatial mean (default: ``'operative_temperature'``).

    Returns
    -------
    ThermalNicheResult
    """
    data = env.get(variable)                          # (n_points, 8760)

    # Select occupied hours
    if schedule is not None:
        mask = schedule.mask                          # (8760,) bool
        data_occ = data[:, mask]                      # (n_points, n_occ)
    else:
        data_occ = data

    n_hours = data_occ.shape[1]
    if n_hours == 0:
        return ThermalNicheResult(
            niche_fraction=0.0, discomfort_fraction=0.0, uniform_fraction=0.0,
            hourly_n_niches=np.array([]), hourly_spatial_range=np.array([]),
            delta_t_min=delta_t_min, delta_t_max=delta_t_max,
            niche_deviation=niche_deviation,
        )

    spatial_mean = data_occ.mean(axis=0)              # (n_occ,)
    spatial_range = data_occ.max(axis=0) - data_occ.min(axis=0)  # (n_occ,)

    # Classify each point each hour
    warm = data_occ > spatial_mean[None, :] + niche_deviation    # (n, n_occ)
    cool = data_occ < spatial_mean[None, :] - niche_deviation    # (n, n_occ)

    has_warm = warm.any(axis=0)      # (n_occ,) bool
    has_cool = cool.any(axis=0)      # (n_occ,) bool
    n_niches = has_warm.astype(int) + has_cool.astype(int) + 1   # ≥1 (neutral always)

    multi_niche = n_niches >= 2
    discomfort = spatial_range > delta_t_max
    uniform = spatial_range < delta_t_min

    return ThermalNicheResult(
        niche_fraction=float(multi_niche.mean()),
        discomfort_fraction=float(discomfort.mean()),
        uniform_fraction=float(uniform.mean()),
        hourly_n_niches=n_niches,
        hourly_spatial_range=spatial_range,
        delta_t_min=delta_t_min,
        delta_t_max=delta_t_max,
        niche_deviation=niche_deviation,
    )


def context_score(
    env: ThermalEnvironment,
    comfort_result: ComfortResult,
    epw_path: Optional[str] = None,
    schedule: Optional[OccupancySchedule] = None,
    variable: str = "operative_temperature",
) -> ContextScoreResult:
    """Per-point per-hour context-dependent comfort classification.

    Classifies each ``(point, hour)`` as:

    * ``+1`` **beneficial** — discomfort that alliesthesia theory predicts is
      enjoyable: mild warmth in a cold season (outdoor running mean < 15°C) or
      mild coolness in a hot season (outdoor running mean > 22°C).
    * ``0`` **neutral** — within the comfort model's band.
    * ``-1`` **problematic** — discomfort without compensating seasonal context.

    The seasonal context comes from the EPW running mean outdoor temperature
    (7-day exponential weighted mean, α=0.8, per ASHRAE 55).

    Parameters
    ----------
    env:
        ThermalEnvironment.
    comfort_result:
        Output of any capy comfort model. Used to identify which
        ``(point, hour)`` pairs are outside the comfort band.
    epw_path:
        Path to EPW file for seasonal context. Falls back to ``env.epw_path``.
    schedule:
        If provided, unoccupied hours are set to ``0`` (neutral) in the output.
    variable:
        Indoor variable for seasonal warm/cool context (default:
        ``'operative_temperature'``).

    Returns
    -------
    ContextScoreResult
    """
    resolved_epw = epw_path or env.epw_path
    if resolved_epw is None:
        raise ValueError(
            "context_score() requires an EPW file for seasonal context. "
            "Pass epw_path= or set env.epw_path."
        )

    t_op = env.get(variable)                          # (n_points, 8760)
    n_points, n_hours = t_op.shape

    # Outdoor running mean from EPW
    t_outdoor = _load_dry_bulb(resolved_epw)           # (8760,)
    t_rm = _running_mean_epw(t_outdoor, alpha=0.8)    # (8760,) 7-day exp mean

    # Comfort band mask from comfort_result
    in_comfort = comfort_result.is_comfortable()       # (n_points, 8760) bool

    # Context rules:
    # Warm season (t_rm >= 22°C): feeling cool is beneficial, warm is problematic
    # Cold season (t_rm <= 15°C): feeling warm is beneficial, cool is problematic
    # Shoulder (15 < t_rm < 22): discomfort is just discomfort → -1
    warm_season = t_rm >= 22.0                         # (8760,) bool
    cold_season = t_rm <= 15.0                         # (8760,) bool

    t_op_cool = t_op < t_op.mean(axis=0)              # (n, 8760) cooler than spatial mean
    t_op_warm = t_op > t_op.mean(axis=0)

    scores = np.zeros((n_points, n_hours), dtype=np.int8)

    # Outside comfort → potentially +1 or -1
    out = ~in_comfort
    beneficial = (
        (out & warm_season[None, :] & t_op_cool)  # cool spot on hot day
        | (out & cold_season[None, :] & t_op_warm)  # warm spot on cold day
    )
    problematic = out & ~beneficial

    scores[beneficial] = 1
    scores[problematic] = -1

    # Zero out unoccupied hours
    if schedule is not None:
        occ_mask = schedule.mask                       # (8760,)
        scores[:, ~occ_mask] = 0

    total = scores.size
    return ContextScoreResult(
        scores=scores,
        beneficial_fraction=float((scores == 1).sum() / total),
        neutral_fraction=float((scores == 0).sum() / total),
        problematic_fraction=float((scores == -1).sum() / total),
    )

