"""PMV/PPD comfort model using pythermalcomfort."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from capy.comfort.base import ComfortResult
from capy.core.environment import ThermalEnvironment


@dataclass
class PMVParameters:
    """Personal parameters for the PMV model."""

    met: float = 1.1          # metabolic rate (met) — typical office work
    clo: float = 0.8          # clothing insulation (clo) — light trousers + shirt
    wme: float = 0.0          # external mechanical work (usually 0)
    standard: str = "ASHRAE"  # 'ASHRAE' or 'ISO'
    limit_inputs: bool = True  # clip inputs to standard's valid range


_COMFORT_BAND = (-0.5, 0.5)


def calculate_pmv(
    env: ThermalEnvironment,
    params: Optional[PMVParameters] = None,
    default_velocity: float = 0.1,
) -> ComfortResult:
    """Calculate PMV for all sensor points over all hours.

    Args:
        env: ThermalEnvironment with air_temperature, mean_radiant_temperature,
             and relative_humidity. Air velocity is optional — falls back to
             default_velocity (0.1 m/s) if not present.
        params: PMVParameters for met, clo, standard, etc.
        default_velocity: Fallback air speed in m/s when not in env.variables.
                          Matches the EnergyPlus default for still indoor air.

    Returns:
        ComfortResult with continuous PMV values and {-1, 0, +1} discrete classes.
    """
    try:
        from pythermalcomfort.utilities import v_relative
    except ImportError as exc:
        raise ImportError(
            "pythermalcomfort is required for PMV calculations. "
            "Install it with: pip install pythermalcomfort"
        ) from exc

    if params is None:
        params = PMVParameters()

    std = params.standard.upper()
    try:
        if std == "ASHRAE":
            from pythermalcomfort.models import pmv_ppd_ashrae as _pmv_fn
        else:
            from pythermalcomfort.models import pmv_ppd_iso as _pmv_fn
    except ImportError as exc:
        raise ImportError(
            "pythermalcomfort is required for PMV calculations. "
            "Install it with: pip install pythermalcomfort"
        ) from exc

    ta = env.air_temperature()          # (n_points, n_hours)
    tr = env.mrt()
    rh = env.relative_humidity()

    try:
        v = env.air_velocity()
    except KeyError:
        v = np.full_like(ta, default_velocity)

    n_points, n_hours = ta.shape

    vr = v_relative(v=v.ravel(), met=params.met)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _pmv_fn(
            tdb=ta.ravel(),
            tr=tr.ravel(),
            vr=vr,
            rh=rh.ravel(),
            met=params.met,
            clo=params.clo,
            wme=params.wme,
            limit_inputs=params.limit_inputs,
            round_output=False,
        )

    pmv_values = np.asarray(result.pmv, dtype=float).reshape(n_points, n_hours)

    lo, hi = _COMFORT_BAND
    discrete = np.where(pmv_values < lo, -1, np.where(pmv_values > hi, 1, 0)).astype(
        np.int8
    )

    return ComfortResult(
        values=pmv_values,
        discrete=discrete,
        model="PMV",
        comfort_band=_COMFORT_BAND,
        env=env,
    )
