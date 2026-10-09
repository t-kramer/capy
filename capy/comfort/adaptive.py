"""Adaptive comfort models: ASHRAE 55 and EN 15251 / EN 16798-1.

Evidence base:
  ASHRAE 55-2023 Section 5.4:
    T_comf = 0.31 × T_prevailing + 17.8 [°C]
    90 % acceptability: ±2.5 °C; 80 %: ±3.5 °C

  EN 16798-1 (formerly EN 15251), Appendix B:
    T_comf = 0.33 × T_prevailing + 18.8 [°C]
    Class I: ±2 °C; Class II: ±3 °C; Class III: ±4 °C

  Applicability: T_prevailing must fall within 10–33.5 °C (ASHRAE 55)
  or 10–30 °C (EN 16798-1); outside this range the model is not valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from capy.comfort.base import ComfortResult
from capy.core.environment import ThermalEnvironment
from capy.io.epw import load_dry_bulb as _load_dry_bulb
from capy.io.epw import running_mean as _running_mean_epw

# ---------------------------------------------------------------------------
# Standards configuration
# ---------------------------------------------------------------------------

_STANDARDS = {
    "ashrae55": {
        "bands": {"90%": 2.5, "80%": 3.5},
        "default_band": "90%",
        "valid_range": (10.0, 33.5),
        "band_attr": {
            "90%": "acceptability_90",
            "80%": "acceptability_80",
        },
    },
    "en15251": {
        "bands": {"I": 2.0, "II": 3.0, "III": 4.0},
        "default_band": "II",
        "valid_range": (10.0, 30.0),
        "band_attr": {
            "I": "acceptability_cat_i",
            "II": "acceptability_cat_ii",
            "III": "acceptability_cat_iii",
        },
    },
}
# Alias
_STANDARDS["en16798"] = _STANDARDS["en15251"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveResult(ComfortResult):
    """Extended ComfortResult for adaptive models.

    Additional attributes:
        t_comf:             (n_hours,) neutral/comfort temperature per hour [°C]
        t_prevailing:       (n_hours,) prevailing mean outdoor temperature [°C]
        acceptance_band:    half-width of comfort band [°C]
        applicability_mask: (n_hours,) bool — True where model is valid
                            (T_prevailing within allowable range)
    """
    t_comf: np.ndarray
    t_prevailing: np.ndarray
    acceptance_band: float
    applicability_mask: np.ndarray

    def __repr__(self) -> str:  # type: ignore[override]
        valid_pct = self.applicability_mask.mean() * 100
        return (
            f"AdaptiveResult(model='{self.model}', band=±{self.acceptance_band}°C, "
            f"valid_hours={valid_pct:.0f}%)"
        )


def calculate_adaptive(
    env: ThermalEnvironment,
    standard: str = "ashrae55",
    epw_path: Optional[str] = None,
    prevailing_mean_outdoor_temp: Optional[float] = None,
    acceptance_band: Optional[str] = None,
    alpha: float = 0.8,
    n_days: int = 7,
    default_velocity: float = 0.1,
) -> AdaptiveResult:
    """Calculate adaptive comfort for all sensor points over all hours.

    Supports ASHRAE 55-2023 (Section 5.4) and EN 16798-1 (formerly EN 15251)
    adaptive comfort models via pythermalcomfort. The prevailing outdoor running
    mean temperature is derived from the EPW weather file using exponential
    weighting.

    Parameters
    ----------
    env:
        ThermalEnvironment with operative temperature available.
    standard:
        ``'ashrae55'`` (default) or ``'en15251'`` / ``'en16798'``.
    epw_path:
        Path to EPW weather file. Overrides ``env.epw_path``. Required unless
        *prevailing_mean_outdoor_temp* is supplied.
    prevailing_mean_outdoor_temp:
        Override the outdoor running mean with a single annual scalar value
        (°C). When supplied, the EPW is not read. Useful for quick sensitivity
        analysis.
    acceptance_band:
        Band label (e.g. ``'90%'`` or ``'80%'`` for ASHRAE 55; ``'I'``, ``'II'``,
        ``'III'`` for EN 15251). Uses the standard's default if not provided.
    alpha:
        Exponential weighting factor for the running mean (default 0.8,
        per ASHRAE 55).
    n_days:
        Number of preceding days used in the running mean (default 7).
    default_velocity:
        Fallback air speed in m/s when not present in env.variables (default 0.1).

    Returns
    -------
    AdaptiveResult
        ComfortResult subclass with additional adaptive model fields.

    Raises
    ------
    ValueError
        If neither *epw_path* nor *prevailing_mean_outdoor_temp* is provided,
        or if *standard* is not recognised.
    """
    try:
        if standard.lower().replace("-", "").replace(" ", "") == "ashrae55":
            from pythermalcomfort.models import adaptive_ashrae as _ptc_adaptive
        else:
            from pythermalcomfort.models import adaptive_en as _ptc_adaptive
    except ImportError as exc:
        raise ImportError(
            "pythermalcomfort is required for adaptive comfort calculations. "
            "Install it with: pip install pythermalcomfort"
        ) from exc

    std_key = standard.lower().replace("-", "").replace(" ", "")
    if std_key not in _STANDARDS:
        raise ValueError(
            f"Unknown standard '{standard}'. "
            f"Choose from: {list(_STANDARDS)}"
        )
    cfg = _STANDARDS[std_key]

    if acceptance_band is None:
        acceptance_band = cfg["default_band"]
    if acceptance_band not in cfg["bands"]:
        raise ValueError(
            f"Band '{acceptance_band}' not valid for {standard}. "
            f"Choose from: {list(cfg['bands'])}"
        )
    half_width = cfg["bands"][acceptance_band]
    band_attr = cfg["band_attr"][acceptance_band]

    n_hours = len(env.time_index)

    # ── Prevailing mean outdoor temperature ─────────────────────────────
    if prevailing_mean_outdoor_temp is not None:
        t_prev = np.full(n_hours, float(prevailing_mean_outdoor_temp))
    else:
        resolved_epw = epw_path or env.epw_path
        if resolved_epw is None:
            raise ValueError(
                "calculate_adaptive() needs an EPW file. Pass epw_path= or "
                "set env.epw_path, or supply prevailing_mean_outdoor_temp=."
            )
        t_outdoor = _load_dry_bulb(resolved_epw)
        t_prev = _running_mean_epw(t_outdoor, alpha=alpha, n_days=n_days)

    # ── Applicability ────────────────────────────────────────────────────
    valid_lo, valid_hi = cfg["valid_range"]
    applicability = (t_prev >= valid_lo) & (t_prev <= valid_hi)

    # ── Environment arrays ───────────────────────────────────────────────
    ta = env.air_temperature()
    tr = env.mrt()
    try:
        v = env.air_velocity()
    except KeyError:
        v = np.full_like(ta, default_velocity)

    n_points = env.grid.n_points
    t_op = env.operative_temperature()

    # ── Call pythermalcomfort per point ──────────────────────────────────
    t_comf: Optional[np.ndarray] = None
    discrete = np.zeros((n_points, n_hours), dtype=np.int8)

    for i in range(n_points):
        r = _ptc_adaptive(
            tdb=ta[i], tr=tr[i], t_running_mean=t_prev, v=v[i], limit_inputs=True
        )
        if t_comf is None:
            t_comf = np.asarray(r.tmp_cmf, dtype=float)

        acc = np.asarray(getattr(r, band_attr), dtype=bool)
        dev = t_op[i] - t_comf
        discrete[i] = np.where(acc, 0, np.where(dev < 0, -1, 1))

    # Outside applicability range → unclassified
    discrete[:, ~applicability] = 0

    values = t_op - t_comf[None, :]

    return AdaptiveResult(
        values=values,
        discrete=discrete,
        model=f"adaptive_{std_key}_{acceptance_band}",
        comfort_band=(-half_width, half_width),
        env=env,
        t_comf=t_comf,
        t_prevailing=t_prev,
        acceptance_band=half_width,
        applicability_mask=applicability,
    )

