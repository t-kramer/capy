"""Empirical comfort model based on ASHRAE Global Thermal Comfort Database II."""
from __future__ import annotations

from capy.comfort.base import ComfortResult
from capy.core.environment import ThermalEnvironment

# Evidence-based comfort range from ASHRAE GTCDB II:
# 10th–90th percentile of operative temperatures where occupants
# expressed 'No change' thermal preference.
DEFAULT_LOWER = 21.0  # °C
DEFAULT_UPPER = 28.0  # °C


def calculate_empirical(
    env: ThermalEnvironment,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> ComfortResult:
    """Classify comfort using an operative temperature band.

    A simple but robust model: any operative temperature within [lower, upper]
    is considered comfortable. Derived from the 10th–90th percentile of the
    ASHRAE Global Thermal Comfort Database II for occupants expressing
    'No change' thermal preference (Földváry Ličina et al., 2018).

    Args:
        env: ThermalEnvironment with operative_temperature.
        lower: Lower bound of comfortable operative temperature (°C).
        upper: Upper bound of comfortable operative temperature (°C).

    Returns:
        ComfortResult.
    """
    from capy.comfort.base import ComfortResult

    T_op = env.operative_temperature()  # (n_points, n_hours)
    comfort_band = (lower, upper)

    discrete = (
        (T_op < lower).astype(int) * -1
        + (T_op > upper).astype(int) * 1
    ).astype("int8")

    return ComfortResult(
        values=T_op,
        discrete=discrete,
        model="Empirical",
        comfort_band=comfort_band,
        env=env,
    )
