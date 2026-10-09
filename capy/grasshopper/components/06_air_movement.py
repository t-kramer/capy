"""capy | 06 | Natural Air Movement
Paste this script into a GHPython component.

Assesses natural air movement pleasure potential. Three modes:
  Mode 1 — Uniform velocity   : supply a single float for Velocity
  Mode 2 — Point velocity data: supply a (n_points × 8760) DataTree
  Mode 3 — Geometry proxy     : supply Window objects via Windows list
  Mode 4 — Unavailable        : no inputs → returns NaN with guidance

Inputs:
    env             : capy ThermalEnvironment (from component 00 or 01)
    Velocity        : float (m/s) OR DataTree (n_points × 8760)  (optional)
    Windows         : list of capy Window objects               (optional)
    Schedule        : capy OccupancySchedule                    (optional)
    TThreshold      : float — T_op threshold for air movement pleasure [°C] (default 24)
    VMin            : float — minimum pleasant velocity [m/s] (default 0.2)
    VMax            : float — maximum pleasant velocity [m/s] (default 0.8)

Outputs:
    data_mode          : str  — 'uniform', 'point_data', 'geometry_proxy', or 'unavailable'
    pleasure_fraction  : float — fraction of (occupied) point-hours in pleasure zone [0–1]
    nv_potential       : float — natural ventilation potential score [0–1]
                         (geometry_proxy mode only; None otherwise)
    mean_velocity      : list  — mean hourly air velocity per point [m/s]
                         (velocity modes only; None in geometry_proxy mode)
"""

from capy.metrics.air_movement import assess_air_movement, Window
from capy.grasshopper.convert import datatree_to_array

if env is None:
    raise ValueError("Connect a capy ThermalEnvironment to env.")

_v_threshold = float(TThreshold) if TThreshold is not None else 24.0
_v_min = float(VMin) if VMin is not None else 0.2
_v_max = float(VMax) if VMax is not None else 0.8

# Resolve velocity input
_velocity = None
if Velocity is not None:
    try:
        # DataTree path
        _velocity = datatree_to_array(Velocity)
    except Exception:
        # Scalar float
        _velocity = float(Velocity)

result = assess_air_movement(
    env,
    velocity=_velocity,
    windows=Windows if Windows else None,
    T_threshold=_v_threshold,
    v_min=_v_min,
    v_max=_v_max,
    schedule=Schedule,
)

data_mode = result.data_mode
pleasure_fraction = result.pleasure_zone_fraction
nv_potential = result.nv_potential
mean_velocity = (
    result.mean_velocity.mean(axis=1).tolist()
    if result.mean_velocity is not None
    else None
)

print(f"capy | Air movement mode: {data_mode}  pleasure={pleasure_fraction:.1%}")
