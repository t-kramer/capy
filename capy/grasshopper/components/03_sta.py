"""capy | 03 | Spatial Thermal Autonomy (sTA)
Paste this script into a GHPython component.

Computes the fraction of occupied hours where ≥ tau of the floor area
is thermally comfortable without mechanical cooling/heating.

Inputs:
    comfort     : capy ComfortResult (from component 02)
    Tau         : float — comfort area threshold (default 0.8 = 80 %)
    Schedule    : capy OccupancySchedule  (optional; default = all hours)
    AreaWeighted: bool — weight by sensor area (default True)

Outputs:
    annual_sta  : float — annual sTA [%]
    hourly_sta  : list  — sTA per hour (8760 values) [%]
    point_values: list  — per-point annual comfort fraction [0–1]
    colored_mesh: LBMesh3D for GH viewport (requires ladybug-geometry)
"""

from capy.metrics.autonomy import spatial_thermal_autonomy
from capy.grasshopper.convert import to_colored_mesh, sensor_values_to_list

if comfort is None:
    raise ValueError("Connect a capy ComfortResult to comfort.")

_tau = float(Tau) if Tau is not None else 0.8
_weighted = bool(AreaWeighted) if AreaWeighted is not None else True

sta = spatial_thermal_autonomy(
    comfort,
    tau=_tau,
    schedule=Schedule,
    area_weighted=_weighted,
)

annual_sta = sta.annual_pct
hourly_sta = (sta.hourly_array * 100).tolist()
point_values = sensor_values_to_list(sta.comfort_hours_matrix, annual_stat="mean")

try:
    colored_mesh = to_colored_mesh(comfort.env.grid, [v for v in point_values])
except Exception:
    colored_mesh = None

print(f"capy | sTA = {annual_sta:.1f}%  (tau={_tau})")
