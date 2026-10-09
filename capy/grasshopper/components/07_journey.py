"""capy | 07 | Thermal Journey
Paste this script into a GHPython component.

Maps thermal conditions along a circulation path defined by ordered points.
Requires a Divide Curve or Point List from Grasshopper.

Inputs:
    env             : capy ThermalEnvironment (from component 00 or 01)
    PathPoints      : list of Rhino Point3d — ordered path waypoints
    comfort         : capy ComfortResult (optional — for journey comfort score)
    Schedule        : capy OccupancySchedule  (optional)
    Variable        : str — variable to map (default 'operative_temperature')
    Method          : str — 'nearest' or 'idw' (default 'nearest')
    NeutralThreshold: float — ΔT below which step is neutral [°C] (default 0.5)

Outputs:
    annual_mean     : list  — annual mean value at each path step [°C]
    annual_std      : list  — annual std at each path step [°C]
    transitions     : list  — transition type per step: +1 warm / 0 neutral / -1 cool
    comfort_score   : float — mean comfort fraction along path during occupied hours
    journey         : JourneyResult — full result for further analysis
"""

from capy.metrics.journey import thermal_journey
from capy.grasshopper.convert import points_from_rhino

if env is None:
    raise ValueError("Connect a capy ThermalEnvironment to env.")
if PathPoints is None or len(PathPoints) == 0:
    raise ValueError("Connect an ordered list of Rhino Point3d to PathPoints.")

_pts = points_from_rhino(PathPoints)
_var = str(Variable) if Variable is not None else "operative_temperature"
_method = str(Method).lower() if Method is not None else "nearest"
_threshold = float(NeutralThreshold) if NeutralThreshold is not None else 0.5

journey = thermal_journey(
    env,
    path_points=_pts,
    variable=_var,
    method=_method,
    schedule=Schedule,
    neutral_threshold=_threshold,
    comfort_result=comfort,
)

annual_mean = journey.profile.mean(axis=1).tolist()
annual_std = journey.profile.std(axis=1).tolist()
transitions = journey.transitions.tolist()
comfort_score = journey.journey_comfort_score

print(
    f"capy | Journey: {len(annual_mean)} steps  "
    f"ΔT={max(annual_mean)-min(annual_mean):.1f}°C  "
    f"comfort={comfort_score:.1%}" if not isinstance(comfort_score, float) or
    not __import__('math').isnan(comfort_score)
    else f"capy | Journey: {len(annual_mean)} steps  ΔT={max(annual_mean)-min(annual_mean):.1f}°C"
)
