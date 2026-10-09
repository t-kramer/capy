"""capy | 08 | Thermal Niches & Context Score (Alliesthesia)
Paste this script into a GHPython component.

Two alliesthesia-based metrics:
  Thermal Niches  — fraction of occupied hours with co-existing warm/cool zones
  Context Score   — per-point per-hour +1/0/-1 classification (seasonal context)

Inputs:
    env             : capy ThermalEnvironment (from component 00 or 01)
    comfort         : capy ComfortResult (required for context_score)
    EPW             : Ladybug EPW or path string (required for context_score)
    Schedule        : capy OccupancySchedule  (optional)
    DeltaTMin       : float — min spatial range for 'varied' environment [°C] (default 2)
    DeltaTMax       : float — max spatial range before discomfort [°C] (default 4)
    NicheDeviation  : float — deviation from mean to count as niche [°C] (default 1)

Outputs:
    niche_fraction     : float — % of occupied hours with ≥2 thermal niches [0–1]
    discomfort_fraction: float — % of occupied hours with excessive heterogeneity [0–1]
    uniform_fraction   : float — % of occupied hours with insufficient variety [0–1]
    beneficial_fraction: float — % of all point-hours classified as +1 (beneficial) [0–1]
    point_context_mean : list  — mean context score per point [-1, +1]
"""

from capy.metrics.alliesthesia import thermal_niches

if env is None:
    raise ValueError("Connect a capy ThermalEnvironment to env.")

_dt_min = float(DeltaTMin) if DeltaTMin is not None else 2.0
_dt_max = float(DeltaTMax) if DeltaTMax is not None else 4.0
_dev = float(NicheDeviation) if NicheDeviation is not None else 1.0

niches = thermal_niches(
    env,
    schedule=Schedule,
    delta_t_min=_dt_min,
    delta_t_max=_dt_max,
    niche_deviation=_dev,
)

niche_fraction = niches.niche_fraction
discomfort_fraction = niches.discomfort_fraction
uniform_fraction = niches.uniform_fraction

# Context score — optional, requires comfort result + EPW
beneficial_fraction = None
point_context_mean = None

if comfort is not None and EPW is not None:
    from capy.metrics.alliesthesia import context_score
    _epw_path = str(EPW._file_path) if hasattr(EPW, "_file_path") else str(EPW)
    ctx = context_score(env, comfort, epw_path=_epw_path, schedule=Schedule)
    beneficial_fraction = ctx.beneficial_fraction
    point_context_mean = ctx.scores.mean(axis=1).tolist()

print(
    f"capy | Niches={niche_fraction*100:.1f}%  "
    f"Discomfort={discomfort_fraction*100:.1f}%  "
    f"Uniform={uniform_fraction*100:.1f}%"
    + (f"  Beneficial={beneficial_fraction*100:.1f}%" if beneficial_fraction is not None else "")
)
