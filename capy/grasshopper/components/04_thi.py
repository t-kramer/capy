"""capy | 04 | Thermal Heterogeneity Index (THI) + BHI
Paste this script into a GHPython component.

Computes spatial thermal variety metrics:
  THI_area  — median hourly spatial range across the floor plate [°C]
  THI_point — median annual thermal range per point [°C]
  BHI       — fraction of hours in the beneficial 2–4 °C heterogeneity band [%]

Inputs:
    env         : capy ThermalEnvironment (from component 00 or 01)
    Schedule    : capy OccupancySchedule  (optional)
    BandMin     : float — lower bound of beneficial heterogeneity band [°C] (default 2)
    BandMax     : float — upper bound of beneficial heterogeneity band [°C] (default 4)

Outputs:
    thi_area    : float — THI_area [°C]
    thi_point   : float — THI_point [°C]
    bhi_beneficial : float — % of hours in beneficial band
    bhi_discomfort : float — % of hours above band (spatial discomfort)
    bhi_uniform    : float — % of hours below band (thermally uniform)
"""

from capy.metrics.heterogeneity import thi_area, thi_point, beneficial_heterogeneity_index

if env is None:
    raise ValueError("Connect a capy ThermalEnvironment to env.")

_band_min = float(BandMin) if BandMin is not None else 2.0
_band_max = float(BandMax) if BandMax is not None else 4.0

_thi_area = thi_area(env, schedule=Schedule)
_thi_point = thi_point(env, schedule=Schedule)
_bhi = beneficial_heterogeneity_index(
    env, schedule=Schedule, band_min=_band_min, band_max=_band_max
)

thi_area_val = _thi_area.thi
thi_point_val = _thi_point.thi
bhi_beneficial = _bhi.bhi_beneficial
bhi_discomfort = _bhi.bhi_discomfort
bhi_uniform = _bhi.bhi_uniform

print(
    f"capy | THI_area={thi_area_val:.2f}°C  THI_point={thi_point_val:.2f}°C"
    f"  BHI_beneficial={bhi_beneficial:.1f}%"
)
