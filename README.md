
<p align="left">
  <img src="assets/img/logo-v1.png" alt="capy logo" width="200">
</p>

**Spatial thermal comfort analysis for biophilic building design.**

Named after the capybara, an animal that navigates between warm and cool environments as a daily pleasure-seeking behaviour, a natural demonstration of spatial thermal variety and thermal alliesthesia.

Part of the *Decoding Delight* project (Autodesk Faculty Research 2026, UC Berkeley Center for the Built Environment).

---

## Install

```bash
pip install capy
```

Or from source:

```bash
git clone https://github.com/t-kramer/capy
cd capy
pip install -e ".[dev]"
```

## Quick start

```python
from capy.io.csv import from_csv_folder
from capy.comfort.pmv import calculate_pmv
from capy.metrics.autonomy import spatial_thermal_autonomy
from capy.metrics.heterogeneity import thi_area, beneficial_heterogeneity_index
from capy.core.schedule import OccupancySchedule

# Load simulation results (Honeybee CSV format)
env = from_csv_folder('path/to/sim/results/')

# Occupancy schedule
schedule = OccupancySchedule.work_hours()

# Comfort model
comfort = calculate_pmv(env)

# Spatial Thermal Autonomy
sta = spatial_thermal_autonomy(comfort, tau=0.8, schedule=schedule)
print(f'sTA = {sta.annual_pct:.1f}%')

# Thermal Heterogeneity
thi = thi_area(env, schedule=schedule)
print(f'THI_area = {thi.thi:.2f} °C')

# Beneficial Heterogeneity (2–4°C band)
bhi = beneficial_heterogeneity_index(env, schedule=schedule)
print(f'Beneficial hours: {bhi.bhi_beneficial:.1f}%')
```

## Metrics

### Comfort models
| Model | Function |
|---|---|
| PMV (Fanger) | `capy.comfort.pmv.calculate_pmv()` |
| Adaptive ASHRAE 55 | `capy.comfort.adaptive.calculate_adaptive()` *(coming)* |
| Empirical (ASHRAE GTCDB II) | `capy.comfort.empirical.calculate_empirical()` |

### Spatial metrics
| Metric | Function |
|---|---|
| Spatial Thermal Autonomy (sTA) | `capy.metrics.autonomy.spatial_thermal_autonomy()` |
| THI area | `capy.metrics.heterogeneity.thi_area()` |
| THI point | `capy.metrics.heterogeneity.thi_point()` |
| Beneficial Heterogeneity Index | `capy.metrics.heterogeneity.beneficial_heterogeneity_index()` |

### Biophilic pleasure metrics
| Metric | Function |
|---|---|
| Natural air movement | `capy.metrics.air_movement.assess_air_movement()` |
| Diurnal amplitude | `capy.metrics.diurnal.diurnal_amplitude()` *(coming)* |
| Thermal niches | `capy.metrics.alliesthesia.thermal_niches()` *(coming)* |
| Thermal journey | `capy.metrics.journey.thermal_journey()` *(coming)* |

## Grasshopper integration

`capy` is designed to work from GHPython components. Copy the component scripts from `capy/grasshopper/components/` into GHPython nodes in your definition.

```python
# Inside a GHPython component
import capy
from capy.io.csv import from_csv_folder
from capy.comfort.pmv import calculate_pmv
from capy.metrics.autonomy import spatial_thermal_autonomy

env = from_csv_folder(csv_folder_path)
comfort = calculate_pmv(env)
sta = spatial_thermal_autonomy(comfort, tau=tau)

# Outputs
a_sta = sta.annual_pct
hourly_sta = sta.hourly_array.tolist()
```

## License

MIT
