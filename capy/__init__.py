"""capy — Spatial thermal comfort analysis for biophilic building design.

Named after the capybara, which navigates between warm and cool thermal
environments as a daily pleasure-seeking behaviour — a living demonstration
of spatial thermal variety and alliesthesia.

Quick start::

    from capy import ThermalEnvironment, SpatialGrid, OccupancySchedule
    from capy.io.csv import from_csv_folder
    from capy.comfort.pmv import calculate_pmv
    from capy.metrics.autonomy import spatial_thermal_autonomy
    from capy.metrics.heterogeneity import thi_area, beneficial_heterogeneity_index

    env = from_csv_folder('path/to/sim/results/')
    comfort = calculate_pmv(env)
    sta = spatial_thermal_autonomy(comfort)
    print(f'sTA = {sta.annual_pct:.1f}%')
"""

__version__ = "0.1.0"

from capy import grasshopper
from capy.comfort import adaptive, empirical, pmv
from capy.comfort.base import ComfortResult
from capy.core.environment import ThermalEnvironment
from capy.core.grid import SpatialGrid
from capy.core.schedule import OccupancySchedule
from capy.io import csv
from capy import pcs
from capy.metrics import air_movement, alliesthesia, autonomy, diurnal, heterogeneity, journey
from capy.viz import heatmap, spatial
from capy.viz import journey as viz_journey

__all__ = [
    "SpatialGrid",
    "ThermalEnvironment",
    "OccupancySchedule",
    "ComfortResult",
    "__version__",
]
