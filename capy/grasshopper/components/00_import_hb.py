"""capy | 00 | Import from Honeybee
Paste this script into a GHPython component.

Inputs:
    SensorGrid      : Honeybee SensorGrid object (from HB Model or HB Grid component)
    AirTemperature  : HourlyContinuousCollection — indoor air temperature [°C]
    MRT             : HourlyContinuousCollection — mean radiant temperature [°C]
    RelHumidity     : HourlyContinuousCollection — relative humidity [%]
    AirVelocity     : HourlyContinuousCollection — air velocity [m/s]  (optional)
    EPW             : Ladybug EPW object or path string  (optional but recommended)
    UnitArea        : float — floor area per sensor in m² (default: 1.0)

Outputs:
    env             : capy ThermalEnvironment — pass to comfort components
"""

import capy
from capy.io.honeybee import from_honeybee_grid

if SensorGrid is None:
    raise ValueError("Connect a Honeybee SensorGrid to SensorGrid input.")
if AirTemperature is None or MRT is None:
    raise ValueError("AirTemperature and MRT inputs are required.")

_unit_area = float(UnitArea) if UnitArea is not None else 1.0

# Resolve EPW path
_epw_path = None
if EPW is not None:
    _epw_path = str(EPW._file_path) if hasattr(EPW, "_file_path") else str(EPW)

# Build data_collections dict
_collections = {
    "air_temperature": AirTemperature,
    "mean_radiant_temperature": MRT,
    "relative_humidity": RelHumidity,
}
if AirVelocity is not None:
    _collections["air_velocity"] = AirVelocity

env = from_honeybee_grid(
    sensor_grid=SensorGrid,
    data_collections=_collections,
    epw_path=_epw_path,
    unit_area=_unit_area,
)

print(f"capy | Loaded: {env}")
