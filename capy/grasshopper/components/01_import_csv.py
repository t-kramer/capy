"""capy | 01 | Import from CSV folder
Paste this script into a GHPython component.

Compatible with comfortSIM and Honeybee CSV exports where each variable
is stored as a separate CSV file (rows = sensor points, columns = hours).

Inputs:
    FolderPath  : str — path to folder containing CSV files
                  e.g. 'C:/sim_results/grid_01/'
    EPW         : Ladybug EPW object or path string  (optional)

Outputs:
    env         : capy ThermalEnvironment — pass to comfort components
"""

from capy.io.csv import from_csv_folder

if FolderPath is None:
    raise ValueError("Connect a folder path string to FolderPath.")

_epw_path = None
if EPW is not None:
    _epw_path = str(EPW._file_path) if hasattr(EPW, "_file_path") else str(EPW)

env = from_csv_folder(str(FolderPath), epw_path=_epw_path)

print(f"capy | Loaded: {env}")
