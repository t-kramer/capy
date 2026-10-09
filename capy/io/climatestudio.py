"""ClimateStudio I/O adapter.

To be implemented once the ClimateStudio export format is confirmed.
ClimateStudio is a Solemma product that runs thermal and daylighting simulations
inside Rhino/Grasshopper.
"""
from __future__ import annotations

from capy.core.environment import ThermalEnvironment


def from_climatestudio(export_path: str) -> ThermalEnvironment:
    """Load a ThermalEnvironment from a ClimateStudio export.

    Args:
        export_path: Path to the ClimateStudio export file or folder.

    Returns:
        ThermalEnvironment.

    Note:
        Not yet implemented — export format to be confirmed.
        Use capy.io.csv.from_csv_folder() or capy.io.honeybee.from_honeybee_grid()
        in the meantime.
    """
    raise NotImplementedError(
        "ClimateStudio I/O is not yet implemented. "
        "Export format needs to be confirmed before implementation."
    )
