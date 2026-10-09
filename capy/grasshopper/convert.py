"""Rhino / Grasshopper ↔ numpy conversion utilities.

Import this module inside a GHPython component after installing capy:
    import capy.grasshopper.convert as gh_convert

These helpers bridge Rhino geometry types and DataTree structures with
capy's numpy-based data model.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def points_from_rhino(point3d_list) -> np.ndarray:
    """Convert a list of Rhino Point3d to a (n, 3) numpy array.

    Args:
        point3d_list: List or GH tree of Rhino.Geometry.Point3d objects.

    Returns:
        (n, 3) float array of x, y, z coordinates.
    """
    return np.array([[p.X, p.Y, p.Z] for p in point3d_list], dtype=float)


def points_to_rhino(array: np.ndarray):
    """Convert a (n, 3) numpy array to a list of Rhino Point3d.

    Args:
        array: (n, 3) float array.

    Returns:
        List of Rhino.Geometry.Point3d objects.
    """
    try:
        import Rhino.Geometry as rg
    except ImportError as exc:
        raise ImportError(
            "Rhino.Geometry is only available inside Rhino/Grasshopper."
        ) from exc
    return [rg.Point3d(float(row[0]), float(row[1]), float(row[2])) for row in array]


def datatree_to_array(gh_datatree, dtype=float) -> np.ndarray:
    """Flatten a GH DataTree of numbers to a (n_branches, branch_length) array.

    Args:
        gh_datatree: Grasshopper DataTree.
        dtype: numpy dtype for the output array.

    Returns:
        2D numpy array where rows = branches, columns = items.
    """
    rows = [list(gh_datatree.Branch(i)) for i in range(gh_datatree.BranchCount)]
    return np.array(rows, dtype=dtype)


def array_to_datatree(array: np.ndarray):
    """Convert a 2D numpy array to a GH DataTree.

    Row i → branch {i}.

    Args:
        array: 2D numpy array.

    Returns:
        Grasshopper.DataTree[object].
    """
    try:
        import Grasshopper as gh
        from Grasshopper.Kernel.Data import GH_Path
    except ImportError as exc:
        raise ImportError(
            "Grasshopper is only available inside Rhino/Grasshopper."
        ) from exc

    tree = gh.DataTree[object]()
    for i, row in enumerate(array):
        path = GH_Path(i)
        tree.AddRange([float(v) for v in row], path)
    return tree


def to_colored_mesh(grid, values: np.ndarray, legend_params=None):
    """Create a colored Ladybug mesh for Grasshopper viewport display.

    Maps a ``(n_points,)`` array of scalar values onto the sensor grid using
    the Ladybug ``VisualizationData`` / ``LegendParameters`` system. Each
    sensor point becomes a small quad face; the mesh is returned as an
    ``LBMesh3D`` with vertex colours for direct GH viewport rendering.

    Parameters
    ----------
    grid:
        capy ``SpatialGrid`` with ``points`` attribute (model coordinates in m).
    values:
        ``(n_points,)`` scalar array (e.g. annual mean operative temperature).
    legend_params:
        Optional ``ladybug.legend.LegendParameters`` object to control colour
        scale, title, and number of segments.

    Returns
    -------
    ``ladybug_geometry.mesh3d.Mesh3D``
        Coloured mesh ready to connect to a Grasshopper Mesh parameter.

    Raises
    ------
    ImportError
        If ``ladybug-geometry`` is not installed.

    Note
    ----
    Requires ``ladybug-geometry`` (``pip install ladybug-geometry``), available
    in Rhino 8 / GH2 via the LBT package. The sensor height in
    ``grid.sensor_height`` is added to each point's Z coordinate.

    When running outside Rhino and needing a quick visual, use
    ``capy.viz.spatial.plot_floor_map()`` instead.
    """
    try:
        from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
        from ladybug_geometry.geometry3d.face import Face3D
        from ladybug_geometry.geometry3d.mesh import Mesh3D
    except ImportError as exc:
        raise ImportError(
            "to_colored_mesh() requires ladybug-geometry. "
            "Install with: pip install ladybug-geometry\n"
            "Alternatively, use capy.viz.spatial.plot_floor_map() "
            "for a matplotlib-based 2D map."
        ) from exc

    values = np.asarray(values, dtype=float)
    pts = grid.points                        # (n, 3)
    n = pts.shape[0]

    if values.shape[0] != n:
        raise ValueError(
            f"values length {values.shape[0]} != grid n_points {n}"
        )

    zh = getattr(grid, "sensor_height", 0.0)

    # Represent each sensor as a small quad (half-size of typical cell)
    # Default cell half-size: sqrt(mean_area) / 2
    mean_area = float(grid.areas.mean()) if hasattr(grid, "areas") else 1.0
    half = np.sqrt(mean_area) / 2.0

    faces = []
    vertices = []
    v_idx = 0
    for i in range(n):
        x, y, z = pts[i, 0], pts[i, 1], pts[i, 2] + zh
        # Quad: bottom-left → bottom-right → top-right → top-left
        v0 = Point3D(x - half, y - half, z)
        v1 = Point3D(x + half, y - half, z)
        v2 = Point3D(x + half, y + half, z)
        v3 = Point3D(x - half, y + half, z)
        vertices.extend([v0, v1, v2, v3])
        faces.append((v_idx, v_idx + 1, v_idx + 2, v_idx + 3))
        v_idx += 4

    mesh = Mesh3D(vertices, faces)

    # Apply colour mapping via LegendParameters
    try:
        from ladybug.legend import LegendParameters
        from ladybug.graphic import GraphicContainer

        if legend_params is None:
            legend_params = LegendParameters(
                min_point=None,
                max_point=None,
            )

        container = GraphicContainer(
            values,
            min_point=mesh.min,
            max_point=mesh.max,
            legend_parameters=legend_params,
        )
        mesh.colors = container.value_colors
    except ImportError:
        # ladybug-core not available: return uncoloured mesh
        pass

    return mesh


def sensor_values_to_list(values: np.ndarray, annual_stat: str = "mean") -> list:
    """Reduce a (n_points, 8760) array to a flat Python list for GH outputs.

    Parameters
    ----------
    values:
        2-D array of shape ``(n_points, n_hours)``.
    annual_stat:
        Aggregation to apply over the time axis — ``'mean'``, ``'median'``,
        ``'max'``, or ``'min'``.

    Returns
    -------
    list of ``float`` of length ``n_points``.
    """
    stat_funcs = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
    }
    if annual_stat not in stat_funcs:
        raise ValueError(
            f"annual_stat must be one of {list(stat_funcs)}, got '{annual_stat}'"
        )
    return stat_funcs[annual_stat](values, axis=1).tolist()
