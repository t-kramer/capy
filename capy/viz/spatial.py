"""Static 2D floor plan visualizations for reports and publications.

For live Grasshopper viewport visualization, use the colored mesh output
from capy.grasshopper.components instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from capy.viz._style import apply_chart_style, apply_top_right_colorbar


def plot_floor_map(
    source=None,
    variable: str = "operative_temperature",
    hour: Optional[int] = None,
    annual_stat: str = "mean",
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    point_size: float = 80.0,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    rotate: bool = True,
    values: Optional[np.ndarray] = None,
    grid=None,
    label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """2D scatter plot of sensor points coloured by thermal variable.

    When the grid carries a non-zero ``north_angle`` and *rotate* is ``True``
    (the default), coordinates are rotated so that geographic North aligns with
    the top of the plot and a North arrow is added.

    Parameters
    ----------
    source:
        ``ThermalEnvironment`` or ``ComfortResult``. Points are taken from
        ``source.grid`` (``ThermalEnvironment``) or ``source.env.grid``
        (``ComfortResult``). Mutually exclusive with *values* + *grid*.
    variable:
        Variable to colour by. When *source* is a ``ComfortResult`` and
        *variable* is ``'comfort_fraction'``, per-point comfort fraction is used.
    hour:
        Specific hour index (0–8759) to plot. When ``None``, uses *annual_stat*.
    annual_stat:
        Annual aggregation when *hour* is ``None`` — ``'mean'``, ``'median'``,
        ``'max'``, or ``'min'``.
    figsize:
        Matplotlib figure size (width, height) in inches.
    save_path:
        Optional path to save the figure (PNG/SVG/PDF).
    cmap:
        Colormap name (default ``'RdYlBu_r'``).
    point_size:
        Scatter point size in points² (default 80).
    show_colorbar:
        Whether to add a colorbar (default ``True``).
    title:
        Figure title. Auto-generated when ``None``.
    rotate:
        Apply the grid's ``north_angle`` rotation (default ``True``). Set to
        ``False`` to plot raw grid coordinates regardless of ``north_angle``.
    values:
        ``(n_points,)`` numpy array of pre-computed per-point values. When
        supplied, *grid* must also be provided and *source* is ignored.
    grid:
        ``SpatialGrid`` used to locate sensor points when *values* is supplied.
    label:
        Colorbar label used when *values* is supplied directly.
    vmin:
        Colorbar minimum (overrides data range). Most useful in raw-array mode
        — e.g. ``vmin=-1, vmax=1`` for phase alignment scores.
    vmax:
        Colorbar maximum (overrides data range).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

    # Raw array mode: values + grid provided directly
    if values is not None:
        if grid is None:
            raise ValueError("grid is required when values is provided.")
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.shape[0] != grid.n_points:
            raise ValueError(
                f"values shape {values.shape} does not match grid.n_points={grid.n_points}."
            )
        effective_angle = getattr(grid, "north_angle", 0.0) if rotate else 0.0
        xs, ys = _rotated_xy(grid, effective_angle)
        _vmin = vmin if vmin is not None else float(np.nanmin(values))
        _vmax = vmax if vmax is not None else float(np.nanmax(values))
        _label = label or "Value"
        _title = title or "Floor map"

        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(xs, ys, c=values, s=point_size,
                        cmap=cmap, vmin=_vmin, vmax=_vmax,
                        edgecolors="none", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=4, width=0.8)
        if show_colorbar:
            apply_top_right_colorbar(ax, fig, sc, _label)
        if effective_angle != 0.0:
            _add_north_arrow(ax)
        apply_chart_style(ax, _title, subtitle, colorbar=show_colorbar)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    if source is None:
        raise ValueError("Either source or (values, grid) must be provided.")

    # Resolve grid and data from source
    is_comfort_result = hasattr(source, "env")
    if is_comfort_result:
        grid = source.env.grid
    else:
        grid = source.grid

    effective_angle = getattr(grid, "north_angle", 0.0) if rotate else 0.0
    xs, ys = _rotated_xy(grid, effective_angle)

    # Resolve values to colour
    if is_comfort_result and variable == "comfort_fraction":
        values = source.comfort_fraction()
        label = "Comfort fraction"
        cmap = "RdYlGn"
        vmin, vmax = 0.0, 1.0
    else:
        env = source.env if is_comfort_result else source
        raw = env.get(variable)                # (n_points, n_hours)

        if hour is not None:
            if not (0 <= hour < raw.shape[1]):
                raise ValueError(
                    f"hour={hour} is out of range for {raw.shape[1]} hours."
                )
            values = raw[:, hour]
            label = f"{variable} [°C] at hour {hour}"
        else:
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
            values = stat_funcs[annual_stat](raw, axis=1)
            label = f"{annual_stat.capitalize()} {variable} [°C]"

        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        xs, ys, c=values, s=point_size,
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="none", zorder=3,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)

    # Remove frame, keep ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=4, width=0.8)

    if show_colorbar:
        apply_top_right_colorbar(ax, fig, sc, label)

    if effective_angle != 0.0:
        _add_north_arrow(ax)

    if title is None:
        if hour is not None:
            title = f"Floor map — {variable} (hour {hour})"
        else:
            title = f"Floor map — {annual_stat} {variable}"
    apply_chart_style(ax, title, subtitle, colorbar=show_colorbar)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotated_xy(grid, angle: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) display coordinates rotated *angle* degrees clockwise.

    Rotating by the grid's ``north_angle`` clockwise places geographic North
    at the top of the plot.
    """
    raw_x = grid.points[:, 0]
    raw_y = grid.points[:, 1]

    if angle == 0.0:
        return raw_x, raw_y

    cx, cy = raw_x.mean(), raw_y.mean()
    dx, dy = raw_x - cx, raw_y - cy

    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # Clockwise rotation: x' = x·cos + y·sin,  y' = −x·sin + y·cos
    rot_x = cx + dx * cos_a + dy * sin_a
    rot_y = cy - dx * sin_a + dy * cos_a
    return rot_x, rot_y


def _add_north_arrow(ax):
    """Draw a compact North arrow in the upper-right corner of the axes."""
    ax.annotate(
        "N",
        xy=(0.95, 0.95), xycoords="axes fraction",
        xytext=(0.95, 0.87), textcoords="axes fraction",
        ha="center", va="bottom",
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        annotation_clip=False,
    )
