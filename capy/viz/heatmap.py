"""Hourly heatmap visualizations."""
from __future__ import annotations

from typing import Optional

import numpy as np

from capy.viz._style import apply_chart_style, apply_top_right_colorbar


def _apply_heatmap_style(ax, fig):
    """Remove frame, keep ticks; apply Arial font."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=4, width=0.8)


def plot_hourly_heatmap(
    source,
    variable: str = "operative_temperature",
    schedule=None,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """24h × 365d calendar heatmap.

    Day of year on the x-axis; hour of day on the y-axis with hour 0 at the
    bottom and 24 at the top. Each hour fills the gap between its integer tick
    marks (hour 1 spans y=0 to y=1). Unoccupied hours are transparent when a
    schedule is provided.

    Accepts a ``ThermalEnvironment``, ``ComfortResult``,
    ``OccupancySchedule``, or a pre-computed ``(8760,)`` numpy array.

    Parameters
    ----------
    source:
        ``ThermalEnvironment``, ``ComfortResult``, ``OccupancySchedule``, or
        a ``np.ndarray`` of shape ``(8760,)`` containing any hourly scalar
        series (e.g. ``SpatialVariabilityResult.hourly_range``).
    variable:
        Variable to plot when *source* is a ``ThermalEnvironment`` or
        ``ComfortResult`` (default: ``'operative_temperature'``).
    schedule:
        ``OccupancySchedule``; unoccupied cells are made transparent. Ignored
        when *source* is itself an ``OccupancySchedule``.
    figsize:
        Matplotlib figure size.
    save_path:
        Optional path to save the figure (PNG/SVG/PDF).
    cmap:
        Colormap name. Defaults to ``'Greens'`` for occupancy schedules and
        ``'RdYlBu_r'`` for all other sources.
    title:
        Figure title. Auto-generated when ``None``.
    label:
        Colorbar label. Auto-generated when ``None``. Most useful when
        *source* is a raw numpy array.
    vmin:
        Colorbar minimum. Derived from data when ``None``.
    vmax:
        Colorbar maximum. Derived from data when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

    is_raw_array = isinstance(source, np.ndarray)
    is_schedule = not is_raw_array and hasattr(source, "as_boolean") and hasattr(source, "n_occupied")

    if is_raw_array:
        arr = np.asarray(source, dtype=float)
        if arr.ndim != 1 or len(arr) != 8760:
            raise ValueError(
                f"When source is a numpy array it must have shape (8760,), got {arr.shape}."
            )
        grid_data = arr.reshape(365, 24).T
        _label = label or "Value"
        auto_title = "Hourly heatmap"
        default_cmap = "RdYlBu_r"
    elif is_schedule:
        grid_data = source.as_boolean().astype(float).reshape(365, 24).T  # (24, 365)
        _label = label or "Occupied"
        auto_title = "Occupancy schedule"
        default_cmap = "Greens"
    else:
        env = source.env if hasattr(source, "env") else source
        spatial_mean = env.get(variable).mean(axis=0)          # (8760,)
        grid_data = spatial_mean.reshape(365, 24).T             # (24, 365)

        if schedule is not None:
            occ = schedule.as_boolean().reshape(365, 24).T
            grid_data = np.where(occ, grid_data, np.nan)

        _label = label or f"Spatial mean {variable}"
        auto_title = f"Hourly heatmap — {variable}"
        default_cmap = "RdYlBu_r"

    if cmap is None:
        cmap = default_cmap

    # Transparent NaN (unoccupied hours show through to axes background)
    cmap_obj = mpl.colormaps[cmap].with_extremes(bad=(0, 0, 0, 0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        grid_data,
        aspect="auto", interpolation="nearest",
        cmap=cmap_obj,
        origin="lower",
        extent=[0, 365, 0, 24],
    )

    # Colorbar — use ScalarMappable for explicit range
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    _vmin = vmin if vmin is not None else (0.0 if is_schedule else float(np.nanmin(grid_data)))
    _vmax = vmax if vmax is not None else (1.0 if is_schedule else float(np.nanmax(grid_data)))
    sm = mcm.ScalarMappable(cmap=cmap_obj, norm=mcolors.Normalize(vmin=_vmin, vmax=_vmax))
    sm.set_array([])
    apply_top_right_colorbar(ax, fig, sm, _label)

    month_starts_day, month_labels = _month_ticks_days()
    ax.set_xticks(month_starts_day)
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_xlabel("Day of year", fontsize=10)

    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_yticklabels(["0h", "6h", "12h", "18h", "24h"], fontsize=8)
    ax.set_ylabel("Hour of day", fontsize=10)

    apply_chart_style(ax, title if title is not None else auto_title, subtitle, colorbar=True)

    _apply_heatmap_style(ax, fig)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_ticks_days():
    """(starts_in_days, labels) for monthly x-axis ticks on a day axis."""
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    starts, d = [], 0
    for nd in days:
        starts.append(d)
        d += nd
    return starts, labels
