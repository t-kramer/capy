"""Temporal timeline and seasonal profile visualizations."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from capy.viz._style import apply_chart_style, apply_top_right_legend


def plot_timeline(
    data: np.ndarray,
    time_index=None,
    threshold: Optional[float] = None,
    threshold_label: Optional[str] = None,
    ylabel: str = "Value",
    aggregation: str = "mean",
    fill: bool = True,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    color: str = "#4a90d9",
):
    """Seasonal ribbon chart for any hourly time series.

    Reduces an hourly array to daily values and draws a filled area chart
    over the year. When a threshold is provided, hours above it are tinted
    green and hours below are tinted red — useful for sTA fraction vs tau,
    PMV vs 0, or any value vs a target.

    Parameters
    ----------
    data:
        ``(n_hours,)`` hourly time series (typically 8760).
    time_index:
        ``pd.DatetimeIndex`` aligned with *data*. Used for axis labels when
        supplied; falls back to a Jan–Dec generic year when ``None``.
    threshold:
        Optional horizontal reference line value. When provided, the fill
        colour switches above / below it (green / red).
    threshold_label:
        Legend label for the threshold line. Auto-generated when ``None``.
    ylabel:
        Y-axis label.
    aggregation:
        Daily reduction method: ``'mean'``, ``'median'``, ``'max'``, or
        ``'min'`` (default ``'mean'``).
    fill:
        Fill area under the curve (default ``True``).
    figsize:
        Matplotlib figure size.
    save_path:
        Optional path to save the figure (PNG/SVG/PDF).
    title:
        Figure title.
    color:
        Line and fill colour when no threshold is set.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from capy.viz.heatmap import _month_ticks_days

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })

    data = np.asarray(data, dtype=float)
    n_hours = len(data)
    n_days = n_hours // 24

    _agg = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
    if aggregation not in _agg:
        raise ValueError(f"aggregation must be one of {list(_agg)}, got '{aggregation}'")
    daily = _agg[aggregation](data[: n_days * 24].reshape(n_days, 24), axis=1)

    x = np.arange(n_days)

    fig, ax = plt.subplots(figsize=figsize)

    if fill:
        if threshold is not None:
            ax.fill_between(
                x, threshold, daily,
                where=(daily >= threshold),
                alpha=0.55, color="#66bb6a", interpolate=True,
            )
            ax.fill_between(
                x, daily, threshold,
                where=(daily < threshold),
                alpha=0.45, color="#ef9a9a", interpolate=True,
            )
        else:
            ax.fill_between(x, 0, daily, alpha=0.45, color=color)

    line_color = "#333333" if threshold is not None else color
    ax.plot(x, daily, color=line_color, linewidth=1.2)

    if threshold is not None:
        lbl = threshold_label or f"= {threshold}"
        ax.axhline(threshold, color="#444444", linewidth=1.0, linestyle="--", label=lbl)
        apply_top_right_legend(ax)

    month_starts, month_labels = _month_ticks_days()
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_xlim(0, n_days)
    ax.set_ylabel(ylabel, fontsize=10)
    apply_chart_style(ax, title, subtitle)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=4, width=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_seasonal_profile(
    data: np.ndarray,
    time_index=None,
    target_range: Optional[Tuple[float, float]] = None,
    band_percentiles: Tuple[int, int] = (25, 75),
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    color: str = "#4a90d9",
):
    """Seasonal profile line chart with optional target band and point spread.

    When *data* is ``(n_points, 365)``, the median across sensor points is
    drawn as a line and a shaded band covers the inter-percentile spread.
    When *data* is ``(365,)``, only the line is drawn.

    An optional horizontal target band (e.g. the 2–4°C beneficial amplitude
    range) is drawn behind the data.

    Parameters
    ----------
    data:
        ``(365,)`` daily scalar values, or ``(n_points, 365)`` per-point
        per-day values (e.g. ``DiurnalAmplitudeResult.amplitude_matrix``).
    time_index:
        ``pd.DatetimeIndex`` (unused today; reserved for future sub-year data).
    target_range:
        ``(low, high)`` — shaded horizontal band marking a target or reference
        range (e.g. ``(2.0, 4.0)`` for the beneficial diurnal amplitude band).
    band_percentiles:
        ``(p_low, p_high)`` percentile range for the point spread band when
        *data* is 2-D (default ``(25, 75)``).
    ylabel:
        Y-axis label.
    figsize:
        Matplotlib figure size.
    save_path:
        Optional path to save the figure (PNG/SVG/PDF).
    title:
        Figure title.
    color:
        Colour for the median line and spread band.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from capy.viz.heatmap import _month_ticks_days

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })

    data = np.asarray(data, dtype=float)
    has_spread = data.ndim == 2

    if has_spread:
        n_days = data.shape[1]
        center = np.median(data, axis=0)
        p_lo = np.percentile(data, band_percentiles[0], axis=0)
        p_hi = np.percentile(data, band_percentiles[1], axis=0)
    else:
        n_days = data.shape[0]
        center = data
        p_lo = p_hi = None

    x = np.arange(n_days)

    fig, ax = plt.subplots(figsize=figsize)

    if target_range is not None:
        ax.axhspan(
            target_range[0], target_range[1],
            alpha=0.12, color="#66bb6a", zorder=0,
            label=f"Target {target_range[0]}–{target_range[1]}",
        )

    if has_spread:
        ax.fill_between(
            x, p_lo, p_hi,
            alpha=0.25, color=color,
            label=f"P{band_percentiles[0]}–P{band_percentiles[1]}",
        )

    ax.plot(
        x, center, color=color, linewidth=1.8,
        label="Median" if has_spread else None,
    )

    if target_range is not None or has_spread:
        apply_top_right_legend(ax)

    month_starts, month_labels = _month_ticks_days()
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_xlim(0, n_days - 1)
    ax.set_ylabel(ylabel, fontsize=10)
    apply_chart_style(ax, title, subtitle)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=4, width=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
