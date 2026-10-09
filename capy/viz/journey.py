"""Journey profile visualization."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from capy.viz._style import apply_chart_style, apply_top_right_legend


def plot_journey_profile(
    result,
    hours: Optional[List[int]] = None,
    show_transitions: bool = True,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
    variable_label: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    cmap_transitions: Optional[dict] = None,
):
    """Temperature profile along a circulation path.

    X-axis: cumulative path distance from the start (metres, derived from
    ``result.path_points`` Euclidean distances).
    Y-axis: variable value (e.g. operative temperature in °C).

    Plots the annual **mean ± 1 std** as a shaded band. Additional hour
    lines can be overlaid. Coloured background bands indicate the dominant
    thermal transition type at each step.

    Parameters
    ----------
    result:
        ``JourneyResult`` from ``capy.metrics.journey.thermal_journey()``.
    hours:
        List of specific hour indices (0–8759) to overlay as individual
        lines. When ``None``, only the mean ± 1 std band is shown.
    show_transitions:
        Shade step intervals by transition type:
        green = warming, blue = cooling, grey = neutral.
    figsize:
        Matplotlib figure size (width, height) in inches.
    save_path:
        Optional path to save the figure (PNG/SVG/PDF).
    variable_label:
        Y-axis label. Defaults to ``result.variable``.
    cmap_transitions:
        Dict mapping transition value to colour:
        ``{1: 'red', 0: 'lightgrey', -1: 'steelblue'}``.
        Overrides defaults.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    profile = result.profile     # (n_steps, n_hours)
    n_steps = profile.shape[0]
    path_pts = result.path_points

    # Cumulative path distances
    if n_steps > 1:
        step_vecs = np.diff(path_pts, axis=0)              # (n_steps-1, 3)
        step_dists = np.linalg.norm(step_vecs, axis=1)    # (n_steps-1,)
        cum_dist = np.concatenate([[0.0], np.cumsum(step_dists)])
    else:
        cum_dist = np.array([0.0])

    # Annual mean ± std per step
    mean_profile = profile.mean(axis=1)    # (n_steps,)
    std_profile = profile.std(axis=1)      # (n_steps,)

    # Transition colours
    _default_colors = {1: "#e8846b", 0: "#d4d4d4", -1: "#6baed6"}
    tc = {**_default_colors, **(cmap_transitions or {})}

    fig, ax = plt.subplots(figsize=figsize)

    # Coloured step bands behind the profile
    if show_transitions and n_steps > 1:
        for i, trans in enumerate(result.transitions):
            x0, x1 = cum_dist[i], cum_dist[i + 1]
            ax.axvspan(x0, x1, alpha=0.25, color=tc[int(trans)], linewidth=0)

    # ± 1 std shaded band
    ax.fill_between(
        cum_dist,
        mean_profile - std_profile,
        mean_profile + std_profile,
        alpha=0.3, color="#555555", label="Mean ± 1 std",
    )

    # Mean profile line
    ax.plot(cum_dist, mean_profile, color="#222222", linewidth=1.8, label="Annual mean")

    # Individual hour lines
    if hours:
        for h in hours:
            if not (0 <= h < profile.shape[1]):
                raise ValueError(f"hour {h} is out of range for {profile.shape[1]} hours")
            ax.plot(cum_dist, profile[:, h], linewidth=0.9, alpha=0.7, label=f"Hour {h}")

    # Transition legend patches
    if show_transitions and n_steps > 1:
        legend_patches = [
            mpatches.Patch(color=tc[1], alpha=0.5, label="Warming →"),
            mpatches.Patch(color=tc[0], alpha=0.5, label="Neutral"),
            mpatches.Patch(color=tc[-1], alpha=0.5, label="← Cooling"),
        ]
        apply_top_right_legend(ax, handles=legend_patches + list(ax.get_lines()[:2]))
    else:
        apply_top_right_legend(ax)

    ax.set_xlabel("Path distance [m]", fontsize=10)
    ylabel = variable_label or result.variable.replace("_", " ").capitalize()
    ax.set_ylabel(f"{ylabel} [°C]", fontsize=10)
    _title = title or f"Thermal journey profile — {result.variable}"
    apply_chart_style(ax, _title, subtitle)
    ax.set_xlim(cum_dist[0], cum_dist[-1])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
