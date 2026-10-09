"""Category summary visualizations — generic stacked bar charts."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from capy.viz._style import apply_chart_style, apply_top_right_legend


def plot_category_summary(
    categories: List[dict],
    rows: Optional[List[str]] = None,
    values_matrix: Optional[np.ndarray] = None,
    show_values: bool = True,
    bar_height: float = 0.5,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
):
    """Horizontal stacked bar chart for any set of named categories.

    Accepts a list of category definitions (label + colour) and either
    embeds the values directly in that list (single-row mode) or takes a
    ``(n_rows, n_categories)`` matrix for side-by-side comparison.

    Parameters
    ----------
    categories:
        List of dicts, each with:

        * ``'label'`` — display name shown in the legend.
        * ``'color'`` — hex or named matplotlib colour.
        * ``'value'`` — numeric value (**required for single-row mode only**).

    rows:
        Row labels for multi-row mode. When supplied, *values_matrix* must
        also be provided and the ``'value'`` key in *categories* is ignored.
    values_matrix:
        ``(n_rows, n_categories)`` array of values used in multi-row mode.
        Each row is normalised independently so segments sum to 100 %.
    show_values:
        Annotate each segment with its percentage (default ``True``).
        Labels narrower than ~3 % of the total bar are suppressed.
    bar_height:
        Height of each bar in axes units (default 0.5).
    figsize:
        Matplotlib figure size. Auto-sized when ``None``.
    save_path:
        Optional path to save the figure (PNG/SVG/PDF).
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    Single-row (spatial variability result)::

        plot_category_summary([
            {'label': 'Variable',  'value': 38.2, 'color': '#66bb6a'},
            {'label': 'Uniform',   'value': 41.0, 'color': '#bdbdbd'},
            {'label': 'Excessive', 'value': 20.8, 'color': '#ffa726'},
        ])

    Multi-row (comparing two zones)::

        plot_category_summary(
            categories=[
                {'label': 'Variable',  'color': '#66bb6a'},
                {'label': 'Uniform',   'color': '#bdbdbd'},
                {'label': 'Excessive', 'color': '#ffa726'},
            ],
            rows=['Office', 'Atrium'],
            values_matrix=np.array([[38.2, 41.0, 20.8], [55.0, 30.0, 15.0]]),
        )
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })

    labels = [c["label"] for c in categories]
    colors = [c["color"] for c in categories]
    n_cats = len(categories)

    if rows is None:
        # Single-row: values embedded in categories
        raw = np.array([c["value"] for c in categories], dtype=float)
        all_values = raw[np.newaxis, :]   # (1, n_cats)
        row_labels = None
    else:
        if values_matrix is None:
            raise ValueError("values_matrix is required when rows is provided.")
        all_values = np.asarray(values_matrix, dtype=float)
        if all_values.shape != (len(rows), n_cats):
            raise ValueError(
                f"values_matrix shape {all_values.shape} does not match "
                f"({len(rows)}, {n_cats})."
            )
        row_labels = rows

    n_rows = all_values.shape[0]

    if figsize is None:
        figsize = (8, max(1.8, n_rows * 0.9 + 1.2))

    fig, ax = plt.subplots(figsize=figsize)

    for row_idx in range(n_rows):
        vals = all_values[row_idx]
        total = vals.sum()
        if total == 0:
            continue
        pct = vals / total * 100.0

        y = float(n_rows - 1 - row_idx)
        x_left = 0.0

        for cat_idx, (p, v, col) in enumerate(zip(pct, vals, colors)):
            ax.barh(y, p, left=x_left, height=bar_height, color=col, edgecolor="none")

            if show_values and p > 3.0:
                text = f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"
                text_color = "white" if _is_dark(col) else "#333333"
                ax.text(
                    x_left + p / 2, y, text,
                    ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold",
                )
            x_left += p

    if row_labels is not None:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels[::-1], fontsize=9)
    else:
        ax.set_yticks([])

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_xlabel("% of occupied hours", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=col, label=lbl)
        for col, lbl in zip(colors, labels)
    ]
    apply_top_right_legend(ax, handles=legend_patches, labels=labels)

    apply_chart_style(ax, title, subtitle)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _is_dark(hex_color: str) -> bool:
    """Return True when the hex colour has low luminance (use white label text)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return False
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5
