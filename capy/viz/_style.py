"""Shared chart style helpers."""
from __future__ import annotations

from typing import Optional


def apply_chart_style(
    ax,
    title: Optional[str],
    subtitle: Optional[str] = None,
    colorbar: bool = False,
) -> None:
    """Bold left-aligned title with optional subtitle above the axes.

    colorbar=True increases the title pad to leave room for a top-right
    horizontal colorbar placed by apply_top_right_colorbar().
    """
    if title:
        if subtitle and colorbar:
            pad = 46
        elif colorbar:
            pad = 44
        elif subtitle:
            pad = 22
        else:
            pad = 8
        ax.set_title(title, loc='left', fontweight='bold', fontsize=11, pad=pad)
    if subtitle:
        ax.text(
            0.0, 1.0, subtitle,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, color='#666666', clip_on=False,
        )


def apply_top_right_colorbar(ax, fig, mappable, label: str):
    """Horizontal colorbar at top right, aligned with the subtitle/legend band.

    Places a thin horizontal colorbar above the axes at the right edge — the
    same vertical band occupied by legends in other chart types. The label sits
    above the bar, right-aligned. Call apply_chart_style() with colorbar=True
    to ensure the title pad is large enough to avoid overlap.
    """
    import matplotlib as mpl

    cax = ax.inset_axes([0.56, 1.06, 0.43, 0.038])
    cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cb.ax.spines[:].set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, width=0.7)
    cb.locator = mpl.ticker.MaxNLocator(nbins=4)
    cb.update_ticks()
    cax.set_title(label, loc='right', fontsize=8, pad=2, color='#444444')
    return cb


def apply_top_right_legend(ax, handles=None, labels=None) -> None:
    """Horizontal legend at the top-right, vertically aligned with the subtitle."""
    if handles is None:
        handles, auto_labels = ax.get_legend_handles_labels()
        if not handles:
            return
        if labels is None:
            labels = auto_labels
    elif labels is None:
        labels = [h.get_label() for h in handles]

    ax.legend(
        handles=handles,
        labels=labels,
        loc='lower right',
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=ax.transAxes,
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
