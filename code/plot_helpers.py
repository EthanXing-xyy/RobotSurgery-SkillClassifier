"""Shared plotting helpers (house style in bn5211_style.py)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from bn5211_style import fs


def plot_confusion(fig, ax, cm, names, cbar_label="Share of true class", empty_note="no samples"):
    """Confusion matrix: colour = row-normalised share, number = trial count."""
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    im = ax.imshow(cmn, cmap="bn_seq", vmin=0, vmax=1)
    k = len(names)
    for r in range(k):
        if row_sum[r, 0] == 0:
            ax.add_patch(plt.Rectangle((-0.5, r - 0.5), k, 1, color=fs.GRID, zorder=2, linewidth=0))
            ax.text((k - 1) / 2, r, empty_note, ha="center", va="center",
                    fontsize=fs.FS_SMALL, color=fs.MUTED, zorder=3)
            continue
        for c in range(k):
            if cm[r, c] > 0:
                ax.text(c, r, "%d" % cm[r, c], ha="center", va="center", fontsize=fs.FS_SMALL,
                        color="white" if cmn[r, c] > 0.5 else fs.INK,
                        fontweight="semibold" if r == c else "normal")
    ax.set_xticks(range(k), names)
    ax.set_yticks(range(k), names)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set(xlabel="Predicted class", ylabel="True class")
    fs.colorbar(fig, im, ax, cbar_label, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    return cmn


def per_class_dots(ax, names, metrics, support=None, xlim=(0, 1.0), xlabel="F1 score"):
    """Dot plot of per-class scores; metrics = [(label, values), ...]."""
    k = len(names)
    yy = np.arange(k)
    for y in yy:
        ax.axhline(y, color=fs.GRID, lw=0.6, zorder=0)
    offsets = np.linspace(-0.2, 0.2, len(metrics)) if len(metrics) > 1 else [0.0]
    for i, (lab, v) in enumerate(metrics):
        v = np.asarray(v, dtype=float)
        ax.plot(v, yy + offsets[i], linestyle="none", marker=fs.MARKERS[i], markersize=4.0,
                color=fs.CYCLE[i], label=lab, zorder=3 + i, markeredgecolor="white",
                markeredgewidth=0.3, clip_on=False)
    ax.set_yticks(yy, names if support is None else
                  ["%s (%d)" % (n, s) for n, s in zip(names, support)])
    ax.set_ylim(k - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(xlim[0], xlim[1] + 0.01 * (xlim[1] - xlim[0]))
    ax.set_xticks(np.round(np.linspace(xlim[0], xlim[1], 6), 2))
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=min(len(metrics), 3),
              borderaxespad=0.1, handletextpad=0.2)


def model_bars(ax, labels, values, cis=None, ylabel="Macro-F1", chance=None, chance_label=None,
               colors=None, fmt="{:.3f}", rotation=20):
    """Horizontal comparison of model scores with optional error bars and a chance line."""
    x = np.arange(len(labels))
    cols = colors or [fs.CYCLE[i % len(fs.CYCLE)] for i in range(len(labels))]
    bars = ax.bar(x, values, width=0.68, color=cols)
    if cis is not None:
        lo = [v - c[0] for v, c in zip(values, cis)]
        hi = [c[1] - v for v, c in zip(values, cis)]
        ax.errorbar(x, values, yerr=[lo, hi], fmt="none", ecolor=fs.AXIS, elinewidth=0.7, capsize=2)
    if chance is not None:
        ax.axhline(chance, color=fs.MUTED, lw=0.7, linestyle=(0, (4, 2)), zorder=0)
        if chance_label:
            fs.note(ax, chance_label, x=0.01, y=chance + 0.02, va="bottom")
    fs.bar_labels(ax, bars, fmt=fmt)
    ax.set_xticks(x, labels, rotation=rotation, ha="right" if rotation else "center",
                  rotation_mode="anchor" if rotation else None)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    return bars
