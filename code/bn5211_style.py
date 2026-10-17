"""Figure house style for the BN5211 final-project report (Topic 1).

Adapted from the figure style of BN5211 Assignment 2 (same author/group), so that both
documents share one visual language. Every figure is drawn at the width it is printed in
the report (168 mm text width) and placed at 100 %, so all figures share one type scale:
7.5 pt labels, 6.8 pt ticks, 6.3 pt annotations, 9.5 pt panel letters. Figures carry no
titles; the report caption says what each panel shows. IBM Plex Sans, a muted 8-colour
palette checked for colour-vision deficiencies, open axes, and colour always paired with
a marker or a dash pattern.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FONT_DIR = ROOT / "fonts"

FIG_W_FULL = 6.61   # inches = 168 mm, the report text width
FIG_W_HALF = 3.22   # inches, two figures side by side

# --------------------------------------------------------------------------
# Palette. Order matters: series 0/1 are the most separable pair under all
# three colour-vision simulations, so the commonest comparison uses them first.
# --------------------------------------------------------------------------
PALETTE = {
    "teal": "#1F6F78", "terra": "#C8553D", "indigo": "#3F4B9A", "ochre": "#B07A22",
    "sage": "#5E8F4E", "rose": "#B0588A", "ink": "#2E3A4F", "clay": "#8A6A4F",
}
CYCLE = list(PALETTE.values())

INK = "#232834"        # text, primary strokes
AXIS = "#4A5160"       # spines and ticks
MUTED = "#6B7280"      # secondary text
RULE = "#D9DBE0"       # light rules
GRID = "#ECEDF0"       # guide lines, empty cells
PAPER = "#FFFFFF"      # figure / axes background
WASH = "#F1ECE1"       # warm highlight band

FS_TEXT, FS_TICK, FS_SMALL, FS_TAG = 7.5, 6.8, 6.3, 9.5

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h"]
DASHES = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)),
          "-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]

SEQ = LinearSegmentedColormap.from_list(
    "bn_seq", ["#FBFAF7", "#CFE3E2", "#6FA8AC", "#1F6F78", "#123F45"])
DIV = LinearSegmentedColormap.from_list(
    "bn_div", ["#8C3A26", "#C8553D", "#EFC7B8", "#F7F6F2", "#B9D6D6", "#1F6F78", "#123F45"])
for _cmap in (SEQ, DIV):
    try:
        mpl.colormaps.register(_cmap)
    except ValueError:
        pass  # already registered in this session


def _register_fonts(font_dir: Path | None = None) -> str:
    font_dir = Path(font_dir) if font_dir else FONT_DIR
    if font_dir.is_dir():
        for ttf in font_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(ttf))
    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("IBM Plex Sans", "Arial", "Liberation Sans", "DejaVu Sans"):
        if family in available:
            return family
    return "sans-serif"


def _apply(font_dir: str | os.PathLike | None = None) -> str:
    """Install the house rcParams globally. Returns the font family used."""
    family = _register_fonts(font_dir)
    mpl.rcParams.update({
        "font.family": family, "font.size": FS_TEXT,
        "axes.titlesize": FS_TEXT, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 3,
        "axes.labelsize": FS_TEXT, "axes.labelcolor": INK, "axes.labelpad": 2.5,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "legend.title_fontsize": FS_TICK, "text.color": INK, "mathtext.default": "regular",
        "axes.prop_cycle": mpl.cycler(color=CYCLE), "figure.facecolor": PAPER,
        "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "image.cmap": "bn_seq",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.2, "lines.markersize": 3.2, "lines.markeredgewidth": 0,
        "lines.solid_capstyle": "round", "patch.linewidth": 0,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3, "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.3, "legend.borderpad": 0.2,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "standard",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    return family


def _series(i: int, markevery: int | None = None, **overrides) -> dict:
    """Colour + marker + dash for series i (redundant encoding)."""
    style = {"color": CYCLE[i % len(CYCLE)], "marker": MARKERS[i % len(MARKERS)],
             "linestyle": DASHES[i % len(DASHES)]}
    if markevery is not None:
        style["markevery"] = markevery
    style.update(overrides)
    return style


def _panel_tag(ax, letter: str) -> None:
    """Register a panel letter; fs.save places it at the panel's top-left corner."""
    ax._bn_tag = letter
    if not (ax.get_title(loc="left") or ax.get_title(loc="center") or ax.get_title(loc="right")):
        ax.set_title(" ", loc="left", fontsize=FS_TAG, pad=1)   # reserve the line


def _freeze(fig) -> None:
    """Run the layout engine once and keep the resulting axes positions."""
    from matplotlib.layout_engine import PlaceHolderLayoutEngine
    engine = fig.get_layout_engine()
    if engine is not None and not isinstance(engine, PlaceHolderLayoutEngine):
        fig.canvas.draw()
        fig.set_layout_engine("none")
    if fig.get_layout_engine() is None:
        fig.set_layout_engine(PlaceHolderLayoutEngine(adjust_compatible=True, colorbar_gridspec=True))


def _align_height(fig, ref_ax, *axes) -> None:
    """Give axes the vertical extent of ref_ax (e.g. a dot plot beside a square matrix)."""
    _freeze(fig)
    p = ref_ax.get_position()
    for ax in axes:
        q = ax.get_position()
        ax.set_position([q.x0, p.y0, q.width, p.height])


def _save(fig, name: str, fig_dir: Path | None = None) -> Path:
    """Freeze the layout, place the panel letters, then write PNG + PDF."""
    fig_dir = Path(fig_dir) if fig_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    tagged = [ax for ax in fig.axes if getattr(ax, "_bn_tag", None)]
    _freeze(fig)
    if tagged and not getattr(fig, "_bn_tagged", False):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        W, H = fig.bbox.width, fig.bbox.height
        rows: dict[float, list] = {}
        for ax in tagged:
            rows.setdefault(round(ax.get_position().y1, 2), []).append((ax, ax.get_tightbbox(r)))
        for items in rows.values():
            top = max(bb.y1 for _, bb in items)
            for ax, bb in items:
                fig.text(max(bb.x0, 1) / W, min(top, H - 1) / H, ax._bn_tag, ha="left", va="top",
                         fontsize=FS_TAG, fontweight="semibold", color=INK)
        fig._bn_tagged = True
    png = fig_dir / f"{name}.png"
    fig.savefig(png)
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.close(fig)
    return png


def _halo(width: float = 2.0, color: str = PAPER):
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=color)]


def _note(ax, text: str, x: float = 0.02, y: float = 0.97, ha: str = "left", va: str = "top",
          color: str = MUTED, **kw):
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FS_SMALL,
                   color=color, **kw)


def _bar_labels(ax, bars, fmt: str = "{:.2f}", inside: bool = False) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, -10 if inside else 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_SMALL,
                    color="white" if inside else b.get_facecolor())


def _colorbar(fig, mappable, ax, label: str | None = None, ticks=None):
    cax = ax.inset_axes([1.04, 0.0, 0.045, 1.0])
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.6, labelsize=FS_SMALL, pad=1.5)
    if label:
        cb.set_label(label, fontsize=FS_SMALL, labelpad=3)
    return cb


def _mark_best(ax, x, y, color: str = INK, size: float = 5.5) -> None:
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor="none", markeredgecolor=color,
            markeredgewidth=0.9, linestyle="none", zorder=6, clip_on=False)


fs = types.SimpleNamespace(
    PALETTE=PALETTE, CYCLE=CYCLE, INK=INK, AXIS=AXIS, MUTED=MUTED, RULE=RULE, GRID=GRID,
    PAPER=PAPER, WASH=WASH, MARKERS=MARKERS, DASHES=DASHES, SEQ=SEQ, DIV=DIV,
    FS_TEXT=FS_TEXT, FS_TICK=FS_TICK, FS_SMALL=FS_SMALL, FS_TAG=FS_TAG,
    FIG_W_FULL=FIG_W_FULL, FIG_W_HALF=FIG_W_HALF, FIG_DIR=FIG_DIR,
    apply=_apply, series=_series, panel_tag=_panel_tag, freeze=_freeze, align_height=_align_height,
    save=_save, halo=_halo, note=_note, bar_labels=_bar_labels, colorbar=_colorbar,
    mark_best=_mark_best)
