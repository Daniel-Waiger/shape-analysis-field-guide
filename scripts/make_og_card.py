"""Make docs/figures/og-card.png: the 1200 x 630 poster that link previews show.

Chat apps (WhatsApp, iMessage, Slack, X) show a *still* image in a link
preview, so the card is one frame's worth of the hero animation: the live
worm, the mean worm and the dead worm, the same three shapes as
``docs/figures/hero.gif``, with the guide's title.  Run from the repo root::

    python scripts/make_og_card.py

Kept under 300 kB so WhatsApp fetches it.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, so `shapekit` imports
from shapekit.datasets import load_celegans
from shapekit.preshape import from_complex, preshape, to_complex
from shapekit.procrustes import align
from shapekit.shape_space import frechet_mean

PAPER, INK, MUTED = "#FAF7F2", "#1F2A37", "#5B6673"
TEAL, CORAL = "#1B998B", "#E4572E"
LIVE_WORM, DEAD_WORM = 622, 2          # the same two worms the hero animation uses
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures" / "og-card.png"


def _font(weight: str) -> font_manager.FontProperties:
    for path in font_manager.findSystemFonts():
        name = pathlib.Path(path).name
        if name == f"Inter-{weight}.otf":
            return font_manager.FontProperties(fname=path)
    return font_manager.FontProperties(family="DejaVu Sans", weight="bold" if weight != "Regular" else "normal")


def main() -> None:
    points, _ = load_celegans()
    Z = to_complex(preshape(points))
    mean = frechet_mean(Z)
    live = align(Z[LIVE_WORM], mean)
    dead = align(Z[DEAD_WORM], mean)

    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0.42, 0.12, 0.55, 0.76], facecolor=PAPER)
    for z, color, lw, ms, zorder in ((live, TEAL, 6, 11, 3), (mean, INK, 6, 11, 4), (dead, CORAL, 6, 11, 5)):
        xy = from_complex(z)
        ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, lw=lw, ms=ms, solid_capstyle="round", zorder=zorder)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    pad = 0.06
    ax.set_xlim(-0.6 - pad, 0.6 + pad)

    fig.text(0.06, 0.76, "Shape Analysis", fontproperties=_font("SemiBold"), fontsize=44, color=INK)
    fig.text(0.06, 0.64, "Field Guide", fontproperties=_font("SemiBold"), fontsize=44, color=INK)
    fig.text(0.06, 0.45, "From outlines to shape space,\nfor biologists.", fontproperties=_font("Regular"),
             fontsize=19, color=MUTED, linespacing=1.4)
    legend_y = 0.24
    for i, (label, color) in enumerate((("live worm", TEAL), ("mean", INK), ("dead worm", CORAL))):
        y = legend_y - i * 0.065
        fig.add_artist(plt.Line2D([0.06, 0.095], [y + 0.012, y + 0.012], color=color, lw=5, transform=fig.transFigure,
                                  solid_capstyle="round"))
        fig.text(0.105, y, label, fontproperties=_font("Medium"), fontsize=15, color=INK)
    fig.text(0.06, 0.06, "daniel-waiger.github.io/shape-analysis-field-guide", fontproperties=_font("Regular"),
             fontsize=12.5, color=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor=PAPER)
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(OUT.parents[2])}  {size_kb:.0f} kB")
    assert size_kb < 300, "WhatsApp needs the preview image under 300 kB"


if __name__ == "__main__":
    main()
