#!/usr/bin/env python3
"""Regenerate the chapter 08 / 09 figures and ``docs/data/embeddings.json``.

Outputs (all deterministic for a given seed, CPU only):

    docs/figures/08_pointnet.png            architecture diagram
    docs/figures/08_maxpool_invariance.png  shuffled points -> identical embedding
    docs/figures/08_temperature.png         softmax of 8 similarities at tau = 1, 0.3, 0.07
    docs/figures/08_embedding_pca.png       PCA of embeddings, untrained vs trained
    docs/figures/09_center_axis_bug.png     correct centering vs the tutorial's wrong axis
    docs/data/embeddings.json               2-D PCA of <= 400 trained embeddings + probe accuracies

Run from anywhere::

    python3 scripts/make_figures_embed.py            # ~3-4 min on 4 CPU threads
    python3 scripts/make_figures_embed.py --quick    # 2 epochs, for a smoke test

Needs torch, scikit-learn and matplotlib. Visual identity follows SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from shapekit.embed import (  # noqa: E402
    PointNet,
    center_point_clouds,
    linear_probe_accuracy,
    train_pointnet,
    tutorial_center_bug,
)

DATA = ROOT / "data" / "mef_outlines_subset.npz"
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "docs" / "data"

# ----------------------------------------------------------------------------- #
# visual identity (SPEC.md)
# ----------------------------------------------------------------------------- #
PAPER, SURFACE, INK, MUTED, LINE = "#FAF7F2", "#FFFFFF", "#1F2A37", "#5B6673", "#D9D4CC"
TEAL, CORAL, GOLD, LAVENDER, SLATE = "#1B998B", "#E4572E", "#F2A541", "#7B6CF6", "#6B7A8F"
CLASS_COLOR = {0: CORAL, 1: TEAL}
CLASS_NAME = {0: "wild-type", 1: "LMNA-/-"}
DPI = 160


def set_style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.edgecolor": LINE, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "axes.titlepad": 10, "axes.titleweight": "medium",
    })


def new_figure(width_px, height_px):
    return plt.figure(figsize=(width_px / DPI, height_px / DPI))


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    return path


def bare(ax):
    """An axes with nothing drawn by matplotlib itself (for diagrams)."""
    ax.set_axis_off()
    ax.set_facecolor(PAPER)


# ----------------------------------------------------------------------------- #
# small drawing helpers
# ----------------------------------------------------------------------------- #


def box(ax, x, y, w, h, fc=SURFACE, ec=LINE, lw=1.2, ls="-", radius=1.2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                           fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=11, color=INK, ha="center", va="center", weight="normal", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, **kw)


# ----------------------------------------------------------------------------- #
# figures
# ----------------------------------------------------------------------------- #


def fig_pointnet(cell):
    """Architecture diagram: points -> shared MLP -> max-pool -> MLP -> embedding -> head."""
    fig = new_figure(1600, 1060)
    ax = fig.add_axes([0, 0, 1, 1])
    bare(ax)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 79.5)

    text(ax, 60, 76.1, "PointNet: from an unordered cloud of points to one vector", size=15, weight="semibold")
    text(ax, 60, 72.5, "every stage treats each point the same way; the max-pool is where the order stops mattering",
         size=11, color=MUTED)

    y0, h = 37.5, 26  # box bottom and height
    mid = y0 + h / 2
    title_y = y0 + h + 2.4

    # ---- 1. input ---------------------------------------------------------- #
    x, w = 2, 15
    box(ax, x, y0, w, h)
    text(ax, x + w / 2, title_y, "1  input", size=12, weight="semibold")
    pts = center_point_clouds(cell)
    pts = pts / np.abs(pts).max() * 5.4
    ax.scatter(x + w / 2 + pts[:, 0], mid + 1.2 - pts[:, 1], s=8, color=INK, lw=0)
    text(ax, x + w / 2, y0 + 3.2, "N × 2", size=11, color=MUTED)
    arrow(ax, x + w + 0.4, mid, x + w + 2.6, mid)

    # ---- 2. shared per-point MLP ------------------------------------------- #
    x, w = 20, 27
    box(ax, x, y0, w, h)
    text(ax, x + w / 2, title_y, "2  shared per-point MLP", size=12, weight="semibold")
    widths = [2.2, 3.6, 7.2]
    labels = ["N×64", "N×128", "N×1024"]
    cx = x + 2.4
    rows_h, rows_y = 12.0, y0 + 8.6
    for i, (bw, lab) in enumerate(zip(widths, labels)):
        ax.add_patch(Rectangle((cx, rows_y), bw, rows_h, fc="#EEF1F4", ec=LINE, lw=1))
        # one point's row, highlighted: it travels through the network on its own
        ax.add_patch(Rectangle((cx, rows_y + rows_h * 0.62), bw, 0.9, fc=GOLD, ec="none"))
        text(ax, cx + bw / 2, rows_y - 2.0, lab, size=11, color=MUTED)
        if i < len(widths) - 1:
            arrow(ax, cx + bw + 0.5, rows_y + rows_h / 2, cx + bw + 2.6, rows_y + rows_h / 2, lw=1.1)
            cx += bw + 3.5
    text(ax, x + w / 2, y0 + h - 3.0, "2 → 64 → 128 → 1024", size=11)
    arrow(ax, x + w + 0.4, mid, x + w + 2.6, mid)

    # ---- 3. max-pool --------------------------------------------------------- #
    x, w = 50, 15
    box(ax, x, y0, w, h)
    text(ax, x + w / 2, title_y, "3  max-pool", size=12, weight="semibold")
    bw = 8.5
    bx = x + w / 2 - bw / 2
    ax.add_patch(Rectangle((bx, y0 + 12.5), bw, 8.5, fc="#EEF1F4", ec=LINE, lw=1))
    text(ax, x + w / 2, y0 + h - 3.0, "N × 1024", size=11, color=MUTED)
    arrow(ax, x + w / 2, y0 + 12.1, x + w / 2, y0 + 8.4, lw=1.2)
    text(ax, x + w / 2 + 1.6, y0 + 10.2, "max", size=11, color=MUTED, ha="left")
    ax.add_patch(Rectangle((bx, y0 + 6.4), bw, 1.6, fc=GOLD, ec="none"))
    text(ax, x + w / 2, y0 + 3.2, "1 × 1024", size=11, color=MUTED)
    arrow(ax, x + w + 0.4, mid, x + w + 2.6, mid)

    # ---- 4. MLP -> embedding --------------------------------------------------- #
    x, w = 68, 17
    box(ax, x, y0, w, h)
    text(ax, x + w / 2, title_y, "4  MLP → embedding", size=12, weight="semibold")
    text(ax, x + w / 2, y0 + h - 3.0, "1024 → 512 → 256", size=11)
    ax.add_patch(Rectangle((x + 3, y0 + 10.5), w - 6, 2.6, fc=SURFACE, ec=GOLD, lw=1.8))
    text(ax, x + w / 2, y0 + 16.4, "e, 256 numbers", size=11, weight="medium")
    text(ax, x + w / 2, y0 + 5.0, "the one you keep", size=11, color=MUTED)
    arrow(ax, x + w + 0.4, mid, x + w + 2.6, mid)

    # ---- 5. projection head ----------------------------------------------------- #
    x, w = 88, 17
    box(ax, x, y0, w, h)
    text(ax, x + w / 2, title_y, "5  projection head", size=12, weight="semibold")
    text(ax, x + w / 2, y0 + h - 3.0, "256 → 256 → 64", size=11)
    ax.add_patch(Rectangle((x + 4.5, y0 + 10.5), w - 9, 2.6, fc=SURFACE, ec=INK, lw=1.2))
    text(ax, x + w / 2, y0 + 16.4, "z, 64 numbers", size=11, weight="medium")
    text(ax, x + w / 2, y0 + 5.0, "L2-normalized: ‖z‖ = 1", size=11, color=MUTED)
    arrow(ax, x + w + 0.4, mid, x + w + 2.6, mid)

    # ---- loss (training only) ---------------------------------------------------- #
    x, w = 108, 10
    box(ax, x, y0 + 7, w, h - 14, fc=PAPER, ec=MUTED, ls=(0, (3, 2)))
    text(ax, x + w / 2, mid, "SupCon\nloss", size=11, color=INK)
    text(ax, x + w / 2, title_y, "training only", size=11, color=MUTED)

    # ---- numbered legend -------------------------------------------------------- #
    notes = [
        "1   N = 120 points with (x, y) each, in any order. Position has already been removed (centred).",
        "2   The same small network is applied to every point on its own (a Conv1d with kernel size 1); one row = one point.\n"
        "     The gold row follows a single point through the layers: nothing mixes points yet.",
        "3   For each of the 1024 features, keep the largest value any point produced. A maximum does not depend on order,\n"
        "     so from here on the point order is gone.",
        "4   Two dense layers turn the 1024 pooled values into the embedding e: the fingerprint used for PCA, k-means and the probe.",
        "5   A small head maps e to z and rescales it to length 1, so z·z′ is a cosine similarity. The contrastive loss is computed\n"
        "     on z; after training, the head and z are dropped.",
    ]
    ty = 31.0
    for note in notes:
        text(ax, 2, ty, note, size=11, color=INK, ha="left", va="top", linespacing=1.5)
        ty -= 3.4 * (note.count("\n") + 1) + 0.9
    return save(fig, "08_pointnet.png")


def fig_maxpool_invariance(model, cell, seed=0):
    """Shuffling the point order gives exactly the same embedding."""
    rng = np.random.default_rng(seed)
    centred = center_point_clouds(cell)
    ordered = cell[np.argsort(np.arctan2(centred[:, 1], centred[:, 0]))]  # walk around the outline
    shuffled = ordered[rng.permutation(len(cell))]
    emb_a = model.embed(ordered[None])[0]
    emb_b = model.embed(shuffled[None])[0]
    diff = np.abs(emb_a - emb_b)

    fig = new_figure(1200, 980)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], left=0.1, right=0.975, top=0.83, bottom=0.085,
                          hspace=0.55, wspace=0.32)
    cmap = LinearSegmentedColormap.from_list("index", [LINE, INK])
    order_colors = cmap(np.linspace(0, 1, len(cell)))

    for col, (pts, title) in enumerate([(ordered, "points ordered around the outline"),
                                        (shuffled, "the same points, order shuffled")]):
        ax = fig.add_subplot(gs[0, col])
        c = center_point_clouds(pts)
        ax.plot(c[:, 0], c[:, 1], color=SLATE, lw=0.8, alpha=0.5, zorder=1)
        ax.scatter(c[:, 0], c[:, 1], s=18, c=order_colors, zorder=2, edgecolor=SURFACE, lw=0.4)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("x (px, centred)")
        if col == 0:
            ax.set_ylabel("y (px, centred)")

    n_show = 48
    ax = fig.add_subplot(gs[1, 0])
    idx = np.arange(n_show)
    ax.bar(idx, emb_a[:n_show], color=SLATE, width=0.72, label="ordered")
    ax.scatter(idx, emb_b[:n_show], s=16, color=GOLD, zorder=3, edgecolor=INK, lw=0.4, label="shuffled")
    ax.axhline(0, color=LINE, lw=1)
    top = np.abs(emb_a[:n_show]).max()
    ax.set_ylim(-top * 1.15, top * 1.75)
    ax.set_xlabel(f"embedding dimension (first {n_show} of {len(emb_a)})")
    ax.set_ylabel("value")
    ax.set_title("embedding e for both orders")
    ax.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=1.0)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(np.arange(len(diff)), diff, color=INK, width=1.0)
    ax.set_ylim(0, 1e-5)
    ax.set_yticks([0, 5e-6, 1e-5])
    ax.set_yticklabels(["0", "5e-6", "1e-5"])
    ax.set_xlabel(f"embedding dimension (all {len(diff)})")
    ax.set_ylabel("|difference|")
    ax.set_title(f"difference, max = {diff.max():.1e}")
    ax.text(0.5, 0.55, "identical to float precision:\na maximum does not depend on order",
            transform=ax.transAxes, ha="center", fontsize=11, color=MUTED)

    fig.suptitle("Max-pool makes the embedding permutation-invariant", fontsize=14, weight="semibold", y=0.98)
    fig.text(0.5, 0.905, "one real MEF outline, 120 points, through the trained PointNet\n"
             "dot shade = position in the array, line = array order", ha="center", va="center",
             fontsize=11, color=MUTED, linespacing=1.5)
    return save(fig, "08_maxpool_invariance.png"), diff.max()


def fig_temperature():
    """Softmax of the same similarities at three temperatures."""
    sims = np.array([0.95, 0.80, 0.60, 0.40, 0.20, 0.00, -0.20, -0.40])
    taus = [1.0, 0.3, 0.07]
    fig = new_figure(1600, 640)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.19, wspace=0.14)
    for ax, tau in zip(axes, taus):
        logits = sims / tau
        p = np.exp(logits - logits.max())
        p /= p.sum()
        effective = float(np.exp(-(p * np.log(p)).sum()))  # exp(entropy): "how many neighbours count"
        colors = [GOLD] + [SLATE] * (len(sims) - 1)
        ax.bar(np.arange(len(sims)), p, color=colors, width=0.72)
        ax.text(0, p[0] + 0.02, f"{p[0]:.2f}", ha="center", va="bottom", fontsize=11, color=INK)
        ax.set_xticks(np.arange(len(sims)))
        ax.set_xticklabels([f"{s:g}".replace("-", "−") for s in sims])
        ax.set_xlabel("cosine similarity to the anchor")
        ax.set_title(f"τ = {tau:g}", fontsize=13)
        ax.text(0.97, 0.95, f"about {effective:.1f} neighbours\nget real weight", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=MUTED)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("softmax probability")
    fig.suptitle("The same 8 similarities through a softmax at three temperatures", fontsize=14,
                 weight="semibold", y=0.97)
    fig.text(0.5, 0.885, "softmax(s / τ): a small τ stretches the differences, so the closest neighbour (gold) takes "
             "almost all of the probability", ha="center", fontsize=11, color=MUTED)
    return save(fig, "08_temperature.png")


def pca_2d(features):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0).fit(features)
    return pca.transform(features), pca.explained_variance_ratio_


def fig_embedding_pca(emb_untrained, emb_trained, labels, acc_untrained, acc_trained, epochs, best_epoch, minutes):
    fig = new_figure(1600, 760)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.76, bottom=0.12, wspace=0.2)
    panels = [
        (emb_untrained, f"untrained PointNet (random weights)\nlinear probe: {acc_untrained:.3f}"),
        (emb_trained, f"trained {epochs} epochs on CPU ({minutes:.1f} min), best epoch {best_epoch}\n"
                      f"linear probe: {acc_trained:.3f}"),
    ]
    coords_out = None
    for ax, (emb, title) in zip(axes, panels):
        coords, ratio = pca_2d(emb)
        for k in (0, 1):
            m = labels == k
            ax.scatter(coords[m, 0], coords[m, 1], s=22, color=CLASS_COLOR[k], alpha=0.85,
                       edgecolor=SURFACE, lw=0.6, label=f"{CLASS_NAME[k]} (class {k})")
        ax.set_title(title, fontsize=12)
        lo, hi = np.percentile(coords, [0.5, 99.5], axis=0)
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        off = int(((coords < lo) | (coords > hi)).any(axis=1).sum())
        if off:
            ax.text(0.98, 0.03, f"{off} point{'s' if off > 1 else ''} off-scale (axes clipped to 0.5–99.5 %)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=MUTED)
        ax.set_xlabel(f"PC1 ({ratio[0] * 100:.0f} % of variance)")
        ax.set_ylabel(f"PC2 ({ratio[1] * 100:.0f} % of variance)")
        coords_out = coords
    axes[0].legend(loc="best")

    gain = acc_trained - acc_untrained
    if gain >= 0.03:
        verdict = f"training separates the classes better (probe {gain:+.3f})"
    elif gain > 0:
        verdict = f"training helps only a little here (probe {gain:+.3f})"
    else:
        verdict = f"training did not improve the probe here ({gain:+.3f})"
    fig.suptitle(f"PCA of 256-d embeddings, {len(labels)} held-out cells: {verdict}", fontsize=13.5,
                 weight="semibold", y=0.965)
    fig.text(0.5, 0.885, "linear probe = logistic regression, 5-fold stratified CV on these held-out embeddings; "
             "PCA fitted separately per panel", ha="center", fontsize=11, color=MUTED)
    return save(fig, "08_embedding_pca.png"), coords_out


def _square_range(ax, pts, pad=1.18):
    """Equal aspect with a square data window, so all panels get the same box."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    r = float((hi - lo).max()) / 2 * pad
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] + r, centre[1] - r)  # inverted: image coordinates
    ax.set_aspect("equal")


def fig_center_axis_bug(cell, label):
    color = CLASS_COLOR[int(label)]
    correct = center_point_clouds(cell)
    wrong = tutorial_center_bug(cell)
    fig = new_figure(1600, 700)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.69, bottom=0.13, wspace=0.32)

    ax = axes[0]
    ax.scatter(cell[:, 0], cell[:, 1], s=16, color=color, edgecolor=SURFACE, lw=0.4)
    cx, cy = cell.mean(axis=0)
    ax.plot(cx, cy, marker="+", color=GOLD, ms=14, mew=2.2)
    ax.annotate("centroid", (cx, cy), xytext=(9, 9), textcoords="offset points", fontsize=11, color=MUTED)
    ax.set_title("as stored\n(image pixel coordinates)", pad=26)
    ax.text(0.5, 1.03, "X, shape (N, 2)", transform=ax.transAxes, ha="center", fontsize=11,
            family="monospace", color=MUTED)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    _square_range(ax, cell)

    ax = axes[1]
    ax.scatter(correct[:, 0], correct[:, 1], s=16, color=color, edgecolor=SURFACE, lw=0.4)
    ax.plot(0, 0, marker="+", color=GOLD, ms=14, mew=2.2)
    ax.set_title("correct: mean over\nthe points axis", pad=26)
    ax.text(0.5, 1.03, "X - X.mean(axis=-2)", transform=ax.transAxes, ha="center", fontsize=11,
            family="monospace", color=MUTED)
    ax.set_xlabel("x − x̄ (px)")
    ax.set_ylabel("y − ȳ (px)")
    _square_range(ax, correct)

    ax = axes[2]
    _square_range(ax, wrong)
    lim = max(abs(v) for v in (*ax.get_xlim(), *ax.get_ylim()))
    ax.plot([-lim, lim], [lim, -lim], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.scatter(wrong[:, 0], wrong[:, 1], s=16, color=color, edgecolor=SURFACE, lw=0.4, zorder=3)
    ax.set_title("wrong: mean over the\n(x, y) pair of each point", pad=26)
    ax.text(0.5, 1.03, "X.T - X.T.mean(axis=-2)", transform=ax.transAxes, ha="center", fontsize=11,
            family="monospace", color=MUTED)
    ax.set_xlabel("(x − y) / 2")
    ax.set_ylabel("(y − x) / 2")
    ax.text(0.97, 0.05, "every point lands on y = −x:\nhalf of the shape is gone", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=INK)

    fig.suptitle("One cell, centred two ways: the axis you average over matters", fontsize=14,
                 weight="semibold", y=0.965)
    fig.text(0.5, 0.885, "the original tutorial transposed each cell to (2, N) and then averaged over axis −2, "
             "which is the pair (x, y), not the N points", ha="center", fontsize=11, color=MUTED)
    return save(fig, "09_center_axis_bug.png")


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #


def pick_cell(points, labels, want_label=0):
    """A deterministic, typical cell: the class-0 cell whose extent is closest to the class median."""
    extents = (points.max(axis=1) - points.min(axis=1)).max(axis=1)
    candidates = np.flatnonzero(labels == want_label)
    median = np.median(extents[candidates])
    return int(candidates[np.argmin(np.abs(extents[candidates] - median))])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="2 epochs, for a smoke test")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 2

    torch.set_num_threads(args.threads)
    set_style()
    t0 = time.perf_counter()
    log = lambda msg: print(f"[{time.perf_counter() - t0:6.1f} s] {msg}", flush=True)  # noqa: E731

    data = np.load(DATA)
    points = data["points"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    log(f"loaded {DATA.name}: points {points.shape}, labels {np.bincount(labels).tolist()}")

    cell_index = pick_cell(points, labels, want_label=0)
    cell = points[cell_index]
    log(f"example cell: index {cell_index}, class {labels[cell_index]} ({CLASS_NAME[int(labels[cell_index])]})")

    # figures that need no training
    log(f"wrote {fig_pointnet(cell).name}")
    log(f"wrote {fig_temperature().name}")
    log(f"wrote {fig_center_axis_bug(cell, labels[cell_index]).name}")

    # training (seeded; model.train()/eval() handled inside train_pointnet)
    log(f"training PointNet: epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"wd={args.weight_decay} seed={args.seed} threads={args.threads}")
    t_train = time.perf_counter()
    result = train_pointnet(points, labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            seed=args.seed, weight_decay=args.weight_decay, augment=True, log=log)
    train_minutes = (time.perf_counter() - t_train) / 60
    model = result["model"]
    val_idx, scale = result["val_idx"], result["scale"]
    log(f"training done in {train_minutes:.1f} min; best epoch {result['best_epoch']} "
        f"(hold-out probe {result['best_val_acc']:.3f}, untrained {result['untrained_val_acc']:.3f})")

    # an untrained twin with the same initial weights (same seed, same construction order)
    torch.manual_seed(args.seed)
    untrained = PointNet(embed_dim=model.embed_dim, proj_dim=model.proj_dim)
    untrained.input_scale.fill_(scale)
    untrained.eval()

    val_points, val_labels = points[val_idx], labels[val_idx]
    emb_untrained = untrained.embed(val_points)
    emb_trained = model.embed(val_points)
    acc_untrained = linear_probe_accuracy(emb_untrained, val_labels, n_splits=5, seed=args.seed)
    acc_trained = linear_probe_accuracy(emb_trained, val_labels, n_splits=5, seed=args.seed)
    log(f"5-fold linear probe on {len(val_idx)} held-out cells: untrained {acc_untrained:.4f}, "
        f"trained {acc_trained:.4f}")

    path, max_diff = fig_maxpool_invariance(model, cell, seed=args.seed)
    log(f"wrote {path.name} (max |difference| after shuffling: {max_diff:.2e})")
    path, coords = fig_embedding_pca(emb_untrained, emb_trained, val_labels, acc_untrained, acc_trained,
                                     args.epochs, result["best_epoch"], train_minutes)
    log(f"wrote {path.name}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    keep = min(400, len(coords))
    payload = {
        "points": [[round(float(x), 4), round(float(y), 4)] for x, y in coords[:keep]],
        "labels": [int(v) for v in val_labels[:keep]],
        "probe_accuracy": {"untrained": round(acc_untrained, 4), "trained": round(acc_trained, 4)},
    }
    out = DATA_DIR / "embeddings.json"
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    log(f"wrote {out.relative_to(ROOT)} ({keep} points)")
    log("done")


if __name__ == "__main__":
    main()
