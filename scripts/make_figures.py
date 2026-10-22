"""Regenerate the static figures in ``docs/figures/`` (deterministic, seed 0).

Run from the repository root::

    python scripts/make_figures.py                       # every figure
    python scripts/make_figures.py --only 06_scree 07_local_control
    python scripts/make_figures.py --list

Every figure is computed with ``shapekit`` from the shipped datasets; nothing
is drawn from guessed numbers.  Figures 08_* and 09_* (learned embeddings,
field notes) come from ``scripts/make_figures_embed.py`` and are not listed here.

Conventions (SPEC.md, "Visual identity"): paper background, ink text, no
top/right spines, class 0 = coral, class 1 = teal, means in ink, "before"
states in slate at 50 % alpha, highlights in gold, lavender for the tangent
plane and splines.  Shapes are drawn with equal aspect and an inverted y axis
so worms look as they do in the image.  PNGs: dpi 160, 1200 px wide (1600 for
wide ones).  GIFs: 12 fps, loop forever, last frame held about 1 s, < 3 MB.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapekit import (  # noqa: E402
    BSplineCurve,
    TangentPCA,
    align,
    align_dataset,
    arc_length,
    basis_matrix,
    chord_distance,
    cubic_bspline,
    frechet_mean,
    from_complex,
    geodesic_distance,
    geodesic_path,
    gpa_mean,
    load_celegans,
    log_map,
    mask_to_contours,
    optimal_rotation_angle,
    order_skeleton,
    preshape,
    procrustes_distance,
    resample_by_arclength,
    rotate,
    skeletonize,
    to_complex,
)

OUT_DIR = ROOT / "docs" / "figures"
SEED = 0

# ----------------------------------------------------------------- identity
PAPER = "#FAF7F2"
INK = "#1F2A37"
MUTED = "#5B6673"
LINE = "#D9D4CC"
TEAL = "#1B998B"
CORAL = "#E4572E"
GOLD = "#F2A541"
LAVENDER = "#7B6CF6"
SLATE = "#6B7A8F"
CLASS_COLOUR = {0: CORAL, 1: TEAL}
CLASS_NAME = {0: "dead", 1: "live"}

DPI = 160
GIF_DPI = 100
FPS = 12
HOLD = FPS  # frames repeated at the end of a GIF (about 1 s)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.titlelocation": "center",
        "axes.titlepad": 6,
        "axes.labelsize": 11,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#B9B3AA",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": LINE,
        "grid.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)

# Worm indices used throughout, chosen so the figures reproduce the numbers
# quoted in the chapters:
#   ROT_A, ROT_B: rotating worm B onto worm A gives theta* = -1.4922899425 rad
#   DIST_WORM:    geodesic 0.35815 / chord 0.35624 to the Frechet mean
#   HERO_LIVE:    a clearly arched live worm (mode-1 score about +2.4 SD)
#   HERO_DEAD:    a nearly straight dead worm with a slight bend
#   BENT_WORM:    a clean U-shaped live worm, 0.86 rad from the mean (chapter 04 morph)
ROT_A, ROT_B = 87, 818
DIST_WORM = 899
HERO_LIVE, HERO_DEAD = 622, 2
BENT_WORM = 1236
GPA_SEEDS = (1, 2, 3)


# ------------------------------------------------------------------ helpers
def figure(width_px: int, height_px: int, dpi: int = DPI) -> plt.Figure:
    return plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)


def save_png(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=fig.dpi, facecolor=PAPER)
    plt.close(fig)
    return path


def grab_frame(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3].copy())


def save_gif(frames: list[Image.Image], name: str, hold: int = HOLD) -> Path:
    """Write frames with one shared palette (no flicker, small files) at
    12 fps, looping forever, the last frame repeated ``hold`` times."""
    frames = list(frames) + [frames[-1]] * hold
    sample_idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    strip = Image.fromarray(np.vstack([np.asarray(frames[i]) for i in sample_idx]))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = OUT_DIR / name
    quantised[0].save(
        path, save_all=True, append_images=quantised[1:], duration=int(round(1000 / FPS)),
        loop=0, optimize=False, disposal=1,
    )
    return path


def fit_text(fig, x, y, text, size=11, weight="normal", color=INK, ha="left", va="center", width=0.95):
    """Figure text that is wrapped if it would run past ``width`` of the figure."""
    t = fig.text(x, y, text, fontsize=size, weight=weight, color=color, ha=ha, va=va, linespacing=1.25)
    renderer = fig.canvas.get_renderer()
    extent = t.get_window_extent(renderer).width
    max_px = width * fig.get_figwidth() * fig.dpi
    if extent > max_px:
        n = max(20, int(len(text) * max_px / extent) - 1)
        t.set_text("\n".join(textwrap.wrap(text, n)))
    return t


def headline(fig, title, sub=None, y=0.95, x=0.03, size=13.5):
    fit_text(fig, x, y, title, size=size, weight="bold", width=1 - 2 * x)
    if sub:
        fit_text(fig, x, y - 0.045, sub, size=11, color=MUTED, va="top", width=1 - 2 * x)


def caption(fig, text, y=0.012):
    fit_text(fig, 0.5, y, text, size=10.5, color=MUTED, ha="center", va="bottom", width=0.96)


def shape_axes(ax, xlim=(-0.6, 0.6), ylim=(-0.3, 0.3)):
    """A bare panel for a preshape: equal aspect, y inverted, no frame."""
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])  # inverted y: image convention
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_worm(ax, z, color, lw=2.4, ms=5.5, alpha=1.0, zorder=3, ls="-", label=None):
    """Plot a shape given as complex (N,) or real (N, 2)."""
    xy = from_complex(z) if np.iscomplexobj(z) else np.asarray(z)
    (line,) = ax.plot(
        xy[:, 0], xy[:, 1], ls, color=color, lw=lw, alpha=alpha, zorder=zorder,
        marker="o", ms=ms, markeredgecolor=color, markerfacecolor=color, label=label,
    )
    return line


def recolour(line, c):
    line.set_color(c)
    line.set_markerfacecolor(c)
    line.set_markeredgecolor(c)


def set_xy(line, z):
    xy = from_complex(z) if np.iscomplexobj(z) else np.asarray(z)
    line.set_data(xy[:, 0], xy[:, 1])


def blend(c1: str, c2: str, t: float) -> tuple:
    a = np.array(matplotlib.colors.to_rgb(c1))
    b = np.array(matplotlib.colors.to_rgb(c2))
    return tuple((1 - t) * a + t * b)


def handle(color, lw=2.4, marker="o", ls="-", alpha=1.0, label=""):
    return plt.Line2D([], [], color=color, lw=lw, marker=marker, ls=ls, alpha=alpha, label=label)


# --------------------------------------------------------------------- data
class Data:
    """Everything derived from the worm dataset, computed once."""

    def __init__(self) -> None:
        self.points, self.labels = load_celegans()
        self.Z = to_complex(preshape(self.points))
        self.mean = frechet_mean(self.Z)
        self.Z_al = align_dataset(self.Z, self.mean)
        self.pca = TangentPCA(mean=self.mean).fit(self.Z)
        self.scores = self.pca.transform(self.Z)

    def aligned(self, i: int) -> np.ndarray:
        return self.Z_al[i]


_DATA: Data | None = None


def data() -> Data:
    global _DATA
    if _DATA is None:
        _DATA = Data()
    return _DATA


# ---------------------------------------------------------- synthetic masks
def synthetic_cell_image(rng: np.random.Generator, size: int = 160) -> np.ndarray:
    """A plausible fluorescence-like image: one big blob with a wavy edge,
    one blob cut by the frame, one speck of debris, a soft background and
    noise.  No real image ships with the repository."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)

    def blob(cx, cy, r0, wobble, phase):
        ang = np.arctan2(yy - cy, xx - cx)
        rad = np.hypot(xx - cx, yy - cy)
        r = r0 * (1 + wobble[0] * np.sin(3 * ang + phase) + wobble[1] * np.cos(5 * ang - phase)
                  + wobble[2] * np.sin(2 * ang + 2 * phase))
        return (rad < r).astype(float)

    img = 0.10 + 0.06 * (xx / size) + 0.04 * (yy / size)
    img += 0.70 * blob(78, 84, 40, (0.16, 0.09, 0.07), 0.6)
    img += 0.55 * blob(150, 28, 26, (0.10, 0.06, 0.0), 1.9)
    img += 0.50 * blob(24, 142, 4, (0.0, 0.0, 0.0), 0.0)
    img = ndimage.gaussian_filter(img, 2.2)
    img += rng.normal(0, 0.035, img.shape)
    img = ndimage.gaussian_filter(img, 0.6)
    return np.clip(img, 0, 1)


def worm_mask_from_landmarks(xy: np.ndarray, scale: float, half_width: float, margin: int = 6):
    """Rasterise a band around the spline through the 12 landmarks (scaled by
    ``scale`` so the worm is a modest number of pixels long)."""
    xy = np.asarray(xy, dtype=float) * scale
    origin = np.floor(xy.min(axis=0)) - margin
    local = xy - origin
    curve = BSplineCurve(len(local), closed=False).fit(local).sample(600)
    w = int(np.ceil(local[:, 0].max())) + margin
    h = int(np.ceil(local[:, 1].max())) + margin
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)
    d = np.full(len(grid), np.inf)
    for start in range(0, len(curve), 60):  # chunked to keep memory small
        seg = curve[start:start + 60]
        d = np.minimum(d, np.min(np.linalg.norm(grid[:, None, :] - seg[None, :, :], axis=2), axis=1))
    return d.reshape(h, w) <= half_width


# ------------------------------------------------------------- 3-D sphere
class SphereView:
    """Orthographic drawing of a unit sphere.  Coordinates are given directly
    in camera space: (u, v) is the screen position, w the depth towards the
    viewer, so a point is visible when w > 0."""

    def __init__(self, ax, light=(-0.45, 0.55, 0.70)):
        self.ax = ax
        self.light = np.asarray(light) / np.linalg.norm(light)
        ax.set_aspect("equal")
        ax.set_axis_off()

    def draw_sphere(self, n: int = 700):
        u = np.linspace(-1, 1, n)
        U, V = np.meshgrid(u, u)
        inside = U**2 + V**2 <= 1
        W = np.sqrt(np.clip(1 - U**2 - V**2, 0, None))
        normals = np.stack([U, V, W], axis=-1)
        lam = np.clip(normals @ self.light, 0, 1)
        shade = 0.40 + 0.60 * lam**1.2
        lit = np.array(matplotlib.colors.to_rgb("#FFFFFF"))
        dark = np.array(matplotlib.colors.to_rgb("#B3AC9F"))
        rgb = dark + (lit - dark) * shade[..., None]
        rgba = np.concatenate([rgb, inside[..., None].astype(float)], axis=-1)
        self.ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="lower", zorder=1, interpolation="bilinear")
        self.ax.add_patch(patches.Circle((0, 0), 1, fill=False, ec="#A39C90", lw=1.0, zorder=2))

    @staticmethod
    def _split(pts):
        """Split a polyline on the sphere into front and back runs."""
        front = pts[:, 2] >= 0
        runs, cur = [], []
        for p, f in zip(pts, front):
            if cur and cur[0][1] != f:
                runs.append(cur)
                cur = []
            cur.append((p, f))
        if cur:
            runs.append(cur)
        return [(np.array([p for p, _ in r]), r[0][1]) for r in runs]

    def draw_circle(self, normal, color="#A39C90", lw=0.9, back=True):
        normal = np.asarray(normal) / np.linalg.norm(normal)
        a = np.cross(normal, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(normal, [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        t = np.linspace(0, 2 * np.pi, 400)
        pts = np.outer(np.cos(t), a) + np.outer(np.sin(t), b)
        for run, f in self._split(pts):
            if f:
                self.ax.plot(run[:, 0], run[:, 1], color=color, lw=lw, zorder=3)
            elif back:
                self.ax.plot(run[:, 0], run[:, 1], color=color, lw=lw * 0.8, ls=(0, (2, 3)), alpha=0.55, zorder=2.5)

    def draw_arc(self, p, q, color=GOLD, lw=3.0, zorder=5, n=120):
        p = np.asarray(p) / np.linalg.norm(p)
        q = np.asarray(q) / np.linalg.norm(q)
        omega = np.arccos(np.clip(p @ q, -1, 1))
        t = np.linspace(0, 1, n)[:, None]
        pts = (np.sin((1 - t) * omega) * p + np.sin(t * omega) * q) / np.sin(omega)
        for run, f in self._split(pts):
            self.ax.plot(run[:, 0], run[:, 1], color=color, lw=lw if f else lw * 0.6,
                         alpha=1 if f else 0.35, zorder=zorder, solid_capstyle="round")
        return pts

    def scatter(self, pts, colors, s=16, alpha=0.85, zorder=4, back=False):
        pts = np.asarray(pts)
        front = pts[:, 2] > 0.02
        colors = np.asarray(colors, dtype=object)
        self.ax.scatter(pts[front, 0], pts[front, 1], s=s, c=list(colors[front]), alpha=alpha,
                        linewidths=0, zorder=zorder)
        if back:
            self.ax.scatter(pts[~front, 0], pts[~front, 1], s=s * 0.6, c=list(colors[~front]),
                            alpha=0.18, linewidths=0, zorder=2.6)

    @staticmethod
    def tangent_basis(m, twist_deg=0.0):
        """Orthonormal basis of the tangent plane at m, turned by ``twist_deg``
        inside the plane."""
        m = np.asarray(m) / np.linalg.norm(m)
        e1 = np.cross([0, 1.0, 0], m)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(m, e1)
        phi = np.radians(twist_deg)
        f1 = np.cos(phi) * e1 + np.sin(phi) * e2
        f2 = -np.sin(phi) * e1 + np.cos(phi) * e2
        return m, f1, f2

    @staticmethod
    def exp(m, e1, e2, a, b):
        """Point on the sphere reached by walking the tangent vector a e1 + b e2."""
        t = np.hypot(a, b)
        if t < 1e-12:
            return m
        return np.cos(t) * m + np.sin(t) * (a * e1 + b * e2) / t


# =============================================================== figures
def fig_hero():
    d = data()
    live = d.aligned(HERO_LIVE)
    dead = d.aligned(HERO_DEAD)
    n = 15
    leg1 = geodesic_path(live, d.mean, n)  # live -> mean
    leg2 = geodesic_path(d.mean, dead, n)  # mean -> dead
    seq = []
    for k in range(n + 1):
        seq.append((leg1[k], blend(TEAL, INK, k / n), "live worm  →  mean"))
    seq += [seq[-1]] * 6
    for k in range(n + 1):
        seq.append((leg2[k], blend(INK, CORAL, k / n), "mean  →  dead worm"))
    seq += [seq[-1]] * 10
    back = []
    for k in range(n, -1, -1):
        back.append((leg2[k], blend(INK, CORAL, k / n), "dead worm  →  mean"))
    back += [back[-1]] * 6
    for k in range(n, -1, -1):
        back.append((leg1[k], blend(TEAL, INK, k / n), "mean  →  live worm"))
    seq += back

    fig = figure(900, 400, GIF_DPI)
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.72])
    shape_axes(ax, (-0.6, 0.6), (-0.34, 0.34))
    for z, c in ((live, TEAL), (d.mean, INK), (dead, CORAL)):
        draw_worm(ax, z, c, lw=1.4, ms=3, alpha=0.28, zorder=2)
    headline(fig, "Live worm → mean → dead worm, along the shortest paths in shape space", size=13, y=0.94)
    ax.legend(handles=[handle(TEAL, 2.5, None, label="live worm"), handle(INK, 2.5, None, label="mean worm"),
                       handle(CORAL, 2.5, None, label="dead worm")],
              loc="lower left", ncol=3, bbox_to_anchor=(0.0, -0.17), handlelength=1.6)
    stage = ax.text(0.99, 1.0, "", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=11.5)
    moving = draw_worm(ax, seq[0][0], seq[0][1], lw=3.2, ms=7, zorder=5)
    frames = []
    for z, c, name in seq:
        set_xy(moving, z)
        recolour(moving, c)
        stage.set_text(name)
        frames.append(grab_frame(fig))
    plt.close(fig)
    return save_gif(frames, "hero.gif")


def fig_01_pipeline():
    rng = np.random.default_rng(SEED)
    img = synthetic_cell_image(rng)
    mask = img > 0.42
    contours = mask_to_contours(mask, min_len=0)
    kept = [c for c in contours if len(c) >= 40]
    dropped = [c for c in contours if len(c) < 40]
    main = max(kept, key=len)
    landmarks = resample_by_arclength(main, 20, closed=True)
    spline = BSplineCurve(20, closed=True).fit(landmarks)
    curve = spline.sample(400)

    fig = figure(1600, 480)
    axes = fig.subplots(1, 5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.76, bottom=0.12, wspace=0.08)
    ext = (-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5)
    for ax in axes:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(LINE)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    axes[0].set_title("1  Synthetic image\n(160 × 160 px)")
    mask_rgba = np.zeros(mask.shape + (4,))
    mask_rgba[mask] = matplotlib.colors.to_rgba(SLATE)
    axes[1].imshow(mask_rgba, extent=ext, interpolation="nearest")
    axes[1].set_title("2  Binary mask\n(intensity > 0.42)")
    axes[2].imshow(mask_rgba * [1, 1, 1, 0.25], extent=ext, interpolation="nearest")
    for c in kept:
        loop = np.vstack([c, c[:1]])
        axes[2].plot(loop[:, 0], loop[:, 1], color=INK, lw=1.6)
    for c in dropped:
        axes[2].plot(c[:, 0], c[:, 1], color=CORAL, lw=1.6)
    axes[2].set_title(f"3  Traced contours\n({len(kept)} kept, {len(dropped)} dropped)")
    axes[3].plot(np.append(main[:, 0], main[0, 0]), np.append(main[:, 1], main[0, 1]), color=SLATE, lw=1.0, alpha=0.5)
    axes[3].plot(landmarks[:, 0], landmarks[:, 1], "o", color=CORAL, ms=6, zorder=4)
    axes[3].set_title("4  20 landmarks\n(equal arc length)")
    axes[4].plot(landmarks[:, 0], landmarks[:, 1], "o", color=CORAL, ms=5, zorder=4, alpha=0.9)
    loop = np.vstack([curve, curve[:1]])
    axes[4].plot(loop[:, 0], loop[:, 1], color=LAVENDER, lw=2.4, zorder=3)
    axes[4].set_title("5  Cubic B-spline\n(20 control points)")
    headline(fig, "From pixels to an outline model: image → mask → contour → landmarks → spline", x=0.01, y=0.94)
    caption(fig, "Synthetic image (no real image ships with the repo). The contour shorter than min_len = 40 points is "
                 "dropped (coral); the spline interpolates the 20 landmarks exactly.")
    return save_png(fig, "01_pipeline.png")


def fig_01_arclength():
    d = data()
    mask = worm_mask_from_landmarks(d.points[HERO_LIVE], scale=0.5, half_width=2.6)
    skel = skeletonize(mask)
    pixels = order_skeleton(skel)
    N = 12
    even = resample_by_arclength(pixels, N)
    steps_raw = np.linalg.norm(np.diff(pixels, axis=0), axis=1)
    steps_even = np.linalg.norm(np.diff(even, axis=0), axis=1)
    L = arc_length(pixels)[-1]
    n_diag = int(np.sum(steps_raw > 1.2))

    fig = figure(1200, 660)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.2, wspace=0.06)
    h, w = mask.shape
    mask_rgba = np.zeros(mask.shape + (4,))
    mask_rgba[mask] = matplotlib.colors.to_rgba("#E6E1D8")
    for ax in axes:
        ax.imshow(mask_rgba, extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax = axes[0]
    ax.plot(pixels[:, 0], pixels[:, 1], "s", color=SLATE, ms=3, zorder=4)
    for p, q, s in zip(pixels[:-1], pixels[1:], steps_raw):
        diag = s > 1.2
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GOLD if diag else SLATE, lw=3.4 if diag else 1.6, zorder=5 if diag else 3)
    ax.set_title(f"{len(pixels)} skeleton pixels\n{len(pixels) - 1 - n_diag} straight steps, {n_diag} diagonal (gold)")
    fig.legend(handles=[handle(SLATE, 1.6, "s", label="straight step, 1 px"), handle(GOLD, 3.4, None, label="diagonal step, 1.41 px")],
               loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=2)
    ax = axes[1]
    ax.plot(pixels[:, 0], pixels[:, 1], color=SLATE, lw=1.2, alpha=0.45, zorder=2)
    ax.plot(even[:, 0], even[:, 1], "-", color=CORAL, lw=1.8, alpha=0.6, zorder=3)
    ax.plot(even[:, 0], even[:, 1], "o", color=CORAL, ms=7.5, zorder=4)
    ax.set_title(f"{N} landmarks by arc length\nevery step {steps_even.mean():.2f} px = L / {N - 1}, L = {L:.1f} px")
    headline(fig, "Uneven pixel steps become evenly spaced landmarks")
    caption(fig, "The mask is a band around the spline through worm 622's landmarks, shrunk to half size "
                 "(no raw worm mask ships with the repo); skeleton, ordering and resampling by shapekit.contours.")
    return save_png(fig, "01_arclength_resampling.png")


def fig_01_representations():
    d = data()
    rng = np.random.default_rng(SEED)
    xy = from_complex(d.aligned(HERO_LIVE))
    spline = BSplineCurve(12, closed=False).fit(xy)
    curve = spline.sample(300)
    t = rng.uniform(0, spline.t_max, 120)
    cloud = spline.evaluate(np.sort(t)) + rng.normal(0, 0.006, (120, 2))
    cloud = cloud[rng.permutation(120)]

    fig = figure(1200, 440)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.2, wspace=0.05)
    for ax in axes:
        shape_axes(ax, (-0.52, 0.52), (-0.33, 0.33))
    ax = axes[0]
    ax.plot(xy[:, 0], xy[:, 1], color=CORAL, lw=1.4, alpha=0.5)
    ax.plot(xy[:, 0], xy[:, 1], "o", color=CORAL, ms=7, zorder=4)
    for k, (x, y) in enumerate(xy):
        ax.annotate(str(k + 1), (x, y), xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=10, color=INK)
    ax.set_title("12 landmarks, in order")
    axes[1].plot(cloud[:, 0], cloud[:, 1], "o", color=TEAL, ms=4.2, alpha=0.85, zorder=4)
    axes[1].set_title("120 points, no order")
    axes[2].plot(curve[:, 0], curve[:, 1], color=INK, lw=2.8, zorder=4)
    axes[2].set_title("A continuous curve")
    headline(fig, "The same worm, three ways to write it down", y=0.93)
    caption(fig, "Worm 622 (live), centred and scaled. The point cloud is sampled at random along the curve with a little jitter.")
    return save_png(fig, "01_representations.png")


def fig_02_preshape_steps():
    d = data()
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(d.labels == 0), 3, replace=False),
                          rng.choice(np.flatnonzero(d.labels == 1), 3, replace=False)])
    raw = d.points[idx]
    centred = raw - raw.mean(axis=1, keepdims=True)
    pre = preshape(raw)

    fig = figure(1200, 560)
    ph = 0.46  # panel height, figure fraction
    px_h = ph * 560
    w1 = px_h / (520 / 696) / 1200
    w2 = px_h / 1200
    ax1 = fig.add_axes([0.05, 0.2, w1, ph])
    ax2 = fig.add_axes([0.05 + w1 + 0.09, 0.2, w2, ph])
    ax3 = fig.add_axes([0.05 + w1 + w2 + 0.18, 0.2, w2, ph])
    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.6)
        ax.tick_params(length=0)
    ax1.add_patch(patches.Rectangle((0, 0), 696, 520, fill=False, ec="#B9B3AA", lw=1.2))
    for k, i in enumerate(idx):
        draw_worm(ax1, raw[k], CLASS_COLOUR[d.labels[i]], lw=1.6, ms=2.8)
    ax1.set_xlim(0, 696)
    ax1.set_ylim(520, 0)
    ax1.set_title("1  In the image")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    for k, i in enumerate(idx):
        draw_worm(ax2, centred[k], CLASS_COLOUR[d.labels[i]], lw=1.6, ms=2.8, alpha=0.9)
    ax2.plot(0, 0, "+", color=INK, ms=10, mew=1.5, zorder=6)
    ax2.set_xlim(-85, 85)
    ax2.set_ylim(85, -85)
    ax2.set_title("2  Centroid at (0, 0)")
    ax2.set_xlabel("x (pixels)")
    for k, i in enumerate(idx):
        draw_worm(ax3, pre[k], CLASS_COLOUR[d.labels[i]], lw=1.6, ms=2.8, alpha=0.9)
    ax3.plot(0, 0, "+", color=INK, ms=10, mew=1.5, zorder=6)
    ax3.set_xlim(-0.62, 0.62)
    ax3.set_ylim(0.62, -0.62)
    ax3.set_title("3  Scaled to norm 1")
    ax3.set_xlabel("x (no units)")
    fig.legend(handles=[handle(CORAL, 2, label="dead"), handle(TEAL, 2, label="live")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.97, 0.93))
    headline(fig, "Removing position and size: six worms, step by step",
             sub="Preshapes (right) still differ in rotation; chapter 03 removes it.", x=0.05)
    return save_png(fig, "02_preshape_steps.png")


def fig_02_sphere():
    d = data()
    fig = figure(1200, 780)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.12, 1.12)
    view = SphereView(ax)
    view.draw_sphere()
    axis = np.array([0.0, 0.92, 0.39])  # a tilted "north pole", for depth cues
    view.draw_circle(axis)
    view.draw_circle(np.cross(axis, [1.0, 0, 0]))
    m, e1, e2 = view.tangent_basis([0.05, 0.28, 0.96])
    s = d.scores[:, :2]  # tangent-PCA scores: radians along the sphere
    pts = np.array([view.exp(m, e1, e2, a, b) for a, b in s])
    colours = np.array([CLASS_COLOUR[l] for l in d.labels], dtype=object)
    order = np.argsort(pts[:, 2])  # back first
    view.scatter(pts[order], colours[order], s=11, alpha=0.75, back=True)
    ax.scatter([m[0]], [m[1]], s=70, color=INK, zorder=6, edgecolor=PAPER, linewidths=1.2)
    z = pts[DIST_WORM]
    view.draw_arc(m, z, GOLD, lw=3.2, zorder=5)
    ax.scatter([z[0]], [z[1]], s=55, color=GOLD, zorder=7, edgecolor=INK, linewidths=1.0)
    ax.annotate("mean worm", (m[0], m[1]), xytext=(-80, 60), textcoords="offset points", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), ha="center", fontsize=11.5)
    ax.annotate("worm 899", (z[0], z[1]), xytext=(70, 50), textcoords="offset points", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), ha="center", fontsize=11.5)
    mid = view.exp(m, e1, e2, *(0.5 * s[DIST_WORM]))
    ax.annotate("great-circle arc between them\n(the geodesic, chapter 04)", (mid[0], mid[1]), xytext=(150, -150),
                textcoords="offset points", color=GOLD, ha="center", fontsize=11.5,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.9, shrinkB=4))
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=CORAL, label="dead worm"),
                       plt.Line2D([], [], marker="o", ls="", color=TEAL, label="live worm")],
              loc="lower left", bbox_to_anchor=(0.02, 0.04))
    headline(fig, "Every preshape has norm 1, so it lives on a sphere",
             sub="The real sphere has 21 dimensions. This 3-D stand-in places each of the 1,377 worms by its "
                 "first two tangent-PCA scores (chapter 06), so the spread is the real spread.")
    return save_png(fig, "02_sphere.png")


def fig_03_rotation_search():
    d = data()
    zA, zB = d.Z[ROT_A], d.Z[ROT_B]
    theta_star = optimal_rotation_angle(zB, zA)
    thetas = np.linspace(-np.pi, np.pi, 48)
    errors = np.array([np.linalg.norm(zA - rotate(zB, t)) for t in thetas])
    fine = np.linspace(-np.pi, np.pi, 721)
    errors_fine = np.array([np.linalg.norm(zA - rotate(zB, t)) for t in fine])

    fig = figure(900, 430, GIF_DPI)
    ax_s = fig.add_axes([0.02, 0.06, 0.44, 0.80])
    ax_e = fig.add_axes([0.55, 0.19, 0.42, 0.66])
    shape_axes(ax_s, (-0.6, 0.6), (-0.6, 0.6))
    draw_worm(ax_s, zA, INK, lw=2.6, ms=6, zorder=4)
    b_line = draw_worm(ax_s, zB, SLATE, lw=2.4, ms=5.5, alpha=0.75, zorder=3)
    leg = ax_s.legend(handles=[handle(INK, 2.5, label="worm A (fixed)"), handle(SLATE, 2.5, label="worm B, turned by θ")],
                      loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.8)
    ax_e.set_xlim(-np.pi, np.pi)
    ax_e.set_ylim(0, 2.1)
    ax_e.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_e.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
    ax_e.set_xlabel("rotation angle θ (rad)")
    ax_e.set_ylabel("mismatch  E(θ) = ‖A − e^{iθ} B‖")
    ax_e.grid(True, alpha=0.6)
    (curve,) = ax_e.plot([], [], color=INK, lw=2.2)
    (dot,) = ax_e.plot([], [], "o", color=SLATE, ms=8, zorder=5)
    readout = ax_e.text(0.98, 0.95, "", transform=ax_e.transAxes, ha="right", va="top", color=INK, fontsize=12)
    headline(fig, "Rotation by trial and error: turn B over A and watch the mismatch", size=13, y=0.94, x=0.02)
    frames = []
    for t, e in zip(thetas, errors):
        set_xy(b_line, rotate(zB, t))
        sel = fine <= t
        curve.set_data(fine[sel], errors_fine[sel])
        dot.set_data([t], [e])
        readout.set_text(f"θ = {t:+.2f} rad\nE = {e:.3f}")
        frames.append(grab_frame(fig))
    # the answer: B at the best angle, minimum marked
    e_min = np.linalg.norm(zA - rotate(zB, theta_star))
    set_xy(b_line, rotate(zB, theta_star))
    recolour(b_line, GOLD)
    b_line.set_alpha(1.0)
    curve.set_data(fine, errors_fine)
    dot.set_data([theta_star], [e_min])
    dot.set_color(GOLD)
    dot.set_markersize(11)
    dot.set_markeredgecolor(INK)
    ax_e.axvline(theta_star, color=GOLD, lw=1.2, ls="--", zorder=2)
    readout.set_text(f"minimum at θ* = {theta_star:.4f} rad\nE = {e_min:.3f}")
    leg.remove()
    ax_s.legend(handles=[handle(INK, 2.5, label="worm A (fixed)"), handle(GOLD, 2.5, label="worm B at θ*")],
                loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.8)
    frames.append(grab_frame(fig))
    plt.close(fig)
    return save_gif(frames, "03_rotation_search.gif", hold=HOLD + 6)


def fig_03_votes():
    d = data()
    zA, zB = d.Z[ROT_A], d.Z[ROT_B]
    votes = np.conj(zB) * zA  # conj(b_k) a_k : rotates B onto A
    total = votes.sum()
    theta_star = np.angle(total)
    assert np.isclose(theta_star, optimal_rotation_angle(zB, zA))

    fig = figure(1600, 640)
    ax_s = fig.add_axes([0.02, 0.10, 0.28, 0.70])
    ax_v = fig.add_axes([0.37, 0.16, 0.26, 0.64])
    ax_t = fig.add_axes([0.70, 0.16, 0.28, 0.64])
    shape_axes(ax_s, (-0.6, 0.6), (-0.6, 0.6))
    draw_worm(ax_s, zA, INK, lw=2.2, ms=5)
    draw_worm(ax_s, zB, SLATE, lw=2.0, ms=4.5, alpha=0.75)
    for k in (0, 5, 11):
        for z, c in ((zA, INK), (zB, SLATE)):
            ax_s.annotate(str(k + 1), (z[k].real, z[k].imag), xytext=(7, -6), textcoords="offset points",
                          fontsize=10, color=c)
    ax_s.set_title("Worms A (ink) and B (slate)")
    for ax in (ax_v, ax_t):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.tick_params(length=0)
    r = np.abs(votes).max() * 1.2
    ax_v.set_xlim(-r, r)
    ax_v.set_ylim(-r, r)
    for k, v in enumerate(votes):
        ax_v.annotate("", xy=(v.real, v.imag), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6, mutation_scale=12, shrinkA=0, shrinkB=0))
        if k in (0, 5, 11):
            ax_v.annotate(str(k + 1), (v.real, v.imag), xytext=(5, 4), textcoords="offset points", fontsize=10, color=INK)
    ax_v.set_title("12 votes: arrow k = conj(bₖ) · aₖ")
    ax_v.text(0.03, 0.97, "the far-out landmarks (1, 12)\ncast the longest arrows", transform=ax_v.transAxes,
              fontsize=10.5, color=MUTED, va="top")
    chain = np.concatenate([[0], np.cumsum(votes)])
    lo = min(chain.real.min(), chain.imag.min())
    hi = max(chain.real.max(), chain.imag.max())
    pad = 0.15 * (hi - lo)
    ax_t.set_xlim(-0.55, 0.75)
    ax_t.set_ylim(lo - pad, max(hi, 0.15) + pad)
    for p, q in zip(chain[:-1], chain[1:]):
        ax_t.annotate("", xy=(q.real, q.imag), xytext=(p.real, p.imag),
                      arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4, mutation_scale=10, shrinkA=0, shrinkB=0))
    ax_t.annotate("", xy=(total.real, total.imag), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3, mutation_scale=18, shrinkA=0, shrinkB=0))
    arc_r = 0.22
    ang = np.linspace(0, theta_star, 60)
    ax_t.plot(arc_r * np.cos(ang), arc_r * np.sin(ang), color=INK, lw=1.2)
    ax_t.text(arc_r * 1.25 * np.cos(theta_star / 2) + 0.03, arc_r * 1.25 * np.sin(theta_star / 2), "θ*",
              color=INK, ha="left", va="center", fontsize=12)
    ax_t.set_title("Head to tail; the sum in gold")
    ax_t.text(0.97, 0.03, f"angle of the sum\nθ* = {theta_star:.4f} rad\n= {np.degrees(theta_star):.1f}°\n"
                          f"|sum| = {abs(total):.3f}",
              transform=ax_t.transAxes, ha="right", va="bottom", fontsize=11, color=INK)
    headline(fig, "Each landmark pair votes for a rotation; the angle of the summed votes is the answer", x=0.02)
    caption(fig, "Worms 87 (A) and 818 (B); rotating B by θ* gives the least mismatch, the same −1.4923 rad the trial-and-error search found.")
    return save_png(fig, "03_votes.png")


def fig_04_distances():
    d = data()
    z = d.Z[DIST_WORM]
    rho = geodesic_distance(z, d.mean)
    chord = chord_distance(z, d.mean)
    proc = procrustes_distance(z, d.mean)
    inner = abs(np.vdot(z, d.mean))
    assert abs(rho - 0.35815) < 1e-4 and abs(chord - 0.35624) < 1e-4

    fig = figure(1200, 640)
    ax = fig.add_axes([0.01, 0.03, 0.56, 0.82])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.15, 1.3)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="#A39C90", lw=1.4, zorder=2)
    ax.plot(0, 0, "+", color=INK, ms=10, mew=1.4, zorder=4)
    ax.text(0.0, -0.08, "centre of the sphere", ha="center", va="top", color=MUTED, fontsize=10.5)
    a_m = np.pi / 2 + rho / 2
    a_z = np.pi / 2 - rho / 2
    pm = np.array([np.cos(a_m), np.sin(a_m)])
    pz = np.array([np.cos(a_z), np.sin(a_z)])
    for p in (pm, pz):
        ax.plot([0, p[0]], [0, p[1]], color=SLATE, lw=1.0, ls="--", zorder=3)
    arc = np.linspace(a_z, a_m, 100)
    ax.plot(np.cos(arc), np.sin(arc), color=GOLD, lw=5, zorder=5, solid_capstyle="round")
    ax.plot([pm[0], pz[0]], [pm[1], pz[1]], color=LAVENDER, lw=2.8, zorder=4)
    ax.plot([pm[0], pz[0]], [pm[1], pz[1]], "o", color=INK, ms=8, zorder=6)
    small = np.linspace(a_z, a_m, 40)
    ax.plot(0.3 * np.cos(small), 0.3 * np.sin(small), color=INK, lw=1.0, zorder=4)
    ax.text(0.0, 0.36, f"ρ = {np.degrees(rho):.1f}°", ha="center", va="bottom", color=INK, fontsize=11)
    ax.annotate(f"arc = geodesic, {rho:.5f}", xy=(0, 1.0), xytext=(0, 1.2), ha="center", va="bottom", color=INK,
                fontsize=11.5, arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.4))
    ax.annotate(f"chord = Procrustes, {chord:.5f}", xy=(0, np.cos(rho / 2)), xytext=(0, 0.74), ha="center", va="top",
                color=INK, fontsize=11.5, arrowprops=dict(arrowstyle="-", color=LAVENDER, lw=1.4))
    ax.text(pm[0] - 0.06, pm[1] + 0.02, "m", ha="right", va="bottom", color=INK, fontsize=12.5)
    ax.text(pz[0] + 0.06, pz[1] + 0.02, "z", ha="left", va="bottom", color=INK, fontsize=12.5)
    ax.text(0, -1.05, "the great circle through m and z, drawn to scale", ha="center", va="top", color=MUTED, fontsize=10.5)
    # right column: the two shapes and the numbers
    for z_shape, box, colour, name in ((d.mean, [0.58, 0.50, 0.19, 0.26], INK, "m: the mean worm"),
                                       (align(z, d.mean), [0.79, 0.50, 0.19, 0.26], TEAL, "z: worm 899 (live)")):
        ins = fig.add_axes(box)
        shape_axes(ins, (-0.55, 0.55), (-0.45, 0.45))
        draw_worm(ins, z_shape, colour, lw=2, ms=4)
        ins.set_title(name, fontsize=11)
    lines = [
        f"|⟨z, m⟩| = cos ρ = {inner:.5f}",
        f"geodesic: ρ = arccos|⟨z, m⟩| = {rho:.5f}",
        f"chord: √(2 − 2|⟨z, m⟩|) = {chord:.5f}",
        f"Procrustes: ‖m − e^{{iθ*}}z‖ = {proc:.5f}",
        f"gap: {rho - chord:.5f} ≈ ρ³/24 = {rho**3 / 24:.5f}",
    ]
    fig.text(0.58, 0.47, "\n".join(lines), ha="left", va="top", fontsize=11, color=INK, linespacing=1.5)
    fig.text(0.58, 0.03, "Chord and Procrustes: one expression.\nThe arc is always a little longer.",
             ha="left", va="bottom", fontsize=10.5, color=MUTED)
    headline(fig, "Along the sphere or straight through it: geodesic versus chord")
    return save_png(fig, "04_distances.png")


def fig_04_geodesic_morph():
    d = data()
    target = d.aligned(BENT_WORM)
    path = geodesic_path(d.mean, target, 10)  # the 11 steps
    n_frames = 40
    fine = geodesic_path(d.mean, target, n_frames - 1)
    rho = geodesic_distance(d.mean, target)

    fig = figure(900, 520, GIF_DPI)
    ax = fig.add_axes([0.02, 0.33, 0.96, 0.54])
    shape_axes(ax, (-0.62, 0.62), (-0.36, 0.36))
    draw_worm(ax, d.mean, SLATE, lw=1.6, ms=3.5, alpha=0.5, zorder=2)
    draw_worm(ax, target, SLATE, lw=1.6, ms=3.5, alpha=0.5, zorder=2)
    moving = draw_worm(ax, path[0], INK, lw=3.2, ms=7, zorder=5)
    readout = ax.text(0.99, 0.04, "", transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=12)
    thumbs = []
    for k in range(11):
        tax = fig.add_axes([0.02 + k * 0.0873, 0.07, 0.085, 0.2])
        shape_axes(tax, (-0.62, 0.62), (-0.42, 0.42))
        ln = draw_worm(tax, path[k], SLATE, lw=1.2, ms=2, alpha=0.5)
        tax.text(0.5, -0.12, f"t = {k / 10:.1f}", transform=tax.transAxes, ha="center", va="top",
                 fontsize=10, color=MUTED)
        thumbs.append(ln)
    headline(fig, "Walking the geodesic from the mean worm to a strongly bent worm", size=13, y=0.95, x=0.02,
             sub=f"All 11 steps t = 0, 0.1, …, 1 below. Geodesic distance ρ = {rho:.3f} rad; every step has norm 1.")
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        set_xy(moving, fine[i])
        recolour(moving, blend(INK, TEAL, t))
        readout.set_text(f"t = {t:.2f}")
        for k, ln in enumerate(thumbs):
            reached = k / 10 <= t + 1e-9
            recolour(ln, blend(INK, TEAL, k / 10) if reached else SLATE)
            ln.set_alpha(1.0 if reached else 0.5)
            ln.set_linewidth(1.8 if reached else 1.2)
        frames.append(grab_frame(fig))
    plt.close(fig)
    return save_gif(frames, "04_geodesic_morph.gif", hold=HOLD + 6)


def fig_05_means():
    d = data()
    plain = d.Z_al.mean(axis=0)  # not renormalised: shorter than 1
    plain_norm = np.linalg.norm(plain)
    gpa = align(gpa_mean(d.Z, seed=SEED), d.mean)
    gap = geodesic_distance(gpa, d.mean)
    gap_text = f"{gap:.1e}" if gap > 1e-9 else "< 1e-9"

    fig = figure(1200, 560)
    ax = fig.add_axes([0.02, 0.10, 0.62, 0.66])
    shape_axes(ax, (-0.56, 0.56), (-0.3, 0.3))
    draw_worm(ax, gpa, GOLD, lw=7, ms=12, alpha=0.9, zorder=3)
    draw_worm(ax, d.mean, INK, lw=2.6, ms=6, zorder=5)
    draw_worm(ax, plain, SLATE, lw=2.6, ms=6, alpha=0.9, zorder=4)
    zoom = fig.add_axes([0.68, 0.14, 0.30, 0.58])
    x0, y0 = d.mean[-1].real, d.mean[-1].imag
    half = 0.045
    zoom.set_xlim(x0 - 1.6 * half, x0 + 0.6 * half)
    zoom.set_ylim(y0 + half, y0 - half)
    zoom.set_aspect("equal")
    zoom.set_xticks([])
    zoom.set_yticks([])
    for s in zoom.spines.values():
        s.set_visible(True)
        s.set_color("#B9B3AA")
    draw_worm(zoom, gpa, GOLD, lw=12, ms=22, alpha=0.9, zorder=3)
    draw_worm(zoom, d.mean, INK, lw=3, ms=10, zorder=5)
    draw_worm(zoom, plain, SLATE, lw=3, ms=10, alpha=0.9, zorder=4)
    zoom.set_title("zoom on the right tip", fontsize=11)
    zoom.annotate("", xy=(d.mean[-1].real, y0 - 0.02), xytext=(plain[-1].real, y0 - 0.02),
                  arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.2, shrinkA=0, shrinkB=0))
    zoom.text(0.5 * (d.mean[-1].real + plain[-1].real), y0 - 0.025, f"{d.mean[-1].real - plain[-1].real:.3f}",
              ha="center", va="bottom", color=SLATE, fontsize=10.5)
    ax.add_patch(patches.Rectangle((x0 - 1.6 * half, y0 - half), 2.2 * half, 2 * half, fill=False, ec="#B9B3AA", lw=1))
    fig.legend(handles=[handle(INK, 2.6, label="Fréchet mean (top eigenvector of S)"),
                        handle(SLATE, 2.6, label=f"plain average of the aligned worms (norm {plain_norm:.4f})"),
                        handle(GOLD, 6, None, label=f"GPA result ({gap_text} rad from the Fréchet mean)")],
               loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=1, handlelength=2.2)
    headline(fig, "Three means of the same 1,377 worms, overlaid",
             sub="The GPA mean is drawn wide so the Fréchet mean can sit on top of it; the plain average points the same way but is 3.6 % shorter.")
    return save_png(fig, "05_means.png")


def fig_05_power_iteration():
    d = data()
    runs = []
    for seed in GPA_SEEDS:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(len(d.Z)))
        m, hist = gpa_mean(d.Z, seed=seed, return_history=True)
        hist = align_dataset(hist, d.mean)  # rotate the whole history onto the Frechet mean
        dist = np.array([geodesic_distance(h, d.mean) for h in hist])
        runs.append((seed, start, hist, dist))
    floor = 1e-9
    last_visible = max(int(np.max(np.flatnonzero(r[3] > floor))) for r in runs)
    n_rounds = last_visible + 2
    colours = [LAVENDER, GOLD, TEAL]
    per_round = 6

    fig = figure(900, 560, GIF_DPI)
    tops = [fig.add_axes([0.02 + k * 0.325, 0.48, 0.31, 0.32]) for k in range(3)]
    ax_c = fig.add_axes([0.09, 0.10, 0.88, 0.29])
    lines, labels_r = [], []
    for ax, (seed, start, hist, dist), c in zip(tops, runs, colours):
        shape_axes(ax, (-0.6, 0.6), (-0.42, 0.42))
        draw_worm(ax, d.mean, INK, lw=1.8, ms=4, zorder=5, ls="--")
        draw_worm(ax, hist[0], SLATE, lw=1.4, ms=3, alpha=0.5, zorder=2)
        ln = draw_worm(ax, hist[0], c, lw=2.8, ms=6, zorder=4)
        lines.append(ln)
        ax.set_title(f"start: worm {start}", fontsize=11.5)
        labels_r.append(ax.text(0.5, 0.0, "", transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=11))
    ax_c.set_yscale("log")
    ax_c.set_ylim(3e-10, 3)
    ax_c.set_xlim(-0.3, n_rounds - 0.7)
    ax_c.set_xlabel("GPA round")
    ax_c.set_ylabel("distance to Fréchet mean (rad)")
    ax_c.grid(True, alpha=0.6, which="major")
    ax_c.set_xticks(range(n_rounds))
    ax_c.axhline(floor, color=LINE, lw=1)
    ax_c.text(-0.2, floor * 1.8, "floor: numerically identical to the mean", ha="left", va="bottom", color=MUTED, fontsize=10)
    tracks = []
    for (seed, start, hist, dist), c in zip(runs, colours):
        (t,) = ax_c.plot([], [], "-o", color=c, lw=2, ms=6, label=f"start worm {start}")
        tracks.append(t)
    ax_c.text(0.99, 0.95, "each round shrinks the error by about λ₂/λ₁ ≈ 0.05",
              transform=ax_c.transAxes, ha="right", va="top", color=MUTED, fontsize=10.5)
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 0.13), ncol=1, handlelength=1.6, labelspacing=0.2)
    headline(fig, "GPA from three random starting worms converges to the Fréchet mean", size=13, y=0.95, x=0.02,
             sub="Dashed ink: the Fréchet mean. Slate: the starting worm. Each round is rotated onto the mean for display.")
    frames = []
    for r in range(n_rounds):
        for (seed, start, hist, dist), ln, lab, tr in zip(runs, lines, labels_r, tracks):
            rr = min(r, len(hist) - 1)
            set_xy(ln, hist[rr])
            tr.set_data(np.arange(rr + 1), np.maximum(dist[: rr + 1], floor))
            lab.set_text(f"round {rr}: {dist[rr]:.1e} rad" if dist[rr] > floor else f"round {rr}: converged")
        frames.extend([grab_frame(fig)] * per_round)
    plt.close(fig)
    return save_gif(frames, "05_power_iteration.gif")


def fig_06_tangent_plane():
    fig = figure(1200, 780)
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.74])
    ax.set_xlim(-1.1, 1.9)
    ax.set_ylim(-1.12, 1.12)
    view = SphereView(ax)
    view.draw_sphere()
    axis = np.array([0.0, 0.92, 0.39])
    view.draw_circle(axis)
    view.draw_circle(np.cross(axis, [1.0, 0, 0]))
    m, e1, e2 = view.tangent_basis([0.10, 0.62, 0.78], twist_deg=20)
    s = 0.46
    corners = np.array([m + a * e1 + b * e2 for a, b in ((-s, -s), (s, -s), (s, s), (-s, s))])
    ax.add_patch(patches.Polygon(corners[:, :2], closed=True, fc=LAVENDER, ec="none", alpha=0.22, zorder=6))
    ax.add_patch(patches.Polygon(corners[:, :2], closed=True, fill=False, ec=LAVENDER, lw=1.6, zorder=6))
    a, b = 0.74, -0.44  # tangent vector of the worm, in the plane basis
    rho = np.hypot(a, b)
    z = view.exp(m, e1, e2, a, b)
    v = m + a * e1 + b * e2
    view.draw_arc(m, z, GOLD, lw=3.0, zorder=5)
    ax.plot([m[0], v[0]], [m[1], v[1]], color=GOLD, lw=3.0, zorder=7, solid_capstyle="round")
    ax.scatter([m[0]], [m[1]], s=80, color=INK, zorder=9, edgecolor=PAPER, linewidths=1.2)
    ax.scatter([z[0]], [z[1]], s=70, color=TEAL, zorder=9, edgecolor=INK, linewidths=1.0)
    ax.scatter([v[0]], [v[1]], s=70, color=TEAL, zorder=9, edgecolor=INK, linewidths=1.0)
    ax.add_patch(FancyArrowPatch((z[0], z[1]), (v[0], v[1]), connectionstyle="arc3,rad=0.5", arrowstyle="-|>",
                                 mutation_scale=18, color=INK, lw=2.0, zorder=10, shrinkA=9, shrinkB=9))
    ax.add_patch(FancyArrowPatch((v[0], v[1]), (z[0], z[1]), connectionstyle="arc3,rad=0.5", arrowstyle="-|>",
                                 mutation_scale=18, color=LAVENDER, lw=2.0, zorder=10, shrinkA=9, shrinkB=9))
    mid = 0.5 * (z + v)
    ax.text(mid[0] + 0.30, mid[1] - 0.02, "log map\nsphere → sheet", color=INK, ha="left", va="center", fontsize=11.5)
    ax.text(mid[0] - 0.30, mid[1] + 0.0, "exp map\nsheet → sphere", color=LAVENDER, ha="right", va="center", fontsize=11.5)
    ax.annotate("mean m: where the sheet touches", (m[0], m[1]), xytext=(-135, 30), textcoords="offset points",
                color=INK, ha="center", fontsize=11, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("a worm z, on the sphere", (z[0], z[1]), xytext=(60, -80), textcoords="offset points",
                color=INK, ha="center", fontsize=11, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("its tangent vector\nv = log_m(z), on the sheet,\nlength ρ", (v[0], v[1]), xytext=(40, 40),
                textcoords="offset points", color=INK, ha="left", fontsize=11,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fit_text(fig, 0.03, 0.085, f"gold: arc m → z on the sphere and segment m → v on the sheet, both of length "
                               f"ρ = {rho:.2f}", color=GOLD, width=0.94)
    fit_text(fig, 0.03, 0.035, "lavender square: the tangent space at m, a flat copy of the neighbourhood of the mean",
             color=LAVENDER, width=0.94)
    headline(fig, "A flat sheet pressed against the sphere at the mean",
             sub="Log map: sphere → sheet, where PCA works. Exp map: sheet → sphere.")
    return save_png(fig, "06_tangent_plane.png")


def flat_pca_ratio(d: Data) -> np.ndarray:
    """PCA straight on the aligned coordinates (no log map), for comparison."""
    R = np.concatenate([d.Z_al.real, d.Z_al.imag], axis=1)
    R = R - R.mean(axis=0)
    s = np.linalg.svd(R, compute_uv=False)
    var = s**2
    return var / var.sum()


def fig_06_scree():
    d = data()
    tangent = d.pca.explained_variance_ratio_[:10] * 100
    flat = flat_pca_ratio(d)[:10] * 100
    cum = np.cumsum(tangent)
    x = np.arange(1, 11)
    fig = figure(1200, 540)
    ax = fig.add_axes([0.08, 0.14, 0.9, 0.68])
    w = 0.38
    ax.bar(x - w / 2, tangent, w, color=INK, label="tangent PCA (log map first)", zorder=3)
    ax.bar(x + w / 2, flat, w, color=SLATE, alpha=0.5, label="flat PCA on the aligned coordinates (no log map)", zorder=3)
    for xi, t in zip(x, tangent):
        ax.text(xi - w / 2, t + 0.8, f"{t:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
    for xi, f in zip(x[:4], flat[:4]):
        ax.text(xi + w / 2, f + 0.8, f"{f:.1f}", ha="center", va="bottom", fontsize=10, color=SLATE)
    ax.set_xticks(x)
    ax.set_xlabel("mode")
    ax.set_ylabel("variance explained (%)")
    ax.set_ylim(0, 78)
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    ax.legend(loc="upper right")
    ax.text(0.98, 0.56, f"cumulative (tangent PCA): {cum[0]:.1f} → {cum[1]:.1f} → {cum[2]:.1f} → {cum[3]:.1f} % "
                        f"after four modes;\n{cum[9]:.1f} % after ten. Only 20 of the 24 modes can carry variance.",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=MUTED)
    headline(fig, "Mode 1, bending, carries two thirds of the variance", x=0.08)
    return save_png(fig, "06_scree.png")


def fig_06_modes():
    d = data()
    fig = figure(1200, 760)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.09, wspace=0.05, hspace=0.2)
    ratio = d.pca.explained_variance_ratio_
    for i, ax in enumerate(axes.flat):
        plus, minus = d.pca.mode_shape(i, 1.0)
        shape_axes(ax, (-0.6, 0.6), (-0.3, 0.3))
        draw_worm(ax, minus, LAVENDER, lw=2.4, ms=5.5, zorder=3)
        draw_worm(ax, plus, GOLD, lw=2.4, ms=5.5, zorder=3)
        draw_worm(ax, d.mean, INK, lw=2.4, ms=5.5, zorder=4)
        ax.set_title(f"Mode {i + 1}: {ratio[i] * 100:.1f} %, 1 SD = {d.pca.sdev_[i]:.3f} rad")
    fig.legend(handles=[handle(GOLD, label="+1 SD"), handle(INK, label="mean"), handle(LAVENDER, label="−1 SD")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    headline(fig, "The first four modes, one standard deviation each way",
             sub="modeᵢ± = exp_m(± sᵢ cᵢ); all 1,377 worms, Fréchet mean as the base point")
    return save_png(fig, "06_modes.png")


def fig_06_mode_sweep():
    d = data()
    n = 48
    sd = d.pca.sdev_[0]
    fig = figure(900, 420, GIF_DPI)
    ax = fig.add_axes([0.02, 0.2, 0.96, 0.66])
    shape_axes(ax, (-0.62, 0.62), (-0.4, 0.4))
    draw_worm(ax, d.mean, SLATE, lw=1.6, ms=3.5, alpha=0.5, zorder=2)
    moving = draw_worm(ax, d.mean, LAVENDER, lw=3.2, ms=7, zorder=5)
    ax_s = fig.add_axes([0.2, 0.09, 0.6, 0.06])
    ax_s.set_xlim(-2.3, 2.3)
    ax_s.set_ylim(-1, 1)
    ax_s.set_axis_off()
    ax_s.plot([-2, 2], [0, 0], color=LINE, lw=4, solid_capstyle="round")
    for v in (-2, -1, 0, 1, 2):
        ax_s.plot([v, v], [-0.35, 0.35], color=MUTED, lw=1)
        ax_s.text(v, -0.9, f"{v:+d} SD" if v else "mean", ha="center", va="top", fontsize=10.5, color=MUTED)
    (knob,) = ax_s.plot([-2], [0], "o", color=GOLD, ms=14, markeredgecolor=INK, zorder=5)
    readout = ax.text(0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=12.5)
    headline(fig, "Mode 1 swept from −2 SD to +2 SD and back", size=13, y=0.95, x=0.02,
             sub=f"shape = exp_m(w · s₁ c₁), s₁ = {sd:.3f} rad; the faint worm is the mean")
    frames = []
    for i in range(n):
        w = -2 * np.cos(2 * np.pi * i / n)
        set_xy(moving, d.pca.inverse_transform(np.array([w * sd])))
        knob.set_data([w], [0])
        readout.set_text(f"w = {w:+.2f} SD")
        frames.append(grab_frame(fig))
    plt.close(fig)
    return save_gif(frames, "06_mode_sweep.gif")


def fig_06_pca_scatter():
    d = data()
    s = d.scores
    ratio = d.pca.explained_variance_ratio_
    fig = figure(1200, 700)
    ax = fig.add_axes([0.09, 0.12, 0.88, 0.76])
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(s))
    for cls in (0, 1):
        sel = order[d.labels[order] == cls]
        ax.scatter(s[sel, 0], s[sel, 1], s=16, color=CLASS_COLOUR[cls], alpha=0.6, linewidths=0,
                   label=f"{CLASS_NAME[cls]} ({sel.size})", zorder=3)
    ax.axhline(0, color=LINE, lw=1, zorder=1)
    ax.axvline(0, color=LINE, lw=1, zorder=1)
    ax.set_xlabel(f"mode 1 score (rad), {ratio[0] * 100:.1f} % of the variance: bending")
    ax.set_ylabel(f"mode 2 score (rad), {ratio[1] * 100:.1f} %")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left", markerscale=1.8)
    ax.text(0.98, 0.04, "dense core: nearly straight worms of both classes\narch: strongly bent live worms",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=11)
    headline(fig, "Every worm on the first two modes, coloured by class", x=0.09)
    return save_png(fig, "06_pca_scatter.png")


def fig_06_dead_vs_live():
    d = data()
    fig = figure(1200, 1040)
    axes = fig.subplots(3, 2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.06, wspace=0.05, hspace=0.3)
    for col, cls in enumerate((0, 1)):
        Zc = d.Z[d.labels == cls]
        mean_c = align(frechet_mean(Zc), d.mean)
        pca_c = TangentPCA(mean=mean_c).fit(Zc)
        colour = CLASS_COLOUR[cls]
        fig.text(0.27 + 0.48 * col, 0.855, f"{CLASS_NAME[cls].capitalize()} worms ({Zc.shape[0]})", ha="center",
                 va="center", fontsize=13, color=colour, weight="bold")
        ax = axes[0, col]
        shape_axes(ax, (-0.6, 0.6), (-0.3, 0.3))
        draw_worm(ax, d.mean, SLATE, lw=1.4, ms=3, alpha=0.5, zorder=2)
        draw_worm(ax, mean_c, colour, lw=2.8, ms=6, zorder=4)
        ax.set_title(f"mean: {geodesic_distance(mean_c, d.mean):.3f} rad from the overall mean")
        for row in (1, 2):
            i = row - 1
            plus, minus = pca_c.mode_shape(i, 1.0)
            ax = axes[row, col]
            shape_axes(ax, (-0.6, 0.6), (-0.3, 0.3))
            draw_worm(ax, mean_c, SLATE, lw=1.4, ms=3, alpha=0.5, zorder=2)
            draw_worm(ax, minus, colour, lw=2.4, ms=5.5, zorder=3, ls="--")
            draw_worm(ax, plus, colour, lw=2.4, ms=5.5, zorder=4)
            ax.set_title(f"mode {i + 1}: {pca_c.explained_variance_ratio_[i] * 100:.1f} %, 1 SD = {pca_c.sdev_[i]:.3f} rad")
    fig.legend(handles=[handle(INK, label="+1 SD"), handle(INK, ls="--", label="−1 SD"),
                        handle(SLATE, 1.6, alpha=0.6, label="overall mean (row 1) / class mean (rows 2, 3)")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    headline(fig, "Dead worms barely bend; live worms bend about nine times more",
             sub="Own Fréchet mean and tangent PCA per class; same axes, so amplitudes compare.")
    return save_png(fig, "06_dead_vs_live.png")


def fig_06_sampling():
    d = data()
    samples = d.pca.sample(6, n_modes=3, seed=SEED)
    lengths = [np.linalg.norm(log_map(d.mean, z)) for z in samples]
    fig = figure(1200, 640)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.70, bottom=0.03, wspace=0.05, hspace=0.35)
    for k, (ax, z) in enumerate(zip(axes.flat, samples)):
        shape_axes(ax, (-0.6, 0.6), (-0.32, 0.32))
        draw_worm(ax, d.mean, SLATE, lw=1.4, ms=3, alpha=0.5, zorder=2)
        draw_worm(ax, z, LAVENDER, lw=2.6, ms=6, zorder=4)
        ax.set_title(f"sample {k + 1}: {lengths[k]:.3f} rad away")
    headline(fig, "Six worms that never existed",
             sub="z = exp_m(Σ wᵢ sᵢ cᵢ), wᵢ ~ N(0, 1) on the first 3 modes, sᵢ the data's own standard "
                 "deviations (no √(n−1) factor). Faint: the mean.")
    return save_png(fig, "06_sampling.png")


def fig_07_bspline_basis():
    x = np.linspace(-3, 8, 1101)
    fig = figure(1200, 780)
    ax1 = fig.add_axes([0.08, 0.60, 0.9, 0.24])
    ax2 = fig.add_axes([0.08, 0.10, 0.9, 0.36])
    beta = cubic_bspline(x)
    ax1.plot(x, beta, color=INK, lw=2.6)
    ax1.fill_between(x, 0, beta, color=INK, alpha=0.06)
    for xv, yv, lab in ((0, 2 / 3, "2/3"), (1, 1 / 6, "1/6"), (-1, 1 / 6, "1/6"), (2, 0, "0"), (-2, 0, "0")):
        ax1.plot([xv], [yv], "o", color=GOLD, ms=7, markeredgecolor=INK, zorder=5)
        ax1.annotate(lab, (xv, yv), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10.5, color=INK)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.03, 0.85)
    ax1.set_xticks(range(-3, 4))
    ax1.grid(True, alpha=0.6)
    ax1.set_ylabel("β(x)")
    ax1.set_title("The cubic B-spline bump β: four cubic pieces, 4 units wide, smooth at the joins", loc="left")
    for k in range(-2, 8):
        ax2.plot(x, cubic_bspline(x - k), color=LAVENDER, lw=1.6, alpha=0.8)
    total = sum(cubic_bspline(x - k) for k in range(-5, 12))
    ax2.plot(x, total, color=GOLD, lw=3.2, zorder=5)
    ax2.set_xlim(-1, 6)
    ax2.set_ylim(-0.03, 1.15)
    ax2.set_xticks(range(-1, 7))
    ax2.grid(True, alpha=0.6)
    ax2.set_xlabel("t  (position along the curve, in control-point spacings)")
    ax2.set_ylabel("weight")
    ax2.set_title("Copies shifted by whole numbers, β(t − k), and their sum", loc="left")
    ax2.text(-0.9, 1.04, "sum of all copies = 1 everywhere (partition of unity)", ha="left", va="bottom", color=GOLD, fontsize=11)
    ax2.annotate("at t = 2.5 four bumps are non-zero:\n23/48 + 23/48 + 1/48 + 1/48 = 1", xy=(2.5, 23 / 48), xytext=(2.95, 0.80),
                 fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax2.axvline(2.5, color=MUTED, lw=0.8, ls=":")
    headline(fig, "One bump, its shifted copies, and their sum", x=0.08)
    return save_png(fig, "07_bspline_basis.png")


def fig_07_spline_fit():
    d = data()
    pts = from_complex(d.aligned(HERO_LIVE))
    fig = figure(1200, 540)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.16, wspace=0.05)
    for ax, M in zip(axes, (6, 12)):
        spline = BSplineCurve(M, closed=False).fit(pts)
        curve = spline.sample(300)
        C = spline.control_points_
        resid = np.linalg.norm(spline.evaluate(spline.t_fit_) - pts, axis=1).max()
        shape_axes(ax, (-0.62, 0.62), (-0.38, 0.38))
        ax.plot(C[:, 0], C[:, 1], "--", color=LAVENDER, lw=1.6, zorder=2)
        ax.plot(C[:, 0], C[:, 1], "s", color=LAVENDER, ms=7, zorder=5, markeredgecolor=PAPER)
        ax.plot(curve[:, 0], curve[:, 1], color=INK, lw=2.6, zorder=4)
        ax.plot(pts[:, 0], pts[:, 1], "o", color=CORAL, ms=6.5, zorder=6)
        how = "least-squares fit" if M != len(pts) else "exact interpolation"
        ax.set_title(f"M = {M}: {how}")
        ax.text(0.5, 0.0, f"largest miss at a measured point: {resid:.4f}", transform=ax.transAxes, ha="center",
                va="bottom", color=MUTED, fontsize=10.5)
    fig.legend(handles=[plt.Line2D([], [], color=CORAL, ls="", marker="o", label="12 measured landmarks"),
                        handle(LAVENDER, 1.6, "s", "--", label="control points and polygon"),
                        handle(INK, 2.6, None, label="fitted cubic B-spline")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    headline(fig, "The same 12 landmarks fitted with 6 and with 12 control points",
             sub="Worm 622. M = 12 interpolates every landmark; M = 6 only passes near them.")
    return save_png(fig, "07_spline_fit.png")


def fig_07_local_control():
    d = data()
    pts = from_complex(d.aligned(HERO_LIVE))
    M = 12
    spline = BSplineCurve(M, closed=False).fit(pts)
    C0 = spline.control_points_.copy()
    k = 6
    n = 48
    t_dense = np.linspace(0, M - 1, 400)
    B = basis_matrix(t_dense, M)
    touched = (t_dense >= k - 2) & (t_dense <= k + 2)
    fig = figure(900, 480, GIF_DPI)
    ax = fig.add_axes([0.02, 0.16, 0.96, 0.70])
    shape_axes(ax, (-0.62, 0.62), (-0.42, 0.42))
    ax.plot((B @ C0)[:, 0], (B @ C0)[:, 1], color=SLATE, lw=1.8, alpha=0.5, zorder=2)
    (poly,) = ax.plot(C0[:, 0], C0[:, 1], "--", color=LAVENDER, lw=1.5, zorder=3)
    (ctrl,) = ax.plot(C0[:, 0], C0[:, 1], "s", color=LAVENDER, ms=7, markeredgecolor=PAPER, zorder=6)
    (curve,) = ax.plot([], [], color=INK, lw=2.8, zorder=4)
    (hot,) = ax.plot([], [], color=GOLD, lw=4.5, zorder=5, solid_capstyle="round")
    (knob,) = ax.plot([], [], "s", color=GOLD, ms=12, markeredgecolor=INK, zorder=8)
    for j in (k - 2, k + 2):
        p = (B @ C0)[np.argmin(np.abs(t_dense - j))]
        ax.plot([p[0]], [p[1]], "|", color=INK, ms=14, mew=1.6, zorder=7)
        ax.annotate(f"t = {j}", p, xytext=(-14 if j == k - 2 else 14, -4), textcoords="offset points",
                    ha="right" if j == k - 2 else "left", va="center", fontsize=10.5, color=INK)
    ax.legend(handles=[handle(SLATE, 1.8, None, alpha=0.6, label="curve before"),
                       handle(INK, 2.6, None, label="curve now"),
                       handle(GOLD, 4, None, label=f"the part that can move: t in [{k - 2}, {k + 2}]"),
                       handle(LAVENDER, 1.5, "s", "--", label="control points")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, handlelength=2)
    headline(fig, "Local control: dragging control point c₆ moves only the four segments it touches", size=13, y=0.95, x=0.02,
             sub="Open cubic B-spline with 12 control points through worm 622. Beyond two knots away the curve does not notice.")
    frames = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        C = C0.copy()
        C[k] += 0.09 * np.array([np.cos(ang + np.pi / 2), -np.sin(ang + np.pi / 2)])
        xy = B @ C
        curve.set_data(xy[:, 0], xy[:, 1])
        hot.set_data(xy[touched, 0], xy[touched, 1])
        poly.set_data(C[:, 0], C[:, 1])
        ctrl.set_data(C[:, 0], C[:, 1])
        knob.set_data([C[k, 0]], [C[k, 1]])
        frames.append(grab_frame(fig))
    plt.close(fig)
    return save_gif(frames, "07_local_control.gif")


# ----------------------------------------------------------------- registry
FIGURES = {
    "hero.gif": fig_hero,
    "01_pipeline.png": fig_01_pipeline,
    "01_arclength_resampling.png": fig_01_arclength,
    "01_representations.png": fig_01_representations,
    "02_preshape_steps.png": fig_02_preshape_steps,
    "02_sphere.png": fig_02_sphere,
    "03_rotation_search.gif": fig_03_rotation_search,
    "03_votes.png": fig_03_votes,
    "04_distances.png": fig_04_distances,
    "04_geodesic_morph.gif": fig_04_geodesic_morph,
    "05_means.png": fig_05_means,
    "05_power_iteration.gif": fig_05_power_iteration,
    "06_tangent_plane.png": fig_06_tangent_plane,
    "06_scree.png": fig_06_scree,
    "06_modes.png": fig_06_modes,
    "06_mode_sweep.gif": fig_06_mode_sweep,
    "06_pca_scatter.png": fig_06_pca_scatter,
    "06_dead_vs_live.png": fig_06_dead_vs_live,
    "06_sampling.png": fig_06_sampling,
    "07_bspline_basis.png": fig_07_bspline_basis,
    "07_spline_fit.png": fig_07_spline_fit,
    "07_local_control.gif": fig_07_local_control,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="make only these figures (file name, with or without extension)")
    parser.add_argument("--list", action="store_true", help="list the figure names and exit")
    args = parser.parse_args(argv)
    if args.list:
        for name in FIGURES:
            print(name)
        return 0
    names = list(FIGURES)
    if args.only:
        names = []
        for n in args.only:
            matches = [f for f in FIGURES if f == n or f.rsplit(".", 1)[0] == n]
            if not matches:
                parser.error(f"unknown figure {n!r}; use --list to see the names")
            names += matches
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    t_all = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        path = FIGURES[name]()
        size = path.stat().st_size
        flag = "  (over 3 MB!)" if name.endswith(".gif") and size > 3 * 1024 * 1024 else ""
        print(f"{name:32s} {size / 1024:8.0f} kB  {time.perf_counter() - t0:5.1f} s{flag}")
    print(f"done: {len(names)} figures in {time.perf_counter() - t_all:.1f} s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
