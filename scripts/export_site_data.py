"""Write the precomputed numbers that the interactive demos in docs/ use.

Outputs (all floats rounded to 5 decimals, everything seeded with 0):

- ``docs/data/worms.json``: the Fréchet mean worm, three dead and three
  live example preshapes, the tangent-PCA numbers for the ``pca-modes``
  demo, and the per-class means.
- ``docs/data/spline.json``: one worm as 20 landmarks and the M = 6
  control points of the least-squares cubic B-spline through them.

Run from the repository root::

    python scripts/export_site_data.py [--out docs/data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapekit import (  # noqa: E402
    BSplineCurve,
    TangentPCA,
    align,
    frechet_mean,
    from_complex,
    load_celegans,
    preshape,
    resample_by_arclength,
    to_complex,
)

SEED = 0
DECIMALS = 5


def rounded(obj):
    """Recursively turn numpy arrays into lists of floats rounded to 5 decimals."""
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return rounded(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), DECIMALS)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def xy(z: np.ndarray) -> np.ndarray:
    """Complex (N,) -> (N, 2) list of [x, y] rows."""
    return from_complex(z)


def export_worms(points: np.ndarray, labels: np.ndarray) -> dict:
    Z = to_complex(preshape(points))
    mean = frechet_mean(Z)
    rng = np.random.default_rng(SEED)
    dead_idx = rng.choice(np.flatnonzero(labels == 0), 3, replace=False)
    live_idx = rng.choice(np.flatnonzero(labels == 1), 3, replace=False)

    pca = TangentPCA(mean=mean).fit(Z)
    class_means = {}
    for name, cls in (("dead", 0), ("live", 1)):
        m_cls = frechet_mean(Z[labels == cls])
        class_means[name] = xy(align(m_cls, mean))  # rotated onto the global mean for overlay

    return {
        "note": (
            "All shapes are preshapes (centred, unit Frobenius norm) in image orientation "
            "(x right, y down). 'examples' keep their original rotation; 'class_means' are "
            "rotated onto 'mean'. Tangent vectors are 24 reals: 12 x-parts then 12 y-parts. "
            "A shape at mode scores w_i (in SD units) is exp_map(mean, mean_tangent + sum_i "
            "w_i * sdev[i] * components[i])."
        ),
        "n_landmarks": int(points.shape[1]),
        "mean": xy(mean),
        "examples": {
            "dead": [xy(Z[i]) for i in dead_idx],
            "live": [xy(Z[i]) for i in live_idx],
            "indices": {"dead": dead_idx.tolist(), "live": live_idx.tolist()},
        },
        "pca": {
            "tangent_layout": "re_then_im",
            "mean_tangent": pca.mean_tangent_,
            "components": pca.components_[:4],
            "sdev": pca.sdev_[:4],
            "explained": pca.explained_variance_ratio_[:10],
        },
        "class_means": class_means,
    }


def export_spline(points: np.ndarray, index: int = 0, n_landmarks: int = 20, M: int = 6) -> dict:
    """One worm as 20 landmarks and the M = 6 control points fitted to them.

    The dataset stores 12 landmarks per worm.  To get 20, the 12 points are
    interpolated by a cubic B-spline (exact, 12 control points), that curve
    is sampled densely and resampled to 20 points evenly spaced along it.
    """
    twelve = points[index]
    dense = BSplineCurve(len(twelve), closed=False).fit(twelve).sample(400)
    landmarks = resample_by_arclength(dense, n_landmarks)
    curve = BSplineCurve(M, closed=False).fit(landmarks)
    return {
        "note": (
            f"Worm {index} of celegans_points.npy, image pixel coordinates (x right, y down). "
            "'landmarks' are 20 points evenly spaced along the worm; 'control_points' are the "
            "least-squares cubic B-spline with M control points, open curve with mirror ends "
            "(c[-1] = c[1], c[M] = c[M-2]), parameter t in [0, M-1]: curve(t) = sum_k c_k beta(t-k)."
        ),
        "worm_index": index,
        "M": M,
        "closed": False,
        "boundary": "mirror",
        "landmarks": landmarks,
        "control_points": curve.control_points_,
        "curve": curve.sample(101),
        "twelve_landmarks": twelve,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs" / "data"), help="output folder")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points, labels = load_celegans()
    with open(out / "worms.json", "w") as f:
        json.dump(rounded(export_worms(points, labels)), f, separators=(",", ":"))
    with open(out / "spline.json", "w") as f:
        json.dump(rounded(export_spline(points)), f, separators=(",", ":"))


if __name__ == "__main__":
    main()
