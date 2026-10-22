"""Loaders for the two datasets shipped in ``data/``.

Both return landmarks with the shapekit axis convention: ``(K, N, 2)`` with
``(x, y)`` columns in image pixel units (x = column, y = row).

Explained in chapter 00.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Default location of the data folder: ``<repo>/data``.  Override with the
#: environment variable ``SHAPEKIT_DATA`` or the ``data_dir`` argument.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """Resolve the folder that holds the ``.npy`` / ``.npz`` files.

    Order of preference: the ``data_dir`` argument, then the environment
    variable ``SHAPEKIT_DATA``, then ``<repo>/data``.
    """
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("SHAPEKIT_DATA")
    return Path(env) if env else DEFAULT_DATA_DIR


def load_celegans(data_dir: str | os.PathLike | None = None) -> tuple[NDArray, NDArray]:
    """The C. elegans landmark dataset (BBBC010, 12 points per worm).

    - ``points``: ``(1377, 12, 2)`` float64, ``(x, y)`` pixel coordinates
      of 12 landmarks evenly spaced along each worm's midline, head to
      tail (which end is the head is arbitrary).
    - ``labels``: ``(1377,)`` int, 0 = dead, 1 = live.

    Explained in chapter 00.
    """
    folder = resolve_data_dir(data_dir)
    points = np.load(folder / "celegans_points.npy").astype(np.float64)
    labels = np.load(folder / "celegans_labels.npy").astype(int)
    return points, labels


def load_mef_subset(
    data_dir: str | os.PathLike | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """A subset of the mouse embryonic fibroblast (MEF) outlines.

    - ``points``: ``(1200, 120, 2)`` float32, ``(x, y)`` pixel coordinates
      of 120 points on each cell outline (unordered point cloud).
    - ``labels``: ``(1200,)`` int, 0 = wild-type, 1 = LMNA-/- (lamin A/C
      knock-out).
    - ``keys``: ``(1200,)`` strings naming the source image and cell.

    Explained in chapter 00 (used in chapter 08).
    """
    folder = resolve_data_dir(data_dir)
    with np.load(folder / "mef_outlines_subset.npz") as d:
        points = d["points"].astype(np.float32)
        labels = d["labels"].astype(int)
        keys = d["keys"].astype(str)
    return points, labels, keys
