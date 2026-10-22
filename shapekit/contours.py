"""From a mask to ordered points: outlines, midlines and even spacing.

Image arrays are indexed ``mask[row, col]``; row grows downward and
corresponds to ``y``, column to ``x``.  Every function here returns
points as ``(N, 2)`` arrays of ``(x, y)`` rows, i.e. ``(col, row)``, so
they plot correctly with ``plt.plot(points[:, 0], points[:, 1])`` and an
inverted y axis.

Explained in chapter 01.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


def mask_to_contours(mask: ArrayLike, min_len: int = 0, level: float = 0.5) -> list[NDArray]:
    """Trace the outline of every object in a mask.

    Method: marching squares.  Think of the mask as a landscape whose
    height at each pixel is the pixel's value; the outline is the line
    where the height crosses ``level`` (0.5 for a 0/1 mask), traced
    through each 2 x 2 block of pixel centres (``contourpy``, which ships
    with matplotlib).

    - ``mask``: 2-D array.  A binary mask (0/1 or bool) is traced once.
      An instance mask (0 = background, 1, 2, ... = one number per
      object) is traced one object at a time.
    - ``min_len``: outlines with fewer points than this are dropped (small
      fragments of segmentation debris).
    - ``level``: the height at which the outline is drawn.

    Returns a list of ``(N, 2)`` arrays of ``(x, y)`` points, each an
    ordered loop (first point not repeated at the end).

    Explained in chapter 01.
    """
    import contourpy

    mask = np.asarray(mask)
    labels = np.unique(mask[mask != 0])
    if mask.dtype == bool or len(labels) <= 1:
        layers = [mask.astype(float)]
    else:
        layers = [(mask == lab).astype(float) for lab in labels]
    contours: list[NDArray] = []
    for layer in layers:
        gen = contourpy.contour_generator(z=layer, line_type="Separate")
        for line in gen.lines(level):
            line = np.asarray(line, dtype=float)
            if len(line) > 1 and np.allclose(line[0], line[-1]):
                line = line[:-1]
            if len(line) >= min_len:
                contours.append(line)
    return contours


# Zhang-Suen thinning, written with 8-neighbour lookups on a padded array.
def skeletonize(mask: ArrayLike) -> NDArray:
    """Peel an object down to a one-pixel-wide midline (Zhang-Suen thinning).

    Method: repeat two passes until nothing changes.  In each pass a
    boundary pixel is removed when (a) it has 2 to 6 object neighbours,
    (b) going around its 8 neighbours there is exactly one 0 -> 1
    change, and (c)/(d) two side conditions that remove pixels from the
    north/east sides in pass 1 and south/west sides in pass 2, so the
    line ends up in the middle.

    - ``mask``: 2-D binary array (bool or 0/1).

    Returns a bool array of the same shape with the skeleton pixels True.

    Explained in chapter 01.
    """
    img = np.asarray(mask).astype(bool).copy()
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            # Eight neighbours, clockwise from north: P2 .. P9.
            P2 = p[:-2, 1:-1]
            P3 = p[:-2, 2:]
            P4 = p[1:-1, 2:]
            P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]
            P7 = p[2:, :-2]
            P8 = p[1:-1, :-2]
            P9 = p[:-2, :-2]
            ring = [P2, P3, P4, P5, P6, P7, P8, P9]
            count = sum(r.astype(int) for r in ring)
            transitions = sum(
                ((~ring[i]) & ring[(i + 1) % 8]).astype(int) for i in range(8)
            )
            if step == 0:
                cond = ~(P2 & P4 & P6) & ~(P4 & P6 & P8)
            else:
                cond = ~(P2 & P4 & P8) & ~(P2 & P6 & P8)
            remove = img & (count >= 2) & (count <= 6) & (transitions == 1) & cond
            if remove.any():
                img[remove] = False
                changed = True
        if not changed:
            return img


def skeleton_endpoints(skeleton: ArrayLike) -> NDArray:
    """Pixels of a skeleton that have exactly one neighbour (the tips).

    - ``skeleton``: 2-D bool array, one pixel wide.

    Returns ``(n_tips, 2)`` points as ``(x, y)``.

    Explained in chapter 01.
    """
    sk = np.asarray(skeleton).astype(bool)
    neighbours = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    rows, cols = np.nonzero(sk & (neighbours == 1))
    return np.stack([cols, rows], axis=1).astype(float)


def order_skeleton(mask: ArrayLike) -> NDArray:
    """List the pixels of a one-pixel-wide line from one tip to the other.

    Method: start at an endpoint (a pixel with a single neighbour; if
    there is none, at the pixel nearest the top-left corner), then
    repeatedly step to the nearest pixel not yet visited.  This is the
    "greedy walk" of the original tutorial.

    - ``mask``: 2-D bool array holding the skeleton (see :func:`skeletonize`).
      If it is thicker than one pixel it is thinned first.

    Returns ``(N, 2)`` ``(x, y)`` points, ordered head to tail.  Which tip
    is the head is arbitrary (the first endpoint in row-major order).

    Explained in chapter 01.
    """
    sk = np.asarray(mask).astype(bool)
    sk = skeletonize(sk) if sk.sum() > 0 else sk
    rows, cols = np.nonzero(sk)
    pts = np.stack([cols, rows], axis=1).astype(float)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    tips = skeleton_endpoints(sk)
    if len(tips):
        start = int(np.argmin(np.linalg.norm(pts - tips[0], axis=1)))
    else:
        start = 0
    remaining = np.ones(len(pts), dtype=bool)
    order = [start]
    remaining[start] = False
    current = pts[start]
    for _ in range(len(pts) - 1):
        idx = np.flatnonzero(remaining)
        d = np.linalg.norm(pts[idx] - current, axis=1)
        nxt = idx[int(np.argmin(d))]
        order.append(nxt)
        remaining[nxt] = False
        current = pts[nxt]
    return pts[order]


def arc_length(points: ArrayLike, closed: bool = False) -> NDArray:
    """Cumulative distance along a polyline.

    Formula::

        s_0 = 0,   s_i = s_{i-1} + || p_i - p_{i-1} ||

    - ``p_i``: the i-th point.
    - ``s_i``: the distance walked from the first point to point i.

    Returns ``(N,)`` (or ``(N + 1,)`` for a closed curve, where the last
    entry is the distance back to the start).

    Explained in chapter 01.
    """
    P = np.asarray(points, dtype=float)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_by_arclength(points: ArrayLike, N: int, closed: bool = False) -> NDArray:
    """``N`` points evenly spaced *along* a polyline.

    Formula::

        s_i   = cumulative length up to point i           (see arc_length)
        s*_j  = j * L / (N - 1),  j = 0..N-1              (open; L = total length)
        p*_j  = linear interpolation of the polyline at distance s*_j

    - ``points``: ``(P, 2)`` ordered points, e.g. a traced outline or an
      ordered skeleton, usually unevenly spaced (a diagonal pixel step is
      longer than a straight one).
    - ``N``: how many landmarks to produce.
    - ``closed``: for a loop the spacing is ``L / N`` and the start point
      is not repeated.
    - ``p*_j``: the new points.  Consecutive ones are the same distance
      apart measured along the curve.

    Returns ``(N, 2)``.

    Explained in chapter 01.
    """
    P = np.asarray(points, dtype=float)
    if closed:
        P = np.vstack([P, P[:1]])
    s = arc_length(P)
    L = s[-1]
    if closed:
        targets = np.arange(N) * L / N
    else:
        targets = np.linspace(0.0, L, N)
    x = np.interp(targets, s, P[:, 0])
    y = np.interp(targets, s, P[:, 1])
    return np.stack([x, y], axis=1)


def mask_to_landmarks(mask: ArrayLike, N: int, kind: str = "skeleton") -> NDArray:
    """The whole pipeline: mask -> ordered points -> ``N`` even landmarks.

    - ``kind="skeleton"``: thin the object to its midline, order it head
      to tail, resample (for elongated open shapes such as worms).
    - ``kind="contour"``: trace the longest outline and resample it as a
      closed loop (for blob-like shapes such as cells).

    Returns ``(N, 2)`` ``(x, y)`` landmarks.

    Explained in chapter 01.
    """
    if kind == "skeleton":
        return resample_by_arclength(order_skeleton(skeletonize(mask)), N, closed=False)
    if kind == "contour":
        contours = mask_to_contours(mask)
        if not contours:
            raise ValueError("no outline found in the mask")
        longest = max(contours, key=len)
        return resample_by_arclength(longest, N, closed=True)
    raise ValueError("kind must be 'skeleton' or 'contour'")
