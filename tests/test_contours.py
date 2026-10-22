import numpy as np

from shapekit import (
    arc_length,
    mask_to_contours,
    mask_to_landmarks,
    order_skeleton,
    resample_by_arclength,
    skeleton_endpoints,
    skeletonize,
)


def _ellipse_mask(a=50, b=10, shape=(60, 120)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((xx - 60) ** 2 / a**2 + (yy - 30) ** 2 / b**2) < 1


def test_resample_by_arclength_gives_equal_segment_lengths():
    rng = np.random.default_rng(0)
    # unevenly spaced points on a straight line
    s = np.sort(rng.uniform(0, 1, 30))
    s[0], s[-1] = 0.0, 1.0
    line = np.stack([3 * s, 4 * s], axis=1)
    out = resample_by_arclength(line, 11)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(seg, 0.5)
    assert np.allclose(out[0], line[0]) and np.allclose(out[-1], line[-1])
    # a bent polyline: equal arc-length spacing, total length preserved
    zig = np.array([[0, 0], [1, 0], [1, 1], [3, 1], [3, 3]], dtype=float)
    out = resample_by_arclength(zig, 7)
    assert np.isclose(arc_length(out)[-1], arc_length(zig)[-1])
    assert np.allclose(np.diff(arc_length(out)), 1.0)
    # closed loop: N equal steps around a circle, no repeated start point
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    out = resample_by_arclength(circle, 8, closed=True)
    seg = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1)
    assert np.allclose(seg, seg[0], rtol=1e-4)
    assert out.shape == (8, 2)


def test_contours_are_xy_loops_and_filter_short_ones():
    mask = _ellipse_mask()
    contours = mask_to_contours(mask, min_len=10)
    assert len(contours) == 1
    c = contours[0]
    assert c.shape[1] == 2
    assert c[:, 0].max() < 120 and c[:, 0].min() > 0  # x spans the long axis
    assert 15 < c[:, 1].min() and c[:, 1].max() < 45  # y the short axis
    # instance mask: two objects -> two outlines
    inst = mask.astype(int)
    inst[:, 60:] *= 2
    assert len(mask_to_contours(inst)) == 2
    assert mask_to_contours(mask, min_len=10_000) == []


def test_skeleton_is_thin_ordered_and_runs_along_the_long_axis():
    mask = _ellipse_mask()
    sk = skeletonize(mask)
    assert sk.sum() < mask.sum() / 5
    assert len(skeleton_endpoints(sk)) == 2
    pts = order_skeleton(sk)
    assert pts.shape[1] == 2 and len(pts) == sk.sum()
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() <= np.sqrt(2) + 1e-9  # consecutive pixels touch
    assert pts[0, 0] < pts[-1, 0]  # left tip to right tip
    lm = mask_to_landmarks(mask, 12, kind="skeleton")
    assert lm.shape == (12, 2)
    assert np.allclose(np.diff(arc_length(lm)), np.diff(arc_length(lm))[0])
    lm2 = mask_to_landmarks(mask, 20, kind="contour")
    assert lm2.shape == (20, 2)
