import numpy as np

from shapekit import load_celegans, load_mef_subset


def test_celegans_shapes_and_labels():
    points, labels = load_celegans()
    assert points.shape == (1377, 12, 2) and points.dtype == np.float64
    assert labels.shape == (1377,) and labels.dtype.kind == "i"
    assert set(np.unique(labels)) == {0, 1}
    assert np.isfinite(points).all()
    # landmarks are evenly spaced along most worms (median variation ~2 %)
    seg = np.linalg.norm(np.diff(points, axis=1), axis=2)
    assert np.median(seg.std(axis=1) / seg.mean(axis=1)) < 0.05


def test_mef_subset_shapes_and_labels():
    points, labels, keys = load_mef_subset()
    assert points.shape == (1200, 120, 2) and points.dtype == np.float32
    assert labels.shape == (1200,) and set(np.unique(labels)) == {0, 1}
    assert keys.shape == (1200,) and keys.dtype.kind == "U"
    assert all(k.startswith("wildtype") for k in keys[labels == 0])
