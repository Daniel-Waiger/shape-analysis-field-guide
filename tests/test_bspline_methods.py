"""The two ways of un-blurring (direct solve, two-sweep recursive filter) agree."""

import numpy as np
import pytest

from shapekit.bspline import BSplineCurve, interpolation_filter


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("M", [4, 6, 12, 37])
def test_solve_and_recursive_agree(closed, M):
    rng = np.random.default_rng(M)
    s = rng.normal(size=(M, 2))
    c_solve = interpolation_filter(s, closed=closed, method="solve")
    c_rec = interpolation_filter(s, closed=closed, method="recursive")
    assert np.allclose(c_solve, c_rec, atol=1e-10)


@pytest.mark.parametrize("closed", [False, True])
def test_recursive_passes_through_knots(closed):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(9, 2))
    c = interpolation_filter(pts, closed=closed, method="recursive")
    curve = BSplineCurve(9, closed)
    curve.control_points_ = c
    t = np.arange(9) if closed else np.arange(9)
    assert np.allclose(curve.evaluate(t), pts, atol=1e-10)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        interpolation_filter(np.zeros((5, 2)), method="magic")
