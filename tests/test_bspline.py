import numpy as np

from shapekit import BSplineCurve, basis_matrix, cubic_bspline, gram_matrix, interpolation_filter
from shapekit.bspline import POLE


def test_bump_values_and_partition_of_unity():
    assert np.isclose(cubic_bspline(0.0), 2.0 / 3.0)
    assert np.isclose(cubic_bspline(1.0), 1.0 / 6.0)
    assert np.isclose(cubic_bspline(2.0), 0.0)
    assert np.isclose(cubic_bspline(-0.5), cubic_bspline(0.5))
    t = np.linspace(-0.49, 0.49, 50)
    total = sum(cubic_bspline(t - k) for k in range(-3, 4))
    assert np.allclose(total, 1.0)
    assert np.isclose(POLE**2 + 4 * POLE + 1, 0.0)


def test_interpolation_filter_undoes_the_141_blur_closed():
    rng = np.random.default_rng(0)
    s = rng.normal(size=9)
    c = interpolation_filter(s, closed=True)
    blurred = (np.roll(c, 1) + 4 * c + np.roll(c, -1)) / 6
    assert np.allclose(blurred, s, atol=1e-12)


def test_interpolation_filter_undoes_the_141_blur_open_with_mirror():
    rng = np.random.default_rng(1)
    s = rng.normal(size=7)
    c = interpolation_filter(s, closed=False)
    ext = np.concatenate([[c[1]], c, [c[-2]]])  # mirror: c[-1] = c[1], c[M] = c[M-2]
    blurred = (ext[:-2] + 4 * ext[1:-1] + ext[2:]) / 6
    assert np.allclose(blurred, s, atol=1e-10)


def test_curve_passes_through_the_knots_open_and_closed():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(8, 2)) * 10
    for closed in (False, True):
        curve = BSplineCurve(8, closed=closed).fit(P)
        assert curve.control_points_.shape == (8, 2)
        assert np.allclose(curve.knots(), P, atol=1e-10)
        assert np.allclose(curve.evaluate(np.arange(8.0)), P, atol=1e-10)
        dense = curve.sample(100)
        assert dense.shape == (100, 2)
        if not closed:
            assert np.allclose(dense[0], P[0]) and np.allclose(dense[-1], P[-1])


def test_basis_rows_sum_to_one_and_least_squares_fit_is_close():
    for closed in (False, True):
        B = basis_matrix(np.linspace(0, 5 if closed else 4, 37, endpoint=not closed), 5, closed)
        assert np.allclose(B.sum(axis=1), 1.0)
    # a dense circle, fitted with 8 control points, is reproduced closely
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    curve = BSplineCurve(8, closed=True).fit(circle)
    r = np.linalg.norm(curve.sample(300), axis=1)
    assert np.all(np.abs(r - 1.0) < 0.01)
    # fitting exactly M points interpolates, more points gives a least-squares fit
    line = np.stack([np.linspace(0, 10, 25), np.linspace(0, 10, 25) ** 2 / 10], axis=1)
    curve = BSplineCurve(6, closed=False).fit(line)
    dense = curve.sample(200)
    truth = np.stack([np.linspace(0, 10, 2000), np.linspace(0, 10, 2000) ** 2 / 10], axis=1)
    gaps = np.linalg.norm(dense[:, None, :] - truth[None, :, :], axis=2).min(axis=1)
    assert gaps.max() < 0.1  # the curve hugs the parabola (extent 10 x 10)


def test_gram_matrix_is_symmetric_positive_definite_with_known_entries():
    for M, closed in ((6, True), (10, True), (6, False), (12, False)):
        Phi = gram_matrix(M, closed=closed)
        assert Phi.shape == (M, M)
        assert np.allclose(Phi, Phi.T)
        assert np.linalg.eigvalsh(Phi).min() > 0
    Phi = gram_matrix(8, closed=True)
    expected = [151 / 315, 397 / 1680, 1 / 42, 1 / 5040, 0.0]
    for d, val in enumerate(expected):
        assert np.isclose(Phi[0, d], val, atol=1e-12)
    # every basis function covers the same amount of curve on a closed loop
    assert np.allclose(Phi.sum(axis=1), 1.0)
