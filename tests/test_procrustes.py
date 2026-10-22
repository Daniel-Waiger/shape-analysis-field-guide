import numpy as np
from scipy.optimize import minimize_scalar

from shapekit import (
    align,
    align_dataset,
    canonical_orientation,
    frechet_mean,
    geodesic_distance,
    gpa_mean,
    optimal_rotation_angle,
    rotate,
)


def _brute_force_angle(z, w):
    """Minimise ||w - e^{i theta} z|| by a fine grid plus local polish."""
    grid = np.linspace(-np.pi, np.pi, 3601)
    errs = [np.linalg.norm(w - np.exp(1j * t) * z) for t in grid]
    t0 = grid[int(np.argmin(errs))]
    res = minimize_scalar(
        lambda t: np.linalg.norm(w - np.exp(1j * t) * z),
        bounds=(t0 - 0.01, t0 + 0.01),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return res.x


def test_closed_form_angle_equals_brute_force_minimum(worm_preshapes):
    Z, _ = worm_preshapes
    rng = np.random.default_rng(0)
    for _ in range(5):
        i, j = rng.choice(len(Z), 2, replace=False)
        theta = optimal_rotation_angle(Z[i], Z[j])
        theta_bf = _brute_force_angle(Z[i], Z[j])
        diff = np.angle(np.exp(1j * (theta - theta_bf)))
        assert abs(diff) < 1e-6


def test_angle_recovers_a_known_rotation():
    rng = np.random.default_rng(1)
    z = rng.normal(size=12) + 1j * rng.normal(size=12)
    z -= z.mean()
    z /= np.linalg.norm(z)
    w = rotate(z, 0.7)
    assert abs(optimal_rotation_angle(z, w) - 0.7) < 1e-12
    assert np.allclose(align(z, w), w)
    # worked example from the guide: (1, 0) onto (0, 1) is a +90 degree turn
    assert np.isclose(optimal_rotation_angle(np.array([1.0 + 0j]), np.array([1j])), np.pi / 2)


def test_align_dataset_matches_align_row_by_row(worm_preshapes):
    Z, _ = worm_preshapes
    ref = Z[0]
    A = align_dataset(Z[:20], ref)
    for k in range(20):
        assert np.allclose(A[k], align(Z[k], ref))
        # the alignment never changes the norm
        assert np.isclose(np.linalg.norm(A[k]), 1.0)


def test_gpa_mean_equals_frechet_mean_from_any_start(worm_preshapes):
    Z, _ = worm_preshapes
    m = frechet_mean(Z)
    for seed in range(4):
        g, history = gpa_mean(Z, n_iter=20, seed=seed, return_history=True)
        assert geodesic_distance(m, g) < 1e-6
        assert np.allclose(align(g, m), m, atol=1e-6)
        assert history.shape[1] == Z.shape[1] and len(history) >= 2
        assert np.allclose(np.linalg.norm(history, axis=1), 1.0)
        # iterates are returned already rotated onto the final mean
        assert np.allclose(history[-1], g)
        for h in history:
            assert abs(optimal_rotation_angle(h, g)) < 1e-9


def test_rotation_only_gpa_converges_but_to_a_slightly_different_mean(worm_preshapes):
    Z, _ = worm_preshapes
    m = frechet_mean(Z)
    g0 = gpa_mean(Z, seed=0, weighted=False)
    g1 = gpa_mean(Z, seed=3, weighted=False)
    assert geodesic_distance(g0, g1) < 1e-6  # same fixed point from any start
    d = geodesic_distance(m, g0)
    assert 1e-4 < d < 0.02  # close to, but not identical with, the eigenvector mean


def test_canonical_orientation_is_a_rotation_and_idempotent():
    rng = np.random.default_rng(4)
    z = rng.normal(size=12) + 1j * rng.normal(size=12)
    z -= z.mean()
    c = canonical_orientation(z)
    assert np.isclose(abs(np.vdot(z, c)), np.linalg.norm(z) ** 2)  # same shape, rotated
    assert np.allclose(canonical_orientation(rotate(z, 1.3)), c)
    assert c[0].real <= 0
