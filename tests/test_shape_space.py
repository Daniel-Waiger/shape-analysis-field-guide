import numpy as np

from shapekit import (
    align,
    chord_distance,
    exp_map,
    frechet_mean,
    geodesic_distance,
    geodesic_path,
    log_map,
    procrustes_distance,
    rotate,
)


def test_procrustes_equals_chord_and_geodesic_is_longer(worm_preshapes):
    Z, _ = worm_preshapes
    rng = np.random.default_rng(0)
    for _ in range(50):
        i, j = rng.choice(len(Z), 2, replace=False)
        dp = procrustes_distance(Z[i], Z[j])
        dc = chord_distance(Z[i], Z[j])
        dg = geodesic_distance(Z[i], Z[j])
        assert abs(dp - dc) < 1e-12
        assert dg >= dc
        assert dg <= np.pi / 2 + 1e-12


def test_distances_are_rotation_invariant_and_zero_for_same_shape(worm_preshapes):
    Z, _ = worm_preshapes
    z, w = Z[3], Z[7]
    for fn in (geodesic_distance, chord_distance, procrustes_distance):
        assert abs(fn(z, w) - fn(rotate(z, 2.0), rotate(w, -1.0))) < 1e-12
        assert fn(z, rotate(z, 0.5)) < 1e-6


def test_frechet_mean_is_a_preshape_and_beats_the_plain_average(worm_preshapes):
    Z, _ = worm_preshapes
    m = frechet_mean(Z)
    assert np.isclose(np.linalg.norm(m), 1.0) and abs(m.mean()) < 1e-12
    # the plain average of aligned shapes is shorter than 1
    aligned = np.stack([align(z, m) for z in Z])
    plain = aligned.mean(axis=0)
    assert np.linalg.norm(plain) < 1.0
    # the Fréchet mean maximises the summed squared projections
    score = lambda c: np.sum(np.abs(Z.conj() @ c) ** 2)
    assert score(m) >= score(plain / np.linalg.norm(plain)) - 1e-9


def test_geodesic_path_endpoints_norms_and_spacing(worm_preshapes):
    Z, _ = worm_preshapes
    z, w = Z[0], Z[100]
    path = geodesic_path(z, w, n_steps=10)
    assert path.shape == (11, Z.shape[1])
    assert np.allclose(np.linalg.norm(path, axis=1), 1.0)
    assert np.allclose(path[-1], w)
    assert np.allclose(align(path[0], z), z, atol=1e-9)
    steps = [geodesic_distance(path[k], path[k + 1]) for k in range(10)]
    assert np.allclose(steps, geodesic_distance(z, w) / 10, atol=1e-9)


def test_exp_log_roundtrip_up_to_rotation(worm_preshapes):
    Z, _ = worm_preshapes
    m = frechet_mean(Z)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(Z), 20, replace=False):
        v = log_map(m, Z[i])
        assert abs(np.vdot(m, v)) < 1e-9  # tangent: orthogonal to the mean
        assert np.isclose(np.linalg.norm(v), geodesic_distance(m, Z[i]))
        back = exp_map(m, v)
        assert np.isclose(np.linalg.norm(back), 1.0)
        assert chord_distance(back, Z[i]) < 1e-7
        assert np.allclose(align(back, Z[i]), Z[i], atol=1e-9)
    assert np.allclose(exp_map(m, np.zeros_like(m)), m)
