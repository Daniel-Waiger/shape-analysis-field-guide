import numpy as np

from shapekit import center, from_complex, frobenius_norm, normalize, preshape, to_complex


def test_center_removes_centroid_single_and_batch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2)) + 100.0
    assert np.allclose(center(X).mean(axis=0), 0.0)
    B = rng.normal(size=(5, 12, 2)) + 50.0
    assert np.allclose(center(B).mean(axis=1), 0.0)
    assert center(B).shape == B.shape


def test_normalize_gives_unit_frobenius_norm():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(5, 12, 2)) * 30.0
    P = normalize(center(B))
    assert np.allclose(frobenius_norm(P), 1.0)
    assert np.allclose(np.sqrt((P**2).sum(axis=(1, 2))), 1.0)


def test_preshape_is_invariant_to_translation_and_scale():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 2))
    Y = 3.0 * X + np.array([10.0, -7.0])
    assert np.allclose(preshape(X), preshape(Y))


def test_complex_roundtrip_and_complex_preshape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 12, 2))
    z = to_complex(X)
    assert z.shape == (4, 12) and np.iscomplexobj(z)
    assert np.allclose(from_complex(z), X)
    # preshape works the same on the complex representation
    assert np.allclose(to_complex(preshape(X)), preshape(z))
    assert np.allclose(np.abs(preshape(z)).sum(axis=1) ** 0 * np.linalg.norm(preshape(z), axis=1), 1.0)
