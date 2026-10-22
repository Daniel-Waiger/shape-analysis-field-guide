import numpy as np

from shapekit import TangentPCA, align, frechet_mean, geodesic_distance, real_to_tangent, tangent_to_real


def test_tangent_pca_shapes_ratios_and_rank(worm_preshapes):
    Z, _ = worm_preshapes
    pca = TangentPCA().fit(Z)
    assert pca.components_.shape == (24, 24)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)
    assert np.all(np.diff(pca.explained_variance_) <= 1e-12)
    # 24 numbers minus 2 (position), 1 (size), 1 (rotation): 20 live dimensions
    n_nonzero = int(np.sum(pca.explained_variance_ratio_ > 1e-10))
    assert n_nonzero == 20
    # the first mode dominates (bending), matching the tutorial's ~66 %
    assert 0.6 < pca.explained_variance_ratio_[0] < 0.72
    # rows of components are orthonormal
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(24))


def test_transform_scores_have_the_fitted_variance(worm_preshapes):
    Z, _ = worm_preshapes
    pca = TangentPCA().fit(Z)
    S = pca.transform(Z)
    assert S.shape == (len(Z), 24)
    assert np.allclose(S.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(S.var(axis=0, ddof=1), pca.explained_variance_, atol=1e-12)
    assert np.allclose(pca.transform(Z[5]), S[5])


def test_inverse_transform_reconstructs_shapes(worm_preshapes):
    Z, _ = worm_preshapes
    pca = TangentPCA().fit(Z)
    S = pca.transform(Z[:10])
    R = pca.inverse_transform(S)
    for z, r in zip(Z[:10], R):
        assert np.allclose(align(r, z), z, atol=1e-9)


def test_mode_shape_and_sample(worm_preshapes):
    Z, _ = worm_preshapes
    pca = TangentPCA().fit(Z)
    plus, minus = pca.mode_shape(0, n_sd=1.0)
    assert plus.shape == (12,) and np.isclose(np.linalg.norm(plus), 1.0)
    # walking one SD along mode 1 moves the shape by about sdev_[0] on the sphere
    assert np.isclose(geodesic_distance(pca.mean_, plus), pca.sdev_[0], rtol=0.05)
    assert geodesic_distance(plus, minus) > geodesic_distance(pca.mean_, plus)
    samples = pca.sample(6, n_modes=3, seed=0)
    assert samples.shape == (6, 12)
    assert np.allclose(np.linalg.norm(samples, axis=1), 1.0)
    assert np.allclose(samples, pca.sample(6, n_modes=3, seed=0))  # deterministic
    assert not np.allclose(samples, pca.sample(6, n_modes=3, seed=1))


def test_given_mean_is_used_and_helpers_roundtrip(worm_preshapes):
    Z, _ = worm_preshapes
    m = frechet_mean(Z)
    pca = TangentPCA(mean=m).fit(Z[:100])
    assert np.allclose(pca.mean_, m)
    v = np.arange(12) + 1j * np.arange(12, 24)
    assert np.allclose(real_to_tangent(tangent_to_real(v)), v)
