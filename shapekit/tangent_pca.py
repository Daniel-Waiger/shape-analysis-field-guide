"""Tangent-space PCA: the main ways a set of shapes varies.

PCA needs a flat space, and shape space is curved.  So every shape is
first flattened onto the tangent plane at the mean (:func:`log_map`),
PCA is run on those flat vectors, and results are mapped back with
:func:`exp_map`.

A tangent vector of N complex numbers is stored as 2N real numbers:
first the N real parts (x), then the N imaginary parts (y).

Explained in chapter 06.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .shape_space import exp_map, frechet_mean, log_map


def tangent_to_real(v: ArrayLike) -> NDArray:
    """Store complex tangent vectors as real ones: ``[Re v, Im v]``.

    Formula::

        r = ( Re v_1, ..., Re v_N, Im v_1, ..., Im v_N )

    - ``v``: a complex tangent vector ``(N,)`` or a stack ``(K, N)``.
    - ``r``: the same numbers as a real vector of length 2N (or ``(K, 2N)``).

    Explained in chapter 06.
    """
    v = np.asarray(v, dtype=complex)
    return np.concatenate([v.real, v.imag], axis=-1)


def real_to_tangent(r: ArrayLike) -> NDArray:
    """Inverse of :func:`tangent_to_real`: ``v = r[:N] + i r[N:]``.

    - ``r``: a real vector of length 2N (or ``(K, 2N)``).
    - ``v``: the complex tangent vector ``(N,)`` (or ``(K, N)``).

    Explained in chapter 06.
    """
    r = np.asarray(r, dtype=float)
    n = r.shape[-1] // 2
    return r[..., :n] + 1j * r[..., n:]


class TangentPCA:
    """PCA of shapes on the tangent plane at their mean.

    Formula (fit)::

        v_k = log_map(mean, z_k)                 (K tangent vectors)
        R   = [ Re v_k, Im v_k ]_k               (K x 2N real matrix)
        mu  = (1/K) sum_k R[k]                   (mean tangent vector)
        R - mu = U S C^T                         (singular value decomposition)
        lambda_i = S_i^2 / (K - 1)               (variance of mode i)

    - ``mean``: the preshape at which the tangent plane touches the
      sphere.  Pass one to use it, or leave ``None`` to compute the
      Fréchet mean of the fitted data.
    - ``v_k``: shape k flattened onto the tangent plane.
    - ``mu``: the centre of the flattened cloud (close to zero, because
      the Fréchet mean is nearly the centre of its own tangent cloud).
    - ``C`` (``components_``): unit vectors, one per row, each a
      *mode of variation*; mode 1 is the direction along which the
      shapes spread most.
    - ``lambda_i`` (``explained_variance_``): how much of the spread lies
      along mode i; ``explained_variance_ratio_`` divides by the total.
    - ``sdev_``: ``sqrt(lambda_i)``, one standard deviation along mode i.

    Only 2N - 4 modes can carry variance: 2 directions are used up by
    position, 1 by size and 1 by rotation, which the preshape and the
    alignment have removed.  For 12 landmarks that leaves 20 of 24.

    Explained in chapter 06.
    """

    def __init__(self, mean: ArrayLike | None = None) -> None:
        self.mean_: NDArray | None = None if mean is None else np.asarray(mean, dtype=complex)
        self.mean_tangent_: NDArray | None = None
        self.components_: NDArray | None = None
        self.explained_variance_: NDArray | None = None
        self.explained_variance_ratio_: NDArray | None = None
        self.sdev_: NDArray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, Z: ArrayLike) -> "TangentPCA":
        """Flatten the dataset and run PCA.  See the class docstring.

        - ``Z``: complex preshapes ``(K, N)``.

        Explained in chapter 06.
        """
        Z = np.asarray(Z, dtype=complex)
        if self.mean_ is None:
            self.mean_ = frechet_mean(Z)
        R = self._flatten(Z)
        self.mean_tangent_ = R.mean(axis=0)
        centred = R - self.mean_tangent_
        # SVD of the centred cloud: rows of vt are the principal directions.
        _, s, vt = np.linalg.svd(centred, full_matrices=True)
        n_modes = vt.shape[0]
        variance = np.zeros(n_modes)
        variance[: len(s)] = s**2 / max(len(Z) - 1, 1)
        # Make each mode's sign reproducible: its largest entry is positive.
        signs = np.sign(vt[np.arange(n_modes), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        self.components_ = vt * signs[:, None]
        self.explained_variance_ = variance
        self.explained_variance_ratio_ = variance / variance.sum()
        self.sdev_ = np.sqrt(variance)
        return self

    def _check_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None or self.mean_tangent_ is None:
            raise RuntimeError("call fit(Z) first")

    def _flatten(self, Z: NDArray) -> NDArray:
        return tangent_to_real(np.stack([log_map(self.mean_, z) for z in Z]))

    # ------------------------------------------------------------ transform
    def transform(self, Z: ArrayLike) -> NDArray:
        """Scores of each shape along each mode.

        Formula::

            score_k = ( [Re v_k, Im v_k] - mu ) C^T,   v_k = log_map(mean, z_k)

        - ``Z``: complex preshapes ``(K, N)`` (or one shape ``(N,)``).
        - ``score_k``: ``(2N,)`` real numbers; entry i says how far shape k
          lies along mode i, in the same units as ``sdev_``.

        Returns ``(K, 2N)`` (or ``(2N,)`` for one shape).

        Explained in chapter 06.
        """
        self._check_fitted()
        Z = np.asarray(Z, dtype=complex)
        single = Z.ndim == 1
        R = self._flatten(Z[None, :] if single else Z)
        scores = (R - self.mean_tangent_) @ self.components_.T
        return scores[0] if single else scores

    def inverse_transform(self, scores: ArrayLike) -> NDArray:
        """From mode scores back to preshapes.

        Formula::

            r = mu + sum_i score_i C_i
            z = exp_map(mean, r[:N] + i r[N:])

        - ``scores``: ``(n_modes,)`` or ``(K, n_modes)``; missing trailing
          modes count as zero.
        - ``C_i``: the i-th mode (row of ``components_``).
        - ``z``: the reconstructed complex preshape(s).

        Explained in chapter 06.
        """
        self._check_fitted()
        single = np.ndim(scores) == 1
        scores = np.atleast_2d(np.asarray(scores, dtype=float))
        r = self.mean_tangent_ + scores @ self.components_[: scores.shape[1]]
        shapes = np.stack([exp_map(self.mean_, v) for v in real_to_tangent(r)])
        return shapes[0] if single else shapes

    def mode_shape(self, i: int, n_sd: float = 1.0) -> tuple[NDArray, NDArray]:
        """The shapes ``n_sd`` standard deviations along mode ``i``, both ways.

        Formula::

            z_plus  = exp_map(mean, mu + n_sd * sdev_i * C_i)
            z_minus = exp_map(mean, mu - n_sd * sdev_i * C_i)

        - ``i``: mode index, 0 for the first (largest) mode.
        - ``n_sd``: how many standard deviations to walk.
        - ``sdev_i C_i``: one standard deviation of movement along mode i,
          as a tangent vector.

        Returns ``(z_plus, z_minus)``, two complex preshapes ``(N,)``.

        Explained in chapter 06.
        """
        self._check_fitted()
        step = np.zeros(self.components_.shape[0])
        step[i] = n_sd * self.sdev_[i]
        plus = self.inverse_transform(step)
        minus = self.inverse_transform(-step)
        return plus, minus

    def sample(self, n: int, n_modes: int = 3, seed: int = 0) -> NDArray:
        """Draw synthetic shapes from a Gaussian on the tangent plane.

        Formula::

            score_i ~ Normal(0, sdev_i^2)   for i < n_modes, else 0
            z = exp_map(mean, mu + sum_i score_i C_i)

        - ``n``: how many shapes to draw.
        - ``n_modes``: how many modes get random scores; the rest stay
          at zero (the mean).
        - ``sdev_i``: the standard deviation of the data along mode i,
          so samples have the same spread as the data.  Multiplying by
          an extra ``sqrt(K - 1)`` (as the original tutorial did) throws
          the samples off the far side of the sphere.
        - ``seed``: random seed for reproducibility.

        Returns complex preshapes ``(n, N)``.

        Explained in chapter 06.
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        scores = rng.standard_normal((n, n_modes)) * self.sdev_[:n_modes]
        return np.atleast_2d(self.inverse_transform(scores))
