"""Procrustes alignment: remove rotation with one complex formula.

Shapes are given as complex preshapes, ``(N,)`` for one shape and
``(K, N)`` for a dataset (see :mod:`shapekit.preshape`).  Multiplying a
complex vector by ``exp(i * theta)`` rotates every landmark by ``theta``
radians about the origin, counter-clockwise in a y-up coordinate system
(clockwise when the y axis points down, as in images).

Explained in chapter 03 (the closed-form angle) and chapter 05 (GPA).

Credits: the complex-number formulation follows notebook 2A of *Shape
Analysis 101* by Johannes Hugger and Virginie Uhlmann (2021, BSD-3) and
Dryden & Mardia (2016).  Re-implemented, not copied; see ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .preshape import preshape


def hermitian_product(z: ArrayLike, w: ArrayLike) -> complex:
    """The complex inner product of two shapes.

    Formula::

        <z, w> = sum_k conj(z_k) * w_k

    - ``z_k, w_k``: the k-th landmark of each shape, as complex numbers.
    - ``conj(z_k)``: the complex conjugate of z_k (flip the sign of the
      imaginary part), which is what makes the product measure *how well
      the landmark pairs point the same way*.
    - ``<z, w>``: one complex number.  Its magnitude says how similar the
      shapes are after the best rotation; its angle says what that
      rotation is.

    Explained in chapter 03.
    """
    z = np.asarray(z)
    w = np.asarray(w)
    return complex(np.vdot(z, w))


def optimal_rotation_angle(z: ArrayLike, w: ArrayLike) -> float:
    """The angle that rotates ``z`` onto ``w`` with least squared error.

    Formula::

        theta* = angle( sum_k conj(z_k) * w_k ) = angle( <z, w> )

    - ``z, w``: two preshapes, complex ``(N,)``.
    - ``conj(z_k) * w_k``: one "vote" per landmark pair, an arrow whose
      direction is the rotation that would bring z_k onto w_k and whose
      length grows with how far both points lie from the centre.
    - ``sum_k``: adding all the votes gives the consensus arrow.
    - ``angle(.)``: the direction of that arrow, in radians, in (-pi, pi].
    - ``theta*``: rotating z by this angle, ``exp(i theta*) * z``,
      minimises ``||w - exp(i theta) z||^2`` over all theta.

    Why: ``||w - e^{i theta} z||^2 = ||w||^2 + ||z||^2 - 2 Re(e^{i theta}
    <w, z>)``, and the last term is largest when ``e^{i theta}`` cancels
    the angle of ``<w, z>``, i.e. ``theta = angle(<z, w>)``.

    Explained in chapter 03.
    """
    return float(np.angle(hermitian_product(z, w)))


def rotate(z: ArrayLike, theta: float) -> NDArray:
    """Rotate a complex shape by ``theta`` radians.

    Formula::

        z' = exp(i * theta) * z

    - ``exp(i theta) = cos(theta) + i sin(theta)``: a complex number of
      magnitude 1 at angle theta.
    - ``z``: the shape as complex numbers; every landmark is multiplied by
      the same factor, so the whole shape turns rigidly about the origin.

    Explained in chapter 03.
    """
    return np.exp(1j * theta) * np.asarray(z)


def align(z: ArrayLike, w: ArrayLike) -> NDArray:
    """Rotate ``z`` so that it matches ``w`` as closely as possible.

    Formula::

        z_aligned = exp(i * theta*) * z,   theta* = angle(<z, w>)

    - ``z``: the shape to rotate (complex preshape).
    - ``w``: the fixed target shape.
    - ``theta*``: the closed-form angle from :func:`optimal_rotation_angle`.

    Explained in chapter 03.
    """
    return rotate(z, optimal_rotation_angle(z, w))


def align_dataset(Z: ArrayLike, ref: ArrayLike) -> NDArray:
    """Rotate every shape in a dataset onto one reference shape.

    Formula::

        Z_aligned[k] = exp(i * theta_k) * Z[k],   theta_k = angle(<Z[k], ref>)

    - ``Z``: the dataset, complex ``(K, N)``, one preshape per row.
    - ``ref``: the reference preshape, complex ``(N,)``, typically a mean.
    - ``theta_k``: the best angle for shape k, computed independently for
      each shape.

    Explained in chapter 03 (used again in chapters 05 and 06).
    """
    Z = np.asarray(Z)
    ref = np.asarray(ref)
    # <Z[k], ref> for every k at once: (K, N) conj times (N,) summed over N.
    products = Z.conj() @ ref
    thetas = np.angle(products)
    return np.exp(1j * thetas)[:, None] * Z


def canonical_orientation(z: ArrayLike) -> NDArray:
    """Rotate a shape into a fixed, reproducible orientation.

    Formula::

        phi = angle( sum_k z_k^2 ) / 2
        z' = exp(-i * phi) * z,  then flipped by pi if Re(z'_0) > 0

    - ``sum_k z_k^2``: the complex "second moment" of a centred shape.
      Its angle is twice the direction of the shape's long axis (because
      squaring doubles angles), so ``phi`` is the long-axis direction.
    - ``z'``: the shape with its long axis horizontal.
    - The flip makes the first landmark sit on the left (negative real
      part), so a worm reads head-left, tail-right.

    This is only a convention: it changes nothing about the shape, it
    just chooses one of the infinitely many rotated copies so that means
    and figures come out the same every time.

    Explained in chapter 05 (where the mean shape needs an orientation).
    """
    z = np.asarray(z, dtype=complex)
    phi = np.angle(np.sum(z**2)) / 2.0
    z = np.exp(-1j * phi) * z
    if z[0].real > 0:
        z = -z
    return z


def gpa_mean(
    Z: ArrayLike,
    n_iter: int = 20,
    seed: int = 0,
    return_history: bool = False,
    weighted: bool = True,
    tol: float = 1e-10,
) -> NDArray | tuple[NDArray, NDArray]:
    """Generalised Procrustes Analysis: iterate "fit every shape to the mean, average".

    Algorithm::

        m_0  = a randomly chosen shape of the dataset
        repeat:
            fit_k = w_k * Z[k],
                    w_k = <Z[k], m>                     if weighted  (rotation and scale)
                    w_k = exp(i angle(<Z[k], m>))       otherwise    (rotation only)
            m_new = preshape( (1/K) sum_k fit_k )
        until ||m_new - m|| < tol  (or n_iter rounds)

    - ``Z``: the dataset of complex preshapes, ``(K, N)``.
    - ``m``: the current estimate of the mean shape.
    - ``<Z[k], m> = sum_j conj(Z[k]_j) m_j``: the complex inner product.
      Its angle is the best rotation of shape k onto the mean (chapter
      03); its magnitude ``|<Z[k], m>|`` is the best *scale*.
    - ``weighted=True`` (the original tutorial's loop): each shape is
      rotated *and* shrunk by that magnitude, so a shape that matches the
      mean poorly counts for less.  ``weighted=False`` (textbook GPA):
      rotation only, every shape keeps its unit size.
    - ``(1/K) sum_k``: a plain point-by-point average of the fitted shapes.
    - ``preshape(.)``: re-centre and rescale, because the average of unit
      vectors that point in slightly different directions is shorter than 1.
    - ``seed``: which shape starts the iteration (``numpy`` random seed).
    - ``return_history``: also return every intermediate mean, as an
      array ``(n_rounds + 1, N)``, each already rotated onto the final
      mean so a plot of the convergence does not spin.

    Two facts worth knowing:

    1. With ``weighted=True`` each round computes ``sum_k <m, Z[k]> Z[k]
       = S m`` and renormalises.  That is power iteration for the top
       eigenvector of ``S = sum_k z_k z_k^H``, so it converges to the
       Fréchet mean of :func:`shapekit.shape_space.frechet_mean` from any
       start (up to an overall rotation).
    2. With ``weighted=False`` the fixed point maximises
       ``sum_k |<m, z_k>|`` instead of ``sum_k |<m, z_k>|^2``: the mean
       under the chord (partial Procrustes) distance rather than the full
       Procrustes distance.  It is close to, but not the same as, the
       eigenvector mean: about 0.007 (Procrustes distance) away on the
       worm data.

    Compare means with a distance, not coordinate by coordinate, because
    of the arbitrary rotation.

    Explained in chapter 05.
    """
    Z = np.asarray(Z, dtype=complex)
    rng = np.random.default_rng(seed)
    m = Z[rng.integers(len(Z))].copy()
    history = [m]
    for _ in range(n_iter):
        products = Z.conj() @ m  # <Z[k], m> for every k
        factors = products if weighted else np.exp(1j * np.angle(products))
        fitted = factors[:, None] * Z
        m_new = preshape(fitted.mean(axis=0))
        history.append(m_new)
        change = np.linalg.norm(m_new - m)
        m = m_new
        if change < tol:
            break
    if return_history:
        return m, align_dataset(np.stack(history), m)
    return m
