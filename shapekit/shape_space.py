"""Kendall shape space: means, distances, geodesics and the tangent maps.

All functions take complex preshapes: ``(N,)`` for one shape, ``(K, N)``
for a dataset (see :mod:`shapekit.preshape`).  Two preshapes that differ
only by a rotation are the *same shape*; every formula below is built so
that the rotation does not matter.

Explained in chapters 04 (distances and geodesics), 05 (Fréchet mean) and
06 (log and exp maps).

Credits: the geometry is Kendall's (1984) as set out in Dryden & Mardia
(2016).  The functions follow notebook 2A of *Shape Analysis 101* by Johannes
Hugger and Virginie Uhlmann (2021, BSD-3); the spline-curve version of the
same geometry (chapter 07) is from Song, Uhlmann, Fageot & Unser (2020) and the
tutorial's ``spline_shape_space.py`` by Virginie Uhlmann and Anna Song.  This
module is a re-implementation; see ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .preshape import preshape
from .procrustes import align, canonical_orientation, hermitian_product, optimal_rotation_angle


def frechet_mean(Z: ArrayLike) -> NDArray:
    """The Fréchet mean shape: the top eigenvector of the summed outer products.

    Formula::

        S = sum_k z_k z_k^H          (an N x N complex matrix)
        m = top eigenvector of S,    then re-centred, normalised and
                                     put in canonical orientation

    - ``z_k``: the k-th preshape, a complex column of N landmarks.
    - ``z_k^H``: its conjugate transpose, so ``z_k z_k^H`` is an N x N
      table of "how landmark a relates to landmark b" for that shape.
    - ``S``: those tables added over all K shapes.
    - ``top eigenvector``: the single direction ``m`` that maximises
      ``m^H S m = sum_k |<m, z_k>|^2``, which is the same as minimising
      ``sum_k (1 - |<m, z_k>|^2)``, the summed squared full Procrustes
      distance to all shapes.  That is the definition of a Fréchet mean
      (Dryden & Mardia, Result 8.2).
    - The eigenvector comes with an arbitrary rotation; we fix it with
      :func:`shapekit.procrustes.canonical_orientation` so results are
      reproducible.

    Returns a complex ``(N,)`` preshape.

    Explained in chapter 05.
    """
    Z = np.asarray(Z, dtype=complex)
    if Z.ndim != 2:
        raise ValueError(f"expected a dataset of shape (K, N), got {Z.shape}")
    S = Z.T @ Z.conj()
    eigvals, eigvecs = np.linalg.eigh(S)
    m = eigvecs[:, np.argmax(eigvals)]
    return canonical_orientation(preshape(m))


def _abs_product(z: ArrayLike, w: ArrayLike) -> float:
    """|<z, w>| clipped to [0, 1] so that arccos never sees 1.0000000002."""
    return float(min(1.0, abs(hermitian_product(z, w))))


def geodesic_distance(z: ArrayLike, w: ArrayLike) -> float:
    """Distance along the sphere between two shapes (an arc length).

    Formula::

        rho(z, w) = arccos( |<z, w>| )

    - ``z, w``: two complex preshapes (unit norm, centred).
    - ``<z, w> = sum_k conj(z_k) w_k``: the complex inner product.
    - ``|<z, w>|``: its magnitude, the cosine of the angle between the
      two shapes once the best rotation is applied (taking the magnitude
      is what makes the rotation irrelevant).
    - ``rho``: that angle in radians, between 0 (identical shapes) and
      pi/2.  It is the length of the shortest arc on the preshape sphere,
      the "great-circle route".

    Explained in chapter 04.
    """
    return float(np.arccos(_abs_product(z, w)))


def chord_distance(z: ArrayLike, w: ArrayLike) -> float:
    """Straight-line distance through the sphere between two shapes.

    Formula::

        d_chord(z, w) = sqrt( 2 - 2 |<z, w>| )

    - ``|<z, w>|``: the cosine of the angle between the shapes after the
      best rotation, see :func:`geodesic_distance`.
    - ``2 - 2 cos(rho)``: the squared length of the chord that cuts
      across the arc of angle rho on a unit sphere.

    This equals :func:`procrustes_distance` exactly (not just closely).

    Explained in chapter 04.
    """
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * _abs_product(z, w))))


def procrustes_distance(z: ArrayLike, w: ArrayLike) -> float:
    """Euclidean distance after rotating ``z`` onto ``w`` as well as possible.

    Formula::

        d_P(z, w) = || w - exp(i theta*) z ||,   theta* = angle(<z, w>)

    - ``theta*``: the closed-form best rotation of chapter 03.
    - ``exp(i theta*) z``: z turned by that angle.
    - ``||.||``: the ordinary Euclidean (Frobenius) norm of the difference.

    Expanding the square at the best angle gives ``2 - 2 |<z, w>|``, so
    this is the same number as :func:`chord_distance`; the two functions
    exist to show that the "align and subtract" recipe and the formula on
    the sphere agree.  (This is the *partial* Procrustes distance; the
    *full* one, ``sqrt(1 - |<z, w>|^2)``, also allows a rescaling.)

    Explained in chapter 04.
    """
    return float(np.linalg.norm(np.asarray(w) - align(z, w)))


def geodesic_path(z: ArrayLike, w: ArrayLike, n_steps: int = 10) -> NDArray:
    """Shapes evenly spaced along the shortest arc from ``z`` to ``w`` (slerp).

    Formula::

        z~      = exp(i theta*) z                      (z rotated onto w)
        rho     = arccos( |<z, w>| )
        path(t) = [ sin((1 - t) rho) z~ + sin(t rho) w ] / sin(rho),
                  t = 0, 1/n, 2/n, ..., 1

    - ``z~``: z after the best rotation, so that the arc does not spin.
    - ``rho``: the geodesic distance between the shapes.
    - ``t``: the fraction of the way travelled, 0 at z and 1 at w.
    - ``sin((1-t) rho) / sin(rho)`` and ``sin(t rho) / sin(rho)``: weights
      chosen so that every ``path(t)`` has unit norm, i.e. stays on the
      sphere.  A plain linear blend would cut through the inside.

    Returns a complex array ``(n_steps + 1, N)``: row 0 is ``z~``, the
    last row is ``w``.

    Explained in chapter 04.
    """
    z_al = align(z, w)
    w = np.asarray(w, dtype=complex)
    rho = geodesic_distance(z_al, w)
    t = np.linspace(0.0, 1.0, n_steps + 1)[:, None]
    if rho < 1e-12:
        return np.repeat(z_al[None, :], n_steps + 1, axis=0)
    return (np.sin((1.0 - t) * rho) * z_al + np.sin(t * rho) * w) / np.sin(rho)


def log_map(mean: ArrayLike, z: ArrayLike) -> NDArray:
    """Flatten a shape onto the tangent plane at the mean.

    Formula::

        z~  = exp(i theta*) z,   theta* = angle(<z, mean>)
        rho = arccos( Re <mean, z~> )
        v   = (rho / sin(rho)) * ( z~ - cos(rho) * mean )

    - ``mean``: the preshape where the flat tangent plane touches the
      sphere (usually the Fréchet mean).
    - ``z~``: z rotated onto the mean, so the rotation is removed first.
    - ``rho``: the geodesic distance from the mean to z.
    - ``z~ - cos(rho) mean``: the part of z~ that is perpendicular to the
      mean, i.e. the direction in which you must walk to reach z.
    - ``rho / sin(rho)``: a stretch that makes ``||v|| = rho``, so the
      length of the tangent vector equals the arc length on the sphere.
    - ``v``: a complex ``(N,)`` tangent vector satisfying ``<mean, v> = 0``.

    Explained in chapter 06.
    """
    mean = np.asarray(mean, dtype=complex)
    z_al = align(z, mean)
    cos_rho = float(np.clip(np.vdot(mean, z_al).real, -1.0, 1.0))
    rho = float(np.arccos(cos_rho))
    if rho < 1e-12:
        return z_al - mean  # numerically zero
    return (rho / np.sin(rho)) * (z_al - cos_rho * mean)


def exp_map(mean: ArrayLike, v: ArrayLike) -> NDArray:
    """Walk from the mean along a tangent vector and land back on the sphere.

    Formula::

        t = ||v||
        z = cos(t) * mean + (sin(t) / t) * v

    - ``mean``: the preshape at which the tangent plane touches the sphere.
    - ``v``: a tangent vector at the mean (complex ``(N,)``, with
      ``<mean, v> = 0``), e.g. produced by :func:`log_map` or by PCA.
    - ``t``: its length, the distance to walk along the sphere.
    - ``cos(t) mean + sin(t) v/||v||``: the point on the great circle that
      starts at the mean, heads in direction v, and has travelled an arc
      of length t.  It has unit norm again.

    ``exp_map(mean, log_map(mean, z))`` gives back z up to rotation.
    Walking too far (t > pi/2) wraps around the sphere, which is the
    pitfall of chapter 06's synthetic worms.

    Explained in chapter 06.
    """
    mean = np.asarray(mean, dtype=complex)
    v = np.asarray(v, dtype=complex)
    t = float(np.linalg.norm(v))
    if t < 1e-12:
        return mean.copy()
    return np.cos(t) * mean + (np.sin(t) / t) * v
