"""Preshape: remove position and size from a set of landmarks.

A *preshape* is what is left of a shape after you move its centre of
mass to the origin and rescale it to unit size.  Rotation is still there;
chapter 03 deals with that.

Axis convention (used everywhere in shapekit): points are rows and the
(x, y) coordinates are columns, so one shape is an array of shape (N, 2)
and a dataset is (K, N, 2).  The same shape can also be stored as N
complex numbers z = x + i y, shape (N,) or (K, N) for a dataset.  Every
function here accepts either representation.

Explained in chapter 02 (and chapter 03 for the complex form).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _points_axis(X: NDArray) -> int:
    """Axis along which the landmarks are listed: -2 for real (…, N, 2)
    arrays, -1 for complex (…, N) arrays."""
    return -1 if np.iscomplexobj(X) else -2


def center(X: ArrayLike) -> NDArray:
    """Move the centre of mass of each shape to the origin.

    Formula::

        X_c[k] = X[k] - (1/N) * sum_j X[j]

    - ``X[k]``: the k-th landmark of a shape, as an (x, y) pair (or as the
      complex number x + i y).
    - ``N``: the number of landmarks.
    - ``(1/N) * sum_j X[j]``: the centroid, the plain average position of
      all landmarks.
    - ``X_c``: the centred shape; its landmarks now sum to zero.

    Works on one shape, ``(N, 2)`` or complex ``(N,)``, or on a whole
    dataset, ``(K, N, 2)`` or complex ``(K, N)``: each shape is centred
    on its own centroid.  Subtracting a constant from every landmark
    removes *position* and nothing else.

    Explained in chapter 02.
    """
    X = np.asarray(X)
    axis = _points_axis(X)
    return X - X.mean(axis=axis, keepdims=True)


def frobenius_norm(X: ArrayLike) -> NDArray:
    """Size of each shape: the root of the summed squared coordinates.

    Formula::

        ||X|| = sqrt( sum_k ( x_k^2 + y_k^2 ) ) = sqrt( sum_k |z_k|^2 )

    - ``x_k, y_k``: the coordinates of landmark k (``z_k`` in complex form).
    - ``||X||``: the Frobenius norm.  For a centred shape it is the root
      of the summed squared distances of the landmarks from the centroid,
      so it grows with the overall size of the shape.

    Returns a scalar for one shape or a ``(K,)`` array for a dataset.

    Explained in chapter 02.
    """
    X = np.asarray(X)
    if np.iscomplexobj(X):
        return np.sqrt(np.sum(np.abs(X) ** 2, axis=-1))
    return np.sqrt(np.sum(X.real**2, axis=(-2, -1)))


def normalize(X: ArrayLike) -> NDArray:
    """Rescale each shape to unit Frobenius norm.

    Formula::

        X_n = X / ||X||

    - ``X``: a shape (usually already centred).
    - ``||X||``: its Frobenius norm, see :func:`frobenius_norm`.
    - ``X_n``: the same shape, scaled so that its norm is exactly 1.

    Dividing by the norm removes *size*: a worm and a copy twice as large
    become identical.  Works on ``(N, 2)``, ``(K, N, 2)``, complex ``(N,)``
    and complex ``(K, N)`` arrays.

    Explained in chapter 02.
    """
    X = np.asarray(X)
    norm = frobenius_norm(X)
    if np.iscomplexobj(X):
        return X / np.expand_dims(norm, -1)
    return X / np.expand_dims(norm, (-2, -1))


def preshape(X: ArrayLike) -> NDArray:
    """Centre and normalize: the preshape of a landmark configuration.

    Formula::

        P = (X - centroid(X)) / ||X - centroid(X)||

    - ``X``: the raw landmarks, ``(N, 2)`` or ``(K, N, 2)`` (or complex).
    - ``centroid(X)``: the average landmark, see :func:`center`.
    - ``||.||``: the Frobenius norm, see :func:`frobenius_norm`.
    - ``P``: the preshape.  Position and size are gone; rotation remains.

    All preshapes of N landmarks live on the unit sphere in 2N - 2
    dimensions (the "preshape sphere"), which is why chapter 04 can talk
    about arcs and chords between shapes.

    Explained in chapter 02.
    """
    return normalize(center(X))


def to_complex(X: ArrayLike) -> NDArray:
    """Turn (x, y) rows into complex numbers z = x + i y.

    Formula::

        z_k = x_k + i * y_k

    - ``x_k, y_k``: the coordinates of landmark k.
    - ``i``: the imaginary unit, i^2 = -1.
    - ``z_k``: the same point as one complex number.

    Input ``(N, 2)`` gives ``(N,)``; input ``(K, N, 2)`` gives ``(K, N)``.
    Complex numbers make rotation a single multiplication, which is the
    whole point of chapter 03.

    Explained in chapter 03.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 2:
        raise ValueError(f"expected last axis of length 2 (x, y), got shape {X.shape}")
    return X[..., 0] + 1j * X[..., 1]


def from_complex(z: ArrayLike) -> NDArray:
    """Turn complex numbers back into (x, y) rows.

    Formula::

        (x_k, y_k) = (Re z_k, Im z_k)

    - ``Re z_k``: the real part of z_k, the x coordinate.
    - ``Im z_k``: the imaginary part, the y coordinate.

    Input ``(N,)`` gives ``(N, 2)``; input ``(K, N)`` gives ``(K, N, 2)``.

    Explained in chapter 03.
    """
    z = np.asarray(z)
    return np.stack([z.real, z.imag], axis=-1)
