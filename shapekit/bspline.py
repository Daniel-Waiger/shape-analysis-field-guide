"""Cubic B-splines: shapes as smooth curves with a few control points.

A curve is a weighted sum of shifted copies of one smooth "bump", the
cubic B-spline ``beta(x)``.  The weights are the *control points*, each an
(x, y) pair, so the curve is ``c(t) = sum_k c_k beta(t - k)``.

Conventions used here:

- ``M`` is the number of control points, which is also the number of
  knots the curve passes through when fitted by interpolation.
- Open curves use parameter ``t`` in ``[0, M - 1]`` and *mirror* boundary
  conditions (``c_{-1} = c_1``, ``c_M = c_{M-2}``).  The original tutorial
  pads the knots instead and stores ``M + 4`` coefficients; both make a
  curve through the same knots.
- Closed curves use ``t`` in ``[0, M)`` and wrap around.

Explained in chapter 07.

Credits: the B-spline model and the interpolation filter are from Unser,
Aldroubi & Eden (1993) and Unser (1999).  The reference implementation that
was read while writing this module is Virginie Uhlmann's Spline Fitting
Toolbox v1.0 (``spline_models/spline_curve_model.py`` in *Shape Analysis
101*, BSD-3); nothing from it is reproduced verbatim.  See ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Pole of the cubic B-spline interpolation filter: the root of
#: z^2 + 4 z + 1 = 0 with magnitude below 1.
POLE: float = -2.0 + np.sqrt(3.0)


def cubic_bspline(x: ArrayLike) -> NDArray:
    """The cubic B-spline bump, centred at 0, 4 units wide.

    Formula::

        beta(x) = 2/3 - |x|^2 + |x|^3 / 2      if |x| < 1
        beta(x) = (2 - |x|)^3 / 6              if 1 <= |x| < 2
        beta(x) = 0                            otherwise

    - ``x``: distance from the bump's centre (any array shape).
    - ``beta(x)``: the height of the bump there; 2/3 at the centre, 1/6 at
      x = +-1, 0 at and beyond x = +-2.  It is made of four cubic pieces
      that join smoothly (continuous value, slope and curvature).

    At integer positions the shifted bumps have weights 1/6, 4/6, 1/6,
    which sum to 1, so moving all control points by the same amount moves
    the whole curve by that amount.

    Explained in chapter 07.
    """
    a = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(a)
    inner = a < 1.0
    outer = (a >= 1.0) & (a < 2.0)
    out[inner] = 2.0 / 3.0 - a[inner] ** 2 + a[inner] ** 3 / 2.0
    out[outer] = (2.0 - a[outer]) ** 3 / 6.0
    return out


def interpolation_filter(samples: ArrayLike, closed: bool = False, method: str = "solve") -> NDArray:
    """Control points whose cubic B-spline curve passes through the samples.

    At integer parameters the curve equals a blur of the control points,
    ``s_k = (c_{k-1} + 4 c_k + c_{k+1}) / 6``.  Fitting means un-blurring,
    and there are two exact ways to do it:

    ``method="solve"`` (default) writes the blur as a banded linear system,
    one row per sample, and solves it directly::

        [4 2 0 ...]           [c_0]       [s_0]
        [1 4 1 ...]  / 6  @   [c_1]   =   [s_1]      (open: mirror ends)
        [...      ]           [...]       [...]

    ``method="recursive"`` is the classic two-sweep filter of Unser,
    Aldroubi and Eden (1993), which the source tutorial's spline toolbox
    also uses::

        p       = -2 + sqrt(3)                       (the pole)
        c+[k]   = s[k] + p * c+[k - 1]               (forward sweep)
        c-[k]   = p * (c-[k + 1] - c+[k])            (backward sweep)
        c       = 6 * c-

    Both give the same control points to machine precision; the recursive
    one is what chapter 07 walks through step by step, and it is written
    here with ``scipy.signal.lfilter`` so each sweep is a one-line call.

    - ``samples``: the values to pass through, shape ``(M,)`` or
      ``(M, d)`` (each column is filtered on its own, so pass ``(M, 2)``
      for x and y at once).
    - ``p``: the pole, about -0.268: each sweep is a running average whose
      memory fades by that factor per step.  It is the root of
      ``p^2 + 4p + 1 = 0`` and exactly cancels the 1-4-1 blur.
    - ``closed``: ``True`` wraps around (periodic curve); ``False`` uses
      the mirror extension of the samples at both ends.
    - ``c``: the control points, same shape as ``samples``.

    Explained in chapter 07.
    """
    s = np.asarray(samples, dtype=float)
    if method not in ("solve", "recursive"):
        raise ValueError("method must be 'solve' or 'recursive'")
    one_d = s.ndim == 1
    cols = s[:, None] if one_d else s
    if method == "solve":
        c = _solve_1_4_1(cols, closed)
    else:
        c = np.stack([_recursive_1d(cols[:, j], closed) for j in range(cols.shape[1])], axis=1)
    return c[:, 0] if one_d else c


def _solve_1_4_1(s: NDArray, closed: bool) -> NDArray:
    """Un-blur by solving ``(1/6) * [1 4 1] @ c = s`` as a linear system.

    ``s`` is ``(M, d)``; every column is solved at once.  For a closed curve
    the blur wraps around (a circulant matrix); for an open curve the mirror
    rule ``c_{-1} = c_1`` and ``c_M = c_{M-2}`` folds the two outside
    neighbours back in, which doubles the off-diagonal entry in the first
    and last row.
    """
    M = s.shape[0]
    if M < 2:
        return s.astype(float).copy()  # a single sample interpolates itself
    A = np.zeros((M, M))
    idx = np.arange(M)
    A[idx, idx] = 4.0
    if closed:
        A[idx, (idx + 1) % M] += 1.0
        A[idx, (idx - 1) % M] += 1.0
    else:
        A[idx[:-1], idx[:-1] + 1] += 1.0
        A[idx[1:], idx[1:] - 1] += 1.0
        A[0, 1] += 1.0          # c_{-1} = c_1
        A[M - 1, M - 2] += 1.0  # c_M = c_{M-2}
    return np.linalg.solve(A / 6.0, s)


def _recursive_1d(s: NDArray, closed: bool) -> NDArray:
    """The two-sweep recursive filter for one column, via ``scipy.signal.lfilter``.

    Each sweep is a first-order recursion, which is exactly what ``lfilter``
    with denominator ``[1, -p]`` computes.  Only the *starting values* of the
    two sweeps depend on the boundary rule; they are the standard ones from
    Unser, Aldroubi and Eden (1993): a geometric sum of the extended samples
    for the forward sweep, and a closed-form combination of the last two
    forward values (open) or another geometric sum (closed) for the backward
    sweep.
    """
    from scipy.signal import lfilter

    M = len(s)
    p = POLE
    if M < 2:
        return s.astype(float).copy()
    # Extended sample sequence read backwards from index 0, as the forward
    # sweep's start value needs it: periodic or mirrored.
    if closed:
        period = M
        back = s[(-np.arange(M)) % M]
    else:
        period = 2 * M - 2
        mirrored = np.concatenate([s, s[-2:0:-1]])           # one full period
        back = mirrored[(-np.arange(period)) % period]
    start_fwd = np.sum(back * p ** np.arange(len(back))) / (1.0 - p**period)

    forward = np.empty(M)
    forward[0] = start_fwd
    forward[1:] = lfilter([1.0], [1.0, -p], s[1:], zi=[p * start_fwd])[0]

    if closed:
        tail = np.sum(p ** np.arange(M) * forward) * p / (1.0 - p**M)
        start_bwd = -p * (tail + forward[-1])
    else:
        start_bwd = (p / (p**2 - 1.0)) * (forward[-1] + p * forward[-2])

    # Backward sweep c-[k] = p * (c-[k+1] - c+[k]), run on the reversed arrays.
    rev = lfilter([-p], [1.0, -p], forward[-2::-1], zi=[p * start_bwd])[0]
    backward = np.concatenate([rev[::-1], [start_bwd]])
    return 6.0 * backward


def basis_matrix(t: ArrayLike, M: int, closed: bool = False) -> NDArray:
    """Weights of every control point at every parameter value.

    Formula::

        B[i, k] = sum_m beta(t_i - k - m M)           (closed, m = -1, 0, 1)
        B[i, k] = beta(t_i - k) + mirror terms          (open)

    - ``t``: parameter values, shape ``(n,)``; ``[0, M)`` for closed and
      ``[0, M - 1]`` for open curves.
    - ``M``: number of control points.
    - ``beta``: the bump :func:`cubic_bspline`.
    - mirror terms: for open curves the phantom control points
      ``c_{-1} = c_1`` and ``c_M = c_{M-2}`` are folded into columns 1 and
      M - 2.
    - ``B``: an ``(n, M)`` matrix; the curve is ``B @ C`` for control
      points ``C`` of shape ``(M, 2)``.  Every row sums to 1.

    Explained in chapter 07.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    k = np.arange(M)
    if closed:
        diff = t[:, None] - k[None, :]
        B = cubic_bspline(diff) + cubic_bspline(diff - M) + cubic_bspline(diff + M)
    else:
        B = cubic_bspline(t[:, None] - k[None, :])
        B[:, 1] += cubic_bspline(t + 1.0)  # c_{-1} = c_1
        B[:, M - 2] += cubic_bspline(t - M)  # c_M = c_{M-2}
    return B


def gram_matrix(M: int, closed: bool = True) -> NDArray:
    """The Gram matrix Phi: how much every pair of basis bumps overlaps.

    Formula::

        Phi[k, l] = integral  B_k(t) B_l(t)  dt      over the curve's domain

    - ``B_k``: the k-th basis function (the bump centred at k, wrapped
      for closed curves, mirrored for open ones).
    - ``Phi``: an ``M x M`` symmetric positive-definite matrix.  With it,
      the overlap of two whole curves with control points ``z`` and ``w``
      (as complex vectors) is ``z^H Phi w``: the same inner product that
      :mod:`shapekit.shape_space` uses for landmarks, but between
      continuous curves rather than between control points.
    - For closed curves ``Phi[k, l]`` depends only on ``k - l`` (mod M):
      151/315 on the diagonal, 397/1680 for neighbours, 1/42 and 1/5040 for
      the next two, 0 beyond.

    The integral is done exactly with 4-point Gauss-Legendre quadrature on
    each unit interval (the integrand is a polynomial of degree 6 there).
    The original tutorial scales the parameter to ``[0, 1]``, which only
    divides Phi by M.

    Explained in chapter 07.
    """
    nodes, weights = np.polynomial.legendre.leggauss(4)
    n_intervals = M if closed else M - 1
    Phi = np.zeros((M, M))
    for j in range(n_intervals):
        t = j + 0.5 * (nodes + 1.0)
        B = basis_matrix(t, M, closed)
        Phi += (B * (0.5 * weights)[:, None]).T @ B
    return Phi


class BSplineCurve:
    """A cubic B-spline curve with ``M`` control points.

    Formula::

        c(t) = sum_k C[k] beta(t - k)        (plus wrap or mirror terms)

    - ``M``: number of control points (at least 4).
    - ``closed``: a loop (cell outline) or an open line (worm midline).
    - ``C`` (``control_points_``): the ``(M, 2)`` control points, set by
      :meth:`fit`.
    - ``t``: the curve parameter; ``[0, M)`` closed, ``[0, M - 1]`` open.

    Explained in chapter 07.
    """

    def __init__(self, M: int, closed: bool = False) -> None:
        if M < 4:
            raise ValueError("a cubic B-spline curve needs at least 4 control points")
        self.M = int(M)
        self.closed = bool(closed)
        self.control_points_: NDArray | None = None
        #: parameter value assigned to each fitted point (set by :meth:`fit`)
        self.t_fit_: NDArray | None = None

    @property
    def t_max(self) -> float:
        """Upper end of the parameter range (``M`` closed, ``M - 1`` open)."""
        return float(self.M if self.closed else self.M - 1)

    def fit(self, points: ArrayLike) -> "BSplineCurve":
        """Find control points for a curve through (or close to) the points.

        Two cases:

        - ``len(points) == M``: exact interpolation with
          :func:`interpolation_filter`; the curve passes through every
          point, at ``t = 0, 1, ..., M - 1``.
        - ``len(points) != M``: least squares.  Each point i gets a
          parameter ``t_i`` proportional to its distance along the
          polyline, and ``C = argmin ||B(t) C - points||^2`` is solved
          with ``numpy.linalg.lstsq``.  The curve then passes *near* the
          points, smoothing them with only M control points.

        - ``points``: ``(P, 2)`` array of (x, y) rows.

        Explained in chapter 07.
        """
        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError(f"expected points of shape (P, 2), got {P.shape}")
        if len(P) == self.M:
            self.t_fit_ = np.arange(self.M, dtype=float)
            self.control_points_ = interpolation_filter(P, self.closed)
        else:
            self.t_fit_ = self._arclength_parameters(P)
            B = basis_matrix(self.t_fit_, self.M, self.closed)
            self.control_points_, *_ = np.linalg.lstsq(B, P, rcond=None)
        return self

    def _arclength_parameters(self, P: NDArray) -> NDArray:
        """Parameter per point, proportional to distance along the polyline."""
        if self.closed:
            seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            return s[:-1] / s[-1] * self.M
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return s / s[-1] * (self.M - 1)

    def evaluate(self, t: ArrayLike) -> NDArray:
        """Points on the curve at the given parameter values, ``B(t) @ C``.

        - ``t``: parameter values ``(n,)``.
        - returns ``(n, 2)`` (x, y) rows.

        Explained in chapter 07.
        """
        if self.control_points_ is None:
            raise RuntimeError("call fit(points) first, or set control_points_")
        return basis_matrix(t, self.M, self.closed) @ self.control_points_

    def sample(self, n_points: int = 200) -> NDArray:
        """``n_points`` evenly spaced (in parameter) points along the curve.

        - open curves: ``t`` from 0 to M - 1 inclusive;
        - closed curves: ``t`` from 0 up to (not including) M, so the
          first point is not repeated.

        Returns ``(n_points, 2)``.

        Explained in chapter 07.
        """
        t = np.linspace(0.0, self.t_max, n_points, endpoint=not self.closed)
        return self.evaluate(t)

    def knots(self) -> NDArray:
        """The curve at the integer parameters ``t = 0, 1, ...``: the points an
        interpolating fit passes through.  Returns ``(M, 2)``.

        Explained in chapter 07.
        """
        return self.evaluate(np.arange(self.M, dtype=float))
