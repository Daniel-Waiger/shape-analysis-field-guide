"""The Phenotype Lab's classical engine, in plain numpy (chapter 10).

``docs/lab/engines/classical.js`` classifies a shape in the browser with the
numbers that ``scripts/run_lab.py`` writes into ``<dataset>.json["engine"]``.
This module is the *reference implementation* of exactly that arithmetic, so
the JavaScript can be checked against it and the JSON can be checked against
the scikit-learn pipeline it was distilled from.

The engine dictionary (see LAB_SPEC.md, "JSON contract") holds::

    mean           [[re, im] x N]        the Fréchet mean preshape
    pca_mean       [2N]                  mean tangent vector (re then im)
    components     [[2N] x n_components] tangent PCA modes, one per row
    scaler_mean    [n_components]        StandardScaler.mean_
    scaler_scale   [n_components]        StandardScaler.scale_
    lr_coef        [[n_components] x C]  LogisticRegression.coef_ (C rows)
    lr_intercept   [C]
    tangent_layout "re_then_im"

and :func:`classify` runs, for one shape ``X`` of ``(N, 2)`` points::

    z      = preshape(X)                              centre, unit Frobenius norm
    theta  = angle( sum_k conj(z_k) m_k )             best rotation onto the mean
    z~     = exp(i theta) z
    rho    = arccos( Re <m, z~> )                     geodesic distance to the mean
    v      = (rho / sin rho) (z~ - cos(rho) m)        log map
    r      = [Re v, Im v] - pca_mean                  2N reals
    s      = r @ components^T                         n_components scores
    x      = (s - scaler_mean) / scaler_scale
    logits = lr_coef @ x + lr_intercept
    p      = softmax(logits)

For two classes scikit-learn stores one row ``w`` and uses ``sigmoid(w·x + b)``;
the engine stores the equivalent two rows ``[-w/2, w/2]`` and intercepts
``[-b/2, b/2]``, so that ``softmax`` gives exactly the same probabilities and
the JavaScript needs a single code path (``binary_rows: "symmetric"``).

Everything here uses the same functions as the rest of shapekit; nothing is
re-derived.  Axis convention: points are rows, (x, y) are columns.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .preshape import from_complex, preshape, to_complex
from .procrustes import optimal_rotation_angle
from .shape_space import log_map
from .tangent_pca import tangent_to_real

__all__ = ["classify", "classify_many", "explain", "engine_from_fit", "softmax"]


def softmax(logits: ArrayLike) -> NDArray:
    """``exp(l_c - max) / sum_c exp(l_c - max)``: probabilities that sum to 1."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _engine_arrays(engine: dict) -> dict:
    mean = np.asarray(engine["mean"], dtype=float)
    return {
        "mean": mean[:, 0] + 1j * mean[:, 1],
        "pca_mean": np.asarray(engine["pca_mean"], dtype=float),
        "components": np.asarray(engine["components"], dtype=float),
        "scaler_mean": np.asarray(engine["scaler_mean"], dtype=float),
        "scaler_scale": np.asarray(engine["scaler_scale"], dtype=float),
        "lr_coef": np.asarray(engine["lr_coef"], dtype=float),
        "lr_intercept": np.asarray(engine["lr_intercept"], dtype=float),
    }


def explain(shape_xy: ArrayLike, engine: dict) -> dict:
    """Run the engine on one shape and return every intermediate quantity.

    - ``shape_xy``: ``(N, 2)`` points (any position, size and rotation).
    - ``engine``: the dictionary described in the module docstring.

    Returns a dict with ``centroid`` (2,), ``size`` (Frobenius norm of the
    centred shape), ``angle`` (radians, the rotation applied to reach the
    mean), ``aligned`` (``(N, 2)`` aligned preshape), ``rho`` (geodesic
    distance to the mean), ``tangent`` (2N), ``scores`` (n_components),
    ``standardised`` (n_components), ``logits`` (C) and ``proba`` (C).
    This is what the Classify screen's stepper shows.  Chapter 10.
    """
    E = _engine_arrays(engine)
    X = np.asarray(shape_xy, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"expected one shape of (N, 2) points, got {X.shape}")
    if X.shape[0] != E["mean"].shape[0]:
        raise ValueError(f"engine expects {E['mean'].shape[0]} points, got {X.shape[0]}")
    centroid = X.mean(axis=0)
    centred = X - centroid
    size = float(np.sqrt(np.sum(centred**2)))
    z = to_complex(preshape(X))
    theta = optimal_rotation_angle(z, E["mean"])
    aligned = np.exp(1j * theta) * z
    rho = float(np.arccos(np.clip(np.vdot(E["mean"], aligned).real, -1.0, 1.0)))
    v = log_map(E["mean"], z)
    tangent = tangent_to_real(v) - E["pca_mean"]
    scores = E["components"] @ tangent
    standardised = (scores - E["scaler_mean"]) / E["scaler_scale"]
    logits = E["lr_coef"] @ standardised + E["lr_intercept"]
    return {
        "centroid": centroid,
        "size": size,
        "angle": float(theta),
        "aligned": from_complex(aligned),
        "rho": rho,
        "tangent": tangent,
        "scores": scores,
        "standardised": standardised,
        "logits": logits,
        "proba": softmax(logits),
    }


def classify(shape_xy: ArrayLike, engine: dict) -> NDArray:
    """Class probabilities of one ``(N, 2)`` shape under the engine: ``(C,)``.

    This is the function ``docs/lab/engines/classical.js`` reproduces; see the
    module docstring for the arithmetic.  Chapter 10.
    """
    return explain(shape_xy, engine)["proba"]


def classify_many(shapes_xy: ArrayLike, engine: dict) -> NDArray:
    """:func:`classify` for a stack ``(K, N, 2)``; returns ``(K, C)``."""
    return np.stack([classify(s, engine) for s in np.asarray(shapes_xy, dtype=float)])


def engine_from_fit(mean, pca, keep: int, scaler, lr, method: str, classes: list[str]) -> dict:
    """Distil a fitted mean / TangentPCA / StandardScaler / LogisticRegression into the engine dict.

    - ``mean``: complex ``(N,)`` Fréchet mean preshape.
    - ``pca``: a fitted :class:`shapekit.TangentPCA` with ``mean_ == mean``.
    - ``keep``: how many leading modes feed the classifier.
    - ``scaler``, ``lr``: the fitted scikit-learn steps (``lr.coef_`` has one
      row for two classes; it is expanded to the symmetric two-row form).
    - ``method``, ``classes``: labels copied into the dict.

    Returns plain lists (JSON-ready, unrounded).  Chapter 10.
    """
    mean = np.asarray(mean, dtype=complex)
    coef = np.asarray(lr.coef_, dtype=float)
    intercept = np.asarray(lr.intercept_, dtype=float)
    binary = coef.shape[0] == 1
    if binary:
        coef = np.vstack([-coef[0] / 2.0, coef[0] / 2.0])
        intercept = np.array([-intercept[0] / 2.0, intercept[0] / 2.0])
    engine = {
        "method": method,
        "classes": list(classes),
        "n_points": int(mean.shape[0]),
        "mean": np.stack([mean.real, mean.imag], axis=1).tolist(),
        "n_components": int(keep),
        "pca_mean": np.asarray(pca.mean_tangent_, dtype=float).tolist(),
        "components": np.asarray(pca.components_[:keep], dtype=float).tolist(),
        "tangent_layout": "re_then_im",
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "lr_coef": coef.tolist(),
        "lr_intercept": intercept.tolist(),
        "decision": "p = softmax(lr_coef @ x + lr_intercept), x = (scores - scaler_mean) / scaler_scale",
    }
    if binary:
        engine["binary_rows"] = "symmetric"
        engine["binary_note"] = ("scikit-learn fits one row w and p1 = sigmoid(w.x + b); stored as the two rows "
                                 "[-w/2, w/2] with intercepts [-b/2, b/2], whose softmax is the same p1")
    return engine
