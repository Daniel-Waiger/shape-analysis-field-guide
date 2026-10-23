"""The Phenotype Lab's computations (chapter 10): one dataset in, one results dict out.

``scripts/run_lab.py`` calls :func:`run_dataset` for every dataset of the
registry (:mod:`shapekit.lab_datasets`) and hands the result to
:mod:`shapekit.lab_export`, which rounds it and writes the JSON of the
contract in LAB_SPEC.md.  Everything in here is deterministic for a given
seed (the PointNet part also for a given torch thread count, torch build and
CPU); nothing is typed in by hand.

The method families:

- ``landmarks_lr``: preshape → Fréchet mean → tangent PCA (modes with
  variance > 1e-10, at most 40) → StandardScaler → multinomial logistic
  regression, scored with stratified, shuffled 5-fold cross-validation.
- ``spline_lr``: the same geometry on the control points of a cubic
  B-spline (:class:`shapekit.BSplineCurve`) fitted to each ordered curve.
- ``pointnet_probe``: :func:`shapekit.embed.train_pointnet` (PointNet +
  supervised contrastive loss) and a linear probe on the embeddings, plus
  the untrained network as a baseline and an ONNX export for the browser.
  The clouds are scaled to unit RMS radius each, so the network sees shape
  only; the epoch is chosen on a slice of the training part, never on the
  scored hold-out.
- ``size_lr``: the baseline that uses *only* size (log centroid size →
  StandardScaler → logistic regression, same folds).  It says how much of a
  dataset's separability is size, which every other method has removed.

Diagnostic (not a method, never in ``methods`` or ``best_accuracy``):

- ``roundness_baseline`` (datasets in ``ROUNDNESS_BASELINE_DATASETS``, i.e.
  ``mef``): circularity + log axis ratio → StandardScaler → logistic
  regression, same folds (:func:`roundness_baseline`), circularity measured
  on each cloud put in loop order (:func:`shapekit.contours.order_loop`).
  Two hand-made numbers; it shows how much of the MEF separability is plain roundness,
  which is what the dataset's genotype/substrate confound predicts.

Preprocessing by representation:

- ``landmarks``: used as they are (anatomical correspondence);
- ``outline``: arc-length resampled to 100 points when longer (order kept);
  a *closed* outline is also re-rooted label-free at the cyclic shift that
  brings it closest to the Fréchet mean (:func:`align_cyclic_shifts`), so
  that point k means the same place on every outline;
- ``pointcloud``: only ``pointnet_probe`` and ``size_lr``; for the *display*
  blocks (PCA scatter, examples, neighbours) the points are ordered by angle
  around the centroid, which is an approximation that is labelled as such
  and never used for classification.

Axis convention: points are rows, (x, y) are columns.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .bspline import BSplineCurve
from .contours import order_loop, resample_by_arclength
from .datasets import resolve_data_dir
from .lab_datasets import LabDataset
from .lab_engine import classify, engine_from_fit
from .preshape import from_complex, preshape, to_complex
from .procrustes import align_dataset
from .shape_space import frechet_mean
from .tangent_pca import TangentPCA

__all__ = [
    "N_OUTLINE_POINTS",
    "SPLINE_M",
    "EPOCHS",
    "POINTNET_HP",
    "VAL_FRACTION",
    "SELECT_FRACTION",
    "resolve_classes",
    "dataset_notes",
    "align_cyclic_shifts",
    "prepare_points",
    "classical_method",
    "spline_method",
    "size_method",
    "circularity",
    "axis_ratio",
    "roundness_baseline",
    "pca_block",
    "examples_block",
    "neighbours_block",
    "export_onnx",
    "pointnet_method",
    "run_dataset",
    "default_epochs",
]

#: outlines longer than this are resampled by arc length to this many points
N_OUTLINE_POINTS = 100
#: rounds of "re-root every closed outline at the mean's best cyclic shift, recompute the mean"
CYCLIC_ROUNDS = 3
#: a best shift within this fraction of the perimeter of the source's start counts as "already there"
START_NEAR_ZERO = 0.1
#: when at least this fraction of the outlines is already there, the source's start is a landmark and is kept
START_LANDMARK_MAJORITY = 0.5
#: number of spline control points per dataset (closed outlines 12, the worms' open midline 6)
SPLINE_M = {"celegans": 6, "leaves": 12, "rbc": 12}
#: PointNet training epochs per dataset (chosen by size; see LAB_SPEC.md)
EPOCHS = {"celegans": 40, "mef": 60, "leaves": 60, "rbc": 40}
#: PointNet hyper-parameters, typed once (the JSON strings are built from this dict)
POINTNET_HP = {"temperature": 0.07, "lr": 3e-4, "weight_decay": 1e-3, "batch_size": 64, "embed_dim": 256,
               "proj_dim": 64, "augment": True}
#: PointNet splits: the hold-out that is scored, and the slice of the training part that picks the epoch
VAL_FRACTION = 0.25
SELECT_FRACTION = 0.2
#: fewer shapes than this and PointNet is skipped
MIN_POINTNET_N = 100
VARIANCE_FLOOR = 1e-10
MAX_COMPONENTS = 40
N_FOLDS = 5
N_WORST = 3
N_EXAMPLES = 5
N_SCATTER = 400
N_NEIGHBOURS = 300
LR_C = 1.0
LR_MAX_ITER = 3000
#: datasets that get the display-only ``roundness_baseline`` block (the MEF confound, LAB_SPEC.md Round 3b)
ROUNDNESS_BASELINE_DATASETS = ("mef",)
ROUNDNESS_LABEL = "roundness only (two hand-made numbers)"
#: how ``roundness_baseline`` orders each point cloud before measuring its circularity
POINT_ORDER = "nearest-neighbor + 2-opt loop"

#: the ``protocol`` strings every method block carries
CV_PROTOCOL = "stratified {k}-fold CV, seed {seed}, pooled out-of-fold"
HOLDOUT_PROTOCOL = ("stratified {train}/{val} hold-out, seed {seed}; epoch chosen on a {select} % slice of the "
                    "training part")
SD_DEF = "sd of fold_accuracies over the folds, ddof=0 (population sd)"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def default_epochs(dataset_id: str, n: int) -> int:
    """Epochs for PointNet: the per-dataset table, else 60 (40 when n > 2000)."""
    if dataset_id in EPOCHS:
        return EPOCHS[dataset_id]
    return 40 if n > 2000 else 60


def cv_protocol(seed: int) -> str:
    return CV_PROTOCOL.format(k=N_FOLDS, seed=seed)


def holdout_protocol(seed: int) -> str:
    return HOLDOUT_PROTOCOL.format(train=round(100 * (1 - VAL_FRACTION)), val=round(100 * VAL_FRACTION),
                                   seed=seed, select=round(100 * SELECT_FRACTION))


def resolve_classes(ds: LabDataset, labels: NDArray) -> list[str]:
    """Class names: the registry's, else ``class k`` (every registered dataset names its classes)."""
    n_classes = int(labels.max()) + 1 if len(labels) else 0
    names = [str(c) for c in ds.classes]
    if len(names) >= n_classes:
        return names
    return [f"class {k}" for k in range(n_classes)]


def _dataset_npz_files(ds: LabDataset) -> list[Path]:
    base = Path(resolve_data_dir())
    return sorted(p for folder in (base, base / "_local") for p in folder.glob(f"{ds.id}_*.npz"))


def dataset_notes(ds: LabDataset) -> list[str]:
    """Preprocessing notes the fetch script stored with the data (a ``preprocessing`` string array in the npz).

    ``fetch_datasets.py`` records what it did to the raw source (e.g. the
    red-cell extraction: fields read, masks missing, components dropped) in
    the dataset's ``.npz``; those lines go into the JSON's ``preprocessing``
    so the counts on the page are traceable.  Empty when there is no such key.
    """
    notes: list[str] = []
    for path in _dataset_npz_files(ds):
        try:
            with np.load(path, allow_pickle=False) as f:
                if "preprocessing" in f.files:
                    notes.extend(str(s) for s in np.asarray(f["preprocessing"]).reshape(-1) if str(s))
        except (OSError, ValueError):
            continue
    return notes


def _order_by_angle(points: NDArray) -> NDArray:
    """Sort each cloud's points by angle around its centroid (display only)."""
    out = np.empty_like(points)
    for k, cloud in enumerate(points):
        c = cloud - cloud.mean(axis=0)
        order = np.argsort(np.arctan2(c[:, 1], c[:, 0]), kind="stable")
        out[k] = cloud[order]
    return out


def _mean_geodesic(Z: NDArray, mean: NDArray) -> float:
    return float(np.mean(np.arccos(np.clip(np.abs(Z @ np.conj(mean)), 0.0, 1.0))))


def align_cyclic_shifts(landmarks: NDArray, rounds: int = CYCLIC_ROUNDS) -> tuple[NDArray, dict]:
    """Re-root closed outlines at the cyclic shift that brings each closest to the Fréchet mean.

    Arc-length resampling puts the points of a closed outline in
    correspondence *up to a cyclic shift*: where the traced contour happened
    to start.  A rotation-only Procrustes cannot undo that shift, so this
    canonicalises the start point label-free: starting from the current
    Fréchet mean, every outline is rolled (same direction, no reversal) to
    the shift that minimises its Procrustes distance to the mean (the shift
    maximising ``|<z_shifted, mean>|``, which is the full Procrustes fit over
    rotations), the mean is recomputed, and this is repeated ``rounds`` times.

    ``landmarks`` is ``(K, N, 2)`` ordered closed outlines.  Returns the
    rolled copy and a dict: ``shifts`` (per outline, cumulative), ``rounds``,
    ``n_shifted``, ``fraction_near_zero`` (outlines whose best shift is
    within ``START_NEAR_ZERO`` of the perimeter of where the source started:
    about 2 × ``START_NEAR_ZERO`` when the source's starts are arbitrary,
    near 1 when they already agree), ``mean_distance_before`` /
    ``mean_distance_after`` (mean geodesic distance to the mean) and
    ``changed_per_round``.  :func:`prepare_points` uses ``fraction_near_zero``
    to decide, label-free, whether the source's start point is a landmark
    worth keeping (the leaves: every outline starts at the leaf base) or an
    artefact worth removing (the red cells: wherever the contour tracer
    began).  Chapter 10.
    """
    X = np.array(landmarks, dtype=float)   # a copy: the input is never rolled in place
    K, N, _ = X.shape
    Z = to_complex(preshape(X))
    mean = frechet_mean(Z)
    before = _mean_geodesic(Z, mean)
    shifts = np.zeros(K, dtype=int)
    changed_per_round = []
    for _ in range(rounds):
        # |<roll(z, -s), mean>| for every outline and every shift s
        score = np.empty((K, N))
        for s in range(N):
            score[:, s] = np.abs(np.roll(Z, -s, axis=1) @ np.conj(mean))
        best = score.argmax(axis=1)
        changed_per_round.append(int(np.sum(best != 0)))
        for k in np.flatnonzero(best):
            Z[k] = np.roll(Z[k], -best[k])
            X[k] = np.roll(X[k], -best[k], axis=0)
        shifts = (shifts + best) % N
        mean = frechet_mean(Z)
    near_zero = np.minimum(shifts, N - shifts) <= START_NEAR_ZERO * N
    info = {
        "rounds": int(rounds),
        "shifts": shifts.tolist(),
        "n_shifted": int(np.sum(shifts != 0)),
        "fraction_near_zero": float(np.mean(near_zero)),
        "mean_distance_before": before,
        "mean_distance_after": _mean_geodesic(Z, mean),
        "changed_per_round": changed_per_round,
    }
    return X, info


def prepare_points(ds: LabDataset, points: NDArray) -> dict:
    """Preprocess by representation; see the module docstring.

    Returns ``landmarks`` (``(K, N, 2)`` ordered points with correspondence,
    or ``None`` for a point cloud), ``display`` (ordered points for the
    display blocks; the angle-ordered cloud for a point cloud), ``cloud``
    (the raw points for PointNet), ``spline`` (``{"M", "closed"}`` or
    ``None``), ``n_points``, ``correspondence`` (how point k of one shape is
    matched to point k of another) and ``notes``.
    """
    points = np.asarray(points, dtype=float)
    notes = []
    spline = None
    cyclic = None
    if ds.representation == "landmarks":
        landmarks = points
        correspondence = "anatomical landmarks"
        if ds.id in SPLINE_M:
            spline = {"M": SPLINE_M[ds.id], "closed": bool(ds.closed)}
        else:
            notes.append("spline_lr not applicable: anatomical landmarks are not an ordered curve")
        display = landmarks
    elif ds.representation == "outline":
        if points.shape[1] > N_OUTLINE_POINTS:
            landmarks = np.stack([resample_by_arclength(p, N_OUTLINE_POINTS, closed=ds.closed) for p in points])
            notes.append(f"outline resampled by arc length from {points.shape[1]} to {N_OUTLINE_POINTS} points")
        else:
            landmarks = points
        correspondence = "arc length"
        if ds.closed:
            aligned, cyclic = align_cyclic_shifts(landmarks)
            cyclic["applied"] = cyclic["fraction_near_zero"] < START_LANDMARK_MAJORITY
            cyclic["criterion"] = (f"applied unless at least {START_LANDMARK_MAJORITY:.0%} of the outlines already "
                                   f"start within {START_NEAR_ZERO:.0%} of the perimeter of their best shift")
            if cyclic["applied"]:
                landmarks = aligned
                correspondence = "arc length + cyclic-shift alignment to the mean"
                notes.append(
                    f"closed outlines re-rooted at the cyclic shift closest to the Fréchet mean ({cyclic['rounds']} "
                    f"rounds; {cyclic['n_shifted']} of {len(landmarks)} outlines shifted, only "
                    f"{cyclic['fraction_near_zero']:.0%} already started near their best shift; mean geodesic "
                    f"distance to the mean {cyclic['mean_distance_before']:.4f} -> {cyclic['mean_distance_after']:.4f})")
            else:
                correspondence = "arc length from the source's start point (kept: it is a landmark)"
                notes.append(
                    f"closed outlines keep the source's start point: {cyclic['fraction_near_zero']:.0%} of them "
                    f"already start within {START_NEAR_ZERO:.0%} of the perimeter of their best cyclic shift to the "
                    f"Fréchet mean, so the start is a landmark (re-rooting would only blur it)")
        spline = {"M": SPLINE_M.get(ds.id, 12), "closed": bool(ds.closed)}
        display = landmarks
    elif ds.representation == "pointcloud":
        landmarks = None
        correspondence = "none (unordered points)"
        display = _order_by_angle(points)
        notes.append("points are unordered: landmarks_lr and spline_lr not applicable; "
                     "display blocks use points ordered by angle around the centroid")
    else:
        raise ValueError(f"unknown representation {ds.representation!r}")
    return {
        "landmarks": landmarks,
        "display": display,
        "cloud": points,
        "spline": spline,
        "n_points": int(points.shape[1] if landmarks is None else landmarks.shape[1]),
        "correspondence": correspondence,
        "cyclic": cyclic,
        "notes": notes,
    }


def _spline_control_points(landmarks: NDArray, M: int, closed: bool) -> NDArray:
    return np.stack([BSplineCurve(M, closed=closed).fit(p).control_points_ for p in landmarks])


def _fold_accuracies(cv, X, y, pred) -> list[float]:
    return [float(np.mean(pred[test] == y[test])) for _, test in cv.split(X, y)]


def _confusion(y, pred, n_classes: int) -> list[list[int]]:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y, pred, labels=list(range(n_classes))).astype(int).tolist()


def _worst_indices(y, pred, proba, n=N_WORST) -> NDArray:
    """Indices of the ``n`` wrong predictions with the highest confidence in the wrong class."""
    wrong = np.flatnonzero(pred != y)
    conf = proba[wrong, pred[wrong]]
    order = np.lexsort((wrong, -conf))  # by confidence desc, then index asc (deterministic)
    return wrong[order][:n]


def _chance(labels: NDArray) -> float:
    return float(np.bincount(labels).max() / len(labels))


def _lr_pipeline(C: float = LR_C, max_iter: int = LR_MAX_ITER):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C))


def _cv_predict(features: NDArray, y: NDArray, seed: int, C: float = LR_C, max_iter: int = LR_MAX_ITER):
    """Out-of-fold probabilities of the scaler + LR pipeline; returns ``(pipe, cv, proba, pred, folds)``."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    pipe = _lr_pipeline(C, max_iter)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, features, y, cv=cv, method="predict_proba")
    pred = proba.argmax(axis=1)
    return pipe, cv, proba, pred, _fold_accuracies(cv, features, y, pred)


def _cv_summary(y: NDArray, pred: NDArray, folds: list[float], n_classes: int, seed: int, C: float,
                max_iter: int) -> dict:
    """The fields every cross-validated method block shares."""
    return {
        "status": "ready",
        "accuracy": float(np.mean(pred == y)),
        "accuracy_sd": float(np.std(folds)),
        "accuracy_sd_def": SD_DEF,
        "fold_accuracies": folds,
        "chance": _chance(y),
        "n": int(len(y)),
        "n_folds": N_FOLDS,
        "protocol": cv_protocol(seed),
        "cv": f"StratifiedKFold({N_FOLDS}, shuffle=True, random_state={seed}); cross_val_predict",
        "classifier": f"StandardScaler + LogisticRegression(C={C}, max_iter={max_iter}, lbfgs, multinomial)",
        "cm": _confusion(y, pred, n_classes),
        "cm_layout": "rows true, cols predicted, class order",
    }


# --------------------------------------------------------------------------- #
# classical methods
# --------------------------------------------------------------------------- #


def shape_geometry(X: NDArray) -> dict:
    """Preshapes, Fréchet mean, tangent PCA, aligned shapes and scores of ordered points ``(K, N, 2)``.

    No labels are involved.  Returns ``Z`` (complex preshapes), ``mean``,
    ``pca`` (a fitted :class:`TangentPCA`), ``keep`` (modes with variance
    above the floor, at most ``MAX_COMPONENTS``), ``scores`` (all modes) and
    ``aligned`` (``(K, N, 2)`` preshapes rotated onto the mean).
    """
    Z = to_complex(preshape(X))
    mean = frechet_mean(Z)
    pca = TangentPCA(mean=mean).fit(Z)
    keep = int(min(MAX_COMPONENTS, np.sum(pca.explained_variance_ > VARIANCE_FLOOR)))
    return {"Z": Z, "mean": mean, "pca": pca, "keep": keep, "scores": pca.transform(Z),
            "aligned": from_complex(align_dataset(Z, mean))}


def classical_method(X: NDArray, labels: NDArray, classes: list[str], method: str, seed: int = 0,
                     C: float = LR_C, max_iter: int = LR_MAX_ITER) -> tuple[dict, dict]:
    """``landmarks_lr`` / ``spline_lr`` on ordered points ``X`` of ``(K, N, 2)``.

    Returns ``(result, extras)``: ``result`` is the method block of the JSON
    (accuracy, folds, chance, cm, worst, engine, ...), ``extras`` is
    :func:`shape_geometry` of ``X``.  The mean and the tangent PCA are fitted
    once on all shapes (they use no labels); the scaler and the regression
    are refit inside every fold by ``cross_val_predict``.
    """
    y = np.asarray(labels).astype(int)
    n_classes = len(classes)
    extras = shape_geometry(X)
    keep, aligned = extras["keep"], extras["aligned"]
    features = extras["scores"][:, :keep]
    pipe, _cv, proba, pred, folds = _cv_predict(features, y, seed, C, max_iter)

    # the engine: the same pipeline fitted on every shape
    pipe.fit(features, y)
    engine = engine_from_fit(extras["mean"], extras["pca"], keep, pipe[0], pipe[1], method, classes)
    worst_idx = _worst_indices(y, pred, proba)
    p_full = pipe.predict_proba(features[worst_idx]) if len(worst_idx) else np.zeros((0, n_classes))
    worst = []
    for i, k in enumerate(worst_idx):
        p_engine = classify(aligned[k], engine)
        if not np.allclose(p_engine, p_full[i], atol=1e-6):
            raise AssertionError(f"{method}: lab_engine.classify differs from the sklearn pipeline on shape {k}: "
                                 f"{p_engine} vs {p_full[i]}")
        worst.append({
            "index": int(k), "true": int(y[k]), "pred": int(pred[k]),
            "p": float(proba[k, pred[k]]), "p_cv": proba[k].tolist(), "p_engine": p_full[i].tolist(),
            "shape": aligned[k],
        })
    result = _cv_summary(y, pred, folds, n_classes, seed, C, max_iter)
    result.update({
        "n_components": keep,
        "size_information": "removed",
        "note": "mean and tangent PCA fitted on all shapes (label-free); scaler and regression refit in every fold",
        "worst": worst,
        "worst_note": "p / p_cv: out-of-fold probabilities; p_engine: the engine (fitted on all shapes) on the same shape",
        "engine": engine,
    })
    return result, extras


def spline_method(landmarks: NDArray, labels: NDArray, classes: list[str], M: int, closed: bool,
                  seed: int = 0) -> tuple[dict, dict]:
    """``spline_lr``: control points of a cubic B-spline through each curve, then :func:`classical_method`."""
    control = _spline_control_points(landmarks, M, closed)
    result, extras = classical_method(control, labels, classes, "spline_lr", seed=seed)
    result["M"] = int(M)
    result["closed"] = bool(closed)
    result["input"] = "control_points"
    result["spline"] = (f"BSplineCurve(M={M}, closed={closed}).fit(points): least-squares control points with "
                        f"arc-length parameters; the engine classifies the (M, 2) control polygon")
    result["engine"]["M"] = int(M)
    result["engine"]["closed"] = bool(closed)
    result["engine"]["input"] = "control_points"
    # the fitted curve of each worst mistake, for drawing (control points alone look like a polygon)
    for w in result["worst"]:
        curve = BSplineCurve(M, closed=closed)
        curve.control_points_ = np.asarray(w["shape"], dtype=float)
        w["curve"] = curve.sample(60)
    return result, extras


def centroid_size(points: NDArray) -> NDArray:
    """Centroid size of each shape: the Frobenius norm of the centred points, ``(K,)``."""
    P = np.asarray(points, dtype=float)
    centred = P - P.mean(axis=1, keepdims=True)
    return np.sqrt(np.sum(centred**2, axis=(1, 2)))


def size_method(points: NDArray, labels: NDArray, classes: list[str], seed: int = 0, C: float = LR_C,
                max_iter: int = LR_MAX_ITER) -> dict:
    """``size_lr`` ("Size only"): log centroid size → StandardScaler → logistic regression, same folds.

    The one feature is the logarithm of the centroid size (chapter 02) of
    the points as given (landmarks, the resampled outline, or the cloud);
    everything else is as in :func:`classical_method`.  No shapes, no engine:
    the block says how much of the separability is size, which every other
    method has removed.
    """
    y = np.asarray(labels).astype(int)
    n_classes = len(classes)
    size = centroid_size(points)
    features = np.log(size)[:, None]
    _pipe, _cv, _proba, pred, folds = _cv_predict(features, y, seed, C, max_iter)
    result = _cv_summary(y, pred, folds, n_classes, seed, C, max_iter)
    result.update({
        "feature": "log centroid size (Frobenius norm of the centred points, in the data's units)",
        "size_information": "only",
        "n_components": 1,
        "size_median_by_class": [float(np.median(size[y == c])) for c in range(n_classes)],
        "note": "baseline: size is the only input; never a headline number (excluded from best_accuracy)",
    })
    return result


def circularity(ordered: NDArray) -> NDArray:
    """``4πA / P²`` of each closed polygon ``(K, N, 2)`` (shoelace area, perimeter incl. the closing edge).

    1 for a circle, smaller for anything elongated or ragged.  The points
    must be in loop order: for a point cloud use
    :func:`shapekit.contours.order_loop` (as :func:`roundness_baseline`
    does), not angle order around the centroid, under which a
    non-star-shaped outline zig-zags and its circularity comes out too low.
    """
    P = np.asarray(ordered, dtype=float)
    x, y = P[..., 0], P[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    perimeter = np.sum(np.linalg.norm(P - np.roll(P, -1, axis=1), axis=2), axis=1)
    return 4.0 * np.pi * area / perimeter**2


def axis_ratio(points: NDArray) -> NDArray:
    """``sqrt(λ_max / λ_min)`` of each shape's 2 × 2 point covariance (≥ 1; order-free), ``(K,)``."""
    P = np.asarray(points, dtype=float)
    centred = P - P.mean(axis=1, keepdims=True)
    cov = np.einsum("kni,knj->kij", centred, centred) / P.shape[1]
    ev = np.linalg.eigvalsh(cov)  # ascending
    return np.sqrt(ev[:, 1] / np.maximum(ev[:, 0], np.finfo(float).tiny))


def roundness_baseline(points: NDArray, labels: NDArray, classes: list[str], seed: int = 0, C: float = LR_C,
                       max_iter: int = LR_MAX_ITER) -> dict:
    """Display-only diagnostic: circularity + log axis ratio → StandardScaler → logistic regression.

    Same folds and classifier as the methods (stratified 5-fold, shuffled,
    ``random_state=seed``; pooled out-of-fold accuracy).  Circularity is
    :func:`circularity` of each cloud put in loop order by
    :func:`shapekit.contours.order_loop` (nearest-neighbor walk + 2-opt; angle
    order around the centroid would zig-zag on cells with protrusions and
    push their circularity down); the axis ratio is :func:`axis_ratio` of
    the points as given (order-free).  Not a method: it lives outside
    ``methods`` and never counts for ``best_accuracy``.
    """
    y = np.asarray(labels).astype(int)
    n_classes = len(classes)
    looped = np.stack([order_loop(cloud) for cloud in np.asarray(points, dtype=float)])
    circ = circularity(looped)
    ratio = axis_ratio(points)
    features = np.column_stack([circ, np.log(ratio)])
    _pipe, _cv, _proba, pred, folds = _cv_predict(features, y, seed, C, max_iter)
    result = _cv_summary(y, pred, folds, n_classes, seed, C, max_iter)
    result.update({
        "label": ROUNDNESS_LABEL,
        "features": ["circularity 4πA/P² of the outline with the points in loop order",
                     "log axis ratio, sqrt(λ_max/λ_min) of the point covariance"],
        "point_order": POINT_ORDER,
        "n_features": 2,
        "circularity_median_by_class": [float(np.median(circ[y == c])) for c in range(n_classes)],
        "axis_ratio_median_by_class": [float(np.median(ratio[y == c])) for c in range(n_classes)],
        "role": "diagnostic",
        "note": ("display-only diagnostic, not a method (excluded from methods and best_accuracy): how far plain "
                 "roundness separates the classes"),
    })
    return result


# --------------------------------------------------------------------------- #
# display blocks
# --------------------------------------------------------------------------- #


def pca_block(extras: dict, labels: NDArray, seed: int = 0, note: str | None = None) -> dict:
    """Explained variance (10), mean, two modes at ±1 SD, a seeded scatter of ≤ 400 shapes."""
    pca: TangentPCA = extras["pca"]
    scores = extras["scores"]
    rng = np.random.default_rng(seed)
    n = len(labels)
    pick = np.sort(rng.choice(n, size=min(N_SCATTER, n), replace=False))
    modes = []
    for i in range(2):
        plus, minus = pca.mode_shape(i, 1.0)
        modes.append({"sd": float(pca.sdev_[i]), "plus": from_complex(plus), "minus": from_complex(minus)})
    block = {
        "explained": pca.explained_variance_ratio_[:10].tolist(),
        "sdev": pca.sdev_[:10].tolist(),
        "n_components": int(extras["keep"]),
        "tangent_layout": "re_then_im",
        "mean": from_complex(pca.mean_),
        "modes": modes,
        "scatter": [[float(scores[k, 0]), float(scores[k, 1]), int(labels[k])] for k in pick],
        "scatter_indices": pick.tolist(),
    }
    if note:
        block = {"note": note, **block}
    return block


def examples_block(aligned: NDArray, labels: NDArray, n_classes: int, seed: int = 0) -> tuple[dict, dict]:
    """Five aligned preshapes per class (seeded); returns ``(examples, indices)`` keyed by class id."""
    rng = np.random.default_rng(seed)
    examples, indices = {}, {}
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        pick = np.sort(rng.choice(idx, size=min(N_EXAMPLES, len(idx)), replace=False))
        examples[str(c)] = [aligned[k] for k in pick]
        indices[str(c)] = pick.tolist()
    return examples, indices


def neighbours_block(aligned: NDArray, labels: NDArray, seed: int = 0) -> dict:
    """≤ 300 aligned preshapes with labels and indices, for the nearest-neighbour display."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    pick = np.sort(rng.choice(n, size=min(N_NEIGHBOURS, n), replace=False))
    return {"shapes": [aligned[k] for k in pick], "labels": labels[pick].astype(int).tolist(),
            "indices": pick.tolist(), "aligned_to": "mean"}


# --------------------------------------------------------------------------- #
# PointNet + linear probe
# --------------------------------------------------------------------------- #


def _probe_dict(probe, classes: list[str]) -> dict:
    scaler, lr = probe[0], probe[1]
    coef = np.asarray(lr.coef_, dtype=float)
    intercept = np.asarray(lr.intercept_, dtype=float)
    out = {"classes": list(classes), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
           "decision": "p = softmax(coef @ ((embedding - scaler_mean) / scaler_scale) + intercept)"}
    if coef.shape[0] == 1:
        coef = np.vstack([-coef[0] / 2.0, coef[0] / 2.0])
        intercept = np.array([-intercept[0] / 2.0, intercept[0] / 2.0])
        out["binary_rows"] = "symmetric"
    out["coef"] = coef.tolist()
    out["intercept"] = intercept.tolist()
    return out


def export_onnx(model, path: Path, example: NDArray, opset: int = 17) -> dict:
    """Export the embedding half of a :class:`shapekit.embed.PointNet` to ONNX and verify it.

    Input ``points`` ``(B, N, 2)`` float32, *already* centred, divided by the
    cloud's own RMS radius when the model was trained with ``normalize_size``,
    and divided by ``model.input_scale`` (the browser does that); output
    ``embedding`` ``(B, embed_dim)``.  Batch and N are dynamic.  The file is
    checked with onnxruntime against torch on ``example`` (preprocessed
    clouds) and the maximum absolute difference is returned.
    """
    import warnings

    import torch

    class EmbedOnly(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, points):
            return self.net(points)[1]

    wrapper = EmbedOnly(model).eval()
    sample = torch.as_tensor(np.asarray(example[:2], dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, (sample,), str(path), input_names=["points"], output_names=["embedding"],
                          opset_version=opset, dynamic_axes={"points": {0: "batch", 1: "n_points"},
                                                             "embedding": {0: "batch"}}, dynamo=False)
    info = {"opset": opset,
            "input": "points (batch, n_points, 2) float32, centred, "
                     + ("divided by the cloud's own RMS radius, then " if model.normalize_size else "")
                     + "divided by input_scale",
            "output": "embedding", "bytes": int(path.stat().st_size)}
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover
        info["verified"] = False
        info["note"] = "onnxruntime not installed; export not verified"
        return info
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.asarray(example, dtype=np.float32)
    with torch.no_grad():
        ref = wrapper(torch.as_tensor(x)).numpy()
    got = session.run(None, {"points": x})[0]
    diff = float(np.abs(got - ref).max())
    if diff > 1e-4:
        raise AssertionError(f"ONNX embeddings differ from torch by {diff:.2e} (> 1e-4)")
    info["verified"] = True
    info["max_abs_diff_vs_torch"] = diff
    return info


def pointnet_method(points: NDArray, labels: NDArray, classes: list[str], epochs: int, seed: int = 0,
                    onnx_path: Path | None = None, onnx_rel: str | None = None, threads: int = 4,
                    log=None) -> dict:
    """``pointnet_probe``: train, probe, baseline, confusion, worst mistakes, ONNX, probe weights.

    Protocol (LAB_SPEC.md, Round 2): a stratified 75/25 split; the 75 % is
    split again into fit (80 %) and select (20 %); the network trains on
    fit, the epoch is chosen on select, and ``accuracy`` is the probe fitted
    on the fit embeddings scored on the 25 % hold-out that nothing else has
    seen.  Every cloud is scaled to unit RMS radius, so size is removed.
    """
    import torch

    from .embed import PointNet, linear_probe_accuracy, preprocess_points, train_pointnet

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    y = np.asarray(labels).astype(int)
    n_classes = len(classes)
    hp = POINTNET_HP
    t0 = time.perf_counter()
    result = train_pointnet(points, y, epochs=epochs, batch_size=hp["batch_size"], lr=hp["lr"],
                            weight_decay=hp["weight_decay"], augment=hp["augment"], seed=seed,
                            embed_dim=hp["embed_dim"], proj_dim=hp["proj_dim"], temperature=hp["temperature"],
                            val_fraction=VAL_FRACTION, select_fraction=SELECT_FRACTION, normalize_size=True, log=log)
    train_seconds = time.perf_counter() - t0
    model, scale = result["model"], float(result["scale"])
    fit_idx, select_idx, val_idx = result["fit_idx"], result["select_idx"], result["val_idx"]
    y_fit, y_val = y[fit_idx], y[val_idx]

    emb_fit = model.embed(points[fit_idx]).astype(np.float64)
    emb_val = model.embed(points[val_idx]).astype(np.float64)
    probe = _lr_pipeline(max_iter=2000).fit(emb_fit, y_fit)
    proba_val = probe.predict_proba(emb_val)
    pred_val = proba_val.argmax(axis=1)
    holdout = float(np.mean(pred_val == y_val))
    cv_val, cv_folds = linear_probe_accuracy(emb_val, y_val, n_splits=N_FOLDS, seed=seed, return_folds=True)

    # the untrained twin: same seed, same construction order, same scale
    torch.manual_seed(seed)
    untrained = PointNet(embed_dim=model.embed_dim, proj_dim=model.proj_dim, normalize_size=True)
    untrained.input_scale.fill_(scale)
    untrained.eval()
    u_fit = untrained.embed(points[fit_idx]).astype(np.float64)
    u_val = untrained.embed(points[val_idx]).astype(np.float64)
    u_probe = _lr_pipeline(max_iter=2000).fit(u_fit, y_fit)
    u_holdout = float(u_probe.score(u_val, y_val))
    u_cv, _ = linear_probe_accuracy(u_val, y_val, n_splits=N_FOLDS, seed=seed, return_folds=True)

    worst_idx = _worst_indices(y_val, pred_val, proba_val)
    worst = [{"index": int(val_idx[k]), "true": int(y_val[k]), "pred": int(pred_val[k]),
              "p": float(proba_val[k, pred_val[k]]), "p_engine": proba_val[k].tolist(),
              "shape": preshape(points[val_idx[k]])} for k in worst_idx]

    # permutation invariance, measured: shuffle the points of one validation cloud
    rng = np.random.default_rng(seed)
    cloud = points[val_idx[0]]
    shuffled = cloud[rng.permutation(len(cloud))]
    perm_delta = float(np.abs(model.embed(cloud[None]) - model.embed(shuffled[None])).max())

    out = {
        "status": "ready",
        "accuracy": holdout,
        "accuracy_measure": "linear probe fitted on the fit embeddings, scored on the untouched 25 % hold-out",
        "accuracy_select": float(result["best_select_acc"]),
        "accuracy_select_measure": "the same probe scored on the 20 % select slice: the number that chose the epoch",
        "accuracy_cv_val": float(cv_val),
        "accuracy_sd": float(np.std(cv_folds)),
        "accuracy_sd_def": f"sd of fold_accuracies of the {N_FOLDS}-fold probe inside the hold-out, ddof=0; "
                           "not an uncertainty of `accuracy`",
        "fold_accuracies": [float(f) for f in cv_folds],
        "protocol": holdout_protocol(seed),
        "size_information": "removed",
        "cv": f"{N_FOLDS}-fold StratifiedKFold(shuffle, random_state={seed}) linear probe on the validation "
              f"embeddings (as the tutorial did)",
        "untrained": u_holdout,
        "untrained_select": float(result["untrained_select_acc"]),
        "untrained_cv_val": float(u_cv),
        "chance": _chance(y),
        "chance_val": _chance(y_val),
        "n": int(len(y)), "n_train": int(len(result["train_idx"])), "n_fit": int(len(fit_idx)),
        "n_select": int(len(select_idx)), "n_val": int(len(val_idx)),
        "split": f"train_test_split(test_size={VAL_FRACTION}, stratify=labels, random_state={seed}); the training "
                 f"part split again with test_size={SELECT_FRACTION} (stratified, random_state={seed}) into fit / "
                 f"select",
        "embed_dim": int(model.embed_dim), "proj_dim": int(model.proj_dim),
        "epochs": int(epochs), "best_epoch": int(result["best_epoch"]),
        "training": (f"SupConLoss(temperature {hp['temperature']}), Adam lr {hp['lr']}, weight decay "
                     f"{hp['weight_decay']}, batch {hp['batch_size']}, "
                     f"{'random rotation + jitter augmentation' if hp['augment'] else 'no augmentation'}, "
                     f"best epoch by the probe on the select slice"),
        "history_select_acc": [float(h["select_acc"]) for h in result["history"]],
        "threads": int(threads),
        "n_points": int(points.shape[1]),
        "input_scale": scale,
        "input_normalisation": "per_cloud_rms",
        "preprocessing": "centre each cloud, divide it by its own RMS radius (unit size per cloud: size removed), "
                         "then by input_scale (1 by construction)",
        "permutation_max_delta": perm_delta,
        "cm": _confusion(y_val, pred_val, n_classes),
        "cm_layout": "rows true, cols predicted, class order; validation split only",
        "worst": worst,
        "worst_note": "p / p_engine: the hold-out probe's probabilities (one fit, no cross-validation)",
        "probe": _probe_dict(probe, classes),
    }
    if onnx_path is not None:
        prepared, _ = preprocess_points(points[val_idx][:64], scale=scale, normalize_size=True)
        out["onnx"] = onnx_rel
        out["onnx_info"] = export_onnx(model, Path(onnx_path), prepared)
    else:
        out["onnx"] = None
    out["_timing"] = {"train": float(train_seconds), "total": float(time.perf_counter() - t0)}
    return out


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #


def run_dataset(ds: LabDataset, points: NDArray, labels: NDArray, *, seed: int = 0, skip_pointnet: bool = False,
                epochs: int | None = None, onnx_dir: Path | None = None, threads: int = 4, log=None) -> dict:
    """Every block of the JSON contract for one dataset, unrounded (see :mod:`shapekit.lab_export`).

    ``onnx_dir`` is where ``<id>.onnx`` goes (``None`` disables the export,
    as does ``ds.shapes_shown == False``).  Coordinates are still present in
    the returned dict when ``shapes_shown`` is false; the exporter strips
    them, so that every number is computed the same way for every dataset.
    Wall-clock times go under ``timing`` (never inside a method block).
    """
    t_start = time.perf_counter()
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels).astype(int).reshape(-1)
    if points.ndim != 3 or points.shape[-1] != 2 or len(points) != len(labels):
        raise ValueError(f"{ds.id}: expected points (K, N, 2) and labels (K,), got {points.shape}, {labels.shape}")
    classes = resolve_classes(ds, labels)
    n_classes = len(classes)
    prep = prepare_points(ds, points)
    say = log if log is not None else (lambda msg: None)
    timing: dict = {}

    doc = {
        "id": ds.id, "title": ds.title, "question": ds.question, "representation": ds.representation,
        "closed": bool(ds.closed), "n_points": prep["n_points"], "n": int(len(labels)),
        "status": "ready", "shapes_shown": bool(ds.shapes_shown),
        "source": {"name": ds.source.name, "citation": ds.source.citation, "url": ds.source.url,
                   "license": ds.source.license, "redistributed": bool(ds.source.redistributed)},
        "classes": [{"id": c, "name": classes[c], "n": int(np.sum(labels == c))} for c in range(n_classes)],
        "chance": _chance(labels),
        "seed": seed,
        "correspondence": prep["correspondence"],
        "preprocessing": dataset_notes(ds) + prep["notes"],
        "methods": {},
    }
    if ds.caveats:
        doc["caveats"] = list(ds.caveats)
    if prep["cyclic"] is not None:
        doc["cyclic_shift_alignment"] = {k: v for k, v in prep["cyclic"].items() if k != "shifts"}

    # --- classical ------------------------------------------------------- #
    display_extras = None
    if prep["landmarks"] is not None:
        t = time.perf_counter()
        res, extras = classical_method(prep["landmarks"], labels, classes, "landmarks_lr", seed=seed)
        res["engine"]["correspondence"] = prep["correspondence"]
        timing["landmarks_lr"] = time.perf_counter() - t
        doc["methods"]["landmarks_lr"] = res
        doc["engine"] = res["engine"]
        display_extras = extras
        say(f"  landmarks_lr  acc {res['accuracy']:.4f} ± {res['accuracy_sd']:.4f}  "
            f"({res['n_components']} components, {timing['landmarks_lr']:.1f} s)")
        if prep["spline"] is not None:
            t = time.perf_counter()
            res, _ = spline_method(prep["landmarks"], labels, classes, prep["spline"]["M"], prep["spline"]["closed"],
                                   seed=seed)
            res["engine"]["correspondence"] = prep["correspondence"]
            timing["spline_lr"] = time.perf_counter() - t
            doc["methods"]["spline_lr"] = res
            say(f"  spline_lr     acc {res['accuracy']:.4f} ± {res['accuracy_sd']:.4f}  "
                f"(M = {res['M']}, {res['n_components']} components, {timing['spline_lr']:.1f} s)")
        else:
            doc["methods"]["spline_lr"] = {"status": "not_applicable",
                                           "reason": "anatomical landmarks are not an ordered curve"}
    else:
        reason = "points are unordered"
        doc["methods"]["landmarks_lr"] = {"status": "not_applicable", "reason": reason}
        doc["methods"]["spline_lr"] = {"status": "not_applicable", "reason": reason}
        doc["engine"] = {"status": "not_applicable", "reason": reason + "; classify with pointnet_probe (ONNX)"}
        # display-only geometry on the angle-ordered points (no classifier involved)
        display_extras = shape_geometry(prep["display"])

    # --- size only -------------------------------------------------------- #
    t = time.perf_counter()
    size_input = prep["landmarks"] if prep["landmarks"] is not None else prep["cloud"]
    res = size_method(size_input, labels, classes, seed=seed)
    timing["size_lr"] = time.perf_counter() - t
    doc["methods"]["size_lr"] = res
    say(f"  size_lr       acc {res['accuracy']:.4f} ± {res['accuracy_sd']:.4f}  (size only, {timing['size_lr']:.1f} s)")

    # --- roundness diagnostic (display only, not a method) --------------- #
    if ds.id in ROUNDNESS_BASELINE_DATASETS:
        t = time.perf_counter()
        res = roundness_baseline(prep["cloud"], labels, classes, seed=seed)
        timing["roundness_baseline"] = time.perf_counter() - t
        doc["roundness_baseline"] = res
        say(f"  roundness     acc {res['accuracy']:.4f} ± {res['accuracy_sd']:.4f}  (diagnostic: circularity + "
            f"axis ratio, {timing['roundness_baseline']:.1f} s)")

    # --- display blocks -------------------------------------------------- #
    note = None
    if prep["landmarks"] is None:
        note = "points ordered by angle around the centroid for display; not used for classification"
    doc["pca"] = pca_block(display_extras, labels, seed=seed, note=note)
    examples, example_indices = examples_block(display_extras["aligned"], labels, n_classes, seed=seed)
    doc["examples"] = examples
    doc["example_indices"] = example_indices
    doc["neighbours"] = neighbours_block(display_extras["aligned"], labels, seed=seed)
    if note:
        doc["neighbours"]["note"] = note

    # --- PointNet -------------------------------------------------------- #
    n = len(labels)
    if skip_pointnet:
        doc["methods"]["pointnet_probe"] = {"status": "skipped", "reason": "--skip-pointnet"}
    elif n < MIN_POINTNET_N:
        doc["methods"]["pointnet_probe"] = {"status": "skipped", "reason": f"n < {MIN_POINTNET_N}", "n": int(n)}
        say(f"  pointnet_probe skipped (n = {n} < {MIN_POINTNET_N})")
    else:
        n_epochs = int(epochs) if epochs is not None else default_epochs(ds.id, n)
        onnx_path = onnx_rel = None
        if onnx_dir is not None and ds.shapes_shown:
            onnx_path = Path(onnx_dir) / f"{ds.id}.onnx"
            onnx_rel = f"models/{ds.id}.onnx"
        say(f"  pointnet_probe: {n_epochs} epochs on {n} clouds of {points.shape[1]} points")
        res = pointnet_method(prep["cloud"], labels, classes, n_epochs, seed=seed, onnx_path=onnx_path,
                              onnx_rel=onnx_rel, threads=threads, log=log)
        if ds.representation == "landmarks":
            res["note"] = "landmarks fed as an unordered point cloud (the fair comparison)"
        pn_timing = res.pop("_timing")
        timing["pointnet_train"] = pn_timing["train"]
        timing["pointnet_probe"] = pn_timing["total"]
        doc["methods"]["pointnet_probe"] = res
        say(f"  pointnet_probe acc {res['accuracy']:.4f} (hold-out), select {res['accuracy_select']:.4f}, "
            f"{res['accuracy_cv_val']:.4f} (5-fold on val), untrained {res['untrained']:.4f}, "
            f"{pn_timing['total']:.0f} s")
    timing["total"] = time.perf_counter() - t_start
    doc["timing"] = {"seconds": timing, "threads": int(threads),
                     "note": "wall-clock seconds; not deterministic and not part of the results"}
    return doc
