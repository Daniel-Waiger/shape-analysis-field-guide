"""Learned shape embeddings: PointNet + supervised contrastive loss (chapter 08).

This module turns a cell outline, stored as an unordered cloud of points, into a
fixed-length vector (an "embedding") with a small PointNet, trains it with the
supervised contrastive loss, and measures the result with a linear probe.

Axis convention (as everywhere in shapekit): points are rows, (x, y) are columns.
One cloud is ``(N, 2)``; a dataset is ``(K, N, 2)``. Inside this module the only
place that departs from it is the PyTorch ``Conv1d`` layer, which wants
``(batch, channels, N)``; :meth:`PointNet.forward` transposes on the way in and
never hands that layout back to you.

PyTorch is optional. ``import shapekit`` and the numpy helpers in this file
(:func:`center_point_clouds`, :func:`preprocess_points`,
:func:`linear_probe_accuracy`) work without it. :class:`PointNet`,
:class:`SupConLoss`, :class:`MEFPointDataset` and :func:`train_pointnet` raise a
clear ``ImportError`` when torch is missing.

Chapter 09 ("Field notes") lists the mistakes of the original tutorial that this
file is written to avoid:

* centering over the points axis, not the (x, y) axis (see
  :func:`center_point_clouds` and :func:`tutorial_center_bug`);
* ``model.train()`` at the start of every training epoch and ``model.eval()`` +
  ``torch.no_grad()`` only for validation (see :func:`train_pointnet`);
* the "best" weights are a ``copy.deepcopy`` of the ``state_dict``, not a
  reference to the live tensors;
* a seeded, stratified train/validation split.

Reference implementations read while writing this (not copied):
``fxia22/pointnet.pytorch`` (MIT) for the PointNet layout and
``HobbitLong/SupContrast`` (BSD-2) for the loss and its no-positive guard.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

try:  # torch is optional for the package as a whole
    import torch
    import torch.utils.data
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

__all__ = [
    "TORCH_AVAILABLE",
    "center_point_clouds",
    "tutorial_center_bug",
    "preprocess_points",
    "InputTransform2d",
    "PointNet",
    "SupConLoss",
    "MEFPointDataset",
    "linear_probe_accuracy",
    "train_pointnet",
]


def _require_torch(what: str) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"{what} needs PyTorch, which is not installed. The rest of shapekit works "
            "without it; to use the learned embeddings install the CPU build, e.g. "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        )


# A class can only subclass ``nn.Module`` when torch exists. Without torch the
# classes below still exist (so docs and ``from shapekit.embed import PointNet``
# work) but their constructors raise the ImportError above.
_Module = nn.Module if TORCH_AVAILABLE else object
_Dataset = torch.utils.data.Dataset if TORCH_AVAILABLE else object


# --------------------------------------------------------------------------- #
# numpy helpers (no torch needed)
# --------------------------------------------------------------------------- #


def center_point_clouds(points):
    """Subtract each cloud's centroid: ``X - mean over the points axis``.

    Formula, for one cloud ``X`` of shape ``(N, 2)``::

        X_c[k] = X[k] - (1/N) * sum_j X[j]        for every point k

    * ``X[k]`` is the k-th point, a pair ``(x, y)``.
    * The mean runs over the *points* axis (``axis=-2``), so the result is the
      same cloud shifted so that its centre of mass sits at ``(0, 0)``.

    Accepts ``(N, 2)`` or ``(K, N, 2)`` and returns ``float32`` of the same
    shape. It is the same operation as ``shapekit.preshape.center``; it is
    repeated here so that this module stays importable on its own. Chapter 02
    explains centering; chapter 09 explains why the axis matters.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim not in (2, 3) or points.shape[-1] != 2:
        raise ValueError(f"expected (N, 2) or (K, N, 2), got {points.shape}")
    return points - points.mean(axis=-2, keepdims=True)


def tutorial_center_bug(points):
    """Reproduce the *wrong-axis* centering of the original tutorial (chapter 09).

    The tutorial stored a cell as ``(2, N)`` and subtracted the mean over
    ``axis=-2``, i.e. over the pair ``(x, y)`` of each point instead of over the
    N points. For a point ``(x, y)`` that gives::

        ((x - y) / 2, (y - x) / 2)

    so every point lands on the line ``y = -x`` and carries a single number,
    ``(x - y) / 2``. Half of each point's position is thrown away and the cloud
    is not centred at all. This function exists only to show the effect
    (figure ``09_center_axis_bug.png``); never use it for real work.

    Accepts ``(N, 2)`` or ``(K, N, 2)``; returns the damaged points in the usual
    points-as-rows layout.
    """
    points = np.asarray(points, dtype=np.float32)
    transposed = np.swapaxes(points, -1, -2)  # (..., 2, N): what the tutorial fed in
    damaged = transposed - transposed.mean(axis=-2, keepdims=True)  # bug: axis of length 2
    return np.swapaxes(damaged, -1, -2)


def preprocess_points(points, scale=None):
    """Center each cloud and divide all of them by one shared scale.

    Returns ``(prepared, scale)`` where ``prepared = center(points) / scale``.

    * Centering removes the position of the cell in the image (chapter 02).
    * ``scale`` is a single number for the whole dataset, so cells keep their
      *relative* sizes (a large cell stays larger than a small one). When
      ``scale`` is ``None`` it is set to the root-mean-square distance of
      points from their centroid over the given clouds, which puts the
      coordinates in a comfortable range of about ±3 for the network.

    Pass the ``scale`` returned on the training cells when preparing other
    cells, so that all cells are measured with the same ruler.
    """
    centered = center_point_clouds(points)
    if scale is None:
        scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=-1))))
        if not scale > 0:
            scale = 1.0
    return centered / np.float32(scale), float(scale)


def linear_probe_accuracy(features, labels, n_splits=5, seed=0, return_folds=False):
    """Linear-probe accuracy: logistic regression with stratified, shuffled K-fold.

    A *linear probe* asks "can a straight line separate the classes in this
    embedding?". It fits

        p(class 1 | f) = sigmoid(w · f + b)

    * ``f`` is one embedding vector (one row of ``features``);
    * ``w`` and ``b`` are the only learned numbers; nothing non-linear is added,
      so a high accuracy means the embedding itself organised the classes well.

    The data is cut into ``n_splits`` folds that keep the class ratio
    (``StratifiedKFold``), shuffled with ``seed``; each fold serves once as the
    test set and the mean test accuracy is returned. Features are standardised
    inside each fold before the regression (this only helps the solver converge;
    it does not change what a linear probe can do).

    Returns the mean accuracy as a float, or ``(mean, per_fold_scores)`` when
    ``return_folds`` is true. Chapter 08.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, features, labels, cv=cv)
    mean = float(scores.mean())
    return (mean, scores) if return_folds else mean


def _holdout_probe_accuracy(train_features, train_labels, test_features, test_labels):
    """Fit the probe on one set of embeddings and score it on another."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(np.asarray(train_features, dtype=np.float64), np.asarray(train_labels).reshape(-1))
    return float(clf.score(np.asarray(test_features, dtype=np.float64), np.asarray(test_labels).reshape(-1)))


# --------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------- #


class InputTransform2d(_Module):
    """PointNet's "T-Net" for 2-D points: predicts one 2×2 matrix per cloud.

    A mini PointNet looks at the whole cloud and proposes a matrix ``T`` that is
    applied to every point before the main network. The last layer starts at
    zero, so ``T`` starts as the identity ("do nothing") and only moves away if
    training finds it useful. Off by default in :class:`PointNet`; it is here
    so the reader can see what the option does. Chapter 08.
    """

    def __init__(self):
        _require_torch("InputTransform2d")
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 4),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        """``x``: ``(B, 2, N)`` channels-first. Returns ``(B, 2, 2)``."""
        pooled = self.point_mlp(x).max(dim=2).values
        delta = self.fc(pooled).view(-1, 2, 2)
        return delta + torch.eye(2, device=x.device, dtype=x.dtype)


class PointNet(_Module):
    """A small PointNet that maps a 2-D point cloud to an embedding (chapter 08).

    Pipeline for one cloud ``X`` of shape ``(N, 2)``::

        h_k = MLP_point(X[k])            for each point k   -> (N, 1024)
        g   = max over k of h_k          (feature-wise)     -> (1024,)
        e   = MLP_embed(g)               the embedding      -> (embed_dim,)
        z   = normalize(Head(e))         the projection     -> (proj_dim,), length 1

    * ``MLP_point`` is one small network, 2 → 64 → 128 → 1024 (with BatchNorm
      and ReLU), applied to *every point separately with the same weights*.
      In code it is a ``Conv1d`` with kernel size 1, which is exactly that.
    * ``max over k`` keeps, for each of the 1024 features, the largest value
      any point produced. A maximum does not care about order, so shuffling the
      points cannot change ``g``: this is where permutation invariance comes
      from.
    * ``MLP_embed`` is 1024 → 512 → ``embed_dim``; ``e`` is the vector you keep
      after training (for PCA, k-means, the linear probe).
    * ``Head`` is ``embed_dim`` → ``embed_dim`` → ``proj_dim`` and
      ``normalize`` rescales to unit length, so dot products of two ``z`` are
      cosine similarities in ``[-1, 1]``. The loss is computed on ``z``, which
      is thrown away afterwards.

    ``forward(points)`` takes ``(B, N, 2)`` and returns ``(projection, embedding)``.
    ``input_transform=True`` adds the optional 2×2 :class:`InputTransform2d`
    before the first layer. Defaults (``embed_dim=256, proj_dim=64``) give about
    0.9 M weights, comfortable on a laptop CPU.
    """

    def __init__(self, embed_dim=256, proj_dim=64, input_transform=False):
        _require_torch("PointNet")
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.proj_dim = int(proj_dim)
        self.transform = InputTransform2d() if input_transform else None
        self.point_mlp = nn.Sequential(
            nn.Conv1d(2, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.embed_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, self.embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, self.proj_dim),
        )
        # The scale used by preprocess_points at training time, stored with the
        # weights so that embed() prepares new cells with the same ruler.
        self.register_buffer("input_scale", torch.tensor(1.0))

    def per_point_features(self, points):
        """The ``(B, 1024, N)`` features *before* max-pooling (for figures)."""
        x = points.transpose(1, 2)
        if self.transform is not None:
            x = torch.bmm(self.transform(x), x)
        return self.point_mlp(x)

    def forward(self, points):
        """``points``: ``(B, N, 2)``. Returns ``(projection (B, proj_dim), embedding (B, embed_dim))``."""
        if points.dim() != 3 or points.shape[-1] != 2:
            raise ValueError(f"expected (B, N, 2), got {tuple(points.shape)}")
        per_point = self.per_point_features(points)           # (B, 1024, N)
        pooled = per_point.max(dim=2).values                   # (B, 1024)
        embedding = self.embed_mlp(pooled)                     # (B, embed_dim)
        projection = F.normalize(self.head(embedding), dim=1)  # (B, proj_dim), unit length
        return projection, embedding

    def embed(self, points, batch_size=256, preprocess=True):
        """Numpy in, numpy out: ``(K, N, 2)`` clouds → ``(K, embed_dim)`` embeddings.

        Runs in eval mode with no gradients and restores the previous mode.
        With ``preprocess=True`` (default) the clouds are centred and divided by
        the stored ``input_scale`` first, exactly as during training.
        """
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2:
            points = points[None]
        if preprocess:
            points, _ = preprocess_points(points, scale=float(self.input_scale))
        was_training = self.training
        self.eval()
        device = self.input_scale.device
        out = []
        with torch.no_grad():
            for start in range(0, len(points), batch_size):
                batch = torch.as_tensor(points[start:start + batch_size], device=device)
                _, emb = self(batch)
                out.append(emb.cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.embed_dim), np.float32)


# --------------------------------------------------------------------------- #
# the loss
# --------------------------------------------------------------------------- #


class SupConLoss(_Module):
    """Supervised contrastive loss, single view (Khosla et al. 2020). Chapter 08.

    For a batch of ``B`` unit-length vectors ``z_i`` with labels ``y_i``, call
    ``P(i)`` the *positives* of anchor ``i``: every other item in the batch with
    the same label. The loss is

        L = (1/B) · Σ_i  L_i,
        L_i = -(1/|P(i)|) · Σ_{p in P(i)}  log [ exp(z_i·z_p / τ) / Σ_{a ≠ i} exp(z_i·z_a / τ) ]

    Reading it from the inside out:

    * ``z_i·z_p`` is the cosine similarity of two cells, between −1 and 1
      (the vectors have length 1, see :class:`PointNet`).
    * ``/ τ`` ("temperature", default 0.07) stretches the similarities by about
      14×. Small τ makes the softmax picky: only the very closest neighbours
      get any probability. Large τ makes it indifferent.
    * ``exp(·) / Σ_a exp(·)`` is a **softmax** over all the *other* items in the
      batch: "if anchor i had to pick one neighbour, what is the probability it
      picks p?" The probabilities over all a ≠ i sum to 1.
    * ``-log`` turns a probability into a penalty: 0 if the probability is 1,
      large if it is near 0. So anchor i is punished whenever a same-class cell
      p gets a low probability, i.e. whenever some other-class cell looks more
      similar than p does.
    * The ``mask`` in the code is the ``B×B`` table "same label, and not the
      anchor itself". Averaging over the positives (``1/|P(i)|``) and then over
      the anchors (``1/B``) gives one number.

    Numerical details, as in the reference implementation: each row's maximum
    is subtracted before ``exp`` (changes nothing mathematically, prevents
    overflow), and an anchor with *no* positive in the batch contributes 0
    instead of ``0/0 = NaN``.

    Benchmark: if all ``B`` vectors are identical, every neighbour is equally
    likely, the probability of each positive is ``1/(B−1)`` and the loss is
    ``ln(B−1)`` (about 5.29 for B = 200). A trained model gets well below that.
    """

    def __init__(self, temperature=0.07):
        _require_torch("SupConLoss")
        super().__init__()
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, features, labels):
        """``features``: ``(B, D)`` unit-length rows; ``labels``: ``(B,)``. Returns a scalar."""
        if features.dim() != 2:
            raise ValueError(f"features must be (B, D), got {tuple(features.shape)}")
        batch = features.shape[0]
        if batch < 2:
            raise ValueError("need at least two items in the batch")
        labels = torch.as_tensor(labels, device=features.device).reshape(-1)
        if labels.shape[0] != batch:
            raise ValueError("number of labels does not match number of features")

        # similarities, sharpened by the temperature
        logits = features @ features.T / self.temperature                   # (B, B)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()    # stability only

        not_self = ~torch.eye(batch, dtype=torch.bool, device=features.device)
        # log of the softmax over all a ≠ i:  logit - log(sum_a exp(logit_a))
        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        positives = (labels[:, None] == labels[None, :]) & not_self          # the mask
        n_pos = positives.sum(dim=1)
        mean_log_prob_pos = (log_prob * positives).sum(dim=1) / n_pos.clamp(min=1)  # 0 when no positive
        return -mean_log_prob_pos.mean()


# --------------------------------------------------------------------------- #
# data and training
# --------------------------------------------------------------------------- #


class MEFPointDataset(_Dataset):
    """A tiny ``torch.utils.data.Dataset`` of (cloud, label) pairs.

    ``points`` must already be preprocessed (see :func:`preprocess_points`).
    With ``augment=True`` every access rotates the cloud by a random angle and
    adds Gaussian jitter (``jitter`` in the preprocessed units); the cloud is
    centred, so a rotation about the origin is a rotation about the cell.
    """

    def __init__(self, points, labels, augment=False, jitter=0.01, seed=0):
        _require_torch("MEFPointDataset")
        self.points = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)
        if len(self.points) != len(self.labels):
            raise ValueError("points and labels have different lengths")
        self.augment = bool(augment)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        cloud = self.points[index]
        if self.augment:
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            cloud = cloud @ rot.T
            if self.jitter > 0:
                cloud = cloud + self.rng.normal(0.0, self.jitter, size=cloud.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(cloud, dtype=np.float32)), int(self.labels[index])


def train_pointnet(points, labels, epochs=30, batch_size=64, lr=1e-3, seed=0, device="cpu", log=None,
                   val_fraction=0.25, temperature=0.07, embed_dim=256, proj_dim=64, weight_decay=1e-4,
                   input_transform=False, augment=False):
    """Train a :class:`PointNet` with :class:`SupConLoss` and keep the best weights.

    ``points`` is ``(K, N, 2)`` raw coordinates (points are rows), ``labels`` is
    ``(K,)`` integers. Steps, each one deliberately the opposite of a bug in the
    original tutorial (chapter 09):

    1. seed numpy and torch;
    2. split into train / validation with ``sklearn.train_test_split``,
       stratified by class and seeded (the split is the same on every run);
    3. centre every cloud over the **points** axis and divide by one scale
       computed on the training cells (:func:`preprocess_points`);
    4. for each epoch call ``model.train()`` first, then run the batches;
    5. validate with ``model.eval()`` and ``torch.no_grad()``: validation loss,
       plus a linear probe fitted on the training embeddings and scored on the
       validation embeddings ("val_acc");
    6. the untrained model is evaluated the same way at epoch 0 and counts as a
       candidate; whenever ``val_acc`` improves the ``state_dict`` is
       ``copy.deepcopy``-ed.

    ``log`` is any callable taking a string (``print`` is fine); ``None`` is silent.

    Returns a dict: ``model`` (best weights loaded, in eval mode, ``input_scale``
    set), ``scale``, ``train_idx``, ``val_idx``, ``history`` (one dict per epoch
    including epoch 0: ``epoch, train_loss, val_loss, val_acc, seconds``),
    ``best_epoch``, ``best_val_acc``, ``untrained_val_acc``, ``seconds``.
    """
    _require_torch("train_pointnet")
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    t_start = time.perf_counter()
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"points must be (K, N, 2), got {points.shape}")
    if len(points) != len(labels):
        raise ValueError("points and labels have different lengths")

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    idx = np.arange(len(points))
    train_idx, val_idx = train_test_split(idx, test_size=val_fraction, stratify=labels, random_state=seed)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    train_points, scale = preprocess_points(points[train_idx])      # scale from training cells only
    val_points, _ = preprocess_points(points[val_idx], scale=scale)
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    model = PointNet(embed_dim=embed_dim, proj_dim=proj_dim, input_transform=input_transform).to(device)
    model.input_scale.fill_(scale)
    criterion = SupConLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = MEFPointDataset(train_points, train_labels, augment=augment, seed=seed)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size,
                        generator=loader_gen, num_workers=0)
    train_t = torch.as_tensor(train_points, device=device)
    val_t = torch.as_tensor(val_points, device=device)
    val_labels_t = torch.as_tensor(val_labels, device=device)

    def evaluate():
        # eval mode: BatchNorm uses its running statistics; no_grad: no autograd bookkeeping
        model.eval()
        with torch.no_grad():
            _, train_emb = model(train_t)
            val_proj, val_emb = model(val_t)
            # the loss depends on the batch size (ln(B-1) when nothing is learned), so
            # score the validation cells in chunks of the training batch size
            chunks = [(criterion(val_proj[i:i + batch_size], val_labels_t[i:i + batch_size]).item())
                      for i in range(0, len(val_idx), batch_size) if i + 2 <= len(val_idx)]
            val_loss = float(np.mean(chunks))
        acc = _holdout_probe_accuracy(train_emb.cpu().numpy(), train_labels, val_emb.cpu().numpy(), val_labels)
        return val_loss, acc

    history = []
    val_loss, val_acc = evaluate()
    history.append({"epoch": 0, "train_loss": float("nan"), "val_loss": val_loss, "val_acc": val_acc,
                    "seconds": time.perf_counter() - t_start})
    best_acc, best_epoch = val_acc, 0
    best_state = copy.deepcopy(model.state_dict())  # a snapshot, not a reference
    if log is not None:
        log(f"epoch   0  untrained   val_loss {val_loss:.3f}  val_acc {val_acc:.3f}")

    for epoch in range(1, epochs + 1):
        model.train()  # back to training mode every epoch (BatchNorm learns batch statistics)
        running, n_batches = 0.0, 0
        for batch_points, batch_labels in loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            projection, _ = model(batch_points)
            loss = criterion(projection, batch_labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        val_loss, val_acc = evaluate()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                        "seconds": time.perf_counter() - t_start})
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if log is not None:
            log(f"epoch {epoch:3d}  train_loss {train_loss:.3f}  val_loss {val_loss:.3f}  "
                f"val_acc {val_acc:.3f}  ({history[-1]['seconds']:.0f} s)")

    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "scale": scale,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "untrained_val_acc": history[0]["val_acc"],
        "seconds": time.perf_counter() - t_start,
    }
