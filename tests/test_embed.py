"""Tests for shapekit.embed (chapter 08). Torch-dependent tests skip cleanly without torch."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapekit.embed import (  # noqa: E402
    center_point_clouds,
    linear_probe_accuracy,
    preprocess_points,
    tutorial_center_bug,
)


# ----------------------------------------------------------------------------- #
# numpy-only parts (always run)
# ----------------------------------------------------------------------------- #


def test_center_point_clouds_uses_points_axis():
    rng = np.random.default_rng(0)
    clouds = rng.normal(size=(5, 30, 2)) + np.array([100.0, 300.0])
    centered = center_point_clouds(clouds)
    assert centered.shape == clouds.shape
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-4)
    # the shape itself is untouched: pairwise differences are preserved
    assert np.allclose(centered[0, 1] - centered[0, 0], clouds[0, 1] - clouds[0, 0], atol=1e-4)


def test_tutorial_center_bug_collapses_onto_diagonal():
    pts = np.array([[100.0, 300.0], [120.0, 310.0], [90.0, 250.0]])
    damaged = tutorial_center_bug(pts)
    expected = np.array([[-100.0, 100.0], [-95.0, 95.0], [-80.0, 80.0]])
    assert np.allclose(damaged, expected)
    assert np.allclose(damaged[:, 0], -damaged[:, 1])  # every point on y = -x


def test_preprocess_points_shared_scale():
    rng = np.random.default_rng(1)
    clouds = rng.normal(size=(8, 20, 2)) * 50 + 200
    prepared, scale = preprocess_points(clouds)
    assert scale > 0
    assert np.isclose(np.sqrt(np.mean(np.sum(prepared**2, axis=-1))), 1.0, atol=1e-5)
    again, scale2 = preprocess_points(clouds, scale=scale)
    assert scale2 == scale and np.allclose(again, prepared)


def test_linear_probe_accuracy_separable():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 5)) + np.array([4, 0, 0, 0, 0])
    b = rng.normal(size=(60, 5)) - np.array([4, 0, 0, 0, 0])
    features = np.concatenate([a, b])
    labels = np.array([0] * 60 + [1] * 60)
    acc, folds = linear_probe_accuracy(features, labels, n_splits=5, seed=0, return_folds=True)
    assert len(folds) == 5
    assert acc >= 0.98
    # seeded: same answer twice
    assert linear_probe_accuracy(features, labels, n_splits=5, seed=0) == acc


def test_import_without_torch_is_clean():
    """``import shapekit.embed`` must work when torch is absent; using the classes must not."""
    code = r"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError('torch blocked for this test')
sys.meta_path.insert(0, Block())
sys.path.insert(0, %r)
import shapekit.embed as e
assert e.TORCH_AVAILABLE is False
for make in (e.PointNet, e.SupConLoss, lambda: e.MEFPointDataset([[[0.0, 0.0]]], [0])):
    try:
        make()
    except ImportError as ex:
        assert 'PyTorch' in str(ex)
    else:
        raise SystemExit('expected ImportError')
try:
    e.train_pointnet([[[0.0, 0.0]]], [0])
except ImportError:
    pass
else:
    raise SystemExit('expected ImportError from train_pointnet')
import numpy as np
assert e.center_point_clouds(np.zeros((3, 2))).shape == (3, 2)
print('ok')
""" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


# ----------------------------------------------------------------------------- #
# torch parts (skipped without torch)
# ----------------------------------------------------------------------------- #

torch = pytest.importorskip("torch")

from shapekit.embed import MEFPointDataset, PointNet, SupConLoss, train_pointnet  # noqa: E402


def _toy_clouds(n_per_class=24, n_points=40, seed=0):
    """Two classes of 2-D clouds: circles (0) and squashed ellipses (1), random order."""
    rng = np.random.default_rng(seed)
    clouds, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            t = rng.uniform(0, 2 * np.pi, size=n_points)
            r = rng.uniform(0.9, 1.1)
            x = r * np.cos(t)
            y = r * np.sin(t) * (1.0 if label == 0 else 0.4)
            cloud = np.stack([x, y], axis=1) + rng.normal(scale=0.02, size=(n_points, 2))
            clouds.append(cloud[rng.permutation(n_points)])
            labels.append(label)
    return np.asarray(clouds, dtype=np.float32), np.asarray(labels)


def test_pointnet_shapes_and_unit_projection():
    torch.manual_seed(0)
    model = PointNet(embed_dim=32, proj_dim=8).eval()
    x = torch.randn(4, 25, 2)
    proj, emb = model(x)
    assert proj.shape == (4, 8) and emb.shape == (4, 32)
    assert torch.allclose(proj.norm(dim=1), torch.ones(4), atol=1e-5)


def test_pointnet_permutation_invariance():
    torch.manual_seed(0)
    model = PointNet(embed_dim=32, proj_dim=8).eval()
    x = torch.randn(3, 50, 2)
    perm = torch.randperm(50)
    with torch.no_grad():
        proj_a, emb_a = model(x)
        proj_b, emb_b = model(x[:, perm])
    assert torch.allclose(emb_a, emb_b, atol=1e-5)
    assert torch.allclose(proj_a, proj_b, atol=1e-5)
    # the numpy path agrees too
    e1 = model.embed(x.numpy(), preprocess=False)
    e2 = model.embed(x[:, perm].numpy(), preprocess=False)
    assert np.allclose(e1, e2, atol=1e-5)


def test_pointnet_with_input_transform_runs_and_starts_at_identity():
    torch.manual_seed(0)
    model = PointNet(embed_dim=16, proj_dim=8, input_transform=True).eval()
    x = torch.randn(2, 20, 2)
    with torch.no_grad():
        t = model.transform(x.transpose(1, 2))
        proj, emb = model(x)
    assert torch.allclose(t, torch.eye(2).expand(2, 2, 2), atol=1e-6)
    assert proj.shape == (2, 8) and emb.shape == (2, 16)


def test_supcon_identical_features_gives_log_b_minus_1():
    torch.manual_seed(0)
    for batch in (8, 20):
        one = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
        features = one.repeat(batch, 1)
        labels = torch.tensor([0] * (batch // 2) + [1] * (batch // 2))
        loss = SupConLoss(temperature=0.07)(features, labels).item()
        assert math.isclose(loss, math.log(batch - 1), rel_tol=1e-5)


def test_supcon_no_positive_is_finite():
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    loss = SupConLoss()(features, torch.tensor([0, 1, 1, 2]))
    assert torch.isfinite(loss)


def test_supcon_prefers_clustered_features():
    """Same-class vectors close together should score lower than a shuffled labelling."""
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.tensor([1.0, 0.0]) + 0.05 * torch.randn(10, 2), dim=1)
    b = torch.nn.functional.normalize(torch.tensor([-1.0, 0.0]) + 0.05 * torch.randn(10, 2), dim=1)
    features = torch.cat([a, b])
    good = torch.tensor([0] * 10 + [1] * 10)
    bad = good[torch.randperm(20)]
    crit = SupConLoss()
    assert crit(features, good) < crit(features, bad)


def test_loss_decreases_on_toy_data():
    torch.manual_seed(0)
    clouds, labels = _toy_clouds()
    x = torch.from_numpy(clouds)
    y = torch.from_numpy(labels)
    model = PointNet(embed_dim=32, proj_dim=16)
    crit = SupConLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(12):
        opt.zero_grad()
        proj, _ = model(x)
        loss = crit(proj, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_dataset_augmentation_keeps_shape():
    clouds, labels = _toy_clouds(n_per_class=3)
    ds = MEFPointDataset(clouds, labels, augment=True, seed=0)
    cloud, label = ds[0]
    assert cloud.shape == (40, 2) and label == 0
    assert len(ds) == 6


def test_train_pointnet_end_to_end():
    clouds, labels = _toy_clouds(n_per_class=20)
    clouds = clouds * 50 + np.array([300.0, 200.0], dtype=np.float32)  # pixel-like coordinates
    messages = []
    result = train_pointnet(clouds, labels, epochs=3, batch_size=16, lr=1e-3, seed=0, log=messages.append,
                            embed_dim=16, proj_dim=8)
    model = result["model"]
    assert model.training is False
    assert len(result["history"]) == 4 and result["history"][0]["epoch"] == 0
    assert len(messages) == 4
    assert 0.0 <= result["best_val_acc"] <= 1.0
    assert set(result["train_idx"]).isdisjoint(result["val_idx"])
    assert len(result["train_idx"]) + len(result["val_idx"]) == len(clouds)
    # the stored scale lets embed() prepare raw cells itself
    emb = model.embed(clouds[:5])
    assert emb.shape == (5, 16) and np.isfinite(emb).all()
    # seeded: same split and same numbers on a second run
    again = train_pointnet(clouds, labels, epochs=3, batch_size=16, lr=1e-3, seed=0, embed_dim=16, proj_dim=8)
    assert np.array_equal(again["val_idx"], result["val_idx"])
    assert math.isclose(again["history"][-1]["train_loss"], result["history"][-1]["train_loss"], rel_tol=1e-4)
