"""The Phenotype Lab registry: shapes, labels and class counts of every dataset.

Datasets fetched by ``scripts/fetch_datasets.py`` (brain, gorilla, leaves) are
skipped cleanly when their npz is missing; the fetch-locally dataset (rbc) must
raise ``DatasetNotFetched`` when absent.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from shapekit.datasets import resolve_data_dir
from shapekit.lab_datasets import DATASETS, DatasetNotFetched, LabDataset, get, leaves_classes

ROOT = Path(__file__).resolve().parents[1]


def _load_or_skip(dataset_id: str):
    try:
        return get(dataset_id).load()
    except DatasetNotFetched as e:
        pytest.skip(f"{dataset_id} not fetched: {e}")


def _check_basic(ds: LabDataset, points: np.ndarray, labels: np.ndarray, n: int, n_points: int):
    assert points.shape == (n, n_points, 2) and points.dtype == np.float64
    assert np.isfinite(points).all()
    assert labels.shape == (n,) and labels.dtype.kind == "i"
    assert set(np.unique(labels)) == set(range(len(ds.classes)))


# ------------------------------------------------------------------ registry

def test_registry_ids_and_api():
    assert list(DATASETS) == ["celegans", "mef", "brain", "gorilla", "leaves", "rbc"]
    for ds in DATASETS.values():
        assert isinstance(ds, LabDataset)
        assert ds.representation in {"landmarks", "outline", "pointcloud"}
        assert ds.title and ds.question and ds.classes
        assert ds.source.name and ds.source.citation and ds.source.url and ds.source.license
        assert callable(ds.loader) and ds.load is not None
    with pytest.raises(KeyError):
        get("no-such-dataset")


def test_redistribution_flags_match_the_spec():
    assert all(DATASETS[k].source.redistributed for k in ("celegans", "mef", "brain", "gorilla", "leaves"))
    assert DATASETS["rbc"].source.redistributed is False
    assert DATASETS["rbc"].shapes_shown is False
    assert "CC BY-NC-ND" in DATASETS["rbc"].source.license
    assert "GPL-2" in DATASETS["brain"].source.license and "GPL-2" in DATASETS["gorilla"].source.license
    assert DATASETS["leaves"].source.license == "CC BY 4.0"
    for k in ("brain", "gorilla", "leaves", "rbc"):
        assert DATASETS[k].fetch_command == f"python scripts/fetch_datasets.py --{k}"


def test_licence_notices_exist_next_to_the_gpl_datasets():
    for name in ("brain_landmarks", "gorilla_landmarks"):
        npz = ROOT / "data" / f"{name}.npz"
        if not npz.exists():
            pytest.skip(f"{npz} not fetched")
        notice = (ROOT / "data" / f"{name}.LICENSE").read_text(encoding="utf-8")
        assert "GPL-2" in notice and "shapes" in notice and "LICENSES/GPL-2.txt" in notice
        gpl = (ROOT / "LICENSES" / "GPL-2.txt").read_text(encoding="utf-8")
        assert "GNU GENERAL PUBLIC LICENSE" in gpl[:200] and "Version 2, June 1991" in gpl[:200]


# ------------------------------------------------------------------ datasets in the repo

def test_celegans():
    ds = get("celegans")
    points, labels = ds.load()
    _check_basic(ds, points, labels, 1377, 12)
    assert ds.classes == ("dead", "live") and ds.representation == "landmarks"
    assert Counter(labels.tolist()) == {0: 579, 1: 798}


def test_mef():
    ds = get("mef")
    points, labels = ds.load()
    _check_basic(ds, points, labels, 1200, 120)
    assert ds.classes == ("wild-type", "LMNA−/−") and ds.representation == "pointcloud"
    assert Counter(labels.tolist()) == {0: 600, 1: 600}


def test_brain():
    ds = get("brain")
    points, labels = _load_or_skip("brain")
    _check_basic(ds, points, labels, 28, 13)
    assert ds.classes == ("control", "schizophrenia") and ds.representation == "landmarks"
    # the package's order: first 14 controls, last 14 patients
    assert labels.tolist() == [0] * 14 + [1] * 14


def test_gorilla():
    ds = get("gorilla")
    points, labels = _load_or_skip("gorilla")
    _check_basic(ds, points, labels, 59, 8)
    assert ds.classes == ("female", "male") and ds.representation == "landmarks"
    assert labels.tolist() == [0] * 30 + [1] * 29


def test_leaves():
    ds = get("leaves")
    points, labels = _load_or_skip("leaves")
    _check_basic(ds, points, labels, 440, 150)
    assert ds.representation == "outline" and ds.closed is True
    assert len(ds.classes) == 15 and leaves_classes() == ds.classes
    assert ds.classes[9] == "Sorbus Aucuparia" and ds.classes[14] == "Fagus Sylvatica"
    counts = Counter(labels.tolist())
    assert sum(counts.values()) == 440 and min(counts.values()) == 27 and max(counts.values()) == 30
    assert counts[9] == 30 and counts[13] == 27
    # the authors scaled every outline to ±1.5; arc-length resampling stays inside that box
    assert points.min() >= -1.5 - 1e-9 and points.max() <= 1.5 + 1e-9
    # closed outlines resampled by arc length: roughly even steps (equal along the polygon,
    # so chords vary a little where a corner falls between two samples), closing step included
    closed = np.concatenate([points, points[:, :1]], axis=1)
    seg = np.linalg.norm(np.diff(closed, axis=1), axis=2)
    assert np.median(seg.std(axis=1) / seg.mean(axis=1)) < 0.1


def test_leaves_loader_refuses_a_mismatched_npz(tmp_path, monkeypatch):
    src = ROOT / "data" / "leaves_outlines.npz"
    if not src.exists():
        pytest.skip("leaves not fetched")
    with np.load(src, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    d["classes"] = np.array(["wrong"] * len(d["classes"]))
    np.savez(tmp_path / "leaves_outlines.npz", **d)
    monkeypatch.setenv("SHAPEKIT_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="classes"):
        get("leaves").load()


# ------------------------------------------------------------------ fetch-locally dataset

def test_rbc_raises_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAPEKIT_DATA", str(tmp_path))
    assert resolve_data_dir() == tmp_path
    with pytest.raises(DatasetNotFetched, match="--rbc"):
        get("rbc").load()
    for k in ("brain", "gorilla", "leaves"):
        with pytest.raises(DatasetNotFetched, match=f"--{k}"):
            get(k).load()


def test_rbc_local_file_if_present():
    """Only runs on a machine where --rbc was executed; nothing from it is in the repo."""
    ds = get("rbc")
    try:
        points, labels = ds.load()
    except DatasetNotFetched:
        pytest.skip("rbc not fetched locally")
    _check_basic(ds, points, labels, len(labels), 150)
    assert ds.classes == ("circular", "elongated", "other")
    assert all(Counter(labels.tolist())[k] > 0 for k in range(3))
    assert not (ROOT / "data" / "rbc_outlines.npz").exists(), "rbc data must never leave data/_local/"
