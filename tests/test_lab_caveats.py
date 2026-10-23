"""The MEF confound (LAB_SPEC.md Round 3b): registry caveats, the JSON and index flags, the roundness baseline.

The tutorial's wild-type fields all come from ``Control/LMNA++1`` (unpatterned) and its LMNA−/− fields from
``Circle/LMNA--1`` (circle micropattern), so genotype is confounded with substrate.  These tests check that
the caveat travels from the registry into ``mef.json`` and ``index.json``, that the display-only
``roundness_baseline`` is there, deterministic and kept out of the methods, and that the page and the sample
manifest say so.
"""

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from shapekit.lab_datasets import DATASETS, MEF, DatasetNotFetched, LabDataset
from shapekit.lab_export import best_method, fetch_locally_document, finalize, index_entry
from shapekit.lab_pipeline import (
    ROUNDNESS_BASELINE_DATASETS,
    POINT_ORDER,
    ROUNDNESS_LABEL,
    axis_ratio,
    circularity,
    roundness_baseline,
    run_dataset,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB = ROOT / "docs" / "lab"
MEF_JSON = LAB / "data" / "mef.json"
INDEX_JSON = LAB / "data" / "index.json"


def _json(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not written yet")
    return json.loads(path.read_text(encoding="utf-8"))


def _clouds(n_per_class=40, n_points=60, seed=0):
    """Round clouds (class 0) vs elongated ragged clouds (class 1), unordered points."""
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for c in (0, 1):
        for _ in range(n_per_class):
            t = rng.uniform(0, 2 * np.pi, n_points)
            a, b = (1.0, 0.95) if c == 0 else (1.0, 0.45)
            r = 1 + rng.normal(0, 0.02 if c == 0 else 0.12, n_points)
            pts.append(np.stack([a * r * np.cos(t), b * r * np.sin(t)], axis=1) * 30 + rng.uniform(0, 400, 2))
            labels.append(c)
    return np.stack(pts), np.array(labels)


# ------------------------------------------------------------------ registry

def test_every_dataset_has_a_caveats_tuple():
    assert LabDataset.__dataclass_fields__["caveats"].default == ()
    for ds in DATASETS.values():
        assert isinstance(ds.caveats, tuple) and all(isinstance(c, str) and c.strip() for c in ds.caveats)
    assert [d.id for d in DATASETS.values() if d.caveats] == ["mef"]


def test_mef_caveat_names_the_folders_and_the_evidence():
    text = " ".join(MEF.caveats)
    for needle in ("Control/LMNA++1", "Circle/LMNA--1", "unpatterned", "circle micropattern", "28 of 28",
                   "14 sampled wild-type", "14 sampled LMNA−/−", "cannot be attributed to genotype", "7cbe667"):
        assert needle in text, needle
    # the folder suffixes are undefined upstream: no "replicate" claim, the suffixes named as unknown
    assert "replicate" not in text.lower()
    assert "does not say what its folder suffixes 1 and 2 mean" in text
    assert "99.5th percentile" in text and "647 px" in text   # the clip and the field that fails without it
    assert MEF.question == ("Can a classifier tell these two groups of fibroblasts apart from outlines alone, "
                            "and what is it actually seeing?")
    assert MEF.title == "Mouse fibroblasts, wild-type vs LMNA−/−"   # the page adds the note, the title stays


# ------------------------------------------------------------------ pipeline / export

def test_circularity_and_axis_ratio_are_sane():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)[None] * 7
    ellipse = np.stack([3 * np.cos(t), np.sin(t)], axis=1)[None]
    assert circularity(circle)[0] == pytest.approx(1.0, abs=1e-3)
    assert circularity(ellipse)[0] < 0.7
    assert axis_ratio(circle)[0] == pytest.approx(1.0, abs=1e-9)
    assert axis_ratio(ellipse)[0] == pytest.approx(3.0, rel=1e-6)
    # order-free: shuffling the points does not change the axis ratio
    rng = np.random.default_rng(1)
    assert axis_ratio(ellipse[:, rng.permutation(200)])[0] == pytest.approx(3.0, rel=1e-6)


def test_roundness_baseline_measures_circularity_in_loop_order():
    """A shuffled cloud gets the circularity of its outline, not of an angle-sorted zig-zag."""
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    r = 1 + 0.6 * np.cos(5 * t)
    phi = t + 1.2 * (r - 1)                                   # twisted star: angle order fails on it
    star = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1) * 40
    rng = np.random.default_rng(0)
    clouds = np.stack([star[rng.permutation(120)] + rng.uniform(0, 100, 2) for _ in range(20)])
    labels = np.repeat([0, 1], 10)
    rb = roundness_baseline(clouds, labels, ["a", "b"], seed=0)
    true_circ = circularity(star[None])[0]
    assert rb["circularity_median_by_class"][0] == pytest.approx(true_circ, rel=1e-6)
    assert rb["point_order"] == POINT_ORDER


def test_roundness_baseline_is_deterministic_and_separates_round_from_elongated():
    pts, labels = _clouds()
    a = roundness_baseline(pts, labels, ["round", "elongated"], seed=0)
    b = roundness_baseline(pts.copy(), labels.copy(), ["round", "elongated"], seed=0)
    assert a == b
    assert a["label"] == ROUNDNESS_LABEL == "roundness only (two hand-made numbers)"
    assert a["role"] == "diagnostic" and a["n_features"] == 2 and a["n_folds"] == 5
    assert "StratifiedKFold(5, shuffle=True, random_state=0)" in a["cv"]
    assert a["accuracy"] > 0.9
    assert np.sum(a["cm"]) == len(labels)
    circ, ratio = a["circularity_median_by_class"], a["axis_ratio_median_by_class"]
    assert circ[0] > circ[1] and ratio[1] > ratio[0]


def test_run_dataset_writes_caveats_and_the_roundness_block_outside_the_methods():
    assert ROUNDNESS_BASELINE_DATASETS == ("mef",)
    pts, labels = _clouds(n_per_class=30)
    ds = dataclasses.replace(MEF, loader=lambda: (pts, labels))
    doc = finalize(run_dataset(ds, pts, labels, seed=0, skip_pointnet=True), "test")
    assert doc["caveats"] == list(MEF.caveats)
    assert "roundness_baseline" not in doc["methods"]
    rb = doc["roundness_baseline"]
    assert rb["status"] == "ready" and 0 <= rb["accuracy"] <= 1
    assert best_method(doc)[0] != "roundness_baseline"
    row = index_entry(ds, doc)
    assert row["has_caveat"] is True
    # a dataset without caveats: no key in its JSON, has_caveat false in the index
    plain = dataclasses.replace(MEF, id="plain", caveats=())
    doc2 = finalize(run_dataset(plain, pts, labels, seed=0, skip_pointnet=True), "test")
    assert "caveats" not in doc2 and "roundness_baseline" not in doc2
    assert index_entry(plain, doc2)["has_caveat"] is False
    assert index_entry(plain, None)["has_caveat"] is False and index_entry(ds, None)["has_caveat"] is True


def test_fetch_locally_stub_carries_the_caveats():
    stub = fetch_locally_document(MEF, "scripts/run_lab.py --dataset mef", "not fetched")
    assert stub["caveats"] == list(MEF.caveats)
    assert "caveats" not in fetch_locally_document(DATASETS["rbc"], "x", "not fetched")


# ------------------------------------------------------------------ shipped files

def test_shipped_mef_json_carries_the_caveats():
    doc = _json(MEF_JSON)
    assert doc["caveats"] == list(MEF.caveats)
    assert doc["question"] == MEF.question
    for other in DATASETS.values():
        path = LAB / "data" / f"{other.id}.json"
        if other.id != "mef" and path.exists():
            assert json.loads(path.read_text(encoding="utf-8")).get("caveats", []) == list(other.caveats)


def test_shipped_index_flags_the_caveat():
    index = _json(INDEX_JSON)
    assert {r["id"]: r["has_caveat"] for r in index} == {d.id: bool(d.caveats) for d in DATASETS.values()}


def test_shipped_roundness_baseline_is_present_and_reproducible():
    doc = _json(MEF_JSON)
    rb = doc["roundness_baseline"]
    assert rb["label"] == ROUNDNESS_LABEL and rb["role"] == "diagnostic"
    assert "roundness_baseline" not in doc["methods"]
    assert set(rb) >= {"accuracy", "accuracy_sd", "fold_accuracies", "cm", "protocol", "circularity_median_by_class",
                       "axis_ratio_median_by_class"}
    assert np.sum(rb["cm"]) == doc["n"]
    try:
        points, labels = MEF.load()
    except DatasetNotFetched as exc:  # pragma: no cover - mef ships in data/
        pytest.skip(str(exc))
    again = finalize({"roundness_baseline": roundness_baseline(points, labels, list(MEF.classes), seed=doc["seed"])},
                     "x")["roundness_baseline"]
    assert again == rb   # same numbers as the shipped file, to the stored decimals
    circ = rb["circularity_median_by_class"]
    assert circ[1] > circ[0]   # LMNA−/− (on circles) rounder than wild-type (unpatterned)
    assert rb["point_order"] == POINT_ORDER == "nearest-neighbor + 2-opt loop"


def test_mef_sample_captions_name_the_original_folders():
    man = _json(LAB / "data" / "images" / "mef" / "manifest.json")
    caps = {im["class"]: im["caption"] for im in man["images"]}
    assert "original folder Control/LMNA++1 (unpatterned)" in caps["wild-type"]
    assert "original folder Circle/LMNA--1 (circle micropattern)" in caps["LMNA−/−"]
    assert all("confounded" in c for c in caps.values())


# ------------------------------------------------------------------ page

def test_page_shows_the_caveat():
    js = (LAB / "lab.js").read_text(encoding="utf-8")
    css = (LAB / "lab.css").read_text(encoding="utf-8")
    for needle in ("function caveatBox(", "function caveatMark(", "has_caveat", "roundness_baseline",
                   "titleNote: 'genotype confounded with substrate'", "caveat-foot", "card-caveat", "aria-label"):
        assert needle in js, needle
    # the box goes above panel 0: appended to main before the panels grid that panel 0 opens
    assert js.index("if (cavBox) main.appendChild(cavBox);\n  const panels") < js.index(
        "panels.appendChild(pixelsPanel(")
    for needle in (".caveat-box", ".caveat-mark", ".card-caveat", ".caveat-foot", ".title-note"):
        assert needle in css, needle
    # the badge colors are tokens redefined for both dark-mode selectors
    assert css.count("--caveat-badge-bg:") == 3 and css.count("--caveat-badge-fg:") == 3
