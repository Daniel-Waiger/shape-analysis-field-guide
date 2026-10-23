"""Tests for the Phenotype Lab (chapter 10): registry, run_lab.py, the classical engine, ONNX."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapekit import preshape  # noqa: E402
from shapekit.lab_datasets import DATASETS, DatasetNotFetched, LabDataset, Source, get  # noqa: E402
from shapekit.lab_engine import classify, explain  # noqa: E402
from shapekit.lab_export import best_method, finalize, index_entry, round_floats, strip_coordinates  # noqa: E402
from shapekit.lab_pipeline import (  # noqa: E402
    START_LANDMARK_MAJORITY,
    align_cyclic_shifts,
    classical_method,
    prepare_points,
    run_dataset,
    size_method,
)

EXPECTED_SHAPES = {"celegans": (1377, 12, 2), "mef": (1200, 120, 2), "brain": (28, 13, 2), "gorilla": (59, 8, 2)}
RUN_LAB = ROOT / "scripts" / "run_lab.py"


def load_run_lab():
    """Import scripts/run_lab.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("run_lab", RUN_LAB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def walk(obj, path=""):
    """Yield ``(path, value)`` for every leaf and container of a JSON document."""
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def looks_like_coordinates(value) -> bool:
    """A non-empty list whose items are pairs of numbers (or lists of such): shapes, weights, scatters."""
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if isinstance(first, list) and len(first) == 2 and all(isinstance(x, (int, float)) for x in first):
        return any(isinstance(x, float) for x in first)   # a confusion matrix of ints is not a coordinate
    if isinstance(first, list) and first and isinstance(first[0], list):
        return looks_like_coordinates(first)
    return False


def synthetic_outlines(n_per_class=60, n_points=40, seed=0):
    """Two classes of noisy closed outlines (rounder vs more elongated ellipses), random start points."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points, labels = [], []
    for c, (a, b) in enumerate([(1.0, 0.9), (1.0, 0.6)]):
        for _ in range(n_per_class):
            r = 1 + rng.normal(0, 0.03, n_points)
            x = np.stack([a * r * np.cos(t), b * r * np.sin(t)], axis=1) * rng.uniform(20, 40)
            theta = rng.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            x = x @ rot.T + rng.uniform(0, 500, 2)
            points.append(np.roll(x, rng.integers(0, n_points), axis=0))
            labels.append(c)
    return np.stack(points), np.array(labels)


SYNTHETIC_RESTRICTED = LabDataset(
    id="synthetic_restricted", title="Synthetic restricted outlines", question="test",
    representation="outline", classes=("round", "elongated"),
    source=Source(name="synthetic", citation="none", url="none", license="CC BY-NC-ND 4.0 (pretend)",
                  redistributed=False),
    loader=lambda: synthetic_outlines(), closed=True, shapes_shown=False,
    fetch_command="python scripts/fetch_datasets.py --synthetic",
)


# ----------------------------------------------------------------------------- #
# registry
# ----------------------------------------------------------------------------- #


def test_registry_entries_are_complete():
    for ds in DATASETS.values():
        assert ds.id and ds.title and ds.question
        assert ds.representation in ("landmarks", "outline", "pointcloud")
        assert ds.source.name and ds.source.license and ds.source.url
        assert get(ds.id) is ds
    with pytest.raises(KeyError):
        get("nope")


@pytest.mark.parametrize("dataset_id", [d.id for d in DATASETS.values() if d.source.redistributed])
def test_redistributed_datasets_load_with_stated_shapes(dataset_id):
    ds = get(dataset_id)
    try:
        points, labels = ds.load()
    except DatasetNotFetched as exc:
        pytest.skip(f"{dataset_id} not fetched yet: {exc}")
    assert points.ndim == 3 and points.shape[-1] == 2
    assert labels.shape == (len(points),)
    assert labels.min() == 0
    if dataset_id in EXPECTED_SHAPES:
        assert points.shape == EXPECTED_SHAPES[dataset_id]
    if ds.classes:
        assert labels.max() + 1 == len(ds.classes)


def test_fetch_locally_datasets_raise_or_load():
    ds = get("rbc")
    assert not ds.source.redistributed and not ds.shapes_shown and ds.fetch_command
    try:
        points, labels = ds.load()
    except DatasetNotFetched as exc:
        assert "fetch_datasets.py" in str(exc)
    else:
        assert points.ndim == 3 and labels.shape == (len(points),)


# ----------------------------------------------------------------------------- #
# the classical engine reproduces scikit-learn
# ----------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def worm_landmarks_lr(worms):
    points, labels = worms
    result, extras = classical_method(points, labels, ["dead", "live"], "landmarks_lr", seed=0)
    return points, labels, result, extras


def test_engine_reproduces_pipeline_probabilities(worm_landmarks_lr):
    points, labels, result, extras = worm_landmarks_lr
    engine = result["engine"]
    assert engine["n_components"] == 20 and len(engine["lr_coef"]) == 2
    for w in result["worst"]:
        p = classify(w["shape"], engine)
        assert np.allclose(p, w["p_engine"], atol=1e-6)
        assert np.isclose(p.sum(), 1.0)
    # any raw worm, in its original position, size and rotation, gives the same answer
    k = 7
    p_raw = classify(points[k], engine)
    p_aligned = classify(extras["aligned"][k], engine)
    assert np.allclose(p_raw, p_aligned, atol=1e-8)
    # a rotated, scaled, shifted copy too
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = (points[k] @ rot.T) * 3.0 + np.array([100.0, -40.0])
    assert np.allclose(classify(moved, engine), p_raw, atol=1e-8)


def test_engine_survives_json_rounding(worm_landmarks_lr):
    _, _, result, _ = worm_landmarks_lr
    rounded = finalize({"shapes_shown": True, "methods": {"landmarks_lr": result}}, "test")
    block = json.loads(json.dumps(rounded["methods"]["landmarks_lr"]))
    for w in block["worst"]:
        p = classify(np.asarray(w["shape"]), block["engine"])
        assert np.allclose(p, w["p_engine"], atol=1e-3)


def test_explain_reports_the_steps(worm_landmarks_lr):
    points, _, result, _ = worm_landmarks_lr
    steps = explain(points[0], result["engine"])
    assert np.allclose(steps["centroid"], points[0].mean(axis=0))
    assert np.isclose(steps["size"], np.sqrt(np.sum((points[0] - points[0].mean(axis=0)) ** 2)))
    assert np.isclose(np.sqrt(np.sum(steps["aligned"] ** 2)), 1.0)
    assert 0.0 <= steps["rho"] <= np.pi / 2
    assert steps["scores"].shape == (20,) and steps["proba"].shape == (2,)


def test_binary_rows_are_symmetric(worm_landmarks_lr):
    _, _, result, _ = worm_landmarks_lr
    coef = np.asarray(result["engine"]["lr_coef"])
    assert np.allclose(coef[0], -coef[1])
    assert result["engine"]["binary_rows"] == "symmetric"


# ----------------------------------------------------------------------------- #
# preprocessing and export helpers
# ----------------------------------------------------------------------------- #


def test_prepare_points_by_representation():
    worms = get("celegans")
    prep = prepare_points(worms, np.random.default_rng(0).normal(size=(5, 12, 2)))
    assert prep["landmarks"].shape == (5, 12, 2) and prep["spline"] == {"M": 6, "closed": False}
    cells = get("mef")
    prep = prepare_points(cells, np.random.default_rng(0).normal(size=(5, 120, 2)))
    assert prep["landmarks"] is None and prep["spline"] is None and prep["display"].shape == (5, 120, 2)
    assert prep["correspondence"].startswith("none")
    leaves = get("leaves")
    prep = prepare_points(leaves, np.random.default_rng(0).normal(size=(5, 150, 2)))
    assert prep["landmarks"].shape == (5, 100, 2) and prep["spline"] == {"M": 12, "closed": True}
    assert prep["cyclic"] is not None and "applied" in prep["cyclic"] and prep["correspondence"].startswith("arc length")
    gorilla = get("gorilla")
    prep = prepare_points(gorilla, np.random.default_rng(0).normal(size=(5, 8, 2)))
    assert prep["landmarks"].shape == (5, 8, 2) and prep["spline"] is None
    assert prep["correspondence"] == "anatomical landmarks"


def test_align_cyclic_shifts_recovers_random_start_points():
    points, labels = synthetic_outlines(n_per_class=40, n_points=40, seed=1)
    original = points.copy()
    rolled, info = align_cyclic_shifts(points)
    assert np.array_equal(points, original), "the input must not be rolled in place"
    assert rolled.shape == points.shape and info["rounds"] == 3
    # the starts were random: few outlines were already at their best shift, most had to move
    assert info["fraction_near_zero"] < START_LANDMARK_MAJORITY
    assert info["n_shifted"] > 0.7 * len(points)
    assert info["mean_distance_after"] < 0.8 * info["mean_distance_before"]
    # every rolled outline is a cyclic shift of the original (same direction, no reversal)
    for k in range(len(points)):
        assert np.allclose(rolled[k], np.roll(original[k], -info["shifts"][k], axis=0))
    # a second pass finds (almost) nothing left to do: the starts now agree
    _, again = align_cyclic_shifts(rolled)
    assert again["fraction_near_zero"] > 0.9
    # and the classifier sees it: aligned outlines separate the two classes, unaligned ones barely
    prep = prepare_points(SYNTHETIC_RESTRICTED, points)
    assert prep["cyclic"]["applied"] is True and "cyclic-shift" in prep["correspondence"]
    acc_aligned = classical_method(prep["landmarks"], labels, ["a", "b"], "landmarks_lr")[0]["accuracy"]
    acc_raw = classical_method(points, labels, ["a", "b"], "landmarks_lr")[0]["accuracy"]
    assert acc_aligned > acc_raw
    # a source whose start point is consistent keeps it
    prep2 = prepare_points(SYNTHETIC_RESTRICTED, rolled)
    assert prep2["cyclic"]["applied"] is False and "kept" in prep2["correspondence"]
    assert np.array_equal(prep2["landmarks"], rolled)


def test_size_method_is_the_size_only_baseline(worms):
    points, labels = worms
    res = size_method(points, labels, ["dead", "live"], seed=0)
    assert res["status"] == "ready" and res["size_information"] == "only" and res["n_components"] == 1
    assert res["protocol"] == "stratified 5-fold CV, seed 0, pooled out-of-fold"
    assert "worst" not in res and "engine" not in res and "shape" not in json.dumps(round_floats(res, 4))
    assert np.isclose(res["accuracy"], 0.7466, atol=5e-4)   # pooled out-of-fold, centroid size (content review)
    assert np.array(res["cm"]).sum() == 1377 and len(res["fold_accuracies"]) == 5
    assert res["size_median_by_class"][0] > res["size_median_by_class"][1]   # dead worms are larger (chapter 02)


def test_best_method_excludes_the_size_baseline():
    doc = {"methods": {"landmarks_lr": {"accuracy": 0.8, "protocol": "cv"}, "size_lr": {"accuracy": 0.95, "protocol": "cv"},
                       "pointnet_probe": {"status": "skipped"}}}
    assert best_method(doc) == ("landmarks_lr", 0.8, "cv")
    row = index_entry(get("celegans"), {**doc, "status": "ready", "classes": [], "chance": 0.5})
    assert row["best_accuracy"] == 0.8 and row["best_method"] == "landmarks_lr" and row["best_protocol"] == "cv"
    assert best_method({"methods": {"size_lr": {"accuracy": 0.95}}}) == (None, None, None)


def test_round_floats_and_strip():
    doc = {"a": np.float64(1.23456789), "b": [np.int64(3), -0.00001], "c": {"d": np.zeros((2, 2))}}
    out = round_floats(doc, 4)
    assert out == {"a": 1.2346, "b": [3, 0.0], "c": {"d": [[0.0, 0.0], [0.0, 0.0]]}}
    stripped = strip_coordinates({
        "shapes_shown": False, "pca": {}, "examples": {}, "neighbours": {}, "engine": {},
        "methods": {"landmarks_lr": {"accuracy": 0.9, "engine": {}, "worst": [{"index": 1, "shape": [[0, 0]]}]},
                    "pointnet_probe": {"accuracy": 0.8, "probe": {}, "onnx": "models/x.onnx"}},
    })
    assert "pca" not in stripped and "engine" not in stripped
    assert stripped["methods"]["landmarks_lr"]["worst"] == [{"index": 1}]
    assert "engine" not in stripped["methods"]["landmarks_lr"]
    assert stripped["methods"]["pointnet_probe"]["onnx"] is None and "probe" not in stripped["methods"]["pointnet_probe"]


# ----------------------------------------------------------------------------- #
# the script writes the contract
# ----------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lab_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("lab")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_lab.py"), "--dataset", "celegans", "--skip-pointnet",
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out, proc.stdout


def test_run_lab_writes_the_contract(lab_output):
    out, stdout = lab_output
    doc = json.loads((out / "data" / "celegans.json").read_text())
    for key in ("id", "title", "question", "representation", "n_points", "n", "status", "shapes_shown", "source",
                "classes", "computed_at", "script", "examples", "pca", "methods", "engine", "neighbours"):
        assert key in doc, key
    assert doc["id"] == "celegans" and doc["status"] == "ready" and doc["n"] == 1377 and doc["n_points"] == 12
    assert doc["script"] == "scripts/run_lab.py --dataset celegans --skip-pointnet"   # the exact flags
    assert doc["computed_at"].endswith("Z")
    assert doc["correspondence"] == "anatomical landmarks"
    assert set(doc["timing"]["seconds"]) >= {"landmarks_lr", "spline_lr", "size_lr", "total"}
    assert doc["timing"]["threads"] == 4
    assert [c["name"] for c in doc["classes"]] == ["dead", "live"]
    assert sum(c["n"] for c in doc["classes"]) == 1377
    assert doc["source"]["redistributed"] is True and doc["source"]["license"]

    lm = doc["methods"]["landmarks_lr"]
    assert 0.7 < lm["accuracy"] < 1.0 and lm["accuracy"] > lm["chance"]
    assert np.isclose(lm["chance"], 798 / 1377, atol=1e-4)
    assert np.array(lm["cm"]).shape == (2, 2) and np.array(lm["cm"]).sum() == 1377
    assert len(lm["fold_accuracies"]) == 5 and len(lm["worst"]) == 3
    for w in lm["worst"]:
        assert w["true"] != w["pred"] and 0.5 <= w["p"] <= 1.0
        assert np.array(w["shape"]).shape == (12, 2) and len(w["p_engine"]) == 2
    sp = doc["methods"]["spline_lr"]
    assert sp["M"] == 6 and 0.6 < sp["accuracy"] < 1.0
    assert np.array(sp["worst"][0]["shape"]).shape == (6, 2) and np.array(sp["worst"][0]["curve"]).shape == (60, 2)
    assert doc["methods"]["pointnet_probe"] == {"status": "skipped", "reason": "--skip-pointnet"}
    sz = doc["methods"]["size_lr"]
    assert sz["size_information"] == "only" and 0.7 < sz["accuracy"] < lm["accuracy"] and "worst" not in sz
    for name in ("landmarks_lr", "spline_lr", "size_lr"):
        block = doc["methods"][name]
        assert block["protocol"] == "stratified 5-fold CV, seed 0, pooled out-of-fold"
        assert block["size_information"] in ("removed", "only")
        assert "ddof=0" in block["accuracy_sd_def"]
        assert "seconds" not in block, "wall-clock times live under timing, not in the method blocks"
        assert np.isclose(np.trace(block["cm"]) / np.sum(block["cm"]), block["accuracy"], atol=1e-4)

    engine = doc["engine"]
    assert engine["method"] == "landmarks_lr" and engine["tangent_layout"] == "re_then_im"
    assert len(engine["mean"]) == 12 and len(engine["pca_mean"]) == 24
    assert len(engine["components"]) == engine["n_components"] == 20 and len(engine["components"][0]) == 24
    assert len(engine["scaler_mean"]) == len(engine["scaler_scale"]) == 20
    assert len(engine["lr_coef"]) == 2 and len(engine["lr_coef"][0]) == 20 and len(engine["lr_intercept"]) == 2

    assert len(doc["pca"]["explained"]) == 10 and len(doc["pca"]["modes"]) == 2
    assert np.array(doc["pca"]["mean"]).shape == (12, 2) and len(doc["pca"]["scatter"]) == 400
    assert len(doc["pca"]["scatter"][0]) == 3
    assert all(len(doc["examples"][c]) == 5 for c in ("0", "1"))
    assert len(doc["neighbours"]["shapes"]) == 300 == len(doc["neighbours"]["labels"]) == len(doc["neighbours"]["indices"])

    # rounding: coordinates to 4 decimals, weights to 6
    assert all(len(str(v).split(".")[-1]) <= 4 for row in doc["examples"]["0"][0] for v in row)
    assert all(len(repr(v).split(".")[-1]) <= 6 for v in engine["lr_coef"][0])

    index = json.loads((out / "data" / "index.json").read_text())
    assert [r["id"] for r in index] == list(DATASETS)
    row = next(r for r in index if r["id"] == "celegans")
    assert row["status"] == "ready" and row["best_accuracy"] == lm["accuracy"] and row["classes"] == ["dead", "live"]
    assert row["best_method"] == "landmarks_lr" and row["best_protocol"] == lm["protocol"]
    assert all(r["status"] in ("ready", "proposed", "fetch_locally") for r in index)
    assert "landmarks_lr" in stdout
    # the redistributable data files were copied next to the JSON for the Pyodide panel
    files = out / "data" / "files"
    for src in (ROOT / "data").iterdir():
        if src.is_file():
            assert (files / src.name).read_bytes() == src.read_bytes()
    assert not any("rbc" in p.name for p in files.iterdir())


def test_written_engine_reproduces_probabilities(lab_output):
    out, _ = lab_output
    doc = json.loads((out / "data" / "celegans.json").read_text())
    for name in ("landmarks_lr", "spline_lr"):
        block = doc["methods"][name]
        for w in block["worst"]:
            p = classify(np.asarray(w["shape"]), block["engine"])
            assert np.allclose(p, w["p_engine"], atol=1e-3), (name, p, w["p_engine"])
            # p (cross-validated) and p_engine (fitted on all shapes) may disagree; both are kept
            assert int(np.argmax(p)) == int(np.argmax(w["p_engine"]))
            assert w["true"] != w["pred"]


def _run_lab(args, env_data=None, timeout=300):
    env = dict(os.environ)
    if env_data is not None:
        env["SHAPEKIT_DATA"] = str(env_data)
    return subprocess.run([sys.executable, str(RUN_LAB), *args], capture_output=True, text=True, cwd=ROOT,
                          timeout=timeout, env=env)


def test_fetch_locally_stub_and_overwrite_guard(tmp_path):
    """With no data at all (SHAPEKIT_DATA empty) rbc gets a stub, but never over a ready JSON without --force."""
    empty = tmp_path / "nodata"
    empty.mkdir()
    out = tmp_path / "out"
    proc = _run_lab(["--dataset", "rbc", "--out", str(out)], env_data=empty)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    path = out / "data" / "rbc.json"
    doc = json.loads(path.read_text())
    assert doc["status"] == "fetch_locally" and doc["shapes_shown"] is False
    assert doc["fetch_command"] == "python scripts/fetch_datasets.py --rbc"
    assert "examples" not in doc and "engine" not in doc
    index = json.loads((out / "data" / "index.json").read_text())
    assert next(r for r in index if r["id"] == "rbc")["status"] == "fetch_locally"

    # a ready JSON (someone's real rbc results) survives a run on a machine without the data
    ready = {"id": "rbc", "title": "x", "status": "ready", "shapes_shown": False, "n": 5, "chance": 0.6,
             "classes": [{"id": 0, "name": "a", "n": 3}, {"id": 1, "name": "b", "n": 2}],
             "methods": {"landmarks_lr": {"accuracy": 0.9, "protocol": "cv"}}}
    path.write_text(json.dumps(ready))
    proc = _run_lab(["--dataset", "rbc", "--out", str(out)], env_data=empty)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(path.read_text()) == ready and "kept the existing" in proc.stdout
    index = json.loads((out / "data" / "index.json").read_text())
    row = next(r for r in index if r["id"] == "rbc")
    assert row["status"] == "ready" and row["best_accuracy"] == 0.9 and row["best_method"] == "landmarks_lr"
    # --force replaces it
    proc = _run_lab(["--dataset", "rbc", "--out", str(out), "--force"], env_data=empty)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads(path.read_text())
    assert doc["status"] == "fetch_locally" and doc["script"].endswith("--force")


def test_partial_runs_refuse_the_default_output_folder():
    for flags in (["--skip-pointnet"], ["--epochs", "1"]):
        proc = _run_lab(["--dataset", "celegans", *flags], timeout=60)
        assert proc.returncode == 2 and "--out DIR" in proc.stderr, proc.stderr
    # nothing in docs/lab was touched by those refusals: the self-check below reads the shipped files
    assert (ROOT / "docs" / "lab" / "data" / "index.json").exists()


def test_self_check_failure_is_an_error(lab_output, monkeypatch, tmp_path):
    out, _ = lab_output
    run_lab = load_run_lab()
    path = out / "data" / "celegans.json"
    assert run_lab.self_check(path) < 1e-3
    doc = json.loads(path.read_text())
    doc["methods"]["landmarks_lr"]["worst"][0]["p_engine"] = [0.5, 0.5]
    bad = tmp_path / "celegans.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(run_lab.SelfCheckError):
        run_lab.self_check(bad)
    # and the script exits non-zero when the check fails during a run
    monkeypatch.setattr(run_lab, "classify", lambda shape, engine: np.array([0.5, 0.5]))
    code = run_lab.main(["--dataset", "celegans", "--skip-pointnet", "--out", str(tmp_path / "run")])
    assert code == 1


def test_script_command_records_every_flag():
    run_lab = load_run_lab()
    args = argparse.Namespace(skip_pointnet=True, epochs=3, seed=1, threads=2, force=True)
    assert run_lab.script_command("mef", args) == ("scripts/run_lab.py --dataset mef --skip-pointnet --epochs 3 "
                                                   "--seed 1 --threads 2 --force")
    plain = argparse.Namespace(skip_pointnet=False, epochs=None, seed=0, threads=4, force=False)
    assert run_lab.script_command("mef", plain) == "scripts/run_lab.py --dataset mef"


def test_copy_data_files_refuses_restricted_names(tmp_path):
    run_lab = load_run_lab()
    src = tmp_path / "data"
    src.mkdir()
    (src / "ok.npz").write_bytes(b"x")
    (src / "_local").mkdir()
    (src / "_local" / "rbc_outlines.npz").write_bytes(b"secret")
    out = tmp_path / "out"
    assert run_lab.copy_data_files(out, data_dir=src, log=lambda m: None) == ["ok.npz"]
    assert (out / "data" / "files" / "ok.npz").read_bytes() == b"x"
    (src / "rbc_outlines.npz").write_bytes(b"secret")
    with pytest.raises(RuntimeError, match="licence-restricted"):
        run_lab.copy_data_files(out, data_dir=src, log=lambda m: None)


# ----------------------------------------------------------------------------- #
# the licence guard, end to end (no torch needed)
# ----------------------------------------------------------------------------- #


def assert_no_coordinates(doc: dict, onnx_dir: Path):
    for key in ("pca", "examples", "example_indices", "neighbours", "engine"):
        assert key not in doc, key
    for name, block in doc["methods"].items():
        if isinstance(block, dict):
            for key in ("engine", "probe", "onnx_info"):
                assert key not in block, (name, key)
            for w in block.get("worst", []):
                assert set(w) <= {"index", "true", "pred", "p", "p_cv", "p_engine"}, (name, sorted(w))
            if "onnx" in block:
                assert block["onnx"] is None
    for path, value in walk(doc):
        assert not looks_like_coordinates(value), f"coordinate-like array at {path}"
    assert not list(Path(onnx_dir).glob("*.onnx")), "no ONNX file may be written for a restricted dataset"
    assert doc["shapes_shown"] is False and "shapes_note" in doc
    assert doc["methods"]["size_lr"]["status"] == "ready"


def test_licence_guard_end_to_end_classical(tmp_path):
    points, labels = SYNTHETIC_RESTRICTED.load()
    raw = run_dataset(SYNTHETIC_RESTRICTED, points, labels, seed=0, skip_pointnet=True, onnx_dir=tmp_path)
    assert "pca" in raw and "engine" in raw   # computed like every other dataset ...
    doc = json.loads(json.dumps(finalize(raw, "test")))   # ... and stripped on export
    assert_no_coordinates(doc, tmp_path)
    assert doc["methods"]["landmarks_lr"]["accuracy"] > 0.9 and doc["cyclic_shift_alignment"]["applied"] is True
    assert doc["preprocessing"] and any("re-rooted" in n for n in doc["preprocessing"])
    row = index_entry(SYNTHETIC_RESTRICTED, doc)
    assert row["best_method"] in ("landmarks_lr", "spline_lr") and row["shapes_shown"] is False


# ----------------------------------------------------------------------------- #
# PointNet + ONNX
# ----------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="PointNet needs torch")
onnxruntime = pytest.importorskip("onnxruntime", reason="ONNX round trip needs onnxruntime")


def test_onnx_round_trip(tmp_path):
    from shapekit.embed import PointNet, preprocess_points
    from shapekit.lab_pipeline import export_onnx

    torch.manual_seed(0)
    model = PointNet(embed_dim=256, proj_dim=64).eval()
    clouds = np.random.default_rng(0).normal(size=(5, 30, 2)).astype(np.float32) * 20 + 100
    prepared, scale = preprocess_points(clouds)
    model.input_scale.fill_(scale)
    path = tmp_path / "m.onnx"
    info = export_onnx(model, path, prepared)
    assert path.exists() and info["verified"] and info["max_abs_diff_vs_torch"] < 1e-4 and info["opset"] == 17
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "points" and session.get_outputs()[0].name == "embedding"
    # dynamic batch and point count, and permutation invariance
    one = session.run(None, {"points": prepared[:1]})[0]
    assert one.shape == (1, 256)
    shuffled = prepared[:1, np.random.default_rng(1).permutation(30)]
    assert np.abs(session.run(None, {"points": shuffled})[0] - one).max() < 1e-4
    other = session.run(None, {"points": np.zeros((2, 17, 2), np.float32)})[0]
    assert other.shape == (2, 256)
    assert np.allclose(model.embed(clouds), session.run(None, {"points": prepared})[0], atol=1e-4)


def test_pointnet_method_small(worms, tmp_path):
    from shapekit.lab_pipeline import pointnet_method

    points, labels = worms
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(labels), 160, replace=False))
    res = pointnet_method(points[pick], labels[pick], ["dead", "live"], epochs=1, seed=0,
                          onnx_path=tmp_path / "w.onnx", onnx_rel="models/w.onnx", threads=2)
    assert res["status"] == "ready" and res["n_train"] + res["n_val"] == 160
    assert res["n_fit"] + res["n_select"] == res["n_train"] and res["n_select"] == round(0.2 * res["n_train"])
    assert 0.0 <= res["accuracy"] <= 1.0 and 0.0 <= res["untrained"] <= 1.0 and 0.0 <= res["accuracy_select"] <= 1.0
    assert res["protocol"] == "stratified 75/25 hold-out, seed 0; epoch chosen on a 20 % slice of the training part"
    assert res["size_information"] == "removed" and res["input_normalisation"] == "per_cloud_rms"
    assert np.isclose(res["input_scale"], 1.0, atol=1e-5)
    assert len(res["history_select_acc"]) == 2 and "seconds" not in res and res["_timing"]["train"] > 0
    assert np.array(res["cm"]).sum() == res["n_val"]
    assert res["onnx"] == "models/w.onnx" and res["onnx_info"]["verified"]
    probe = res["probe"]
    assert len(probe["coef"]) == 2 and len(probe["coef"][0]) == 256 and len(probe["scaler_mean"]) == 256
    # the probe weights classify an embedding exactly like the sklearn pipeline did
    session = onnxruntime.InferenceSession(str(tmp_path / "w.onnx"), providers=["CPUExecutionProvider"])
    w = res["worst"][0]
    subset = points[pick]  # worst[].index indexes the clouds given to pointnet_method
    cloud = subset[w["index"]].astype(np.float32)
    centred = cloud - cloud.mean(axis=0)
    prepared = centred / np.sqrt(np.mean(np.sum(centred**2, axis=1))) / np.float32(res["input_scale"])
    emb = session.run(None, {"points": prepared[None]})[0][0].astype(np.float64)
    x = (emb - np.asarray(probe["scaler_mean"])) / np.asarray(probe["scaler_scale"])
    logits = np.asarray(probe["coef"]) @ x + np.asarray(probe["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert np.allclose(p, w["p_engine"], atol=1e-3)
    assert np.array(w["shape"]).shape == (12, 2) and np.allclose(preshape(subset[w["index"]]), w["shape"])
    # the network's input has unit RMS radius whatever the cell's size was (size removed before the model)
    assert np.isclose(np.sqrt(np.mean(np.sum(prepared**2, axis=1))), 1.0, atol=1e-5)


def test_licence_guard_end_to_end_with_pointnet(tmp_path):
    """A shapes_shown=False dataset through run_dataset + finalize, PointNet branch included: no ONNX, no shapes."""
    points, labels = SYNTHETIC_RESTRICTED.load()
    raw = run_dataset(SYNTHETIC_RESTRICTED, points, labels, seed=0, epochs=1, onnx_dir=tmp_path, threads=2)
    pn = raw["methods"]["pointnet_probe"]
    assert pn["status"] == "ready" and pn["onnx"] is None and "onnx_info" not in pn and "probe" in pn
    doc = json.loads(json.dumps(finalize(raw, "test")))
    assert_no_coordinates(doc, tmp_path)
    pn = doc["methods"]["pointnet_probe"]
    assert pn["protocol"].startswith("stratified 75/25 hold-out") and pn["size_information"] == "removed"
    assert "permutation_max_delta" in pn and "probe" not in pn
    assert set(doc["timing"]["seconds"]) >= {"pointnet_train", "pointnet_probe", "total"}
