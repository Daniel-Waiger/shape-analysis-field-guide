"""The JSON, ONNX and index files shipped under docs/lab/ agree with each other and with the code.

These tests read what the page reads.  They do not recompute the lab (that is
``scripts/run_lab.py``); they check that what was written is internally
consistent, that the browser engines can reproduce the stored probabilities,
and that the licence guard held.
"""

import json
import pathlib

import numpy as np
import pytest

from shapekit.lab_datasets import DATASETS
from shapekit.lab_engine import classify
from shapekit.lab_export import BASELINE_METHODS, best_method

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB = ROOT / "docs" / "lab"
DOCS = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((LAB / "data").glob("*.json"))
        if p.stem != "index"}
READY = [k for k, d in DOCS.items() if d.get("status") == "ready"]


def ready_methods(doc):
    return {m: b for m, b in doc["methods"].items() if isinstance(b, dict) and b.get("status") == "ready"}


@pytest.mark.skipif(not DOCS, reason="no lab JSON on disk")
def test_index_matches_the_documents():
    index = json.loads((LAB / "data" / "index.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in index] == list(DATASETS)
    for row in index:
        doc = DOCS.get(row["id"])
        if doc is None:
            assert row["status"] == "proposed"
            continue
        assert row["status"] == doc["status"] and row["n"] == doc["n"] and row["chance"] == doc.get("chance")
        name, acc, protocol = best_method(doc)
        assert (row["best_method"], row["best_accuracy"], row["best_protocol"]) == (name, acc, protocol)
        if acc is not None:
            assert name not in BASELINE_METHODS
            assert acc == max(b["accuracy"] for m, b in ready_methods(doc).items() if m not in BASELINE_METHODS)
        assert row["methods"] == {m: (b.get("status", "ready") if isinstance(b, dict) else "ready")
                                  for m, b in doc["methods"].items()}


@pytest.mark.parametrize("dataset_id", READY)
def test_document_is_internally_consistent(dataset_id):
    doc = DOCS[dataset_id]
    assert sum(c["n"] for c in doc["classes"]) == doc["n"]
    assert np.isclose(doc["chance"], max(c["n"] for c in doc["classes"]) / doc["n"], atol=1e-4)
    assert doc["script"].startswith(f"scripts/run_lab.py --dataset {dataset_id}")
    assert "--skip-pointnet" not in doc["script"] and "--epochs" not in doc["script"], "partial run shipped"
    assert "correspondence" in doc and "timing" in doc and "seconds" in doc["timing"]
    assert "size_lr" in doc["methods"]
    for name, block in ready_methods(doc).items():
        assert "protocol" in block and block["size_information"] in ("removed", "only"), name
        assert "seconds" not in block, name
        assert "ddof=0" in block["accuracy_sd_def"]
        cm = np.array(block["cm"])
        assert np.isclose(np.trace(cm) / cm.sum(), block["accuracy"], atol=1e-4), name
        if name == "pointnet_probe":
            assert cm.sum() == block["n_val"] and "accuracy_select" in block and "untrained" in block
            assert block["n_fit"] + block["n_select"] == block["n_train"]
            assert len(block["history_select_acc"]) == block["epochs"] + 1
            assert block["input_normalisation"] == "per_cloud_rms" and abs(block["input_scale"] - 1.0) < 1e-5
            assert block["permutation_max_delta"] < 1e-4
        else:
            assert cm.sum() == doc["n"]
            assert np.isclose(np.mean(block["fold_accuracies"]), block["accuracy"], atol=2e-2)
            assert np.isclose(np.std(block["fold_accuracies"]), block["accuracy_sd"], atol=1e-4)
        for w in block.get("worst", []):
            # pred is the out-of-fold prediction (p_cv); p_engine is the full-data engine's vector (PointNet: the same)
            assert w["true"] != w["pred"] and w["pred"] == int(np.argmax(w.get("p_cv", w["p_engine"])))
            assert np.isclose(w["p"], w.get("p_cv", w["p_engine"])[w["pred"]], atol=1e-4)
    if doc["representation"] == "outline" and doc["closed"]:
        assert "cyclic_shift_alignment" in doc and isinstance(doc["cyclic_shift_alignment"]["applied"], bool)


@pytest.mark.parametrize("dataset_id", READY)
def test_shipped_engines_reproduce_their_probabilities(dataset_id):
    doc = DOCS[dataset_id]
    checked = 0
    for name, block in ready_methods(doc).items():
        if "engine" not in block:
            continue
        for w in block.get("worst", []):
            p = classify(np.asarray(w["shape"]), block["engine"])
            assert np.abs(p - np.asarray(w["p_engine"])).max() < 1e-3, (dataset_id, name, w["index"])
            checked += 1
    if doc["shapes_shown"]:
        assert doc["engine"] == doc["methods"]["landmarks_lr"].get("engine", doc["engine"])
    if not doc["shapes_shown"]:
        assert checked == 0


@pytest.mark.parametrize("dataset_id", [k for k in READY if not DOCS[k]["shapes_shown"]])
def test_restricted_documents_carry_no_coordinates(dataset_id):
    doc = DOCS[dataset_id]
    for key in ("pca", "examples", "example_indices", "neighbours", "engine"):
        assert key not in doc
    for name, block in doc["methods"].items():
        if isinstance(block, dict):
            assert not {"engine", "probe", "onnx_info"} & set(block), name
            assert block.get("onnx") is None
            for w in block.get("worst", []):
                assert not {"shape", "curve"} & set(w)
    assert not (LAB / "models" / f"{dataset_id}.onnx").exists()


@pytest.mark.parametrize("dataset_id", [k for k in READY if DOCS[k]["shapes_shown"]
                                        and DOCS[k]["methods"].get("pointnet_probe", {}).get("status") == "ready"])
def test_shipped_onnx_reproduces_the_probe(dataset_id):
    onnxruntime = pytest.importorskip("onnxruntime")
    from shapekit.lab_datasets import DatasetNotFetched, get

    doc = DOCS[dataset_id]
    pn = doc["methods"]["pointnet_probe"]
    path = LAB / pn["onnx"]
    assert path.exists() and path.stat().st_size == pn["onnx_info"]["bytes"]
    try:
        points, labels = get(dataset_id).load()
    except DatasetNotFetched as exc:
        pytest.skip(str(exc))
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probe = pn["probe"]
    for w in pn["worst"]:
        cloud = points[w["index"]].astype(np.float32)
        centred = cloud - cloud.mean(axis=0)
        prepared = centred / np.sqrt(np.mean(np.sum(centred**2, axis=1))) / np.float32(pn["input_scale"])
        emb = session.run(None, {"points": prepared[None]})[0][0].astype(np.float64)
        x = (emb - np.asarray(probe["scaler_mean"])) / np.asarray(probe["scaler_scale"])
        logits = np.asarray(probe["coef"]) @ x + np.asarray(probe["intercept"])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        assert np.abs(p - np.asarray(w["p_engine"])).max() < 2e-3, (dataset_id, w["index"])
        assert labels[w["index"]] == w["true"]
        # a shuffled, translated and scaled copy gives the same embedding (order and size removed)
        rng = np.random.default_rng(w["index"])
        other = (cloud[rng.permutation(len(cloud))] * 3.0 + 100.0)
        oc = other - other.mean(axis=0)
        op = oc / np.sqrt(np.mean(np.sum(oc**2, axis=1))) / np.float32(pn["input_scale"])
        assert np.abs(session.run(None, {"points": op[None]})[0][0] - emb).max() < 1e-4
