"""Write the Phenotype Lab's JSON files (chapter 10).

:func:`finalize` turns the unrounded dict of :func:`shapekit.lab_pipeline.run_dataset`
into the document of the JSON contract in LAB_SPEC.md: numpy arrays become
lists, coordinates are rounded to 4 decimals, engine / probe weights and
``input_scale`` to 6, tiny diagnostics (``permutation_max_delta``,
``max_abs_diff_vs_torch``) to 3 significant digits, and for a dataset whose
licence forbids publishing derived coordinates (``shapes_shown: false``)
every coordinate-carrying block is removed.
:func:`write_index` rebuilds ``index.json`` from the registry and the JSON
files present.  :func:`fetch_locally_document` is the stub written when a
dataset has not been downloaded.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .lab_datasets import DATASETS, LabDataset

__all__ = ["COORD_DECIMALS", "WEIGHT_DECIMALS", "BASELINE_METHODS", "round_floats", "round_significant",
           "finalize", "strip_coordinates", "fetch_locally_document", "write_json", "write_index", "index_entry",
           "best_method", "timestamp"]

COORD_DECIMALS = 4
WEIGHT_DECIMALS = 6
#: diagnostics that are tiny by design: kept to 3 significant digits so "0.0" never hides "3.2e-07"
DIAGNOSTIC_SIGNIFICANT = 3
_DIAGNOSTIC_KEYS = ("permutation_max_delta", "max_abs_diff_vs_torch")
#: model inputs rounded like weights (6 decimals) although they sit outside the weight blocks
_SIX_DECIMAL_KEYS = ("input_scale",)
#: blocks that hold model weights (rounded to 6 decimals, not 4)
_WEIGHT_KEYS = ("engine", "probe")
#: methods that are baselines: never the headline of a dataset
BASELINE_METHODS = ("size_lr",)
#: blocks that carry coordinates, removed when ``shapes_shown`` is false
_COORDINATE_KEYS = ("pca", "examples", "example_indices", "neighbours", "engine")
_WORST_COORDINATE_KEYS = ("shape", "curve", "landmarks")
_METHOD_COORDINATE_KEYS = ("engine", "probe", "onnx_info")


def timestamp() -> str:
    """UTC time as ``2026-10-22T12:34:56Z``."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def round_floats(obj, decimals: int):
    """Recursively turn numpy into plain Python and round every float."""
    if isinstance(obj, dict):
        return {str(k): round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, decimals) for v in obj]
    if isinstance(obj, np.ndarray):
        return round_floats(obj.tolist(), decimals)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = round(float(obj), decimals)
        return 0.0 if v == 0 else v  # no "-0.0"
    return obj


def round_significant(x: float, digits: int = DIAGNOSTIC_SIGNIFICANT) -> float:
    """``float(f"{x:.3g}")``: a tiny diagnostic keeps its magnitude instead of rounding to 0.0."""
    v = float(x)
    return 0.0 if v == 0 else float(f"{v:.{digits}g}")


def _round_doc(doc: dict) -> dict:
    """Coordinates and statistics to 4 decimals, weights (engine, probe) and ``input_scale`` to 6, diagnostics to 3 s.f."""
    out = {}
    for key, value in doc.items():
        if key in _WEIGHT_KEYS:
            out[key] = round_floats(value, WEIGHT_DECIMALS)
        elif key in _SIX_DECIMAL_KEYS:
            out[key] = round_floats(value, WEIGHT_DECIMALS)
        elif key in _DIAGNOSTIC_KEYS and isinstance(value, (float, np.floating)):
            out[key] = round_significant(value)
        elif key == "onnx_info" and isinstance(value, dict):
            out[key] = _round_doc(value)
        elif key == "methods" and isinstance(value, dict):
            out[key] = {m: _round_doc(block) if isinstance(block, dict) else round_floats(block, COORD_DECIMALS)
                        for m, block in value.items()}
        elif key == "worst" and isinstance(value, list):
            # the worst mistakes' shapes are the engine's inputs for the page's self-check:
            # 4-decimal coordinates get amplified by the low-variance modes, so keep 6
            out[key] = [{k: round_floats(v, WEIGHT_DECIMALS if k == "shape" else COORD_DECIMALS)
                         for k, v in w.items()} if isinstance(w, dict) else round_floats(w, COORD_DECIMALS)
                        for w in value]
        else:
            out[key] = round_floats(value, COORD_DECIMALS)
    return out


def strip_coordinates(doc: dict) -> dict:
    """Remove every block that carries coordinates or weights (``shapes_shown: false``)."""
    doc = dict(doc)
    for key in _COORDINATE_KEYS:
        doc.pop(key, None)
    methods = {}
    for name, block in doc.get("methods", {}).items():
        if not isinstance(block, dict):
            methods[name] = block
            continue
        block = dict(block)
        for key in _METHOD_COORDINATE_KEYS:
            block.pop(key, None)
        if "onnx" in block:
            block["onnx"] = None
        if "worst" in block:
            block["worst"] = [{k: v for k, v in w.items() if k not in _WORST_COORDINATE_KEYS} for w in block["worst"]]
        methods[name] = block
    doc["methods"] = methods
    doc["shapes_shown"] = False
    doc["shapes_note"] = ("the licence of this dataset forbids redistributing derived data: no coordinates, "
                          "no shapes, no model weights are published; only counts and accuracies")
    return doc


def finalize(doc: dict, script: str, computed_at: str | None = None) -> dict:
    """Round, strip (when ``shapes_shown`` is false) and stamp a dataset document."""
    doc = dict(doc)
    doc["computed_at"] = computed_at or timestamp()
    doc["script"] = script
    if not doc.get("shapes_shown", True):
        doc = strip_coordinates(doc)
    return _round_doc(doc)


def fetch_locally_document(ds: LabDataset, script: str, reason: str, computed_at: str | None = None) -> dict:
    """The stub written when a dataset's files are absent."""
    return {
        "id": ds.id, "title": ds.title, "question": ds.question, "representation": ds.representation,
        "closed": bool(ds.closed), "n_points": None, "n": None,
        "status": "fetch_locally", "shapes_shown": bool(ds.shapes_shown),
        "source": {"name": ds.source.name, "citation": ds.source.citation, "url": ds.source.url,
                   "license": ds.source.license, "redistributed": bool(ds.source.redistributed)},
        "classes": [{"id": i, "name": name, "n": None} for i, name in enumerate(ds.classes)],
        "fetch_command": ds.fetch_command,
        "reason": reason,
        "computed_at": computed_at or timestamp(),
        "script": script,
        "methods": {},
    }


def write_json(path: Path, doc: dict) -> int:
    """Write compact JSON; returns the size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path.stat().st_size


def best_method(doc: dict) -> tuple[str | None, float | None, str | None]:
    """``(method, accuracy, protocol)`` of the best ready method, baselines (``size_lr``) excluded.

    Ties go to the first method in the document's order.  ``(None, None,
    None)`` when no method is ready.
    """
    best = None
    for name, m in doc.get("methods", {}).items():
        if name in BASELINE_METHODS or not isinstance(m, dict):
            continue
        acc = m.get("accuracy")
        if isinstance(acc, bool) or not isinstance(acc, (int, float)):
            continue
        if best is None or acc > best[1]:
            best = (name, float(acc), m.get("protocol"))
    return best if best is not None else (None, None, None)


def index_entry(ds: LabDataset, doc: dict | None) -> dict:
    """One row of ``index.json`` for a registry entry and its JSON (``None`` → proposed)."""
    if doc is None:
        return {"id": ds.id, "title": ds.title, "status": "proposed", "representation": ds.representation,
                "n": None, "classes": list(ds.classes), "best_accuracy": None, "best_method": None,
                "best_protocol": None, "chance": None, "license": ds.source.license,
                "shapes_shown": bool(ds.shapes_shown)}
    name, acc, protocol = best_method(doc)
    return {
        "id": ds.id, "title": doc.get("title", ds.title), "status": doc.get("status", "proposed"),
        "representation": doc.get("representation", ds.representation), "n": doc.get("n"),
        "classes": [c["name"] for c in doc.get("classes", [])],
        "best_accuracy": acc, "best_method": name, "best_protocol": protocol, "chance": doc.get("chance"),
        "license": doc.get("source", {}).get("license", ds.source.license),
        "shapes_shown": bool(doc.get("shapes_shown", ds.shapes_shown)),
        "methods": {m: (b.get("status", "ready") if isinstance(b, dict) else "ready")
                    for m, b in doc.get("methods", {}).items()},
        "computed_at": doc.get("computed_at"),
    }


def write_index(data_dir: Path, registry: dict[str, LabDataset] | None = None) -> list[dict]:
    """Rebuild ``index.json`` from the registry order and whatever ``<id>.json`` files exist."""
    registry = DATASETS if registry is None else registry
    data_dir = Path(data_dir)
    rows = []
    for ds in registry.values():
        path = data_dir / f"{ds.id}.json"
        doc = None
        if path.exists():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                doc = None
        rows.append(index_entry(ds, doc))
    rows = round_floats(rows, COORD_DECIMALS)
    write_json(data_dir / "index.json", rows)
    return rows
