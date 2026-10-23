"""The files the published site serves must be identical to data/, and nothing restricted may be among them.

GitHub Pages serves only docs/, so the Pyodide panel of the Phenotype Lab reads
datasets from docs/lab/data/files/.  ``scripts/run_lab.py`` copies every
top-level file of data/ there; this test keeps the copies from drifting, and
makes sure the licence-restricted red-cell data (data/_local/) never lands
anywhere under docs/lab/: not as a data file, not as an ONNX model, not as
coordinates inside a JSON.
"""

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SERVED = ROOT / "docs" / "lab"
RESTRICTED_NAME_PARTS = ("rbc", "erythrocyte")
COORDINATE_KEYS = ("pca", "examples", "example_indices", "neighbours", "engine", "probe", "onnx_info", "shape",
                   "curve", "mean", "modes", "scatter", "components")


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_site_copies_match_data():
    sources = sorted(p for p in DATA.iterdir() if p.is_file())
    assert sources, "data/ is empty?"
    for src in sources:
        copy = SERVED / "data" / "files" / src.name
        assert copy.exists(), f"missing site copy {copy} (run scripts/run_lab.py)"
        assert _sha(src) == _sha(copy), f"{src.name} differs between data/ and docs/lab/data/files/"


def test_no_restricted_data_in_site():
    names = [p.name.lower() for p in SERVED.rglob("*") if p.is_file()]
    offenders = [n for n in names if any(part in n for part in RESTRICTED_NAME_PARTS)
                 and not n.endswith(".json")]
    assert not offenders, f"licence-restricted artefacts under docs/lab/: {offenders}"
    assert not (SERVED / "models" / "rbc.onnx").exists()


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_restricted_json_carries_only_numbers():
    for path in (SERVED / "data").glob("*.json"):
        if path.stem == "index":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("shapes_shown", True):
            continue
        keys = set(_keys(doc))
        leaked = keys & set(COORDINATE_KEYS)
        assert not leaked, f"{path.name} (shapes_shown false) carries {sorted(leaked)}"
        for name, block in doc.get("methods", {}).items():
            if isinstance(block, dict):
                assert block.get("onnx") is None, name
