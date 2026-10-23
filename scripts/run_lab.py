"""Compute everything the Phenotype Lab (chapter 10) shows.

For every dataset of the registry (``shapekit.lab_datasets``), or the one
named with ``--dataset``, this script runs the method families of
LAB_SPEC.md and writes

- ``docs/lab/data/<id>.json``      the JSON contract (numbers, shapes, engine)
- ``docs/lab/models/<id>.onnx``    the trained PointNet (embedding only)
- ``docs/lab/data/index.json``     one row per dataset
- ``docs/lab/data/files/``         byte-identical copies of the redistributable
                                   ``data/`` files, for the page's Pyodide panel

Nothing on the page is typed in by hand: every number comes from here.

Determinism.  The classical methods are deterministic for a given seed
(default 0) on any machine.  The PointNet results are deterministic for a
given seed **and** torch thread count (``--threads``, default 4), torch build
and CPU: a different thread count changes the floating-point reduction order
and with it the trained weights, the chosen epoch and the ONNX file.  The
JSON records ``seed``, ``timing.threads`` and the exact command in ``script``;
the wall-clock ``timing`` block is the only part that is never reproducible.

Guards.  A dataset whose files are not downloaded gets a stub with
``"status": "fetch_locally"``, but an existing ``"status": "ready"`` JSON is
never replaced by a stub unless ``--force`` is given (rbc can only be computed
by someone who fetched 594 MB; a plain run on a fresh clone must not destroy
its results).  ``--skip-pointnet`` and ``--epochs`` produce partial or
non-canonical results, so they refuse to write into the default output
folder unless ``--force`` is given; use ``--out DIR``.  A failed engine
self-check (the page's own check, |dp| > 1e-3) counts as a failed dataset
and the script exits non-zero.

Run time: 8.7 min measured for all six datasets on an idle 4-CPU machine
(PointNet: celegans 19 s, mef 87 s, leaves 158 s, rbc 257 s; the classical
methods take seconds); ``--skip-pointnet`` runs in a few seconds.

Run from the repository root::

    python scripts/run_lab.py                       # every dataset available
    python scripts/run_lab.py --dataset celegans    # one dataset
    python scripts/run_lab.py --skip-pointnet --out /tmp/lab   # classical methods only (fast)
    python scripts/run_lab.py --epochs 5 --out /tmp/lab        # override the PointNet epochs
    python scripts/run_lab.py --force               # also overwrite ready JSON with stubs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapekit.datasets import resolve_data_dir  # noqa: E402
from shapekit.lab_datasets import DATASETS, DatasetNotFetched, get  # noqa: E402
from shapekit.lab_engine import classify  # noqa: E402
from shapekit.lab_export import fetch_locally_document, finalize, timestamp, write_index, write_json  # noqa: E402
from shapekit.lab_pipeline import run_dataset  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "lab"
DEFAULT_THREADS = 4
SELF_CHECK_TOLERANCE = 1e-3
#: file names that must never be copied to the site (licence-restricted data lives in data/_local/ anyway)
RESTRICTED_NAME_PARTS = ("rbc", "erythrocyte")


class SelfCheckError(AssertionError):
    """The written JSON's engine does not reproduce its own ``worst[].p_engine``."""


def script_command(dataset_id: str, args: argparse.Namespace) -> str:
    """The exact command that reproduces one dataset's file (recorded in the JSON's ``script``)."""
    parts = [f"scripts/run_lab.py --dataset {dataset_id}"]
    if args.skip_pointnet:
        parts.append("--skip-pointnet")
    if args.epochs is not None:
        parts.append(f"--epochs {args.epochs}")
    if args.seed != 0:
        parts.append(f"--seed {args.seed}")
    if args.threads != DEFAULT_THREADS:
        parts.append(f"--threads {args.threads}")
    if args.force:
        parts.append("--force")
    return " ".join(parts)


def self_check(path: Path, tolerance: float = SELF_CHECK_TOLERANCE) -> float | None:
    """Re-read a written JSON and run the engine on its worst mistakes (what the page does on load).

    Returns the largest |classify(shape) - p_engine| over the classical
    methods, or ``None`` when there is nothing to check.  Raises
    :class:`SelfCheckError` above ``tolerance``.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    worst_delta = None
    for name, block in doc.get("methods", {}).items():
        if not isinstance(block, dict) or "engine" not in block:
            continue
        for w in block.get("worst", []):
            if "shape" not in w:
                continue
            delta = float(np.abs(classify(np.asarray(w["shape"]), block["engine"]) - np.asarray(w["p_engine"])).max())
            worst_delta = delta if worst_delta is None else max(worst_delta, delta)
    if worst_delta is not None and worst_delta > tolerance:
        raise SelfCheckError(f"engine self-check off by {worst_delta:.2e} (> {tolerance}) in {path.name}")
    return worst_delta


def existing_status(path: Path) -> str | None:
    """``status`` of a JSON already on disk, or ``None``."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (ValueError, AttributeError):
        return None


def copy_data_files(out: Path, data_dir: Path | None = None, log=print) -> list[str]:
    """Copy the redistributable ``data/`` files (top level only, never ``_local/``) to ``<out>/data/files/``.

    The page's Pyodide panel reads them from there, since GitHub Pages serves
    only ``docs/``.  Names that look like licence-restricted data are refused.
    """
    src_dir = Path(data_dir) if data_dir is not None else Path(resolve_data_dir())
    dest_dir = out / "data" / "files"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(p for p in src_dir.iterdir() if p.is_file()):
        if any(part in src.name.lower() for part in RESTRICTED_NAME_PARTS):
            raise RuntimeError(f"refusing to copy {src.name} to the site: it looks like licence-restricted data")
        dest = dest_dir / src.name
        if not dest.exists() or dest.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dest)
        copied.append(src.name)
    log(f"copied {len(copied)} data files to {dest_dir}")
    return copied


def run_one(dataset_id: str, out: Path, args: argparse.Namespace, log) -> dict:
    """Compute one dataset and write its JSON (and ONNX); returns a summary row."""
    ds = get(dataset_id)
    data_dir = out / "data"
    models_dir = out / "models"
    path = data_dir / f"{ds.id}.json"
    script = script_command(ds.id, args)
    computed_at = timestamp()
    t0 = time.perf_counter()
    log(f"{ds.id}: {ds.title}")
    try:
        points, labels = ds.load()
    except DatasetNotFetched as exc:
        if existing_status(path) == "ready" and not args.force:
            log(f"  not fetched here: {exc}")
            log(f"  kept the existing {path} (status ready); pass --force to replace it with a stub")
            return {"id": ds.id, "status": "kept", "bytes": path.stat().st_size}
        doc = fetch_locally_document(ds, script, str(exc), computed_at)
        size = write_json(path, doc)
        log(f"  not fetched: {exc}")
        log(f"  wrote {path} ({size} bytes, status fetch_locally)")
        return {"id": ds.id, "status": "fetch_locally", "bytes": size}
    log(f"  loaded points {points.shape}, labels {np.bincount(labels).tolist()}")
    doc = run_dataset(ds, points, labels, seed=args.seed, skip_pointnet=args.skip_pointnet, epochs=args.epochs,
                      onnx_dir=models_dir, threads=args.threads, log=log)
    doc = finalize(doc, script, computed_at)
    size = write_json(path, doc)
    delta = self_check(path)   # raises SelfCheckError -> the dataset counts as failed
    seconds = time.perf_counter() - t0
    log(f"  wrote {path} ({size / 1024:.0f} KB) in {seconds:.0f} s"
        + (f"; engine self-check max |dp| = {delta:.1e}" if delta is not None else ""))
    row = {"id": ds.id, "status": "ready", "n": doc["n"], "chance": doc["chance"], "bytes": size,
           "seconds": seconds, "self_check": delta}
    for name, block in doc["methods"].items():
        if isinstance(block, dict) and "accuracy" in block:
            row[name] = block["accuracy"]
            row[name + "_sd"] = block.get("accuracy_sd")
        else:
            row[name] = block.get("status") if isinstance(block, dict) else None
    pn = doc["methods"].get("pointnet_probe", {})
    if isinstance(pn, dict) and "accuracy" in pn:
        row["pointnet_seconds"] = doc.get("timing", {}).get("seconds", {}).get("pointnet_probe")
        row["pointnet_epochs"] = pn["epochs"]
        row["pointnet_untrained"] = pn["untrained"]
        row["pointnet_select"] = pn["accuracy_select"]
        row["onnx"] = pn.get("onnx")
    return row


def print_table(rows: list[dict]) -> None:
    cols = [("dataset", "id", "{:<10}"), ("status", "status", "{:<14}"), ("n", "n", "{:>5}"),
            ("chance", "chance", "{:>7}"), ("landmarks_lr", "landmarks_lr", "{:>14}"),
            ("spline_lr", "spline_lr", "{:>14}"), ("size_lr", "size_lr", "{:>14}"),
            ("pointnet", "pointnet_probe", "{:>9}"), ("select", "pointnet_select", "{:>7}"),
            ("untrained", "pointnet_untrained", "{:>9}"), ("epochs", "pointnet_epochs", "{:>6}"),
            ("pn s", "pointnet_seconds", "{:>6}")]
    print()
    print("  ".join(fmt.format(title) for title, _, fmt in cols))
    for row in rows:
        cells = []
        for _, key, fmt in cols:
            v = row.get(key)
            if v is None:
                s = "-"
            elif isinstance(v, float):
                if key in ("landmarks_lr", "spline_lr", "size_lr") and isinstance(row.get(key + "_sd"), float):
                    s = f"{v:.3f} ± {row[key + '_sd']:.3f}"   # sd over the CV folds (ddof=0)
                elif key == "pointnet_seconds":
                    s = f"{v:.0f}"
                else:
                    s = f"{v:.3f}"
            else:
                s = str(v)
            cells.append(fmt.format(s))
        print("  ".join(cells))
    print("  classical: pooled 5-fold CV ± sd over folds; pointnet: 25 % hold-out (select = the 20 % slice that "
          "chose the epoch); size_lr is a baseline, never a headline")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), metavar="ID",
                        help="one dataset (repeatable); default: all of " + ", ".join(DATASETS))
    parser.add_argument("--skip-pointnet", action="store_true", help="do not train PointNet (needs --out or --force)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the per-dataset PointNet epochs (needs --out or --force)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"torch CPU threads (default {DEFAULT_THREADS}; another value changes the PointNet results)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default docs/lab)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a ready JSON with a stub, and allow partial runs into the default folder")
    args = parser.parse_args(argv)

    partial = args.skip_pointnet or args.epochs is not None
    if partial and args.out.resolve() == DEFAULT_OUT.resolve() and not args.force:
        parser.error("--skip-pointnet / --epochs produce partial results: write them with --out DIR, "
                     "or pass --force to overwrite docs/lab anyway")

    t0 = time.perf_counter()
    log = lambda msg: print(f"[{time.perf_counter() - t0:6.1f} s] {msg}", flush=True)  # noqa: E731
    if args.threads != DEFAULT_THREADS:
        log(f"WARNING: --threads {args.threads} (default {DEFAULT_THREADS}): the PointNet results will differ "
            f"from the shipped ones; the JSON records the thread count")
    ids = args.dataset or list(DATASETS)
    rows, failed = [], []
    for dataset_id in ids:
        try:
            rows.append(run_one(dataset_id, args.out, args, log))
        except Exception as exc:  # keep going: one broken dataset must not lose the others
            traceback.print_exc()
            log(f"  FAILED {dataset_id}: {type(exc).__name__}: {exc}")
            failed.append(dataset_id)
            rows.append({"id": dataset_id, "status": f"error: {type(exc).__name__}"})
    index = write_index(args.out / "data")
    log(f"wrote {args.out / 'data' / 'index.json'} ({len(index)} datasets)")
    try:
        copy_data_files(args.out, log=log)
    except Exception as exc:  # noqa: BLE001
        log(f"  FAILED copying data files: {exc}")
        failed.append("data/files")
    print_table(rows)
    log(f"done in {(time.perf_counter() - t0) / 60:.1f} min" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
