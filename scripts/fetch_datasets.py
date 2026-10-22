#!/usr/bin/env python
"""Fetch the Phenotype Lab datasets that are not typed into the repository.

Every array this script writes is traced back to a download whose size,
MD5 and SHA-256 are printed, and every dataset gets a notice file next to
it saying where it came from and under which licence.  Nothing from the
downloads is ever executed: tarballs, zips, CSV and ``.rda`` files are
read as data with plain Python readers, and the download folders are
never put on ``sys.path`` (the script is safe under ``python -I``).

Run from the repository root::

    python scripts/fetch_datasets.py --brain --gorilla --leaves      # redistributable
    python scripts/fetch_datasets.py --rbc                           # local only
    python scripts/fetch_datasets.py --all

What each flag produces (see LAB_SPEC.md and ATTRIBUTION.md):

``--brain``
    ``data/brain_landmarks.npz`` (points ``(28, 13, 2)``, labels 0 = control
    for subjects 1-14, 1 = schizophrenia for 15-28) from ``schizophrenia.dat``
    of the CRAN package ``shapes`` (GPL-2), plus ``data/brain_landmarks.LICENSE``
    and ``LICENSES/GPL-2.txt``.
``--gorilla``
    ``data/gorilla_landmarks.npz`` (points ``(59, 8, 2)``, labels 0 = female
    for the 30 ``gorf.dat`` skulls, 1 = male for the 29 ``gorm.dat`` skulls)
    from the same package, plus ``data/gorilla_landmarks.LICENSE``.
``--leaves``
    ``data/leaves_outlines.npz`` (points ``(440, 150, 2)``, labels 0-14,
    ``classes`` = 15 species names) from the figshare CSV "Outlines of Swedish
    leaves" (CC BY 4.0), plus ``data/leaves_outlines.ATTRIBUTION.md``.
``--rbc``
    **Local only.**  erythrocytesIDB v2 (Zenodo 18299474) is licensed
    CC BY-NC-ND 4.0: no derived data may be redistributed.  The 594 MB zip is
    downloaded to ``data/_local/`` (git-ignored), unpacked there, and one
    closed outline per cell is traced from the per-class field masks of
    erythrocytesIDB2 and erythrocytesIDB3 into ``data/_local/rbc_outlines.npz``
    (labels 0 = circular, 1 = elongated, 2 = other).  Never commit or publish
    anything under ``data/_local/``.

Options: ``--download-dir`` (where the small downloads are kept; default
``data/_local/downloads``), ``--skip-download`` (reuse files that are already
there without re-checking them against the network), ``--n-points`` (points per
resampled outline, default 150).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapekit.contours import mask_to_contours, resample_by_arclength  # noqa: E402

DATA = ROOT / "data"
LOCAL = DATA / "_local"
LICENSES = ROOT / "LICENSES"
SCRIPT = "scripts/fetch_datasets.py"

# ---------------------------------------------------------------- sources
SHAPES_VERSION = "1.2.8"
SHAPES_URLS = (
    f"https://cran.r-project.org/src/contrib/shapes_{SHAPES_VERSION}.tar.gz",
    f"https://cran.r-project.org/src/contrib/Archive/shapes/shapes_{SHAPES_VERSION}.tar.gz",
)
SHAPES_MD5 = "9e81c00a7ad99c973e2eae37cc6cce63"   # as downloaded from CRAN on 2026-10-22
GPL2_URL = "https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt"

LEAVES_ARTICLE = "https://api.figshare.com/v2/articles/14551005"
LEAVES_FILE_NAME = "Contours_SwedishLeaves.csv"
LEAVES_URL_FALLBACK = "https://ndownloader.figshare.com/files/27917577"
LEAVES_MD5 = "c7984a894d3cbc017805c6ec27a60f5d"   # figshare's computed_md5 for the file

# CC BY-NC-ND 4.0 -- LOCAL USE ONLY.  See the notice printed by fetch_rbc().
RBC_RECORD = "https://zenodo.org/api/records/18299474"
RBC_URL = "https://zenodo.org/api/records/18299474/files/Version%202,%20erythrocytesIDB%202017.zip/content"
RBC_MD5 = "9d480fe1763f6ddb2b8bb4983cdb51e7"      # from the Zenodo record's file metadata
RBC_ZIP = LOCAL / "erythrocytesIDB_v2.zip"
RBC_DIR = LOCAL / "erythrocytesIDB_v2"
RBC_NPZ = LOCAL / "rbc_outlines.npz"
RBC_CLASSES = ("circular", "elongated", "other")
RBC_NOTICE = """
================================================================================
  erythrocytesIDB v2 is licensed CC BY-NC-ND 4.0 (no derivatives, no commercial
  use).  Everything written under data/_local/ -- the zip, the unpacked images,
  rbc_outlines.npz and anything computed from them that contains coordinates --
  is for local use only.  data/_local/ is git-ignored: NEVER commit, publish or
  redistribute any of it.  The lab page shows only accuracies and counts.
================================================================================
"""

# RBC outline extraction parameters (documented here and in ATTRIBUTION.md).
RBC_THRESHOLD = 127          # the masks are JPEGs: pixel > 127 is "cell"
RBC_MIN_ABS_AREA = 30        # pixels; smaller blobs are JPEG debris, dropped before the median is taken
RBC_AREA_RANGE = (0.3, 2.5)  # keep components with area in [0.3, 2.5] x median area of their database
RBC_MIN_CONTOUR = 20         # drop components whose traced outline has fewer points than this


# ---------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def checksums(path: Path) -> tuple[str, str, int]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha.hexdigest(), size


def report(path: Path, expected_md5: str | None = None) -> tuple[str, str]:
    md5, sha, size = checksums(path)
    status = ""
    if expected_md5:
        status = "  (md5 matches)" if md5 == expected_md5 else f"  (md5 MISMATCH, expected {expected_md5})"
    print(f"  {path.name}: {size:,} bytes\n    md5    {md5}{status}\n    sha256 {sha}")
    if expected_md5 and md5 != expected_md5:
        raise SystemExit(f"checksum mismatch for {path}; delete it and re-run, or update the pinned md5")
    return md5, sha


def _open(url: str, headers: dict | None = None, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": "shapekit-fetch/0.1", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, dest: Path, *, md5: str | None = None, skip: bool = False,
             resume: bool = False, attempts: int = 6) -> Path:
    """Download ``url`` to ``dest`` (via ``dest.part``), following redirects.

    - a complete ``dest`` that matches ``md5`` (or exists, with ``skip``) is reused;
    - ``resume=True`` continues a partial ``dest.part`` with an HTTP Range request
      (for the large zip);
    - HTTP 202 (figshare "preparing the file") and transient errors are retried.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if skip:
            print(f"  reusing {dest} (--skip-download)")
            return dest
        if md5 and checksums(dest)[0] == md5:
            print(f"  reusing {dest} (md5 verified)")
            return dest
        if not md5:
            print(f"  reusing {dest} (already present; no pinned checksum)")
            return dest
        print(f"  {dest} exists but its md5 differs from the pinned value; downloading again")
        dest.unlink()
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if (resume and part.exists()) else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with _open(url, headers) as r:
                status = getattr(r, "status", 200)
                if status == 202:
                    raise urllib.error.HTTPError(url, 202, "not ready", r.headers, None)
                if offset and status != 206:        # server ignored the Range header: start over
                    offset = 0
                total = r.headers.get("Content-Length")
                total = int(total) + offset if total else None
                mode = "ab" if offset else "wb"
                done = offset
                last = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 5:
                            pct = f" ({100 * done / total:.0f}%)" if total else ""
                            print(f"    ... {done / 1e6:,.0f} MB{pct}", flush=True)
                            last = time.time()
            if total and done != total:
                raise OSError(f"short read: {done} of {total} bytes")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ConnectionError) as e:
            code = getattr(e, "code", None)
            if attempt == attempts:
                raise
            wait = 5 * attempt if code == 202 else 3 * attempt
            print(f"    attempt {attempt} failed ({e}); retrying in {wait} s", flush=True)
            if not resume and part.exists():
                part.unlink()
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_first(urls, dest: Path, **kw) -> Path:
    last: Exception | None = None
    for url in urls:
        try:
            return download(url, dest, **kw)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last = e
            print(f"    {url} failed: {e}")
    raise SystemExit(f"could not download {dest.name}: {last}")


def safe_extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Unpack a zip as data, refusing paths that escape ``out_dir``; skips macOS junk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store") or info.is_dir():
                continue
            target = (out_dir / name).resolve()
            if out_dir.resolve() not in target.parents:
                raise SystemExit(f"refusing to extract {name!r}: path escapes {out_dir}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def read_rda_array(rda_path: Path, name: str) -> np.ndarray:
    """Read one numeric array object from an R ``.rda`` file as data (no R, no code)."""
    try:
        import pyreadr  # type: ignore
        try:
            obj = pyreadr.read_r(str(rda_path))[name]
            return np.asarray(getattr(obj, "values", obj), dtype=float)
        except Exception as e:  # pyreadr needs xarray for >2-D arrays; fall through to rdata
            first_error: Exception | None = e
    except ImportError as e:
        first_error = e
    try:
        import rdata  # type: ignore
    except ImportError:
        raise SystemExit(
            f"cannot read {rda_path.name}: install 'pyreadr' and 'xarray' (or 'rdata'); "
            f"last error: {first_error}"
        ) from None
    obj = rdata.read_rda(str(rda_path))[name]
    return np.asarray(getattr(obj, "values", obj), dtype=float)


def read_rda_list(rda_path: Path, name: str):
    """Read an R list (e.g. ``schizophrenia`` = list(x, group)) with ``rdata``; None if unavailable."""
    try:
        import rdata  # type: ignore
    except ImportError:
        return None
    return rdata.read_rda(str(rda_path))[name]


def gpl2_notice(dataset_file: str, flag: str, what: str, tarball: Path, md5: str, sha: str) -> str:
    return f"""Notice for data/{dataset_file}
(written by {SCRIPT} {flag} on {now_iso()})

{what}

The arrays were extracted from the source tarball of the R package "shapes"
version {SHAPES_VERSION} by Ian L. Dryden (https://cran.r-project.org/package=shapes,
https://github.com/iandryden/shapes), which is distributed under the GNU
General Public License version 2 (GPL-2).  They are redistributed here under
the same licence, GPL-2, with this notice.  The full licence text is in
LICENSES/GPL-2.txt and at https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.
The GPL-2 applies to this data file; the rest of this repository is MIT-licensed
(see LICENSE).

Source tarball: {tarball.name}
  md5    {md5}
  sha256 {sha}
Reference: Dryden, I. L. & Mardia, K. V. (2016). Statistical Shape Analysis,
with Applications in R (2nd ed.). Wiley.
"""


# ---------------------------------------------------------------- shapes package (brain, gorilla)
def fetch_shapes_tarball(download_dir: Path, skip: bool) -> tuple[Path, str, str]:
    print(f"[shapes] CRAN package shapes {SHAPES_VERSION} (GPL-2)")
    tarball = download_first(SHAPES_URLS, download_dir / f"shapes_{SHAPES_VERSION}.tar.gz", md5=SHAPES_MD5, skip=skip)
    md5, sha = report(tarball, SHAPES_MD5)
    return tarball, md5, sha


def extract_rda(tarball: Path, member: str, out_dir: Path) -> Path:
    """Pull one ``data/*.rda`` member out of the tarball as a plain file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
        if member not in names:
            listing = sorted(n for n in names if "/data/" in n)
            raise SystemExit(f"{member} not in {tarball.name}; data members: {listing}")
        f = tar.extractfile(member)
        if f is None:
            raise SystemExit(f"{member} is not a regular file")
        dest = out_dir / Path(member).name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f, out)
    return dest


def ensure_gpl2(download_dir: Path, skip: bool) -> Path:
    LICENSES.mkdir(exist_ok=True)
    target = LICENSES / "GPL-2.txt"
    if target.exists():
        print(f"  {target.relative_to(ROOT)} already present")
        return target
    src = download(GPL2_URL, download_dir / "gpl-2.0.txt", skip=skip)
    head = src.read_text(encoding="utf-8", errors="replace")[:200]
    if "GNU GENERAL PUBLIC LICENSE" not in head or "Version 2, June 1991" not in head:
        raise SystemExit(f"{src} does not look like the GPL-2 text; refusing to install it")
    report(src)
    shutil.copyfile(src, target)
    print(f"  wrote {target.relative_to(ROOT)}")
    return target


def fetch_brain(download_dir: Path, skip: bool) -> None:
    tarball, md5, sha = fetch_shapes_tarball(download_dir, skip)
    rda = extract_rda(tarball, "shapes/data/schizophrenia.dat.rda", download_dir / "shapes_data")
    arr = read_rda_array(rda, "schizophrenia.dat")          # R array 13 x 2 x 28
    if arr.shape != (13, 2, 28):
        raise SystemExit(f"schizophrenia.dat has shape {arr.shape}, expected (13, 2, 28)")
    points = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float64)   # (28, 13, 2)
    labels = np.array([0] * 14 + [1] * 14, dtype=np.int64)
    # Cross-check the group factor carried by the list object `schizophrenia` (x, group).
    verified = "labels from the package documentation (first 14 = con, last 14 = scz)"
    try:
        lst = read_rda_list(extract_rda(tarball, "shapes/data/schizophrenia.rda", download_dir / "shapes_data"),
                            "schizophrenia")
    except SystemExit:
        lst = None
    if lst is not None:
        group = [str(g) for g in lst["group"]]
        x = np.asarray(getattr(lst["x"], "values", lst["x"]), dtype=float)
        if group != ["con"] * 14 + ["scz"] * 14 or not np.array_equal(x, arr):
            raise SystemExit(f"schizophrenia$group / $x disagree with schizophrenia.dat: {Counter(group)}")
        verified = "labels verified against schizophrenia$group (con x 14, scz x 14); schizophrenia$x == schizophrenia.dat"
    out = DATA / "brain_landmarks.npz"
    np.savez(out, points=points, labels=labels, classes=np.array(["control", "schizophrenia"]),
             landmark_names=np.array([
                 "splenium", "genu", "top of corpus callosum", "top of head", "tentorium of cerebellum at dura",
                 "top of cerebellum", "tip of fourth ventricle", "bottom of cerebellum", "top of pons",
                 "bottom of pons", "optic chiasm", "frontal pole", "superior colliculus"]))
    print(f"  wrote {out.relative_to(ROOT)}: points {points.shape}, labels {dict(Counter(labels.tolist()))} "
          f"(0 = control, 1 = schizophrenia); 13 landmarks x 28 subjects as documented; {verified}")
    what = ("The file holds the `schizophrenia.dat` array of the package (Bookstein's schizophrenia\n"
            "data: 13 near-midsagittal brain landmarks from MR images of 28 subjects, data kindly\n"
            "provided by Fred Bookstein; source: Bookstein, F. L. (1996) Biometrics, biomathematics\n"
            "and the morphometric synthesis, Bulletin of Mathematical Biology 58:313-365).  R's\n"
            "13 x 2 x 28 array was rearranged to points (28, 13, 2) float64 in the package's\n"
            "subject and landmark order, with labels 0 = control (subjects 1-14) and\n"
            "1 = schizophrenia (subjects 15-28) as the package documents.  Values are unchanged.")
    (DATA / "brain_landmarks.LICENSE").write_text(
        gpl2_notice("brain_landmarks.npz", "--brain", what, tarball, md5, sha), encoding="utf-8")
    print("  wrote data/brain_landmarks.LICENSE (GPL-2 notice)")
    ensure_gpl2(download_dir, skip)


def fetch_gorilla(download_dir: Path, skip: bool) -> None:
    tarball, md5, sha = fetch_shapes_tarball(download_dir, skip)
    f = read_rda_array(extract_rda(tarball, "shapes/data/gorf.dat.rda", download_dir / "shapes_data"), "gorf.dat")
    m = read_rda_array(extract_rda(tarball, "shapes/data/gorm.dat.rda", download_dir / "shapes_data"), "gorm.dat")
    if f.shape != (8, 2, 30) or m.shape != (8, 2, 29):
        raise SystemExit(f"gorf.dat {f.shape} / gorm.dat {m.shape}: expected (8, 2, 30) and (8, 2, 29)")
    points = np.concatenate([f.transpose(2, 0, 1), m.transpose(2, 0, 1)]).astype(np.float64)   # (59, 8, 2)
    labels = np.array([0] * 30 + [1] * 29, dtype=np.int64)
    out = DATA / "gorilla_landmarks.npz"
    np.savez(out, points=np.ascontiguousarray(points), labels=labels, classes=np.array(["female", "male"]))
    print(f"  wrote {out.relative_to(ROOT)}: points {points.shape}, labels {dict(Counter(labels.tolist()))} "
          f"(0 = female from gorf.dat, 1 = male from gorm.dat); 8 landmarks, 30 + 29 skulls as documented")
    what = ("The file holds the `gorf.dat` (30 female) and `gorm.dat` (29 male) gorilla skull\n"
            "landmark arrays of the package (8 landmarks in 2-D per skull; data from Paul O'Higgins,\n"
            "Hull-York Medical School; source: O'Higgins, P. & Dryden, I. L. (1993) Sexual dimorphism\n"
            "in hominoids: further studies of craniofacial shape differences in Pan, Gorilla, Pongo,\n"
            "Journal of Human Evolution 24:183-205).  R's 8 x 2 x n arrays were rearranged to\n"
            "points (59, 8, 2) float64 (the 30 females first, then the 29 males, each in the\n"
            "package's order) with labels 0 = female, 1 = male.  Values are unchanged.")
    (DATA / "gorilla_landmarks.LICENSE").write_text(
        gpl2_notice("gorilla_landmarks.npz", "--gorilla", what, tarball, md5, sha), encoding="utf-8")
    print("  wrote data/gorilla_landmarks.LICENSE (GPL-2 notice)")
    ensure_gpl2(download_dir, skip)


# ---------------------------------------------------------------- leaves
def leaves_download_url() -> tuple[str, dict]:
    """Ask the figshare API for the file's real download URL and the article metadata."""
    import json
    try:
        with _open(LEAVES_ARTICLE) as r:
            meta = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  figshare API unavailable ({e}); using the recorded download URL")
        return LEAVES_URL_FALLBACK, {}
    for f in meta.get("files", []):
        if f.get("name") == LEAVES_FILE_NAME:
            return f["download_url"], meta
    raise SystemExit(f"{LEAVES_FILE_NAME} not listed in {LEAVES_ARTICLE}: {[f.get('name') for f in meta.get('files', [])]}")


def fetch_leaves(download_dir: Path, skip: bool, n_points: int) -> None:
    print("[leaves] Outlines of Swedish leaves, figshare 14551005 (CC BY 4.0)")
    url, meta = leaves_download_url()
    lic = (meta.get("license") or {}).get("name", "CC BY 4.0 (per the figshare record)")
    authors = ", ".join(a.get("full_name", "") for a in meta.get("authors", [])) or "Armand Leroi, Stephen Marsland, Arianna Salili-James"
    doi = meta.get("doi", "10.6084/m9.figshare.14551005")
    print(f"  licence (figshare API): {lic}; authors: {authors}; DOI: {doi}")
    csv_path = download(url, download_dir / LEAVES_FILE_NAME, md5=LEAVES_MD5, skip=skip)
    md5, sha = report(csv_path, LEAVES_MD5)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    print(f"  header: {header}; rows: {len(rows):,}")
    for r in rows[:3]:
        print(f"    {dict(r)}")
    needed = {"indexNumber", "indexName", "speciesNumber", "speciesName", "X", "Y", "pointOrder"}
    if not needed <= set(header or []):
        raise SystemExit(f"unexpected CSV columns {header}; the loader below assumes {sorted(needed)}")

    outlines: "OrderedDict[int, dict]" = OrderedDict()
    for r in rows:
        o = outlines.setdefault(int(r["indexNumber"]), {
            "name": r["indexName"], "species": int(r["speciesNumber"]), "species_name": r["speciesName"].strip(), "pts": []})
        if o["species"] != int(r["speciesNumber"]) or o["name"] != r["indexName"]:
            raise SystemExit(f"outline {r['indexNumber']} has inconsistent labels")
        o["pts"].append((int(r["pointOrder"]), float(r["X"]), float(r["Y"])))

    species = {}
    for o in outlines.values():
        species.setdefault(o["species"], o["species_name"])
        if species[o["species"]] != o["species_name"]:
            raise SystemExit(f"species number {o['species']} maps to two names")
    numbers = sorted(species)
    if numbers != list(range(1, len(numbers) + 1)):
        raise SystemExit(f"species numbers are not 1..K: {numbers}")
    classes = np.array([species[k] for k in numbers])

    raw_counts = Counter(len(o["pts"]) for o in outlines.values())
    pts_list, labels, names, closing_dup = [], [], [], 0
    for o in outlines.values():
        pts = np.array([(x, y) for _, x, y in sorted(o["pts"])], dtype=float)
        if np.allclose(pts[0], pts[-1]):            # the file repeats the first point at the end
            pts = pts[:-1]
            closing_dup += 1
        pts_list.append(resample_by_arclength(pts, n_points, closed=True))
        labels.append(o["species"] - 1)
        names.append(o["name"])
    points = np.stack(pts_list).astype(np.float64)
    labels_arr = np.array(labels, dtype=np.int64)
    counts = Counter(labels_arr.tolist())
    out = DATA / "leaves_outlines.npz"
    np.savez(out, points=points, labels=labels_arr, classes=classes, names=np.array(names),
             species_number=np.array([n + 1 for n in labels], dtype=np.int64))
    print(f"  wrote {out.relative_to(ROOT)}: points {points.shape}, {len(classes)} species; "
          f"raw points per outline {dict(raw_counts)}, closing point repeated in {closing_dup} outlines, "
          f"resampled to {n_points} by arc length")
    for i, c in enumerate(classes):
        print(f"    {i:2d} {c:26s} {counts.get(i, 0)}")

    lines = "\n".join(f"| {i} | {c} | {counts.get(i, 0)} |" for i, c in enumerate(classes))
    raw_desc = ", ".join(f"{n} points x {k} outlines" for n, k in sorted(raw_counts.items()))
    (DATA / "leaves_outlines.ATTRIBUTION.md").write_text(f"""# data/leaves_outlines.npz

Written by `{SCRIPT} --leaves` on {now_iso()}.

## Source

- **Outlines of Swedish leaves**, {authors}. figshare, DOI {doi},
  <https://figshare.com/articles/dataset/Outlines_of_Swedish_leaves/14551005>.
  Licence: **{lic}** (<https://creativecommons.org/licenses/by/4.0/>).
  figshare's description: "A CSV containing the x and y coordinates of 440 leaf outlines extracted
  from images found in the Swedish Leaf Dataset. All outlines have been scaled to +-1.5 in x and y
  direction, and each outline has 150 points."
- The outlines were traced by those authors from the **Swedish Leaf dataset** of Oskar J. O. Söderkvist
  (2001), *Computer vision classification of leaves from Swedish trees*, MSc thesis, Linköping University
  (<https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/>; the author's Zenodo copy,
  record 19663514, is CC BY 4.0; the Linköping page states no terms).
- File downloaded: `{LEAVES_FILE_NAME}`, md5 {md5}, sha256 {sha}.

## What the CSV contains

Columns `{', '.join(header or [])}`: {len(outlines)} outlines, each with a species number and
species name ({len(classes)} species), an outline name (e.g. `l10nr001` = species 10, leaf 1) and ordered
`(X, Y)` points in `pointOrder`.  Species labels therefore **come with the file**; nothing was inferred.
The file actually holds {raw_desc} (not the 150 of the description); the first point is repeated at the
end in {closing_dup} of {len(outlines)} outlines (a closed polygon written with its closing vertex).

## What was done to the data

1. Rows grouped by `indexNumber`, points sorted by `pointOrder`, the repeated closing vertex dropped.
2. Each closed outline resampled to **{n_points} points equally spaced along its arc length**
   (`shapekit.contours.resample_by_arclength(points, {n_points}, closed=True)`, linear interpolation along
   the polygon; the start point is the file's first point).  Coordinates are otherwise unchanged
   (still within ±1.5).
3. Stored as `points` ({points.shape[0]}, {n_points}, 2) float64, `labels` (0–{len(classes) - 1}, = speciesNumber − 1),
   `classes` (species names in speciesNumber order), `names` (indexName) and `species_number`.

| label | species | outlines |
|---|---|---|
{lines}
""", encoding="utf-8")
    print("  wrote data/leaves_outlines.ATTRIBUTION.md")


# ---------------------------------------------------------------- rbc (local only)
def _find_mask(field_dir: Path, cls: str) -> Path | None:
    for p in field_dir.iterdir():
        if p.name.lower() == f"mask-{cls}.jpg":
            return p
    return None


def _components(mask_path: Path) -> list[tuple[np.ndarray, tuple, float]]:
    """Connected components of a JPEG mask: (component mask crop, slices, area), border-touching ones dropped."""
    from PIL import Image
    from scipy import ndimage
    grey = np.asarray(Image.open(mask_path).convert("L"))
    binary = ndimage.binary_fill_holes(grey > RBC_THRESHOLD)
    lab, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == grey.shape[0] or sl[1].stop == grey.shape[1]
        comp = lab[sl] == i
        area = float(comp.sum())
        if touches or area < RBC_MIN_ABS_AREA:
            continue
        out.append((comp, sl, area))
    return out


def fetch_rbc(skip: bool, n_points: int) -> None:
    print("[rbc] erythrocytesIDB v2, Zenodo 18299474")
    print(RBC_NOTICE)
    LOCAL.mkdir(parents=True, exist_ok=True)
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    if "data/_local" not in gitignore:
        raise SystemExit("data/_local/ is not in .gitignore; refusing to download CC BY-NC-ND data into the repository")
    if RBC_ZIP.exists() and skip:
        print(f"  reusing {RBC_ZIP} (--skip-download)")
    else:
        download(RBC_URL, RBC_ZIP, md5=RBC_MD5, resume=True)
    report(RBC_ZIP, None if skip else RBC_MD5)

    if any(RBC_DIR.rglob("erythrocytesIDB2")):
        print(f"  {RBC_DIR} already unpacked")
    else:
        n = safe_extract_zip(RBC_ZIP, RBC_DIR)
        print(f"  unpacked {n} files into {RBC_DIR}")

    # One outline per connected component of each per-class field mask of IDB2 and IDB3.
    # (erythrocytesIDB1's "individual cells" are 80 x 80 image crops, not masks, so they are not used.)
    points, labels, dbs, fields, areas = [], [], [], [], []
    for db in ("erythrocytesIDB2", "erythrocytesIDB3"):
        roots = [p for p in RBC_DIR.rglob(db) if p.is_dir()]
        if not roots:
            raise SystemExit(f"{db} not found under {RBC_DIR}")
        field_dirs = sorted(p for p in roots[0].iterdir() if p.is_dir())
        comps = []   # (label, field, comp, area)
        missing = Counter()
        for fd in field_dirs:
            for k, cls in enumerate(RBC_CLASSES):
                mp = _find_mask(fd, cls)
                if mp is None:
                    missing[cls] += 1
                    continue
                for comp, _sl, area in _components(mp):
                    comps.append((k, fd.name, comp, area))
        median = float(np.median([a for *_, a in comps]))
        lo, hi = RBC_AREA_RANGE[0] * median, RBC_AREA_RANGE[1] * median
        kept = Counter()
        dropped = Counter()
        for k, field, comp, area in comps:
            if not (lo <= area <= hi):
                dropped["area"] += 1
                continue
            padded = np.pad(comp, 1)
            contours = mask_to_contours(padded)
            if not contours:
                dropped["no contour"] += 1
                continue
            outer = max(contours, key=len)
            if len(outer) < RBC_MIN_CONTOUR:
                dropped["short contour"] += 1
                continue
            points.append(resample_by_arclength(outer - 1.0, n_points, closed=True))
            labels.append(k)
            dbs.append(int(db[-1]))
            fields.append(field)
            areas.append(area)
            kept[RBC_CLASSES[k]] += 1
        print(f"  {db}: {len(field_dirs)} fields (class masks missing: {dict(missing) or 'none'}); "
              f"median component area {median:.0f} px, kept area in [{lo:.0f}, {hi:.0f}]; "
              f"kept {dict(kept)}; dropped {dict(dropped) or 'none'}")

    pts = np.stack(points).astype(np.float64)
    lab = np.array(labels, dtype=np.int64)
    np.savez(RBC_NPZ, points=pts, labels=lab, classes=np.array(RBC_CLASSES), database=np.array(dbs, dtype=np.int64),
             field=np.array(fields), component_area=np.array(areas, dtype=np.float64),
             license=np.array("CC BY-NC-ND 4.0 -- local use only, never redistribute"))
    counts = Counter(lab.tolist())
    print(f"  wrote {RBC_NPZ} (LOCAL ONLY): points {pts.shape}; class counts "
          + ", ".join(f"{i} {c} = {counts.get(i, 0)}" for i, c in enumerate(RBC_CLASSES)))
    print(RBC_NOTICE)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brain", action="store_true")
    p.add_argument("--gorilla", action="store_true")
    p.add_argument("--leaves", action="store_true")
    p.add_argument("--rbc", action="store_true", help="LOCAL ONLY (CC BY-NC-ND 4.0); writes under data/_local/")
    p.add_argument("--all", action="store_true", help="all four")
    p.add_argument("--skip-download", action="store_true", help="reuse files already in the download dir / data/_local")
    p.add_argument("--download-dir", type=Path, default=LOCAL / "downloads",
                   help="where the small downloads are kept (default data/_local/downloads, git-ignored)")
    p.add_argument("--n-points", type=int, default=150, help="points per resampled outline (leaves, rbc)")
    a = p.parse_args(argv)
    if a.all:
        a.brain = a.gorilla = a.leaves = a.rbc = True
    if not (a.brain or a.gorilla or a.leaves or a.rbc):
        p.error("nothing to do: pass --brain, --gorilla, --leaves, --rbc or --all")
    DATA.mkdir(exist_ok=True)
    a.download_dir.mkdir(parents=True, exist_ok=True)
    if a.brain:
        fetch_brain(a.download_dir, a.skip_download)
    if a.gorilla:
        fetch_gorilla(a.download_dir, a.skip_download)
    if a.leaves:
        fetch_leaves(a.download_dir, a.skip_download, a.n_points)
    if a.rbc:
        fetch_rbc(a.skip_download, a.n_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
