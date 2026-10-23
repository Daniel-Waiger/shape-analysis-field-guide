#!/usr/bin/env python
"""Fetch the Phenotype Lab datasets that are not typed into the repository.

Every array this script writes is traced back to a download whose size,
MD5 and SHA-256 are printed, and every dataset gets a notice file next to
it saying where it came from and under which license.  Nothing from the
downloads is ever executed: tarballs, zips, CSV and ``.rda`` files are
read as data with plain Python readers, and the download folders are
never put on ``sys.path`` (the script is safe under ``python -I``).

Run from the repository root::

    python scripts/fetch_datasets.py --brain --gorilla --leaves      # redistributable
    python scripts/fetch_datasets.py --rbc                           # local only
    python scripts/fetch_datasets.py --all
    python scripts/fetch_datasets.py --images celegans mef            # original images, local only

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

``--images celegans|mef|leaves|rbc|all``
    **Local only.**  The original *images* behind the datasets, into
    ``data/_local/images/<id>/`` (git-ignored, never committed); each set
    prints its license and the attribution it requires and gets a
    ``SOURCE.txt`` with the checksums.  ``celegans``: BBBC010 images zip and
    both ground-truth zips (foreground, one mask per worm) from
    data.broadinstitute.org (CC0).  ``mef``: a bounded subset (``--max-files``,
    default 40 files, about 107 MB, never more than 150 MB) of the TIFFs of
    github.com/kukionfr/Micropattern_MEF_LMNA_Image at a pinned commit (MIT ©
    2020 Kyu Sang Han), the two fields the tutorial used first; every file is
    checked against the SHA-256 in its Git LFS pointer.  ``leaves``: the
    per-species zips of the Swedish Leaf dataset from the Linköping page
    (``--species 5,7`` by default, the two smallest; ``--species all`` for
    all 15, about 3.9 GB).  ``rbc``: nothing is downloaded; the already
    fetched erythrocytesIDB tree (``--rbc``) is reported.
    ``scripts/make_sample_images.py`` turns the celegans and mef originals
    into the small web samples under ``docs/lab/data/images/``.

Options: ``--download-dir`` (where the small downloads are kept; default
``data/_local/downloads``), ``--skip-download`` (reuse files that are already
there without contacting the network; their pinned checksums are still
enforced), ``--n-points`` (points per resampled outline, default 150),
``--max-files`` and ``--species`` (for ``--images``, see above).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
LEAVES_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

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
      (for the large zip); a 416 answer to that request means the ``.part`` is
      already complete and it is promoted to ``dest`` and verified;
    - HTTP 202 (figshare "preparing the file") and transient errors are retried;
    - a finished download is verified against ``md5`` and **deleted** on a
      mismatch (``SystemExit``), so a bad file never stays on disk.
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
            try:
                response = _open(url, headers)
            except urllib.error.HTTPError as e:
                if offset and e.code == 416:          # the .part already holds the whole file
                    print(f"    {part.name} is already complete (HTTP 416); verifying it")
                    part.replace(dest)
                    return _verify_download(dest, md5)
                raise
            with response as r:
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
            return _verify_download(dest, md5)
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


def _verify_download(dest: Path, md5: str | None) -> Path:
    """Check a finished download against its pinned md5; delete it and stop on a mismatch."""
    if md5:
        got = checksums(dest)[0]
        if got != md5:
            dest.unlink()
            raise SystemExit(f"checksum mismatch for {dest.name}: got md5 {got}, expected {md5}; the file was "
                             f"deleted, re-run to download it again (or update the pinned md5 if the source changed)")
    return dest


def gitignore_covers(text: str, path: str = "data/_local") -> bool:
    """True when ``.gitignore`` has a real (uncommented, unnegated) line ignoring ``path``."""
    wanted = {path, path + "/", "/" + path, "/" + path + "/"}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line in wanted:
            return True
    return False


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
        wanted = []
        for info in z.infolist():            # validate every name before writing anything
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store") or info.is_dir():
                continue
            target = (out_dir / name).resolve()
            if out_dir.resolve() not in target.parents:
                raise SystemExit(f"refusing to extract {name!r}: path escapes {out_dir}")
            wanted.append((info, target))
        for info, target in wanted:
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


def gpl2_notice(dataset_file: str, flag: str, what: str, provider: str, tarball: Path, md5: str, sha: str) -> str:
    """GPL-2 notice for a dataset extracted from the ``shapes`` package.

    ``provider`` names who supplied the data to the package (as its documentation
    says); the wording of the rest is fixed so that re-running the script
    reproduces the reviewed notices in data/ (only the timestamp changes).
    """
    return f"""Notice for data/{dataset_file}
(written by {SCRIPT} {flag} on {now_iso()})

{what}

Copyright: the R package "shapes" is by Ian L. Dryden (its DESCRIPTION names him
as author and maintainer and carries no separate copyright line); the data were
provided to the package by {provider}, as the package documents.

The arrays were extracted from the source tarball of the R package "shapes"
version {SHAPES_VERSION} by Ian L. Dryden (https://cran.r-project.org/package=shapes,
https://github.com/iandryden/shapes), which is distributed under the GNU
General Public License version 2 (GPL-2).  They are redistributed here under
the same license, GPL-2, with this notice.  The full license text is at
{GPL2_URL} and a byte-identical copy
is in LICENSES/GPL-2.txt of this repository.  This data is provided WITHOUT ANY
WARRANTY, without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE (GPL-2 sections 11 and 12).  The GPL-2 applies to this data
file; the rest of this repository is MIT-licensed (see LICENSE).
[Unverified] The package's GPL-2 is the only license statement found for these
data; whether the data providers themselves granted that license to the data,
separately from the package, was not verified.

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
        gpl2_notice("brain_landmarks.npz", "--brain", what, "Fred Bookstein", tarball, md5, sha), encoding="utf-8")
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
        gpl2_notice("gorilla_landmarks.npz", "--gorilla", what, "Paul O'Higgins", tarball, md5, sha), encoding="utf-8")
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
    print(f"  license (figshare API): {lic}; authors: {authors}; DOI: {doi}")
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
  License: **{lic}** (<{LEAVES_LICENSE_URL}>).
  figshare's description: "A CSV containing the x and y coordinates of 440 leaf outlines extracted
  from images found in the Swedish Leaf Dataset. All outlines have been scaled to +-1.5 in x and y
  direction, and each outline has 150 points."
- The outlines were extracted by those authors (figshare's wording; the method is not described) from the
  photographs of the **Swedish Leaf dataset** of Oskar J. O. Söderkvist (2001), *Computer vision
  classification of leaves from Swedish trees*, MSc thesis, Linköping University
  (<https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/>). The Linköping page states no terms.
  A Zenodo record listing Söderkvist as creator (19663514, CC BY 4.0, deposited 2026-04-20; uploader not
  identified) carries the photographs under CC BY 4.0; the license chain for the underlying photographs
  rests on that record, which was not verified to come from the author.
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
   the polygon; the start point is the file's first point; the lab later re-roots each outline at the
   cyclic shift that best matches the mean, see chapter 10).  Coordinates are otherwise unchanged
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
    if not gitignore_covers(gitignore, "data/_local"):
        raise SystemExit("data/_local/ is not in .gitignore; refusing to download CC BY-NC-ND data into the repository")
    if RBC_ZIP.exists() and skip:
        print(f"  reusing {RBC_ZIP} (--skip-download)")
    else:
        download(RBC_URL, RBC_ZIP, md5=RBC_MD5, resume=True)
    report(RBC_ZIP, RBC_MD5)             # always enforced, --skip-download or not

    marker = RBC_DIR / ".unpacked-ok"     # written only after a complete extraction
    if marker.exists():
        print(f"  {RBC_DIR} already unpacked")
    else:
        if RBC_DIR.exists():
            print(f"  {RBC_DIR} exists without the completion marker (interrupted unpack?); unpacking again")
            shutil.rmtree(RBC_DIR)
        n = safe_extract_zip(RBC_ZIP, RBC_DIR)
        marker.write_text(f"{n} files unpacked from {RBC_ZIP.name} (md5 {RBC_MD5}) on {now_iso()}\n", encoding="utf-8")
        print(f"  unpacked {n} files into {RBC_DIR}")

    # One outline per connected component of each per-class field mask of IDB2 and IDB3.
    # (erythrocytesIDB1's "individual cells" are 80 x 80 image crops, not masks, so they are not used.)
    points, labels, dbs, fields, areas = [], [], [], [], []
    extraction_notes = [
        f"outlines traced by {SCRIPT} --rbc from the per-class field masks of erythrocytesIDB2 and erythrocytesIDB3 "
        f"(erythrocytesIDB1's single-cell crops are images, not masks, and are not used): connected components of "
        f"each mask (pixel > {RBC_THRESHOLD}, holes filled), border-touching components and blobs under "
        f"{RBC_MIN_ABS_AREA} px dropped, components kept when their area is within [{RBC_AREA_RANGE[0]}, "
        f"{RBC_AREA_RANGE[1]}] x the median area of their database, outer contour resampled to {n_points} points "
        f"by arc length"]
    for db in ("erythrocytesIDB2", "erythrocytesIDB3"):
        roots = sorted(p for p in RBC_DIR.rglob(db) if p.is_dir())
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
        line = (f"{db}: {len(field_dirs)} fields (class masks missing: {dict(missing) or 'none'}); "
                f"median component area {median:.0f} px, kept area in [{lo:.0f}, {hi:.0f}]; "
                f"kept {dict(kept)}; dropped {dict(dropped) or 'none'}")
        print(f"  {line}")
        extraction_notes.append(line)

    pts = np.stack(points).astype(np.float64)
    lab = np.array(labels, dtype=np.int64)
    extraction_notes.append("the class counts are this extraction's, not the database's cell counts")
    np.savez(RBC_NPZ, points=pts, labels=lab, classes=np.array(RBC_CLASSES), database=np.array(dbs, dtype=np.int64),
             field=np.array(fields), component_area=np.array(areas, dtype=np.float64),
             preprocessing=np.array(extraction_notes),
             license=np.array("CC BY-NC-ND 4.0 -- local use only, never redistribute"))
    counts = Counter(lab.tolist())
    print(f"  wrote {RBC_NPZ} (LOCAL ONLY): points {pts.shape}; class counts "
          + ", ".join(f"{i} {c} = {counts.get(i, 0)}" for i, c in enumerate(RBC_CLASSES)))
    print(RBC_NOTICE)


# ---------------------------------------------------------------- original images (local only)
IMAGES_DIR = LOCAL / "images"
IMAGE_SETS = ("celegans", "mef", "leaves", "rbc")

BBBC010_BASE = "https://data.broadinstitute.org/bbbc/BBBC010/"
BBBC010_PAGE = "https://bbbc.broadinstitute.org/BBBC010"
# name -> (sub-folder it is unpacked into, size in bytes, md5).  BBBC publishes no checksums; the
# sizes are the server's Content-Length and the md5s are of the files as downloaded on 2026-10-23.
BBBC010_FILES = OrderedDict([
    ("BBBC010_v2_images.zip", ("images", 68389595, "ef94b142c6dbc64ffc2d90028b130305")),
    ("BBBC010_v1_foreground.zip", ("foreground", 1097766, "1925b6a88a23431fd433a5c42d5e6da6")),
    ("BBBC010_v1_foreground_eachworm.zip", ("foreground_eachworm", 2734642, "ae73910ed293397b4ea5082883d278b1")),
])
BBBC010_NOTICE = """
  License: CC0 1.0.  BBBC010's page: "To the extent possible under law, the various
  contributors of the imagesets have waived all copyright and related or neighboring
  rights to BBBC010."  No attribution is legally required; the page asks for this
  citation: "We used the C.elegans infection live/dead image set version 1 provided by
  Fred Ausubel and available from the Broad Bioimage Benchmark Collection
  [Ljosa et al., Nature Methods, 2012]."  (Ljosa, V., Sokolnicki, K. L. & Carpenter, A. E.
  (2012) Annotated high-throughput microscopy image sets for validation. Nat. Methods 9:637.)
"""

MEF_OWNER_REPO = "kukionfr/Micropattern_MEF_LMNA_Image"
MEF_COMMIT = "7cbe6676825d3389c66b1da1102a7ddc2fcaef4a"   # master on 2026-10-23
MEF_API_TREE = f"https://api.github.com/repos/{MEF_OWNER_REPO}/git/trees/{MEF_COMMIT}?recursive=1"
MEF_RAW = f"https://raw.githubusercontent.com/{MEF_OWNER_REPO}/{MEF_COMMIT}/"
MEF_MEDIA = f"https://media.githubusercontent.com/media/{MEF_OWNER_REPO}/{MEF_COMMIT}/"
MEF_FOLDERS = tuple(f"{p}/LMNA{g}{k}" for p in ("Control", "Circle", "Triangle") for g in ("++", "--") for k in (1, 2))
# The two fields Shape Analysis 101 ships as wildtype/ and lmna_deficient/ raw_images/xy001c*.tif
# (identified by their SHA-256, which equals the LFS oid of these paths): fetched first.
MEF_TUTORIAL_FIELDS = ("Control/LMNA++1/xy001", "Circle/LMNA--1/xy001")
MEF_MAX_BYTES = 150_000_000
# The tutorial's outlines and instance mask for those fields (Shape Analysis 101, BSD-3), read by
# scripts/make_sample_images.py; mef_data.npy is a pickled dict and is only ever opened with a
# restricted unpickler that admits plain numpy arrays.
TUTORIAL_COMMIT = "bfccc0cd35d5fec7a73d2b43c906d4199e02c9da"
TUTORIAL_RAW = f"https://raw.githubusercontent.com/uhlmanngroup/shapeanalysis101/{TUTORIAL_COMMIT}/"
MEF_TUTORIAL_FILES = OrderedDict([
    ("data/MEF_LMNA/mef_data.npy", (10783452, "9748f2056fdede1a01847c1b40dac88a")),
    ("data/MEF_LMNA/xy001c1_instances.h5", (6143968, "afad98f567939287fcc67271ff50e99f")),
])
MEF_NOTICE = """
  License: MIT, Copyright (c) 2020 Kyu Sang Han.  The MIT notice ("The above copyright
  notice and this permission notice shall be included in all copies or substantial
  portions of the Software") must accompany every copy: the full text is saved next to
  the images as LICENSE.  The repository's README names the paper the images were made
  for (it says 2020): Phillip, Han, Chen, Wirtz & Wu, "A robust unsupervised machine-learning
  method to quantify the morphological heterogeneity of cells and nuclei", Nature Protocols
  16, 754-774 (2021), doi:10.1038/s41596-020-00432-x.  The README names no separate license
  for the images; [Unverified] whether the author intended the MIT software license to
  cover the image data.  Images: MEF LMNA++ (wild-type) and LMNA-- (lamin A/C deficient) on
  Circle / Triangle micropatterns or unpatterned Control, in folders suffixed 1 and 2 (the
  README does not say what the suffix means); c1 = phalloidin (actin), c2 = DAPI (nuclei).
"""

LEAVES_PAGE = "https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/"
# species number -> (name as on the Linköping page, zip size in bytes (Content-Length), md5 or None)
LEAVES_ZIPS = OrderedDict([
    (1, ("Ulmus carpinifolia", 250780097, "a2be49500d6719a152c86a5a321c72b6")), (2, ("Acer", 691147424, None)),
    (3, ("Salix aurita", 238409222, None)), (4, ("Quercus", 144067810, None)),
    (5, ("Alnus incana", 74427080, None)), (6, ("Betula pubescens", 213508706, None)),
    (7, ("Salix alba 'Sericea'", 98867138, None)), (8, ("Populus tremula", 212801360, None)),
    (9, ("Ulmus glabra", 335120681, None)), (10, ("Sorbus aucuparia", 581191233, None)),
    (11, ("Salix sinerea", 126555720, None)), (12, ("Populus", 255321931, None)),
    (13, ("Tilia", 189169741, None)), (14, ("Sorbus intermedia", 292517046, None)),
    (15, ("Fagus silvatica", 180097468, None)),
])
LEAVES_DEFAULT_SPECIES = (5, 7)      # the two smallest zips (74 MB + 99 MB)
LEAVES_IMAGE_NOTICE = """
  License: the Linköping page states no terms.  A Zenodo record listing Söderkvist as
  creator (19663514, CC BY 4.0, deposited 2026-04-20, uploader not identified) is the only
  license statement found for the photographs.  Treat them as local-only (they are never
  copied into the repository) and cite: Oskar J. O. Söderkvist, "Computer vision
  classification of leaves from Swedish trees", Master's thesis, Linköping University, 2001.
"""


def _require_local_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").exists() else ""
    if not gitignore_covers(gitignore, "data/_local"):
        raise SystemExit("data/_local/ is not in .gitignore; refusing to download images into the repository")


def _check_size(path: Path, size: int | None) -> None:
    if size is not None and path.stat().st_size != size:
        got = path.stat().st_size
        path.unlink()
        raise SystemExit(f"{path.name}: {got:,} bytes, expected {size:,}; the file was deleted, re-run to download it again")


def _write_source(out_dir: Path, title: str, notice: str, lines: list[str]) -> None:
    (out_dir / "SOURCE.txt").write_text(
        f"{title}\nFetched by {SCRIPT} on {now_iso()} -- LOCAL ONLY (data/_local/ is git-ignored).\n"
        f"{notice}\nFiles (size, md5, sha256):\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _report_line(path: Path, base: Path) -> str:
    md5, sha, size = checksums(path)
    return f"  {path.relative_to(base)}  {size:,} bytes  md5 {md5}  sha256 {sha}"


def fetch_images_celegans(skip: bool) -> None:
    print("[images: celegans] BBBC010, C. elegans infection live/dead image set (CC0)")
    out = IMAGES_DIR / "celegans"
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, (sub, size, md5) in BBBC010_FILES.items():
        zpath = download(BBBC010_BASE + name, out / name, md5=md5, skip=skip)
        _check_size(zpath, size)
        report(zpath, md5)
        lines.append(_report_line(zpath, out))
        marker = out / sub / ".unpacked-ok"
        if marker.exists():
            print(f"  {out.relative_to(ROOT) / sub} already unpacked")
        else:
            if (out / sub).exists():
                shutil.rmtree(out / sub)
            n = safe_extract_zip(zpath, out / sub)
            marker.write_text(f"{n} files from {name} ({checksums(zpath)[0]}) on {now_iso()}\n", encoding="utf-8")
            print(f"  unpacked {n} files into {(out / sub).relative_to(ROOT)}")
    for sub in ("images", "foreground", "foreground_eachworm"):
        files = [p for p in (out / sub).rglob("*")
                 if p.is_file() and not p.name.startswith(".") and ".svn" not in p.parts]
        kinds = Counter(p.suffix.lower() for p in files)
        print(f"  {sub}: {len(files)} files {dict(kinds)} (the zips also carry .svn/ metadata, ignored)")
    _write_source(out, f"BBBC010 ({BBBC010_PAGE})", BBBC010_NOTICE, lines)
    print(BBBC010_NOTICE)


def _lfs_pointer(path: str) -> tuple[str, int] | None:
    """Read the Git LFS pointer of ``path`` at the pinned commit: (sha256, size), None if absent."""
    from urllib.parse import quote
    try:
        with _open(MEF_RAW + quote(path), timeout=60) as r:
            text = r.read(1024).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines() if " " in line)
    if not fields.get("version", "").startswith("https://git-lfs.github.com/spec/v1"):
        raise SystemExit(f"{path}: expected a Git LFS pointer, got {text[:80]!r}")
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:"):
        raise SystemExit(f"{path}: LFS pointer without a sha256 oid")
    return oid.split(":", 1)[1], int(fields["size"])


def _mef_listing() -> set[str] | None:
    """Every path at the pinned commit via the GitHub API, or None when the API is unreachable."""
    import json
    try:
        with _open(MEF_API_TREE, {"Accept": "application/vnd.github+json"}, timeout=60) as r:
            tree = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - rate limits, proxies: fall back to probing raw pointers
        print(f"  GitHub API listing unavailable ({e}); probing raw.githubusercontent.com instead")
        return None
    if tree.get("truncated"):
        print("  GitHub API listing is truncated; probing raw.githubusercontent.com instead")
        return None
    return {t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"}


def mef_candidates() -> list[str]:
    """Field stems in fetch order: the tutorial's two fields, then xy001, xy002, ... round-robin
    over the 12 folders (Control / Circle / Triangle x LMNA++ / LMNA-- x suffix 1 / 2)."""
    order = list(MEF_TUTORIAL_FIELDS)
    for i in range(1, 200):
        for folder in MEF_FOLDERS:
            stem = f"{folder}/xy{i:03d}"
            if stem not in order:
                order.append(stem)
    return order


def fetch_images_mef(skip: bool, max_files: int) -> None:
    print(f"[images: mef] Micropattern_MEF_LMNA_Image @ {MEF_COMMIT[:12]} (MIT (c) 2020 Kyu Sang Han)")
    if max_files < 2:
        raise SystemExit("--max-files must be at least 2 (one field = c1 + c2)")
    from urllib.parse import quote
    out = IMAGES_DIR / "mef"
    out.mkdir(parents=True, exist_ok=True)
    lic = download(MEF_RAW + "LICENSE", out / "LICENSE", skip=skip)
    if "Copyright (c) 2020 Kyu Sang Han" not in lic.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(f"{lic} does not carry the expected MIT copyright line; refusing to continue")
    listing = _mef_listing()
    n_fields = max_files // 2
    if max_files % 2:
        print(f"  --max-files {max_files} is odd; fetching {n_fields} whole fields ({2 * n_fields} files)")
    chosen, total, lines = [], 0, []
    for stem in mef_candidates():
        if len(chosen) == n_fields:
            break
        paths = [f"{stem}c1.tif", f"{stem}c2.tif"]
        if listing is not None and not all(p in listing for p in paths):
            continue
        ptrs = [_lfs_pointer(p) for p in paths]
        if any(p is None for p in ptrs):
            continue
        size = sum(s for _, s in ptrs)
        if total + size > MEF_MAX_BYTES:
            print(f"  stopping at {len(chosen)} fields: the next would pass {MEF_MAX_BYTES / 1e6:.0f} MB")
            break
        chosen.append((stem, paths, ptrs))
        total += size
    print(f"  {len(chosen)} fields, {2 * len(chosen)} files, {total / 1e6:.1f} MB")
    for stem, paths, ptrs in chosen:
        for path, (sha, size) in zip(paths, ptrs):
            dest = out / path
            if dest.exists() and checksums(dest)[1] == sha:
                print(f"  reusing {path} (sha256 matches its LFS pointer)")
            else:
                if dest.exists():
                    dest.unlink()
                download(MEF_MEDIA + quote(path), dest)
                _check_size(dest, size)
                got = checksums(dest)[1]
                if got != sha:
                    dest.unlink()
                    raise SystemExit(f"{path}: sha256 {got} differs from its LFS pointer {sha}; deleted")
                print(f"  {path}: downloaded, sha256 matches its LFS pointer")
            report(dest)
            lines.append(_report_line(dest, out))
    print("  the tutorial's outlines for these fields (Shape Analysis 101, BSD-3, Hugger & Uhlmann 2021):")
    for rel, (size, md5) in MEF_TUTORIAL_FILES.items():
        dest = download(TUTORIAL_RAW + rel, out / "tutorial" / Path(rel).name, md5=md5, skip=skip)
        _check_size(dest, size)
        report(dest, md5)
        lines.append(_report_line(dest, out))
    tutorial = [s for s, *_ in chosen if s in MEF_TUTORIAL_FIELDS]
    print(f"  tutorial fields included: {tutorial or 'none'} (wildtype = {MEF_TUTORIAL_FIELDS[0]}, "
          f"lmna_deficient = {MEF_TUTORIAL_FIELDS[1]})")
    _write_source(out, f"Micropattern_MEF_LMNA_Image (https://github.com/{MEF_OWNER_REPO}, commit {MEF_COMMIT})",
                  MEF_NOTICE, lines)
    print(MEF_NOTICE)


def parse_species(text: str) -> list[int]:
    if text.strip().lower() == "all":
        return list(LEAVES_ZIPS)
    try:
        nums = sorted({int(t) for t in text.replace(" ", "").split(",") if t})
    except ValueError:
        raise SystemExit(f"--species expects species numbers 1-15 separated by commas, or 'all'; got {text!r}") from None
    bad = [n for n in nums if n not in LEAVES_ZIPS]
    if bad or not nums:
        raise SystemExit(f"--species: unknown species {bad or text!r}; valid: 1-15 (see {LEAVES_PAGE})")
    return nums


def fetch_images_leaves(skip: bool, species: list[int]) -> None:
    print("[images: leaves] Swedish Leaf dataset, O. J. O. Söderkvist (2001), Linköping University")
    print(LEAVES_IMAGE_NOTICE)
    out = IMAGES_DIR / "leaves"
    out.mkdir(parents=True, exist_ok=True)
    total = sum(LEAVES_ZIPS[s][1] for s in species)
    print(f"  species {species}: {total / 1e6:,.0f} MB of zips")
    lines = []
    for s in species:
        name, size, md5 = LEAVES_ZIPS[s]
        print(f"  species {s}: {name} ({size / 1e6:,.0f} MB)")
        zpath = download(LEAVES_PAGE + f"leaf{s}.zip", out / f"leaf{s}.zip", md5=md5, skip=skip, resume=True)
        _check_size(zpath, size)
        report(zpath, md5)
        lines.append(_report_line(zpath, out))
        sub = out / f"leaf{s}"
        marker = sub / ".unpacked-ok"
        if marker.exists():
            print(f"    {sub.relative_to(ROOT)} already unpacked")
        else:
            if sub.exists():
                shutil.rmtree(sub)
            n = safe_extract_zip(zpath, sub)
            marker.write_text(f"{n} files from leaf{s}.zip on {now_iso()}\n", encoding="utf-8")
            print(f"    unpacked {n} files into {sub.relative_to(ROOT)}")
        files = [p for p in sub.rglob("*") if p.is_file() and not p.name.startswith(".")]
        print(f"    {len(files)} files {dict(Counter(p.suffix.lower() for p in files))}")
    _write_source(out, f"Swedish Leaf dataset ({LEAVES_PAGE})", LEAVES_IMAGE_NOTICE, lines)


def report_images_rbc() -> None:
    print("[images: rbc] erythrocytesIDB v2 (CC BY-NC-ND 4.0): nothing to download here")
    print(RBC_NOTICE)
    if not (RBC_DIR / ".unpacked-ok").exists():
        print(f"  {RBC_DIR.relative_to(ROOT)} is not there yet: run `python {SCRIPT} --rbc` (594 MB, local only)")
        return
    files = [p for p in RBC_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")]
    per_db = Counter(next((part for part in p.relative_to(RBC_DIR).parts
                           if part in ("erythrocytesIDB1", "erythrocytesIDB2", "erythrocytesIDB3")), "other")
                     for p in files)
    kinds = Counter(p.suffix.lower() for p in files)
    print(f"  the images are already unpacked in {RBC_DIR.relative_to(ROOT)}: {len(files)} files {dict(kinds)}; "
          f"per database {dict(sorted(per_db.items()))}")
    print("  Cite: Marrero Fernández, P. D. et al., erythrocytesIDB, Universitat de les Illes Balears, Zenodo "
          "18299474; IEEE J. Biomed. Health Inform. 19(4):1514-1525 (2015).  No images or derived coordinates "
          "may be redistributed (NC, ND).")


def fetch_images(which: list[str], skip: bool, max_files: int, species: list[int]) -> None:
    _require_local_ignored()
    sets = list(IMAGE_SETS) if "all" in which else [w for w in IMAGE_SETS if w in which]
    for s in sets:
        if s == "celegans":
            fetch_images_celegans(skip)
        elif s == "mef":
            fetch_images_mef(skip, max_files)
        elif s == "leaves":
            fetch_images_leaves(skip, species)
        elif s == "rbc":
            report_images_rbc()
    print(f"Images are under {IMAGES_DIR.relative_to(ROOT)}/ (git-ignored): never commit them.")
    sampled = [s for s in sets if s in ("celegans", "mef")]
    if sampled:
        print(f"`python scripts/make_sample_images.py` builds the small redistributable samples from them "
              f"({', '.join(sampled)}).")


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
    p.add_argument("--images", nargs="+", choices=IMAGE_SETS + ("all",), metavar="SET",
                   help="LOCAL ONLY: original images into data/_local/images/<set>/ "
                        "(celegans, mef, leaves, rbc or all)")
    p.add_argument("--max-files", type=int, default=40,
                   help="--images mef: number of TIFFs (c1 + c2 per field; default 40, about 107 MB)")
    p.add_argument("--species", default=",".join(map(str, LEAVES_DEFAULT_SPECIES)),
                   help="--images leaves: species numbers 1-15, comma-separated, or 'all' "
                        "(default 5,7: the two smallest zips)")
    a = p.parse_args(argv)
    if a.all:
        a.brain = a.gorilla = a.leaves = a.rbc = True
    if not (a.brain or a.gorilla or a.leaves or a.rbc or a.images):
        p.error("nothing to do: pass --brain, --gorilla, --leaves, --rbc, --all or --images SET")
    species = parse_species(a.species) if a.images else []
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
    if a.images:
        fetch_images(a.images, a.skip_download, a.max_files, species)
    return 0


if __name__ == "__main__":
    sys.exit(main())
