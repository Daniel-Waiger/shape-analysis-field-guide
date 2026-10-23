#!/usr/bin/env python
"""Build the small, redistributable sample images of the Phenotype Lab (deterministic).

Run from the repository root, after ``python scripts/fetch_datasets.py --images celegans mef``::

    python scripts/make_sample_images.py
    python scripts/make_sample_images.py --tutorial-dir PATH/TO/shapeanalysis101/data   # fallback source

Writes (LAB_SPEC.md, "Round 3: raw images"):

``docs/lab/data/images/celegans/``
    One live well (B07, ampicillin) and one dead well (A23, untreated) of BBBC010, brightfield
    channel ``w2``, contrast-stretched to 8-bit PNG, plus the matching human-corrected foreground
    masks; overlays computed by ``shapekit`` from the per-worm ground-truth masks: the traced
    outline and the chapter-01 pipeline (mask -> skeleton -> 12 arc-length landmarks).
    ``manifest.json`` and ``CREDIT.md`` (CC0 + the citation BBBC010 asks for).
``docs/lab/data/images/mef/``
    The wild-type and the LMNA-/- field the tutorial used (``xy001c1``, phalloidin / actin),
    8-bit PNG, 900 px wide; overlays = the tutorial's ``mef_data.npy`` outlines for that field if
    they register on the image (measured, see ``registration`` in the manifest), otherwise
    outlines traced from ``xy001c1_instances.h5``.  ``manifest.json`` and ``LICENSE`` (MIT).
``docs/figures/01_real_pipeline.png``
    The chapter 01 pipeline on one real worm: image -> ground-truth mask -> skeleton ->
    12 landmarks -> spline.

Sources, in order of preference: ``data/_local/images/`` (from ``fetch_datasets.py --images``),
then ``--tutorial-dir`` (a Shape Analysis 101 ``data/`` folder; the same BBBC010 and MEF files,
only one per-worm mask).  Everything read from those folders is data: TIFF/PNG/HDF5 are read with
tifffile / Pillow / h5py, and the tutorial's pickled ``mef_data.npy`` only through a restricted
unpickler that admits plain numpy arrays.  Nothing random is used; two runs give identical bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shapekit.bspline import BSplineCurve  # noqa: E402
from shapekit.contours import (  # noqa: E402
    mask_to_contours,
    order_loop,
    order_skeleton,
    resample_by_arclength,
    skeleton_endpoints,
    skeletonize,
)

SCRIPT = "scripts/make_sample_images.py"
LOCAL_IMAGES = ROOT / "data" / "_local" / "images"
OUT = ROOT / "docs" / "lab" / "data" / "images"
FIGURE = ROOT / "docs" / "figures" / "01_real_pipeline.png"

MAX_WIDTH = 900
N_LANDMARKS = 12                 # the worm dataset's landmark count
OUTLINE_POINTS = 100             # traced outlines are resampled to this many points for the manifest
MAX_WALK_STEP = 3.0              # px: a longer step in the ordered skeleton means the greedy walk jumped
CEL_STRETCH = (0.5, 99.5)        # percentiles for the brightfield contrast stretch
MEF_STRETCH = (1.0, 99.7)        # percentiles for the actin contrast stretch
REGISTRATION_MAX_MEDIAN_PX = 3.0  # outlines "register" when their median distance to the image's edges is below this
COVERED_MEDIAN_PX = 3.0          # mef_data.npy lists most cells twice (a few px apart): an outline whose points lie,
                                 # in the median, within this distance of an outline already kept is not drawn
FIGURE_WORM = "B07_06"           # the worm the tutorial ships as its ground-truth example

CELEGANS_WELLS = (
    # well, class id, class name, what the well is
    ("B07", 1, "live", "column 7: positive control, treated with ampicillin (mostly live, curved worms)"),
    ("A23", 0, "dead", "column 23: negative control, untreated (mostly dead, rod-like worms)"),
)
MEF_FIELDS = (
    # repo path stem, class id, class name, tutorial folder, mef_data.npy key prefix, PNG name, substrate
    # (LAB_SPEC.md Round 3b: every sampled wild-type field of the tutorial registers on Control/LMNA++1 and every
    # LMNA−/− field on Circle/LMNA--1, so genotype is confounded with substrate and with whatever else differs
    # between the two folders; the captions say so)
    ("Control/LMNA++1/xy001", 0, "wild-type", "wildtype", "wildtype_xy001c1_", "wildtype_control_xy001_actin.png",
     "unpatterned"),
    ("Circle/LMNA--1/xy001", 1, "LMNA−/−", "lmna_deficient", "lmna_xy001c1_", "lmna_circle_xy001_actin.png",
     "circle micropattern"),
)
#: fields per class whose source folder was verified by registration (LAB_SPEC.md Round 3b and "Round 3 fixes")
MEF_SAMPLED_PER_CLASS = 14
#: the folder the tutorial's other class comes from (for the captions)
MEF_OTHER_FOLDER = {0: "Circle/LMNA--1 (circle micropattern)", 1: "Control/LMNA++1 (unpatterned)"}

MIT_LICENSE = """MIT License

Copyright (c) 2020 Kyu Sang Han

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#: the paper the MEF repository's README says the images are for (it says "Nature Protocol paper (2020)"); cited as a
#: courtesy, details from PubMed 33424024
MEF_PAPER = ("Phillip, J. M., Han, K.-S., Chen, W.-C., Wirtz, D. & Wu, P.-H. A robust unsupervised machine-learning "
             "method to quantify the morphological heterogeneity of cells and nuclei. Nature Protocols 16, 754-774 "
             "(2021), doi:10.1038/s41596-020-00432-x (the repository README calls it a 2020 paper)")

BBBC010_CITATION = ('"We used the C.elegans infection live/dead image set version 1 provided by Fred Ausubel and '
                    'available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]."')


# ------------------------------------------------------------------ helpers
def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def stretch(a: np.ndarray, percentiles: tuple[float, float]) -> tuple[np.ndarray, float, float]:
    """Linear contrast stretch: the given percentiles map to 0 and 1, values outside are clipped."""
    lo, hi = np.percentile(a, percentiles)
    return np.clip((a.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0), float(lo), float(hi)


def to_png8(x01: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(np.round(x01 * 255).astype(np.uint8), mode="L").save(path, optimize=True)


def resize01(x01: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Lanczos resize of a [0, 1] float image to ``width`` px (aspect kept); returns image and scale."""
    from PIL import Image
    h, w = x01.shape
    if w <= width:
        return x01, 1.0
    scale = width / w
    height = int(round(h * scale))
    im = Image.fromarray(x01.astype(np.float32), mode="F").resize((width, height), Image.Resampling.LANCZOS)
    return np.clip(np.asarray(im, dtype=np.float64), 0.0, 1.0), scale


def to_png_frame(xy: np.ndarray, scale: float) -> np.ndarray:
    """Pixel-center coordinates (x, y) of the original image -> the resized PNG's frame."""
    return (np.asarray(xy, float) + 0.5) * scale - 0.5


def pts_json(xy: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in xy]


def dump_manifest(manifest: dict, path: Path) -> None:
    """Indented JSON with every point list on one line (readable, about a third of the plain size)."""
    import re
    text = json.dumps(manifest, indent=1, ensure_ascii=False)
    text = re.sub(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", r"[\1,\2]", text)
    text = re.sub(r"\[\s*((?:\[-?[\d.]+,-?[\d.]+\],?\s*)+)\]", lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", text)
    if json.loads(text) != json.loads(json.dumps(manifest, ensure_ascii=False)):
        raise AssertionError("manifest compaction changed the content")
    path.write_text(text + "\n", encoding="utf-8")


def read_mask(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L")) > 127


class _NumpyOnly(pickle.Unpickler):
    """Unpickler for the tutorial's ``mef_data.npy`` (a pickled dict of arrays): numpy arrays only."""

    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            import importlib
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"mef_data.npy: refusing to load {module}.{name}")


def load_mef_data(path: Path) -> dict:
    import numpy.lib.format as fmt
    with open(path, "rb") as f:
        version = fmt.read_magic(f)
        (fmt.read_array_header_1_0 if version == (1, 0) else fmt.read_array_header_2_0)(f)
        obj = _NumpyOnly(f).load()
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected a dict of arrays, got {type(obj)}")
    return obj


def otsu_threshold(a: np.ndarray) -> float:
    hist, edges = np.histogram(a, bins=1024)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def edge_distance(binary: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to the nearest boundary pixel of a binary/label image."""
    from scipy import ndimage
    b = np.asarray(binary)
    edge = np.zeros(b.shape, bool)
    edge[1:, :] |= b[1:, :] != b[:-1, :]
    edge[:-1, :] |= b[1:, :] != b[:-1, :]
    edge[:, 1:] |= b[:, 1:] != b[:, :-1]
    edge[:, :-1] |= b[:, 1:] != b[:, :-1]
    return ndimage.distance_transform_edt(~edge)


def sample_at(dist: np.ndarray, xy: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return ndimage.map_coordinates(dist, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def covered_by(a: np.ndarray, loop: np.ndarray) -> float:
    """Median distance from the points of ``a`` to the closed polyline ``loop`` (points in loop order)."""
    p, q = loop, np.roll(loop, -1, axis=0)
    seg = q - p
    t = np.clip((((a[:, None, :] - p[None]) * seg[None]).sum(-1)) / np.maximum((seg ** 2).sum(-1), 1e-12)[None], 0, 1)
    nearest = p[None] + t[..., None] * seg[None]
    return float(np.median(np.sqrt(((a[:, None, :] - nearest) ** 2).sum(-1)).min(1)))


# ------------------------------------------------------------------ sources
class Sources:
    def __init__(self, local: Path, tutorial: Path | None):
        self.local = local
        self.tutorial = tutorial

    def _first(self, *cands: Path | None) -> Path | None:
        for c in cands:
            if c is not None and c.exists():
                return c
        return None

    def celegans_image(self, well: str) -> tuple[Path | None, str]:
        hits = sorted((self.local / "celegans" / "images").glob(f"*_{well}_w2_*.tif"))
        if hits:
            return hits[0], "BBBC010_v2_images.zip"
        if self.tutorial is not None:
            hits = sorted((self.tutorial / "C. elegans").glob(f"*_{well}_w2_*.tif"))
            if hits:
                return hits[0], "Shape Analysis 101 data/C. elegans/ (a copy of the BBBC010 file)"
        return None, ""

    def celegans_foreground(self, well: str) -> Path | None:
        return self._first(self.local / "celegans" / "foreground" / "BBBC010_v1_foreground" / f"{well}_binary.png")

    def celegans_worms(self, well: str) -> tuple[list[Path], str]:
        folder = self.local / "celegans" / "foreground_eachworm" / "BBBC010_v1_foreground_eachworm"
        hits = sorted(folder.glob(f"{well}_*_ground_truth.png"))
        if hits:
            return hits, "BBBC010_v1_foreground_eachworm.zip"
        if self.tutorial is not None:
            hits = sorted((self.tutorial / "C. elegans").glob(f"{well}_*_ground_truth.png"))
            if hits:
                return hits, "Shape Analysis 101 data/C. elegans/ (only this worm's mask is in the tutorial)"
        return [], ""

    def mef_image(self, stem: str, folder: str, channel: str = "c1") -> Path | None:
        name = Path(stem).name + f"{channel}.tif"
        tut = self.tutorial / "MEF_LMNA" / folder / "raw_images" / name if self.tutorial else None
        return self._first(self.local / "mef" / f"{stem}{channel}.tif", tut)

    def mef_tutorial_file(self, name: str) -> Path | None:
        tut = self.tutorial / "MEF_LMNA" / name if self.tutorial else None
        return self._first(self.local / "mef" / "tutorial" / name, tut)


# ------------------------------------------------------------------ C. elegans
def worm_pipeline(mask: np.ndarray) -> dict:
    """Chapter 01 on one ground-truth worm mask, on a padded crop (the result is the same as on the
    full frame: thinning and the walk are local) shifted back to image coordinates."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    removed = 0
    if n > 1:                                  # stray pixels in the ground truth: keep the worm itself
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        keep = lab == (int(np.argmax(sizes)) + 1)
        removed = int(mask.sum() - keep.sum())
        mask = keep
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - 3, 0), min(rows.max() + 4, mask.shape[0])
    c0, c1 = max(cols.min() - 3, 0), min(cols.max() + 4, mask.shape[1])
    crop = mask[r0:r1, c0:c1]
    offset = np.array([c0, r0], float)
    outline = max(mask_to_contours(crop), key=len) + offset
    skel = skeletonize(crop)
    walk = order_skeleton(skel)
    steps = np.linalg.norm(np.diff(walk, axis=0), axis=1)
    landmarks = resample_by_arclength(walk, N_LANDMARKS, closed=False) + offset
    return {"outline": outline, "skeleton_mask": skel, "skeleton_offset": offset, "walk": walk + offset,
            "landmarks": landmarks, "n_tips": len(skeleton_endpoints(skel)), "max_step": float(steps.max()),
            "n_skeleton": int(skel.sum()), "area": int(mask.sum()), "stray_removed": removed}


def build_celegans(src: Sources, out: Path) -> tuple[dict | None, dict]:
    out.mkdir(parents=True, exist_ok=True)
    images, worms_by_id, notes = [], {}, []
    for well, cid, cname, what in CELEGANS_WELLS:
        tif, origin = src.celegans_image(well)
        if tif is None:
            print(f"  [celegans] no w2 image for {well}; run fetch_datasets.py --images celegans")
            return None, {}
        import tifffile
        raw = tifffile.imread(tif)
        img01, lo, hi = stretch(raw, CEL_STRETCH)
        img01, scale = resize01(img01, MAX_WIDTH)
        h, w = img01.shape
        png = f"{well}_w2_brightfield.png"
        to_png8(img01, out / png)

        worm_files, worm_origin = src.celegans_worms(well)
        overlays, failed, done, cleaned = [], [], [], []
        for wf in worm_files:
            wid = wf.name.replace("_ground_truth.png", "")
            res = worm_pipeline(read_mask(wf))
            worms_by_id[wid] = res
            if res["stray_removed"]:
                cleaned.append(f"{wid} ({res['stray_removed']} px)")
            outline = resample_by_arclength(res["outline"], OUTLINE_POINTS, closed=True)
            overlays.append({"kind": "outline", "points": pts_json(to_png_frame(outline, scale)), "closed": True,
                             "label": f"worm {wid}: outline of its ground-truth mask", "class": cid})
            if res["max_step"] <= MAX_WALK_STEP:
                overlays.append({"kind": "landmarks", "points": pts_json(to_png_frame(res["landmarks"], scale)),
                                 "closed": False, "label": f"worm {wid}: {N_LANDMARKS} landmarks (skeleton, arc length)",
                                 "class": cid})
                done.append(wid)
            else:
                failed.append(f"{wid} (the ordered skeleton jumps {res['max_step']:.0f} px: the greedy walk "
                              f"left the midline, so no landmarks are drawn)")
        worm_note = (f"Overlays from the {len(worm_files)} per-worm ground-truth masks of this well "
                     f"({worm_origin}): outline = shapekit.contours.mask_to_contours, resampled to {OUTLINE_POINTS} "
                     f"points; landmarks = skeletonize -> order_skeleton -> resample_by_arclength({N_LANDMARKS}) "
                     f"for {len(done)} of {len(worm_files)} worms")
        if cleaned:
            worm_note += ("; only the largest 8-connected component of each mask is used, which removed stray "
                          "pixels from " + ", ".join(cleaned))
        if failed:
            worm_note += "; not drawn: " + "; ".join(failed)
        if not worm_files:
            worm_note = "No per-worm ground-truth masks were available for this well, so there is no overlay"
        processing = (f"BBBC010 channel w2 (brightfield; the BBBC010 page calls channel 1 brightfield, but in the "
                      f"files w2 is the brightfield and w1 the GFP image), 16-bit TIFF {raw.shape[1]} x {raw.shape[0]}; "
                      f"linear contrast stretch from the {CEL_STRETCH[0]}th to the {CEL_STRETCH[1]}th percentile of the "
                      f"image ({lo:.0f} -> 0, {hi:.0f} -> 255), clipped, 8-bit PNG; no crop, no resize "
                      f"(scale {scale:g}). {worm_note}.")
        notes.append(processing)
        common = {"channel": "w2 (brightfield)", "class": cname, "class_id": cid, "well": well}
        images.append({"file": png, "alt": f"Brightfield image of BBBC010 well {well}: {what.split(': ')[1]}",
                       "caption": f"Well {well} ({cname} control), {what}", "width": w, "height": h, **common,
                       "source_file": tif.name, "source_archive": origin, "source_md5": md5_of(tif),
                       "overlays": overlays, "processing": processing})

        fg = src.celegans_foreground(well)
        if fg is not None:
            m = read_mask(fg)
            mpng = f"{well}_foreground_mask.png"
            to_png8(m.astype(float), out / mpng)
            images.append({"file": mpng, "alt": f"Human-corrected foreground mask of well {well}: worms white, background black",
                           "caption": f"Well {well}: BBBC010's human-corrected foreground mask (all worms in one binary image)",
                           "width": m.shape[1], "height": m.shape[0], **common, "channel": "ground truth (foreground)",
                           "source_file": fg.name, "source_archive": "BBBC010_v1_foreground.zip", "source_md5": md5_of(fg),
                           "overlays": overlays,
                           "processing": (f"BBBC010 foreground ground truth {fg.name}, thresholded at 127 and written as "
                                          f"0 / 255 8-bit PNG, same frame as the brightfield image; overlays as on "
                                          f"{png}.")})
    manifest = {
        "id": "celegans",
        "title": "C. elegans, live vs dead: two BBBC010 wells",
        "generated_by": SCRIPT,
        "coordinates": "overlay points are (x, y) in the PNG's pixel frame: x to the right, y down, pixel centers at integers",
        "images": images,
        "credit": {"title": "BBBC010: C. elegans infection live/dead image set, version 1",
                   "authors": "Fred Ausubel (Massachusetts General Hospital), via the Broad Bioimage Benchmark Collection "
                              "(Ljosa, Sokolnicki & Carpenter 2012)",
                   "url": "https://bbbc.broadinstitute.org/BBBC010", "license": "CC0 1.0",
                   "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "notice_file": "CREDIT.md",
                   "citation": BBBC010_CITATION},
    }
    dump_manifest(manifest, out / "manifest.json")
    (out / "CREDIT.md").write_text(celegans_credit(images), encoding="utf-8")
    return manifest, worms_by_id


def celegans_credit(images: list[dict]) -> str:
    rows = "\n".join(f"| `{im['file']}` | `{im['source_file']}` ({im['source_archive']}) | {im['well']} | {im['class']} |"
                     for im in images)
    return f"""# Credit: BBBC010 sample images

The files in this folder are modified copies of images from **BBBC010**, the *C. elegans* infection
live/dead image set of the Broad Bioimage Benchmark Collection, <https://bbbc.broadinstitute.org/BBBC010>:
the brightfield images were contrast-stretched and converted to 8-bit PNG, and the foreground masks were
thresholded at 127 and written as 0 / 255 8-bit PNGs. The outlines and landmarks in `manifest.json` are
derived data, computed from BBBC010's per-worm ground-truth masks (see the end of this file). They were
made by `{SCRIPT}`; `manifest.json` records every processing step and the source file of each image.

Channel: the BBBC010 page says "Channel 1 = brightfield; channel 2 = GFP", but in the files `w2` is the
brightfield image (bright well, dark worms) and `w1` the low-signal fluorescence one, so these samples use
`w2`.

## License

**CC0 1.0** (<https://creativecommons.org/publicdomain/zero/1.0/>). BBBC010's page: "Copyright: CC0. To the extent
possible under law, the various contributors of the imagesets have waived all copyright and related or
neighboring rights to BBBC010. This work is published from: United States."

## Citation (as BBBC010 requests)

{BBBC010_CITATION}

Ljosa, V., Sokolnicki, K. L. & Carpenter, A. E. (2012). Annotated high-throughput microscopy image sets
for validation. *Nature Methods* 9:637. The images were acquired for a screen in Fred Ausubel's lab at MGH
(Moy et al., *ACS Chemical Biology*, 2009).

## Files

| file | source | well | class |
|---|---|---|---|
{rows}

The overlays in `manifest.json` (outlines and 12 landmarks per worm) were computed by `shapekit` from
BBBC010's per-worm ground-truth masks (`BBBC010_v1_foreground_eachworm.zip`).
"""


# ------------------------------------------------------------------ MEF
def registration(points_rc: list[np.ndarray], dist: np.ndarray) -> dict:
    allp = np.vstack(points_rc)
    xy = allp[:, ::-1]                       # (row, col) -> (x, y)
    swapped = allp                           # the wrong reading, (row, col) taken as (x, y)
    inb = (swapped[:, 0] < dist.shape[1]) & (swapped[:, 1] < dist.shape[0])
    return {"median_px": round(float(np.median(sample_at(dist, xy))), 2),
            "p90_px": round(float(np.percentile(sample_at(dist, xy), 90)), 2),
            "median_px_if_axes_swapped": round(float(np.median(sample_at(dist, swapped[inb]))), 2) if inb.any() else None}


def build_mef(src: Sources, out: Path) -> dict | None:
    out.mkdir(parents=True, exist_ok=True)
    data_path = src.mef_tutorial_file("mef_data.npy")
    h5_path = src.mef_tutorial_file("xy001c1_instances.h5")
    mef = load_mef_data(data_path) if data_path else None
    import tifffile
    from scipy import ndimage
    images = []
    for stem, cid, cname, tut_folder, prefix, png, substrate in MEF_FIELDS:
        tif = src.mef_image(stem, tut_folder)
        if tif is None:
            print(f"  [mef] {stem}c1.tif not found; run fetch_datasets.py --images mef")
            return None
        raw = tifffile.imread(tif).astype(np.float64)
        img01, lo, hi = stretch(raw, MEF_STRETCH)
        small, scale = resize01(img01, MAX_WIDTH)
        h, w = small.shape
        to_png8(small, out / png)

        # Registration: distance from each outline point to the nearest edge of an Otsu mask of the
        # (lightly smoothed) actin image, and to the edges of the tutorial's instance mask (wild-type).
        smooth = ndimage.gaussian_filter(raw, 1.0)
        thr = otsu_threshold(smooth)
        otsu_dist = edge_distance(smooth > thr)
        keys = sorted((k for k in (mef or {}) if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1]))
        outlines = {k: np.asarray(mef[k], float) for k in keys}
        reg = {"otsu_threshold": round(thr, 1), "n_outlines": len(keys)}
        if keys:
            reg["vs_otsu_edges"] = registration(list(outlines.values()), otsu_dist)
        lab = None
        if h5_path is not None and tut_folder == "wildtype":
            import h5py
            with h5py.File(h5_path, "r") as f5:
                lab = np.asarray(f5["exported_data"])[..., 0]
            if keys:
                reg["vs_instance_mask_edges"] = registration(list(outlines.values()), edge_distance(lab))
        registered = bool(keys) and reg["vs_otsu_edges"]["median_px"] <= REGISTRATION_MAX_MEDIAN_PX
        reg["registered"] = registered
        reg["criterion"] = (f"median distance from the outline points to the nearest edge of an Otsu mask of the "
                            f"Gaussian-smoothed (sigma 1 px) actin image <= {REGISTRATION_MAX_MEDIAN_PX} px")

        overlays = []
        if registered:
            kept, dropped, loops = [], [], {}
            for k in keys:
                P = outlines[k][:, ::-1]                        # (row, col) -> (x, y)
                loops[k] = order_loop(P)
                twin = next((j for j in kept if covered_by(P, loops[j]) < COVERED_MEDIAN_PX), None)
                (dropped if twin else kept).append(k)
            for k in kept:
                P = loops[k]
                overlays.append({"kind": "outline", "points": pts_json(to_png_frame(P, scale)), "closed": True,
                                 "label": f"{cname} cell {k}: {len(P)} outline points from the tutorial's mef_data.npy",
                                 "class": cid})
            overlay_note = (f"Overlays: the {len(keys)} outlines that the tutorial's mef_data.npy holds for this field "
                            f"(keys {prefix}*, 120 unordered points each, (row, col) -> (x, y)); they register on this "
                            f"image (median distance to the actin edges {reg['vs_otsu_edges']['median_px']} px, "
                            f"{reg['vs_otsu_edges']['median_px_if_axes_swapped']} px with the axes swapped"
                            + (f"; {reg['vs_instance_mask_edges']['median_px']} px to the edges of the tutorial's "
                               f"xy001c1_instances.h5 mask" if "vs_instance_mask_edges" in reg else "")
                            + f"). The file lists most cells twice, a few px apart, and some cells also as parts "
                            f"(a cell with and without a long protrusion); {len(dropped)} outlines that lie on an outline "
                            f"with a lower key (median distance < {COVERED_MEDIAN_PX:g} px: keys "
                            f"{', '.join(k.rsplit('_', 1)[1] for k in dropped) or 'none'}) are not drawn, leaving "
                            f"{len(kept)}. Points are put in loop order for drawing (nearest-neighbor walk + 2-opt, shapekit.contours.order_loop); the "
                            f"Lab's PointNet never uses an order.")
        elif lab is not None:
            for v in [int(u) for u in np.unique(lab) if u != 0]:
                cs = mask_to_contours(lab == v)
                if not cs:
                    continue
                P = resample_by_arclength(max(cs, key=len), 120, closed=True)
                overlays.append({"kind": "outline", "points": pts_json(to_png_frame(P, scale)), "closed": True,
                                 "label": f"{cname} object {v}: outline traced from xy001c1_instances.h5", "class": cid})
            overlay_note = ("Overlays: mef_data.npy's outlines did not register on this image, so the outlines were "
                            "traced by shapekit.contours.mask_to_contours from the tutorial's xy001c1_instances.h5 "
                            "and resampled to 120 points.")
        else:
            overlay_note = "No overlay: mef_data.npy's outlines are missing or do not register, and no instance mask exists for this field."
        processing = (f"Micropattern_MEF_LMNA_Image {stem}c1.tif (phalloidin / actin channel c1; nuclei c2 not shown), "
                      f"16-bit TIFF {raw.shape[1]} x {raw.shape[0]}; linear contrast stretch from the {MEF_STRETCH[0]}th "
                      f"to the {MEF_STRETCH[1]}th percentile ({lo:.0f} -> 0, {hi:.0f} -> 255), clipped; resized by "
                      f"{scale:g} to {w} x {h} (Lanczos); 8-bit PNG; no crop. {overlay_note}")
        folder = str(Path(stem).parent)
        genotype = "LMNA++" if cid == 0 else "lamin A/C deficient"
        images.append({"file": png, "alt": f"Actin (phalloidin) fluorescence of {cname} mouse embryonic fibroblasts "
                                           f"({substrate}, original folder {folder}), field xy001",
                       "caption": f"{cname[0].upper() + cname[1:]} MEFs ({genotype}), original folder {folder} "
                                  f"({substrate}), field xy001, actin channel: the field the tutorial used. All {MEF_SAMPLED_PER_CLASS} sampled "
                                  f"{cname} fields of the tutorial come from this folder and all {MEF_SAMPLED_PER_CLASS} "
                                  f"sampled {'LMNA−/−' if cid == 0 else 'wild-type'} fields from "
                                  f"{MEF_OTHER_FOLDER[cid]}, so genotype is confounded with substrate and with whatever "
                                  f"else differs between the two folders.",
                       "width": w, "height": h, "channel": "c1 (phalloidin, actin)", "source_file": f"{stem}c1.tif",
                       "source_md5": md5_of(tif), "class": cname, "class_id": cid, "scale": scale,
                       "overlays": overlays, "registration": reg, "processing": processing})
    manifest = {
        "id": "mef",
        "title": "Mouse embryonic fibroblasts, wild-type vs LMNA−/−: the tutorial's two fields",
        "generated_by": SCRIPT,
        "coordinates": "overlay points are (x, y) in the PNG's pixel frame: x to the right, y down, pixel centers at "
                       "integers; original pixel (x, y) maps to ((x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5)",
        "images": images,
        "credit": {"title": "Micropattern_MEF_LMNA_Image", "authors": "Kyu Sang Han",
                   "url": "https://github.com/kukionfr/Micropattern_MEF_LMNA_Image", "license": "MIT",
                   "license_url": "https://opensource.org/licenses/MIT", "notice_file": "LICENSE",
                   "copyright": "Copyright (c) 2020 Kyu Sang Han",
                   "license_note": ("The repository's only license is this MIT software license, and its README names "
                                    "no separate license for the images; [Unverified] whether the author intended the "
                                    "MIT software license to cover the image data."),
                   "related_paper": MEF_PAPER,
                   "overlay_credit": "outlines from mef_data.npy of Shape Analysis 101 (Hugger & Uhlmann 2021, BSD-3), "
                                     "derived by the tutorial authors from these images"},
    }
    dump_manifest(manifest, out / "manifest.json")
    upstream = src.local / "mef" / "LICENSE"
    if upstream.exists() and upstream.read_text(encoding="utf-8").strip() != MIT_LICENSE.strip():
        raise SystemExit(f"{upstream} differs from the MIT text this script writes; check the upstream license")
    (out / "LICENSE").write_text(MIT_LICENSE, encoding="utf-8")
    return manifest


# ------------------------------------------------------------------ figure
def make_figure(src: Sources, worms: dict, path: Path) -> Path | None:
    import make_figures as mf  # SPEC identity: colors, rcParams, headline/caption helpers
    import matplotlib
    import tifffile
    if FIGURE_WORM not in worms:
        print(f"  [figure] worm {FIGURE_WORM} not available; figure not written")
        return None
    res = worms[FIGURE_WORM]
    tif, _ = src.celegans_image(FIGURE_WORM.split("_")[0])
    raw = tifffile.imread(tif)
    img01, _, _ = stretch(raw, CEL_STRETCH)
    worm_files, _ = src.celegans_worms(FIGURE_WORM.split("_")[0])
    mask = read_mask(next(p for p in worm_files if p.name.startswith(FIGURE_WORM)))
    lm = res["landmarks"]
    spline = BSplineCurve(N_LANDMARKS, closed=False).fit(lm)
    curve = spline.sample(300)
    sk_rows, sk_cols = np.nonzero(res["skeleton_mask"])
    sk = np.stack([sk_cols, sk_rows], 1) + res["skeleton_offset"]

    rows, cols = np.nonzero(mask)
    pad = 14
    cx, cy = (cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2
    half = max(cols.max() - cols.min(), rows.max() - rows.min()) / 2 + pad
    xlim, ylim = (cx - half, cx + half), (cy + half, cy - half)       # inverted y: image convention

    fig = mf.figure(1600, 520)
    axes = fig.subplots(1, 5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.74, bottom=0.12, wspace=0.06)
    ext = (-0.5, mask.shape[1] - 0.5, mask.shape[0] - 0.5, -0.5)
    faint = np.zeros(mask.shape + (4,))
    faint[mask] = matplotlib.colors.to_rgba(mf.SLATE, 0.22)
    solid = np.zeros(mask.shape + (4,))
    solid[mask] = matplotlib.colors.to_rgba(mf.SLATE)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(mf.LINE)
    axes[0].imshow(img01, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    axes[0].set_title("1  Brightfield image\n(BBBC010 well B07, w2)")
    axes[1].imshow(solid, extent=ext, interpolation="nearest")
    axes[1].set_title(f"2  Ground-truth mask\n(worm {FIGURE_WORM.split('_')[1]}, {res['area']} px)")
    axes[2].imshow(faint, extent=ext, interpolation="nearest")
    axes[2].plot(sk[:, 0], sk[:, 1], "s", color=mf.INK, ms=1.6, zorder=4)
    axes[2].set_title(f"3  Skeleton\n({res['n_skeleton']} pixels, {res['n_tips']} tips)")
    axes[3].imshow(faint, extent=ext, interpolation="nearest")
    axes[3].plot(res["walk"][:, 0], res["walk"][:, 1], color=mf.SLATE, lw=1.0, alpha=0.6, zorder=3)
    axes[3].plot(lm[:, 0], lm[:, 1], "o", color=mf.TEAL, ms=6.5, zorder=4)
    axes[3].plot(lm[0, 0], lm[0, 1], "o", color=mf.GOLD, ms=8.5, zorder=5)
    axes[3].set_title(f"4  {N_LANDMARKS} landmarks\n(equal arc length)")
    axes[4].imshow(img01, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest", alpha=0.55)
    axes[4].plot(curve[:, 0], curve[:, 1], color=mf.LAVENDER, lw=2.6, zorder=3)
    axes[4].plot(lm[:, 0], lm[:, 1], "o", color=mf.TEAL, ms=5, zorder=4)
    axes[4].set_title(f"5  Cubic B-spline\n(through {N_LANDMARKS} landmarks)")
    mf.headline(fig, "The chapter 01 pipeline on a real worm: image → mask → skeleton → landmarks → spline", x=0.01, y=0.94)
    mf.fit_text(fig, 0.5, 0.012, f"A live worm (well B07, ampicillin) from BBBC010 (CC0), with its ground-truth per-worm mask. "
                    f"shapekit.contours thins, orders and resamples the midline; gold = landmark 1 (which tip is first is arbitrary).",
                size=11, color=mf.MUTED, ha="center", va="bottom", width=0.96)   # SPEC: >= 11 pt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=fig.dpi, facecolor=mf.PAPER, metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", type=Path, default=LOCAL_IMAGES, help="fetched originals (default data/_local/images)")
    p.add_argument("--tutorial-dir", type=Path, default=None,
                   help="fallback: a Shape Analysis 101 data/ folder (C. elegans/, MEF_LMNA/)")
    p.add_argument("--out", type=Path, default=OUT, help="where the samples go (default docs/lab/data/images)")
    p.add_argument("--figure", type=Path, default=FIGURE, help="the chapter 01 real-pipeline figure")
    p.add_argument("--no-figure", action="store_true")
    a = p.parse_args(argv)
    src = Sources(a.source, a.tutorial_dir)
    ok = True
    print("[celegans]")
    cel, worms = build_celegans(src, a.out / "celegans")
    ok &= cel is not None
    print("[mef]")
    mef = build_mef(src, a.out / "mef")
    ok &= mef is not None
    if not a.no_figure and worms:
        fig = make_figure(src, worms, a.figure)
        print(f"[figure] {fig}")
    total = 0
    for man in (cel, mef):
        if not man:
            continue
        for im in man["images"]:
            f = a.out / man["id"] / im["file"]
            total += f.stat().st_size
            n_ov = len(im["overlays"])
            print(f"  {man['id']}/{im['file']}: {im['width']} x {im['height']}, {f.stat().st_size / 1e3:.0f} KB, {n_ov} overlays")
    print(f"  samples total {total / 1e6:.2f} MB (budget 2.5 MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
