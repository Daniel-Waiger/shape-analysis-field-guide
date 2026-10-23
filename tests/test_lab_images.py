"""Round 3 sample images: docs/lab/data/images/<id>/ (manifests, files, budget, licences, determinism)
and the offline parts of ``fetch_datasets.py --images``."""

import hashlib
import http.server
import importlib.util
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "lab" / "data" / "images"
LOCAL = ROOT / "data" / "_local" / "images"
SAMPLE_IDS = ("celegans", "mef")
BUDGET = 2_500_000
MAX_WIDTH = 900
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CLASSES = {"celegans": {"dead", "live"}, "mef": {"wild-type", "LMNA−/−"}}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp"}


def manifest(ds):
    return json.loads((IMAGES / ds / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("ds", SAMPLE_IDS)
def test_manifest_matches_contract(ds):
    m = manifest(ds)
    assert m["id"] == ds
    assert isinstance(m["images"], list) and m["images"]
    for im in m["images"]:
        for key in ("file", "alt", "caption", "width", "height", "channel", "source_file", "class", "overlays",
                    "processing"):
            assert key in im, (ds, im.get("file"), key)
        for key in ("file", "alt", "caption", "channel", "source_file", "processing"):
            assert isinstance(im[key], str) and im[key].strip(), (ds, im["file"], key)
        assert isinstance(im["width"], int) and isinstance(im["height"], int)
        assert im["class"] in CLASSES[ds]
        assert isinstance(im["overlays"], list)
        for ov in im["overlays"]:
            assert ov["kind"] in ("outline", "landmarks")
            assert isinstance(ov["closed"], bool) and isinstance(ov["label"], str) and ov["label"]
            assert len(ov["points"]) >= 2 and all(len(p) == 2 for p in ov["points"])
    c = m["credit"]
    for key in ("title", "authors", "url", "license", "license_url", "notice_file"):
        assert isinstance(c[key], str) and c[key], key
    assert (IMAGES / ds / c["notice_file"]).is_file()


@pytest.mark.parametrize("ds", SAMPLE_IDS)
def test_files_are_small_8bit_pngs_matching_the_manifest(ds):
    from PIL import Image
    for im in manifest(ds)["images"]:
        path = IMAGES / ds / im["file"]
        assert path.is_file(), path
        assert path.read_bytes()[:8] == PNG_MAGIC, path
        with Image.open(path) as img:
            assert img.format == "PNG"
            assert img.size == (im["width"], im["height"])
            assert img.mode in ("L", "RGB", "RGBA", "P"), img.mode        # 8 bits per channel
        assert im["width"] <= MAX_WIDTH


def test_samples_within_budget():
    total = sum(p.stat().st_size for p in IMAGES.rglob("*") if p.is_file())
    assert total <= BUDGET, f"{total:,} bytes in {IMAGES}"


@pytest.mark.parametrize("ds", SAMPLE_IDS)
def test_overlay_points_inside_the_image(ds):
    for im in manifest(ds)["images"]:
        w, h = im["width"], im["height"]
        for ov in im["overlays"]:
            for x, y in ov["points"]:
                assert -0.5 <= x <= w - 0.5 and -0.5 <= y <= h - 0.5, (im["file"], ov["label"], x, y)


def test_every_worm_image_carries_landmarks_and_mef_outlines_registered():
    cel = manifest("celegans")
    bright = [im for im in cel["images"] if im["file"].endswith("_brightfield.png")]
    assert {im["class"] for im in bright} == {"dead", "live"}
    for im in bright:
        lms = [ov for ov in im["overlays"] if ov["kind"] == "landmarks"]
        assert lms and all(len(ov["points"]) == 12 and not ov["closed"] for ov in lms)
    mef = manifest("mef")
    assert {im["class"] for im in mef["images"]} == {"wild-type", "LMNA−/−"}
    for im in mef["images"]:
        assert im["overlays"], im["file"]
        reg = im.get("registration", {})
        if reg.get("registered"):
            assert reg["vs_otsu_edges"]["median_px"] <= 3.0
            assert "mef_data.npy" in im["processing"]
        else:
            assert "xy001c1_instances.h5" in im["processing"]


def test_licence_and_credit_files():
    lic = (IMAGES / "mef" / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License")
    assert "Copyright (c) 2020 Kyu Sang Han" in lic
    assert "The above copyright notice and this permission notice shall be included" in lic
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in lic
    credit = (IMAGES / "celegans" / "CREDIT.md").read_text(encoding="utf-8")
    assert "CC0" in credit
    assert "Broad Bioimage Benchmark Collection" in credit and "Ljosa" in credit
    assert "provided by Fred Ausubel" in credit


def test_no_restricted_images_under_docs():
    assert not (IMAGES / "rbc").exists() and not (IMAGES / "leaves").exists()
    pattern = re.compile(r"(rbc|leaf|leaves|erythro|swedish|^l\d+nr\d+)", re.IGNORECASE)
    bad = [p for p in (ROOT / "docs").rglob("*")
           if p.is_file() and p.suffix.lower() in IMAGE_EXT and pattern.search(str(p.relative_to(ROOT / "docs")))]
    assert not bad, bad
    # only the two redistributable sets have an images folder
    assert {p.name for p in IMAGES.iterdir() if p.is_dir()} <= set(SAMPLE_IDS)


def test_figure_exists():
    from PIL import Image
    path = ROOT / "docs" / "figures" / "01_real_pipeline.png"
    with Image.open(path) as img:
        assert img.size[0] == 1600


# ------------------------------------------------------------------ determinism (needs the fetched originals)
def _sources_present():
    worms = LOCAL / "celegans" / "foreground_eachworm" / "BBBC010_v1_foreground_eachworm"
    need = [LOCAL / "celegans" / "images", worms / "B07_06_ground_truth.png",
            LOCAL / "mef" / "Control" / "LMNA++1" / "xy001c1.tif", LOCAL / "mef" / "Circle" / "LMNA--1" / "xy001c1.tif",
            LOCAL / "mef" / "tutorial" / "mef_data.npy", LOCAL / "mef" / "tutorial" / "xy001c1_instances.h5"]
    return all(p.exists() for p in need)


def _run(out: Path) -> dict:
    subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "make_sample_images.py"),
                    "--out", str(out / "images"), "--figure", str(out / "fig.png")],
                   check=True, cwd=ROOT, capture_output=True, timeout=300)
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file()}


@pytest.mark.skipif(not _sources_present(),
                    reason="the original images are not fetched: run `python scripts/fetch_datasets.py --images "
                           "celegans mef` (data/_local/images/, git-ignored) to enable this test")
def test_make_sample_images_is_deterministic_and_matches_the_shipped_manifests(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a == b
    for ds in SAMPLE_IDS:
        got = (tmp_path / "a" / "images" / ds / "manifest.json").read_text(encoding="utf-8")
        assert json.loads(got) == manifest(ds), f"docs/lab/data/images/{ds}/manifest.json is stale: re-run the script"
        for name in ("LICENSE", "CREDIT.md"):
            shipped = IMAGES / ds / name
            if shipped.exists():
                assert (tmp_path / "a" / "images" / ds / name).read_bytes() == shipped.read_bytes()


# ------------------------------------------------------------------ fetch_datasets.py --images (offline parts)
@pytest.fixture(scope="module")
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_datasets_images", ROOT / "scripts" / "fetch_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_images_go_under_the_ignored_local_folder(fetch):
    assert fetch.IMAGES_DIR == ROOT / "data" / "_local" / "images"
    assert fetch.gitignore_covers((ROOT / ".gitignore").read_text(encoding="utf-8"), "data/_local")


def test_species_parsing(fetch):
    assert fetch.parse_species("1") == [1]
    assert fetch.parse_species("7, 5") == [5, 7]
    assert fetch.parse_species("all") == list(range(1, 16))
    for bad in ("0", "16", "x", ""):
        with pytest.raises(SystemExit):
            fetch.parse_species(bad)


def test_mef_candidates_start_with_the_tutorial_fields(fetch):
    c = fetch.mef_candidates()
    assert c[:2] == list(fetch.MEF_TUTORIAL_FIELDS)
    assert len(c) == len(set(c))
    assert c[2].endswith("xy001") and len(fetch.MEF_FOLDERS) == 12
    # the default subset stays under the size cap: 40 files of ~2.67 MB
    assert 20 * 2 * 2_667_154 < fetch.MEF_MAX_BYTES


class PointerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.endswith("present.tif"):
            body = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"ab" * 32 + b"\nsize 1234\n"
            self.send_response(200)
        elif self.path.endswith("notlfs.tif"):
            body = b"II*\x00 not a pointer"
            self.send_response(200)
        else:
            body = b"404: Not Found"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_lfs_pointer_parsing(fetch, monkeypatch):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), PointerHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fetch, "MEF_RAW", f"http://127.0.0.1:{httpd.server_address[1]}/")
        assert fetch._lfs_pointer("Control/LMNA++1/present.tif") == ("ab" * 32, 1234)
        assert fetch._lfs_pointer("Control/LMNA++1/absent.tif") is None
        with pytest.raises(SystemExit):
            fetch._lfs_pointer("Control/LMNA++1/notlfs.tif")
    finally:
        httpd.shutdown()
