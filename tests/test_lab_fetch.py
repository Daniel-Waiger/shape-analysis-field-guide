"""scripts/fetch_datasets.py: the parts that guard the repository (no network: a local HTTP server)."""

import hashlib
import http.server
import importlib.util
import pathlib
import threading
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_datasets", ROOT / "scripts" / "fetch_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves one payload with HTTP Range support (206 / 416), like the Zenodo download."""

    payload = b""

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        total = len(self.payload)
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= total:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.end_headers()
                return
            body = self.payload[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
        else:
            body = self.payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    RangeHandler.payload = bytes(range(256)) * 40
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.bin", RangeHandler.payload
    httpd.shutdown()


def test_gitignore_guard_is_a_real_line_match(fetch):
    assert fetch.gitignore_covers("data/_local/\n")
    assert fetch.gitignore_covers("*.pyc\n/data/_local\n")
    assert not fetch.gitignore_covers("# data/_local/\n")
    assert not fetch.gitignore_covers("!data/_local/\n")
    assert not fetch.gitignore_covers("data/_local_old/\n")
    assert not fetch.gitignore_covers("")
    assert fetch.gitignore_covers((ROOT / ".gitignore").read_text(encoding="utf-8")), ".gitignore must ignore data/_local/"


def test_download_verifies_and_deletes_on_mismatch(fetch, server, tmp_path, monkeypatch):
    url, payload = server
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    good = hashlib.md5(payload).hexdigest()
    dest = tmp_path / "file.bin"
    assert fetch.download(url, dest, md5=good) == dest and dest.read_bytes() == payload
    dest.unlink()
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch.download(url, dest, md5="0" * 32, attempts=1)
    assert not dest.exists(), "a file with the wrong checksum must not stay on disk"
    assert not dest.with_suffix(".bin.part").exists()


def test_download_resume_and_complete_part(fetch, server, tmp_path, monkeypatch):
    url, payload = server
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    good = hashlib.md5(payload).hexdigest()
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    # a partial .part is continued with a Range request
    part.write_bytes(payload[:1000])
    assert fetch.download(url, dest, md5=good, resume=True) == dest and dest.read_bytes() == payload
    dest.unlink()
    # a .part that already holds the whole file gets HTTP 416: it is promoted and verified, not retried to death
    part.write_bytes(payload)
    assert fetch.download(url, dest, md5=good, resume=True, attempts=1) == dest and dest.read_bytes() == payload
    assert not part.exists()
    dest.unlink()
    # ... and a complete .part with the wrong content is deleted
    part.write_bytes(payload[:-1] + b"x")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch.download(url, dest, md5=good, resume=True, attempts=1)
    assert not dest.exists()


def test_safe_extract_zip_validates_before_writing(fetch, tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("ok.txt", "fine")
        z.writestr("../escape.txt", "evil")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="escapes"):
        fetch.safe_extract_zip(bad, out)
    assert not (out / "ok.txt").exists(), "nothing is written before every entry has been validated"
    assert not (tmp_path / "escape.txt").exists()
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("a/b.txt", "x")
        z.writestr("__MACOSX/junk", "y")
    assert fetch.safe_extract_zip(good, out) == 1 and (out / "a" / "b.txt").read_text() == "x"
