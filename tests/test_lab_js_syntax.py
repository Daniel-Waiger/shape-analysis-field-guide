"""Every JavaScript file under docs/ must parse.

The lab imports its engines statically, so one syntax error in any module leaves the
whole page on "Loading the lab..." forever.  ``node --check`` parses each file (ES
modules included, which Node detects from their import/export statements) without
running it.  Skipped when node is not installed.
"""

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
JS_FILES = sorted((ROOT / "docs").rglob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("path", JS_FILES, ids=[str(p.relative_to(ROOT)) for p in JS_FILES])
def test_js_parses(path: pathlib.Path):
    proc = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"{path.relative_to(ROOT)} does not parse:\n{proc.stderr}"


def test_js_files_found():
    assert any(p.name == "lab.js" for p in JS_FILES), "docs/lab/lab.js should exist"
