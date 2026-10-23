"""The wheel the Lab's Pyodide panel installs must be built from the current sources.

``docs/lab/py/shapekit-*.whl`` is produced by ``scripts/build_lab_wheel.sh``;
setuptools stamps file times into the zip, so the wheel is not byte-reproducible
and this compares its members with the tree, not its hash.
"""

import pathlib
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WHEELS = sorted((ROOT / "docs" / "lab" / "py").glob("shapekit-*-py3-none-any.whl"))


@pytest.mark.skipif(not WHEELS, reason="no wheel built yet (bash scripts/build_lab_wheel.sh)")
def test_wheel_members_match_the_tree():
    assert len(WHEELS) == 1, f"exactly one wheel is served: {[w.name for w in WHEELS]}"
    tree = {p.name: p.read_bytes() for p in (ROOT / "shapekit").glob("*.py")}
    with zipfile.ZipFile(WHEELS[0]) as z:
        members = {n for n in z.namelist() if n.startswith("shapekit/") and n.endswith(".py")}
        assert {m.split("/", 1)[1] for m in members} == set(tree), "the wheel's module list differs from shapekit/"
        for member in sorted(members):
            name = member.split("/", 1)[1]
            assert z.read(member) == tree[name], f"{name} in the wheel differs from shapekit/{name}; rebuild the wheel"
        wheel_meta = next(n for n in z.namelist() if n.endswith(".dist-info/WHEEL"))
        text = z.read(wheel_meta).decode()
        assert "Root-Is-Purelib: true" in text and "py3-none-any" in text
        assert not any(n.endswith((".so", ".pyd", ".dylib")) for n in z.namelist())
