import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_site_data_writes_the_two_json_files(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_site_data.py"), "--out", str(tmp_path)],
        check=True,
    )
    worms = json.loads((tmp_path / "worms.json").read_text())
    assert len(worms["mean"]) == 12 and len(worms["mean"][0]) == 2
    assert len(worms["examples"]["dead"]) == 3 and len(worms["examples"]["live"]) == 3
    assert len(worms["pca"]["mean_tangent"]) == 24
    assert len(worms["pca"]["components"]) == 4 and len(worms["pca"]["components"][0]) == 24
    assert len(worms["pca"]["sdev"]) == 4 and len(worms["pca"]["explained"]) == 10
    assert set(worms["class_means"]) == {"dead", "live"}
    spline = json.loads((tmp_path / "spline.json").read_text())
    assert len(spline["landmarks"]) == 20 and len(spline["control_points"]) == 6
    # floats are rounded to 5 decimals
    assert all(round(v, 5) == v for row in spline["landmarks"] for v in row)
    # running twice gives identical output (seeded)
    again = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_site_data.py"), "--out", str(tmp_path / "b")],
        check=True,
    )
    assert again.returncode == 0
    assert (tmp_path / "b" / "worms.json").read_text() == (tmp_path / "worms.json").read_text()
