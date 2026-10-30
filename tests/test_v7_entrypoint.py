from pathlib import Path
import json
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def test_v7_build_entrypoint_writes_compiled_scene_and_manifest(tmp_path):
    scene = tmp_path / "workbench.xml"
    manifest = tmp_path / "workbench.json"
    completed = subprocess.run(
        [PYTHON, str(PROJECT / "scripts/build_v7_workbench.py"),
         "--output", str(scene), "--manifest", str(manifest), "--seed", "9"],
        cwd=PROJECT, check=True, capture_output=True, text=True,
    )
    report = json.loads(completed.stdout)
    assert scene.is_file() and manifest.is_file()
    assert report["actionable_objects"] == ["tool_screwdriver", "tool_wrench", "tool_marker"]
    assert report["compiled_model"]["bodies"] >= 7
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["layout"]["seed"] == 9

