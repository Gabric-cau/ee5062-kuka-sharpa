"""Wrist-RGBD evaluation must retain contact evidence and metric depth."""

import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_v7_rgbd import run


def test_rgbd_episode_has_a_complete_physical_success_denominator(tmp_path):
    output = tmp_path / "evaluation"
    run(output, 450, 1, ("rgbd_shared",))
    summary = json.loads((output / "summary.json").read_text())
    result = summary["results"][0]["metrics"]
    assert summary["complete"]
    assert result["unknown_reasons"] == []
    assert result["lift_evidence"]["reasons"] == []
    assert result["strict_physical_success"] is True
    assert result["evidence_samples"] == result["expected_evidence_samples"]
    observation = output / "seed_450" / "rgbd_shared"
    assert (observation / "wrist_depth.png").stat().st_size > 0
    assert np.isfinite(np.load(observation / "wrist_depth_m.npy", allow_pickle=False)).any()
