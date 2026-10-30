"""Physical paired regression for supported finger rolling."""

from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from run_v7_supported_rolling import INDEX_PUSH, run_arm


def test_index_push_rolls_marker_with_fixed_wrist_and_table_support(tmp_path):
    control = run_arm(tmp_path / "control", override=False)
    push = run_arm(tmp_path / "push", override=True)
    assert push["success"] and not control["success"]
    assert push["failure_reason"] is None and control["failure_reason"] is None
    assert push["sample_count"] == control["sample_count"] == 751
    assert push["live_object_qpos_writes"] == control["live_object_qpos_writes"] == 0
    assert push["in_air_regrasp_claim"] is False
    assert push["regrasp_evidence"]["relative_rotation_deg"] > 30.
    assert control["regrasp_evidence"]["relative_rotation_deg"] < 2.
    assert push["regrasp_evidence"]["continuous_support_contact"]
    assert push["regrasp_evidence"]["continuous_hand_contact"]
    assert push["regrasp_evidence"]["max_object_drop_m"] < .001
    assert push["regrasp_evidence"]["wrist_translation_m"] < .002
    assert push["fixed_wrist_arm_goal"] == push["final_wrist_arm_goal"]
    assert np.allclose(push["initial_rolling_snapshot"]["arm_joint_position"],
                       control["initial_rolling_snapshot"]["arm_joint_position"])
    assert {row["joint"]: row["target_rad"] for row in push["joint_target_commands"]} == INDEX_PUSH
    assert control["joint_target_commands"] == []
    assert push["force_safe_12n"]
    assert (tmp_path / "push" / "native_contacts.jsonl.gz").stat().st_size > 0
