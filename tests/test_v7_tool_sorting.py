"""Previously sorted tools must remain docked after later physical actions."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_v7_tool_sorting import run


def test_three_tools_are_sorted_without_resets_and_remain_in_their_slots(tmp_path):
    summary = run(tmp_path / "sorting", 0)
    assert summary["success"]
    assert summary["completed_tools"] == 3
    assert summary["resets_between_tools"] == 0
    assert len(summary["final_tools"]) == 3
    assert all(item["still_docked"] for item in summary["final_tools"].values())
    assert all(item["accepted"] and item["collision_free"] for item in summary["results"])
    assert not summary["source_changed_during_run"]
    for item in summary["results"]:
        physical = item["physical_metrics"]
        assert physical["strict_physical_success"] is True
        assert physical["stable_released_in_slot"] is True
        assert physical["release_evidence"]["stable_seconds"] >= 3.
        assert physical["release_evidence"]["final_target_hand_force_n"] <= .05
