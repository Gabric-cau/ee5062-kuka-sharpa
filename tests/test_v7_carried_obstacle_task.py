"""A complete carried-tool route must survive actual integrated contact checks."""

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from run_v7_carried_obstacle_task import run


def test_carried_screwdriver_crosses_guard_and_is_released_stably(tmp_path):
    pytest.importorskip("ompl.base")
    output = tmp_path / "carried"
    result = run(output, seed=0)
    assert result["success"], result["acceptance_checks"]
    assert result["lift_evidence"]["accepted"]
    assert result["lift_evidence"]["sustained_lift_seconds"] >= .1
    assert result["dock_error_m"] <= .025
    assert result["task_spec"]["release_verify_seconds"] == 3.
    assert max(result["channel_peak_force_n"].values()) <= 12.
    assert result["direct_blocked_states"] > 0
    assert all(item["status"] == "exact_solution" for item in result["planner_records"])
    assert result["physical_tool_proxy_containment_failures"] == 0
    assert result["native_transfer_collision_events"] == 0
    assert result["native_non_hand_self_contact_events"] == 0
    assert result["live_object_qpos_writes"] == 0
    assert not result["source_changed_during_run"]
    with gzip.open(output / "physical_trace.jsonl.gz", "rt") as stream:
        trace = [json.loads(line) for line in stream]
    with gzip.open(output / "native_contacts.jsonl.gz", "rt") as stream:
        contacts = [json.loads(line) for line in stream]
    assert len(trace) == len(contacts) == result["native_contact_samples"]
    times = np.asarray([row["time_s"] for row in trace])
    assert np.allclose(np.diff(times), .002, atol=1e-10)
    assert [row["time_s"] for row in trace] == [row["time_s"] for row in contacts]
    carried = [row for row in trace if row["phase"] == "OMPL_TRANSFER"]
    assert len(carried) > 1000
    assert all(row["physical_tool_inside_declared_proxy"] for row in carried)
    assert all(row["target_hand_force_n"] >= .2 for row in carried)
    verification = [row for row in trace if row["phase"] == "VERIFY"]
    assert verification[-1]["time_s"] - verification[0]["time_s"] >= 3.
    assert all(row["target_hand_force_n"] <= .05 for row in verification)
    assert "TOOL_ALIGN" in {row["phase"] for row in trace}
