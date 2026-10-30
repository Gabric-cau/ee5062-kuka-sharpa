"""Exact native-clock baselines for offline probe object-height visualization."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_public_ycb_provisional_preload as plots


ATTEMPT = {"attempt_start_tick": 0, "attempt_end_tick": 4}


def observation(tick, *, active=True, start=.004):
    return {"timestamp_s": tick * .002, "probe_active": active, "probe_start_s": start}


def test_object_height_uses_actual_start_sample_instead_of_initial_or_first_active_height():
    heights = {1: (.002, .10), 2: (.004, .20), 3: (.006, .203), 4: (.008, .205)}
    result, series = plots.object_probe_motion(
        [observation(1, active=False, start=None), observation(3), observation(4)], ATTEMPT, heights)
    segment = result["probe_segments"][0]
    assert segment["baseline_native_tick"] == 2
    assert segment["baseline_native_timestamp_s"] == .004
    assert segment["baseline_free_joint_height_m"] == .20
    assert series["displacement_m"][1:] == pytest.approx([.003, .005])
    assert np.isnan(series["displacement_m"][0])
    assert segment["object_vertical_displacement_m"]["max"] == pytest.approx(.005)
    assert result["native_height_field"] == "target_qpos[2]"
    assert result["interpolated_or_extrapolated"] is False


@pytest.mark.parametrize("missing_tick", [2, 3])
def test_missing_start_or_active_native_sample_is_not_interpolated(missing_tick):
    heights = {2: (.004, .20), 3: (.006, .203)}
    del heights[missing_tick]
    with pytest.raises(ValueError, match="exact native start and observation ticks"):
        plots.object_probe_motion([observation(3)], ATTEMPT, heights)


def test_same_clock_final_probe_state_controls_curve_without_fabricating_admission_motion():
    heights = {2: (.004, .20), 3: (.006, .203)}
    result, series = plots.object_probe_motion(
        [observation(2, active=False, start=None), observation(2),
         observation(3), observation(3, active=False)], ATTEMPT, heights)
    assert series["time"] == [.004, .006]
    assert series["displacement_m"][0] == 0.
    assert np.isnan(series["displacement_m"][1])
    assert result["active_unique_observations"] == 1
    result, series = plots.object_probe_motion(
        [observation(2, active=False, start=None), observation(3, active=False, start=None)], ATTEMPT, {})
    assert result["recorded_probe_starts"] == result["active_unique_observations"] == 0
    assert result["probe_segments"] == []
    assert np.isnan(series["displacement_m"]).all()
