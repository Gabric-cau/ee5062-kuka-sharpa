import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_public_ycb_strict import channel_pressures, longest_interval


def test_missing_native_time_breaks_continuous_stability():
    assert longest_interval([0., .002, 3.1, 3.102], [True]*4) == pytest.approx(.002)
    assert longest_interval([0., .002, .004, .006], [True, True, False, True]) == pytest.approx(.002)


def test_contact_channels_sum_patches_and_exclude_internal_hand_contacts():
    def contact(a, b, force):
        return {"geom_ids": [a, b], "force_torque_contact_frame": [force, 0., 0., 0., 0., 0.]}
    pressures = channel_pressures([contact(1, 3, 7.), contact(1, 3, 6.), contact(1, 2, 40.)], {1: "thumb", 2: "index"})
    assert pressures["thumb"] == 13.
    assert pressures["index"] == 0.
    with pytest.raises(ValueError):
        channel_pressures([contact(1, 3, float("nan"))], {1: "thumb"})
