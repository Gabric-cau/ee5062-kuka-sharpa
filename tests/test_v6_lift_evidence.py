"""A completed wrist trajectory cannot substitute for actual held object lift."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuka_sharpa_grasp.pick_place_evaluator import held_lift_evidence


def row(timestamp, *, lifted=True, hand_contact=True, table_support=False):
    return dict(timestamp=timestamp, wrist_position=[.65, -.18, .85],
                oracle=dict(lifted=lifted, hand_contact=hand_contact,
                            table_support=table_support))


def test_wrist_lift_and_final_placement_never_substitute_for_object_lift():
    trace = [row(index * .002, lifted=False) for index in range(100)]
    trace[-1]["oracle"].update(success=True, released=True)
    evidence = held_lift_evidence(trace)
    assert not evidence["observed"]
    assert evidence["held_seconds"] == 0


def test_continuous_off_table_object_lift_with_hand_contact_is_accepted():
    trace = [row(index * .002) for index in range(61)]
    evidence = held_lift_evidence(trace)
    assert evidence["observed"]
    assert evidence["held_seconds"] == pytest.approx(.12)


@pytest.mark.parametrize("interruption", ["lifted", "hand_contact", "table_support"])
def test_interrupted_support_cannot_sum_disconnected_hold_intervals(interruption):
    trace = [row(index * .002) for index in range(91)]
    trace[45]["oracle"][interruption] = interruption == "table_support"
    evidence = held_lift_evidence(trace)
    assert not evidence["observed"]
    assert evidence["held_seconds"] == pytest.approx(.088)


def test_sparse_samples_cannot_certify_an_unobserved_continuous_lift():
    trace = [row(0), row(.15), row(.16), row(.17)]
    evidence = held_lift_evidence(trace)
    assert not evidence["observed"]
    assert evidence["held_seconds"] == pytest.approx(.02)


def test_duplicate_endpoint_records_do_not_create_extra_hold_duration():
    trace = [row(0), *[row(.002) for _ in range(100)]]
    evidence = held_lift_evidence(trace)
    assert not evidence["observed"]
    assert evidence["held_seconds"] == pytest.approx(.002)


@pytest.mark.parametrize("trace", [
    [row(.04), row(.02)], [row(0), row(float("nan"))],
    [row(0), row(float("inf"))], [row(-.002), row(0)],
])
def test_invalid_trace_time_is_rejected(trace):
    with pytest.raises(ValueError, match="timestamp"):
        held_lift_evidence(trace)


def test_missing_oracle_labels_remain_unknown_not_success():
    trace = [dict(timestamp=index * .002, oracle={}) for index in range(100)]
    assert not held_lift_evidence(trace)["observed"]
    assert not held_lift_evidence([])["observed"]
