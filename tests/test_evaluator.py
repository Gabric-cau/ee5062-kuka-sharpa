import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_sharpa_grasp.evaluator import FailureReason, TrialStage, evaluate_trial


def test_success_requires_final_contact_after_lift():
    result = evaluate_trial(max_contacts=2, final_contacts=1, object_height=0.60)
    assert result.status == "success"
    assert result.stage is TrialStage.HOLD
    assert result.failure_reason is FailureReason.NONE


def test_contact_without_lift_is_reported_as_incomplete():
    result = evaluate_trial(max_contacts=2, final_contacts=2, object_height=0.52)
    assert result.status == "contact_only"
    assert result.stage is TrialStage.GRASP_SETTLE
    assert result.failure_reason is FailureReason.INSUFFICIENT_LIFT


def test_drop_and_no_contact_have_distinct_reasons():
    dropped = evaluate_trial(max_contacts=2, final_contacts=0, object_height=0.40)
    empty = evaluate_trial(max_contacts=0, final_contacts=0, object_height=0.40)
    assert (dropped.status, dropped.failure_reason) == ("dropped", FailureReason.DROP)
    assert (empty.status, empty.failure_reason) == ("no_contact", FailureReason.NO_CONTACT)
