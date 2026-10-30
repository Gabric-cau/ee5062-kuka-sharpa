"""Small, deterministic evaluation layer for grasp trials.

The simulator still has no place phase, so a successful trial currently ends
in ``HOLD``.  Keeping the vocabulary here makes that limitation explicit and
gives later recovery/place controllers a stable result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrialStage(str, Enum):
    """Last meaningful stage reached by a rollout."""

    APPROACH = "approach"
    CONTACT = "contact"
    GRASP_SETTLE = "grasp_settle"
    LIFT = "lift"
    HOLD = "hold"
    FAILED = "failed"


class FailureReason(str, Enum):
    """Machine-readable reason for a non-successful result."""

    NONE = "none"
    NO_CONTACT = "no_contact"
    INSUFFICIENT_LIFT = "insufficient_lift"
    DROP = "drop"


@dataclass(frozen=True)
class TrialEvaluation:
    """Normalized outcome independent of MuJoCo data structures."""

    status: str
    stage: TrialStage
    failure_reason: FailureReason


def evaluate_trial(
    *,
    max_contacts: int,
    final_contacts: int,
    object_height: float,
    lift_threshold: float = 0.55,
    drop_height: float = 0.50,
) -> TrialEvaluation:
    """Classify a rollout from contact and final-pose signals.

    ``success`` requires contact at the end of the lift, not merely a contact
    that happened earlier.  The thresholds are explicit arguments so later
    object-scale or placement experiments can report their assumptions.
    """

    if object_height > lift_threshold and final_contacts > 0:
        return TrialEvaluation("success", TrialStage.HOLD, FailureReason.NONE)
    if max_contacts == 0:
        return TrialEvaluation("no_contact", TrialStage.CONTACT, FailureReason.NO_CONTACT)
    if object_height < drop_height:
        return TrialEvaluation("dropped", TrialStage.LIFT, FailureReason.DROP)
    return TrialEvaluation(
        "contact_only", TrialStage.GRASP_SETTLE, FailureReason.INSUFFICIENT_LIFT
    )
