"""Finite known-can planner. Inputs are beliefs and robot/contact observations only."""
from enum import Enum
import numpy as np


class State(str, Enum):
    SEARCH = 'SEARCH'
    APPROACH = 'APPROACH'
    ALIGN = 'ALIGN'
    CLOSE = 'CLOSE'
    TRIAL_LIFT = 'TRIAL_LIFT'
    TRANSFER = 'TRANSFER'
    PLACE = 'PLACE'
    RELEASE = 'RELEASE'
    VERIFY = 'VERIFY'
    RECOVER = 'RECOVER'
    ABORT = 'ABORT'


def candidates(belief, now):
    reasons = []
    if belief is None or belief.centroid is None or belief.confidence < .15:
        return [], ['NO_BELIEF']
    if now - belief.timestamp > .25:
        return [], ['STALE_BELIEF']
    if belief.object_kind != 'ycb_can':
        return [], ['UNCALIBRATED_OBJECT']
    # Partial front surface bias, calibrated for the upright 005 can and wrist view.
    center = belief.centroid.copy()
    center[:2] += [-.005, .008]
    center[2] = .38
    result = []
    for dx in (0., -.008, .008):
        point = center + [dx, 0., 0.]
        if not (.55 < point[0] < .85 and abs(point[1]) < .2):
            reasons.append('WORKSPACE')
        else:
            result.append(point)
    return result, reasons


class TaskController:
    def __init__(self):
        self.state = None
        self.transitions = []
        self.failure = None

    def enter(self, state, timestamp, reason='none'):
        self.transitions.append(dict(previous=self.state, state=state.value,
                                     timestamp=float(timestamp), reason=reason))
        self.state = state.value

    def fail(self, timestamp, reason):
        self.failure = reason
        self.enter(State.RECOVER, timestamp, reason)

    def accept_lift(self, timestamp, contacts, lifted):
        if contacts == 0 or not lifted:
            self.fail(timestamp, 'FAILED_TRIAL_LIFT')
            return False
        return True


def interpolate(start, end, alpha):
    alpha = np.clip(alpha, 0., 1.)
    return start + (end - start) * (3 * alpha**2 - 2 * alpha**3)


def completed_success(transitions, physical_success, failure):
    required = [state.value for state in (State.APPROACH, State.ALIGN, State.CLOSE,
                State.TRIAL_LIFT, State.TRANSFER, State.PLACE, State.RELEASE, State.VERIFY)]
    visited = [item['state'] for item in transitions]
    index = 0
    for state in visited:
        if index < len(required) and state == required[index]:
            index += 1
    return bool(physical_success and failure is None and index == len(required))
