"""Oracle labels only: never supply a pose to the task planner."""
import math
import mujoco
import numpy as np


class PlacementEvaluator:
    def __init__(self, model, zone):
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        self.zone = np.asarray(zone)
        self.initial_height = None
        self.stable_since = None
        self.target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
        self.table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
        self.hand_geoms = [i for i in range(model.ngeom)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or '').startswith('left_')
                           and model.geom_contype[i] != 0]

    def clearance(self, model, data):
        return min(mujoco.mj_geomDistance(model, data, self.target_geom, gid, 1., None)
                   for gid in self.hand_geoms)

    def sample(self, model, data, hand, released):
        position = data.xpos[self.body].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.body, velocity, 0)
        if self.initial_height is None:
            self.initial_height = float(position[2])
        error = float(np.linalg.norm(position[:2] - self.zone[:2]))
        linear_speed = float(np.linalg.norm(velocity[3:]))
        angular_speed = float(np.linalg.norm(velocity[:3]))
        clearance = self.clearance(model, data)
        pairs = [{int(c.geom1), int(c.geom2)} for c in data.contact[:data.ncon]]
        hand_contact = any(self.target_geom in p and bool(p & set(self.hand_geoms)) for p in pairs)
        table_support = False
        wrench = np.zeros(6)
        for i, pair in enumerate(pairs):
            if pair == {self.target_geom, self.table_geom}:
                mujoco.mj_contactForce(model, data, i, wrench)
                table_support |= abs(wrench[0]) > .05
        disengaged = not hand_contact and clearance > .01
        retreat = clearance > .10
        valid = (released and error < .045 and abs(position[2] - .38) < .025
                 and linear_speed < .04 and angular_speed < .2 and retreat
                 and disengaged and table_support)
        if not valid:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = float(data.time)
        return dict(position=position.tolist(), placement_error=error,
                    linear_speed_m_s=linear_speed, angular_speed_rad_s=angular_speed,
                    hand_clearance_m=clearance, hand_contact=hand_contact,
                    table_support=bool(table_support), disengaged=disengaged,
                    retreat=retreat, release_commanded=released, released=bool(released and disengaged),
                    lifted=bool(position[2] > self.initial_height + .07),
                    success=bool(valid and data.time - self.stable_since >= .5))


def held_lift_evidence(trace, min_continuous=.1, maximum_sample_gap=.02):
    """Evaluation-only proof of sustained object lift, not wrist motion.

    The oracle's lifted label requires the actual object to rise over 70 mm.
    A qualifying interval also requires hand contact and no table support.
    Missing labels and gaps cannot establish continuous physical support.
    """
    if (not math.isfinite(min_continuous) or min_continuous <= 0
            or not math.isfinite(maximum_sample_gap) or maximum_sample_gap <= 0):
        raise ValueError('positive finite lift duration and maximum sample gap are required')
    previous_time = None
    held_since = None
    longest = 0.
    for row in trace:
        try:
            timestamp = float(row['timestamp'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('a finite nonnegative lift-evidence timestamp is required') from error
        if (not math.isfinite(timestamp) or timestamp < 0
                or previous_time is not None and timestamp < previous_time):
            raise ValueError('lift-evidence timestamps must be finite, nonnegative, and nondecreasing')
        if previous_time is not None and timestamp - previous_time > maximum_sample_gap + 1e-12:
            held_since = None
        labels = row.get('oracle') or {}
        qualified = (labels.get('lifted') is True and labels.get('hand_contact') is True
                     and labels.get('table_support') is False)
        if qualified:
            if held_since is None:
                held_since = timestamp
            longest = max(longest, timestamp - held_since)
        else:
            held_since = None
        previous_time = timestamp
    return dict(observed=bool(longest + 1e-12 >= min_continuous), held_seconds=longest,
                minimum_required_seconds=min_continuous,
                maximum_sample_gap_seconds=maximum_sample_gap,
                criteria='actual object >70 mm above initial height, hand contact, no table support')
