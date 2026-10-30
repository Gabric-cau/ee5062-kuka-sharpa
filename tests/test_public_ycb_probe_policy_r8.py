"""Probe-policy contracts without physics integration or device jobs."""

import ast
from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_force_band_r8 as r8


def fixture():
    hand = object.__new__(r8.LoadTransferProbeHand)
    hand.profile = {"contact_closure": {"thumb": .4, "index": 0., "middle": .3, "ring": .3, "pinky": 0.}}
    hand.estimated = SimpleNamespace(pose_timestamp_s=0., mass_kg=.349,
        kin=SimpleNamespace(body=0), payload_confirmed=False, probe_active=False)
    hand.pressures = dict.fromkeys((*r8.r4.FINGERS, "palm"), 0.)
    for finger in ("thumb", "middle", "ring"):
        hand.pressures[finger] = .8
    raw = [{"finger": finger, "hand_body_name": finger + "_link",
        "position_world_m": [0., 0., 0.], "normal_on_hand_world": [1., 0., 0.], "pressure_n": .8}
        for finger in ("thumb", "middle", "ring")]
    hand.reader = SimpleNamespace(escape=None, raw_contacts=raw)
    hand.reader.read = lambda data: {key: SimpleNamespace(normal=value) for key, value in hand.pressures.items()}
    hand.probe_association = {"accepted": True, "patches": [{"finger": row["finger"],
        "measured_pressure_n": row["pressure_n"]} for row in raw]}
    hand.probe_geometry_15 = {"feasible": True, "measured_feasible": False,
        "measured_capacity_feasible": False}
    hand.probe_allocation_1g = {"measured_capacity_feasible": True, "measured_feasible": False}
    hand.probe_solved_sample = hand.contact_sample(raw)
    hand.last_probe_solve_s = 0.
    hand.last_probe_visual_s = -np.inf
    hand.emergency_active = hand.emergency_phase_hold = hand.probe_active = False
    hand.probe_stage = "ACQUIRING"
    hand.probe_eligible_seconds = hand.probe_confirmed_seconds = 0.
    hand.pressure_window = deque()
    hand.probe_start_world_m = None
    hand.probe_start_s = None
    hand.probe_required_fingers = set()
    hand.last_wrist_position = None
    hand.last_wrist_timestamp_s = None
    hand.actual_wrist_speed_m_s = 0.
    hand.latest_probe = {}
    hand.probe_events = []
    hand.allocation = {"measured_feasible": False}
    hand.ready_seconds = 0.
    hand.phase = "CLOSE"
    hand.table_load = .349 * 9.81
    hand.table_support = lambda data: hand.table_load
    hand.force_solves = []
    hand.auto_visual_refresh = True

    def refresh(data, *, force=False):
        hand.last_probe_solve_s = float(data.time)
        hand.force_solves.append(force)
        return hand.reader.read(data)

    hand.refresh_probe_solve = refresh
    return hand, SimpleNamespace(time=0., xpos=np.zeros((1, 3)))


def step(hand, data, phase="CLOSE", count=1):
    result = None
    for _ in range(count):
        data.time += .002
        if hand.auto_visual_refresh and data.time - hand.estimated.pose_timestamp_s >= .25:
            hand.estimated.pose_timestamp_s = float(data.time)
        result = hand.after_step(SimpleNamespace(value=phase), data, .002)
    return result


class ProbePolicyContracts(unittest.TestCase):
    def test_admission_is_capacity_experiment_not_equilibrium_claim(self):
        hand, data = fixture()
        step(hand, data, count=260)
        self.assertGreaterEqual(hand.probe_eligible_seconds, .25)
        self.assertFalse(hand.allocation["measured_feasible"])
        self.assertFalse(hand.probe_geometry_15["measured_capacity_feasible"])
        self.assertFalse(hand.probe_allocation_1g["measured_feasible"])
        self.assertTrue(hand.admit_probe(data))
        self.assertTrue(hand.force_solves[-1])
        self.assertFalse(hand.estimated.payload_confirmed)
        self.assertEqual(hand.ready_seconds, 0.)

    def test_decisive_fresh_solve_can_reject_cached_capacity(self):
        hand, data = fixture()
        step(hand, data, count=260)
        original = hand.refresh_probe_solve

        def changed(data, *, force=False):
            result = original(data, force=force)
            if force:
                hand.probe_allocation_1g["measured_capacity_feasible"] = False
            return result

        hand.refresh_probe_solve = changed
        self.assertFalse(hand.admit_probe(data))
        self.assertFalse(hand.probe_active)

    def test_stability_and_three_associated_fingers_are_required(self):
        hand, data = fixture()
        step(hand, data, count=260)
        hand.pressures["middle"] = .2
        step(hand, data)
        self.assertEqual(hand.probe_eligible_seconds, 0.)
        self.assertFalse(hand.admit_probe(data))
        hand, data = fixture()
        for index in range(260):
            hand.pressures["ring"] = .8 if index % 2 else 1.2
            step(hand, data)
        self.assertFalse(hand.admit_probe(data))

    def test_moving_wrist_cannot_start_or_decisively_confirm_probe(self):
        hand, data = fixture()
        step(hand, data, count=260)
        hand.actual_wrist_speed_m_s = .0101
        self.assertFalse(hand.admit_probe(data))
        hand, data = fixture()
        step(hand, data, count=260)
        self.assertTrue(hand.admit_probe(data))
        hand.probe_allocation_1g["measured_feasible"] = True
        hand.table_load = 0.
        data.xpos[0, 2] = .005
        hand.last_wrist_position = data.xpos[0].copy()
        step(hand, data, "TRIAL_LIFT", 130)
        hand.actual_wrist_speed_m_s = .0101
        self.assertFalse(hand.confirm_probe(data))

    def test_probe_stops_on_pressure_speed_contact_and_emergency(self):
        for condition in ("pressure", "speed", "contact", "emergency", "visual", "downward"):
            with self.subTest(condition=condition):
                hand, data = fixture()
                step(hand, data, count=260)
                self.assertTrue(hand.admit_probe(data))
                if condition == "pressure": hand.pressures["ring"] = 6.01
                if condition == "speed": data.xpos[0, 2] += .0001
                if condition == "contact": hand.pressures["middle"] = 0.
                if condition == "emergency": hand.emergency_active = True
                if condition == "visual":
                    hand.auto_visual_refresh = False
                    hand.estimated.pose_timestamp_s = -2.
                if condition == "downward":
                    data.xpos[0, 2] = -.0006
                    hand.last_wrist_position = data.xpos[0].copy()
                failure = step(hand, data, "TRIAL_LIFT")
                self.assertIsNotNone(failure)
                self.assertTrue(failure.startswith("PROBE_"))
                hand.abort_probe(failure)
                self.assertFalse(hand.probe_active)
                self.assertFalse(hand.estimated.payload_confirmed)

    def test_confirmation_requires_actual_motion_unload_and_fixed_normal_1g(self):
        hand, data = fixture()
        step(hand, data, count=260)
        self.assertTrue(hand.admit_probe(data))
        hand.probe_allocation_1g["measured_feasible"] = True
        hand.table_load = 0.
        step(hand, data, "TRIAL_LIFT", 130)
        self.assertFalse(hand.confirm_probe(data))
        data.xpos[0, 2] = .005
        hand.last_wrist_position = data.xpos[0].copy()
        hand.table_load = .349 * 9.81
        step(hand, data, "TRIAL_LIFT", 130)
        self.assertFalse(hand.confirm_probe(data))
        hand.table_load = 0.
        step(hand, data, "TRIAL_LIFT", 130)
        self.assertTrue(hand.confirm_probe(data))
        self.assertTrue(hand.force_solves[-1])
        self.assertTrue(hand.estimated.payload_confirmed)
        self.assertFalse(hand.probe_active)
        self.assertEqual(hand.ready_seconds, 0.)

    def test_confirmation_rechecks_current_patch_equilibrium(self):
        hand, data = fixture()
        step(hand, data, count=260)
        self.assertTrue(hand.admit_probe(data))
        hand.probe_allocation_1g["measured_feasible"] = True
        hand.table_load = 0.
        data.xpos[0, 2] = .005
        hand.last_wrist_position = data.xpos[0].copy()
        step(hand, data, "TRIAL_LIFT", 130)
        hand.probe_allocation_1g["measured_feasible"] = False
        self.assertFalse(hand.confirm_probe(data))
        self.assertFalse(hand.estimated.payload_confirmed)

    def test_preprobe_image_cannot_certify_probe_even_if_not_stale(self):
        hand, data = fixture()
        step(hand, data, count=260)
        hand.estimated.pose_timestamp_s = float(data.time)
        self.assertTrue(hand.admit_probe(data))
        hand.auto_visual_refresh = False
        hand.probe_allocation_1g["measured_feasible"] = True
        hand.table_load = 0.
        data.xpos[0, 2] = .005
        hand.last_wrist_position = data.xpos[0].copy()
        step(hand, data, "TRIAL_LIFT", 130)
        self.assertLess(hand.latest_probe["visual_age_s"], 1.)
        self.assertFalse(hand.latest_probe["accepted_visual_after_probe_start"])
        self.assertFalse(hand.confirm_probe(data))

    def test_actual_refresh_uses_current_native_input_and_keeps_original_gate(self):
        hand, data = fixture()
        del hand.refresh_probe_solve
        hand.estimated.surface = object()
        hand.estimated.estimated_pose = lambda data, carrying: {"accepted": True}
        hand.pressures["thumb"] = .9
        hand.reader.raw_contacts[0]["pressure_n"] = .9
        hand.reader.raw_contacts[0]["position_world_m"] = [.003, 0., 0.]
        calls = []

        def associate(raw, pose, surface, **kwargs):
            self.assertEqual(raw[0]["pressure_n"], .9)
            self.assertEqual(raw[0]["position_world_m"], [.003, 0., 0.])
            self.assertEqual(kwargs["max_pose_age_s"], 1.)
            return {"accepted": True, "patches": [{"finger": "thumb", "measured_pressure_n": .9}],
                "estimated_com_world_m": [.01, .02, .05]}

        def allocate(patches, com, mass, **kwargs):
            self.assertEqual(patches[0]["measured_pressure_n"], .9)
            self.assertEqual(com, [.01, .02, .05])
            self.assertEqual(mass, .349)
            self.assertEqual(kwargs["measured_pressures_n"]["thumb"], .9)
            calls.append(kwargs["gravity_margin"])
            return {"feasible": True, "measured_capacity_feasible": True, "measured_feasible": True}

        with patch.object(r8.r4, "associate_tactile_contacts", associate), patch.object(r8.r4, "allocate_contact_wrench", allocate):
            hand.refresh_probe_solve(data, force=True)
        self.assertEqual(calls, [1.5, 1.])
        self.assertFalse(hand.allocation["measured_feasible"])
        self.assertEqual(hand.ready_seconds, 0.)
        self.assertEqual(next(row for row in hand.probe_solved_sample if row["finger"] == "thumb")["position_world_m"], [.003, 0., 0.])

    def test_native_tick_logs_are_strict_json_before_and_during_probe(self):
        hand, data = fixture()
        hand.last_probe_solve_s = -np.inf
        step(hand, data, "SEARCH")
        self.assertIsNone(hand.latest_probe["solve_age_s"])
        json.dumps(hand.probe_events, allow_nan=False)
        step(hand, data, count=260)
        self.assertTrue(hand.admit_probe(data))
        data.xpos[0, 2] = .000002
        step(hand, data, "TRIAL_LIFT")
        self.assertIs(type(hand.latest_probe["actual_probe_endpoint_reached"]), bool)
        json.dumps(hand.probe_events, allow_nan=False)
        json.dumps({"probe_policy": hand.latest_probe.copy()}, allow_nan=False)

    def test_unconfirmed_probe_does_not_move_visual_pose_with_wrist(self):
        estimated = object.__new__(r8.ProbeEstimatedState)
        estimated.pose_fit = {"accepted": True, "origin_m": [0., 0., .05],
            "rotation_world_from_object": np.eye(3).tolist()}
        estimated.anchor_wrist_position = np.zeros(3)
        estimated.anchor_wrist_rotation = np.eye(3)
        estimated.kin = SimpleNamespace(body=0)
        data = SimpleNamespace(xpos=np.array([[0., 0., .005]]), xmat=np.eye(3).reshape((1, 9)))
        estimated.probe_active = True
        self.assertEqual(estimated.estimated_pose(data, True)["origin_m"], [0., 0., .05])
        estimated.probe_active = False
        self.assertAlmostEqual(estimated.estimated_pose(data, True)["origin_m"][2], .055)

    def test_effective_controller_keeps_safety_and_outcome_code(self):
        profile = json.loads((r8.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
        sentinel = SimpleNamespace()
        controller, source, _, estimated = r8.controller_for("ycb_can", Path("scratch/r8_contract"), [], sentinel, profile, .349)
        _, base, _, _ = r8.r6.controller_for("ycb_can", Path("scratch/r8_contract"), [], SimpleNamespace(), profile, .349)
        self.assertIs(controller.__globals__["HandContactReader"](None), sentinel)
        self.assertIs(controller.__globals__["_estimated"], estimated)
        self.assertIs(r8.LoadTransferProbeHand.write, r8.r6.ActualJointReliefHand.write)
        self.assertIn("prohibit_downward=True", source)
        self.assertIn("probe_failure = hand.after_step", source)
        self.assertLess(source.index("_estimated.bind(kin)"), source.index("for attempt in range(max_attempts)"))
        self.assertIn("hand.abort_probe(probe_failure)\n                raise MotionFailure(probe_failure)", source)
        for name in ("lift_evidence", "physical_success", "success"):
            def assignments(text):
                return [ast.dump(node) for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
            self.assertEqual(assignments(source), assignments(base))
        self.assertEqual(r8.PHASE_SECONDS["CLOSE"], 8.)


if __name__ == "__main__":
    unittest.main()
