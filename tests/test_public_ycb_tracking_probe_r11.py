"""Contracts for the declared R11 pressure-tracking-only experiment."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_tracking_probe_r11 as r11


def tracking_fixture(pressures=None, goals=None):
    hand = object.__new__(r11.BoundedTrackingProbeHand)
    fingers = r11.r9.r8.r4.FINGERS
    values = dict.fromkeys((*fingers, "palm"), .5)
    values["palm"] = 0.
    values.update(pressures or {})
    readings = {finger: SimpleNamespace(normal=value) for finger, value in values.items()}
    data = SimpleNamespace(time=1.)
    hand.reader = SimpleNamespace(last_data=data, raw_contacts=[], read=lambda data: readings)
    hand.config = SimpleNamespace(close_rate=.12, relief_rate=.2)
    hand.profile = {"contact_closure": dict(zip(fingers, (1., 0., 1., 1., 0.), strict=True))}
    hand.elapsed = 0.
    hand.last_allocation_s = 1.
    hand.allocation = {"feasible": True, "measured_feasible": True,
        "target_pressures_n": dict.fromkeys(fingers, 2.) | (goals or {})}
    hand.association = {"accepted": True}
    hand.solved_contact_sample = []
    hand.closure = dict.fromkeys(fingers, .3)
    hand.engaged = set()
    hand.previous_pressure = values.copy()
    hand.pressure_rate = dict.fromkeys(values, 0.)
    hand.relief_offsets = {}
    hand.ready_seconds = .2
    hand.slip_adjustments = 0
    hand.feedback_events = []
    hand.feedback_applied = False
    hand.emergency_active = hand.emergency_phase_hold = False
    hand.latched_fingers = {}
    hand.actual_joint_references = {}
    hand.phase = "CLOSE"
    hand.filtered_tracking_goals = {}
    hand.raw_tracking_goals = {}
    hand.reserved_tracking_goals = {}
    hand.measurements_match_solve = lambda: True
    hand.joint_pressure_relief = lambda readings, dt: None
    return hand, readings, data


class TrackingProbeContracts(unittest.TestCase):
    def test_generated_regulator_only_changes_goal_and_proportional_gain(self):
        original = ast.parse(textwrap.dedent(inspect.getsource(r11.r9.r8.r4.WrenchAllocatedHand.regulate)))
        generated = ast.parse(r11.TRACKING_REGULATOR_SOURCE)
        original_goal = next(node.value for node in ast.walk(original) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "goal" for target in node.targets)
            and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "min")
        for node in ast.walk(generated):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "self.pressure_goal_for":
                node.value = original_goal
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == .1:
                node.value = .03
        generated.body[0].body.pop()
        self.assertEqual(ast.dump(generated), ast.dump(original))

    def test_all_active_fingers_get_filtered_reserve_with_six_newton_ceiling(self):
        hand, _, _ = tracking_fixture()
        for finger in ("thumb", "middle", "ring"):
            self.assertAlmostEqual(hand.pressure_goal_for(finger, {finger: 1.}, .15), 1.2)
            self.assertEqual(hand.raw_tracking_goals[finger], 1.)
            self.assertEqual(hand.reserved_tracking_goals[finger], 1.4)
        hand.filtered_tracking_goals.clear()
        self.assertEqual(hand.pressure_goal_for("middle", {"middle": 20.}, .002), 6.)

    def test_zero_nominal_finger_has_no_reserve_or_filter_tail(self):
        hand, readings, _ = tracking_fixture()
        hand.filtered_tracking_goals["index"] = 5.
        self.assertEqual(hand.pressure_goal_for("index", {"index": 6.}, .002), 0.)
        hand.regulate(readings, .002, False)
        self.assertLess(hand.closure["index"], .3)
        self.assertEqual(hand.feedback_events[-1]["force_targets_n"]["index"], 0.)

    def test_rate_ceiling_and_pressure_relief_override_remain_effective(self):
        hand, readings, _ = tracking_fixture({"middle": .2}, {"middle": 6.})
        hand.regulate(readings, .002, True)
        self.assertAlmostEqual(hand.closure["middle"], .3 + .12 * .002)
        hand, readings, _ = tracking_fixture({"middle": 8.}, {"middle": 6.})
        hand.regulate(readings, .002, True)
        self.assertAlmostEqual(hand.closure["middle"], .3 - .2 * .002)

    def test_pressure_tracking_uses_declared_gain_on_unsaturated_command(self):
        hand, readings, _ = tracking_fixture({"middle": .5}, {"middle": .6})
        hand.regulate(readings, .002, False)
        filtered_goal = .6 + .002 / (.15 + .002) * .4
        self.assertAlmostEqual(hand.closure["middle"], .3 + .1 * (filtered_goal - .5) * .002)

    def test_original_free_contact_acquisition_is_unchanged(self):
        hand, readings, _ = tracking_fixture(dict.fromkeys(r11.r9.r8.r4.FINGERS, 0.))
        original, old_readings, _ = tracking_fixture(dict.fromkeys(r11.r9.r8.r4.FINGERS, 0.))
        r11.r9.r8.r4.WrenchAllocatedHand.regulate(original, old_readings, .002, False)
        hand.regulate(readings, .002, False)
        self.assertEqual(hand.closure, original.closure)

    def test_r6_emergency_targets_and_readiness_take_precedence(self):
        hand, readings, _ = tracking_fixture()

        def latch(readings, dt):
            hand.emergency_active = True
            hand.actual_joint_references = {"left_middle_MCP_FE": .123}
            hand.latched_fingers = {"middle": {}}

        hand.joint_pressure_relief = latch
        hand.regulate(readings, .002, False)
        self.assertEqual(hand.ready_seconds, 0.)
        self.assertEqual(hand.angles(), {"left_middle_MCP_FE": .123})
        self.assertTrue(hand.feedback_events[-1]["safety_closure_hold"])
        self.assertEqual(hand.feedback_events[-1]["actual_joint_anchored_targets_rad"], hand.angles())

    def test_close_phase_resets_only_tracking_target_state(self):
        hand, _, data = tracking_fixture()
        hand.phase = "ALIGN"
        hand.filtered_tracking_goals = {"middle": 3.}
        hand.raw_tracking_goals = {"middle": 2.6}
        hand.reserved_tracking_goals = {"middle": 3.}
        with patch.object(r11.r9.CurrentPatchProbeHand, "phase_begin", lambda self, phase, data: None):
            hand.phase_begin("CLOSE", data)
        self.assertEqual(hand.filtered_tracking_goals, {})
        self.assertEqual(hand.raw_tracking_goals, {})
        self.assertEqual(hand.reserved_tracking_goals, {})

    def test_gates_motion_and_current_input_solver_are_inherited_without_changes(self):
        profile = json.loads((r11.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
        sentinel = SimpleNamespace()
        controller, source, _, _ = r11.controller_for("ycb_can", Path("scratch/r11_contract"), [], sentinel, profile, .349)
        _, old_source, _, _ = r11.r9.controller_for("ycb_can", Path("scratch/r11_contract"), [], SimpleNamespace(), profile, .349)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(old_source)))
        factory = controller.__globals__["HandGraspController"]
        self.assertIs(factory.__globals__["LoadTransferProbeHand"], r11.BoundedTrackingProbeHand)
        self.assertIs(controller.__globals__["HandContactReader"](None), sentinel)
        for method in ("after_step", "admit_probe", "confirm_probe", "abort_probe", "held_revalidated",
                "refresh_probe_solve", "angles", "write", "update_pressure_relief"):
            with self.subTest(method=method):
                self.assertIs(getattr(r11.BoundedTrackingProbeHand, method), getattr(r11.r9.CurrentPatchProbeHand, method))
        self.assertEqual(controller.__globals__["_phase_seconds"]["CLOSE"], 8.)

    def test_telemetry_and_runner_freeze_declared_generated_regulator(self):
        hand, readings, _ = tracking_fixture()
        hand.regulate(readings, .002, False)
        encoded = json.dumps(hand.feedback_events, allow_nan=False)
        self.assertIn("raw_allocation_targets_n", encoded)
        self.assertIn("reserved_tracking_goals_n", encoded)
        self.assertIn(r11.TRACKING_REGULATOR_SHA256, encoded)
        self.assertEqual(hashlib.sha256(r11.TRACKING_REGULATOR_SOURCE.encode()).hexdigest(), r11.TRACKING_REGULATOR_SHA256)
        self.assertIn("public_ycb_bounded_tracking_probe_development_r11", r11.run.__code__.co_consts)
        self.assertIn("tracking_regulator_source", r11.run.__code__.co_consts)
        self.assertIn("probe_refresh.json.gz", r11.run.__code__.co_consts)
        snapshot = r11.snapshot(r11.PROJECT / "config/public_ycb_sidewall_20261021.json")
        self.assertIn("scripts/develop_public_ycb_tracking_probe_r11.py", snapshot)
        self.assertIn("tests/test_public_ycb_tracking_probe_r11.py", snapshot)


if __name__ == "__main__":
    unittest.main()
