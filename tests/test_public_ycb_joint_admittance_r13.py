"""Local contracts only: no task dynamics or success is inferred."""

import ast
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_joint_admittance_r13 as r13


def synthetic_patches():
    return [dict(finger=finger, input_index=index, position_world_m=position,
        normal_on_object_world=normal, measured_pressure_n=pressure,
        friction_coefficient=.7745966692414834, hand_body_name=f"left_{finger}_DP")
        for finger, index, position, normal, pressure in (
            ("thumb", 2, [-.03, 0., 0.], [1., 0., 0.], 2.),
            ("middle", 7, [.03, .015, 0.], [-1., 0., 0.], 1.),
            ("ring", 11, [.03, -.015, 0.], [-1., 0., 0.], 1.))]


def pressures():
    return dict(thumb=2., middle=1., ring=1., index=0., pinky=0., palm=0.)


def fake_hand():
    hand = object.__new__(r13.JointAdmittanceProbeHand)
    hand.model = SimpleNamespace(actuator_trnid=np.array([[0, 0]]), jnt_qposadr=np.array([0]),
        jnt_range=np.array([[0., 1.7]]))
    hand.actuators = {"left_middle_PIP": 0}
    hand.flexion_joints = lambda: {"left_middle_PIP": ("middle", 1.4)}
    hand.config = SimpleNamespace(close_rate=.12, relief_rate=.12)
    hand.joint_references = {"left_middle_PIP": .5}
    hand.initial_command_bias = {"left_middle_PIP": .126}
    hand.joint_mode = "JOINT_CONTROL"
    hand.joint_activation_s = 0.
    hand.previous_feasible_targets = None
    hand.previous_tracking_sample = None
    hand.joint_target = {"feasible": True}
    hand.joint_velocities = {"left_middle_PIP": .5}
    hand.joint_solves = []
    hand.joint_events = []
    hand.last_joint_write_s = 1.
    hand.target_timestamp_s = 1.002
    hand.emergency_active = hand.emergency_phase_hold = False
    hand.phase = "CLOSE"
    hand.reader = SimpleNamespace(read=lambda data: {})
    hand.update_pressure_relief = lambda data, readings: None
    return hand


def base_write(hand, data):
    for name, value in hand.angles().items():
        data.ctrl[hand.actuators[name]] = value


class TrackingTargetContracts(unittest.TestCase):
    def test_lp_preserves_original_constraints_and_raw_indices(self):
        patches = synthetic_patches()
        patches.append(dict(patches[1], input_index=99, measured_pressure_n=.01))
        result = r13.allocate_tracking_targets(patches, [0., 0., 0.], .1,
            dict(pressures(), middle=1.01))
        self.assertTrue(result["feasible"], result)
        self.assertEqual([p["raw_input_index"] for p in result["patches"]], [2, 7, 11])
        self.assertEqual(result["excluded_tracking_raw_indices"], [99])
        self.assertEqual(patches[-1]["measured_pressure_n"], .01)
        for finger in r13.ACTIVE:
            self.assertGreaterEqual(result["targets_n"][finger], .75 - 1e-6)
            self.assertLessEqual(result["targets_n"][finger], 6. + 1e-6)
        np.testing.assert_allclose(result["effective_patch_mu"], [.7745966692414834 * .85] * 3)
        np.testing.assert_allclose(result["target_wrench_residual"], np.zeros(6), atol=1e-6)
        self.assertTrue(result["hypothetical_target_fixed_normal_1g_feasible"])
        self.assertIn("not measured", result["hypothetical_check_scope"])
        self.assertFalse(result["runtime_target_state_used"])

    def test_soft_reference_and_invalid_three_finger_inputs(self):
        result = r13.allocate_tracking_targets(synthetic_patches(), [0., 0., 0.], .1,
            pressures(), dict(thumb=3., middle=2., ring=2.))
        self.assertTrue(result["feasible"], result)
        self.assertEqual(result["reference_n"], dict(thumb=2.75, middle=1.75, ring=1.75))
        self.assertFalse(r13.allocate_tracking_targets(synthetic_patches()[:2], [0., 0., 0.], .1, pressures())["feasible"])
        self.assertFalse(r13.allocate_tracking_targets(synthetic_patches(), [0., 0., 0.], -1., pressures())["feasible"])
        duplicate = synthetic_patches()
        duplicate[1]["input_index"] = 2
        self.assertEqual(r13.allocate_tracking_targets(duplicate, [0., 0., 0.], .1, pressures())["reason"], "DUPLICATE_RAW_PATCH_INDEX")

    def test_grouping_preserves_raw_mapping_and_separates_opposing_normals(self):
        patches = [dict(p, raw_input_index=p["input_index"], target_normal_n=1.) for p in synthetic_patches()]
        patches.extend([dict(patches[1], raw_input_index=23, position_world_m=[.03, .016, 0.]),
            dict(patches[1], raw_input_index=24, normal_on_object_world=[1., 0., 0.])])
        groups = r13.group_tracking_patches(patches)
        middle = [g for g in groups if g["finger"] == "middle"]
        self.assertEqual(len(middle), 2)
        self.assertEqual(sorted(sorted(g["raw_input_indices"]) for g in middle), [[7, 23], [24]])
        self.assertAlmostEqual(sum(g["measured_normal_n"] for g in middle), 3.)

    def test_damped_solution_can_compress_mp_while_relaxing_dp_and_logs_residual(self):
        jac = np.array([[.056346892, .014132945, 0.], [.090042705, .049252557, .018655125]])
        velocity, row = r13.damped_joint_velocity(jac, [.2, -.2])
        self.assertGreater((jac @ velocity)[0], 0.)
        self.assertLess((jac @ velocity)[1], 0.)
        self.assertFalse(row["incompatible_direction_hold"])
        self.assertTrue(np.isfinite(velocity).all())
        velocity, row = r13.damped_joint_velocity([[1.], [1.]], [1., -2.])
        self.assertTrue(row["incompatible_direction_hold"])
        np.testing.assert_equal(velocity, [0.])
        with self.assertRaises(ValueError):
            r13.damped_joint_velocity([[float("nan")]], [1.])


class ReferenceContracts(unittest.TestCase):
    def test_captured_loaded_bias_is_not_truncated_and_new_bias_is_bounded(self):
        command, row = r13.bounded_reference_step(.5, .374, .126, 0., .002, [0., 1.7], .168, .168)
        self.assertEqual(command, .5)
        self.assertAlmostEqual(row["captured_initial_command_minus_actual_rad"], .126)
        self.assertAlmostEqual(row["additional_command_bias_rad"], 0.)
        command, row = r13.bounded_reference_step(.5199, .374, .126, 10., .002, [0., 1.7], .168, .168)
        self.assertAlmostEqual(command, .52)
        self.assertLessEqual(abs(row["command_delta_rad"]), .168 * .002 + 1e-12)

    def test_interval_conflict_holds_without_command_jump(self):
        command, row = r13.bounded_reference_step(.5, .2, .126, 1., .002, [0., 1.7], .168, .168)
        self.assertEqual(command, .5)
        self.assertTrue(row["bounds_conflict_hold"])
        self.assertGreater(abs(row["additional_command_bias_rad"]), .02)

    def test_activation_preserves_prior_command_and_actual_anchor(self):
        hand = fake_hand()
        hand.joint_mode = "SCALAR_ACQUISITION"
        data = SimpleNamespace(time=1., ctrl=np.array([.5]), qpos=np.array([.374]))
        hand._activate(data)
        self.assertEqual(hand.joint_references, {"left_middle_PIP": .5})
        self.assertAlmostEqual(hand.initial_command_bias["left_middle_PIP"], .126)
        self.assertEqual(hand.joint_mode, "JOINT_CONTROL")
        self.assertTrue(any(e.get("event") == "ACTIVATION_CAPTURE" for e in hand.joint_events))

    def test_first_write_is_continuous_and_nominal_rate_is_never_exceeded(self):
        hand = fake_hand()
        data = SimpleNamespace(time=1.002, ctrl=np.array([.5]), qpos=np.array([.374]))
        hand.joint_mode = "SCALAR_ACQUISITION"
        hand._activate(data)
        with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": 1.}), \
                patch.object(r13.r11.BoundedTrackingProbeHand, "write", base_write):
            hand.write(data)
        self.assertGreaterEqual(data.ctrl[0], .5)
        self.assertLessEqual(data.ctrl[0] - .5, .168 * .002 + 1e-12)
        self.assertAlmostEqual(hand.initial_command_bias["left_middle_PIP"], .126)

    def test_invalid_or_stale_target_holds_instead_of_scalar_fallback(self):
        for mode, feasible, timestamp in (("HOLD_INVALID", False, 1.002), ("JOINT_CONTROL", True, 1.)):
            with self.subTest(mode=mode, timestamp=timestamp):
                hand = fake_hand()
                hand.joint_mode, hand.joint_target, hand.target_timestamp_s = mode, {"feasible": feasible}, timestamp
                data = SimpleNamespace(time=1.002, ctrl=np.array([.5]), qpos=np.array([.374]))
                with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": 1.}), \
                        patch.object(r13.r11.BoundedTrackingProbeHand, "write", base_write):
                    hand.write(data)
                self.assertEqual(data.ctrl[0], .5)
                self.assertNotEqual(hand.joint_mode, "SCALAR_ACQUISITION")

    def test_emergency_at_write_overrides_nominal_and_release_resumes_after_clear(self):
        hand = fake_hand()
        data = SimpleNamespace(time=1.002, ctrl=np.array([.5]), qpos=np.array([.374]))
        hand.update_pressure_relief = lambda data, readings: setattr(hand, "emergency_active", True)
        with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": .3}), \
                patch.object(r13.r11.BoundedTrackingProbeHand, "write", base_write):
            hand.write(data)
        self.assertEqual(data.ctrl[0], .3)
        self.assertEqual(hand.joint_references["left_middle_PIP"], .3)
        self.assertEqual(hand.joint_mode, "EMERGENCY_HOLD")
        hand.phase = "RECOVER"
        hand.emergency_active = False
        hand.update_pressure_relief = lambda data, readings: None
        data.time += .002
        with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": 1.}), \
                patch.object(r13.r11.BoundedTrackingProbeHand, "write", base_write):
            hand.write(data)
        self.assertEqual(hand.joint_mode, "RELEASING")
        self.assertLess(data.ctrl[0], .3)
        self.assertGreaterEqual(data.ctrl[0], .3 - .168 * .002 - 1e-12)
        self.assertEqual(hand.joint_events[-1]["applied_command_rad"], hand.joint_references)

    def test_final_write_new_emergency_is_recorded_and_carried(self):
        hand = fake_hand()
        data = SimpleNamespace(time=1.002, ctrl=np.array([.5]), qpos=np.array([.374]))
        def final_override(self, data):
            self.emergency_phase_hold = True
            data.ctrl[0] = .25
        with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": 1.}), \
                patch.object(r13.r11.BoundedTrackingProbeHand, "write", final_override):
            hand.write(data)
        self.assertEqual(hand.joint_references["left_middle_PIP"], .25)
        self.assertEqual(hand.joint_mode, "EMERGENCY_HOLD")
        self.assertTrue(hand.joint_events[-1]["safety_override_at_final_write"])
        self.assertNotEqual(hand.joint_events[-1]["candidate_command_rad"], hand.joint_events[-1]["applied_command_rad"])

    def test_phase_retry_reconciles_and_release_goes_to_zero_without_scalar_snap(self):
        hand = fake_hand()
        hand.phase = "RECOVER"
        data = SimpleNamespace(time=1.002, ctrl=np.array([.5]), qpos=np.array([.374]))
        def phase(self, phase, data):
            self.phase = phase
        with patch.object(r13.r11.BoundedTrackingProbeHand, "phase_begin", phase):
            hand.phase_begin("CLOSE", data)
        self.assertEqual(hand.joint_mode, "SCALAR_RECONCILING")
        with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": .5}), \
                patch.object(r13.r11.BoundedTrackingProbeHand, "write", base_write):
            hand.write(data)
        self.assertEqual(data.ctrl[0], .5)
        self.assertEqual(hand.joint_mode, "SCALAR_ACQUISITION")
        self.assertEqual(hand.joint_references, {})
        hand.joint_references = {"left_middle_PIP": .0001}
        with patch.object(r13.r11.BoundedTrackingProbeHand, "phase_begin", phase):
            hand.phase_begin("RELEASE", data)
        data.time += .002
        with patch.object(r13.r11.BoundedTrackingProbeHand, "angles", return_value={"left_middle_PIP": 1.}), \
                patch.object(r13.r11.BoundedTrackingProbeHand, "write", base_write):
            hand.write(data)
            data.time += .002
            hand.write(data)
        self.assertEqual(hand.joint_mode, "OPEN_HOLD")
        self.assertEqual(data.ctrl[0], 0.)
        json.dumps(hand.joint_events, allow_nan=False)

    def test_invalid_association_before_activation_keeps_scalar_acquisition(self):
        hand = fake_hand()
        hand.joint_mode, hand.joint_activation_s, hand.joint_references = "SCALAR_ACQUISITION", None, {}
        data = SimpleNamespace(time=1.002)
        hand.reader = SimpleNamespace(last_data=data, read=lambda data: {key: SimpleNamespace(normal=value)
            for key, value in pressures().items()}, raw_contacts=[])
        hand.estimated = SimpleNamespace(estimated_pose=lambda data, carrying: {}, surface=None, pose_timestamp_s=1.)
        with patch.object(r13.r11.BoundedTrackingProbeHand, "regulate"), \
                patch.object(r13.r11.r9.r8.r4, "associate_tactile_contacts", return_value={"accepted": False, "reason": "STALE_POSE"}):
            hand.regulate({}, .002, False)
        self.assertEqual(hand.joint_mode, "SCALAR_ACQUISITION")
        self.assertEqual(hand.joint_references, {})
        self.assertFalse(hand.joint_target["feasible"])
        hand.joint_mode, hand.joint_activation_s, hand.joint_references = "JOINT_CONTROL", 1., {"left_middle_PIP": .5}
        with patch.object(r13.r11.BoundedTrackingProbeHand, "regulate"), \
                patch.object(r13.r11.r9.r8.r4, "associate_tactile_contacts", return_value={"accepted": False, "reason": "STALE_POSE"}):
            hand.regulate({}, .002, False)
        self.assertEqual(hand.joint_mode, "HOLD_INVALID")
        self.assertEqual(hand.joint_references, {"left_middle_PIP": .5})


class InheritanceAndRunnerContracts(unittest.TestCase):
    def test_only_motion_edit_is_obsolete_close_exit_removal(self):
        profile = json.loads((r13.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
        current = r13.controller_for("ycb_can", Path("scratch/r13_contract"), [], SimpleNamespace(), profile, .349)
        previous = r13.r11.controller_for("ycb_can", Path("scratch/r13_contract"), [], SimpleNamespace(), profile, .349)
        tree, old = ast.parse(current[1]), ast.parse(previous[1])
        move = next(n for n in ast.walk(old) if isinstance(n, ast.FunctionDef) and n.name == "move")
        loop = next(n for n in move.body if isinstance(n, ast.For))
        obsolete = next(n for n in loop.body if isinstance(n, ast.If)
            and ast.unparse(n.test) == "state == State.CLOSE and hand.ready_seconds >= 0.25")
        loop.body.remove(obsolete)
        self.assertEqual(ast.dump(tree), ast.dump(old))
        self.assertIn("hand.probe_eligible_seconds >= _probe_parameters['preprobe_seconds']", current[1])
        self.assertEqual(current[0].__globals__["_phase_seconds"]["CLOSE"], 8.)
        self.assertNotIn("probe_cartesian_target", current[1])
        self.assertIs(current[0].__globals__["HandGraspController"].__globals__["LoadTransferProbeHand"], r13.JointAdmittanceProbeHand)

    def test_original_gates_and_native_safety_are_inherited(self):
        for method in ("after_step", "admit_probe", "confirm_probe", "abort_probe", "held_revalidated",
                "refresh_probe_solve", "update_pressure_relief", "joint_pressure_relief", "release_pressure_relief"):
            with self.subTest(method=method):
                self.assertIs(getattr(r13.JointAdmittanceProbeHand, method), getattr(r13.r11.BoundedTrackingProbeHand, method))
        self.assertEqual(r13.PROBE_PARAMETERS, r13.r11.PROBE_PARAMETERS)
        source = inspect.getsource(r13.JointAdmittanceProbeHand)
        self.assertNotIn("data.qpos[", source.split("def _actual", 1)[1].split("def _mode", 1)[-1])
        self.assertIn('group["normal_on_object_world"]', source)

    def test_runner_records_own_identity_raw_events_and_snapshot(self):
        constants = r13.run.__code__.co_consts
        self.assertIn("public_ycb_joint_admittance_development_r13", constants)
        self.assertIn("joint_admittance.json.gz", constants)
        self.assertIn("probe_refresh.json.gz", constants)
        self.assertIn("joint_admittance", constants)
        sources = r13.snapshot(r13.PROJECT / "config/public_ycb_sidewall_20261021.json")
        self.assertIn("scripts/develop_public_ycb_joint_admittance_r13.py", sources)
        self.assertIn("tests/test_public_ycb_joint_admittance_r13.py", sources)
        self.assertIn("scripts/develop_public_ycb_tracking_probe_r11.py", sources)


if __name__ == "__main__":
    unittest.main()
