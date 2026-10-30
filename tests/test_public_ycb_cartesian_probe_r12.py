"""Bounded actual-wrist probe changes preserve physical and acceptance guards."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_cartesian_probe_r12 as r12


def controllers():
    profile = json.loads((r12.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
    sentinel = SimpleNamespace()
    current = r12.controller_for("ycb_can", Path("scratch/r12_contract"), [], sentinel, profile, .349)
    previous = r12.r11.controller_for("ycb_can", Path("scratch/r12_contract"), [], SimpleNamespace(), profile, .349)
    return current, previous, sentinel


class CartesianProbeContracts(unittest.TestCase):
    def test_reference_is_anchored_to_actual_admission_and_five_mm_endpoint(self):
        start = np.array([.623, .021, .624])
        row = r12.bounded_probe_reference(start, start, 0.)
        np.testing.assert_allclose(row["desired_actual_path_world_m"], start)
        np.testing.assert_allclose(row["cartesian_ik_target_world_m"], start)
        halfway = start + [0., 0., .0025]
        row = r12.bounded_probe_reference(start, halfway, 1.5)
        np.testing.assert_allclose(row["desired_actual_path_world_m"], halfway)
        endpoint = start + [0., 0., .005]
        for elapsed in (3., 4.5, 6., 10.):
            with self.subTest(elapsed=elapsed):
                row = r12.bounded_probe_reference(start, endpoint, elapsed)
                np.testing.assert_allclose(row["desired_actual_path_world_m"], endpoint)
                np.testing.assert_allclose(row["cartesian_ik_target_world_m"], endpoint)

    def test_correction_uses_gain_two_and_vector_norm_two_mm_limit(self):
        start = np.array([.623, .021, .624])
        endpoint = start + [0., 0., .005]
        actual = endpoint - [.0001, -.0002, .0003]
        row = r12.bounded_probe_reference(start, actual, 3.)
        np.testing.assert_allclose(row["bounded_reference_correction_world_m"], [.0002, -.0004, .0006])
        row = r12.bounded_probe_reference(start, endpoint - [.01, -.02, .03], 6.)
        correction = np.asarray(row["bounded_reference_correction_world_m"])
        self.assertAlmostEqual(np.linalg.norm(correction), .002)
        np.testing.assert_allclose(correction / np.linalg.norm(correction), np.array([1., -2., 3.]) / np.sqrt(14.))

    def test_reference_rejects_invalid_inputs_and_remains_finite_with_bounded_progress(self):
        for start, actual, elapsed in (([0., 0.], [0., 0., 0.], 0.),
                ([0., 0., float("nan")], [0., 0., 0.], 0.),
                ([0., 0., 0.], [0., float("inf"), 0.], 0.),
                ([0., 0., 0.], [0., 0., 0.], -1.),
                ([0., 0., 0.], [0., 0., 0.], float("inf"))):
            with self.subTest(start=start, actual=actual, elapsed=elapsed):
                with self.assertRaises(ValueError):
                    r12.bounded_probe_reference(start, actual, elapsed)
        for elapsed in np.linspace(0., 6., 31):
            row = r12.bounded_probe_reference([.6, 0., .6], [.59, .01, .59], float(elapsed))
            self.assertTrue(0. <= row["cubic_progress"] <= 1.)
            self.assertLessEqual(np.linalg.norm(row["bounded_reference_correction_world_m"]), .002 + 1e-12)
            json.dumps(row, allow_nan=False)

    def test_helper_logs_actual_wrist_reference_and_sample_age(self):
        hand = object.__new__(r12.CartesianProbeHand)
        hand.probe_start_world_m = np.array([.6, .02, .62])
        hand.estimated = SimpleNamespace(kin=SimpleNamespace(body=0))
        hand.probe_cartesian_events = []
        hand.latest_probe_cartesian = None
        hand.probe_active = True
        data = SimpleNamespace(time=1., xpos=np.array([[.6, .02, .62]]))
        self.assertIsNone(hand.probe_cartesian_snapshot(data))
        target = hand.probe_cartesian_target(data, .02)
        np.testing.assert_allclose(target, hand.latest_probe_cartesian["cartesian_ik_target_world_m"])
        data.time += .012
        row = hand.probe_cartesian_snapshot(data)
        self.assertAlmostEqual(row["sample_age_s"], .012)
        self.assertEqual(len(hand.probe_cartesian_events), 1)
        self.assertFalse(row["target_pose_used"])
        json.dumps(row, allow_nan=False)

    def test_motion_ast_changes_only_probe_tracking_duration_and_declared_exit_conditions(self):
        (controller, source, _, _), (_, old_source, _, _), sentinel = controllers()
        tree, original = ast.parse(source), ast.parse(old_source)
        move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
        duration_if = next(node for node in move.body if isinstance(node, ast.If)
            and ast.unparse(node.test) == "state == State.TRIAL_LIFT and hand.probe_active")
        self.assertEqual(ast.unparse(duration_if.body[0]), "duration = _cartesian_probe_parameters['maximum_probe_seconds']")
        move.body.remove(duration_if)
        loop = next(node for node in move.body if isinstance(node, ast.For))
        self.assertFalse(any(isinstance(node, ast.If) and ast.unparse(node.test) == "state == State.CLOSE and hand.ready_seconds >= 0.25" for node in loop.body))
        eligible_index = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "state == State.CLOSE and hand.probe_eligible_seconds >= _probe_parameters['preprobe_seconds']")
        old_move = next(node for node in ast.walk(original) if isinstance(node, ast.FunctionDef) and node.name == "move")
        old_loop = next(node for node in old_move.body if isinstance(node, ast.For))
        old_ready = next(node for node in old_loop.body if isinstance(node, ast.If)
            and ast.unparse(node.test) == "state == State.CLOSE and hand.ready_seconds >= 0.25")
        loop.body.insert(eligible_index, old_ready)
        segment = next(node for node in loop.body if isinstance(node, ast.If)
            and ast.unparse(node.test) == "cartesian_motion and step % segment_steps == 0")
        reference_index = next(index for index, node in enumerate(segment.body) if isinstance(node, ast.If)
            and ast.unparse(node.test) == "state == State.TRIAL_LIFT and hand.probe_active")
        branch = segment.body[reference_index]
        self.assertIn("hand.probe_cartesian_target(data, end_step * model.opt.timestep)", ast.unparse(branch.body[0]))
        segment.body[reference_index] = branch.orelse[0]
        tracking_row = next(node for node in loop.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Subscript) and ast.unparse(target) == "trace[-1]['probe_cartesian_tracking']" for target in node.targets))
        loop.body.remove(tracking_row)
        confirmation_exit = loop.body.pop()
        self.assertEqual(ast.unparse(confirmation_exit.test), "state == State.TRIAL_LIFT and hand.probe_active and (hand.probe_confirmed_seconds >= _probe_parameters['confirmation_seconds'])")
        self.assertEqual(ast.dump(tree), ast.dump(original))
        self.assertIs(controller.__globals__["HandContactReader"](None), sentinel)
        self.assertIs(controller.__globals__["HandGraspController"].__globals__["LoadTransferProbeHand"], r12.CartesianProbeHand)
        self.assertEqual(controller.__globals__["_phase_seconds"]["CLOSE"], 8.)

    def test_sensor_pressure_emergency_and_acceptance_methods_are_inherited(self):
        for method in ("regulate", "pressure_goal_for", "after_step", "admit_probe", "confirm_probe",
                "abort_probe", "held_revalidated", "refresh_probe_solve", "angles", "write", "update_pressure_relief"):
            with self.subTest(method=method):
                self.assertIs(getattr(r12.CartesianProbeHand, method), getattr(r12.r11.BoundedTrackingProbeHand, method))
        self.assertEqual(r12.PROBE_PARAMETERS, r12.r11.PROBE_PARAMETERS)
        self.assertEqual(r12.CARTESIAN_PROBE_PARAMETERS["maximum_probe_seconds"], 6.)

    def test_runner_identity_records_full_reference_artifact_and_frozen_sources(self):
        constants = r12.run.__code__.co_consts
        self.assertIn("public_ycb_actual_wrist_probe_development_r12", constants)
        self.assertIn("probe_cartesian.json.gz", constants)
        self.assertIn("actual_wrist_probe_motion", constants)
        self.assertIn("probe_refresh.json.gz", constants)
        self.assertIs(r12.run.__globals__["_tracking_parameters"], r12.r11.TRACKING_PARAMETERS)
        snapshot = r12.snapshot(r12.PROJECT / "config/public_ycb_sidewall_20261021.json")
        self.assertIn("scripts/develop_public_ycb_tracking_probe_r11.py", snapshot)
        self.assertIn("scripts/develop_public_ycb_cartesian_probe_r12.py", snapshot)
        self.assertIn("tests/test_public_ycb_cartesian_probe_r12.py", snapshot)


if __name__ == "__main__":
    unittest.main()
