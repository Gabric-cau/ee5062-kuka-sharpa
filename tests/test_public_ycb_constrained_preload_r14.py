"""R14 source/numerical contracts; no task dynamics."""

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
import develop_public_ycb_constrained_preload_r14 as r14


J = np.array([[.08935580524392883, .048986061483409615, .01857043395932789],
    [.05708146702803219, .014612111754878232, 0.]])


def original_gate():
    return {"preprobe_reasons": [], "sampled_solve_current": True,
        "cached_measurements_match": True, "solve_age_s": 0.,
        "sampled_fixed_normal_1g_feasible": True}


def preload_hand():
    hand = object.__new__(r14.ConstrainedPreloadHand)
    hand.preload_ready = False
    hand.preload_start_s = hand.preload_anchor_targets = hand.preload_previous_sample = None
    hand.preload_events = []
    hand.probe_events = []
    hand.phase = "CLOSE"
    hand.probe_eligible_seconds = .25
    hand.joint_mode = "JOINT_CONTROL"
    hand.current_preload_observation = lambda data: (original_gate(), {"feasible": True},
        {"thumb:DP:0": {"target_n": 3., "measured_n": 2.8}}, [])
    return hand


def write_hand():
    hand = object.__new__(r14.ConstrainedPreloadHand)
    names = ["left_middle_MCP_FE", "left_middle_PIP", "left_middle_DIP"]
    refs = np.array([.35638750540772596, .4667835724988831, .261133587267851])
    actual = np.array([.3035301835833512, .37989592158833985, .2418478550529146])
    bias = np.array([.043322199157757424, .10689498048687313, .0315121863127078])
    hand.actuators = dict(zip(names, range(3), strict=True))
    hand.model = SimpleNamespace(actuator_trnid=np.array([[0, 0], [1, 0], [2, 0]]),
        jnt_qposadr=np.arange(3), jnt_range=np.array([[-.174533, 1.5708], [0., 1.7453], [0., 1.3963]]))
    hand.flexion_joints = lambda: dict(zip(names, [("middle", 1.), ("middle", 1.4), ("middle", .8)], strict=True))
    hand.joint_references = dict(zip(names, refs.tolist(), strict=True))
    hand.initial_command_bias = dict(zip(names, bias.tolist(), strict=True))
    hand.config = SimpleNamespace(close_rate=.12, relief_rate=.6)
    hand.reader = SimpleNamespace(read=lambda data: {})
    hand.update_pressure_relief = lambda data, readings: None
    hand.last_joint_write_s = 33.534
    hand.target_timestamp_s = 33.536
    hand.joint_target = {"feasible": True}
    hand.joint_mode = "JOINT_CONTROL"
    hand.phase = "CLOSE"
    hand.joint_velocities = dict.fromkeys(names, 0.)
    hand.joint_solves = [{"finger": "middle", "joint_names": names,
        "compression_jacobian_m_rad": J.tolist(), "desired_normal_velocity_m_s": [0., -5.65108870498654e-6],
        "unlimited_joint_velocity_rad_s": [.11, -.12, .13],
        "predicted_unbounded_normal_velocity_m_s": [1e-5, -2e-5],
        "incompatible_direction_hold": True}]
    hand.qp_events = []
    hand.joint_events = []
    hand.emergency_active = hand.emergency_phase_hold = False
    data = SimpleNamespace(time=33.536, qpos=actual, ctrl=refs.copy())
    return hand, data


class ConstrainedSolverContracts(unittest.TestCase):
    def test_current_moving_bias_example_progresses_without_postclip_reversal(self):
        hand, data = write_hand()
        names = list(hand.actuators)
        lower, upper = r14.current_joint_box(hand, data, names, .002)
        self.assertGreater(lower[1], 0.)
        velocity, row = r14.constrained_joint_velocity(J, [0., -5.65108870498654e-6], lower, upper)
        self.assertTrue(row["accepted"], row)
        self.assertFalse(row["zero_velocity_in_box"])
        self.assertLess((J @ velocity)[1], 0.)
        self.assertGreater(np.linalg.norm(velocity), 0.)
        self.assertTrue(np.all(velocity >= lower - 1e-9))
        self.assertTrue(np.all(velocity <= upper + 1e-9))
        for index, name in enumerate(names):
            joint = hand.model.actuator_trnid[index, 0]
            ref = hand.joint_references[name]
            _, ratio = hand.flexion_joints()[name]
            cmd, _ = r14.r13.bounded_reference_step(ref, hand._actual(data, name), hand.initial_command_bias[name],
                velocity[index], .002, hand.model.jnt_range[joint], .12 * ratio, .6 * ratio)
            self.assertAlmostEqual(cmd, ref + velocity[index] * .002, places=12)

    def test_prebound_weak_error_retains_direction_and_nonzero_progress(self):
        jac = [[.08993684944925022, .049037249341353895, .018531396378600584],
            [.05667143248322911, .013807482065957121, 0.]]
        desired = [-4.077492978026731e-6, .0005552128774480706]
        velocity, row = r14.constrained_joint_velocity(jac, desired, [-.6, -.84, -.48], [.12, .168, .096])
        self.assertTrue(row["accepted"], row)
        predicted = np.asarray(jac) @ velocity
        self.assertLessEqual(predicted[0], 1e-10)
        self.assertGreater(predicted[1], 0.)

    def test_tiny_normal_error_is_not_promoted_to_a_direction_constraint(self):
        velocity, row = r14.constrained_joint_velocity([[1., 0.], [0., 1.]],
            [1e-13, 1e-4], [-1., -1.], [1., 1.])
        self.assertTrue(row["accepted"], row)
        self.assertEqual(row["direction_error_epsilon_m_s"], 1e-12)
        self.assertGreater(velocity[1], 0.)

    def test_unconstrained_solution_matches_original_damped_objective(self):
        desired = np.array([.0001, .0002])
        expected = J.T @ np.linalg.solve(J @ J.T + .005 ** 2 * np.eye(2), desired)
        velocity, row = r14.constrained_joint_velocity(J, desired, [-1.] * 3, [1.] * 3)
        self.assertTrue(row["accepted"], row)
        np.testing.assert_allclose(velocity, expected, atol=1e-7)
        self.assertEqual(row["damping_m_rad"], .005)

    def test_infeasible_and_invalid_inputs_hold_truthfully(self):
        for jac, desired, low, high in (([[1.]], [-.001], [.1], [.2]),
                ([[1.]], [.001], [.2], [.1]), ([[float("nan")]], [.001], [-.1], [.1]),
                (np.empty((0, 0)), [], [], [])):
            with self.subTest(jac=jac, low=low):
                velocity, row = r14.constrained_joint_velocity(jac, desired, low, high)
                self.assertFalse(row["accepted"], row)
                np.testing.assert_equal(velocity, np.zeros_like(low))
        velocity, row = r14.constrained_joint_velocity([[1.]], [0.], [0.], [0.])
        self.assertTrue(row["accepted"], row)
        np.testing.assert_equal(velocity, [0.])

    def test_actual_write_preserves_parent_safety_and_projection_identity(self):
        hand, data = write_hand()
        def parent_angles(self):
            return self.joint_references.copy()
        def parent_write(self, data):
            for name, value in self.angles().items():
                data.ctrl[self.actuators[name]] = value
        with patch.object(r14.r13.r11.BoundedTrackingProbeHand, "angles", parent_angles), \
                patch.object(r14.r13.r11.BoundedTrackingProbeHand, "write", parent_write):
            hand.write(data)
        self.assertEqual(hand.joint_mode, "JOINT_CONTROL")
        self.assertLessEqual(hand.qp_events[-1]["parent_projection_difference_rad"], 1e-12)
        self.assertFalse(hand.joint_events[-1]["bounded_motion_predictions"][0]["incompatible_bounded_direction_hold"])
        self.assertTrue(hand.joint_solves[0]["incompatible_direction_hold"])
        self.assertEqual(hand.joint_solves[0]["unlimited_joint_velocity_rad_s"], [.11, -.12, .13])
        self.assertEqual(hand.joint_solves[0]["predicted_unbounded_normal_velocity_m_s"], [1e-5, -2e-5])
        self.assertIn("constrained_joint_velocity_rad_s", hand.joint_solves[0])
        self.assertIn("constrained_predicted_normal_velocity_m_s", hand.joint_solves[0])
        self.assertTrue(hand.qp_events[-1]["applied_motion_checks"][0]["satisfies_nominal_constraints"])
        json.dumps(hand.qp_events, allow_nan=False)

    def test_late_safety_override_still_wins(self):
        hand, data = write_hand()
        def final_override(self, data):
            self.emergency_active = True
            data.ctrl[:] = .2
        with patch.object(r14.r13.r11.BoundedTrackingProbeHand, "angles", return_value=hand.joint_references.copy()), \
                patch.object(r14.r13.r11.BoundedTrackingProbeHand, "write", final_override):
            hand.write(data)
        self.assertEqual(hand.joint_mode, "EMERGENCY_HOLD")
        self.assertTrue(hand.qp_events[-1]["original_safety_override"])
        np.testing.assert_equal(data.ctrl, [.2] * 3)
        self.assertTrue(hand.qp_events[-1]["original_safety_override"])


class PreloadContracts(unittest.TestCase):
    def test_current_good_inputs_require_full_continuous_window(self):
        hand = preload_hand()
        for time, ready in ((1., False), (1.002, False), (1.248, False), (1.25, True)):
            with self.subTest(time=time):
                row = hand.update_preload(SimpleNamespace(time=time))
                self.assertEqual(row["ready"], ready)
                self.assertEqual(row["target_age_s"], 0.)
        json.dumps(hand.preload_events, allow_nan=False)

    def test_all_original_and_new_conditions_reset_the_timer(self):
        cases = [dict(preprobe_reasons=["PRESSURE_NOT_STABLE"]),
            dict(sampled_fixed_normal_1g_feasible=False), dict(cached_measurements_match=False),
            dict(solve_age_s=.002), dict(sampled_solve_current=False)]
        for changed in cases:
            with self.subTest(changed=changed):
                hand = preload_hand()
                hand.update_preload(SimpleNamespace(time=1.))
                hand.current_preload_observation = lambda data: ({**original_gate(), **changed},
                    {"feasible": True}, {"g": {"target_n": 3., "measured_n": 3.}}, [])
                row = hand.update_preload(SimpleNamespace(time=1.25))
                self.assertFalse(row["ready"])
                self.assertIsNone(hand.preload_start_s)
        hand = preload_hand()
        hand.probe_eligible_seconds = .248
        self.assertIn("ORIGINAL_CONTINUOUS_ELIGIBILITY", hand.update_preload(SimpleNamespace(time=1.))["reasons"])

    def test_target_drift_is_measured_from_window_origin_and_group_order_is_stable(self):
        hand = preload_hand()
        for index, target in enumerate((3., 3.2, 3.36)):
            hand.current_preload_observation = lambda data, target=target: (original_gate(), {"feasible": True},
                {"g": {"target_n": target, "measured_n": target}}, [])
            row = hand.update_preload(SimpleNamespace(time=1. + index * .1))
        self.assertEqual(row["window_reset_reason"], "TARGET_MOVED_FROM_WINDOW_ORIGIN")
        self.assertEqual(row["continuous_seconds"], 0.)
        first = {"finger": "middle", "hand_body_name": "MP", "raw_input_indices": [1, 2],
            "target_normal_n": 1., "measured_normal_n": 1.}
        self.assertEqual(r14.preload_group_values([first]),
            r14.preload_group_values([{**first, "raw_input_indices": [2, 1]}]))

    def test_pressure_error_or_group_identity_changes_prevent_readiness(self):
        hand = preload_hand()
        hand.update_preload(SimpleNamespace(time=1.))
        hand.current_preload_observation = lambda data: (original_gate(), {"feasible": True},
            {"different": {"target_n": 3., "measured_n": 2.64}}, [])
        row = hand.update_preload(SimpleNamespace(time=1.25))
        self.assertIn("GROUP_PRESSURE_ERROR", row["reasons"])
        self.assertEqual(row["window_reset_reason"], "GROUP_IDENTITY_CHANGED")
        self.assertFalse(row["ready"])

    def test_admission_forces_current_gate_and_never_uses_counterfactual_as_measurement(self):
        hand = preload_hand()
        hand.update_preload(SimpleNamespace(time=1.))
        with patch.object(r14.r13.JointAdmittanceProbeHand, "admit_probe", return_value=True) as parent:
            self.assertTrue(hand.admit_probe(SimpleNamespace(time=1.25)))
            self.assertTrue(hand.preload_events[-1]["admission_recheck"])
            self.assertEqual(parent.call_count, 1)
        hand.current_preload_observation = lambda data: ({**original_gate(), "sampled_fixed_normal_1g_feasible": False},
            {"feasible": True, "hypothetical_target_fixed_normal_1g_feasible": True},
            {"g": {"target_n": 3., "measured_n": 3.}}, [])
        with patch.object(r14.r13.JointAdmittanceProbeHand, "admit_probe") as parent:
            self.assertFalse(hand.admit_probe(SimpleNamespace(time=1.252)))
            parent.assert_not_called()

    def test_post_integration_gate_recomputes_actual_current_target_and_measurements(self):
        hand = preload_hand()
        hand.reader = SimpleNamespace(raw_contacts=[{"hand_body_name": "DP"}])
        hand.refresh_probe_solve = unittest.mock.Mock(return_value={})
        hand._snapshot_probe = lambda data, readings: {**original_gate(), "pressures_n": {"thumb": 3.}}
        hand.estimated = SimpleNamespace(estimated_pose=lambda data, carrying: {}, payload_confirmed=False,
            surface=None, pose_timestamp_s=1., mass_kg=.349)
        hand.previous_feasible_targets = {"thumb": 4.}
        hand.contact_sample = lambda raw: []
        target = {"feasible": True, "patches": [{"finger": "thumb", "raw_input_index": 0,
            "hand_body_name": "DP", "position_world_m": [0., 0., 0.], "normal_on_object_world": [1., 0., 0.],
            "target_normal_n": 3., "measured_pressure_n": 2.95}]}
        association = {"accepted": True, "estimated_com_world_m": [0., 0., 0.],
            "patches": [{"input_index": 0, "finger": "thumb"}]}
        data = SimpleNamespace(time=1.002)
        with patch.object(r14.r13.r11.r9.r8.r4, "associate_tactile_contacts", return_value=association) as associate, \
                patch.object(r14.r13, "allocate_tracking_targets", return_value=target) as allocate:
            _, _, groups, _ = r14.ConstrainedPreloadHand.current_preload_observation(hand, data)
        hand.refresh_probe_solve.assert_called_once_with(data, force=True)
        self.assertEqual(associate.call_args.kwargs["now_s"], 1.002)
        self.assertEqual(next(iter(groups.values()))["measured_n"], 2.95)
        self.assertEqual(allocate.call_args.args[-1], {"thumb": 4.})
        self.assertEqual(hand.previous_feasible_targets, {"thumb": 4.})


class RunnerContracts(unittest.TestCase):
    def test_only_motion_change_is_stricter_close_exit_and_declared_deadline(self):
        profile = json.loads((r14.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
        new = r14.controller_for("ycb_can", Path("scratch/r14_contract"), [], SimpleNamespace(), profile, .349)
        old = r14.r13.controller_for("ycb_can", Path("scratch/r14_contract"), [], SimpleNamespace(), profile, .349)
        tree = ast.parse(new[1])
        condition = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
            and ast.unparse(n.test) == "state == State.CLOSE and hand.preload_ready")
        condition.test = ast.parse("state == State.CLOSE and hand.probe_eligible_seconds >= _probe_parameters['preprobe_seconds']", mode="eval").body
        self.assertEqual(ast.dump(tree), ast.dump(ast.parse(old[1])))
        self.assertEqual(new[0].__globals__["_phase_seconds"]["CLOSE"], 16.)
        self.assertEqual(old[0].__globals__["_phase_seconds"]["CLOSE"], 8.)
        self.assertIs(new[0].__globals__["HandGraspController"].__globals__["LoadTransferProbeHand"], r14.ConstrainedPreloadHand)

    def test_unchanged_force_physics_confirmation_gates_and_full_artifacts(self):
        for method in ("regulate", "confirm_probe", "held_revalidated", "abort_probe", "update_pressure_relief", "angles"):
            with self.subTest(method=method):
                self.assertIs(getattr(r14.ConstrainedPreloadHand, method), getattr(r14.r13.JointAdmittanceProbeHand, method))
        self.assertEqual(r14.PROBE_PARAMETERS, r14.r13.PROBE_PARAMETERS)
        self.assertIn("force=True", inspect.getsource(r14.ConstrainedPreloadHand.current_preload_observation))
        constants = r14.run.__code__.co_consts
        for value in ("public_ycb_constrained_preload_development_r14", "preload.json.gz", "constrained_joint.json.gz",
                "joint_admittance.json.gz", "probe_refresh.json.gz", "constrained_preload"):
            self.assertIn(value, constants)
        sources = r14.snapshot(r14.PROJECT / "config/public_ycb_sidewall_20261021.json")
        for name in ("scripts/develop_public_ycb_joint_admittance_r13.py", "tests/test_public_ycb_joint_admittance_r13.py",
                "scripts/develop_public_ycb_constrained_preload_r14.py", "tests/test_public_ycb_constrained_preload_r14.py"):
            self.assertIn(name, sources)


if __name__ == "__main__":
    unittest.main()
