"""Input freshness repairs preserve the frozen R8 probe acceptance criteria."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_force_band_r9 as r9
from test_public_ycb_probe_policy_r8 import fixture


def freshness_fixture():
    original, data = fixture()
    hand = object.__new__(r9.CurrentPatchProbeHand)
    hand.__dict__.update(original.__dict__)
    del hand.refresh_probe_solve
    hand.estimated.pose_fit = {"accepted": True}
    hand.last_solved_visual_pose = hand.estimated.pose_fit
    hand.probe_refresh_events = []
    return hand, data


class FreshnessContracts(unittest.TestCase):
    def test_contact_change_forces_current_solve_before_periodic_deadline(self):
        hand, data = freshness_fixture()
        data.time = .002
        hand.reader.raw_contacts[0]["pressure_n"] = 1.3
        seen = []

        def base(self, data, *, force=False):
            seen.append((force, self.reader.raw_contacts[0]["pressure_n"]))
            self.probe_solved_sample = self.contact_sample(self.reader.raw_contacts)
            return self.reader.read(data)

        with patch.object(r9.r8.LoadTransferProbeHand, "refresh_probe_solve", base):
            hand.refresh_probe_solve(data)
        self.assertEqual(seen, [(True, 1.3)])
        self.assertTrue(hand.probe_refresh_events[-1]["changed_contact"])

    def test_same_timestamp_new_accepted_visual_pose_forces_recompute(self):
        hand, data = freshness_fixture()
        hand.estimated.pose_fit = dict(hand.estimated.pose_fit)
        seen = []
        with patch.object(r9.r8.LoadTransferProbeHand, "refresh_probe_solve",
                lambda self, data, *, force=False: seen.append(force) or self.reader.read(data)):
            hand.refresh_probe_solve(data)
        self.assertEqual(seen, [True])
        self.assertTrue(hand.probe_refresh_events[-1]["changed_accepted_visual_pose"])

    def test_unchanged_inputs_retain_periodic_cache(self):
        hand, data = freshness_fixture()
        data.time = .002
        seen = []
        with patch.object(r9.r8.LoadTransferProbeHand, "refresh_probe_solve",
                lambda self, data, *, force=False: seen.append(force) or self.reader.read(data)):
            hand.refresh_probe_solve(data)
        self.assertEqual(seen, [False])
        self.assertFalse(hand.probe_refresh_events[-1]["solve_executed"])
        self.assertEqual(hand.probe_refresh_events[-1]["reactive_solve_reason"], ["MATCHING_INPUT_CACHE"])

    def test_failed_current_capacity_is_not_relabelled_as_eligible(self):
        hand, data = freshness_fixture()
        hand.reader.raw_contacts[0]["pressure_n"] = 1.3
        data.time = .002

        def rejected(self, data, *, force=False):
            self.probe_allocation_1g = {"measured_capacity_feasible": False, "measured_feasible": False}
            self.probe_solved_sample = self.contact_sample(self.reader.raw_contacts)
            self.last_probe_solve_s = float(data.time)
            return self.reader.read(data)

        with patch.object(r9.r8.LoadTransferProbeHand, "refresh_probe_solve", rejected):
            readings = hand.refresh_probe_solve(data)
        row = hand._snapshot_probe(data, readings)
        self.assertIn("CAPACITY_1G_INFEASIBLE", row["preprobe_reasons"])
        json.dumps(hand.probe_refresh_events, allow_nan=False)
        hand.probe_active = hand.estimated.probe_active = True
        hand.probe_start_world_m = data.xpos[0].copy()
        hand.probe_start_s = 0.
        hand.probe_required_fingers = {"thumb", "middle", "ring"}
        with patch.object(r9.r8.LoadTransferProbeHand, "refresh_probe_solve", rejected):
            failure = hand.after_step(SimpleNamespace(value="TRIAL_LIFT"), data, .002)
        self.assertEqual(failure, "PROBE_CAPACITY_1G_INFEASIBLE")

    def test_runner_uses_r9_identity_and_preserves_strict_json_telemetry(self):
        hand, data = freshness_fixture()
        data.time = .002
        with patch.object(r9.r8.LoadTransferProbeHand, "refresh_probe_solve",
                lambda self, data, *, force=False: self.reader.read(data)):
            hand.refresh_probe_solve(data)
        encoded = json.dumps(hand.probe_refresh_events, allow_nan=False)
        self.assertIn("MATCHING_INPUT_CACHE", encoded)
        self.assertIn("current_tactile_sample", encoded)
        self.assertIn("current_accepted_visual_pose", encoded)
        self.assertIn("public_ycb_current_contact_probe_development_r9", r9.run.__code__.co_consts)
        self.assertIn("probe_refresh.json.gz", r9.run.__code__.co_consts)

    def test_effective_controller_and_gates_are_identical_to_frozen_r8(self):
        profile = json.loads((r9.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
        sentinel = SimpleNamespace()
        controller, source, _, _ = r9.controller_for("ycb_can", Path("scratch/r9_contract"), [], sentinel, profile, .349)
        _, original, _, _ = r9.r8.controller_for("ycb_can", Path("scratch/r9_contract"), [], SimpleNamespace(), profile, .349)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(original)))
        factory = controller.__globals__["HandGraspController"]
        self.assertIs(factory.__globals__["LoadTransferProbeHand"], r9.CurrentPatchProbeHand)
        self.assertIs(controller.__globals__["HandContactReader"](None), sentinel)
        for name in ("after_step", "admit_probe", "confirm_probe", "angles", "write", "regulate"):
            self.assertIs(getattr(r9.CurrentPatchProbeHand, name), getattr(r9.r8.LoadTransferProbeHand, name))
        self.assertIsNot(r9.r8.LoadTransferProbeHand, r9.CurrentPatchProbeHand)


if __name__ == "__main__":
    unittest.main()
