"""Profile isolation must preserve R9's physical controller and evidence gates."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_profile_probe_r10 as r10


class ProfileProbeContracts(unittest.TestCase):
    def test_only_declared_profile_bytes_are_accepted(self):
        path = r10.PROJECT / "config/public_ycb_sidewall_20261021.json"
        actual = r10.r9.r8.r4.previous.sha256(path)
        r10.validate_profile(path, actual)
        for rejected in ("0" * 64, actual.upper(), "", actual + "0"):
            with self.subTest(rejected=rejected), self.assertRaises(ValueError):
                r10.validate_profile(path, rejected)

    def test_generated_task_uses_the_unchanged_r9_controller(self):
        profile = json.loads((r10.PROJECT / "config/public_ycb_sidewall_20261021.json").read_text())["ycb_can"]
        guard = SimpleNamespace()
        controller, source, _, _ = r10.run.__globals__["controller_for"](
            "ycb_can", Path("scratch/r10_contract"), [], guard, profile, .349)
        _, baseline, _, _ = r10.r9.controller_for(
            "ycb_can", Path("scratch/r10_contract"), [], SimpleNamespace(), profile, .349)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(baseline)))
        self.assertIs(controller.__globals__["HandContactReader"](None), guard)
        factory = controller.__globals__["HandGraspController"]
        self.assertIs(factory.__globals__["LoadTransferProbeHand"], r10.r9.CurrentPatchProbeHand)

    def test_identity_snapshot_and_refresh_artifact_are_bound(self):
        self.assertIn("public_ycb_profile_probe_development_r10", r10.run.__code__.co_consts)
        self.assertIn("probe_refresh.json.gz", r10.run.__code__.co_consts)
        self.assertIs(r10.run.__globals__["snapshot"], r10.snapshot)
        path = r10.PROJECT / "config/public_ycb_sidewall_20261021.json"
        sources = r10.snapshot(path)
        self.assertIn("scripts/develop_public_ycb_profile_probe_r10.py", sources)
        self.assertIn("tests/test_public_ycb_profile_probe_r10.py", sources)
        self.assertEqual(sources[path.relative_to(r10.PROJECT).as_posix()],
            r10.r9.r8.r4.previous.sha256(path))
        for name, value in r10.r9.PROBE_PARAMETERS.items():
            self.assertEqual(r10.PROBE_PARAMETERS[name], value)


if __name__ == "__main__":
    unittest.main()
