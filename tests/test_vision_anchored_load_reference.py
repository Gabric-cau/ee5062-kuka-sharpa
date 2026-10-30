"""Recorded anchor selection, integrity and robot-only rigid propagation."""

import copy
from dataclasses import FrozenInstanceError
import gzip
import hashlib
import json

import numpy as np
import pytest

from kuka_sharpa_grasp.vision_anchored_load_reference import (
    VisionAnchoredLoadReference, load_vision_anchored_load_reference,
)


IDENTITY = np.eye(3).tolist()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_files(tmp_path, change=None):
    result_path, trace_path, prior_path, mesh = [tmp_path / name for name in
        ("result.json", "trace.json.gz", "prior.json", "public.obj")]
    mesh.write_bytes(b"declared public mesh fixture\n")
    fit = {"accepted": True, "origin_m": [1., 0., 0.], "rotation_world_from_object": IDENTITY,
        "rms_m": .001, "distance_p95_m": .002,
        "near_best_origin_xy_spread_m": .001, "near_best_origin_z_spread_m": .001}
    def observed(timestamp, accepted=True):
        return {"timestamp_s": timestamp, "phase": "CLOSE", "path": "observations/accepted",
            "runtime_target_pose_input": False, "registration": dict(fit, accepted=accepted)}
    result = {"object_kind": "ycb_can", "registration_observations": [observed(1.), observed(2.),
        observed(2.5, False), observed(4.)]}
    trace = [{"timestamp": t, "wrist_position": [0., 0., 0.], "wrist_rotation_world": IDENTITY}
        for t in (1., 2., 4.)]
    prior = {"format": "r28_saved_checkpoint_static_load_diagnostic_v1", "visual_timestamp_s": 2.,
        "mesh_com_local_m": [0., 1., 0.], "catalogue_mass_kg": .349, "declared_pair_friction_prior": .7,
        "mass_source": "Declared catalogue", "friction_source": "Declared engineering prior",
        "mesh_prior": {"kind": "ycb_can", "mesh_file": str(mesh), "mesh_sha256": digest(mesh),
            "com_local_m": [0., 1., 0.], "com_scope": "Isolated public mesh; not live target COM"},
        "boundaries": {"runtime_target_pose_or_com_input": False, "propagation_is_new_visual_measurement": False}}
    if change:
        change(result, trace, prior)
    result_path.write_text(json.dumps(result))
    trace_path.write_bytes(gzip.compress(json.dumps(trace).encode()))
    prior["source_and_input_hashes_before"] = {str(p): digest(p) for p in (result_path, trace_path, mesh)}
    prior["source_and_input_hashes_after"] = dict(prior["source_and_input_hashes_before"])
    prior_path.write_text(json.dumps(prior))
    hashes = {"result": digest(result_path), "trace": digest(trace_path), "prior": digest(prior_path), "mesh": digest(mesh)}
    return (result_path, trace_path, prior_path), hashes, mesh


def test_last_accepted_before_cutoff_and_same_timestamp_immutable_anchor(tmp_path):
    paths, hashes, mesh = fixture_files(tmp_path)
    reference, sources = load_vision_anchored_load_reference(*paths, expected_hashes=hashes, cutoff_timestamp_s=3.)
    assert reference.visual_timestamp_s == 2. and sources["selected_visual_observation_index"] == 1
    assert sources["visual_age_at_cutoff_s"] == 1.
    assert sources["anchor_trace_timestamp_s"] == 2. and sources["source_hashes_verified"]
    assert sources["inputs"]["mesh"]["sha256"] == digest(mesh)
    with pytest.raises(FrozenInstanceError):
        reference.visual_timestamp_s = 3.
    assert isinstance(reference.visual_origin_world_m, tuple)
    saved = reference.as_dict()
    saved["anchor_wrist_position_m"][0] = 100.
    assert reference.anchor_wrist_position_m == (0., 0., 0.)


def test_pure_rotation_translation_and_old_visual_age_are_preserved():
    origin = [1., 0., 0.]
    reference = VisionAnchoredLoadReference(2., origin, IDENTITY, [0., 0., 0.], IDENTITY, [0., 1., 0.], .349, .7)
    origin[0] = 500.
    rotation = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    result = reference.evaluate([2., 3., 4.], rotation, timestamp_s=25.)
    np.testing.assert_allclose(result["estimated_origin_world_m"], [2., 4., 4.])
    np.testing.assert_allclose(result["estimated_com_world_m"], [1., 4., 4.])
    assert result["visual_timestamp_s"] == 2. and result["robot_pose_timestamp_s"] == 25.
    assert result["visual_pose_age_s"] == 23. and not result["propagation_is_new_visual_measurement"]
    assert not result["rigid_holding_assumption_verified"] and result["task_success"] is None
    with pytest.raises(ValueError, match="predates"):
        reference.evaluate([0., 0., 0.], IDENTITY, timestamp_s=1.)
    with pytest.raises(ValueError, match="rotation"):
        reference.evaluate([0., 0., 0.], [[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], timestamp_s=3.)


@pytest.mark.parametrize("change,match", [
    (lambda r, t, p: r["registration_observations"].clear(), "No accepted"),
    (lambda r, t, p: t.pop(1), "same-timestamp"),
    (lambda r, t, p: t.append(copy.deepcopy(t[1])), "unique"),
    (lambda r, t, p: r["registration_observations"][1].update(runtime_target_pose_input=True), "target-pose"),
    (lambda r, t, p: r["registration_observations"][1]["registration"].update(rms_m=.01), "quality"),
    (lambda r, t, p: p.update(visual_timestamp_s=1.), "different visual"),
    (lambda r, t, p: p.update(mesh_com_local_m=[9., 9., 9.]), "COM differs"),
    (lambda r, t, p: r.update(object_kind="another_object"), "object kind"),
])
def test_missing_or_incompatible_evidence_fails_closed(tmp_path, change, match):
    paths, hashes, _ = fixture_files(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        load_vision_anchored_load_reference(*paths, expected_hashes=hashes, cutoff_timestamp_s=3.)


def test_expected_digest_and_prior_binding_cannot_be_bypassed(tmp_path):
    paths, hashes, _ = fixture_files(tmp_path)
    paths[0].write_text(paths[0].read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_vision_anchored_load_reference(*paths, expected_hashes=hashes, cutoff_timestamp_s=3.)
    hashes["result"] = digest(paths[0])
    with pytest.raises(ValueError, match="bind input"):
        load_vision_anchored_load_reference(*paths, expected_hashes=hashes, cutoff_timestamp_s=3.)


def test_mesh_relocation_keeps_content_identity(tmp_path):
    paths, hashes, mesh = fixture_files(tmp_path)
    relocated = tmp_path / "linux" / mesh.name
    relocated.parent.mkdir()
    relocated.write_bytes(mesh.read_bytes())
    mesh.unlink()
    _, sources = load_vision_anchored_load_reference(*paths, expected_hashes=hashes,
        cutoff_timestamp_s=3., mesh_path=relocated)
    assert sources["inputs"]["mesh"]["path"] == str(relocated.resolve())


@pytest.mark.parametrize("field,value", [("catalogue_mass_kg", 0.), ("visual_timestamp_s", float("nan")),
    ("declared_friction_coefficient", -1.), ("visual_rotation_world_from_object", [[0.] * 3] * 3)])
def test_invalid_anchor_values_are_rejected(field, value):
    args = dict(visual_timestamp_s=2., visual_origin_world_m=[1., 0., 0.],
        visual_rotation_world_from_object=IDENTITY, anchor_wrist_position_m=[0., 0., 0.],
        anchor_wrist_rotation_world=IDENTITY, mesh_com_local_m=[0., 1., 0.],
        catalogue_mass_kg=.349, declared_friction_coefficient=.7)
    args[field] = value
    with pytest.raises(ValueError):
        VisionAnchoredLoadReference(**args)
