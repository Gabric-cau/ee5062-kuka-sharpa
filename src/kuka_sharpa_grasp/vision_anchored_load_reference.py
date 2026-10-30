"""Immutable recorded-vision anchor propagated by robot wrist poses only.

The public-mesh COM and catalogue mass are declared priors, loaded from an
explicitly hash-bound receipt. No simulator or live target state is accepted.
Rigid propagation is a holding hypothesis, not a new visual measurement or
evidence of grasp, force equilibrium, retention, or placement success.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite nonnegative scalar")
    number = float(value)
    if not math.isfinite(number) or number < 0. or positive and number == 0.:
        raise ValueError(name + " must be finite and nonnegative")
    return number


def _vector(values, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(name + " must be a finite three-vector")
    return tuple(map(float, array))


def _rotation(values, name):
    array = np.asarray(values, dtype=float)
    if (array.shape != (3, 3) or not np.isfinite(array).all()
            or not np.allclose(array.T @ array, np.eye(3), atol=1e-6, rtol=0.)
            or abs(float(np.linalg.det(array)) - 1.) > 1e-6):
        raise ValueError(name + " must be a finite proper orthonormal rotation")
    return tuple(tuple(map(float, row)) for row in array)


@dataclass(frozen=True)
class VisionAnchoredLoadReference:
    visual_timestamp_s: float
    visual_origin_world_m: tuple[float, float, float]
    visual_rotation_world_from_object: tuple[tuple[float, ...], ...]
    anchor_wrist_position_m: tuple[float, float, float]
    anchor_wrist_rotation_world: tuple[tuple[float, ...], ...]
    mesh_com_local_m: tuple[float, float, float]
    catalogue_mass_kg: float
    declared_friction_coefficient: float

    def __post_init__(self):
        for name in ("visual_origin_world_m", "anchor_wrist_position_m", "mesh_com_local_m"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        for name in ("visual_rotation_world_from_object", "anchor_wrist_rotation_world"):
            object.__setattr__(self, name, _rotation(getattr(self, name), name))
        for name in ("visual_timestamp_s", "catalogue_mass_kg", "declared_friction_coefficient"):
            object.__setattr__(self, name, _number(getattr(self, name), name,
                positive=name == "catalogue_mass_kg"))

    def as_dict(self):
        return {"visual_timestamp_s": self.visual_timestamp_s,
            "visual_origin_world_m": list(self.visual_origin_world_m),
            "visual_rotation_world_from_object": [list(row) for row in self.visual_rotation_world_from_object],
            "anchor_wrist_position_m": list(self.anchor_wrist_position_m),
            "anchor_wrist_rotation_world": [list(row) for row in self.anchor_wrist_rotation_world],
            "mesh_com_local_m": list(self.mesh_com_local_m), "catalogue_mass_kg": self.catalogue_mass_kg,
            "declared_friction_coefficient": self.declared_friction_coefficient,
            "runtime_target_pose_or_com_input": False,
            "propagation_is_new_visual_measurement": False, "rigid_holding_assumption_verified": False}

    def evaluate(self, current_wrist_position_m, current_wrist_rotation_world, *, timestamp_s):
        """Estimate origin/COM using current robot proprioception and the old anchor.

        ``timestamp_s`` identifies the supplied robot pose. It never replaces
        ``visual_timestamp_s`` or resets the old observation's visual age.
        """
        timestamp = _number(timestamp_s, "timestamp_s")
        if timestamp < self.visual_timestamp_s:
            raise ValueError("Current robot pose predates the recorded visual observation")
        wrist = np.asarray(_vector(current_wrist_position_m, "current_wrist_position_m"))
        wrist_rotation = np.asarray(_rotation(current_wrist_rotation_world, "current_wrist_rotation_world"))
        delta = wrist_rotation @ np.asarray(self.anchor_wrist_rotation_world).T
        origin = wrist + delta @ (np.asarray(self.visual_origin_world_m) - self.anchor_wrist_position_m)
        rotation = delta @ np.asarray(self.visual_rotation_world_from_object)
        com = origin + rotation @ np.asarray(self.mesh_com_local_m)
        if not np.isfinite(np.r_[origin, com, rotation.ravel()]).all():
            raise ValueError("Nonfinite rigid propagation arithmetic")
        return {"estimated_origin_world_m": origin.tolist(),
            "estimated_rotation_world_from_object": rotation.tolist(), "estimated_com_world_m": com.tolist(),
            "catalogue_mass_kg": self.catalogue_mass_kg,
            "declared_friction_coefficient": self.declared_friction_coefficient,
            "robot_pose_timestamp_s": timestamp, "visual_timestamp_s": self.visual_timestamp_s,
            "visual_pose_age_s": timestamp - self.visual_timestamp_s,
            "runtime_target_pose_or_com_input": False, "propagation_is_new_visual_measurement": False,
            "rigid_holding_assumption_verified": False, "task_success": None,
            "scope": "Old accepted RGB-D and public-mesh COM rigidly propagated by current wrist proprioception"}


def _captured(path, expected):
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if not isinstance(expected, str) or digest != expected:
        raise ValueError("Input SHA256 mismatch: " + str(path))
    return content, {"path": str(path), "sha256": digest, "bytes": len(content)}


def _bound_prior_input(prior, path, expected):
    before, after = prior["source_and_input_hashes_before"], prior["source_and_input_hashes_after"]
    if before != after:
        raise ValueError("The saved prior receipt reports changed inputs")
    matches = [digest for name, digest in before.items() if Path(name).name == Path(path).name]
    if len(matches) != 1 or matches[0] != expected:
        raise ValueError("Prior receipt does not uniquely bind input: " + str(path))


def load_vision_anchored_load_reference(result_path, trace_path, prior_receipt_path, *,
                                      expected_hashes, cutoff_timestamp_s=22., mesh_path=None):
    """Return ``(immutable_reference, source_receipt)`` from hash-bound files.

    ``expected_hashes`` must contain ``result``, ``trace``, ``prior``, ``mesh``.
    ``prior_receipt_path`` is the saved R28 static-load diagnostic, which records
    the public-mesh local COM and catalogue mass/friction without live state.
    ``mesh_path`` can relocate that same SHA-bound public mesh to another host.
    The selected visual observation is independently chosen from the result;
    its wrist anchor must be a unique trace row at the same native timestamp.
    """
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {"result", "trace", "prior", "mesh"}:
        raise ValueError("expected_hashes needs exactly result, trace, prior and mesh")
    cutoff = _number(cutoff_timestamp_s, "cutoff_timestamp_s")
    result_bytes, result_record = _captured(result_path, expected_hashes["result"])
    trace_bytes, trace_record = _captured(trace_path, expected_hashes["trace"])
    prior_bytes, prior_record = _captured(prior_receipt_path, expected_hashes["prior"])
    result, prior = json.loads(result_bytes), json.loads(prior_bytes)
    trace = json.loads(gzip.decompress(trace_bytes))
    if (prior["format"] != "r28_saved_checkpoint_static_load_diagnostic_v1"
            or prior["boundaries"]["runtime_target_pose_or_com_input"] is not False
            or prior["boundaries"]["propagation_is_new_visual_measurement"] is not False):
        raise ValueError("Expected the explicitly bounded public-mesh prior receipt")
    mesh = Path(prior["mesh_prior"]["mesh_file"]) if mesh_path is None else Path(mesh_path)
    _, mesh_record = _captured(mesh, expected_hashes["mesh"])
    if prior["mesh_prior"]["mesh_sha256"] != expected_hashes["mesh"]:
        raise ValueError("Public-mesh identity differs from the prior receipt")
    for path, key in ((result_path, "result"), (trace_path, "trace"), (mesh, "mesh")):
        _bound_prior_input(prior, path, expected_hashes[key])

    accepted = []
    for index, observation in enumerate(result["registration_observations"]):
        timestamp = _number(observation["timestamp_s"], "visual timestamp")
        if timestamp <= cutoff and observation["registration"].get("accepted") is True:
            accepted.append((timestamp, index, observation))
    if not accepted:
        raise ValueError("No accepted recorded RGB-D observation at or before cutoff")
    timestamp, observation_index, observation = max(accepted, key=lambda item: item[0])
    if sum(abs(item[0] - timestamp) <= 1e-9 for item in accepted) != 1:
        raise ValueError("Ambiguous latest accepted visual timestamp")
    if observation["runtime_target_pose_input"] is not False:
        raise ValueError("Visual observation declares runtime target-pose input")
    fit = observation["registration"]
    for field, limit in (("rms_m", .004), ("distance_p95_m", .008),
                         ("near_best_origin_xy_spread_m", .005), ("near_best_origin_z_spread_m", .005)):
        if _number(fit[field], field) > limit:
            raise ValueError("Recorded accepted pose fails its quality/ambiguity gate: " + field)
    anchors = [row for row in trace if abs(_number(row["timestamp"], "trace timestamp") - timestamp) <= 1e-9]
    if len(anchors) != 1:
        raise ValueError("A unique same-timestamp wrist trace anchor is required")
    anchor = anchors[0]
    if abs(float(prior["visual_timestamp_s"]) - timestamp) > 1e-9:
        raise ValueError("The prior receipt was built from a different visual observation")
    reference = VisionAnchoredLoadReference(timestamp, fit["origin_m"], fit["rotation_world_from_object"],
        anchor["wrist_position"], anchor["wrist_rotation_world"], prior["mesh_com_local_m"],
        prior["catalogue_mass_kg"], prior["declared_pair_friction_prior"])
    if not np.allclose(reference.mesh_com_local_m, prior["mesh_prior"]["com_local_m"], atol=1e-12, rtol=0.):
        raise ValueError("Public mesh COM differs within the saved prior receipt")
    if result.get("object_kind") != prior["mesh_prior"]["kind"]:
        raise ValueError("Recorded object kind differs from the declared public-mesh prior")
    sources = {"format": "vision_anchored_load_reference_sources_v1", "inputs": {
        "result": result_record, "trace": trace_record, "prior": prior_record, "mesh": mesh_record},
        "cutoff_timestamp_s": cutoff, "selected_visual_observation_index": observation_index,
        "selected_visual_path": observation["path"], "selected_visual_phase": observation["phase"],
        "visual_timestamp_s": timestamp, "anchor_trace_timestamp_s": float(anchor["timestamp"]),
        "visual_age_at_cutoff_s": cutoff - timestamp, "anchor": reference.as_dict(),
        "mass_source": prior["mass_source"], "friction_source": prior["friction_source"],
        "mesh_com_scope": prior["mesh_prior"]["com_scope"], "source_hashes_verified": True,
        "runtime_target_pose_or_com_input": False, "propagation_is_new_visual_measurement": False,
        "rgbd_pixel_reprocessing_performed": False, "rigid_holding_assumption_verified": False,
        "scope": "Saved accepted registration and matching wrist trace; recorded priors are not remeasured"}
    return reference, sources
