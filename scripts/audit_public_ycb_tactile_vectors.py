#!/usr/bin/env python3
"""Stream recorded tactile force vectors without changing simulation or gates."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path

import numpy as np
import mujoco

from analyze_public_ycb_vertical_load import digest, json_array, vector
from audit_public_ycb_bias_recovery import checked_path, equal, tick
from public_ycb_evidence import load_evidence_model


CHANNELS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def hand_inventory(model):
    result = {}
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        if body.startswith("left_"):
            finger = next((name for name in CHANNELS[:-1] if body.startswith("left_" + name + "_")), "palm")
            result[geom] = (finger, body)
    if not result:
        raise ValueError("Missing known hand geometry")
    return result


def rebuild_patches(sample, model, hand):
    """Apply the frozen r4 hand-vs-external filter in original contact order."""
    patches = []
    for contact in sample["native_contacts"]:
        ids = contact["geom_ids"]
        if (len(ids) != 2 or any(type(item) is not int or not 0 <= item < model.ngeom for item in ids)
                or ids[0] == ids[1]):
            raise ValueError("Invalid native contact geometry pair")
        a, b = ids
        if (a in hand) == (b in hand):
            continue
        side = a if a in hand else b
        sign = 1. if side == b else -1.
        frame = np.asarray(contact["contact_frame_world_rows"], dtype=float)
        if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                or not np.allclose(frame @ frame.T, np.eye(3), atol=1e-8, rtol=0.)):
            raise ValueError("Invalid native contact frame")
        local = vector(contact["force_torque_contact_frame"], 6, "contact wrench")
        point = vector(contact["position_world_m"], 3, "tactile contact point")
        normal = sign * frame[0]
        force = sign * frame.T @ local[:3]
        pressure, tangent = abs(float(local[0])), float(np.linalg.norm(local[1:3]))
        equal(float(force @ normal), pressure, "Tactile pressure/force sign")
        equal(np.linalg.norm(force - pressure * normal), tangent, "Tactile tangent norm")
        patches.append({"finger": hand[side][0], "hand_body_name": hand[side][1],
            "position_world_m": point.tolist(), "normal_on_hand_world": normal.tolist(),
            "force_on_hand_world_n": force.tolist(), "pressure_n": pressure,
            "tangential_force_n": tangent})
    return patches


def forces(patches, indices=None):
    chosen = range(len(patches)) if indices is None else indices
    result = {key: np.zeros(3) for key in ("force_on_hand_world_n", "normal_on_hand_world_n", "shear_on_hand_world_n")}
    for index in chosen:
        patch = patches[index]
        force = np.asarray(patch["force_on_hand_world_n"])
        normal = np.asarray(patch["normal_on_hand_world"]) * patch["pressure_n"]
        result["force_on_hand_world_n"] += force
        result["normal_on_hand_world_n"] += normal
        result["shear_on_hand_world_n"] += force - normal
    equal(result["force_on_hand_world_n"], result["normal_on_hand_world_n"] + result["shear_on_hand_world_n"],
        "Tactile normal/shear force decomposition")
    return {**{key: value.tolist() for key, value in result.items()},
        **{key.replace("on_hand", "hand_on_external"): (-value).tolist() for key, value in result.items()}}


class TimedStream:
    def __init__(self, path):
        self.stream, self.next, self.previous = iter(json_array(path)), None, -1
        self._advance()

    def _advance(self):
        self.next = next(self.stream, None)
        if self.next is not None:
            index = tick(self.next["timestamp_s"])
            if index < self.previous:
                raise ValueError("Diagnostic stream is not ordered")
            self.previous = index

    def at(self, index):
        if self.next is not None and tick(self.next["timestamp_s"]) < index:
            raise ValueError("Diagnostic observation has no contemporaneous native sample")
        rows = []
        while self.next is not None and tick(self.next["timestamp_s"]) == index:
            rows.append(self.next)
            self._advance()
        return rows


def check_pressures(patches, snapshot):
    measured = {finger: sum(row["pressure_n"] for row in patches if row["finger"] == finger) for finger in CHANNELS}
    if set(snapshot["pressures_n"]) != set(CHANNELS):
        raise ValueError("Incomplete contemporaneous tactile pressure channels")
    equal([snapshot["pressures_n"][finger] for finger in CHANNELS],
        [measured[finger] for finger in CHANNELS], "Native/reported tactile pressure", 1e-8)


def association_indices(patches, recorded, *, tracking=False):
    indices = []
    for patch in recorded:
        index = patch["raw_input_index"] if tracking else patch["input_index"]
        if type(index) is not int or index < 0 or index >= len(patches) or index in indices:
            raise ValueError("Invalid or duplicate contemporaneous association raw index")
        actual = patches[index]
        if patch["finger"] != actual["finger"]:
            raise ValueError("Associated raw index differs from known hand channel")
        if tracking and patch["hand_body_name"] != actual["hand_body_name"]:
            raise ValueError("Associated raw index differs from known hand link")
        equal(patch["position_world_m"], actual["position_world_m"], "Associated tactile point", 1e-8)
        equal(patch["normal_on_object_world"], -np.asarray(actual["normal_on_hand_world"]), "Associated tactile normal", 1e-8)
        equal(patch["measured_pressure_n"], actual["pressure_n"], "Associated tactile pressure", 1e-8)
        indices.append(index)
    return indices


def evaluation_only(sample, hand, target):
    """Privileged target identities are used only in this isolated comparison."""
    indices, offset = [], 0
    for contact in sample["native_contacts"]:
        a, b = contact["geom_ids"]
        if (a in hand) == (b in hand):
            continue
        if target in (a, b):
            indices.append(offset)
        offset += 1
    return indices


def summarize(rows, ordinal, start, frames, seconds):
    end = rows[-1]["timestamp_s"]
    selected = [row for row in rows if row["timestamp_s"] >= end - seconds - 1e-9]
    def stats(values):
        if not values:
            return None
        array = np.asarray(values)
        return {"mean": array.mean(axis=0).tolist(), "minimum": array.min(axis=0).tolist(), "maximum": array.max(axis=0).tolist()}
    groups = {}
    for label in ("all_external", "associated_hypothesis", "unassociated_external", "evaluation_only_target"):
        available = [row[label] for row in selected if row[label] is not None]
        groups[label] = {"frames": len(available), "force_vectors_n": {
            key: stats([row[key] for row in available]) for key in selected[0]["all_external"]}}
    return {"close_segment_ordinal": ordinal, "segment_start_s": start, "segment_end_s": end,
        "segment_frames": frames, "window_seconds": seconds, "window_start_s": selected[0]["timestamp_s"],
        "window_end_s": end, "window_frames": len(selected), "groups": groups,
        "association_scope_counts": dict(Counter(row["association_scope"] for row in selected)),
        "frames": selected}


def sign_sanity():
    patch = {"force_on_hand_world_n": [2., 0., 3.], "normal_on_hand_world": [1., 0., 0.], "pressure_n": 2.}
    result = forces([patch])
    equal(result["force_hand_on_external_world_n"], [-2., 0., -3.], "Synthetic action/reaction sign")
    equal(result["normal_hand_on_external_world_n"], [-2., 0., 0.], "Synthetic normal sign")
    equal(result["shear_hand_on_external_world_n"], [0., 0., -3.], "Synthetic shear sign")
    class TwoGeometries:
        ngeom = 2
    for hand_geom, sign in ((0, -1.), (1, 1.)):
        native = {"native_contacts": [{"geom_ids": [0, 1], "contact_frame_world_rows": np.eye(3).tolist(),
            "force_torque_contact_frame": [2., .3, .4, 0., 0., 0.], "position_world_m": [0., 0., 0.]}]}
        rebuilt = rebuild_patches(native, TwoGeometries(), {hand_geom: ("thumb", "left_thumb_link")})
        equal(rebuilt[0]["force_on_hand_world_n"], sign * np.asarray([2., .3, .4]),
            "Synthetic native geom-order force sign")


def analyze(directory, seconds=.25):
    sign_sanity()
    directory = Path(directory).resolve(strict=True)
    summary_path, protocol_path = checked_path(directory, "summary.json"), checked_path(directory, "protocol.json")
    summary, protocol = json.loads(summary_path.read_text()), json.loads(protocol_path.read_text())
    if (summary["complete"] is not True or summary["protocol"] != protocol
            or protocol["source_changed_during_run"] is not False
            or protocol["source_hashes_before"] != protocol["source_hashes_after"]
            or len(summary["rows"]) != 1 or protocol["format"] not in {
                "public_ycb_bias_recovery_development_r16", "public_ycb_bias_budget_development_r17"}):
        raise ValueError("Expected one completed unchanged-source R16/R17 case")
    if not np.isfinite(seconds) or seconds <= 0.:
        raise ValueError("Positive finite tail window required")
    row = summary["rows"][0]
    if row["runner_error"]:
        raise ValueError("Runner-error case cannot establish tactile diagnostics")
    case = checked_path(directory, row["path"])
    scene, metadata = checked_path(case, "scene.xml"), checked_path(case, "metadata.json")
    if json.loads(metadata.read_text()) != row or (row["scene_unchanged_since_load"] is not True
            or row["scene_sha256_before_load"] != digest(scene) or row["scene_sha256_at_finalize"] != digest(scene)):
        raise ValueError("Case metadata/runtime scene hash differs")
    paths = {"summary": summary_path, "protocol": protocol_path, "metadata": metadata, "scene": scene,
        "script": Path(__file__).resolve(),
        "helper_vertical": Path(digest.__code__.co_filename).resolve(),
        "helper_auditor": Path(checked_path.__code__.co_filename).resolve(),
        "helper_evidence_model": Path(load_evidence_model.__code__.co_filename).resolve(),
        "reference_r4_reader": Path(__file__).resolve().with_name("develop_public_ycb_force_band_r4.py")}
    if digest(paths["reference_r4_reader"]) != protocol["source_hashes_before"]["scripts/develop_public_ycb_force_band_r4.py"]:
        raise ValueError("Reference tactile reader differs from recorded execution source")
    for name in ("native", "preload", "load_transfer_probe"):
        artifact = row["artifacts"][name]
        path = checked_path(case, artifact["path"])
        if digest(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError("Artifact compressed hash/size differs: " + name)
        paths[name] = path
    before = {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for key, path in paths.items()}
    model, counts = load_evidence_model(scene), Counter()
    equal(model.opt.timestep, .002, "Native timestep", 1e-12)
    hand, target = hand_inventory(model), model.geom("target_geom").id
    preload, probe = TimedStream(paths["preload"]), TimedStream(paths["load_transfer_probe"])
    tails, windows, close_start, close_count = deque(maxlen=int(np.ceil(seconds / .002)) + 3), [], None, 0
    for index, sample in enumerate(json_array(paths["native"]), 1):
        if tick(sample["timestamp"]) != index:
            raise ValueError("Native stream must remain contiguous at 2ms from first integration")
        counts["native_samples"] += 1
        patches = rebuild_patches(sample, model, hand)
        counts["all_external_contact_observations"] += len(patches)
        choices = []
        for observed in probe.at(index):
            check_pressures(patches, observed)
            counts["contemporaneous_probe_pressure_checks"] += 1
            if abs(float(observed["solve_age_s"] or 0.)) <= 1e-9 and observed["solve_age_s"] is not None:
                association = observed["sampled_public_mesh_association"]
                if association["accepted"]:
                    choices.append(("current_probe_association", association_indices(patches, association["patches"])))
                    counts["current_probe_association_checks"] += 1
        for observed in preload.at(index):
            check_pressures(patches, observed["original_probe_snapshot"])
            counts["contemporaneous_preload_pressure_checks"] += 1
            current = observed["current_target_lp"]
            if current.get("feasible"):
                indices = association_indices(patches, current["patches"], tracking=True)
                counts["current_tracking_retained_patch_checks"] += len(indices)
                if not choices:
                    choices.append(("current_tracking_retained_association", indices))
        scope, associated = choices[0] if choices else ("no_current_association_record", None)
        complement = None if associated is None else [i for i in range(len(patches)) if i not in associated]
        record = {"timestamp_s": sample["timestamp"], "association_scope": scope,
            "tactile_patches_native_order": patches, "associated_raw_input_indices": associated,
            "all_external": forces(patches),
            "associated_hypothesis": None if associated is None else forces(patches, associated),
            "unassociated_external": None if complement is None else forces(patches, complement),
            "evaluation_only_target": forces(patches, evaluation_only(sample, hand, target)),
            "external_patches": len(patches), "associated_patches": None if associated is None else len(associated)}
        if associated is not None:
            equal(np.asarray(record["associated_hypothesis"]["force_on_hand_world_n"])
                + record["unassociated_external"]["force_on_hand_world_n"],
                record["all_external"]["force_on_hand_world_n"], "Associated/complement force partition")
        if sample["state"] == "CLOSE":
            if close_start is None:
                close_start = sample["timestamp"]
            close_count += 1
            tails.append(record)
        elif tails:
            windows.append(summarize(tails, len(windows) + 1, close_start, close_count, seconds))
            tails.clear()
            close_start, close_count = None, 0
    if tails:
        windows.append(summarize(tails, len(windows) + 1, close_start, close_count, seconds))
    if preload.next is not None or probe.next is not None or not windows:
        raise ValueError("Unmatched diagnostic observations or missing CLOSE window")
    if counts["native_samples"] != row["native_audit"]["native_samples"]:
        raise ValueError("Native sample denominator differs")
    if any(digest(path) != before[name]["sha256"] for name, path in paths.items()):
        raise ValueError("An input changed during offline analysis")
    return {"format": "public_ycb_tactile_vector_reconstruction_v1", "inputs": before,
        "case": row["path"], "counts": dict(counts), "native_timestep_s": .002,
        "synthetic_sign_sanity_passed": True, "windows": windows, "inputs_unchanged_during_analysis": True,
        "runtime": {"numpy": np.__version__, "mujoco": mujoco.__version__},
        "field_definitions": {
            "perception_filter": "Exact r4 native-order reconstruction: include a contact iff exactly one geom belongs to the known left hand. External geom identity, target pose, target COM and target motion are never used by this filter or association-index selection.",
            "force_sign": "force_on_hand=sign*contact_frame.T@local_force; sign=+1 if hand is geom2 else -1. force_hand_on_external is its negative. Associated-hypothesis forces are hand-on-the-estimated-object only as an association hypothesis.",
            "decomposition": "normal_on_hand=pressure*unit_normal_on_hand; shear=actual_force_on_hand-normal_on_hand. Action/reaction negates all three. These are 3D forces, not a full six-axis contact/wrist wrench; native torsional/rolling contact moments are not included.",
            "association": "Only a probe association solved at this exact timestamp, or a current preload target's raw_input_index, is joined to native-order patches. Retained tracking patches use active-finger/.05N filters; their complement is outside that retained set, not a proof of wrong-object contact. Missing current association is unknown, never zero/unassociated.",
            "timing": "Same-observation native cache and probe/preload reads are checked for exact pressure and accepted-patch point/normal/pressure agreement. A stale sampled LP is not treated as a current association. Original contact_sample omits force vectors and is not persisted separately; no comparison against a nonexistent raw sensor snapshot is claimed.",
            "evaluation_only_target": "A separate privileged target-geom selection is evaluated after perception reconstruction and never supplies the perception filter/association or controller. Comparison only, not a deployable measurement.",
            "local_frame": "World-coordinate tactile vectors are sufficient for vertical-load decomposition. No link-local force/moment or hand FK reconstruction is claimed by this minimal diagnostic."},
        "scope": "Offline recorded-force diagnostic only, not a new sensor, physical hardware validation, causal intervention, force-control success, grasp success or permission to bypass preload/probe/task gates. Downward shear in a supported grasp can change sign during later motion. No full case or additional integration was run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=.25)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = analyze(args.directory, args.window_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "counts": result["counts"], "windows": len(result["windows"])}))


if __name__ == "__main__":
    main()
