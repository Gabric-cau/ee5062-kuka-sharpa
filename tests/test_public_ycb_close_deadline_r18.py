"""Deadline isolation and arm-clock arithmetic; no full task is run here."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import develop_public_ycb_close_deadline_r18 as r18
from test_public_ycb_bias_budget_r17 import PROFILE, real_model

r17 = r18.r17

STUB_SOURCE = """
def run_task(model, state, cartesian_motion=True):
    def move(state, target, duration, closure):
        duration = max(duration, _phase_seconds.get(state.value, duration))
        steps = int(duration / model.opt.timestep)
        segment_steps = max(1, round(0.02 / model.opt.timestep))
        start = start_position = 0.0
        end_position = target
        output = []
        next_position = 0.0
        for step in range(steps):
            if cartesian_motion and step % segment_steps == 0:
                end_step = min(steps, step + segment_steps)
                next_position = interpolate(start_position, end_position, end_step / steps)
            if cartesian_motion:
                desired_command = next_position
            else:
                desired_command = interpolate(start, target, (step + 1) / steps)
            alpha = (step + 1) / steps
            output.append((desired_command, alpha))
            if early_ready and step == 20:
                break
        return output
    return move(state, 1.0, 3.0, 0.85)
"""


def stub_task(source, close_seconds=16., early=False):
    namespace = {"State": SimpleNamespace(CLOSE=SimpleNamespace(value="CLOSE")),
        "_phase_seconds": {"CLOSE": close_seconds}, "early_ready": early,
        "interpolate": lambda start, end, fraction: start + (end - start) * fraction}
    exec(compile(source, "<deadline_test>", "exec"), namespace)
    return namespace["run_task"], namespace


@pytest.mark.parametrize("cartesian", [True, False])
def test_original_arm_fractions_match_first8000_and_hold_during_extension(cartesian):
    tree, counts = r18.transform_controller(STUB_SOURCE)
    original, old = stub_task(STUB_SOURCE)
    extended, new = stub_task(tree, close_seconds=32.)
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    before = original(model, old["State"].CLOSE, cartesian)
    after = extended(model, new["State"].CLOSE, cartesian)
    assert len(before) == 8000 and len(after) == 16000
    assert [row[0] for row in before] == [row[0] for row in after[:8000]]
    assert all(row[0] == 1. for row in after[8000:])
    assert counts == {"arm_step_injections": 1, "arm_fraction_replacements": 2, "hand_fraction_replacements": 0}
    # The hand fraction expression is deliberately untouched by the arm transform.
    assert after[-1][1] == before[-1][1] == 1.
    assert after[7999][1] == .5
    assert ast.unparse(tree).count("alpha = (step + 1) / steps") == 1


@pytest.mark.parametrize("cartesian", [True, False])
def test_nonclose_fractions_and_existing_early_exit_are_unchanged(cartesian):
    tree, _ = r18.transform_controller(STUB_SOURCE)
    original, old = stub_task(STUB_SOURCE)
    extended, new = stub_task(tree, close_seconds=32.)
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    state = SimpleNamespace(value="TRANSFER")
    assert original(model, state, cartesian) == extended(model, state, cartesian)
    for close in (16., 32.):
        early, namespace = stub_task(tree, close_seconds=close, early=True)
        assert len(early(model, namespace["State"].CLOSE, cartesian)) == 21


@pytest.mark.parametrize("timestep", [.001, .004, float("nan")])
def test_physics_timestep_changes_are_rejected(timestep):
    tree, _ = r18.transform_controller(STUB_SOURCE)
    task, namespace = stub_task(tree, close_seconds=32.)
    with pytest.raises(ValueError):
        task(SimpleNamespace(opt=SimpleNamespace(timestep=timestep)), namespace["State"].CLOSE)


@pytest.mark.parametrize("source", [
    STUB_SOURCE.replace("end_step / steps", "end_step / (steps + 1)"),
    STUB_SOURCE.replace("(step + 1) / steps)", "(step + 2) / steps)"),
    STUB_SOURCE.replace("segment_steps = max(1, round(0.02 / model.opt.timestep))", "segment_steps = 3"),
    STUB_SOURCE.replace("steps = int(duration / model.opt.timestep)", "steps = 42"),
])
def test_source_structure_drift_is_rejected(source):
    with pytest.raises(RuntimeError, match="structure changed"):
        r18.transform_controller(source)


def test_transform_is_single_use_and_only_expected_ast_changes():
    tree, _ = r18.transform_controller(STUB_SOURCE)
    with pytest.raises(RuntimeError, match="unmodified"):
        r18.transform_controller(ast.unparse(tree))
    move = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "move")
    assignment = next(i for i, node in enumerate(move.body) if isinstance(node, ast.Assign)
        and ast.unparse(node) == "steps = int(duration / model.opt.timestep)")
    del move.body[assignment + 1:assignment + 4]
    for node in ast.walk(move):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "next_position" and isinstance(node.value, ast.Call):
            node.value.args[2] = ast.parse("end_step / steps", mode="eval").body
        elif isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "desired_command" and isinstance(node.value, ast.Call):
            node.value.args[2] = ast.parse("(step + 1) / steps", mode="eval").body
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(STUB_SOURCE), include_attributes=False)


def test_real_factory_preserves_hand_retry_budget_gates_and_module_globals(real_model, tmp_path):
    import mujoco
    from kuka_sharpa_grasp.pick_place import reset_task, Scenario
    data = mujoco.MjData(real_model)
    _, kin, _ = reset_task(real_model, data, Scenario(seed=3414, object_kind="ycb_can"))
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    guard = r17.r16.r13.r11.r9.r8.r6.LatchedPressureGuard(real_model, .349)
    parameters = (r17.r16.r14.PRELOAD_PARAMETERS, r17.r16.r13.JOINT_PARAMETERS,
        r17.r16.r14.PROBE_PARAMETERS, r17.r16.r13.r11.r9.r8.PHASE_SECONDS)
    before = copy.deepcopy(parameters)
    arguments = ("ycb_can", tmp_path, [], guard, profile, .349)
    old, source, _, _ = r17.controller_for(*arguments, additional_bias_limit_rad=.04)
    task, rendered, hands, estimated = r18.controller_for(*arguments)
    assert old.__globals__["_phase_seconds"]["CLOSE"] == 16.
    assert task.__globals__["_phase_seconds"] == dict(old.__globals__["_phase_seconds"], CLOSE=32.)
    assert task.__globals__["_phase_seconds"] is not old.__globals__["_phase_seconds"]
    for name in ("_probe_parameters", "_phase_seconds"):
        if name != "_phase_seconds":
            assert task.__globals__[name] == old.__globals__[name]
    config = r17.r16.r13.r11.r9.r8.r4.previous.selected_profile("ycb_can", profile)[1]
    estimated.bind(kin)
    first = task.__globals__["HandGraspController"](real_model, config)
    second = task.__globals__["HandGraspController"](real_model, config)
    assert hands == [first, second] and first is not second
    assert all(type(hand) is r17.BiasBudgetHand and hand.additional_bias_limit_rad == .04 for hand in hands)
    assert first.write.__func__ is r17.r16.BiasRecoveryHand.write
    assert first.after_step.__func__ is r17.r16.BiasRecoveryHand.after_step
    assert first.admit_probe.__func__ is r17.r16.BiasRecoveryHand.admit_probe
    assert parameters == before
    expected, _ = r18.transform_controller(source)
    assert ast.dump(ast.parse(rendered)) == ast.dump(expected)


def protocol_stub():
    baseline = json.loads(r18.R17_SUMMARY.read_text())["protocol"]
    return {key: copy.deepcopy(baseline[key]) for key in ("format", "nominal_profiles", "schedule",
        "joint_admittance", "phase_seconds", "constrained_preload", "force_control", "close_termination",
        "scoring", "load_transfer_probe", "probe_confirmation", "r16_bias_recovery_experiment",
        "r17_bias_budget_experiment")}


def test_preflight_and_snapshot_include_immutable_r17_dependencies(monkeypatch):
    declaration = r18.validate_inputs(PROFILE)
    baseline = json.loads(r18.R17_SUMMARY.read_text())
    assert declaration["inherited_r17_declaration"] == baseline["protocol"]["r17_bias_budget_experiment"]
    assert declaration["effective_phase_seconds"]["CLOSE"] == 32.
    assert declaration["effective_constrained_preload"]["close_seconds"] == 32.
    assert declaration["inherited_r17_phase_seconds"]["CLOSE"] == 16.
    assert declaration["additional_bias_limit_rad"] == .04
    sources = r18.snapshot(PROFILE)
    assert all(sources[key] == value for key, value in baseline["protocol"]["source_hashes_before"].items())
    for relative, expected in r18.PREFLIGHT_HASHES.items():
        assert sources[relative] == expected
    for relative in r18.NEW_SOURCES:
        assert sources[relative] == r17.r16.sha256(r18.PROJECT / relative)
    original = r17.r16.sha256
    monkeypatch.setattr(r17.r16, "sha256", lambda path: "0" * 64 if Path(path) == r18.R17_SUMMARY else original(path))
    with pytest.raises(ValueError, match="R18 preflight input"):
        r18.validate_inputs(PROFILE)


def test_nested_writers_preserve_true_r17_controller_and_effective_r18_deadline(tmp_path):
    declaration = r18.validate_inputs(PROFILE)
    runner = r18.isolated_runner(PROFILE, declaration)
    protocol = protocol_stub()
    saved_before = copy.deepcopy(protocol)
    assert runner.__globals__["previous"].RichNativeAudit is r17.TorqueNativeAudit
    runner.__globals__["previous"].atomic_json(tmp_path / "protocol.json", protocol)
    saved = json.loads((tmp_path / "protocol.json").read_text())
    assert saved["format"] == saved["runner_revision"] == r18.PROTOCOL_FORMAT
    assert saved["phase_seconds"] == dict(saved_before["phase_seconds"], CLOSE=32.)
    assert saved["constrained_preload"] == dict(saved_before["constrained_preload"], close_seconds=32.)
    assert "declared 32s" in saved["close_termination"]
    for key in ("joint_admittance", "scoring", "load_transfer_probe", "probe_confirmation",
            "r16_bias_recovery_experiment", "r17_bias_budget_experiment"):
        assert saved[key] == saved_before[key]
    assert saved["schedule"][0]["profile"] == "bias_budget_r17"
    assert saved["schedule"][0]["runner_revision"] == r18.PROTOCOL_FORMAT
    assert saved["schedule"][0]["close_seconds"] == 32.
    assert saved["schedule"][0]["additional_bias_limit_rad"] == .04
    assert saved["inherited_r17_phase_seconds"]["CLOSE"] == 16.
    assert saved["inherited_r17_force_control"] == saved_before["force_control"]
    assert "R18 32s" in saved["force_control"] and "R17 .04rad hand" in saved["force_control"]
    summary = {"protocol": saved, "rows": [{"kind": "ycb_can", "seed": 3414}],
        "controller_revision": "public_ycb_constrained_preload_r14"}
    runner.__globals__["previous"].atomic_json(tmp_path / "summary.json", summary)
    again = json.loads((tmp_path / "summary.json").read_text())
    assert again["protocol"] == saved
    assert again["controller_revision"] == "public_ycb_bias_budget_development_r17"
    assert again["runner_revision"] == r18.PROTOCOL_FORMAT
    assert again["rows"][0]["close_seconds"] == 32.
    assert runner.__globals__["PHASE_SECONDS"]["CLOSE"] == 32.


def test_metadata_without_factory_binding_preserves_original_runner_error(tmp_path):
    declaration = r18.validate_inputs(PROFILE)
    runner = r18.isolated_runner(PROFILE, declaration)
    value = {"kind": "ycb_can", "seed": 3414, "runner_error": "CONSTRUCTION_FAILED"}
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json", value)
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["runner_error"] == "CONSTRUCTION_FAILED"
    assert saved["missing_runtime_controller_binding"] is True
    assert "effective_task_phase_seconds" not in saved and "effective_runner_phase_seconds" not in saved


def test_actual_factory_binding_and_scene_hash_survive_nested_writers(real_model, tmp_path):
    declaration = r18.validate_inputs(PROFILE)
    runner = r18.isolated_runner(PROFILE, declaration)
    profile = json.loads(PROFILE.read_text())["ycb_can"]
    task, source, _, _ = runner.__globals__["controller_for"]("ycb_can", tmp_path, [],
        SimpleNamespace(), profile, .349)
    builder = runner.__globals__["build_pick_place_scene"]
    scene = tmp_path / "scene.xml"
    builder(scene, "ycb_can", physics_profile="physical-v6")
    value = {"kind": "ycb_can", "seed": 3414, "runner_error": None}
    runner.__globals__["previous"].atomic_json(tmp_path / "metadata.json", value)
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["effective_task_phase_seconds"] == task.__globals__["_phase_seconds"]
    assert saved["effective_runner_phase_seconds"] == runner.__globals__["PHASE_SECONDS"]
    assert saved["effective_task_phase_seconds"]["CLOSE"] == 32.
    assert saved["arm_source_transform"]["arm_fraction_replacements"] == 2
    assert saved["arm_source_transform"]["derived_controller_sha256"] == r18.hashlib.sha256(source.encode()).hexdigest()
    assert saved["scene_unchanged_since_load"] and saved["scene_sha256_before_load"] == r17.r16.sha256(scene)
    assert "missing_runtime_controller_binding" not in saved


def test_one_declared_case_and_no_automatic_budget_or_time_sweep(monkeypatch):
    declaration = r18.validate_inputs(PROFILE)
    monkeypatch.setattr(r18, "validate_inputs", lambda *args: declaration)
    monkeypatch.setattr(r18, "isolated_runner", lambda *args: pytest.fail("Invalid case must not run"))
    with pytest.raises(ValueError, match="frozen ycb_can:3414"):
        r18.run(Path("unused"), PROFILE, (("ycb_can", 3415),))
