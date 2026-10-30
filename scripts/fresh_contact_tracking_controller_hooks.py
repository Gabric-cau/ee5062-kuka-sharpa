"""Rebuild current contact Jacobians after an accepted fresh tracking target."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import provisional_preload_controller_hooks as provisional


hooks, r17 = provisional.hooks, provisional.r17
r13 = hooks.r13
allocate_tracking_targets = hooks.allocate_tracking_targets
R13_SOURCE_SHA256 = "ea37244e99045959c6ad841477b2d275823c1599f1ac9624af9e94c16bf866ed"
REMOVED_REASON = "CURRENT_PATCH_SET_CHANGED_WAIT_ONE_SAMPLE"
_ORIGINAL = r13.JointAdmittanceProbeHand.regulate
_source_path = Path(inspect.getsourcefile(_ORIGINAL)).resolve(strict=True)
if hashlib.sha256(_source_path.read_bytes()).hexdigest() != R13_SOURCE_SHA256:
    raise RuntimeError("Frozen R13 contact-tracking source changed")
_class_ast = next(node for node in ast.parse(_source_path.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "JointAdmittanceProbeHand")
_ORIGINAL_AST = next(node for node in _class_ast.body
    if isinstance(node, ast.FunctionDef) and node.name == "regulate")
if ast.dump(ast.parse(textwrap.dedent(inspect.getsource(_ORIGINAL))).body[0]) != ast.dump(_ORIGINAL_AST):
    raise RuntimeError("Loaded R13 regulate differs from its pinned source")

_OBSOLETE_IF = ast.parse('''if self.joint_activation_s is not None and changed:
    self.joint_target.update(feasible=False, reason="CURRENT_PATCH_SET_CHANGED_WAIT_ONE_SAMPLE")
    self._mode("HOLD_INVALID", data, self.joint_target["reason"])
    return
''').body[0]
_FRESH_TELEMETRY = ast.parse('''self.joint_target["fresh_contact_tracking"] = {
    "format": "r24_current_contact_tracking_v1",
    "timestamp_s": float(data.time),
    "changed_from_previous_sample": bool(changed),
    "activated_before_current_solve": self.joint_activation_s is not None,
    "current_contact_sample_count": len(sample),
    "current_target_raw_input_indices": [patch["raw_input_index"] for patch in target["patches"]],
    "status": "CURRENT_ASSOCIATION_AND_TARGET_ACCEPTED_JACOBIANS_PENDING",
}
''').body[0]
_REBUILT_TELEMETRY = ast.parse('''self.joint_target["fresh_contact_tracking"].update(
    status="CURRENT_CONTACT_JACOBIANS_REBUILT",
    rebuilt_group_count=len(groups),
    rebuilt_finger_solves=[{
        "finger": solve["finger"],
        "joint_names": list(solve["joint_names"]),
        "raw_input_indices_by_group": [list(group["raw_input_indices"]) for group in solve["groups"]],
        "jacobian_shape": [len(solve["compression_jacobian_m_rad"]), len(solve["joint_names"])],
    } for solve in self.joint_solves],
)
''').body[0]


def _transform_regulate(tree):
    if ast.dump(tree) != ast.dump(_ORIGINAL_AST):
        raise RuntimeError("Frozen R13 regulate AST changed")
    transformed = copy.deepcopy(tree)
    indices = [index for index, node in enumerate(transformed.body)
        if ast.dump(node) == ast.dump(_OBSOLETE_IF)]
    if len(indices) != 1:
        raise RuntimeError("Expected exactly the obsolete contact-change early return")
    transformed.body[indices[0]] = copy.deepcopy(_FRESH_TELEMETRY)
    transformed.body.append(copy.deepcopy(_REBUILT_TELEMETRY))
    return ast.fix_missing_locations(transformed)


def restore_original_ast(tree):
    """Remove only the declared observations and restore the one removed branch."""
    restored = copy.deepcopy(tree)
    indices = [index for index, node in enumerate(restored.body)
        if ast.dump(node) == ast.dump(_FRESH_TELEMETRY)]
    if len(indices) != 1 or ast.dump(restored.body[-1]) != ast.dump(_REBUILT_TELEMETRY):
        raise RuntimeError("Fresh-contact telemetry structure changed")
    restored.body[indices[0]] = copy.deepcopy(_OBSOLETE_IF)
    restored.body.pop()
    if ast.dump(restored) != ast.dump(_ORIGINAL_AST):
        raise RuntimeError("Fresh-contact method changes more than the declared branch")
    return restored


_REGULATE_AST = _transform_regulate(_ORIGINAL_AST)
restore_original_ast(_REGULATE_AST)


def _compile_regulate():
    # Compile a class cell, then bind the code to the original super() closure.
    tree = ast.parse("class _CompiledContactMethod:\n    pass\n")
    tree.body[0].body = [copy.deepcopy(_REGULATE_AST)]
    namespace = {"__name__": __name__}
    exec(compile(ast.fix_missing_locations(tree), "<fresh_contact_tracking_policy>", "exec"), namespace)
    code = namespace["_CompiledContactMethod"].regulate.__code__
    if code.co_freevars != _ORIGINAL.__code__.co_freevars:
        raise RuntimeError("Fresh-contact method must preserve the original super closure")
    inherited = hooks.SharedNormalBudgetHand.regulate
    method = FunctionType(code, dict(inherited.__globals__), _ORIGINAL.__name__,
        _ORIGINAL.__defaults__, _ORIGINAL.__closure__)
    method.__kwdefaults__ = _ORIGINAL.__kwdefaults__
    method.__annotations__ = dict(_ORIGINAL.__annotations__)
    method.__doc__ = _ORIGINAL.__doc__
    method.__qualname__ = _ORIGINAL.__qualname__
    return method


_fresh_regulate = _compile_regulate()


def tracking_method_binding():
    """Bind the inspectable wrapper and disclose the separately compiled inner method."""
    original_globals = _ORIGINAL.__globals__
    inner_globals = _fresh_regulate.__globals__
    keys = (original_globals.keys() | inner_globals.keys()) - {"allocate_tracking_targets", "__warningregistry__"}
    ast_sha = lambda tree: hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    wrapper = FreshContactTrackingHand.regulate
    wrapper_tree = ast.parse(textwrap.dedent(inspect.getsource(wrapper))).body[0]
    expected_wrapper = ast.parse("def regulate(self, readings, dt, carrying):\n    return _fresh_regulate(self, readings, dt, carrying)\n").body[0]
    path = Path(__file__).resolve(strict=True)
    return {"format": "r24_fresh_contact_method_binding_v1",
        "wrapper_source_file": path.relative_to(r17.PROJECT).as_posix(),
        "wrapper_source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "wrapper_method_source_sha256": hashlib.sha256(inspect.getsource(wrapper).encode()).hexdigest(),
        "wrapper_is_thin_delegate": ast.dump(wrapper_tree) == ast.dump(expected_wrapper),
        "inner_compilation_filename": _fresh_regulate.__code__.co_filename,
        "inner_bytecode_sha256": hashlib.sha256(_fresh_regulate.__code__.co_code).hexdigest(),
        "inner_bytecode_scope": "co_code; Python-version dependent, not the portable source identity",
        "ast_hash_format": "sha256(ast.dump(tree) without location attributes)",
        "original_r13_source_sha256": R13_SOURCE_SHA256,
        "original_method_ast_sha256": ast_sha(_ORIGINAL_AST),
        "transformed_method_ast_sha256": ast_sha(_REGULATE_AST),
        "restored_method_ast_sha256": ast_sha(restore_original_ast(_REGULATE_AST)),
        "inner_same_closure_as_original": _fresh_regulate.__closure__ is _ORIGINAL.__closure__,
        "inner_original_class_cell_retained": _fresh_regulate.__closure__[0].cell_contents is r13.JointAdmittanceProbeHand,
        "inner_other_globals_match_except_allocator_and_warning_registry": all(
            key in inner_globals and key in original_globals and inner_globals[key] is original_globals[key] for key in keys),
        "inner_allocator_is_shared_normal": inner_globals["allocate_tracking_targets"] is hooks.allocate_tracking_targets,
        "wrapper_allocator_metadata_matches_inner": allocate_tracking_targets is inner_globals["allocate_tracking_targets"],
        "scope": "The wrapper is new source, code and globals; only the inner method retains the original super closure and shared allocator bindings."}


class FreshContactTrackingHand(provisional.ProvisionalPreloadBudgetHand):
    """Use the accepted current patches; retain the inherited QP, writer and gates."""

    def regulate(self, readings, dt, carrying):
        return _fresh_regulate(self, readings, dt, carrying)

    def fresh_contact_telemetry(self):
        """Expose the accepted-target records already retained by the final writer."""
        events = [copy.deepcopy(event["current_target"]["fresh_contact_tracking"])
            for event in self.joint_events if event.get("event") == "NOMINAL_REFERENCE_WRITE"
            and "fresh_contact_tracking" in event.get("current_target", {})]
        return {"format": "r24_current_contact_tracking_v1", "method_binding": tracking_method_binding(), "events": events,
            "accepted_reconstruction_count": len(events),
            "changed_current_sample_count": sum(event["changed_from_previous_sample"] for event in events),
            "scope": "Accepted current-target reconstruction retained at writes; subsequent QP acceptance and task success are separate."}


_factory = hooks._isolated_function(r17.controller_for, BiasBudgetHand=FreshContactTrackingHand)


def controller_for(*args, additional_bias_limit_rad, **kwargs):
    return _factory(*args, additional_bias_limit_rad=additional_bias_limit_rad, **kwargs)
