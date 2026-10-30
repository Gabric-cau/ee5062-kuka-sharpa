"""Manifest-backed, resumable experiments on one local CPU physics host."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from .rollout import GraspRollout
from .scenarios import ScenarioSampler
from .scene import OBJECT_KINDS, PROJECT_ROOT, YCB_OBJECTS, build_scene


SCHEMA_VERSION = 1
PHYSICS_BACKEND = "classic_mujoco_cpu"
RENDERER_BACKEND = "none/headless"
CONTROLLERS = ("fixed", "force")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(project: Path = PROJECT_ROOT) -> dict:
    """Hash executable physics code and model assets, never mutable outputs."""
    names = ("__init__", "control", "evaluator", "physics", "rollout", "scenarios", "scene", "experiments")
    paths = [project / "src" / "kuka_sharpa_grasp" / f"{name}.py" for name in names]
    paths += [project / "scripts" / name for name in ("run_benchmark.py", "run_mass_sweep.py", "summarize_results.py")]
    paths += [project / "requirements.txt"]
    vendor = project / "vendor" / "mujoco_menagerie"
    for model, xml, assets in (("kuka_iiwa_14", "iiwa14.xml", "assets"), ("sharpa_wave", "left_hand.xml", "assets/left")):
        paths.append(vendor / model / xml)
        paths.extend(path for path in (vendor / model / assets).rglob("*") if path.is_file())
    paths.extend((vendor / spec["mesh_file"]).resolve() for spec in YCB_OBJECTS.values())
    files = {path.relative_to(project).as_posix(): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
             for path in sorted(set(paths))}
    return {"sha256": content_hash(files), "files": files}


def software_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
            "system": platform.system(), "system_release": platform.release(), "system_version": platform.version(),
            "architecture": platform.machine()}


def machine_identity() -> dict[str, str]:
    host = socket.gethostname()
    machine_guid = ""
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            machine_guid = str(winreg.QueryValueEx(key, "MachineGuid")[0])
    else:
        machine_id = Path("/etc/machine-id")
        if machine_id.exists():
            machine_guid = machine_id.read_text(encoding="utf-8").strip()
    return {"host": host, "machine_id": content_hash({"host": host, "guid": machine_guid,
                                                       "architecture": platform.machine()})}


@dataclass(frozen=True)
class ExperimentConfig:
    object_kind: str = "box"
    seeds: tuple[int, ...] = (0, 1, 2)
    masses: tuple[float, ...] = (0.15,)
    duration_s: float = 2.5
    xy_jitter: float = 0.0
    friction_range: tuple[float, float] | None = None
    mass_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.object_kind not in OBJECT_KINDS:
            raise ValueError(f"unsupported object kind: {self.object_kind}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonempty, unique and non-negative")
        if not self.masses or len(set(self.masses)) != len(self.masses) or any(mass <= 0 or not math.isfinite(mass) for mass in self.masses):
            raise ValueError("masses must be nonempty, unique and positive")
        if self.mass_range is not None and len(self.masses) != 1:
            raise ValueError("mass randomization cannot be combined with a multi-mass sweep")
        if not math.isfinite(self.duration_s) or self.duration_s < 0.002:
            raise ValueError("duration must allow at least one 0.002 s physics step")
        if not math.isfinite(self.xy_jitter) or self.xy_jitter < 0:
            raise ValueError("jitter must be finite and non-negative")
        for name, limits in (("friction", self.friction_range), ("mass", self.mass_range)):
            if limits is not None and (len(limits) != 2 or not all(math.isfinite(x) for x in limits)
                                       or limits[0] <= 0 or limits[0] > limits[1]):
                raise ValueError(f"{name} range must be finite, positive and ordered")

    def serialized(self) -> dict:
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, values: dict) -> "ExperimentConfig":
        result = dict(values)
        for name in ("seeds", "masses", "friction_range", "mass_range"):
            if result.get(name) is not None:
                result[name] = tuple(result[name])
        return cls(**result)


def make_trial_plan(config: ExperimentConfig, config_hash: str) -> list[dict]:
    base = (0.70, 0.0, 0.43) if config.object_kind == "box" else tuple(map(float, YCB_OBJECTS[config.object_kind]["target_pos"].split()))
    sampler = ScenarioSampler(base, config.xy_jitter, config.friction_range, config.mass_range)
    plan = []
    for mass in config.masses:
        mass_group = f"uniform[{float(config.mass_range[0])},{float(config.mass_range[1])}]" if config.mass_range else str(float(mass))
        for seed in config.seeds:
            spec = sampler.sample(seed, "fixed", mass)
            scenario = {"object_kind": config.object_kind, "seed": seed, "mass_group": mass_group,
                        "object_pos": list(spec.object_pos), "object_mass": spec.object_mass,
                        "object_friction": 0.8 if spec.object_friction is None else spec.object_friction}
            scenario_id = content_hash(scenario)
            for controller in CONTROLLERS:
                trial_id = content_hash({"config_hash": config_hash, "scenario_id": scenario_id, "controller": controller})
                plan.append({**scenario, "scenario_id": scenario_id, "controller": controller, "trial_id": trial_id})
    return plan


def _write_json(path: Path, value: object, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            # Atomic no-replace publication prevents a retry from overwriting evidence.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _run_lock(run_dir: Path):
    with (run_dir / ".run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("another local process is already running this run directory") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def load_manifest(run_dir: Path) -> dict:
    with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    if manifest.get("physics_backend") != PHYSICS_BACKEND or manifest.get("renderer_backend") != RENDERER_BACKEND:
        raise ValueError("manifest backend mismatch")
    if manifest["config_hash"] != content_hash(manifest["config"]):
        raise ValueError("manifest configuration hash mismatch")
    if manifest["source"]["sha256"] != content_hash(manifest["source"]["files"]):
        raise ValueError("manifest source hash mismatch")
    expected = make_trial_plan(ExperimentConfig.from_dict(manifest["config"]), manifest["config_hash"])
    if manifest["planned_trials"] != expected:
        raise ValueError("manifest trial plan mismatch")
    return manifest


def validate_resume(manifest: dict, config: ExperimentConfig, source: dict, versions: dict, machine: dict) -> None:
    for name, recorded, current in (("configuration", manifest["config_hash"], content_hash(config.serialized())),
                                    ("source", manifest["source"], source),
                                    ("software versions", manifest["versions"], versions),
                                    ("machine", manifest["machine"], machine)):
        if recorded != current:
            raise ValueError(f"resume {name} mismatch; create a new run instead")


def read_trial_records(run_dir: Path, manifest: dict) -> list[dict]:
    plan = {trial["trial_id"]: trial for trial in manifest["planned_trials"]}
    records = []
    for path in sorted((run_dir / "trials").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        trial_id = row.get("trial_id")
        if path.stem != trial_id or trial_id not in plan:
            raise ValueError(f"unexpected trial record: {path}")
        digest = row.pop("record_sha256", None)
        if digest != content_hash(row):
            raise ValueError(f"trial record hash mismatch: {path}")
        row["record_sha256"] = digest
        trial = plan[trial_id]
        expected = {"config_hash": manifest["config_hash"], "source_sha256": manifest["source"]["sha256"],
                    "run_id": manifest["run_id"], **manifest["machine"], "versions": manifest["versions"],
                    **{f"{key}_version": manifest["versions"][key] for key in ("python", "mujoco", "numpy")},
                    "physics_backend": PHYSICS_BACKEND, "renderer_backend": RENDERER_BACKEND,
                    "object_kind": trial["object_kind"], "seed": trial["seed"], "controller": trial["controller"],
                    "scenario_id": trial["scenario_id"], "mass_group": trial["mass_group"],
                    "object_mass": trial["object_mass"], "object_friction": trial["object_friction"],
                    "sampled_object_pos": trial["object_pos"], "duration_s": manifest["config"]["duration_s"],
                    **dict(zip(("object_x", "object_y", "object_z"), trial["object_pos"]))}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"trial metadata mismatch: {path}")
        if row.get("completed") is not True or row.get("status") not in ("success", "dropped", "no_contact", "contact_only", "error"):
            raise ValueError(f"incomplete or invalid trial record: {path}")
        records.append(row)
    order = {trial["trial_id"]: index for index, trial in enumerate(manifest["planned_trials"])}
    return sorted(records, key=lambda row: order[row["trial_id"]])


def _csv_rows(rows: list[dict]) -> list[dict]:
    return [{key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
             for key, value in row.items()} for row in rows]


def write_csv(path: Path, rows: list[dict], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            if rows:
                fields = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(_csv_rows(rows))
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_experiment(config: ExperimentConfig | None = None, *, run_dir: Path | None = None,
                   resume: bool = False, results_root: Path | None = None,
                   max_new_trials: int | None = None, runner: Callable | None = None) -> Path:
    """Run sequential paired trials; a killed process resumes from published JSON."""
    invocation_started = time.perf_counter()
    if max_new_trials is not None and max_new_trials < 0:
        raise ValueError("max-new-trials must be non-negative")
    source, versions, machine = source_identity(), software_versions(), machine_identity()
    identity_elapsed = time.perf_counter() - invocation_started
    if resume:
        if run_dir is None:
            raise ValueError("resume requires a run directory")
        run_dir = run_dir.resolve()
        manifest = load_manifest(run_dir)
        config = config or ExperimentConfig.from_dict(manifest["config"])
        validate_resume(manifest, config, source, versions, machine)
    else:
        config = config or ExperimentConfig()
        run_id = uuid.uuid4().hex
        if run_dir is None:
            root = results_root or PROJECT_ROOT / "results" / "local_runs"
            run_dir = root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{run_id[:8]}"
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        config_hash = content_hash(config.serialized())
        manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "config": config.serialized(),
                    "config_hash": config_hash, "source": source, "versions": versions, "machine": machine,
                    "physics_backend": PHYSICS_BACKEND, "renderer_backend": RENDERER_BACKEND,
                    "task_scope": "grasp/lift/hold from authored hand-aligned initial pose",
                    "mass_model": "body_mass varies; principal body_inertia scales from the compiled reference in proportion to mass",
                    "mass_model_assumption": "Fixed shape and center of mass; homogeneous-density approximation, not measured object inertia",
                    "physics_profile": "legacy-v5",
                    "randomized_inputs": [name for name, enabled in (("xy_pose", config.xy_jitter > 0),
                        ("sliding_friction", config.friction_range is not None), ("body_mass", config.mass_range is not None)) if enabled],
                    "planned_trials": make_trial_plan(config, config_hash)}
        manifest["unique_physical_scenarios"] = len({content_hash({key: trial[key]
            for key in ("object_kind", "object_pos", "object_mass", "object_friction")}) for trial in manifest["planned_trials"]})
        _write_json(run_dir / "manifest.json", manifest, immutable=True)
    physical_scenarios = {content_hash({key: trial[key] for key in ("object_kind", "object_pos", "object_mass", "object_friction")})
                          for trial in manifest["planned_trials"]}
    print(f"xy_jitter={config.xy_jitter:g} unique_physical_scenarios={len(physical_scenarios)} "
          f"planned_pairs={len(manifest['planned_trials']) // 2}", flush=True)
    with _run_lock(run_dir):
        records = read_trial_records(run_dir, manifest)
        finished = {row["trial_id"] for row in records}
        remaining = [trial for trial in manifest["planned_trials"] if trial["trial_id"] not in finished]
        compile_started = time.perf_counter()
        scene = build_scene(run_dir / "scene.xml", config.object_kind)
        if runner is None and remaining:
            model = mujoco.MjModel.from_xml_path(str(scene))
            runner = GraspRollout(model, mujoco.MjData(model)).run
        model_build_elapsed = time.perf_counter() - compile_started
        added = 0
        for trial in remaining:
            if max_new_trials is not None and added >= max_new_trials:
                break
            started = time.perf_counter()
            try:
                result = runner(seed=trial["seed"], controller=trial["controller"], duration=config.duration_s,
                                object_mass=trial["object_mass"], object_pos=tuple(trial["object_pos"]),
                                object_friction=trial["object_friction"])
                outcome = asdict(result)
                canonical_json(outcome)
            except Exception as exc:
                outcome = {"status": "error", "failure_reason": "rollout_error", "stage": "failed",
                           "lifted": False, "error_type": type(exc).__name__, "error_message": str(exc)}
            row = {**outcome, **{key: value for key, value in trial.items() if key != "object_pos"},
                   "sampled_object_pos": trial["object_pos"],
                   **dict(zip(("object_x", "object_y", "object_z"), trial["object_pos"])),
                   "run_id": manifest["run_id"], **machine, "config_hash": manifest["config_hash"],
                   "source_sha256": source["sha256"], "versions": versions,
                   **{f"{key}_version": versions[key] for key in ("python", "mujoco", "numpy")},
                   "physics_backend": PHYSICS_BACKEND, "renderer_backend": RENDERER_BACKEND,
                   "duration_s": config.duration_s, "completed": True,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.perf_counter() - started}
            row["record_sha256"] = content_hash(row)
            _write_json(run_dir / "trials" / f"{trial['trial_id']}.json", row, immutable=True)
            records.append(row)
            added += 1
            write_csv(run_dir / "results.csv", records)
            print(f"{trial['object_kind']} seed={trial['seed']} {trial['controller']} mass={trial['object_mass']:.6g} "
                  f"status={row['status']} elapsed={row['elapsed_s']:.3f}s", flush=True)
        records = read_trial_records(run_dir, manifest)
        write_csv(run_dir / "results.csv", records)
        finished = {row["trial_id"] for row in records}
        missing = [trial["trial_id"] for trial in manifest["planned_trials"] if trial["trial_id"] not in finished]
        invocation_id = uuid.uuid4().hex
        invocation = {"invocation_id": invocation_id, "run_id": manifest["run_id"], "resumed": resume,
                      "new_trials": added, "identity_elapsed_s": identity_elapsed,
                      "model_build_elapsed_s": model_build_elapsed,
                      "elapsed_wall_s": time.perf_counter() - invocation_started,
                      "finished_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(run_dir / "invocations" / f"{invocation_id}.json", invocation, immutable=True)
        invocations = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "invocations").glob("*.json")]
        _write_json(run_dir / "completion.json", {"run_id": manifest["run_id"], "planned": len(manifest["planned_trials"]),
                                                  "completed": len(records), "incomplete_trial_ids": missing,
                                                  "complete": not missing, "statuses": dict(Counter(row["status"] for row in records)),
                                                  "total_trial_elapsed_s": sum(row["elapsed_s"] for row in records),
                                                  "total_recorded_invocation_wall_s": sum(item["elapsed_wall_s"] for item in invocations),
                                                  "invocations": invocations})
    print(f"run_dir={run_dir}\ncompleted={len(records)}/{len(manifest['planned_trials'])}", flush=True)
    return run_dir


def aggregate_runs(inputs: list[Path]) -> tuple[list[dict], dict]:
    """Reject duplicated evidence and mixed protocols before producing statistics."""
    rows, planned, seen_runs, seen_trials = [], [], set(), set()
    identity = None
    for supplied in inputs:
        run_dir = supplied.parent if supplied.is_file() else supplied
        manifest = load_manifest(run_dir)
        comparison = (manifest["config_hash"], manifest["source"]["sha256"], canonical_json(manifest["versions"]))
        if identity is None:
            identity = comparison
        elif comparison != identity:
            raise ValueError("aggregation configuration/source/software versions mismatch; summarize incompatible runs separately")
        if manifest["run_id"] in seen_runs:
            raise ValueError(f"duplicate run: {manifest['run_id']}")
        seen_runs.add(manifest["run_id"])
        records = read_trial_records(run_dir, manifest)
        for row in records:
            key = (row["machine_id"], row["trial_id"])
            if key in seen_trials:
                raise ValueError(f"duplicate trial for host {row['host']}: {row['trial_id']}")
            seen_trials.add(key)
            rows.append(row)
        planned.extend({**trial, **manifest["machine"], "run_id": manifest["run_id"]} for trial in manifest["planned_trials"])
    completed_ids = {(row["run_id"], row["trial_id"]) for row in rows}
    groups = defaultdict(list)
    completed_groups = defaultdict(list)
    for trial in planned:
        groups[(trial["object_kind"], trial["mass_group"], trial["controller"], trial["host"], trial["machine_id"])].append(trial)
    for row in rows:
        completed_groups[(row["object_kind"], row["mass_group"], row["controller"], row["host"], row["machine_id"])].append(row)
    summary = []
    for key, trials in sorted(groups.items()):
        records = completed_groups[key]
        statuses = Counter(row["status"] for row in records)
        missing = [trial["trial_id"] for trial in trials if (trial["run_id"], trial["trial_id"]) not in completed_ids]
        summary.append({**dict(zip(("object_kind", "mass_group", "controller", "host", "machine_id"), key)),
                        "planned_trials": len(trials), "completed_trials": len(records), "incomplete_trials": len(missing),
                        "successes": statuses["success"], "success_rate_planned": statuses["success"] / len(trials),
                        "success_rate_completed": statuses["success"] / len(records) if records else None,
                        "dropped": statuses["dropped"], "no_contact": statuses["no_contact"],
                        "contact_only": statuses["contact_only"], "errors": statuses["error"],
                        "measured_contacts_trials": sum("final_contacts" in row for row in records),
                        "measured_force_trials": sum("peak_force" in row for row in records),
                        "mean_final_contacts": float(np.mean([row["final_contacts"] for row in records if "final_contacts" in row]))
                            if any("final_contacts" in row for row in records) else None,
                        "mean_peak_force": float(np.mean([row["peak_force"] for row in records if "peak_force" in row]))
                            if any("peak_force" in row for row in records) else None,
                        "incomplete_trial_ids": missing})
    details = {"runs": sorted(seen_runs), "planned_trials": len(planned), "completed_trials": len(rows),
               "incomplete_trials": len(planned) - len(rows), "config_hash": identity[0] if identity else None,
               "source_sha256": identity[1] if identity else None, "physics_backend": PHYSICS_BACKEND,
               "renderer_backend": RENDERER_BACKEND}
    planned_pairs = {}
    for trial in planned:
        planned_pairs[(trial["run_id"], trial["scenario_id"])] = trial
    outcomes = defaultdict(dict)
    for row in rows:
        outcomes[(row["run_id"], row["scenario_id"])][row["controller"]] = row["status"]
    paired_groups = defaultdict(Counter)
    for pair_id, trial in planned_pairs.items():
        pair = outcomes[pair_id]
        if set(pair) != set(CONTROLLERS):
            status = "incomplete_pairs"
        else:
            fixed, force = pair["fixed"] == "success", pair["force"] == "success"
            status = "both_succeed" if fixed and force else "fixed_only" if fixed else "force_only" if force else "neither"
        key = (trial["object_kind"], trial["mass_group"], trial["host"], trial["machine_id"])
        paired_groups[key][status] += 1
    details["paired_outcomes"] = [{**dict(zip(("object_kind", "mass_group", "host", "machine_id"), key)),
                                  **{name: counts[name] for name in ("both_succeed", "fixed_only", "force_only", "neither", "incomplete_pairs")}}
                                 for key, counts in sorted(paired_groups.items())]
    details["unique_physical_scenarios"] = len({content_hash({key: trial[key]
        for key in ("object_kind", "object_pos", "object_mass", "object_friction")}) for trial in planned})
    return summary, details


def experiment_cli(*, mass_sweep: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Single-host CPU grasp experiments with immutable trial records")
    parser.add_argument("--seeds", type=int, help="number of seeds from seed-start (default: 3, or 1 for mass sweep)")
    parser.add_argument("--seed-start", type=int, help="inclusive first seed (default: 0)")
    parser.add_argument("--seed-stop", type=int, help="exclusive final seed; incompatible with --seeds")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--object", choices=OBJECT_KINDS)
    parser.add_argument("--randomize", action="store_true", default=None, help="enable paired XY pose jitter")
    parser.add_argument("--jitter", type=float)
    parser.add_argument("--friction", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--mass-range", type=float, nargs=2, metavar=("LOW", "HIGH"))
    if mass_sweep:
        parser.add_argument("--masses", type=float, nargs="+")
    else:
        parser.add_argument("--mass", type=float, help="fixed object mass in kg (default: 0.15)")
    parser.add_argument("--run-dir", type=Path, help="new isolated run directory; existing directories rejected")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="continue a manifest-backed run without rerunning completed trials")
    parser.add_argument("--max-new-trials", type=int, help="publish at most this many new trials, then stop for a later resume")
    parser.add_argument("--output", type=Path, help="optional fresh CSV export; never overwrite an existing CSV")
    args = parser.parse_args()
    try:
        if args.run_dir and args.resume:
            raise ValueError("choose --run-dir or --resume")
        if args.output and args.output.exists():
            raise ValueError(f"output already exists: {args.output}; choose a new path")
        if args.seed_stop is not None and args.seeds is not None:
            raise ValueError("--seed-stop and --seeds cannot be combined")
        baseline = ExperimentConfig.from_dict(load_manifest(args.resume)["config"]) if args.resume else ExperimentConfig(
            seeds=(0,) if mass_sweep else (0, 1, 2), masses=(0.05, 0.15, 0.30, 0.60) if mass_sweep else (0.15,))
        seeds = baseline.seeds
        if any(value is not None for value in (args.seed_start, args.seed_stop, args.seeds)):
            start = args.seed_start if args.seed_start is not None else 0
            stop = args.seed_stop if args.seed_stop is not None else start + (args.seeds if args.seeds is not None else len(baseline.seeds))
            seeds = tuple(range(start, stop))
        jitter = baseline.xy_jitter if args.jitter is None else args.jitter
        if args.randomize and args.jitter is None:
            jitter = baseline.xy_jitter or 0.005
        if args.randomize is None and not args.resume:
            if jitter:
                raise ValueError("--jitter requires --randomize")
            jitter = 0.0
        masses = (tuple(args.masses) if args.masses is not None else baseline.masses) if mass_sweep else (
            (args.mass,) if args.mass is not None else baseline.masses)
        if args.mass_range and not args.resume and mass_sweep and args.masses is None:
            masses = (0.15,)
        config = ExperimentConfig(object_kind=args.object or baseline.object_kind, seeds=seeds, masses=masses,
                                  duration_s=baseline.duration_s if args.duration is None else args.duration,
                                  xy_jitter=jitter,
                                  friction_range=tuple(args.friction) if args.friction else baseline.friction_range,
                                  mass_range=tuple(args.mass_range) if args.mass_range else baseline.mass_range)
        run_dir = run_experiment(config, run_dir=args.resume or args.run_dir, resume=bool(args.resume),
                                 max_new_trials=args.max_new_trials)
        if args.output:
            write_csv(args.output, read_trial_records(run_dir, load_manifest(run_dir)), immutable=True)
            print(f"exported {args.output}")
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
