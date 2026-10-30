#!/usr/bin/env python3
"""Keep one experiment's stdout and terminal status independent of SSH."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def start(job_directory, cwd, command, environment):
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir() or not command:
        raise ValueError("An existing working directory and command are required")
    job_directory = Path(job_directory).resolve()
    job_directory.mkdir(parents=True, exist_ok=False)
    request = {"format": "ssh_independent_experiment_v1", "created_utc": now(),
        "cwd": str(cwd), "command": list(command), "environment_overrides": environment,
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Launcher success is not experiment success; inspect terminal status and task artifacts."}
    atomic_json(job_directory / "request.json", request)
    atomic_json(job_directory / "status.json", {"state": "STARTING", "terminal": False,
        "created_utc": request["created_utc"], "returncode": None})
    try:
        with (job_directory / "supervisor.log").open("ab", buffering=0) as log:
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "worker", str(job_directory)], cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    except Exception as exc:
        atomic_json(job_directory / "status.json", {"state": "LAUNCH_ERROR", "terminal": True,
            "finished_utc": now(), "returncode": None, "error": repr(exc)})
        raise
    atomic_json(job_directory / "launcher.json", {"worker_pid": worker.pid,
        "started_utc": now(), "job_directory": str(job_directory)})
    return {"launched": True, "worker_pid": worker.pid, "job_directory": str(job_directory),
        "experiment_returncode": None}


def work(job_directory):
    job_directory = Path(job_directory).resolve(strict=True)
    request = json.loads((job_directory / "request.json").read_text())
    # Exclusive claim prevents accidentally launching a second physical run.
    with (job_directory / "worker_claim.json").open("x", encoding="utf-8") as claim:
        json.dump({"pid": os.getpid(), "claimed_utc": now()}, claim)
    started = now()
    status = {"state": "RUNNING", "terminal": False, "worker_pid": os.getpid(),
        "started_utc": started, "returncode": None, "child_exit_observed": False}
    atomic_json(job_directory / "status.json", status)
    child = None
    try:
        environment = dict(os.environ, **request["environment_overrides"])
        with (job_directory / "experiment.log").open("xb", buffering=0) as log:
            child = subprocess.Popen(request["command"], cwd=request["cwd"], env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            status["child_pid"] = child.pid
            atomic_json(job_directory / "status.json", status)
            returncode = child.wait()
        status.update(state="FINISHED", terminal=True, returncode=returncode, child_exit_observed=True,
            finished_utc=now())
    except Exception as exc:
        returncode = child.poll() if child is not None else None
        exit_observed = child is not None and returncode is not None
        status.update(state="SUPERVISOR_ERROR", terminal=child is None or exit_observed,
            returncode=returncode, child_exit_observed=exit_observed,
            error=repr(exc), child_state_unknown=child is not None and not exit_observed)
        if status["terminal"]:
            status["finished_utc"] = now()
    atomic_json(job_directory / "status.json", status)
    return status


def process_observation(pid, expected_arguments):
    result = {"pid": pid}
    try:
        os.kill(pid, 0)
        result["pid_exists"] = True
    except ProcessLookupError:
        result["pid_exists"] = False
    except PermissionError:
        result["pid_exists"] = None
    process_command = Path(f"/proc/{pid}/cmdline")
    if process_command.exists():
        arguments = process_command.read_bytes().rstrip(b"\0").split(b"\0")
        result["identity_matches"] = arguments == [part.encode() for part in expected_arguments]
    return result


def inspect(job_directory):
    job_directory = Path(job_directory).resolve(strict=True)
    status = json.loads((job_directory / "status.json").read_text())
    observation = {"status": status, "observed_utc": now()}
    worker_pid = status.get("worker_pid")
    if worker_pid is None and (job_directory / "launcher.json").exists():
        worker_pid = json.loads((job_directory / "launcher.json").read_text())["worker_pid"]
    if not status["terminal"] and worker_pid is not None:
        observation["worker"] = process_observation(worker_pid,
            [sys.executable, str(Path(__file__).resolve()), "worker", str(job_directory)])
    if status.get("child_pid") is not None and not status.get("child_exit_observed", False):
        request = json.loads((job_directory / "request.json").read_text())
        observation["child"] = process_observation(status["child_pid"], request["command"])
    observation["child_exit_verified"] = status.get("child_exit_observed", False)
    observation["restart_safe_from_this_receipt"] = bool(status["terminal"]
        and (status.get("child_pid") is None or observation["child_exit_verified"]))
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    launch = subparsers.add_parser("start")
    launch.add_argument("--job-dir", required=True, type=Path)
    launch.add_argument("--cwd", required=True, type=Path)
    launch.add_argument("--env", action="append", default=[])
    launch.add_argument("command", nargs=argparse.REMAINDER)
    for mode in ("worker", "status"):
        subparsers.add_parser(mode).add_argument("job_directory", type=Path)
    args = parser.parse_args()
    if args.mode == "start":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        environment = {}
        for item in args.env:
            key, separator, value = item.partition("=")
            if not separator or not key or "\0" in item:
                parser.error("Each --env requires NAME=value")
            environment[key] = value
        result = start(args.job_dir, args.cwd, command, environment)
    elif args.mode == "worker":
        result = work(args.job_directory)
    else:
        result = inspect(args.job_directory)
    print(json.dumps(result, indent=2), flush=True)
    return int(args.mode == "worker" and result["state"] == "SUPERVISOR_ERROR")


if __name__ == "__main__":
    raise SystemExit(main())
