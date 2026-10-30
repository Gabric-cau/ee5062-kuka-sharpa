from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_supervised_experiment.py"
SPEC = importlib.util.spec_from_file_location("supervised_experiment", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def terminal(job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        observed = supervisor.inspect(job)
        if observed["status"]["terminal"]:
            return observed["status"]
        time.sleep(.02)
    pytest.fail("Detached quick child did not reach a terminal status")


def test_detached_child_retains_nonzero_exit_and_output(tmp_path):
    job = tmp_path / "job"
    result = supervisor.start(job, tmp_path, [sys.executable, "-c",
        "import os,sys; print(os.environ['EE5062_JOB_TEST']); sys.exit(7)"],
        {"EE5062_JOB_TEST": "captured-output"})
    assert result["launched"] is True and result["experiment_returncode"] is None
    status = terminal(job)
    assert status["state"] == "FINISHED" and status["returncode"] == 7
    assert status["child_pid"] > 0
    assert (job / "experiment.log").read_text().strip() == "captured-output"
    assert json.loads((job / "request.json").read_text())["cwd"] == str(tmp_path)
    with pytest.raises(FileExistsError):
        supervisor.start(job, tmp_path, [sys.executable, "-c", "pass"], {})
    with pytest.raises(FileExistsError):
        supervisor.work(job)


def test_missing_executable_is_terminal_without_invented_child_exit(tmp_path):
    job = tmp_path / "missing"
    supervisor.start(job, tmp_path, [str(tmp_path / "nonexistent")], {})
    status = terminal(job)
    assert status["state"] == "SUPERVISOR_ERROR"
    assert status["returncode"] is None and "FileNotFoundError" in status["error"]


def test_launcher_can_exit_while_worker_continues(tmp_path):
    job = tmp_path / "independent"
    subprocess.run([sys.executable, str(SCRIPT), "start", "--job-dir", str(job),
        "--cwd", str(tmp_path), "--", sys.executable, "-c",
        "import time; time.sleep(.3); print('after-launcher')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=5)
    status = terminal(job)
    assert status["returncode"] == 0
    assert (job / "experiment.log").read_text().strip() == "after-launcher"


def test_supervisor_error_does_not_mark_live_child_terminal(tmp_path, monkeypatch):
    job = tmp_path / "write-error"
    job.mkdir()
    supervisor.atomic_json(job / "request.json", {"command": [sys.executable, "-c", "pass"],
        "cwd": str(tmp_path), "environment_overrides": {}})
    original = supervisor.atomic_json
    child = type("Child", (), {"pid": 123456789, "poll": lambda self: None})()
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: child)
    failed = []

    def write(path, value):
        if value.get("state") == "RUNNING" and "child_pid" in value and not failed:
            failed.append(True)
            raise OSError("injected status write failure")
        return original(path, value)

    monkeypatch.setattr(supervisor, "atomic_json", write)
    status = supervisor.work(job)
    assert status["state"] == "SUPERVISOR_ERROR" and status["terminal"] is False
    assert status["returncode"] is None and status["child_exit_observed"] is False
    observed = supervisor.inspect(job)
    assert observed["child"]["pid"] == child.pid
    assert observed["restart_safe_from_this_receipt"] is False
