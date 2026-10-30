"""Immutable-on-disk V7 teleoperation episode recording.

The recorder keeps operator commands and physical MuJoCo snapshots together.
It does not infer task success: a task evaluator must provide that value, and
the default is deliberately ``None``.  This makes an interactive recording a
reproducible experiment artifact rather than an unsupported success claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .teleop import TeleopActionLog, TeleopCommand


FORMAT = "kuka-sharpa-v7-teleop-episode"
FORMAT_VERSION = 1


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write one JSON artifact without leaving a partial final file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _finite_time(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite and nonnegative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


class TeleopEpisodeRecorder:
    """Record commands and actual post-step observations for one episode.

    ``initial_commands`` is used by replay runs so the emitted artifact keeps
    the exact input log.  Interactive runs pass no initial commands and append
    each command at the same cadence as the physics loop.  Snapshots are
    copied through JSON validation, which catches NumPy/scalar leakage before
    an experiment is presented as portable evidence.
    """

    def __init__(self, output: str | Path, *, seed: int, mode: str,
                 shared_control: bool, initial_commands: Iterable[TeleopCommand] = (),
                 metadata: dict[str, Any] | None = None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.mode = str(mode)
        if not self.mode:
            raise ValueError("mode must be non-empty")
        self.shared_control = bool(shared_control)
        self.action_log = TeleopActionLog(initial_commands)
        self.trace: list[dict[str, Any]] = []
        self.metadata = dict(metadata or {})
        self._closed = False
        self._last_snapshot_time = -math.inf

    @property
    def closed(self) -> bool:
        return self._closed

    def record_command(self, command: TeleopCommand) -> None:
        if self._closed:
            raise RuntimeError("episode recorder is already finalized")
        self.action_log.append(command)

    def record_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("episode recorder is already finalized")
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dictionary")
        copied = json.loads(json.dumps(snapshot, allow_nan=False))
        timestamp = _finite_time(copied.get("time_s"), "snapshot.time_s")
        if timestamp + 1e-12 < self._last_snapshot_time:
            raise ValueError("snapshots must have nondecreasing time_s")
        self._last_snapshot_time = timestamp
        self.trace.append(copied)

    def finalize(self, *, task_success: bool | None = None,
                 task_success_note: str = "No evaluator result was supplied.",
                 events: Iterable[dict[str, Any]] = (),
                 extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist action log, trace and metadata once and return metadata."""
        if self._closed:
            return json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        if task_success is not None and not isinstance(task_success, bool):
            raise TypeError("task_success must be True, False, or None")
        event_list = json.loads(json.dumps(list(events), allow_nan=False))
        metadata = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "mode": self.mode,
            "shared_control": self.shared_control,
            "duration_s": self.trace[-1]["time_s"] if self.trace else 0.0,
            "commands": len(self.action_log.commands),
            "frames": len(self.trace),
            "events": len(event_list),
            "task_success": task_success,
            "task_success_note": str(task_success_note),
            **self.metadata,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        _write_json_atomic(self.output / "action_log.json", {
            "format": "kuka-sharpa-v7-teleop",
            "commands": [command.as_dict() for command in self.action_log.commands],
        })
        _write_json_atomic(self.output / "trace.json", self.trace)
        _write_json_atomic(self.output / "events.json", event_list)
        _write_json_atomic(self.output / "metadata.json", metadata)
        self._closed = True
        return metadata


__all__ = ["FORMAT", "FORMAT_VERSION", "TeleopEpisodeRecorder"]
