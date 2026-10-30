"""Deterministic keyboard-command and action-log replay primitives.

This module intentionally has no GUI, curses, or device dependency.  A future
viewer can feed currently-held key names to :class:`KeyboardTeleop`; the same
commands can be recorded and replayed on a 2060, the lab 4090, or in headless
tests with identical numeric results.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


def _vector3(value: Sequence[float], field: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(x) for x in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain three finite numbers") from error
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f"{field} must contain three finite numbers")
    return values


@dataclass(frozen=True)
class TeleopCommand:
    """One timestamped Cartesian/hand command.

    ``translation_m`` and ``rotation_rad`` are increments in the controller's
    selected reference frame.  ``grip_delta`` is normalized closure change;
    positive values close the hand.  Commands are data only and do not write
    MuJoCo state, preserving the existing command/physics boundary.
    """

    timestamp: float
    translation_m: tuple[float, float, float] = (0., 0., 0.)
    rotation_rad: tuple[float, float, float] = (0., 0., 0.)
    grip_delta: float = 0.
    buttons: tuple[str, ...] = ()
    selected_target: str | None = None
    source: str = "keyboard"

    def __post_init__(self):
        if not math.isfinite(float(self.timestamp)) or float(self.timestamp) < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "translation_m", _vector3(self.translation_m, "translation_m"))
        object.__setattr__(self, "rotation_rad", _vector3(self.rotation_rad, "rotation_rad"))
        if not math.isfinite(float(self.grip_delta)):
            raise ValueError("grip_delta must be finite")
        object.__setattr__(self, "grip_delta", float(self.grip_delta))
        if self.selected_target is not None and (not isinstance(self.selected_target, str)
                                                  or not self.selected_target):
            raise ValueError("selected_target must be a non-empty string or None")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        normalized = tuple(sorted({str(button) for button in self.buttons}))
        if any(not button for button in normalized):
            raise ValueError("buttons cannot contain empty names")
        object.__setattr__(self, "buttons", normalized)

    def as_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "translation_m": list(self.translation_m),
            "rotation_rad": list(self.rotation_rad),
            "grip_delta": float(self.grip_delta),
            "buttons": list(self.buttons),
            "selected_target": self.selected_target,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, record: dict) -> "TeleopCommand":
        if not isinstance(record, dict):
            raise ValueError("teleoperation command must be an object")
        try:
            return cls(**record)
        except TypeError as error:
            raise ValueError("invalid teleoperation command fields") from error


class TeleopActionLog:
    """Monotonic command recorder with deterministic JSON persistence/replay."""

    def __init__(self, commands: Iterable[TeleopCommand] = ()):
        self._commands: list[TeleopCommand] = []
        self._cursor = 0
        self._last_due_time = -math.inf
        for command in commands:
            self.append(command)

    @property
    def commands(self) -> tuple[TeleopCommand, ...]:
        return tuple(self._commands)

    def append(self, command: TeleopCommand) -> None:
        if not isinstance(command, TeleopCommand):
            raise TypeError("command must be a TeleopCommand")
        if self._commands and command.timestamp < self._commands[-1].timestamp:
            raise ValueError("commands must have nondecreasing timestamps")
        self._commands.append(command)

    def reset(self) -> None:
        self._cursor = 0
        self._last_due_time = -math.inf

    def due(self, timestamp: float) -> tuple[TeleopCommand, ...]:
        """Return each command due at ``timestamp`` exactly once."""
        if not math.isfinite(float(timestamp)) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if timestamp < self._last_due_time:
            raise ValueError("replay time moved backwards; reset the action log")
        self._last_due_time = float(timestamp)
        due: list[TeleopCommand] = []
        while self._cursor < len(self._commands) and self._commands[self._cursor].timestamp <= timestamp + 1e-12:
            due.append(self._commands[self._cursor])
            self._cursor += 1
        return tuple(due)

    def replay(self) -> Iterator[TeleopCommand]:
        """Yield a fresh deterministic sequence without changing the cursor."""
        yield from self._commands

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "kuka-sharpa-v7-teleop", "commands": [item.as_dict() for item in self._commands]}
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_json(cls, path: str | Path) -> "TeleopActionLog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != "kuka-sharpa-v7-teleop":
            raise ValueError("unsupported teleoperation log format")
        commands = payload.get("commands")
        if not isinstance(commands, list):
            raise ValueError("teleoperation log commands must be a list")
        return cls(TeleopCommand.from_dict(item) for item in commands)


@dataclass(frozen=True)
class KeyboardTeleopConfig:
    translation_step_m: float = .01
    rotation_step_rad: float = .08
    grip_step: float = .08

    def __post_init__(self):
        values = (self.translation_step_m, self.rotation_step_rad, self.grip_step)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("keyboard teleop steps must be finite and positive")


class KeyboardTeleop:
    """Map a set of held key names to one deterministic command."""

    def __init__(self, config: KeyboardTeleopConfig = KeyboardTeleopConfig()):
        self.config = config

    def command(self, keys: Iterable[str], timestamp: float, *, source: str = "keyboard") -> TeleopCommand:
        held = {str(key).lower() for key in keys}
        s, r, g = self.config.translation_step_m, self.config.rotation_step_rad, self.config.grip_step
        translation = (
            s * (int("d" in held) - int("a" in held)),
            s * (int("w" in held) - int("s" in held)),
            s * (int("e" in held) - int("q" in held)),
        )
        rotation = (
            r * (int("j" in held) - int("l" in held)),
            r * (int("i" in held) - int("k" in held)),
            r * (int("o" in held) - int("u" in held)),
        )
        selected = next((key[len("target:"):] for key in sorted(held) if key.startswith("target:")), None)
        buttons = tuple(sorted(key for key in held if key in {"space", "r", "escape"}))
        return TeleopCommand(timestamp=float(timestamp), translation_m=translation,
                             rotation_rad=rotation,
                             grip_delta=g * (int("]" in held) - int("[" in held)),
                             buttons=buttons, selected_target=selected, source=source)


__all__ = ["TeleopCommand", "TeleopActionLog", "KeyboardTeleop", "KeyboardTeleopConfig"]
