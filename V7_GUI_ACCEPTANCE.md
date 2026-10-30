# V7 automated GUI recording and replay acceptance

This is a functional teleoperation-interface check, performed on macOS with
an actual Tk 9.0.4 window and programmatically generated Tk input. It is not a
human-operated experiment and is not a grasp-success result.

## Reproduce

Run from the project root with the installed project Python environment.
On Linux the renderer can use `MUJOCO_GL=egl`; omit that variable on macOS.

```bash
PYTHONPATH=src python scripts/run_v7_workbench.py \
  --view --automated-gui-check --duration 3 --fps 20 --seed 0 \
  --output results/v7_automated_gui_new

PYTHONPATH=src python scripts/run_v7_workbench.py \
  --replay results/v7_automated_gui_new/action_log.json \
  --duration 3 --fps 20 --seed 0 \
  --output results/v7_automated_gui_replay_new
```

For an operator-produced capture, omit `--automated-gui-check` and operate the
window. Such a run records `unverified_gui_input`; any report must identify
who actually operated it. A headless SSH session needs a graphical display to
open Tk; its EGL renderer alone does not provide that display.

The automated sequence invokes the target menu and shared/manual radio
buttons, changes actual finger sliders, generates movement and bracket-grip
keypress/release events, then exercises Escape, resume and focus-loss clearing.
The capture metadata explicitly records `input_provenance=automated_gui_input`.

## Recorded evidence

`results/v7_automated_gui_final_20261014` contains the GUI-produced `workbench.mp4`,
`operator_events.json`, `action_log.json`, `trace.json`, `events.json`,
`metadata.json` and scene. `results/v7_automated_gui_final_replay_20261014` contains
the headless replay. The GUI run's `acceptance_summary.json` records their
comparison and video-pixel checks; first/last PNGs provide inspection frames.

- Each trace has 1,500 post-step physics samples at 2 ms intervals, covering
  three simulation seconds without the former 50 ms trace gaps.
- The GUI captured 21 input events: keys, modes, target, finger sliders and
  focus loss. It saved 60 nonblank 1280x720 video frames.
- At every matching simulation time, maximum absolute error is zero for
  wrist position/quaternion, arm position/velocity/command, finger closure
  and hand joint position. Mode, target and status match throughout.
- The input/replay source hashes match. Workbench-script SHA256 is
  `5abe82ae5576503b91c49c4da43b8837ad8b30e3160a01cfee15099f4280dfb8`.
- `tests/test_v7_episode.py` and `tests/test_v7_stop.py`: 14 passed.
- The repaired stop freezes arm and hand actuator references, discards pending
  movement while stopped and preserves native physics integration. Seven
  focused regressions exercise operator, collision and tactile stop paths.

Recompute equivalence, required input coverage and decoded video checks:

```bash
python scripts/verify_v7_gui_replay.py \
  --gui results/v7_automated_gui_final_20261014 \
  --replay results/v7_automated_gui_final_replay_20261014 \
  --output results/my_gui_replay_receipt.json
```

The initial stop-fix recording with only 16 input events lacks complete
focus-loss coverage and is retained as a development attempt. The final
recording requires all interaction classes as well as physical equivalence.

The headless entry point automatically reads the recorded action log's sibling
`operator_events.json`. It applies mode changes, target selection, individual
finger targets and Escape at their recorded simulation times before physics.
Ordinary held-key movement already exists in the command log, so it is not
applied twice. Preserve both input files, the seed, initial manual/shared
setting and duration when replaying an operator recording.

## Claim boundary

This establishes a working GUI input/recording/replay path with measured
equivalence. It does not establish human usability, a human demonstration,
force-feedback hardware, remote-display responsiveness or successful grasping.
Both recordings deliberately preserve `task_success: null`.
The frozen r2a and r2b ZIP files were not modified by this check.

## Dell native desktop reproduction

`results/dell_v7_final_20261014_r3c/v7_windows_verify_20261014_r3c/final_dell_verification.json`
independently accepts the actual RTX 2060/WGL context, interactive desktop
session, source integrity, three-tool task and GUI/replay check. Dell captured
21 automated input events and 1,500 native 2 ms physics samples in each GUI
and replay trace, with zero error for the compared states. The input remains
automated, and the GUI recording does not establish task success.

The workbench displays five finger bars. Native contact aggregation and force
gates use six derived channels including the palm; the palm is logged and
controlled even though it has no displayed bar. Fingertip touch sensors are
separate from these derived force channels.
