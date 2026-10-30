# V7 Workbench Runbook

V7 is an independent extension of the validated V6 project. It provides a
compilable cluttered workbench, a six-object registry, native
per-fingertip MuJoCo touch channels, target-independent contact aggregation,
and deterministic keyboard-command replay. The physical task runner and
calibration sweep below produce explicit success/failure evidence; they do
not turn contact or a video into a success label.

## Continuous tool sorting

Run all three tools in one physical scene, without a reset between tools:

```bash
PYTHONPATH=src python scripts/run_v7_tool_sorting.py --seed 0 --video \
  --output results/my_continuous_sorting
```

On Windows, `START_V7_SORTING.cmd` runs this workflow and saves a synchronized
video. It uses declared upright staging guides and 25 mm dock tolerance,
checks native target-held lift, every-step collision and 12 N channel force,
and observes three seconds of stable release. The final check confirms that
earlier tools remain docked after the later actions. This is the primary
multi-tool demonstration; `START_V7_WORKBENCH.cmd` opens the separate
interactive teleoperation interface.

Install `requirements-motion.txt` for OMPL obstacle experiments. On headless
Linux use `MUJOCO_GL=egl`; on macOS omit that environment variable. Classic
MuJoCo physics uses the CPU, while EGL rendering can use the NVIDIA GPU.

For the actual carried screwdriver pick-transfer-dock experiment:

```bash
PYTHONPATH=src python scripts/run_v7_carried_obstacle_task.py --seed 0 --video \
  --output results/my_carried_screwdriver
```

This uses declared over-guard waypoints with collision-checked OMPL joint
connections. The actual free tool remains grasped during transfer; native
contact, proxy containment and release checks are retained. See
`V7_OBSTACLE_EXECUTION.md` for development failures and the scope of the
three recorded physical scenes. Object state and native contact attribution
also supply supervisory phase and safety guards.

On Windows run `SETUP_V7.cmd` once before either launcher. No physical robot
is needed for keyboard teleoperation. `V7_DELL_SETUP.md` lists the controls
and recording/replay files. A real person must operate the panel before
describing a recording as human teleoperation.

## Frozen Comparisons

The precise marker protocol runs the same newly seeded scene in nominal-map,
wrist-RGB-D feedback, and wrist-RGB-D without feedback conditions:

```bash
PYTHONPATH=src python scripts/evaluate_v7_rgbd.py --start-seed 900 --count 30 \
  --output results/my_paired_rgbd
```

It uses 25 mm docking tolerance, three seconds of stable release and a 12 N
channel force limit. Its perception assumes known marker appearance, radius,
height and upright staging. The task supervisor still uses common physical
lift/dock state guards in all conditions; body poses do not localize the grasp.
The recorded seeds 600-629 must remain held-out results rather than calibration
data. `V7_PROTOCOL_RESULTS.md` reports every failure and denominator.

For the separate same-seed three-tool feedback comparison:

```bash
PYTHONPATH=src python scripts/evaluate_v7_tool_feedback.py --start-seed 50 --count 5 \
  --output results/my_three_tool_feedback
```

This initializes an independent fixture for every tool/condition and preserves
all native contact and release-speed samples. It is separate from the
continuous sorting demonstration, which does not reset between tools.

For offline sensor-stress replay on saved camera observations:

```bash
PYTHONPATH=src python scripts/evaluate_v7_sensor_stress.py \
  --source results/my_paired_rgbd --output results/my_sensor_stress
```

This adds deterministic synthetic depth noise, dropout and appearance-bbox
occlusion to the same observations. It measures perception only, not the
closed-loop task success rate or performance of a physical camera.

## Build the scene

Run from the project root. Outputs must be new paths because the entry point
refuses to overwrite experiments.

```bash
.venv/bin/python scripts/build_v7_workbench.py \
  --output results/v7_workbench_seed0.xml \
  --manifest results/v7_workbench_seed0.json --seed 0
```

The scene contains three actionable primitive tool proxies (screwdriver,
wrench and marker), three physical distractors, two dividers, a rack base and
three accepted-object slots. Tool dimensions, mass and friction are recorded
in the manifest. Primitive tools are engineering proxies; the vendored YCB
can is the only current mesh distractor.

## Use the contracts

```python
from kuka_sharpa_grasp.object_registry import default_workbench_registry
from kuka_sharpa_grasp.tactile import TactileContactReader
from kuka_sharpa_grasp.teleop import KeyboardTeleop, TeleopActionLog

registry = default_workbench_registry()
target = registry.select_target("tool_marker")
keyboard = KeyboardTeleop()
log = TeleopActionLog([keyboard.command({"w", "target:tool_marker"}, 0.0)])
```

After loading the V7 XML, construct `TactileContactReader` with the optional
`v7_touch_thumb`, `v7_touch_index`, `v7_touch_middle`, `v7_touch_ring` and
`v7_touch_pinky` names. The reader aggregates hand-side contacts against any
non-hand geometry and reports normal/tangential force, contact centroid and a
bounded slip score. It never receives a target geometry ID.

## Run the workbench

For a reproducible headless demonstration with an overview, hand close-up,
wrist RGB-D pair and five tactile bars:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_v7_workbench.py \
  --duration 8 --fps 20 --output results/v7_demo_seed0
```

On macOS omit `MUJOCO_GL=egl`; on Windows use `START_V7_WORKBENCH.cmd` for
the interactive panel. The output includes `workbench.mp4`, `trace.json`,
`action_log.json` and `metadata.json`. This entry point is a workbench and
teleoperation demonstration: it intentionally writes `task_success: null`.
For a bounded interactive capture, use the Tk panel with `--view --duration`:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/run_v7_workbench.py \
  --view --duration 30 --output results/v7_operator_seed0
```

The panel writes the same files when the window closes or the duration is
reached, plus `operator_events.json` containing the raw Tk key, focus, target,
mode and finger-slider events. `metadata.json` records the event count and
explicitly says that human identity and intent are not verified. A real human
operation may be described as teleoperation only when the video and this log
were produced while a person operated the panel; scripted headless runs and
replays remain reproducible command demonstrations. Replay a recorded command stream with
`--replay path/to/action_log.json`; replay output is a new directory and keeps
the original command timestamps.

## Obstacle planning gate

The declared upright screwdriver gate adds a small transfer barrier and plans
with the held-tool proxy, so the route is checked against both the arm and the
carried tool:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v7_obstacle_gate.py \
  --output results/v7_obstacle_gate_seed0
```

The acceptance output records the blocked direct route, planner status, every
validated state, minimum clearance and collision count. This is a planning and
geometry gate; it does not label a physical pick-dock episode through the
barrier as completed.

## Batch metrics

Every recorded run contains `trace.json`, `events.json` and `metadata.json`.
Evaluate one or more runs with explicit denominators:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_v7_batch.py \
  --input results/v7_demo_seed0 results/v7_demo_seed1 \
  --output results/v7_batch/summary.json \
  --csv results/v7_batch/episodes.csv
```

The evaluator reports target selection, contact-based grasp evidence, lift,
drop, obstacle/non-target/target collisions, force peaks, docking error,
interventions and (when continuous fields were recorded) supported regrasp.
Missing `events.json` or missing contact/docking fields produce `null` and an
explicit unknown reason. `success_rate` uses only episodes with a complete
success decision; the JSON records both numerator and denominator. A video
or a scene compile therefore cannot inflate the batch rate.

The same command accepts `task_results.json` from `scripts/run_v7_tasks.py`
and expands its per-tool traces. The runner writes a sibling `events.json`
stream, which the evaluator splits at `OPERATOR_TARGET` markers before
computing each episode's collision denominator.

## Physical multi-tool task and calibration

Run the three actionable tools with the same seeded workbench. The runner
executes `approach -> align -> close -> tactile hold -> lift -> transfer ->
dock -> release`; it never writes an object's free-joint qpos during an
episode. Use a new output directory for every run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v7_tasks.py \
  --seed 0 --phase-timeout 6 --hold-seconds .1 \
  --output results/v7_protocol_seed0_shared

PYTHONPATH=src .venv/bin/python scripts/run_v7_tasks.py \
  --seed 0 --phase-timeout 6 --hold-seconds .1 --manual \
  --output results/v7_protocol_seed0_manual
```

The process exits with code 2 when any task fails. Preserve that directory and
evaluate it with `scripts/evaluate_v7_batch.py`; named failures stay in the
denominator. To tune wrist rotation, approach direction and grasp offset using
the same physical executor, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_v7_grasp.py \
  --object tool_marker --seed 7 --count 12 --phase-timeout 6 \
  --output results/v7_calibration_marker
```

The calibration score requires sustained target contact in the lift phase,
support-free motion, multiple contacting fingers and no overload/collision.
Whole-episode maximum height is diagnostic only. A lift candidate is separate
from full task success, which also requires docking, release and settled
verification. The sweep records compressed traces and source hashes; qpos
edits are limited to episode reset.

## Synchronized physical task video

Add `--video --fps 20` to the physical task runner. Each video is sampled
directly after the same MuJoCo integration step recorded in the task trace:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v7_tasks.py \
  --object tool_marker --seed 100 --video --fps 20 \
  --output results/v7_marker_live_video_seed100
```

On a Linux 4090 worker prefix the command with `MUJOCO_GL=egl`. The output
includes `video/tool_marker.mp4`, synchronized overview/hand/wrist RGB-D
views and tactile bars. First, peak-tactile and final PNGs plus per-view pixel
statistics are saved beside the video. `video/metadata.json` declares
`synchronized_recorded_physics`; the recorder never advances physics or
reconstructs object qpos. Task success comes from the executor's measured
lift/dock/release criteria.

Each target also gets its own `action_log.json`, `trace.json`, `events.json`
and `metadata.json` directory. These logs accept the existing teleoperation
replay entry point; use the original seed and the recorded `duration_s`:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v7_workbench.py \
  --seed 100 --replay results/v7_marker_live_video_seed100/tool_marker/action_log.json \
  --duration 12.08 --fps 20 --output results/v7_marker_replay_seed100
```

The replay workbench retains `task_success: null` until an evaluator is run.
The parent `scene_manifest.json` records seeded geometry,
physical object parameters, compiled scene hash and Python source hashes.
Video writers and renderers close in a `finally` block, including failed tasks.

`V7Kinematics`, `CollisionChecker` and `RRTConnectPlanner` provide the
collision-aware motion boundary. The planner uses OMPL RRT-Connect when the
optional `ompl` Python package is installed; otherwise it returns the explicit
`ompl_unavailable` status and no path is treated as a success.

The supported regrasp evaluator accepts only a continuous trace with minimum
hand force, support contact, finger aperture motion and a contact-count
transition. Endpoint-only pose changes are rejected. Passive table/object
contacts are ignored by the collision audit, while unknown robot contacts are
reported as unsafe.

## Tests

```bash
.venv/bin/python -m pytest tests/test_v7_workbench.py \
  tests/test_v7_tactile_teleop.py tests/test_v7_entrypoint.py \
  tests/test_v7_evaluator.py tests/test_v7_motion.py -q
.venv/bin/python -m pytest tests -q
```

The full suite protects the V6 baseline. V7 task experiments must use new
result folders and report target-selection, non-target collisions, dropped
objects, force peaks, docking error, operator interventions and relative
object-to-wrist motion during regrasp. A successful static scene compile or
teleoperation replay is not a grasp success.
