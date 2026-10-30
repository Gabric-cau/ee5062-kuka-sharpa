# Physical Obstacle Route Acceptance

This experiment drives the real MuJoCo KUKA arm and Sharpa hand actuators
through an OMPL RRT-Connect path. The hand is empty. The original six free
workbench objects remain in the scene, together with the declared transfer
barrier from the planning gate. It demonstrates physical route execution,
not a completed grasp, carried-tool transfer, or docking task.

Run from the project root:

```bash
PYTHONPATH=src python scripts/run_v7_obstacle_execution.py \
  --output results/my_obstacle_execution --timebudget 6 --seed 0
```

Only reset initializes the robot arm and hand joints. During execution,
commands change motor targets, gravity compensation uses the existing robot
controller, and `mj_step` integrates every free body. No live object qpos is
written. The nominal route is blocked by a physical 16 mm box. The planner
checks the complete arm and hand plus an explicitly hypothetical tool box
at the wrist; that proxy is additional planning geometry and is never
presented as an actual grasped tool.

## Recorded Evidence

Frozen empty-hand revision, local runs on 2026-10-14:

| Run | Planned States | Physics Steps | Duration (s) | Actual Min Clearance (mm) | Max Tracking Error (mrad) | Final Goal Error (mrad) | Robot/Environment Contact Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `results/v7_obstacle_execution_final_seed0_20261014` | 917 | 17654 | 35.308 | 8.854 | 11.723 | 0.059 | 0 |
| `results/v7_obstacle_execution_final_seed1_20261014` | 1669 | 26338 | 52.676 | 8.838 | 11.652 | 0.073 | 0 |
| `results/v7_obstacle_execution_final_seed2_20261014` | 1093 | 17230 | 34.460 | 8.840 | 11.696 | 0.074 | 0 |

All three runs returned `exact_solution` and reached the goal with no native
robot self-contact. Maximum actual joint speed was below 0.120 rad/s, and
maximum arm torque was 48.85-52.10 Nm. RRT sampling is nondeterministic, so
the saved `planned_path.json` is the authority for each recorded path.

Each run also executes the straight joint interpolation as a physical
baseline. It reaches the barrier after 184 physics steps (0.368 s), produces
8 robot/environment contact steps with peak normal force 5.569 N, then
stops. The baseline is intentionally unsuccessful and is retained alongside
the successful planned route.

`planned/contacts.jsonl.gz` and `direct/contacts.jsonl.gz` record every native
contact, including robot/environment, robot self, and free-object support
contacts, at every 2 ms physics step. `physical_trace.jsonl.gz` records arm
commands, physical qpos/qvel, motor torque, tracking error, and actual
robot-to-environment geometry distance at 50 Hz. The reported minimum
clearance is therefore sampled at 50 Hz; the no-contact result is audited
at every physics step. `result.json` contains the experiment boundary,
source hashes, model manifest, and separate planned/baseline outcomes.

## Verification

```bash
PYTHONPATH=src python -m pytest -q tests/test_v7_obstacle_execution.py \
  tests/test_v7_obstacle_gate.py tests/test_v7_motion.py
```

These 9 tests passed locally. The physical regression verifies that the
blocked route causes native contact and stops; it does not replace the
three saved complete physical executions above.

## Physical Carried-Tool Task

The additional `scripts/run_v7_carried_obstacle_task.py` completes a physical
screwdriver pick, held lift, carried transfer, docking and stable release.
It preserves the six free workbench objects and the upright source/receiving
guides. An additional declared 70 x 18 x 310 mm guard lies between source and
receiver. This is a separate workcell profile; the empty-hand experiment and
all unsuccessful development probes remain available.

```bash
PYTHONPATH=src python scripts/run_v7_carried_obstacle_task.py \
  --seed 0 --output results/my_carried_route --video
```

Install `requirements-motion.txt` for the OMPL dependency on the validated
Linux host. The same script entry point requires working OMPL Python
bindings; the pinned PyPI 2.0.1 release has no native Windows wheel and OMPL
is absent from the Dell base environment. Use the validated Linux/4090
environment for this experiment. The script inserts its own source path.
Saved scene XML files record the
original machine's asset directory; rerun the source command to generate the
scene with the local asset path.

The controller uses declared map poses and calibrated grip geometry. It
does not use live evaluation object poses to select commanded wrist goals.
The shared task supervisor still reads object-state evaluation guards for
lift and dock completion. Over-guard Cartesian waypoints, spaced at most
30 mm apart, define a task-specific high corridor. Each local connection
is solved by OMPL RRT-Connect in a bounded joint-space neighborhood and
checked against the complete robot, non-target objects and a declared
oriented held-tool box. A dense, independently collision-checked local
shortcut is executed when valid; otherwise the returned OMPL path is used.
This is a declared-map route with known geometry, not unconstrained scene
exploration or learned motion planning.

MuJoCo motor targets execute the route at at most 0.08 rad/s commanded joint
speed. No live object qpos is written. A high-position wrist roll of 0.10 rad
over 2 s, followed by a 0.6 s settling period, prepares the tool for the
receiving guide. Docking uses an 8 N unloading setpoint while the acceptance
limit remains 12 N per native tactile channel, including the palm. These
parameters were developed on seed 0 and frozen before final seeds 0, 1 and 2.

## Current 4090 Reproduction

On 22 October 2026 the current checkout was rerun on the laboratory RTX4090
with seed 0 after installing the pinned `ompl==2.0.1` wheel into the project
Python 3.10 environment. The first attempt is retained separately in
`results/v7_carried_obstacle_current_20261021_seed0` and stopped at
`WAYPOINT_0_OMPL_UNAVAILABLE`. That failed attempt remains an explicit task
failure due to a missing dependency; it is separate from the successful rerun
in `results/v7_carried_obstacle_current_20261022_seed0`.

The rerun returned `success=true`: all 24 local waypoint queries were
`exact_solution`, all dense shortcuts passed the declared robot/hand/tool
collision checker, and the straight joint interpolation was blocked in 163 of
301 checked states. It integrated 23,389 contiguous 2 ms samples with zero
transfer or whole-task environment collision samples, zero tool-proxy
containment failures, 42.515 mm minimum actual guard clearance, 14.448 mm dock
error, and 5.783 N maximum native channel force. The saved MP4 and phase frames
are under the same result directory. This is a current single-seed reproduction
with declared geometry, not a randomized obstacle-generalization estimate.

Frozen carried-experiment revision on 2026-10-14 (before subsequent stop and
fixture contact-motion repairs):

| Seed | Accepted | Physics Samples | Dock Error (mm) | Max Channel Force (N) | Actual Guard Clearance (mm) | Stable Verification (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | yes | 23389 | 14.315 | 5.783 | 42.515 | 3.002 |
| 1 | yes | 23424 | 14.557 | 5.803 | 42.509 | 3.002 |
| 2 | yes | 23384 | 14.270 | 5.798 | 42.513 | 3.002 |

Results are in `results/v7_carried_obstacle_final_seed{0,1,2}_20261014`.
All three executions passed target-specific sustained held-lift checks,
25 mm dock tolerance, release without hand contact, and final linear/angular
speed limits of 0.02 m/s and 0.25 rad/s. Native transfer environment-contact
events, whole-task evaluated robot/environment collision samples, non-hand
robot self-contact events, and held-tool containment failures were all zero.
The physical tool remained inside the declared 45/45/145 mm proxy half-size
at every transfer step; the seed-0 maximum actual bounds were
29.3/34.0/118.6 mm. All three runs reported unchanged source hashes, with
the final entry point SHA256
`6a496ccba8ad9f6ba717d4e5c8bd50600ec9b42857bf0260a12375564fdb302f`.

Every 2 ms step records complete native contacts and a matching physical
snapshot in `native_contacts.jsonl.gz` and `physical_trace.jsonl.gz`.
Snapshots contain commanded arm goals, actual arm/hand state, requested
finger closures, target-attributed contact evidence and object poses.
Native contacts and object state also supply supervisory phase and safety
guards; declared priors supply grasp localization and route targets.
Neighboring closed fingers make shallow self-contact; this
is recorded rather than removed. Across final runs its peak force was
1.297-1.299 N, with maximum penetration about 0.615 mm. Thus the result is
collision-free transfer against the environment, not zero contact between
every pair of robot surfaces.

The straight carried joint interpolation is rejected by geometry checks
at 163-164 of 301 sampled states. This carried straight-path comparison is
planning evidence; it was not physically executed. The separate empty-hand
experiment above retains an actual blocked-route physical baseline.

The seed-0 video `video/tool_screwdriver.mp4` is sampled from the live
post-step physics at 20 Hz: 936 frames, 1280 x 720, 46.8 s. Overview, hand
close-up, wrist RGB/depth and tactile readings are synchronized. Video
metadata and `pixel_qa.json` confirm decoded, nonblank and changing frames;
success is computed from physics logs. These are three fixed-workcell
demonstrations with small seeded placement variation, including the
development seed. They are not a broad grasp-success benchmark.

Development failures `results/v7_carried_obstacle_probe{0..5}_20261014`
remain intact: probes 0 and 1 rejected local planning endpoints, probe 2
completed the carried route but failed stable docking (112 mm final error),
and probes 3-5 stopped after receiving-guide contact raised the palm channel
to about 31 N. Probe 6 first completed the calibrated slow-roll sequence.
These records document unsuccessful development candidates and are not
counted as held-out evaluation.

The added physical regression is:

```bash
PYTHONPATH=src python -m pytest -q tests/test_v7_carried_obstacle_task.py
```

It runs the complete carried task, checks the strict final criteria, and
audits matching 500 Hz native-contact and physical traces. It does not
replace the three saved executions or establish arbitrary-tool robustness.
The test passed locally on 2026-10-14: 1 passed in 113.02 s.

## New Frozen-Release 4090 Verification

The source subsequently received stop/focus-loss and fixture contact-motion
repairs. The tables above remain evidence for their saved frozen revisions;
their source hashes must not be replaced with the current checkout hashes.
The new [4090 release receipt](results/lab4090_v7_final_source_verify_r3_20261014/final_validation_receipt.json)
is accepted with all nine checks passing: frozen source checks before/after
execution, 292 software tests, actual RTX 4090 EGL rendering with four
nonblank panes, and all three carried processes and independent native-log
audits. The initial hardware probe import failure is retained; a new process
using the same immutable source completed the EGL probe.

| Seed | Accepted | Physics Samples | Dock Error (mm) | Max Channel Force (N) | Actual Guard Clearance (mm) | Native Stable Window (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | yes | 23389 | 14.448 | 5.783 | 42.515 | 3.000 |
| 1 | yes | 23424 | 14.521 | 5.803 | 42.509 | 3.000 |
| 2 | yes | 23384 | 14.435 | 5.797 | 42.513 | 3.000 |

These are new-source reruns of the development seed 0 and validation seeds
1/2, separate from the older table above. Each native VERIFY window has
1,501 contiguous samples at 2 ms, zero target-hand force, and linear/angular
speeds below the acceptance limits. The phase duration is 3.002 s; the
independent first-to-last sample window is 3.000 s. Recorded controller
hashes match the source used by the continuous sorting evaluation. The new
carried entry point SHA256 is
`a51b823550989db970d4b6482ae3da65ebf4f71f1ace4e968b266319c39fc555`.
The new [seed-0 video](results/lab4090_v7_final_source_verify_r3_20261014/carried_seed0/video/tool_screwdriver.mp4)
is actual RTX 4090 rendered simulation evidence, not human teleoperation.
Its video metadata peak of 12.038 N is the sum of six native contact channels
(five fingers plus palm), not a single channel. The frozen video displays
only the five finger bars. The 12 N acceptance gate includes the palm and
uses the maximum individual native hand channel reported in the table above;
the seed-0 maximum of 5.783 N is on the palm channel.

The 292-test run covers the frozen 336-file release archive (SHA256
`3e54242324e8b38dae4e5e3b30aee0c7f10e06844b0844ea4d535be2bd33e648`).
Later isolated public-object prototypes and Windows-launcher changes have
separate checks; this receipt does not establish another complete 292-test
run of the eventual final package. The
[local transfer verification](results/lab4090_v7_final_source_verify_r3_20261014/local_transfer_verification.json)
also passes: all 49 inventory files match byte counts and SHA256, and the
six complete raw logs retain matching contiguous 2 ms samples, zero native
target-hand contact throughout VERIFY, stable speeds and the accepted
per-channel force peaks. The separate
[Dell desktop receipt](results/dell_v7_final_20261014_r3c/v7_windows_verify_20261014_r3c/final_dell_verification.json)
accepts actual RTX 2060 rendering, GUI/replay and three continuous tool
transfers. OMPL is absent from that environment; physical OMPL avoidance
evidence comes from the lab 4090 runs above.

The repaired continuous sorting controller has its own 30-sequence native
evaluation: 29/30 complete sequences and 89/90 tool transfers, with maximum
channel force 11.707 N. This is the six-object upright fixture at +/-1.5 mm
initial XY jitter, fixed mass/friction and the declared map, rather than
30 additional obstacle-transfer trials. Its sequence confidence interval
and the retained seed-1004 placement failure are recorded in
[V7_PROTOCOL_RESULTS.md](V7_PROTOCOL_RESULTS.md).
