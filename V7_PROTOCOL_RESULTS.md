# V7 protocol evidence

This file separates development experiments by protocol and source revision.
The complete project acceptance gate is in `V7_REQUIREMENTS_MATRIX.md`.

## Current 4090 Marker Perception And Tactile Comparison

New seeds 1600-1629 completed 90 physical episodes with unchanged source
hashes and no runtime-error or unknown outcomes. Every condition uses the same
seeded marker XY map error (+/-5 mm), mass factor (0.8-1.2) and friction factor
(0.8-1.2) in the complete six-body workbench.

| Condition | Strict Success | Retained Failures |
| --- | ---: | --- |
| Nominal map + tactile | 25/30 | 1 lift, 2 close and 2 verification timeouts |
| Wrist RGB-D + tactile | 30/30 | None |
| Wrist RGB-D without tactile | 23/30 | 7 verification timeouts |

All 90 episodes pass the declared collision and 12 N channel-force gates.
The RGB-D estimator assumes known marker appearance, radius, height and upright
staging. This is a narrow known-object perturbation experiment, not public-YCB
grasp success, hardware accuracy or arbitrary-clutter generalization. Full
observations, estimates, traces, actions and failures are retained in
[`v7_rgbd_randomized_current_20261022_1600_1629`](results/v7_rgbd_randomized_current_20261022_1600_1629/).

## Current 4090 Three-Tool Paired Evaluation

The current checkout was rerun on the laboratory RTX4090 at seeds 40-44 using
the same fresh six-object upright fixture for the shared-control and no-tactile
conditions. Every pair completed with a known outcome: 30 episodes, 15 per
mode, and no runner errors or unknown samples.

| Tool | Shared tactile | No tactile | No-tactile failure |
| --- | ---: | ---: | --- |
| Marker | 5/5 | 5/5 | None |
| Screwdriver | 5/5 | 5/5 | None |
| Wrench | 5/5 | 0/5 | `LIFT_TIMEOUT` in all five |

All 25 successful episodes passed native held-lift; all 30 attempts passed
the collision and 12 N per-channel force gates. The largest channel force was 9.287 N with shared
control and 8.052 N in the completed screwdriver no-tactile episodes; the
wrench no-tactile episodes ended before lift and are not successes. The paired
summary and all per-episode scene manifests, actions, events and 2 ms traces
are in
[`v7_three_tool_feedback_current_20261021_40_44`](results/v7_three_tool_feedback_current_20261021_40_44/).

The synchronized continuous seed-0 demonstration for the same current source is
[`v7_tool_sorting_current_20261021_seed0/tool_sorting.mp4`](results/v7_tool_sorting_current_20261021_seed0/tool_sorting.mp4).
It contains 1,608 post-step frames at 20 FPS over 80.366 s, transfers marker,
screwdriver and wrench without scene resets, and is an autonomous visualization;
it is not human teleoperation evidence and is excluded from the paired batch.

This is a small fixed upright-fixture ablation with a declared map and known
tool identities. It is evidence that the current tactile retention path matters
for this wrench configuration under these conditions, not a randomized-clutter
generalization result, a learned-policy result, or human teleoperation evidence.

## Current Continuous Sorting Evaluation

The frozen contact-motion repair was evaluated on 30 fresh continuous
sequences at seeds 1000-1029. Each sequence transfers marker, screwdriver
and wrench in the same six-object workcell without resets and checks that
previously placed tools remain docked. The main sampling unit is a complete
sequence, rather than three independent transfers.

| Measure | Result |
| --- | ---: |
| Complete three-tool sequences | 29/30 (96.7%) |
| Accepted tool transfers | 89/90 (98.9%) |
| Target-held lift / collision gate / force gate | 90/90 each |
| Unknown or unattempted transfers | 0 |
| Maximum native hand-channel force | 11.707 N |
| Maximum accepted docking error | 24.689 mm |
| Minimum accepted stable release | 3.000 s |
| Complete-sequence 95% Wilson interval | 83.33-99.41% |

The unchanged acceptance gates are 12 N per native hand channel (including
the palm), 25 mm docking tolerance and three seconds of stable release,
with a 12 s phase timeout. Collision-free means no non-target
robot/environment contact exceeding both 2 mm penetration and 0.5 N normal
force. It permits recorded shallow contacts.

Seed 1004 remains a failure: the wrench ends 26.677 mm from its dock,
causing `VERIFY_TIMEOUT` / `NO_STABLE_RELEASE_WINDOW`. Its held lift,
collision and force checks pass, with maximum channel force 9.433 N, but
no qualifying release window is obtained. No threshold was relaxed. The
batch maximum of 11.707 N occurs during wrench docking at seed 1009.

This is a known-tool upright-fixture protocol with the declared correct
map, initial XY jitter of +/-1.5 mm (wrench Y fixed), and fixed masses and
friction. It is not the +/-5 mm / randomized-physics RGB-D protocol below.
The observed sequence rate exceeds 85%, but its confidence lower bound is
below 85%, so this sample does not establish a population guarantee of at
least 85%. Transfer outcomes within a sequence are dependent; 89/90 is a
descriptive transfer rate, not 90 independent scene trials.

Evidence is in
[the batch summary](results/v7_continuous_force_holdout1000_1029_20261014/summary.json),
[independent native verification](results/v7_continuous_force_holdout1000_1029_20261014/independent_native_verification.json)
and [confidence analysis](results/v7_continuous_force_holdout1000_1029_20261014/confidence_analysis.json).
All 90 compressed traces/action logs/event logs retain 1,210,444 contiguous
2 ms native samples. Their byte counts and SHA256 hashes, all 30 scene
hashes, and 44 saved/live source hashes were rechecked against the receipt;
source hashes agree before and after the batch. The frozen fixture entry
point SHA256 is
`a964483611f1f090d15090908f42b915ee4b893a133f136338c0e132395e4923`.

The preceding fresh-seed check at 90-94 is a separate 5/5 sequences and
15/15 transfers, with peak 9.516 N and maximum docking error 24.943 mm.
Its [independent receipt](results/v7_continuous_force_crossval90_94_20261014/independent_native_verification.json)
is preserved; its denominator is not pooled with the 30-sequence batch.
The current [seed-0 sorting video](results/v7_continuous_sorting_current_motion_relief_video_seed0_20261014/tool_sorting.mp4)
is an autonomous 3/3-transfer demonstration, excluded from both batches.

## Frozen Precise Paired Evaluation

`results/lab4090_v7_rgbd_paired_precise600_629_20261014` contains all 90
episodes, captured on the lab RTX 4090 with frozen controller calibration,
before the subsequent stop/contact-motion repairs.
The gate is a native target-held lift, collision-free motion, 12 N maximum
per channel, 25 mm final dock tolerance and three seconds of stable release.
Each seed has the same +/-5 mm initial XY error and 0.8-1.2 mass/friction
factors across all three conditions.

| Condition | Strict success | Unknown |
| --- | ---: | ---: |
| Nominal map plus tactile | 24/30 (80.0%) | 0 |
| Wrist RGB-D plus tactile | 30/30 (100%) | 0 |
| Wrist RGB-D, feedback disabled | 20/30 (66.7%) | 0 |

All 90 trials are collision-free and below the 12 N channel force limit.
RGB-D+tactile docking errors are 16.33-19.34 mm. Its finite-trial 95%
Wilson interval is approximately 88.65-100%; it is not proof of 100%
general accuracy. Known colour, radius, height and upright priors constrain
perception. The ablation disables the feedback bundle (retention, shared
relief/inhibition/stops and tactile phase guards), so it does not isolate one
tactile mechanism. Shared object-state supervisory lift/dock guards remain
in every condition; body poses do not localize the camera-guided grasp. The
retrieval receipt verifies all raw logs and metric-depth exports.

The earlier seeds 500-529 remain a separate 65 mm / 0.5 s development
comparison. Do not pool their denominators with this precise evaluation.

## Continuous Sorting And Three-Tool Feedback

Continuous scene seeds 30-34 completed 5/5 three-tool sequences, or 15/15
transfers, with no scene reset between marker, screwdriver and wrench.
Already placed tools remain docked at the end. The synchronized seed-0 video
is a separate demonstration and is excluded from that denominator. These are
historical results from the earlier release instrumentation.

The earlier stricter native-release protocol at seeds 80-84 completes 4/5 sequences
and 14/15 transfers. At seed 82, the thumb base contacts the previously docked
screwdriver during wrench docking: the native channel reaches 12.262 N,
exceeding the unchanged 12 N gate. This episode remains a failure even though
the task state machine completes.

`results/v7_fixture_motion_relief_summary_20261014` preserves the diagnosis:
an initial zero-speed pressure pause avoids the force overshoot but causes
`DOCK_TIMEOUT`. Taring the initial held-tool pressure, maintaining a slow
contact crawl and retaining finger relief then passes development seed 82,
with DOCK peak 8.944 N and all three tools still docked. Fresh-seed validation
at 90-94 and 1000-1029 is reported above, separately from this development
replay. Shallow non-target contact still
occurs; collision-free means the declared penetration/force gate passes,
not that every surface has zero physical contact.

`results/v7_three_tool_feedback_paired40_44_20261014` records 30 independently
initialized fixture episodes: three tools, five same-seed pairs,
feedback enabled/disabled. Feedback completes 15/15 versus 10/15 without
feedback; marker and screwdriver each complete 5/5 in both conditions, while
wrench completes 5/5 versus 0/5 (`LIFT_TIMEOUT`). This small fixture
experiment supports wrench retention under its declared conditions, not a
universal benefit for every object. Unknown outcomes are zero; all trials
satisfy the collision and 12 N safety gate. Each accepted result independently
passes a full three-second release window and 25 mm dock check.

The repaired receiving-force controller is documented in
`results/v7_fixture_force_relief_20261014`: 20/20 task and force-safe
episodes, with temporary 8 N dock/release relief and the unchanged 12 N
acceptance limit.

## Earlier Marker Transfer Development

The calibrated marker protocol uses a top-down approach, wrist offset
`(-0.065, 0.037, 0.233) m`, closure `0.65`, a two-second close-settle period,
100 mm lift, Cartesian interpolation, tactile retention and explicit release
verification. Objects remain free bodies throughout the episode.

`results/v7_marker_heldout100_109_shared_20261014/summary.json` records ten
complete episodes at new seeds 100-109: executor and independent evaluator
both accepted **10/10**, with no unknown outcomes. Peak aggregate hand normal
force was 8.30-8.44 N; final centre-to-slot error was 16.8-20.0 mm. This is a
small development check with only +/-1.5 mm seeded placement variation. It is
not a multi-tool success rate, a large uncertainty benchmark, or evidence of
learning-policy training. This marker-only revision predates the later
frozen evaluations reported above.

`results/v7_marker_live_video_seed100_20261014` contains one synchronized
physical marker episode accepted by both evaluators. Its 242-frame MP4
shows overview, hand, wrist RGB-D and tactile channels from the same
post-step states. It is a demonstration and is not added to the ten-episode
denominator. The workbench replay retains `task_success: null` until evaluated.

The frozen marker-revision paired experiment is in
`results/v7_marker_robustness_final2_20261014/`. It perturbs actual initial
marker XY position by up to 5 mm and mass/friction by up to 20%, while the
controller retains the nominal map and all three distractors remain present.
Both modes have complete denominators and no unknown outcomes: shared-control
is 23/30 full success (76.7%) and 27/30 target-held lift (90.0%); the
manual-with-retention comparison is 22/30 full success (73.3%) and 27/30
target-held lift (90.0%). Both modes are 30/30 collision-free and force-safe.
Failures are `CLOSE_TIMEOUT` (3 shared, 3 manual) and `VERIFY_TIMEOUT` (4
shared, 5 manual). This is a fixed primitive-marker robustness protocol, not
a learned-policy success rate. The final run reports `source_changed=false`.
Current `manual` mode disables shared safety interventions but still uses
tactile retention, so this comparison is not a complete no-tactile ablation.

## Wrist RGB-D Correction Probe

`results/v7_rgbd_probe_20261014/summary.json` contains ten construction-time
perturbation probes. A wrist RGB-D frame is segmented using the declared
marker colour and known cylinder radius; the controller receives the fitted
XY map, while the physical body pose is used only for post-run error. The fit
error is 0.051-0.057 mm in these synthetic observations and all ten probes
completed with 16.3-19.0 mm docking error. This is a known-object detector
under an upright staging prior, not a general RGB-D perception result.

The reproducible entry point is `scripts/run_v7_rgbd_marker.py`. Its output
records the policy input boundary and saves the RGB-D images.

## Upright Tool Fixture

The original flat screwdriver/wrench poses remain a negative baseline. A
separate declared receiving fixture profile is implemented by
`src/kuka_sharpa_grasp/v7_fixture.py` and
`scripts/run_v7_fixture_task.py`. It preserves free joints and uses explicit
source guides, object-specific receiving-guide heights, physical rack slot
centres, tactile retention and a 3-second post-release verification window.
Earlier frozen fixture runs accepted screwdriver seeds 0-3 (4/4, dock error
14.65-14.74 mm) and wrench seeds 0-4 (5/5, 13.83-15.47 mm), with zero
collision events. These are fixture-supported tool-sorting results and do not
replace the flat-layout denominator.

## Public YCB Extension And Registration Development

These are separate V6-based single-object trials on the public tomato can,
potted-meat can and cracker-box meshes, with known identities, authored
appearance and upright staging. They do not use the V7 six-object sorting
denominator or its strict release/collision criteria. Each trial permits
at most two physical attempts without resetting between attempts.

| Frozen batch | Complete task | Held lift | Additional <=12 N channel reference |
| --- | ---: | ---: | ---: |
| `public_ycb_current_random18_20261014`, seeds 3200-3217 | 3/18 | 3/18 | Not recorded in this receipt |
| `public_ycb_improved_random18_20261014`, seeds 3400-3417 | 6/18 | 14/18 | 0/18 |

Both batches preserve all 18 trials, have no unknown outcomes or runner
errors, and have independent
[original](results/public_ycb_current_random18_20261014/verification_receipt.json)
and [improved](results/public_ycb_improved_random18_20261014/verification_receipt.json)
native-log verification receipts. The batches
use different fresh seeds; their rates are not a paired estimate of the
controller effect. In the improved batch, independent native maximum
hand-channel forces range from 13.616 to 132.622 N. None of its six
completed tasks passes the additional 12 N reference. A 12 N finger-relief
setpoint in the V6 controller is not a whole-task force limit; approach and
placement contacts can still produce excessive loads. These are retained
negative safety results, not force-safe V7 successes.

The separate registration development captures nine wrist RGB-D clouds
(three known meshes at seeds 0, 3402 and 3406). Replaying those same saved
clouds with a bounded 3-degree roll/pitch fit reduces XY origin error from
0.974-1.059 to 0.057-0.111 mm for the tomato can, 1.745-1.869 to
0.036-0.073 mm for the meat can, and 4.657-4.785 to 0.225-0.399 mm for
the cracker box. The estimator uses visible metric-depth points and static
CAD geometry, with known identity/appearance, near-upright orientation and
height priors. Live body poses and simulator segmentation are not estimator
inputs; body poses are used only for post-fit error measurement. The
[nine-cloud replay](results/public_ycb_registration_small_tilt_replay_20261014/summary.json)
is a reused development dataset, not nine new grasp trials, unseen-object
accuracy, unconstrained 6-DoF localization or hardware validation.

A subsequent bounded-tilt closed-loop development check at the reused
can-3414, meat-3402 and box-3406 cases, each in two transfer profiles,
records 2/6 complete tasks and 6/6 held lifts in
`public_ycb_tilt_registered_controller_development_20261014`. Meat-can
placement loses contact, and the alternate can/box transfer profiles fail
placement or perception. Its two successful tasks do not establish the
V7 force gate. Localization and force-safe execution remain distinct
development problems; these six cases are excluded from both fresh
18-trial denominators.

## Earlier Baseline Before Controller Corrections

- MuJoCo classic CPU dynamics, physical-v6 actuator/contact profile
- seed `0`, all three actionable objects, `phase_timeout=5 s`, tactile hold
  `0.08 s`
- one shared-control run and one manual-control run
- a second shared-control run at seed `7` for the distractor-heavy layout
- object free joints are advanced only by MuJoCo; the task runner does not
  write object qpos during an episode

## Earlier Baseline Results

The six complete seed-0 episodes have `success=0/6` and `unknown=0`. The
batch evaluator therefore reports `success_rate=0.0` with denominator `6`.
The recorded failures are:

| mode | object | failure |
| --- | --- | --- |
| shared | screwdriver | `CLOSE_TIMEOUT` |
| shared | wrench | `ALIGN_SAFETY_STOP` |
| shared | marker | `GRASP_CONTACT_NOT_PERSISTENT` |
| manual | screwdriver | `CLOSE_TIMEOUT` |
| manual | wrench | `ALIGN_TIMEOUT` |
| manual | marker | `GRASP_CONTACT_NOT_PERSISTENT` |

The seed-7 clutter run also completed all three episodes and retained the same
failure classes. No contact impulse is labelled as a lift or dock success.

## Earlier Calibration And Corrected Scoring

`results/v7_calibration_marker_20261014/calibration.json` contains a seeded
12-candidate sweep over grasp offset, wrist rotation, approach direction and
closure. Its best candidate produced no measured lift and failed with
`ALIGN_TIMEOUT`; the maximum-height score is retained for the next controller
iteration. The sweep uses the same task executor and has the same qpos-write
policy as the protocol.

The initial maximum-height score was unsuitable: it could reward an impact
or a tipped object. The revised calibration requires phase-specific,
sustained target-hand contact, a support-free lift, multiple contacting
fingers, and absence of collision/overload. Full lift-and-dock success is
reported separately. The 128-candidate neighbourhood search found 22 strict
lift candidates; 22/128 is search yield, not the success rate of a policy.
The saved sweep explicitly records source changes during development.

## Dell Windows Reproduction

The [final Dell receipt](results/dell_v7_final_20261014_r3c/v7_windows_verify_20261014_r3c/final_dell_verification.json)
passes all eight checks in desktop session 1 using the actual
`NVIDIA GeForce RTX 2060/PCIe/SSE2` native rendering context. The frozen
340-file deployment is unchanged after execution and the shared runtime
source hashes match the current controller. Seed 0 completes a continuous
three-tool sequence without resets, with 40,163 native samples and zero
unknown or unattempted transfers.

| Tool | Strict Accepted | Maximum Native Channel (N) | Dock Error (mm) | Stable Release (s) |
| --- | --- | ---: | ---: | ---: |
| Marker | yes | 4.306 | 17.598 | 3.000 |
| Screwdriver | yes | 6.505 | 14.582 | 3.000 |
| Wrench | yes | 9.427 | 14.424 | 3.002 |

This reproduction is excluded from the fresh-seed success denominators.
The automated GUI logs contain 21 input events with required coverage;
GUI/replay each retain 1,500 physics frames and pass replay equivalence.
`human_operator` is false and GUI `task_success` remains null.

Windows software checks retain their full history: the initial immutable
suite records 279 passed, 1 skipped and 12 failed. Eleven failures arose
from the wrapper's Windows temporary-path parsing; corrected-path recovery
records 11 passed and 1 optional OMPL integration case deselected. OMPL is
absent from this native Windows base environment. This is device-specific
reproduction with targeted recovery, not a repeated 292-pass Windows suite.
The checkout is `C:\Users\13514\Desktop\kuka_sharpa_v7_20261014_r3`; its desktop
entries are `EE5062 V7 Teleoperation 20261014_r3.lnk` and
`EE5062 V7 Tool Sorting 20261014_r3.lnk`.

## Remaining Acceptance Work

The original flat screwdriver/wrench layout has not completed the full task.
Upright tool-guide experiments are a separate fixture-supported sorting
protocol and cannot replace those failures in the original denominator. A
separate obstacle gate now provides task-level planning/geometry evidence:
the direct route is blocked (150/301 collision states), while OMPL
RRTConnect validates an 835-state held-tool route with zero collision states
and 6.31 mm minimum clearance. This standalone geometry gate does not score
physical task execution. The later frozen carried-tool experiment does
complete physical pick, transfer and dock in three scenes, as recorded in
`V7_OBSTACLE_EXECUTION.md`; its source revision is separate from the latest
stop/contact-motion repair. A new frozen-release 4090 run passes all nine
[release checks](results/lab4090_v7_final_source_verify_r3_20261014/final_validation_receipt.json):
292 software tests, actual RTX 4090 EGL rendering, and 3/3 physical carried
reruns at seeds 0-2 with independent remote native-log recomputation. Seed 0
is the development case and seeds 1/2 are validation cases; these are not
three newly sampled scenes. The new dock errors are 14.448/14.521/14.435 mm,
and each independently measured native release window is 3.000 s.

The 292-test result belongs to the frozen 336-file release archive (SHA256
`3e54242324e8b38dae4e5e3b30aee0c7f10e06844b0844ea4d535be2bd33e648`).
Subsequent isolated public-object prototypes and Windows-launcher edits
are checked separately; the receipt is not a repeated whole-suite check of
the eventual final package. Complete local raw-log retrieval and
[verification](results/lab4090_v7_final_source_verify_r3_20261014/local_transfer_verification.json)
also pass: all 49 inventory files match size/SHA256, and all six raw logs
were fully parsed with the three independent native audits accepted.
The Dell desktop reproduction above is accepted; final package and student
contribution review remain before submission. The frozen feedback comparisons above retain
their original denominators and source versions.
A human-operated teleoperation recording still needs its own evidence; the
GUI and replay functional check does not establish human operation.

Body-origin height alone cannot distinguish a tipped tool from contact
penetration; both object position and quaternion remain in the trace. The
declared contact baseline is retained until a controlled numerical check
supports a change.

## Reproduction

See `V7_RUNBOOK.md` for the exact runner, batch evaluator and calibration
commands. The episode directories contain `task_results.json`, `events.json`
and `metadata.json`; the batch directory contains the JSON and CSV summaries.
The current continuous native batch and its existing-artifact verification
use the retained standalone orchestrator:

```bash
PYTHONPATH=src python scratch/v7_physics_sensitivity/crossvalidate_continuous.py \
  --output results/my_continuous30 --start-seed 1000 --count 30
PYTHONPATH=src python scratch/v7_physics_sensitivity/crossvalidate_continuous.py \
  --output results/my_continuous30 --start-seed 1000 --count 30 --verify-existing
```

Use a new output directory for a new physics run. The archived batch contains
its frozen orchestrator and source tree; its `trace.json.gz`,
`action_log.json.gz` and `events.json.gz` are retained alongside the receipts.
