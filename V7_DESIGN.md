# V7 Tactile-Assisted Tool Workbench

V7 changes the project question from “can one known object be picked up?” to:

> Can a KUKA iiwa and Sharpa hand retrieve a selected tool from a cluttered
> maintenance bench, use contact feedback to maintain a safe grasp, perform a
> supported finger-level regrasp, and dock the tool in an orientation-specific
> slot under shared teleoperation?

This is a simulation research extension of the V6 physical baseline. It does
not claim real haptic hardware, sim-to-real transfer, or general-object
accuracy. The first dexterity milestone is table/V-block-supported regrasp;
free-space in-hand rotation is a stretch experiment.

## Task contract

Each scene contains multiple actionable objects, not decorative distractors.
The target is selected from an object registry and the remaining objects,
dividers and rack slots remain collision geometry. The task records target
selection, approach, grasp, lift, supported regrasp, obstacle-aware transfer,
docking and release as separate outcomes.

The first supported regrasp target is either a 15 mm axial shift or a 15--30
degree orientation change while wrist motion is bounded. A run is only credited
with finger-level manipulation when the object pose changes relative to the
wrist while the wrist pose remains within its configured tolerance.

## Sensor and control contract

V7 tactile values are generated from hand-local contact geometry: per-finger
normal force, tangential force, contact count, contact centroid and a bounded
slip-risk score. The controller must not receive a target body ID or ground
truth object pose as a tactile input. MuJoCo contact wrenches are used as a
simulation measurement; they are not a calibrated tactile camera.

Teleoperation supplies a replayable wrist command stream and target-selection
events. Shared control may clamp unsafe commands, regulate finger closure,
stop on overload or collision, and propose a local docking correction. A
keyboard, mouse or ordinary gamepad is sufficient; visualized force is not
physical force feedback.

## Evaluation matrix

Freeze the scene seeds and compare the same cases under:

1. scripted baseline without tactile regulation;
2. teleoperation replay with vision only;
3. teleoperation replay with tactile grasp regulation;
4. shared control with tactile regulation and obstacle checking.

Report whole-task completion, target-selection correctness, dropped objects,
non-target collision count and displacement, peak and sustained finger force,
docking position/orientation error, time, operator interventions, and the
relative object-to-wrist pose change during regrasp. The selected successful
video is presentation evidence; it is never a replacement for a batch
denominator.

## Scope gates

The minimum V7 release is a compiling multi-object workbench, target registry,
target-independent tactile trace, deterministic teleoperation replay, and a
supported regrasp contract with tests. A three-tool batch, full collision-aware
transfer and a visual demonstration follow those gates. Unstable free-space
rotation, threaded screwdriving and uncalibrated haptic claims remain optional
extensions.
