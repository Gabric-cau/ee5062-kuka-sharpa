# Physical model and contact-control evidence

Engineering evidence addendum, 21 October 2026. This is a draft section for the
project report, not a completed submission or a hardware-validation claim.

## Actual compiled model

The exporter reads saved experiment scenes through MuJoCo's compiler. The three
JSON records in `results/compiled_physics_20261021` contain the actual target
mass, principal inertia and inertial frame, collision geometries, explicit
contact pairs, hinge constraints and actuator gains/limits. They retain the
scene, exporter and physics-source hashes and separate values from assumptions.

| Target | Mass (kg) | Principal inertia (kg m^2) |
| --- | ---: | --- |
| YCB can | 0.349 | 0.000359131, 0.000357194, 0.000204549 |
| YCB potted meat can | 0.370 | 0.000457402, 0.000348420, 0.000268273 |
| YCB cracker box | 0.411 | 0.002614136, 0.001887854, 0.001000984 |

These inertias are compiled uniform-density approximations, not measured object
inertias. Body-relative inertial positions and quaternions are in each JSON.

The `physical-v6` model uses gravity `(0, 0, -9.81)` m/s^2, a 0.002 s step,
`implicitfast`, Newton, 100 solver iterations, tolerance `1e-10`, and `impratio`
100. Sliding-friction coefficients are 0.6 for the two cans, 0.65 for the box,
and 1.0 for the hand. Explicit target pairs use the geometric mean, giving
approximately 0.774597 and 0.806226 respectively. This mixing rule and the
material coefficients are engineering assumptions.

Hand rigid and compliant-pad `solref` values are `(0.01, 1)` and `(0.06, 0.9)`.
The exported resolved pair parameters are authoritative where mixing occurs.
Detailed visual meshes, hand collision proxies, and convex target collision
geometry are distinct. A visually convincing mesh is not proof of accurate
contact mechanics. Soft joint constraints can be crossed temporarily, even
when position commands remain in their declared ranges.

Arm torque caps are `(320, 320, 176, 176, 110, 40, 40)` Nm. Command velocity caps
are `(85, 85, 100, 75, 130, 135, 135)` degrees/s and acceleration caps are
0.8 rad/s^2. The acceleration cap is an engineering command constraint.
The actual per-actuator and per-joint compiled values are retained in JSON.

## Sensor and control flow

1. A simulated wrist RGB-D camera supplies a visible surface cloud. Registration
   to a known public mesh estimates object pose; known identity, appearance,
   near-upright and catalogue priors limit generalization.
2. Native external hand contacts supply normal pressure, shear, position and
   contact normal. These are simulated tactile measurements, not calibrated
   hardware sensors. The controller does not identify external target geoms for
   tactile association; it associates contacts against the estimated surface.
3. A friction-cone linear program allocates normal-pressure goals using estimated
   geometry and catalogue mass/COM. Its default 0.8 friction prior with 0.85
   safety factor is an allocator assumption, distinct from the actual pair.
4. Position-reference increments track those pressure goals within the existing
   servo/rate/torque limits. R6 safety overrides ordinary tracking on overload.
5. The R8/R9 experimental 1g-capacity gate permits only a bounded 5 mm probe.
   Confirmation additionally requires measured-load compatibility, fresh vision,
   maintained contacts and actual virtual-table unloading. The target-specific
   table load sensor and supervisory geometry checks are declared privileged
   simulation instruments. Probe confirmation is not full task success.

The LP's unmeasured tangential forces remain decision variables. Feasible
capacity and fixed-normal feasibility do not measure actual six-axis equilibrium.
The independent task evaluation still checks placement within 25 mm, at least
3 s released stability, contact peaks within 12 N, and physical held-lift evidence.
All retries, recovery and native 2 ms force spikes stay in the evaluation.

## Current result and reproduction

R9 remains a failed development episode: no confirmed lift, no full task success,
and a whole-episode maximum of 4.834046 N. Its middle-finger contact disappeared
on both attempts. Lower force peaks alone do not establish successful grasping.
See `results/r9_independent_native_audit_20261021/DIAGNOSIS.md`.

Subsequent actual R11 tracking sustained zero table load for 1.064 s, with
1.856346 mm evaluated object ascent and a 5.920831 N whole-episode peak. It
still failed the 5 mm endpoint and fixed-normal force-reserve criteria. R12's
bounded wrist-reference feedback regressed to contact loss after 0.574 s;
its whole-episode peak was 6.000560 N. Both full tasks remain failures. Neither
partial transfer nor a static feasible allocation is a successful grasp.
The complete current results and provenance are in `ACTIVE_DEVELOPMENT.md`.

The read-only [contact-distribution comparison](results/contact_distribution_20261021_r11_r12/contact_distribution.png)
shows first-probe native channel pressure, middle MP/DP load distribution,
patch friction demand, and commanded versus actual middle-joint motion.
Friction demand is normalized by the unchanged conservative LP coefficient;
values above one describe individual recorded patches exceeding that assumed
margin, not a task verdict or a different actual contact coefficient. The
plot retains native 2 ms samples and verifies raw input hashes before loading.
Joint traces show change from admission; initial servo offsets are not zeroed
in the underlying records. This analysis does not advance the simulation.

Offline R13 target design finds current conservative targets for all 76 sampled
R11 probe inputs. Substituting the hypothetical target patch normals into the
original 1g LP passes for all 76 samples in each of three target designs. These
are counterfactual compatibility checks, not measured force attainment. A local
middle MP/DP Jacobian has rank two for independent MCP/PIP/DIP motion, while
the existing scalar closure cannot independently press MP and relieve DP.
See `results/r13_force_target_design_20261021/PROPOSAL.md`.

R13 subsequently ran its frozen independent-joint controller on the RTX4090.
Both independent verifiers accepted the artifact consistency checks, including
23,285 contiguous 2 ms native samples and no external target-position writes.
Task success and held lift are false. The first attempt did enter the probe at
9.568 s, then stopped after 0.502 s with current `PROBE_CAPACITY_1G_INFEASIBLE`.
The second attempt exhausted its 8 s CLOSE without the required 0.25 s stable
window. The final `PROBE_NOT_ELIGIBLE` label does not mean no first probe occurred.
The maximum whole-episode external channel pressure was 4.194977 N.

Independent joint motion increased proximal middle loading while reducing tip
loading. However, at the last first-probe write, thumb pressure was 3.303585 N
against a hypothetical target of 5.459637 N. The controller had not attained its
static target. Second-attempt post-limit direction rollback also left the middle
PIP added bias outside its declared 0.02 rad envelope for 406 writes, reaching
0.020291 rad in magnitude. Thus neither pressure safety alone nor the source
tests prove every intended command constraint. R13 remains a failed development
controller and is not deployed to Dell as an upgrade.

The [three-controller contact comparison](results/contact_distribution_20261021_r11_r12_r13/contact_distribution.png)
and [R13 probe plot](results/probe_diagnosis_20261021_r13/probe_diagnosis.png)
use the completed native records. R14 is an isolated proposal to solve joint
constraints together and require preload convergence before probe admission;
the measured R14 outcome is recorded below. The engineering report
appendix is [EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md](docs/EE5062_CONTACT_CONTROL_ADDENDUM_20261021.md).

R14 then ran the isolated direction-constrained joint-box solver and stricter
preload gate on the same can case. Both CLOSE phases exhausted their declared
16 s deadline; current measured fixed-normal 1g feasibility never became true,
so no probe was admitted. The run ended `PROBE_NOT_ELIGIBLE` with 31,950 native
2 ms samples, a 4.767377 N external-channel peak, zero external target qpos
writes, and false task success/held lift. Of 6,530 accepted active QP writes,
the independent native command audit reconstructed 52,240 next-step actuator
commands and found no range, rate, or added-bias violation (maximum bias
0.020000000000000073 rad, within floating-point tolerance). 194 infeasible QP
solves were explicitly held as `HOLD_INVALID`; they are excluded from the
accepted-command claim and remain evidence that the moving bias box can become
infeasible. The longer CLOSE did not create a false success criterion.

R14 source/test/design hashes are recorded in the active status document. Its
full raw artifacts, preload/QP telemetry, strict receipt, and
`native_command_bounds.json` are under
`results/public_ycb_constrained_preload_20261021_r14`. This is an actual
4090-rendered MuJoCo run, but it remains a negative development result.

The following re-exports a saved scene to a new path; it never advances physics:

```sh
PYTHONPATH=src .venv/bin/python scripts/export_public_ycb_physics.py \
  results/public_ycb_current_contact_20261021_r9/ycb_can_current_contact_probe_r9_seed3414/scene.xml \
  --output results/compiled_physics_repeat/ycb_can.json
```

Use the checkout's configured Python environment if `.venv` is absent locally.
Native MuJoCo integration runs on CPU. The actual R9 OpenGL context reported
an NVIDIA GeForce RTX 4090; rendering capability does not correct a contact
posture or force-tracking failure. No learned policy or neural training is
claimed. Codex assisted code and analysis; student contributions and conclusions
must be reviewed and stated by the team before submitting a report.
