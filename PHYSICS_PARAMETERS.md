# Physical Parameter Profiles

This project uses rigid-body MuJoCo dynamics. `physical-v6` is a new experimental
profile with traceable object masses and arm actuation limits. `legacy-v5` remains
the default of the existing scene builder so archived studies and existing
entry points retain their original physical settings. V6 must be selected
explicitly; old V5 success rates do not validate V6.

```python
from kuka_sharpa_grasp.scene import build_scene
scene = build_scene(output_path, object_kind="ycb_can",
                    physics_profile="physical-v6", timestep=0.002)
```

The profile definitions and source URLs are in `src/kuka_sharpa_grasp/physics.py`.
The compiled model records `custom/numeric/physics_profile_version` as 5 or 6.
V6 contact revision 2 is recorded in `physics_contact_revision`; source snapshots
retain the earlier development contact settings rather than silently rewriting results.
Vendor robot XML files are not modified.

## Object Masses and Inertia

| Object | Legacy V5 mass | Physical V6 mass | Mass provenance |
| --- | ---: | ---: | --- |
| Synthetic box | 0.150 kg | 0.150 kg | Engineering test-object assumption |
| YCB 005 tomato soup can | 0.150 kg | 0.349 kg | YCB Table II, item 5 |
| YCB 010 potted meat can | 0.150 kg | 0.370 kg | YCB Table II, item 10 |
| YCB 003 cracker box | 0.150 kg | 0.411 kg | YCB Table II, item 3 |

The three YCB values are reported specimen masses from Calli et al.,
*Benchmarking in Manipulation Research: The YCB Object and Model Set and
Benchmarking Protocols*, Table II, PDF page 7:
[author-hosted paper](https://people.eecs.berkeley.edu/~pabbeel/papers/2015-RAM-YCB-object-set.pdf).
They are reference values for the published object set, not measurements of a
physical specimen owned by this project. Dataset/product versions can differ.

MuJoCo computes each target's center of mass and inertia from its mesh or box
under a uniform-density rigid-body approximation. This does not reproduce the
internal distribution of soup, meat, packaging or loose cracker contents.
Current collision meshes retain MuJoCo's convex collision approximation;
visual mesh detail does not imply concave collision fidelity.

`GraspRollout.configure_object(mass=..., friction=...)` changes density at fixed
shape and center of mass. It scales all three principal inertia values by
`new_mass / reference_mass`, rather than changing mass alone. Every change is
computed from the captured reference, so repeated sweeps do not accumulate
scaling error. `configure_object()` restores mass, inertia, geom friction and
V6 explicit contact-pair friction. `reset()` resets state and retains the selected
physical parameters, allowing a configured randomized trial to start correctly.
Invalid or nonfinite mass/friction values are rejected before model mutation.
This fixes a historical mass-sweep inconsistency in both profiles; old frozen
source snapshots remain the authority for reproducing old variable-mass runs.

## Contact Parameters

All friction values below are engineering assumptions. No local material-pair
calibration has been performed. Sliding friction is dimensionless; torsional and
rolling coefficients have MuJoCo's length units and are listed for completeness.

| V6 geom | Sliding | Torsional | Rolling |
| --- | ---: | ---: | ---: |
| Tomato / potted meat can | 0.60 | 0.005 m | 0.0001 m |
| Cracker box | 0.65 | 0.005 m | 0.0001 m |
| Synthetic box | 0.80 | 0.005 m | 0.0001 m |
| Table | 0.60 | 0.005 m | 0.0001 m |
| Hand collision geoms | 1.00 | 0.005 m | 0.0001 m |

V6 creates explicit target-table and target-hand contact pairs. Each pair's
coefficient is the geometric mean of its two geom coefficients, a declared
simulation assumption. This ensures object friction sweeps affect grasping:
MuJoCo's ordinary equal-priority maximum mixing would otherwise hide lower
object friction behind the hand's value of 1.0. Updating the object friction
also updates every associated V6 pair. Legacy V5 uses the original mixing rule.

`condim=3` activates normal contact and two sliding directions. Torsional and
rolling friction are not active in these target pairs. This explicit choice
keeps the contact model inspectable; it is not a claim of complete contact physics.

| Pair | `solref` (time constant, damping ratio) | `solimp` |
| --- | --- | --- |
| Target-table | `0.02 1.0` | `0.9 0.95 0.001 0.5 2` |
| Target-fitted finger capsule | `0.02 1.0` | Same |
| Target-rigid palm collision segment | `0.015 1.0` | Same |
| Target-fingertip pad | `0.04 0.95` | Same |

The explicit pairs preserve MuJoCo's native `solref` and `solimp` mixing from
the compiled geom parameters, including nested XML default inheritance.
Equal-priority positive-format parameters are weighted by `solmix`; a higher
priority selects that geom's parameters, and direct-format `solref` uses the
elementwise minimum at equal priority. The vendor fingertip itself has
`solref="0.06 0.9"`, the rigid palm has `0.01 1.0`, and fitted capsules without
a collision class inherit `0.02 1.0`. With the object's `0.02 1.0`, the default
equal weights give the effective values above. Assigning the raw hand value
directly to each explicit pair changed the original contact compliance;
revision 2 corrects that confound. This preserves compliance while the declared
geometric-mean friction remains an independent engineering assumption.

These soft
contact parameters approximate contact compliance numerically; they are not
Young's modulus, measured silicone deformation, or a validated tactile sensor.

## Robot Limits and Integration

V6 applies joint-level `actuatorfrcrange` limits. The arm controller must route
feedforward/gravity compensation through actuator forces so it shares those
limits. An external `qfrc_applied` torque bypasses an actuator force limit.
Sharpa's existing vendor joint torque ranges are retained.

| KUKA iiwa 14 joint | Maximum actuator torque | Command velocity cap |
| --- | ---: | ---: |
| A1 | 320 Nm | 85 degrees/s |
| A2 | 320 Nm | 85 degrees/s |
| A3 | 176 Nm | 100 degrees/s |
| A4 | 176 Nm | 75 degrees/s |
| A5 | 110 Nm | 130 degrees/s |
| A6 | 40 Nm | 135 degrees/s |
| A7 | 40 Nm | 135 degrees/s |

Source: KUKA's [LBR iiwa technical data](https://www.kuka.com/-/media/kuka-downloads/imported/87f2706ce77c4318877932fb36f6002d/kuka_lbr_iiwa_ja.pdf),
iiwa 14 axis-data table. Velocity is also documented in the
[iiwa 14 R820 product data](https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000246833_en.pdf).
Published peak torque is not a continuous thermal operating limit.

MuJoCo has no native hard joint velocity limit. V6 stores these velocity caps in
`jnt_user[:, 0]` and a conservative 0.8 rad/s^2 command acceleration cap in
`jnt_user[:, 1]`. The acceleration value is an engineering assumption. The
controller must rate-limit motion commands and record actual joint velocities;
the metadata alone does not constrain motion. Do not clip simulated `qvel` to
manufacture compliance. Contact impulses can still produce actual excursions.

Both profiles use gravity `(0, 0, -9.81) m/s^2`, `implicitfast` integration,
Newton solver, elliptic friction cone, 100 solver iterations,
and default timestep 0.002 s. Legacy V5 retains `impratio=10` and tolerance
`1e-8`; physical V6 explicitly uses `impratio=100` and tolerance `1e-10`.
MuJoCo's [contact-slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)
recommends an elliptic cone with increased `impratio` and a sufficiently accurate
Newton solve when soft contact constraints creep. These are numerical solver
choices, not increased material friction or a calibrated material parameter.
The 100 value is an engineering selection; larger values and the ad-hoc no-slip
postsolver were development diagnostics and are not enabled by this profile.
Passing static/numerical checks does not establish cross-object grasp accuracy.

`timestep=0.001` supports a convergence comparison.
The scene builder accepts steps in `(0, 0.01]` seconds; larger admissible steps
are not asserted to be accurate for grasp contacts.

MuJoCo's [joint force-limit documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-actuatorfrcrange)
and [contact parameter documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
define the numerical meaning of these settings.

## Validation Boundaries

`tests/test_physics.py` checks compiled masses, positive physically consistent
principal inertia, density scaling/restoration, effective friction updates,
actual actuator torque saturation in both directions, profile compatibility,
and explicit timestep validation. These are software and model-contract checks.

Physical acceptance should additionally include analytic free-fall, sliding
friction and static support comparisons, plus timestep/contact sensitivity.
Passing those checks establishes internal numerical plausibility. It does not
establish hardware calibration, silicone deformation fidelity, sim-to-real
transfer, or a new grasp success rate.
