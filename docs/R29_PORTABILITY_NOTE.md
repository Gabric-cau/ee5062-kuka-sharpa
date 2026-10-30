# R29 Runtime And Reproduction Boundaries

The recorded laboratory Linux run completed R29. Independent macOS and Linux
audits accepted its saved native states and actual commands. Those audits did
not establish that a new physics rollout would match on every operating system.

## Fresh macOS run, 30 October 2026

A fresh run from this GitHub checkout, after installing the exact runtime
inputs, stopped at native step 3,599 (simulation time 7.198 s) with
`replay_mismatch`. It did not reach the 22-second prefix endpoint or execute
the new R29 continuation. No successful pick/place is claimed for this run.

All 45 frozen controller/source files and all six input hashes match the
successful laboratory experiment. Neither source nor inputs changed during
this run. The one failing comparison was robot joint velocity:

| Replay comparison | Observed maximum error | Existing limit | Outcome |
| --- | ---: | ---: | --- |
| Joint velocity | 0.00001237931 rad/s | 0.000010 rad/s | Reject |
| Joint position | 0.0000000228914 rad | 0.000001 rad | Pass |
| Channel pressure | 0.0000214679 N | 0.01 N | Pass |
| Individual patch pressure | 0.000102471 N | 0.01 N | Pass |
| Patch position | 0.00000000261781 m | 0.00001 m | Pass |
| Patch normal | 0.0000000582705 | 0.0001 | Pass |

Patch count and hand-side identity also match. The failure is a numerical
replay discrepancy before the continuation; it is not evidence that an RTX4090
is required, or that the physical grasp failed after lifting. The absolute
velocity excess is about 0.00000237931 rad/s. The existing gate remains intact;
this one observation does not justify enlarging the replay tolerance.

The fresh runtime is macOS arm64, Python 3.13.15, MuJoCo 3.15.0, NumPy 2.4.6
and SciPy 1.17.1. The successful saved rollout used Linux x86_64, Python 3.10.12,
MuJoCo 3.15.0, NumPy 2.2.6 and SciPy 1.15.3. These observations do not isolate
which operating system, architecture, Python version or numerical build causes
the discrepancy. The `requirements-r29.txt` core pins describe the locally
tested Python 3.13 environment; they do not promise Windows rollout success.

## Team Reproduction Gate

For a Windows PC, the first reproduction attempt should use WSL2 Ubuntu 22.04
with Python 3.10 and the saved Linux core versions above. This is a proposed
environment match, not an already verified WSL2 result. The physics runner does
not render and does not require a CUDA GPU. Use a separate native Windows
environment for the interactive V7 GUI and record that environment separately.

Accept a new R29 reproduction only when:

1. `prepare_r29_inputs.py --check` verifies the five replay/prior inputs and
   public mesh.
2. The full 11,000-step prefix passes the existing comparison gates.
3. The continuation finishes with a recorded physical outcome and complete
   native logs.
4. `audit_coordinated_pick_place_r29.py` and `audit_r29_task_sequence.py`
   independently accept the new output and its task-success label.

Preserve a mismatch or task failure in its own output directory. Passing the
13 input-installer tests or the 11 small-model runner tests establishes their
specific contracts, not this full reproduction gate. A saved Linux video can
be viewed without rerunning the physics, and remains evidence of that original
Linux experiment only.
