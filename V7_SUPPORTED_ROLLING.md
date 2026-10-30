# Reproducible Supported Finger Rolling

`scripts/run_v7_supported_rolling.py` is the official entry point for the
table-supported marker rolling experiment. It uses the existing physical
MuJoCo KUKA/Sharpa session and the calibrated marker approach/close. The
wrist lowers by 6 mm, then its commanded arm goal stays fixed for 1.5 s.
The index finger receives MCP_FE = 0.95 rad and PIP = 1.15 rad. A paired
control repeats the identical setup without individual joint overrides.

```bash
PYTHONPATH=src python scripts/run_v7_supported_rolling.py \
  --output results/my_supported_rolling --start-seed 0 --count 5

# Synchronized overview, hand detail, wrist RGB/depth and native tactile views.
PYTHONPATH=src python scripts/run_v7_supported_rolling.py \
  --output results/my_supported_rolling_video --count 1 --video --fps 20
```

Both output paths must be new. Exit code 0 requires all push episodes to
pass the existing evaluator and the 12 N per-channel force limit, all
controls to fail the rolling-motion criterion, and stable source hashes.

## Evidence

The current 4090 reproduction is in
[`results/v7_supported_rolling_current_20261022/summary.json`](results/v7_supported_rolling_current_20261022/summary.json).
Seeds 0-4 completed both conditions with unchanged source hashes: index push
passed 5/5 and no-override control passed 0/5. Active relative rotation was
48.119-48.622 degrees, measured wrist translation 0.680-0.766 mm, and maximum
marker drop below 0.096 mm. Both conditions retained continuous hand/table
support and passed the 12 N channel-force limit. Each episode includes its
own synchronized MP4, full physical trace and native contact stream. These
are current reruns of development seeds, not a new held-out cohort.

The following historical figures remain separate. Final-source local results are in
`results/v7_supported_rolling_final_20261014/summary.json`. All five seeds
(0-4) completed both arms. Index push passed 5/5; no-override control passed
0/5, as expected. These are small declared-layout trials with the scene's
existing +/-1.5 mm seeded placement variation, not generalization across
tool types or a held-out robotics benchmark.

| Metric | Index Push | No-Override Control |
| --- | ---: | ---: |
| Relative marker rotation | 48.06-48.50 deg | 0.50-0.68 deg |
| Maximum wrist translation | 0.666-0.751 mm | 0.405-0.423 mm |
| Finger aperture change | 3.644-3.759 mm | 0.412-0.431 mm |
| Finger contact-count transitions | 2 | 1 |
| Maximum marker drop | 0.0925-0.0953 mm | 0.0993-0.1009 mm |
| Continuous hand and table support | All samples | All samples |

The largest finger-channel force across approach, close, support and rolling
was 8.731 N. The marker retained at least 7.910 N of total hand contact force
during rolling. The arm goal is unchanged during both 1.5 s evaluation
windows; measured wrist motion is reported separately from the goal.
No live object generalized position is written. The native simulator
poses and contact identities are used only for measurement, not commands.

Every episode saves:

- `result.json`: evaluator thresholds, raw metric outcomes, phases, source
  hashes, scene manifest, force peaks and explicit scope.
- `physical_trace.jsonl.gz`: every 2 ms snapshot, joint positions, joint
  override targets/motor commands, tactile values, object/wrist relative
  pose, support and contact-count measurements.
- `native_contacts.jsonl.gz`: every 2 ms native contact list, geometry
  names, positions, local frames, distance and six-component contact wrench.
- `joint_targets.json` and `cartesian_commands.json`: the issued commands.

The historical seed 0 synchronized videos are in
`results/v7_supported_rolling_video_20261014/seed_0/index_push/supported_rolling.mp4`
and the matching `no_override_control` folder. Both decoded as 134 moving,
nonblank 1280x720 frames at 20 FPS. `rolling_start.png` and
`rolling_final.png` provide inspectable frames. The marker has a uniform
primitive surface, so recorded relative rotation is established by the
physical pose/contact trace rather than texture tracking in the video.

## Scope And Test

This is table-supported finger rolling with a single known cylindrical tool
proxy. It is not an in-air regrasp, real tactile hardware measurement,
human teleoperation, or a learned dexterous policy. The evaluator requires
relative motion, limited wrist movement, finger movement, a contact
transition, continuous hand/table contact and less than 5 mm drop.

```bash
PYTHONPATH=src python -m pytest -q tests/test_v7_supported_rolling.py
```

The physical paired regression passed locally. It verifies that the two
arms start identically, preserve the commanded wrist goal, and produce the
expected active/control difference with continuous support.
