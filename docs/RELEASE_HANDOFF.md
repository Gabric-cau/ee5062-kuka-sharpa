# Team Handoff, 30 October 2026

This public repository contains code, robot/object assets, focused tests, a current report and a team plan.
The `handoff-20261030` Release contains:

| Asset | Purpose |
| --- | --- |
| `r29-runtime-inputs.zip` | Five exact replay/prior inputs omitted from the older source ZIP. Install with `scripts/prepare_r29_inputs.py`. |
| `r29-recorded-evidence.zip` | Full original Linux R29 native/controller streams, model, source snapshot, video, plots and independent receipts, retaining original project-relative paths. |
| `mac-reproduction-failure.zip` | Full fresh Mac reproduction attempt, stopped at 7.198 s by the unchanged joint-velocity replay gate. |
| `PROJECT_REPORT.pdf` | Updated report, correcting the official/optional requirement distinction and documenting the fresh Mac failure. |
| `SHA256SUMS.txt` | Checksums for the attached ZIP/PDF assets. |

See [the team plan](ASSIGNMENT_AND_TEAM_PLAN_ZH.md) and [setup instructions](GETTING_STARTED.md).
Large artifacts are separate from Git history. Source tags and saved artifacts are evidence of their recorded scope, not proof of randomized task reliability.
The original frozen files contain historical filesystem paths, preserved byte-for-byte for audit. The older pre-submission v4 ZIP is retained locally and excluded from the public Release because its nested source archive contains laboratory connection instructions. Use this repository and the runtime-input asset for reproduction.
This Release is a team handoff, not a Canvas submission or a claim that all coursework requirements are complete.
