# A: Public YCB Randomized Baseline And Evaluation

Primary official-baseline work. See `docs/ASSIGNMENT_AND_TEAM_PLAN_ZH.md`, section 3.

- Independently install the exact inputs and record environment/commit/commands. Keep any replay mismatch.
- With the lead, replace fixed replay assumptions with a fresh object-selection, perception, grasp-pose and approach interface. Lead owns controller integration; A owns schedules, evaluation and failure reproduction.
- Plan 3 known YCB objects x 5 randomized poses for development (15 episodes), followed by a frozen 3 x 10 independent-seed evaluation (30 episodes), only after the fresh interface works.
- Preserve every failure/runtime error and the planned denominator. Do not retune on evaluation seeds.
- Deliver protocol JSON, per-episode CSV, traces/contact logs, videos, failure categories and a results table via `workA/public-ycb-benchmark` PR.

Acceptance: a fresh process executes random selection/placement, generated grasp, physical task and evaluation. Merely changing the seed of R29's recorded prefix does not satisfy this requirement.
Assignee pending GitHub username; no invitation has been sent.
