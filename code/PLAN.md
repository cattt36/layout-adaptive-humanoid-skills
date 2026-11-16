# HNU pipeline v2 — plan (started )

Goal: ThreeStepPoleRun, FiveStepPoleRun, CrouchedTraverse on G1 → three expert tracking teachers → one MoE teacher
→ five latent-conditioned students (Z0–Z4) covering all three actions.

Remote root: `/root/gpufree-data/hnu_v2/{code,data,runs}`. Code source of truth: this folder, synced by scp.
Simulator: mjlab 1.6.0 (MuJoCo-Warp, BeyondMimic tracking task, G1 29-DoF MJCF). All kinematics/dynamics
in stage 1 use the **same mjlab G1 MJCF** that is used for training.

## Stage 1 — physically consistent G1 references (`stage1_reference/`)
Input: existing `Retargeted/Data/<task>/*_Dynamics.npz` (geometry already scaled to G1; feet within ~5 cm of
scaled human toe targets), force-plate phase files, obstacle CSV.
1. Contacts: plate phases (ground truth) + kinematic detection outside the plates.
2. Contact-locked IK: stance foot flat on ground at fixed xy/yaw; swing feet never below ground; joint limits
   with margin; smooth corrections.
3. Flight phases: COM made exactly ballistic (g = 9.81).
4. Contact-aware inverse dynamics: sole-point contact forces in friction pyramids solved by QP; joint torques
   include GRF; residual base wrench reported as infeasibility.
5. Outputs: new `*_Dynamics.npz` (old keys, recomputed) + mjlab motion npz (50 Hz) + per-trial report.
Gate: no stance foot slip >1 cm, penetration <5 mm, joint limits respected, report torque ratios.

## Stage 2 — expert teachers (mjlab tracking, multi-clip, adaptive sampling, DR)
One expert per task over all its accepted trials. CrouchedTraverse adds the overhead beam; ThreeStepPoleRun/FiveStepPoleRun adds
foothold markers (flat ground as captured). Eval: success rate over all clips × N seeds with pushes.

## Stage 3 — MoE teacher
Three experts behind a task-id router (hard gating). Rollouts with perturbations → execution labels
(success, real contact forces/torques, failure time) for z_quality.

## Stage 4 — annotations + latents
Regenerate robot JSON metrics/semantics from stage-1 data and stage-3 rollouts; human parts kept.
Fix quality-branch leakage (no inverse-dynamics torque in quality inputs). Train Z0–Z4 (same splits).

## Stage 5 — students
Five students (Z0–Z4), deployable observations + z_skill, DAgger from the MoE teacher. Sim eval per task.

## Stage 6 — cleanup
List superseded files; delete only after the user confirms the list.
