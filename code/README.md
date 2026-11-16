# pipeline_v2 — G1 ThreeStepPoleRun / FiveStepPoleRun / CrouchedTraverse

Started . Replaces the old per-action RL patch-work (`RL/`, frozen residual actors, time gates) with a
standard pipeline: physically checked references → motion-tracking experts → MoE teacher → latent-conditioned
students (Z0–Z4 ablation) + executability (z_quality) module.

Remote layout (`/root/gpufree-data/hnu_v2`): `code/` (this folder, synced with `sync_run.ps1`), `data/`, `runs/`.
Simulator: mjlab 1.6.0 (MuJoCo-Warp), G1 29-DoF MJCF shipped with mjlab. Python: `/root/gpufree-data/venvs/mjlab`.

## Stages and entry points

| Stage | Code | Output (remote) |
| --- | --- | --- |
| 1 references | `stage1_reference/build_reference.py` (`run_all.sh`) | `data/ref_v1/<task>/<trial>_{Dynamics,motion}.npz`, `_report.json` |
| 2 experts | `hnu_tasks/`, `train_hnu.py`, `launch_expert.sh` | `runs/logs/rsl_rl/hnu_track_<task>/` |
| 3 expert eval / rollouts | `eval_expert.py`, `run_post_expert.sh` | `runs/expert_eval/<task>/` |
| 4 annotations | `stage4_annotations/regen_robot_metrics.py`, `make_latent_npz.py` | `data/annot_*/{LLM_annotation,Retargeted}/…`, `data/latent_npz_v1` |
| 4 latents | `stage4_latent/` (patched copy of `Latent/`), `train_all.sh`, `export_z_tables.py` | `runs/latent_v*/Z*/`, `data/z_tables/Z*/` |
| 5 students | `hnu_tasks/distill.py`, `distill_train.py`, `eval_student.py` | `runs/students/Z*/` |
| 5 z_quality | `stage5_quality/quality.py` | `runs/quality/Z*/` |

## Stage 1 — what is done to the references (mode `light`)
* Source: existing `Retargeted/Data/<task>/*_Dynamics.npz` (already scaled to G1, ≈0.75–0.84 × human).
* Gaussian low-pass (σ = 2 frames @120 Hz), ground alignment, crop to the task (ready pause → deceleration).
* Contacts: force-plate phases (foot label checked kinematically, clock offset ≤ ±0.15 s estimated) + kinematic
  detection elsewhere; heel/flat/toe roll phases.
* Contact-aware inverse dynamics on the mjlab model: sole-corner forces in friction pyramids (μ = 0.6) solved by QP,
  joint torques include the ground reaction, unexplained base wrench stored as residual.
* The `full` mode (contact-locked IK + ballistic flights) exists but made the references *worse* on every metric
  (see git history of this README / chat log); it is not used.
* Excluded: `CrouchedTraverse_033` (broken source retarget, root speed 71 m/s).

Reference difficulty (median over trials, contact-aware ID vs mjlab effort limits):
ThreeStepPoleRun peak torque ≈ 5× limit, root speed 4–5.5 m/s; FiveStepPoleRun ≈ 2–2.5×, 2.7–3 m/s;
CrouchedTraverse ≈ 2–3×, 2.3 m/s, waist beyond 90 % soft range in 60–85 % of frames (deep crouch).

## Stage 4 — annotation changes
* Human fields untouched. Robot fields recomputed with the stage-1 data:
  support-leg moments are now physical (e.g. FiveStepPoleRun_001 P02 support ankle 0.013 → 2.65 N·m/kg; human 2.21),
  `floating_base_wrench` is now the *residual* (old: the whole body weight, up to 900 N),
  new `ground_reaction` and `execution` (simulated expert rollouts) blocks.
* `robot_semantics` regenerated with deterministic evidence-backed rules (`rules_v1`), same vocabulary and
  feasibility-tag/objective derivation as the original generator; only skill atoms used an LLM before.
* Latent code changes (`stage4_latent/motion_latent_z.py`): inverse-dynamics torque removed from the quality
  encoder input (target leakage); `execution` added to quality *labels*; Z0–Z4 share one set of hyper-parameters
  (old Z4-only tuning behind `HNU_Z4_TUNING=1`); latents are trained jointly on the three actions so that one
  student can consume one z space.

## Stage 2/3 — expert results (10k PPO iterations, 4096 envs, mjlab BeyondMimic defaults)

Every clip played from its first frame, 8 envs per clip; success = reaching the last frame without termination.

| Expert | clips | success clean | success push+obs-noise | body pos err |
| --- | --- | --- | --- | --- |
| ThreeStepPoleRun | 25 | 98.0 % | 97.5 % | 10.8 cm |
| FiveStepPoleRun | 28 | 90.2 % | 85.3 % | 10.5 cm |
| CrouchedTraverse | 31 | 100 % | 98.4 % | 8.3 cm |

Executed actuator effort saturates (ratio 1.0) at every push-off: the reference demands ≈2× the limits (offline ID),
the policy realises the motion at the torque limit.  W1 was resumed from model_6000 on the second GPU (same run
settings; `run_v1`).

## Stage 5 — round v2 (latents without skill-level transfer, student sees z at t and t+0.5 s)

All-clip success (clean env), 8 rollouts per clip from frame 0, 95 % cluster-bootstrap CI:
teacher bank 95.5 [93.2, 97.6]; student + explicit reference 90.9 [87.0, 94.3] (95 % retention);
L0 44.8, L1 35.2, L3 38.4, L4 42.0; no-latent 0.1.  Findings:
* the latent carries the motion (no-latent ≈ 0 %), but two 16-d samples are too sparse for precise tracking;
* with lambda_skill_metric = lambda_skill_atoms = 0 for every level, supervision cannot reach z_skill
  (Z3 ≈ Z4 bit-for-bit up to 1e-3), so level differences are training noise;
* Q_exec (rollout-trained) AUROC 0.84–0.86 vs offline heuristic 0.52–0.54; at 80 % coverage the fall rate drops
  from 9.3–10.8 % to 3.8–4.3 %; isotonic ECE 0.003–0.007.

Round v3 therefore (a) enables the skill-level transfer terms (0.10 / 0.08) for **all** levels and (b) feeds the
student z_skill at t, +0.2, +0.5, +0.8 s (offsets 0/10/25/40 control steps).  v2 is kept as the mechanism ablation.
Two v2 students (Z2, Z4-MoE) hit CUDA OOM in the first pass and were re-run (`tools/fix_z2_moe.sh`).

## Known issues / open items
* DuMuQiao.usd is not registered to the mocap lab frame (ramp on the opposite end, different extent than the CSV
  bridge marks); a clearance check with it reports collisions even for the reference.  Needs the lab-frame bridge
  geometry (post positions, plank underside height) before obstacle-aware training/evaluation of CrouchedTraverse.
* ThreeStepPoleRun/FiveStepPoleRun.usd = 5 stakes Ø0.30 m × 0.10 m at the W2 zig-zag (0.80 m forward / 0.90 m lateral) — already in robot
  scale and consistent with the retargeted W2 steps; no layout for W1 yet.
* ThreeStepPoleRun force-plate phases vs retarget timing do not match well (plate_match ≈ 0.2–0.5); contact labels of
  W1 plate steps are unreliable (tracking rewards do not use them).
* CrouchedTraverse overhead beam is not yet in the simulation scene (clearance is only checked geometrically).
