# Decoupling Rhythm from Skill: Layout-Adaptive Execution of Humanoid Skills from Human Demonstrations

Code, trained models, processed data and the real-robot deployment bundle of the project. A Unitree G1 learns three
obstacle-traversal skills from human motion capture (crouched traverse under a beam, three-step and five-step pole
runs), retrieves a demonstrated skill from a text instruction, re-times it for the current obstacle layout, previews the
candidates in simulation and executes the best one or refuses.

The human data (optical motion capture, force plates, gait events, layouts, retargeted G1 motion) are released
separately: <https://github.com/cattt36/obstacle-traversal-mocap-g1>.

## Contents

| Path | What |
|---|---|
| `code/` | The full research code (simulation is mjlab on MuJoCo-Warp): references, motion-tracking experts, skill encoders, students, planner, router, preview, evaluation, sim2sim and ONNX export. See `code/README.md` for the stage-by-stage notes |
| `deployment/Deployment_mocap/` | The bundle that was used on the real G1: two ONNX policies, the planned layouts with the default skills and the pre-computed candidate plans, the router and skill library, the motion-capture obstacle measurement and the robot health check. Its `README.md` is the field handbook |
| `docs/MANIFEST_models.json` | Every model file of the release: archive path, original run path, role, size |
| `docs/requirements_mjlab_venv.txt` | Exact package versions of the training environment |
| `scripts/download_assets.sh` | Downloads and unpacks the release assets |
| `scripts/setup_layout.py` | Recreates the run layout of the training server from the unpacked assets (symlinks) and copies the code with the path prefix replaced |

The text encoder of the router (`paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0) is not stored here. Put it into
`deployment/Deployment_mocap/text_encoder/` with
`huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir deployment/Deployment_mocap/text_encoder/paraphrase-multilingual-MiniLM-L12-v2`.

## Release assets (models, data, results)

The models and the processed data are too large for git and are attached to the GitHub release `v1.0` as tar parts
(each below 2 GB). Only the final checkpoint of every run is included, not the intermediate ones.

| Asset | Size | Content |
|---|---|---|
| `models_experts_part1.tar` | 1.4 GB | Motion-tracking experts (rsl_rl PPO, one per task) with `params/{env,agent}.yaml` and ONNX. `models/experts/paper_teachers/` = the three teachers of every student in the paper; `other_runs/` = all other expert runs (earlier rounds, fine-tunes and real-robot retraining) |
| `models_skill_encoders_part1.tar` | 1.4 GB | Skill encoders (motion latent, levels Z0–Z4, seeds 42–44) with their latent tables and router heads. `latent_v5t` = fold 1, `latent_cv_f2`–`f4` = cross-validation folds 2–4, `latent_kin_*`, `latent_noq_*`, `latent_rom_*` = human-evidence ablations, folds 1–4 |
| `models_students_part1.tar`, `_part2.tar` | 3.6 GB | All distilled students (`student_<iter>.pt` + `args.json` + `train_log.jsonl`) |
| `models_deploy_exports_part1.tar` | 1.4 GB | ONNX exports with `policy_meta.json` and parity checks; MuJoCo scenes stored once under `_scenes/<md5>.mjb` (the manifest maps each export to its scene) |
| `data_part1.tar`, `data_part2.tar` | 3.6 GB | `data/` of the pipeline: physically checked references, obstacle geometry, CV splits, tempo and cross-trial augmentation, z tables, skill libraries, planner outputs, annotations. Augmented copies are relative symlinks, so unpack all parts into the same directory |
| `results_part1.tar` | 0.15 GB | Evaluation outputs (`runs/results_*`, closed-loop and end-to-end reports, pre-registrations) |

```bash
bash scripts/download_assets.sh /path/to/hnu_v2      # downloads and unpacks into data/, runs/, models/
python scripts/setup_layout.py /path/to/hnu_v2 --prefix-copy /path/to/hnu_v2/code   # run layout + code with your paths
```

Expert runs are stored as `run_<name>` (for example `run_rs10c`). The scripts find a run by the suffix `*_<name>`, so
they work both with the released runs and with new runs, which rsl_rl names `<time stamp>_<name>`.

### Student names

`students_<round>[_f<k>][_d2]_s<seed>/<level>/student_<iter>.pt`

| Part | Meaning |
|---|---|
| `v6g10xm` | rhythm-decoupled students (cross-trial skill augmentation, recipe "mix"), `run_v6g10xm_l40.sh` |
| `v6g10n` | conventionally distilled students (tempo copies only), `run_v6g10n_l40.sh` / `run_v6g10n_l40b.sh` |
| `Z0`–`Z3` | skill-code levels L0–L3 (`Z4` = dropped level L4) |
| `ZR` | explicit-reference student: ER-dec under `v6g10xm` (`run_zr_l40.sh`, `run_zr_l40_s23.sh`), ER-conv under `v6g10n` (`run_zrn_l40.sh`) |
| `REF`, `NL` | fold-1 diagnostics of the conventional round: student with the tracking reference, student without skill input |
| `_f2`–`_f4` | cross-validation fold (no tag = fold 1) |
| `_d2` | second DAgger stage (2000 iterations after 4000). These are the evaluated students |
| `_s1`–`_s5` | RL seed |
| `v6g10nnt` | perception ablation (time-to-step entries of the stake features zeroed) |
| `v6g10x`, `v6g10xw`, `v6g10x25` | pilots of the cross-trial augmentation |
| `rt11xx`, `v6g10xm_base_*` | real-robot retraining |

The real robot ran `deployment/Deployment_mocap`: `model_bridge` = `rtJb` iteration 2000 (crouched traverse),
`model_stakes` = `rtJs` iteration 2000 (pole runs). Their source checkpoints are
`students/students_rtJb_s5/Z3/` and `students/students_rtJs_s5/Z3/`, the exports
`deploy_exports/rtJb_c2000/` and `deploy_exports/rtJs_c2000/`.

## Pipeline and entry points

| Stage | Code |
|---|---|
| References (filtering, contacts from force plates, contact-aware inverse dynamics) | `code/stage1_reference/build_reference.py`, `run_all.sh` |
| Motion-tracking experts | `code/hnu_tasks/`, `code/train_hnu.py`, `code/launch_expert.sh`, `code/eval_expert.py` |
| Annotations and skill encoders | `code/stage4_annotations/`, `code/stage4_latent/` (`train_all.sh`, `export_z_tables.py`) |
| Students (DAgger from the expert bank) | `code/hnu_tasks/distill.py`, `code/distill_train.py`, `code/eval_student.py` |
| Main study (folds 1–4, L0–L3, seeds 1–3) | `code/run_v6g10xm_l40.sh` (decoupled), `code/run_v6g10n_l40.sh` + `run_v6g10n_l40b.sh` (conventional), `code/run_zr_l40.sh`, `run_zr_l40_s23.sh` (ER-dec), `code/run_zrn_l40.sh` (ER-conv) |
| Cross-trial augmentation, splits | `code/tools/make_xz.py`, `make_aug_ts.py`, `make_cv_folds.py` |
| Geometric planner (re-timing a retrieved skill for a layout) | `code/tools/make_z_plan.py`, `make_z_cands_plan.py`, `std_plan.py` (standard start, real robot) |
| Router (gate + task) and retrieval | `code/stage4_latent/router_gate4.py`, `retrieve_skill.py` |
| Success predictor Q | `code/stage5_quality/quality.py` |
| Preview selection and end-to-end evaluation with refusal | `code/tools/preview_select.py`, `e2e_refusal_pool.py`, `e2e_chain_v2.py` |
| Statistics (cluster bootstrap, closed-loop CIs) | `code/tools/closed_loop_ci.py`, `cv_fold_table.py`, `aggregate_*.py` |
| Sim2sim (CPU MuJoCo), ONNX export, parity | `code/deploy/sim2sim.py`, `export_student.py`, `parity_check.py` |
| Real robot (placement check with motion capture, obstacle measurement, health check) | `code/deploy/place_live.py`, `obstacles_live.py`, `code/deploy/robot_side/` |

`code/tools/` also holds the analysis, figure and diagnostic scripts of the project as they were used. Most scripts
start with a docstring or comment that says what they do and when they were written.

## Environment

Python 3.11, mjlab 1.6.0, MuJoCo 3.11.0, MuJoCo-Warp 3.11.0, warp-lang 1.17.0, PyTorch 2.14.0, rsl-rl-lib 5.4.2,
onnx 1.23.0 (full list in `docs/requirements_mjlab_venv.txt`). Training ran on 5 × NVIDIA L40. The deployment code
needs onnxruntime and mujoco on the robot computer.

The scripts use absolute paths of the training server: code in `/root/gpufree-data/hnu_v2/code`, data in
`/root/gpufree-data/hnu_v2/data`, runs in `/root/gpufree-data/hnu_v2/runs`, Python in
`/root/gpufree-data/venvs/mjlab/bin/python`. `scripts/setup_layout.py` (see above) links every model file of
`docs/MANIFEST_models.json` from `models/` back to its original path (`runs/students_*`,
`runs/logs/rsl_rl/hnu_track_<task>/<run>`, `runs/latent_*`, `deploy_exports/`) and, with `--prefix-copy`, writes a copy of
`code/` in which `/root/gpufree-data/hnu_v2` is replaced by your directory. Set `PY` in the shell scripts to your
Python, or link your environment to `/root/gpufree-data/venvs/mjlab`.

Example check of an unpacked release (fold 1, level L3, seed 1, condition A without DR and pushes, as in
`code/run_v6g10xm_l40.sh`). Run from a fresh download, it reproduced `runs/results_v6g10xm_s1_dr0/eval_student_Z3.json`
of the release within the rollout randomness (success over all clips of each task within 2 points, e.g. 85.4 % vs
85.3 % for the three-step pole run, larger spread on the small val subsets):

```bash
H=/path/to/hnu_v2; cd $H/code
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
HNU_SPLIT_JSON=$H/data/clip_split_cv/fold1/split.json HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 \
  python eval_student.py --student $H/runs/students_v6g10xm_d2_s1/Z3/student_2000.pt \
  --latent-dir $H/data/z_tables_cv_f1/Z3 --per-clip 32 --out $H/runs/check_eval --tag _Z3
```

Not included: the robot-side control program of the lab (`hnu_pipeline.py`, it loads the ONNX policies of the bundle,
see the deployment handbook), the raw recordings of the real-robot runs and the third-party text encoder.

## License

Code and models: MIT (see `LICENSE`). The dataset has its own license in its repository.
