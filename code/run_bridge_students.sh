#!/bin/bash
# Students L0-L4 (v4d latents, RL seed 1) distilled in the bridge scene: CrouchedTraverse under the capture-video bridge
# (teacher: bridge-perception expert), W1/W2 on flat ground (flat experts).  Every student additionally reads the
# low-dimensional obstacle features (execution-layer perception); the deployment chain is unchanged.
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v4d
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_bridge
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/bridge_students.log; }
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
S=$H/runs/students_bridge
R=$H/runs/results_bridge
mkdir -p $S $R
student() {  # P gpu
  [ -f $S/$1/student_2500.pt ] && return
  CUDA_VISIBLE_DEVICES=$2 $PY distill_train.py --profile bridge_$1 $CK --num-envs 4096 --iters 2500 --seed 1000 --scene bridge \
    --latent-dir $H/data/z_tables_$V/$1 --out $S/$1 > $N/bridge_E_$1.log 2>&1
}
log "teacher eval"
for X in "" "--push"; do
  CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene bridge --z-offsets 0,10,25,40 \
    --latent-dir $H/data/z_tables_$V/Z4 --per-clip 8 $X --out $R --tag _teacher >> $N/bridge_F_teacher.log 2>&1
done
log "students"
student Z0 0 & student Z1 0 & student Z2 0 & student Z3 1 & student Z4 1 & wait
for P in Z0 Z1 Z2 Z3 Z4; do student $P 0; done  # retry failures one by one
log "eval"
i=0
for P in Z0 Z1 Z2 Z3 Z4; do
  ( for X in "" "--push"; do
      CUDA_VISIBLE_DEVICES=$((i % 2)) $PY eval_student.py --student $S/$P/student_2500.pt --latent-dir $H/data/z_tables_$V/$P \
        --per-clip 8 $X --out $R --tag _$P >> $N/bridge_F_$P.log 2>&1
    done ) &
  i=$((i + 1))
done
wait
log "videos"
export MUJOCO_GL=egl
for C in CrouchedTraverse_007 CrouchedTraverse_021; do
  CUDA_VISIBLE_DEVICES=0 $PY record_grid.py --clip $C --students-root $S --z-root $H/data/z_tables_$V $CK \
    --out $H/videos/bridge_students/grid_$C.mp4 >> $N/bridge_videos.log 2>&1
done
log "bridge students finished"
