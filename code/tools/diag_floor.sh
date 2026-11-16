#!/bin/bash
# 4090, (user request): diagnostic of floor contacts during the stake run (eval_student.py HNU_EVAL_FLOOR=1:
# robot foot on the floor at a step where no reference foot is at floor level, between the first and last stake
# contact).  Fold 1, RL seed 1, nominal, 16 rollouts per clip, criterion v2.  Policies: teacher bank (own reference),
# new students L0 / L3, old student L3, explicit-reference baseline ZR -- condition A (own z / reference, human clock)
# and B (library + planner plan4).  Table: runs/diag_floor.md
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; O=$H/runs/diag_floor
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_DR=0 HNU_START_FRAC=0 HNU_EVAL_CRIT=task HNU_EVAL_FLOOR=1 HNU_SPLIT_JSON=$D/clip_split_cv/fold1/split.json
[ -d $D/z_lib_f1/plan4/ZR ] || { HNU_LEVELS=ZR $PY tools/make_z_library.py 1 && HNU_LEVELS=ZR HNU_SKIP_GEOM=1 $PY tools/make_z_plan.py 1 plan4; } > /dev/null 2>&1
GA="HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10"
GB="HNU_GEOM_W1=$D/geom_plan4_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/geom_plan4_f1/ref_stakes_s83_h10"
ev() {  # name geom-env latent-dir policy-args...
  local NAME=$1 GE=$2 LD=$3; shift 3
  [ -f $O/$NAME/done ] && return
  env $GE CUDA_VISIBLE_DEVICES=1 nice $PY eval_student.py "$@" --latent-dir $LD --per-clip 16 --out $O/$NAME --tag _x \
    >> $O.log 2>&1 && touch $O/$NAME/done
}
mkdir -p $O
NEW=$H/runs/students_v6g10xm_d2_s1 OLD=$H/runs/students_v6g10n_d2_s1
ev teacher_A "$GA" $D/z_tables_cv_f1/Z0 --teacher $(cat $H/runs/night/v6g10n_teachers.txt) --scene geom
for P in Z0 Z3; do
  ev new_${P}_A "$GA" $D/z_tables_cv_f1/$P --student $NEW/$P/student_2000.pt
  ev new_${P}_B "$GB" $D/z_lib_f1/plan4/$P --student $NEW/$P/student_2000.pt
done
ev old_Z3_A "$GA" $D/z_tables_cv_f1/Z3 --student $OLD/Z3/student_2000.pt
ev old_Z3_B "$GB" $D/z_lib_f1/plan4/Z3 --student $OLD/Z3/student_2000.pt
ev zr_A "$GA" $D/z_tables_cv_f1/ZR --student $NEW/ZR/student_2000.pt
ev zr_B "$GB" $D/z_lib_f1/plan4/ZR --student $NEW/ZR/student_2000.pt
$PY - <<'EOF' > $H/runs/diag_floor.md
import json, os, numpy as np
O = "/root/gpufree-data/hnu_v2/runs/diag_floor"
test = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_cv/fold1/split.json"))["test"])
names = [("teacher_A", "teacher（自身参考）"), ("new_Z0_A", "新学生 L0，条件 A"), ("new_Z3_A", "新学生 L3，条件 A"),
         ("old_Z3_A", "旧学生 L3，条件 A"), ("zr_A", "ZR，条件 A"), ("new_Z0_B", "新学生 L0，条件 B"),
         ("new_Z3_B", "新学生 L3，条件 B"), ("old_Z3_B", "旧学生 L3，条件 B"), ("zr_B", "ZR，条件 B")]
print("# 踩桩阶段的非参考触地（第 1 折 test 试次，种子 1，理想环境，每片段 16 次）\n")
print("指标：从第一次碰桩到最后一次碰桩之间，机器人有脚碰地、而真人参考此刻没有脚着地的情况。次数 = 每回合的触地起始次数（两只脚合计）；时长 = 每回合累计秒数；参考着地占比 = 同一时间窗内真人参考有脚着地的比例。\n")
print("| 策略 | 动作 | 成功率 v2 | 触地次数 / 回合 | 触地时长 s / 回合 | 有非参考触地的回合 % | 参考着地占比 |")
print("|---|---|---|---|---|---|---|")
for n, lab in names:
    fs = [f for f in os.listdir(f"{O}/{n}") if f.startswith("eval_") and f.endswith(".json")] if os.path.isdir(f"{O}/{n}") else []
    if not fs:
        continue
    rows = [r for r in json.load(open(f"{O}/{n}/{fs[0]}"))["clips"] if r["clip"] in test]
    for t in ("ThreeStepPoleRun", "FiveStepPoleRun"):
        r = [x for x in rows if x["task"] == t and x.get("floor_touch_events") is not None]
        if not r:
            continue
        m = lambda k: np.nanmean([x[k] for x in r])  # noqa: E731
        print(f"| {lab} | {t} | {100 * m('success_v2'):.0f} | {m('floor_touch_events'):.2f} | {m('floor_touch_s'):.3f} | "
              f"{100 * m('floor_touch_any'):.0f} | {100 * m('ref_floor_frac'):.0f}% |")
EOF
echo "diag done" >> $O.log
