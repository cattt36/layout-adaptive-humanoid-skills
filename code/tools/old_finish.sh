#!/bin/bash
# L40, (user decision): after the cross-trial z pilot (tools/xz_l40.sh, xz_l40_plan4.sh), finish the five
# v6g10n students interrupted at 15:00 (round 1 done, round 2 + evaluation missing) with the unchanged recipe:
#   fold 4 seed 2 L1 / L2 / L3  (old method complete for folds 1-4 x seeds 1-2; ablation "without cross-trial z")
#   fold 1 seed 3 L0 / L3       (old vs new L0 / L3 for fold 1 seeds 1-3; deployable-condition / E2E ablation)
# one() and the environment are taken verbatim from run_v6g10n_l40b.sh (lines up to the job list).
H=/root/gpufree-data/hnu_v2
cd $H/code
while pgrep -fx "bash tools/xz_l40.sh" > /dev/null || pgrep -fx "bash tools/xz_l40_plan4.sh" > /dev/null; do sleep 60; done
source <(sed -n '17,77p' run_v6g10n_l40b.sh | sed '/^# teacher bank on this hardware/,/^log "teacher bank evaluated"$/d')
log "old_finish: resume the 5 interrupted students (round 2 + evaluation)"
one 4 Z1 0 2 & one 4 Z2 1 2 & one 4 Z3 2 2 & one 1 Z0 3 3 & one 1 Z3 4 3 & wait
log "old_finish: complete"
