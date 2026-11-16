"""Table of the tempo check (runs/ts_check): success % per action at playback speeds 0.85 ... 1.15 (1.0 = the
regular v6g10j / teacher-bank results).  Teacher: train trials; students: train and test trials."""
import json
import os

H = "/root/gpufree-data/hnu_v2/runs"
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
SPEEDS = ["085", "090", "100", "110", "115"]


def path(sp, dr, who):
    f = "eval_teacher_teacher.json" if who == "teacher" else f"eval_student_{who}.json"
    if sp == "100":
        d = f"{H}/results_v6g10j_bank_teacher_dr{dr}" if who == "teacher" else f"{H}/results_v6g10j_s1_dr{dr}"
    else:
        d = f"{H}/ts_check/ts{sp}/dr{dr}"
    return f"{d}/{f}"


for dr in ("0", "1"):
    print(f"\n=== {'nominal' if dr == '0' else 'randomised'}: success % DuMu / ThreeStepPoleRun / FiveStepPoleRun (ThreeStepPoleRun / FiveStepPoleRun stakes all hit %)")
    print("model   split " + "".join(f"| x{int(sp) / 100:.2f}                     " for sp in SPEEDS))
    for who, splits in (("teacher", ("train",)), ("Z0", ("train", "test")), ("Z4", ("train", "test"))):
        for sp_ in splits:
            cells = []
            for sp in SPEEDS:
                f = path(sp, dr, who)
                if not os.path.exists(f):
                    cells.append(f"{'-':26s}")
                    continue
                s = json.load(open(f))["summary"]
                succ = "/".join(f"{100 * s[f'{t}/{sp_}']['success']:.0f}" for t in TASKS)
                hit = "/".join(f"{100 * (s[f'{t}/{sp_}'].get('stakes_all_hit') or 0):.0f}" for t in TASKS[1:])
                cells.append(f"{succ + ' (' + hit + ')':26s}")
            name = {"Z0": "L0 s1", "Z4": "L4 s1"}.get(who, who)
            print(f"{name:7s} {sp_:5s} " + "".join(f"| {c}" for c in cells))
