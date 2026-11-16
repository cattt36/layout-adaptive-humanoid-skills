#!/bin/bash
# Short progress of the student batch currently training on the L40 server (for the user): per running student its
# level and overall progress (round 1 = 4000 + round 2 = 2000 iterations), the estimated end of the batch, and the
# number of finished students of the study (L0-L3 main experiment).
H=/root/gpufree-data/hnu_v2
python3 - <<'EOF'
import glob, json, os, re, subprocess, datetime
H = "/root/gpufree-data/hnu_v2"
rows = []
for pid in subprocess.run(["pgrep", "-f", "distill_train.py --profile"], capture_output=True, text=True).stdout.split():
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().split("\0")
    except OSError:
        continue
    prof = cmd[cmd.index("--profile") + 1]
    out = cmd[cmd.index("--out") + 1]
    it, mins = 0, 0.0
    try:
        last = json.loads(open(os.path.join(out, "train_log.jsonl")).read().strip().splitlines()[-1])
        it, mins = last.get("it", 0), last.get("elapsed_min", 0.0)
    except Exception:
        pass
    r2 = "d2_" in prof
    done = (4000 if r2 else 0) + it
    lvl = re.search(r"_(Z\d|REF|NL)$", prof).group(1)
    name = {"Z0": "L0", "Z1": "L1", "Z2": "L2", "Z3": "L3"}.get(lvl, lvl) + ("（感知消融）" if "nnt" in prof else "")
    rate = it / mins if mins > 0 else 21.0
    left_min = ((2000 - it) if r2 else (4000 - it + 2000)) / max(rate, 1e-6) + 10
    rows.append((name, 100 * done / 6000, left_min))
now = datetime.datetime.now()
if rows:
    print(f"正在训练 {len(rows)} 个学生：" + "  |  ".join(f"{n} {p:.0f}%" for n, p, _ in sorted(rows)))
    print(f"本批预计完成：{(now + datetime.timedelta(minutes=max(r[2] for r in rows))).strftime('%m-%d %H:%M')}（含评估）")
else:
    print("当前没有学生在训练")
n = len([f for f in glob.glob(f"{H}/runs/students_v6g10n*_d2_s*/Z[0-3]/student_2000.pt") if "nnt" not in f])
print(f"主实验 L0–L3 已完成 {min(n, 48)}/48 个学生")
EOF
