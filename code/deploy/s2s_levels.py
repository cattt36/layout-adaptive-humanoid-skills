"""sim2sim success of the selected deployment students per level (deploy_exports/<V>/L*_s*)."""
import glob
import json
import os
import sys

V = sys.argv[1] if len(sys.argv) > 1 else "v6g10s"
base = f"/root/gpufree-data/hnu_v2/deploy_exports/{V}"
CONDS = [("理想", "sim2sim_none_dt0.005.json"), ("推扰", "sim2sim_none_push_dt0.005.json"),
         ("延迟+参数随机", "sim2sim_0-20_dr_dt0.005.json"), ("延迟+参数随机+推扰", "sim2sim_0-20_push_dr_dt0.005.json"),
         ("延迟+参数随机(1ms)", "sim2sim_0-20_dr_dt0.001.json")]
for T, cn in (("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")):
    print(f"\n{cn}\n| 层级 | " + " | ".join(c for c, _ in CONDS) + " |")
    for d in sorted(glob.glob(f"{base}/L*_s*")):
        if not os.path.isdir(d):
            continue
        cells = []
        for _, f in CONDS:
            p = os.path.join(d, f)
            cells.append(f"{100 * json.load(open(p))['summary'][T]['success']:.1f}" if os.path.exists(p) else "-")
        par = open(os.path.join(d, "parity.txt")).read().strip().splitlines()[-1] if os.path.exists(os.path.join(d, "parity.txt")) else "?"
        print(f"| {os.path.basename(d)} | " + " | ".join(cells) + f" | {par}")
