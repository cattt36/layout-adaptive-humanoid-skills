"""Motion fidelity on the stake task: mean body / joint tracking error of the non-robust 0.10 m expert (g2_h10) vs the
latency-robust one (h10c, zero latency) -- does the robust expert still move like the reference?"""
import glob
import json

H = "/root/gpufree-data/hnu_v2/runs/stakes_eval"
for T in ("ThreeStepPoleRun", "FiveStepPoleRun"):
    for tag, pat in (("g2_h10 (non-robust)", f"{H}/g2_h10/{T}/eval_{T}_stakes.json"),
                     ("h10c (latency-robust, lat0)", f"{H}/robust_stakes_h10c/{T}/lat0/eval_{T}_stakes.json")):
        f = glob.glob(pat)
        if not f:
            continue
        j = json.load(open(f[0]))
        jerr = sum(c["mean_joint_pos_err"] for c in j["clips"]) / len(j["clips"])
        print(f"{T} {tag}: all-stakes {j['success_all_stakes']:.3f}  body err {100 * j['mean_body_pos_err']:.1f} cm  "
              f"joint err {jerr:.3f} rad  anchor xy {100 * j['mean_anchor_xy_err']:.1f} cm")
