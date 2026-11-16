"""Stake task at the deployment latency (0-20 ms): success and tracking error of h10c vs h10d (clean and push)."""
import glob
import json

H = "/root/gpufree-data/hnu_v2/runs/stakes_eval"
for T in ("ThreeStepPoleRun", "FiveStepPoleRun"):
    for v in ("robust_stakes_h10c", "robust_stakes_h10d"):
        for X in ("", "_push_noise"):
            f = glob.glob(f"{H}/{v}/{T}/lat0-20/eval_{T}_stakes{X}.json")
            if f:
                j = json.load(open(f[0]))
                print(f"{T} {v[-4:]} {'push' if X else 'clean'}: all-stakes {100 * j['success_all_stakes']:.1f}%  "
                      f"body err {100 * j['mean_body_pos_err']:.1f} cm  anchor xy {100 * j['mean_anchor_xy_err']:.1f} cm")
