"""Where in the motion do held-out students fail on test-subject clips?  (sim2sim rows of diag_v6g10h_s1_Z0)
For every failed test clip: failure time, cause, and the motion phase at that moment (time to the first stake
stance / the bridge, pelvis speed and height)."""
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
rows = json.load(open(f"{H}/deploy_exports/diag_v6g10h_s1_Z0/nominal.json"))["rows"]
meta = json.load(open(f"{H}/deploy_exports/diag_v6g10h_s1_Z0/policy_meta.json"))
test = set(open(f"{H}/data/clip_split_v5/clips_test.txt").read().split())
for r in rows:
    if r["clip"] not in test or r["success"] or r["trial"] > 0:
        continue
    for d in meta["motion_dirs"]:
        f = f"{d}/{r['clip']}_motion.npz"
        if os.path.exists(f):
            break
    z = np.load(f)
    t = r["steps"] / 50
    k = min(r["steps"], len(z["body_pos_w"]) - 1)
    v = np.linalg.norm(z["body_lin_vel_w"][k, 0, :2])
    h = z["body_pos_w"][k, 0, 2]
    note = ""
    sj = f.replace("_motion.npz", "_stakes.json")
    if os.path.exists(sj):
        j = json.load(open(sj))
        first = j["stakes"][0]["stance"][0] / j["fps"]
        note = f"first stake stance at {first:.2f} s ({t - first:+.2f} s)"
    bj = f.replace("_motion.npz", "_bridge.json")
    if os.path.exists(bj):
        b = json.load(open(bj))
        parts = b["parts"] if "parts" in b else b["boxes"]
        bx = parts["plank"]["center"][:2]
        dist = np.linalg.norm(np.array(bx) - z["body_pos_w"][k, 0, :2])
        note = f"pelvis {dist:.2f} m from the beam centre"
    print(f"{r['clip']:18s} fail at {t:5.2f} s / {r['T'] / 50:5.2f} s  cause {r['cause']:12s} ref speed {v:4.2f} m/s "
          f"ref pelvis h {h:.2f} m  {note}")
