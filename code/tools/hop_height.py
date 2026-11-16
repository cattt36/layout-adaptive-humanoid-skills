"""DEMO: hop height of a blind 5PR tracker vs the reference (the vK model hopped ~13 cm lower with the left foot).
Ankle-roll link height above the floor (FK of the sim2sim traces, HNU_S2S_TRACE, 12 s hold) in reference frames 120-300
(run-up end .. stake 3), first 32 runs.
  python hop_height.py <export_dir> <motion_npz> <trace_dir>"""
import glob
import json
import os
import sys

import mujoco
import numpy as np

ex, mfile, tdir = sys.argv[1:4]
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
scene = os.path.join(ex, "scene.mjb")
model = mujoco.MjModel.from_binary_path(scene if os.path.exists(scene) else os.path.join(ex, "..", "scene.mjb"))
data = mujoco.MjData(model)
nm = {n: i for i, n in enumerate(meta["body_names"])}
bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{s}_ankle_roll_link") for s in ("left", "right")}
ref = np.load(mfile)
H, W = 12 * 50, range(120, 300)
rz = {s: ref["body_pos_w"][:, nm[f"{s}_ankle_roll_link"], 2] for s in ("left", "right")}
g = min(rz["left"].min(), rz["right"].min())
sim = {"left": [], "right": []}
for f in sorted(glob.glob(os.path.join(tdir, "*.npz")))[:32]:
    qp = np.load(f)["qpos"]
    z = {"left": [], "right": []}
    for k in W:
        if H + k >= len(qp):
            break
        data.qpos[:] = qp[H + k][:model.nq]
        mujoco.mj_kinematics(model, data)
        for s in z:
            z[s].append(data.xpos[bid[s], 2])
    for s in z:
        if z[s]:
            sim[s].append(max(z[s]) - g)
out = []
for s in ("left", "right"):
    a = np.array(sim[s])
    out.append(f"{s} apex: ref {rz[s][120:300].max() - g:.2f} m, sim mean {a.mean():.2f} m (min {a.min():.2f}, n {len(a)})")
print("\n".join(out))
