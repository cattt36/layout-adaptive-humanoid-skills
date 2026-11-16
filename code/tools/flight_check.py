"""DEMO: does the 5PR tracker really take off on the hop onto stake 1?  Replays sim2sim traces (qpos + stake
mocap poses) with mj_forward and counts the control steps in [stake-1 touchdown - 0.6 s, touchdown + 0.1 s] with NO robot
foot contact (= flight), plus the reference flight from its foot_contact flags.
  python flight_check.py <export_dir> <layout_dir> <trace_dir> [<trace_dir> ...]"""
import glob
import json
import os
import sys

import mujoco
import numpy as np

ex, lay = sys.argv[1:3]
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
scene = os.path.join(ex, "scene.mjb")
m = mujoco.MjModel.from_binary_path(scene if os.path.exists(scene) else os.path.join(ex, "..", "scene.mjb"))
d = mujoco.MjData(m)
clip = sorted(glob.glob(os.path.join(lay, "FiveStepPoleRun_*_motion.npz")))[0]
ref = np.load(clip)
s1 = json.load(open(clip.replace("_motion.npz", "_stakes.json")))["stakes"][0]
nm = {n: i for i, n in enumerate(meta["body_names"])}
fc = ref["foot_contact"].astype(bool)
c = fc[:, 0 if s1["foot"] == "L" else 1]
a = nm["left_ankle_roll_link" if s1["foot"] == "L" else "right_ankle_roll_link"]
dist = np.linalg.norm(ref["body_pos_w"][:, a, :2] - [s1["x"], s1["y"]], axis=1)
k = int(np.argmin(np.where(c, dist, 9.0)))
while k > 0 and c[k - 1]:
    k -= 1
td, fps = k, float(ref["fps"])
W = range(td - int(0.6 * fps), td + int(0.1 * fps))
rf = ~fc[W.start:W.stop].any(1)
print(f"reference: flight {rf.sum()} steps = {1000 * rf.sum() / fps:.0f} ms before stake-1 touchdown (frame {td})")
foot_b = set()
for b in range(m.nbody):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if n.startswith("robot/") and "ankle_roll" in n:
        foot_b.add(b)
H = 12 * 50
for tdir in sys.argv[3:]:
    fl, ok = [], []
    for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
        z = np.load(f)
        qp = z["qpos"]
        if H + W.stop >= len(qp):
            continue
        d.mocap_pos[:], d.mocap_quat[:] = z["mocap_pos"], z["mocap_quat"]
        n = 0
        for kk in W:
            d.qpos[:] = qp[H + kk][:m.nq]
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            touch = any(m.geom_bodyid[d.contact[i].geom1] in foot_b or m.geom_bodyid[d.contact[i].geom2] in foot_b
                        for i in range(d.ncon))
            n += not touch
        fl.append(n)
        ok.append(str(z["cause"]) == "None")
    fl = np.array(fl)
    print(f"  {os.path.basename(os.path.dirname(tdir))}/{os.path.basename(tdir)} (n {len(fl)}): flight mean "
          f"{1000 * fl.mean() / fps:.0f} ms (min {1000 * fl.min() / fps:.0f}, max {1000 * fl.max() / fps:.0f}), "
          f"runs with >= 40 ms flight {100 * np.mean(fl >= 2):.0f} %, with none {100 * np.mean(fl == 0):.0f} %")
