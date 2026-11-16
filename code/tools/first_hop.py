"""DEMO: the first hop onto stake 1 (user: on the robot the first hop "does not take off").  From sim2sim traces
(HNU_S2S_TRACE + HNU_S2S_TORQUE, 12 s hold) in the window [stake-1 touchdown - 0.6 s, touchdown + 0.1 s] of the reference:
  landing-foot apex and pelvis rise (sim vs reference), push-off leg (opposite foot) peak joint speed, peak torque /
  effort limit, share of physics steps at the limit, peak positive power, and the outcome on stake 1.
  python first_hop.py <export_dir> <layout_dir> <task> <trace_dir> [<trace_dir> ...]"""
import glob
import json
import os
import sys

import mujoco
import numpy as np

ex, lay, task = sys.argv[1:4]
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
scene = os.path.join(ex, "scene.mjb")
m = mujoco.MjModel.from_binary_path(scene if os.path.exists(scene) else os.path.join(ex, "..", "scene.mjb"))
dd = mujoco.MjData(m)
clip = sorted(glob.glob(os.path.join(lay, f"{task}_*_motion.npz")))[0]
ref = np.load(clip)
S = json.load(open(clip.replace("_motion.npz", "_stakes.json")))["stakes"]
nm = {n: i for i, n in enumerate(meta["body_names"])}
ANK = {"L": nm["left_ankle_roll_link"], "R": nm["right_ankle_roll_link"]}
fc = ref["foot_contact"].astype(bool)
s1 = S[0]
F, O = s1["foot"], "R" if s1["foot"] == "L" else "L"
c = fc[:, 0 if F == "L" else 1]
dist = np.linalg.norm(ref["body_pos_w"][:, ANK[F], :2] - [s1["x"], s1["y"]], axis=1)
k = int(np.argmin(np.where(c, dist, 9.0)))
while k > 0 and c[k - 1]:
    k -= 1
td = k  # reference touchdown frame on stake 1
fps = float(ref["fps"])
W = range(td - int(0.6 * fps), td + int(0.1 * fps))
g = ref["body_pos_w"][:, [ANK["L"], ANK["R"]], 2].min()
ref_apex = ref["body_pos_w"][W.start:td + 1, ANK[F], 2].max() - g
ref_pel = ref["body_pos_w"][W.start:td + 1, 0, 2].max() - ref["body_pos_w"][W.start, 0, 2]
side = "left" if O == "L" else "right"
J = [f"{side}_{j}_joint" for j in ("hip_pitch", "knee", "ankle_pitch")]
ji = [meta["joint_names"].index(j) for j in J]
qa = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{j}")] for j in J]
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{'left' if F == 'L' else 'right'}_ankle_roll_link")
H = 12 * 50
dt = float(meta["sim"]["control_dt"])
print(f"{task}: stake 1 foot {F}, push-off {O}, ref touchdown frame {td}; ref apex {ref_apex:.2f} m, ref pelvis rise {ref_pel:.2f} m")
for tdir in sys.argv[4:]:
    R = {"apex": [], "pel": [], "on": [], "qd": [], "tq": [], "sat": [], "pw": []}
    for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
        z = np.load(f)
        qp, tau, eff, dec = z["qpos"], z["tau"], z["eff"], int(z["dec"])
        if H + W.stop >= len(qp) or not len(tau):
            continue
        a = []
        for kk in range(W.start, td + 1):
            dd.qpos[:] = qp[H + kk][:m.nq]
            mujoco.mj_kinematics(m, dd)
            a.append(dd.xpos[bid, 2])
        R["apex"].append(max(a) - g)
        R["pel"].append(qp[H + W.start:H + td + 1, 2].max() - qp[H + W.start, 2])
        q = qp[H + W.start:H + W.stop][:, qa]
        qd = np.abs(np.diff(q, axis=0)) / dt
        # tau row i = physics step of control step i (trace row 0 = the initial state, then one row per control step)
        t = tau[(H + W.start) * dec:(H + W.stop) * dec][:, ji]
        e = eff[ji]
        qdp = np.repeat(np.diff(q, axis=0) / dt, dec, axis=0)[:len(t)]
        tc = np.clip(t[:len(qdp)], -e, e)
        R["qd"].append(qd.max(0))
        R["tq"].append(np.abs(t).max(0) / e)
        R["sat"].append((np.abs(t) >= e).mean(0))
        R["pw"].append((tc * qdp).max(0))
        R["on"].append(str(z["cause"]))
    if not R["apex"]:
        print(f"  {tdir}: no traces with torque")
        continue
    n = len(R["apex"])
    print(f"  {os.path.basename(tdir)} (n {n}): landing-foot apex {np.mean(R['apex']):.2f} m (min {np.min(R['apex']):.2f}), "
          f"pelvis rise {np.mean(R['pel']):.3f} m, causes {dict(zip(*np.unique(R['on'], return_counts=True)))}")
    for i, j in enumerate(J):
        print(f"    {j:24s} peak speed {np.mean([x[i] for x in R['qd']]):5.1f} rad/s  peak |tau|/limit "
              f"{np.mean([x[i] for x in R['tq']]):.2f} (limit {eff[ji][i]:.0f})  at limit {100 * np.mean([x[i] for x in R['sat']]):4.1f}% "
              f"  peak power {np.mean([x[i] for x in R['pw']]):5.0f} W")
