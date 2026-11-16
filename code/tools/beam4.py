"""DEMO (flat CT, user: the bridge must keep FOUR posts): search straight beam lines (reference plank axis
rotated / shifted) and four post positions on them (spacing >= GAP, end posts <= beam length - 0.1 m apart) that
maximise the smallest post clearance (min over robot bodies and frames, 2nd percentile over the simulated runs; body
origins), and require that the robot passes under the beam between the end posts.  Writes the best layouts as bridge
jsons (posts moved, plank on top of them along the line, underside = max(reference, robot top + margin); ramp / leg
parked far away) for a physical check in sim2sim (contacts, falls).
  python beam4.py <export_dir> <motion_npz> <trace_dir> <hold_s> <bridge_json> <out_dir> [n_best]"""
import copy
import glob
import json
import os
import sys

import mujoco
import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


ex, mfile, tdir, hold_s, bj, od = sys.argv[1:7]
NB = int(sys.argv[7]) if len(sys.argv) > 7 else 4
GAP, MARGIN = 0.70, 0.08
# (user: post heights cannot change): DEMO_UNDER_FIX=1 keeps the reference beam height and only accepts lines
# whose robot top under the beam (98th pct) + DEMO_MARGIN (default 0.05 m) stays below it; wider search
FIX = os.environ.get("DEMO_UNDER_FIX", "0") == "1"
if FIX:
    MARGIN = float(os.environ.get("DEMO_MARGIN", "0.05"))
os.makedirs(od, exist_ok=True)
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
model = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
data = mujoco.MjData(model)
bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in meta["body_names"]]
rob = {b for b in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("robot/")}
gids = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in rob])
grb = model.geom_rbound[gids]
ref = np.load(mfile)
T = len(ref["joint_pos"])
a0 = yaw(ref["body_quat_w"][0, 0])
o0 = ref["body_pos_w"][0, 0, :2]
H = int(round(float(hold_s) / float(meta["sim"]["control_dt"])))
# everything in the CLIP WORLD frame here (the json frame); start-frame output only for printing
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T  # noqa: E731
BXY, GXY, GZ = [], [], []
for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
    qp = np.load(f)["qpos"]
    b, g, z = [], [], []
    for k in range(0, T, 2):
        if H + k >= len(qp):
            break
        data.qpos[:] = qp[H + k][:model.nq]
        mujoco.mj_kinematics(model, data)
        b.append(data.xpos[bid, :2].copy())
        g.append(data.geom_xpos[gids, :2].copy())
        z.append(data.geom_xpos[gids, 2] + grb)
    BXY.append(np.array(b)), GXY.append(np.array(g)), GZ.append(np.array(z))
pel = [x[:, 0] for x in BXY]
J = json.load(open(bj))
P = J["parts"]
pl = P["plank"]
r = float(P["post0"]["radius"])
bw = float(pl["half"][1])
c0 = np.array(pl["center"][:2])
L = float(pl["half"][0])
ref_under = pl["center"][2] - pl["half"][2]


def clr(c):
    return float(np.percentile([np.min(np.linalg.norm(x - c, axis=-1)) - r for x in BXY], 2))


cands = []
for dyaw in (range(-45, 46, 5) if FIX else range(-30, 31, 5)):
    for off in (np.arange(-1.2, 1.21, 0.1) if FIX else np.arange(-0.8, 0.81, 0.1)):
        yw = pl["yaw"] + np.radians(dyaw)
        ax = np.array([np.cos(yw), np.sin(yw)])
        nrm = np.array([-ax[1], ax[0]])
        ts = np.round(np.arange(-L - 1.0, L + 1.001, 0.05), 3)
        cl = np.array([clr(c0 + off * nrm + t * ax) for t in ts])
        # best 4 posts: maximise the minimum clearance; brute force over thresholds
        best = None
        for th in np.sort(np.unique(np.round(cl, 3)))[::-1]:
            ok = ts[cl >= th]
            if len(ok) < 4:
                continue
            # greedy inside every window of length <= 2L - 0.1
            for t1 in ok:
                sel = [t1]
                for t in ok:
                    if t - sel[-1] >= GAP and t - t1 <= 2 * L - 0.10:
                        sel.append(t)
                    if len(sel) == 4:
                        break
                if len(sel) == 4:
                    best = (float(th), sel)
                    break
            if best:
                break
        if not best:
            continue
        th, sel = best
        p = [c0 + off * nrm + t * ax for t in sel]
        # the robot must pass under the beam between the end posts: pelvis crosses the line inside [t1, t4]
        cross = 0
        for q in pel:
            s = (q - c0 - off * nrm) @ nrm
            a = (q - c0 - off * nrm) @ ax
            idx = np.where(np.sign(s[1:]) != np.sign(s[:-1]))[0]
            cross += int(np.sum((a[idx] > sel[0]) & (a[idx] < sel[-1])))
        cross /= max(len(pel), 1)
        if cross < 1:
            continue
        tops = []
        for g, z in zip(GXY, GZ):
            d = g - p[0]
            sa = d @ ax
            lat = np.abs(d @ nrm)
            under = (sa > 0) & (sa < sel[-1] - sel[0]) & (lat < bw + grb[None])
            tops.append(float(np.max(np.where(under, z, 0.0))))
        need = max(ref_under, float(np.percentile(tops, 98)) + MARGIN)
        if FIX and need > ref_under + 1e-6:
            continue
        cands.append(dict(clear=th, dyaw=dyaw, off=float(off), ts=[float(t) for t in sel], posts=[q.tolist() for q in p],
                          yaw=float(yw), under=need, crossings=cross))
cands.sort(key=lambda c: (-c["clear"], c["under"]))
print(f"reference: post clearance (body origins) {[round(clr(np.array(P[k]['center'][:2])), 3) for k in sorted(P) if k.startswith('post')]}, "
      f"underside {ref_under:.2f} m, {len(BXY)} runs")
print("| # | min post clearance (m) | rot | shift | posts (start frame x, y) | beam underside (m) | crossings |\n|---|---|---|---|---|---|---|")
for i, c in enumerate(cands[:NB]):
    ps = "; ".join(f"({to_s(q)[0]:.2f}, {to_s(q)[1]:.2f})" for q in c["posts"])
    print(f"| {i} | {c['clear']:+.3f} | {c['dyaw']:+d} | {c['off']:+.1f} | {ps} | {c['under']:.2f} | {c['crossings']:.1f} |")
    j = copy.deepcopy(J)
    Q = j["parts"]
    hh = c["under"] / 2
    for k, q in zip(sorted(k for k in Q if k.startswith("post")), c["posts"]):
        Q[k]["center"] = [q[0], q[1], hh]
        Q[k]["half_height"] = hh
    mid = (np.array(c["posts"][0]) + np.array(c["posts"][-1])) / 2
    Q["plank"]["center"] = [float(mid[0]), float(mid[1]), c["under"] + Q["plank"]["half"][2]]
    Q["plank"]["yaw"], Q["plank"]["pitch"] = c["yaw"], 0.0
    for k in ("ramp", "leg"):
        if k in Q:
            Q[k]["center"] = [Q[k]["center"][0] + 60.0, Q[k]["center"][1] + 60.0, Q[k]["center"][2]]
    j["demo_layout"] = c
    json.dump(j, open(os.path.join(od, f"cand{i}.json"), "w"), indent=1)
json.dump(cands[:NB], open(os.path.join(od, "cands.json"), "w"), indent=1)
