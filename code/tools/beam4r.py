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
# DEMO_TIGHT=1: robot top from the COLLISION geoms' oriented boxes (geom_aabb) instead of the bounding spheres of
# all geoms (too conservative: rejected every fixed-height line although the real robot never touched the beam)
TIGHT = os.environ.get("DEMO_TIGHT", "0") == "1"
if TIGHT:
    gids = np.array([g for g in gids if model.geom_contype[g] or model.geom_conaffinity[g]])
grb = model.geom_rbound[gids]
gab = model.geom_aabb[gids].reshape(-1, 2, 3) if TIGHT else None  # [G, (center, half), 3] in the geom frame
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
        if TIGHT:
            m3 = data.geom_xmat[gids].reshape(-1, 3, 3)
            z.append(data.geom_xpos[gids, 2] + np.einsum("gj,gj->g", m3[:, 2], gab[:, 0])
                     + np.einsum("gj,gj->g", np.abs(m3[:, 2]), gab[:, 1]))
        else:
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


# DEMO_RIGID=1 (real bridge = one rigid piece, post spacing fixed): move / rotate the WHOLE bridge (also turned
# end for end); the posts keep their reference positions along the beam, shifted together by s
RIGID = os.environ.get("DEMO_RIGID", "0") == "1"
ax0 = np.array([np.cos(pl["yaw"]), np.sin(pl["yaw"])])
t_ref = np.array(sorted(float((np.array(P[k]["center"][:2]) - c0) @ ax0) for k in P if k.startswith("post")))
# rigid bridge keeps the sloped beam (reference pitch, post heights); DEMO_DZ = height correction to the real bridge
# (mocap D2 post tops 1.178 / 1.146 / 1.106 / 1.082 m = sim reference of clip 013 + 0.027 m)
DZ = float(os.environ.get("DEMO_DZ", "0.0"))
SLOPE = -np.tan(float(pl["pitch"]))  # underside rise per m along the beam axis
if RIGID:
    from scipy.spatial import cKDTree  # mjlab venv
    TREES = [cKDTree(x.reshape(-1, 2)) for x in BXY]
cands = []
for dyaw in (range(-180, 180, 5) if RIGID else range(-45, 46, 5) if FIX else range(-30, 31, 5)):
    for off in (np.arange(-1.2, 1.21, 0.1) if FIX else np.arange(-0.8, 0.81, 0.1)):
        yw = pl["yaw"] + np.radians(dyaw)
        ax = np.array([np.cos(yw), np.sin(yw)])
        nrm = np.array([-ax[1], ax[0]])
        if RIGID:
            S = np.arange(-1.5, 1.501, 0.1)
            q = c0 + off * nrm + (t_ref[None, :] + S[:, None])[..., None] * ax  # [S, 4, 2]
            dist = np.stack([tr.query(q.reshape(-1, 2))[0] for tr in TREES])  # [runs, S * 4]
            cs = (np.percentile(dist, 2, axis=0) - r).reshape(len(S), len(t_ref)).min(1)
            k = int(np.argmax(cs))
            best = (float(cs[k]), [float(t + S[k]) for t in t_ref])
            ts, cl = np.array([]), np.array([])
        else:
            ts = np.round(np.arange(-L - 1.0, L + 1.001, 0.05), 3)
            cl = np.array([clr(c0 + off * nrm + t * ax) for t in ts])
            best = None
        # best 4 posts: maximise the minimum clearance; brute force over thresholds
        for th in ([] if RIGID else np.sort(np.unique(np.round(cl, 3)))[::-1]):
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
            if RIGID:  # robot top minus the LOCAL underside of the sloped beam (reference pitch, heights + DZ)
                z = z - (ref_under + DZ + SLOPE * (sa + t_ref[0]))
                tops.append(float(np.max(np.where(under, z, -1.0))))
            else:
                tops.append(float(np.max(np.where(under, z, 0.0))))
        if RIGID:
            need = ref_under + DZ
            if FIX and float(np.percentile(tops, 98)) + MARGIN > 0:
                continue
        else:
            need = max(ref_under, float(np.percentile(tops, 98)) + MARGIN)
            if FIX and need > ref_under + 1e-6:
                continue
        cands.append(dict(clear=th, dyaw=dyaw, off=float(off), ts=[float(t) for t in sel], posts=[q.tolist() for q in p],
                          yaw=float(yw), under=need, crossings=cross, top98=float(np.percentile(tops, 98))))
cands.sort(key=lambda c: (-c["clear"], c["under"]))
if RIGID:  # distinct layouts only (neighbouring rotations / shifts of the same layout are near-duplicates)
    keep = []
    for c in cands:
        if all(abs(c["dyaw"] - k["dyaw"]) >= 20 or abs(c["off"] - k["off"]) >= 0.4 for k in keep):
            keep.append(c)
    cands = keep
print(f"reference: post clearance (body origins) {[round(clr(np.array(P[k]['center'][:2])), 3) for k in sorted(P) if k.startswith('post')]}, "
      f"underside {ref_under:.2f} m, {len(BXY)} runs")
print("| # | min post clearance (m) | rot | shift | posts (start frame x, y) | beam underside (m) | crossings |\n|---|---|---|---|---|---|---|")
for i, c in enumerate(cands[:NB]):
    ps = "; ".join(f"({to_s(q)[0]:.2f}, {to_s(q)[1]:.2f})" for q in c["posts"])
    print(f"| {i} | {c['clear']:+.3f} | {c['dyaw']:+d} | {c['off']:+.1f} | {ps} | {c['under']:.2f} (robot top {c['top98']:.2f}) | {c['crossings']:.1f} |")
    j = copy.deepcopy(J)
    Q = j["parts"]
    hh = c["under"] / 2
    for k, q in zip(sorted(k for k in Q if k.startswith("post")), c["posts"]):
        Q[k]["center"] = [q[0], q[1], hh]
        Q[k]["half_height"] = hh
    mid = (np.array(c["posts"][0]) + np.array(c["posts"][-1])) / 2
    if RIGID:  # plank keeps its reference position relative to the posts
        mid = c0 + c["off"] * np.array([-np.sin(c["yaw"]), np.cos(c["yaw"])]) + (c["ts"][0] - t_ref[0]) * np.array(
            [np.cos(c["yaw"]), np.sin(c["yaw"])])
    Q["plank"]["center"] = [float(mid[0]), float(mid[1]), c["under"] + Q["plank"]["half"][2]]
    Q["plank"]["yaw"], Q["plank"]["pitch"] = c["yaw"], 0.0
    if RIGID:  # sloped beam, every post keeps its own reference height (+ DZ)
        for k in (k for k in Q if k.startswith("post")):
            hh = (P[k]["center"][2] + P[k]["half_height"] + DZ) / 2
            Q[k]["center"][2], Q[k]["half_height"] = hh, hh
        Q["plank"]["center"][2] = pl["center"][2] + DZ
        Q["plank"]["pitch"] = pl["pitch"]
    for k in ("ramp", "leg"):
        if k in Q:
            Q[k]["center"] = [Q[k]["center"][0] + 60.0, Q[k]["center"][1] + 60.0, Q[k]["center"][2]]
    j["demo_layout"] = c
    json.dump(j, open(os.path.join(od, f"cand{i}.json"), "w"), indent=1)
json.dump(cands[:NB], open(os.path.join(od, "cands.json"), "w"), indent=1)
