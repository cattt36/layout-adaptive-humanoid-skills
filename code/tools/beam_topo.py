"""DEMO (user: the robot must weave around the four posts IN ORDER, as the human did; do not change the task).
Rigid real bridge (4 posts + sloped beam, post spacing fixed) moved only slightly around its REFERENCE pose (rotation about
the beam centre + shift across / along the beam, no flip) so that
  - every simulated run keeps the reference weaving: posts passed in the order 1-2-3-4 with the reference sides
    (closest approach of the pelvis, post left / right of the travel direction), and
  - the smallest post clearance (all robot body origins, 2nd percentile over runs, minus post radius) is maximised.
Writes the best layouts as bridge jsons (same format as tools/beam4.py DEMO_RIGID: posts keep their reference heights + DZ,
sloped beam, ramp / leg parked) for the physical check in sim2sim.
  python beam_topo.py <export_dir> <motion_npz> <trace_dir> <hold_s> <bridge_json> <out_dir> [n_best]
  env: DEMO_DZ (real post height correction, 013: 0.027), TOPO_MIN (fraction of runs that must keep the weaving, 0.98)"""
import copy
import glob
import json
import os
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


ex, mfile, tdir, hold_s, bj, od = sys.argv[1:7]
NB = int(sys.argv[7]) if len(sys.argv) > 7 else 6
DZ = float(os.environ.get("DEMO_DZ", "0.0"))
TOPO_MIN = float(os.environ.get("TOPO_MIN", "0.98"))
os.makedirs(od, exist_ok=True)
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
model = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
data = mujoco.MjData(model)
bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in meta["body_names"]]
ref = np.load(mfile)
T = len(ref["joint_pos"])
a0 = yaw(ref["body_quat_w"][0, 0])
o0 = ref["body_pos_w"][0, 0, :2]
Rs = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ Rs.T  # noqa: E731
H = int(round(float(hold_s) / float(meta["sim"]["control_dt"])))
BXY, PEL = [], []
for f in sorted(glob.glob(os.path.join(tdir, "*.npz"))):
    qp = np.load(f)["qpos"]
    b = []
    for k in range(0, T, 2):
        if H + k >= len(qp):
            break
        data.qpos[:] = qp[H + k][:model.nq]
        mujoco.mj_kinematics(model, data)
        b.append(data.xpos[bid, :2].copy())
    b = np.array(b)
    BXY.append(b), PEL.append(b[:, 0])
J = json.load(open(bj))
P = J["parts"]
pl = P["plank"]
names = sorted((k for k in P if k.startswith("post")), key=lambda k: -(P[k]["center"][2] + P[k]["half_height"]))  # 1 = high
post0 = np.array([P[k]["center"][:2] for k in names])
r = float(P[names[0]]["radius"])
c0 = np.array(pl["center"][:2])


def signature(path, posts):
    """order of closest approach + side (L: post left of the travel direction) for every post."""
    d = np.linalg.norm(path[None] - posts[:, None], axis=-1)  # [4, T]
    k = np.argmin(d, axis=1)
    k0, k1 = np.clip(k - 3, 0, len(path) - 1), np.clip(k + 3, 0, len(path) - 1)
    v = path[k1] - path[k0]
    w = posts - path[k]
    side = "".join("L" if v[i, 0] * w[i, 1] - v[i, 1] * w[i, 0] > 0 else "R" for i in range(len(posts)))
    return "".join(str(i + 1) for i in np.argsort(k)), side


ref_sig = signature(ref["body_pos_w"][:, 0, :2], post0)
TREES = [cKDTree(x.reshape(-1, 2)) for x in BXY]
print(f"reference weaving {ref_sig}, {len(BXY)} runs")


def evaluate(posts):
    keep = np.mean([signature(p, posts) == ref_sig for p in PEL])
    dist = np.stack([tr.query(posts)[0] for tr in TREES])  # [runs, 4]
    cl = np.percentile(dist, 2, axis=0) - r
    return float(keep), float(cl.min()), cl


k0, c0cl, cl0 = evaluate(post0)
print(f"reference pose: weaving kept {k0:.2f}, clearance per post {np.round(cl0, 3)} (min {c0cl:+.3f})")
ax_ref = np.array([np.cos(pl["yaw"]), np.sin(pl["yaw"])])
cands = []
for dyaw in np.arange(-15, 15.01, 1.0):
    th = np.radians(dyaw)
    Rr = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ax = Rr @ ax_ref
    nrm = np.array([-ax[1], ax[0]])
    base = (post0 - c0) @ Rr.T + c0
    for off in np.arange(-0.40, 0.401, 0.02):
        for s in np.arange(-0.40, 0.401, 0.02):
            posts = base + off * nrm + s * ax
            dist = np.stack([tr.query(posts)[0] for tr in TREES])
            cl = float((np.percentile(dist, 2, axis=0) - r).min())
            cands.append(dict(clear=cl, dyaw=float(dyaw), off=float(off), s=float(s), posts=posts.tolist(),
                              yaw=float(pl["yaw"] + th)))
cands.sort(key=lambda c: -c["clear"])
good = []
for c in cands:
    keep, cl, clp = evaluate(np.array(c["posts"]))
    if keep < TOPO_MIN:
        continue
    c["weaving_kept"], c["clear_per_post"] = keep, [float(x) for x in clp]
    c["move"] = float(np.max(np.linalg.norm(np.array(c["posts"]) - post0, axis=1)))
    if all(abs(c["dyaw"] - g["dyaw"]) >= 3 or np.hypot(c["off"] - g["off"], c["s"] - g["s"]) >= 0.08 for g in good):
        good.append(c)
    if len(good) >= NB:
        break
print("| # | min clearance (m) | per post 1-4 | rot (deg) | across / along (m) | max post move (m) | weaving kept | posts (start frame) |")
print("|---|---|---|---|---|---|---|---|")
for i, c in enumerate(good):
    ps = "; ".join(f"({to_s(q)[0]:.2f}, {to_s(q)[1]:.2f})" for q in c["posts"])
    print(f"| {i} | {c['clear']:+.3f} | {' / '.join(f'{x:+.3f}' for x in c['clear_per_post'])} | {c['dyaw']:+.0f} | "
          f"{c['off']:+.2f} / {c['s']:+.2f} | {c['move']:.2f} | {c['weaving_kept']:.2f} | {ps} |")
    j = copy.deepcopy(J)
    Q = j["parts"]
    for k, q in zip(names, c["posts"]):
        hh = (P[k]["center"][2] + P[k]["half_height"] + DZ) / 2
        Q[k]["center"] = [q[0], q[1], hh]
        Q[k]["half_height"] = hh
    th = np.radians(c["dyaw"])
    Rr = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ax = Rr @ ax_ref
    nrm = np.array([-ax[1], ax[0]])
    mid = c0 + c["off"] * nrm + c["s"] * ax  # rotation about the beam centre
    Q["plank"]["center"] = [float(mid[0]), float(mid[1]), pl["center"][2] + DZ]
    Q["plank"]["yaw"], Q["plank"]["pitch"] = c["yaw"], pl["pitch"]
    for k in ("ramp", "leg"):
        if k in Q:
            Q[k]["center"] = [Q[k]["center"][0] + 60.0, Q[k]["center"][1] + 60.0, Q[k]["center"][2]]
    j["demo_layout"] = c
    json.dump(j, open(os.path.join(od, f"cand{i}.json"), "w"), indent=1)
# the reference pose itself (real heights, ramp parked) for comparison
j = copy.deepcopy(J)
Q = j["parts"]
for k in names:
    hh = (P[k]["center"][2] + P[k]["half_height"] + DZ) / 2
    Q[k]["center"][2], Q[k]["half_height"] = hh, hh
Q["plank"]["center"][2] = pl["center"][2] + DZ
for k in ("ramp", "leg"):
    if k in Q:
        Q[k]["center"] = [Q[k]["center"][0] + 60.0, Q[k]["center"][1] + 60.0, Q[k]["center"][2]]
json.dump(j, open(os.path.join(od, "refpose.json"), "w"), indent=1)
json.dump(good, open(os.path.join(od, "cands.json"), "w"), indent=1)
