"""DEMO: where does the blind tracker actually go?  From sim2sim traces (HNU_S2S_TRACE, one npz per trial)
of one tracker on its own clip, in the clip's START FRAME (origin = frame-0 pelvis of the reference, +x = its heading,
i.e. the floor mark the robot stands on, facing the obstacles):
  - pelvis: mean / sd of the deviation from the reference every 0.5 s of the motion
  - foot landings: per reference stance (after the start), robot foot (same side) at mid-stance vs the reference
    foothold: mean offset, 95th-percentile radius
  - bridge: per post, horizontal clearance of the robot (min over bodies of distance - post radius), p5 / median,
    and the same for the reference
Prints a markdown block; also writes <out>.json.
  python prop_spread.py <export_dir> <motion_npz> <trace_dir> <hold_s> <out_prefix> [bridge_json]
"""
import glob
import json
import os
import sys

import mujoco
import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


ex, mfile, tdir, hold_s, out = sys.argv[1:6]
bj = sys.argv[6] if len(sys.argv) > 6 else None
meta = json.load(open(os.path.join(ex, "policy_meta.json")))
model = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
data = mujoco.MjData(model)
bnames = meta["body_names"]
bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in bnames]
feet = [bnames.index("left_ankle_roll_link"), bnames.index("right_ankle_roll_link")]
ref = np.load(mfile)
rp = ref["body_pos_w"].astype(float)
T = len(rp)
fps = float(ref["fps"])
a0 = yaw(ref["body_quat_w"][0, 0])
o0 = rp[0, 0, :2].copy()
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p)[..., :2] - o0) @ R.T  # noqa: E731  world (reference) -> start frame
H = int(round(float(hold_s) / float(meta["sim"]["control_dt"])))
fc = ref["foot_contact"].astype(bool)
# reference stance segments (>= 3 frames, stake stances of the running clips are short) that start after the standing phase (first foot lift)
lift = int(np.argmax(~fc.all(1)))
segs = []
for s in range(2):
    on = fc[:, s]
    k = lift
    while k < T:
        if on[k]:
            e = k
            while e < T and on[e]:
                e += 1
            if e - k >= 3:
                segs.append((s, k, e))
            k = e
        else:
            k += 1
segs.sort(key=lambda x: x[1])
posts = []
if bj:
    parts = json.load(open(bj))["parts"]
    posts = [(n, np.array(p["center"][:2]), float(p["radius"])) for n, p in sorted(parts.items()) if n.startswith("post")]


def clearance(xy_bodies, c, r):
    return float(np.min(np.linalg.norm(xy_bodies - c, axis=-1)) - r)


files = sorted(glob.glob(os.path.join(tdir, "*.npz")))
pel, land, clr, ok = [], [], [], []
for f in files:
    t = np.load(f)
    qp = t["qpos"]
    ok.append(bool(t["success_v2"]) or str(t["cause"]) == "None")
    xs = np.full((T, len(bid), 3), np.nan)
    for k in range(T):
        i = H + k
        if i >= len(qp):
            break
        data.qpos[:] = qp[i][:model.nq] if len(qp[i]) >= model.nq else np.pad(qp[i], (0, model.nq - len(qp[i])))
        mujoco.mj_kinematics(model, data)
        xs[k] = data.xpos[bid]
    pel.append(xs[:, 0, :2])
    land.append([xs[(k + e) // 2, feet[s], :2] for s, k, e in segs])
    clr.append([np.nanmin([clearance(xs[k, :, :2], c, r) for k in range(T) if not np.isnan(xs[k, 0, 0])]) for _, c, r in posts])
pel = np.array(pel)
res = dict(trials=len(files), success=float(np.mean(ok)) if ok else None)
lines = [f"**{os.path.basename(out)}** ({len(files)} trials, success {100 * np.mean(ok):.0f} %; start frame, m)\n"]
lines.append("| t (s) | ref x | ref y | mean dx | mean dy | sd x | sd y |\n|---|---|---|---|---|---|---|")
rows = []
for k in range(0, T, int(fps / 2)):
    rs = to_s(rp[k, 0])
    ps = to_s(pel[:, k])
    good = ~np.isnan(ps[:, 0])
    if good.sum() < 3:
        continue
    dv = ps[good] - rs
    rows.append(dict(t=k / fps, ref=rs.tolist(), mean=dv.mean(0).tolist(), sd=dv.std(0).tolist()))
    lines.append(f"| {k / fps:.1f} | {rs[0]:.2f} | {rs[1]:.2f} | {dv[:, 0].mean():+.3f} | {dv[:, 1].mean():+.3f} | "
                 f"{dv[:, 0].std():.3f} | {dv[:, 1].std():.3f} |")
res["pelvis"] = rows
lines.append("\n| landing | foot | t (s) | ref x | ref y | mean dx | mean dy | r95 |\n|---|---|---|---|---|---|---|---|")
lr = []
for j, (s, k, e) in enumerate(segs):
    rs = to_s(rp[(k + e) // 2, feet[s]])
    ps = to_s(np.array([l[j] for l in land]))
    good = ~np.isnan(ps[:, 0])
    if good.sum() < 3:
        continue
    dv = ps[good] - rs
    r95 = float(np.percentile(np.linalg.norm(dv - dv.mean(0), axis=1), 95))
    lr.append(dict(foot="LR"[s], t=(k + e) / 2 / fps, ref=rs.tolist(), mean=dv.mean(0).tolist(), r95=r95))
    lines.append(f"| {j + 1} | {'LR'[s]} | {(k + e) / 2 / fps:.2f} | {rs[0]:.2f} | {rs[1]:.2f} | {dv[:, 0].mean():+.3f} | "
                 f"{dv[:, 1].mean():+.3f} | {r95:.3f} |")
res["landings"] = lr
if posts:
    refc = [min(clearance(rp[k, :, :2], c, r) for k in range(T)) for _, c, r in posts]
    C = np.array(clr)
    lines.append("\n| post | x | y | ref clearance | robot p5 | robot median |\n|---|---|---|---|---|---|")
    res["posts"] = []
    for i, (n, c, r) in enumerate(posts):
        cs = to_s(c)
        res["posts"].append(dict(post=n, xy=cs.tolist(), ref=refc[i], p5=float(np.percentile(C[:, i], 5)), med=float(np.median(C[:, i]))))
        lines.append(f"| {n} | {cs[0]:.2f} | {cs[1]:.2f} | {refc[i]:+.3f} | {np.percentile(C[:, i], 5):+.3f} | {np.median(C[:, i]):+.3f} |")
if posts:
    # suggested post positions (props must not be touched): slide each post ALONG the beam axis (the beam still rests
    # on the post tops) to the nearest position whose clearance (min over bodies and frames, 2nd percentile over runs)
    # is >= CLR; if none within +-0.8 m, the nearest 2-D position with that clearance
    CLR = 0.25
    pl = json.load(open(bj))["parts"].get("plank")
    ax = np.array([np.cos(pl["yaw"]), np.sin(pl["yaw"])]) if pl else np.array([1.0, 0.0])
    XY = []  # [trials, frames, bodies, 2]
    for f in files:
        t = np.load(f)
        qp = t["qpos"]
        fr = []
        for k in range(0, T, 2):
            i = H + k
            if i >= len(qp):
                break
            data.qpos[:] = qp[i][:model.nq] if len(qp[i]) >= model.nq else np.pad(qp[i], (0, model.nq - len(qp[i])))
            mujoco.mj_kinematics(model, data)
            fr.append(data.xpos[bid, :2].copy())
        XY.append(np.array(fr).reshape(-1, len(bid), 2))

    def clr_at(c, r):
        per = [np.min(np.linalg.norm(x - c, axis=-1)) - r for x in XY if len(x)]
        return float(np.percentile(per, 2))

    res["post_suggest"] = []
    lines.append(f"\n建议立柱位置（{len(files)} 次中 98% 的余量 >= {CLR:.2f} m；身体连杆原点，实际身体表面再近约 0.10-0.15 m）\n"
                 "| post | ref x | ref y | 建议 x | 建议 y | 沿梁挪动 (m) | 余量 (m) |\n|---|---|---|---|---|---|---|")
    for n, c, r in posts:
        best = None
        for s in sorted(np.arange(-0.8, 0.801, 0.05), key=abs):
            cc = c + s * ax
            if clr_at(cc, r) >= CLR:
                best = (cc, s)
                break
        if best is None:
            cand = sorted(((dx, dy) for dx in np.arange(-0.8, 0.81, 0.05) for dy in np.arange(-0.8, 0.81, 0.05)),
                          key=lambda v: np.hypot(*v))
            for dx, dy in cand:
                cc = c + np.array([dx, dy])
                if clr_at(cc, r) >= CLR:
                    best = (cc, None)
                    break
        if best is None:
            lines.append(f"| {n} | {to_s(c)[0]:.2f} | {to_s(c)[1]:.2f} | - | - | - | - |")
            continue
        cs, rs = to_s(best[0]), to_s(c)
        res["post_suggest"].append(dict(post=n, ref=rs.tolist(), xy=cs.tolist(), slide=best[1], clr=clr_at(best[0], r)))
        lines.append(f"| {n} | {rs[0]:.2f} | {rs[1]:.2f} | {cs[0]:.2f} | {cs[1]:.2f} | "
                     f"{'%+.2f' % best[1] if best[1] is not None else '离开梁线'} | {clr_at(best[0], r):.2f} |")
print("\n".join(lines))
json.dump(res, open(out + ".json", "w"), indent=1)
# arrays for the placement figure (tools/prop_figure.py), all in the start frame
np.savez_compressed(out + "_xy.npz", ref_pelvis=to_s(rp[:, 0]), robot_pelvis=to_s(pel),
                    ref_feet=np.array([to_s(rp[(k + e) // 2, feet[s]]) for s, k, e in segs]).reshape(-1, 2),
                    feet_side=np.array([s for s, _, _ in segs]),
                    robot_feet=np.array([[to_s(l[j]) for j in range(len(segs))] for l in land]).reshape(len(land), len(segs), 2),
                    posts=np.array([to_s(c) for _, c, _ in posts]).reshape(-1, 2),
                    post_r=np.array([r for _, _, r in posts]), ok=np.array(ok))
