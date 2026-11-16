"""CrouchedTraverse failure analysis (runs/dumu_fail) + reference / bridge features of every trial.
Per trial (all 27, from the reference): bridge scale, beam underside height, reference pelvis height and highest body
point when crossing the beam line, clearance, distance of the crossing point to the nearest post, crossing angle,
approach speed, crouch duration.  Per failing rollout: time relative to the reference beam crossing, cause (pelvis
low / tilt), robot pelvis offset from the beam line and nearest post at failure, first bridge contact time."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
D = f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"
O = f"{H}/runs/dumu_fail"
split = json.load(open(f"{H}/data/clip_split_trial/split.json"))
where = {n: k for k in ("train", "val", "test") for n in split[k]}
man = json.load(open(f"{H}/runs/latent_v5/seed42/Z0/split_manifest.json"))
c2s = {f"{a}_{int(t):03d}": s for a, g in man["subject_trial_groups"].items() for s, ts in g.items() for t in ts}


def geom(clip):
    b = json.load(open(f"{D}/{clip}_bridge.json"))
    p = b["parts"]["plank"]
    c = np.array(p["center"])
    yaw = p["yaw"]
    d = np.array([np.cos(yaw), np.sin(yaw)])
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    under = c[2] - p["half"][2]
    posts = np.array([v["center"][:2] for k, v in b["parts"].items() if k.startswith("post")])
    return b, c, d, n, under, posts


def feats(clip):
    m = np.load(os.path.realpath(f"{D}/{clip}_motion.npz"))
    b, c, d, n, under, posts = geom(clip)
    pel = m["body_pos_w"][:, 0]
    sd = (pel[:, :2] - c[:2]) @ n
    cross = np.where(np.sign(sd[:-1]) != np.sign(sd[1:]))[0]
    k = int(cross[0]) if len(cross) else int(np.argmin(np.abs(sd)))
    top = m["body_pos_w"][k, :, 2].max()
    v = m["body_lin_vel_w"][max(0, k - 50):k - 10, 0, :2]
    vel = np.linalg.norm(v, axis=-1).mean()
    dirv = v.mean(0) / (np.linalg.norm(v.mean(0)) + 1e-9)
    ang = np.degrees(np.arccos(min(1.0, abs(dirv @ n))))
    h0 = np.median(pel[:30, 2])
    crouch = (pel[:, 2] < h0 - 0.15).sum() / 50
    pdist = np.linalg.norm(posts - pel[k, :2], axis=-1).min()
    return dict(k=k, scale=b.get("scale", np.nan), under=under, pel_h=pel[k, 2], top=top, clear=under - top, post=pdist,
                ang=ang, v=vel, crouch=crouch, T=len(pel), c=c, n=n, posts=posts)


clips = sorted(os.path.basename(f).replace("_motion.npz", "") for f in glob.glob(f"{D}/*_motion.npz"))
F = {c: feats(c) for c in clips}
print("trial            split subj      scale under  pel_h  top   clear  post_d angle  v_app crouch_s")
for c in clips:
    f = F[c]
    print(f"{c:16s} {where.get(c, '?'):5s} {c2s.get(c, '?'):9s} {f['scale']:.2f}  {f['under']:.2f}  {f['pel_h']:.2f}  "
          f"{f['top']:.2f}  {f['clear']:+.2f}  {f['post']:.2f}   {f['ang']:4.1f}  {f['v']:.2f}  {f['crouch']:.2f}")
tr = [c for c in clips if where.get(c) == "train"]
for k in ("scale", "under", "pel_h", "clear", "post", "ang", "v", "crouch"):
    a = np.array([F[c][k] for c in tr])
    print(f"train {k:6s} mean {a.mean():.3f} sd {a.std():.3f} min {a.min():.3f} max {a.max():.3f}")

MODELS = [("teacher", "eval_teacher_teacher.json"), ("old L0", "eval_student_t_Z0.json"), ("old L4", "eval_student_t_Z4.json"),
          ("new L0", "eval_student_j_Z0.json"), ("new L4", "eval_student_j_Z4.json")]
data = {n: {c["clip"]: c for c in json.load(open(f"{O}/{f}"))["clips"]} for n, f in MODELS if os.path.exists(f"{O}/{f}")}
cos85 = np.cos(np.radians(85))
for clip in sorted(next(iter(data.values()))):
    f = F[clip]
    print(f"\n=== {clip} ({where.get(clip)}, {c2s.get(clip)}): reference crosses the beam at {f['k'] / 50:.2f} s of {f['T'] / 50:.2f} s")
    for name in data:
        d = data[name][clip]["detail"]
        ok = np.array(d["success"])
        ff = np.array(d["fail_frame"])
        st = np.array(d["fail_state"])
        tf = np.array(d["touch_frame"])
        fails = [(ff[i], st[i]) for i in range(len(ok)) if not ok[i] and ff[i] >= 0]
        if fails:
            dt = np.array([(fr - f["k"]) / 50 for fr, _ in fails])
            low = sum(s[2] < 0.30 for _, s in fails)
            tilt = sum(s[3] > -cos85 for _, s in fails)
            off = np.array([(s[:2] - f["c"][:2]) @ f["n"] for _, s in fails])
            pd = np.array([np.linalg.norm(f["posts"] - s[:2], axis=-1).min() for _, s in fails])
            desc = (f"fail t-cross median {np.median(dt):+.2f} s [{dt.min():+.2f}, {dt.max():+.2f}], low {low} tilt {tilt}, "
                    f"pelvis-beam offset median {np.median(off):+.2f} m, nearest post {np.median(pd):.2f} m")
        else:
            desc = ""
        touch = tf[tf >= 0]
        tdesc = f"touch {100 * len(touch) / len(tf):.0f}% (first at t-cross {np.median((touch - f['k']) / 50):+.2f} s)" if len(touch) else "touch 0%"
        print(f"  {name:8s} succ {100 * ok.mean():3.0f}%  {tdesc}  {desc}")
