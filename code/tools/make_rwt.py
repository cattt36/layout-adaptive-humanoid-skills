"""External baseline "retrieve - warp - track" (user decision: one external baseline).
The mainstream alternative to a z-conditioned student is to retrieve a reference motion from the library and give it
to a whole-body motion tracker (H2O / OmniH2O / BeyondMimic).  Tracker = our BeyondMimic-style expert bank (mjlab
tracking defaults + stake / bridge look-ahead; the MoE teacher the students are distilled from), so tracker capability
is identical -- only the interface differs (explicit retrieved reference vs. z + perception student).  The experts
were trained on all trials (incl. test), which can only favour the baseline.
For every candidate-sweep pair (layout n, library candidate c) of tools/make_z_cands_plan.py the reference of n is
replaced by the reference of c, adapted to n with the SAME planner as the students (plan4 rules).  All trials share
the capture (world) frame -- the stakes / bridge are physically fixed, only footholds and start poses differ (start
headings even differ by up to 180 deg, so a rigid start-pose alignment would throw the stakes metres away).  Hence:
  time  : piecewise-linear time warp, anchors = planned stance windows of n <- c's own stance windows (stakes), or
          planned entry frame under the beam <- c's entry frame (beam); length = n's clip length
  start : horizontal offset p0_n - p0_c at frame 0 (the robot starts where trial n starts, as the students do; posture
          and heading of frame 0 are c's), world frame otherwise (no rotation)
  t     : the start offset decays linearly to 0 at the first planned stance / beam entry; c's world path afterwards
  s     : the offset moves on to stake_n - stake_c during each planned stance window (linear in between, held after
          the last stake), so that c's footholds land on n's stakes; beam: plank_n - plank_c (centre) at the entry
Velocities are scaled by the local time rate and get the offset derivative.  Environment clock / success windows = the planned windows of geom_cand_f<k>
(identical to the students' sweep).  Writes data/geom_rwt<var>_f<F>/k<kk>/{ref_stakes_s93_h10/ThreeStepPoleRun,
ref_stakes_s83_h10/FiveStepPoleRun, bridge/CrouchedTraverse}; pairs without a plan link the original files (never scored).
  python make_rwt.py <fold> <t|s>"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
F = int(sys.argv[1])
VAR = sys.argv[2]
assert VAR in ("t", "s")
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10"}
BRIDGE = f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"
GC = f"{H}/data/geom_cand_f{F}"
OUT = f"{H}/data/geom_rwt{VAR}_f{F}"
PJ = json.load(open(f"{H}/data/z_cand_plan_f{F}/pairs.json"))
LIB, K = PJ["library"], PJ["K"]


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def rot2(v, a):
    c, s = np.cos(a), np.sin(a)
    out = v.copy()
    out[..., 0], out[..., 1] = c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]
    return out


def path(name):
    t = name.split("_")[0]
    return BRIDGE if t == "CrouchedTraverse" else os.path.join(GEO[t], t)


def load(name):
    return dict(np.load(os.path.join(path(name), f"{name}_motion.npz")))


def stance(name):
    j = json.load(open(os.path.join(path(name), f"{name}_stakes.json")))
    return np.array([p[:2] for p in j["stake_xyz_50hz"]]), \
        np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])


def plank(name):
    p = json.load(open(os.path.realpath(os.path.join(BRIDGE, f"{name}_bridge.json"))))["parts"]["plank"]
    return np.array(p["center"][:2]), float(p["yaw"])


def adapt(n, c, ad, asrc, offset_knots):
    """reference of candidate c adapted to layout n (world frame): time warp + horizontal offset field."""
    mn, mc = load(n), load(c)
    T = len(mn["body_pos_w"])
    p0n, p0c = mn["body_pos_w"][0, 0, :2], mc["body_pos_w"][0, 0, :2]
    dyaw = 0.0  # no rotation (shared capture frame); kept for the transform helpers
    qy = np.array([1.0, 0, 0, 0], np.float32)
    if not (np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)):
        ad, asrc = [0.0, T - 1.0], [0.0, len(mc["body_pos_w"]) - 1.0]
    tt = np.clip(np.interp(np.arange(T), ad, asrc), 0, len(mc["body_pos_w"]) - 1)
    rate = np.gradient(tt)  # source frames per target frame
    i0 = np.floor(tt).astype(int)
    i1 = np.minimum(i0 + 1, len(mc["body_pos_w"]) - 1)
    w = (tt - i0).astype(np.float32)

    def lerp(x):
        ww = w.reshape((-1,) + (1,) * (x.ndim - 1))
        return (1 - ww) * x[i0] + ww * x[i1]

    out = {"fps": mn["fps"]}
    pos = lerp(mc["body_pos_w"]).copy()
    q0, q1 = mc["body_quat_w"][i0], mc["body_quat_w"][i1]
    q1 = np.where((np.sum(q0 * q1, -1, keepdims=True) < 0), -q1, q1)
    q = (1 - w[:, None, None]) * q0 + w[:, None, None] * q1
    q = qmul(np.broadcast_to(qy, q.shape), q / np.linalg.norm(q, axis=-1, keepdims=True))
    lv = rot2(lerp(mc["body_lin_vel_w"]), dyaw) * rate[:, None, None]
    av = rot2(lerp(mc["body_ang_vel_w"]), dyaw) * rate[:, None, None]
    if offset_knots is not None:
        kt, kv = offset_knots
        off = np.stack([np.interp(np.arange(T), kt, kv[:, d]) for d in range(2)], 1)
        pos[..., :2] += off[:, None, :]
        lv[..., :2] += (np.gradient(off, axis=0) * float(mn["fps"]))[:, None, :]
    out.update(joint_pos=lerp(mc["joint_pos"]).astype(np.float32), joint_vel=(lerp(mc["joint_vel"]) * rate[:, None]).astype(np.float32),
               body_pos_w=pos.astype(np.float32), body_quat_w=q.astype(np.float32), body_lin_vel_w=lv.astype(np.float32),
               body_ang_vel_w=av.astype(np.float32))
    if "foot_contact" in mc:
        out["foot_contact"] = mc["foot_contact"][np.rint(tt).astype(int)]
    return out, (p0n, p0c, dyaw)


def beam(name):
    """(distance start pelvis -> nearer plank end, entry frame under the beam, clip length) = make_z_cands_plan.py"""
    m = np.load(os.path.join(BRIDGE, f"{name}_motion.npz"))
    p = json.load(open(os.path.realpath(os.path.join(BRIDGE, f"{name}_bridge.json"))))["parts"]["plank"]
    c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, nn = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - c[:2]
    inside = (np.abs(rel @ nn) < 0.25) & (np.abs(rel @ d) < half[0])
    entry = int(np.where(inside.any(1))[0][0]) if inside.any() else -1
    p0 = m["body_pos_w"][0, 0, :2]
    return float(min(np.linalg.norm(c[:2] + s * d * half[0] - p0) for s in (1, -1))), entry, len(m["body_pos_w"])


BM = {n: beam(n) for n in PJ["split"] if n.startswith("CrouchedTraverse") and os.path.exists(os.path.join(BRIDGE, f"{n}_motion.npz"))}
_bl = [n for n in LIB["CrouchedTraverse"] if BM[n][1] > 0]
RD = np.polyfit([BM[n][0] for n in _bl], [BM[n][1] for n in _bl], 1)


def start_offset(n, c):
    return load(n)["body_pos_w"][0, 0, :2] - load(c)["body_pos_w"][0, 0, :2]


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


diag = []
for k in range(K):
    kk = f"k{k:02d}"
    pairs = PJ["pairs"].get(kk, {})
    dirs = {"ThreeStepPoleRun": os.path.join(OUT, kk, "ref_stakes_s93_h10", "ThreeStepPoleRun"),
            "FiveStepPoleRun": os.path.join(OUT, kk, "ref_stakes_s83_h10", "FiveStepPoleRun"),
            "CrouchedTraverse": os.path.join(OUT, kk, "bridge", "CrouchedTraverse")}
    for t, d in dirs.items():
        os.makedirs(d, exist_ok=True)
        src_dir = os.path.join(GC, kk, os.path.basename(GEO[t]), t) if t in GEO else BRIDGE
        for f in os.listdir(src_dir):
            n = f.rsplit("_", 1)[0]
            if f.endswith("_motion.npz") and n in pairs:
                continue  # written below
            link(os.path.join(src_dir, f), os.path.join(d, f))
        for n, c in pairs.items():
            if not n.startswith(t):
                continue
            mn_T = len(load(n)["body_pos_w"])
            if t in GEO:
                pd, _ = stance(n)
                pc, wc = stance(c)
                jn = json.load(open(os.path.join(GC, kk, os.path.basename(GEO[t]), t, f"{n}_stakes.json")))
                wn = np.array([(s["stance"][0] * 50.0 / jn["fps"], s["stance"][1] * 50.0 / jn["fps"]) for s in jn["stakes"]])
                ad = [0.0] + wn.flatten().tolist() + [mn_T - 1.0]
                asrc = [0.0] + wc.flatten().tolist() + [len(load(c)["body_pos_w"]) - 1.0]
                d0 = start_offset(n, c)
                delta = pd - pc  # world frame
                if VAR == "s":
                    kt = [0.0] + [x for a, b in wn for x in (a, b)]
                    kv = np.vstack([d0] + [x for dd in delta for x in (dd, dd)])
                else:
                    kt, kv = [0.0, wn[0, 0]], np.vstack([d0, np.zeros(2)])
                ref, _ = adapt(n, c, ad, asrc, (np.array(kt), kv))
                diag.append(dict(k=kk, n=n, c=c, task=t, start_offset_m=float(np.linalg.norm(d0)),
                                 foothold_mismatch_m=float(np.linalg.norm(delta, axis=1).mean())))
            else:
                dn, en, Tn = BM[n]
                dc, ec, Tc = BM[c]
                pred = float(np.clip(np.polyval(RD, dn), 10, Tn - 30))  # = make_z_cands_plan.py
                ad, asrc = [0.0, pred, Tn - 1.0], [0.0, float(ec), Tc - 1.0]
                d0 = start_offset(n, c)
                delta = plank(n)[0] - plank(c)[0]
                ref, _ = adapt(n, c, ad, asrc, (np.array([0.0, pred]), np.vstack([d0, delta if VAR == "s" else np.zeros(2)])))
                diag.append(dict(k=kk, n=n, c=c, task=t, start_offset_m=float(np.linalg.norm(d0)),
                                 foothold_mismatch_m=float(np.linalg.norm(delta))))
            np.savez(os.path.join(d, f"{n}_motion.npz"), **ref)
for t in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"):
    x = [r for r in diag if r["task"] == t]
    print(f"fold {F} variant {VAR} {t}: {len(x)} pairs, start offset {np.mean([r['start_offset_m'] for r in x]):.2f} m, "
          f"world foothold / plank mismatch {np.mean([r['foothold_mismatch_m'] for r in x]):.3f} m")
json.dump(diag, open(os.path.join(OUT, "diag.json"), "w"), indent=1)
