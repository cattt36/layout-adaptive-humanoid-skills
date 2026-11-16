"""DEPLOYMENT ONLY (paper models / data untouched): standard start for the real robot.

The robot stands at the layout origin in a NORMAL stance (feet parallel, knees symmetric, upright), facing the obstacles,
and the reference motion itself brings it into the retrieved skill:

  normal stance (held while the rope is released)  ->  step-in: weight shift + 1-2 small steps onto the footprints of
  the cut frame  ->  blend into the cut-frame pose  ->  slow-in: playback speed rises smoothly 0 -> 1 (the clip
  plays in slow motion first)  ->  the clip from there on as recorded

Cut frame k_c = the first frame where the human stands on both feet facing the obstacles (|heading - obstacle
direction| <= MAX_DH), i.e. the opening turn of the capture (CrouchedTraverse: most trials start with the back to the
bridge) is dropped.  The new layout origin is the stance pelvis, +x = the start heading (= the heading at k_c).

Outputs per clip: <clip>_motion.npz (new timeline; key `src_frame` = source frame (float) of every new frame,
`start_info`), and remapped <clip>_z.npy / <clip>_stakes.json (stance windows through src_frame), <clip>_bridge.json
copied.  For a deployment layout also layout_aligned.json (start = origin, heading 0, obstacles in the new frame).

  python make_start.py clip   <scene.mjb> <policy_meta.json> <in_dir> <clip> <out_dir> [--z <z.npy> ...]
  python make_start.py layout <scene.mjb> <policy_meta.json> <layout_dir> <out_layout_dir>
  python make_start.py render <scene.mjb> <policy_meta.json> <motion.npz> <out.png> [<out.mp4>]
"""
import argparse
import glob
import json
import os
import shutil

import mujoco
import numpy as np

from make_leadin import Robot, quat_yaw, slerp, smooth, yaw_quat

FPS = 50
MAX_DH = 30.0          # deg, heading tolerance of the cut frame (the step-in turns the feet by up to this)
K_MAX = 160            # frames: the cut frame must be within the first 3.2 s
T_SHIFT, T_STEP, T_BLEND, T_SLOW = 0.5, 0.7, 0.6, 2.0  # s: weight shift, one step, blend into the cut pose, slow-in
STEP_H = 0.06          # m, swing-foot clearance of a synthesized step
STANCE_W = None        # half foot distance of the normal stance (None: from the default pose)
ANCHOR_YAW = 25.0      # deg: the anchor foot keeps its cut-frame heading up to this (else it turns with a step)
ARM_DOWN = 0.45        # rad: arms count as "down" (shoulder roll within this of the default)
MAX_ARM = 9.0          # (arm rule of the cut frame off: T-pose arms are lowered by lower_arms())
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def obstacle_direction(motion, stakes_json, bridge_json):
    """angle (rad, world) the robot has to face at the start: stakes -> towards the first stake; bridge -> along the
    beam axis in the direction of travel (sign: towards the beam centre from the clip start)"""
    p0 = motion["body_pos_w"][0, 0, :2]
    if stakes_json and os.path.exists(stakes_json):
        s = json.load(open(stakes_json))["stakes"][0]
        d = np.array([s["x"], s["y"]]) - p0
        return float(np.arctan2(d[1], d[0]))
    pl = json.load(open(bridge_json))["parts"]["plank"]
    ax = np.array([np.cos(pl["yaw"]), np.sin(pl["yaw"])])
    d = np.array(pl["center"][:2]) - p0
    if abs(d @ ax) < 0.2:
        return float(np.arctan2(d[1], d[0]))
    ax = ax * np.sign(d @ ax)
    return float(np.arctan2(ax[1], ax[0]))


def first_event(stakes_json, bridge_json=None, motion=None):
    """first 50 Hz frame of an obstacle event -- the cut must come well before it: stakes -> the first stance window;
    bridge -> the pelvis within 0.8 m (horizontal) of a support post"""
    if stakes_json and os.path.exists(stakes_json):
        j = json.load(open(stakes_json))
        return int(j["stakes"][0]["stance"][0] * FPS / j["fps"])
    if bridge_json and os.path.exists(bridge_json) and motion is not None:
        parts = json.load(open(bridge_json))["parts"]
        posts = np.array([v["center"][:2] for k, v in parts.items() if k.startswith("post")])
        d = np.linalg.norm(motion["body_pos_w"][:, 0, None, :2] - posts[None], axis=2).min(1)
        near = np.nonzero(d < 0.8)[0]
        return int(near[0]) if len(near) else None
    return None


def cut_frame(R, motion, a, ev):
    """first frame (before the obstacle event) on both feet -- contact label, or both ankles within 2.5 cm of the stance
    height -- with the heading within MAX_DH of the obstacle direction"""
    bn = R.meta["body_names"]
    fi = [bn.index(n) for n in FEET]
    Q, fc, P = motion["body_quat_w"], motion["foot_contact"], motion["body_pos_w"]
    up = [i for i, j in enumerate(R.jn) if "shoulder_roll" in j]  # spread arms (S2 frame 0: roll 1.35 rad)
    zf = float(np.percentile(P[:, fi, 2], 5))
    kmax = min(K_MAX, len(Q) - 60, (ev - 20) if ev else K_MAX)
    # arms (shoulder roll) within MAX_ARM of the default; a clip that starts with the arms spread (S2: 1.18 rad at frame 0)
    # is cut only once they are down (<= 0.3 rad)
    arm_dev = np.abs(motion["joint_pos"][:, up] - R.default[up]).max(1)
    arm_max = MAX_ARM
    for k in range(0, max(kmax, 0)):
        both = fc[k].all() or bool((P[k, fi, 2] <= zf + 0.025).all())
        arms = float(np.abs(motion["joint_pos"][k, up] - R.default[up]).max())  # e.g. S2 starts with the arms spread
        if both and abs(np.degrees(wrap(quat_yaw(Q[k, 0]) - a))) <= MAX_DH and arms <= arm_max:
            return k
    return None


def sample(motion, s):
    """reference state at a fractional source frame s (joints / positions linear, quaternions slerp)"""
    T = len(motion["joint_pos"])
    i = int(np.clip(np.floor(s), 0, T - 1))
    j = min(i + 1, T - 1)
    f = float(np.clip(s - i, 0.0, 1.0))
    out = {}
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        out[k] = (1 - f) * motion[k][i] + f * motion[k][j]
    qa, qb = motion["body_quat_w"][i], motion["body_quat_w"][j]
    out["body_quat_w"] = np.array([slerp(qa[b], qb[b], f) for b in range(qa.shape[0])])
    out["foot_contact"] = motion["foot_contact"][i if f < 0.5 else j]
    return out


def lower_arms(R, motion):
    """Many captures start in a T-pose (shoulder roll ~1.5 rad) and only lower the arms when the motion starts: the arm
    joints of those first frames are replaced by the default (arms down), blended back to the recorded arms over 0.5 s
    once they have come down (shoulder roll within ARM_DOWN of the default for 10 frames); arm bodies by forward
    kinematics.  -> (motion, k_down or None)"""
    up = [i for i, j in enumerate(R.jn) if "shoulder_roll" in j]
    arm = [i for i, j in enumerate(R.jn) if any(s in j for s in ("shoulder", "elbow", "wrist"))]
    dev = np.abs(motion["joint_pos"][:, up] - R.default[up]).max(1)
    if dev[0] <= 0.8:
        return motion, None
    low = dev <= ARM_DOWN
    k_down = next((k for k in range(len(dev) - 10) if low[k:k + 10].all()), None)
    if k_down is None:
        return motion, None
    m = {k: np.array(v) for k, v in motion.items()}
    nb = int(0.5 * FPS)
    k_end = min(k_down + nb, len(dev) - 2)  # arms down until k_down, then 0.5 s back to the recorded (already low) arms
    for k in range(k_end):
        s = smooth((k - k_down) / nb) if k > k_down else 0.0
        q = m["joint_pos"][k].astype(float).copy()
        q[arm] = (1 - s) * R.default[arm] + s * q[arm]
        m["joint_pos"][k] = q
        R.set(m["body_pos_w"][k, 0], m["body_quat_w"][k, 0], q)
        bp, bq = R.bodies()
        m["body_pos_w"][k], m["body_quat_w"][k] = bp, bq
    dt = 1.0 / FPS
    m["joint_vel"][:k_end + 1] = np.gradient(m["joint_pos"][:k_end + 2].astype(float), dt, axis=0)[:k_end + 1]
    m["body_lin_vel_w"][:k_end + 1] = np.gradient(m["body_pos_w"][:k_end + 2].astype(float), dt, axis=0)[:k_end + 1]
    bq = m["body_quat_w"]
    for k in range(1, k_end + 1):
        for b in range(bq.shape[1]):
            qe = np.zeros(4)
            mujoco.mju_mulQuat(qe, bq[k, b].astype(float), np.array([bq[k - 1, b, 0], *(-bq[k - 1, b, 1:])], float))
            wv = np.zeros(3)
            mujoco.mju_quat2Vel(wv, qe, dt)
            m["body_ang_vel_w"][k, b] = wv
    return m, k_down


def make_start(R, motion, stakes_json, bridge_json):
    motion, k_down = lower_arms(R, motion)
    bn = R.meta["body_names"]
    fi = [bn.index(n) for n in FEET]
    a = obstacle_direction(motion, stakes_json, bridge_json)
    kc = cut_frame(R, motion, a, first_event(stakes_json, bridge_json, motion))
    if kc is None:
        return None, dict(reason="no frame on both feet facing the obstacles")
    P, Q = motion["body_pos_w"][kc], motion["body_quat_w"][kc]
    hc = quat_yaw(Q[0])
    fwd, lat = np.array([np.cos(hc), np.sin(hc)]), np.array([-np.sin(hc), np.cos(hc)])
    zf = float(np.percentile(motion["body_pos_w"][:, fi, 2], 5))  # stance ankle height
    tgt = {n: (np.array(P[i]), np.array(Q[i])) for n, i in zip(FEET, fi)}
    mid = 0.5 * (P[fi[0], :2] + P[fi[1], :2])
    # normal stance width: the default pose
    if STANCE_W:
        w = STANCE_W
    else:
        R.set(np.array([0, 0, 0.78]), np.array([1.0, 0, 0, 0]), R.default)
        bp, _ = R.bodies()
        w = 0.5 * abs(bp[fi[0], 1] - bp[fi[1], 1])
    # normal stance = one foot ALREADY on its cut-frame footprint (anchor: the one whose heading is closer to the start
    # heading), the other parallel beside it at the normal width -> at most one synthesized step (no shuffle)
    dyaw = {n: abs(np.degrees(wrap(quat_yaw(tgt[n][1]) - hc))) for n in FEET}
    anc = min(FEET, key=lambda n: dyaw[n])
    oth = FEET[1] if anc == FEET[0] else FEET[0]
    sgn = -1.0 if anc == FEET[0] else 1.0  # the right foot lies at -lat of the left one
    anc_q = tgt[anc][1] if dyaw[anc] <= ANCHOR_YAW else yaw_quat(hc)
    anc_p = np.r_[tgt[anc][0][:2], zf]
    st = {anc: (anc_p, anc_q), oth: (np.r_[anc_p[:2] + sgn * 2 * w * lat, zf], yaw_quat(hc))}
    mid = 0.5 * (st[FEET[0]][0][:2] + st[FEET[1]][0][:2])
    pel_z = min(float(P[0, 2]), 0.78)
    pel_q0 = yaw_quat(hc)
    q_up_c = motion["joint_pos"][kc].astype(float)
    # which feet have to move (distance to the cut-frame footprint)
    def need(n):
        dp = np.linalg.norm(st[n][0][:2] - tgt[n][0][:2])
        dy = abs(np.degrees(wrap(quat_yaw(tgt[n][1]) - quat_yaw(st[n][1]))))
        return dp > 0.03 or dy > 5.0, dp
    order = sorted([n for n in FEET if need(n)[0]], key=lambda n: -need(n)[1])
    # ---- key poses -> per-frame targets
    frames = []  # (feet dict, pelvis p, pelvis q, upper-body blend)
    cur = {n: (st[n][0].copy(), st[n][1].copy()) for n in FEET}
    pel = np.r_[mid, pel_z]
    n_shift, n_step, n_blend = int(T_SHIFT * FPS), int(T_STEP * FPS), int(T_BLEND * FPS)
    n_lead = 1 + len(order) * (n_shift + n_step) + n_blend
    t = 0
    frames.append(({n: cur[n] for n in FEET}, pel.copy(), pel_q0, 0.0))  # frame 0: the normal stance (hold)
    for sw in order:
        sup = FEET[1] if sw == FEET[0] else FEET[0]
        p_from = pel.copy()
        p_to = np.r_[cur[sup][0][:2] + 0.25 * (cur[sw][0][:2] - cur[sup][0][:2]), pel_z]  # pelvis over the support foot
        for i in range(n_shift):
            s = smooth((i + 1) / n_shift)
            t += 1
            frames.append(({n: cur[n] for n in FEET}, p_from + s * (p_to - p_from), pel_q0, t / n_lead))
        a0, q0f = cur[sw]
        a1, q1f = tgt[sw]
        for i in range(n_step):
            s = smooth((i + 1) / n_step)
            p = a0 + s * (a1 - a0)
            p = np.r_[p[:2], a0[2] + s * (a1[2] - a0[2]) + STEP_H * np.sin(np.pi * s)]
            t += 1
            f = dict(cur)
            f[sw] = (p, slerp(q0f, q1f, s))
            frames.append((f, p_to, pel_q0, t / n_lead))
        cur[sw] = (a1.copy(), q1f.copy())
        pel = p_to
    for i in range(n_blend):
        s = smooth((i + 1) / n_blend)
        t += 1
        frames.append(({n: cur[n] for n in FEET}, pel + s * (P[0] - pel), slerp(pel_q0, Q[0], s), t / n_lead))
    # ---- IK of the lead-in frames
    up = [i for i in range(len(R.jn)) if i not in R.leg]
    bp, bq, q = np.r_[mid, pel_z], pel_q0, R.default.copy()
    L_q, L_p, L_r, errs = [], [], [], []
    for k, (feet, pp, pq, ub) in enumerate(frames):
        qr = R.default.copy()
        u = smooth(ub)
        qr[up] = (1 - u) * R.default[up] + u * q_up_c[up]
        if k == len(frames) - 1:  # exactly the cut frame
            L_q.append(motion["joint_pos"][kc].astype(float))
            L_p.append(motion["body_pos_w"][kc].astype(float))
            L_r.append(motion["body_quat_w"][kc].astype(float))
            continue
        bp, bq, qq, e = R.ik(bp, bq, q if k else qr, {n: (feet[n][0], feet[n][1], 1.0) for n in FEET}, pp, pq,
                             (1 - ub) * R.default + ub * motion["joint_pos"][kc], iters=200 if k == 0 else 50)
        qq[up] = qr[up]
        q = qq
        R.set(bp, bq, q)
        b_p, b_q = R.bodies()
        L_q.append(q.copy())
        L_p.append(b_p)
        L_r.append(b_q)
        errs.append(e)
    # ---- slow-in (time warp) and the rest of the clip
    n_slow = int(T_SLOW * FPS)
    r = np.array([smooth((i + 1) / n_slow) for i in range(n_slow)])
    tau = np.cumsum(r)  # source frames advanced
    srcs = [float(kc)] * len(frames) + list(kc + tau)
    S_q, S_p, S_r, S_v, S_lv, S_av, S_fc = [], [], [], [], [], [], []
    for i, s in enumerate(kc + tau):
        x = sample(motion, s)
        S_q.append(x["joint_pos"])
        S_p.append(x["body_pos_w"])
        S_r.append(x["body_quat_w"])
        S_v.append(x["joint_vel"] * r[i])
        S_lv.append(x["body_lin_vel_w"] * r[i])
        S_av.append(x["body_ang_vel_w"] * r[i])
        S_fc.append(x["foot_contact"])
    s_end = kc + tau[-1]
    k_rest = int(np.ceil(s_end + 1e-6))
    rest = slice(k_rest, len(motion["joint_pos"]))
    srcs += list(np.arange(k_rest, len(motion["joint_pos"]), dtype=float))
    lead_q, lead_p, lead_r = np.array(L_q), np.array(L_p), np.array(L_r)
    dt = 1.0 / FPS
    lead_v = np.gradient(lead_q, dt, axis=0)
    lead_lv = np.gradient(lead_p, dt, axis=0)
    lead_av = np.zeros_like(lead_lv)
    for i in range(1, len(lead_r)):
        for b in range(lead_r.shape[1]):
            qe = np.zeros(4)
            mujoco.mju_mulQuat(qe, lead_r[i, b], np.array([lead_r[i - 1, b, 0], *(-lead_r[i - 1, b, 1:])]))
            wv = np.zeros(3)
            mujoco.mju_quat2Vel(wv, qe, dt)
            lead_av[i, b] = wv
    lead_v[0], lead_lv[0], lead_av[0] = 0, 0, 0  # the stance is at rest
    lead_v[-1], lead_lv[-1], lead_av[-1] = 0, 0, 0  # slow-in starts from rest
    out = dict(
        fps=np.float64(FPS),
        joint_pos=np.concatenate([lead_q, S_q, motion["joint_pos"][rest]]).astype(np.float32),
        joint_vel=np.concatenate([lead_v, S_v, motion["joint_vel"][rest]]).astype(np.float32),
        body_pos_w=np.concatenate([lead_p, S_p, motion["body_pos_w"][rest]]).astype(np.float32),
        body_quat_w=np.concatenate([lead_r, S_r, motion["body_quat_w"][rest]]).astype(np.float32),
        body_lin_vel_w=np.concatenate([lead_lv, S_lv, motion["body_lin_vel_w"][rest]]).astype(np.float32),
        body_ang_vel_w=np.concatenate([lead_av, S_av, motion["body_ang_vel_w"][rest]]).astype(np.float32),
        foot_contact=np.concatenate([np.ones((len(frames), 2), bool), np.array(S_fc), motion["foot_contact"][rest]]),
        src_frame=np.array(srcs, np.float32))
    info = dict(cut_frame=int(kc), cut_heading_vs_obstacle_deg=round(float(np.degrees(wrap(hc - a))), 1),
                steps=[n.split("_")[0] for n in order], lead_frames=len(frames), slow_frames=n_slow,
                start_pelvis_xy=[float(mid[0]), float(mid[1])], start_heading_rad=float(hc), stance_half_width=float(w),
                max_ik_err=float(max(errs)) if errs else 0.0, frames_total=int(len(out["joint_pos"])),
                source_frames=int(len(motion["joint_pos"])))
    out["start_info"] = np.array(json.dumps(info))
    return out, info


def remap_z(z, src):
    return z[np.clip(np.round(src).astype(int), 0, len(z) - 1)]


def remap_stakes(sj, src, out_f):
    """stance windows / csv frames (capture fps) of a stakes json through the new timeline"""
    j = json.load(open(sj))
    fps = j["fps"]
    new = np.arange(len(src), dtype=float)

    def m(x_csv):  # csv frame -> new csv frame (monotone interpolation; flat lead-in parts map to its end)
        s = x_csv * FPS / fps
        k = float(np.interp(s, src + 1e-6 * new, new, left=0.0))  # src is flat during the lead-in
        return int(round(k * fps / FPS))
    for s in j["stakes"]:
        s["stance"] = [m(v) for v in s["stance"]]
        if "csv_frame" in s:
            s["csv_frame"] = m(s["csv_frame"])
    if "ramp" in j:
        j["ramp"] = [m(v) for v in j["ramp"]]
    j["standard_start"] = "make_start.py (deployment): timeline remapped"
    json.dump(j, open(out_f, "w"), indent=1)


def do_clip(R, in_dir, clip, out_dir, z_files=()):
    mf = os.path.join(in_dir, f"{clip}_motion.npz")
    sj, bj = os.path.join(in_dir, f"{clip}_stakes.json"), os.path.join(in_dir, f"{clip}_bridge.json")
    motion = dict(np.load(mf))
    out, info = make_start(R, motion, sj if os.path.exists(sj) else None, bj if os.path.exists(bj) else None)
    if out is None:
        return None, info
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{clip}_motion.npz"), **out)
    src = out["src_frame"]
    if os.path.exists(sj):
        remap_stakes(sj, src, os.path.join(out_dir, f"{clip}_stakes.json"))
    if os.path.exists(bj):
        shutil.copy(bj, os.path.join(out_dir, f"{clip}_bridge.json"))
    for zf in z_files:
        z = np.load(zf)
        np.save(zf if os.path.dirname(zf) == out_dir else os.path.join(out_dir, os.path.basename(zf)), remap_z(z, src))
    return out, info


def do_layout(R, L, out_L):
    """a deployment layout dir: motion / z / stakes|bridge + layout_aligned.json in the new standard-start frame"""
    la = json.load(open(os.path.join(L, "layout_aligned.json")))
    clip = la["clip"]
    zf = os.path.join(L, f"{clip}_z.npy")
    os.makedirs(out_L, exist_ok=True)
    out, info = do_clip(R, L, clip, out_L, [zf] if os.path.exists(zf) else [])
    if out is None:
        return info
    # new aligned frame: origin = stance pelvis, +x = start heading
    s0 = np.array(info["start_pelvis_xy"])
    h = info["start_heading_rad"]
    Rm = np.array([[np.cos(-h), -np.sin(-h)], [np.sin(-h), np.cos(-h)]])
    to_al = lambda p: Rm @ (np.asarray(p, float)[:2] - s0)  # noqa: E731
    new = dict(la)
    new.update(theta_deg=0.0, start=dict(x=0.0, y=0.0, heading_deg=0.0),
               frame="standard start: origin = robot start (pelvis projection, normal stance), +x = start "
                     "heading (towards the obstacles); clip frame rotated by -start heading about the start")
    bn = R.meta["body_names"]
    P0 = out["body_pos_w"][0]
    new["feet0"] = {"L": to_al(P0[bn.index(FEET[0])]).round(3).tolist(), "R": to_al(P0[bn.index(FEET[1])]).round(3).tolist()}
    if la.get("posts"):
        parts = json.load(open(os.path.join(L, f"{clip}_bridge.json")))["parts"]
        # match the planned posts (old aligned frame) to the bridge parts, then express them in the new frame
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
        from obstacles_live import Layout
        old = Layout(L)
        names = [k for k in parts if k.startswith("post")]
        for p in new["posts"]:
            k = min(names, key=lambda k: np.linalg.norm(old.to_al(parts[k]["center"]) - np.array([p["x"], p["y"]])))
            p["x"], p["y"] = [round(float(v), 3) for v in to_al(parts[k]["center"])]
        pl = parts["plank"]
        xy = to_al(pl["center"])
        new["plank"] = dict(la["plank"], x=round(float(xy[0]), 3), y=round(float(xy[1]), 3),
                            yaw_deg=round(float(np.degrees(wrap(pl["yaw"] - h))), 1))
    if la.get("stakes"):
        sj = json.load(open(os.path.join(out_L, f"{clip}_stakes.json")))
        for p, s in zip(new["stakes"], sorted(sj["stakes"], key=lambda s: s["stance"][0])):
            p["x"], p["y"] = [round(float(v), 3) for v in to_al([s["x"], s["y"]])]
    new["standard_start"] = info
    json.dump(new, open(os.path.join(out_L, "layout_aligned.json"), "w"), indent=1, ensure_ascii=False)
    return info


def render(R, mf, out_png, out_mp4=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import subprocess
    from PIL import Image, ImageDraw
    m = np.load(mf)
    info = json.loads(str(m["start_info"]))
    n_lead, n_slow = info["lead_frames"], info["slow_frames"]
    W, H = 400, 480
    R.m.vis.global_.offwidth, R.m.vis.global_.offheight = W, H
    rnd = mujoco.Renderer(R.m, H, W)
    h0 = info["start_heading_rad"]

    def img(k, lab):
        R.set(m["body_pos_w"][k, 0], m["body_quat_w"][k, 0], m["joint_pos"][k])
        R.d.mocap_pos[:, 2] = -10.0
        mujoco.mj_forward(R.m, R.d)
        cam = mujoco.MjvCamera()
        cam.type, cam.distance, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 2.6, -12.0
        cam.azimuth = np.degrees(h0) + 150.0
        cam.lookat[:] = [m["body_pos_w"][k, 0, 0], m["body_pos_w"][k, 0, 1], 0.55]
        rnd.update_scene(R.d, cam)
        im = Image.fromarray(rnd.render())
        ImageDraw.Draw(im).text((8, 8), lab, fill=(255, 255, 255))
        return im
    keys = [(0, "normal stance (hold)"), (n_lead // 3, "step-in"), (n_lead - 1, "cut frame pose"),
            (n_lead + n_slow // 2, "slow-in"), (min(n_lead + n_slow + 25, len(m["joint_pos"]) - 1), "clip")]
    ims = [img(k, f"{lab}  t={k / FPS:.1f}s") for k, lab in keys]
    canvas = Image.new("RGB", (W * len(ims), H))
    for i, im in enumerate(ims):
        canvas.paste(im, (i * W, 0))
    canvas.save(out_png)
    if out_mp4:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4], stdin=subprocess.PIPE)
        end = min(n_lead + n_slow + 150, len(m["joint_pos"]))
        for k in range(0, end, 2):
            ph = "stance" if k == 0 else ("step-in" if k < n_lead else ("slow-in" if k < n_lead + n_slow else "clip"))
            ff.stdin.write(np.asarray(img(k, f"{ph}  t={k / FPS:.1f}s")).tobytes())
        ff.stdin.close()
        ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clip", "layout", "render"])
    ap.add_argument("scene")
    ap.add_argument("meta")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("c", nargs="?")
    ap.add_argument("d", nargs="?")
    ap.add_argument("--z", nargs="*", default=[])
    x = ap.parse_args()
    R = Robot(x.scene, x.meta)
    if x.cmd == "clip":
        out, info = do_clip(R, x.a, x.b, x.c, x.z)
        print(x.b, json.dumps(info))
    elif x.cmd == "layout":
        print(os.path.basename(x.a.rstrip("/")), json.dumps(do_layout(R, x.a, x.b)))
    else:
        render(R, x.a, x.b, x.c)


if __name__ == "__main__":
    main()
