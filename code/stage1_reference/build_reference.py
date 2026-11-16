"""Stage 1: physically consistent G1 references from the existing retargeted NPZ.

For every trial:
  1. contact labels   : force-plate phases (ground truth) + kinematic detection off the plates
  2. contact-locked IK: stance sole flat on the ground at a fixed xy/yaw, swing sole above ground,
                        joint limits with margin; corrections are smoothed in time
  3. flight phases    : whole-body COM made exactly ballistic (g = 9.81)
  4. inverse dynamics : sole-corner contact forces (friction pyramid, unilateral) solved by QP so
                        that joint torques include the ground reaction; the unexplained base wrench
                        is reported as dynamic infeasibility
  5. outputs          : <trial>_Dynamics.npz (old keys, recomputed + new keys), <trial>_motion.npz
                        (mjlab tracking format, 50 Hz) and <trial>_report.json

All kinematics/dynamics use the mjlab G1 MJCF, i.e. the model used for training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import quadprog
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, get_spec

G = 9.81
SOLE_Z = -0.035  # sole plane in ankle_roll_link frame (capsules at z=-0.025, radius 0.01)
SOLE_CORNERS = np.array(
    [[0.13, 0.028, SOLE_Z], [0.13, -0.028, SOLE_Z], [-0.055, 0.022, SOLE_Z], [-0.055, -0.022, SOLE_Z]]
)
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
TOE_LOCAL = np.array([0.13, 0.0, SOLE_Z])
HEEL_LOCAL = np.array([-0.055, 0.0, SOLE_Z])
ROLL_DZ = 0.02  # m height difference toe vs heel that counts as a toe/heel roll
MU = 0.6
OUT_FPS = 50.0
# trials whose source retarget is broken (checked )
EXCLUDE = {"CrouchedTraverse_033": "source retarget root jumps (root speed 71 m/s, torque 96x)"}
SMOOTH_IN_SIGMA = 2.0  # frames @120 Hz on the raw retarget (~11 Hz cutoff)
SMOOTH_CORR_SIGMA = 3.0  # frames @120 Hz on the IK correction


def smooth_traj(Q, sigma):
    Q = Q.copy()
    Q[:, :3] = gaussian_filter1d(Q[:, :3], sigma, axis=0, mode="nearest")
    Q[:, 7:] = gaussian_filter1d(Q[:, 7:], sigma, axis=0, mode="nearest")
    quat = Q[:, 3:7].copy()
    for t in range(1, len(quat)):
        if np.dot(quat[t], quat[t - 1]) < 0:
            quat[t] *= -1
    quat = gaussian_filter1d(quat, sigma, axis=0, mode="nearest")
    Q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return Q


# ----------------------------------------------------------------------------- model

@dataclass
class Robot:
    m: mujoco.MjModel
    d: mujoco.MjData
    joint_names: list
    lo: np.ndarray
    hi: np.ndarray
    effort: np.ndarray
    vel_limit: np.ndarray
    foot_ids: tuple
    mass: float


def load_robot() -> Robot:
    spec = get_spec()
    m = spec.compile()
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
    lo, hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
    effort = np.zeros(len(names))
    vel = np.zeros(len(names))
    import re
    vmap = {25.0: 37.0, 88.0: 32.0, 139.0: 20.0, 5.0: 22.0, 50.0: 37.0}
    for a in G1_ARTICULATION.actuators:
        for expr in a.target_names_expr:
            for j, n in enumerate(names):
                if re.fullmatch(expr, n):
                    effort[j] = a.effort_limit
                    vel[j] = vmap[a.effort_limit]
    assert (effort > 0).all(), "missing effort limits"
    foot_ids = tuple(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES)
    return Robot(m, d, names, lo, hi, effort, vel, foot_ids, float(m.body_subtreemass[1]))


_INV = {}


def inverse_model(r: Robot):
    """Copy of the model with constraints/passive forces disabled: qfrc_inverse = M qacc + bias."""
    if "m" not in _INV:
        mi = get_spec().compile()
        for flag in ("mjDSBL_CONSTRAINT", "mjDSBL_PASSIVE", "mjDSBL_SPRING", "mjDSBL_DAMPER"):
            if hasattr(mujoco.mjtDisableBit, flag):
                mi.opt.disableflags |= int(getattr(mujoco.mjtDisableBit, flag))
        _INV["m"], _INV["d"] = mi, mujoco.MjData(mi)
    return _INV["m"], _INV["d"]


def fk(r: Robot, q):
    r.d.qpos[:] = q
    mujoco.mj_kinematics(r.m, r.d)
    mujoco.mj_comPos(r.m, r.d)


def sole_points(r: Robot, fid, d=None):
    d = r.d if d is None else d
    R = d.xmat[fid].reshape(3, 3)
    return d.xpos[fid] + SOLE_CORNERS @ R.T


# ----------------------------------------------------------------------------- contacts

def read_plate_contacts(phase_csv, fps, T):
    """Return bool (T,2) contact from force-plate phase files, and list of (foot, s, e)."""
    c = np.zeros((T, 2), bool)
    events = []
    if not phase_csv or not os.path.exists(phase_csv):
        return c, events
    with open(phase_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "PhaseEN" in row:  # ThreeStepPoleRun/FiveStepPoleRun action cycles
            if row["PhaseEN"] not in ("LeftFootContact", "RightFootContact"):
                continue
            foot = 0 if row["PhaseEN"].startswith("Left") else 1
            t0, t1 = float(row["StartTime_s"]), float(row["EndTime_s"])
        else:  # CrouchedTraverse steps
            foot = 0 if row["Foot"] == "L" else 1
            t0, t1 = float(row["IC_Time_s"]), float(row["TO_Time_s"])
        s, e = int(round(t0 * fps)), int(round(t1 * fps))
        s, e = max(s, 0), min(e, T - 1)
        if e > s:
            c[s : e + 1, foot] = True
            events.append((foot, s, e))
    return c, events


def detect_kinematic_contacts(r: Robot, Q, fps):
    T = len(Q)
    low = np.zeros((T, 2))
    pos = np.zeros((T, 2, 3))
    for t in range(T):
        fk(r, Q[t])
        for k, fid in enumerate(r.foot_ids):
            low[t, k] = sole_points(r, fid)[:, 2].min()
            pos[t, k] = r.d.xpos[fid]
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=-1) * fps
    return low, speed


def clean_runs(mask, min_len):
    out = mask.copy()
    for k in range(mask.shape[1]):
        m = mask[:, k]
        t = 0
        while t < len(m):
            if m[t]:
                s = t
                while t < len(m) and m[t]:
                    t += 1
                if t - s < min_len:
                    out[s:t, k] = False
            else:
                t += 1
    return out


def intervals(mask):
    out = []
    t = 0
    while t < len(mask):
        if mask[t]:
            s = t
            while t < len(mask) and mask[t]:
                t += 1
            out.append((s, t - 1))
        else:
            t += 1
    return out


# ----------------------------------------------------------------------------- IK

def yaw_of(R):
    return np.arctan2(R[1, 0], R[0, 0])


def rotz(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_err(R_target, R):
    """Orientation error (world frame) such that R_target ~= exp(e) R."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_target @ R.T).flatten())
    e = np.zeros(3)
    mujoco.mju_quat2Vel(e, q, 1.0)
    return e


def ik_frame(r: Robot, q_ref, q_init, foot_targets, iters=30):
    """Damped least squares on nv; foot_targets[k] = ('pose', p, R, w) | ('lock', p, R) | ('above', 0)."""
    m, d = r.m, r.d
    nv = m.nv
    q = q_init.copy()
    jp = np.zeros((3, nv))
    jr = np.zeros((3, nv))
    # posture weights: root pos/ori, legs, waist, arms
    w_post = np.full(nv, 0.05)
    w_post[0:3] = 1.0
    w_post[3:6] = 1.0
    w_post[6 + 12 : 6 + 15] = 0.2  # waist
    w_post[6 + 15 :] = 2.0  # arms stay as retargeted
    for _ in range(iters):
        fk(r, q)
        rows, errs = [], []
        for k, fid in enumerate(r.foot_ids):
            tgt = foot_targets[k]
            if tgt is None:
                continue
            if tgt[0] == "point":  # ('point', local_point, world_target, R_soft, w)
                lp, p_t, R_t, w = tgt[1], tgt[2], tgt[3], tgt[4]
                R = d.xmat[fid].reshape(3, 3)
                pw = d.xpos[fid] + R @ lp
                mujoco.mj_jac(m, d, jp, jr, pw, fid)
                rows += [w * jp, 0.25 * jr]
                errs += [w * (p_t - pw), 0.25 * rot_err(R_t, R)]
                pts = sole_points(r, fid)
                i = int(np.argmin(pts[:, 2]))
                if pts[i, 2] < 0.0:
                    mujoco.mj_jac(m, d, jp, jr, pts[i], fid)
                    rows.append(w * jp[2:3])
                    errs.append(w * np.array([-pts[i, 2]]))
            elif tgt[0] in ("lock", "pose"):
                p_t, R_t = tgt[1], tgt[2]
                w = tgt[3] if tgt[0] == "pose" else 4.0
                mujoco.mj_jacBody(m, d, jp, jr, fid)
                rows += [w * jp, 0.5 * w * jr]
                errs += [w * (p_t - d.xpos[fid]), 0.5 * w * rot_err(R_t, d.xmat[fid].reshape(3, 3))]
            else:  # keep sole above ground
                pts = sole_points(r, fid)
                i = int(np.argmin(pts[:, 2]))
                if pts[i, 2] < 0.0:
                    mujoco.mj_jac(m, d, jp, jr, pts[i], fid)
                    rows.append(4.0 * jp[2:3])
                    errs.append(4.0 * np.array([-pts[i, 2]]))
        # posture: difference to reference
        dq_ref = np.zeros(nv)
        mujoco.mj_differentiatePos(m, dq_ref, 1.0, q, q_ref)
        A = np.vstack(rows + [np.diag(w_post)]) if rows else np.diag(w_post)
        b = np.concatenate(errs + [w_post * dq_ref]) if errs else w_post * dq_ref
        dq = np.linalg.solve(A.T @ A + 1e-4 * np.eye(nv), A.T @ b)
        mujoco.mj_integratePos(m, q, dq, 1.0)
        q[7:] = np.clip(q[7:], r.lo + 0.01, r.hi - 0.01)
        if np.linalg.norm(dq) < 1e-6:
            break
    return q


# ----------------------------------------------------------------------------- dynamics

def qvel_qacc(r: Robot, Q, fps):
    T, nv = len(Q), r.m.nv
    V = np.zeros((T, nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(r.m, V[t], 1.0 / fps, Q[t], Q[t + 1])
    V[-1] = V[-2]
    # central-ish: average forward differences, then smooth
    V[1:] = 0.5 * (V[1:] + V[:-1])
    V = savgol_filter(V, 9, 3, axis=0)
    A = savgol_filter(V, 9, 3, deriv=1, delta=1.0 / fps, axis=0)
    return V, A


CORNER_SETS = {1: [0, 1, 2, 3], 2: [0, 1], 3: [2, 3]}


def solve_contact_forces(r: Robot, q, v, a, contact, phase=None):
    """Return per-foot world force (2,3), joint torques (29,), base residual (6,), qfrc_inverse (35,)."""
    m, d = inverse_model(r)
    d.qpos[:] = q
    d.qvel[:] = v
    d.qacc[:] = a
    mujoco.mj_inverse(m, d)
    tau_full = d.qfrc_inverse.copy()
    Js, owners = [], []
    jp = np.zeros((3, m.nv))
    for k, fid in enumerate(r.foot_ids):
        if not contact[k]:
            continue
        corners = CORNER_SETS.get(int(phase[k]) if phase is not None else 1, CORNER_SETS[1])
        for p in sole_points(r, fid, d)[corners]:
            mujoco.mj_jac(m, d, jp, None, p, fid)
            Js.append(jp.copy())
            owners.append(k)
    foot_f = np.zeros((2, 3))
    if not Js:
        return foot_f, tau_full[6:].copy(), tau_full[:6].copy(), tau_full
    n = len(Js)
    Jb = np.hstack([J[:, :6].T for J in Js])  # 6 x 3n  (J^T f projected on base)
    # objective: ||Jb f - tau_base||^2 + eps ||f||^2
    H = Jb.T @ Jb + 1e-6 * np.eye(3 * n)
    g = Jb.T @ tau_full[:6]
    # constraints C^T f >= 0 : fz>=0, mu fz +- fx >=0, mu fz +- fy >=0
    C = []
    for i in range(n):
        for row in ([0, 0, 1], [1, 0, MU], [-1, 0, MU], [0, 1, MU], [0, -1, MU]):
            c = np.zeros(3 * n)
            c[3 * i : 3 * i + 3] = row
            C.append(c)
    C = np.array(C).T
    f = quadprog.solve_qp(H, g, C, np.zeros(C.shape[1]))[0]
    gen = np.zeros(m.nv)
    for i, J in enumerate(Js):
        fi = f[3 * i : 3 * i + 3]
        gen += J.T @ fi
        foot_f[owners[i]] += fi
    tau_joint = tau_full[6:] - gen[6:]
    residual = tau_full[:6] - gen[:6]
    return foot_f, tau_joint, residual, tau_full


def self_collisions(r: Robot, q):
    m, d = r.m, r.d
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    worst, n = 0.0, 0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < -0.005:
            n += 1
            worst = min(worst, c.dist)
    return n, -worst


# ----------------------------------------------------------------------------- resampling / mjlab export

def resample(r: Robot, Q, fps_in, fps_out):
    T = len(Q)
    dur = (T - 1) / fps_in
    ts = np.arange(0.0, dur + 1e-9, 1.0 / fps_out)
    out = np.zeros((len(ts), Q.shape[1]))
    for i, t in enumerate(ts):
        x = t * fps_in
        i0 = min(int(np.floor(x)), T - 1)
        i1 = min(i0 + 1, T - 1)
        w = x - i0
        out[i, :3] = (1 - w) * Q[i0, :3] + w * Q[i1, :3]
        qq = np.zeros(4)
        a, b = Q[i0, 3:7].copy(), Q[i1, 3:7].copy()
        if np.dot(a, b) < 0:
            b = -b
        qq = (1 - w) * a + w * b
        out[i, 3:7] = qq / np.linalg.norm(qq)
        out[i, 7:] = (1 - w) * Q[i0, 7:] + w * Q[i1, 7:]
    return out


def export_mjlab_motion(r: Robot, Q50, fps):
    T = len(Q50)
    nb = r.m.nbody - 1
    pos = np.zeros((T, nb, 3))
    quat = np.zeros((T, nb, 4))
    for t in range(T):
        fk(r, Q50[t])
        pos[t] = r.d.xpos[1:]
        quat[t] = r.d.xquat[1:]
    for t in range(1, T):  # hemisphere continuity
        flip = (quat[t] * quat[t - 1]).sum(-1) < 0
        quat[t, flip] *= -1
    lin = np.gradient(pos, 1.0 / fps, axis=0)
    ang = np.zeros_like(lin)
    for t in range(T):
        t0, t1 = max(t - 1, 0), min(t + 1, T - 1)
        for b in range(nb):
            qd = np.zeros(3)
            mujoco.mju_subQuat(qd, quat[t1, b], quat[t0, b])  # body-frame
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat[t, b])
            ang[t, b] = R.reshape(3, 3) @ qd / ((t1 - t0) / fps)
    V = np.gradient(Q50[:, 7:], 1.0 / fps, axis=0)
    return dict(
        fps=np.array(fps),
        joint_pos=Q50[:, 7:].astype(np.float32),
        joint_vel=V.astype(np.float32),
        body_pos_w=pos.astype(np.float32),
        body_quat_w=quat.astype(np.float32),
        body_lin_vel_w=lin.astype(np.float32),
        body_ang_vel_w=ang.astype(np.float32),
    )


# ----------------------------------------------------------------------------- main per trial

def process_trial(r: Robot, npz_path, phase_csv, out_dir, time_scale=1.0, mode="light"):
    no_ik = mode != "full"
    name = os.path.basename(npz_path).replace("_Dynamics.npz", "")
    src = np.load(npz_path, allow_pickle=True)
    Q0 = src["qpos"].astype(np.float64)
    fps = float(src["fps"])
    assert list(src["actuator_joint_names"]) == r.joint_names, "joint order mismatch"
    T = len(Q0)

    # --- ground offset & contacts
    low, speed = detect_kinematic_contacts(r, Q0, fps)
    plate, plate_events = read_plate_contacts(phase_csv, fps, T)
    kin = (low < 0.03) & (speed < 0.35)
    kin = clean_runs(kin, 4)
    ground = np.median(low[kin]) if kin.any() else 0.0
    Q0[:, 2] -= ground
    low -= ground
    # force-plate clock vs retarget clock: find the shift that best matches plate stances with the
    # kinematically "planted" foot (low and slow), then shift all plate events by it
    planted = (low < 0.04) & (speed < 0.6)
    # foot labels of the plate files are not always right (e.g. swapped in ThreeStepPoleRun): take the foot
    # that is planted during the event window
    n_swapped = 0
    fixed = []
    for foot, s, e in plate_events:
        m_own, m_other = planted[s : e + 1, foot].mean(), planted[s : e + 1, 1 - foot].mean()
        if m_other > m_own + 0.2:
            foot, n_swapped = 1 - foot, n_swapped + 1
        fixed.append((foot, s, e))
    plate_events = fixed
    best_off, best_score = 0, -1.0
    for off in range(-int(0.15 * fps), int(0.15 * fps) + 1):
        sc = 0.0
        for foot, s, e in plate_events:
            a, b = s + off, e + off
            if a < 0 or b >= T:
                sc -= 1.0
                continue
            sc += planted[a : b + 1, foot].mean()
        if sc > best_score + 1e-9 or (abs(sc - best_score) < 1e-9 and abs(off) < abs(best_off)):
            best_off, best_score = off, sc
    plate_offset_s = best_off / fps
    plate_match = best_score / max(len(plate_events), 1)
    plate_events = [(f, s + best_off, e + best_off) for f, s, e in plate_events
                    if 0 <= s + best_off and e + best_off < T]
    plate = np.zeros_like(plate)
    for f, s, e in plate_events:
        plate[s : e + 1, f] = True
    contact = kin.copy()
    plate_any = plate.any(1)
    # within plate-covered windows (plus margin) trust plates; outside use kinematics
    cover = np.zeros(T, bool)
    for _, s, e in plate_events:
        cover[max(s - 6, 0) : min(e + 7, T)] = True
    contact[cover] = plate[cover]
    contact = clean_runs(contact, 4)

    # --- crop to the task: from the last quiet moment before the first plate step (the "ready"
    # pause after walking to the start line) to when the subject has decelerated after the last
    # plate step (before walking back / turning around).
    root_speed = gaussian_filter1d(np.linalg.norm(np.gradient(Q0[:, :2], axis=0), axis=1) * fps, 4)
    first_plate = min((s for _, s, _ in plate_events), default=0)
    last_plate = max((e for _, _, e in plate_events), default=T - 1)
    quiet = root_speed < 0.3
    s0 = 0
    for t in range(first_plate, 0, -1):
        if quiet[max(t - int(0.2 * fps), 0) : t].all():
            s0 = max(t - int(0.4 * fps), 0)
            break
    s1 = min(last_plate + int(2.0 * fps), T - 1)
    for t in range(last_plate, T):
        if root_speed[t] < 0.5:
            s1 = min(t + int(0.5 * fps), T - 1)
            break
    Q0, contact, plate = Q0[s0 : s1 + 1], contact[s0 : s1 + 1], plate[s0 : s1 + 1]
    T = len(Q0)
    if mode != "raw":
        Q0 = smooth_traj(Q0, SMOOTH_IN_SIGMA)

    # --- stance targets (flat sole at locked xy / yaw)
    P_orig = np.zeros((T, 2, 3))
    R_orig = np.zeros((T, 2, 3, 3))
    for t in range(T):
        fk(r, Q0[t])
        for k, fid in enumerate(r.foot_ids):
            P_orig[t, k] = r.d.xpos[fid]
            R_orig[t, k] = r.d.xmat[fid].reshape(3, 3)
    nb = int(round(0.10 * fps))  # blend window before touchdown / after lift-off
    targets = [[None] * 2 for _ in range(T)]
    stance_list = []
    phase_type = np.zeros((T, 2), int)  # 0 swing, 1 flat, 2 toe, 3 heel
    stance_pose = np.zeros((T, 2, 4))  # x, y, z, yaw of the flat lock pose
    for k, fid in enumerate(r.foot_ids):
        w_lock = np.zeros(T)
        lock_p = np.zeros((T, 3))
        lock_R = np.tile(np.eye(3), (T, 1, 1))
        for s, e in intervals(contact[:, k]):
            # roll phase from the original foot: toe lower than heel -> toe contact, and vice versa
            dz = np.array([((R_orig[t, k] @ TOE_LOCAL)[2] - (R_orig[t, k] @ HEEL_LOCAL)[2]) for t in range(s, e + 1)])
            ph = np.where(dz < -ROLL_DZ, 2, np.where(dz > ROLL_DZ, 3, 1))
            flat = ph == 1
            if not flat.any():
                flat[:] = True
            ps = P_orig[s : e + 1, k][flat]
            ys = np.array([yaw_of(R) for R in R_orig[s : e + 1, k]])[flat]
            w = np.hanning(len(ps) + 2)[1:-1] + 1e-3
            p = (ps * w[:, None]).sum(0) / w.sum()
            yaw = np.arctan2((np.sin(ys) * w).sum(), (np.cos(ys) * w).sum())
            p[2] = -SOLE_Z
            R = rotz(yaw)
            phase_type[s : e + 1, k] = ph
            stance_pose[s : e + 1, k, :3] = p
            stance_pose[s : e + 1, k, 3] = yaw
            lo_t, hi_t = max(s - nb, 0), min(e + nb, T - 1)
            for t in range(lo_t, hi_t + 1):
                if t < s:
                    u = (t - (s - nb)) / nb
                elif t > e:
                    u = ((e + nb) - t) / nb
                else:
                    u = 1.0
                u = float(np.clip(u, 0, 1))
                u = u * u * (3 - 2 * u)
                if u >= w_lock[t]:
                    w_lock[t], lock_p[t], lock_R[t] = u, p, R
            stance_list.append(dict(foot="LR"[k], start=s, end=e, xy=p[:2].tolist(), yaw=float(yaw),
                                    plate=bool(plate[s:e + 1, k].any())))
        for t in range(T):
            u = w_lock[t]
            pt = (1 - u) * P_orig[t, k] + u * lock_p[t]
            qa, qb, qo = np.zeros(4), np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qa, R_orig[t, k].flatten())
            mujoco.mju_mat2Quat(qb, lock_R[t].flatten())
            if u > 0:
                dv = np.zeros(3)
                mujoco.mju_subQuat(dv, qb, qa)
                qo[:] = qa
                mujoco.mju_quatIntegrate(qo, dv, u)
            else:
                qo[:] = qa
            Rt = np.zeros(9)
            mujoco.mju_quat2Mat(Rt, qo)
            Rt = Rt.reshape(3, 3)
            zmin = (pt + SOLE_CORNERS @ Rt.T)[:, 2].min()
            if zmin < 0:
                pt = pt.copy()
                pt[2] -= zmin
            wgt = 4.0 if contact[t, k] else 1.0 + 3.0 * u
            targets[t][k] = ("pose", pt, Rt, wgt)
            if phase_type[t, k] in (2, 3):
                lp = TOE_LOCAL if phase_type[t, k] == 2 else HEEL_LOCAL
                Rl = rotz(stance_pose[t, k, 3])
                anchor = stance_pose[t, k, :3] + Rl @ lp
                anchor[2] = 0.0
                targets[t][k] = ("point", lp, anchor, R_orig[t, k], 4.0)

    if no_ik:
        Q2 = Q0.copy()
    else:
        # --- IK pass
        Q1 = np.zeros_like(Q0)
        prev_delta = np.zeros(r.m.nv)
        for t in range(T):
            q_init = Q0[t].copy()
            mujoco.mj_integratePos(r.m, q_init, prev_delta, 1.0)
            Q1[t] = ik_frame(r, Q0[t], q_init, targets[t])
            mujoco.mj_differentiatePos(r.m, prev_delta, 1.0, Q0[t], Q1[t])
        # --- smooth the correction in time (no re-projection: keeps the result smooth)
        D = np.zeros((T, r.m.nv))
        for t in range(T):
            mujoco.mj_differentiatePos(r.m, D[t], 1.0, Q0[t], Q1[t])
        D = gaussian_filter1d(D, SMOOTH_CORR_SIGMA, axis=0, mode="nearest")
        Q2 = np.zeros_like(Q0)
        for t in range(T):
            q = Q0[t].copy()
            mujoco.mj_integratePos(r.m, q, D[t], 1.0)
            q[7:] = np.clip(q[7:], r.lo + 0.005, r.hi - 0.005)
            Q2[t] = q

    # --- ballistic flight COM
    flight = ~contact.any(1)
    com = np.zeros((T, 3))
    for t in range(T):
        fk(r, Q2[t])
        com[t] = r.d.subtree_com[1]
    flight_report = []
    for s, e in intervals(flight):
        a, b = s - 1, e + 1
        if no_ik or a < 0 or b >= T or (b - a) < 3:
            continue
        Tf = (b - a) / fps
        tt = np.arange(0, b - a + 1) / fps
        c0, c1 = com[a], com[b]
        vz0 = (c1[2] - c0[2] + 0.5 * G * Tf**2) / Tf
        bal = c0[None] + (c1 - c0)[None] * (tt / Tf)[:, None]
        bal[:, 2] = c0[2] + vz0 * tt - 0.5 * G * tt**2
        delta = bal - com[a : b + 1]
        Q2[a : b + 1, :3] += delta
        flight_report.append(dict(start=int(s), end=int(e), dur_s=round(Tf, 3), max_fix_mm=round(float(np.abs(delta).max() * 1000), 1)))

    # --- time scaling (default 1.0 = keep human timing)
    fps_eff = fps / time_scale

    # --- derivatives + contact-aware inverse dynamics
    V, A = qvel_qacc(r, Q2, fps_eff)
    nu = len(r.joint_names)
    tau = np.zeros((T, nu))
    foot_force = np.zeros((T, 2, 3))
    residual = np.zeros((T, 6))
    qfrc = np.zeros((T, r.m.nv))
    for t in range(T):
        foot_force[t], tau[t], residual[t], qfrc[t] = solve_contact_forces(r, Q2[t], V[t], A[t], contact[t], phase_type[t])

    # --- diagnostics
    joint_q = Q2[:, 7:]
    excess = np.maximum(joint_q - r.hi, 0) + np.maximum(r.lo - joint_q, 0)
    mid, half = 0.5 * (r.lo + r.hi), 0.5 * (r.hi - r.lo)
    soft_excess = np.maximum(np.abs(joint_q - mid) - 0.9 * half, 0)
    tau_ratio = np.abs(tau) / r.effort
    vel_ratio = np.abs(V[:, 6:]) / r.vel_limit
    slip, pen = [], []
    low2 = np.zeros((T, 2))
    for t in range(T):
        fk(r, Q2[t])
        for k, fid in enumerate(r.foot_ids):
            low2[t, k] = sole_points(r, fid)[:, 2].min()
            if contact[t, k]:
                tg = targets[t][k]
                if tg[0] == "point":
                    pw = r.d.xpos[fid] + r.d.xmat[fid].reshape(3, 3) @ tg[1]
                    slip.append(np.linalg.norm(pw[:2] - tg[2][:2]))
                else:
                    slip.append(np.linalg.norm(r.d.xpos[fid][:2] - tg[1][:2]))
                for st in stance_list:
                    if st["foot"] == "LR"[k] and st["start"] <= t <= st["end"]:
                        st["slip_mm"] = max(st.get("slip_mm", 0.0), round(slip[-1] * 1000, 1))
    selfc = np.array([self_collisions(r, Q2[t]) for t in range(0, T, 2)])
    res_force = np.linalg.norm(residual[:, :3], axis=1)
    weight = r.mass * G

    report = dict(
        trial=name, frames=T, fps=fps_eff, crop=[int(s0), int(s1)], ground_offset_m=round(float(ground), 4),
        plate_offset_s=round(float(plate_offset_s), 3), plate_match=round(float(plate_match), 3),
        plate_feet_swapped=int(n_swapped),
        n_stances=len(stance_list), n_plate_events=len(plate_events), n_flights=len(flight_report),
        stance_slip_max_mm=round(float(np.max(slip) * 1000), 2) if slip else 0.0,
        sole_penetration_max_mm=round(float(max(-low2.min(), 0) * 1000), 2),
        joint_limit_excess_max_rad=round(float(excess.max()), 4),
        soft_limit_frames_pct=round(float((soft_excess > 1e-3).any(1).mean() * 100), 2),
        torque_ratio_max=round(float(tau_ratio.max()), 3),
        torque_over_frames_pct=round(float((tau_ratio > 1).any(1).mean() * 100), 2),
        torque_worst=[(r.joint_names[j], round(float(tau_ratio[:, j].max()), 2), round(float((tau_ratio[:, j] > 1).mean() * 100), 2))
                      for j in np.argsort(-tau_ratio.max(0))[:4]],
        vel_ratio_max=round(float(vel_ratio.max()), 3),
        vel_worst=[(r.joint_names[j], round(float(vel_ratio[:, j].max()), 2)) for j in np.argsort(-vel_ratio.max(0))[:3]],
        soft_limit_joints=[(r.joint_names[j], round(float((soft_excess[:, j] > 1e-3).mean() * 100), 1))
                           for j in np.argsort(-(soft_excess > 1e-3).mean(0))[:5]],
        root_lin_acc_p99=round(float(np.percentile(np.linalg.norm(A[:, :3], axis=1), 99)), 1),
        residual_stance_p95=round(float(np.percentile(res_force[contact.any(1)], 95) / weight), 3) if contact.any() else 0.0,
        residual_flight_p95=round(float(np.percentile(res_force[~contact.any(1)], 95) / weight), 3) if (~contact.any(1)).any() else 0.0,
        base_residual_force_over_weight_p95=round(float(np.percentile(res_force, 95) / weight), 3),
        base_residual_force_over_weight_max=round(float(res_force.max() / weight), 3),
        peak_grf_over_weight=round(float(np.linalg.norm(foot_force, axis=-1).sum(1).max() / weight), 2),
        self_collision_frames=int((selfc[:, 0] > 0).sum() * 2), self_penetration_max_mm=round(float(selfc[:, 1].max() * 1000), 1),
        root_speed_max=round(float(np.linalg.norm(V[:, :2], axis=1).max()), 2),
        flights=flight_report, stances=stance_list,
    )

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_Dynamics.npz"),
        qpos=Q2, qvel=V, qacc=A, qfrc_inverse=qfrc, actuator_tau=tau,
        actuator_names=np.array(r.joint_names), actuator_joint_names=np.array(r.joint_names),
        actuator_torque_limits=r.effort, actuator_velocity_limits=r.vel_limit, fps=np.array(fps_eff),
        robot_mass_kg=np.array(r.mass), foot_contact=contact, foot_force_world=foot_force,
        base_residual_wrench=residual, joint_limit_excess=excess, joint_limit_names=np.array(r.joint_names),
        sole_min_height_m=low2, source_crop=np.array([s0, s1]), time_scale=np.array(time_scale),
    )
    Q50 = resample(r, Q2, fps_eff, OUT_FPS)
    motion = export_mjlab_motion(r, Q50, OUT_FPS)
    idx = np.clip(np.round(np.arange(len(Q50)) * fps_eff / OUT_FPS).astype(int), 0, T - 1)
    motion["foot_contact"] = contact[idx]
    np.savez_compressed(os.path.join(out_dir, f"{name}_motion.npz"), **motion)
    with open(os.path.join(out_dir, f"{name}_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with <trial>_Dynamics.npz")
    ap.add_argument("--phase", required=True, help="dir with phase csv files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", nargs="*", default=None)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--mode", choices=["raw", "light", "full"], default="light",
                    help="raw: unmodified retarget; light: smoothing only; full: + contact IK + ballistic flights")
    args = ap.parse_args()
    r = load_robot()
    files = sorted(f for f in os.listdir(args.src) if f.endswith("_Dynamics.npz"))
    for f in files:
        name = f.replace("_Dynamics.npz", "")
        if args.trials and name not in args.trials:
            continue
        if name in EXCLUDE:
            print(json.dumps({"trial": name, "excluded": EXCLUDE[name]}), flush=True)
            continue
        ph = [p for p in os.listdir(args.phase) if p.startswith(name + "_")]
        phase_csv = os.path.join(args.phase, ph[0]) if ph else None
        rep = process_trial(r, os.path.join(args.src, f), phase_csv, args.out, args.time_scale, args.mode)
        brief = {k: rep[k] for k in rep if k not in ("flights", "stances")}
        print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
