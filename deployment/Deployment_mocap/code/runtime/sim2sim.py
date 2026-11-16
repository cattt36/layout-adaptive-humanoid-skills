"""Sim2sim: run an exported student (policy.onnx + hnu_deploy runtime) in plain CPU MuJoCo.

Differences from training (mjlab / MuJoCo-Warp): CPU MuJoCo solver, explicit joint PD torques computed by the
controller (tau = kp (q* - q) - kd dq, clipped to the effort limits) instead of MuJoCo position actuators with
implicit damping, optional finer physics step (--physics-dt 0.001, closer to the motor drivers), the observation
built by the deployment runtime.  Same success criterion as the training evaluation (no fall: anchor height /
orientation, end-effector height; strict stakes; from the clip's first frame, start joint noise +-0.05 rad).

  python deploy/sim2sim.py <export_dir> [--latency none|0-20|20] [--push] [--dr] [--trials 2] [--physics-dt 0.005]
                           [--tasks CrouchedTraverse,...] [--workers 64] [--out result.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import zlib
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hnu_deploy import ClipContext, RobotState, StudentPolicy, TrackerPolicy, quat_rotate_inv  # noqa: E402

HIDDEN_Z = -10.0


# ---------------------------------------------------------------- quaternion helpers (wxyz)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def qapply(q, v):
    return qmul(qmul(q, np.array([0.0, *v])), qinv(q))[1:]


def _yaw_np(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


# ---------------------------------------------------------------- setup
G = {}


def init_worker(ex, physics_dt, override=None):
    meta = json.load(open(os.path.join(ex, "policy_meta.json")))
    meta.update(override or {})  # --motion-dirs / --latent-dir (planner clock, perturbed layouts)
    for ev, key in (("HNU_BASE_MID", "mid"), ("HNU_BASE_END", "end")):  # shortened bases (inner / end posts)
        if os.environ.get(ev) and meta["bridge"].get("post_base"):
            meta["bridge"]["post_base"][key] = json.loads(os.environ[ev])
    m = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
    m.opt.timestep = physics_dt
    # position actuators -> torque motors: the controller computes the PD torque explicitly
    m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_gainprm[:] = 0.0
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    m.actuator_biasprm[:] = 0.0
    m.actuator_ctrlrange[:] = m.actuator_forcerange
    m.actuator_ctrllimited[:] = 1
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in meta["joint_names"]]
    act_of_joint = {int(m.actuator_trnid[i, 0]): i for i in range(m.nu)}
    G.update(ex=ex, meta=meta, m=m, qadr=np.array([m.jnt_qposadr[j] for j in jid]),
             vadr=np.array([m.jnt_dofadr[j] for j in jid]), act=np.array([act_of_joint[j] for j in jid]),
             body=lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n),
             sens={k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, v) for k, v in meta["sensors"].items()},
             mocap=lambda n: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n + "/mocap_base")],
             pol=TrackerPolicy(ex) if meta.get("policy_type") == "tracker" else StudentPolicy(ex), mass0=m.body_mass.copy(), fric0=m.geom_friction.copy(), ipos0=m.body_ipos.copy(),
             condim0=m.geom_condim.copy(), solref0=m.geom_solref.copy(), jrange0=m.jnt_range.copy(), jlim0=m.jnt_limited.copy(),
             waist=[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in ("waist_roll_joint", "waist_pitch_joint")],
             steps=np.load(os.environ["HNU_STEP_STATES"]) if os.environ.get("HNU_STEP_STATES") else None,
             feet_geoms=[i for i in range(m.ngeom) if re.fullmatch(r"robot/(left|right)_foot[1-7]_collision",
                                                                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")],
             torso=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"),
             bridge_geoms={i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])).split("/")[0]
                           for i in range(m.ngeom)
                           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or "").startswith("bridge_")},
             robot_bodies={i for i in range(m.nbody)
                           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("robot/")},
             # criterion v2: stake i stepped on = a foot geom touches stake i's geom with the ankle raised
             stake_geoms={i: int(re.match(r"stake(\d+)", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i]))).group(1))
                          for i in range(m.ngeom)
                          if re.match(r"stake\d+", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or "")})


def clip_paths(meta, clip):
    for dd in meta["motion_dirs"]:
        f = os.path.join(dd, f"{clip}_motion.npz")
        if os.path.exists(f):
            return f, os.path.join(dd, f"{clip}_stakes.json"), os.path.join(dd, f"{clip}_bridge.json")
    raise FileNotFoundError(clip)


def read_state(m, d, enc_bias=0.0):
    s = G["sens"]
    sd = lambda i: d.sensordata[m.sensor_adr[i]: m.sensor_adr[i] + m.sensor_dim[i]].copy()  # noqa: E731
    return RobotState(ang_vel=sd(s["ang_vel"]), lin_vel=sd(s["lin_vel"]), quat=d.qpos[3:7].copy(),
                      pos=d.qpos[0:3].copy(), q=d.qpos[G["qadr"]].copy() + enc_bias, dq=d.qvel[G["vadr"]].copy())


def run(job):
    clip, trial, opt = job
    meta, m, pol = G["meta"], G["m"], G["pol"]
    rng = np.random.default_rng(zlib.crc32(f"{clip}/{trial}/{opt}".encode()))
    mf, sj, bj = clip_paths(meta, clip)
    ref = np.load(mf)
    ctx = ClipContext(meta, np.load(os.path.join(meta["latent_dir"], f"{clip}_z.npy")), sj, bj)
    ctx.set_reference(mf)  # planned pelvis trajectory (used when meta obstacle_input = "reference")
    ctx_plan = ctx  # planned obstacle layout (before the misplacement below)
    T = ref["joint_pos"].shape[0]
    if opt.get("stake_h_meta"):  # stakes as high as the exported scene (meta), perceived correctly (real-stake size test)
        for s_ in ctx.stakes:
            s_["c"] = np.r_[s_["c"][:2], 0.5 * meta["stakes"]["height"]]
    # dynamics randomisation (level 1, as in training): PD +-10 %, effort 0.95-1.05, link masses +-5 %
    kp, kd, eff = (np.array(meta[k]) for k in ("kp", "kd", "effort_limit"))
    m.body_mass[:], m.geom_friction[:], m.body_ipos[:] = G["mass0"], G["fric0"], G["ipos0"]
    m.geom_condim[:], m.geom_solref[:] = G["condim0"], G["solref0"]
    m.jnt_range[:], m.jnt_limited[:] = G["jrange0"], G["jlim0"]
    if opt.get("waist_lock"):  # waist roll / pitch locked at 0 (faulty real waist; training HNU_WAIST_LOCK)
        for j in G["waist"]:
            m.jnt_range[j], m.jnt_limited[j] = [-0.02, 0.02], 1
        # (faulty real waist): HNU_S2S_WAIST_OFF="roll,pitch" rad -> locked around that offset instead of 0
        _wo = [float(v) for v in os.environ.get("HNU_S2S_WAIST_OFF", "0,0").split(",")]
        for j, o_ in zip(G["waist"], _wo):
            m.jnt_range[j] = [o_ - 0.02, o_ + 0.02]
    enc_bias = np.zeros(len(G["qadr"]))
    if opt["base_rand"]:  # always on in the training evaluation ("nominal" included)
        m.geom_friction[G["feet_geoms"], 0] = rng.uniform(0.3, 1.2)  # one value for all foot geoms
        m.body_ipos[G["torso"]] += rng.uniform([-0.025, -0.05, -0.05], [0.025, 0.05, 0.05])
        enc_bias = rng.uniform(-0.01, 0.01, len(G["qadr"]))
    if opt["dr"]:
        kp, kd = kp * rng.uniform(0.9, 1.1, kp.shape), kd * rng.uniform(0.9, 1.1, kd.shape)
        eff = eff * rng.uniform(0.95, 1.05, eff.shape)
        rb = [i for i in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i).startswith("robot/")]
        m.body_mass[rb] = G["mass0"][rb] * rng.uniform(0.95, 1.05, len(rb))
    if opt.get("dr_x"):  # real-robot gap (hnu_tasks.add_contact_actuator_dr): contact / actuator / payload
        kp0, kd0, eff0 = (np.array(meta[k]) for k in ("kp", "kd", "effort_limit"))
        kp, kd = kp0 * rng.uniform(0.8, 1.2, kp0.shape), kd0 * rng.uniform(0.8, 1.2, kd0.shape)
        ank = np.array(["ankle" in j for j in meta["joint_names"]])
        eff = eff0 * np.where(ank, rng.uniform(0.7, 1.0, eff0.shape), rng.uniform(0.85, 1.05, eff0.shape))
        hk = np.array([("hip_pitch" in j) or ("knee" in j) for j in meta["joint_names"]])
        eff = np.where(hk, eff0 * rng.uniform(0.8, 1.0, eff0.shape), eff)
        rb = [i for i in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i).startswith("robot/")]
        m.body_mass[rb] = G["mass0"][rb] * rng.uniform(0.9, 1.1, len(rb))
        m.body_mass[G["torso"]] += rng.uniform(0.0, 1.5)
        fg = G["feet_geoms"]
        m.geom_friction[fg, 0] = rng.uniform(0.3, 1.5)
        m.geom_condim[fg] = 4
        m.geom_friction[fg, 1] = rng.uniform(0.0, 0.05)
        m.geom_solref[fg, 0] = rng.uniform(0.02, 0.06)
    if os.environ.get("HNU_S2S_LANKLE"):  # weak left ankle pitch of the real robot (logs: 38-40 Nm), env: seeds unchanged
        eff = eff.copy()
        eff[meta["joint_names"].index("left_ankle_pitch_joint")] = float(os.environ["HNU_S2S_LANKLE"])
    if os.environ.get("HNU_S2S_WAIST_KP"):  # real vH logs: waist roll / pitch on PD "kp,kd[,effort Nm]"
        _wk = [float(v) for v in os.environ["HNU_S2S_WAIST_KP"].split(",")]  # (soft waist; effort = weak waist motor)
        kp, kd, eff = kp.copy(), kd.copy(), eff.copy()
        for jn in ("waist_roll_joint", "waist_pitch_joint"):
            kp[meta["joint_names"].index(jn)], kd[meta["joint_names"].index(jn)] = _wk[0], _wk[1]
            if len(_wk) > 2:
                eff[meta["joint_names"].index(jn)] = _wk[2]
    if opt.get("torsion", 0) > 0:  # fixed torsional friction of the soles (condim 4), e.g. 0.03 m
        m.geom_condim[G["feet_geoms"]] = 4
        m.geom_friction[G["feet_geoms"], 1] = opt["torsion"]
    pb = meta["bridge"].get("post_base") or {}
    if ctx.bridge is not None and (pb.get("mid") or pb.get("end")):
        # base / brace boxes resized to this layout's dims (the compiled scene has one size for all posts)
        for gi in G["bridge_geoms"]:
            p = G["bridge_geoms"][gi][len("bridge_"):]
            if p in ctx.base_half and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
                m.geom_size[gi] = ctx.base_half[p]
                m.geom_rbound[gi] = float(np.linalg.norm(ctx.base_half[p]))
                m.geom_aabb[gi] = np.r_[0.0, 0.0, 0.0, ctx.base_half[p]]
    d = mujoco.MjData(m)
    # misplaced obstacles (true positions moved; the robot perceives them correctly): whole layout shifted by
    # --obst-shift in a random horizontal direction and / or every stake moved independently by up to --stake-jitter
    if opt.get("obst_shift", 0) > 0 or opt.get("stake_jitter", 0) > 0:
        import copy
        ctx = copy.deepcopy(ctx)
        ang = rng.uniform(0, 2 * np.pi)
        off = opt.get("obst_shift", 0) * np.array([np.cos(ang), np.sin(ang), 0.0])
        for s_ in ctx.stakes:
            j = opt.get("stake_jitter", 0)
            o = off + (np.append(rng.uniform(-j, j, 2), 0.0) if j > 0 else 0.0)
            s_["c"], s_["xy"] = s_["c"] + o, s_["xy"] + o[:2]
        if ctx.bridge is not None:
            ctx.bridge = {p: (c + off, q) for p, (c, q) in ctx.bridge.items()}
    # bridge built higher / lower than in the reference (perceived correctly): all parts moved by --bridge-dz
    if opt.get("bridge_dz", 0) != 0 and ctx.bridge is not None:
        import copy
        ctx = copy.deepcopy(ctx)
        ctx.bridge = {p: (c + np.array([0.0, 0.0, 0.0 if p.startswith(("base", "brace")) else opt["bridge_dz"]]), q)
                      for p, (c, q) in ctx.bridge.items()}  # post bases stay on the floor
    # obstacles of this clip (others parked below the floor)
    d.mocap_pos[:, 2] = HIDDEN_Z
    for i, s in enumerate(ctx.stakes):
        d.mocap_pos[G["mocap"](f"stake{i}")] = s["c"]
    if ctx.bridge is not None:
        for p, (c, q) in ctx.bridge.items():
            k = G["mocap"](f"bridge_{p}")
            d.mocap_pos[k], d.mocap_quat[k] = c, q
    # initial state = reference frame 0 (+ joint noise as in the evaluation)
    bn = meta["body_names"]
    d.qpos[0:3], d.qpos[3:7] = ref["body_pos_w"][0, 0], ref["body_quat_w"][0, 0]
    d.qvel[0:3] = ref["body_lin_vel_w"][0, 0]
    d.qvel[3:6] = quat_rotate_inv(ref["body_quat_w"][0, 0], ref["body_ang_vel_w"][0, 0])  # free joint: local frame
    lo, hi = meta["tracking"]["reset_joint_noise"]
    d.qpos[G["qadr"]] = ref["joint_pos"][0] + rng.uniform(lo, hi, len(G["qadr"]))
    d.qvel[G["vadr"]] = ref["joint_vel"][0]
    # start modes (real robot): "rest" = frame-0 pose standing still (rope hold released); "step" = a
    # stepping-in-place state of a locomotion controller (HNU_STEP_STATES, tools/make_step_states.py), the robot's
    # heading = reference start heading; "ref" = frame-0 pose WITH the reference velocities (training evaluation)
    if opt.get("start", "ref") != "ref":
        d.qvel[:] = 0.0
        d.qpos[2] += 0.005
        if opt["start"] == "step":
            S = G["steps"]
            i = int(rng.integers(len(S["z"])))
            sn = [str(x) for x in S["joint_names"]]
            q_s, dq_s = S["q"][i][[sn.index(j) for j in meta["joint_names"]]], S["dq"][i][[sn.index(j) for j in meta["joint_names"]]]
            legs = np.array([any(k in j for k in ("hip", "knee", "ankle")) for j in meta["joint_names"]])
            if rng.random() < 0.5:  # arms of the frame-0 pose (else the stepping state's arms)
                q_s = np.where(legs | np.array([j.startswith("waist") for j in meta["joint_names"]]), q_s, d.qpos[G["qadr"]])
            d.qpos[G["qadr"]], d.qvel[G["vadr"]] = q_s, dq_s
            qy = yaw_quat(ref["body_quat_w"][0, 0])
            d.qpos[3:7] = qmul(qy, S["rp_quat"][i])
            d.qpos[2] = float(S["z"][i]) + 0.005
            d.qvel[0:3] = qapply(qy, S["v_yaw"][i])
            d.qvel[3:6] = S["w_body"][i]
    if opt.get("waist_lock"):
        for j in G["waist"]:
            d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]] = float(np.mean(m.jnt_range[j])), 0.0
    if opt.get("start_rp", 0) > 0:  # set down tilted: roll / pitch ~U(-start_rp, start_rp) deg
        r_, p_ = np.deg2rad(rng.uniform(-opt["start_rp"], opt["start_rp"], 2))
        qrp = np.zeros(4)
        mujoco.mju_euler2Quat(qrp, np.array([r_, p_, 0.0]), "xyz")
        d.qpos[3:7] = qmul(qrp, d.qpos[3:7])
    if opt.get("start_xy", 0) > 0:  # start position error, uniform per axis
        d.qpos[0:2] += rng.uniform(-opt["start_xy"], opt["start_xy"], 2)
    if opt.get("start_yaw", 0) > 0:  # robot placed with a heading error of +-start_yaw degrees (random sign)
        psi = np.deg2rad(opt["start_yaw"]) * rng.choice([-1.0, 1.0])
        qz = np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])
        d.qpos[3:7] = qmul(qz, d.qpos[3:7])
        c_, s_ = np.cos(psi), np.sin(psi)
        d.qvel[0:2] = [c_ * d.qvel[0] - s_ * d.qvel[1], s_ * d.qvel[0] + c_ * d.qvel[1]]
    mujoco.mj_forward(m, d)
    tb = [bn.index(n) for n in meta["tracking"]["body_names"]]
    rb = [G["body"](n) for n in meta["tracking"]["body_names"]]
    ia = meta["tracking"]["body_names"].index(meta["tracking"]["anchor"])
    ee = [meta["tracking"]["body_names"].index(n) for n in meta["tracking"]["ee_bodies"]]
    feet = [G["body"](n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    st = meta["stakes"]
    s0 = read_state(m, d, enc_bias)
    # perception error (policy input only; physics and success checks use the true obstacles):
    # constant horizontal offset of every obstacle in a random direction + per-step noise on the positions
    pctx = ctx_plan if meta.get("obstacle_input") == "reference" else ctx  # plan-based policy: sheet positions
    if os.environ.get("HNU_S2S_UNSEEN") == "1":
        # obstacles misplaced (--obst-shift / --stake-jitter) but the robot perceives the PLAN (layout files
        # not updated); default = perceived where they are (= obstacles measured by the mocap, deploy/obstacles_live.py)
        pctx = ctx_plan
    if float(os.environ.get("HNU_S2S_MEAS_NOISE", "0")) > 0 and pctx is ctx:
        # measured obstacles with a constant measurement error per stake / bridge part, N(0, sigma) in x and y
        import copy
        sg = float(os.environ["HNU_S2S_MEAS_NOISE"])
        mrng = np.random.default_rng(zlib.crc32(f"{clip}/{trial}/meas".encode()))
        pctx = copy.deepcopy(ctx)
        for s_ in pctx.stakes:
            e_ = np.append(mrng.normal(0, sg, 2), 0.0)
            s_["c"], s_["xy"] = s_["c"] + e_, s_["xy"] + e_[:2]
        if pctx.bridge is not None:
            pctx.bridge = {p: (c + np.append(mrng.normal(0, sg, 2), 0.0), q) for p, (c, q) in pctx.bridge.items()}
    if os.environ.get("HNU_S2S_PERC_BRIDGE") and ctx.bridge is not None:
        # the robot perceives the bridge of this json (e.g. the sheet) while the physical bridge is the
        # layout's (e.g. measured posts): bridge moved on site without updating the plan.  Env only: seeds unchanged.
        import copy
        pctx = copy.copy(ctx)
        pctx.bridge = ClipContext(meta, ctx.z, None,
                                  os.environ["HNU_S2S_PERC_BRIDGE"]).bridge
    if opt.get("perc_bias", 0) > 0:
        import copy
        pctx = copy.deepcopy(ctx)
        ang = rng.uniform(0, 2 * np.pi)
        off = opt["perc_bias"] * np.array([np.cos(ang), np.sin(ang), 0.0])
        for s_ in pctx.stakes:
            s_["c"] = s_["c"] + off
        if pctx.bridge is not None:
            pctx.bridge = {p: (c + off, q) for p, (c, q) in pctx.bridge.items()}
    pol.obst_noise = opt.get("perc_noise", 0.0)
    pol.rng = np.random.default_rng(rng.integers(2 ** 32))
    # localisation drift (obstacle features only): the robot knows the obstacle layout relative to its start pose
    # up to an initial offset, and tracks itself by odometry with a systematic scale error kappa and heading error
    # dpsi (both ~U(-r, r)): estimated displacement = (1 + kappa) R(dpsi) (true displacement)
    r, b0 = opt.get("drift", 0.0), opt.get("init_off", 0.0)
    loc_err = [0.0]
    if r > 0 or b0 > 0:
        kappa, dpsi = rng.uniform(-r, r), rng.uniform(-r, r)
        ang = rng.uniform(0, 2 * np.pi)
        e0 = b0 * np.array([np.cos(ang), np.sin(ang)])
        p0 = s0.pos[:2].copy()
        Rz = np.array([[np.cos(dpsi), -np.sin(dpsi)], [np.sin(dpsi), np.cos(dpsi)]])
        qz = np.array([np.cos(dpsi / 2), 0.0, 0.0, np.sin(dpsi / 2)])

        def localise(s):
            est = p0 + e0 + (1 + kappa) * Rz @ (s.pos[:2] - p0)
            s.perc_pos = np.array([est[0], est[1], s.pos[2]])
            s.perc_quat = qmul(qz, s.quat)
            loc_err[0] = max(loc_err[0], float(np.linalg.norm(est - s.pos[:2])))
            return s
    else:
        def localise(s):
            return s
    # mocap pelvis pose bias: heading +-head_bias deg (random sign), height +-height_bias m (random sign);
    # obstacle features and the estimated velocity use the biased pose
    hb = np.deg2rad(opt.get("head_bias", 0.0)) * rng.choice([-1.0, 1.0])
    zb = opt.get("height_bias", 0.0) * rng.choice([-1.0, 1.0])
    qb = np.array([np.cos(hb / 2), 0.0, 0.0, np.sin(hb / 2)])
    if hb != 0 or zb != 0:
        _loc0 = localise

        def localise(s):
            s = _loc0(s)
            pp = (s.pos if s.perc_pos is None else s.perc_pos).copy()
            s.perc_pos = pp + np.array([0.0, 0.0, zb])
            s.perc_quat = qmul(qb, s.quat if s.perc_quat is None else s.perc_quat)
            return s
    # LiDAR localisation (env only -> seeds unchanged): the pose for the obstacle features is updated at
    # HNU_S2S_POSE_HZ (e.g. 10 = one LiDAR scan) with a delay of HNU_S2S_POSE_LAG_S and held in between
    pose_hz = float(os.environ.get("HNU_S2S_POSE_HZ", "0"))
    if pose_hz > 0:
        _cdt = meta["sim"]["control_dt"]
        lag_n = int(round(float(os.environ.get("HNU_S2S_POSE_LAG_S", "0")) / _cdt))
        every = max(1, int(round(1.0 / (pose_hz * _cdt))))
        pbuf, held, cnt = collections.deque(maxlen=lag_n + 2), [None], [0]
        _loc1 = localise

        def localise(s):
            s = _loc1(s)
            pbuf.append(((s.pos if s.perc_pos is None else s.perc_pos).copy(),
                         (s.quat if s.perc_quat is None else s.perc_quat).copy()))
            if held[0] is None or cnt[0] % every == 0:
                held[0] = pbuf[0] if len(pbuf) <= lag_n else pbuf[-1 - lag_n]
            cnt[0] += 1
            s.perc_pos, s.perc_quat = held[0][0].copy(), held[0][1].copy()
            return s
    # velocity estimated like on the robot (--vel-est): mocap pelvis position at ~100 Hz (every 2nd physics
    # step at 5 ms) with 1.5 mm noise, least-squares slope over 60 ms, 10-40 ms transport delay, rotated into the body
    # frame with the (biased) heading; replaces the simulator's IMU-site velocity
    vest = None
    if opt.get("vel_est"):
        vest = dict(t=collections.deque(maxlen=64), p=collections.deque(maxlen=64), n=0,
                    delay=rng.uniform(0.01, 0.04))

        def vel_sample(t):
            vest["n"] += 1
            if vest["n"] % 2 == 0:
                vest["t"].append(t)
                vest["p"].append(d.qpos[0:3] + rng.normal(0.0, 0.0015, 3))

        def vel_estimate(s, t):
            tt, pp = np.array(vest["t"]), np.array(vest["p"])
            if len(tt) < 3:
                return s
            sel = (tt <= t - vest["delay"]) & (tt > t - vest["delay"] - 0.06)
            if sel.sum() < 3:
                sel = np.zeros(len(tt), bool)
                sel[-3:] = True
            x = tt[sel] - tt[sel].mean()
            v_w = (x[:, None] * (pp[sel] - pp[sel].mean(0))).sum(0) / max((x * x).sum(), 1e-9)
            s.lin_vel = quat_rotate_inv(qmul(qb, s.quat), v_w)
            return s
    odo_trace = []
    if opt.get("odom"):  # obstacle features from onboard leg odometry + IMU, started at the PLANNED pose
        if "odo" not in G:
            from odometry import LegOdometry
            G["odo"] = LegOdometry(G["ex"])
        odo = G["odo"]
        ydr = np.deg2rad(opt.get("odom_yaw_drift", 0.0)) * rng.choice([-1.0, 1.0])  # gyro heading drift, rad/s
        tclk = [0.0]

        def imu_q(s):
            return qmul(np.array([np.cos(ydr * tclk[0] / 2), 0, 0, np.sin(ydr * tclk[0] / 2)]), s.quat)
        odo.reset(s0.q, imu_q(s0), ref["body_pos_w"][0, 0], float(np.arctan2(*(lambda q: (2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))(ref["body_quat_w"][0, 0]))))

        def localise(s):
            tclk[0] += meta["sim"]["control_dt"]
            pp_, pq_ = odo.update(s.q, imu_q(s))
            e_ = float(np.linalg.norm(pp_[:2] - s.pos[:2]))
            loc_err[0] = max(loc_err[0], e_)
            odo_trace.append((round(tclk[0], 2), round(e_, 3), round(float(pp_[2] - s.pos[2]), 3),
                              round(float(np.degrees((_yaw_np(pq_) - _yaw_np(s.quat) + np.pi) % (2 * np.pi) - np.pi)), 2)))
            if not os.environ.get("HNU_S2S_ODOM_SHADOW"):  # shadow: estimate only recorded, the policy sees the true pose
                s.perc_pos, s.perc_quat = pp_, pq_
            return s
        if not os.environ.get("HNU_S2S_ODOM_SHADOW"):
            s0.perc_pos, s0.perc_quat = odo.pos.copy(), odo.quat.copy()
    else:
        s0 = localise(s0)
    if meta.get("policy_type") == "tracker":  # blind tracker: plays the layout reference; heading origin = planned start
        pol.reset(mf, s0, start_yaw=_yaw_np(ref["body_quat_w"][0, 0]))
    else:
        pol.reset(pctx, s0, 0)
    hist_states = [s0, s0, s0]  # [t-2, t-1, t] for a sensor delay of up to 2 control steps
    lag, period = 0, 25
    phase = int(rng.integers(period))
    dec = int(round(meta["sim"]["control_dt"] / m.opt.timestep))
    push_t = rng.uniform(*meta["push"]["interval_s"]) if opt["push"] else np.inf
    # hold: the policy gets frame 0 for the first H control steps (robot settles from rest), then the clock runs
    H = int(round(opt.get("hold", 0.0) / meta["sim"]["control_dt"]))
    tlog = [] if os.environ.get("HNU_S2S_TORQUE") else None  # env, not opt: per-trial seeds unchanged
    rope = tuple(float(x) for x in opt["rope"].split(",")) if opt.get("rope") else None
    errs, cause = [], None
    # diagnostics: per stake, the stance foot relative to the stake centre during its core window (distance,
    # along-track / lateral components w.r.t. the approach direction, share of frames on top); bridge contacts
    srec = [dict(d=[], along=[], lat=[], on=[]) for _ in ctx.stakes]
    prev_xy = ref["body_pos_w"][0, 0, :2]
    sdir = []
    for s_ in ctx.stakes:
        v = s_["xy"] - prev_xy
        sdir.append(v / max(np.linalg.norm(v), 1e-6))
        prev_xy = s_["xy"]
    bcont = collections.Counter()
    floor_ev = []  # --floor-log: (frame, foot, stakes passed, normal force N, contact x in ankle frame, ref ankle z)
    stepped_v2 = set()
    # swing-foot floor contacts (robot foot on the floor while the reference ankle of that side is >= 7 cm up)
    scuff_n, terr_id = 0, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    ank_i = [bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")]
    foot_side = {g: (0 if "left" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) else 1) for g in G["feet_geoms"]}
    # HNU_S2S_TRACE=<dir> (videos): qpos per control step + obstacle poses -> <dir>/<clip>_<trial>.npz,
    # rendered separately (tools/e2e_robot_demo.py render).  Not part of opt, so the per-trial seed is unchanged.
    trace = [d.qpos.copy()] if os.environ.get("HNU_S2S_TRACE") else None
    rope_len = float(os.environ.get("HNU_S2S_ROPE_LEN", "0"))
    rope_anchor = d.xpos[G["torso"]][:2].copy()
    p_start, y_start, hold_drift = d.qpos[0:2].copy(), _yaw_np(d.qpos[3:7]), None
    evlog = [] if os.environ.get("HNU_S2S_EVLOG") else None
    _wp = m.jnt_qposadr[G["waist"][1]]
    waist_pitch_max = 0.0
    for kk in range(T - 1 + H):
        k = max(0, kk - H)  # clip frame given to the policy
        # sensor latency: one lag (0 / 1 (/ 2) control steps) per 0.5 s segment, shared by all sensor inputs
        if opt["latency"] in ("0-20", "0-40") and (kk + phase) % period == 0:
            lag = int(rng.integers(2 if opt["latency"] == "0-20" else 3))
        elif opt["latency"] == "20":
            lag = 1
        s = hist_states[-1 - lag] if opt["latency"] != "none" else hist_states[-1]
        target = pol.step(s, k, kk < H) if meta.get("policy_type") == "tracker" else pol.step(s, k)
        for j_ in range(dec):
            if rope is not None:  # rope pull on the torso, released linearly (--rope frac,seconds)
                tr_ = (kk * dec + j_) * m.opt.timestep
                fz_ = rope[0] * 33.3 * 9.81 * max(0.0, 1.0 - tr_ / rope[1])
                d.xfrc_applied[G["torso"], 2] = fz_
                if rope_len > 0:  # pendulum rope (HNU_S2S_ROPE_LEN m): horizontal pull back to the start
                    d.xfrc_applied[G["torso"], :2] = -fz_ / rope_len * (d.xpos[G["torso"]][:2] - rope_anchor)
            q, dq = d.qpos[G["qadr"]], d.qvel[G["vadr"]]
            tau_cmd = kp * (target - q) - kd * dq
            d.ctrl[G["act"]] = np.clip(tau_cmd, -eff, eff)
            if tlog is not None:  # --torque-log: commanded PD torque per joint (before the effort clip)
                tlog.append(tau_cmd.copy())
            mujoco.mj_step(m, d)
            if vest is not None:
                vel_sample((kk * dec + j_ + 1) * m.opt.timestep)
        if (kk + 1) * meta["sim"]["control_dt"] >= push_t:  # push: random root velocity change
            vr = meta["push"]["velocity_range"]
            dv = np.array([rng.uniform(*vr.get(a, (0.0, 0.0))) for a in ("x", "y", "z", "roll", "pitch", "yaw")])
            d.qvel[0:3] += dv[:3]
            d.qvel[3:6] += quat_rotate_inv(d.qpos[3:7], dv[3:])
            push_t += rng.uniform(*meta["push"]["interval_s"])
        mujoco.mj_forward(m, d)
        waist_pitch_max = max(waist_pitch_max, float(d.qpos[_wp]))
        if kk == H - 1:  # end of the standing hold: how far the robot moved / turned while it should stand still
            hold_drift = (round(float(np.linalg.norm(d.qpos[0:2] - p_start)), 3),
                          round(float(np.degrees((_yaw_np(d.qpos[3:7]) - y_start + np.pi) % (2 * np.pi) - np.pi)), 1))
        if trace is not None:
            trace.append(d.qpos.copy())
        s_new = read_state(m, d, enc_bias)
        if vest is not None:
            s_new = vel_estimate(s_new, (kk + 1) * dec * m.opt.timestep)
        hist_states = hist_states[1:] + [localise(s_new)]
        f = kk + 1 - H if kk >= H else 0
        # termination rules of the training evaluation
        ref_p, ref_q = ref["body_pos_w"][f][tb], ref["body_quat_w"][f][tb]
        rob_p = np.array([d.xpos[b] for b in rb])
        rob_qa = d.xquat[rb[ia]].copy()
        if opt.get("crit", "track") == "task":
            # TASK criterion (paper definition: no fall, every stake hit, obstacle passed): failure only on a fall
            # (pelvis below 0.40 m or body tilted > 1 rad), numerical blow-up or a missed stake; no reference-tracking
            # terminations.  Completion is checked at the end of the clip.
            up = quat_rotate_inv(d.qpos[3:7], [0, 0, 1.0])[2]
            if d.qpos[2] < float(os.environ.get("HNU_FALL_H", "0.30")) or up < np.cos(float(os.environ.get("HNU_FALL_TILT", "1.4835"))):
                cause = "fall"
            elif not np.all(np.isfinite(d.qvel)) or np.linalg.norm(d.qvel[0:3]) > 30:
                cause = "bad_state"
            # (missed / edge stakes are NOT failures under the task criterion; recorded as a secondary metric)
            if cause is None and f == T - 1 and np.linalg.norm(d.qpos[0:2] - ref["body_pos_w"][f, 0, :2]) > 1.0:
                cause = "not_completed"
        elif abs(ref_p[ia, 2] - rob_p[ia, 2]) > meta["tracking"]["anchor_pos_z"]:
            cause = "anchor_pos"
        elif abs(quat_rotate_inv(ref_q[ia], [0, 0, -1.0])[2] - quat_rotate_inv(rob_qa, [0, 0, -1.0])[2]) > \
                meta["tracking"]["anchor_ori"]:
            cause = "anchor_ori"
        elif np.any(np.abs(ref_p[ee, 2] - rob_p[ee, 2]) > meta["tracking"]["ee_z"]):
            cause = "ee_body_pos"
        elif not np.all(np.isfinite(d.qvel)) or np.linalg.norm(d.qvel[0:3]) > 30:
            cause = "bad_state"
        else:
            for s_ in ctx.stakes:
                if s_["f0"] <= f <= s_["f1"]:
                    p = d.xpos[feet[s_["foot"]]]
                    if np.linalg.norm(p[:2] - s_["xy"]) > st["tol"] or p[2] < st["height"] - 0.02:
                        cause = "stake_missed"
        for i, s_ in enumerate(ctx.stakes):
            if s_["f0"] <= f <= s_["f1"]:
                p = d.xpos[feet[s_["foot"]]]
                e = p[:2] - s_["xy"]
                srec[i]["d"].append(float(np.linalg.norm(e)))
                srec[i]["along"].append(float(e @ sdir[i]))
                srec[i]["lat"].append(float(e @ np.array([-sdir[i][1], sdir[i][0]])))
                srec[i]["on"].append(bool(np.linalg.norm(e) <= st["tol"] and p[2] >= st["height"] - 0.02))
        if ctx.stakes:
            fs = set(G["feet_geoms"])
            for ci in range(d.ncon):
                g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                for ga, gb in ((g1, g2), (g2, g1)):
                    if ga in G["stake_geoms"] and gb in fs and d.xpos[m.geom_bodyid[gb]][2] >= 0.07:
                        stepped_v2.add(G["stake_geoms"][ga])
        if f > 0:
            ref_up = ref["body_pos_w"][f][ank_i, 2] >= 0.07
            on_floor = [False, False]
            for ci in range(d.ncon):
                g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                for ga, gb in ((g1, g2), (g2, g1)):
                    if ga == terr_id and gb in foot_side:
                        on_floor[foot_side[gb]] = True
            scuff_n += sum(1 for s_ in (0, 1) if on_floor[s_] and ref_up[s_])
        ev_b = set()
        if ctx.bridge is not None:
            for ci in range(d.ncon):
                g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
                for ga, gb in ((g1, g2), (g2, g1)):
                    if ga in G["bridge_geoms"] and gb not in G["bridge_geoms"] and m.geom_bodyid[gb] in G["robot_bodies"]:
                        bcont[G["bridge_geoms"][ga]] += 1
                        if evlog is not None:
                            ev_b.add(G["bridge_geoms"][ga] + ":" + mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gb]))[6:])
        if evlog is not None and f > 0:  # HNU_S2S_EVLOG: per-step events for the fall analysis
            nf = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gb]))[6:]
                         for ci in range(d.ncon) for ga, gb in ((d.contact[ci].geom1, d.contact[ci].geom2),
                                                                (d.contact[ci].geom2, d.contact[ci].geom1))
                         if ga == terr_id and gb not in foot_side and m.geom_bodyid[gb] in G["robot_bodies"]})
            evlog.append([int(f), int(on_floor[0]), int(on_floor[1]), int(ref_up[0]), int(ref_up[1]), sorted(ev_b), nf,
                          round(float(d.qpos[2]), 3), round(float(quat_rotate_inv(d.qpos[3:7], [0, 0, 1.0])[2]), 3)])
        if opt.get("floor_log") and ctx.stakes:  # robot foot on the floor while no reference foot is low
            refz = ref["body_pos_w"][f][[bn.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")], 2]
            ok_win = ctx.stakes[0]["f0"] - 20 <= f <= ctx.stakes[-1]["f1"] + 10
            terr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
            if ok_win and refz.min() >= 0.06:
                fs = set(G["feet_geoms"])
                for ci in range(d.ncon):
                    con = d.contact[ci]
                    for ga, gb in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                        if gb in fs and ga == terr:  # foot geom <-> floor
                            fr6 = np.zeros(6)
                            mujoco.mj_contactForce(m, d, ci, fr6)
                            side = 0 if m.geom_bodyid[gb] == feet[0] else 1
                            R_ = d.xmat[feet[side]].reshape(3, 3)
                            loc = R_.T @ (con.pos - d.xpos[feet[side]])  # contact point in the ankle frame (+x toe)
                            gap = sum(1 for s_ in ctx.stakes if s_["f1"] < f)  # stakes already passed
                            floor_ev.append((f, side, gap, float(abs(fr6[0])), float(loc[0]), float(refz[side])))
        # tracking error as in training: reference re-anchored to the robot anchor (xy, yaw)
        dq_ = yaw_quat(qmul(rob_qa, qinv(ref_q[ia])))
        dp = np.array([rob_p[ia, 0], rob_p[ia, 1], ref_p[ia, 2]])
        rel = np.array([dp + qapply(dq_, ref_p[i] - ref_p[ia]) for i in range(len(tb))])
        errs.append(float(np.linalg.norm(rel - rob_p, axis=1).mean()))
        if cause:
            break
    stakes = [dict(idx=i, foot=ctx.stakes[i]["foot"], reached=bool(r["d"]),
                   dist=float(np.mean(r["d"])) if r["d"] else None, along=float(np.mean(r["along"])) if r["d"] else None,
                   lat=float(np.mean(r["lat"])) if r["d"] else None, on=float(np.mean(r["on"])) if r["d"] else None)
              for i, r in enumerate(srec)]
    fail_xy = d.qpos[0:2].copy()
    beam_rel = None
    if ctx.bridge is not None and "plank" in ctx.bridge:
        bc, bq = ctx.bridge["plank"]
        ax = qapply(bq, [1.0, 0.0, 0.0])[:2]
        ax = ax / max(np.linalg.norm(ax), 1e-6)
        beam_rel = float((fail_xy - bc[:2]) @ ax)  # signed position along the beam axis at the end / failure
    all_stepped = all(i in stepped_v2 for i in range(len(ctx.stakes)))
    if trace is not None:
        np.savez_compressed(os.path.join(os.environ["HNU_S2S_TRACE"], f"{clip}_{trial}.npz"), qpos=np.array(trace),
                            mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(), control_dt=meta["sim"]["control_dt"],
                            cause=str(cause), success_v2=bool(cause is None and all_stepped))
    tq = {}
    if tlog:  # per joint: peak |tau|, 95th percentile, share of physics steps at the effort limit (1 control step = dec)
        T_ = np.abs(np.array(tlog))
        for jn_i, jn in enumerate(meta["joint_names"]):
            if any(k in jn for k in ("ankle", "knee", "hip_pitch")):
                tq[jn] = [round(float(T_[:, jn_i].max()), 1), round(float(np.percentile(T_[:, jn_i], 95)), 1),
                          round(float(np.mean(T_[:, jn_i] >= eff[jn_i])), 3)]
        tq["eff"] = {jn: round(float(eff[i]), 1) for i, jn in enumerate(meta["joint_names"]) if "ankle" in jn or "knee" in jn}
    return dict(clip=clip, task=clip.rsplit("_", 1)[0], trial=trial, success=cause is None, cause=cause, torque=tq,
                success_v2=bool(cause is None and all_stepped), stakes_all_stepped=all_stepped if ctx.stakes else None,
                swing_scuff_steps=scuff_n, hold_drift=hold_drift, waist_pitch_max=round(waist_pitch_max, 3),
                steps=k + 1, T=T, body_err=float(np.mean(errs)) if errs else float("nan"),
                loc_err_max=loc_err[0], odo_trace=odo_trace[::25], stakes=stakes, bridge_contacts=dict(bcont), beam_rel=beam_rel,
                pelvis_h=float(d.qpos[2]), ref_pelvis_h=float(ref["body_pos_w"][min(k + 1, T - 1), 0, 2]),
                **(dict(floor_events=floor_ev) if opt.get("floor_log") else {}),
                **(dict(evlog=evlog) if evlog is not None else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--latency", default="none", choices=["none", "0-20", "0-40", "20"])
    # real-robot start and sensing (see run())
    ap.add_argument("--start", default="ref", choices=["ref", "rest", "step"],
                    help="ref: frame 0 with reference velocities; rest: frame-0 pose at rest; step: stepping-in-place "
                         "state (HNU_STEP_STATES)")
    ap.add_argument("--hold", type=float, default=0.0, help="s, policy clock held at frame 0 after the start")
    ap.add_argument("--start-xy", type=float, default=0.0, help="m, start position error (uniform per axis)")
    ap.add_argument("--dr-x", action="store_true", help="contact / actuator / payload randomisation of the training")
    ap.add_argument("--torsion", type=float, default=0.0, help="m, fixed sole torsional friction (condim 4)")
    ap.add_argument("--vel-est", action="store_true", help="velocity estimated from noisy mocap positions (60 ms LSQ)")
    ap.add_argument("--head-bias", type=float, default=0.0, help="deg, mocap heading bias (random sign)")
    ap.add_argument("--height-bias", type=float, default=0.0, help="m, mocap pelvis height bias (random sign)")
    ap.add_argument("--rope", default="", help="frac,seconds: rope pull on the torso (x body weight), released linearly")
    ap.add_argument("--start-rp", type=float, default=0.0, help="deg, start roll / pitch error (uniform)")
    ap.add_argument("--waist-lock", action="store_true", help="waist roll / pitch locked at 0 (model)")
    ap.add_argument("--odom", action="store_true", help="obstacle features from leg odometry + IMU (deploy/odometry.py)")
    ap.add_argument("--odom-yaw-drift", type=float, default=0.0, help="deg/s, IMU heading drift (random sign)")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--no-base-rand", action="store_true", help="disable friction / CoM / encoder-bias randomisation")
    ap.add_argument("--perc-bias", type=float, default=0.0, help="m, constant horizontal obstacle offset (random direction)")
    ap.add_argument("--perc-noise", type=float, default=0.0, help="m, per-step uniform noise on obstacle positions")
    ap.add_argument("--obst-shift", type=float, default=0.0, help="m, true obstacle layout shifted (random direction)")
    ap.add_argument("--stake-jitter", type=float, default=0.0, help="m, every stake moved independently (uniform)")
    ap.add_argument("--drift", type=float, default=0.0, help="odometry drift r: scale and heading error ~U(-r, r)")
    ap.add_argument("--init-off", type=float, default=0.0, help="m, initial localisation offset (random direction)")
    ap.add_argument("--bridge-dz", type=float, default=0.0, help="m, bridge built higher (+) / lower (-) than the reference")
    ap.add_argument("--start-yaw", type=float, default=0.0, help="deg, robot start heading error (random sign)")
    ap.add_argument("--crit", default="track", choices=["track", "task"], help="success criterion (task = no fall, stakes hit, obstacle passed)")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--floor-log", action="store_true", help="record floor contacts between the stakes (no reference foot low)")
    ap.add_argument("--stake-h-meta", action="store_true", help="stake centres at the exported stake height (meta), e.g. a 0.15 m scene")
    ap.add_argument("--physics-dt", type=float, default=0.005)
    ap.add_argument("--tasks", default="CrouchedTraverse,ThreeStepPoleRun,FiveStepPoleRun")
    ap.add_argument("--max-clips", type=int, default=0, help="per task, 0 = all")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--motion-dirs", default="", help="comma list replacing meta motion_dirs (geometry + clock), e.g. "
                    "the planner geometry geom_plan4_f<k> or a perturbed layout of tools/make_layout_perturb.py")
    ap.add_argument("--latent-dir", default="", help="replaces meta latent_dir (e.g. z_lib_f<k>/plan4/<P>)")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.export_dir, "policy_meta.json")))
    override = {}
    if a.motion_dirs:
        override["motion_dirs"] = a.motion_dirs.split(",")
    if a.latent_dir:
        override["latent_dir"] = a.latent_dir
    meta.update(override)
    clips = []
    for t in a.tasks.split(","):
        cs = sorted({f[:-len("_motion.npz")] for dd in meta["motion_dirs"] for f in os.listdir(dd)
                     if f.startswith(t + "_") and f.endswith("_motion.npz")})
        clips += cs[: a.max_clips] if a.max_clips else cs
    opt = dict(latency=a.latency, push=a.push, dr=a.dr, base_rand=not a.no_base_rand, perc_bias=a.perc_bias,
               perc_noise=a.perc_noise, obst_shift=a.obst_shift, stake_jitter=a.stake_jitter,
               drift=a.drift, init_off=a.init_off, bridge_dz=a.bridge_dz, start_yaw=a.start_yaw, crit=a.crit)
    if a.floor_log:  # only when set (the per-trial seed hashes opt)
        opt["floor_log"] = True
    if a.stake_h_meta:
        opt["stake_h_meta"] = True
    # new options only when set (the per-trial seed hashes opt: old results stay reproducible)
    for k_, v_ in dict(start=a.start if a.start != "ref" else None, hold=a.hold or None, start_xy=a.start_xy or None,
                       dr_x=a.dr_x or None, torsion=a.torsion or None, vel_est=a.vel_est or None,
                       head_bias=a.head_bias or None, height_bias=a.height_bias or None, rope=a.rope or None,
                       start_rp=a.start_rp or None, waist_lock=a.waist_lock or None, odom=a.odom or None,
                       odom_yaw_drift=a.odom_yaw_drift or None).items():
        if v_ is not None:
            opt[k_] = v_
    jobs = [(c, i, opt) for c in clips for i in range(a.trials)]
    with Pool(a.workers, initializer=init_worker, initargs=(a.export_dir, a.physics_dt, override)) as p:
        rows = p.map(run, jobs, chunksize=1)
    summ = {}
    for t in a.tasks.split(","):
        r = [x for x in rows if x["task"] == t]
        if r:
            causes = {}
            for x in r:
                if x["cause"]:
                    causes[x["cause"]] = causes.get(x["cause"], 0) + 1
            summ[t] = dict(success=round(float(np.mean([x["success"] for x in r])), 3),
                           success_v2=round(float(np.mean([x["success_v2"] for x in r])), 3),
                           body_err=round(float(np.nanmean([x["body_err"] for x in r])), 4), n=len(r), causes=causes)
            print(f"{t:12s} success {100 * summ[t]['success']:5.1f}%  body err {100 * summ[t]['body_err']:.1f} cm  "
                  f"n={len(r)}  failures {causes}")
    out = a.out or os.path.join(a.export_dir, f"sim2sim_{a.latency}{'_push' if a.push else ''}{'_dr' if a.dr else ''}"
                                              f"{'_nobase' if a.no_base_rand else ''}"
                                              f"{f'_pb{a.perc_bias}' if a.perc_bias else ''}{f'_pn{a.perc_noise}' if a.perc_noise else ''}"
                                              f"_dt{a.physics_dt}.json")
    json.dump(dict(opt=opt, physics_dt=a.physics_dt, summary=summ, rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
