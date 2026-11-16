"""Exteroceptive ("perception") observations for the geometry experts, in the robot's yaw-aligned base frame.

stake_lookahead : next 2 stakes of the clip (those whose core stance window has not ended): stake-top position
                  relative to the pelvis (dx, dy, dz), time until the reference steps on it (s, clipped to 2) and a
                  validity flag  -> 10 dims.  Stand-in for a depth camera / stake detector.
bridge_lookahead: beam (closest point of its axis, horizontally) relative to the pelvis (dx, dy, dz of the beam
                  underside) + the 2 nearest support posts (dx, dy)  -> 7 dims.
Both are appended at the END of the actor / critic groups so that an existing expert can be warm-started by
zero-padding its first layer (tools/pad_checkpoint.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

N_AHEAD = 2


def _yaw_frame(env, perturb: bool = False, command_name: str = "motion"):
    d = env.scene["robot"].data
    p = d.root_link_pos_w
    q = d.root_link_quat_w  # wxyz
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    if perturb and getattr(env, "_hnu_perr", None) is not None:
        # localisation error of the student's obstacle features (version B, HNU_PERC_ERR=1): the robot pose used for
        # the features = reference start + initial offset + (1 + kappa) R(dpsi) (true displacement from the start)
        e = env._hnu_perr
        cmd = env.command_manager.get_term(command_name)
        p0 = cmd.motion.body_pos_w[cmd.motion.clip_start[cmd.clip_id], 0] + env.scene.env_origins  # pelvis, frame 0
        dxy = p[:, :2] - p0[:, :2]
        c, s = torch.cos(e["dpsi"]), torch.sin(e["dpsi"])
        rot = torch.stack([c * dxy[:, 0] - s * dxy[:, 1], s * dxy[:, 0] + c * dxy[:, 1]], -1)
        est = p0[:, :2] + e["e0"] + (1 + e["kappa"])[:, None] * rot
        p = torch.cat([est, p[:, 2:3]], -1)
        yaw = yaw + e["dpsi"]
    if perturb and os.environ.get("HNU_OBST_REF", "0") == "1":
        # (onboard sensors only, no localisation): the obstacle features are computed from the PLANNED pelvis
        # pose (reference at the current clip frame) instead of the measured one -- plan information like z_skill
        c = env.command_manager.get_term(command_name)
        return c.body_pos_w[:, 0], *(lambda y: (torch.cos(y), torch.sin(y)))(_yaw_of(c.body_quat_w[:, 0]))
    odom = os.environ.get("HNU_ODOM", "")
    if perturb and odom:
        p, yaw = _odom_pose(env, p, yaw, command_name, odom)
    if perturb and getattr(env, "_hnu_pbias", None) is not None:
        # motion-capture pose bias (HNU_PERC_BIAS): constant per episode, pelvis position (x, y, z) and heading
        b = env._hnu_pbias
        p = p + b["pos"]
        yaw = yaw + b["yaw"]
    if perturb and os.environ.get("HNU_POSE_RATE"):
        p, yaw = _lidar_pose(env, p, yaw, command_name, os.environ["HNU_POSE_RATE"])
    return p, torch.cos(yaw), torch.sin(yaw)


def _lidar_pose(env, p, yaw, command_name, spec):
    """(LiDAR localisation instead of the motion capture): the pose used for the obstacle features is a
    LiDAR-odometry pose -- updated once per scan and delayed.  HNU_POSE_RATE = "period,lag_lo,lag_hi,fast_frac,
    fast_lag_lo,fast_lag_hi,sig_pos,sig_yaw_deg" (control steps of 20 ms): per episode either SCAN mode (update every
    `period` steps with a random phase, delay ~U[lag_lo, lag_hi] steps) or, with probability fast_frac, FAST mode (an
    IMU-propagated pose: update every step, delay ~U[fast_lag_lo, fast_lag_hi]); every update gets N(0, sig_pos) on x, y
    and N(0, sig_yaw) on the heading.  Buffers refilled at episode starts and clip / frame jumps."""
    per, llo, lhi, ffr, flo, fhi, sp, sy = (float(x) for x in spec.split(","))
    per, llo, lhi, flo, fhi = int(per), int(llo), int(lhi), int(flo), int(fhi)
    c = env.command_manager.get_term(command_name)
    B, dev = env.num_envs, env.device
    K = max(lhi, fhi) + 1
    st = getattr(env, "_hnu_lidar", None)
    if st is None:
        st = env._hnu_lidar = dict(bp=p[None].repeat(K, 1, 1).clone(), by=yaw[None].repeat(K, 1).clone(), hp=p.clone(),
                                   hy=yaw.clone(), lag=torch.zeros(B, dtype=torch.long, device=dev),
                                   per=torch.ones(B, dtype=torch.long, device=dev), ph=torch.zeros(B, dtype=torch.long, device=dev),
                                   ts=c.time_steps.clone(), clip=c.clip_id.clone(), step=-1)
    step = int(env.common_step_counter)
    # refill at episode starts / clip or frame jumps on EVERY call (idempotent within a step: the env's first
    # observation is computed at step 0 after a call made while the managers are built)
    jump = (c.time_steps < st["ts"]) | (c.time_steps > st["ts"] + 1) | (c.clip_id != st["clip"]) | (env.episode_length_buf == 0)
    if jump.any():
        i = jump
        n = int(i.sum())
        fast = torch.rand(n, device=dev) < ffr
        lag_s = torch.randint(llo, lhi + 1, (n,), device=dev)
        lag_f = torch.randint(flo, fhi + 1, (n,), device=dev)
        st["lag"][i] = torch.where(fast, lag_f, lag_s)
        st["per"][i] = torch.where(fast, torch.ones_like(lag_s), torch.full_like(lag_s, per))
        st["ph"][i] = torch.randint(0, per, (n,), device=dev)
        st["bp"][:, i], st["by"][:, i] = p[i][None], yaw[i][None]
        st["hp"][i], st["hy"][i] = p[i], yaw[i]
    st["ts"], st["clip"] = c.time_steps.clone(), c.clip_id.clone()
    if step == st["step"]:  # second feature term of the same step: buffers already advanced
        return st["hp"].clone(), st["hy"].clone()
    st["step"] = step
    w = step % K
    st["bp"][w], st["by"][w] = p, yaw
    upd = ((step + st["ph"]) % st["per"]) == 0
    if upd.any():
        ar = torch.arange(B, device=dev)
        r = (w - st["lag"]) % K
        noise_p = torch.randn(B, 2, device=dev) * sp
        noise_y = torch.randn(B, device=dev) * float(np.radians(sy))
        newp = st["bp"][r, ar].clone()
        newp[:, :2] += noise_p
        st["hp"] = torch.where(upd[:, None], newp, st["hp"])
        st["hy"] = torch.where(upd, st["by"][r, ar] + noise_y, st["hy"])
    return st["hp"].clone(), st["hy"].clone()


def _yaw_of(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _odom_pose(env, p, yaw, command_name, spec):
    """(user: the actor may only use the robot's own sensors): the pose used for the student's obstacle
    features is dead reckoning from the PLANNED start pose -- leg odometry + IMU heading on the robot
    (deploy/odometry.py).  Error model, HNU_ODOM="kappa,drift_deg_s,rw_m_sqrt_s,clean": per episode a displacement scale
    error ~U(+-kappa), a heading drift rate ~U(+-drift) deg/s and a horizontal random walk rw m/sqrt(s); the start pose
    is the reference pose for the rest starts (the real placement error, HNU_REST_XY / _YAW, is not seen) and the true
    pose for the other resets; a share `clean` of the episodes has no odometry error (start error only).  Re-initialised
    at every episode start and when the command jumps to a new clip / frame (teleport)."""
    kap, drift, rw, clean = (float(x) for x in spec.split(","))
    c = env.command_manager.get_term(command_name)
    B, dev = env.num_envs, env.device
    st = getattr(env, "_hnu_odom", None)
    if st is None:
        st = env._hnu_odom = dict(p0=p.clone(), y0=yaw.clone(), pa=p.clone(), ya=yaw.clone(), k=torch.zeros(B, device=dev),
                                  dr=torch.zeros(B, device=dev), rw=torch.zeros(B, 2, device=dev), t0=torch.zeros(B, device=dev),
                                  ts=c.time_steps.clone(), clip=c.clip_id.clone(), step=-1)
    step = int(env.common_step_counter)
    jump = (c.time_steps < st["ts"]) | (c.time_steps > st["ts"] + 1) | (c.clip_id != st["clip"])
    init = ((env.episode_length_buf == 0) | jump) & (st["t0"] != float(step))  # once per step (2 feature terms)
    if init.any():
        i = init
        n = int(i.sum())
        st["p0"][i], st["y0"][i] = p[i], yaw[i]
        rest = getattr(c, "rest_flag", None)
        rest = rest[i] if rest is not None else torch.zeros(n, dtype=torch.bool, device=dev)
        ref_p = c.body_pos_w[i, 0]
        ref_y = _yaw_of(c.body_quat_w[i, 0])
        st["pa"][i] = torch.where(rest[:, None], ref_p, p[i])
        st["ya"][i] = torch.where(rest, ref_y, yaw[i])
        on = (torch.rand(n, device=dev) >= clean).float()
        u = lambda: torch.rand(n, device=dev) * 2 - 1  # noqa: E731
        st["k"][i] = u() * kap * on
        st["dr"][i] = u() * float(np.radians(drift)) * on
        st["rw"][i] = 0.0
        st["rwon"] = st.get("rwon", torch.ones(B, device=dev))
        st["rwon"][i] = on
        st["t0"][i] = float(step)
    if step != st["step"]:
        st["step"] = step
        st["rw"] += torch.randn(B, 2, device=dev) * rw * float(np.sqrt(env.step_dt)) * st.get("rwon", torch.ones(B, device=dev))[:, None]
    st["ts"], st["clip"] = c.time_steps.clone(), c.clip_id.clone()
    t = (step - st["t0"]).clamp(min=0) * env.step_dt
    dpsi = st["ya"] - st["y0"] + st["dr"] * t
    d = p[:, :2] - st["p0"][:, :2]
    cs, sn = torch.cos(dpsi), torch.sin(dpsi)
    est_xy = st["pa"][:, :2] + (1 + st["k"])[:, None] * torch.stack([cs * d[:, 0] - sn * d[:, 1], sn * d[:, 0] + cs * d[:, 1]], -1) + st["rw"]
    est_z = st["pa"][:, 2] + (p[:, 2] - st["p0"][:, 2])
    return torch.cat([est_xy, est_z[:, None]], -1), yaw + dpsi


def zero_lin_vel(env) -> torch.Tensor:
    """(onboard sensors only): the base linear velocity input is not available on the robot -> constant 0
    (input slot kept so that the deployed network can be fine-tuned)."""
    return torch.zeros(env.num_envs, 3, device=env.device)


def resample_perc_bias(env, env_ids, pos: float = 0.04, height: float = 0.04, yaw_deg: float = 5.0,
                       clean_frac: float = 0.3):
    """Reset event (real-robot D1 runs: the mocap pelvis was 4.4-5.2 cm too high and its heading 6-12 deg off
    before calibration, residual +2.6 deg after): per-episode bias of the pelvis pose used for the student's obstacle
    features and estimated velocity -- x, y ~U(-pos, pos), z ~U(-height, height), heading ~U(-yaw, yaw); a share
    clean_frac of the episodes is bias-free."""
    B, dev = env.num_envs, env.device
    if getattr(env, "_hnu_pbias", None) is None:
        env._hnu_pbias = dict(pos=torch.zeros(B, 3, device=dev), yaw=torch.zeros(B, device=dev))
    if env_ids is None:
        env_ids = torch.arange(B, device=dev)
    n = len(env_ids)
    on = (torch.rand(n, device=dev) >= clean_frac).float()
    u = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # noqa: E731
    b = env._hnu_pbias
    b["pos"][env_ids] = torch.stack([u(n) * pos, u(n) * pos, u(n) * height], -1) * on[:, None]
    b["yaw"][env_ids] = u(n) * float(np.radians(yaw_deg)) * on


def est_lin_vel(env, win: int = 3, max_delay: int = 1, sigma: float = 0.04) -> torch.Tensor:
    """Base linear velocity as estimated on the real robot: least-squares slope of the mocap pelvis
    position over 60 ms (= mean velocity of the last `win` control steps, i.e. ~30 ms behind), a further transport delay
    of 0..max_delay steps per episode, Gaussian noise sigma (m/s); rotated into the body frame with the (biased, see
    resample_perc_bias) heading.  Velocity of the IMU site as the training sensor robot/imu_lin_vel.  The shared sensor
    latency of the student observation comes on top."""
    from mjlab.envs import mdp as env_mdp
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    q = env.scene["robot"].data.root_link_quat_w
    v_b = env_mdp.builtin_sensor(env, sensor_name="robot/imu_lin_vel")
    v_w = quat_apply(q, v_b)
    B, dev = env.num_envs, env.device
    L = win + max_delay
    st = getattr(env, "_hnu_vest", None)
    if st is None:
        st = env._hnu_vest = dict(buf=v_w[:, None].repeat(1, L, 1).clone(), step=-1,
                                  delay=torch.randint(0, max_delay + 1, (B,), device=dev))
    step = int(env.common_step_counter)
    if step != st["step"]:
        st["buf"] = torch.cat([st["buf"][:, 1:], v_w[:, None]], 1)
        st["step"] = step
    fresh = env.episode_length_buf == 0
    if fresh.any():
        st["buf"][fresh] = v_w[fresh, None]
        st["delay"][fresh] = torch.randint(0, max_delay + 1, (int(fresh.sum()),), device=dev)
    idx = (L - 1 - st["delay"])[:, None] - torch.arange(win, device=dev)[None]
    est_w = torch.gather(st["buf"], 1, idx[..., None].expand(-1, -1, 3)).mean(1)
    b = getattr(env, "_hnu_pbias", None)
    if b is not None:  # heading bias: the world velocity is expressed with a wrong heading
        c, s = torch.cos(-b["yaw"]), torch.sin(-b["yaw"])
        est_w = torch.stack([c * est_w[:, 0] - s * est_w[:, 1], s * est_w[:, 0] + c * est_w[:, 1], est_w[:, 2]], -1)
    return quat_apply_inverse(q, est_w) + sigma * torch.randn_like(est_w)


def resample_perc_err(env, env_ids, bias_max: float = 0.05, drift_max: float = 0.02, clean_frac: float = 0.3):
    """Reset event (version B): per-episode localisation error of the student's obstacle features -- initial offset
    (radius ~U(0, bias_max), random direction), odometry scale / heading error ~U(-drift_max, drift_max); a share
    clean_frac of the episodes is error-free."""
    B, dev = env.num_envs, env.device
    if getattr(env, "_hnu_perr", None) is None:
        env._hnu_perr = dict(e0=torch.zeros(B, 2, device=dev), kappa=torch.zeros(B, device=dev),
                             dpsi=torch.zeros(B, device=dev))
    if env_ids is None:
        env_ids = torch.arange(B, device=dev)
    n = len(env_ids)
    on = (torch.rand(n, device=dev) >= clean_frac).float()
    ang = torch.rand(n, device=dev) * 2 * torch.pi
    r = torch.rand(n, device=dev) * bias_max * on
    e = env._hnu_perr
    e["e0"][env_ids] = torch.stack([r * torch.cos(ang), r * torch.sin(ang)], -1)
    e["kappa"][env_ids] = (torch.rand(n, device=dev) * 2 - 1) * drift_max * on
    e["dpsi"][env_ids] = (torch.rand(n, device=dev) * 2 - 1) * drift_max * on


def _freeze(env, cmd, key: str, out: torch.Tensor) -> torch.Tensor:
    """Motion-capture dropout (evaluation, pre-registered secondary experiment C): HNU_PERC_FREEZE=<seconds> holds the
    obstacle features at their last value from the clip's trigger frame on (HNU_PERC_FREEZE_TABLE: json {clip:
    frame}, tools/make_freeze_table.py -- 0.5 s before the first stake stance / before entering under the beam)."""
    dur = float(os.environ.get("HNU_PERC_FREEZE", "0"))
    if dur <= 0:
        return out
    st = getattr(env, "_hnu_frz", None)
    if st is None:
        tab = json.load(open(os.environ["HNU_PERC_FREEZE_TABLE"]))
        trig = torch.tensor([tab.get(n, -10 ** 6) for n in cmd.motion.names], device=env.device)
        st = env._hnu_frz = dict(trig=trig)
    frame = cmd.time_steps - cmd.motion.clip_start[cmd.clip_id]
    t0 = st["trig"][cmd.clip_id]
    frozen = ((frame >= t0) & (frame < t0 + int(round(dur * 50))))[:, None]
    held = st.get(key)
    if held is None or held.shape != out.shape:
        held = out.clone()
    res = torch.where(frozen, held, out)
    st[key] = res.clone()
    return res


def _to_base(v, c, s):  # world xy delta [B, ..., 2] -> yaw frame
    shape = (-1,) + (1,) * (v.dim() - 2)
    c, s = c.view(shape), s.view(shape)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def stake_lookahead(env, command_name: str = "motion", perturb: bool = False, no_tts: bool = False) -> torch.Tensor:
    """no_tts (perception-channel ablation, , HNU_STAKE_NO_TTS=1, students only): the time until the reference
    steps on the stake is zeroed, so the timing of the footsteps is not given by the obstacle features."""
    cmd = env.command_manager.get_term(command_name)
    B = env.num_envs
    out = torch.zeros(B, N_AHEAD, 5, device=env.device)
    if getattr(cmd, "stake_table", None) is None:
        return out.flatten(1)
    foot, f0, f1, _ = cmd.stake_windows
    frame = cmd.time_steps - cmd.motion.clip_start[cmd.clip_id]
    tab = cmd.stake_table[cmd.clip_id] + env.scene.env_origins[:, None]  # [B, S, 3] centres (z = height / 2)
    if getattr(cmd, "stake_offset", None) is not None and not (perturb and os.environ.get("HNU_OBST_REF", "0") == "1"):
        # layout randomisation: perceive the stakes where they are (plan-based features, HNU_OBST_REF: the sheet positions)
        tab = tab.clone()
        tab[..., :2] += cmd.stake_offset
    ff0, ff1, fo = f0[cmd.clip_id], f1[cmd.clip_id], foot[cmd.clip_id]
    upcoming = (fo >= 0) & (ff1 >= frame[:, None])
    # rank upcoming stakes by order (stakes are stored in time order): k-th upcoming = k-th True
    rank = torch.cumsum(upcoming.long(), 1) - 1
    p, c, s = _yaw_frame(env, perturb, command_name)
    for k in range(N_AHEAD):
        sel = upcoming & (rank == k)
        has = sel.any(1)
        idx = sel.float().argmax(1)
        ar = torch.arange(B, device=env.device)
        pos = tab[ar, idx]
        top = pos[:, 2] * 2.0  # stake top height
        rel = _to_base(pos[:, :2] - p[:, :2], c, s)
        dt = ((ff0[ar, idx] - frame).float() / 50.0).clamp(0.0, 2.0)
        if no_tts:
            dt = torch.zeros_like(dt)
        v = torch.stack([rel[:, 0], rel[:, 1], top - p[:, 2], dt, torch.ones_like(dt)], -1)
        out[:, k] = torch.where(has[:, None], v, torch.zeros_like(v))
    return _freeze(env, cmd, f"stake{int(perturb)}", out.flatten(1))


def bridge_lookahead(env, command_name: str = "motion", beam_half: float = 0.075, perturb: bool = False) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    B = env.num_envs
    if getattr(cmd, "bridge_table", None) is None:
        return torch.zeros(B, 7, device=env.device)
    names = cmd.bridge_parts
    tab = cmd.bridge_table[cmd.clip_id]  # [B, P, 7] (pos, quat wxyz)
    plan_feat = perturb and os.environ.get("HNU_OBST_REF", "0") == "1"  # plan-based features: the sheet positions
    if getattr(cmd, "bridge_part_off", None) is not None and bool((cmd.bridge_part_off != 0).any()) and not plan_feat:
        # placement randomisation: perceive the bridge where it actually is (the per-part offsets used by place_bridge)
        tab = tab.clone()
        tab[..., :3] += cmd.bridge_part_off
    org = env.scene.env_origins
    p, c, s = _yaw_frame(env, perturb, command_name)
    # beam axis: plank centre + direction from its quaternion (yaw, pitch): x-axis of the box
    i = names.index("plank")
    bc = tab[:, i, :3] + org
    q = tab[:, i, 3:]
    ax = torch.stack([1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2), 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3]),
                      2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])], -1)
    axh = ax[:, :2] / ax[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t = ((p[:, :2] - bc[:, :2]) * axh).sum(-1)
    closest = bc + ax * (t / ax[:, :2].norm(dim=-1).clamp(min=1e-6))[:, None]
    rel = _to_base(closest[:, :2] - p[:, :2], c, s)
    beam = torch.stack([rel[:, 0], rel[:, 1], closest[:, 2] - beam_half - p[:, 2]], -1)
    post_ids = [k for k, n in enumerate(names) if n.startswith("post")]
    if not post_ids:
        return torch.cat([beam, torch.zeros(B, 4, device=env.device)], -1)
    pp = tab[:, post_ids, :2] + org[:, None, :2]
    d = (pp - p[:, None, :2]).norm(dim=-1)
    order = d.argsort(1)[:, :2]
    near = torch.gather(pp, 1, order[..., None].expand(-1, -1, 2))
    relp = _to_base(near - p[:, None, :2], c, s).flatten(1)
    if relp.shape[1] < 4:
        relp = torch.cat([relp, torch.zeros(B, 4 - relp.shape[1], device=env.device)], -1)
    present = (tab[:, i, 2] > -5.0).float()[:, None]  # clips without a bridge (hidden parts): zeros
    return _freeze(env, cmd, f"bridge{int(perturb)}", torch.cat([beam, relp], -1) * present)


def zero_features(env, dim: int) -> torch.Tensor:
    """(HNU_OBST_ZERO=1): the obstacle features are not available on the robot (no reliable localisation)
    -> constant 0, input slot kept (the deployed network is fine-tuned)."""
    return torch.zeros(env.num_envs, dim, device=env.device)
