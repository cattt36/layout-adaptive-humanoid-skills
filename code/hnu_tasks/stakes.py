"""Stepping stakes (ThreeStepPoleRun/FiveStepPoleRun.usd spec: cylinder diameter 0.30 m, height 0.10 m) as per-env mocap entities.

Each env places the stakes of the clip it is tracking (from <clip>_stakes.json next to the motion file); envs
whose clip has fewer stakes (or none) park the unused stakes 10 m below the floor.
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg

N_STAKES = 5
HIDDEN_Z = -10.0
STAKE_H = float(os.environ.get("HNU_STAKE_H", "0.10"))  # height curriculum: 0.03 -> 0.06 -> 0.10 m
STAKE_TOL = float(os.environ.get("HNU_STAKE_TOL", "0.17"))  # foot-to-centre tolerance of stake_missed (precision curriculum)
STAKE_R = float(os.environ.get("HNU_STAKE_R", "0.15"))  # stake radius (real-stake size test; training 0.15)


STAKE_SHAPE = os.environ.get("HNU_STAKE_SHAPE", "cylinder")  # cylinder | mesh<N> (N-gon prism) | box (equal area)


def stake_spec_fn(radius: float, height: float):
    """Stake collision shape.  MuJoCo-Warp loses sphere-cylinder contacts once a foot sphere is deep inside
    the cylinder (CPU MuJoCo keeps them) -> feet fall through stakes; HNU_STAKE_SHAPE selects an alternative
    (mesh<N>: convex N-gon prism circumscribing the same radius; box: square of equal area)."""
    def fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="stake")
        g = body.add_geom()
        g.name = "stake_collision"
        if STAKE_SHAPE.startswith("mesh"):
            nside = int(STAKE_SHAPE[4:] or 16)
            ang = np.arange(nside) * 2 * np.pi / nside
            ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], -1)
            verts = np.concatenate([np.c_[ring, np.full(nside, -0.5 * height)], np.c_[ring, np.full(nside, 0.5 * height)]])
            m = spec.add_mesh()
            m.name = "stake_prism"
            m.uservert = verts.astype(np.float64).flatten().tolist()
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = "stake_prism"
        elif STAKE_SHAPE == "box":
            half = 0.5 * radius * np.sqrt(np.pi)  # equal area to the circle
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [half, half, 0.5 * height]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [radius, 0.5 * height, 0.0]
        g.rgba = [0.55, 0.42, 0.30, 1.0]
        g.friction = [1.0, 0.005, 0.0001]
        g.condim = 3
        g.contype = 1
        g.conaffinity = 1
        return spec
    return fn


def stake_entity_cfgs(radius: float = STAKE_R, height: float = STAKE_H) -> dict[str, EntityCfg]:
    cfgs = {}
    for i in range(N_STAKES):
        c = EntityCfg(spec_fn=stake_spec_fn(radius, height))
        c.init_state = EntityCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z))
        cfgs[f"stake{i}"] = c
    return cfgs


def load_stake_table(files: list[str], device: str) -> torch.Tensor:
    """[num_clips, N_STAKES, 3] stake centre positions in motion coordinates (hidden ones at HIDDEN_Z)."""
    tab = np.zeros((len(files), N_STAKES, 3), np.float32)
    tab[:, :, 2] = HIDDEN_Z
    for c, f in enumerate(files):
        j = f.replace("_motion.npz", "_stakes.json")
        if os.path.exists(j):
            meta = json.load(open(j))
            for i, p in enumerate(meta["stake_xyz_50hz"][:N_STAKES]):
                tab[c, i] = p
    return torch.tensor(tab, device=device)


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def feet_global_position_error_exp(env, command_name: str, std: float) -> torch.Tensor:
    """Foot placement in the WORLD frame (xy included): exp(-mean_feet ||p_foot - p_ref||^2 / std^2)."""
    c = env.command_manager.get_term(command_name)
    idx = [c.cfg.body_names.index(n) for n in FEET]
    err = torch.sum(torch.square(c.body_pos_w[:, idx] - c.robot_body_pos_w[:, idx]), dim=-1).mean(-1)
    return torch.exp(-err / std ** 2)


def feet_height_error_exp(env, command_name: str, std: float) -> torch.Tensor:
    """(blind 5PR tracker hopped ~13 cm lower than the reference): foot HEIGHT only (world z), unaffected by the
    horizontal drift that saturates feet_global_pos: exp(-mean_feet (z_foot - z_ref)^2 / std^2)."""
    c = env.command_manager.get_term(command_name)
    idx = [c.cfg.body_names.index(n) for n in FEET]
    err = torch.square(c.body_pos_w[:, idx, 2] - c.robot_body_pos_w[:, idx, 2]).mean(-1)
    return torch.exp(-err / std ** 2)


def anchor_height_error_exp(env, command_name: str, std: float) -> torch.Tensor:
    """(blind 5PR tracker: first hop onto stake 1 lifts the pelvis 5 cm vs 13 cm in the reference; the robot
    "does not take off"): anchor HEIGHT only (world z), unaffected by the horizontal drift that saturates the 3D anchor
    position reward: exp(-(z_anchor - z_ref)^2 / std^2)."""
    c = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]) / std ** 2)


def anchor_xy_too_far(env, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when the torso drifts horizontally away from the reference (needed for footstep tasks)."""
    c = env.command_manager.get_term(command_name)
    return torch.norm(c.anchor_pos_w[:, :2] - c.robot_anchor_pos_w[:, :2], dim=-1) > threshold


def load_stake_windows(files: list[str], device: str):
    """Per clip and stake: foot index (0 L, 1 R, -1 none), core stance window [f0, f1] in 50 Hz clip frames
    (middle 50 % of the reference stance on the stake) and stake centre xy."""
    C = len(files)
    foot = torch.full((C, N_STAKES), -1, dtype=torch.long)
    f0 = torch.zeros((C, N_STAKES), dtype=torch.long)
    f1 = torch.full((C, N_STAKES), -1, dtype=torch.long)
    xy = torch.zeros((C, N_STAKES, 2))
    for c, f in enumerate(files):
        j = f.replace("_motion.npz", "_stakes.json")
        if not os.path.exists(j):
            continue
        meta = json.load(open(j))
        for i, s in enumerate(meta["stakes"][:N_STAKES]):
            a, b = s["stance"][0] * 50.0 / meta["fps"], s["stance"][1] * 50.0 / meta["fps"]
            foot[c, i] = 0 if s["foot"] == "L" else 1
            f0[c, i] = int(round(a + 0.25 * (b - a)))
            f1[c, i] = int(round(b - 0.25 * (b - a)))
            xy[c, i] = torch.tensor([s["x"], s["y"]])
    return foot.to(device), f0.to(device), f1.to(device), xy.to(device)


def stake_missed(env, command_name: str) -> torch.Tensor:
    """Terminate if, while the reference stands on a stake, the robot's corresponding foot is not on that stake top."""
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    foot, f0, f1, xy = c.stake_windows
    frame = c.time_steps - c.motion.clip_start[c.clip_id]
    idx = [c.cfg.body_names.index(n) for n in FEET]
    feet = c.robot_body_pos_w[:, idx] - env.scene.env_origins[:, None]  # [B, 2, 3]
    missed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first = 1 if os.environ.get("HNU_STAKE_SKIP_FIRST", "0") == "1" else 0  # training only: relaxed first stake
    for i in range(first, N_STAKES):
        k = foot[c.clip_id, i]
        active = (k >= 0) & (frame >= f0[c.clip_id, i]) & (frame <= f1[c.clip_id, i])
        p = feet[torch.arange(env.num_envs, device=env.device), k.clamp(min=0)]
        cxy = xy[c.clip_id, i] + (c.stake_offset[:, i] if getattr(c, "stake_offset", None) is not None else 0.0)
        far = torch.norm(p[:, :2] - cxy, dim=-1) > STAKE_TOL
        low = p[:, 2] < STAKE_H - 0.02
        missed |= active & (far | low)
    return missed


def stake_foot_reward(env, command_name: str, std: float) -> torch.Tensor:
    """Task reward of the footstep problem: while the reference stands on a stake, exp(-d^2/std^2) of the
    corresponding robot foot's horizontal distance d to the stake centre (0 outside stake windows)."""
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return torch.zeros(env.num_envs, device=env.device)
    foot, f0, f1, xy = c.stake_windows
    frame = c.time_steps - c.motion.clip_start[c.clip_id]
    idx = [c.cfg.body_names.index(n) for n in FEET]
    feet = c.robot_body_pos_w[:, idx] - env.scene.env_origins[:, None]
    r = torch.zeros(env.num_envs, device=env.device)
    for i in range(N_STAKES):
        k = foot[c.clip_id, i]
        active = (k >= 0) & (frame >= f0[c.clip_id, i]) & (frame <= f1[c.clip_id, i])
        p = feet[torch.arange(env.num_envs, device=env.device), k.clamp(min=0)]
        cxy = xy[c.clip_id, i] + (c.stake_offset[:, i] if getattr(c, "stake_offset", None) is not None else 0.0)
        d2 = torch.sum(torch.square(p[:, :2] - cxy), dim=-1)
        r += active.float() * torch.exp(-d2 / std ** 2)
    return r


# ---------------------------------------------------------------------------------------------------------------
# Contact-based stake objective (success criterion v2: every stake must be stepped on, top or edge).
# The old stake_missed / stake checks used ankle height, which also accepts a foot HOVERING over a stake, and
# feet_ground_contact only sees the terrain.  Per stake and episode a contact level is tracked:
#   2 = a foot stood fully on the stake (foot-stake contact, no floor contact of that foot)
#   1 = edge / straddle (foot-stake contact while that foot also touches the floor)
#   0 = not stepped on
# over the reference stance window +-0.2 s (STK_MARGIN frames), with the ankle above ground level (>= 0.07 m: a foot
# on the floor kicking the stake side does not count).  At window end + margin the stake is settled once.
STK_MARGIN = 10


def stake_contact_sensors():
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    return tuple(ContactSensorCfg(
        name=f"foot_stake{i}",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern=r"^stake$", entity=f"stake{i}"),
        fields=("found", "force"), reduce="netforce", num_slots=1) for i in range(N_STAKES))


def stake_state(env):
    st = getattr(env, "_hnu_stk", None)
    if st is None:
        B, dev = env.num_envs, env.device
        st = env._hnu_stk = dict(level=torch.zeros(B, N_STAKES, dtype=torch.long, device=dev),
                                 start=torch.zeros(B, dtype=torch.long, device=dev), step=-1)
    return st


def reset_stake_state(env, env_ids, start_frames):
    """Called by the motion command at every (re)sample: new episode / clip -> levels 0, remember the start frame."""
    st = stake_state(env)
    st["level"][env_ids] = 0
    st["start"][env_ids] = start_frames


def _stake_contact_update(env, command_name: str):
    """Idempotent per simulation step: raise the per-stake contact level from the current contacts."""
    st = stake_state(env)
    stepc = int(getattr(env, "common_step_counter", 0))
    if st["step"] == stepc:
        return st
    st["step"] = stepc
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return st
    foot, f0, f1, _ = c.stake_windows
    frame = c.time_steps - c.motion.clip_start[c.clip_id]
    idx = [c.cfg.body_names.index(n) for n in FEET]
    raised = (c.robot_body_pos_w[:, idx, 2] - env.scene.env_origins[:, None, 2]) >= 0.07  # [B, 2]
    gnd = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    for i in range(N_STAKES):
        name = f"foot_stake{i}"
        if name not in env.scene.sensors:
            continue
        fnd = env.scene.sensors[name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
        on = fnd & raised
        lv = torch.where((on & ~gnd).any(1), 2, torch.where(on.any(1), 1, 0))
        k = foot[c.clip_id, i]
        win = (k >= 0) & (frame >= f0[c.clip_id, i] - STK_MARGIN) & (frame <= f1[c.clip_id, i] + STK_MARGIN)
        st["level"][:, i] = torch.where(win, torch.maximum(st["level"][:, i], lv), st["level"][:, i])
    return st


def _settling(env, command_name: str):
    """[B, N_STAKES] bool: stake i is settled at this step (window end + margin) and was scoreable in this episode
    (its window started after the episode's start frame)."""
    c = env.command_manager.get_term(command_name)
    foot, f0, f1, _ = c.stake_windows
    frame = c.time_steps - c.motion.clip_start[c.clip_id]
    st = stake_state(env)
    ok = foot[c.clip_id] >= 0
    begin = f0[c.clip_id] - STK_MARGIN
    return ok & (frame[:, None] == f1[c.clip_id] + STK_MARGIN) & (begin >= st["start"][:, None])


def stake_event_reward(env, command_name: str, r_full: float, r_edge: float, r_miss: float) -> torch.Tensor:
    """One payout per stake at window end + margin: r_full (stood fully on it), r_edge (edge / straddle) or r_miss
    (never stepped on it; r_miss < 0).  Paid once, so lingering on a stake is not rewarded."""
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return torch.zeros(env.num_envs, device=env.device)
    st = _stake_contact_update(env, command_name)
    s = _settling(env, command_name)
    lv = st["level"]
    val = torch.where(lv == 2, r_full, torch.where(lv == 1, r_edge, r_miss))
    return (val * s.float()).sum(1)


def floor_scuff(env, command_name: str, ref_low: float = 0.06) -> torch.Tensor:
    """(FiveStepPoleRun teacher scuffing the floor between stakes, see the floor-touch diagnostic):
    number of robot feet touching the floor while no reference foot is at floor level (ankle >= ref_low), from the first
    stake window start - 2 margins to the last window end + margin (same definition as eval_student.py HNU_EVAL_FLOOR)."""
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return torch.zeros(env.num_envs, device=env.device)
    foot, f0, f1, _ = c.stake_windows
    frame = c.time_steps - c.motion.clip_start[c.clip_id]
    ok = foot[c.clip_id] >= 0
    lo = torch.where(ok, f0[c.clip_id], 1 << 30).min(1).values - 2 * STK_MARGIN
    hi = torch.where(ok, f1[c.clip_id], -1).max(1).values + STK_MARGIN
    idx = [c.cfg.body_names.index(n) for n in FEET]
    ref_up = ((c.body_pos_w[:, idx, 2] - env.scene.env_origins[:, None, 2]) >= ref_low).all(1)
    gnd = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    run = ok.any(1) & (frame >= lo) & (frame <= hi)
    return gnd.float().sum(1) * (run & ref_up).float()


def stake_missed_contact(env, command_name: str) -> torch.Tensor:
    """Terminate when a stake is settled without ever having been stepped on (contact-based stake_missed)."""
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    st = _stake_contact_update(env, command_name)
    return (_settling(env, command_name) & (st["level"] == 0)).any(1)


def stake_level_metric(env, command_name: str, level: int) -> torch.Tensor:
    """Diagnostics (reward weight 0): share of the stakes settled at this step that reached `level`."""
    c = env.command_manager.get_term(command_name)
    if getattr(c, "stake_windows", None) is None:
        return torch.zeros(env.num_envs, device=env.device)
    st = _stake_contact_update(env, command_name)
    s = _settling(env, command_name)
    return ((st["level"] == level) & s).float().sum(1)


RUNUP_FREE_S = float(os.environ.get("HNU_RUNUP_FREE_S", "-1"))  # <0: off


def global_gate(env, command_name: str = "motion") -> torch.Tensor:
    """1 from RUNUP_FREE_S before the first stake on, 0 during the run-up (or when the clip has no stakes):
    the run-up is tracked relative to the body only (as the flat expert does), world-frame position objectives
    start shortly before the first stake, where the perceived stake position guides the approach."""
    c = env.command_manager.get_term(command_name)
    if RUNUP_FREE_S < 0 or getattr(c, "stake_windows", None) is None:
        return torch.ones(env.num_envs, device=env.device)
    foot, f0, _, _ = c.stake_windows
    frame = c.time_steps - c.motion.clip_start[c.clip_id]
    first = f0[c.clip_id, 0]
    has = foot[c.clip_id, 0] >= 0
    # clips without stakes (rehearsal of the original flat motion) are tracked body-relative only, as the flat expert
    return ((frame >= first - int(RUNUP_FREE_S * 50)) & has).float()


def gated_feet_global(env, command_name: str, std: float) -> torch.Tensor:
    return feet_global_position_error_exp(env, command_name, std) * global_gate(env, command_name)


def gated_root_global(env, command_name: str, std: float) -> torch.Tensor:
    from mjlab.tasks.tracking import mdp as tmdp
    return tmdp.motion_global_anchor_position_error_exp(env, command_name=command_name, std=std) * global_gate(env, command_name)


def gated_anchor_xy(env, command_name: str, threshold: float) -> torch.Tensor:
    return anchor_xy_too_far(env, command_name, threshold) & (global_gate(env, command_name) > 0)


def place_stakes(env, clip_ids: torch.Tensor, env_ids: torch.Tensor, table: torch.Tensor, offset=None) -> None:
    origins = env.scene.env_origins[env_ids]
    quat = torch.zeros(len(env_ids), 4, device=origins.device)
    quat[:, 0] = 1.0
    for i in range(N_STAKES):
        name = f"stake{i}"
        try:
            ent = env.scene[name]
        except (KeyError, AttributeError):
            return
        pos = table[clip_ids, i] + origins
        if offset is not None:  # layout randomisation (horizontal)
            pos = pos.clone()
            pos[:, :2] += offset[:, i]
        ent.write_mocap_pose_to_sim(torch.cat([pos, quat], -1), env_ids=env_ids)
