"""HNU tracking tasks for mjlab (expert teachers, one per action)."""
import copy
import os

from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .multi_motion import MultiMotionCommandCfg

DATA_ROOT = os.environ.get("HNU_REF_ROOT", "/root/gpufree-data/hnu_v2/data/ref_v1")
TASKS = ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse")


def _to_multi(cmd) -> MultiMotionCommandCfg:
    fields = {k: getattr(cmd, k) for k in cmd.__dataclass_fields__ if hasattr(cmd, k)}
    return MultiMotionCommandCfg(**fields)


def feet_ground_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


def _sync_delay_buffers():
    """mjlab gives every observation term its own DelayBuffer drawing lags from the global RNG, so the terms of one
    observation are delayed independently (fresh joint_pos with 20 ms-old joint_vel, ...): an all-fresh observation
    (zero latency) is practically never seen in training.  A real robot delays the whole observation at once.  Give
    every buffer its own generator with the same seed -> identical draws (same shapes, same call order) -> one
    latency per env shared by all terms (HNU_DR_LAG_SYNC=1)."""
    import torch
    from mjlab.utils.buffers import delay_buffer as db
    if getattr(db.DelayBuffer, "_hnu_sync", False):
        return
    init = db.DelayBuffer.__init__
    seed = int(os.environ.get("HNU_DR_LAG_SEED", "12345"))

    def __init__(self, *a, generator=None, **kw):
        dev = kw.get("device", a[3] if len(a) > 3 else "cpu")
        init(self, *a, generator=generator or torch.Generator(device=dev).manual_seed(seed), **kw)

    db.DelayBuffer.__init__ = __init__
    db.DelayBuffer._hnu_sync = True


def add_robust_dr(cfg, delay_groups=("actor",)):
    """Sim-to-real robustness (HNU_DR=1), identical for teachers and every student level, on top of the tracking
    defaults (pushes, CoM offset, encoder bias, foot friction 0.3-1.2):
      PD gains x[0.8, 1.2], actuator effort limits x[0.9, 1.05], link masses x[0.9, 1.1] (per env, at start-up),
      sensing / compute latency 0-2 control steps (0-40 ms) on the policy observation groups."""
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    # curriculum level (HNU_DR_LEVEL): 1 = mild (PD +-10 %, effort 0.95-1.05, mass +-5 %, 0-20 ms), 2 = full
    lvl = int(os.environ.get("HNU_DR_LEVEL", "2"))
    pd, ef, ms, lag = ((0.9, 1.1), (0.95, 1.05), (0.95, 1.05), 1) if lvl == 1 else ((0.8, 1.2), (0.9, 1.05), (0.9, 1.1), 2)
    parts = os.environ.get("HNU_DR_PARTS", "dyn,lag").split(",")  # diagnostics: dynamics only / latency only
    lag_min = int(os.environ.get("HNU_DR_LAG_MIN", "0"))
    lag = int(os.environ.get("HNU_DR_LAG_MAX", str(lag)))
    cfg.events = dict(cfg.events)
    if "dyn" in parts or "pd" in parts:
        cfg.events["pd_gains"] = EventTermCfg(mode="startup", func=dr.pd_gains, params={"kp_range": pd, "kd_range": pd})
    if "dyn" in parts or "effort" in parts:
        cfg.events["effort_limits"] = EventTermCfg(mode="startup", func=dr.effort_limits, params={"effort_limit_range": ef})
    if "dyn" in parts or "mass" in parts:
        cfg.events["body_mass"] = EventTermCfg(mode="startup", func=dr.body_mass, params={"ranges": ms})
    # latency is piecewise constant: resampled every HNU_DR_LAG_PERIOD steps (default 25 = 0.5 s, staggered), so
    # sustained 0 / 20 / 40 ms all occur.  (Per-step jitter, period 0, produced students that need >= 1 step of lag.)
    period = int(os.environ.get("HNU_DR_LAG_PERIOD", "25"))
    if "lag" in parts and os.environ.get("HNU_DR_LAG_SYNC", "0") == "1":
        _sync_delay_buffers()
    # HNU_DR_LAG_SENSORS=1: delay only what is measured (IMU, joints, estimated velocity, obstacle features); the
    # policy's own last action and the clock-indexed z_skill / reference / progress inputs have no sensing latency
    no_lag = ("actions", "task", "progress", "ref_command")
    sensors_only = os.environ.get("HNU_DR_LAG_SENSORS", "0") == "1"
    if "lag" in parts:
        for g in delay_groups:
            if g in cfg.observations:
                for n, t in cfg.observations[g].terms.items():
                    if sensors_only and (n in no_lag or n.startswith("z_")):
                        continue
                    t.delay_min_lag, t.delay_max_lag = min(lag_min, lag), lag
                    t.delay_update_period = period
                    t.delay_per_env_phase = True
    if os.environ.get("HNU_DR_X", "0") == "1":
        add_contact_actuator_dr(cfg)
    return cfg


FEET_RE = r"^(left|right)_foot[1-7]_collision$"


from mjlab.managers.event_manager import requires_model_fields  # noqa: E402


@requires_model_fields("geom_solref")
def foot_solref(env, env_ids, ranges=(0.02, 0.06), asset_cfg=None):
    """Soft / hard soles: contact time constant (solref[0]) of the foot geoms, one value per env (startup)."""
    from mjlab.envs.mdp.dr._core import _randomize_model_field
    _randomize_model_field(env, env_ids, "geom_solref", entity_type="geom", ranges=ranges, operation="abs",
                           asset_cfg=asset_cfg, axes=[0], shared_random=True)


@requires_model_fields("actuator_forcerange")
def joint_effort_scale(env, env_ids, pattern: str = "ankle", ranges=(0.7, 1.0)):
    """Effort limits of the actuators whose joint name contains `pattern` x U(ranges) of the default, per joint and
    env (startup; dr.effort_limits selects whole actuator groups only)."""
    import torch
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = [int(c) for a in robot.actuators for n, c in zip(a.target_names, a.global_ctrl_ids.tolist()) if pattern in n]
    ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    s = ranges[0] + (ranges[1] - ranges[0]) * torch.rand(len(env_ids), len(ids), device=env.device)
    dflt = env.sim.get_default_field("actuator_forcerange")
    e = env_ids.long()[:, None]
    env.sim.model.actuator_forcerange[e, ids, 0] = dflt[ids, 0] * s
    env.sim.model.actuator_forcerange[e, ids, 1] = dflt[ids, 1] * s


def add_contact_actuator_dr(cfg):
    """, real-robot gap (G1 D1 runs the robot turns 7-18 deg less than the simulator with the same
    joint targets; left ankle pitch saturates at 38-40 Nm): HNU_DR_X=1 adds
      feet condim 4 + torsional friction U(HNU_TORSION) m (default 0-0.05), sliding friction U(HNU_FRIC) (0.3-1.5),
      sole contact time constant U(0.02, 0.06) s, back payload U(0, HNU_PAYLOAD) kg on the torso (1.5),
      effort limits x U(0.85, 1.05) per joint and x U(0.7, 1.0) for the ankles (G1 ankle limit 50 Nm -> 35-50 Nm).
    PD gains / link masses / latency follow HNU_DR_LEVEL (level 2: PD x0.8-1.2 per joint, masses x0.9-1.1)."""
    import dataclasses
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    robot = cfg.scene.entities["robot"]
    cols = []
    for c in robot.collisions:
        cd = dict(c.condim) if isinstance(c.condim, dict) else c.condim
        if isinstance(cd, dict):
            cd = {k: (4 if "foot" in k else v) for k, v in cd.items()}
        else:
            cd = 4
        cols.append(dataclasses.replace(c, condim=cd))
    robot.collisions = tuple(cols)
    feet = SceneEntityCfg("robot", geom_names=FEET_RE)
    fr = tuple(float(x) for x in os.environ.get("HNU_FRIC", "0.3,1.5").split(","))
    tor = tuple(float(x) for x in os.environ.get("HNU_TORSION", "0.0,0.05").split(","))
    cfg.events = dict(cfg.events)
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["ranges"] = fr
    cfg.events["foot_torsion"] = EventTermCfg(mode="startup", func=dr.geom_friction, params={
        "asset_cfg": feet, "operation": "abs", "ranges": tor, "axes": [1], "shared_random": True})
    if os.environ.get("HNU_SOLREF", "1") == "1":
        cfg.events["foot_solref"] = EventTermCfg(mode="startup", func=foot_solref, params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=FEET_RE), "ranges": (0.02, 0.06)})
    pl = float(os.environ.get("HNU_PAYLOAD", "1.5"))
    if pl > 0:  # after body_mass (scale of the defaults); "add" also starts from the default torso mass
        cfg.events["payload"] = EventTermCfg(mode="startup", func=dr.body_mass, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add", "ranges": (0.0, pl)})
    cfg.events["effort_limits"] = EventTermCfg(mode="startup", func=dr.effort_limits,
                                               params={"effort_limit_range": (0.85, 1.05)})
    cfg.events["effort_ankle"] = EventTermCfg(mode="startup", func=joint_effort_scale,
                                              params={"pattern": "ankle", "ranges": (0.7, 1.0)})
    # real vD fall (log 141607): left hip pitch at its 88 Nm limit for 0.1 s before the collapse, real torque
    # 0.8-1.0 x the PD command -> hip pitch and knee limits x U(0.8, 1.0) (HNU_DR_X_HIPKNEE=1)
    if os.environ.get("HNU_DR_X_HIPKNEE", "0") == "1":
        for pat in ("hip_pitch", "knee"):
            cfg.events[f"effort_{pat}"] = EventTermCfg(mode="startup", func=joint_effort_scale,
                                                       params={"pattern": pat, "ranges": (0.8, 1.0)})
    return cfg


WAIST_LOCKED = ("waist_roll_joint", "waist_pitch_joint")


def apply_waist_lock(cfg):
    """(real G1 with a faulty waist: waist pitch / roll do not follow their targets): HNU_WAIST_LOCK=1 locks
    waist roll and pitch at 0 -- joint range +-0.02 rad in the simulator and action scale 0 (the policy's output for them
    has no effect; the exported meta has action_scale 0 there, so the robot gets the constant target 0 = default pose)."""
    import re
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
    robot = cfg.scene.entities["robot"]
    base_fn = robot.spec_fn

    def spec_fn():
        s = base_fn()
        for j in s.joints:
            if j.name.split("/")[-1] in WAIST_LOCKED:
                j.range = [-0.02, 0.02]
                j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        return s
    robot.spec_fn = spec_fn
    s0 = base_fn()  # keep the spec alive while reading its joints
    names = [j.name.split("/")[-1] for j in s0.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    scale = {}
    for n in names:
        v = next((s for p, s in G1_ACTION_SCALE.items() if re.fullmatch(p, n)), None)
        if v is not None:
            scale[n] = 0.0 if n in WAIST_LOCKED else v
    cfg.actions["joint_pos"].scale = scale
    wo = os.environ.get("HNU_WAIST_OFF", "")
    if wo:
        # (the faulty real waist may stick away from 0: logs pitch +0.06..+0.20, roll -0.06..-0.13 rad;
        # sim2sim with the waist stuck at pitch +0.20 -> bridge 34-44 %): HNU_WAIST_OFF="roll_lo,roll_hi,pitch_lo,pitch_hi,
        # zero_frac" -> per episode the waist is locked (+-0.02 rad) around a random offset (a share zero_frac at 0); the
        # policy sees the actual angle in its joint observations
        from mjlab.managers.event_manager import EventTermCfg
        rl, rh, pl, ph, zf = (float(x) for x in wo.split(","))
        lo, hi = min(rl, pl, 0.0) - 0.02, max(rh, ph, 0.0) + 0.02

        def spec_fn_wide():
            s = base_fn()
            for j in s.joints:
                if j.name.split("/")[-1] in WAIST_LOCKED:
                    j.range = [lo, hi]
                    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            return s
        robot.spec_fn = spec_fn_wide
        cfg.events = dict(cfg.events)
        cfg.events["waist_offset"] = EventTermCfg(mode="reset", func=waist_offset_lock,
                                                  params={"roll": (rl, rh), "pitch": (pl, ph), "zero_frac": zf})
    return cfg


@requires_model_fields("jnt_range")
def waist_offset_lock(env, env_ids, roll=(0.0, 0.0), pitch=(0.0, 0.0), zero_frac=0.3):
    """reset event of HNU_WAIST_OFF (apply_waist_lock): waist roll / pitch locked +-0.02 rad around a random offset"""
    import torch
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    jn = list(robot.joint_names)
    jid = robot.indexing.joint_ids[[jn.index(n) for n in WAIST_LOCKED]].long()
    n = len(env_ids)
    u = lambda r: r[0] + (r[1] - r[0]) * torch.rand(n, device=env.device)  # noqa: E731
    off = torch.stack([u(roll), u(pitch)], -1) * (torch.rand(n, 1, device=env.device) >= zero_frac).float()
    if getattr(env, "_hnu_waist_off", None) is None:
        env._hnu_waist_off = torch.zeros(env.num_envs, 2, device=env.device)
    env._hnu_waist_off[env_ids] = off
    e = env_ids.long()[:, None]
    env.sim.model.jnt_range[e, jid[None], 0] = off - 0.02
    env.sim.model.jnt_range[e, jid[None], 1] = off + 0.02
    if os.environ.get("HNU_DEBUG_WAIST") and not getattr(env, "_hnu_waist_dbg", False):
        env._hnu_waist_dbg = True
        r = env.sim.model.jnt_range[env_ids[:4].long()][:, jid]
        print(f"[waist_offset] joints {jid.tolist()} offsets (first 4) {off[:4].tolist()} ranges {r.tolist()}", flush=True)


def hnu_env_cfg(task: str, play: bool = False):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    if os.environ.get("HNU_WAIST_LOCK", "0") == "1":
        apply_waist_lock(cfg)
    cmd = _to_multi(cfg.commands["motion"])
    cmd.motion_file = os.path.join(DATA_ROOT, task)
    cfg.commands["motion"] = cmd
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (feet_ground_sensor(),)
    # HNU_EP_LEN: longer episodes for the slow rope release (up to 12 s) before the motion
    cfg.episode_length_s = float(os.environ.get("HNU_EP_LEN", "10")) if not play else cfg.episode_length_s
    if os.environ.get("HNU_DR", "0") == "1":
        add_robust_dr(cfg, ("actor",))
    return cfg


def hnu_rl_cfg(task: str):
    cfg = unitree_g1_tracking_ppo_runner_cfg()
    cfg.experiment_name = f"hnu_track_{task}"
    cfg.max_iterations = 20_000
    cfg.save_interval = 500
    cfg.logger = "tensorboard"
    return cfg


STAKES_ROOT = os.environ.get("HNU_STAKES_ROOT", "/root/gpufree-data/hnu_v2/data/ref_stakes")


def bad_physics_state(env):
    """Terminate envs whose state diverged (e.g. a foot initialised inside a stake) before NaNs reach the obs."""
    import torch
    d = env.scene["robot"].data
    v, w, q = d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.joint_vel
    finite = torch.isfinite(v).all(-1) & torch.isfinite(w).all(-1) & torch.isfinite(q).all(-1)
    return ~finite | (v.norm(dim=-1) > 30.0) | (w.norm(dim=-1) > 60.0) | (q.abs().max(dim=-1).values > 200.0)


def hnu_stakes_env_cfg(task: str, play: bool = False):
    """ThreeStepPoleRun/FiveStepPoleRun tracking on per-clip stakes (diameter 0.30 m, height 0.10 m) with raised references."""
    from mjlab.managers.termination_manager import TerminationTermCfg
    from .stakes import stake_entity_cfgs
    cfg = hnu_env_cfg(task, play=play)
    cfg.terminations["bad_state"] = TerminationTermCfg(func=bad_physics_state)
    # footstep task: world-frame foot placement reward, stronger global root tracking, horizontal drift limit
    from mjlab.managers.reward_manager import RewardTermCfg
    from .stakes import anchor_xy_too_far, feet_global_position_error_exp
    cfg.rewards["feet_global_pos"] = RewardTermCfg(func=feet_global_position_error_exp,
                                                   weight=float(os.environ.get("HNU_FEET_W", "3.0")),
                                                   params={"command_name": "motion",
                                                           "std": float(os.environ.get("HNU_FEET_STD", "0.10"))})
    fz_w = float(os.environ.get("HNU_FEET_Z_W", "0"))  # foot height tracking (5PR blind tracker hopped too low)
    if fz_w > 0:
        from .stakes import feet_height_error_exp
        cfg.rewards["feet_height"] = RewardTermCfg(func=feet_height_error_exp, weight=fz_w,
                                                   params={"command_name": "motion",
                                                           "std": float(os.environ.get("HNU_FEET_Z_STD", "0.05"))})
    az_w = float(os.environ.get("HNU_ANCHOR_Z_W", "0"))  # anchor height tracking (5PR first hop too flat)
    if az_w > 0:
        from .stakes import anchor_height_error_exp
        cfg.rewards["anchor_height"] = RewardTermCfg(func=anchor_height_error_exp, weight=az_w,
                                                     params={"command_name": "motion",
                                                             "std": float(os.environ.get("HNU_ANCHOR_Z_STD", "0.05"))})
    stake_rw = float(os.environ.get("HNU_STAKE_RW", "0"))
    if stake_rw > 0:  # direct footstep objective (foot-to-stake-centre during the reference stance on each stake)
        from .stakes import stake_foot_reward
        cfg.rewards["stake_foot"] = RewardTermCfg(func=stake_foot_reward, weight=stake_rw,
                                                  params={"command_name": "motion", "std": 0.08})
    if os.environ.get("HNU_STAKE_MISS_TERM", "1") == "1":
        from .stakes import stake_missed
        cfg.terminations["stake_missed"] = TerminationTermCfg(func=stake_missed, params={"command_name": "motion"})
    if os.environ.get("HNU_STAKE_CONTACT", "0") == "1":
        # , success criterion v2: contact-based stake objective (stakes.py) -- one payout per stake
        # (fully on it / edge / missed) and termination when a stake passes without being stepped on
        from .stakes import stake_contact_sensors, stake_event_reward, stake_missed_contact
        cfg.scene.sensors = tuple(cfg.scene.sensors) + stake_contact_sensors()
        cfg.rewards["stake_event"] = RewardTermCfg(func=stake_event_reward, weight=1.0, params={
            "command_name": "motion", "r_full": float(os.environ.get("HNU_STK_FULL", "2.0")),
            "r_edge": float(os.environ.get("HNU_STK_EDGE", "0.5")), "r_miss": float(os.environ.get("HNU_STK_MISS", "-2.0"))})
        cfg.terminations["stake_missed_contact"] = TerminationTermCfg(func=stake_missed_contact,
                                                                      params={"command_name": "motion"})
    scuff_w = float(os.environ.get("HNU_SCUFF_W", "0"))
    if scuff_w != 0:  # floor contact between the stakes (reference feet raised), per foot and step
        from .stakes import floor_scuff
        cfg.rewards["floor_scuff"] = RewardTermCfg(func=floor_scuff, weight=scuff_w, params={"command_name": "motion"})
    add_robustness_terms(cfg)
    cfg.rewards["motion_global_root_pos"].weight = 2.0
    cfg.rewards["motion_global_root_pos"].params["std"] = float(os.environ.get("HNU_GROOT_STD", "0.2"))  # demo blind trackers: wider (0.4)
    cfg.terminations["anchor_xy"] = TerminationTermCfg(func=anchor_xy_too_far,
                                                       params={"command_name": "motion",
                                                               "threshold": float(os.environ.get("HNU_ANCHOR_XY", "0.4"))})
    # never initialise a foot inside a stake: lift the reset pose slightly and use smaller joint noise
    m = cfg.commands["motion"]
    if m.pose_range:
        m.pose_range = dict(m.pose_range) | {"z": (0.02, 0.04)}
    m.joint_position_range = (-0.05, 0.05)
    cfg.commands["motion"].motion_file = os.path.join(STAKES_ROOT, task)
    if os.environ.get("HNU_STAKES_REHEARSAL", "0") == "1":  # + the original flat clips (no stakes): rehearsal
        cfg.commands["motion"].motion_file += "," + os.path.join(DATA_ROOT, task)
    cfg.commands["motion"].use_stakes = True
    cfg.scene.entities = dict(cfg.scene.entities) | stake_entity_cfgs()
    if float(os.environ.get("HNU_RUNUP_FREE_S", "-1")) >= 0:
        # run-up tracked body-relative only; world-frame objectives / drift limit from shortly before the first stake
        from .stakes import gated_anchor_xy, gated_feet_global, gated_root_global
        cfg.rewards["feet_global_pos"].func = gated_feet_global
        cfg.rewards["motion_global_root_pos"].func = gated_root_global
        cfg.terminations["anchor_xy"].func = gated_anchor_xy
    if os.environ.get("HNU_PERCEPTION", "0") == "1":
        add_perception(cfg, "stakes")
    return blind_actor(cfg)


def add_perception(cfg, kind: str, **params):
    """Append the exteroceptive term (stakes / bridge look-ahead) to the actor (noisy) and critic (clean) groups."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
    from .perception import bridge_lookahead, stake_lookahead
    func = stake_lookahead if kind == "stakes" else bridge_lookahead
    p = {"command_name": "motion", **params}
    cfg.observations["actor"].terms["perception"] = ObservationTermCfg(
        func=func, params=p, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02))
    cfg.observations["critic"].terms["perception"] = ObservationTermCfg(func=func, params=p)


BRIDGE_ROOT = os.environ.get("HNU_BRIDGE_ROOT", "/root/gpufree-data/hnu_v2/data/ref_bridge")


def bridge_contact_sensor(part: str = "plank") -> ContactSensorCfg:
    return ContactSensorCfg(
        name="bridge_contact" if part == "plank" else f"bridge_contact_{part}",
        primary=ContactMatch(mode="subtree", pattern=r"^pelvis$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern=r"^bridge_part$", entity=f"bridge_{part}"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )


def bridge_touch(env):
    """[num_envs] bool: any robot body touches the plank or a support post."""
    import torch
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for k in list(env.scene.sensors.keys()):
        data = getattr(env.scene.sensors[k], "data", None)
        if k.startswith("bridge_contact") and hasattr(data, "found"):
            out |= (data.found.reshape(env.num_envs, -1) > 0).any(-1)
    return out


def bridge_contact(env):
    """1 while any robot body touches the plank / posts (the task is to pass under it without contact)."""
    return bridge_touch(env).float()


def bridge_contact_force(env):
    """Graded contact penalty: 0.2 for any touch, rising to 1.0 with the contact force (1.0 at one body
    weight, ~340 N) -- grazing the beam / a post is allowed by the success criterion, hitting it is not."""
    import torch
    f = torch.zeros(env.num_envs, device=env.device)
    for k in list(env.scene.sensors.keys()):
        data = getattr(env.scene.sensors[k], "data", None)
        if k.startswith("bridge_contact") and hasattr(data, "force"):
            f = f + data.force.reshape(env.num_envs, -1, 3).norm(dim=-1).sum(-1)
    touch = bridge_touch(env).float()
    return touch * (0.2 + 0.8 * (f / 340.0).clamp(max=1.0))


def feet_slip(env):
    """Squared horizontal speed of feet that are in contact with the floor (slipping), summed over both feet."""
    import torch
    robot = env.scene["robot"]
    ids = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    v = robot.data.body_link_lin_vel_w[:, ids, :2]
    c = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    return (c.float() * torch.sum(torch.square(v), -1)).sum(-1)


def add_robustness_terms(cfg):
    """Fine-tuning knobs: HNU_SLIP_W (< 0) foot-slip penalty; HNU_PUSH_SCALE scales the push velocities."""
    from mjlab.managers.reward_manager import RewardTermCfg
    w = float(os.environ.get("HNU_SLIP_W", "0"))
    if w != 0:
        cfg.rewards["feet_slip"] = RewardTermCfg(func=feet_slip, weight=w)
    s = float(os.environ.get("HNU_PUSH_SCALE", "1"))
    if s != 1 and "push_robot" in cfg.events:
        vr = cfg.events["push_robot"].params["velocity_range"]
        cfg.events["push_robot"].params["velocity_range"] = {k: (v[0] * s, v[1] * s) for k, v in vr.items()}


def blind_actor(cfg):
    """(user: BeyondMimic-style blind tracker for the real robot): HNU_ACTOR_BLIND=1 -> the ACTOR (the policy
    that runs on the robot) gets only what the robot measures itself plus the reference: the global anchor position,
    the base linear velocity and the obstacle perception are constant 0 (slots kept, so the existing teacher is
    fine-tuned by PPO); reference command, anchor orientation (IMU), base angular velocity, joints and last action stay.
    The critic keeps the privileged observations (asymmetric actor-critic)."""
    if os.environ.get("HNU_ACTOR_BLIND", "0") != "1":
        return cfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from .perception import zero_features
    terms = cfg.observations["actor"].terms
    for name, dim in (("motion_anchor_pos_b", 3), ("base_lin_vel", 3), ("perception", None)):
        if name in terms:
            if dim is None:  # stakes 10 / bridge 7
                dim = 10 if terms[name].func.__name__ == "stake_lookahead" else 7
            terms[name] = ObservationTermCfg(func=zero_features, params={"dim": dim})
    return cfg


def swing_floor_scuff(env, command_name: str, ref_up: float = 0.07) -> "torch.Tensor":
    """(real G1, D1: a swing foot touched the floor under the beam and the robot fell): per foot, the robot
    foot is on the floor while the reference foot of the same side is in swing (contact label off) and raised (ankle
    >= ref_up; stance ankle ~0.045 m, so >= ~2.5 cm clearance).  Shuffle steps that stay on the floor are not counted."""
    c = env.command_manager.get_term(command_name)
    idx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    ref_z = c.body_pos_w[:, idx, 2] - env.scene.env_origins[:, None, 2]
    swing = (~c.ref_foot_contact.bool()) & (ref_z >= ref_up)
    gnd = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    moving = c._moving()
    out = (gnd & swing).float().sum(1)
    return out if moving is None else out * moving[:, 0]


def hnu_bridge_env_cfg(play: bool = False):
    """CrouchedTraverse under the marker-defined bridge (plank + rear ramp + front leg, per-clip registration)."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from .bridge import bridge_entity_cfgs, part_names
    cfg = hnu_env_cfg("CrouchedTraverse", play=play)
    d = os.path.join(BRIDGE_ROOT, "CrouchedTraverse")
    cfg.commands["motion"].motion_file = d
    cfg.commands["motion"].use_bridge = True
    cfg.scene.entities = dict(cfg.scene.entities) | bridge_entity_cfgs(d)
    touch_parts = [p for p in part_names(d) if p == "plank" or p.startswith("post")]
    cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(bridge_contact_sensor(p) for p in touch_parts)
    cfg.terminations["bad_state"] = TerminationTermCfg(func=bad_physics_state)
    cfg.rewards["bridge_contact"] = RewardTermCfg(
        func=bridge_contact_force if os.environ.get("HNU_BRIDGE_FORCE", "0") == "1" else bridge_contact,
        weight=float(os.environ.get("HNU_BRIDGE_W", "-1.0")))
    add_robustness_terms(cfg)
    sw = float(os.environ.get("HNU_SWING_SCUFF_W", "0"))
    if sw != 0:  # swing-foot floor contact (real D1 fall)
        cfg.rewards["swing_scuff"] = RewardTermCfg(func=swing_floor_scuff, weight=sw, params={"command_name": "motion"})
    # the bridge is fixed in the reference frame: track the global path (as for the stakes)
    from .stakes import anchor_xy_too_far
    cfg.rewards["motion_global_root_pos"].weight = 2.0
    cfg.rewards["motion_global_root_pos"].params["std"] = float(os.environ.get("HNU_GROOT_STD", "0.2"))
    cfg.terminations["anchor_xy"] = TerminationTermCfg(func=anchor_xy_too_far,
                                                       params={"command_name": "motion", "threshold": float(os.environ.get("HNU_ANCHOR_XY_BRIDGE", "0.5"))})
    if os.environ.get("HNU_PERCEPTION", "0") == "1":
        from .bridge import _metas, _parts
        beam_half = float(_parts(_metas(d)[0])["plank"]["half"][2])
        add_perception(cfg, "bridge", beam_half=beam_half)
    return blind_actor(cfg)


if os.path.isdir(os.path.join(BRIDGE_ROOT, "CrouchedTraverse")):
    register_mjlab_task(
        task_id="HNU-Track-CrouchedTraverse-Bridge",
        env_cfg=hnu_bridge_env_cfg(),
        play_env_cfg=hnu_bridge_env_cfg(play=True),
        rl_cfg=hnu_rl_cfg("CrouchedTraverse"),
        runner_cls=MotionTrackingOnPolicyRunner,
    )

for _t in ("ThreeStepPoleRun", "FiveStepPoleRun"):
    register_mjlab_task(
        task_id=f"HNU-Track-{_t}-Stakes",
        env_cfg=hnu_stakes_env_cfg(_t),
        play_env_cfg=hnu_stakes_env_cfg(_t, play=True),
        rl_cfg=hnu_rl_cfg(_t + "_stakes"),
        runner_cls=MotionTrackingOnPolicyRunner,
    )

for _t in TASKS:
    register_mjlab_task(
        task_id=f"HNU-Track-{_t}",
        env_cfg=hnu_env_cfg(_t),
        play_env_cfg=hnu_env_cfg(_t, play=True),
        rl_cfg=hnu_rl_cfg(_t),
        runner_cls=MotionTrackingOnPolicyRunner,
    )
