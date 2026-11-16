"""Distillation environment: all three actions, MoE teacher observations and deployable student observations.

Observation groups
  teacher : exactly the expert actor observation (reference joints, anchor pose, base velocities, joints, last
            action), noise-free
  student : deployable only -- IMU angular velocity, projected gravity, joint pos/vel, last action (history 5)
            + z_skill(t) and z_skill(t + 0.5 s) from the chosen latent profile
  task    : one-hot action id (used by the MoE router, not by the student)
"""
from __future__ import annotations

import os

import torch

from mjlab.envs import mdp as env_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import DATA_ROOT, TASKS, _to_multi, feet_ground_sensor

Z_FUTURE_STEPS = 25  # 0.5 s at 50 Hz


def obs_z_skill(env, command_name: str, offset: int, jitter: int = 0) -> torch.Tensor:
    """z_skill at t + offset.  jitter > 0 (training-time augmentation, HNU_Z_JITTER): the offset is moved by a random
    integer in [-jitter, jitter] per env and step, so the student cannot key on exact z values of a clip."""
    if jitter > 0:
        off = offset + torch.randint(-jitter, jitter + 1, (env.num_envs,), device=env.device)
        return env.command_manager.get_term(command_name).z_skill(off.clamp(min=0))
    return env.command_manager.get_term(command_name).z_skill(offset)


def obs_task_onehot(env, command_name: str) -> torch.Tensor:
    tid = env.command_manager.get_term(command_name).task_id
    return torch.nn.functional.one_hot(tid.clamp(min=0), len(TASKS)).float()


def obs_progress(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).clip_progress


def obs_clip_frame(env, command_name: str) -> torch.Tensor:
    """(clip index, frame within clip) -- bookkeeping for labels, never fed to a policy."""
    c = env.command_manager.get_term(command_name)
    m = c.motion
    return torch.stack([c.clip_id.float(), (c.time_steps - m.clip_start[c.clip_id]).float()], -1)


def hnu_distill_env_cfg(latent_dir: str, play: bool = False, motion_dirs: str | None = None,
                        z_offsets: tuple[int, ...] = (0, Z_FUTURE_STEPS), scene: str = "flat",
                        hist: int = 5, lin_vel: bool = False):
    """scene = "flat" (original) or "bridge": CrouchedTraverse clips under the capture-video bridge (HNU_BRIDGE_ROOT),
    W1/W2 on flat ground; the CrouchedTraverse expert reads the bridge look-ahead ("teacher_perc" group) and every
    student variant gets the same low-dimensional obstacle features (execution-layer perception input)."""
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    if os.environ.get("HNU_WAIST_LOCK", "0") == "1":  # waist roll / pitch locked (faulty real waist)
        from . import apply_waist_lock
        apply_waist_lock(cfg)
    cmd = _to_multi(cfg.commands["motion"])
    if scene == "bridge" and motion_dirs is None:
        from . import BRIDGE_ROOT
        motion_dirs = ",".join(os.path.join(BRIDGE_ROOT if t == "CrouchedTraverse" else DATA_ROOT, t) for t in TASKS)
    if scene == "geom" and motion_dirs is None:
        # unified geometry scene: ThreeStepPoleRun / FiveStepPoleRun on 0.06 m stakes (robot-scale spacing), CrouchedTraverse under the bridge
        from . import BRIDGE_ROOT
        geo = {"ThreeStepPoleRun": os.environ.get("HNU_GEOM_W1", "/root/gpufree-data/hnu_v2/data/ref_stakes_s93_h06"),
               "FiveStepPoleRun": os.environ.get("HNU_GEOM_W2", "/root/gpufree-data/hnu_v2/data/ref_stakes_s83_h06"),
               "CrouchedTraverse": BRIDGE_ROOT}
        motion_dirs = ",".join(os.path.join(geo[t], t) for t in TASKS)
    cmd.motion_file = motion_dirs or ",".join(os.path.join(DATA_ROOT, t) for t in TASKS)
    cmd.latent_dir = latent_dir
    cfg.commands["motion"] = cmd
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (feet_ground_sensor(),)
    teacher_terms = {k: ObservationTermCfg(func=v.func, params=dict(v.params))
                     for k, v in cfg.observations["actor"].terms.items()}
    student_terms = {
        "base_ang_vel": ObservationTermCfg(func=env_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
                                           noise=Unoise(n_min=-0.2, n_max=0.2), history_length=hist),
        "projected_gravity": ObservationTermCfg(func=env_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
                                                history_length=hist),
        "joint_pos": ObservationTermCfg(func=env_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                                        history_length=hist),
        "joint_vel": ObservationTermCfg(func=env_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5),
                                        history_length=hist),
        "actions": ObservationTermCfg(func=env_mdp.last_action, history_length=hist),
    }
    # (real robot): HNU_VEL_EST=1 -> velocity estimated like on the robot (mocap position slope, delayed,
    # noisy; perception.est_lin_vel); HNU_PERC_BIAS="pos,height,yaw_deg[,clean]" -> per-episode mocap pose bias on the
    # student's obstacle features and estimated velocity (training only)
    vel_est = os.environ.get("HNU_VEL_EST", "0") == "1" and not play
    pbias = os.environ.get("HNU_PERC_BIAS", "") if not play else ""
    if lin_vel:
        # v6: base linear velocity as provided by the robot's state estimator (noisy: +-0.2 m/s)
        if os.environ.get("HNU_LINVEL_ZERO", "0") == "1":  # onboard sensors only -> no velocity input
            from .perception import zero_lin_vel
            student_terms["base_lin_vel"] = ObservationTermCfg(func=zero_lin_vel, history_length=hist)
        elif vel_est:
            from .perception import est_lin_vel
            student_terms["base_lin_vel"] = ObservationTermCfg(func=est_lin_vel, params={"win": 3, "max_delay": 1,
                                                                                         "sigma": 0.04},
                                                               noise=Unoise(n_min=-0.1, n_max=0.1), history_length=hist)
        else:
            student_terms["base_lin_vel"] = ObservationTermCfg(func=env_mdp.builtin_sensor,
                                                               params={"sensor_name": "robot/imu_lin_vel"},
                                                               noise=Unoise(n_min=-0.2, n_max=0.2), history_length=hist)
    # z_skill samples at the given future offsets (control steps).  Legacy layout (0, 25) keeps the old names.
    # z augmentation for generalisation to unseen clips (training only; same for every level):
    # HNU_Z_NOISE = uniform noise amplitude on every z sample, HNU_Z_JITTER = random time shift of the samples (steps)
    z_noise = float(os.environ.get("HNU_Z_NOISE", "0")) if not play else 0.0
    z_jit = int(os.environ.get("HNU_Z_JITTER", "0")) if not play else 0
    for i, off in enumerate(z_offsets):
        name = "z_now" if (tuple(z_offsets) == (0, Z_FUTURE_STEPS) and i == 0) else \
               "z_future" if tuple(z_offsets) == (0, Z_FUTURE_STEPS) else f"z_{off}"
        student_terms[name] = ObservationTermCfg(
            func=obs_z_skill, params={"command_name": "motion", "offset": int(off), "jitter": z_jit},
            noise=Unoise(n_min=-z_noise, n_max=z_noise) if z_noise > 0 else None)
    teacher_perc_terms = None
    teacher_stk_terms = None
    if scene in ("bridge", "geom"):
        from mjlab.managers.termination_manager import TerminationTermCfg
        from . import BRIDGE_ROOT, bad_physics_state, bridge_contact_sensor
        from .bridge import _metas, _parts, bridge_entity_cfgs, part_names
        from .perception import bridge_lookahead, stake_lookahead
        bdir = os.path.join(BRIDGE_ROOT, "CrouchedTraverse")
        cmd.use_bridge = True
        cfg.scene.entities = dict(cfg.scene.entities) | bridge_entity_cfgs(bdir)
        cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
            bridge_contact_sensor(p) for p in part_names(bdir) if p == "plank" or p.startswith("post"))
        cfg.terminations["bad_state"] = TerminationTermCfg(func=bad_physics_state)
        bp = {"command_name": "motion", "beam_half": float(_parts(_metas(bdir)[0])["plank"]["half"][2])}
        teacher_perc_terms = dict(teacher_terms) | {"perception": ObservationTermCfg(func=bridge_lookahead, params=bp)}
        # low-dimensional obstacle features for every student (stakes slot kept for the stake scene: zeros here)
        # version B (HNU_PERC_ERR=1): per-episode localisation error on the student's obstacle features only
        perr = (os.environ.get("HNU_PERC_ERR", "0") == "1" or bool(pbias) or bool(os.environ.get("HNU_ODOM"))) and not play
        if os.environ.get("HNU_OBST_REF", "0") == "1":  # plan-based features also in evaluation / export (play)
            perr = True
        # perception-channel ablation (HNU_STAKE_NO_TTS=1, training AND evaluation): no time-to-step in the features
        no_tts = os.environ.get("HNU_STAKE_NO_TTS", "0") == "1"
        student_terms["obst_stakes"] = ObservationTermCfg(func=stake_lookahead,
                                                          params={"command_name": "motion", "perturb": perr,
                                                                  "no_tts": no_tts},
                                                          noise=Unoise(n_min=-0.02, n_max=0.02))
        student_terms["obst_bridge"] = ObservationTermCfg(func=bridge_lookahead, params=bp | {"perturb": perr},
                                                          noise=Unoise(n_min=-0.02, n_max=0.02))
        if os.environ.get("HNU_OBST_ZERO", "0") == "1":  # proprioception only (no localisation on the robot)
            from .perception import zero_features
            student_terms["obst_stakes"] = ObservationTermCfg(func=zero_features, params={"dim": 10})
            student_terms["obst_bridge"] = ObservationTermCfg(func=zero_features, params={"dim": 7})
        if pbias:
            from mjlab.managers.event_manager import EventTermCfg
            from .perception import resample_perc_bias
            pb = [float(x) for x in pbias.split(",")]
            cfg.events = dict(cfg.events)
            cfg.events["perc_bias"] = EventTermCfg(mode="reset", func=resample_perc_bias, params=dict(
                pos=pb[0], height=pb[1], yaw_deg=pb[2], clean_frac=pb[3] if len(pb) > 3 else 0.3))
        if perr and os.environ.get("HNU_PERC_ERR", "0") == "1":
            from mjlab.managers.event_manager import EventTermCfg
            from .perception import resample_perc_err
            cfg.events = dict(cfg.events)
            cfg.events["perc_err"] = EventTermCfg(mode="reset", func=resample_perc_err, params={
                "bias_max": float(os.environ.get("HNU_PERC_BIAS_MAX", "0.05")),
                "drift_max": float(os.environ.get("HNU_PERC_DRIFT_MAX", "0.02")),
                "clean_frac": float(os.environ.get("HNU_PERC_CLEAN_FRAC", "0.3"))})
        if scene == "geom":
            from .stakes import stake_entity_cfgs, stake_missed
            cmd.use_stakes = True
            cfg.scene.entities = dict(cfg.scene.entities) | stake_entity_cfgs()  # height HNU_STAKE_H (0.06 m)
            # strict footstep criterion: missing a stake while the reference stands on it ends the episode
            cfg.terminations["stake_missed"] = TerminationTermCfg(func=stake_missed, params={"command_name": "motion"})
            if os.environ.get("HNU_STAKE_CONTACT", "0") == "1" and not play:
                # (criterion v2): a stake that passes without a foot on it (contact-based) ends the
                # DAgger episode, as for the fine-tuned teachers (training only; eval_student adds its own sensors)
                from .stakes import stake_contact_sensors, stake_missed_contact
                cfg.scene.sensors = tuple(cfg.scene.sensors) + stake_contact_sensors()
                cfg.terminations["stake_missed_contact"] = TerminationTermCfg(func=stake_missed_contact,
                                                                              params={"command_name": "motion"})
            if cmd.pose_range:  # never initialise a foot inside a stake
                cmd.pose_range = dict(cmd.pose_range) | {"z": (0.02, 0.04)}
            cmd.joint_position_range = (-0.05, 0.05)
            teacher_stk_terms = dict(teacher_terms) | {
                "perception": ObservationTermCfg(func=stake_lookahead, params={"command_name": "motion"})}
    # upper-bound baseline: same deployable proprioception, explicit reference joints instead of z_skill
    ref_terms = {k: v for k, v in student_terms.items() if not k.startswith("z_")}
    ref_terms["ref_command"] = ObservationTermCfg(func=env_mdp.generated_commands, params={"command_name": "motion"})
    # lower-bound baseline: no latent, only the action id and the normalised clip progress
    nolat_terms = {k: v for k, v in student_terms.items() if not k.startswith("z_")}
    nolat_terms["task"] = ObservationTermCfg(func=obs_task_onehot, params={"command_name": "motion"})
    nolat_terms["progress"] = ObservationTermCfg(func=obs_progress, params={"command_name": "motion"})
    cfg.observations = {
        "teacher": ObservationGroupCfg(terms=teacher_terms, concatenate_terms=True, enable_corruption=False),
        "student": ObservationGroupCfg(terms=student_terms, concatenate_terms=True, enable_corruption=not play),
        "student_ref": ObservationGroupCfg(terms=ref_terms, concatenate_terms=True, enable_corruption=not play),
        "student_nolatent": ObservationGroupCfg(terms=nolat_terms, concatenate_terms=True, enable_corruption=not play),
        "meta": ObservationGroupCfg(terms={
            "clip": ObservationTermCfg(func=obs_clip_frame, params={"command_name": "motion"})},
            concatenate_terms=True, enable_corruption=False),
        "task": ObservationGroupCfg(terms={"onehot": ObservationTermCfg(func=obs_task_onehot, params={"command_name": "motion"})},
                                    concatenate_terms=True, enable_corruption=False),
    }
    if teacher_perc_terms is not None:
        cfg.observations["teacher_perc"] = ObservationGroupCfg(terms=teacher_perc_terms, concatenate_terms=True,
                                                               enable_corruption=False)
    if teacher_stk_terms is not None:
        cfg.observations["teacher_stk"] = ObservationGroupCfg(terms=teacher_stk_terms, concatenate_terms=True,
                                                              enable_corruption=False)
    if os.environ.get("HNU_DR", "0") == "1":  # robustness: dynamics randomisation + latency on the student inputs
        from . import add_robust_dr
        add_robust_dr(cfg, ("student", "student_ref", "student_nolatent"))
    if not play:  # recovery training: larger start-state perturbations and pushes during DAgger, so the
        # students visit more "after a mistake" states and the teacher shows how to recover (training only)
        rs = float(os.environ.get("HNU_RESET_SCALE", "1"))
        if rs != 1:
            m = cfg.commands["motion"]
            if m.pose_range:  # z keeps its (stake-safe) range
                m.pose_range = {k: (v if k == "z" else (v[0] * rs, v[1] * rs)) for k, v in m.pose_range.items()}
            if m.velocity_range:
                m.velocity_range = {k: (v[0] * rs, v[1] * rs) for k, v in m.velocity_range.items()}
            m.joint_position_range = (m.joint_position_range[0] * rs, m.joint_position_range[1] * rs)
        from . import add_robustness_terms
        add_robustness_terms(cfg)  # HNU_PUSH_SCALE (the slip term is a reward: no effect in DAgger)
    # HNU_EP_LEN: longer episodes for the slow rope release (up to 12 s) before the motion
    cfg.episode_length_s = float(os.environ.get("HNU_EP_LEN", "10")) if not play else cfg.episode_length_s
    return cfg


class ExpertActor(torch.nn.Module):
    """Deterministic actor from an rsl_rl checkpoint (obs normalizer + ELU MLP)."""

    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)["actor_state_dict"]
        self.mean = sd["obs_normalizer._mean"]
        self.std = sd["obs_normalizer._std"]
        layers = []
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
        for n, i in enumerate(idx):
            w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if n < len(idx) - 1:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers).to(device)

    @torch.no_grad()
    def forward(self, obs):
        return self.mlp((obs - self.mean) / (self.std + 1e-2))


class MoETeacher(torch.nn.Module):
    """Three action experts behind a hard router on the task id.  `groups` maps an action to the observation group
    its expert reads (default "teacher"; the bridge expert reads "teacher_perc"); then forward() takes the obs dict."""

    def __init__(self, ckpts: dict[str, str], device: str, groups: dict[str, str] | None = None):
        super().__init__()
        self.experts = torch.nn.ModuleList([ExpertActor(ckpts[t], device) for t in TASKS])
        self.groups = [(groups or {}).get(t, "teacher") for t in TASKS]

    @torch.no_grad()
    def forward(self, teacher_obs, task_onehot):
        if isinstance(teacher_obs, dict):
            out = torch.stack([e(teacher_obs[g]) for e, g in zip(self.experts, self.groups)], 1)
        else:
            out = torch.stack([e(teacher_obs) for e in self.experts], 1)  # [B, K, A]
        return (out * task_onehot.unsqueeze(-1)).sum(1)


def scene_teacher_groups(scene: str) -> dict[str, str] | None:
    if scene == "bridge":
        return {"CrouchedTraverse": "teacher_perc"}
    if scene == "geom":
        return {"CrouchedTraverse": "teacher_perc", "ThreeStepPoleRun": "teacher_stk", "FiveStepPoleRun": "teacher_stk"}
    return None
