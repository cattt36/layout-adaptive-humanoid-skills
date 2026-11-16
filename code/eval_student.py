"""Evaluate a distilled student (or the MoE teacher) on every clip of all three actions from the first frame.

  python eval_student.py --student <student_N.pt> --latent-dir <z_tables/Zk> [--push] --per-clip 4 --out <dir>
  python eval_student.py --teacher --ckpt-... (MoE teacher baseline)
Reports success rates per action and separately for train / val / test (unseen-subject) trials.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import hnu_tasks  # noqa: F401
from hnu_tasks import TASKS
from hnu_tasks.distill import MoETeacher, hnu_distill_env_cfg, scene_teacher_groups
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from distill_train import student_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=None)
    ap.add_argument("--teacher", action="store_true")
    for t in TASKS:
        ap.add_argument(f"--ckpt-{t}", default=None)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--per-clip", type=int, default=4)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--scene", default=None, choices=[None, "flat", "bridge", "geom"], help="default: from the student ckpt")
    ap.add_argument("--z-offsets", default=None, help="teacher eval only: z layout of the env (default 0,25)")
    args = ap.parse_args()
    device = "cuda:0"
    ck = None if args.teacher else torch.load(args.student, map_location=device, weights_only=False)
    z_offsets = tuple(ck.get("z_offsets", (0, 25))) if ck else \
        (tuple(int(x) for x in args.z_offsets.split(",")) if args.z_offsets else (0, 25))
    scene = args.scene or (ck.get("scene", "flat") if ck else "flat")
    cfg = hnu_distill_env_cfg(args.latent_dir, play=True, z_offsets=z_offsets, scene=scene,
                              hist=ck.get("hist", 5) if ck else 5, lin_vel=ck.get("lin_vel", False) if ck else False)
    cmd = cfg.commands["motion"]
    cmd.start_clip_mode = "env_mod"
    n_clips = len(resolve_motion_files(cmd.motion_file))
    cfg.scene.num_envs = n_clips * args.per_clip
    if args.push:
        cfg.events["push_robot"] = load_env_cfg("HNU-Track-FiveStepPoleRun").events["push_robot"]
    # success criterion.  "task" (paper definition, ): the motion is completed without a fall -- pelvis
    # below 0.30 m or body tilt > 85 deg ends the episode as a failure (the earlier 0.40 m / 57 deg flagged
    # recoverable forward leans in the bridge crouch); missed / edge stakes, stumbles that are
    # recovered and contact with the bridge (incl. a supporting hand) are NOT failures; at the end the pelvis must be
    # within 1 m of the reference end point.  Stake hits and foot-to-centre distance are recorded as secondary
    # metrics.  "track" = the reference-tracking terminations of the training environment (secondary).
    crit = os.environ.get("HNU_EVAL_CRIT", "track")
    if crit == "task":
        from mjlab.managers.termination_manager import TerminationTermCfg

        def fall(env):
            d = env.scene["robot"].data
            h = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            return (h < 0.30) | (d.projected_gravity_b[:, 2] > -float(np.cos(np.radians(85.0))))

        cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "bad_state")}
        cfg.terminations["fall"] = TerminationTermCfg(func=fall)
    # foot <-> stake contact sensors (success criterion v2): feet_ground_contact only sees the terrain.
    # Measurement only -- verified not to change the rollouts (tools/sens_ab.sh).
    from hnu_tasks.stakes import N_STAKES as _NS
    stake_sens = all(f"stake{i}" in cfg.scene.entities for i in range(_NS))
    if stake_sens:
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(ContactSensorCfg(
            name=f"foot_stake{i}",
            primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern=r"^stake$", entity=f"stake{i}"),
            fields=("found", "force"), reduce="netforce", num_slots=1) for i in range(_NS))
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset()
    if args.teacher:
        pol = MoETeacher({t: getattr(args, f"ckpt_{t}") for t in TASKS}, device, scene_teacher_groups(scene))
        act_fn = lambda o: pol(o, o["task"])  # noqa: E731
    else:
        st = student_from_ckpt(ck, env.action_manager.total_action_dim, device)
        grp = ck.get("obs_group", "student")
        act_fn = lambda o: st(o[grp])  # noqa: E731
    m = env.command_manager.get_term("motion").motion
    B = env.num_envs
    clip_of_env = torch.arange(B, device=device) % n_clips
    clip_len = m.clip_len[clip_of_env]
    active = torch.ones(B, dtype=torch.bool, device=device)
    success = torch.zeros(B, dtype=torch.bool, device=device)
    err_sum = torch.zeros(B, device=device)
    err_n = torch.zeros(B, device=device)
    cmd_term = env.command_manager.get_term("motion")
    touched = torch.zeros(B, dtype=torch.bool, device=device)
    # execution quality (while the episode is active): landing impact, loading rate, foot slip in contact,
    # actuator saturation, contact-state mismatch w.r.t. the reference contact labels
    robot = env.scene["robot"]
    feet_sensor = env.scene.sensors["feet_ground_contact"]
    fid = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    import mujoco as _mj
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as _spec
    _m = _spec().compile()
    bw = float(_m.body_mass.sum()) * 9.81
    import re as _re
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as _gc
    _acts = [v for v in vars(_gc).values() if hasattr(v, "target_names_expr") and hasattr(v, "effort_limit")]
    _lim = []
    for jn in robot.joint_names:  # per-joint effort limit from the G1 actuator groups
        _lim.append(next((float(a.effort_limit) for a in _acts
                          if any(_re.fullmatch(p, jn) for p in a.target_names_expr)), float("inf")))
    lim = torch.tensor(_lim, device=device, dtype=torch.float32)
    q = {k: torch.zeros(B, device=device) for k in ("peak_force", "max_rate", "slip_sum", "slip_n", "sat_sum", "mis_sum", "q_n")}
    prev_f = None
    dt = env.step_dt
    # secondary stake metrics: per stake, share of its core stance window in which the reference foot is on the
    # stake top (within the tolerance radius, above the top) and the mean foot-to-centre distance in that window
    from hnu_tasks.stakes import N_STAKES, STAKE_H, STAKE_TOL
    sw = getattr(cmd_term, "stake_windows", None)
    st_win = torch.zeros(B, N_STAKES, device=device)
    st_on = torch.zeros(B, N_STAKES, device=device)
    st_dist = torch.zeros(B, N_STAKES, device=device)
    # simulator check: MuJoCo-Warp loses sphere-cylinder contacts once a foot sphere is deep inside a
    # stake (CPU MuJoCo still reports them) -> the foot falls through the stake.  Flag rollouts in which a foot sphere
    # centre is > 2 cm inside a stake volume (horizontal and vertical).
    pass_through = torch.zeros(B, dtype=torch.bool, device=device)
    if stake_sens:
        import mujoco as _mjc
        _m = env.sim.mj_model
        _gn = lambda g: _mjc.mj_id2name(_m, _mjc.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
        foot_geoms = [g for g in range(_m.ngeom) if _gn(g).startswith(("robot/left_foot", "robot/right_foot"))]
        stake_geoms = [g for g in range(_m.ngeom) if _gn(g).endswith("/stake_collision")]
        # shape-independent volume: inscribed radius / half height from the stake definition (stakes.py)
        from hnu_tasks.stakes import STAKE_SHAPE as _SS
        stake_hh = 0.5 * STAKE_H
        stake_r = 0.15 * (np.cos(np.pi / int(_SS[4:] or 16)) if _SS.startswith("mesh") else
                          (0.5 * np.sqrt(np.pi) if _SS == "box" else 1.0))
    stepped = torch.zeros(B, N_STAKES, dtype=torch.bool, device=device)  # v2: foot on the stake (top / edge) at any time
    stood_full = torch.zeros(B, N_STAKES, dtype=torch.bool, device=device)  # ... touching the stake but not the floor
    stepped_rf = torch.zeros(B, N_STAKES, dtype=torch.bool, device=device)  # experiment B: stepped on by the REFERENCE foot
    near_dist = torch.full((B, N_STAKES), 9.0, device=device)  # diagnostics: closest loaded foot to each stake centre
    near_d = torch.zeros(B, N_STAKES, 2, device=device)  # ... its (along-course, across) offset
    fidx = [cmd_term.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    ar = torch.arange(B, device=device)
    # completion uses the state of the PREVIOUS step: at the last frame the command resamples and the robot is
    # teleported to the start of the next clip inside env.step
    prev_rob = torch.zeros(B, 2, device=device)
    prev_ref = torch.zeros(B, 2, device=device)
    fail_frame = torch.full((B,), -1, device=device, dtype=torch.long)  # clip frame of the termination (-1 = none)
    prev_frame = cmd_term.time_steps - m.clip_start[cmd_term.clip_id]
    prev_state = torch.zeros(B, 4, device=device)  # robot root xyz (env frame) and projected gravity z, previous step
    fail_state = torch.zeros(B, 4, device=device)
    touch_frame = torch.full((B,), -1, device=device, dtype=torch.long)  # first bridge contact (clip frame)
    # semantic consistency of the executed motion (pre-registered secondary experiment B, HNU_EVAL_SEM=1): a fixed,
    # level-independent rule-based evaluator on foot touchdowns (no latent encoder involved)
    SEM = bool(os.environ.get("HNU_EVAL_SEM"))
    KTD = 64
    td_rob = torch.full((B, 2, KTD), float("nan"), device=device)
    td_ref = torch.full((B, 2, KTD), float("nan"), device=device)
    n_rob = torch.zeros(B, 2, dtype=torch.long, device=device)
    n_ref = torch.zeros(B, 2, dtype=torch.long, device=device)
    prev_inc = torch.ones(B, 2, dtype=torch.bool, device=device)  # standing start: no touchdown at frame 0
    prev_refc = torch.ones(B, 2, dtype=torch.bool, device=device)
    # Q_exec along the closed-loop rollouts (exploratory item 5): HNU_EVAL_QEXEC = comma-separated
    # quality_net.pt files (stage5_quality/quality.py).  Measurement only -- each net reads the student observation the
    # action is computed from, nothing is fed back.  Per step and rollout the isotonic-calibrated P(no failure in the
    # next 1 s) of every net (NaN once the rollout has ended) -> <out>/qexec<tag>[_push].npz
    QX = [p for p in os.environ.get("HNU_EVAL_QEXEC", "").split(",") if p]
    if QX:
        import importlib.util as _ilu
        _qspec = _ilu.spec_from_file_location("_quality", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                      "stage5_quality", "quality.py"))
        _qm = _ilu.module_from_spec(_qspec)
        _qspec.loader.exec_module(_qm)
        qnets, qiso = [], []
        for p in QX:
            qc = torch.load(p, map_location=device, weights_only=False)
            qn_ = _qm.QualityNet(qc["dim"]).to(device)
            qn_.load_state_dict(qc["state"])
            qnets.append(qn_.eval())
            qiso.append(tuple(np.asarray(v, dtype=np.float64) for v in qc["isotonic"]))
        assert obs["student"].shape[1] == qc["dim"], (obs["student"].shape, qc["dim"])
        q_trace, fr_trace = [], []
        fell = torch.zeros(B, dtype=torch.bool, device=device)  # terminated (not time-out), while active
    # diagnostic (HNU_EVAL_FLOOR=1): floor contacts of the robot's feet during the stake run that the
    # reference does not have (no reference foot at floor level at that step)
    FLOOR = bool(os.environ.get("HNU_EVAL_FLOOR"))
    fl_g, fl_s, fl_r = [], [], []
    with torch.inference_mode():
        hold0 = getattr(cmd_term, "hold0", None)  # start from rest with a held clock (HNU_REST_FRAC / HNU_HOLD)
        hold0 = hold0.clone() if hold0 is not None else torch.zeros(B, dtype=torch.long, device=device)
        for step in range(int(m.clip_len.max() + hold0.max())):
            if QX:
                xq = obs["student"].float()
                qs = np.stack([np.interp(torch.sigmoid(n_(xq)).cpu().numpy(), kx, ky) for n_, (kx, ky) in zip(qnets, qiso)], 1)
                q_trace.append(np.where(active.cpu().numpy()[:, None], qs, np.nan).astype(np.float32))
                fr_trace.append((cmd_term.time_steps - m.clip_start[cmd_term.clip_id]).to(torch.int32).cpu().numpy())
            obs, _, term, trunc, _ = env.step(act_fn(obs))
            if scene in ("bridge", "geom"):
                touched |= hnu_tasks.bridge_touch(env) & active
            err_sum += torch.where(active, cmd_term.metrics["error_body_pos"], 0.0)
            err_n += active.float()
            a = active.float()
            fz = feet_sensor.data.force.reshape(B, 2, -1)[..., -1].abs() / bw  # [B, 2] vertical load (BW)
            q["peak_force"] = torch.maximum(q["peak_force"], fz.max(1).values * a)
            if prev_f is not None:
                q["max_rate"] = torch.maximum(q["max_rate"], ((fz - prev_f).clamp(min=0).max(1).values / dt) * a)
            prev_f = fz
            inc = fz > 0.05
            vxy = robot.data.body_link_lin_vel_w[:, fid, :2].norm(dim=-1)
            q["slip_sum"] += (vxy * inc).sum(1) * a
            q["slip_n"] += inc.float().sum(1) * a
            tau = robot.data.qfrc_actuator[:, robot.data.qfrc_actuator.shape[1] - len(lim):].abs()
            q["sat_sum"] += (tau >= 0.98 * lim).float().mean(1) * a
            ref_c = cmd_term.ref_foot_contact.float() > 0.5
            q["mis_sum"] += (ref_c != inc).float().mean(1) * a
            q["q_n"] += a
            if SEM:  # touchdown events (contact rising edges) of robot and reference, per foot, while active
                fr = (cmd_term.time_steps - m.clip_start[cmd_term.clip_id]).float()
                for buf, cnt, now, before in ((td_rob, n_rob, inc, prev_inc), (td_ref, n_ref, ref_c, prev_refc)):
                    rise = now & ~before & active[:, None]
                    for ft in range(2):
                        e = rise[:, ft] & (cnt[:, ft] < KTD)
                        idx = cnt[:, ft].clamp(max=KTD - 1)
                        buf[ar[e], ft, idx[e]] = fr[e]
                        cnt[:, ft] += e.long()
                prev_inc, prev_refc = inc.clone(), ref_c.clone()
            if sw is not None:
                foot, f0, f1, sxy = sw
                frame = cmd_term.time_steps - m.clip_start[cmd_term.clip_id]
                feet_p = cmd_term.robot_body_pos_w[:, fidx] - env.scene.env_origins[:, None]
                for i in range(N_STAKES):
                    k = foot[cmd_term.clip_id, i]
                    win = active & (k >= 0) & (frame >= f0[cmd_term.clip_id, i]) & (frame <= f1[cmd_term.clip_id, i])
                    p = feet_p[ar, k.clamp(min=0)]
                    cxy = sxy[cmd_term.clip_id, i]
                    if getattr(cmd_term, "stake_offset", None) is not None:
                        cxy = cxy + cmd_term.stake_offset[:, i]
                    dxy = torch.norm(p[:, :2] - cxy, dim=-1)
                    on = (dxy <= STAKE_TOL) & (p[:, 2] >= STAKE_H - 0.02)
                    st_win[:, i] += win.float()
                    st_on[:, i] += (win & on).float()
                    st_dist[:, i] += torch.where(win, dxy, 0.0)
                    if stake_sens:  # v2: a foot physically on stake i (top or edge), ankle above ground level
                        fnd = env.scene.sensors[f"foot_stake{i}"].data.found.reshape(B, -1)[:, :2] > 0
                        raised = feet_p[..., 2] >= 0.07
                        gnd = feet_sensor.data.found.reshape(B, -1)[:, :2] > 0
                        stepped[:, i] |= (fnd & raised).any(1) & active
                        stood_full[:, i] |= (fnd & raised & ~gnd).any(1) & active
                        kk = k.clamp(min=0)
                        stepped_rf[:, i] |= fnd[ar, kk] & raised[ar, kk] & active & (k >= 0)
                        # diagnostics: closest LOADED foot (floor or stake contact) to the centre, around the window
                        wide = active & (k >= 0) & (frame >= f0[cmd_term.clip_id, i] - 15) & (frame <= f1[cmd_term.clip_id, i] + 15)
                        loaded = fnd | gnd
                        dd = torch.norm(feet_p[..., :2] - cxy[:, None], dim=-1)  # [B, 2]
                        dd = torch.where(loaded, dd, torch.full_like(dd, 9.0))
                        dmin, kmin = dd.min(1)
                        better = wide & (dmin < near_dist[:, i])
                        rel = feet_p[ar, kmin, :2] - cxy
                        u = cmd_term._course_dirs()[cmd_term.clip_id] if hasattr(cmd_term, "_course_dirs") else rel * 0
                        along = (rel * u).sum(-1)
                        across = rel[:, 0] * -u[:, 1] + rel[:, 1] * u[:, 0]
                        near_dist[:, i] = torch.where(better, dmin, near_dist[:, i])
                        near_d[:, i] = torch.where(better[:, None], torch.stack([along, across], -1), near_d[:, i])
            if FLOOR and sw is not None and stake_sens:  # per step: robot floor / stake contact, reference foot low
                gnd_ = feet_sensor.data.found.reshape(B, -1)[:, :2] > 0
                raised_ = feet_p[..., 2] >= 0.07
                stk_ = torch.zeros_like(gnd_)
                for i in range(N_STAKES):
                    stk_ |= (env.scene.sensors[f"foot_stake{i}"].data.found.reshape(B, -1)[:, :2] > 0) & raised_
                refz = cmd_term.body_pos_w[:, fidx, 2] - env.scene.env_origins[:, None, 2]
                fl_g.append((gnd_ & active[:, None]).cpu().numpy())
                fl_s.append((stk_ & active[:, None]).cpu().numpy())
                fl_r.append(((refz < 0.06) & active[:, None]).cpu().numpy())
            if stake_sens and foot_geoms and stake_geoms:
                gx = env.sim.data.geom_xpos  # [B, ngeom, 3]
                fs = gx[:, foot_geoms]  # [B, F, 3]
                sc = gx[:, stake_geoms]  # [B, S, 3]
                dxy = torch.norm(fs[:, :, None, :2] - sc[:, None, :, :2], dim=-1)  # [B, F, S]
                inside = (dxy < stake_r - 0.02) & (fs[:, :, None, 2] < sc[:, None, :, 2] + stake_hh - 0.02) \
                    & (fs[:, :, None, 2] > sc[:, None, :, 2] - stake_hh)
                pass_through |= inside.flatten(1).any(1) & active
            done = (term | trunc) & active
            if QX:
                fell |= term & active
            fail_frame = torch.where(done, prev_frame + 1, fail_frame)  # terminated envs are already reset here
            fail_state = torch.where(done[:, None], prev_state, fail_state)
            touch_frame = torch.where((touch_frame < 0) & touched & active, prev_frame + 1, touch_frame)
            if os.environ.get("HNU_EVAL_DEBUG") and (step < 3 or step % 100 == 0):
                d_ = robot.data
                print(f"[debug] step {step}: active {int(active.sum())}, done {int(done.sum())}, root z mean "
                      f"{float((d_.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean()):.3f}, pg_z mean "
                      f"{float(d_.projected_gravity_b[:, 2].mean()):.3f}, term {int(term.sum())}, trunc {int(trunc.sum())}")
            active &= ~done
            fin = active & (step + 1 >= clip_len - 1 + hold0)
            if os.environ.get("HNU_EVAL_DEBUG") and fin.any():
                dist = torch.norm(prev_rob - prev_ref, dim=-1)
                print(f"[debug] step {step}: {int(fin.sum())} finishing, end distance mean {float(dist[fin].mean()):.2f} m")
            if crit == "task":  # completion: the pelvis ends within 1 m of the reference end point
                fin_ok = fin & (torch.norm(prev_rob - prev_ref, dim=-1) < 1.0)
                success |= fin_ok
            else:
                success |= fin
            active &= ~fin
            prev_rob = robot.data.root_link_pos_w[:, :2].clone()
            prev_ref = cmd_term.body_pos_w[:, 0, :2].clone()
            prev_frame = cmd_term.time_steps - m.clip_start[cmd_term.clip_id]
            prev_state = torch.cat((robot.data.root_link_pos_w - env.scene.env_origins,
                                    robot.data.projected_gravity_b[:, 2:3]), dim=-1)
            if not active.any():
                break
    succ = success.cpu().numpy()
    nocontact = succ & ~touched.cpu().numpy()
    err = (err_sum / err_n.clamp(min=1)).cpu().numpy()
    qn = q["q_n"].clamp(min=1)
    qual = dict(peak_force_bw=q["peak_force"], max_loading_rate_bw_s=q["max_rate"],
                foot_slip_m_s=q["slip_sum"] / q["slip_n"].clamp(min=1), torque_sat_frac=q["sat_sum"] / qn,
                contact_mismatch=q["mis_sum"] / qn)
    has = st_win > 0
    n_st = has.sum(1).clamp(min=1)
    # every stake of the clip hit (foot on top >= 50 % of its core window) AND the episode successful
    n_clip = (sw[0][clip_of_env] >= 0).sum(1) if sw is not None else torch.zeros(B, device=device, dtype=torch.long)
    n_hit = (((st_on / st_win.clamp(min=1)) >= 0.5) & has).sum(1)
    # LEGACY (flawed, ): ankle height >= 0.08 m within the radius also counts a foot hovering over the stake;
    # kept for comparison only -- use stakes_all_stepped / success_v2
    qual["stakes_all_hit"] = torch.where(n_clip > 0, ((n_hit == n_clip) & success).float(),
                                         torch.full_like(n_st, float("nan"), dtype=torch.float32))
    qual["stake_centre_dist_m"] = torch.where(has.any(1), (st_dist / st_win.clamp(min=1)).sum(1) / n_st,
                                              torch.full_like(n_st, float("nan"), dtype=torch.float32))
    # success criterion v2 (user definition after the video review): ThreeStepPoleRun / FiveStepPoleRun additionally require that
    # EVERY stake was stepped on (top or edge; any time, rhythm does not matter) -- passing a stake over the floor is a
    # failure even when the motion is completed.  CrouchedTraverse unchanged (bridge contact allowed, no fall).
    valid = (sw[0][clip_of_env] >= 0) if sw is not None else torch.zeros(B, N_STAKES, dtype=torch.bool, device=device)
    all_stepped = (stepped | ~valid).all(1)
    success_v2 = success & all_stepped if stake_sens else success.clone()
    nan_t = torch.full((B,), float("nan"), device=device)
    qual["stakes_stepped_frac"] = torch.where(n_clip > 0, (stepped & valid).sum(1).float() / n_clip.clamp(min=1), nan_t)
    qual["stakes_all_stepped"] = torch.where(n_clip > 0, all_stepped.float(), nan_t)
    qual["stakes_all_full"] = torch.where(n_clip > 0, (stood_full | ~valid).all(1).float(), nan_t)
    qual["success_v2"] = success_v2.float()
    qual["stake_pass_through"] = torch.where(n_clip > 0, pass_through.float(), nan_t)  # simulator artefact rate
    if QX:
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, f"qexec{args.tag}{'_push' if args.push else ''}.npz"),
                            q=np.stack(q_trace), frame=np.stack(fr_trace), nets=np.array(QX), clip=clip_of_env.cpu().numpy(),
                            names=np.array(m.names), success=success.cpu().numpy(), success_v2=success_v2.cpu().numpy(),
                            fell=fell.cpu().numpy(), fail_frame=fail_frame.cpu().numpy())
    elif os.environ.get("HNU_EVAL_EPISODES"):  # per-rollout outcomes without Q_exec nets (e.g. the ZR baseline)
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, f"qexec{args.tag}{'_push' if args.push else ''}.npz"),
                            clip=clip_of_env.cpu().numpy(), names=np.array(m.names), success=success.cpu().numpy(),
                            success_v2=success_v2.cpu().numpy(), fell=np.full(B, np.nan), fail_frame=fail_frame.cpu().numpy())
    qual = {k: v.cpu().numpy() for k, v in qual.items()}
    if FLOOR and fl_g:
        Gf, Sf, Rf = np.stack(fl_g), np.stack(fl_s), np.stack(fl_r)  # [T, B, 2]
        ev, frac, rfrac, dur = (np.full(B, np.nan) for _ in range(4))
        for b in range(B):
            on = np.where(Sf[:, b].any(1))[0]
            if len(on) < 2:
                continue
            lo, hi = on[0], on[-1]
            g = Gf[lo:hi + 1, b] & ~Rf[lo:hi + 1, b].any(1, keepdims=True)  # robot foot on the floor, reference foot not
            ev[b] = float(sum(((g[1:, k] & ~g[:-1, k]).sum() + g[0, k]) for k in range(2)))
            frac[b] = float(g.any(1).mean())
            dur[b] = float(g.any(1).sum() / 50.0)
            rfrac[b] = float(Rf[lo:hi + 1, b].any(1).mean())
        qual.update(floor_touch_events=ev, floor_touch_frac=frac, floor_touch_s=dur, ref_floor_frac=rfrac,
                    floor_touch_any=np.where(np.isnan(ev), np.nan, (ev > 0).astype(float)))
    sem_env = None
    if SEM:  # per rollout: median |touchdown time error| (s) of matched touchdowns (same foot, within 0.3 s), share of
        # reference touchdowns without a robot touchdown, stakes all hit with the reference foot
        tr, tf = td_rob.cpu().numpy(), td_ref.cpu().numpy()
        dev, miss = np.full(B, np.nan), np.full(B, np.nan)
        for i in range(B):
            errs, n_tot, n_miss = [], 0, 0
            for ft in range(2):
                r, b = tf[i, ft][~np.isnan(tf[i, ft])], tr[i, ft][~np.isnan(tr[i, ft])]
                for t in r:
                    n_tot += 1
                    dmin = np.abs(b - t).min() if len(b) else np.inf
                    if dmin <= 15:
                        errs.append(dmin / 50.0)
                    else:
                        n_miss += 1
            dev[i] = np.median(errs) if errs else np.nan
            miss[i] = n_miss / n_tot if n_tot else np.nan
        allhit = ((n_hit == n_clip) & (n_clip > 0)).cpu().numpy() if sw is not None else np.zeros(B, bool)
        has_st = (n_clip > 0).cpu().numpy() if sw is not None else np.zeros(B, bool)
        # stakes_ok: LEGACY window/height rule (hovering counts); stakes_foot_ok: contact-based (criterion v2) -- every
        # stake touched (top or edge, ankle raised) by the foot the reference uses on it, at any time
        foot_ok = (stepped_rf | ~valid).all(1).cpu().numpy() if stake_sens else np.zeros(B, bool)
        sem_env = dict(tempo_dev_s=dev, td_miss=miss, contact_agree=1.0 - qual["contact_mismatch"],
                       stakes_ok=np.where(has_st, allhit.astype(float), np.nan),
                       stakes_foot_ok=np.where(has_st & stake_sens, foot_ok.astype(float), np.nan))
    # split labels: the latent subject split by default; HNU_SPLIT_JSON overrides (trial-level held-out protocol)
    split = json.load(open(os.environ.get("HNU_SPLIT_JSON") or os.path.join(args.latent_dir, "split.json")))
    where = {n: k for k in ("train", "val", "test") for n in split[k]}
    rows = []
    for c in range(n_clips):
        idx = np.arange(c, B, n_clips)
        rows.append(dict(clip=m.names[c], task=m.names[c].split("_")[0], split=where.get(m.names[c], "?"),
                         success=float(succ[idx].mean()), success_no_contact=float(nocontact[idx].mean()),
                         body_err=float(err[idx].mean()),
                         **{k: (float(np.nanmean(v[idx])) if k.startswith(("floor_", "ref_floor")) and not np.all(np.isnan(v[idx]))
                                else float(v[idx].mean())) for k, v in qual.items()}))
        if sem_env is not None:  # semantic consistency over the SUCCESSFUL rollouts of the clip (n = sem_n)
            ok = idx[succ[idx]]
            rows[-1]["sem_n"] = int(len(ok))
            for k, v in sem_env.items():
                vv = v[ok]
                rows[-1][f"sem_{k}"] = float(np.nanmean(vv)) if len(vv) and not np.all(np.isnan(vv)) else None
        if os.environ.get("HNU_EVAL_DETAIL"):  # per rollout: termination frame, per stake share of the window on top
            ff = fail_frame.cpu().numpy()
            ratio = (st_on / st_win.clamp(min=1)).cpu().numpy()
            fs = fail_state.cpu().numpy()
            tf = touch_frame.cpu().numpy()
            stp = stepped.cpu().numpy()
            nd = near_d.cpu().numpy()
            rows[-1]["detail"] = dict(fail_frame=[int(ff[i]) for i in idx], success=[bool(succ[i]) for i in idx],
                                      stepped=[[bool(x) for x in stp[i]] for i in idx],
                                      # per stake: closest loaded-foot landing to the stake centre in its window
                                      # (stake frame: along-course / across, m) -> how a stake was missed
                                      near_xy=[[[round(float(v), 3) for v in nd[i, s]] for s in range(N_STAKES)] for i in idx],
                                      fail_state=[[round(float(x), 3) for x in fs[i]] for i in idx],
                                      touch_frame=[int(tf[i]) for i in idx],
                                      stake_on=[[round(float(x), 2) for x in ratio[i]] for i in idx],
                                      stake_win=([[int(sw[1][c, i]), int(sw[2][c, i])] for i in range(N_STAKES)]
                                                 if sw is not None else None))
    summary = {}
    for t in TASKS:
        for sp in ("all", "train", "val", "test"):
            r = [x for x in rows if x["task"] == t and (sp == "all" or x["split"] == sp)]
            if r:
                summary[f"{t}/{sp}"] = dict(success=round(float(np.mean([x["success"] for x in r])), 3),
                                            success_no_contact=round(float(np.mean([x["success_no_contact"] for x in r])), 3),
                                            body_err=round(float(np.mean([x["body_err"] for x in r])), 3), n=len(r),
                                            **{k: round(float(np.nanmean([x[k] for x in r])), 4) if not all(np.isnan(x[k]) for x in r) else None for k in qual})
    out = dict(policy="teacher" if args.teacher else args.student, latent_dir=args.latent_dir, push=args.push, scene=scene,
               per_clip=args.per_clip, summary=summary, clips=rows)
    os.makedirs(args.out, exist_ok=True)
    name = f"eval_{'teacher' if args.teacher else 'student'}{args.tag}{'_push' if args.push else ''}.json"
    json.dump(out, open(os.path.join(args.out, name), "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:24s} success {v['success']:.3f}  no-contact {v['success_no_contact']:.3f}  "
              f"body_err {v['body_err']:.3f}  n={v['n']}")


if __name__ == "__main__":
    main()
