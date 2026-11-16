"""Evaluate / roll out an expert tracking policy on every clip of a task.

Each environment tracks one clip from its first frame (env i -> clip i % n_clips).  An episode is a
success if the policy reaches the last frame of the clip without triggering a termination.
Per-step physics (joint torques, foot contact forces, root state, tracking errors) is saved for later
use as execution labels (robot annotations, z_quality).

  python eval_expert.py --task FiveStepPoleRun --ckpt <model.pt> --per-clip 8 [--push] [--noise] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch

import hnu_tasks  # noqa: F401
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def task_id(task, stakes=False, bridge=False):
    return f"HNU-Track-{task}" + ("-Stakes" if stakes else "") + ("-Bridge" if bridge else "")


def build(task, motion_dir, per_clip, push, noise, device, stakes=False, bridge=False):
    tid = task_id(task, stakes, bridge)
    env_cfg = load_env_cfg(tid, play=True)
    train_cfg = load_env_cfg(tid, play=False)
    cmd = env_cfg.commands["motion"]
    cmd.motion_file = motion_dir
    cmd.start_clip_mode = "env_mod"
    n_clips = len(resolve_motion_files(motion_dir))
    env_cfg.scene.num_envs = n_clips * per_clip
    if push:
        env_cfg.events["push_robot"] = train_cfg.events["push_robot"]
    if noise:
        env_cfg.observations["actor"].enable_corruption = True
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(tid).clip_actions)
    return env, wrapped, n_clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--motion-dir", default=None)
    ap.add_argument("--per-clip", type=int, default=8)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--save-rollouts", action="store_true")
    ap.add_argument("--stakes", action="store_true", help="evaluate on the per-clip stakes scene")
    ap.add_argument("--stake-json-dir", default=None,
                    help="score stake hits from these <clip>_stakes.json (also on flat ground, see --stake-top)")
    ap.add_argument("--stake-top", type=float, default=None,
                    help="stake top height used by the hit criterion (default: HNU_STAKE_H or 0.10; 0 = flat ground)")
    ap.add_argument("--bridge", action="store_true", help="CrouchedTraverse under the marker-defined bridge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    score_stakes = args.stakes or args.stake_json_dir is not None
    stake_dir = args.stake_json_dir
    stake_top = args.stake_top if args.stake_top is not None else float(os.environ.get("HNU_STAKE_H", "0.10"))
    device = "cuda:0"
    root = hnu_tasks.STAKES_ROOT if args.stakes else hnu_tasks.BRIDGE_ROOT if args.bridge else hnu_tasks.DATA_ROOT
    motion_dir = args.motion_dir or os.path.join(root, args.task)
    env, wrapped, n_clips = build(args.task, motion_dir, args.per_clip, args.push, args.noise, device, args.stakes,
                                  args.bridge)
    tid = task_id(args.task, args.stakes, args.bridge)
    runner_cls = load_runner_cls(tid) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(load_rl_cfg(tid)), device=device)
    runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    feet = env.scene.sensors["feet_ground_contact"]
    m = cmd.motion
    B = env.num_envs
    clip_of_env = torch.arange(B, device=device) % n_clips
    clip_len = m.clip_len[clip_of_env]
    T = int(m.clip_len.max())

    obs = wrapped.get_observations()
    active = torch.ones(B, dtype=torch.bool, device=device)
    success = torch.zeros(B, dtype=torch.bool, device=device)
    fail_step = torch.full((B,), -1, dtype=torch.long, device=device)
    touched = torch.zeros(B, dtype=torch.bool, device=device)  # bridge plank contact at any time
    keys =["root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel", "joint_torque",
            "feet_force", "body_pos_err", "joint_pos_err", "ref_contact", "frame"]
    rec = {k: [] for k in keys}
    with torch.inference_mode():
        for step in range(T):
            d = robot.data
            if args.save_rollouts:
                rec["root_pos"].append(d.root_link_pos_w.clone())
                rec["root_quat"].append(d.root_link_quat_w.clone())
                rec["root_lin_vel"].append(d.root_link_lin_vel_w.clone())
                rec["root_ang_vel"].append(d.root_link_ang_vel_w.clone())
                rec["joint_pos"].append(d.joint_pos.clone())
                rec["joint_vel"].append(d.joint_vel.clone())
                rec["joint_torque"].append(d.qfrc_actuator.clone())
                rec["feet_force"].append(feet.data.force.clone())
                rec["ref_contact"].append(cmd.ref_foot_contact.clone())
                rec["frame"].append((cmd.time_steps - m.clip_start[cmd.clip_id]).clone())
            rec["body_pos_err"].append(cmd.metrics["error_body_pos"].clone())
            rec["joint_pos_err"].append(cmd.metrics["error_joint_pos"].clone())
            rec.setdefault("anchor_xy_err", []).append(
                torch.norm(cmd.anchor_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2], dim=-1).clone())
            if score_stakes:
                fi = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
                rec.setdefault("feet_pos", []).append((d.body_link_pos_w[:, fi] - env.scene.env_origins[:, None]).clone())
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            if args.bridge:
                touched |= hnu_tasks.bridge_touch(env) & active
            done_now = dones.bool() & active
            fail_step[done_now] = step
            if done_now.any():  # which termination term ended the episode
                tm = env.termination_manager
                causes = rec.setdefault("_causes", {})
                for nm in tm.active_terms:
                    hit = (tm.get_term(nm).bool() & done_now)
                    if hit.any():
                        causes[nm] = causes.get(nm, 0) + int(hit.sum())
            active &= ~done_now
            finished = active & (step + 1 >= clip_len - 1)
            success |= finished
            active &= ~finished
            if not active.any():
                break

    n = len(rec["body_pos_err"])
    err = torch.stack(rec["body_pos_err"]).cpu().numpy()  # [n, B]
    jerr = torch.stack(rec["joint_pos_err"]).cpu().numpy()
    valid = np.zeros((n, B), bool)
    last = torch.where(fail_step >= 0, fail_step, clip_len - 1).cpu().numpy()
    for b in range(B):
        valid[: min(last[b] + 1, n), b] = True
    succ = success.cpu().numpy()
    axy = torch.stack(rec["anchor_xy_err"]).cpu().numpy()
    # stake hits: the stake's foot must stand on the stake top (within radius, ankle 0-0.10 m above the top) for >= 3
    # control steps inside the reference stance window (+-8 steps); a clip succeeds on stakes only if all are hit
    stake_hit = np.ones((B,), bool)
    stake_rate = np.full((B,), np.nan)
    if score_stakes:
        fp = torch.stack(rec["feet_pos"]).cpu().numpy()  # [n, B, 2, 3]
        for b in range(B):
            j = os.path.join(stake_dir or motion_dir, f"{m.names[int(clip_of_env[b])]}_stakes.json")
            if not os.path.exists(j):
                continue
            meta = json.load(open(j))
            hits = []
            for s in meta["stakes"]:
                k = 0 if s["foot"] == "L" else 1
                a = max(int(round(s["stance"][0] * 50 / meta["fps"])) - 1 - 8, 0)
                e = min(int(round(s["stance"][1] * 50 / meta["fps"])) - 1 + 8, n - 1)
                seg = fp[a:e + 1, b, k][valid[a:e + 1, b]]
                ok = (np.hypot(seg[:, 0] - s["x"], seg[:, 1] - s["y"]) < 0.18) & (seg[:, 2] > stake_top) & (seg[:, 2] < stake_top + 0.10)
                hits.append(ok.sum() >= 3)
            stake_hit[b] = bool(np.all(hits))
            stake_rate[b] = float(np.mean(hits))
    succ_strict = succ & stake_hit
    no_touch = ~touched.cpu().numpy()
    if args.bridge:  # strict bridge success: completed without touching the plank
        succ_strict = succ & no_touch
    per_clip = []
    for c in range(n_clips):
        idx = np.arange(c, B, n_clips)
        per_clip.append(dict(
            clip=m.names[c], success_rate=float(succ[idx].mean()), success_all_stakes=float(succ_strict[idx].mean()),
            stake_hit_rate=float(np.nanmean(stake_rate[idx])) if score_stakes else None,
            mean_body_pos_err=float(err[:, idx][valid[:, idx]].mean()),
            mean_anchor_xy_err=float(axy[:, idx][valid[:, idx]].mean()),
            mean_joint_pos_err=float(jerr[:, idx][valid[:, idx]].mean()),
            fail_frac_of_clip=[round(float(fail_step[i].item() / clip_len[i].item()), 2) for i in idx if fail_step[i] >= 0],
        ))
    summary = dict(task=args.task, ckpt=args.ckpt, push=args.push, noise=args.noise, per_clip_envs=args.per_clip,
                   success_rate=float(succ.mean()), success_all_stakes=float(succ_strict.mean()),
                   stake_hit_rate=float(np.nanmean(stake_rate)) if score_stakes else None,
                   bridge_contact_rate=float(1 - no_touch.mean()) if args.bridge else None,
                   success_no_contact=float(succ_strict.mean()) if args.bridge else None,
                   mean_anchor_xy_err=float(axy[valid].mean()),
                   mean_body_pos_err=float(err[valid].mean()), termination_causes=rec.pop("_causes", {}),
                   clips=per_clip)
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.task}{'_stakes' if args.stakes else ''}{'_bridge' if args.bridge else ''}{'_push' if args.push else ''}{'_noise' if args.noise else ''}"
    with open(os.path.join(args.out, f"eval_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}))
    for pc in per_clip:
        print(f"  {pc['clip']:18s} success {pc['success_rate']:.2f}  body_err {pc['mean_body_pos_err']:.3f}  fails@{pc['fail_frac_of_clip']}")
    if args.save_rollouts:
        arr = {k: torch.stack(v).cpu().numpy() for k, v in rec.items() if v}
        for c in range(n_clips):
            idx = np.arange(c, B, n_clips)
            out = {k: v[:, idx] for k, v in arr.items()}
            out["valid"] = valid[:, idx]
            out["success"] = succ[idx]
            out["fail_step"] = fail_step.cpu().numpy()[idx]
            out["joint_names"] = np.array(robot.joint_names)
            np.savez_compressed(os.path.join(args.out, f"rollout_{m.names[c]}{'_push' if args.push else ''}.npz"), **out)


if __name__ == "__main__":
    main()
