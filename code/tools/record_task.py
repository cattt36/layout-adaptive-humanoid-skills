"""Video of student rollouts on one clip under the TASK success criterion (fall = pelvis < 0.30 m or tilt > 85 deg;
success = no fall and the pelvis ends within 1 m of the reference end), with per-stake hit / miss labels
(reference foot on the stake top >= 50 % of the stake's core stance window, as in eval_student.py).
For every student, rollouts are repeated (new reset randomisation) until one matches the wanted outcome:
  any | success_miss (success with >= 1 stake missed) | success_clean (success, all stakes hit) | fail
Students are shown side by side; a top-down inset is not used, the camera follows the robot from the side.

  python record_task.py --clip ThreeStepPoleRun_012 --out v.mp4 \
      --student L0=<pt>:<z_dir>:success_miss --student L4=<pt>:<z_dir>:success_clean
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from hnu_tasks.multi_motion import resolve_motion_files
from hnu_tasks.stakes import N_STAKES, STAKE_H, STAKE_TOL
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg

from distill_train import student_from_ckpt


def fall(env):
    d = env.scene["robot"].data
    h = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (h < 0.30) | (d.projected_gravity_b[:, 2] > -float(np.cos(np.radians(85.0))))


def rollout(clip, pt, zdir, label, want, tries, size):
    ck = torch.load(pt, map_location="cuda:0", weights_only=False)
    cfg = hnu_distill_env_cfg(zdir, play=True, z_offsets=tuple(ck["z_offsets"]), scene=ck.get("scene", "geom"),
                              hist=ck.get("hist", 5), lin_vel=ck.get("lin_vel", False))
    cmd = cfg.commands["motion"]
    names = [os.path.basename(f).replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
    cmd.fixed_clip = names.index(clip)
    cmd.sampling_mode = "start"
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "bad_state")}
    cfg.terminations["fall"] = TerminationTermCfg(func=fall)
    cfg.scene.num_envs = 1
    cfg.viewer.height, cfg.viewer.width = size
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 3.6, -30.0, 90.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    st = student_from_ckpt(ck, env.action_manager.total_action_dim, "cuda:0")
    grp = ck.get("obs_group", "student")
    c = env.command_manager.get_term("motion")
    m = c.motion
    T = int(m.clip_len[cmd.fixed_clip])
    foot, f0, f1, sxy = c.stake_windows
    fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    font = ImageFont.load_default(size=20)
    best = None
    for k in range(tries):
        torch.manual_seed(1000 + k)
        win = np.zeros(N_STAKES)
        on = np.zeros(N_STAKES)
        frames, fell = [], None
        with torch.inference_mode():
            obs, _ = env.reset()
            for t in range(T - 1):
                obs, _, term, _, _ = env.step(st(obs[grp]))
                if term.any():
                    fell = t / 50
                    break
                fr = int(c.time_steps[0] - m.clip_start[c.clip_id[0]])
                fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
                cid = int(c.clip_id[0])
                for i in range(N_STAKES):
                    kf = int(foot[cid, i])
                    if kf >= 0 and int(f0[cid, i]) <= fr <= int(f1[cid, i]):
                        p = fp[kf]
                        d = np.linalg.norm(p[:2] - sxy[cid, i].cpu().numpy())
                        win[i] += 1
                        on[i] += float(d <= STAKE_TOL and p[2] >= STAKE_H - 0.02)
                img = Image.fromarray(env.render())
                dr = ImageDraw.Draw(img)
                dr.text((10, 8), f"{label}   {clip}   t = {t / 50:.2f} s", fill=(255, 255, 255), font=font)
                tags = []
                for i in range(N_STAKES):
                    if win[i] > 0 and fr > int(f1[cid, i]):
                        tags.append((f"S{i + 1}", on[i] / win[i] >= 0.5))
                x = 10
                for name, ok in tags:
                    dr.text((x, 34), f"{name} {'HIT' if ok else 'MISS'}", fill=(80, 255, 80) if ok else (255, 90, 90), font=font)
                    x += 95
                frames.append(np.asarray(img))
        if not frames:
            continue
        end_ok = fell is None
        if end_ok:  # completion: pelvis within 1 m of the reference end (as in eval_student)
            rob = c.robot_body_pos_w[0, 0, :2]
            ref = c.body_pos_w[0, 0, :2]
            end_ok = float(torch.norm(rob - ref)) < 1.0
        hits = [on[i] / win[i] >= 0.5 for i in range(N_STAKES) if win[i] > 0]
        n_hit, n_st = sum(hits), len(hits)
        outcome = "fail" if not end_ok else ("success_clean" if n_hit == n_st else "success_miss")
        verdict = (f"FELL at {fell:.2f} s -> FAILURE" if fell is not None else
                   ("SUCCESS" if end_ok else "did not finish -> FAILURE")) + f"   stakes hit {n_hit}/{n_st}"
        img = Image.fromarray(frames[-1].copy())
        ImageDraw.Draw(img).text((10, 60), verdict, fill=(80, 255, 80) if outcome != "fail" else (255, 90, 90), font=font)
        frames += [np.asarray(img)] * 75
        print(f"{label} try {k}: {outcome} ({verdict})")
        if best is None:
            best = (frames, outcome, verdict)
        if want == "any" or outcome == want:
            best = (frames, outcome, verdict)
            break
    env.close()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--student", action="append", required=True, help="LABEL=pt:z_dir:want")
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = []
    for s in a.student:
        label, rest = s.split("=", 1)
        pt, zdir, want = rest.split(":")
        frames, outcome, verdict = rollout(a.clip, pt, zdir, label, want, a.tries, (480, 640))
        print(f"{label}: kept {outcome} -- {verdict}")
        cols.append(frames)
    n = max(len(f) for f in cols)
    cols = [f + [f[-1]] * (n - len(f)) for f in cols]
    sep = np.full((cols[0][0].shape[0], 6, 3), 255, np.uint8)
    frames = [np.concatenate(sum([[c[i], sep] for c in cols], [])[:-1], 1) for i in range(n)]
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    imageio.mimsave(a.out, frames[::1], fps=50, codec="libx264", quality=7)
    print("->", a.out)


if __name__ == "__main__":
    main()
