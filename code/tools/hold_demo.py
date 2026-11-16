"""demo: start from rest without a rope -- the robot is set down in the frame-0 pose (or handed over from the
walking controller), the policy keeps the clock at frame 0 for --hold s (standing still on its own), then the motion
runs.  Two policies side by side in the training simulator (mjlab), same start state:
  teacher banks:  --left "label=<bank file or 'cur'>"  --right "label=<bank file>"   (privileged experts)
  students:       --left "label=student:<student.pt>" ...
  MUJOCO_GL=egl python tools/hold_demo.py --clip CrouchedTraverse_024 --hold 3 --start rest --left ... --right ... --out x.mp4
No CJK text on the videos (user, ).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import hnu_tasks  # noqa: F401
from hnu_tasks import TASKS
from hnu_tasks.distill import MoETeacher, hnu_distill_env_cfg, scene_teacher_groups
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv

from distill_train import student_from_ckpt

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
Z_OFF = (0, 5, 10, 15, 20, 25, 30, 40, 50)


def bank(spec: str) -> dict:
    toks = open(spec).read().split() if os.path.exists(spec) else spec.split()
    return {t: toks[toks.index(f"--ckpt-{t}") + 1] for t in TASKS}


def rollout(args, label, policy, size=(720, 640)):
    os.environ.update(HNU_REST_FRAC="1", HNU_STEP_FRAC="1" if args.start == "step" else "0",
                      HNU_HOLD=f"{args.hold},{args.hold}", HNU_HOLD_P="1", HNU_REST_XY="0.0", HNU_REST_YAW="0")
    kind, _, path = policy.partition(":") if policy.startswith("student:") else ("teacher", "", policy)
    ck = torch.load(path, map_location="cuda:0", weights_only=False) if kind == "student" else None
    cfg = hnu_distill_env_cfg(args.latent_dir, play=True, z_offsets=Z_OFF, scene="geom", hist=ck.get("hist", 20) if ck else 20,
                              lin_vel=ck.get("lin_vel", True) if ck else True)
    cmd = cfg.commands["motion"]
    names = [os.path.basename(f).replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
    cmd.fixed_clip = names.index(args.clip)
    cmd.sampling_mode = "start"
    cfg.scene.num_envs = 1
    cfg.viewer.height, cfg.viewer.width = size
    cfg.viewer.distance = 4.0
    from mjlab.managers.termination_manager import TerminationTermCfg

    def fall(env):
        d = env.scene["robot"].data
        h = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return (h < 0.30) | (d.projected_gravity_b[:, 2] > -float(np.cos(np.radians(85.0))))
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "bad_state")}
    cfg.terminations["fall"] = TerminationTermCfg(func=fall)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    obs, _ = env.reset()
    if kind == "student":
        st = student_from_ckpt(ck, env.action_manager.total_action_dim, "cuda:0")
        act = lambda o: st(o[ck.get("obs_group", "student")])  # noqa: E731
    else:
        pol = MoETeacher(bank(path), "cuda:0", scene_teacher_groups("geom"))
        act = lambda o: pol(o, o["task"])  # noqa: E731
    c = env.command_manager.get_term("motion")
    T = int(c.motion.clip_len[cmd.fixed_clip])
    H = int(c.hold0[0])
    font = ImageFont.truetype(FONT, 20)
    frames, status = [], "Completed"
    with torch.inference_mode():
        for t in range(T - 1 + H):
            obs, _, term, trunc, _ = env.step(act(obs))
            if term.any():
                status = f"FELL at t = {t / 50:.1f} s"
                break
            img = Image.fromarray(env.render())
            d = ImageDraw.Draw(img)
            d.text((12, 10), label, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            phase = f"standing still, clock held at frame 0 ({t / 50:.1f} / {H / 50:.0f} s)" if t < H else \
                f"motion running, frame {t - H + 1}"
            d.text((12, 38), f"{args.clip}  t = {t / 50:.1f} s", font=font, fill=(255, 255, 255), stroke_width=2,
                   stroke_fill=(0, 0, 0))
            d.text((12, 66), phase, font=font, fill=(255, 230, 120) if t < H else (150, 220, 255), stroke_width=2,
                   stroke_fill=(0, 0, 0))
            frames.append(np.asarray(img))
    last = Image.fromarray(frames[-1].copy() if frames else np.zeros((*size, 3), np.uint8))
    ImageDraw.Draw(last).text((12, 96), status, font=font, fill=(255, 90, 90) if "FELL" in status else (90, 255, 90),
                              stroke_width=2, stroke_fill=(0, 0, 0))
    frames += [np.asarray(last)] * 50
    env.close()
    return frames, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--start", default="rest", choices=["rest", "step"])
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--left", required=True, help="label=policy")
    ap.add_argument("--right", required=True, help="label=policy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fl, sl = rollout(args, *args.left.split("=", 1))
    fr, sr = rollout(args, *args.right.split("=", 1))
    n = max(len(fl), len(fr))
    fl += [fl[-1]] * (n - len(fl))
    fr += [fr[-1]] * (n - len(fr))
    frames = [np.concatenate([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b], 1) for a, b in zip(fl, fr)]
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=50, codec="libx264", quality=7)
    print(f"{args.clip} {args.start} hold {args.hold}: left {sl}, right {sr} -> {args.out}")


if __name__ == "__main__":
    main()
