"""Record the teacher (expert policy bank) and a student on the same clip, side by side.

  python record_video.py --clip FiveStepPoleRun_022 --student <student.pt> --latent-dir <z_tables_v3/Z2> \
      --ckpt-ThreeStepPoleRun ... --ckpt-FiveStepPoleRun ... --ckpt-CrouchedTraverse ... --out video.mp4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import hnu_tasks  # noqa: F401
from hnu_tasks import TASKS
from hnu_tasks.distill import MoETeacher, hnu_distill_env_cfg
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv

from distill_train import student_from_ckpt


def rollout(args, label, act_builder, z_offsets, size=(540, 720), scene="flat", hist=5, lin_vel=False):
    cfg = hnu_distill_env_cfg(args.latent_dir, play=True, z_offsets=z_offsets, scene=scene, hist=hist, lin_vel=lin_vel)
    cmd = cfg.commands["motion"]
    names = [os.path.basename(f).replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
    cmd.fixed_clip = names.index(args.clip)
    cmd.sampling_mode = "start"
    cfg.scene.num_envs = 1
    cfg.viewer.height, cfg.viewer.width = size
    cfg.viewer.distance = 3.2
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    obs, _ = env.reset()
    act = act_builder(env)
    T = int(env.command_manager.get_term("motion").motion.clip_len[cmd.fixed_clip])
    frames, status = [], "completed"
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    with torch.inference_mode():
        for t in range(T - 1):
            obs, _, term, trunc, _ = env.step(act(obs))
            if term.any():
                status = f"FELL at {t / 50:.2f}s"
                break
            img = Image.fromarray(env.render())
            d = ImageDraw.Draw(img)
            d.text((12, 10), f"{label}", fill=(255, 255, 255), font=font)
            d.text((12, 40), f"{args.clip}   t = {t / 50:.2f} s", fill=(255, 255, 255), font=font)
            frames.append(np.asarray(img))
    last = frames[-1].copy() if frames else np.zeros((*size, 3), np.uint8)
    img = Image.fromarray(last)
    ImageDraw.Draw(img).text((12, 70), status, fill=(255, 80, 80) if "FELL" in status else (80, 255, 80), font=font)
    frames += [np.asarray(img)] * 50  # hold the final frame 1 s
    env.close()
    return frames, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--student-label", default="Student")
    ap.add_argument("--latent-dir", required=True)
    for t in TASKS:
        ap.add_argument(f"--ckpt-{t}", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"

    def teacher_builder(env):
        pol = MoETeacher({t: getattr(args, f"ckpt_{t}") for t in TASKS}, dev)
        return lambda o: pol(o["teacher"], o["task"])

    ck = torch.load(args.student, map_location=dev, weights_only=False)
    z_off = tuple(ck.get("z_offsets", (0, 25)))

    def student_builder(env):
        st = student_from_ckpt(ck, env.action_manager.total_action_dim, dev)
        grp = ck.get("obs_group", "student")
        return lambda o: st(o[grp])

    ft, st_t = rollout(args, "Teacher (expert policy bank, privileged)", teacher_builder, z_off)
    fs, st_s = rollout(args, args.student_label, student_builder, z_off)
    n = max(len(ft), len(fs))
    ft += [ft[-1]] * (n - len(ft))
    fs += [fs[-1]] * (n - len(fs))
    frames = [np.concatenate([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b], 1) for a, b in zip(ft, fs)]
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=50, codec="libx264", quality=7)
    print(f"{args.clip}: teacher {st_t}, student {st_s} -> {args.out}")


if __name__ == "__main__":
    main()
