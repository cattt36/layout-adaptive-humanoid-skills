"""Video of an expert on one clip of any registered HNU task (actor obs, deterministic), with label and final status.

  python record_expert.py <task_id e.g. HNU-Track-CrouchedTraverse-Bridge> <clip> <ckpt> <out.mp4> [label]
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import ExpertActor
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

tid, clip, ckpt, out = sys.argv[1:5]
label = sys.argv[5] if len(sys.argv) > 5 else tid
cfg = load_env_cfg(tid, play=True)
cmd = cfg.commands["motion"]
names = [os.path.basename(f).replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
cmd.fixed_clip = names.index(clip)
cmd.sampling_mode = "start"
cfg.scene.num_envs = 1
cfg.viewer.height, cfg.viewer.width = 540, 960
cfg.viewer.distance, cfg.viewer.elevation = 4.5, -12.0
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
obs, _ = env.reset()
pol = ExpertActor(ckpt, "cuda:0")
T = int(env.command_manager.get_term("motion").motion.clip_len[cmd.fixed_clip])
font = ImageFont.load_default(size=22)
frames, status = [], "completed"
touch = 0
with torch.inference_mode():
    for t in range(T - 1):
        obs, _, term, _, _ = env.step(pol(obs["actor"]))
        if "bridge_contact" in env.scene.sensors:
            touch += int(hnu_tasks.bridge_touch(env).any())
        if term.any():
            status = f"FELL at {t / 50:.2f}s"
            break
        img = Image.fromarray(env.render())
        dr = ImageDraw.Draw(img)
        dr.text((12, 10), label, fill=(255, 255, 255), font=font)
        dr.text((12, 40), f"{clip}   t = {t / 50:.2f} s", fill=(255, 255, 255), font=font)
        frames.append(np.asarray(img))
if touch:
    status += f", bridge contact {touch} steps"
img = Image.fromarray(frames[-1].copy())
ImageDraw.Draw(img).text((12, 70), status, fill=(80, 255, 80) if status == "completed" else (255, 80, 80), font=font)
frames += [np.asarray(img)] * 50
import imageio.v2 as imageio
os.makedirs(os.path.dirname(out), exist_ok=True)
imageio.mimsave(out, frames, fps=50, codec="libx264", quality=7)
print(clip, status, out)
