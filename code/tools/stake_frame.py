"""Render a few frames of the stake scene (reference replay, zero policy) to check stake placement."""
import sys
import numpy as np
import torch
from PIL import Image
import hnu_tasks  # noqa: F401
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task, clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_env_cfg(f"HNU-Track-{task}-Stakes", play=True)
cmd = cfg.commands["motion"]
names = [f.split("/")[-1].replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
cmd.fixed_clip = names.index(clip)
cfg.scene.num_envs = 1
cfg.viewer.height, cfg.viewer.width = 480, 720
cfg.viewer.distance, cfg.viewer.elevation = 4.0, -20.0
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
env.reset()
c = env.command_manager.get_term("motion")
frames = []
T = int(c.motion.clip_len[cmd.fixed_clip])
for f in np.linspace(0, T - 2, 6).astype(int):
    c.reset_to_frame(torch.tensor([0], device="cuda:0"), int(c.motion.clip_start[cmd.fixed_clip] + f))
    env.sim.forward()
    frames.append(env.render())
img = np.concatenate([np.concatenate(frames[:3], 1), np.concatenate(frames[3:], 1)], 0)
Image.fromarray(img).save(out)
print("stakes for clip:", c.stake_table[cmd.fixed_clip].cpu().numpy().round(2).tolist())
