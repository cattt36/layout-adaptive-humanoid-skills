"""Per-frame foot-stake contact, ankle height and distance for one stake of one rollout (same seed / env order as
record_batch.py) -> check a GROUND / ON label.   python stake_contact_diag.py clip stake_index pt z_dir seed"""
import sys

import numpy as np
import torch

from tools.record_batch import make_env

import hnu_tasks  # noqa: F401

clip, si, pt, zdir, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
env, act, names = make_env(pt, zdir, (240, 320))
c = env.command_manager.get_term("motion")
cid = names.index(clip)
c.cfg.fixed_clip = cid
foot, f0, f1, sxy = c.stake_windows
fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
cxy = sxy[cid, si].cpu().numpy()
torch.manual_seed(seed)
print(f"stake {si + 1}: ref foot {'LR'[int(foot[cid, si])]}, window {int(f0[cid, si])}-{int(f1[cid, si])}")
with torch.inference_mode():
    obs, _ = env.reset()
    for t in range(int(c.motion.clip_len[cid]) - 1):
        obs, _, term, _, _ = env.step(act(obs))
        if term.any():
            print("fell", t / 50)
            break
        fr = int(c.time_steps[0] - c.motion.clip_start[cid])
        if int(f0[cid, si]) - 40 <= fr <= int(f1[cid, si]) + 40:
            fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
            fnd = env.scene.sensors[f"foot_stake{si}"].data.found.reshape(-1).cpu().numpy()
            frc = env.scene.sensors[f"foot_stake{si}"].data.force.reshape(-1).cpu().numpy()
            d = [np.linalg.norm(p[:2] - cxy) for p in fp]
            print(f"fr {fr}: L z {fp[0, 2]:.3f} d {d[0]:.2f} | R z {fp[1, 2]:.3f} d {d[1]:.2f} | found {fnd} |F| {np.abs(frc).round(0)}")
