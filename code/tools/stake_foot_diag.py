"""Robot ankle height / distance to the stake centre during each stake window (one rollout, no rendering), plus the
contact force of that foot -> calibrate the 'standing on the stake top' test."""
import sys

import numpy as np
import torch

sys.argv += []
from tools.record_batch import make_env  # noqa: E402

import hnu_tasks  # noqa: E402,F401

clip, pt, zdir = sys.argv[1:4]
env, act, names = make_env(pt, zdir, (240, 320))
c = env.command_manager.get_term("motion")
c.cfg.fixed_clip = names.index(clip)
cid = names.index(clip)
foot, f0, f1, sxy = c.stake_windows
fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
fs = env.scene.sensors["feet_ground_contact"]
torch.manual_seed(1000)
rows = {}
with torch.inference_mode():
    obs, _ = env.reset()
    for t in range(int(c.motion.clip_len[cid]) - 1):
        obs, _, term, _, _ = env.step(act(obs))
        if term.any():
            print("fell at", t / 50)
            break
        fr = int(c.time_steps[0] - c.motion.clip_start[cid])
        fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
        frc = fs.data.force.reshape(1, 2, -1)[0, :, -1].abs().cpu().numpy()
        for i in range(int((foot[cid] >= 0).sum())):
            a, b = int(f0[cid, i]), int(f1[cid, i])
            if a - 25 <= fr <= b + 25:
                k = int(foot[cid, i])
                d = np.linalg.norm(fp[k, :2] - sxy[cid, i].cpu().numpy())
                rows.setdefault(i, []).append((fr - a, fp[k, 2], d, frc[k]))
for i, r in rows.items():
    print(f"stake {i + 1} (window frames {int(f0[cid, i])}-{int(f1[cid, i])}, rel. frame: ankle z / dist to centre / foot force N)")
    print("   " + "  ".join(f"{fr:+d}:{z:.2f}/{d:.2f}/{f:.0f}" for fr, z, d, f in r[::3]))
