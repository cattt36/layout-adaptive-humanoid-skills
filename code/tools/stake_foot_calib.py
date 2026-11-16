"""Calibrate 'foot standing on a stake top' on the ROBOT: ankle height of loaded feet (contact force > 20 N) that are
within 0.17 m of a stake centre vs farther than 0.35 m from every stake (ground), over several ThreeStepPoleRun / FiveStepPoleRun clips."""
import sys

import numpy as np
import torch

from tools.record_batch import make_env

import hnu_tasks  # noqa: F401

pt, zdir = sys.argv[1:3]
clips = sys.argv[3].split(",")
env, act, names = make_env(pt, zdir, (240, 320))
c = env.command_manager.get_term("motion")
fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
fs = env.scene.sensors["feet_ground_contact"]
near, far, air_near = [], [], []
for clip in clips:
    cid = names.index(clip)
    c.cfg.fixed_clip = cid
    xy = c.stake_windows[3][cid].cpu().numpy()
    nst = int((c.stake_windows[0][cid] >= 0).sum())
    torch.manual_seed(1000)
    with torch.inference_mode():
        obs, _ = env.reset()
        for t in range(int(c.motion.clip_len[cid]) - 1):
            obs, _, term, _, _ = env.step(act(obs))
            if term.any():
                break
            fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
            frc = fs.data.force.reshape(1, 2, -1)[0, :, -1].abs().cpu().numpy()
            for k in range(2):
                d = np.linalg.norm(xy[:nst] - fp[k, :2], axis=-1).min()
                if frc[k] > 20 and d < 0.17:
                    near.append(fp[k, 2])
                elif frc[k] > 20 and d > 0.35:
                    far.append(fp[k, 2])
                elif frc[k] < 1 and d < 0.17:
                    air_near.append(fp[k, 2])
q = lambda v: (np.percentile(v, [1, 5, 50, 95, 99]).round(3), len(v)) if len(v) else None  # noqa: E731
print("loaded, within 0.17 m of a stake (on stake?)  1/5/50/95/99 %:", q(near))
print("loaded, > 0.35 m from stakes (ground)         1/5/50/95/99 %:", q(far))
print("unloaded, within 0.17 m of a stake (air)      1/5/50/95/99 %:", q(air_near))
