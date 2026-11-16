"""Diagnostic (user question on the E2E video d1): during the stake run, does the NON-stance foot touch the
floor?  The video labels (record_batch.py) only judge the foot on each stake.  Re-runs rollouts exactly like
record_batch.py (same env, student, z / geometry tables) and logs per control step and foot: floor contact
(feet_ground_contact), stake contact (foot_stake<i>), sole height.  The "stake run" = from the first to the last stake
contact.  Also the reference (human, retargeted, raised) for comparison: its foot_contact labels / sole heights.
Saves frames at the first floor touches.
  python diag_floor_touch.py --student <pt> --zdir <dir> --clip <name> [--seeds 1000,1001,...] --out <dir>
  (env: HNU_GEOM_ThreeStepPoleRun / FiveStepPoleRun of the table, as in tools/e2e_demo.sh)"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_batch as rb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--zdir", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--seeds", default="1000")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
env, act, names = rb.make_env(a.student, a.zdir, (480, 640))
c = env.command_manager.get_term("motion")
c.cfg.fixed_clip = names.index(a.clip)
cid = names.index(a.clip)
m = c.motion
T = int(m.clip_len[cid])
fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
n_st = int((c.stake_windows[0][cid] >= 0).sum())
report = []
for seed in [int(s) for s in a.seeds.split(",")]:
    torch.manual_seed(seed)
    log = []
    saved = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        for t in range(T - 1):
            obs, _, term, trunc, _ = env.step(act(obs))
            if term.any() or trunc.any():
                break
            gnd = env.scene.sensors["feet_ground_contact"].data.found.reshape(-1)[:2].cpu().numpy() > 0
            stk = np.zeros(2, bool)
            for i in range(n_st):
                stk |= env.scene.sensors[f"foot_stake{i}"].data.found.reshape(-1)[:2].cpu().numpy() > 0
            z = (c.robot_body_pos_w[0, fidx, 2] - env.scene.env_origins[0, 2]).cpu().numpy()
            fr = int(c.time_steps[0] - m.clip_start[c.clip_id[0]])
            zr = (c.body_pos_w[0, fidx, 2] - env.scene.env_origins[0, 2]).cpu().numpy()  # reference ankles
            log.append(dict(t=t / 50, fr=fr, gnd=gnd.tolist(), stk=stk.tolist(), z=z.round(3).tolist(), zref=zr.round(3).tolist()))
    st_steps = [i for i, x in enumerate(log) if any(x["stk"])]
    events = []
    if st_steps:
        lo, hi = st_steps[0], st_steps[-1]
        prev = [False, False]
        for i in range(lo, hi + 1):
            x = log[i]
            for k in range(2):
                if x["gnd"][k] and not prev[k]:
                    other = 1 - k
                    events.append(dict(t=x["t"], foot="LR"[k], other_on_stake=bool(x["stk"][other]),
                                       sole_z=x["z"][k], ref_ankle_z=x["zref"][k]))
                prev[k] = x["gnd"][k]
        gnd_frac = [float(np.mean([log[i]["gnd"][k] for i in range(lo, hi + 1)])) for k in range(2)]
        # the reference: ankle height in the same window (raised references: ~0.10 m higher on the stakes)
        ref_low = [float(np.mean([log[i]["zref"][k] < 0.06 for i in range(lo, hi + 1)])) for k in range(2)]
    else:
        lo = hi = None
        gnd_frac = ref_low = [None, None]
    report.append(dict(seed=seed, steps=len(log), stake_run_s=None if lo is None else [log[lo]["t"], log[hi]["t"]],
                       floor_touch_events=events, floor_contact_share_L_R=gnd_frac, ref_ankle_low_share_L_R=ref_low))
    print(json.dumps(report[-1]), flush=True)
    # frames of the first two floor touches of the first seed
    if seed == int(a.seeds.split(",")[0]) and events:
        with torch.inference_mode():
            torch.manual_seed(seed)
            obs, _ = env.reset()
            want = [round(e["t"] * 50) for e in events[:3]]
            for t in range(T - 1):
                obs, _, term, trunc, _ = env.step(act(obs))
                if term.any() or trunc.any():
                    break
                if any(abs(t - w) <= 0 for w in want):
                    Image.fromarray(env.render()).save(os.path.join(a.out, f"touch_{a.clip}_t{t / 50:.2f}.png"))
json.dump(report, open(os.path.join(a.out, f"floor_touch_{a.clip}.json"), "w"), indent=1)
