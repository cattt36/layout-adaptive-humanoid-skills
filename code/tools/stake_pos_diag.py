"""Where are the stakes of a clip in the simulator (table, entity pose, geom size) vs the stake list of the json?"""
import json
import os
import sys

import torch

from tools.record_batch import make_env

import hnu_tasks  # noqa: F401

pt, zdir = sys.argv[1:3]
env, act, names = make_env(pt, zdir, (240, 320))
c = env.command_manager.get_term("motion")
for clip in sys.argv[3].split(","):
    cid = names.index(clip)
    c.cfg.fixed_clip = cid
    with torch.inference_mode():
        env.reset()
    print(f"== {clip}")
    print("   table (x, y, z):", [[round(float(v), 3) for v in p] for p in c.stake_table[cid]])
    print("   windows xy     :", [[round(float(v), 3) for v in p] for p in c.stake_windows[3][cid]])
    org = env.scene.env_origins[0]
    for i in range(5):
        e = env.scene[f"stake{i}"]
        pos = getattr(e.data, "root_link_pos_w", None)
        if pos is None:
            pos = e.data.body_link_pos_w[:, 0]
        print(f"   entity stake{i} pos {[round(float(v), 3) for v in (pos[0] - org)]}")
d = os.environ["HNU_GEOM_W1"] + "/ThreeStepPoleRun"
j = json.load(open(f"{d}/{sys.argv[3].split(',')[0]}_stakes.json"))
print("json keys:", list(j.keys()), "height:", j.get("height"))
