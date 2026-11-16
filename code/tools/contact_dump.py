"""Raw MuJoCo-warp contacts involving the stake geoms around one stake window (same rollout as record_batch.py),
plus the solver's contact count vs capacity -> are foot-stake collisions resolved, or dropped?"""
import sys

import mujoco
import numpy as np
import torch

from tools.record_batch import make_env

import hnu_tasks  # noqa: F401

clip, si, pt, zdir, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
env, act, names = make_env(pt, zdir, (240, 320))
c = env.command_manager.get_term("motion")
cid = names.index(clip)
c.cfg.fixed_clip = cid
f0, f1 = int(c.stake_windows[1][cid, si]), int(c.stake_windows[2][cid, si])
mjm = env.sim.mj_model
gname = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
stake_geoms = [g for g in range(mjm.ngeom) if "stake" in (gname(g) or "")]
print("stake geoms:", [(g, gname(g), mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[g])),
                        int(mjm.geom_contype[g]), int(mjm.geom_conaffinity[g])) for g in stake_geoms])
foot_g = [g for g in range(mjm.ngeom) if "ankle_roll" in (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[g])) or "")]
print("foot geoms:", [(g, gname(g), int(mjm.geom_contype[g]), int(mjm.geom_conaffinity[g])) for g in foot_g])
d = env.sim.data
print("data attrs with 'con':", [a for a in dir(d) if "con" in a.lower()][:20])
torch.manual_seed(seed)
with torch.inference_mode():
    obs, _ = env.reset()
    for t in range(int(c.motion.clip_len[cid]) - 1):
        obs, _, term, _, _ = env.step(act(obs))
        if term.any():
            break
        fr = int(c.time_steps[0] - c.motion.clip_start[cid])
        if f0 - 6 <= fr <= f1 + 6 and fr % 2 == 0:
            try:
                cpu = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)  # noqa: E731
                n = int(cpu(d.nacon).reshape(-1)[0])
                geom = cpu(d.contact.geom)[:n]
                dist = cpu(d.contact.dist)[:n]
                if fr == f0 - 6:
                    print("contact capacity:", cpu(d.contact.geom).shape[0])
                hits = [(gname(a), gname(b), round(float(z), 3)) for (a, b), z in zip(geom, dist)]
                print(f"fr {fr}: ncon {n}, stake contacts {hits}")
            except Exception as e:  # noqa: BLE001
                print("contact read failed:", repr(e))
                break
