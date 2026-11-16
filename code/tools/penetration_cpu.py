"""Cross-check a 'foot inside the stake' frame with standard CPU MuJoCo: copy qpos / qvel / mocap poses of the
MuJoCo-Warp world into a CPU MjData, run mj_forward, list the contacts of the foot (geom pair, distance) and the
foot sphere / stake geometry.  Answers: is the foot really inside the stake, and what does CPU MuJoCo detect?"""
import sys

import mujoco
import numpy as np
import torch

from tools.record_batch import make_env

import hnu_tasks  # noqa: F401

bank, clip = sys.argv[1], sys.argv[2]
env, act, names = make_env(f"TEACHER@{bank}", "/root/gpufree-data/hnu_v2/data/z_tables_v5t/Z4", (120, 160))
c = env.command_manager.get_term("motion")
mjm = env.sim.mj_model
gname = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
cpu = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)  # noqa: E731
feet = {s: [g for g in range(mjm.ngeom) if (gname(g) or "").startswith(f"robot/{s}_foot")] for s in ("left", "right")}
cid = names.index(clip)
c.cfg.fixed_clip = cid
xy_st = c.stake_windows[3][cid].cpu().numpy()
nst = int((c.stake_windows[0][cid] >= 0).sum())
fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
torch.manual_seed(0)
md = mujoco.MjData(mjm)
with torch.inference_mode():
    obs, _ = env.reset()
    for t in range(int(c.motion.clip_len[cid]) - 1):
        obs, _, term, _, _ = env.step(act(obs))
        if term.any():
            break
        fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
        for k, side in enumerate(("left", "right")):
            d = np.linalg.norm(xy_st[:nst] - fp[k, :2], axis=-1)
            if d.min() < 0.08 and fp[k, 2] < 0.115:
                d_ = env.sim.data
                md.qpos[:] = cpu(d_.qpos)[0]
                md.qvel[:] = cpu(d_.qvel)[0]
                if mjm.nmocap:
                    md.mocap_pos[:] = cpu(d_.mocap_pos)[0]
                    md.mocap_quat[:] = cpu(d_.mocap_quat)[0]
                mujoco.mj_forward(mjm, md)
                print(f"t {t / 50:.2f}s {side} foot {d.min():.3f} m from a stake centre, ankle z (warp) {fp[k, 2]:.3f}")
                print("  warp foot sphere z:", [round(float(cpu(d_.geom_xpos)[0][g][2]), 3) for g in feet[side]])
                print("  CPU  foot sphere z:", [round(float(md.geom_xpos[g][2]), 3) for g in feet[side]])
                s = int(d.argmin())
                sg = [g for g in range(mjm.ngeom) if gname(g) == f"stake{s}/stake_collision"][0]
                print(f"  CPU  stake{s} geom xyz {np.round(md.geom_xpos[sg], 3).tolist()} size {np.round(mjm.geom_size[sg][:2], 3).tolist()}")
                cons = [(gname(md.contact[i].geom1), gname(md.contact[i].geom2), round(float(md.contact[i].dist), 4))
                        for i in range(md.ncon) if md.contact[i].geom1 in feet[side] or md.contact[i].geom2 in feet[side]]
                print("  CPU  contacts of this foot:", cons)
                n = int(cpu(d_.nacon).reshape(-1)[0])
                wc = [(gname(a), gname(b), round(float(z), 4)) for (a, b), z in zip(cpu(d_.contact.geom)[:n], cpu(d_.contact.dist)[:n])
                      if a in feet[side] or b in feet[side]]
                print("  warp contacts of this foot:", wc)
                sys.exit()
print("no frame found")
