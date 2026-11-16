"""Does a foot pass THROUGH a stake?  Teacher rollouts on W1 clips; at frames where a foot is within 0.08 m of a stake
centre (xy) and touches the TERRAIN, print the foot collision spheres (world z, radius), the stake geom (centre z,
half height, radius) and the contacts of that foot -> sphere centres below the stake top inside its radius = the
foot is inside the stake volume."""
import sys

import mujoco
import numpy as np
import torch

from tools.record_batch import make_env

import hnu_tasks  # noqa: F401

bank, clips = sys.argv[1], sys.argv[2].split(",")
env, act, names = make_env(f"TEACHER@{bank}", "/root/gpufree-data/hnu_v2/data/z_tables_v5t/Z4", (120, 160))
c = env.command_manager.get_term("motion")
mjm = env.sim.mj_model
gname = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
feet_g = {s: [g for g in range(mjm.ngeom) if (gname(g) or "").startswith(f"robot/{s}_foot")] for s in ("left", "right")}
stake_g = [g for g in range(mjm.ngeom) if "stake" in (gname(g) or "")]
print("foot sphere radii:", {s: [round(float(mjm.geom_size[g][0]), 4) for g in v] for s, v in feet_g.items()})
print("stake geom size (radius, half height):", [tuple(round(float(x), 3) for x in mjm.geom_size[g][:2]) for g in stake_g[:1]])
print("timestep", mjm.opt.timestep, "decimation-free control dt", env.step_dt)
tg = [g for g in range(mjm.ngeom) if gname(g) == "terrain"]
print("margins: terrain", [float(mjm.geom_margin[g]) for g in tg], "foot", float(mjm.geom_margin[feet_g['left'][0]]),
      "stake", float(mjm.geom_margin[stake_g[0]]), "| contype/conaffinity terrain",
      [(int(mjm.geom_contype[g]), int(mjm.geom_conaffinity[g])) for g in tg], "stake",
      (int(mjm.geom_contype[stake_g[0]]), int(mjm.geom_conaffinity[stake_g[0]])), "foot",
      (int(mjm.geom_contype[feet_g['left'][0]]), int(mjm.geom_conaffinity[feet_g['left'][0]])))
print("geom type terrain", [int(mjm.geom_type[g]) for g in tg], "stake", int(mjm.geom_type[stake_g[0]]),
      "(2 sphere, 5 cylinder, 0 plane, 6 box)", "| opt.cone", int(mjm.opt.cone), "nconmax?", getattr(env.sim, "nconmax", None))
fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
cpu = lambda x: x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)  # noqa: E731
shown = 0
for clip in clips:
    cid = names.index(clip)
    c.cfg.fixed_clip = cid
    xy_st = c.stake_windows[3][cid].cpu().numpy()
    nst = int((c.stake_windows[0][cid] >= 0).sum())
    for seed in range(6):
        torch.manual_seed(seed)
        with torch.inference_mode():
            obs, _ = env.reset()
            for t in range(int(c.motion.clip_len[cid]) - 1):
                obs, _, term, _, _ = env.step(act(obs))
                if term.any():
                    break
                fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
                gnd = env.scene.sensors["feet_ground_contact"].data.found.reshape(-1)[:2].cpu().numpy() > 0
                for k, side in enumerate(("left", "right")):
                    d = np.linalg.norm(xy_st[:nst] - fp[k, :2], axis=-1)
                    s = int(d.argmin())
                    if d[s] < 0.08 and gnd[k]:
                        d_ = env.sim.data
                        gx = cpu(d_.geom_xpos)[0] - env.scene.env_origins[0].cpu().numpy()
                        sph = [(round(float(gx[g][2]), 3), round(float(np.linalg.norm(gx[g][:2] - xy_st[s])), 3)) for g in feet_g[side]]
                        sg = stake_g[s]
                        n = int(cpu(d_.nacon).reshape(-1)[0])
                        geom = cpu(d_.contact.geom)[:n]
                        con = sorted({(gname(a), gname(b)) for a, b in geom if a in feet_g[side] or b in feet_g[side]})
                        print(f"{clip} seed {seed} t {t / 50:.2f}s {side} foot {d[s]:.3f} m from stake {s + 1} centre, ankle z {fp[k, 2]:.3f}; "
                              f"stake geom xyz {np.round(gx[sg], 3).tolist()} (table xy {np.round(xy_st[s], 3).tolist()}, top {gx[sg][2] + mjm.geom_size[sg][1]:.3f}); "
                              f"sphere (z, dist to stake axis): {sph}; contacts: {con}")
                        shown += 1
                        if shown >= 6:
                            sys.exit()
print("no pass-through frame found" if shown == 0 else "")
