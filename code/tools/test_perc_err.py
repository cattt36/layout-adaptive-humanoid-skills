"""Check of version B (HNU_PERC_ERR=1): student obstacle features vs the exact teacher features (stake scene)."""
import torch

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from mjlab.envs import ManagerBasedRlEnv

H = "/root/gpufree-data/hnu_v2"
cfg = hnu_distill_env_cfg(f"{H}/data/z_tables_v5/Z0", play=False, z_offsets=(0, 5, 10, 15, 20, 25, 30, 40, 50),
                          scene="geom", hist=20, lin_vel=True)
cfg.scene.num_envs = 256
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
om = env.observation_manager
names = om.active_terms["student"]
dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["student"]]
i0 = sum(dims[:names.index("obst_stakes")])
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0")
for t in range(60):
    obs, *_ = env.step(act)
st = obs["student"][:, i0:i0 + 10].reshape(-1, 2, 5)
te = obs["teacher_stk"][:, -10:].reshape(-1, 2, 5)
valid = te[:, 0, 4] > 0
d = (st[valid, 0, :2] - te[valid, 0, :2]).norm(dim=-1)
print(f"envs with stakes ahead: {int(valid.sum())}")
print(f"|student - teacher| next-stake xy: mean {100 * d.mean():.1f} cm, max {100 * d.max():.1f} cm, "
      f"share < 1 cm {(d < 0.01).float().mean():.2f}")
e = getattr(env, "_hnu_perr", None)
if e is None:
    raise SystemExit("no localisation error state (HNU_PERC_ERR=0)")
print(f"e0 radius mean {100 * e['e0'].norm(dim=-1).mean():.1f} cm, |kappa| max {e['kappa'].abs().max():.3f}, "
      f"|dpsi| max {e['dpsi'].abs().max():.3f}, clean share {(e['e0'].norm(dim=-1) == 0).float().mean():.2f}")
