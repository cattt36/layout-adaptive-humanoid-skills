"""Check HNU_DR_LAG_SYNC: after N steps, are the current lags identical across all delayed observation terms?
  HNU_DR=1 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 python tools/test_lag_sync.py
"""
import torch

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from mjlab.envs import ManagerBasedRlEnv

H = "/root/gpufree-data/hnu_v2"
cfg = hnu_distill_env_cfg(f"{H}/data/z_tables_v5/Z0", play=False, z_offsets=(0, 5, 10, 15, 20, 25, 30, 40, 50),
                          scene="geom", hist=20, lin_vel=True)
cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0")
for t in range(200):
    env.step(act)
    if t % 50 == 49:
        lags = {f"{g}/{n}": b.current_lags.clone() for g, d in env.observation_manager._group_obs_term_delay_buffer.items()
                for n, b in d.items()}
        ref = next(iter(lags.values()))
        same = all(torch.equal(v, ref) for v in lags.values())
        print(f"step {t + 1}: {len(lags)} delayed terms, all identical: {same}, frac lag0 {(ref == 0).float().mean():.2f}",
              "" if same else {k: v[:8].tolist() for k, v in list(lags.items())[:4]})
