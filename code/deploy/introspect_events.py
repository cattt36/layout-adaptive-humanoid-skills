"""Which events (randomisation) are active in the student evaluation env (play=True)?"""
import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg

cfg = hnu_distill_env_cfg("/root/gpufree-data/hnu_v2/data/z_tables_v5/Z0", play=True,
                          z_offsets=(0, 5, 10, 15, 20, 25, 30, 40, 50), scene="geom", hist=20, lin_vel=True)
for k, v in cfg.events.items():
    print(k, v.mode, getattr(v.func, "__name__", v.func), {a: (b.name, b.body_names, b.geom_names, b.joint_names) if a == "asset_cfg" else b for a, b in v.params.items()},
          getattr(v, "interval_range_s", None))
print("episode_length_s", cfg.episode_length_s)
