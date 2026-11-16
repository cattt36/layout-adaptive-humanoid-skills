"""Push events: distillation training env (play=False) vs evaluation push (eval_student.py --push)."""
import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from mjlab.tasks.registry import load_env_cfg

cfg = hnu_distill_env_cfg("/root/gpufree-data/hnu_v2/data/z_tables_v5/Z0", play=False,
                          z_offsets=(0, 5, 10, 15, 20, 25, 30, 40, 50), scene="geom", hist=20, lin_vel=True)
for k, v in cfg.events.items():
    if "push" in k:
        print("train", k, v.mode, getattr(v, "interval_range_s", None), v.params.get("velocity_range"))
print("train events:", list(cfg.events))
p = load_env_cfg("HNU-Track-FiveStepPoleRun").events["push_robot"]
print("eval push", p.interval_range_s, p.params.get("velocity_range"))
