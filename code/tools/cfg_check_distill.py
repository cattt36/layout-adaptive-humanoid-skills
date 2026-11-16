"""Check the recovery-training knobs of the distillation env (training vs play / evaluation)."""
import os

from hnu_tasks.distill import hnu_distill_env_cfg

for play in (False, True):
    c = hnu_distill_env_cfg(os.environ["ZDIR"], play=play, z_offsets=(0, 25), scene="geom", hist=20, lin_vel=True)
    m = c.commands["motion"]
    push = c.events.get("push_robot")
    print(f"play={play}: pose_range {m.pose_range}\n  velocity_range {m.velocity_range}\n  joint_pos {m.joint_position_range}"
          f"\n  push x {push.params['velocity_range']['x'] if push else None}")
