"""Print the reward / termination / sensor set of the fine-tuning env configs (env vars as in tools/tf_v2.sh)."""
import hnu_tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

for task in ("HNU-Track-ThreeStepPoleRun-Stakes", "HNU-Track-CrouchedTraverse-Bridge"):
    c = load_env_cfg(task)
    print(task)
    print("  rewards:", {k: v.weight for k, v in c.rewards.items()})
    print("  terminations:", sorted(c.terminations))
    print("  sensors:", [s.name for s in c.scene.sensors])
    print("  push x:", c.events["push_robot"].params["velocity_range"]["x"])
    if "bridge_contact" in c.rewards:
        print("  bridge_contact func:", c.rewards["bridge_contact"].func.__name__)
    if "stake_event" in c.rewards:
        print("  stake_event params:", c.rewards["stake_event"].params)
