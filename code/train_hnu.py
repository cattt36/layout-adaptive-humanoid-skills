"""mjlab train entry point with the HNU tasks registered.

Example:
  python train_hnu.py HNU-Track-FiveStepPoleRun --env.scene.num-envs 4096 --gpu-ids 0 \
      --env.commands.motion.motion-file /root/gpufree-data/hnu_v2/data/ref_v1/FiveStepPoleRun
"""
import sys

import hnu_tasks  # noqa: F401  (registers HNU-Track-*)
from mjlab.scripts.train import main

if __name__ == "__main__":
    sys.exit(main())
