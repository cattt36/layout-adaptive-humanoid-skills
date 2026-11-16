"""Z2 ablation: Z1 plus retargeted-robot semantic atoms.

Each invocation trains exactly one action. Pass --dataset <action>, for example
--dataset CrouchedTraverse; do not pass all three actions to one run.
"""
from __future__ import annotations

import sys

from latent_train_protocol import resolved_train_args
from motion_latent_z import main


if __name__ == "__main__":
    main(["--ablation-profile", "Z2", *resolved_train_args("Z2", sys.argv[1:])])

