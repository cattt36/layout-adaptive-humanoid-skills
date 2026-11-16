"""Z4 ablation: Z3 plus evidence-grounded human semantics.

Each invocation trains exactly one action. Pass --dataset <action>, for example
--dataset CrouchedTraverse; do not pass all three actions to one run.
"""
from __future__ import annotations

import sys

from latent_train_protocol import resolved_train_args
from motion_latent_z import main


if __name__ == "__main__":
    main(["--ablation-profile", "Z4", *resolved_train_args("Z4", sys.argv[1:])])

