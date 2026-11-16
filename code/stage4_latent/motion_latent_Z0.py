"""Z0 ablation: motion plus the shared task/segment query protocol.

Each invocation trains exactly one action. Pass --dataset <action>, for example
--dataset CrouchedTraverse; do not pass all three actions to one run.
"""
from __future__ import annotations

import sys

from latent_train_protocol import resolved_train_args
from motion_latent_z import main


if __name__ == "__main__":
    main(["--ablation-profile", "Z0", *resolved_train_args("Z0", sys.argv[1:])])

