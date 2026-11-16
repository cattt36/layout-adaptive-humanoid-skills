"""Record the teacher and the five students L0-L4 on the same clip as a 3x2 grid video.

  python record_grid.py --clip ThreeStepPoleRun_030 --students-root <students_v4d> --z-root <z_tables_v4d> \
      --ckpt-ThreeStepPoleRun ... --ckpt-FiveStepPoleRun ... --ckpt-CrouchedTraverse ... --out grid.mp4
Each panel is one deterministic rollout from the first frame (no push, no noise).
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch

from hnu_tasks import TASKS
from hnu_tasks.distill import MoETeacher, scene_teacher_groups
from distill_train import student_from_ckpt
from record_video import rollout

SIZE = (360, 480)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--students-root", required=True)
    ap.add_argument("--z-root", required=True)
    for t in TASKS:
        ap.add_argument(f"--ckpt-{t}", required=True)
    ap.add_argument("--ckpt-name", default="student_2500.pt", help="student file inside <students-root>/<Zk>/")
    ap.add_argument("--title", default="", help="suffix of the student labels (e.g. version)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    panels, status = [], {}
    for P in ["teacher", "Z0", "Z1", "Z2", "Z3", "Z4"]:
        zp = "Z4" if P == "teacher" else P
        ck = torch.load(os.path.join(args.students_root, zp, args.ckpt_name), map_location=dev, weights_only=False)
        z_off = tuple(ck.get("z_offsets", (0, 25)))
        a = SimpleNamespace(clip=args.clip, latent_dir=os.path.join(args.z_root, zp))
        scene = ck.get("scene", "flat")
        if P == "teacher":
            def build(env):
                pol = MoETeacher({t: getattr(args, f"ckpt_{t}") for t in TASKS}, dev, scene_teacher_groups(scene))
                return lambda o: pol(o, o["task"])
            label = "Teacher (expert bank)"
        else:
            def build(env, ck=ck, z_off=z_off):
                st = student_from_ckpt(ck, env.action_manager.total_action_dim, dev)
                grp = ck.get("obs_group", "student")
                return lambda o: st(o[grp])
            label = f"L{P[1]} student {args.title}".strip()
        frames, s = rollout(a, label, build, z_off, size=SIZE, scene=scene, hist=ck.get("hist", 5),
                            lin_vel=ck.get("lin_vel", False))
        panels.append(frames)
        status[P] = s
    n = max(len(f) for f in panels)
    panels = [f + [f[-1]] * (n - len(f)) for f in panels]
    sep_v = np.full((SIZE[0], 4, 3), 255, np.uint8)
    out = []
    for i in range(n):
        rows = [np.concatenate([panels[r * 3][i], sep_v, panels[r * 3 + 1][i], sep_v, panels[r * 3 + 2][i]], 1)
                for r in range(2)]
        out.append(np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), 255, np.uint8), rows[1]], 0))
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, out, fps=50, codec="libx264", quality=7)
    print(args.clip, status, "->", args.out)


if __name__ == "__main__":
    main()
