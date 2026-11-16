"""Probe: how well does z_skill encode contact timing?  Ridge regression from the per-frame z_skill table to the time
until the next touchdown of each foot (reference contact labels, clipped to 1 s); fit on train clips, report R^2 / MAE
on unseen-subject test clips, per level.

  python z_contact_probe.py <z_tables_root> <motion_root> [levels...]
"""
import glob
import json
import os
import sys

import numpy as np

zroot, mroot = sys.argv[1:3]
levels = sys.argv[3:] or ["Z0", "Z1", "Z2", "Z3", "Z4"]


def ttd(contact, fps=50.0):
    T = len(contact)
    out = np.ones((T, 2))
    for k in range(2):
        nxt = 1e9
        for t in range(T - 1, -1, -1):
            if t + 1 < T and contact[t + 1, k] and not contact[t, k]:
                nxt = t + 1
            out[t, k] = min((nxt - t) / fps, 1.0)
    return out


for P in levels:
    d = os.path.join(zroot, P)
    split = json.load(open(os.path.join(d, "split.json")))
    X, Y = {"train": [], "test": []}, {"train": [], "test": []}
    for sp in ("train", "test"):
        for trial in split[sp]:
            zf = os.path.join(d, f"{trial}_z.npy")
            mf = glob.glob(os.path.join(mroot, "*", f"{trial}_motion.npz"))
            if not os.path.exists(zf) or not mf:
                continue
            z = np.load(zf)
            c = np.load(mf[0])["foot_contact"][: len(z)] > 0.5
            X[sp].append(z[: len(c)]); Y[sp].append(ttd(c))
    Xtr, Ytr, Xte, Yte = (np.concatenate(v) for v in (X["train"], Y["train"], X["test"], Y["test"]))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Ytr)
    pred = np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ W
    r2 = 1 - ((pred - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    print(f"{P}: time-to-touchdown probe on test clips  R2 L/R {r2[0]:.3f}/{r2[1]:.3f}  mean {r2.mean():.3f}  "
          f"MAE {np.abs(pred - Yte).mean() * 1000:.0f} ms  (train frames {len(Xtr)}, test {len(Xte)})")
