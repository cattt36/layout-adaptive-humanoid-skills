"""Human-side evidence for the DuMuQiao geometry: where are the actor's head/back/shoulders relative to the
marker-defined bridge (lab frame, Y up, Z forward)?

Boundary markers (LLM_annotation environment): the plank top runs rear_upper -> front_upper; the plank ends are
supported by rear_lower / front_lower.  For every frame we report, per body point, its lateral offset to the plank
centre line and its height below the plank top at that Z.  Frames with |dx| < half width are "under the plank";
the highest body point under the plank bounds the plank underside from below.

  python human_bridge_clearance.py <bvh_dir> <scene_csv>
"""
import glob
import os
import sys

import numpy as np

RL, RU = np.array([-0.209, 0.085, -4.072]), np.array([-0.157, 1.129, -2.617])
FU, FL = np.array([0.017, 1.256, 2.227]), np.array([0.101, 0.051, 2.061])


def parse_bvh(path):
    lines = open(path).read().split("\n")
    joints, stack, order, i = [], [], [], 0
    while not lines[i].strip().startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            joints.append(dict(name=tok[1], parent=stack[-1] if stack else -1))
        elif tok and tok[0] == "End":
            joints.append(dict(name=joints[stack[-1]]["name"] + "_end", parent=stack[-1], end=True))
        elif tok and tok[0] == "{":
            stack.append(len(joints) - 1)
        elif tok and tok[0] == "}":
            stack.pop()
        elif tok and tok[0] == "OFFSET":
            joints[-1]["offset"] = np.array([float(x) for x in tok[1:4]])
        elif tok and tok[0] == "CHANNELS":
            joints[-1]["channels"] = tok[2:]
        i += 1
    n = int(lines[i + 1].split()[1])
    dt = float(lines[i + 2].split()[2])
    data = np.array([[float(x) for x in l.split()] for l in lines[i + 3:i + 3 + n] if l.strip()])
    return joints, data, dt


def rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    o = np.zeros((len(a), 3, 3))
    k = "XYZ".index(axis)
    i, j = [x for x in range(3) if x != k]
    o[:, k, k] = 1
    o[:, i, i] = c; o[:, j, j] = c
    o[:, i, j] = -s; o[:, j, i] = s
    return o


def fk(joints, data):
    T = len(data)
    pos, R = [], []
    col = 0
    for jt in joints:
        ch = jt.get("channels", [])
        vals = data[:, col:col + len(ch)]; col += len(ch)
        Rl = np.tile(np.eye(3), (T, 1, 1))
        p = np.tile(jt["offset"], (T, 1))
        for c, v in zip(ch, vals.T):
            if c.endswith("position"):
                p[:, "XYZ".index(c[0])] = v if jt["parent"] < 0 else p[:, "XYZ".index(c[0])]
            else:
                Rl = Rl @ rot(c[0], v)
        if jt["parent"] < 0:
            pos.append(p); R.append(Rl)
        else:
            Rp = R[jt["parent"]]
            pos.append(pos[jt["parent"]] + np.einsum("tij,tj->ti", Rp, jt["offset"][None].repeat(T, 0)))
            R.append(Rp @ Rl)
    return {jt["name"]: p for jt, p in zip(joints, pos)}


def main():
    bvh_dir = sys.argv[1]
    print("trial | under-plank frames | highest body point under plank (lab m) | its point | plank top there | head min/median (m)")
    for f in sorted(glob.glob(os.path.join(bvh_dir, "*.bvh"))):
        joints, data, dt = parse_bvh(f)
        P = fk(joints, data)
        P = {k: v / 100.0 for k, v in P.items()}  # cm -> m
        names = [k for k in P if any(s in k for s in ("Head", "Neck", "Spine", "Shoulder", "Arm"))]
        best = (-1, None, None)
        under = 0
        for k in names:
            p = P[k]
            u = (p[:, 2] - RU[2]) / (FU[2] - RU[2])
            ok = (u >= 0) & (u <= 1)
            line_x = RU[0] + u * (FU[0] - RU[0])
            top = RU[1] + u * (FU[1] - RU[1])
            inside = ok & (np.abs(p[:, 0] - line_x) < 0.15)
            if k.startswith("Head"):
                under = int(inside.sum())
                head = p[:, 1]
            if inside.any():
                i = np.argmax(np.where(inside, p[:, 1], -9))
                if p[i, 1] > best[0]:
                    best = (p[i, 1], k, top[i])
        toe = P.get("LeftToe", P[list(P)[0]])
        print(f"{os.path.basename(f)} | {under} | {best[0]:.3f} | {best[1]} | {best[2] if best[2] is not None else float('nan'):.3f} | "
              f"{head.min():.2f}/{np.median(head):.2f}   (toe y min {toe[:, 1].min():.3f}, z range {toe[:, 2].min():.2f}..{toe[:, 2].max():.2f})")


if __name__ == "__main__":
    main()
