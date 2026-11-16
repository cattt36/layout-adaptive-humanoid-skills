"""DEMO (user: keep the four posts and the heights, move the CT trajectory away from the posts): push the
reference of a flat CT clip horizontally away from the posts and their floor bars.  Every frame is shifted RIGIDLY by a
smooth offset d(t) (joints / orientations unchanged, velocities updated), so the body shape is untouched; the offset is
built iteratively from the clearance deficits:
  every body origin  >= D_BODY from each post axis
  both ankles        >= D_FOOT from each floor-bar footprint (bar_w x bar_len, long axis across the beam, mid / end
                        lengths as on the real bridge) and from the post
d(t) is Gaussian-smoothed (SIG s) and zero during the standing start (before the first foot lift).
  python warp_ct.py <motion_npz> <bridge_json> <out_npz> [D_BODY] [D_FOOT]"""
import json
import sys

import numpy as np

mfile, bj, out = sys.argv[1:4]
D_BODY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.32
D_FOOT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.16
SIG, IT = 0.25, 25
BAR = {"mid": (0.10, 0.30), "end": (0.10, 0.40)}  # bar_w, bar_len of the real bridge (HNU_BASE_MID / _END)
d = dict(np.load(mfile))
fps = float(d["fps"])
bp = d["body_pos_w"].astype(np.float64)
T, NB = bp.shape[:2]
P = json.load(open(bj))["parts"]
yw = float(P["plank"]["yaw"])
ax = np.array([np.cos(yw), np.sin(yw)])
nrm = np.array([-ax[1], ax[0]])
names = sorted(k for k in P if k.startswith("post"))
posts = [(np.array(P[k]["center"][:2]), float(P[k]["radius"])) for k in names]
# end posts = the two extreme along the beam axis
along = [c @ ax for c, _ in posts]
ends = {int(np.argmin(along)), int(np.argmax(along))}
fc = d["foot_contact"].astype(bool)
lift = int(np.argmax(~fc.all(1)))
t0 = max(lift - int(0.3 * fps), 0)
# ankle indices (body order of the motion = meta body_names; ankle roll links)
ANK = [6, 12]


def deficits(q):
    """[T, 2] push vector from all clearance deficits of positions q [T, NB, 2]."""
    push = np.zeros((T, 2))
    for i, (c, r) in enumerate(posts):
        v = q - c  # [T, NB, 2]
        dist = np.linalg.norm(v, axis=-1)
        e = np.maximum(0.0, D_BODY + r - dist)  # body origins vs post axis
        u = v / np.maximum(dist[..., None], 1e-6)
        push += (e[..., None] * u).max(axis=1) + (e[..., None] * u).min(axis=1)  # strongest push per sign
        bw, bl = BAR["end" if i in ends else "mid"]
        for a in ANK:  # ankle vs bar rectangle (local frame: x along the beam, y across)
            lx, ly = v[:, a] @ ax, v[:, a] @ nrm
            ex = np.maximum(np.abs(lx) - bw / 2, 0.0)
            ey = np.maximum(np.abs(ly) - bl / 2, 0.0)
            dd = np.hypot(ex, ey)
            e2 = np.maximum(0.0, D_FOOT - dd)
            # direction: away from the rectangle (outside -> along the gradient; inside -> along the beam axis)
            gx = np.where(ex > 0, np.sign(lx) * ex, np.sign(lx + 1e-9) * (ey <= 0))
            gy = np.where(ey > 0, np.sign(ly) * ey, 0.0)
            gn = np.maximum(np.hypot(gx, gy), 1e-6)
            push += e2[:, None] * (gx[:, None] * ax + gy[:, None] * nrm) / gn[:, None]
    return push


def smooth(x):
    k = np.arange(-int(3 * SIG * fps), int(3 * SIG * fps) + 1)
    w = np.exp(-0.5 * (k / (SIG * fps)) ** 2)
    w /= w.sum()
    xp = np.pad(x, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], w, mode="valid") for j in range(x.shape[1])], -1)


# freeze the offset after the bridge: from 0.5 s after the last frame with any deficit (no sliding in the run-out)
p0 = deficits(bp[:, :, :2])
hit = np.where(np.linalg.norm(p0, axis=1) > 0)[0]
t1 = min(int(hit[-1] + 0.5 * fps), T - 1) if len(hit) else T - 1


def freeze(o):
    o[:t0] = 0.0
    o[t1:] = o[t1]
    return o


off = np.zeros((T, 2))
for it in range(IT):
    q = bp[:, :, :2] + off[:, None]
    p = deficits(q)
    p[:t0] = 0.0
    off = freeze(off + 1.5 * smooth(p))
    off = freeze(smooth(off))


def report(q, tag):
    mb = min(np.min(np.linalg.norm(q - c, axis=-1)) - r for c, r in posts)
    mf = []
    for i, (c, r) in enumerate(posts):
        bw, bl = BAR["end" if i in ends else "mid"]
        for a in ANK:
            v = q[:, a] - c
            ex = np.maximum(np.abs(v @ ax) - bw / 2, 0.0)
            ey = np.maximum(np.abs(v @ nrm) - bl / 2, 0.0)
            mf.append(np.min(np.hypot(ex, ey)))
    print(f"{tag}: min body-origin clearance to posts {mb:+.3f} m, min ankle distance to bars {min(mf):+.3f} m")


report(bp[:, :, :2], "before")
report(bp[:, :, :2] + off[:, None], "after ")
# stance-foot slide introduced by the shift (per single-support stance, max over stances)
sl = []
for s, a in enumerate(ANK):
    on = fc[:, s]
    k = 0
    while k < T:
        if on[k]:
            e = k
            while e < T and on[e]:
                e += 1
            sl.append(np.linalg.norm(off[e - 1] - off[k]))
            k = e
        else:
            k += 1
print(f"offset: max |d| {np.linalg.norm(off, axis=1).max():.3f} m, max stance slide {max(sl):.3f} m, "
      f"median stance slide {np.median(sl):.3f} m")
bp[:, :, :2] += off[:, None]
vel = np.gradient(off, axis=0) * fps
lv = d["body_lin_vel_w"].astype(np.float64)
lv[:, :, :2] += vel[:, None]
d["body_pos_w"], d["body_lin_vel_w"] = bp.astype(np.float32), lv.astype(np.float32)
d["start_info"] = np.array(str(d["start_info"]) + f" | warped away from the posts (D_BODY {D_BODY}, D_FOOT {D_FOOT})")
np.savez(out, **d)
print(out)
