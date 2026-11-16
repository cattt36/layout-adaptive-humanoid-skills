"""Live placement guide for the bridge posts and the stakes, motion capture (NOKOV XINGYING).

While the posts / stakes are being placed, the screen shows for every one of them how far and in which direction
(+X / -X / +Y / -Y of the layout's aligned frame = mocap frame + shift) it still has to move, refreshed ~4 times a
second.  Ctrl+C ends: final verdict, a record in <layout>/placement_records/, exit code 0 = OK.

  python place_live.py <layout_dir> [--config config.yaml] [--mode rigid|markers] [--shift dx,dy] [--tol 0.03]
  python place_live.py planned_layouts/D1_CrouchedTraverse_024 --config ../real/config.yaml
  python place_live.py planned_layouts/S1_ThreeStepPoleRun_011 --mode markers --shift 0,0.044
  python place_live.py planned_layouts/D1_CrouchedTraverse_024 --source fake          # try the screen without mocap

Targets (layout_aligned.json): bridge -> posts 1..4 (post 1 = first one the robot passes), stakes -> stakes 1..n.
Measurement:
  rigid    rigid bodies post1..post4 / stake1..stake5 (names: --bodies or config mocap.obstacle_bodies), pivot = top
           centre of the post / stake (as in config.yaml)
  markers  unlabeled markers: every persistent marker within --radius of a target (posts 0.3-1.3 m high, stakes
           0.03-0.3 m) belongs to it; target = mean xy of its markers.  Stakes: one marker at the top centre (or >= 3
           evenly on the rim).  Posts: markers on two OPPOSITE sides (or a ring) -- markers on one side only are off
           by the post radius (~4 cm)
Verdict: per item green <= --tol (3 cm), yellow <= --tol2 (5 cm), red beyond.  Stakes: trained +-3 cm per stake
+ 3 cm for the whole layout (stake radius 20 cm).  Posts: either at the sheet (green) or measured and written into
the layout files (--write-plan on exit -> obstacles_live.py; training coverage printed below the table).
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

G_, Y_, R_, B_, N_ = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"


def load_targets(L):
    la = json.load(open(os.path.join(L, "layout_aligned.json")))
    if la.get("posts"):
        kind, items = "posts", [dict(n=p["n"], x=p["x"], y=p["y"], label=f"立柱 {p['n']}") for p in la["posts"]]
    elif la.get("stakes"):
        kind, items = "stakes", [dict(n=s["n"], x=s["x"], y=s["y"], label=f"桩 {s['n']} ({s['foot']})") for s in la["stakes"]]
    else:
        sys.exit(f"{L}: no posts / stakes in layout_aligned.json")
    la["_code"] = ref_code(items)
    return la, kind, items


def ref_code(items):
    """check code of the reference coordinates (n, x, y rounded to mm) -- the same code is printed in the
    README coordinate table, so README and the file the guide reads can be compared at a glance"""
    import hashlib
    s = ";".join(f"{it['n']}:{it['x']:.3f},{it['y']:.3f}" for it in items)
    return hashlib.md5(s.encode()).hexdigest()[:6]


class Source:
    """Latest positions (aligned frame, m) per target, filled by a mocap callback thread."""

    def __init__(self, n, window=0.5):
        self.buf = [collections.deque() for _ in range(n)]
        self.window, self.lock, self.frames, self.t_last = window, threading.Lock(), 0, 0.0

    def put(self, i, xy, t):
        with self.lock:
            b = self.buf[i]
            b.append((t, xy))
            while b and b[0][0] < t - self.window:
                b.popleft()

    def get(self, i, now):
        with self.lock:
            b = [xy for t, xy in self.buf[i] if t >= now - self.window]
        return np.median(np.array(b), 0) if b else None


def start_nokov(a, items, kind, src, shift, scale):
    from nokov.nokovsdk import DataDescriptions, DataDescriptors, PySDKClient
    import ctypes
    c = PySDKClient()
    c.PySetVerbosityLevel(0)
    assert c.Initialize(a.server_ip.encode()) == 0, "NOKOV connect failed (another SDK client running? host reachable?)"
    names = {}
    if a.mode == "rigid":
        ptr = ctypes.POINTER(DataDescriptions)()
        c.PyGetDataDescriptions(ptr)
        dd = ptr.contents
        for i in range(int(dd.nDataDescriptions)):
            d = dd.arrDataDescriptions[i]
            if d.type == DataDescriptors.Descriptor_RigidBody.value:
                rb = d.Data.RigidBodyDescription.contents
                names[int(rb.ID)] = bytes(rb.szName).split(b"\0", 1)[0].decode()
        want = a.bodies or [f"{'post' if kind == 'posts' else 'stake'}{it['n']}" for it in items]
        missing = [w for w in want if w not in names.values()]
        if missing:
            print(f"rigid bodies not defined in XINGYING: {missing} (defined: {sorted(names.values())})")
        idx = {w: i for i, w in enumerate(want)}
    plan = np.array([[it["x"], it["y"]] for it in items])
    zlo, zhi = (0.3, 1.3) if kind == "posts" else (0.03, 0.3)
    lost = lambda v: abs(v) >= 9999.0  # noqa: E731  XINGYING: untracked = +-9999.999 mm (or 9999999)

    def cb(fp, _):
        f = fp.contents
        t = time.monotonic()
        src.frames += 1
        src.t_last = t
        if a.mode == "rigid":
            for k in range(int(f.nRigidBodies)):
                b = f.RigidBodies[k]
                i = idx.get(names.get(int(b.ID)))
                if i is not None and not (lost(b.x) or lost(b.y)):
                    src.put(i, np.array([b.x, b.y]) * scale + shift, t)
        else:
            m = np.array([list(f.OtherMarkers[j]) for j in range(int(f.nOtherMarkers))], float).reshape(-1, 3) \
                if f.OtherMarkers else np.zeros((0, 3))
            m = m[(np.abs(m) < 9999.0).all(1)] * scale
            m = m[(m[:, 2] > zlo) & (m[:, 2] < zhi)]
            if len(m):
                xy = m[:, :2] + shift
                dist = np.linalg.norm(xy[:, None] - plan[None], axis=2)
                own = dist.argmin(1)
                for i in range(len(items)):
                    sel = (own == i) & (dist[:, i] < a.radius)
                    if sel.any():
                        src.put(i, xy[sel].mean(0), t)
    src._cb = cb  # keep a reference (ctypes callback)
    assert c.PySetDataCallback(cb, None) == 0
    return c


def start_fake(items, src):
    rng = np.random.default_rng()
    off = rng.uniform(-0.08, 0.08, (len(items), 2))

    def run():
        t0 = time.monotonic()
        while True:
            t = time.monotonic()
            src.frames += 1
            src.t_last = t
            k = max(0.0, 1.0 - (t - t0) / 40.0)  # the "helpers" move the items to their place within ~40 s
            for i, it in enumerate(items):
                if rng.random() > 0.02:
                    src.put(i, np.array([it["x"], it["y"]]) + k * off[i] + rng.normal(0, 0.002, 2), t)
            time.sleep(1 / 90)
    threading.Thread(target=run, daemon=True).start()


def move_text(d):
    """d = plan - measured (m) -> 'move +X 1.7 cm, -Y 2.8 cm'."""
    out = []
    for v, ax in ((d[0], "X"), (d[1], "Y")):
        if abs(v) >= 0.005:
            out.append(f"往 {'+' if v > 0 else '-'}{ax} 挪 {100 * abs(v):4.1f} cm")
    return "，".join(out) if out else "到位"


def evaluate(items, meas, kind, tol, tol2):
    rows, ok, ideal = [], True, True
    plan = np.array([[it["x"], it["y"]] for it in items])
    for i, it in enumerate(items):
        m = meas[i]
        if m is None:
            rows.append((it, None, None, "看不到", R_))
            ok = ideal = False
            continue
        e = m - plan[i]
        r = float(np.linalg.norm(e))
        col = G_ if r <= tol else (Y_ if r <= tol2 else R_)
        ok &= r <= tol2
        ideal &= r <= tol
        rows.append((it, m, e, move_text(-e), col))
    extra = []
    have = [i for i in range(len(items)) if meas[i] is not None]
    if len(have) == len(items):
        M = np.array(meas)
        sh = (M - plan).mean(0)
        extra.append(f"整体平均偏差 dx {100 * sh[0]:+5.1f}  dy {100 * sh[1]:+5.1f} cm")
        gp, gm = np.linalg.norm(np.diff(plan, axis=0), axis=1), np.linalg.norm(np.diff(M, axis=0), axis=1)
        extra.append("间距 计划 " + " / ".join(f"{g:.3f}" for g in gp) + " m，实测 " + " / ".join(f"{g:.3f}" for g in gm) + " m")
        if kind == "posts":  # coverage of the training if the measured posts are written into the bridge file
            ax = plan[-1] - plan[0]
            ax = ax / np.linalg.norm(ax)
            nx = np.array([-ax[1], ax[0]])
            res = (M - plan) - sh
            al, lat = res @ ax, res @ nx
            c1, c2, c3 = np.all(np.abs(sh) <= 0.05), np.all(np.abs(al) <= 0.10), np.all(np.abs(lat) <= 0.03)
            extra.append(f"若写入规划文件（--write-plan）：整体 ±5 cm {'√' if c1 else '×'}，单根沿横梁 ±10 cm "
                         f"{'√' if c2 else '×'}（{' / '.join(f'{100 * v:+.1f}' for v in al)}），偏离横梁线 ≤ 3 cm "
                         f"{'√' if c3 else '× 重测'}")
        else:  # stakes: along / across the local stake line, after the common shift
            res = (M - plan) - sh
            al, lat = [], []
            for i in range(len(plan)):
                v = plan[min(i + 1, len(plan) - 1)] - plan[max(i - 1, 0)]
                v = v / max(np.linalg.norm(v), 1e-6)
                al.append(float(res[i] @ v))
                lat.append(float(res[i] @ np.array([-v[1], v[0]])))
            c1, c2, c3 = np.all(np.abs(sh) <= 0.05), np.all(np.abs(al) <= 0.03), np.all(np.abs(lat) <= 0.06)
            extra.append(f"若写入规划文件（--write-plan）：整体 ±5 cm {'√' if c1 else '×'}，单根沿前进方向 ±3 cm "
                         f"{'√' if c2 else '×'}，横向 ±6 cm {'√' if c3 else '×'}")
    return rows, ok, ideal, extra


def draw(la, kind, rows, ok, ideal, extra, a, src, shift):
    s = [f"\033[2J\033[H{B_}{la['id']} ({la['clip']})  {'独木桥立柱' if kind == 'posts' else '桩'}{N_}   "
         f"{time.strftime('%H:%M:%S')}   动捕帧 {src.frames}{'' if time.monotonic() - src.t_last < 0.5 else R_ + '  (动捕无数据!)' + N_}",
         f"坐标：布局坐标系 = 动捕 + ({shift[0]:+.3f}, {shift[1]:+.3f}) m；原点 = 机器人起点，+X = 机器人起始朝向；模式 {a.mode}；"
         f"绿 ≤ {100 * a.tol:.0f} cm，黄 ≤ {100 * a.tol2:.0f} cm",
         f"参考坐标：{os.path.basename(a.layout_dir.rstrip('/').rstrip(chr(92)))}/layout_aligned.json，"
         f"校验码 {B_}{la['_code']}{N_}（须与 README 第 2 节该布局的校验码一致）",
         "",
         f"{'':10s} {'计划 x, y (m)':>16s} {'实测 x, y (m)':>16s} {'误差 dx / dy (cm)':>18s}   怎么挪"]
    for it, m, e, txt, col in rows:
        if m is None:
            s.append(f"{col}{it['label']:10s} ({it['x']:6.3f},{it['y']:6.3f})   {'--':>14s}   {'--':>16s}   看不到{N_}")
        else:
            s.append(f"{col}{it['label']:10s} ({it['x']:6.3f},{it['y']:6.3f})  ({m[0]:6.3f},{m[1]:6.3f})  "
                     f"{100 * e[0]:+6.1f} / {100 * e[1]:+6.1f}     {txt}{N_}")
    s.append("")
    s += ["  " + x for x in extra]
    s.append("")
    verdict = f"{G_}全部到位（理想）{N_}" if ideal else (f"{Y_}可以开始（都在 {100 * a.tol2:.0f} cm 内），最好再调到绿色{N_}"
                                                       if ok else f"{R_}继续调整{N_}")
    s.append(f"{B_}判定：{N_}{verdict}        Ctrl+C 结束并保存记录")
    print("\n".join(s), flush=True)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_dir")
    ap.add_argument("--config", default="", help="config.yaml of the real-robot layer (mocap: server_ip, aligned_shift_xy, obstacle_bodies)")
    ap.add_argument("--source", default="nokov", choices=["nokov", "fake"])
    ap.add_argument("--server-ip", default="10.1.1.198")
    ap.add_argument("--mode", default="rigid", choices=["rigid", "markers"])
    ap.add_argument("--bodies", default="", help="comma list of rigid-body names in target order (default post1.. / stake1..)")
    ap.add_argument("--shift", default="", help="dx,dy (m): aligned = mocap + shift (default config aligned_shift_xy or 0,0)")
    ap.add_argument("--radius", type=float, default=0.35, help="markers mode: search radius around a target (m)")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--tol2", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=0.0, help="> 0: run this long, print once, exit (no live screen)")
    ap.add_argument("--write-plan", "--write-bridge", dest="write_plan", action="store_true",
                    help="on exit write the measured posts / stakes into the layout files (obstacles_live.py --write)")
    a = ap.parse_args()
    os.system("")  # ANSI colours in the Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    L = a.layout_dir.rstrip("/\\")
    la, kind, items = load_targets(L)
    scale, shift = 0.001, np.zeros(2)
    if a.config:
        import yaml
        mc = yaml.safe_load(open(a.config)).get("mocap", {})
        a.server_ip = mc.get("server_ip", a.server_ip)
        scale = float(mc.get("unit_scale", scale))
        shift = np.array(mc.get("aligned_shift_xy", [0.0, 0.0]), float)
        ob = (mc.get("obstacle_bodies") or {}).get(kind)
        if ob and not a.bodies:
            a.bodies = ",".join(ob)
    if a.shift:
        shift = np.array([float(v) for v in a.shift.split(",")])
    a.bodies = [b for b in a.bodies.split(",") if b] if isinstance(a.bodies, str) else a.bodies
    src = Source(len(items))
    if a.source == "fake":
        start_fake(items, src)
    else:
        start_nokov(a, items, kind, src, shift, scale)
    time.sleep(0.6)
    meas = [None] * len(items)
    try:
        t_end = time.monotonic() + a.seconds
        while True:
            now = time.monotonic()
            meas = [src.get(i, now) for i in range(len(items))]
            rows, ok, ideal, extra = evaluate(items, meas, kind, a.tol, a.tol2)
            if a.seconds > 0:
                if now >= t_end:
                    break
                time.sleep(0.25)
                continue
            draw(la, kind, rows, ok, ideal, extra, a, src, shift)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    rows, ok, ideal, extra = evaluate(items, meas, kind, a.tol, a.tol2)
    verdict = draw(la, kind, rows, ok, ideal, extra, a, src, shift)
    os.makedirs(os.path.join(L, "placement_records"), exist_ok=True)
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), layout=la["id"], kind=kind, mode=a.mode, shift=shift.tolist(),
               ref_code=la["_code"],
               items=[dict(n=it["n"], plan=[it["x"], it["y"]], measured=None if m is None else m.tolist(),
                           error_cm=None if e is None else (100 * e).round(1).tolist()) for it, m, e, _, _ in rows],
               ok=bool(ok), ideal=bool(ideal))
    rf = os.path.join(L, "placement_records", time.strftime("%m%d_%H%M%S") + f"_{kind}.json")
    json.dump(rec, open(rf, "w"), indent=1, ensure_ascii=False)
    print(f"记录：{rf}")
    if a.write_plan:
        if any(m is None for m in meas):
            print("有目标看不到，未写入规划文件")
        else:
            subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacles_live.py"), L,
                             f"--{kind}", *[f"{m[0]:.4f},{m[1]:.4f}" for m in meas], "--write"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
