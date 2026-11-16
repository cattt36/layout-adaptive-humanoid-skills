"""DEMO: prop coordinates for one demo tracker, in the START FRAME (origin = the robot's floor mark = frame-0
pelvis of the reference, +x = start heading towards the obstacles, +y = left), from the reference clip, its props json
and the simulated repeatability (tools/prop_spread.py <prefix>.json, REP condition).
  stakes: centre (x, y), top height, reference foot on it; simulated landing offset (mean) and 95 % radius nearby
  bridge:  posts (x, y, top height), beam ends and top height, reference / simulated clearance per post
  flat:    footholds of the stepping phase (x, y) with the simulated landing mean / r95 (floor markers / side props)
Markdown to stdout.
  python props_table.py <motion_npz> <spread_prefix> [props_json]"""
import json
import sys

import numpy as np


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


mfile, pre = sys.argv[1:3]
pj = sys.argv[3] if len(sys.argv) > 3 else None
m = np.load(mfile)
a0 = yaw(m["body_quat_w"][0, 0])
o0 = m["body_pos_w"][0, 0, :2]
R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
to_s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T  # noqa: E731
sp = json.load(open(pre + ".json"))
lands = sp.get("landings", [])


def near(xy, rad=0.30):
    best = None
    for l in lands:
        d = np.linalg.norm(np.array(l["ref"]) - xy)
        if d < rad and (best is None or d < best[0]):
            best = (d, l)
    return best[1] if best else None


out = [f"仿真重复性（同一台机器人，起步 1 cm / 1°，{sp['trials']} 次）：成功 {100 * sp['success']:.0f} %\n"]
d = json.load(open(pj)) if pj else {}
if "stakes" in d:
    out.append("| 桩 | x (m) | y (m) | 顶高 (m) | 仿真落点偏移 dx / dy (m) | 落点 95% 半径 (m) |\n|---|---|---|---|---|---|")
    for k, st in enumerate(d["stakes"]):
        c = to_s([st["x"], st["y"]])
        l = near(c)
        off = f"{l['mean'][0]:+.2f} / {l['mean'][1]:+.2f}" if l else "-"
        r95 = f"{l['r95']:.2f}" if l else "-"
        out.append(f"| {k + 1} | {c[0]:.2f} | {c[1]:.2f} | {d.get('height', 0.10):.2f} | {off} | {r95} |")
if "parts" in d:
    P = d["parts"]
    out.append("| 立柱 | x (m) | y (m) | 顶高 (m) | 参考余量 (m) | 仿真余量 p5 / 中位 (m) |\n|---|---|---|---|---|---|")
    pc = {p["post"]: p for p in sp.get("posts", [])}
    for n in sorted(k for k in P if k.startswith("post")):
        c = to_s(P[n]["center"])
        top = P[n]["center"][2] + P[n]["half_height"]
        q = pc.get(n)
        cl = f"{q['p5']:+.2f} / {q['med']:+.2f}" if q else "-"
        rf = f"{q['ref']:+.2f}" if q else "-"
        out.append(f"| {n} | {c[0]:.2f} | {c[1]:.2f} | {top:.2f} | {rf} | {cl} |")
    if "plank" in P:
        p = P["plank"]
        c = to_s(p["center"])
        yw = p["yaw"] - a0
        e = np.array([np.cos(yw), np.sin(yw)]) * p["half"][0]
        out.append(f"\n横梁：两端 ({c[0] - e[0]:.2f}, {c[1] - e[1]:.2f}) → ({c[0] + e[0]:.2f}, {c[1] + e[1]:.2f})，"
                   f"长 {2 * p['half'][0]:.2f} m，梁底高 {p['center'][2] - p['half'][2]:.2f} m / 梁顶高 {p['center'][2] + p['half'][2]:.2f} m"
                   f"（参考动作的身体离梁最近 {d.get('ref_min_dist_m', {}).get('plank', float('nan')):.2f} m）")
    out.append("\n余量 = 身体各连杆原点到柱面的最小水平距离（不含身体本身的宽度约 0.10–0.15 m），只用于比较。")
    sg = sp.get("post_suggest", [])
    if sg:
        out.append("\n**单根立柱的建议位置**（64 次仿真中 98% 的余量 ≥ 0.25 m）：\n\n| 立柱 | 原 x | 原 y | 建议 x | 建议 y | 沿梁挪动 (m) |\n|---|---|---|---|---|---|")
        for q in sg:
            mv = f"{q['slide']:+.2f}" if q["slide"] is not None else "离开梁线"
            out.append(f"| {q['post']} | {q['ref'][0]:.2f} | {q['ref'][1]:.2f} | {q['xy'][0]:.2f} | {q['xy'][1]:.2f} | {mv} |")
        out.append("\n“离开梁线”的立柱架不住横梁，要换一种摆法，见 `<名字>_beam_free.txt`：里面列了沿梁线（以及平移 ±0.2/0.4 m、旋转 ±10° 的梁线）可以放柱子的区间。")
if True:  # footholds of the whole motion (floor marks for the dry runs)
    out.append("\n| 落脚 | 脚 | t (s) | 参考 x | 参考 y | 仿真偏移 dx / dy | 95% 半径 |\n|---|---|---|---|---|---|---|")
    for j, l in enumerate(lands):
        out.append(f"| {j + 1} | {l['foot']} | {l['t']:.2f} | {l['ref'][0]:.2f} | {l['ref'][1]:.2f} | "
                   f"{l['mean'][0]:+.2f} / {l['mean'][1]:+.2f} | {l['r95']:.2f} |")
print("\n".join(out))
