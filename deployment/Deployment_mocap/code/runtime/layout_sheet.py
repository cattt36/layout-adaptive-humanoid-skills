"""Layout sheets for the real-robot experiment: recommended clips per action and, for each, the positions of robot,
stakes and bridge in a start frame (origin = the robot's pelvis at the start projected to the ground, x = initial
heading, y = left), with sizes, tolerances and a top-view drawing.

Clip recommendation: highest mean sim2sim success over the selected deployment students (L0-L4) under 0-20 ms
latency + dynamics randomisation and nominal (rows of sim2sim_0-20_dr_dt0.005.json / sim2sim_none_dt0.005.json).

  python deploy/layout_sheet.py <exports_dir, e.g. deploy_exports/v6g10s> <out_dir> [n_per_action=3]
"""
import glob
import json
import os
import sys

import numpy as np

EXP, OUT = sys.argv[1], sys.argv[2]
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(OUT, exist_ok=True)
dirs = sorted(d for d in glob.glob(f"{EXP}/L*_s*") if os.path.isdir(d))
meta = json.load(open(os.path.join(dirs[0], "policy_meta.json")))
CN = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}


def clip_success():
    acc = {}
    for d in dirs:
        for f in ("sim2sim_0-20_dr_dt0.005.json", "sim2sim_none_dt0.005.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                for r in json.load(open(p))["rows"]:
                    acc.setdefault(r["clip"], []).append(float(r["success"]))
    return {c: (np.mean(v), len(v)) for c, v in acc.items()}


def clip_files(clip):
    for dd in meta["motion_dirs"]:
        f = os.path.join(dd, f"{clip}_motion.npz")
        if os.path.exists(f):
            return f, os.path.join(dd, f"{clip}_stakes.json"), os.path.join(dd, f"{clip}_bridge.json"), dd
    raise FileNotFoundError(clip)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def bridge_sizes(bdir):
    """Part sizes as compiled in the training scene (median over the clips, hnu_tasks/bridge.py)."""
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(bdir, "*_bridge.json")))]
    parts = lambda m: m["parts"] if "parts" in m else {k: dict(type="box", **v) for k, v in m["boxes"].items()}  # noqa: E731
    out = {}
    for p, v in parts(metas[0]).items():
        if v["type"] == "box":
            out[p] = ("box", np.median([parts(m)[p]["half"] for m in metas], axis=0) * 2)  # full lengths
        else:
            out[p] = ("cyl", np.array([np.median([parts(m)[p]["radius"] for m in metas]) * 2,
                                       np.median([parts(m)[p]["half_height"] for m in metas]) * 2]))  # diameter, height
    return out


succ = clip_success()
md = ["# 实物实验摆放图\n",
      "坐标系：原点 = 机器人起始时骨盆在地面上的投影；x 轴 = 机器人起始朝向（正前方）；y 轴 = 左侧；z 轴向上；单位 m。\n",
      f"推荐片段：{len(dirs)} 个部署学生（L0–L4）在 sim2sim 中（理想环境、0–20 ms 延迟 + 参数随机）平均成功率最高的片段。\n",
      "容差（sim2sim 测得，详见实验方案文档）：桩与桩相对位置 ±5 cm；整套布置相对起点 ±10 cm；"
      "感知固定偏差 ≤ 2–3 cm；起始朝向、独木桥高度见 setup 测试结果。\n"]
for T in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
    cands = sorted(((v[0], c) for c, v in succ.items() if c.startswith(T + "_")), reverse=True)[:NPER]
    md.append(f"\n## {CN[T]}\n")
    for rate, clip in cands:
        mf, sj, bj, dd = clip_files(clip)
        ref = np.load(mf)
        p0 = ref["body_pos_w"][0, 0]
        yaw0 = yaw_of(ref["body_quat_w"][0, 0])
        c, s = np.cos(-yaw0), np.sin(-yaw0)

        def to_start(xy):
            v = np.asarray(xy[:2], float) - p0[:2]
            return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

        md.append(f"\n### {clip}（sim2sim 平均成功率 {100 * rate:.0f}%）\n")
        md.append(f"- 机器人：起点 (0, 0)，朝向 +x；起始骨盆高度 {p0[2]:.3f} m；起始姿态 = 参考动作第一帧（关节角见 "
                  f"`{clip}_start_pose.json`）")
        end = to_start(ref["body_pos_w"][-1, 0])
        md.append(f"- 动作结束时骨盆位置约 ({end[0]:.2f}, {end[1]:.2f})；动作时长 {len(ref['joint_pos']) / 50:.1f} s")
        json.dump(dict(joint_names=meta["joint_names"], joint_pos_rad=[float(x) for x in ref["joint_pos"][0]],
                       pelvis_height=float(p0[2])), open(os.path.join(OUT, f"{clip}_start_pose.json"), "w"), indent=1)
        draw = dict(stakes=[], bridge=[], path=np.array([to_start(p) for p in ref["body_pos_w"][:, 0]]))
        if os.path.exists(sj):
            j = json.load(open(sj))
            md.append(f"- 三步桩/五步桩：圆柱桩高 {meta['stakes']['height']:.2f} m、直径 {2 * meta['stakes']['radius']:.2f} m\n")
            md.append("| 桩 | x | y | 踩桩脚 | 踩上时刻 (s) | 与上一个桩中心距离 |\n|---|---|---|---|---|---|")
            prev = None
            for i, (pxyz, st) in enumerate(zip(j["stake_xyz_50hz"], j["stakes"])):
                q = to_start(pxyz)
                dist = "-" if prev is None else f"{np.linalg.norm(q - prev):.3f}"
                md.append(f"| {i + 1} | {q[0]:.3f} | {q[1]:.3f} | {'左' if st['foot'] == 'L' else '右'} | "
                          f"{st['stance'][0] / j['fps']:.2f} | {dist} |")
                draw["stakes"].append(q)
                prev = q
        if os.path.exists(bj):
            j = json.load(open(bj))
            parts = j["parts"] if "parts" in j else {k: dict(type="box", **v) for k, v in j["boxes"].items()}
            sizes = bridge_sizes(dd)
            md.append("- 独木桥部件（中心坐标；尺寸为训练场景中的尺寸）\n")
            md.append("| 部件 | x | y | 中心高度 z | 水平朝向（相对 x 轴, °） | 倾斜（°） | 尺寸 |\n|---|---|---|---|---|---|---|")
            for p in meta["bridge"]["parts"]:
                b = parts[p]
                q = to_start(b["center"])
                kind, sz = sizes[p]
                dims = (f"长 {sz[0]:.3f} × 宽 {sz[1]:.3f} × 高 {sz[2]:.3f}" if kind == "box" else
                        f"直径 {sz[0]:.3f} × 高 {sz[1]:.3f}")
                md.append(f"| {p} | {q[0]:.3f} | {q[1]:.3f} | {b['center'][2]:.3f} | "
                          f"{np.rad2deg(b['yaw'] - yaw0):.1f} | {np.rad2deg(b['pitch']):.1f} | {dims} |")
                draw["bridge"].append((p, q, b["yaw"] - yaw0, kind, sz))
                if p == "plank":
                    md.append(f"|  | 横梁底面离地约 {b['center'][2] - sz[2] / 2:.3f} m |  |  |  |  |  |")
        # top view
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Polygon
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(draw["path"][:, 0], draw["path"][:, 1], "--", color="gray", lw=1)  # reference pelvis path
            for i, q in enumerate(draw["stakes"]):
                ax.add_patch(Circle(q, meta["stakes"]["radius"], color="saddlebrown", alpha=0.6))
                ax.text(q[0], q[1], str(i + 1), ha="center", va="center", color="white", fontsize=9)
            for p, q, yw, kind, sz in draw["bridge"]:
                if kind == "box":
                    L, W = sz[0] / 2, sz[1] / 2
                    cs, sn = np.cos(yw), np.sin(yw)
                    corners = [q + np.array([cs * a - sn * b, sn * a + cs * b]) for a, b in ((L, W), (-L, W), (-L, -W), (L, -W))]
                    ax.add_patch(Polygon(corners, color="peru", alpha=0.5))
                else:
                    ax.add_patch(Circle(q, sz[0] / 2, color="sienna", alpha=0.8))
                ax.text(q[0], q[1], p, fontsize=7)
            ax.arrow(0, 0, 0.3, 0, head_width=0.06, color="blue")
            ax.plot(0, 0, "bo")
            ax.text(0, -0.15, "robot start", color="blue", fontsize=8)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_xlabel("x (m, initial heading)")
            ax.set_ylabel("y (m, left)")
            ax.set_title(clip)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"{clip}_top.png"), dpi=150)
            plt.close(fig)
            md.append(f"\n![{clip}]({clip}_top.png)\n")
        except ImportError:
            md.append("\n（未安装 matplotlib，未生成俯视图）\n")
open(os.path.join(OUT, "layout.md"), "w").write("\n".join(md))
print("written", OUT)
