"""README coordinate tables (+ check codes = place_live.ref_code), CSV and top-view figures of the
standard-start layouts of a deployment bundle.   python coords.py <bundle dir>"""
import glob
import hashlib
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

S = sys.argv[1]
F = f"{S}/figures"
os.makedirs(F, exist_ok=True)
CN = {"CrouchedTraverse": "bridge", "ThreeStepPoleRun": "three-step stakes", "FiveStepPoleRun": "five-step stakes"}
md, csv = [], ["layout,kind,n,foot,x_m,y_m,top_height_m,radius_m,check_code"]
order = ["D1", "D2", "N3", "S1", "S2", "N1", "T1", "T2", "N2"]
dirs = {os.path.basename(d).split("_")[0]: d for d in glob.glob(f"{S}/planned_layouts/*")}
for lid in order:
    d = dirs[lid]
    la = json.load(open(f"{d}/layout_aligned.json"))
    kind = "posts" if la.get("posts") else "stakes"
    items = la[kind]
    code = hashlib.md5(";".join(f"{it['n']}:{it['x']:.3f},{it['y']:.3f}" for it in items).encode()).hexdigest()[:6]
    f0 = la["feet0"]
    md.append(f"#### {lid} ({CN[la['task']]}, `{os.path.basename(d)}`, check code **{code}**)\n")
    md.append(f"Robot: pelvis projection at (0, 0), facing +X. Left foot ({f0['L'][0]:+.3f}, {f0['L'][1]:+.3f}), "
              f"right foot ({f0['R'][0]:+.3f}, {f0['R'][1]:+.3f}). Figure: `figures/{lid}.png`\n")
    xy = np.array([[it["x"], it["y"]] for it in items])
    if kind == "posts":
        md.append("| Post | x (m) | y (m) | Top height (m) | To next (m) |\n|---|---|---|---|---|")
        for i, p in enumerate(items):
            nx = f"{np.linalg.norm(xy[i + 1] - xy[i]):.3f}" if i + 1 < len(items) else "-"
            md.append(f"| {p['n']} | {p['x']:.3f} | {p['y']:.3f} | {p['height_deploy']:.3f} | {nx} |")
            csv.append(f"{lid},post,{p['n']},,{p['x']:.3f},{p['y']:.3f},{p['height_deploy']:.3f},,{code}")
        pk = la["plank"]
        ax = xy[-1] - xy[0]
        ang = np.degrees(np.arctan2(ax[1], ax[0]))
        md.append(f"\nBeam: centre ({pk['x']:.3f}, {pk['y']:.3f}), length {pk['length']:.2f} m, centre height "
                  f"{pk['z_center'] + pk.get('raise_deploy', 0):.3f} m. Line through posts 1→4 at **{ang:+.1f}°** to +X. "
                  f"Start to post 1: {np.linalg.norm(xy[0]):.2f} m.\n")
    else:
        md.append("| Stake | Foot | x (m) | y (m) | To next (m) |\n|---|---|---|---|---|")
        for i, s in enumerate(items):
            nx = f"{np.linalg.norm(xy[i + 1] - xy[i]):.3f}" if i + 1 < len(items) else "-"
            md.append(f"| {s['n']} | {s['foot']} | {s['x']:.3f} | {s['y']:.3f} | {nx} |")
            csv.append(f"{lid},stake,{s['n']},{s['foot']},{s['x']:.3f},{s['y']:.3f},{s['top_z']:.3f},{s['radius']:.3f},{code}")
        md.append(f"\nStakes: diameter {2 * items[0]['radius']:.2f} m, height {items[0]['top_z']:.2f} m, coordinates at "
                  f"the centre of the top face. Start to stake 1: {np.linalg.norm(xy[0]):.2f} m.\n")
    fig, a = plt.subplots(figsize=(7, 5), dpi=90)
    a.add_patch(Polygon([[0.25, 0], [-0.12, 0.12], [-0.12, -0.12]], color="tab:blue", alpha=0.7))
    for k in ("L", "R"):
        a.plot(*f0[k], "s", color="tab:blue", ms=6)
    a.annotate("robot (0,0) facing +X", (0, 0), (0.05, -0.45), fontsize=9, color="tab:blue")
    for it in items:
        if kind == "posts":
            a.add_patch(Circle((it["x"], it["y"]), 0.05, color="tab:red"))
            a.annotate(f"post {it['n']}\n({it['x']:.3f}, {it['y']:.3f})", (it["x"], it["y"]), (it["x"] + 0.08, it["y"] + 0.1), fontsize=8)
        else:
            a.add_patch(Circle((it["x"], it["y"]), it["radius"], fill=False, color="tab:green", lw=2))
            a.annotate(f"stake {it['n']}{it['foot']}\n({it['x']:.3f}, {it['y']:.3f})", (it["x"], it["y"]), (it["x"] + 0.05, it["y"] + 0.25), fontsize=8)
    if kind == "posts":
        a.plot(xy[:, 0], xy[:, 1], "-", color="tab:brown", lw=4, alpha=0.5)
    allp = np.vstack([xy, [[0, 0]]])
    a.set_xlim(allp[:, 0].min() - 0.7, allp[:, 0].max() + 0.9)
    a.set_ylim(allp[:, 1].min() - 0.8, allp[:, 1].max() + 0.8)
    a.set_aspect("equal")
    a.grid(alpha=0.4)
    a.set_xlabel("X (m), origin = robot start")
    a.set_ylabel("Y (m), left positive")
    a.set_title(f"{lid} {CN[la['task']]}  check code {code}")
    fig.tight_layout()
    fig.savefig(f"{F}/{lid}.png")
    plt.close(fig)
open(f"{S}/coords.md", "w").write("\n".join(md) + "\n")
open(f"{S}/point_coordinates.csv", "w", encoding="utf-8-sig").write("\n".join(csv) + "\n")
print("ok")
