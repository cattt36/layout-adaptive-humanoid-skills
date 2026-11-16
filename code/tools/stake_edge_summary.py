"""stake radius sweep (tools/stake_size_s2k.sh) -> success v2 A / C and foot-on-stake quality (condition A).

Per stake step (the foot that stays longest on the stake): foot centre (mean of the 7 sole spheres) distance from the stake
axis, and the share of sole spheres on the stake.  "Partly off" = fewer than 4 of 7 sole spheres on the stake on average.
Out: robot_sim_check/out_stakesize_s2k/summary.md
"""
import glob, json, os, re
import numpy as np, mujoco

H = "/root/gpufree-data/hnu_v2"
O = "/root/gpufree-data/robot_sim_check/out_stakesize_s2k"
EXP = {15: f"{H}/deploy_exports/v6g10xm_base_S2k_s5_mDh40/L3"}
EXP.update({r: f"{H}/deploy_exports/v6g10xm_base_S2k_s5_mDh40_r{r}/L3" for r in (20, 25)})


def steps(m, tr, feet, stk):
    dd = mujoco.MjData(m)
    dd.mocap_pos[:], dd.mocap_quat[:] = tr["mocap_pos"], tr["mocap_quat"]
    acc = {}  # (stake, side) -> [(centre distance, n spheres)]
    for q in tr["qpos"]:
        dd.qpos[:] = q
        mujoco.mj_forward(m, dd)
        on = {}
        for ci in range(dd.ncon):
            cn = dd.contact[ci]
            for ga, gb in ((cn.geom1, cn.geom2), (cn.geom2, cn.geom1)):
                if gb in feet and ga in stk:
                    on.setdefault((stk[ga], feet[gb]), (ga, set()))[1].add(gb)
        for (s, side), (ga, sph) in on.items():
            fc = np.mean([dd.geom_xpos[g][:2] for g in feet if feet[g] == side], 0)
            acc.setdefault((s, side), []).append((np.linalg.norm(fc - dd.geom_xpos[ga][:2]), len(sph)))
    out = {}
    for (s, side), v in acc.items():
        if s not in out or len(v) > len(out[s]):
            out[s] = v
    return [(float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])), len(v)) for v in out.values()]


rows = {}
for r, E in EXP.items():
    m = mujoco.MjModel.from_binary_path(f"{E}/scene.mjb")
    nm = lambda t, i: mujoco.mj_id2name(m, t, i) or ""  # noqa: E731
    feet = {i: nm(5, i)[6] for i in range(m.ngeom) if re.fullmatch(r"robot/(left|right)_foot[1-7]_collision", nm(5, i))}
    stk = {i: int(re.match(r"stake(\d+)", nm(1, int(m.geom_bodyid[i]))).group(1)) for i in range(m.ngeom)
           if re.match(r"stake\d+", nm(1, int(m.geom_bodyid[i]))) and (m.geom_contype[i] or m.geom_conaffinity[i])}
    for fA in sorted(glob.glob(f"{O}/r{r}/*_A.json")):
        n = os.path.basename(fA)[:-len("_A.json")]
        sA = 100 * np.mean([x["success_v2"] for x in json.load(open(fA))["rows"]])
        fC = f"{O}/r{r}/{n}_C.json"
        sC = 100 * np.mean([x["success_v2"] for x in json.load(open(fC))["rows"]]) if os.path.exists(fC) else float("nan")
        st = [s for tf in sorted(glob.glob(f"{O}/r{r}/trace_{n}/*.npz")) for s in steps(m, np.load(tf), feet, stk)]
        dist = np.array([s[0] for s in st]); nsp = np.array([s[1] for s in st])
        rows.setdefault(n, {})[r] = (sA, sC, 100 * dist.mean(), nsp.mean(), 100 * np.mean(nsp < 4))

with open(f"{O}/summary.md", "w") as f:
    f.write("# Stake radius sweep (Deployment_D S2k, sim2sim, 100 trials)\n\n"
            "cell: success v2 % A / C; foot centre to stake axis (cm, A); sole spheres on stake of 7 (A); stake steps partly off %\n\n")
    f.write("| layout | " + " | ".join(f"r = {r} cm" for r in EXP) + " |\n|---|" + "---|" * len(EXP) + "\n")
    for n in sorted(rows):
        f.write(f"| {n} | " + " | ".join("{:.0f} / {:.0f}; {:.1f} cm; {:.1f}/7; {:.0f} %".format(*rows[n][r]) if r in rows[n] else "-"
                                         for r in EXP) + " |\n")
print(open(f"{O}/summary.md").read())
