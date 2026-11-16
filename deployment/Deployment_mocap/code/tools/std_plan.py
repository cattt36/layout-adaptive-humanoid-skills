"""DEPLOYMENT ONLY: planner (plan4 rules of make_z_cands_plan.py) on the STANDARD-START data.

Why: robot_sim_check/std/planned_layouts/*/<clip>_z.npy = the plan (candidate z warped to the ORIGINAL layout
clock) re-timed with the LAYOUT clip's start map.  When the candidate is another clip (D1: 029 planned on 024) the
standing / step-in part then carries the candidate's z of the layout's cut frame (029 mid-walk) -> in sim the robot
walks 0.45 m forward while it should stand (hold drift D1 0.43 m vs 0.04 m with the clip's own std z).
Here: candidate z = its own std table (z_tables_cv_f1/Z3 of data/std: stand / step / slow-in of ITS clip), warped
piecewise linearly with an extra anchor at the end of the standing + step-in part (lead):
  beam   : [0, lead, entry, end]  <- entry under the beam predicted from the distance start -> plank end
            (regression on the std library clips)
  stakes : [0, lead, stance windows ..., end]  <- windows planned as in plan4 (first onset: regression on the std
            library clips; later: candidate's swing / stance durations scaled by the stake distances)
  python std_plan.py <std planned_layouts dir> <out dir>      -> <out>/<layout>/<candidate>/ (a sim2sim layout dir)"""
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
S = f"{H}/data/std"
GEO = {"CrouchedTraverse": f"{S}/ref_bridge_video_lift/CrouchedTraverse", "ThreeStepPoleRun": f"{S}/ref_stakes_s93_h10/ThreeStepPoleRun",
       "FiveStepPoleRun": f"{S}/ref_stakes_s83_h10/FiveStepPoleRun"}
ZD = f"{S}/z_tables_cv_f1/Z3"
split = json.load(open(f"{H}/data/clip_split_cv/fold1/split.json"))


def lead_of(m):
    s = m["src_frame"]
    return int(np.where(s > s[0] + 1e-3)[0][0]) - 1 if (s > s[0] + 1e-3).any() else 0


def beam(mf, bf):
    m = np.load(mf)
    p = json.load(open(os.path.realpath(bf)))["parts"]["plank"]
    c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, nn = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - c[:2]
    inside = (np.abs(rel @ nn) < 0.25) & (np.abs(rel @ d) < half[0])
    entry = int(np.where(inside.any(1))[0][0]) if inside.any() else -1
    p0 = m["body_pos_w"][0, 0, :2]
    return float(min(np.linalg.norm(c[:2] + s * d * half[0] - p0) for s in (1, -1))), entry, len(m["body_pos_w"]), lead_of(m)


def stakes(mf, sf):
    j = json.load(open(sf))
    m = np.load(mf)
    pts = np.array([p[:2] for p in j["stake_xyz_50hz"]])
    win = np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])
    return pts, m["body_pos_w"][0, 0, :2], win, len(m["body_pos_w"]), lead_of(m)


def lib(t):
    return sorted(n for n in split["train"] if n.startswith(t) and os.path.exists(f"{GEO[t]}/{n}_motion.npz")
                  and os.path.exists(f"{ZD}/{n}_z.npy"))


LIB = {t: lib(t) for t in GEO}
BM = {n: beam(f"{GEO['CrouchedTraverse']}/{n}_motion.npz", f"{GEO['CrouchedTraverse']}/{n}_bridge.json") for n in LIB["CrouchedTraverse"]}
bl = [n for n in BM if BM[n][1] > BM[n][3]]
RD = np.polyfit([BM[n][0] for n in bl], [BM[n][1] for n in bl], 1)
ST = {n: stakes(f"{GEO[t]}/{n}_motion.npz", f"{GEO[t]}/{n}_stakes.json") for t in ("ThreeStepPoleRun", "FiveStepPoleRun") for n in LIB[t]}
REG = {t: np.polyfit([np.linalg.norm(ST[n][0][0] - ST[n][1]) for n in LIB[t]], [ST[n][2][0, 0] for n in LIB[t]], 1)
       for t in ("ThreeStepPoleRun", "FiveStepPoleRun")}


def plan_w(dst, src):  # plan4 (make_z_cands_plan.plan_w) on std data
    pd, p0d, _, Td = dst[:4]
    ps, p0s, ws = src[:3]
    seg = lambda p0, p: np.linalg.norm(np.diff(np.vstack([p0, p]), axis=0), axis=1)  # noqa: E731
    r = np.clip(seg(p0d, pd) / np.maximum(seg(p0s, ps), 1e-3), 0.6, 1.6)
    gaps = np.diff(np.concatenate([[0.0], ws.flatten()]))[0::2]
    stance = ws[:, 1] - ws[:, 0]
    out, t = [], 0.0
    for k in range(len(ws)):
        a = float(np.polyval(REG[dst[5]], np.linalg.norm(pd[0] - p0d))) if k == 0 else t + gaps[k] * r[k]
        out.append((a, a + stance[k]))
        t = a + stance[k]
    out = np.array(out)
    if out[-1, 1] > Td - 11:
        out *= (Td - 11) / out[-1, 1]
    return out, ws


def warp(z, T, ad, asrc):
    if not (np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)):
        return None
    tt = np.clip(np.interp(np.arange(T), ad, asrc), 0, len(z) - 1)
    return np.stack([np.interp(tt, np.arange(len(z)), z[:, j]) for j in range(z.shape[1])], 1).astype(z.dtype)


def main(lay_root, out_root):
    info = {}
    for L in sorted(os.listdir(lay_root)):
        ld = os.path.join(lay_root, L)
        clip = [f[:-len("_motion.npz")] for f in os.listdir(ld) if f.endswith("_motion.npz")][0]
        t = clip.split("_")[0]
        info[L] = {}
        if t == "CrouchedTraverse":
            dist, _, T, lead = beam(f"{ld}/{clip}_motion.npz", f"{ld}/{clip}_bridge.json")
            pred = float(np.clip(np.polyval(RD, dist), lead + 25, T - 30))
        else:
            sj = json.load(open(f"{ld}/{clip}_stakes.json"))
            dst = stakes(f"{ld}/{clip}_motion.npz", f"{ld}/{clip}_stakes.json") + (t,)
            T, lead = dst[3], dst[4]
        for c in LIB[t]:
            zc = np.load(f"{ZD}/{c}_z.npy")
            if t == "CrouchedTraverse":
                _, entry, Tc, lead_c = BM[c]
                z = warp(zc, T, [0.0, lead, pred, T - 1.0], [0.0, lead_c, entry, Tc - 1.0])
                plan = dict(lead=lead, entry_pred=round(pred, 1), cand_lead=lead_c, cand_entry=entry)
            else:
                if len(ST[c][0]) != len(dst[0]):
                    continue
                pw, sw = plan_w(dst, ST[c])
                lead_c = ST[c][4]
                z = warp(zc, T, [0.0, lead] + pw.flatten().tolist() + [T - 1.0], [0.0, lead_c] + sw.flatten().tolist() + [len(zc) - 1.0])
                plan = dict(lead=lead, cand_lead=lead_c, windows=pw.round(1).tolist())
            if z is None:
                info[L][c] = "not monotonic (skipped)"
                continue
            o = os.path.join(out_root, L, c)
            os.makedirs(o, exist_ok=True)
            for f in os.listdir(ld):
                if f.endswith("_z.npy"):
                    continue
                shutil.copy(os.path.join(ld, f), o)
            if t != "CrouchedTraverse":  # environment clock = planned windows
                j = json.loads(json.dumps(sj))
                for s, (a, b) in zip(j["stakes"], pw):
                    s["stance"] = [int(round(a * j["fps"] / 50.0)), int(round(b * j["fps"] / 50.0))]
                j["planned_from"] = c
                j["std_plan"] = "tools/std_plan.py "
                json.dump(j, open(os.path.join(o, f"{clip}_stakes.json"), "w"))
            np.save(os.path.join(o, f"{clip}_z.npy"), z)
            json.dump(dict(layout=L, clip=clip, candidate=c, **plan), open(os.path.join(o, "std_plan.json"), "w"), indent=1)
            info[L][c] = plan
    json.dump(dict(library=LIB, beam_reg=RD.tolist(), stake_reg={k: v.tolist() for k, v in REG.items()}, plans=info),
              open(os.path.join(out_root, "std_plans.json"), "w"), indent=1)
    print({L: len([v for v in d.values() if isinstance(v, dict)]) for L, d in info.items()}, "RD", RD.round(2), {k: v.round(2) for k, v in REG.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
