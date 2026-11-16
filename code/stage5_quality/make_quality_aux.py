"""Per-frame auxiliary targets for the executability branch (z_quality), level by level.

Each 50 Hz clip frame is mapped to the annotated gait segment that contains it (segment time in seconds of the
mocap trial; clip frame f <-> source time (crop0 + f * fps / 50) / fps).  The segment's numeric metrics are
flattened and z-scored over all segments of the task family:
  robot  : Retargeted_Robot_Metrics.kinematics / dynamics            (levels L1-L4)
  human  : Human_Metrics.contact_state / step_info / obstacle_relation
           / kinematics / dynamics (incl. GRF, joint moments, stability)  (levels L3-L4)
Only the targets a level is allowed to see (same rule as the latent ablation) are written.  Training-time only:
the Q_exec input stays proprioception + z_skill.

  python make_quality_aux.py --annot <annot_v2/LLM_annotation> --ref <ref_v1> --level Z3 --out <aux_Z3.npz>
"""
import argparse
import glob
import json
import os

import numpy as np

LEVEL_SOURCES = {"Z0": (), "Z1": ("robot",), "Z2": ("robot",), "Z3": ("robot", "human"), "Z4": ("robot", "human")}
NAMESPACES = {"robot": ("Retargeted_Robot_Metrics", ("kinematics", "dynamics")),
              "human": ("Human_Metrics", ("contact_state", "step_info", "obstacle_relation", "kinematics", "dynamics"))}
SKIP = ("time_s", "_frame", "index")


def flatten(x, prefix, out):
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f"{prefix}.{k}", out)
    elif isinstance(x, bool):
        out[prefix] = float(x)
    elif isinstance(x, (int, float)) and np.isfinite(x):
        if not any(s in prefix for s in SKIP):
            out[prefix] = float(x)


def segment_vector(seg, sources):
    v = {}
    for src in sources:
        top, names = NAMESPACES[src]
        block = seg.get(top, {}) or {}
        for n in names:
            flatten(block.get(n, {}), f"{src}.{n}", v)
    return v


def human_grf(csv_path, body_weight_kg, n_mocap):
    """Per mocap frame (120 Hz): total vertical GRF / body weight, horizontal-to-vertical ratio (friction demand),
    vertical loading rate (BW/s).  From the cleaned force-plate export (1000 Hz, plates summed)."""
    import pandas as pd
    ncol = len(open(csv_path, encoding="utf-8-sig").readline().rstrip("\n").split(","))
    raw = pd.read_csv(csv_path, header=None, skiprows=1, names=range(ncol + 2), on_bad_lines="skip").to_numpy(float)
    raw = raw[:, :3 + 9 * ((ncol - 3) // 9)]
    raw = raw[np.isfinite(raw[:, 2])]
    raw = np.nan_to_num(raw)
    mocap = raw[:, 2].astype(int) - 1
    blocks = raw[:, 3:].reshape(len(raw), -1, 9)
    fx, fy, fz = blocks[:, :, 1].sum(1), blocks[:, :, 2].sum(1), blocks[:, :, 3].sum(1)
    out = np.full((n_mocap, 3), np.nan)
    ok = (mocap >= 0) & (mocap < n_mocap)
    bw = 9.81 * body_weight_kg
    for k, v in enumerate((np.abs(fz) / bw, np.hypot(fx, fy) / np.maximum(np.abs(fz), 0.2 * bw))):
        s = np.bincount(mocap[ok], v[ok], n_mocap)
        c = np.bincount(mocap[ok], None, n_mocap)
        out[c > 0, k] = s[c > 0] / c[c > 0]
    out[:, 1] = np.where(out[:, 0] > 0.2, out[:, 1], 0.0)  # friction demand only meaningful under load
    out[1:, 2] = np.diff(out[:, 0]) * 120.0
    # plates only cover the instrumented steps: keep the span between the first and last loaded plate frame
    loaded = np.where(out[:, 0] > 0.2)[0]
    if len(loaded):
        out[: loaded[0]] = np.nan
        out[loaded[-1] + 1:] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--level", required=True, choices=list(LEVEL_SOURCES))
    ap.add_argument("--min-coverage", type=float, default=0.1, help="keep metrics present in >= this share of segments")
    ap.add_argument("--forceplate", default=None, help="<dir>/<task>/<trial>_ForcePlate_CleanUp.csv (human GRF targets)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sources = LEVEL_SOURCES[a.level]
    clips, bw = {}, {}
    for f in sorted(glob.glob(os.path.join(a.annot, "*", "*.json"))):
        doc = json.load(open(f, encoding="utf-8-sig"))
        name = os.path.basename(f)[:-5]
        segs = [(s["time"]["start_s"], s["time"]["end_s"], segment_vector(s, sources)) for s in doc.get("segments", [])]
        clips[name] = segs
        bw[name] = float((doc.get("subject_model") or {}).get("body_weight_kg") or 70.0)
    use_grf = "human" in sources and a.forceplate is not None
    GRF_KEYS = ["human.grf.vertical_bw", "human.grf.friction_demand", "human.grf.loading_rate_bw_s"]
    if not sources:
        np.savez_compressed(a.out, names=np.array([]), keys=np.array([]))
        print("level", a.level, "no auxiliary targets")
        return
    allseg = [v for segs in clips.values() for _, _, v in segs]
    count = {}
    for v in allseg:
        for k in v:
            count[k] = count.get(k, 0) + 1
    keys = sorted(k for k, c in count.items() if c >= a.min_coverage * len(allseg))
    M = np.array([[v.get(k, np.nan) for k in keys] for v in allseg])
    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-6
    keep = sd > 1e-5
    keys = [k for k, ok in zip(keys, keep) if ok]
    mu, sd = mu[keep], sd[keep]
    out = {}
    names = []
    for name, segs in clips.items():
        task = name.rsplit("_", 1)[0]
        dyn = os.path.join(a.ref, task, f"{name}_Dynamics.npz")
        mot = os.path.join(a.ref, task, f"{name}_motion.npz")
        if not (os.path.exists(dyn) and os.path.exists(mot)):
            continue
        d = np.load(dyn)
        crop0, fps = int(d["source_crop"][0]), float(d["fps"])
        n = len(np.load(mot)["joint_pos"])
        tgt = np.full((n, len(keys)), np.nan, np.float32)
        t = (crop0 + np.arange(n) * fps / 50.0) / fps
        for s0, s1, v in segs:
            m = (t >= s0) & (t < s1)
            if m.any():
                row = np.array([v.get(k, np.nan) for k in keys])
                tgt[m] = ((row - mu) / sd).clip(-5, 5)
        if use_grf:
            fp = os.path.join(a.forceplate, task, f"{name}_ForcePlate_CleanUp.csv")
            g = np.full((n, 3), np.nan, np.float32)
            if os.path.exists(fp):
                mi = np.round(crop0 + np.arange(n) * fps / 50.0).astype(int)
                G = human_grf(fp, bw[name], int(mi.max()) + 2)
                g = G[np.clip(mi, 0, len(G) - 1)].astype(np.float32)
            tgt = np.concatenate([tgt, g], 1)
        out[name] = tgt
        names.append(name)
    if use_grf:  # z-score the GRF channels over all clips
        allg = np.concatenate([v[:, -3:] for v in out.values()])
        gm, gs = np.nanmean(allg, 0), np.nanstd(allg, 0) + 1e-6
        print("GRF raw mean/std (vertical BW, friction demand, loading rate BW/s):", np.round(gm, 3), np.round(gs, 3),
              "p95 vertical", np.round(np.nanpercentile(allg[:, 0], 95), 2))
        for k in out:
            out[k][:, -3:] = ((out[k][:, -3:] - gm) / gs).clip(-5, 5)
        keys = keys + GRF_KEYS
    np.savez_compressed(a.out, names=np.array(names), keys=np.array(keys), **{f"t_{k}": v for k, v in out.items()})
    cov = np.mean([np.isfinite(v).any(1).mean() for v in out.values()])
    print(f"level {a.level}: {len(keys)} targets ({sum(k.startswith('human') for k in keys)} human), "
          f"{len(names)} clips, frame coverage {cov:.2f}")
    print("human targets:", [k for k in keys if k.startswith("human")])


if __name__ == "__main__":
    main()
