"""DEPLOYMENT ONLY: standard start (make_start.py) for every training clip / augmented variant + the z tables.

  python make_start_batch.py <scene.mjb> <policy_meta.json> <data_root> <out_root> [workers]

Motion roots (relative to data_root, mirrored under out_root): ref_bridge_video_lift/CrouchedTraverse,
ref_stakes_s93_h10/ThreeStepPoleRun, ref_stakes_s83_h10/FiveStepPoleRun and the aug_xzmix_f1 copies of them.  For every
clip the new motion (with src_frame), the remapped stakes json (stance windows), the copied bridge json; z tables of
z_tables_cv_f1_xzmix/Z3 and z_tables_cv_f1/Z3 remapped with the clip's src_frame (clip name = z file name).  Clips
without a valid cut frame are listed in <out_root>/std_start_skipped.txt and left out (the deployment training uses only
standard-start clips).  Summary in <out_root>/std_start_summary.json.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOTS = ["ref_bridge_video_lift/CrouchedTraverse", "ref_stakes_s93_h10/ThreeStepPoleRun", "ref_stakes_s83_h10/FiveStepPoleRun",
         "aug_xzmix_f1/ref_bridge_video_lift/CrouchedTraverse", "aug_xzmix_f1/ref_stakes_s93_h10/ThreeStepPoleRun",
         "aug_xzmix_f1/ref_stakes_s83_h10/FiveStepPoleRun"]
ZDIRS = ["z_tables_cv_f1_xzmix/Z3", "z_tables_cv_f1/Z3"]
G = {}


def init(scene, meta):
    from make_leadin import Robot
    G["R"] = Robot(scene, meta)


def job(a):
    in_dir, clip, out_dir = a
    from make_start import do_clip
    try:
        out, info = do_clip(G["R"], in_dir, clip, out_dir)
    except Exception as e:  # noqa: BLE001
        return clip, out_dir, None, dict(reason=f"error {e!r}")
    if out is None:
        return clip, out_dir, None, info
    return clip, out_dir, out["src_frame"], info


def main():
    scene, meta, D, O = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    jobs = []
    for r in ROOTS:
        for f in sorted(glob.glob(os.path.join(D, r, "*_motion.npz"))):
            clip = os.path.basename(f)[:-len("_motion.npz")]
            jobs.append((os.path.join(D, r), clip, os.path.join(O, r)))
    print(len(jobs), "clips", flush=True)
    with Pool(workers, initializer=init, initargs=(scene, meta)) as p:
        res = p.map(job, jobs, chunksize=4)
    src = {}
    skipped, summary = [], {}
    for clip, od, s, info in res:
        key = f"{os.path.relpath(od, O)}/{clip}"
        summary[key] = info
        if s is None:
            skipped.append(f"{key}  {info.get('reason')}")
        else:
            src.setdefault(clip, s)  # the same clip name in several roots: identical motion (aug d1 = base)
    # z tables
    nz = 0
    for zd in ZDIRS:
        os.makedirs(os.path.join(O, zd), exist_ok=True)
        for zf in glob.glob(os.path.join(D, zd, "*_z.npy")):
            clip = os.path.basename(zf)[:-len("_z.npy")]
            if clip in src:
                z = np.load(zf)
                np.save(os.path.join(O, zd, os.path.basename(zf)), z[np.clip(np.round(src[clip]).astype(int), 0, len(z) - 1)])
                nz += 1
        for extra in ("split.json",):
            if os.path.exists(os.path.join(D, zd, extra)):
                import shutil
                shutil.copy(os.path.join(D, zd, extra), os.path.join(O, zd, extra))
    open(os.path.join(O, "std_start_skipped.txt"), "w").write("\n".join(skipped) + "\n")
    json.dump(summary, open(os.path.join(O, "std_start_summary.json"), "w"), indent=1)
    ok = len(jobs) - len(skipped)
    print(f"standard start: {ok}/{len(jobs)} clips, {nz} z tables; skipped {len(skipped)} -> std_start_skipped.txt", flush=True)


if __name__ == "__main__":
    main()
