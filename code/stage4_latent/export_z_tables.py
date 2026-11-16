"""Export per-frame z_skill tables (50 Hz, aligned with the mjlab motion clips) from a trained latent model.

  python export_z_tables.py --profile Z0 --latent-root <runs/latent_v1> --motion-root <data/ref_v1> --out <z_tables/Z0>

Every 50 Hz frame f of a clip gets the z_skill of the 96-frame (120 Hz) window centred on the matching
reference frame -- the same encoder input the latent was trained on.  All trials (train/val/test) are exported;
the split is kept in <out>/split.json so students can be evaluated on unseen subjects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motion_latent_z as M  # noqa: E402
from retrieve_skill import build_model  # noqa: E402

H = "/root/gpufree-data/hnu_v2"


def ckpt_normalizers(ck):
    """The normalizers the encoder was TRAINED with (stored in the checkpoint, fitted on its own train split).
    the export used to re-fit them through prepare_data() in its default split mode ('subject'), i.e.
    statistics that can include held-out trials; HNU_Z_REFIT_NORM=1 restores the old behaviour."""
    d = ck.get("normalizers")
    if os.environ.get("HNU_Z_REFIT_NORM") == "1" or not isinstance(d, dict) or "skill" not in d:
        return None
    mk = lambda cls, j: cls(np.asarray(j["mean"], np.float32), np.asarray(j["std"], np.float32),  # noqa: E731
                            np.asarray(j["lower"], np.float32), np.asarray(j["upper"], np.float32), float(j.get("z_clip", 8.0)))
    return M.Normalizers(skill=mk(M.Standardizer, d["skill"]), quality_frame=mk(M.Standardizer, d["quality_frame"]),
                         task=mk(M.Standardizer, d["task"]), semantic=mk(M.MaskedStandardizer, d["semantic"]),
                         quality_target=mk(M.MaskedStandardizer, d["quality_target"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--latent-root", required=True)
    ap.add_argument("--motion-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck_path = Path(a.latent_root) / a.profile / "best_motion_latent.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cka = ck.get("args", {}) or {}
    os.environ["HNU_TEXT_ENCODER"] = str(cka.get("text_encoder", "hash"))
    model, schema_dict, _ = build_model(ck)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    argv = ["--mode", "train", "--datasets", "CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun", "--allow-mixed-datasets",
            "--json-root", f"{H}/data/{os.environ.get('ANNOT', 'annot_v1')}/LLM_annotation", "--npz-root", f"{H}/data/latent_npz_v1",
            "--retargeted-json-root", f"{H}/data/{os.environ.get('ANNOT', 'annot_v1')}/Retargeted/Dynamics", "--ablation-profile", a.profile,
            "--output-dir", str(Path(a.out) / "_prep"), "--seed", str(cka.get("seed", 42)),
            "--text-encoder", str(cka.get("text_encoder", "hash")), "--text-dim", str(cka.get("text_dim", 128))]
    parser = M.build_parser()
    args = parser.parse_args(argv)
    M.apply_profile_defaults(args, argv)
    docs, schema, train, val, test, normalizers, _ = M.prepare_data(args)
    # use the normalizers stored with the checkpoint (the encoder's own training statistics)
    normalizers = ckpt_normalizers(ck) or normalizers
    print("normalizers:", "checkpoint" if ckpt_normalizers(ck) is not None else "re-fitted (subject split)")
    seqs = train + val + test
    ds = M.MotionWindowDataset(seqs, schema, normalizers, window=args.window, stride=args.stride,
                               skill_local_pool_frames=args.skill_local_pool_frames)
    os.makedirs(a.out, exist_ok=True)
    specs, owners = [], []
    for ti, seq in enumerate(seqs):
        motion = os.path.join(a.motion_root, seq.npz_path.parent.name, f"{seq.trial_id}_motion.npz")
        T50 = np.load(motion)["joint_pos"].shape[0]
        T120 = len(seq.skill_frames)
        for f in range(T50):
            c = min(int(round(f * 120.0 / 50.0)), T120 - 1)
            specs.append(M.WindowSpec(trial_index=ti, start_index=c - args.window // 2, center_index=c,
                                      segment_index=M.nearest_segment_index(seq.segments, c),
                                      semantic_purity=1.0, semantic_window_weight=1.0))
            owners.append(ti)
    ds.specs = specs
    for spec in specs:  # same bookkeeping as MotionWindowDataset.__init__
        seq = seqs[spec.trial_index]
        seg = seq.segments[spec.segment_index]
        ds.segment_group_ids.setdefault((seq.trial_id, seg.segment_id), len(ds.segment_group_ids))
        ds.trial_group_ids.setdefault(seq.trial_id, len(ds.trial_group_ids))
        ds.phase_class_ids.setdefault((seq.npz_path.parent.name, seg.gait_phase), len(ds.phase_class_ids))
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    zs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(M.move_batch_to_device(batch, device))
            zs.append(out["z_skill"].cpu().numpy())
    Z = np.concatenate(zs).astype(np.float32)
    owners = np.array(owners)
    for ti, seq in enumerate(seqs):
        np.save(os.path.join(a.out, f"{seq.trial_id}_z.npy"), Z[owners == ti])
    split = dict(train=[s.trial_id for s in train], val=[s.trial_id for s in val], test=[s.trial_id for s in test],
                 latent_checkpoint=str(ck_path), z_dim=int(Z.shape[1]))
    json.dump(split, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(json.dumps(dict(profile=a.profile, clips=len(seqs), frames=int(len(Z)), z_dim=int(Z.shape[1]),
                          z_std=float(Z.std(0).mean()))))


if __name__ == "__main__":
    main()
