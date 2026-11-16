"""z_skill tables of TIME-SCALED trials, computed by the trained latent encoder itself (not by interpolating the
1.0x z): every 120 Hz encoder input sequence is resampled at playback speed s (velocities x s, contact nearest,
segment / span boundaries / s), then exported exactly like export_z_tables.py.  Frame counts match
tools/make_timescale.py (50 Hz length round((T - 1) / s) + 1).

  python export_z_tables_ts.py --profile Z4 --latent-root runs/latent_v5t/seed42 --motion-root data/ref_v1 \
      --speeds 0.9,0.95,1.05,1.1 --only ThreeStepPoleRun,FiveStepPoleRun --out-root data/ts   (-> data/ts/ts090/z_enc/Z4)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
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


def resample(x, s, vel_cols=(), nearest_cols=()):
    T = len(x)
    Tn = int(round((T - 1) / s)) + 1
    t = np.minimum(np.arange(Tn) * s, T - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0)[:, None]
    y = ((1 - w) * x[i0] + w * x[i1]).astype(x.dtype)
    for c in nearest_cols:
        y[:, c] = x[np.where(w[:, 0] < 0.5, i0, i1), c]
    for c in vel_cols:
        y[:, c] *= s
    return y


def scale_seq(seq, s, nu):
    vel = list(range(7, 13)) + list(range(13 + nu, 13 + 2 * nu))  # root lin / ang vel, joint vel
    contact = [13 + 2 * nu]
    q = copy.copy(seq)
    q.skill_frames = resample(seq.skill_frames, s, vel, contact)
    q.quality_frames = resample(seq.quality_frames, s, vel, contact)
    q.task_targets = resample(seq.task_targets, s, list(range(7, 13)), [seq.task_targets.shape[1] - 1])
    q.phase_frames = resample(seq.phase_frames, s)
    T = len(q.skill_frames)
    f = lambda i: min(int(round(i / s)), T)  # noqa: E731
    q.span_start, q.span_end = f(seq.span_start), f(seq.span_end)
    q.segments = [dataclasses.replace(g, start_index=f(g.start_index), end_index=f(g.end_index)) for g in seq.segments]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--latent-root", required=True)
    ap.add_argument("--motion-root", required=True)
    ap.add_argument("--speeds", required=True)
    ap.add_argument("--only", default="ThreeStepPoleRun,FiveStepPoleRun")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--zname", default="z_enc", help="sub-folder name (CV folds: z_enc_f2 ...)")
    a = ap.parse_args()
    ck_path = Path(a.latent_root) / a.profile / "best_motion_latent.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cka = ck.get("args", {}) or {}
    os.environ["HNU_TEXT_ENCODER"] = str(cka.get("text_encoder", "hash"))
    model, _, _ = build_model(ck)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    argv = ["--mode", "train", "--datasets", "CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun", "--allow-mixed-datasets",
            "--json-root", f"{H}/data/{os.environ.get('ANNOT', 'annot_v1')}/LLM_annotation", "--npz-root", f"{H}/data/latent_npz_v1",
            "--retargeted-json-root", f"{H}/data/{os.environ.get('ANNOT', 'annot_v1')}/Retargeted/Dynamics", "--ablation-profile", a.profile,
            "--output-dir", str(Path(a.out_root) / f"_prep_{a.profile}"), "--seed", str(cka.get("seed", 42)),
            "--text-encoder", str(cka.get("text_encoder", "hash")), "--text-dim", str(cka.get("text_dim", 128))]
    parser = M.build_parser()
    args = parser.parse_args(argv)
    M.apply_profile_defaults(args, argv)
    docs, schema, train, val, test, normalizers, _ = M.prepare_data(args)
    from export_z_tables import ckpt_normalizers  # the encoder's own training statistics (see there)
    normalizers = ckpt_normalizers(ck) or normalizers
    only = tuple(a.only.split(","))
    base = [q for q in train + val + test if q.npz_path.parent.name in only]
    nu = (base[0].skill_frames.shape[1] - 1 - 6 - 6 - 1 - base[0].phase_frames.shape[1]) // 2
    for s in map(float, a.speeds.split(",")):
        seqs = [scale_seq(q, s, nu) for q in base]
        ds = M.MotionWindowDataset(seqs, schema, normalizers, window=args.window, stride=args.stride,
                                   skill_local_pool_frames=args.skill_local_pool_frames)
        out = f"{a.out_root}/ts{int(round(100 * s)):03d}/{a.zname}/{a.profile}"
        os.makedirs(out, exist_ok=True)
        specs, owners = [], []
        for ti, seq in enumerate(seqs):
            T50 = np.load(os.path.join(a.motion_root, seq.npz_path.parent.name, f"{seq.trial_id}_motion.npz"))["joint_pos"].shape[0]
            T50 = int(round((T50 - 1) / s)) + 1
            T120 = len(seq.skill_frames)
            for f in range(T50):
                c = min(int(round(f * 120.0 / 50.0)), T120 - 1)
                specs.append(M.WindowSpec(trial_index=ti, start_index=c - args.window // 2, center_index=c,
                                          segment_index=M.nearest_segment_index(seq.segments, c),
                                          semantic_purity=1.0, semantic_window_weight=1.0))
                owners.append(ti)
        ds.specs = specs
        for spec in specs:
            seq = seqs[spec.trial_index]
            seg = seq.segments[spec.segment_index]
            ds.segment_group_ids.setdefault((seq.trial_id, seg.segment_id), len(ds.segment_group_ids))
            ds.trial_group_ids.setdefault(seq.trial_id, len(ds.trial_group_ids))
            ds.phase_class_ids.setdefault((seq.npz_path.parent.name, seg.gait_phase), len(ds.phase_class_ids))
        loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        zs = []
        with torch.no_grad():
            for batch in loader:
                zs.append(model(M.move_batch_to_device(batch, device))["z_skill"].cpu().numpy())
        Z = np.concatenate(zs).astype(np.float32)
        owners = np.array(owners)
        for ti, seq in enumerate(seqs):
            np.save(os.path.join(out, f"{seq.trial_id}_z.npy"), Z[owners == ti])
        print(f"{a.profile} x{s}: {len(seqs)} trials, {len(Z)} frames -> {out}")


if __name__ == "__main__":
    main()
