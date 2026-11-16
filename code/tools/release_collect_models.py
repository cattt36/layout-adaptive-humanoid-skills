"""Model release for GitHub (user ): the final checkpoint of every run plus its configuration, hard-linked into
one tree with a manifest, then packed into tar parts below the 2 GB limit of GitHub release assets.
  python collect_models.py   ->  /root/gpufree-data/release_stage/{models/, MANIFEST_models.json, assets/*.tar}"""
import hashlib
import json
import os
import re
import shutil
import tarfile

G = "/root/gpufree-data"
H = f"{G}/hnu_v2"
OUT = f"{G}/release_stage"
M = f"{OUT}/models"
PART = 1_900_000_000
PAPER_TEACHERS = {"ThreeStepPoleRun": "run_rs10c", "FiveStepPoleRun": "run_tf2_full",
                  "CrouchedTraverse": "run_du3_B"}
manifest = []


def it(f):
    m = re.findall(r"(\d+)", f)
    return int(m[-1]) if m else -1


def link(src, dst, role, note=""):
    if not os.path.isfile(src):  # dangling symlinks in old 4090 copies
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    manifest.append({"file": os.path.relpath(dst, OUT), "source": os.path.relpath(src, G), "role": role,
                     "bytes": os.path.getsize(src), "note": note})


def experts():
    for task in PAPER_TEACHERS:
        root = f"{H}/runs/logs/rsl_rl/hnu_track_{task}"
        for run in sorted(os.listdir(root)):
            d = f"{root}/{run}"
            pts = [f for f in os.listdir(d) if re.fullmatch(r"model_\d+\.pt", f)]
            if not pts:
                continue
            paper = PAPER_TEACHERS[task] == run
            sub = "paper_teachers" if paper else "other_runs"
            dst = f"{M}/experts/{sub}/{task}/{run}"
            note = "teacher of every student in the paper" if paper else ""
            link(f"{d}/{max(pts, key=it)}", f"{dst}/{max(pts, key=it)}", "expert", note)
            for f in os.listdir(d):
                if f.endswith(".onnx"):
                    link(f"{d}/{f}", f"{dst}/{f}", "expert_onnx")
            for f in ("agent.yaml", "env.yaml"):
                if os.path.exists(f"{d}/params/{f}"):
                    link(f"{d}/params/{f}", f"{dst}/params/{f}", "config")


def students():
    root = f"{H}/runs"
    for run in sorted(os.listdir(root)):
        if not run.startswith("students_") or not os.path.isdir(f"{root}/{run}"):
            continue
        for d, _, fs in os.walk(f"{root}/{run}"):
            pts = [f for f in fs if re.fullmatch(r"student_\d+\.pt", f)]
            if not pts:
                continue
            dst = f"{M}/students/{os.path.relpath(d, root)}"
            link(f"{d}/{max(pts, key=it)}", f"{dst}/{max(pts, key=it)}", "student")
            for f in ("args.json", "train_log.jsonl"):
                if f in fs:
                    link(f"{d}/{f}", f"{dst}/{f}", "config" if f == "args.json" else "log")


def encoders():
    """Skill encoders: fold 1 (latent_v5t, the paper's split_v2, and latent_v5) from this server, CV folds 2-4 and the
    evidence ablations (kin / noq / rom, folds 1-4) from the 4090 copy; seeds 42-44, levels Z0-Z4."""
    roots = [f"{H}/runs/latent_v5t", f"{H}/runs/latent_v5"]
    r4 = f"{G}/hnu_v2_4090/runs"
    roots += [f"{r4}/{r}" for r in sorted(os.listdir(r4)) if re.fullmatch(r"latent_(cv|kin|noq|rom)_f\d", r)]
    for root in roots:
        for d, _, fs in os.walk(root):
            for f in fs:
                if not f.endswith(".log"):
                    link(f"{d}/{f}", f"{M}/skill_encoders/{os.path.basename(root)}/{os.path.relpath(d, root)}/{f}",
                         "skill_encoder")
def exports():
    seen = {}
    for d, _, fs in os.walk(f"{H}/deploy_exports"):
        for f in fs:
            src = f"{d}/{f}"
            rel = os.path.relpath(src, f"{H}/deploy_exports")
            if f.endswith(".mjb"):  # scenes repeat across exports: one copy per content
                h = hashlib.md5(open(src, "rb").read()).hexdigest()
                if h not in seen:
                    seen[h] = f"{M}/deploy_exports/_scenes/{h}.mjb"
                    link(src, seen[h], "scene")
                manifest.append({"file": rel, "source": os.path.relpath(src, G), "role": "scene_ref",
                                 "same_as": os.path.relpath(seen[h], OUT)})
            elif f.endswith((".onnx", ".json", ".txt", ".log")):
                link(src, f"{M}/deploy_exports/{rel}", "deploy_export")


def pack():
    os.makedirs(f"{OUT}/assets", exist_ok=True)
    for cat in sorted(os.listdir(M)):
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(f"{M}/{cat}") for f in fs)
        part, size, tf = 0, 0, None
        for fp in files:
            s = os.path.getsize(fp)
            if tf is None or size + s > PART:
                if tf:
                    tf.close()
                part += 1
                tf = tarfile.open(f"{OUT}/assets/models_{cat}_part{part}.tar", "w")
                size = 0
            tf.add(fp, arcname=os.path.relpath(fp, OUT))
            size += s
        tf.close()
        print(cat, len(files), "files", part, "parts")


def main():
    if os.path.exists(M):
        shutil.rmtree(M)
    experts()
    students()
    encoders()
    exports()
    json.dump(manifest, open(f"{OUT}/MANIFEST_models.json", "w"), indent=1)
    tot = sum(x.get("bytes", 0) for x in manifest)
    print("entries", len(manifest), "GB", round(tot / 1e9, 2))
    pack()


if __name__ == "__main__":
    main()
