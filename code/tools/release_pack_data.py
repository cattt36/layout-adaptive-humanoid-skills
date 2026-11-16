"""Processed data and evaluation results for the GitHub release (user ): hnu_v2/data (references, obstacle
geometry, splits, z tables, skill libraries, annotations) and the evaluation outputs of runs/ (result dirs, reports),
packed into tar parts below 2 GB.  Absolute symlinks inside hnu_v2 (augmented copies) become relative links, so the
archives unpack anywhere; links that leave hnu_v2 are stored as files.
  python pack_data.py  ->  /root/gpufree-data/release_stage/assets/{data_part*.tar, results_part*.tar}"""
import os
import tarfile

G = "/root/gpufree-data"
H = f"{G}/hnu_v2"
OUT = f"{G}/release_stage/assets"
PART = 1_900_000_000


def pack(name, files, base):
    part, size, tf = 0, 0, None
    for fp in files:
        real = os.path.realpath(fp)
        inner = os.path.islink(fp) and real.startswith(base + "/")
        s = 0 if inner else os.path.getsize(fp)
        if tf is None or size + s > PART:
            if tf:
                tf.close()
            part += 1
            tf = tarfile.open(f"{OUT}/{name}_part{part}.tar", "w")
            size = 0
        arc = os.path.relpath(fp, base)
        if inner:
            ti = tarfile.TarInfo(arc)
            ti.type, ti.linkname = tarfile.SYMTYPE, os.path.relpath(real, os.path.dirname(fp))
            tf.addfile(ti)
        else:
            tf.add(real, arcname=arc)
        size += s
    tf.close()
    print(name, len(files), "files", part, "parts")


def walk(root, skip_ext=()):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
                  if os.path.isfile(os.path.join(d, f)) and not f.endswith(skip_ext))


def main():
    os.makedirs(OUT, exist_ok=True)
    pack("data", walk(f"{H}/data"), H)
    res = []
    for r in sorted(os.listdir(f"{H}/runs")):
        p = f"{H}/runs/{r}"
        if os.path.isfile(p):
            res.append(p)
        elif not r.startswith(("students_", "latent_", "logs", "videos")):
            res += walk(p, skip_ext=(".pt",))
    pack("results", res, H)


if __name__ == "__main__":
    main()
