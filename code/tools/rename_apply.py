"""task rename (user decision, English names for the paper):
  WuBuZhuang1 -> ThreeStepPoleRun  (三步桩, "Three-step Pole Run", 3 stakes)
  WuBuZhuang2 -> FiveStepPoleRun   (五步桩, "Five-step Pole Run", 5 stakes)
  DuMuQiaoXia -> CrouchedTraverse  (独木桥, "Crouched traverse under a bridge")
Copies <src> to <dst> applying the rename to path names, text files and the strings stored inside
npz / npy / pt / zip / docx files.  Rules:
  EN  identifiers as above (+ case variants, W1_007 / DuMu_004 style ids); a bare "WuBuZhuang" (both stake tasks) ->
      "ThreeStepPoleRun/FiveStepPoleRun"; mocap rigid body DuMuQiao_Da -> Bridge_Da
  ZH  五步桩1 / 五步桩一 / 五步桩动作1 / 五步桩第一组 -> 三步桩, same with 2 -> 五步桩; bare 五步桩 (both) -> 三步桩/五步桩
  router instruction sets (router/data_lang/*.json): EN rules only -- the instruction sentences stay as collected
  obstacle assets Supplement/WuBuZhuang.usd -> Pole.usd, Supplement/DuMuQiao.usd -> Bridge.usd
python rename_apply.py <src> <dst>  -> prints a per-file change report"""
import io
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import PurePath

import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
TEXT = {".md", ".py", ".sh", ".ps1", ".json", ".csv", ".txt", ".yaml", ".yml", ".bvh", ".log", ".jsonl", ".cfg", ".toml", ".ini"}
SPECIAL = [("五步桩1/2", "三步桩/五步桩"), ("W1 56 / W2 76 / DuMu 36", "三步桩 56 / 五步桩 76 / 独木桥 36"),
           ("W1 / W2", "ThreeStepPoleRun / FiveStepPoleRun"), ('startswith("DuMu")', 'startswith("CrouchedTraverse")'),
           ("DuMuQiao_Da", "Bridge_Da"),  # DuMuQiao_Da = mocap rigid body of the bridge
           ("/ DuMu /", "/ CrouchedTraverse /"),
           # server code: prefix tests / globs that meant "both stake tasks"
           ('startswith("WuBu")', 'startswith(("ThreeStepPoleRun", "FiveStepPoleRun"))'),
           ("WuBuZhuang?", "*StepPoleRun"), ("WuBuZhuang[12]", "(Three|Five)StepPoleRun")]
EN_BARE = re.compile(r"WuBuZhuang(?![12])")
EN = [(re.compile(r"WuBuZhuang1"), "ThreeStepPoleRun"), (re.compile(r"WuBuZhuang2"), "FiveStepPoleRun"),
      (re.compile(r"(?i)wubuzhuang1"), "ThreeStepPoleRun"), (re.compile(r"(?i)wubuzhuang2"), "FiveStepPoleRun"),
      # DuMuQiaoXia only: the 5 "DuMuQiao-Xiao_*" BVH recordings in Human metrics are different trials (not the
      # project's bridge clips; renaming them would collide with CrouchedTraverse_001_A) -> kept, asked the user
      (re.compile(r"(?i)dumuqiaoxia"), "CrouchedTraverse"),
      (re.compile(r"(?<![A-Za-z0-9_])W1_(?=\d{3})"), "ThreeStepPoleRun_"),
      (re.compile(r"(?<![A-Za-z0-9_])W2_(?=\d{3})"), "FiveStepPoleRun_"),
      (re.compile(r"(?<![A-Za-z0-9_])DuMu_(?=\d{3})"), "CrouchedTraverse_")]
ZH_NUM = r"(?:\s*1(?![0-9.])|\s*一|动作\s*1(?![0-9.])|第一组)"
ZH_NUM2 = r"(?:\s*2(?![0-9.])|\s*二|动作\s*2(?![0-9.])|第二组)"
ZH_BARE = re.compile(r"五步桩(?!" + ZH_NUM[3:-1] + r")(?!" + ZH_NUM2[3:-1] + r")")
SP = r"(?: (?=[一-鿿]))?"  # "五步桩1 是" -> "三步桩是" (the space only separated the digit from the Chinese text)
ZH = [(re.compile(r"五步桩" + ZH_NUM + SP), "三步桩"), (re.compile(r"五步桩" + ZH_NUM2 + SP), "五步桩")]
P_BOTH_EN, P_BOTH_ZH, P_FIVE = "\x00BOTHEN\x00", "\x00BOTHZH\x00", "\x00FIVE\x00"
log = Counter()
report = []


def ren(s, zh=True, where=""):
    """apply all rules to a string; counts per rule in `log`"""
    for i, (a, b) in enumerate(SPECIAL):  # placeholders, so that the replacement is not renamed again
        if a in s:
            log["special " + a] += s.count(a)
            s = s.replace(a, f"\x00SP{i}\x00")
    n = len(EN_BARE.findall(s))
    if n:
        log["EN bare"] += n
        s = EN_BARE.sub(P_BOTH_EN, s)
    if zh:
        n = len(ZH_BARE.findall(s))
        if n:
            log["ZH bare"] += n
            s = ZH_BARE.sub(P_BOTH_ZH, s)
        for p, r in ZH:
            k = len(p.findall(s))
            if k:
                log[f"ZH -> {r}"] += k
                s = p.sub(P_FIVE if r == "五步桩" else r, s)
    for p, r in EN:
        k = len(p.findall(s))
        if k:
            log[f"EN -> {r}"] += k
            s = p.sub(r, s)
    s = s.replace(P_BOTH_EN, "ThreeStepPoleRun/FiveStepPoleRun").replace(P_BOTH_ZH, "三步桩/五步桩").replace(P_FIVE, "五步桩")
    for i, (a, b) in enumerate(SPECIAL):
        s = s.replace(f"\x00SP{i}\x00", b)
    return s


def ren_path(rel):
    parts = rel.split(os.sep)
    if parts[0] == "Supplement" and parts[-1] in ("WuBuZhuang.usd", "DuMuQiao.usd"):  # obstacle assets
        parts[-1] = {"WuBuZhuang.usd": "Pole.usd", "DuMuQiao.usd": "Bridge.usd"}[parts[-1]]
    return os.sep.join(ren(p, where="path") for p in parts)


def ren_obj(o):  # recursive rename of strings inside python objects (pt / pickled)
    if isinstance(o, str):
        return ren(o)
    if isinstance(o, PurePath):  # pathlib paths stored in checkpoint args
        return type(o)(ren(str(o)))
    if isinstance(o, dict):
        if hasattr(o, "_metadata"):  # torch state_dict: parameter names only, keep
            return o
        items = [(ren_obj(k), ren_obj(v)) for k, v in o.items()]
        try:
            return type(o)(items)
        except TypeError:
            return dict(items)
    if isinstance(o, list):
        return [ren_obj(v) for v in o]
    if isinstance(o, tuple):
        return tuple(ren_obj(v) for v in o)
    if isinstance(o, np.ndarray) and o.dtype.kind in "USO":
        return np.vectorize(lambda x: ren(x) if isinstance(x, str) else x, otypes=[o.dtype])(o) if o.size else o
    return o


def ren_arr(v):
    if v.dtype.kind == "U":
        return np.array([ren(x) for x in v.ravel()], dtype=v.dtype if v.size == 0 else None).reshape(v.shape)
    if v.dtype.kind == "S":
        return np.array([ren(x.decode()).encode() for x in v.ravel()]).reshape(v.shape)
    if v.dtype.kind == "O":
        return ren_obj(v)
    return v


def npz_bytes(data, compressed):
    b = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(b, **data)
    return b.getvalue()


def do_npz(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        comp = any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist())
    d = np.load(io.BytesIO(raw), allow_pickle=True)
    old = re.compile(r"(?i)wubuzhuang|dumuqiao|五步桩")
    arrs = {k: d[k] for k in d.files}
    changed = any(old.search(k) for k in arrs) or any(
        v.dtype.kind in "USO" and old.search(str(v.tolist())) for v in arrs.values())
    if not changed:
        return raw, False
    return npz_bytes({ren(k): ren_arr(v) for k, v in arrs.items()}, comp), True


def do_docx(src, dst):
    from docx import Document
    doc = Document(src)
    n = 0

    def paras(container):
        for p in container.paragraphs:
            yield p
        for t in container.tables:
            for row in t.rows:
                for c in row.cells:
                    yield from paras(c)
    allp = list(paras(doc))
    for s in doc.sections:
        allp += list(s.header.paragraphs) + list(s.footer.paragraphs)
    for p in allp:
        full = "".join(r.text for r in p.runs)
        new = ren(full)
        if new == full:
            continue
        n += 1
        trial = [ren(r.text) for r in p.runs]  # run level keeps the formatting
        if "".join(trial) == new:
            for r, t in zip(p.runs, trial):
                if r.text != t:
                    r.text = t
        else:  # the name is split over runs: put the paragraph text into the first run
            p.runs[0].text = new
            for r in p.runs[1:]:
                r.text = ""
            log["docx merged paragraph"] += 1
    doc.save(dst)
    return n


LINK = os.environ.get("HNU_RENAME_LINK") == "1"  # server mode: unchanged files are hard-linked (no extra space)
OLDB = (b"WuBuZhuang", b"wubuzhuang", b"DuMuQiaoXia", b"dumuqiaoxia", b"DuMuQiao_Da", "五步桩".encode())


def keep(f, out):  # unchanged content
    if LINK:
        os.link(f, out)
    else:
        shutil.copy2(f, out)


def process(rel):
    """rename one file (or symlink) rel -> DST; returns (report line or None, Counter of rule hits)"""
    log.clear()
    if rel.endswith(".pyc"):  # bytecode caches are regenerated, not copied
        return None, Counter()
    f = os.path.join(SRC, rel)
    nrel = ren_path(rel)
    out = os.path.join(DST, nrel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.islink(f):  # recreate the link with the renamed target
        t = os.readlink(f)
        os.symlink(ren_path(t) if not os.path.isabs(t) else "/" + ren_path(t.lstrip("/")), out)
        return (f"{0:6d}  {'symlink':16s} {rel}  ->  {nrel}" if nrel != rel or ren_path(t) != t else None), Counter()
    ext = os.path.splitext(rel)[1].lower()
    what = ""
    if ext not in TEXT and not os.path.islink(f) and os.path.getsize(f) < 5_000_000 and ext not in (".npz", ".npy", ".pt", ".pth", ".zip", ".docx"):
        try:  # unknown extension (e.g. foo.py.bak): treat as text if it is UTF-8 and mentions an old name
            raw = open(f, "rb").read()
            if any(o in raw for o in OLDB):
                raw.decode("utf-8")
                ext = ".txt"
        except UnicodeDecodeError:
            pass
    try:
        if ext in TEXT:
            raw = open(f, "rb").read()
            bom = raw.startswith(b"\xef\xbb\xbf")
            try:
                s, enc = raw.decode("utf-8-sig"), "utf-8"
            except UnicodeDecodeError:  # Supplement CSVs are GBK (Chinese Windows); keep the encoding
                s, enc = raw.decode("gbk"), "gbk"
            zh = not re.search(r"(^|[\\/])data_lang[\\/]", rel)  # instruction sets: sentences stay as collected
            s2 = ren(s, zh=zh)
            if s2 == s:
                keep(f, out)
            else:
                open(out, "wb").write((b"\xef\xbb\xbf" if bom else b"") + s2.encode(enc))
                shutil.copystat(f, out)
            what = "text" + ("" if zh else " (EN only)") + ("" if enc == "utf-8" else " gbk")
        elif ext == ".npz":
            raw = open(f, "rb").read()
            b, ch = do_npz(raw)
            if ch:
                open(out, "wb").write(b)
                shutil.copystat(f, out)
            else:
                keep(f, out)
            what = "npz"
        elif ext == ".npy":
            v = np.load(f, allow_pickle=True)
            if v.dtype.kind in "USO" and re.search(r"(?i)wubuzhuang|dumuqiao|五步桩", str(v.tolist())):
                np.save(out, ren_arr(v))
            else:
                keep(f, out)
            what = "npy"
        elif ext in (".pt", ".pth"):
            with zipfile.ZipFile(f) as z:
                pk = b"".join(z.read(n) for n in z.namelist() if n.endswith(".pkl"))
            if any(o in pk for o in OLDB):
                import torch
                torch.save(ren_obj(torch.load(f, map_location="cpu", weights_only=False)), out)
                shutil.copystat(f, out)
            else:
                keep(f, out)
            what = "pt"
        elif ext == ".zip":
            zin = zipfile.ZipFile(f)
            with zipfile.ZipFile(out, "w") as zout:
                for info in zin.infolist():
                    data = zin.read(info.filename)
                    e2 = os.path.splitext(info.filename)[1].lower()
                    if e2 == ".npz":
                        data, _ = do_npz(data)
                    elif e2 in TEXT:
                        data = ren(data.decode("utf-8-sig")).encode("utf-8")
                    ni = zipfile.ZipInfo(ren(info.filename), date_time=info.date_time)
                    ni.compress_type, ni.external_attr = info.compress_type, info.external_attr
                    zout.writestr(ni, data)
            what = "zip"
        elif ext == ".docx":
            do_docx(f, out)
            what = "docx"
        else:  # binaries (onnx, mjb, mp4, tfevents, ...) are kept as they are
            keep(f, out)
    except Exception as ex:
        if os.path.lexists(out):
            os.remove(out)
        keep(f, out)
        return f"ERROR {rel}: {type(ex).__name__} {ex}"[:300], Counter()
    n = sum(log.values())
    line = f"{n:6d}  {what:16s} {rel}" + (f"  ->  {nrel}" if nrel != rel else "") if (n or nrel != rel) else None
    return line, Counter(log)


if __name__ == "__main__":
    from multiprocessing import Pool
    rels = []
    for dp, dns, fns in os.walk(SRC):
        for d in dns:  # empty directories too; symlinked directories are recreated as links
            p = os.path.join(dp, d)
            if os.path.islink(p):
                rels.append(os.path.relpath(p, SRC))
            else:
                os.makedirs(os.path.join(DST, ren_path(os.path.relpath(p, SRC))), exist_ok=True)
        rels += [os.path.relpath(os.path.join(dp, fn), SRC) for fn in fns]
    total = Counter()
    with Pool(int(os.environ.get("HNU_RENAME_WORKERS", "8"))) as pool:
        for line, c in pool.imap_unordered(process, rels, chunksize=16):
            total.update(c)
            if line:
                report.append(line)
    print("== rule totals")
    for k, v in sorted(total.items()):
        print(f"  {k}: {v}")
    print(f"== files: {len(rels)} processed, {len(report)} changed or renamed, {sum(r.startswith('ERROR') for r in report)} errors")
    print("\n".join(sorted(report)))
