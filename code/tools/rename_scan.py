"""Dry run for the task rename (WuBuZhuang1 -> SanBuZhuang 三步桩, WuBuZhuang2 -> WuBuZhuang 五步桩).
Scans a tree: path names, text files, npz / npy / pt / zip / docx contents.  Prints counts and the context of every
ambiguous hit (bare WuBuZhuang, bare 五步桩, W1 / W2 abbreviations).   python rename_scan.py <root>"""
import io
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np

ROOT = sys.argv[1]
TEXT = {".md", ".py", ".sh", ".ps1", ".json", ".csv", ".txt", ".yaml", ".yml", ".bvh", ".log", ".jsonl", ".cfg", ".toml"}
PAT = {
    "WuBuZhuang1": re.compile(r"WuBuZhuang1|wubuzhuang1|WUBUZHUANG1|Wubuzhuang1"),
    "WuBuZhuang2": re.compile(r"WuBuZhuang2|wubuzhuang2|WUBUZHUANG2|Wubuzhuang2"),
    "bare_WuBuZhuang": re.compile(r"(?i)wubuzhuang(?![12])"),
    "WuBu_short": re.compile(r"(?i)wubu(?!zhuang)"),
    "DuMu_any": re.compile(r"(?i)dumu"),
    "W12_id": re.compile(r"(?<![A-Za-z0-9_])[Ww][12]_\d{3}"),
    "五步桩1/2": re.compile(r"五步桩\s*[12一二]|五步桩动作\s*[12一二]|五步桩第[一二]组"),
    "bare_五步桩": re.compile(r"五步桩(?!\s*[12一二])(?!动作\s*[12一二])(?!第[一二]组)"),
    "W1W2": re.compile(r"(?<![A-Za-z0-9_])[Ww][12](?![0-9A-Za-z_])"),
}
counts = defaultdict(Counter)
ctx = defaultdict(list)
paths = Counter()


def scan_text(rel, s, kind):
    for k, p in PAT.items():
        for m in p.finditer(s):
            counts[kind][k] += 1
            if k in ("bare_WuBuZhuang", "WuBu_short", "bare_五步桩", "W1W2", "DuMu_any", "W12_id") and len(ctx[k]) < 400:
                a, b = max(0, m.start() - 40), min(len(s), m.end() + 40)
                ctx[k].append(f"{rel}: ...{s[a:b]}...".replace("\n", " "))


for dp, dns, fns in os.walk(ROOT):
    for n in dns + fns:
        if re.search(r"(?i)wubu|五步", n):
            paths[re.sub(r"_\d{3}.*", "_###", n)] += 1
    for fn in fns:
        f = os.path.join(dp, fn)
        rel = os.path.relpath(f, ROOT)
        ext = os.path.splitext(fn)[1].lower()
        try:
            if ext in TEXT:
                scan_text(rel, open(f, encoding="utf-8", errors="replace").read(), "text" + ext)
            elif ext in (".npz", ".npy"):
                d = np.load(f, allow_pickle=True)
                items = d.items() if ext == ".npz" else [("arr", d)]
                for key, v in items:
                    scan_text(rel + ":" + key, key, "npz-key")
                    if v.dtype.kind in "USO":
                        scan_text(rel + ":" + key, " ".join(map(str, v.ravel()[:100000])), "npz-str")
            elif ext in (".pt", ".pth", ".pkl"):
                with zipfile.ZipFile(f) as z:
                    for nm in z.namelist():
                        if nm.endswith(".pkl"):
                            scan_text(rel + ":" + nm, z.read(nm).decode("latin1"), "pt-pickle")
            elif ext == ".zip":
                with zipfile.ZipFile(f) as z:
                    for nm in z.namelist():
                        scan_text(rel + ":" + nm, nm, "zip-entry")
                        if os.path.splitext(nm)[1] in TEXT:
                            scan_text(rel + ":" + nm, z.read(nm).decode("utf-8", "replace"), "zip-text")
                        elif nm.endswith(".npz"):
                            d = np.load(io.BytesIO(z.read(nm)), allow_pickle=True)
                            for key, v in d.items():
                                if v.dtype.kind in "USO":
                                    scan_text(rel + ":" + nm + ":" + key, " ".join(map(str, v.ravel()[:1000])), "zip-npz-str")
            elif ext == ".docx":
                with zipfile.ZipFile(f) as z:
                    x = z.read("word/document.xml").decode("utf-8")
                    scan_text(rel, re.sub(r"<[^>]+>", "", x), "docx")
            elif ext == ".usd":
                scan_text(rel, open(f, "rb").read().decode("latin1"), "usd")
        except Exception as e:
            counts["errors"][f"{rel}: {type(e).__name__} {e}"[:200]] += 1

print("== path names (pattern: count)")
for k, v in sorted(paths.items()):
    print(f"  {k}: {v}")
print("== content counts by kind")
for kind, c in sorted(counts.items()):
    print(" ", kind, dict(c))
for k in ("bare_WuBuZhuang", "WuBu_short", "bare_五步桩", "W1W2", "DuMu_any", "W12_id"):
    print(f"== context {k} ({len(ctx[k])} shown)")
    for line in ctx[k]:
        print("  ", line[:220])
