"""Render PDF pages to PNG contact sheets (2 pages side by side) for a visual check.  python pdf_pages.py in.pdf out_prefix"""
import sys

import fitz

doc = fitz.open(sys.argv[1])
print("pages", doc.page_count)
for i in range(0, doc.page_count, 2):
    pix = [doc[j].get_pixmap(dpi=70) for j in range(i, min(i + 2, doc.page_count))]
    w = sum(p.width for p in pix)
    h = max(p.height for p in pix)
    out = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, w, h), False)
    out.clear_with(255)
    x = 0
    for p in pix:
        p2 = fitz.Pixmap(fitz.csRGB, p) if p.n != 3 else p
        p2.set_origin(x, 0)
        out.copy(p2, p2.irect)
        x += p.width
    out.save(f"{sys.argv[2]}_{i // 2:02d}.png")
