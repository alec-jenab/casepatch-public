"""Parser comparison on the measured case (not by brand): PyMuPDF+RapidOCR baseline vs Docling vs LiteParse.

Each parser runs in its own virtualenv/process (heavy, conflicting deps):
    <venv>/bin/python casepatch/bench/parsers.py extract {baseline|docling|liteparse} <case_dir> <out.json>
    python casepatch/bench/parsers.py compare <case_dir> out1.json out2.json ...
Measures per file: seconds, peak RSS, whether word/line boxes are available, recall of labeled sensitive values in
the extracted text, and whether pixel-only text and invisible text were surfaced."""
from __future__ import annotations

import json
import re
import resource
import sys
import time
from pathlib import Path

FILES = (".pdf", ".png", ".jpg", ".jpeg")


def ex_baseline(path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from casepatch.docs.extract import extract_image, extract_pdf
    d = extract_pdf(path, "d") if path.suffix == ".pdf" else extract_image(path, "d")
    lines = [l for p in d.pages for l in p.lines]
    return {"text": "\n".join(l.text for l in lines), "boxes": sum(1 for l in lines if l.chars),
            "box_granularity": "char (native) / word-interpolated char (OCR)",
            "flags": sorted({l.source for l in lines})}


def ex_docling(path):
    from docling.document_converter import DocumentConverter
    global _DC
    try:
        _DC
    except NameError:
        _DC = DocumentConverter()
    r = _DC.convert(str(path))
    doc = r.document
    n_boxes = sum(1 for it, _ in doc.iterate_items() if getattr(it, "prov", None))
    pics = [it for it, _ in doc.iterate_items() if type(it).__name__ == "PictureItem"]
    return {"text": doc.export_to_markdown(), "boxes": n_boxes, "box_granularity": "item (paragraph/table cell)",
            "flags": [f"pictures:{len(pics)}"]}


def ex_liteparse(path):
    from liteparse import LiteParse
    r = LiteParse(output_format="json", quiet=True).parse(str(path))
    items = [ti for p in r.pages for ti in (p.text_items or [])]
    return {"text": r.text if isinstance(r.text, str) else "\n".join(getattr(ti, "text", "") for ti in items),
            "items_text": "\n".join(getattr(ti, "text", "") or getattr(ti, "str", "") for ti in items),
            "boxes": len(items), "box_granularity": "text item (run/word) with bbox", "flags": []}


def extract(tool, case_dir, out):
    fn = {"baseline": ex_baseline, "docling": ex_docling, "liteparse": ex_liteparse}[tool]
    res = {"tool": tool, "files": {}}
    t_all = time.perf_counter()
    for f in sorted(Path(case_dir).iterdir()):
        if f.suffix.lower() not in FILES:
            continue
        t = time.perf_counter()
        try:
            r = fn(f)
            r["seconds"] = round(time.perf_counter() - t, 2)
        except Exception as e:  # report, don't hide
            r = {"error": type(e).__name__ + ": " + str(e)[:200], "seconds": round(time.perf_counter() - t, 2)}
        res["files"][f.name] = r
    res["total_seconds"] = round(time.perf_counter() - t_all, 2)
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


def _n(s):
    return re.sub(r"[\s|*]+", " ", s.lower())


def compare(case_dir, outs):
    labels = json.loads((Path(case_dir) / "labels.json").read_text())
    table = {}
    for o in outs:
        r = json.loads(Path(o).read_text())
        row = {"total_seconds": r["total_seconds"], "peak_rss_mb": r["peak_rss_mb"], "files": {}}
        tot = hit = 0
        for f, info in r["files"].items():
            vals = labels["occurrences"].get(f, [])
            text = _n(info.get("text", "") + "\n" + info.get("items_text", ""))
            found = [v for v in vals if _n(v) in text]
            tot += len(vals); hit += len(found)
            hid = labels.get("hidden_or_non_text", {}).get(f, {})
            row["files"][f] = {"seconds": info.get("seconds"), "boxes": info.get("boxes"), "granularity": info.get("box_granularity"),
                               "recall": f"{len(found)}/{len(vals)}", "missed": [v for v in vals if v not in found],
                               "pixel_only_text_found": all(_n(v) in text for v in hid.get("pixel_only_text", [])) if hid.get("pixel_only_text") else None,
                               "invisible_text_surfaced": ("internal-ref" in text) if hid.get("invisible_text") else None,
                               "error": info.get("error"), "flags": info.get("flags")}
        row["sensitive_value_recall"] = round(hit / max(tot, 1), 3)
        table[r["tool"]] = row
    return table


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(json.dumps(compare(sys.argv[2], sys.argv[3:]), indent=1))
