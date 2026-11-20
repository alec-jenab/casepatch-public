"""Separate verification pass over the OUTPUT files (does not reuse the edit pipeline's state for its verdicts).

Checks: (1) text/object inspection of output files (page text, metadata, XMP, annotations, attachments, JS,
decompressed object streams, EXIF); (2) fresh OCR of rendered output pixels; (3) search for every original
sensitive value (exact + fuzzy for OCR) and a rules-detector rescan; (4) pixel diff outside edited regions;
(5) legibility: read back each replacement; (6) preserved facts: amounts/dates from input still present.
Limits: the OCR engine is the same family as extraction (correlated blind spots); this does NOT certify privacy."""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import numpy as np

from .detect import Chunk, RulesDetector, Unit
from .extract import RENDER_DPI, ocr_lines

MONEY_RX = re.compile(r"[-$]?\d{1,3}(?:,\d{3})*\.\d{2}\b")
DATE_RX = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b")


def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


# Text-layer and bookkeeping methods: the replacement is already in the output PDF
# text (or there is nothing visual to read back). Per-crop RapidOCR on each of these
# dominated verify time on typeset letters (dozens of calls) and pushed a 1-page
# letter past the hosted demo's 180s cap.
TEXT_LAYER_METHODS = frozenset({
    "native_text_rewrite", "native_redact_rewrite", "line_retypeset_reflow",
    "json_string_rewrite", "hidden_text_removed", "hidden_text_line_removed",
    "annotation_removed", "metadata_stripped", "pdf_rebuilt_fresh",
})


def readback_score(replacement, text):
    """Similarity of `replacement` to `text` (1.0 if the normalised value appears as a substring)."""
    r, t = _norm(replacement), _norm(text or "")
    if not r or not t:
        return 0.0
    if r in t:
        return 1.0
    n = len(r)
    return max((difflib.SequenceMatcher(None, r, t[i:i + n]).ratio()
                for i in range(0, max(1, len(t) - n + 1))), default=0.0)


def best_existing_readback(replacement, *texts):
    """Best readback_score against already-available corpora (text layer, full-page OCR, …)."""
    return max((readback_score(replacement, t) for t in texts), default=0.0)


def _fuzzy_in(value, text, threshold=0.88):
    v, t = _norm(value), _norm(text)
    if not v:
        return False
    if v in t:
        return True
    if len(v) < 6:
        return False
    n = len(v)
    for i in range(0, max(1, len(t) - n + 1)):
        w = t[i:i + n]
        if w[:1] != v[:1] and w[-1:] != v[-1:]:
            continue
        if difflib.SequenceMatcher(None, v, w).ratio() >= threshold:
            return True
    return False


def inspect_pdf(path):
    import pymupdf
    d = pymupdf.open(path)
    pages_text = [p.get_text() for p in d]
    objs, dicts = [], []
    for x in range(1, d.xref_length()):
        try:
            o = d.xref_object(x)
            objs.append(o); dicts.append(o)
            if d.xref_is_stream(x):
                objs.append(d.xref_stream(x).decode("latin-1", "replace"))
        except Exception:
            pass
    info = {"metadata": {k: v for k, v in (d.metadata or {}).items() if v and k not in ("format", "encryption")},
            "xmp": bool(d.get_xml_metadata()), "annotations": sum(len(list(p.annots() or [])) for p in d),
            "embedded_files": d.embfile_count(), "javascript": any(re.search(r"/(?:JS|JavaScript)(?![A-Za-z0-9#+_.-])", o) for o in dicts),  # dictionaries only, not stream bytes
            "text_layer_chars": sum(len(t.strip()) for t in pages_text)}
    return d, "\n".join(pages_text), "\n".join(objs), info


def render_pages(path, kind):
    if kind == "image":
        from PIL import Image
        return [np.asarray(Image.open(path).convert("RGB"))]
    import pymupdf
    out = []
    for p in pymupdf.open(path):
        pix = p.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csRGB, alpha=False)
        out.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy())
    return out


def pixel_diff(before, after, rects_px, pad=6):
    import cv2
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    mask = np.zeros(before.shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects_px:
        cv2.rectangle(mask, (int(x0) - pad, int(y0) - pad), (int(x1) + pad, int(y1) + pad), 255, -1)
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2) > 48
    outside = diff & (mask == 0)
    import os
    if os.environ.get("CASEPATCH_DEBUG_DIFF"):
        dbg = np.dstack([diff * 255, mask // 2, outside * 255]).astype(np.uint8)
        cv2.imwrite(os.path.join(os.environ["CASEPATCH_DEBUG_DIFF"], f"diff_{before.shape[1]}x{before.shape[0]}_{int(outside.sum())}.png"), dbg)
    n, _, stats, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8))
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return {"changed_px_outside_regions": int(outside.sum()), "fraction_outside": round(float(outside.sum()) / outside.size, 6),
            "largest_changed_blob_px": largest, "edited_area_fraction": round(float((mask > 0).mean()), 4),
            "verdict": "PASS" if largest < 60 and outside.mean() < 0.0005 else "REVIEW"}


def verify_case(docs, outputs, occs, regions, logs, all_replacements, texts=None, kept=()):
    """outputs: {doc_id: Path}. occs: accepted occurrences (with .text originals). `kept`: values deliberately kept as
    facts (organisation contact lines, places, businesses); the rescan does not report them. Returns (report, local_detail)."""
    kept_n = [k for k in (_norm(x) for x in kept) if len(k) >= 4]
    originals = {}
    for o in occs:
        originals.setdefault(o.doc, set()).add(o.text)
    case_values = {o.text for o in occs}
    for d in docs:  # hidden-layer values that were stripped wholesale must not reappear either
        for r in regions:
            if r.payload:
                case_values.add(r.payload)
    rules = RulesDetector()
    report, detail = {}, {}
    for d in docs:
        out = outputs[d.id]
        rd, dd = {"output_file": out.name}, {"lookalikes": []}
        corpora = {}
        if d.kind == "pdf":
            _, text, objs, info = inspect_pdf(out)
            corpora["text_layer"], corpora["pdf_objects"] = text, objs
            rd["object_inspection"] = info
        elif d.kind == "image":
            from PIL import Image
            ex = Image.open(out).getexif()
            rd["object_inspection"] = {"exif_tags": len(ex), "gps": bool(ex.get_ifd(0x8825))}
        else:
            corpora["json"] = Path(out).read_text()
        ocr_text, rendered = "", []
        if d.kind != "json":
            rendered = render_pages(out, d.kind)
            ocr_text = "\n".join(l.text for img in rendered for l in ocr_lines(img, "v:"))
            corpora["ocr_of_output_pixels"] = ocr_text
        # (3) residual originals
        hits = []
        for v in case_values:
            if len(v) < 4:
                continue
            for name, corpus in corpora.items():
                if name == "pdf_objects" and v.isdigit() and len(v) < 7:
                    continue  # short numbers match font width arrays and xref numbers; text layer and OCR still check it
                found = (_fuzzy_in(v, corpus) if name == "ocr_of_output_pixels" else
                         (re.search(r"(?<!\d)" + re.escape(v) + r"(?!\d)", corpus) is not None if v.isdigit() else _norm(v) in _norm(corpus)))
                if found and name == "ocr_of_output_pixels" and _norm(v) not in _norm(corpus):
                    # fuzzy-only hit: if a replacement explains it at least as well, it is a lookalike, not a leak
                    if any(_fuzzy_in(r, corpus, 0.95) and difflib.SequenceMatcher(None, _norm(v), _norm(r)).ratio() >= 0.7 for r in all_replacements):
                        dd.setdefault("lookalikes", []).append(v); found = False
                if found:
                    hits.append({"layer": name, "value": v})
        # a fuzzy OCR hit that equals/contains a replacement is a lookalike, not a leak
        rd["residual_original_values"] = len(hits)
        dd["residual_hits"] = hits
        # rules rescan on output text: anything PII-shaped that is not one of our replacements
        unexplained = []
        for name, corpus in corpora.items():
            if name == "pdf_objects":
                continue
            units = [Unit(f"v{i}", t, 0, t, "verify") for i, t in enumerate(corpus.splitlines()) if t.strip()]
            line_of = {u.line_id: _norm(u.text) for u in units}
            for p in rules.detect(Chunk(d.id, 0, units)):
                pn, ln = _norm(p.text), line_of.get(p.line_id, "")
                if any(pn and (pn in k or k in pn or k in ln) for k in kept_n):
                    continue  # a kept fact, or on the same line as one (an organisation's contact line)
                if any(p.text in r or r in p.text for r in all_replacements) or re.fullmatch(r"\d{4}", p.text):
                    continue
                # an OCR misreading of one of our own replacements ('hal.ffodd@' for 'hal.fodd@') is not missed PII
                if len(pn) >= 8 and any(abs(len(r) - len(pn)) <= 2 and difflib.SequenceMatcher(None, pn, _norm(r)).ratio() >= 0.9
                                        for r in all_replacements):
                    dd.setdefault("lookalikes", []).append(p.text)
                    continue
                unexplained.append({"layer": name, "kind": p.kind, "value": p.text})
        rd["rescan_pii_like_not_replacements"] = len(unexplained)
        dd["rescan_unexplained"] = unexplained
        # (6) facts
        in_text = "\n".join(l.text for p in d.pages for l in p.lines if l.source != "native_hidden")
        dobs = {o.text for o in occs if o.kind == "DOB"}
        facts = sorted({m.group() for rx in (MONEY_RX, DATE_RX) for m in rx.finditer(in_text)} - dobs)
        all_out = "\n".join(c for k, c in corpora.items() if k != "pdf_objects")
        if texts is not None:  # re-read output text for the post-edit consistency verifier
            texts[d.id] = all_out
        missing = [f for f in facts if f not in all_out and not _fuzzy_in(f, all_out, 0.95)]
        rd["facts_checked"] = len(facts); rd["facts_missing"] = missing  # amounts/dates are non-identifying by policy
        # (4) pixel diff + (5) legibility
        if d.kind != "json":
            dlog = logs.get(d.id, [])
            per_page = []
            for pi, page in enumerate(d.pages):
                rects = []
                for e in dlog:
                    if e.get("page", 0) != pi:
                        continue
                    if "bbox_px" in e:
                        rects.append(e["bbox_px"])
                    elif "bbox_pt" in e:
                        rects.append([v * page.scale for v in e["bbox_pt"]])
                before = page.image
                per_page.append(pixel_diff(before, rendered[pi], rects))
            rd["pixel_diff"] = per_page
            leg = []
            occ_by_id = {o.id: o for o in occs}
            page_ocr = ocr_text
            text_layer = corpora.get("text_layer") or corpora.get("json") or ""
            for e in dlog:
                o = occ_by_id.get(e.get("occurrence"))
                if not o or e.get("method") == "hidden_text_removed" or not ("bbox_px" in e or "bbox_pt" in e):
                    continue
                method = e.get("method")
                existing = best_existing_readback(o.replacement, text_layer, page_ocr)
                # Typeset / text-layer edits: trust the text layer and the one full-page OCR.
                # Raster/signature/inpaint edits still get a crop OCR when the page OCR is weak.
                if method in TEXT_LAYER_METHODS or existing >= 0.8:
                    source = "text_layer_or_page_ocr"
                    score = existing
                else:
                    pi = e.get("page", 0); page = d.pages[pi]
                    bb = e.get("bbox_px") or [v * page.scale for v in e["bbox_pt"]]
                    img = rendered[pi]
                    x0, y0, x1, y1 = [int(v) for v in bb]
                    h = y1 - y0
                    crop = img[max(0, y0 - h // 2):y1 + h // 2, max(0, x0 - 4):x1 + int(0.25 * (x1 - x0)) + 8]
                    read = " ".join(l.text for l in ocr_lines(crop, "c:")) if crop.size else ""
                    score = max(existing, readback_score(o.replacement, read))
                    source = "crop_ocr"
                leg.append({"occurrence": o.id, "method": method, "readback_similarity": round(score, 2),
                            "readback_source": source})
            rd["legibility"] = {"checked": len(leg), "mean_similarity": round(float(np.mean([x["readback_similarity"] for x in leg])), 3) if leg else None,
                                "below_0_8": [x["occurrence"] for x in leg if x["readback_similarity"] < 0.8]}
            dd["legibility"] = leg
        report[d.id], detail[d.id] = rd, dd
    return report, detail
