"""Deterministic checks on the agent's work (no model calls, $0). Driven by the agent's own replacement map, so they work
on customer documents with no labels:
  leak         every mapped original (and each distinctive token of a mapped person name) re-read by OCR on the output
  ghost        the original value re-read from an enhanced crop of each edited area, plus glyph-persistence correlation
  facts        each edit may only touch text that is in the map; pixels outside the recorded edit masks never change
  consistency  casepatch.docs.consistency on a fact graph built from the map (dates/ages/eras/ZIPs/identity ...)
  hidden       output file: PDF text layer / metadata / XMP, image EXIF/XMP/comments
Returns a list of issue dicts; [] means pass."""
from __future__ import annotations

import difflib, re
from pathlib import Path

import numpy as np

from .tools import _bbox_of_chars, _norm

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "monday", "tuesday", "wednesday",
          "thursday", "friday", "saturday", "sunday"}
COMMON = {"will", "mark", "rose", "grace", "hope", "faith", "joy", "may", "june", "april", "august", "bill", "frank", "art", "case",
          "state", "court", "united", "states", "district", "county", "city", "street", "avenue", "north", "south", "east", "west",
          "new", "york", "office", "clerk", "deputy", "judge", "signature", "officer", "inspector", "commissioner", "immigration"}
STOP = {"the", "and", "of", "for", "mr", "mrs", "miss", "dr", "jr", "sr", "von", "van", "de", "la", "del", "der", "le", "st"}


def _fuzzy(value, text, thr=0.86):
    v, t = _norm(value), _norm(text)
    if not v:
        return False
    if len(v) < 6 or re.fullmatch(r"[\d\s./-]+", v):
        return re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", t) is not None
    if v in t:
        return True
    n = len(v)
    for i in range(0, max(1, len(t) - n + 1)):
        w = t[i:i + n]
        if (w[:1] == v[:1] or w[-1:] == v[-1:]) and difflib.SequenceMatcher(None, v, w).ratio() >= thr:
            return True
    return False


def _tokens(entry):
    """Distinctive tokens of a person-like original that the replacement does not reuse (surname-only mentions)."""
    if re.search(r"\d", entry["original"]) or entry.get("kind", "PERSON") not in ("PERSON", "NAME", "", None) or entry["original"].strip()[:1] in "([{<":
        return []
    rep = {t.lower() for t in re.findall(r"[^\W\d_]+", entry["replacement"])}
    return [t for t in re.findall(r"[^\W\d_]+", entry["original"]) if len(t) >= 4 and t[:1].isupper() and t.lower() not in STOP and t.lower() not in rep]


def ocr_text_lines(img):
    from ..docs.extract import ocr_lines
    return ocr_lines(img, "v:")


def check_doc(st, entries, out_path=None, lines_out=None):
    """st: tools.DocState after editing. entries: the whole case map (list of dicts)."""
    import cv2
    issues = []
    mine = [e for e in entries if Path(e["file"]).name == st.name]
    for p, img in enumerate(st.cur):
        out_lines = lines_out[p] if lines_out else ocr_text_lines(img)
        # ---- leak (full-page OCR of the output)
        for e in [e for e in entries if e["original"].strip()[:1] not in "([{<"]:
            date_word = _norm(e["original"]) in MONTHS
            for l in out_lines:
                if date_word and re.search(r"(?i)\b" + re.escape(e["original"]) + r"\.?\s*\d", l.text):
                    continue  # 'August 5, 1951' is a date, not the middle name 'August'
                if _fuzzy(e["original"], l.text) and not _fuzzy(e["replacement"], l.text, 0.95) or (
                        _norm(e["original"]) in _norm(l.text) and _norm(e["original"]) not in _norm(e["replacement"])):
                    issues.append({"check": "leak", "page": p, "value": e["original"], "line": l.text,
                                   "bbox": [round(v) for v in _bbox_of_chars(l.chars)] if l.chars else None})
                    break
            for t in _tokens(e):
                if t.lower() in MONTHS or t.lower() in COMMON:
                    continue
                for l in out_lines:
                    if re.search(r"(?<![^\W\d_])" + re.escape(t.lower()) + r"(?![^\W\d_])", l.text.lower()):
                        issues.append({"check": "leak", "page": p, "value": t, "of": e["original"], "line": l.text,
                                       "bbox": [round(v) for v in _bbox_of_chars(l.chars)] if l.chars else None})
                        break
        # ---- ghost ink at each edited area
        orig = st.orig[p]
        for ed in [x for x in st.edits if x.page == p and not x.undone and x.mask is not None and x.original and not x.original.startswith("(")]:
            x0, y0, x1, y1 = ed.bbox
            pad = max(4, (y1 - y0) // 3)
            X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
            ca, cb = orig[Y0:Y1, X0:X1], img[Y0:Y1, X0:X1]
            if cb.size == 0:
                continue
            ga, gb = cv2.cvtColor(ca, cv2.COLOR_RGB2GRAY), cv2.cvtColor(cb, cv2.COLOR_RGB2GRAY)
            k = 31 if min(gb.shape) > 31 else 3
            dark = np.clip(cv2.medianBlur(gb, k).astype(np.int16) - gb, 0, 255)
            dark0 = np.clip(cv2.medianBlur(ga, k).astype(np.int16) - ga, 1, 255)
            ink = cv2.adaptiveThreshold(ga, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10) > 0
            keep = ink & (dark >= 0.4 * dark0) & (dark >= 12)
            res = np.full_like(gb, 255); res[keep] = 0
            clahe = cv2.createCLAHE(3.0, (8, 8)).apply(gb)
            hit = None
            for name, im in (("residual", res), ("enhanced", clahe)):
                big = cv2.resize(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                t = " ".join(x.text for x in ocr_text_lines(np.ascontiguousarray(big)))
                if _fuzzy(ed.original, t, 0.85) and not _fuzzy(ed.replacement, t, 0.95):
                    hit = name; break
            if hit is None and sum(c.isalpha() for c in ed.original) >= 4 and ga.std() > 5 and gb.std() > 5:
                corr = float(np.corrcoef(ga.ravel().astype(np.float32), gb.ravel().astype(np.float32))[0, 1])
                persist = float(keep.sum()) / max(1.0, float(ink.sum()))
                if corr >= 0.75 and persist >= 0.6:
                    hit = f"persistence corr={corr:.2f}"
            if hit:
                issues.append({"check": "ghost", "page": p, "value": ed.original, "edit_id": ed.id, "bbox": ed.bbox, "how": hit})
        # ---- facts: edits only touch mapped text; nothing changes outside edit masks
        active = [x for x in st.edits if x.page == p and not x.undone and x.mask is not None]
        union = np.zeros(img.shape[:2], bool)
        for ed in active:
            union |= ed.mask
        if st.kind == "image" and ((img != orig).any(axis=2) & ~union).any():
            issues.append({"check": "facts", "page": p, "detail": "pixels changed outside recorded edits"})
        for ed in active:
            x0, y0, x1, y1 = ed.bbox
            touched = []
            for l in [l for l in st.lines(p) if l.conf >= 0.8]:  # garbled reads of handwriting are not facts
                chs = [c for c in l.chars if c.c.strip()]
                inside = "".join(c.c if (x0 <= (c.quad[0][0] + c.quad[2][0]) / 2 <= x1 and y0 <= (c.quad[0][1] + c.quad[2][1]) / 2 <= y1) else "\x00"
                                 for c in l.chars)
                if any(ch != "\x00" for ch in inside):
                    touched += [w for w in inside.split("\x00") if w.strip()]
            text = " ".join(touched)
            rest = text
            for e in mine + entries:
                o = e["original"]
                rest = re.sub(re.escape(o), " ", rest, flags=re.I)
                for t in re.findall(r"[^\W_]+", o):
                    rest = re.sub(r"(?i)(?<![^\W_])" + re.escape(t) + r"(?![^\W_])", " ", rest)
            keep_words = {w.lower() for w in re.findall(r"[^\W_]+", ed.replacement or "")}  # retyped verbatim by the same edit
            left = [w for w in re.findall(r"[^\W_]{3,}", rest) if w.lower() not in keep_words]
            left = [w for w in left if len(w) >= 4 and sum(c.isalpha() for c in w) >= 3]
            if left:  # still there in the output at that place? then nothing non-personal was lost
                pad = max(6, (y1 - y0) // 2)
                crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                after = " ".join(x.text for x in ocr_text_lines(np.ascontiguousarray(crop))) if crop.size else ""
                left = [w for w in left if not _fuzzy(w, after, 0.8)]
            if left and sum(len(w) for w in left) >= 4:
                issues.append({"check": "facts", "page": p, "edit_id": ed.id, "bbox": ed.bbox,
                               "detail": f"edit {ed.id} also covers unmapped text: {' '.join(left)[:80]!r} (a non-personal fact may have changed; "
                                         f"if it is personal, add it to the map, else undo/redo the edit tighter)"})
    # ---- edits that still show a fake the map has since replaced (fix_map here or in another document)
    for e in entries:
        for old in e.get("superseded") or []:
            for ed in st.edits:
                if not ed.undone and ed.replacement and _norm_s(old) in _norm_s(ed.replacement) and _norm_s(e["replacement"]) not in _norm_s(ed.replacement):
                    issues.append({"check": "map_changed", "page": ed.page, "edit_id": ed.id, "bbox": ed.bbox,
                                   "detail": f"edit {ed.id} shows '{old}', but '{e['original']}' now maps to '{e['replacement']}'; undo and redo it"})
    # ---- signatures / faces left unchanged
    issues += region_issues(st)
    # ---- synthetic faces must fit the photo's apparent sex / age band / ethnicity (the agent's estimate of the source)
    for ed in st.edits:
        if ed.kind != "face" or ed.undone or not ed.meta:
            continue
        m = ed.meta.get("match") or {}
        bad = [k for k, v in m.items() if v is False]
        if not ed.meta.get("apparent"):
            issues.append({"check": "face_match", "page": ed.page, "edit_id": ed.id, "bbox": ed.bbox,
                           "detail": "face swapped without an estimate of the original's apparent sex/age/ethnicity; redo replace_face with apparent"})
        elif bad:
            issues.append({"check": "face_match", "page": ed.page, "edit_id": ed.id, "bbox": ed.bbox,
                           "detail": f"synthetic face differs from the photo in {', '.join(bad)} ({ed.meta.get('tile_attrs')} vs {ed.meta.get('apparent')}); "
                                     f"the bank has no closer face - a reviewer should check it"})
    # ---- consistency on the map
    issues += consistency_issues(st, entries)
    # ---- hidden layers / metadata of the written file
    if out_path:
        issues += hidden_issues(out_path, entries)
    return issues


def _norm_s(t):
    return " ".join(str(t).lower().split())


def consistency_issues(st, entries):
    from ..docs import consistency as C
    from ..docs.resolve import Occ
    lines, occs = [], []
    for p in range(len(st.orig)):
        for l in st.lines(p):
            lines.append(C.L(f"{st.name}:{l.id}", st.name, p, l.text, tuple(_bbox_of_chars(l.chars)) if l.chars else None))
    for e in [e for e in entries if Path(e["file"]).name == st.name]:
        line = next((l for l in lines if e["original"].lower() in l.text.lower()), None)
        kind = (e.get("kind") or "").upper() or None
        if kind in ("NAME", "PERSON_NAME"):
            kind = "PERSON"
        if kind is None:
            kind = "PERSON" if not re.search(r"\d", e["original"]) else "ID"
        kind = C.refine_kind(kind, e["original"], line.text if line else "")
        o = Occ(f"{st.name}#{len(occs)}", st.name, line.page if line else 0, line.id if line else f"{st.name}:?", 0,
                len(e["original"]), kind, e["original"], "agent")
        o.entity = f"{kind.lower()}:{e['original'].lower()}"
        occs.append(o)
    if not occs:
        return []
    try:
        g = C.build_graph(lines, occs)
        repl = {o.id: e["replacement"] for o, e in zip(occs, [e for e in entries if Path(e["file"]).name == st.name])}
        vs = C.check(g, repl)
    except Exception as ex:  # a checker crash must not hide the result
        return [{"check": "consistency", "detail": f"checker error {type(ex).__name__}: {str(ex)[:120]}"}]
    out = []
    for v in vs:
        vals = [g.occ[i].text + " -> " + str(repl.get(i)) for i in v["occs"] if i in g.occ]
        out.append({"check": "consistency", "code": v["code"], "detail": f"{C.CODE_TEXT.get(v['code'], v['code'])}: {'; '.join(vals)[:200]}"})
    return out


def hidden_issues(path, entries):
    path = Path(path); out = []
    if path.suffix.lower() == ".pdf":
        import fitz
        d = fitz.open(path)
        text = " ".join(pg.get_text("text") for pg in d)
        for e in entries:
            if _fuzzy(e["original"], text, 0.95):
                out.append({"check": "hidden", "value": e["original"], "detail": "original still in the PDF text layer"})
        meta = {k: v for k, v in (d.metadata or {}).items() if v and k not in ("format", "encryption")}
        if meta or (d.get_xml_metadata() or "").strip():
            out.append({"check": "hidden", "detail": f"PDF metadata not empty: {sorted(meta)}"})
    else:
        from PIL import Image
        im = Image.open(path)
        keys = [k for k in im.info if k in ("exif", "xmp", "comment", "XML:com.adobe.xmp", "icc_profile_text")] + (["exif"] if im.getexif() else [])
        if keys:
            out.append({"check": "hidden", "detail": f"image metadata present: {sorted(set(keys))}"})
    return out


def page_regions(st, p):
    cache = st.__dict__.setdefault("_regions", {})
    if p not in cache:
        from ..docs import visual
        from ..docs.model import Page, quad_bbox
        try:
            pg = Page(p, st.orig[p].shape[1], st.orig[p].shape[0], True, 1.0, st.orig[p], st.lines(p))
            cache[p] = [(r.kind, [int(v) for v in quad_bbox(r.quad)]) for r in visual.find_regions(pg, st.name) if r.kind in ("signature", "face")]
        except Exception:
            cache[p] = []
    return cache[p]


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def _photo_like(img, bb):
    """A real portrait: big enough and a face detector finds a face covering a fair part of the region."""
    x0, y0, x1, y1 = bb
    if min(x1 - x0, y1 - y0) < 60:
        return False
    try:
        from ..docs import native as N
        fs = N.detect_faces(np.ascontiguousarray(img[y0:y1, x0:x1]), thr=0.8)
    except Exception:
        return True   # cannot tell: keep the check
    return any(f is not None and len(f) >= 4 and f[3] >= 0.25 * (y1 - y0) for f in (fs or []))


def compact_review(items):
    """The case-level review list a person reads: one item per problem. Leaks are never dropped, only merged
    (a full-name leak absorbs the leaks of its own tokens in the same file); 'unmapped text' notes are merged per file;
    'two values got the same fake' is dropped when the two values are OCR variants of one name."""
    from ..docs.consistency import CODE_TEXT  # noqa: F401  (keeps the import surface stable)
    out, seen = [], set()
    leaks = {}
    for i in items:
        if i.get("check") == "leak":
            leaks.setdefault(i.get("file"), []).append(i)
    for f, ls in leaks.items():
        vals = sorted({str(i.get("of") or i.get("value")) for i in ls}, key=len, reverse=True)
        for v in vals:
            if any(v.lower() != w.lower() and v.lower() in w.lower() for w in vals):
                continue  # covered by a longer leaked value
            toks = sorted({str(i.get("value")) for i in ls if str(i.get("of") or i.get("value")) == v or str(i.get("value")).lower() in v.lower()})
            first = next(i for i in ls if str(i.get("of") or i.get("value")) == v)
            out.append(dict(first, value=v, detail=f"still readable: '{v}'" + (f" (parts: {', '.join(t for t in toks if t != v)})" if any(t != v for t in toks) else "")))
    facts = {}
    for i in items:
        c = i.get("check")
        if c == "leak":
            continue
        if c == "facts" and "unmapped text" in str(i.get("detail")):
            w = re.search(r"unmapped text: '([^']*)'", str(i.get("detail")))
            facts.setdefault(i.get("file"), []).append((i.get("edit_id"), w.group(1) if w else ""))
            continue
        if c == "consistency" and "same replacement" in str(i.get("detail")):
            vals = re.findall(r"([^:;]+?) -> ", str(i.get("detail")).split(":", 1)[-1])
            vals = [re.sub(r"\(.*?\)", "", v).strip().lower() for v in vals]
            if len(vals) >= 2 and all(_fuzzy(vals[0], v, 0.75) or _fuzzy(v, vals[0], 0.75) for v in vals[1:]):
                continue
        key = (i.get("file"), c, str(i.get("value") or "").lower(), str(i.get("detail"))[:60] if c not in ("signature", "face") else str(i.get("bbox")))
        if key in seen:
            continue
        seen.add(key); out.append(i)
    for f, ws in facts.items():
        out.append({"check": "facts", "file": f, "detail": f"{len(ws)} edit(s) also covered unmapped text; check that nothing non-personal changed: "
                    + "; ".join(f"edit {e}: {w!r}" for e, w in ws)[:300], "edit_ids": [e for e, _ in ws]})
    return out


def region_issues(st):
    """Detected signatures/faces that no edit (or face flag) touched. The agent decides: replace it, or say why not."""
    out = []
    for p in range(len(st.orig)):
        regs = page_regions(st, p)
        if not regs:
            continue
        H, W = st.orig[p].shape[:2]
        for kind, bb in regs:
            x0, y0, x1, y1 = max(0, bb[0]), max(0, bb[1]), min(W, bb[2]), min(H, bb[3])
            if x1 <= x0 or y1 <= y0:
                continue
            changed = float((st.cur[p][y0:y1, x0:x1] != st.orig[p][y0:y1, x0:x1]).any(axis=2).mean())
            flagged = any(f.get("page") == p and _overlap(bb, f["bbox"]) > 0.3 for f in st.flags)
            if kind == "face" and not _photo_like(st.orig[p], [x0, y0, x1, y1]):
                continue  # seals, stamps, ornaments: not a person's photo
            if changed < 0.03 and not flagged:
                out.append({"check": kind, "page": p, "bbox": bb,
                            "detail": f"{kind} region at {bb} is unchanged; if it is a person's {'photo' if kind == 'face' else 'signature'}, replace it "
                                      f"({'replace_face' if kind == 'face' else 'replace_signature, with rotation if sideways'}); otherwise say why in finish"})
    return out
