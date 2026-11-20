"""Extraction: native PDF text with char boxes (PyMuPDF), OCR with word quads (RapidOCR, CPU),
reconciliation of native text against rendered-page OCR, hidden-content inventory, JSON leaves."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .model import Char, Document, Line, Page, quad_bbox, quad_from_rect, rect_inter, rect_overlap

RENDER_DPI = 200
# caption / fragile-line re-OCR is 2x this (400 DPI on a 200 DPI render, or 2x on a scan)
CAPTION_REFINE_FACTOR = 2
PDF_EXT, IMG_EXT = {".pdf"}, {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_OCR = None


def ocr_engine():
    global _OCR
    if _OCR is None:
        import logging
        from rapidocr import RapidOCR
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        _OCR = RapidOCR(params={"Global.log_level": "warning"}) if _accepts_params() else RapidOCR()
    return _OCR


def _accepts_params():
    try:
        from rapidocr import RapidOCR  # noqa
        import inspect
        return "params" in inspect.signature(RapidOCR.__init__).parameters
    except Exception:
        return False


def _lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def _word_chars(word, quad):
    """Interpolate per-character quads along a word quad (handles slant/perspective)."""
    (tl, tr, br, bl), n = quad, max(1, len(word))
    out = []
    for i, ch in enumerate(word):
        a, b = i / n, (i + 1) / n
        out.append(Char(ch, (_lerp(tl, tr, a), _lerp(tl, tr, b), _lerp(bl, br, b), _lerp(bl, br, a))))
    return out


def _lines_from(res, prefix, scale, source):
    lines = []
    if res is None or res.boxes is None:
        return lines
    words_per_line = res.word_results or [()] * len(res.txts)
    for i, (box, txt, score) in enumerate(zip(res.boxes, res.txts, res.scores)):
        words = [w for w in (words_per_line[i] if i < len(words_per_line) else ()) if w and w[0].strip()]
        if not words:
            words = [(txt, score, box.tolist())]
        chars, text = [], ""
        for j, (w, _s, q) in enumerate(words):
            q = tuple((float(x) / scale, float(y) / scale) for x, y in q)
            if j:
                prev = chars[-1].quad
                chars.append(Char(" ", (prev[1], q[0], q[3], prev[2]))); text += " "
            chars.extend(_word_chars(w, q)); text += w
        lines.append(Line(f"{prefix}L{i}", text, chars, source, float(score)))
    return lines


# ---- non-Latin fallback. The default recogniser (PP-OCRv6 multi) drops most Cyrillic and reads Greek as a Latin
# look-alike mix. When a page shows either sign, re-read it with the PP-OCRv5 Greek / East-Slavic recognisers and take
# their line wherever it contains real script letters. Homoglyphs are normalised per word to the word's majority script.
_GREEK = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
_CYR = re.compile(r"[\u0400-\u04ff]")
_LAT_AMBIG = {"greek": dict(zip("ABEZHIKMNOPTYXo", "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧο")),
              "cyrillic": dict(zip("ABEKMHOPCTYXaeopcyxn", "АВЕКМНОРСТУХаеорсухп"))}
_SCRIPT_ENGINES = {}


def _script_engine(lang):
    if lang not in _SCRIPT_ENGINES:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
        _SCRIPT_ENGINES[lang] = RapidOCR(params={"Global.log_level": "warning", "Rec.lang_type": LangRec(lang),
                                                 "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE})
    return _SCRIPT_ENGINES[lang]


def _script_of(ch):
    return "greek" if _GREEK.match(ch) else "cyrillic" if _CYR.match(ch) else "latin" if ch.isascii() and ch.isalpha() else None


def normalize_homoglyphs(text, script):
    """Per word: majority script wins; look-alike letters of the other script are mapped to it."""
    fwd = _LAT_AMBIG[script]; back = {v: k for k, v in fwd.items()}
    def fix(word):
        nat = sum(1 for c in word if _script_of(c) == script and c not in back)
        lat = sum(1 for c in word if _script_of(c) == "latin" and c not in fwd)
        if nat > lat:
            return "".join(fwd.get(c, c) for c in word)
        if lat > nat or not nat:
            return "".join(back.get(c, c) for c in word)
        return word
    return re.sub(r"\S+", lambda m: fix(m.group(0)), text)


def _sparse(line):
    """A line box much wider than its text: the recogniser dropped glyphs it has no alphabet for."""
    if not line.chars:
        return False
    b = quad_bbox((line.chars[0].quad[0], line.chars[-1].quad[1], line.chars[-1].quad[2], line.chars[0].quad[3]))
    h = max(1.0, b[3] - b[1]); expect = (b[2] - b[0]) / (0.55 * h)
    return expect >= 5 and sum(c.isalnum() for c in line.text) < 0.4 * expect


def _bb(line):
    return quad_bbox((line.chars[0].quad[0], line.chars[-1].quad[1], line.chars[-1].quad[2], line.chars[0].quad[3]))


def script_fallback(image, lines, prefix, scale=1.0, source="ocr"):
    if not lines or min(image.shape[:2]) < 24:
        return lines
    txt = "".join(l.text for l in lines)
    langs = []
    if _GREEK.search(txt):
        langs.append("el")
    if _CYR.search(txt):
        langs.append("eslav")
    if not langs and sum(_sparse(l) for l in lines) >= 2:
        langs = ["eslav", "el"]
    best = None
    for lang in langs:
        script = "greek" if lang == "el" else "cyrillic"
        try:
            alt = _lines_from(_script_engine(lang)(image, return_word_box=True), prefix + lang[:2], scale, source)
        except Exception:
            continue
        alt = [l for l in alt if l.chars]
        for l in alt:
            new = normalize_homoglyphs(l.text, script)
            if new != l.text and len(new) == len(l.text):
                for ch, c in zip(l.chars, new):
                    ch.c = c
                l.text = new
        native = sum(1 for l in alt for c in l.text if _script_of(c) == script)
        if native >= 3 and (best is None or native > best[0]):
            best = (native, script, alt)
    if best is None:
        return lines
    _n, script, alt = best
    out, used = list(lines), set()
    for a in alt:
        if a.conf < 0.6 or sum(1 for c in a.text if _script_of(c) == script) < 2:
            continue
        ab = _bb(a)
        hits = [i for i, l in enumerate(out) if l.chars and max(rect_overlap(ab, _bb(l)), rect_overlap(_bb(l), ab)) > 0.5]
        for i in sorted(hits, reverse=True):
            used.add(out[i].id); del out[i]
        a.id = f"{prefix}L{len(lines) + len(out) + 100}"
        out.append(a)
    for i, l in enumerate(out):   # stable ids in reading order
        l.id = f"{prefix}L{i}"
    out.sort(key=lambda l: (_bb(l)[1] if l.chars else 0, _bb(l)[0] if l.chars else 0))
    for i, l in enumerate(out):
        l.id = f"{prefix}L{i}"
    return out


def ocr_lines(image, prefix, scale=1.0, source="ocr", scripts=True):
    """OCR an RGB numpy image. Returns Lines in image-pixel coords divided by `scale`."""
    lines = _lines_from(ocr_engine()(image, return_word_box=True), prefix, scale, source)
    return script_fallback(image, lines, prefix, scale, source) if scripts else lines


def _render(page, dpi=RENDER_DPI):
    import pymupdf
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def native_lines(page, prefix):
    """Group texttrace characters into lines; invisible (render mode 3) or transparent text => native_hidden."""
    runs = []
    for span in page.get_texttrace():
        hidden = span["type"] == 3 or span.get("opacity", 1) == 0
        style = {"font": span["font"], "size": span["size"], "color": span["color"], "flags": span["flags"]}
        for code, _g, origin, bbox in span["chars"]:
            runs.append((chr(code), origin, bbox, hidden, style, span.get("spacewidth") or span["size"] * 0.25))
    lines, cur = [], []
    def flush():
        if cur and "".join(c[0] for c in cur).strip():
            chars = [Char(c[0], quad_from_rect(c[2][0], c[2][1], c[2][2], c[2][3])) for c in cur]
            while chars and chars[-1].c == " ":
                chars.pop()
            while chars and chars[0].c == " ":
                chars.pop(0)
            st = dict(cur[0][4]); st["origin"] = cur[0][1]
            st["chars"] = [(c[0], c[1], c[4]["font"], c[4]["size"], c[4]["color"]) for c in cur if c[0] != " " or True]
            lines.append(Line(f"{prefix}N{len(lines)}", "".join(c.c for c in chars), chars,
                              "native_hidden" if cur[0][3] else "native", 1.0, st))
        cur.clear()
    for r in runs:
        if cur:
            p = cur[-1]
            same_base = abs(r[1][1] - p[1][1]) < 0.5 * p[4]["size"] and r[3] == p[3]
            gap = r[2][0] - p[2][2]
            if not same_base or gap > 2.5 * p[5] or gap < -p[4]["size"]:
                flush()
            elif gap > 0.6 * p[5] and p[0] != " " and r[0] != " ":
                cur.append((" ", (p[2][2], p[1][1]), (p[2][2], p[2][1], r[2][0], p[2][3]), r[3], p[4], p[5]))
        cur.append(r)
    flush()
    return lines


def reconcile(page_obj, lines, ocr, prefix):
    """Compare native text with OCR of the rendered page.
    - OCR text with no native text under it => text exists only in pixels (embedded image) -> add as ocr_pixel_only.
    - Visible native text that OCR cannot see => occluded/white/clipped -> mark native_hidden."""
    notes = []
    visible = [l for l in lines if l.source == "native"]
    vis_boxes = [quad_bbox(c.quad) for l in visible for c in l.chars if c.c.strip()]
    for ol in ocr:
        ob = quad_bbox((ol.chars[0].quad[0], ol.chars[-1].quad[1], ol.chars[-1].quad[2], ol.chars[0].quad[3]))
        covered = sum(rect_inter(ob, vb) for vb in vis_boxes)
        area = max(1e-6, (ob[2] - ob[0]) * (ob[3] - ob[1]))
        if covered / area < 0.15 and sum(ch.isalnum() for ch in ol.text) >= 3:
            ol.source = "ocr_pixel_only"; ol.id = prefix + "P" + ol.id.split("L")[-1]
            lines.append(ol); notes.append({"code": "PIXEL_ONLY_TEXT", "line": ol.id})
    ocr_boxes = [quad_bbox((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3])) for l in ocr]
    for l in visible:
        lb = quad_bbox((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]))
        if sum(ch.isalnum() for ch in l.text) >= 3 and not any(rect_overlap(lb, ob) > 0.2 for ob in ocr_boxes):
            l.source = "native_hidden"; notes.append({"code": "NATIVE_TEXT_NOT_VISIBLE", "line": l.id})
    for l in lines:
        if l.source == "native_hidden" and not any(n.get("line") == l.id for n in notes):
            notes.append({"code": "INVISIBLE_TEXT_LAYER", "line": l.id})
    return notes


def extract_pdf(path, doc_id):
    import pymupdf
    src = pymupdf.open(path)
    d = Document(doc_id, str(path), "pdf", ".pdf")
    meta = {k: v for k, v in (src.metadata or {}).items() if v}
    d.hidden = {"metadata": meta, "xml_metadata": src.get_xml_metadata() or "",
                "embedded_files": {n: src.embfile_get(n).decode("utf-8", "replace")[:5000] for n in src.embfile_names()},
                "annotations": [], "javascript": bool(src.get_page_text if False else _has_js(src))}
    for pno, page in enumerate(src):
        t0 = time.perf_counter()
        prefix = f"{doc_id}:p{pno}:"
        for a in page.annots():
            d.hidden["annotations"].append({k: v for k, v in a.info.items() if v and k in ("content", "title", "subject")})
        lines = native_lines(page, prefix)
        visible_chars = sum(len(l.text) for l in lines if l.source == "native")
        imgs = page.get_image_info()
        big_img = any((i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]) > 0.5 * page.rect.width * page.rect.height for i in imgs)
        t1 = time.perf_counter()
        img = _render(page)
        scale = RENDER_DPI / 72
        if visible_chars < 20 and big_img:   # scanned page: pixels are the document
            ocr = ocr_lines(img, prefix)
            p = Page(pno, img.shape[1], img.shape[0], True, scale, img, ocr)
            p.notes.append({"code": "SCANNED_PAGE", "native_text_chars": visible_chars})
            for l in lines:  # any (OCR) text layer on a scan is discarded in output; detect anyway
                l.source = "native_hidden"; p.lines.append(l)
        else:
            ocr = ocr_lines(img, prefix, scale)
            p = Page(pno, page.rect.width, page.rect.height, False, scale, img, lines)
            p.notes.extend(reconcile(page, p.lines, ocr, prefix))
            p.notes.append({"code": "EMBEDDED_IMAGES", "count": len(imgs)})
            p.embedded_images = [tuple(i["bbox"]) for i in imgs]
        d.timings[f"p{pno}"] = {"native_text_s": round(t1 - t0, 3), "render_ocr_reconcile_s": round(time.perf_counter() - t1, 3)}
        d.pages.append(p)
    t_ref = time.perf_counter()
    for p in d.pages:
        refine_legal_ocr(p, f"{doc_id}:p{p.index}:")
    d.timings["caption_ocr_refine_s"] = round(time.perf_counter() - t_ref, 3)
    return d


def _has_js(doc):
    try:
        return any("/JS" in (doc.xref_object(x) or "") for x in range(1, doc.xref_length()))
    except Exception:
        return False


def extract_image(path, doc_id):
    from PIL import Image, ImageOps
    im = Image.open(path)
    exif = {str(k): str(v)[:200] for k, v in im.getexif().items()}
    gps = im.getexif().get_ifd(0x8825)
    if gps:
        exif["GPS"] = str(dict(gps))[:300]
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.asarray(im).copy()
    d = Document(doc_id, str(path), "image", Path(path).suffix.lower())
    d.hidden = {"exif": exif}
    t0 = time.perf_counter()
    p = Page(0, arr.shape[1], arr.shape[0], True, 1.0, arr, ocr_lines(arr, f"{doc_id}:p0:"))
    d.timings["p0"] = {"ocr_s": round(time.perf_counter() - t0, 3)}
    d.pages.append(p)
    t_ref = time.perf_counter()
    refine_legal_ocr(p, f"{doc_id}:p0:")
    d.timings["caption_ocr_refine_s"] = round(time.perf_counter() - t_ref, 3)
    return d


def extract_json(path, doc_id):
    """A textual document (.json / .jsonl / .ndjson / .txt): validated and bounded by casepatch.docs.textual."""
    from ..core import walk
    from .textual import OUT_EXT, fmt_of, parse
    fmt = fmt_of(Path(path).suffix) or "json"
    parsed = parse(Path(path).read_bytes(), fmt)
    data = parsed["data"]
    d = Document(doc_id, str(path), "json", OUT_EXT[fmt], json=data,
                 textual={k: v for k, v in parsed.items() if k not in ("data",)})
    p = Page(0, 0, 0, False)
    if fmt == "text":
        for i, text in enumerate(data):
            if text.strip():
                p.lines.append(Line(f"{doc_id}:p0:T{i}", text, [], "text", 1.0, {}, (i,)))
    else:
        for i, (jpath, text) in enumerate(walk(data)):
            p.lines.append(Line(f"{doc_id}:p0:J{i}", text, [], "json", 1.0, {}, jpath))
    d.pages.append(p)
    return d


def case_files(input_dir):
    from .textual import TEXT_EXT
    return sorted(f for f in Path(input_dir).iterdir() if f.is_file() and f.suffix.lower() in PDF_EXT | IMG_EXT | set(TEXT_EXT)
                  and f.name != "labels.json" and not f.name.startswith("."))


def extract_dir(input_dir):
    docs = []
    for i, f in enumerate(case_files(input_dir), 1):
        did = f"document-{i:02d}"
        ext = f.suffix.lower()
        docs.append(extract_pdf(f, did) if ext in PDF_EXT else extract_image(f, did) if ext in IMG_EXT else extract_json(f, did))
    return docs


# ---- caption / fragile-line re-OCR (higher DPI + optional PP-OCRv5 Latin fallback)
_LATIN_V5 = None


def _page_img_scale(page):
    """Map page/line coordinates to page.image pixels. Scans and photos already live in image space."""
    if page.image is None:
        return 1.0
    if page.raster or abs(page.width - page.image.shape[1]) < 1.0:
        return 1.0
    return page.scale or 1.0


def _upsample(img, factor=CAPTION_REFINE_FACTOR):
    if factor <= 1 or img is None or img.size == 0:
        return img
    from PIL import Image
    im = Image.fromarray(img)
    w, h = im.size
    return np.asarray(im.resize((max(1, w * factor), max(1, h * factor)), Image.Resampling.LANCZOS))


def _crop_image(image, bbox, pad=10):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    return image[y0:y1, x0:x1], (x0, y0)


def is_fragile_ocr(line):
    """A line whose OCR likely split a hyphen, apostrophe, or short ALL-CAPS word."""
    t = line.text or ""
    if re.search(r"\b[A-Z][A-Za-z]*\s+[-'’]\s+[A-Z]", t):
        return True
    if re.search(r"\b[A-Z]['’]\s+[A-Z]", t):
        return True
    short = re.findall(r"\b[A-Z]{1,3}\b", t)
    if len(short) >= 2 and sum(c.isalpha() for c in t) <= 16:
        return True
    if line.conf < 0.7:
        return True
    if line.chars and _sparse(line):
        return True
    # glued initials ('STEVEN W.MYHRE', 'ANGELAD.CAESAR') and dotted / underscored form-fills
    if re.search(r"[A-Z]\.[A-Z][A-Za-z]{2,}", t):
        return True
    if re.search(r"(?:\.{2,}|_)", t) and re.search(r"[A-Za-z]{3,}", t):
        return True
    return False


def ocr_quality(text):
    """Higher is a more complete reading: more letters, joined hyphen/apostrophe, fewer 1-2 letter shards."""
    t = text or ""
    alnum = sum(c.isalnum() for c in t)
    punct = sum(c in "-'’" for c in t)
    spaced_punct = len(re.findall(r"\s+[-'’]\s+|\s+[-'’]|[-'’]\s+", t))
    short_caps = len(re.findall(r"\b[A-Z]{1,2}\b", t))
    return (alnum + 3 * (punct - spaced_punct), -short_caps, -spaced_punct, alnum)


def better_ocr_text(old, new):
    if not (new or "").strip():
        return False
    # never accept a much shorter reading
    old_n = sum(c.isalnum() for c in (old or ""))
    new_n = sum(c.isalnum() for c in new)
    if old_n and new_n < 0.7 * old_n:
        return False
    return ocr_quality(new) > ocr_quality(old)


def _latin_v5_engine():
    """PP-OCRv5 Latin/English recogniser, used only as a caption fallback. Offline; missing weights => None."""
    global _LATIN_V5
    if _LATIN_V5 is False:
        return None
    if _LATIN_V5 is None:
        try:
            from pathlib import Path as _P
            import rapidocr as _ro
            root = _P(_ro.__file__).parent / "models"
            if not any((root / n).exists() for n in ("latin_PP-OCRv5_rec_mobile.onnx", "en_PP-OCRv5_rec_mobile.onnx")):
                _LATIN_V5 = False
                return None
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
            _LATIN_V5 = RapidOCR(params={"Global.log_level": "warning", "Rec.lang_type": LangRec.LATIN,
                                         "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE})
        except Exception:
            _LATIN_V5 = False
            return None
    return _LATIN_V5


def _shift_lines(lines, dx, dy):
    for l in lines:
        for ch in l.chars:
            ch.quad = tuple((x + dx, y + dy) for x, y in ch.quad)
    return lines


def _ocr_crop(image, bbox_img, prefix, page_scale, factor=CAPTION_REFINE_FACTOR, pad=12):
    """OCR a crop at higher resolution. Returns a list of line-lists (default, optional PP-OCRv5)."""
    crop, origin = _crop_image(image, bbox_img, pad=pad)
    if crop is None:
        return []
    up = _upsample(crop, factor)
    scale = factor * page_scale
    variants = [_lines_from(ocr_engine()(up, return_word_box=True), prefix, scale, "ocr")]
    # script fallback on the same crop (Greek/Cyrillic pages)
    variants[0] = script_fallback(up, variants[0], prefix, scale, "ocr")
    eng = _latin_v5_engine()
    if eng is not None:
        try:
            alt = _lines_from(eng(up, return_word_box=True), prefix + "v5", scale, "ocr")
            if alt:
                variants.append(alt)
        except Exception:
            pass
    dx, dy = origin[0] / page_scale, origin[1] / page_scale
    return [_shift_lines(v, dx, dy) for v in variants]


def _cluster_line_groups(lines, gap_factor=1.5):
    """Group lines into vertical bands (nearby caption / name-list rows)."""
    if not lines:
        return []
    items = sorted(lines, key=lambda l: _bb(l)[1] if l.chars else 0)
    heights = [max(1.0, _bb(l)[3] - _bb(l)[1]) for l in items if l.chars]
    gap = (sorted(heights)[len(heights) // 2] if heights else 12.0) * gap_factor
    groups, cur, prev_y1 = [], [], None
    for l in items:
        if not l.chars:
            continue
        y0, y1 = _bb(l)[1], _bb(l)[3]
        if cur and prev_y1 is not None and y0 - prev_y1 > gap:
            groups.append(cur)
            cur = []
        cur.append(l)
        prev_y1 = y1
    if cur:
        groups.append(cur)
    return groups


def _union_bbox(lines, pad_y=0, pad_x=0):
    boxes = [_bb(l) for l in lines if l.chars]
    if not boxes:
        return None
    return (min(b[0] for b in boxes) - pad_x, min(b[1] for b in boxes) - pad_y,
            max(b[2] for b in boxes) + pad_x, max(b[3] for b in boxes) + pad_y)



def _caption_name_line(t):
    """A caption line that looks like it contains a person name, not a role or court heading."""
    s = (t or "").strip()
    if not s:
        return False
    from .legal import _is_heading_tokens, _is_name_list_line as _nl, _clean, STOP
    if _nl(s):
        return True
    words = [w for w in re.findall(r"[A-Za-z'’-]+", s) if re.search(r"[A-Za-z]", w)]
    rest = [w for w in words if _clean(w) not in STOP and w.lower() not in {"v", "vs", "versus"}]
    if len(rest) < 2 or _is_heading_tokens(words):
        return False
    return 2 <= len(words) <= 8


def _stacked_name_targets(lines, max_gap_factor=2.0):
    """Short typographic lines that sit next to another short name line (a wrapped caption name)."""
    from .legal import _is_stacked_name_line
    items = sorted((l for l in lines if l.chars), key=lambda l: _bb(l)[1] if l.chars else 0)
    if not items:
        return []
    heights = [max(1.0, _bb(l)[3] - _bb(l)[1]) for l in items]
    gap = (sorted(heights)[len(heights) // 2] if heights else 12.0) * max_gap_factor
    out = []
    for i, l in enumerate(items):
        if not _is_stacked_name_line(l):
            continue
        for j in (i - 1, i + 1):
            if not (0 <= j < len(items) and _is_stacked_name_line(items[j])):
                continue
            y0, y1 = _bb(l)[1], _bb(l)[3]
            oy0, oy1 = _bb(items[j])[1], _bb(items[j])[3]
            if min(abs(oy0 - y1), abs(y0 - oy1)) <= gap:
                out.append(l)
                break
    return out


def _name_ocr_targets(page, ocr):
    """Lines worth a higher-DPI re-read: fragile OCR, court captions, letter salutations/sign-offs."""
    from .legal import is_legal, is_court_filing, _caption_flags, _is_name_list_line
    body = "\n".join(l.text for l in page.lines)
    targets = [l for l in ocr if is_fragile_ocr(l)]
    if is_legal(body):
        flags = _caption_flags(page.lines, court_filing=is_court_filing(body))
        by_id = {l.id: cap for l, cap in zip(page.lines, flags)}
        targets.extend(l for l in ocr if _is_name_list_line(l.text) or (by_id.get(l.id) and _caption_name_line(l.text)))
        stacked = _stacked_name_targets(ocr)
        have = {id(l) for l in targets}
        targets.extend(l for l in stacked if id(l) not in have)
    try:
        from .letters import LETTER_CUE, SALUTATION, HON_NAME, CLOSE
    except Exception:
        LETTER_CUE = SALUTATION = HON_NAME = CLOSE = None
    if LETTER_CUE is not None and LETTER_CUE.search(body):
        have = {id(l) for l in targets}
        for i, l in enumerate(ocr):
            if id(l) in have:
                continue
            if SALUTATION.search(l.text) or HON_NAME.search(l.text):
                targets.append(l); have.add(id(l)); continue
            if CLOSE.match(l.text.strip()):
                for nxt in ocr[i + 1:i + 4]:
                    if id(nxt) not in have:
                        targets.append(nxt); have.add(id(nxt))
    # unique, original order
    seen, out = set(), []
    for l in targets:
        if id(l) not in seen:
            seen.add(id(l)); out.append(l)
    return out


def refine_legal_ocr(page, prefix):
    """Re-OCR fragile, caption, and letter name lines at higher DPI (and PP-OCRv5 if it reads better)."""
    from .legal import _is_name_list_line
    if page.image is None:
        return 0
    ocr = [l for l in page.lines if l.source in ("ocr", "ocr_pixel_only") and l.chars and l.text.strip()]
    if not ocr:
        return 0
    targets = _name_ocr_targets(page, ocr)
    if not targets:
        return 0
    img_scale = _page_img_scale(page)
    replaced = 0
    target_ids = {id(l) for l in targets}
    for group in _cluster_line_groups(targets):
        ub = _union_bbox(group)
        if not ub:
            continue
        # pad by about one line height so a name cut at the box edge is still in the crop
        h = max(8.0, ub[3] - ub[1])
        pad = max(8.0, 0.8 * h / max(1, len(group)))
        # image-space bbox
        bbox_img = (ub[0] * img_scale, ub[1] * img_scale, ub[2] * img_scale, ub[3] * img_scale)
        # extra pixel pad for the crop itself
        variants = _ocr_crop(page.image, bbox_img, prefix + "r", img_scale,
                             factor=CAPTION_REFINE_FACTOR, pad=int(pad * img_scale) + 8)
        if not variants:
            continue
        old_txt = " ".join(l.text for l in group)
        best, best_s = None, None
        for v in variants:
            # keep lines that overlap the original (slightly padded) union, plus a name-like line in the pad
            kept = []
            loose = (ub[0] - pad, ub[1] - pad, ub[2] + pad, ub[3] + pad)
            for n in v:
                if not n.chars:
                    continue
                nb = _bb(n)
                if rect_overlap(nb, loose) > 0 or rect_overlap(loose, nb) > 0:
                    kept.append(n)
                elif _is_name_list_line(n.text) and abs((nb[1] + nb[3]) / 2 - (ub[1] + ub[3]) / 2) < 2 * h:
                    kept.append(n)
            if not kept:
                continue
            txt = " ".join(n.text for n in kept)
            if best is None or ocr_quality(txt) > best_s:
                best, best_s = kept, ocr_quality(txt)
        if best is None:
            continue
        new_txt = " ".join(n.text for n in best)
        if not better_ocr_text(old_txt, new_txt):
            continue
        # never drop a 4+ letter name token the original reading had (dotted 'Grubt' / 'Ramona')
        old_toks = {w.upper() for w in re.findall(r"[A-Za-z'’]{4,}", old_txt)}
        new_toks = {w.upper() for w in re.findall(r"[A-Za-z'’]{4,}", new_txt)}
        if old_toks - new_toks:
            continue
        # ... nor a digit run it had: a masked card '************6467' re-read as 'SA' lost the card suffix, so the
        # suffix was never detected and stayed in the output
        if set(re.findall(r"\d{3,}", old_txt)) - set(re.findall(r"\d{3,}", new_txt)):
            continue
        # drop the original group, insert the new lines
        drop = {id(l) for l in group}
        page.lines = [l for l in page.lines if id(l) not in drop]
        page.lines.extend(best)
        replaced += 1
    if replaced:
        page.lines.sort(key=lambda l: (_bb(l)[1] if l.chars else 0, _bb(l)[0] if l.chars else 0))
        i_ocr = 0
        for l in page.lines:
            if l.source in ("ocr", "ocr_pixel_only"):
                l.id = f"{prefix}L{i_ocr}"
                i_ocr += 1
        page.notes.append({"code": "CAPTION_OCR_REFINE", "groups": replaced})
    return replaced
