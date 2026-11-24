"""Separate verification pass over the OUTPUT files (does not reuse the edit pipeline's state for its verdicts).

Checks: (1) text/object inspection of output files (page text, metadata, XMP, annotations, attachments, JS,
decompressed object streams, EXIF); (2) fresh OCR of rendered output pixels; (3) search for every original
sensitive value (exact + fuzzy for OCR), and for the name / street words inside them as standalone words, plus a
rules-detector rescan; (4) pixel diff outside edited regions;
(5) legibility: read back each replacement; (6) preserved facts: amounts/dates from input still present.
Limits: the OCR engine is the same family as extraction (correlated blind spots); this does NOT certify privacy."""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import numpy as np

from .detect import Chunk, RulesDetector, Unit
from .gazetteer import TRADE_WORDS as _TRADE_WORDS
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


# Words inside a replaced name or street line that identify on their own ('Vance' from 'Odalys Brennecke-Vance',
# 'Larkspur' from '4471 Larkspur Court'). The residual check looks for them as standalone words too, so a surname
# part or street name the detectors missed elsewhere is not reported as clean.
_NAME_KINDS = {"PERSON", "GIVEN", "SURNAME"}
_ADDRESS_KINDS = {"ADDRESS", "STREET"}
_STREET_WORDS = {"street", "st", "avenue", "ave", "road", "rd", "drive", "dr", "lane", "ln", "court", "ct", "terrace",
                 "ter", "place", "pl", "boulevard", "blvd", "way", "circle", "cir", "parkway", "pkwy", "highway", "hwy",
                 "trail", "trl", "square", "sq", "loop", "row", "alley", "park", "creek", "ridge", "view", "heights",
                 "north", "south", "east", "west", "apt", "apartment", "suite", "ste", "unit", "floor", "building", "box"}


def identifying_part_sources(values):
    """values: (kind, text) pairs of accepted occurrences -> {standalone identifying word (len >= 4, capitalised):
    {(role, source)}}: given names ("GIVEN", the given name) and surname parts ("SURNAME", the full surname) of names,
    and the street-name words of a street line ("STREET", the street's name as the address fake reads it, or None when
    the line does not parse); before the first comma, numbers, street types, directions, units and the city are left
    out. Everyday-word names are skipped. A spelled-out middle name has role "MIDDLE" (a fake given name of its own);
    other middle words have role None."""
    from .resolve import COMMON_WORD_NAMES, DOC_WORDS, _NAME_PARTICLES, STREET_LINE, middle_name, parse_person, surname_parts, \
        split_role_title, particle_surname
    out = {}
    for kind, text in values:
        text = text or ""
        if kind in _NAME_KINDS:
            core = re.sub(r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof)\.?\s", " ", text)
            core = split_role_title(core.strip())[1]  # 'Captain Velloran': the title is not a name word
            words = re.findall(r"[A-Za-z][A-Za-z']+", core)
            skip = COMMON_WORD_NAMES | DOC_WORDS | _NAME_PARTICLES | {"jr", "sr", "iii"}
            p = parse_person(core) if kind == "PERSON" else None
            given = (p or {}).get("given") if kind == "PERSON" else core.strip() if kind == "GIVEN" else None
            sur = (p or {}).get("surname") if kind == "PERSON" else core.strip() if kind == "SURNAME" else None
            mid = middle_name(core) if kind == "PERSON" else None

            def role(w, given=given, sur=sur, mid=mid):
                if given and w == given:
                    return ("GIVEN", given)
                if mid and w == mid:
                    return ("MIDDLE", mid)
                if sur and len(sur) > 1 and (w.lower() == sur.lower() or w.lower() in surname_parts(sur)):
                    return ("SURNAME", sur)
                ps = particle_surname(sur) if sur else None  # 'Brakel' of 'van der Brakel'
                if ps and (w.lower() == ps[1].lower() or w.lower() in surname_parts(ps[1])):
                    return ("SURNAME", sur)
                return (None, None)
        elif kind in _ADDRESS_KINDS:
            line = text.split(",")[0]
            if not re.match(r"\s*\d", line):
                continue
            words = re.findall(r"[A-Za-z][A-Za-z']+", line)
            skip = _STREET_WORDS | DOC_WORDS
            m = STREET_LINE.match(text.strip())
            name = m.group(2).strip() if m else ""

            def role(w, name=name):
                return ("STREET", name) if w in re.findall(r"[A-Za-z][A-Za-z']+", name) else ("STREET", None)
        else:
            continue
        for w in words:
            if len(w) >= 4 and w[0].isupper() and w.lower() not in skip:
                out.setdefault(w, set()).add(role(w))
    return out


_COURTESY = {"MR", "MRS", "MS", "MISS", "MX", "DR", "PROF"}
_DETERMINER_BEFORE = re.compile(r"(?i)\b(?:the|a|an|some|any|this|that|these|those|each|every|such)\s+$")


def titled_only_words(values):
    """Name words that come only from values written after a rank or office ('Velloran' of 'Captain Velloran', with no
    'Corwin Velloran' in the case). Such a word after a determiner ('the VELLORAN', 'some Velloran members') reads as
    a group's name and is not looked for there."""
    from .resolve import split_role_title
    def rank(t):  # an office or rank ('Captain', 'Officer'), not a courtesy title ('Mr.', 'Dr.')
        title = split_role_title((t or "").strip())[0]
        return bool(title) and title.split()[0].strip(".").upper() not in _COURTESY
    titled = [(k, t) for k, t in values if k == "PERSON" and rank(t)]
    if not titled:
        return set()
    plain = set(identifying_part_sources([(k, t) for k, t in values if (k, t) not in titled]))
    return set(identifying_part_sources(titled)) - plain


def after_determiner(text, start):
    return bool(_DETERMINER_BEFORE.search((text or "")[max(0, start - 12):start]))


def identifying_parts(values):
    """The standalone identifying words of `values` (see identifying_part_sources)."""
    return set(identifying_part_sources(values))


# a word followed by one of these names a business or an organisation ('WRENFIELD HARDWARE', 'Larkspur Medical'):
# a kept fact, not a person's or a street's name standing alone. The business word and up to two words before it are
# capitalised ('Vance Family Dental'); a lowercase phrase ('Vance called the bank') is ordinary prose, not a business.
_ORG_NEXT = re.compile(r"\s+(?:&\s+)?(?:[A-Z][A-Za-z'&]*\s+){0,2}?(?=[A-Z])(?i:" + "|".join((
    "ELECTRONICS", "BANK", "STORES?", "MARKET", "MART", "INC", "LLC", "LTD", "CORP(?:ORATION)?", "CO", "COMPANY", "PHARMACY",
    "RESTAURANT", "CAFE", "COFFEE", "GRILL", "HOTEL", "AUTO", "MOTORS", "SERVICES?", "SUPPLY", "HARDWARE", "GROCERY", "FOODS?",
    "OUTFITTERS", "AUDIO", "MEDICAL", "HEALTH", "DENTAL", "CLINIC", "HOSPITAL", "CENTER", "CENTRE", "GROUP", "PARTNERS",
    "INSURANCE", "CREDIT UNION", "UNIVERSITY", "COLLEGE", "SCHOOL", "LABS?", "STUDIOS?", "BAKERY", "WINES?", "FITNESS",
    "ASSOCIATES", "HOLDINGS", "TRUST", "FOUNDATION") + _TRADE_WORDS) + r")\b")


def standalone_spans(word, text, compound=True):
    """(start, end) of `word` in `text` as a whole word, written as in the source or in capitals, and not the first
    word of a business name. compound=False: also not inside a hyphenated or O'-style compound ('Smith-Vance')."""
    out = []
    rx = (r"(?<![A-Za-z])", r"(?![A-Za-z])") if compound else (r"(?<![A-Za-z'’-])", r"(?![A-Za-z-])")
    for form in {word, word.upper()}:
        for m in re.finditer(rx[0] + re.escape(form) + rx[1], text or ""):
            end = re.compile(r"(?:-[A-Za-z']+)*").match(text, m.end()).end()  # 'ASHCOMBE-ROURKE AUTO': the whole compound
            if not _ORG_NEXT.match(text, end):
                out.append((m.start(), m.end()))
    return sorted(out)


def _without_business(word, text):
    """`text` with each whole-word `word` (any case) that starts a business name ('HOLLOWAY AUTO') blanked out."""
    out = text or ""
    for m in reversed(list(re.finditer(r"(?<![A-Za-z'’-])" + re.escape(word) + r"(?![A-Za-z'’-])", out, re.I))):
        if _ORG_NEXT.match(out, m.end()):
            out = out[:m.start()] + " " * (m.end() - m.start()) + out[m.end():]
    return out


def _without_kept(text, kept):
    """`text` with every kept fact ('Rosalind Thistle', a merchant named like a person) blanked out, any case, so a
    single name word that also starts a kept merchant name is not counted there."""
    out = text or ""
    for k in sorted({x for x in kept if x and len(x) >= 3 and re.search(r"\s", x.strip())}, key=len, reverse=True):
        rx = r"(?<![A-Za-z'’-])" + r"\s+".join(map(re.escape, k.split())) + r"(?![A-Za-z'’-])"
        out = re.sub(rx, lambda m: " " * len(m.group()), out, flags=re.I)
    return out


def part_hits(parts, corpus, replacements=(), kept=(), group_ok=()):
    """Parts still readable in `corpus` as a whole word, written as in the source or in capitals. Text that belongs to a
    replacement (a fake that happens to share the word) or to a kept fact (an organisation's name) is removed first.
    A `group_ok` word after a determiner is not counted (see titled_only_words)."""
    t = corpus or ""
    for r in sorted({x for x in (*replacements, *kept) if x and len(x) >= 3}, key=len, reverse=True):
        t = t.replace(r, " ")
    group_ok = {g.lower() for g in group_ok}
    return {w for w in parts if any(not (w.lower() in group_ok and after_determiner(t, s0)) for s0, _ in standalone_spans(w, t))}


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


def render_pages(path, kind, dpis=None):
    if kind == "image":
        from PIL import Image
        return [np.asarray(Image.open(path).convert("RGB"))]
    import pymupdf
    out = []
    for i, p in enumerate(pymupdf.open(path)):
        # each page at the resolution it was read at, so edit boxes in page-image pixels line up with the output render
        dpi = dpis[i] if dpis and i < len(dpis) else RENDER_DPI
        pix = p.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
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


def verify_case(docs, outputs, occs, regions, logs, all_replacements, texts=None, kept=(), keep=None):
    """outputs: {doc_id: Path}. occs: accepted occurrences (with .text originals). `kept`: values deliberately kept as
    facts (organisation contact lines, places, businesses); the rescan does not report them. `keep`: the user's keep list
    (casepatch.keep.KeepList): values it covers are neither residual originals nor unexplained PII-like text.
    Returns (report, local_detail)."""
    from ..keep import as_keep_list
    keep = as_keep_list(keep)
    kept_n = [k for k in (_norm(x) for x in kept) if len(k) >= 4]
    originals = {}
    for o in occs:
        originals.setdefault(o.doc, set()).add(o.text)
    case_values = {o.text for o in occs}
    for d in docs:  # hidden-layer values that were stripped wholesale must not reappear either
        for r in regions:
            if r.payload:
                case_values.add(r.payload)
    full = {_norm(v) for v in case_values}
    parts = {w for w in identifying_parts((o.kind, o.text) for o in occs) if _norm(w) not in full}
    group_ok = titled_only_words([(o.kind, o.text) for o in occs if not (getattr(o, "meta", None) or {}).get("part_of")])
    # a name or street word replaced on its own as part of a longer value ('Pike' of 'Halvorsen-Pike') is looked for
    # the way it was found: a whole word in Title case or capitals ('the pike in the lake' is not the surname)
    part_values = {o.text.strip() for o in occs if (getattr(o, "meta", None) or {}).get("part_of")}
    parts |= part_values
    # a name word replaced on its own ('Holloway', 'PEMBRIDGE-TALLIS') can also be the name of a business that was kept
    # ('HOLLOWAY AUTO'): those business spans are not counted as the name
    from .resolve import particle_surname
    name_words = {o.text.strip() for o in occs if o.kind in ("PERSON", "GIVEN", "SURNAME") and
                  (not re.search(r"\s", o.text.strip()) or particle_surname(o.text.strip()))}  # also 'van der Brakel'

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
            rendered = render_pages(out, d.kind, [round(p.scale * 72) for p in d.pages] if d.kind == "pdf" else None)
            ocr_text = "\n".join(l.text for img in rendered for l in ocr_lines(img, "v:"))
            corpora["ocr_of_output_pixels"] = ocr_text
        # (3) residual originals
        hits = []
        for v in case_values:
            if len(v) < 4 or keep.covers(v) or v.strip() in part_values:
                continue
            for name, corpus in corpora.items():
                if v.strip() in name_words and isinstance(corpus, str):
                    corpus = _without_kept(_without_business(v.strip(), corpus), kept)
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
        # standalone name / street words from the replaced values (once per word per document)
        for w in sorted(parts):
            if keep.covers(w):
                continue
            layers = [name for name, corpus in corpora.items() if name != "pdf_objects"
                      and part_hits({w}, corpus, all_replacements, kept, group_ok)]
            if layers:
                hits.append({"layer": layers[0], "value": w, "part_of_replaced_value": True})
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
                if keep.covers(p.text):
                    continue  # on the user's keep list
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
