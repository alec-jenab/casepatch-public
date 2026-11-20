"""Score ANY system's transformed output folder against the fictional ground truth.

    python -m casepatch.bench.score --input examples/documents/case-dispute-01 --output <system_out_dir> \
        --system NAME [--map map.json] [--runtime-seconds S] [--usd-per-hour 0.34] [--json result.json]

Inputs needed: original case folder (with labels.json), the system's output folder, and a mapping from input file
names to output file names (auto-detected for CasePatch; otherwise same name/stem or --map). Nothing here calls a
network service. The scorer uses its own extraction/OCR for reading outputs (RapidOCR); OCR blind spots apply.

Metrics (schema casepatch-bench/v1):
- leakage: labeled sensitive values still present (text layer, PDF objects/metadata, OCR of pixels, QR payloads,
  plus any file in the output folder, e.g. reports)
- over_redaction: labeled non-sensitive facts that disappeared
- identity: per-entity consistency of replacements across documents + trace (via anchors), distinct people kept distinct
- evidence: discrepancy/outcome checks (claimed vs actual amount, dates, signer mismatch)
- visual: changed pixels outside sensitive/graphic regions, replacement readability at sensitive locations
- review_actions_estimate, runtime, compute cost per accepted case
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np

SCHEMA = "casepatch-bench/v1"


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _fuzzy(value, text, thr=0.9):
    v, t = _norm(value), _norm(text)
    if v in t:
        return True
    if len(v) < 6 or not re.search(r"[a-z]", v):
        return False
    n = len(v)
    return any(difflib.SequenceMatcher(None, v, t[i:i + n]).ratio() >= thr for i in range(max(1, len(t) - n + 1)) if t[i:i + 1] == v[:1])


def _contains(value, text, ocr=False, objects=False):
    if re.fullmatch(r"[\d\s-]+", value) and objects:
        # PDF object streams are full of numbers (path coordinates like "3.776 2.755"): a digit value only counts as
        # present when it appears as its own token, not stitched across whitespace or inside a decimal
        digits = re.sub(r"\D", "", value)
        return re.search(r"(?<![\d.])" + "-?".join(digits) + r"(?![\d.])", text) is not None
    if re.fullmatch(r"[\d\s-]+", value):
        digits = re.sub(r"\D", "", value)
        pat = r"(?<!\d)" + r"[\s-]?".join(digits) + r"(?!\d)" if len(digits) <= 19 else re.escape(value)
        return re.search(pat, text) is not None
    return _fuzzy(value, text) if ocr else _norm(value) in _norm(text)


class Reader:
    """Reads a file into text layers. Shared by input and output so both are read the same way."""

    def __init__(self):
        from ..docs.extract import ocr_lines
        self.ocr_lines = ocr_lines

    def layers(self, path):
        path = Path(path)
        ext = path.suffix.lower()
        out = {"text": "", "objects": "", "ocr": "", "qr": "", "images": []}
        if ext == ".json":
            out["text"] = path.read_text()
            out["json"] = json.loads(out["text"])
            return out
        from ..docs.verify import inspect_pdf, render_pages
        if ext == ".pdf":
            _, text, objs, info = inspect_pdf(path)
            out["text"], out["objects"] = text, objs + json.dumps(info.get("metadata", {}))
            imgs = render_pages(path, "pdf")
        else:
            from PIL import Image
            im = Image.open(path)
            ex = im.getexif()
            out["objects"] = json.dumps({str(k): str(v) for k, v in ex.items()}) + str(dict(ex.get_ifd(0x8825)))
            imgs = render_pages(path, "image")
        out["images"] = imgs
        lines = [self.ocr_lines(im, f"s{i}:") for i, im in enumerate(imgs)]
        out["ocr_lines"] = lines
        out["ocr"] = "\n".join(l.text for ls in lines for l in ls)
        out["qr"] = "\n".join(self._qr(im) for im in imgs)
        return out

    @staticmethod
    def _qr(img):
        import cv2
        try:
            ok, decoded, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(img)
            return "\n".join(d for d in decoded if d) if ok else ""
        except cv2.error:
            return ""


def auto_map(input_dir, output_dir):
    ins = sorted(f.name for f in Path(input_dir).iterdir() if f.is_file() and f.name != "labels.json" and f.suffix.lower() in
                 {".pdf", ".png", ".jpg", ".jpeg", ".json", ".tif", ".tiff", ".webp"})
    outs = {f.name for f in Path(output_dir).iterdir() if f.is_file()}
    m = {}
    for i, n in enumerate(ins, 1):
        cp = f"document-{i:02d}{Path(n).suffix.lower()}"
        if cp in outs:
            m[n] = cp
        elif n in outs:
            m[n] = n
        else:
            same = [o for o in outs if Path(o).stem == Path(n).stem]
            if same:
                m[n] = same[0]
    return m


def _parse_name(s):
    s = re.sub(r"^(?:Mr|Ms|Mrs|Dr|Ofc)\.?\s+", "", s.strip())
    m = re.fullmatch(r"([A-Za-z'’-]+),\s*([A-Za-z'’-]+)(?:\s+[A-Za-z]\.?)?", s)
    if m:
        return m.group(2).lower(), m.group(1).lower()
    toks = s.split()
    if len(toks) >= 2:
        return toks[0].rstrip(".").lower(), toks[-1].lower()
    return None


def _person_consistent(readings):
    parsed = [p for p in (_parse_name(r) for r in readings) if p]
    if len(parsed) < 2:
        return None
    surnames = {s for _, s in parsed}
    full = {g for g, _ in parsed if len(g) > 1}
    initials = {g[0] for g, _ in parsed}
    return len(surnames) == 1 and len(full) <= 1 and len(initials) == 1


def _json_get(d, path):
    for p in path:
        d = d[p]
    return d


def score(input_dir, output_dir, system, mapping=None, runtime_s=None, usd_per_hour=0.34, reader=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    labels = json.loads((input_dir / "labels.json").read_text())
    if mapping and not (set(mapping) & set(labels["occurrences"])) and (set(mapping.values()) & set(labels["occurrences"])):
        # accept CasePatch's review/file-map.json direction (output id -> original name) as well
        inv = {v: k for k, v in mapping.items()}
        outs = {q.stem: q.name for q in output_dir.iterdir()}
        mapping = {orig: outs.get(oid, oid) for orig, oid in inv.items()}
    mapping = mapping or auto_map(input_dir, output_dir)
    reader = reader or Reader()
    res = {"schema": SCHEMA, "system": system, "case": labels.get("case"), "mapping_found": len(mapping),
           "files_expected": len(labels["occurrences"])}
    ins, outs = {}, {}
    for f in labels["occurrences"]:
        ins[f] = reader.layers(input_dir / f)
        outs[f] = reader.layers(output_dir / mapping[f]) if f in mapping and (output_dir / mapping[f]).exists() else None
    # ---------------- leakage
    leaks, total = [], 0
    for f, values in labels["occurrences"].items():
        o = outs[f]
        for v in values:
            total += 1
            if o is None:
                leaks.append({"file": f, "value_kind": "missing_output", "layer": "none"}); continue
            for layer in ("text", "objects", "ocr", "qr"):
                if _contains(v, o[layer], ocr=(layer == "ocr"), objects=(layer == "objects")):
                    leaks.append({"file": f, "value": v, "layer": layer}); break
    # ---------------- ghost text: the original value still faintly visible under/around the new text. Plain OCR of the
    # output reads the dominant (new) text and misses it, so each original value location is re-read from (a) the
    # residual layer (output pixels kept only where the ORIGINAL had ink) and (b) a contrast-enhanced crop.
    leaked = {(x["file"], x.get("value")) for x in leaks}
    ghosts = []
    for f, values in labels["occurrences"].items():
        o = outs[f]
        if o is None or not o.get("images"):
            continue
        for v in values:
            if (f, v) in leaked or len(re.sub(r"\W", "", v)) < 3:
                continue
            hit = _ghost(reader, ins[f], o, v)
            if hit:
                g = {"file": f, "value": v, "layer": hit}
                ghosts.append(g); leaks.append(g)
    hidden_total, hidden_leaks = 0, []
    for f, groups in labels.get("hidden_or_non_text", {}).items():
        o = outs.get(f)
        for g, values in groups.items():
            for v in values:
                hidden_total += 1
                if o is None:
                    continue
                if v == "GPS":
                    if "GPS" in o["objects"] or "(45.0" in o["objects"]:
                        hidden_leaks.append({"file": f, "group": g, "value": v})
                elif any(_contains(v, o[layer], ocr=(layer == "ocr"), objects=(layer == "objects")) for layer in ("text", "objects", "qr", "ocr")):
                    hidden_leaks.append({"file": f, "group": g, "value": v})
    all_values = {v for e in labels["entities"] for v in e["values"] if len(v) >= 5}
    stray = []
    mapped = set(mapping.values())
    for p in output_dir.rglob("*"):
        if p.is_file() and p.name not in mapped and p.suffix.lower() in (".json", ".txt", ".csv", ".html", ".md", ".log"):
            t = p.read_text(errors="replace")
            stray += [{"file": p.name, "value": v} for v in all_values if v in t]
    res["leakage"] = {"labeled_occurrences": total, "leaked_occurrences": len(leaks), "leak_rate": round(len(leaks) / max(total, 1), 4),
                      "hidden_layer_values": hidden_total, "hidden_layer_leaks": len(hidden_leaks),
                      "other_output_files_with_originals": len(stray), "ghost_text_leaks": len(ghosts),
                      "detail": leaks + hidden_leaks + stray}
    # ---------------- over-redaction
    pres = labels["facts"]["preserved_values"]
    checked, missing = 0, []
    for f, i in ins.items():
        for v in pres:
            if _contains(v, i["text"]) or _contains(v, i["ocr"], ocr=True):
                checked += 1
                o = outs[f]
                if o is None or not (_contains(v, o["text"]) or _contains(v, o["ocr"], ocr=True)):
                    missing.append({"file": f, "value": v})
    res["over_redaction"] = {"preserved_facts_checked": checked, "facts_changed_or_lost": len(missing),
                             "over_redaction_rate": round(len(missing) / max(checked, 1), 4), "detail": missing}
    # ---------------- identity via anchors
    readings = {}
    unread = []
    for a in labels.get("anchors", []):
        o = outs.get(a["file"])
        val = None
        if o is not None:
            if "json_path" in a and "json" in o:
                try:
                    val = str(_json_get(o["json"], a["json_path"]))
                    if a.get("last"):
                        val = re.sub(r"\D", "", val)[-a["last"]:]
                except (KeyError, IndexError, TypeError):
                    val = None
            else:
                for layer in ("text", "ocr"):
                    m = re.search(a["regex"], o[layer]) if o[layer] else None
                    if m:
                        val = m.group(1).strip(); break
        if val is None:
            unread.append({"entity": a["entity"], "file": a["file"]})
        else:
            readings.setdefault(a["entity"], []).append(val)
    ident = {}
    for ent, vals in readings.items():
        if ent.startswith("person:"):
            ok = _person_consistent(vals)
        else:
            norm = {re.sub(r"\D", "", v) for v in vals}
            ok = (len(norm) == 1) if len(vals) >= 2 else None
        originals = {v.lower() for e in labels["entities"] for v in e["values"]}
        kept = sum(1 for v in vals if v.lower() in originals)
        ident[ent] = {"readings": len(vals), "consistent": ok, "original_values_retained": kept, "values": vals}
    scored = [v["consistent"] for v in ident.values() if v["consistent"] is not None]
    res["identity"] = {"entities_scored": len(scored), "consistent": sum(scored), "consistency_rate": round(sum(scored) / max(len(scored), 1), 4),
                       "anchors_unreadable": len(unread), "per_entity": ident, "unreadable": unread}
    # ---------------- evidence
    ev = []
    for e in labels.get("evidence", []):
        if "must_contain" in e:
            o = outs.get(e["file"])
            ok = o is not None and (e["must_contain"] in o["text"] or _contains(e["must_contain"], o["ocr"], ocr=True))
        elif e.get("kind") == "distinct_entities":
            a, b = (readings.get(x, []) for x in e["entities"])
            pa, pb = {_parse_name(x) for x in a} - {None}, {_parse_name(x) for x in b} - {None}
            ok = bool(pa and pb) and not ({g for g, _ in pa if len(g) > 1} & {g for g, _ in pb if len(g) > 1})
        elif e.get("kind") == "shared_surname":
            a, b = (readings.get(x, []) for x in e["entities"])
            sa = {p[1] for p in map(_parse_name, a) if p}; sb = {p[1] for p in map(_parse_name, b) if p}
            ok = bool(sa & sb)
        else:
            ok = None
        ev.append({"id": e["id"], "passed": ok, "optional": e.get("optional", False)})
    req = [x for x in ev if not x["optional"]]
    res["evidence"] = {"checks": len(req), "passed": sum(1 for x in req if x["passed"]), "detail": ev}
    # ---------------- visual
    res["visual"] = _visual(labels, ins, outs)
    res["regions"] = _region_checks(labels, ins, outs)
    # ---------------- review actions, runtime, cost
    sys_flags = 0
    rep = output_dir / "report.json"
    if rep.exists():
        try:
            r = json.loads(rep.read_text())
            sys_flags = len(r.get("needs_review", []))
            if runtime_s is None:
                runtime_s = sum(s.get("seconds", 0) for k, s in r.get("stages", {}).items() if not k.startswith("verify"))
        except (json.JSONDecodeError, AttributeError):
            pass
    fails = res["leakage"]["leaked_occurrences"] + res["leakage"]["hidden_layer_leaks"] + res["over_redaction"]["facts_changed_or_lost"] + \
        (res["identity"]["entities_scored"] - res["identity"]["consistent"]) + (res["evidence"]["checks"] - res["evidence"]["passed"])
    res["review_actions_estimate"] = {"system_flagged_items": sys_flags, "corrections_needed": fails, "total": sys_flags + fails,
                                      "note": "1 action per flagged item to confirm + 1 per error a reviewer must find and fix"}
    accepted = res["leakage"]["leaked_occurrences"] == 0 and res["leakage"]["hidden_layer_leaks"] == 0 and \
        res["evidence"]["passed"] == res["evidence"]["checks"] and res["identity"]["consistent"] == res["identity"]["entities_scored"]
    cost = round(runtime_s / 3600 * usd_per_hour, 5) if runtime_s is not None else None
    res["runtime"] = {"seconds": round(runtime_s, 2) if runtime_s is not None else None, "usd_per_hour_assumed": usd_per_hour,
                      "compute_cost_usd": cost, "accepted_without_correction": accepted,
                      "cost_per_accepted_case_usd": cost if accepted else None}
    return res


MARKER_WORDS = re.compile(r"\b(WITHHELD|REDACTED|REDACTION|CENSORED|ANONYMI[SZ]ED|\[?PII\]?|XXXX+)\b", re.I)


def _span_chars(value, chars):
    """The characters of an OCR line that spell `value` (exact, digit-flexible or best fuzzy window), so the ghost
    check compares only the replaced span, not the rest of the line (unchanged neighbours correlate by definition).
    Falls back to the whole line when the value cannot be located in the character run."""
    s = "".join(c.c for c in chars).lower()
    v = _norm(value)
    if not s or not v:
        return chars
    if re.fullmatch(r"[\d\s-]+", value):
        digits = re.sub(r"\D", "", value)
        pat = r"(?<!\d)" + r"[\s-]?".join(digits) + r"(?!\d)"
    else:
        pat = r"\s*".join(re.escape(ch) for ch in v.replace(" ", ""))
    m = re.search(pat, s)
    if m:
        i, j = m.start(), m.end()
    else:
        n, best, i, j = len(v), 0.0, 0, len(s)
        for k in range(max(1, len(s) - n + 1)):
            r = difflib.SequenceMatcher(None, v, s[k:k + n]).ratio()
            if r > best:
                best, i, j = r, k, min(len(s), k + n)
        if best < 0.85:
            return chars
    span = [c for c in chars[i:j] if c.c.strip()]
    return span or chars


def _ghost(reader, i, o, value):
    """Re-read the original value's location in the output from the residual layer and an enhanced crop."""
    import cv2
    import numpy as np
    v = _norm(value)
    for pi, lines in enumerate(i.get("ocr_lines", [])):
        if pi >= len(o["images"]) or pi >= len(i["images"]):
            continue
        a, b = i["images"][pi], o["images"][pi]
        if a.shape != b.shape:
            continue
        for l in lines:
            if not l.chars or not (_contains(value, l.text, ocr=True)):
                continue
            span = _span_chars(value, l.chars)
            xs = [p[0] for c in span for p in c.quad]; ys = [p[1] for c in span for p in c.quad]
            h = max(ys) - min(ys)
            x0, y0 = int(max(0, min(xs) - h)), int(max(0, min(ys) - 0.5 * h))
            x1, y1 = int(min(a.shape[1], max(xs) + h)), int(min(a.shape[0], max(ys) + 0.5 * h))
            ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
            if ca.size == 0 or min(ca.shape[:2]) < 8:
                continue
            ga, gb = cv2.cvtColor(ca, cv2.COLOR_RGB2GRAY), cv2.cvtColor(cb, cv2.COLOR_RGB2GRAY)
            ink = cv2.adaptiveThreshold(ga, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10) > 0
            k = 31 if min(gb.shape) > 31 else 3
            bg = cv2.medianBlur(gb, k)
            dark = np.clip(bg.astype(np.int16) - gb.astype(np.int16), 0, 255)
            bga = cv2.medianBlur(ga, k)
            dark0 = np.clip(bga.astype(np.int16) - ga.astype(np.int16), 1, 255)
            # an original ink pixel is 'still there' only if the output is at least ~40% as dark as the original was
            keep = ink & (dark >= 0.4 * dark0) & (dark >= 12)
            res = np.full_like(gb, 255)
            res[keep] = 0
            # pixel evidence: inside the value's own box the output still correlates strongly with the original glyphs
            # (a clean replacement draws different glyphs; calibrated on case-real-02: ghosted 0.81 vs clean 0.31-0.55)
            ty0, ty1, tx0, tx1 = int(min(ys)) - y0, int(max(ys)) - y0, int(min(xs)) - x0, int(max(xs)) - x0
            ta, tb = ga[ty0:ty1, tx0:tx1].astype(np.float32), gb[ty0:ty1, tx0:tx1].astype(np.float32)
            # letters only: digit-for-digit replacements in the same font sit on the same grid and correlate by design
            if sum(ch.isalpha() for ch in value) >= 4 and ta.size > 50 and ta.std() > 5 and tb.std() > 5:
                corr = float(np.corrcoef(ta.ravel(), tb.ravel())[0, 1])
                persist = float((keep[ty0:ty1, tx0:tx1]).sum()) / max(1.0, float(ink[ty0:ty1, tx0:tx1].sum()))
                if corr >= 0.75 and persist >= 0.6:
                    return f"ghost_ink_persistence(corr={corr:.2f})"
            clahe = cv2.createCLAHE(3.0, (8, 8)).apply(gb)
            for name, im in (("ghost_residual_layer", res), ("ghost_enhanced_crop", clahe)):
                big = cv2.resize(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                t = " ".join(x.text for x in reader.ocr_lines(big, "g:"))
                tn = _norm(t)
                if re.fullmatch(r"[\d\s.-]+", value):
                    if _contains(value, t):
                        return name
                elif len(v) < 6:
                    if re.search(r"(?<![a-z])" + re.escape(v) + r"(?![a-z])", tn):
                        return name
                elif _fuzzy(value, t, 0.85):
                    return name
    return None


def _crop(img, bb, pad=0):
    x0, y0, x1, y1 = [int(round(v)) for v in bb]
    return img[max(0, y0 - pad):max(0, y1 + pad), max(0, x0 - pad):max(0, x1 + pad)]


def _codes(img):
    try:
        import zxingcpp
        out = []
        for r in zxingcpp.read_barcodes(img):
            p = r.position
            xs = [p.top_left.x, p.top_right.x, p.bottom_right.x, p.bottom_left.x]
            ys = [p.top_left.y, p.top_right.y, p.bottom_right.y, p.bottom_left.y]
            out.append((r.text, (min(xs), min(ys), max(xs), max(ys)), str(r.format)))
        return out
    except Exception:
        return []


def _label_regions(labels, f, img_in, scale):
    """Labelled regions of file f in image pixels: explicit bboxes (labels['regions'], px for images, PDF points
    for PDFs) else auto-located from must_withhold_regions kinds (faces by YuNet, codes by zxing)."""
    regs = []
    for r in (labels.get("regions") or {}).get(f, []):
        bb = r["bbox"] if "bbox" in r else [v * scale for v in r["bbox_pt"]]
        regs.append({"kind": r["kind"], "role": r.get("role", "primary"), "bbox": bb, "page": r.get("page", 0)})
    if regs:
        return regs
    kinds = (labels.get("must_withhold_regions") or {}).get(f, [])
    if "face" in kinds:
        from ..docs.native import detect_faces
        fs = sorted(detect_faces(img_in, 0.6), key=lambda z: -z[2] * z[3])
        for i, z in enumerate(fs):
            regs.append({"kind": "face", "role": "primary" if i == 0 else "ghost", "bbox": [z[0], z[1], z[0] + z[2], z[1] + z[3]], "page": 0})
    for kind in ("qr", "barcode"):
        if kind in kinds:
            for txt, bb, fmt in _codes(img_in):
                if (kind == "qr") == ("QR" in fmt.upper()):
                    regs.append({"kind": kind, "role": "primary", "bbox": list(bb), "page": 0, "payload_in": txt})
    if "signature" in kinds and not any(r["kind"] == "signature" for r in regs):
        regs.append({"kind": "signature", "role": "unlocated", "bbox": None, "page": 0})
    return regs


def _region_checks(labels, ins, outs):
    """Region-specific checks: face replaced (not the same person, still a face), ghost/secondary portraits replaced,
    one synthetic person per document, signature changed (not erased), code payload changed and still decodable, and
    no visible redaction marker (marker words, new solid boxes)."""
    import cv2
    from ..docs import native as N
    files, checks = {}, {"face_replaced": [0, 0], "ghost_face_replaced": [0, 0], "portraits_consistent": [0, 0],
                         "signature_changed": [0, 0], "code_payload_changed": [0, 0], "no_visible_redaction_marker": [0, 0]}
    for f, i in ins.items():
        o = outs.get(f)
        if o is None or f.endswith(".json") or not i.get("images"):
            continue
        a = i["images"][0]; b = o["images"][0] if o["images"] else None
        if b is None:
            continue
        if b.shape != a.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        scale = (200 / 72) if f.lower().endswith(".pdf") else 1.0
        res = []
        out_embs = []
        for r in _label_regions(labels, f, a, scale):
            if r["bbox"] is None:
                res.append({**r, "result": "not_located"}); continue
            ca, cb = _crop(a, r["bbox"], 6), _crop(b, r["bbox"], 6)
            if r["kind"] == "face":
                e0, e1 = N.face_embedding(ca), N.face_embedding(cb)
                sim = N.face_similarity(e0, e1)
                ok = e1 is not None and sim is not None and sim < N.SAME_PERSON_COSINE
                key = "face_replaced" if r["role"] == "primary" else "ghost_face_replaced"
                checks[key][0] += ok; checks[key][1] += 1
                if e1 is not None:
                    out_embs.append(e1)
                res.append({"kind": "face", "role": r["role"], "bbox": [round(v) for v in r["bbox"]], "face_after": e1 is not None,
                            "cosine_to_original": None if sim is None else round(sim, 3), "pass": bool(ok)})
            elif r["kind"] == "signature":
                ga, gb = cv2.cvtColor(ca, cv2.COLOR_RGB2GRAY), cv2.cvtColor(cb, cv2.COLOR_RGB2GRAY)
                ia = cv2.adaptiveThreshold(ga, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12) > 0
                ib = cv2.adaptiveThreshold(gb, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12) > 0
                iou = float((ia & ib).sum() / max(1, (ia | ib).sum()))
                ink_ratio = float(ib.sum() / max(1, ia.sum()))
                ok = iou < 0.6 and ink_ratio > 0.25  # different strokes, and not simply erased
                checks["signature_changed"][0] += ok; checks["signature_changed"][1] += 1
                res.append({"kind": "signature", "bbox": [round(v) for v in r["bbox"]], "ink_iou_vs_original": round(iou, 3),
                            "ink_ratio_after_vs_before": round(ink_ratio, 3), "pass": bool(ok)})
            else:
                pin = r.get("payload_in") or next((t for t, _, _ in _codes(ca)), None)
                pout = next((t for t, _, _ in _codes(cb)), None)
                ok = pout is not None and pin is not None and pout != pin
                checks["code_payload_changed"][0] += ok; checks["code_payload_changed"][1] += 1
                res.append({"kind": r["kind"], "bbox": [round(v) for v in r["bbox"]], "decodes_after": pout is not None,
                            "payload_changed": None if pout is None or pin is None else pout != pin, "pass": bool(ok)})
        if len(out_embs) > 1:
            sims = [N.face_similarity(out_embs[0], e) for e in out_embs[1:]]
            ok = all(x is not None and x >= N.SAME_PERSON_COSINE for x in sims)
            checks["portraits_consistent"][0] += ok; checks["portraits_consistent"][1] += 1
            res.append({"kind": "portrait_consistency", "cosine_primary_vs_others": [round(x, 3) for x in sims], "pass": bool(ok)})
        # visible redaction markers: marker words in OCR of the output that are not in the input, or new solid boxes
        words = sorted({m.group(0) for m in MARKER_WORDS.finditer(o.get("ocr", ""))} - {m.group(0) for m in MARKER_WORDS.finditer(i.get("ocr", ""))})
        da = (a.max(axis=2) < 60).astype(np.uint8); db = (b.max(axis=2) < 60).astype(np.uint8)
        new = (db & (1 - cv2.dilate(da, np.ones((5, 5), np.uint8)))).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(new)
        boxes = [list(map(int, st[k, :4])) for k in range(1, n)
                 if st[k, 4] > 1500 and st[k, 4] / max(1, st[k, 2] * st[k, 3]) > 0.9 and min(st[k, 2], st[k, 3]) > 12]
        ok = not words and not boxes
        checks["no_visible_redaction_marker"][0] += ok; checks["no_visible_redaction_marker"][1] += 1
        res.append({"kind": "redaction_markers", "marker_words": words, "new_solid_boxes": boxes[:10], "pass": bool(ok)})
        files[f] = res
    return {"per_file": files, "summary": {k: {"passed": v[0], "total": v[1]} for k, v in checks.items()}}


def _visual(labels, ins, outs):
    import cv2
    from ..docs.visual import find_regions
    from ..docs.model import Page, quad_bbox, span_quad
    per = {}
    for f, values in labels["occurrences"].items():
        i, o = ins[f], outs[f]
        if f.endswith(".json") or o is None or not i["images"]:
            continue
        pages = []
        for pi, (a, lines) in enumerate(zip(i["images"], i["ocr_lines"])):
            if pi >= len(o["images"]):
                break
            b = o["images"][pi]
            if b.shape != a.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            mask = np.zeros(a.shape[:2], np.uint8)
            boxes = []
            allv = {v for e in labels["entities"] for v in e["values"]}
            for l in lines:
                for v in allv:
                    for m in re.finditer(re.escape(v), l.text):
                        bb = quad_bbox(span_quad(l.chars, m.start(), m.end()))
                        boxes.append(bb)
            pg = Page(pi, a.shape[1], a.shape[0], True, 1.0, a, lines)
            for r in find_regions(pg, "x"):
                boxes.append(quad_bbox(r.quad))
            # pdf annotation icons etc. are not modelled; they show up as outside-region change
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(mask, (int(x0) - 8, int(y0) - 8), (int(x1) + 8, int(y1) + 8), 255, -1)
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 48
            outside = diff & (mask == 0)
            n, _, st, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8))
            readable = blank = 0
            from ..docs.extract import ocr_lines
            for x0, y0, x1, y1 in boxes[:200]:
                crop = b[max(0, int(y0) - 4):int(y1) + 4, max(0, int(x0) - 4):int(x1) + 4]
                if crop.size == 0:
                    continue
                if crop.std() < 10 or "WITHHELD" in " ".join(l.text for l in ocr_lines(crop, "r:")):
                    blank += 1; continue
                ls = ocr_lines(crop, "r:")
                if ls and max(l.conf for l in ls) >= 0.8:
                    readable += 1
            pages.append({"changed_px_outside_sensitive_regions": int(outside.sum()), "fraction_outside": round(float(outside.mean()), 6),
                          "largest_blob_px": int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0,
                          "sensitive_boxes": len(boxes), "replaced_readable": readable, "blanked_or_withheld": blank})
        per[f] = pages
    flat = [p for ps in per.values() for p in ps]
    return {"per_file": per, "mean_fraction_outside": round(float(np.mean([p["fraction_outside"] for p in flat])), 6) if flat else None,
            "readable_rate": round(sum(p["replaced_readable"] for p in flat) / max(1, sum(p["sensitive_boxes"] for p in flat)), 3) if flat else None}


def main():
    ap = argparse.ArgumentParser(prog="casepatch.bench.score")
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True); ap.add_argument("--system", required=True)
    ap.add_argument("--map"); ap.add_argument("--runtime-seconds", type=float); ap.add_argument("--usd-per-hour", type=float, default=0.34)
    ap.add_argument("--json")
    a = ap.parse_args()
    mapping = json.loads(Path(a.map).read_text()) if a.map else None
    r = score(a.input, a.output, a.system, mapping, a.runtime_seconds, a.usd_per_hour)
    s = json.dumps(r, indent=2, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(s)
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("detail", "per_entity", "unreadable", "per_file")} if isinstance(v, dict) else v
               for k, v in r.items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
