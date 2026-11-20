"""Non-text visual regions: QR codes, barcodes, faces/photos, signatures/handwriting, other graphics.
Policy (v0.4): faces, signatures and codes are REPLACED in place by the native engine (synthetic face, different
synthetic signature, regenerated symbol with mapped payload); photos/graphics without a face are kept and flagged
for review. Nothing is silently passed through: every region is listed in the report."""
from __future__ import annotations

import os
import re

import numpy as np

from .model import Region, quad_bbox, quad_from_rect

_FACE = None


def _faces(crop):
    """YuNet (OpenCV zoo, 230 KB ONNX) if present locally; None when no face model is installed."""
    import cv2
    from .models import find_model
    path = find_model("face_detection_yunet_2023mar.onnx")
    if path is None or not hasattr(cv2, "FaceDetectorYN"):
        return None
    h, w = crop.shape[:2]
    det = cv2.FaceDetectorYN.create(str(path), "", (w, h), 0.6)
    _, faces = det.detect(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return [] if faces is None else faces


def _text_mask(page, shape, scale):
    import cv2
    m = np.zeros(shape[:2], np.uint8)
    for l in page.lines:
        if not l.chars or l.source in ("native_hidden", "json"):
            continue
        if l.conf < 0.8:
            continue  # low-confidence OCR over a signature/graphic is not trusted as text
        if _looks_like_scribble(l.text):
            continue
        q = np.array([(l.chars[0].quad[0]), l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]], np.float32) * scale
        cv2.fillPoly(m, [q.astype(np.int32)], 255)
    return cv2.dilate(m, np.ones((9, 9), np.uint8))


def _looks_like_scribble(text):
    t = text.replace(" ", "")
    return len(t) >= 5 and set(t.lower()) <= set("mwnvauil") and not any(v in t.lower() for v in ("ma", "an"))  # 'wMwWWMMAN'


def classify(crop, ink=None):
    """Return (kind, reason, payload)."""
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if min(h, w) >= 21:
        try:
            ok, pts = cv2.QRCodeDetector().detect(crop)
            if ok and pts is not None:
                val = cv2.QRCodeDetector().detectAndDecode(crop)[0]
                return "qr", "QR finder patterns detected" + ("; payload decoded (kept local)" if val else ""), val or None
        except cv2.error:
            pass
    bw = (gray < 128).astype(np.uint8)
    rows = bw[h // 4: 3 * h // 4] if h >= 8 else bw
    if rows.size and w > 60:
        trans = np.abs(np.diff(rows.astype(np.int8), axis=1)).sum(axis=1)
        colfill = bw[h // 4: 3 * h // 4].mean(axis=0)
        if np.median(trans) > 40 and ((colfill > 0.85) | (colfill < 0.15)).mean() > 0.8 and w / max(h, 1) > 2.5:
            # handwriting blocks can look like bars in a column profile: require a decodable symbol, or (no decoder
            # installed) bars that are truly solid over most columns and rows
            if _decode_barcode(crop):
                return "barcode", "dense parallel bars; symbol decoded (payload kept local)", None
            if (colfill > 0.85).mean() > 0.25 and float(np.std(trans)) < 0.25 * float(np.mean(trans)):
                return "barcode", "dense solid parallel bars (not decoded)", None
    faces = _faces(crop) if min(h, w) > 40 else []
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    colorful = float(hsv[..., 1].mean()) > 45 and float(gray.std()) > 25
    if faces is not None and len(faces):
        return "face", "face detector (YuNet) hit", None
    if colorful and h * w > 8000 and 0.5 < w / max(h, 1) < 2.0:
        return "photo", "colorful textured block (possible face/photo; " + ("face detector found no face" if faces is not None else "no face model installed") + ")", None
    inkf = float((gray < 140).mean())
    if w / max(h, 1) > 1.6 and inkf < 0.3:
        return "signature", "thin connected strokes without OCR-legible text (signature/handwriting)", None
    return "graphic", "unclassified graphic region", None


def _decode_barcode(crop):
    """True/False when a 1-D/2-D decoder is installed (zxing-cpp), None when none is available."""
    try:
        import zxingcpp
    except ImportError:
        return None
    try:
        return bool(zxingcpp.read_barcodes(crop))
    except Exception:
        return False


def find_regions(page, doc_id):
    import cv2
    img = page.image
    if img is None:
        return []
    H, W = img.shape[:2]
    out = []
    if not page.raster:
        # Native page: every embedded image is classified; text-only images (stamps) are handled as pixel text.
        tmask = _text_mask(page, img.shape, page.scale)
        for i, bb in enumerate(getattr(page, "embedded_images", [])):
            x0, y0, x1, y1 = [int(v * page.scale) for v in bb]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = img[y0:y1, x0:x1]
            textcov = float((tmask[y0:y1, x0:x1] > 0).mean())
            dark = ((cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) < 200) & (tmask[y0:y1, x0:x1] == 0)).astype(np.uint8)
            lines_only = cv2.bitwise_or(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8)),
                                        cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((40, 1), np.uint8)))
            leftover = float((dark & (1 - lines_only)).mean())
            kind, reason, payload = classify(crop)
            if kind in ("graphic", "signature") and textcov > 0.25 and leftover < 0.01:
                page.notes.append({"code": "TEXT_IMAGE_HANDLED_AS_PIXEL_TEXT", "bbox": [round(v, 1) for v in bb]})
                continue
            if kind == "signature" and (textcov > 0.1 or (x1 - x0) * (y1 - y0) > 0.15 * H * W):
                # a picture of a filled-in form (lots of OCR text, a big share of the page) is not a signature; its
                # text is handled line by line as pixel text and any handwriting in it by the raster rules
                page.notes.append({"code": "TEXT_IMAGE_HANDLED_AS_PIXEL_TEXT", "bbox": [round(v, 1) for v in bb]})
                continue
            out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", kind, quad_from_rect(*bb), "embedded image: " + reason, payload))
        return out
    # Raster page: paper mask, then ink/colour clusters that OCR did not explain as text.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    paper = np.full((H, W), 255, np.uint8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[big, cv2.CC_STAT_AREA] > 0.25 * H * W:
            paper = (lab == big).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            paper = cv2.drawContours(np.zeros_like(paper), [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    paper = cv2.erode(paper, np.ones((25, 25), np.uint8))
    paper_color = np.median(img[(paper > 0) & (bright > 0)], axis=0) if (paper > 0).any() else np.array([255, 255, 255])
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 25)
    colour = (np.abs(img.astype(np.int16) - paper_color.astype(np.int16)).max(axis=2) > 60).astype(np.uint8) * 255
    cand = cv2.bitwise_or(ink, colour)
    cand[paper == 0] = 0
    cand[_text_mask(page, img.shape, 1.0) > 0] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    merged = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.002 * H * W or w * h > 0.45 * H * W or min(w, h) < 18:
            continue
        crop = img[y:y + h, x:x + w]
        sub = cand[y:y + h, x:x + w]
        if (sub > 0).mean() < 0.01:
            continue
        # uniform design fills (header bars) are not evidence regions
        region_px = crop[cv2.dilate(sub, np.ones((5, 5), np.uint8)) > 0]
        if region_px.size and region_px.std(axis=0).mean() < 10 and w / max(h, 1) > 4:
            page.notes.append({"code": "UNIFORM_DESIGN_FILL_IGNORED", "bbox": [x, y, x + w, y + h]}); continue
        kind, reason, payload = classify(crop)
        if kind == "signature":
            # stroke clusters without a printed signature label may be handwriting, stamps or anonymisation blur:
            # never redrawn automatically (that could invent evidence); kept and flagged unless confirmed
            kind, reason = "handwriting", reason + "; no signature label nearby: kept for review"
        pad = 6
        out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", kind,
                          quad_from_rect(max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)), reason, payload))
    out = _add_faces_and_signatures(page, doc_id, out)
    # QR detector over the whole page catches codes split by the clustering.
    try:
        ok, decoded, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(img)
        if ok and pts is not None:
            for val, q in zip(decoded, pts):
                bb = quad_bbox(q)
                same = [r for r in out if _iou(quad_bbox(r.quad), bb) > 0.3]
                for r in same:
                    if r.kind != "qr":
                        r.kind, r.reason, r.payload = "qr", "page-level QR detection", val or None
                if not same:
                    out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", "qr", quad_from_rect(*bb), "page-level QR detection", val or None))
    except cv2.error:
        pass
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-6)


SIG_LABEL = re.compile(r"signature|firma|sign here|signed by|unterschrift|подпись|υπογραφ|\(\s*\w{6,10}\s+or\s+mark\s*\)", re.I)
SIG_WORDS = ("signature", "unterschrift", "firma")


def is_sig_label(text):
    """Printed signature label, tolerant of OCR errors ('eignature'); a long sentence that merely says 'signed'
    ('A signed duplicate of the photograph...') is not a label."""
    import difflib
    t = text.strip()
    if SIG_LABEL.search(t):
        return len(t.split()) <= 8 or t.lower().count("signature") > 0 and t.startswith("(")
    words = re.findall(r"[a-z]{6,}", t.lower())
    if len(t.split()) <= 6 and any(difflib.SequenceMatcher(None, w, s).ratio() >= 0.8 for w in words for s in SIG_WORDS):
        return True
    return bool(re.fullmatch(r"\(?(?:signed|signer|sign)\)?[.:]?", t, re.I))


def _add_faces_and_signatures(page, doc_id, out):
    """Whole-page face detection (portraits on photographed cards are often not separable from the background by
    the paper/ink clustering) and signatures located next to their printed label."""
    import cv2
    img = page.image
    H, W = img.shape[:2]
    try:
        from .native import detect_faces
        faces = detect_faces(img, 0.6)
    except Exception:
        faces = []
    for f in faces:
        x, y, w, h = [float(v) for v in f[:4]]
        if w * h < 300:
            continue
        cx, cy = x + w / 2, y + h / 2
        bw, bh = w * 1.75, h * 2.2
        bb = (max(0, cx - bw / 2), max(0, cy - bh * 0.45), min(W, cx + bw / 2), min(H, cy + bh * 0.55))
        hit = [r for r in out if _iou(quad_bbox(r.quad), bb) > 0.2 or _contains(quad_bbox(r.quad), (cx, cy))]
        for r in hit:
            out.remove(r)
        out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", "face", quad_from_rect(*bb), "face detector (YuNet) on full page", None))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    printed = np.zeros((H, W), np.uint8)
    for l in page.lines:
        if l.chars and l.conf >= 0.9 and len(l.text.strip()) >= 2 and not _looks_like_scribble(l.text):
            q = np.array([l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]], np.float32) * (page.scale if not page.raster else 1.0)
            cv2.fillPoly(printed, [q.astype(np.int32)], 255)
    printed = cv2.dilate(printed, np.ones((5, 5), np.uint8))
    # EU/ISO licence layout: field 7 is the holder's signature (its OCR text, if any, is the scrawl itself)
    try:
        from .forms import eu_licence_fields, eu_field_values, _bb
        sc0 = page.scale if not page.raster else 1.0
        ln = [(l, _bb(l, sc0)) for l in page.lines if l.chars and l.text.strip()]
        marks = eu_licence_fields(ln)
        vals = eu_field_values(ln, marks).get("7", []) if marks else []
        mk = next((m for m in marks if m[0] == "7"), None)
        if mk and vals:
            bbs = [_bb(l, sc0) for l, _ in vals]
            lh = mk[2][3] - mk[2][1]
            bb = (min(b[0] for b in bbs) - 0.2 * lh, min(b[1] for b in bbs) - 0.2 * lh, max(b[2] for b in bbs) + 0.2 * lh, max(b[3] for b in bbs) + 0.2 * lh)
            if not any(_iou(quad_bbox(r.quad), bb) > 0.3 for r in out if r.kind == "signature"):
                out = [r for r in out if not (_iou(quad_bbox(r.quad), bb) > 0.3 and r.kind != "face")]
                out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", "signature", quad_from_rect(*(v / sc0 for v in bb)),
                                  "licence field 7 (holder's signature)", None))
    except Exception:
        pass
    _hs = sorted((max(p[1] for c in l.chars for p in c.quad) - min(p[1] for c in l.chars for p in c.quad)) for l in page.lines if l.chars)
    _medh = _hs[len(_hs) // 2] if _hs else 0
    def _tall(l):
        if not l.chars or len(l.text.split()) != 1 or not _medh:
            return False
        return max(p[1] for c in l.chars for p in c.quad) - min(p[1] for c in l.chars for p in c.quad) > 1.8 * _medh
    # tall specimen scrawls first: a printed signature label next to one is already answered by it
    for l in sorted(page.lines, key=lambda l: not (is_sig_label(l.text) and _tall(l)) if l.chars else True):
        if not l.chars or not is_sig_label(l.text):
            continue
        _lh = max(p[1] for c in l.chars for p in c.quad) - min(p[1] for c in l.chars for p in c.quad)
        if len(l.text.split()) == 1 and _medh and _lh > 1.8 * _medh:
            # one tall word ('Signature' written as a specimen scrawl): that line IS the signature, not its label
            sc1 = page.scale if not page.raster else 1.0
            xs = [p[0] * sc1 for c in l.chars for p in c.quad]; ys = [p[1] * sc1 for c in l.chars for p in c.quad]
            bb = (min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4)
            if not any(_iou(quad_bbox(r.quad), bb) > 0.3 for r in out if r.kind == "signature"):
                out = [r for r in out if not (_iou(quad_bbox(r.quad), bb) > 0.3 and r.kind != "face")]
                out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", "signature", quad_from_rect(*(v / sc1 for v in bb)),
                                  "tall single-word scrawl OCR'd as a signature word", None))
            continue
        sc = page.scale if not page.raster else 1.0
        q = np.array([l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]], np.float32) * sc
        x0, y0 = q[:, 0].min(), q[:, 1].min(); x1, y1 = q[:, 0].max(), q[:, 1].max()
        # a 'label' that lies inside an already located signature is the scrawl itself, OCR'd as a word
        if any(_contains(quad_bbox(r.quad), ((x0 + x1) / 2 / sc, (y0 + y1) / 2 / sc)) for r in out if r.kind == "signature"):
            continue
        lh = max(8.0, y1 - y0); lw = x1 - x0
        if any(r.kind == "signature" and _iou(quad_bbox(r.quad), (x0 / sc - lw * 0.3 / sc, y0 / sc - 3 * lh / sc, x1 / sc + lw * 0.3 / sc, y1 / sc + 3.5 * lh / sc)) > 0
               for r in out):
            continue  # the signature next to this label was already located (e.g. a tall specimen scrawl)
        cands = [(x0 - lw * 0.6, y1, x1 + lw * 1.6, y1 + 3.2 * lh),          # below the label
                 (x1 + lh * 0.3, y0 - 1.5 * lh, x1 + lh * 14, y1 + 1.2 * lh),  # right of the label
                 (x0 - lw * 1.2, y0 - 3.2 * lh, x1 + lw * 1.6, y0)]          # above the label (line under a signature)
        best = None
        for cx0, cy0, cx1, cy1 in cands:
            X0, Y0, X1, Y1 = int(max(0, cx0)), int(max(0, cy0)), int(min(W, cx1)), int(min(H, cy1))
            if X1 - X0 < 10 or Y1 - Y0 < 6:
                continue
            m = (ink[Y0:Y1, X0:X1] > 0) & (printed[Y0:Y1, X0:X1] == 0)
            # drop long straight rules
            rules = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(15, int(lh * 3))), np.uint8))
            m = m & ~(cv2.dilate(rules, np.ones((3, 1), np.uint8)) > 0)
            # dotted / dashed rules ('..........(Signature or mark)'): bridge the dots horizontally, keep only long thin
            # runs, and drop them too, or they chain the signature to the far edge of the window
            thin = m & ~(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((4, 1), np.uint8)) > 0)
            dots = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, max(3, int(lh * 0.5)) | 1), np.uint8))
            dots = cv2.morphologyEx(dots, cv2.MORPH_OPEN, np.ones((1, max(15, int(lh * 4))), np.uint8))
            m = m & ~(cv2.dilate(dots, np.ones((3, 1), np.uint8)) > 0)
            if m.sum() < lh * lh * 0.6:
                continue
            cl = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(lh * 0.6) | 1, int(lh * 0.9) | 1), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(cl)
            comps = []
            for i in range(1, n):
                x, y, w, h = st[i, :4]
                strokes = m[y:y + h, x:x + w].sum()
                if h > 5 * lh or strokes < lh * lh * 0.15 or h >= 0.9 * (Y1 - Y0) and w < 0.5 * h:
                    continue  # (a full-height thin strip is the paper edge / a scan shadow, not pen strokes)
                if X0 + x + w >= W - 0.02 * W and w < lh:
                    continue
                comps.append((x, y, w, h, strokes))
            if os.environ.get("CP_DEBUG_SIG"):
                print("window", (X0, Y0, X1, Y1), "lh", lh, [(X0 + c[0], Y0 + c[1], c[2], c[3], int(c[4])) for c in comps])
            for x, y, w, h, strokes in comps:
                if w < lh * 1.5 or h < lh * 0.5 or strokes < lh * lh * 0.4:
                    continue
                dist = abs((Y0 + y + h / 2) - (y0 + y1) / 2) + abs((X0 + x) - x0) * 0.3
                if best is None or dist < best[0]:
                    # a signature of several words ('Cole Porter') is several stroke clusters on one baseline: grow
                    # the box over neighbouring clusters that share its band and sit within a word gap
                    gx0, gy0, gx1, gy1 = x, y, x + w, y + h
                    grown = len(comps) <= 30  # busy guilloche / security print: no growth, it would chain the pattern
                    while grown:
                        grown = False
                        for cx_, cy_, cw_, ch_, _ in comps:
                            if gx0 <= cx_ and cx_ + cw_ <= gx1 and gy0 <= cy_ and cy_ + ch_ <= gy1:
                                continue
                            ov = min(gy1, cy_ + ch_) - max(gy0, cy_)
                            gap = max(cx_ - gx1, gx0 - (cx_ + cw_), 0)
                            if ov >= 0.3 * min(ch_, gy1 - gy0) and gap <= 2.5 * lh and ch_ <= 2.5 * (gy1 - gy0) + lh and \
                                    max(gx1, cx_ + cw_) - min(gx0, cx_) <= 16 * lh:
                                gx0, gy0, gx1, gy1 = min(gx0, cx_), min(gy0, cy_), max(gx1, cx_ + cw_), max(gy1, cy_ + ch_)
                                grown = True
                    best = (dist, (X0 + gx0 - 4, Y0 + gy0 - 4, X0 + gx1 + 4, Y0 + gy1 + 4))
        if best:
            bb = best[1]
            if any(_iou(quad_bbox(r.quad), bb) > 0.3 for r in out if r.kind == "signature"):
                continue
            out = [r for r in out if not (_iou(quad_bbox(r.quad), bb) > 0.3)]
            rb = tuple(v / sc for v in bb)
            out.append(Region(f"{doc_id}:p{page.index}:R{len(out)}", "signature", quad_from_rect(*rb),
                              f"handwritten strokes next to the printed label '{l.text.strip()[:30]}'", None))
    # re-number ids to stay unique and stable
    for i, r in enumerate(out):
        r.id = f"{doc_id}:p{page.index}:R{i}"
    return out


def _contains(bb, pt):
    return bb[0] <= pt[0] <= bb[2] and bb[1] <= pt[1] <= bb[3]
