"""Constrained editing engine. Builds FRESH output files; edits only reviewed regions.

- Native PDF text: true redaction (glyphs removed from the content stream) + replacement drawn at the original
  baseline with a matched base-14 font, fitted to the original width. Hidden text layers are removed.
- Text that exists only in pixels (stamps, scans, photos, images): paint the region with locally sampled
  background, render replacement text sized and perspective-warped into the original text quad.
- Signatures, faces, QR/barcodes: replaced in place by casepatch.docs.native (synthetic signature/face, regenerated
  symbol with mapped payload). No visible redaction boxes or labels are ever drawn; failures are removed by inpainting
  and flagged for review. Photos/graphics without a face are kept and flagged.
- Output PDFs: fresh document, metadata/XMP/annotations/attachments/JS/links stripped. Images: no EXIF/GPS.
Never regenerates a page with a generative model."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np

from .model import quad_bbox, span_quad

class _RasterFonts(dict):
    """Raster replacement faces by key, resolved on first use for this OS (fonts.resolve: system metric-compatible faces,
    else the bundled Liberation fonts)."""
    KEYS = ("sans", "sans-bold", "serif", "serif-bold", "mono", "mono-bold")

    def __missing__(self, key):
        from . import fonts
        if key not in self.KEYS:
            raise KeyError(key)
        cls, _, b = key.partition("-")
        self[key] = path = fonts.resolve(cls, bold=b == "bold")
        return path

    def items(self):
        return [(k, self[k]) for k in self.KEYS]


RASTER_FONTS = _RasterFonts()


def _font_cache():
    from ..paths import cache_dir
    return cache_dir() / "fonts"


def _pdf_font(name):
    n = (name or "").lower()
    bold = "bold" in n or "black" in n
    if "times" in n or "serif" in n and "sans" not in n or "roman" in n:
        return "tibo" if bold else "tiro"
    if "courier" in n or "mono" in n:
        return "cobo" if bold else "cour"
    return "hebo" if bold else "helv"


def _color(c):
    if isinstance(c, (tuple, list)):
        return tuple(float(x) for x in c[:3]) if len(c) >= 3 else (float(c[0]),) * 3
    if isinstance(c, int):
        return ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
    return (0, 0, 0)


def _sample_colors(img, bbox, pad=6):
    """Background = median of bright pixels in a ring around bbox; ink = median of darkest pixels inside."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    ring = img[Y0:Y1, X0:X1].reshape(-1, 3).astype(np.float32)
    inner = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)].reshape(-1, 3).astype(np.float32)
    if ring.size == 0 or inner.size == 0:
        return np.array([255, 255, 255.]), np.array([0, 0, 0.]), 0.0
    lum = ring.mean(1)
    bgpx = ring[lum >= np.percentile(lum, 60)]
    ilum = inner.mean(1)
    ink = inner[ilum <= np.percentile(ilum, 8)]
    return np.median(bgpx, 0), np.median(ink, 0), float(bgpx.std(0).mean())


def _ink_profile(binary):
    rows = np.where(binary.any(axis=1))[0]; cols = np.where(binary.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1, float(binary.mean())


def match_font(rect_crop, text):
    """Pick font family/weight and size so the ORIGINAL text re-rendered matches the original ink box and density.
    rect_crop: rectified grayscale crop of the original text quad."""
    from PIL import Image, ImageDraw, ImageFont
    H, W = rect_crop.shape
    thr = min(200, (float(rect_crop.max()) + float(rect_crop.min())) / 2)
    ink = rect_crop < thr
    prof = _ink_profile(ink)
    if prof is None:
        return None
    r0, r1, c0, c1, _ = prof
    ink_w, ink_h = c1 - c0, r1 - r0
    density = float(ink[r0:r1, c0:c1].mean())
    best = None
    for key, path in RASTER_FONTS.items():
        f = ImageFont.truetype(path, 100)
        bb = f.getbbox(text)
        if bb[3] - bb[1] <= 0:
            continue
        size = max(6, int(round(100 * ink_h / (bb[3] - bb[1]))))
        f = ImageFont.truetype(path, size)
        bb = f.getbbox(text)
        im = Image.new("L", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), 0)
        ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), text, font=f, fill=255)
        a = np.asarray(im) > 127
        p2 = _ink_profile(a)
        if p2 is None:
            continue
        d2 = float(a[p2[0]:p2[1], p2[2]:p2[3]].mean())
        score = abs(np.log(max(p2[3] - p2[2], 1) / max(ink_w, 1))) + 1.5 * abs(d2 - density)
        if best is None or score < best[0]:
            best = (score, key, size, r0, c0)
    return best


def _pick_font(text, height, width):
    from PIL import ImageFont
    best = None
    for key, path in RASTER_FONTS.items():
        size = max(6, int(height * 0.82))
        f = ImageFont.truetype(path, size)
        w = f.getlength(text)
        score = abs(math.log(max(w, 1) / max(width, 1)))
        if best is None or score < best[0]:
            best = (score, key, size)
    return best[1], best[2]


def _text_patch(text, font_key, size, height):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(RASTER_FONTS[font_key], size)
    w = int(math.ceil(f.getlength(text))) + 2
    im = Image.new("L", (max(w, 4), int(height)), 0)
    asc, desc = f.getmetrics()
    y = (height - (asc + desc)) / 2 + (desc * 0.15)
    ImageDraw.Draw(im).text((1, y), text, font=f, fill=255)
    return np.asarray(im, np.float32) / 255.0


def _text_patch_aligned(text, font_key, size, height, top, left):
    """Render text so its ink top/left match the original ink top/left inside the rectified quad."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(RASTER_FONTS[font_key], size)
    bb = f.getbbox(text)
    w = int(left + bb[2] - bb[0] + 3)
    im = Image.new("L", (max(w, 4), int(height)), 0)
    ImageDraw.Draw(im).text((left - bb[0], top - bb[1]), text, font=f, fill=255)
    return np.asarray(im, np.float32) / 255.0


def raster_replace(img, quad, original, replacement, rng):
    """Paint over `quad` and render `replacement` warped into it. Returns edited bbox."""
    import cv2
    q = np.array(quad, np.float32)
    tl, tr, br, bl = q
    width = float(np.linalg.norm(tr - tl)); height = float(np.linalg.norm(bl - tl))
    if width < 3 or height < 4:
        return None
    # grow quad slightly to cover anti-aliasing
    ux = (tr - tl) / max(width, 1e-6); uy = (bl - tl) / max(height, 1e-6)
    gx, gy = max(3.0, height * 0.18), height * 0.12
    g = np.array([tl - ux * gx - uy * gy, tr + ux * gx - uy * gy, br + ux * gx + uy * gy, bl - ux * gx + uy * gy], np.float32)
    bbox = quad_bbox(g)
    bg, ink, noise = _sample_colors(img, bbox)
    # rectify original text area for font matching (before painting over it)
    Wr, Hr = int(max(width, 4)), int(max(height, 4))
    Minv = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [Wr, 0], [Wr, Hr], [0, Hr]]))
    rect = cv2.warpPerspective(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), Minv, (Wr, Hr), borderMode=cv2.BORDER_REPLICATE)
    fm = match_font(rect, original)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [g.astype(np.int32)], 255)
    fill = np.clip(bg[None, None, :] + rng.normal(0, max(noise * 0.6, 0.5), img.shape), 0, 255)
    img[mask > 0] = fill[mask > 0].astype(np.uint8)
    if fm:
        _, font_key, size, top, left = fm
        patch = _text_patch_aligned(replacement, font_key, size, Hr, top, left)
    else:
        font_key, size = _pick_font(original, height, width)
        patch = _text_patch(replacement, font_key, size, height)
    pw = patch.shape[1]
    height = float(patch.shape[0])
    target_w = min(pw, width + gx * 0.8)  # compress only if longer; otherwise keep natural width
    src = np.float32([[0, 0], [pw, 0], [pw, height], [0, height]])
    end = tl + ux * target_w
    dst = np.float32([tl, end, end + (bl - tl) + (br - bl - (tr - tl)) * (target_w / max(width, 1)), bl])
    M = cv2.getPerspectiveTransform(src, dst)
    alpha = cv2.warpPerspective(patch, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)[..., None]
    img[:] = np.clip(img * (1 - alpha) + ink[None, None, :] * alpha, 0, 255).astype(np.uint8)
    return [round(float(v), 1) for v in bbox]


def _line_index(doc):
    return {l.id: (p, l) for p in doc.pages for l in p.lines}


def render_document(doc, occs, regions, out_path, seed=0, ctx=None):
    """Returns edit log. `occs` already filtered to accepted ones (with .replacement); `regions` to withheld ones."""
    rng = np.random.default_rng(seed)
    if doc.kind == "json":
        return _render_json(doc, occs, out_path)
    if doc.kind == "image":
        return _render_image(doc, occs, regions, out_path, rng, ctx)
    return _render_pdf(doc, occs, regions, out_path, rng, ctx)


def _render_json(doc, occs, out_path):
    lines = _line_index(doc)
    data = json.loads(json.dumps(doc.json))
    edits, log = {}, []
    for o in occs:
        edits.setdefault(o.line_id, []).append(o)
    for lid, os_ in edits.items():
        _, l = lines[lid]
        text = l.text
        for o in sorted(os_, key=lambda o: -o.start):
            text = text[:o.start] + o.replacement + text[o.end:]
            log.append({"occurrence": o.id, "kind": o.kind, "method": "json_string_rewrite", "path_depth": len(l.path)})
        if not l.path:  # the whole document is one JSON string
            data = text
            continue
        node = data
        for part in l.path[:-1]:
            node = node[part]
        node[l.path[-1]] = text
    from .textual import serialize
    fmt = (doc.textual or {}).get("format", "json")
    Path(out_path).write_bytes(serialize(fmt, data, doc.textual or {}).encode("utf-8"))
    return log


RETYPE_CROSSED_WORDS = False  # retyping printed words under leftover strokes doubled glyphs on case-real-02; off until fixed


def _label_protect(shape, lq, line_id, ctx, only=None):
    """Pixels of other text lines (printed labels) that the edited line overlaps: kept as they are."""
    boxes = ctx.get("_line_boxes") or {}
    if only is not None:
        boxes = {k: v for k, v in boxes.items() if k in only}
    x0, y0, x1, y1 = lq[:, 0].min(), lq[:, 1].min(), lq[:, 0].max(), lq[:, 1].max()
    m = None
    for lid, (a0, b0, a1, b1) in boxes.items():
        if lid == line_id:
            continue
        ix, iy = min(x1, a1) - max(x0, a0), min(y1, b1) - max(y0, b0)
        if ix <= 0 or iy <= 0:
            continue
        inter = ix * iy
        if inter > 0.5 * min((x1 - x0) * (y1 - y0), (a1 - a0) * (b1 - b0)):
            continue  # duplicate/containing OCR box, not a neighbouring label
        if m is None:
            m = np.zeros(shape, bool)
        m[int(b0):int(np.ceil(b1)) + 1, int(a0):int(np.ceil(a1)) + 1] = True
    return m


def span_fraction(chars, start, end):
    """Where chars[start:end] sits along its line, as fractions of the line length, from the OCR word boxes. A
    proportional estimate over a whole line ('Signed: Dr. Anand Okafor, MD NPI ...') lands inside the next word and
    erased ', M' of ', MD' (A10b/D11d)."""
    if not chars or not (0 <= start < end <= len(chars)):
        return None
    p0 = np.array(chars[0].quad[0], np.float64); ax = np.array(chars[-1].quad[1], np.float64) - p0
    L2 = float(ax @ ax)
    if L2 < 1:
        return None
    f0 = float((np.array(chars[start].quad[0]) - p0) @ ax) / L2
    f1 = float((np.array(chars[end - 1].quad[1]) - p0) @ ax) / L2
    if not (-0.01 <= f0 < f1 <= 1.01):
        return None
    return (max(0.0, f0), min(1.0, f1))


def _replace_text_any_orientation(img, lq, text, o, rng, ctx, cache, orig):
    """Rotated text (vertical margin numbers, upside-down stamps): edit it in an upright copy of its neighbourhood, so
    the replacement keeps the source's orientation, then write the pixels back."""
    from . import native as N
    d = lq[1] - lq[0]
    ang = np.degrees(np.arctan2(d[1], d[0]))
    k = 0 if abs(ang) <= 45 else 1 if 45 < ang <= 135 else 3 if -135 <= ang < -45 else 2
    bw, bh = float(np.ptp(lq[:, 0])), float(np.ptp(lq[:, 1]))
    vertical_box = k == 0 and len(text.strip()) >= 3 and bh > 1.5 * bw
    if vertical_box:
        # OCR gave an axis box for vertical text: find the reading direction by re-reading both upright rotations
        import difflib
        from .extract import ocr_engine
        x0, y0, x1, y1 = int(lq[:, 0].min()), int(lq[:, 1].min()), int(np.ceil(lq[:, 0].max())), int(np.ceil(lq[:, 1].max()))
        crop = img[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
        best = (-1, 0.0, 0)
        for kk in (1, 3):
            eng = ocr_engine(); saved = (eng.use_det, eng.use_cls, eng.use_rec)
            try:  # recognition only, no angle classifier: an upside-down read scores lower
                res = eng(np.ascontiguousarray(np.rot90(crop, kk)), use_det=False, use_cls=False, use_rec=True)
                got = "".join(res.txts) if res is not None and res.txts else ""
                conf = float(np.mean(res.scores)) if got else 0.0
            except Exception:
                got, conf = "", 0.0
            finally:  # the engine keeps per-call flags; restore them for everyone else
                eng.use_det, eng.use_cls, eng.use_rec = saved
            best = max(best, (round(difflib.SequenceMatcher(None, got, text).ratio(), 3), conf, kk))
        k = best[2] if best[0] > 0.5 else 0
        if k:
            lq = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    if k == 0:
        return N.replace_text(img, lq, text, o.start, o.end, o.replacement, rng, jpeg_q=ctx.get("jpeg_q"), style_cache=cache, orig_img=orig,
                              protect=_label_protect(img.shape[:2], lq, o.line_id, ctx) if N.typewriter_page(orig, cache)
                              else _label_protect(img.shape[:2], lq, o.line_id, ctx, only=ctx.get("_caption_ids", set())),
                              line_conf=(ctx.get("_line_conf") or {}).get(o.line_id), span_frac=ctx.get("_span_frac"))
    H, W = img.shape[:2]
    thick = min(float(np.ptp(lq[:, 0])), float(np.ptp(lq[:, 1]))) if vertical_box else float(np.linalg.norm(lq[3] - lq[0]))
    pad = int(thick * 3 + 10)
    x0, y0 = max(0, int(lq[:, 0].min()) - pad), max(0, int(lq[:, 1].min()) - pad)
    x1, y1 = min(W, int(np.ceil(lq[:, 0].max())) + pad), min(H, int(np.ceil(lq[:, 1].max())) + pad)
    sub, osub = img[y0:y1, x0:x1], orig[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    rs, ro = np.ascontiguousarray(np.rot90(sub, k)), np.ascontiguousarray(np.rot90(osub, k))
    q = lq - [x0, y0]
    if vertical_box:
        # rotate the axis box corners, then re-order as an upright tl,tr,br,bl rectangle in the rotated frame
        if k == 1:
            rq = np.column_stack([q[:, 1], w - 1 - q[:, 0]])
        else:
            rq = np.column_stack([h - 1 - q[:, 1], q[:, 0]])
        a0, b0, a1, b1 = rq[:, 0].min(), rq[:, 1].min(), rq[:, 0].max(), rq[:, 1].max()
        q = np.array([[a0, b0], [a1, b0], [a1, b1], [a0, b1]], np.float32)
    elif k == 1:
        q = np.column_stack([q[:, 1], w - 1 - q[:, 0]])
    elif k == 3:
        q = np.column_stack([h - 1 - q[:, 1], q[:, 0]])
    else:
        q = np.column_stack([w - 1 - q[:, 0], h - 1 - q[:, 1]])
    region, info = N.replace_text(rs, q.astype(np.float32), text, o.start, o.end, o.replacement, rng, jpeg_q=ctx.get("jpeg_q"),
                                  style_cache={}, orig_img=ro)
    if region is None:
        return None, info
    img[y0:y1, x0:x1] = np.rot90(rs, -k)
    full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = np.rot90(region, -k)
    info["rotated_text_deg"] = int(90 * k)
    return full, info


def _apply_raster(page, occs, regions, rng, coord_scale=1.0, ctx=None):
    """Edit page.image in place with the native in-place engine. coord_scale converts quad coords to pixels.
    ctx: {"jpeg_q": int|None, "payload_map": {orig: repl}, "signer": {region_id: fake name}, "seed": str, "orig": ndarray}"""
    from . import native as N
    ctx = ctx or {}
    lines = {l.id: l for l in page.lines}
    img = page.image
    orig = ctx.get("orig") if ctx.get("orig") is not None else img.copy()
    log, cache = [], {}
    ctx["_line_boxes"] = {}
    ctx["_line_conf"] = {l_.id: l_.conf for l_ in lines.values()}
    for l_ in lines.values():
        if l_.chars and sum(c_.isalnum() for c_ in l_.text) >= 2:
            q_ = np.array([pt for ch in l_.chars for pt in ch.quad], np.float32) * coord_scale
            if abs(np.degrees(np.arctan2(*(q_[1] - q_[0])[::-1]))) < 3:  # axis boxes only describe unrotated lines
                ctx["_line_boxes"][l_.id] = (q_[:, 0].min(), q_[:, 1].min(), q_[:, 0].max(), q_[:, 1].max())
                if re.fullmatch(r"\s*\([^()]{2,30}\)\W*", l_.text):
                    ctx.setdefault("_caption_ids", set()).add(l_.id)  # printed field captions: '(Given name)'
    if "_page_style" not in cache:
        probe, seen_ = [], set()
        # the page's own printed lines (longest, confidently read) plus the edited ones: the edited values alone are
        # often a few digits or asterisks, which do not identify a typeface
        cand = sorted((l for l in lines.values() if l.chars and getattr(l, "conf", 1.0) >= 0.85 and len(l.text.strip()) >= 8
                       and sum(c.isalpha() for c in l.text) >= 5), key=lambda l: -len(l.text))[:6]
        cand += [lines[o.line_id] for o in occs if o.line_id in lines]
        for l in cand:
            if l is None or not l.chars or l.id in seen_ or getattr(l, "conf", 1.0) < 0.85:
                continue
            seen_.add(l.id)
            probe.append((np.array((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]), np.float32) * coord_scale, l.text))
        try:
            cache["_page_style"] = N.page_style(orig, probe[:10]) if len(probe) >= 2 else None
        except Exception:
            cache["_page_style"] = None
    for o in sorted(occs, key=lambda o: (o.line_id, o.start)):
        l = lines.get(o.line_id)
        if not l or not l.chars:
            continue
        lq = np.array((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]), np.float32) * coord_scale
        m_ = re.fullmatch(r"([A-Za-z])(\d[\d\-/ ]{5,})", o.text)
        if m_ and o.end - o.start == len(o.text) and o.replacement[:1] == m_.group(1):
            # a lone letter glued to a long number is OCR bleed from an overlapping printed label (e.g. "Phone:"):
            # leave the label's pixels alone and replace only the number
            import copy as _copy
            o = _copy.copy(o); o.start += 1; o.text = o.text[1:]; o.replacement = o.replacement[1:]
        ctx["_span_frac"] = span_fraction(l.chars, o.start, o.end)
        region, info = _replace_text_any_orientation(img, lq, l.text, o, rng, ctx, cache, orig)
        if region is not None:
            ys, xs = np.where(region)
            bb = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            log.append({"occurrence": o.id, "kind": o.kind, "page": page.index, "bbox_px": bb, "method": "native_inpaint_restyle",
                        "style": info, "replacement_len": len(o.replacement)})
        else:
            q = np.array(span_quad(l.chars, o.start, o.end), np.float32) * coord_scale
            bb = raster_replace(img, q, o.text, o.replacement, rng)
            if bb:
                log.append({"occurrence": o.id, "kind": o.kind, "page": page.index, "bbox_px": bb, "method": "raster_patch_rewrite_fallback",
                            "fallback_reason": info.get("error"), "replacement_len": len(o.replacement)})
    # portraits largest first: secondary/ghost portraits on the same page are then made from the new main portrait
    def _farea(r):
        b_ = quad_bbox(r.quad)
        return -(b_[2] - b_[0]) * (b_[3] - b_[1]) if r.kind == "face" else 0
    regions = sorted(regions, key=lambda r: (r.kind != "face", _farea(r)))
    faces_here = [r for r in regions if r.kind == "face" and getattr(r, "decision", "replace") == "replace"]
    if len(faces_here) > 1 and ctx is not None:
        prof = ctx.setdefault("face_profile", {}).setdefault(f"p{page.index}", {})
        fseed0 = (ctx["face_seed"] + f":p{page.index}") if ctx.get("face_seed") else ctx.get("seed", "") + "identity:" + str(prof.get("identity", f"p{page.index}"))
        tile, trials = N.choose_portrait_tile(img, [[float(v) * coord_scale for v in quad_bbox(r.quad)] for r in faces_here], prof, fseed0)
        if tile is not None:
            prof["tile"] = tile
        ctx.setdefault("_portrait_trials", {})[f"p{page.index}"] = trials
    for r in regions:
        bb = [float(v) * coord_scale for v in quad_bbox(r.quad)]
        decision = getattr(r, "decision", "replace")
        seed = ctx.get("seed", "") + r.id
        region, info, method = None, {}, None
        if decision == "replace":
            if r.kind == "face":
                prof = ctx.get("face_profile", {}).get(f"p{page.index}") or {}
                fseed = (ctx["face_seed"] + f":p{page.index}") if ctx.get("face_seed") else \
                    ctx.get("seed", "") + "identity:" + str(prof.get("identity", f"p{page.index}"))
                # faces are processed largest first: any later face on this page is a secondary/ghost portrait
                prev = [e for e in ctx.get("_face_embs", {}).get(f"p{page.index}", []) if e is not None]
                region, info = N.swap_portrait(img, bb, fseed, rng, ctx.get("jpeg_q"), prof, secondary=bool(prev),
                                               primary_emb=prev[0] if prev else None); method = "synthetic_face_swap"
                if region is not None:
                    ctx.setdefault("_face_embs", {}).setdefault(f"p{page.index}", []).append(info.pop("_emb", None))
            elif r.kind == "signature":
                name = ctx.get("signer", {}).get(r.id) or ctx.get("default_signer") or "A. Morgan"
                name = name.title() if name.isupper() else name  # people sign in mixed case
                # printed text near/inside the box must survive the swap (it is not part of the signature)
                printed = np.zeros(img.shape[:2], bool)
                for l_ in lines.values():
                    if not l_.chars or sum(ch.isalnum() for ch in l_.text) < 3:
                        continue
                    lq_ = np.array([pt for ch in l_.chars for pt in ch.quad], np.float32) * coord_scale
                    cx_, cy_ = lq_[:, 0].mean(), lq_[:, 1].mean()
                    inside = bb[0] <= cx_ <= bb[2] and bb[1] <= cy_ <= bb[3]
                    if inside and (getattr(r, "rotated", False) or l_.conf < 0.9):
                        continue
                    lx0, ly0 = np.floor(lq_.min(0)).astype(int); lx1, ly1 = np.ceil(lq_.max(0)).astype(int)
                    printed[max(0, ly0 - 1):ly1 + 2, max(0, lx0 - 1):lx1 + 2] = True
                # the original signature's full ink: stroke pixels outside printed text, grown through connected ink
                # (minus rules/bars) so strokes running over printed words are found too
                import cv2 as _cv2
                H_, W_ = img.shape[:2]
                bx0, by0, bx1, by1 = [int(round(v)) for v in bb]
                ey0 = max(0, by0 - (by1 - by0)); ex0 = max(0, bx0 - 10); ex1 = min(W_, bx1 + 10); ey1 = min(H_, by1 + 10)
                gwin = _cv2.cvtColor(orig[ey0:ey1, ex0:ex1], _cv2.COLOR_RGB2GRAY)
                dwin = (gwin < 140).astype(np.uint8)
                bars_ = (_cv2.morphologyEx(dwin, _cv2.MORPH_OPEN, np.ones((1, 60), np.uint8)) | _cv2.morphologyEx(dwin, _cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))) > 0
                dwin2 = dwin.astype(bool) & ~_cv2.dilate(bars_.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                seeds = dwin2 & ~printed[ey0:ey1, ex0:ex1]
                seeds[:max(0, by0 - ey0), :] = False
                n_, lab_ = _cv2.connectedComponents(dwin2.astype(np.uint8), connectivity=8)
                keep_ids = np.unique(lab_[seeds]); keep_ids = keep_ids[keep_ids > 0]
                sigmask = np.zeros((H_, W_), bool)
                if len(keep_ids):
                    sigmask[ey0:ey1, ex0:ex1] = np.isin(lab_, keep_ids)
                tb = None
                if sigmask.any():
                    ys_, xs_ = np.where(sigmask)
                    tb = (float(np.percentile(xs_, 1)), float(np.percentile(ys_, 1)), float(np.percentile(xs_, 99)), float(np.percentile(ys_, 99)))
                region, info = N.replace_signature(img, bb, name, seed, rng, ctx.get("jpeg_q"), exclude=printed, target_box=tb,
                                                   quad=np.array(r.quad, np.float32) * coord_scale if getattr(r, "rotated", False) else None); method = "synthetic_signature"
                er_ = info.pop("_erase", None) if isinstance(info, dict) else None
                if region is not None and er_ is not None and er_.any():
                    # strokes that crossed printed words were left alone above; retype those words in place (erase the
                    # word box incl. the crossing strokes, redraw the same text in its own matched style)
                    near = _cv2.dilate((sigmask & printed).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    rcache = {"_typewriter": False}
                    redone = []
                    for l_ in lines.values():
                        if not l_.chars or l_.conf < 0.85:
                            continue
                        lq_ = np.array((l_.chars[0].quad[0], l_.chars[-1].quad[1], l_.chars[-1].quad[2], l_.chars[0].quad[3]), np.float32) * coord_scale
                        for wm in re.finditer(r"\S+", l_.text):
                            if sum(ch.isalnum() for ch in wm.group(0)) < 2:
                                continue
                            wq = np.array(span_quad(l_.chars, wm.start(), wm.end()), np.float32) * coord_scale
                            wx0, wy0 = np.floor(wq.min(0)).astype(int); wx1, wy1 = np.ceil(wq.max(0)).astype(int)
                            if RETYPE_CROSSED_WORDS and near[max(0, wy0):wy1, max(0, wx0):wx1].any() and printed[max(0, wy0):wy1, max(0, wx0):wx1].any():
                                r2, i2 = N.replace_text(img, lq_, l_.text, wm.start(), wm.end(), wm.group(0), rng, jpeg_q=ctx.get("jpeg_q"),
                                                        style_cache=rcache, orig_img=orig)
                                if r2 is not None:
                                    region = region | r2
                                    redone.append(wm.group(0))
                    if redone:
                        info["printed_words_retyped"] = redone
            elif r.kind in ("qr", "barcode"):
                region, info = N.replace_code(img, bb, ctx.get("payload_map", {}), seed, rng, ctx.get("jpeg_q")); method = "symbol_regenerated"
                if region is None and info.get("kept"):
                    log.append({"region": r.id, "kind": r.kind, "page": page.index, "bbox_px": bb, "method": "symbol_kept_no_sensitive_payload", **info})
                    continue
        if region is None and decision in ("replace", "erase") and r.kind in ("face", "signature", "qr", "barcode", "photo") and \
                not (decision == "replace" and r.kind == "photo"):
            # could not produce a realistic replacement: remove the content by inpainting (no label, no box) and flag it
            import cv2
            x0, y0, x1, y1 = [int(round(v)) for v in bb]
            m = np.zeros(img.shape[:2], np.uint8)
            sub = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            ink = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
            m[y0:y1, x0:x1] = 255 if r.kind in ("face", "photo", "qr") else cv2.dilate(ink, np.ones((5, 5), np.uint8))
            N.inpaint(img, m)
            log.append({"region": r.id, "kind": r.kind, "page": page.index, "bbox_px": bb, "method": "content_removed_by_inpainting",
                        "reason": info.get("error", decision), "needs_review": True})
            continue
        if region is None:
            log.append({"region": r.id, "kind": r.kind, "page": page.index, "bbox_px": bb, "method": "kept_unchanged",
                        "reason": info.get("error", "decision=" + decision), "needs_review": r.kind in ("face", "signature", "qr", "barcode", "photo")})
            continue
        ys, xs = np.where(region)
        log.append({"region": r.id, "kind": r.kind, "page": page.index, "bbox_px": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                    "method": method, "style": info})
    return log


def _render_image(doc, occs, regions, out_path, rng, ctx=None):
    from PIL import Image
    from .native import jpeg_quality
    page = doc.pages[0]
    page.out_image = page.image.copy()
    orig, page.image = page.image, page.out_image
    q = jpeg_quality(doc.path)
    ctx = dict(ctx or {}, jpeg_q=q, orig=orig)
    log = _apply_raster(page, [o for o in occs if o.page == 0], [r for r in regions], rng, ctx=ctx)
    page.image = orig
    im = Image.fromarray(page.out_image)  # fresh image object: no EXIF, GPS, XMP or ICC text
    if doc.ext in (".jpg", ".jpeg"):
        im.save(out_path, "JPEG", quality=q or 92)
    else:
        im.save(out_path, "PNG")
    log.append({"method": "metadata_stripped", "removed": sorted(doc.hidden.get("exif", {}).keys())})
    return log


def _native_scan(src, index):
    """(pixels, bbox) of the single full-page scan image on a raster PDF page, at its own resolution, else (None, None)."""
    import pymupdf
    pg = src[index]
    ims = pg.get_images(full=True)
    if len(ims) != 1:
        return None, None
    try:
        bb = pg.get_image_bbox(ims[0])
        if bb.is_empty or bb.get_area() < 0.95 * pg.rect.get_area():
            return None, None
        pix = pymupdf.Pixmap(src, ims[0][0])
        if pix.alpha or pix.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()
        return (a[..., 0] if pix.n == 1 else a), bb
    except Exception:
        return None, None


def _merge_into_native(native, before_render, after_render):
    import cv2
    nh, nw = native.shape[:2]
    diff = (np.abs(before_render.astype(np.int16) - after_render.astype(np.int16)).max(axis=2) > 0).astype(np.float32)
    g = native if native.ndim == 2 else native.max(axis=2)
    if _is_bilevel(g):
        # thresholded (fax/typewriter) scan: new ink is thresholded the same way at render resolution, with the ragged
        # edges of scanner ink, then resampled like the source; no grey anti-aliased type
        after_render = after_render.copy()
        dm = cv2.dilate((diff > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        e1 = after_render.min(axis=2).astype(np.float32)
        rng = np.random.default_rng(int(dm.sum()) % (2 ** 31))
        noise = cv2.GaussianBlur(rng.normal(0, 1, e1.shape).astype(np.float32), (0, 0), 1.0)
        lo, hi = float(np.percentile(g, 0.5)), float(np.percentile(g, 99.5))
        # threshold each edited area halfway between paper and its own darkest new ink (thin grey signature strokes
        # survive, anti-aliased black type does not get fattened), with ragged-edge noise scaled to that contrast
        thr = np.full(e1.shape, 128.0, np.float32); amp = np.full(e1.shape, 25.0, np.float32)
        newink = dm & (e1 < before_render.min(axis=2).astype(np.float32) - 20)  # pixels the edit darkened = new ink
        n_, lab = cv2.connectedComponents(cv2.dilate(dm.astype(np.uint8), np.ones((15, 15), np.uint8)))
        for k_ in range(1, n_):
            sel = lab == k_
            vals = e1[sel & newink]
            if vals.size > 10:
                p5 = float(np.percentile(vals, 5)); paper = float(np.percentile(e1[sel & dm], 90))
                thr[sel] = (p5 + paper) / 2; amp[sel] = 0.18 * max(10.0, paper - p5)
        bw = np.where(e1 + noise * amp < thr, lo, hi).astype(np.uint8)
        bw = cv2.medianBlur(bw, 3)
        after_render[dm] = bw[dm][:, None]
    m = cv2.resize(diff, (nw, nh), interpolation=cv2.INTER_AREA) > 0.02
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ed = cv2.resize(after_render, (nw, nh), interpolation=cv2.INTER_AREA)
    if native.ndim == 2:
        ed = cv2.cvtColor(ed, cv2.COLOR_RGB2GRAY)
    out = native.copy()
    out[m] = ed[m]
    return out, int(m.sum())


def _is_bilevel(g):
    """Scan is thresholded black/white (fax/typewriter-style), possibly resampled: almost no mid-tones."""
    if g.size == 0:
        return False
    return bool(np.mean((g > 40) & (g < 215)) < 0.1 and np.mean(g < 40) > 0.01)


class _PageFonts:
    """Embedded-font reuse: the source page's own font program is re-embedded when it contains every glyph needed;
    otherwise the closest installed open font is used (name-family hints); base-14 only as a last resort."""

    def __init__(self, src, page):
        self.src, self.page, self.cache, self.alias = src, page, {}, {}
        self.byname = {}
        self.subset = {}
        for f in page.get_fonts():
            self.byname.setdefault(f[3].split("+")[-1], f[0])
            if "+" in f[3]:
                self.subset[f[3].split("+")[-1]] = True
        # glyphs a subset font is known to contain = characters the page actually draws with it (subset cmaps can
        # claim glyphs whose outlines were dropped)
        self.used = {}
        try:
            import pymupdf
            orig_page = pymupdf.open(src.name)[page.number] if src.name else page  # pre-redaction text of this page
            for b in orig_page.get_text("rawdict")["blocks"]:
                for l in b.get("lines", []):
                    for sp in l["spans"]:
                        self.used.setdefault(sp["font"].split("+")[-1], set()).update(c["c"] for c in sp["chars"])
        except Exception:
            pass

    def font_for(self, fontname, text):
        import pymupdf
        base = (fontname or "").split("+")[-1]
        key = (base, text)
        if key in self.cache:
            return self.cache[key]
        res = None
        xref = self.byname.get(base)
        if xref:
            try:
                _b, _ext, _t, buf = self.src.extract_font(xref)
                if buf:
                    f = pymupdf.Font(fontbuffer=buf)
                    used = self.used.get(base, set())
                    if all(ch == " " or (f.has_glyph(ord(ch)) and (ch in used or not self.subset.get(base))) for ch in text):
                        alias = self.alias.get(xref)
                        if alias is None:
                            alias = f"CPE{xref}"
                            self.page.insert_font(fontname=alias, fontbuffer=buf)
                            self.alias[xref] = alias
                        res = ("embedded:" + base, alias, f)
            except Exception:
                res = None
        if res is None:
            path = _closest_installed(base)
            if path:
                alias = self.alias.get(path)
                if alias is None:
                    alias = "CPI" + hashlib.sha1(path.encode()).hexdigest()[:8]
                    self.page.insert_font(fontname=alias, fontfile=path)
                    self.alias[path] = alias
                res = ("installed:" + Path(path).stem, alias, pymupdf.Font(fontfile=path))
        if res is None:
            n = _pdf_font(fontname)
            res = ("base14:" + n, n, pymupdf.Font(n))
        self.cache[key] = res
        return res


_NAME_HINTS = [("avenir", "Nunito Sans"), ("futura", "Mulish"), ("helvetica", "Arimo"), ("arial", "Arimo"), ("idealsans", "Montserrat"),
               ("gotham", "Montserrat"), ("times", "Tinos"), ("georgia", "PT Serif"), ("garamond", "Libre Baskerville"),
               ("courier", "Cousine"), ("calibri", "Carlito"), ("verdana", "DejaVu Sans"), ("frutiger", "Open Sans"),
               ("myriad", "Open Sans"), ("minion", "PT Serif"), ("roboto", "Roboto"), ("lato", "Lato"), ("liberation", "Liberation Sans")]


def _weight_from_name(b):
    for k, w in (("thin", 100), ("extralight", 200), ("ultralight", 200), ("light", 300), ("medium", 500), ("demibold", 600),
                 ("semibold", 600), ("demi", 600), ("extrabold", 800), ("heavy", 800), ("black", 900), ("bold", 700)):
        if k in b:
            return w
    return 400


def _static_instance(path, weight):
    """Variable fonts embed their default instance (often ExtraLight) in a PDF: pin the wght axis to the source
    font's weight with fontTools (optional dependency) and cache the static instance outside the repo."""
    if not path or "variable" not in Path(path).name.lower():
        return path
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
    except ImportError:
        return path
    out = _font_cache() / f"{Path(path).stem}-w{weight}.ttf"
    if out.exists():
        return str(out)
    try:
        f = TTFont(path)
        axes = {a.axisTag: a for a in f["fvar"].axes}
        loc = {}
        for tag, a in axes.items():
            if tag == "wght":
                loc[tag] = max(a.minValue, min(a.maxValue, weight))
            else:
                loc[tag] = a.defaultValue
        st = instancer.instantiateVariableFont(f, loc)
        out.parent.mkdir(parents=True, exist_ok=True)
        st.save(str(out))
        return str(out)
    except Exception:
        return path


def _one_code_per_glyph(path):
    """A PDF ToUnicode map holds one code point per glyph, and PyMuPDF picks the highest one when a font's cmap maps
    several to the same glyph (U+002D and U+00AD share 'hyphen', U+003B and U+037E share 'semicolon'), so an inserted
    '-' would extract as a soft hyphen. Keep only the lowest code point per glyph in a cached copy outside the repo."""
    if not path:
        return path
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return path
    out = _font_cache() / f"{Path(path).stem}-1cp.ttf"
    if out.exists():
        return str(out)
    try:
        f = TTFont(path)
        changed = False
        for t in f["cmap"].tables:
            if not t.isUnicode():
                continue
            seen = set()
            for cp in sorted(t.cmap):
                g = t.cmap[cp]
                if g in seen:
                    del t.cmap[cp]
                    changed = True
                else:
                    seen.add(g)
        if not changed:
            return path
        out.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(out))
        return str(out)
    except Exception:
        return path


def _closest_installed(base):
    from .native import _fc
    b = base.lower()
    wt = _weight_from_name(b)
    bold = wt >= 600
    for hint, fam in _NAME_HINTS:
        if hint in b:
            p = (_fc(fam, "Bold") if bold else None) or _fc(fam, "Regular")
            if p:
                return _one_code_per_glyph(_static_instance(p, wt))
    fam = re.sub(r"[-,].*$", "", base)
    fam = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", fam)
    return _one_code_per_glyph(_static_instance((_fc(fam, "Bold") if bold else None) or _fc(fam, "Regular"), wt))


def _retypeset_line(sp, fonts, l, occs, page_index):
    """Re-typeset one native line: kept runs at their original origins, replacements at the span origin with the
    reused/closest font; text after a replacement within the same phrase shifts by the width difference (reflow,
    no gap or overlap). A run separated by a wide gap (a new column) keeps its original position."""
    log = []
    raw = l.style.get("chars", [])
    lead = 0
    while lead < len(raw) and raw[lead][0] == " ":
        lead += 1
    raw = raw[lead:lead + len(l.text)]
    if not raw:
        return log
    spans = {o.start: o for o in occs}
    segs, i = [], 0
    while i < len(raw):
        if i in spans:
            o = spans[i]; segs.append(("rep", o, raw[o.start:o.end])); i = o.end; continue
        run = []
        while i < len(raw) and i not in spans:
            if run and (raw[i][2], raw[i][3], raw[i][4]) != (run[0][2], run[0][3], run[0][4]):
                break
            run.append(raw[i]); i += 1
        segs.append(("keep", run))
    shift, prev_end_x = 0.0, None
    max_shift = min_shift = 0.0
    for seg in segs:
        if seg[0] == "keep":
            run = seg[1]
            text = "".join(x[0] for x in run)
            if not text.strip():
                continue
            first = next(k for k, x in enumerate(run) if x[0] != " ")
            x, y = run[first][1]
            size = float(run[first][3])
            if prev_end_x is not None and x - prev_end_x > 2.5 * size:
                shift = 0.0
            kind, alias, f = fonts.font_for(run[0][2], text[first:])
            sp.insert_text((x + shift, y), text[first:], fontname=alias, fontsize=size, color=_color(run[0][4]))
            prev_end_x = x + f.text_length(text[first:], fontsize=size)
        else:
            o, chars = seg[1], seg[2]
            c0 = chars[0]
            x, y = c0[1]
            size = float(c0[3])
            if prev_end_x is not None and x - prev_end_x > 2.5 * size:
                shift = 0.0
            kind, alias, f = fonts.font_for(c0[2], o.replacement)
            old_w = float(l.chars[o.end - 1].quad[1][0]) - x
            new_w = f.text_length(o.replacement, fontsize=size)
            fs = size
            if new_w > old_w * 1.6 and old_w > 0:
                fs = size * old_w * 1.6 / new_w; new_w = old_w * 1.6
            sp.insert_text((x + shift, y), o.replacement, fontname=alias, fontsize=fs, color=_color(c0[4]))
            log.append({"occurrence": o.id, "kind": o.kind, "page": page_index, "method": "native_text_rewrite",
                        "font": kind, "reflow_shift_pt": round(new_w - old_w, 2), "size_ratio": round(fs / size, 3),
                        "bbox_pt": [round(x + min(0.0, shift) - 0.5, 1), round(y - size, 1),
                                    round(x + max(new_w + shift, old_w) + 0.5, 1), round(y + size * 0.3, 1)]})
            shift += new_w - old_w
            max_shift, min_shift = max(max_shift, shift), min(min_shift, shift)
            prev_end_x = x + old_w
    qs = np.array([c.quad for c in l.chars], np.float32).reshape(-1, 2) if l.chars else None
    entry = {"line": l.id, "page": page_index, "method": "line_retypeset_reflow", "segments": len(segs), "max_shift_pt": round(max_shift, 2)}
    if qs is not None and len(qs):
        # the retypeset line (incl. text moved by reflow) is part of the edited area for the out-of-region check
        entry["bbox_pt"] = [round(float(qs[:, 0].min()) + min(0.0, min_shift) - 0.5, 1), round(float(qs[:, 1].min()) - 0.5, 1),
                            round(float(qs[:, 0].max()) + max(0.0, max_shift) + 0.5, 1), round(float(qs[:, 1].max()) + 0.5, 1)]
    log.append(entry)
    return log


def _render_pdf(doc, occs, regions, out_path, rng, ctx=None):
    import pymupdf
    src = pymupdf.open(doc.path)
    log = []
    for page in doc.pages:
        pocc = [o for o in occs if o.page == page.index]
        preg = [r for r in regions if r.id.split(":")[1] == f"p{page.index}"]
        sp = src[page.index]
        if page.raster:
            page.out_image = page.image.copy()
            orig, page.image = page.image, page.out_image
            log += _apply_raster(page, pocc, preg, rng, ctx=dict(ctx or {}, orig=orig, jpeg_q=None))
            page.image = orig
            continue
        lines = {l.id: l for l in page.lines}
        inserts, pixel_edits, retype = [], [], {}
        for a in sp.annots() or []:
            log.append({"page": page.index, "method": "annotation_removed", "bbox_pt": [round(v, 1) for v in a.rect]})
        for o in pocc:
            l = lines[o.line_id]
            if l.source in ("native", "native_hidden"):
                ch = l.chars[o.start:o.end]
                x0 = min(c.quad[0][0] for c in ch); x1 = max(c.quad[1][0] for c in ch)
                y0 = min(c.quad[0][1] for c in ch); y1 = max(c.quad[2][1] for c in ch)
                if l.source == "native":
                    inserts.append((o, l, (x0, y0, x1, y1)))
                    retype.setdefault(l.id, l)
                log.append({"occurrence": o.id, "kind": o.kind, "page": page.index, "bbox_pt": [round(v, 1) for v in (x0, y0, x1, y1)],
                            "method": "native_redact_rewrite" if l.source == "native" else "hidden_text_removed"})
            elif l.source == "ocr_pixel_only":
                pixel_edits.append((o, l))
        # Lines containing a target are redacted whole and re-typeset run-by-run at the ORIGINAL glyph origins:
        # MuPDF glyph removal can drop or shift neighbouring glyphs, so we never rely on partial-line removal.
        for l in retype.values():
            bb = quad_bbox((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]))
            hh = bb[3] - bb[1]
            sp.add_redact_annot(pymupdf.Rect(bb[0] - 0.5, bb[1] + 0.2 * hh, bb[2] + 0.5, bb[3] - 0.2 * hh), fill=False)
        # every hidden text line is removed whole, sensitive or not
        for l in page.lines:
            if l.source == "native_hidden" and l.chars:
                bb = quad_bbox((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3]))
                sp.add_redact_annot(pymupdf.Rect(*bb), fill=False)
                log.append({"line": l.id, "page": page.index, "method": "hidden_text_line_removed", "bbox_pt": [round(v, 1) for v in bb]})
        # pixel-only text and visual regions: edit a page render with the native raster engine, then place the edited
        # pixels back as a patch over exactly the edited area (underlying image pixels are redacted first)
        patches = []
        if pixel_edits or preg:
            work = page.image.copy()
            pg = type(page)(page.index, page.width, page.height, True, page.scale, work, page.lines)
            plog = _apply_raster(pg, [o for o, _ in pixel_edits], preg, rng, coord_scale=page.scale,
                                 ctx=dict(ctx or {}, orig=page.image, jpeg_q=None))
            for e in plog:
                x0, y0, x1, y1 = e["bbox_px"]
                pad = 2
                X0, Y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
                X1, Y1 = min(work.shape[1], int(math.ceil(x1)) + pad), min(work.shape[0], int(math.ceil(y1)) + pad)
                e = dict(e); e.pop("bbox_px", None)
                if e["method"] in ("kept_unchanged", "symbol_kept_no_sensitive_payload") or X1 <= X0 or Y1 <= Y0:
                    e["bbox_pt"] = [round(v / page.scale, 1) for v in (x0, y0, x1, y1)]
                    log.append(e); continue
                rect = pymupdf.Rect(X0 / page.scale, Y0 / page.scale, X1 / page.scale, Y1 / page.scale)
                sp.add_redact_annot(rect, fill=False)
                patches.append((rect, work[Y0:Y1, X0:X1].copy()))
                e["bbox_pt"] = [round(v, 1) for v in rect]
                e["method"] = e["method"] + "+pdf_patch"
                log.append(e)
        sp.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                            text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        from PIL import Image
        for rect, crop in patches:
            bio = io.BytesIO(); Image.fromarray(crop).save(bio, "PNG")
            sp.insert_image(rect, stream=bio.getvalue(), keep_proportion=False)
        fonts = _PageFonts(src, sp)
        for l in retype.values():
            log += _retypeset_line(sp, fonts, l, [o for o, ll, _b in inserts if ll.id == l.id], page.index)
    # ---- fresh output document
    out = pymupdf.open()
    for page in doc.pages:
        if page.raster:
            from PIL import Image
            exts = set()
            for im_ in src[page.index].get_images():
                try:
                    exts.add(src.extract_image(im_[0])["ext"])
                except Exception:
                    pass
            arr = page.out_image
            rect = src[page.index].rect
            native, ibox = _native_scan(src, page.index)
            if native is not None:
                # write edits back at the scan's own resolution: only pixels under edited areas change, everything
                # else stays bit-identical to the source scan (no global resampling, no texture change)
                arr, changed = _merge_into_native(native, page.image, page.out_image)
                place = ibox
                how = "scan_edited_at_native_resolution_no_text_layer"
            else:
                place, how, changed = None, "scan_rebuilt_from_edited_pixels_no_text_layer", None
            gray = bool(np.abs(arr.astype(np.int16) - arr[..., :1]).max() <= 2) if arr.ndim == 3 else True
            pil = Image.fromarray((arr[..., 0] if arr.ndim == 3 else arr) if gray else arr)
            b = io.BytesIO()
            if exts & {"jpeg", "jpg", "jpx"}:
                pil.save(b, "JPEG", quality=ctx.get("scan_jpeg_q", 92) if ctx else 92)
            else:
                pil.save(b, "PNG")  # lossless like the source scan: no recompression grain outside edited regions
            np_ = out.new_page(width=rect.width, height=rect.height)
            np_.insert_image(place or np_.rect, stream=b.getvalue(), keep_proportion=False)
            log.append({"page": page.index, "method": how, **({"native_px": list(arr.shape[1::-1]), "changed_native_px": changed} if native is not None else {})})
        else:
            out.insert_pdf(src, from_page=page.index, to_page=page.index, annots=False, links=False)
    for p in out:
        for a in list(p.annots() or []):
            p.delete_annot(a)
        for w in list(p.widgets() or []):
            p.delete_widget(w)
    out.scrub(attached_files=True, clean_pages=True, embedded_files=True, hidden_text=True, javascript=True, metadata=True,
              redactions=True, redact_images=0, remove_links=True, reset_fields=True, reset_responses=True, thumbnails=True, xml_metadata=True)
    out.set_metadata({})
    out.del_xml_metadata()
    # embed only the glyphs the page uses: full replacement fonts otherwise add 100-300 KB per font
    try:
        out.subset_fonts()
        subset = True
    except Exception:  # keep the full fonts rather than fail the document
        subset = False
    out.save(out_path, garbage=4, deflate=True, clean=True, no_new_id=False)
    log.append({"method": "pdf_rebuilt_fresh", "stripped": ["info_metadata", "xmp", "annotations", "embedded_files", "javascript",
                                                         "links", "form_fields", "thumbnails", "hidden_text"],
                "fonts_subset": subset,
                "source_had": {"metadata_fields": len(doc.hidden.get("metadata", {})), "annotations": len(doc.hidden.get("annotations", [])),
                               "embedded_files": len(doc.hidden.get("embedded_files", {})), "javascript": doc.hidden.get("javascript", False)}})
    return log
