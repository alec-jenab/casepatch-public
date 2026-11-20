"""Native in-place editing engine (local, CPU, $0).

Every edit is constrained to a target mask; pixels outside it are copied from the source unchanged.
- Text in pixels: glyph-compared font match against installed open fonts (sans/serif/mono/typewriter/thermal/
  handwriting), size + horizontal scale + baseline from re-rendering the ORIGINAL text, ink colour, blur, stroke;
  the original glyphs are removed with LaMa inpainting (OpenCV Telea fallback), the replacement is rendered with the
  matched style, warped into the line's perspective quad, composited multiplicatively so the local lighting/paper
  texture shows through, then noise and (for JPEG sources) block grain are re-applied inside the mask only.
- Signatures: strokes inpainted away; a different signature is drawn from the fake name in a handwriting font with
  matched ink colour, stroke width, slant and blur.
- Faces: replaced with a synthetic face (SFHQ, MIT/CC0) aligned by landmarks, colour/contrast/blur/noise matched.
- QR / barcodes: regenerated in the same symbology and size with the case-mapped payload.
Generative models never redraw whole pages; the only learned image model used is LaMa for local background fill.
"""
from __future__ import annotations

import functools
import hashlib
import io
import math
import os
import subprocess
from pathlib import Path

import re

import numpy as np

# ------------------------------------------------------------------ fonts
TEXT_FAMILIES = [
    # (family, class)
    ("Liberation Sans", "sans"), ("Arimo", "sans"), ("Roboto", "sans"), ("Open Sans", "sans"), ("Lato", "sans"),
    ("Noto Sans", "sans"), ("Montserrat", "sans"), ("Nunito Sans", "sans"), ("Work Sans", "sans"), ("Mulish", "sans"),
    ("PT Sans", "sans"), ("Fira Sans", "sans"), ("Carlito", "sans"), ("Nimbus Sans", "sans"), ("DejaVu Sans", "sans"),
    ("Roboto Condensed", "sans-narrow"), ("PT Sans Narrow", "sans-narrow"), ("Archivo Narrow", "sans-narrow"),
    ("Barlow Condensed", "sans-narrow"), ("Oswald", "sans-narrow"),
    ("Liberation Serif", "serif"), ("Tinos", "serif"), ("PT Serif", "serif"), ("Merriweather", "serif"),
    ("Libre Baskerville", "serif"), ("Nimbus Roman", "serif"),
    ("Liberation Mono", "mono"), ("Cousine", "mono"), ("DejaVu Sans Mono", "mono"), ("Roboto Mono", "mono"),
    ("Source Code Pro", "mono"), ("IBM Plex Mono", "mono"), ("Inconsolata", "mono"), ("Space Mono", "mono"),
    ("Fira Mono", "mono"), ("Nimbus Mono PS", "mono"), ("Share Tech Mono", "mono"), ("Anonymous Pro", "mono"),
    ("Courier Prime", "typewriter"), ("Special Elite", "typewriter"), ("Cutive Mono", "typewriter"),
    ("VT323", "thermal"), ("Lekton", "thermal"),
    ("Caveat", "hand"), ("Kalam", "hand"), ("Patrick Hand", "hand"), ("Indie Flower", "hand"), ("Reenie Beanie", "hand"),
    ("Shadows Into Light", "hand"), ("Nothing You Could Do", "hand"),
]
SIGNATURE_FAMILIES = ["Homemade Apple", "Mrs Saint Delafield", "Herr Von Muellerhoff", "Monsieur La Doulaise", "Alex Brush",
                      "Allura", "Great Vibes", "Sacramento", "Mr Dafoe", "La Belle Aurore", "Cedarville Cursive", "Zeyada",
                      "Dawning of a New Day", "Yellowtail", "Pinyon Script", "Qwigley", "Meddon", "Dancing Script"]


@functools.lru_cache(maxsize=None)
def _fc(family, style="Regular"):
    """Installed font file for an exact family (fontconfig, else the OS font folders, else the bundled fonts)."""
    from .fonts import find
    return find(family, style)


@functools.lru_cache(maxsize=None)
def font_candidates(include_hand=True):
    out = []
    for fam, cls in TEXT_FAMILIES:
        if cls == "hand" and not include_hand:
            continue
        reg = _fc(fam, "Regular")
        if not reg:
            continue
        out.append({"family": fam, "class": cls, "weight": "regular", "path": reg})
        bold = _fc(fam, "Bold")
        if bold and bold != reg:
            out.append({"family": fam, "class": cls, "weight": "bold", "path": bold})
        elif "Variable" in reg:
            out.append({"family": fam, "class": cls, "weight": "bold", "path": reg, "variation": "Bold"})
    if not out:  # minimal fallback when fontconfig is unavailable
        from .fonts import fallback_any
        for p in fallback_any():
            if os.path.exists(p):
                out.append({"family": Path(p).stem, "class": "sans", "weight": "regular", "path": p})
    return tuple(tuple(sorted(d.items())) for d in out)


def _font(spec, size):
    from PIL import ImageFont
    spec = dict(spec)
    f = ImageFont.truetype(spec["path"], max(4, int(round(size))))
    if spec.get("variation"):
        try:
            f.set_variation_by_name(spec["variation"])
        except Exception:
            try:
                axes = f.get_variation_axes()
                f.set_variation_by_axes([min(a["maximum"], 700) if a.get("name", b"") in (b"Weight", "Weight") else a["default"] for a in axes])
            except Exception:
                pass
    return f


def render_text_alpha(text, spec, size, ss=4):
    """Supersampled text alpha (float 0..1) with its ink bbox and font-space origin offsets (in output px)."""
    from PIL import Image, ImageDraw
    f = _font(spec, size * ss)
    bb = f.getbbox(text)
    w, h = max(1, bb[2] - bb[0] + 4 * ss), max(1, bb[3] - bb[1] + 4 * ss)
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((2 * ss - bb[0], 2 * ss - bb[1]), text, font=f, fill=255)
    im = im.resize((max(1, w // ss), max(1, h // ss)), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    # origin (pen position x=0, y=baseline-ascent top) in output pixel coords
    return a, (2 - bb[0] / ss, 2 - bb[1] / ss)


def _ink_box(a, thr=0.35):
    ys, xs = np.where(a > thr)
    if not len(xs):
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _resize(a, w, h):
    import cv2
    return cv2.resize(a, (max(1, int(round(w))), max(1, int(round(h)))), interpolation=cv2.INTER_AREA if w < a.shape[1] else cv2.INTER_LINEAR)


def darkness(gray, bg=None, ink=None, text_h=None):
    """Normalised ink coverage 0..1 of a grayscale crop: black-top-hat against a local background estimate
    (robust to coloured security patterns, gradients and glare), scaled by a robust stroke contrast."""
    import cv2
    g = gray.astype(np.float32)
    k = int(max(7, (text_h or g.shape[0] * 0.5) * 0.9)) | 1
    bgimg = cv2.morphologyEx(cv2.medianBlur(gray, 3) if gray.dtype == np.uint8 else g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.float32)
    bgimg = cv2.GaussianBlur(bgimg, (0, 0), k / 4)
    th = np.clip(bgimg - g, 0, None)
    hi = float(np.percentile(th, 99.3))
    if hi < 10:
        return np.zeros_like(g)
    return np.clip(th / hi, 0, 1)


def _vext(a, thr=0.35):
    """Robust vertical text band: 15th percentile of per-column ink tops, 85th of bottoms (ignores stray specks,
    descenders and accents consistently for observed and rendered text)."""
    b = a > thr
    cols = np.where(b.any(axis=0))[0]
    if not len(cols):
        return None
    tops = np.argmax(b[:, cols], axis=0)
    bots = b.shape[0] - np.argmax(b[::-1, cols], axis=0)
    return float(np.percentile(tops, 15)), float(np.percentile(bots, 85)), int(cols.min()), int(cols.max()) + 1


# metric-compatible families (same widths, near-identical shapes): one typeface for page-consistency purposes
FAMILY_GROUP = {"Arimo": "arial", "Liberation Sans": "arial", "Nimbus Sans": "arial",
                "Tinos": "times", "Liberation Serif": "times", "Nimbus Roman": "times",
                "Cousine": "courier", "Liberation Mono": "courier", "Nimbus Mono PS": "courier"}


def family_group(family):
    return FAMILY_GROUP.get(family, family)


def match_font(rect_gray, text, include_hand=True, top=1, max_h=None, prior_class=None, prior_family=None, class_bias=None, candidates=None,
               prior_weight=None, prior_strength=0.03):
    """Glyph comparison: render `text` in each candidate font, align its robust text band and horizontal extent onto
    the observed ink, and compare coverage maps (Dice on blurred maps) + aspect + density error."""
    import cv2
    text = text.strip()
    if not text:
        return None
    obs = rect_gray if rect_gray.dtype == np.float32 else darkness(rect_gray)
    ob = _ink_box(obs, 0.4); ov = _vext(obs, 0.4)
    if ob is None or ov is None:
        return None
    ox0, oy0, ox1, oy1 = ob
    ot, obt, oc0, oc1 = ov
    ow, oh = oc1 - oc0, max(obt - ot, 2.0)
    if max_h and oh > max_h:  # stray ink (glare, neighbouring line) merged into the band: trust the OCR line height
        mid = (ot + obt) / 2 if obt - ot < 1.6 * max_h else obt - max_h / 2
        ot, obt, oh = mid - max_h / 2, mid + max_h / 2, float(max_h)
    if ow < 4 or oh < 3:
        return None
    Hh, Ww = obs.shape
    obl = cv2.GaussianBlur(obs, (0, 0), max(0.8, oh / 16))
    results = []
    for spec in (candidates if candidates is not None else font_candidates(include_hand)):
        a, _ = render_text_alpha(text, spec, 48, ss=2)
        rv = _vext(a)
        if rv is None:
            continue
        rt, rbt, rc0, rc1 = rv
        rh, rw = max(rbt - rt, 1.0), max(rc1 - rc0, 1)
        k = oh / rh
        kx = ow / rw
        aspect_err = abs(math.log(kx / k))
        r = cv2.resize(a, (max(1, int(round(a.shape[1] * kx))), max(1, int(round(a.shape[0] * k)))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros_like(obs)
        dy, dx = int(round(ot - rt * k)), int(round(oc0 - rc0 * kx))
        y0, x0 = max(0, dy), max(0, dx); y1, x1 = min(Hh, dy + r.shape[0]), min(Ww, dx + r.shape[1])
        if y1 <= y0 or x1 <= x0:
            continue
        canvas[y0:y1, x0:x1] = r[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        cbl = cv2.GaussianBlur(canvas, (0, 0), max(0.8, oh / 16))
        dice = 2 * float((cbl * obl).sum()) / max(float((cbl * cbl).sum() + (obl * obl).sum()), 1e-6)
        dens = abs(float(canvas.mean()) - float(obs.mean())) / max(float(obs.mean()), 1e-3)
        score = (1 - dice) + 0.5 * aspect_err + 0.15 * dens + (0.08 if spec_class(spec) == "hand" and candidates is None else 0.0) + \
            (0.06 if prior_class and spec_class(spec) != prior_class else 0.0) + \
            (prior_strength if prior_family and family_group(dict(spec).get("family")) != family_group(prior_family) else 0.0) + \
            (0.6 * prior_strength if prior_weight and dict(spec).get("weight") != prior_weight else 0.0) + \
            ((class_bias or {}).get(spec_class(spec), 0.0))
        results.append((score, dict(spec), k, kx))
    if not results:
        return None
    results.sort(key=lambda x: x[0])
    score, spec, k, kx = results[0]
    return {"font": spec, "family": spec["family"], "class": spec["class"], "weight": spec["weight"], "size_px": float(48 * k),
            "sx": float(np.clip(kx / k, 0.55, 1.8)), "score": round(float(score), 3), "ink_box": (int(ox0), int(oy0), int(ox1), int(oy1)),
            "runner_up": [(r[1]["family"], r[1]["weight"], round(float(r[0]), 3)) for r in results[1:3]]}


def spec_class(spec):
    return dict(spec).get("class")


# ------------------------------------------------------------------ inpainting
_LAMA = None
_LAMA_FAILED = False


def lama():
    global _LAMA, _LAMA_FAILED
    if _LAMA is not None or _LAMA_FAILED or os.environ.get("CASEPATCH_INPAINT") == "opencv":
        return _LAMA
    from .models import find_model
    p = os.environ.get("CASEPATCH_LAMA_MODEL") or find_model("big-lama.pt")
    if not p:
        _LAMA_FAILED = True
        return None
    try:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        m = torch.jit.load(str(p), map_location="cpu")
        m.eval()
        _LAMA = m
    except Exception:
        _LAMA_FAILED = True
    return _LAMA


def inpaint_method():
    return "lama" if lama() is not None else "opencv_telea"


def inpaint(img, mask, max_side=640):
    """Fill mask>0 pixels of img (uint8 RGB) in place. Only masked pixels change. Works per connected cluster crop."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return img
    H, W = m.shape
    grp = cv2.dilate(m, np.ones((41, 41), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grp)
    model = lama()
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        pad = int(max(24, 0.6 * max(w, h) ** 0.9))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        crop = img[y0:y1, x0:x1]
        cm = m[y0:y1, x0:x1] * (lab[y0:y1, x0:x1] == i)
        if not cm.any():
            continue
        filled = None
        if _is_plain_paper(crop, cm):
            # plain paper/scan background: a learned inpainter tends to hallucinate glyph fragments next to text, so
            # fill with the local paper level (max filter removes ink, blur keeps the lighting gradient)
            k = int(np.clip(2 * float(cv2.distanceTransform(cm.astype(np.uint8), cv2.DIST_L2, 3).max()) + 5, 7, 61)) | 1
            filled = _paper_fill(crop, cm, k)
            _PAPER_FILLS[0] += 1
        elif model is not None:
            try:
                filled = _lama_crop(model, crop, cm, max_side)
            except Exception:
                filled = None
        if filled is None:
            filled = cv2.inpaint(crop, cm * 255, 5, cv2.INPAINT_TELEA)
        sel = cm > 0
        crop[sel] = filled[sel]
    return img


_PAPER_FILLS = [0]


def _paper_fill(crop, cm, k):
    """Local paper colour under a mask: a masked, clipped average of the surrounding non-ink pixels. A max filter
    picked up JPEG ringing (the light halo around dark glyphs) and filled erased words with lighter-than-paper
    letter shapes on tinted cards (D11b: white 'HALE' ghosts on a blue licence)."""
    import cv2
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    near = cv2.dilate(cm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(cm.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~near
    if ring.sum() < 20:
        ring = ~near
    gv = g[ring].astype(np.float32)
    valid = ~near & (g >= np.percentile(gv, 35))  # paper, not ink
    ref = crop.astype(np.float32).reshape(crop.shape[0], crop.shape[1], -1)
    pv = ref[valid & ring] if (valid & ring).sum() >= 20 else ref[valid]
    lo, hi = np.percentile(pv, 20, axis=0), np.percentile(pv, 80, axis=0)
    ref = np.clip(ref, lo, hi)  # ringing (lighter) and grain outliers do not set the level
    w = valid.astype(np.float32)
    sig = max(2.0, k / 2)
    num = cv2.GaussianBlur(ref * w[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)[..., None]
    med = np.median(pv, axis=0)
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), med)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out.reshape(crop.shape)


def _is_plain_paper(crop, cm):
    import cv2
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    ring = (cv2.dilate(cm, np.ones((11, 11), np.uint8)) > 0) & ~(cv2.dilate(cm, np.ones((3, 3), np.uint8)) > 0)
    v = g[ring].astype(np.float32)
    if v.size < 30:
        return False
    hi = v[v >= np.percentile(v, 35)]
    if crop.ndim == 3:
        c = crop[ring].astype(np.float32)
        chroma = float(np.percentile(c.max(1) - c.min(1), 60))
    else:
        chroma = 0.0
    return float(np.percentile(v, 75)) > 150 and float(hi.std()) < 7 and chroma < 40


def _lama_crop(model, crop, cm, max_side):
    import cv2
    import torch
    h, w = cm.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        c = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        mm = (cv2.resize(cm.astype(np.uint8) * 255, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST) > 0)
        mm = cv2.dilate(mm.astype(np.uint8), np.ones((3, 3), np.uint8))
    else:
        c, mm = crop, cm
    ph, pw = (8 - c.shape[0] % 8) % 8, (8 - c.shape[1] % 8) % 8
    cp = np.pad(c, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mp = np.pad(mm.astype(np.float32), ((0, ph), (0, pw)), mode="constant")
    with torch.inference_mode():
        ti = torch.from_numpy(cp).permute(2, 0, 1)[None].float() / 255.0
        tm = (torch.from_numpy(mp)[None, None] > 0).float()
        out = model(ti, tm)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    out = (out[:c.shape[0], :c.shape[1]] * 255).round().astype(np.uint8)
    if s < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


# ------------------------------------------------------------------ style estimation helpers
def noise_sigma(gray_crop, sel=None):
    import cv2
    g = gray_crop.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    v = hp[sel] if sel is not None and sel.any() else hp.ravel()
    if v.size < 10:
        return 0.0
    return float(min(12.0, 1.4826 * np.median(np.abs(v - np.median(v)))))  # cap: rules/edges in the sample are not grain


def jpeg_quality(path):
    """Estimate libjpeg quality from the luminance quantisation table (None if not JPEG)."""
    try:
        from PIL import Image
        im = Image.open(path)
        if im.format != "JPEG" or not getattr(im, "quantization", None):
            return None
        q = list(im.quantization.values())[0]
        std = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
               18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
        ratio = float(np.mean([a / b for a, b in zip(sorted(q), sorted(std))])) * 100
        qual = (200 - ratio) / 2 if ratio <= 100 else 5000 / ratio
        return int(np.clip(round(qual), 10, 98))
    except Exception:
        return None


def regrain(img, orig, sel, rng, sigma_target, jpeg_q=None):
    """Add noise inside `sel` so its high-frequency noise matches the source, then (JPEG sources) re-encode the
    surrounding block and keep only selected pixels."""
    import cv2
    ys, xs = np.where(sel)
    if not len(xs):
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = img[y0:y1, x0:x1].astype(np.float32)
    s_now = noise_sigma(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), sel[y0:y1, x0:x1])
    add = math.sqrt(max(sigma_target ** 2 - s_now ** 2, 0.0))
    if add > 0.3:
        n = rng.normal(0, add, patch.shape[:2])[..., None]
        patch = patch + n * np.array([1.0, 1.0, 1.0])
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    if jpeg_q:
        from PIL import Image
        Y0, X0 = (y0 // 8) * 8, (x0 // 8) * 8
        full = img.copy(); full[y0:y1, x0:x1] = patch
        blk = full[Y0:y1 + 8, X0:x1 + 8]
        b = io.BytesIO(); Image.fromarray(blk).save(b, "JPEG", quality=jpeg_q)
        dec = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
        s2 = sel[Y0:y1 + 8, X0:x1 + 8]
        tgt = img[Y0:y1 + 8, X0:x1 + 8]
        tgt[s2] = dec[:s2.shape[0], :s2.shape[1]][s2]
    else:
        tgt = img[y0:y1, x0:x1]
        s2 = sel[y0:y1, x0:x1]
        tgt[s2] = patch[s2]


def _homographies(quad, Wr, Hr):
    import cv2
    q = np.array(quad, np.float32)
    rect = np.float32([[0, 0], [Wr, 0], [Wr, Hr], [0, Hr]])
    return cv2.getPerspectiveTransform(q, rect), cv2.getPerspectiveTransform(rect, q)


def _expand_quad(quad, gx, gy):
    q = np.array(quad, np.float32)
    tl, tr, br, bl = q
    ux = (tr - tl) / max(np.linalg.norm(tr - tl), 1e-6)
    uy = (bl - tl) / max(np.linalg.norm(bl - tl), 1e-6)
    return np.array([tl - ux * gx - uy * gy, tr + ux * gx - uy * gy, br + ux * gx + uy * gy, bl - ux * gx + uy * gy], np.float32)


def _expand_quad_lr(quad, gl, gr, gy):
    q = np.array(quad, np.float32)
    tl, tr, br, bl = q
    ux = (tr - tl) / max(np.linalg.norm(tr - tl), 1e-6)
    uy = (bl - tl) / max(np.linalg.norm(bl - tl), 1e-6)
    return np.array([tl - ux * gl - uy * gy, tr + ux * gr - uy * gy, br + ux * gr + uy * gy, bl - ux * gl + uy * gy], np.float32)


# ------------------------------------------------------------------ handwriting + leftover strokes
CURSIVE_TEXT_FAMILIES = ["Homemade Apple", "La Belle Aurore", "Dancing Script", "Cedarville Cursive", "Dawning of a New Day",
                         "Zeyada", "Sacramento"]


@functools.lru_cache(maxsize=None)
def hand_candidates(cursive=False):
    """Font specs for redrawing a handwritten value: print-style hands, or connected cursive."""
    if cursive:
        out = []
        for fam in CURSIVE_TEXT_FAMILIES:
            pth = _fc(fam, "Regular")
            if pth:
                out.append(tuple(sorted({"family": fam, "class": "hand", "weight": "regular", "path": pth, "cursive": True}.items())))
        if out:
            return tuple(out)
    return tuple(sp for sp in font_candidates(True) if dict(sp).get("class") == "hand")


def handwriting_kind(span_ink, span_text, fm, min_area=4, line_h=None, line_conf=None, blur=None):
    """'cursive' / 'print' when the observed span is handwriting, else None. Typed text matches some installed type
    face well (score < ~0.3); handwriting matches none, and OCR of it is unreliable, so the decision uses the ink:
    connected cursive has far fewer stroke components than letters."""
    import cv2
    if fm is None or not span_ink.any() or "<<" in span_text:
        return None  # (machine-readable zones are OCR-B print by definition)
    if fm.get("class") == "hand":
        return "print"
    if fm.get("score", 0) < 0.38:
        return None
    k = max(3, int(round(line_h * 0.06)) | 1) if line_h else 3
    ink = cv2.morphologyEx(span_ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))  # scan breaks in a stroke
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_a = max(min_area, (0.12 * line_h) ** 2) if line_h else min_area
    comps = int(sum(1 for i in range(1, n) if st[i, 4] >= min_a))
    letters = max(1, sum(ch.isalnum() for ch in span_text))
    if letters < 3:
        return None
    ratio = comps / letters
    if os.environ.get("CP_DEBUG_HW"):
        print("handwriting?", repr(span_text), fm.get("family"), fm.get("score"), comps, letters)
    if ratio < 0.6:
        if blur is not None and line_h and blur > 0.025 * line_h:
            return None  # defocused photo of print: the blur, not a pen, joins the letters (ID held in a hand)
        return "cursive"
    if fm.get("score", 0) > 0.45 and ratio < 1.1 and (line_conf is None or line_conf < 0.95):
        return "print"  # (crisp print that OCR reads with near-certainty is type, however poorly our fonts match it)
    return None


def connected_leftovers(src, erase, eq, lh, protect=None):
    """Ink outside the erase mask that belongs to strokes the mask already cuts through (an ascender, a loop, the
    cross of a 't' above the OCR box): erasing only the in-mask part leaves visible stroke stubs. Returns a bool mask
    of those outside parts (dilated), or None."""
    import cv2
    H, W = erase.shape
    xs, ys = eq[:, 0], eq[:, 1]
    x0, x1 = int(max(0, xs.min() - lh)), int(min(W, xs.max() + lh))
    y0, y1 = int(max(0, ys.min() - lh)), int(min(H, ys.max() + lh))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    e = erase[y0:y1, x0:x1]
    if not e.any():
        return None
    g = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    dk = darkness(g, text_h=lh)
    ref = float(np.percentile(dk[e], 90)) if e.sum() > 10 else 0.0
    if ref < 0.2:
        return None
    ink = dk > max(0.25, 0.35 * ref)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    cols = np.where(e.any(axis=0))[0]
    span_w = int(cols.max() - cols.min() + 1)
    prot = protect[y0:y1, x0:x1] if protect is not None else None
    out = np.zeros_like(e)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 3 or h > 2.5 * lh or w > span_w + 2 * lh or (w > 3 * lh and h < 0.3 * lh):
            continue
        comp = lab == i
        inside = int((comp & e).sum())
        if a - inside < 3 or inside < 0.3 * a:
            continue
        if float(dk[comp].mean()) < 0.5 * ref:
            continue
        if prot is not None and (comp & prot).sum() > 0.05 * a:
            continue
        out |= comp
    if not out.any():
        return None
    out = cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(round(lh / 14)))) > 0
    if prot is not None:
        out &= ~prot
    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = out & ~e
    return full if full.any() else None


GHOST_KEEP = 0.25  # an original ink pixel at >= a quarter of its old darkness is a visible ghost


def ghost_pass(img, src, erase, lh, protect=None, rounds=2):
    """After the inpaint: original glyph ink that is still (faintly) there inside or right around the erased area --
    a guilloche / glare background lets the inpainter rebuild the old letters as a ghost. Those pixels are erased
    again (dilated) and inpainted on their own. Returns the number of re-inpainted pixels."""
    import cv2
    ys, xs = np.where(erase)
    if not len(ys):
        return 0
    H, W = erase.shape
    pad = int(max(3, lh * 0.25))
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    e = erase[y0:y1, x0:x1]
    g0 = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    d0 = darkness(g0, text_h=lh)
    ink0 = (d0 > 0.3) & (cv2.dilate(e.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if ink0.sum() < 10:
        return 0
    total = 0
    for _ in range(rounds):
        g1 = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        d1 = darkness(g1, text_h=lh)
        res = ink0 & (d1 >= GHOST_KEEP * d0) & (d1 > 0.1)
        if protect is not None:
            res &= ~protect[y0:y1, x0:x1]
        if res.sum() < max(8, 0.04 * ink0.sum()):
            break
        m = cv2.dilate(res.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(round(lh / 16))))
        if protect is not None:
            m &= ~protect[y0:y1, x0:x1].astype(np.uint8)
        # small diffusion fill: a learned inpainter would rebuild the letters again from the same context
        sub = np.ascontiguousarray(img[y0:y1, x0:x1])
        img[y0:y1, x0:x1] = cv2.inpaint(sub, m, max(3, int(lh * 0.2)), cv2.INPAINT_TELEA)
        total += int(m.sum())
    return total


# ------------------------------------------------------------------ text replacement
def line_style(img, line_quad, line_text):
    """Font match of a whole printed line (no edit): the page-style pre-pass."""
    import cv2
    q = np.array(line_quad, np.float32)
    lw = float(np.linalg.norm(q[1] - q[0])); lh = float(np.linalg.norm(q[3] - q[0]))
    if lw < 4 or lh < 4 or len(line_text.strip()) < 4:
        return None
    my, mx = lh * 0.35, lh * 0.5
    eq = _expand_quad_lr(q, mx, mx, my)
    S = max(1.0, 28.0 / lh)
    Wr, Hr = int(round((lw + 2 * mx) * S)), int(round((lh + 2 * my) * S))
    Mf, _ = _homographies(eq, Wr, Hr)
    rect = cv2.warpPerspective(img, Mf, (Wr, Hr), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rg = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
    dk = _clean_ink(_remove_rules(darkness(rg, text_h=lh * S), lh * S), my * S, (my + lh) * S)
    return match_font(dk, line_text, include_hand=False, max_h=0.62 * lh * S)


def page_style(img, lines):
    """One typeface for a printed page: font-match up to 8 whole lines that will be edited and keep the family group
    and weight most of them agree on. None when they disagree (mixed pages, handwriting) or there are too few."""
    res = [r for r in (line_style(img, q, t) for q, t in lines[:10]) if r is not None and r["class"] != "hand"]
    if len(res) < 2:
        return None
    from collections import Counter
    grp = Counter(family_group(r["family"]) for r in res)
    g, cnt = grp.most_common(1)[0]
    if cnt < max(2, 0.6 * len(res)):
        return None
    mine = [r for r in res if family_group(r["family"]) == g]
    fam = Counter(r["family"] for r in mine).most_common(1)[0][0]
    wt = Counter(r["weight"] for r in mine).most_common(1)[0][0]
    return {"family": fam, "group": g, "weight": wt, "lines": len(res), "agree": cnt}


def replace_text(img, line_quad, line_text, start, end, replacement, rng, jpeg_q=None, style_cache=None, orig_img=None, protect=None,
                 line_conf=None, span_frac=None):
    """Replace line_text[start:end] (drawn inside line_quad) with `replacement`, in place.
    Returns (edit_mask bool HxW, info dict) or (None, info) when the edit was not possible."""
    import cv2
    H, W = img.shape[:2]
    src = orig_img if orig_img is not None else img
    q = np.array(line_quad, np.float32)
    lw = float(np.linalg.norm(q[1] - q[0])); lh = float(np.linalg.norm(q[3] - q[0]))
    if lw < 4 or lh < 4:
        return None, {"error": "quad_too_small"}
    # rectify with margins so the full ink (ascenders/descenders) is inside
    my, mx = lh * 0.35, lh * 0.5
    n = max(1, len(line_text))
    mr = lh * 3.0 if end >= n else mx  # a span ending the line may grow into the blank space to its right
    eq = _expand_quad_lr(q, mx, mr, my)
    S = max(1.0, 28.0 / lh)  # upsample tiny text for matching
    Wr, Hr = int(round((lw + mx + mr) * S)), int(round((lh + 2 * my) * S))
    Mf, Mb = _homographies(eq, Wr, Hr)
    rect = cv2.warpPerspective(src, Mf, (Wr, Hr), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rg = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
    key = (round(float(q[0][0])), round(float(q[0][1])), line_text)
    dk_raw = darkness(rg, text_h=lh * S)
    # printed rules (underlines, table borders) touching the text are not glyph ink: exclude them from matching and
    # from the erase mask so the rule itself survives untouched
    dk_raw = _remove_rules(dk_raw, lh * S)
    dk = _clean_ink(dk_raw, my * S, (my + lh) * S)
    line_end = int(round((mx + lw) * S))
    dk[:, line_end + int(lh * S * 0.3):] = 0  # growth margin is not part of the observed line
    ib = _ink_box(dk, 0.4)
    if ib is None:
        return None, {"error": "no_ink_found"}
    ox0, oy0, ox1, oy1 = ib
    # span edges: from the OCR word boxes when the caller has them (span_frac), else a proportional estimate
    f0, f1 = span_frac if span_frac else (start / n, end / n)
    sx0 = mx * S + (lw * S) * f0
    sx1 = mx * S + (lw * S) * f1
    col = (dk > 0.3).any(axis=0)
    def snap(x, direction):
        x = int(np.clip(x, 0, Wr - 1)); lim = int(lh * S * 0.6)
        for d in range(lim):
            xx = x + d * direction
            if 0 <= xx < Wr and not col[xx]:
                return xx
        return x
    faint = np.where((dk > 0.08).any(axis=0))[0]
    def word_gap(x, side):
        """Span edge at a word boundary: the widest blank column run near the proportional estimate (letter gaps are
        narrow, word gaps wide). Proportional OCR char boxes can land inside the first glyph ('VISA 8945' -> the '8'
        was left behind and the new '1' drawn over it)."""
        rng_ = int(lh * S * 1.5); best = None
        xs_ = range(max(0, int(x) - rng_), min(Wr, int(x) + rng_))
        g0 = None
        for xx in list(xs_) + [None]:
            blank = xx is not None and not col[xx]
            if blank and g0 is None:
                g0 = xx
            elif not blank and g0 is not None:
                g1 = xx if xx is not None else xs_.stop
                w = g1 - g0
                if g0 <= ox0 or g1 >= ox1:
                    g0 = None
                    continue  # the page margin around the line is not a gap between two words of it
                if w >= max(2, lh * S * 0.18):
                    sc_ = w - abs((g0 + g1) / 2 - x) * 0.15
                    if best is None or sc_ > best[0]:
                        best = (sc_, g0, g1)
                g0 = None
        if best is None:
            return None
        return best[2] - 1 if side < 0 else best[1] + 1
    a0 = snap(sx0, -1) if start > 0 else max(0, (int(faint.min()) if len(faint) else ox0) - 1)
    a1 = snap(sx1, +1) if end < n else min(Wr, (int(faint.max()) + 1 if len(faint) else ox1) + 1)
    if start > 0 and line_text[start - 1].isspace():
        g = word_gap(sx0, -1); a0 = g if g is not None else a0
    if end < n and line_text[end].isspace():
        g = word_gap(sx1, +1); a1 = g if g is not None else a1
    # exact glyph segmentation when every non-space character is its own ink run (typical for print): the span
    # boundaries then fall in the true inter-glyph gaps instead of a proportional estimate
    runs, x = [], 0
    while x < Wr:
        if col[x]:
            x0_ = x
            while x < Wr and col[x]:
                x += 1
            runs.append((x0_, x))
        else:
            x += 1
    glyphs = [i for i, ch in enumerate(line_text) if not ch.isspace()]
    seg_exact = len(runs) == len(glyphs) and len(glyphs) > 0
    if seg_exact:
        gi0 = next((k for k, i in enumerate(glyphs) if i >= start), None)
        gi1 = next((k for k in range(len(glyphs) - 1, -1, -1) if glyphs[k] < end), None)
        if gi0 is not None and gi1 is not None and gi1 >= gi0:
            r0, r1 = runs[gi0], runs[gi1]
            if gi0 > 0:
                a0 = (runs[gi0 - 1][1] + r0[0]) // 2
            if gi1 + 1 < len(runs):
                a1 = (r1[1] + runs[gi1 + 1][0] + 1) // 2
            a0 = max(0, min(a0, r0[0] - 1)); a1 = min(Wr, max(a1, r1[1] + 1))
    span_dk = dk[:, a0:a1]
    span_ink = span_dk > 0.3
    if not span_ink.any():
        return None, {"error": "span_has_no_ink"}
    # font match on the span itself when it is long enough (lines often mix a bold label with a regular value),
    # otherwise on the whole line
    span_text = line_text[start:end]
    fkey = ("fm", key, start, end) if len(span_text.strip()) >= 4 else ("fm", key)
    if style_cache is not None and fkey in style_cache:
        fm = style_cache[fkey]
    else:
        fm = None
        # page-level prior: documents rarely mix type classes, and tiny degraded text is ambiguous on its own
        votes = style_cache.setdefault("_class_votes", {}) if style_cache is not None else {}
        prior = max(votes, key=votes.get) if sum(votes.values()) >= 2 else None
        fvotes = style_cache.setdefault("_family_votes", {}) if style_cache is not None else {}
        pfam = max(fvotes, key=fvotes.get) if sum(fvotes.values()) >= 2 else None
        page_style = (style_cache or {}).get("_page_style")
        pweight, pstrength = None, 0.03
        if page_style:  # one typeface per printed page (D11a/A10c): every value on it is drawn in the page's face
            pfam, pweight, pstrength = page_style["family"], page_style["weight"], 0.08
            if sum(ch.isdigit() for ch in span_text) >= 0.6 * max(1, len(span_text.strip())):
                # digits alone do not identify a typeface (mono and bold faces won on '12/28/1983', drawn grey-bold):
                # numbers on a one-typeface page are drawn in the page's face
                pstrength = 0.3
        tw_page = typewriter_page(orig_img if orig_img is not None else img, style_cache)
        # thresholded fax/typewriter scans: typed values are fixed-pitch typewriter faces, not sans; only a clearly
        # better proportional match (printed form text) may win
        bias = {"sans": 0.14, "sans-narrow": 0.14, "serif": 0.1, "hand": 0.1, "thermal": 0.04} if tw_page else None
        if len(span_text.strip()) >= 4:
            sub = np.zeros_like(dk); sub[:, a0:a1] = span_dk
            fm = match_font(sub, span_text, include_hand=True, max_h=0.62 * lh * S, prior_class=prior, prior_family=pfam, class_bias=bias,
                            prior_weight=pweight, prior_strength=pstrength)
        if fm is None:
            fm = match_font(dk, line_text, include_hand=True, max_h=0.62 * lh * S, prior_class=prior, prior_family=pfam, class_bias=bias,
                            prior_weight=pweight, prior_strength=pstrength)
        _rs0 = rg[:, a0:a1].astype(np.float32)
        _blur = _blur_from_edges(_rs0, span_ink, float(np.percentile(_rs0, 95) - np.percentile(_rs0, 5))) if span_ink.any() else None
        hw = handwriting_kind(span_ink, span_text, fm, line_h=lh * S, line_conf=line_conf, blur=_blur)
        if hw:
            # handwriting: redraw in a hand of the same kind (connected cursive vs print), sized on the observed ink;
            # the page's typed consensus style must not replace it
            sub = np.zeros_like(dk); sub[:, a0:a1] = span_dk
            fh = match_font(sub, span_text, max_h=0.9 * lh * S, candidates=hand_candidates(hw == "cursive"))
            if fh is not None:
                fm = dict(fh, handwritten=hw)
        good = [g for g in (style_cache or {}).get("_good", [])]
        if fm is not None and fm["score"] > 0.33 and len(good) >= 2 and not fm.get("handwritten") and "<<" not in line_text:
            # poor match (stray marks, rules, glare): fall back to the page's consensus style -- the best-scoring
            # family/weight seen on this page and the median size relative to the line height
            best = min(good, key=lambda g: g[0])
            rel = float(np.median([g[2] for g in good]))
            fm = dict(fm, font=dict(best[1]), family=dict(best[1])["family"], weight=dict(best[1]).get("weight"),
                      size_px=rel * lh * S, sx=float(np.median([g[3] for g in good])), score=round(fm["score"], 3), consensus=True)
        if style_cache is not None:
            style_cache[fkey] = fm
            if fm is not None and fm["score"] <= 0.25 and not fm.get("consensus"):
                style_cache.setdefault("_good", []).append((fm["score"], tuple(sorted(fm["font"].items())), fm["size_px"] / (lh * S), fm["sx"]))
            if fm is not None and fm["class"] != "hand":
                votes[fm["class"]] = votes.get(fm["class"], 0) + 1
                fvotes[fm["family"]] = fvotes.get(fm["family"], 0) + 1
    if fm is None:
        return None, {"error": "no_ink_found"}
    if style_cache is not None and style_cache.get("_typewriter") and fm["class"] in ("typewriter", "mono", "thermal"):
        # one typewriter, one type size: snap outliers (squeezed/merged OCR boxes) to the page's typed size
        tws = style_cache.setdefault("_tw_rel", [])
        if tws:  # absolute size: one typewriter has one type size on the page
            ref = float(np.median(tws))
            if abs(fm["size_px"] - ref) / ref > 0.12:
                fm = dict(fm, size_px=ref, size_snapped=True)
        if not fm.get("size_snapped"):
            tws.append(fm["size_px"])
        if fm.get("weight") == "bold" and not fm.get("_w_checked"):
            # typed values are regular strikes; heavy-looking ink on a thresholded scan is the threshold, not bold type
            reg = dict(fm["font"], weight="regular")
            if _fc_spec_exists(reg):
                fm = dict(fm, font=reg, weight="regular", _w_checked=True)
    ys = np.where(span_ink.any(axis=1))[0]
    er = np.zeros((Hr, Wr), np.uint8)
    er[:, a0:a1] = (span_dk > 0.08).astype(np.uint8)
    er = cv2.dilate(er, np.ones((3, 3), np.uint8), iterations=max(1, int(round(S * max(1, lh / 14)))))
    er[:, :max(0, a0 - 1)] = 0; er[:, min(Wr, a1 + 1):] = 0
    _rs = rg[:, a0:a1].astype(np.float32)
    sig_est = _blur_from_edges(_rs, span_ink, float(np.percentile(_rs, 95) - np.percentile(_rs, 5))) / S if span_ink.any() else 0.0
    if sig_est > 0.9 and len(ys):
        # soft (blurred/defocused) glyphs: their halo extends past any darkness threshold; erase the full span band
        pad_y = int(round(0.1 * (ys.max() - ys.min() + 1)))  # tight: ID cards stack labels right above values
        er[max(0, ys.min() - pad_y):min(Hr, ys.max() + 1 + pad_y), max(0, a0 - 2):min(Wr, a1 + 2)] = 1
    core = span_dk > 0.7
    pix = rect[:, a0:a1]
    if core.sum() > 3:
        # darkest third of the core: blurred/glare-washed glyphs have few fully-inked pixels, a plain median is too light
        cp = pix[core].astype(np.float32)
        med = np.median(cp, axis=0)
        # same-coloured ink only (a neighbouring blue label or security print must not set the colour)
        same = np.abs((cp - cp.mean(axis=1, keepdims=True)) - (med - med.mean())).max(axis=1) < 25
        cp = cp[same] if same.sum() > 3 else cp
        lum = cp.mean(axis=1)
        # darkest few percent: a blurred redraw rarely reaches full alpha, so a median-of-darkest-fifth ink drew
        # visibly grey values next to black print (D11a/A10c)
        ink_rgb = np.median(cp[lum <= np.percentile(lum, 5)], axis=0)
    else:
        ink_rgb = np.percentile(pix.reshape(-1, 3), 2, axis=0)
    bgsel = (span_dk < 0.03) & ~cv2.dilate(span_ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    bg_rgb = np.median(pix[bgsel], axis=0) if bgsel.sum() > 3 else np.percentile(pix.reshape(-1, 3), 90, axis=0)
    ratio = np.clip((ink_rgb + 1) / (bg_rgb + 1), 0.0, 1.0)
    size = fm["size_px"]; sx = fm["sx"]; spec = tuple(sorted(fm["font"].items()))
    gray_contrast = float(np.mean(bg_rgb) - np.mean(ink_rgb))
    sigma = _blur_from_edges(rg[:, a0:a1].astype(np.float32), span_ink, gray_contrast)
    src_span = line_text[start:end]
    if re.search(r"\)\d", replacement) and re.search(r"\)\d", src_span) and not re.search(r"\s", src_span):
        # OCR often drops the space in '(415) 555-0142'; the printed gap is still there. Keep it in the new number
        # when the span's ink has a word-sized gap (else the redraw reads '(415)555-0164', D11)
        cols = span_ink.any(axis=0); gaps, run = [], 0
        for c_ in cols[int(np.argmax(cols)):]:
            if not c_:
                run += 1
            elif run:
                gaps.append(run); run = 0
        if gaps and max(gaps) >= 0.16 * lh * S:
            replacement = re.sub(r"\)(?=\d)", ") ", replacement, count=1)
            src_span = re.sub(r"\)(?=\d)", ") ", src_span, count=1)
    alpha, _ = render_text_alpha(replacement, spec, size, ss=4)
    ralpha, _ = render_text_alpha(src_span, spec, size, ss=4)
    rb = _ink_box(ralpha); nb = _ink_box(alpha)
    if rb is None or nb is None:
        return None, {"error": "render_failed"}
    alpha = cv2.resize(alpha, (max(1, int(round(alpha.shape[1] * sx))), alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
    orig_w = (a1 - a0)
    new_w = (nb[2] - nb[0]) * sx
    fit = 1.0
    free = 0
    if end >= n:  # blank columns right of the line (raw darkness, core band) available for a longer value
        band = dk_raw[int(my * S):int((my + lh) * S), :] > 0.3
        busy = band.any(axis=0)
        x = min(Wr - 1, a1)
        while x < Wr and not busy[x]:
            x += 1; free += 1
        free = max(0, free - int(lh * S * 0.4))
    room = orig_w * 1.12 + free
    if end < n:
        # never draw into the next word: the room ends one word space (or a hairline before punctuation) left of the
        # first ink after the span ('94612  (510)' -> a wider fake ZIP overprinted the '(' of the phone, A10a)
        band = dk_raw[int(my * S):int((my + lh) * S), :] > 0.3
        busy = band.any(axis=0)
        li = a0 + int(np.argmax(span_ink.any(axis=0)))
        x = min(Wr - 1, max(a1, int(np.where(span_ink.any(axis=0))[0].max()) + a0 + 1))
        while x < Wr and not busy[x]:
            x += 1
        if x < Wr:
            gap_need = lh * S * (0.3 if line_text[end].isspace() else 0.06)
            avail = x - li - gap_need
            room = min(room, max(avail, orig_w * 0.85))
    if new_w > room:
        fit = room / new_w
        alpha = cv2.resize(alpha, (max(1, int(round(alpha.shape[1] * fit))), alpha.shape[0]), interpolation=cv2.INTER_AREA)
    def _style(al):
        if sigma > 0.15:
            al = cv2.GaussianBlur(al, (0, 0), sigma)
            # stroke cores of blurred print still reach full ink: normalise on the 90th percentile, not the peak
            # (the 99th left the whole redraw lighter than the print around it, D11a)
            pk = float(np.percentile(al[al > 0.05], 90)) if (al > 0.05).any() else 1.0
            al = np.clip(al / max(pk, 0.3), 0, 1)
        return al
    alpha = _style(alpha)
    # stroke-weight match: compare ink mass of the original span with the same text rendered in the matched style
    ref = cv2.resize(ralpha, (max(1, int(round(ralpha.shape[1] * sx))), ralpha.shape[0]), interpolation=cv2.INTER_LINEAR)
    m_ref = float(_style(ref).sum()); m_obs = float(span_dk.sum())
    gain = float(np.clip(m_obs / m_ref, 0.75, 2.2)) if m_ref > 1 else 1.0
    if gain < 0.88 and fm.get("weight") == "bold" and not (style_cache is not None and style_cache.get("_typewriter")):
        # the original strokes are clearly lighter than the matched bold: draw the regular face instead of fading bold
        # type to grey (receipt names/card digits came out grey, heavy and soft next to black regular print, D11a/A10c)
        reg = dict(fm["font"], weight="regular")
        if _fc_spec_exists(reg):
            spec_r = tuple(sorted(reg.items()))
            a_r, _ = render_text_alpha(replacement, spec_r, size, ss=4)
            r_r, _ = render_text_alpha(src_span, spec_r, size, ss=4)
            if _ink_box(a_r) is not None and _ink_box(r_r) is not None:
                a_r = cv2.resize(a_r, (max(1, int(round(a_r.shape[1] * sx))), a_r.shape[0]), interpolation=cv2.INTER_LINEAR)
                if fit < 1.0:
                    a_r = cv2.resize(a_r, (max(1, int(round(a_r.shape[1] * fit))), a_r.shape[0]), interpolation=cv2.INTER_AREA)
                ref_r = cv2.resize(r_r, (max(1, int(round(r_r.shape[1] * sx))), r_r.shape[0]), interpolation=cv2.INTER_LINEAR)
                m_ref_r = float(_style(ref_r).sum())
                alpha, ralpha, rb, spec = _style(a_r), r_r, _ink_box(r_r), spec_r
                fm = dict(fm, font=reg, weight="regular")
                gain = float(np.clip(m_obs / m_ref_r, 0.75, 2.2)) if m_ref_r > 1 else 1.0
    if style_cache is not None and style_cache.get("_typewriter"):
        # thresholded pages: ink mass is dominated by the threshold, not stroke weight; keep type at its own weight
        # (too light vanishes under re-thresholding, too heavy blobs)
        gain = float(np.clip(gain, 1.0, 1.25))
    if abs(gain - 1) > 0.08:
        alpha = np.clip(alpha * gain, 0, 1) if gain > 1 else _thin_to_mass(alpha, gain)
    top_offset = ys.min() - rb[1]
    ov_, rv_ = _vext(span_dk, 0.4), _vext(ralpha, 0.4)
    if ov_ is not None and rv_ is not None:
        top_offset = ov_[1] - rv_[1]  # baseline: robust bottom of the original band = robust bottom of the new text
    nb2 = _ink_box(alpha, 0.2) or (0, 0, alpha.shape[1], alpha.shape[0])
    left_ink = a0 + int(np.argmax(span_ink.any(axis=0)))
    tx = left_ink - nb2[0]
    ty = top_offset
    canvas = np.zeros((Hr, Wr), np.float32)
    y0, x0 = int(round(ty)), int(round(tx))
    ah, aw = alpha.shape
    cy0, cx0 = max(0, y0), max(0, x0)
    cy1, cx1 = min(Hr, y0 + ah), min(Wr, x0 + aw)
    if cy1 <= cy0 or cx1 <= cx0:
        return None, {"error": "placement_outside"}
    canvas[cy0:cy1, cx0:cx1] = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    # back to image space
    erase_img = cv2.warpPerspective(er * 255, Mb, (W, H), flags=cv2.INTER_NEAREST) > 0
    alpha_img = cv2.warpPerspective(canvas, Mb, (W, H), flags=cv2.INTER_LINEAR)
    # constrained region: erase mask U new ink, dilated slightly; never outside the expanded quad
    lim = np.zeros((H, W), np.uint8)
    cv2.fillPoly(lim, [eq.astype(np.int32)], 1)
    region = (erase_img | (alpha_img > 0.01))
    region = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) & lim
    region = region > 0
    if protect is not None:
        # other text (e.g. a printed label the typed value overlaps) is neither erased nor overwritten
        erase_img &= ~protect
        region &= ~protect
        alpha_img[protect] = 0
    left = connected_leftovers(src, erase_img & (lim > 0), eq, lh, protect)
    n_left = int(left.sum()) if left is not None else 0
    if left is not None:
        erase_img = erase_img | left
        region = region | left
        lim = lim | left.astype(np.uint8)
    before = img.copy()
    inpaint(img, erase_img.astype(np.uint8) & lim)
    n_ghost = ghost_pass(img, src, erase_img & (lim > 0), lh, protect)
    base = img.astype(np.float32)
    a = alpha_img[..., None]
    out = base * (1 - a * (1 - ratio[None, None, :]))
    img[region] = np.clip(out[region], 0, 255).astype(np.uint8)
    g0 = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    ring = cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    sig = noise_sigma(g0, ring & ~region)
    regrain(img, src, region, rng, sig, jpeg_q)
    img[~region] = before[~region]
    info = {"font": fm["family"], "font_class": fm["class"], "weight": fm["weight"], "font_score": fm["score"],
            "size_px": round(size / S, 1), "hscale": round(sx * fit, 3), "blur_sigma": round(sigma / S, 2),
            "ink_rgb": [int(v) for v in ink_rgb], "noise_sigma": round(sig, 2), "jpeg_q": jpeg_q, "inpaint": inpaint_method(),
            "fit_ratio": round(fit, 3), "weight_gain": round(gain, 2), "glyph_segmentation": bool(seg_exact), "consensus_style": bool(fm.get("consensus")),
            "handwritten": fm.get("handwritten"), "leftover_stroke_px": n_left, "ghost_pass_px": n_ghost}
    return region, info


def _fc_spec_exists(spec):
    try:
        a, _ = render_text_alpha("Ab", tuple(sorted(spec.items())), 20, ss=1)
        return a is not None and a.max() > 0
    except Exception:
        return False


def typewriter_page(page_img, style_cache=None):
    """Thresholded black/white scan (fax/typewriter era): almost no mid-tones on the page."""
    if style_cache is not None and "_typewriter" in style_cache:
        return style_cache["_typewriter"]
    g = page_img if page_img.ndim == 2 else page_img.max(axis=2)
    g = g[::2, ::2]
    v = bool(np.mean((g > 50) & (g < 205)) < 0.12 and np.mean(g < 50) > 0.01)
    if style_cache is not None:
        style_cache["_typewriter"] = v
    return v


def _remove_rules(dk, text_h):
    """Zero long horizontal runs (underlines, table borders) and their anti-aliased fringe."""
    import cv2
    k = max(9, int(2.2 * text_h))
    rules = cv2.morphologyEx((dk > 0.3).astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    if not rules.any():
        return dk
    n, lab, st, _ = cv2.connectedComponentsWithStats(rules)
    thin = np.zeros(n, np.uint8)
    for i in range(1, n):
        if st[i, 3] <= max(3, 0.22 * text_h):  # a rule is thin; a band of glare/pattern is not
            thin[i] = 1
    rules = thin[lab]
    if not rules.any():
        return dk
    th = max(1, int(round(text_h * 0.06)))
    rules = cv2.dilate(rules, np.ones((2 * th + 3, 1), np.uint8))
    return dk * (1 - rules)


def _clean_ink(dk, core0, core1):
    """Keep ink components that sit on the text line (drop security-pattern and neighbouring-line fragments)."""
    import cv2
    b = (dk > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    keep = np.zeros(n, bool)
    lh = core1 - core0
    for i in range(1, n):
        y, h, area = st[i, 1], st[i, 3], st[i, 4]
        ov = max(0, min(y + h, core1) - max(y, core0))
        cy = y + h / 2
        touches = y <= 0 or y + h >= dk.shape[0]
        if area >= 2 and ov >= 0.5 * min(h, lh) and h < 1.5 * lh and core0 + 0.12 * lh <= cy <= core1 - 0.12 * lh and not (touches and h < 0.5 * lh):
            keep[i] = True
    # flat marks above/below the glyph band (dashed rules, underscores of a form line) are not glyphs; hyphens and
    # dashes inside the band are kept
    tall = [i for i in range(1, n) if keep[i] and st[i, 3] >= 0.35 * lh]
    if len(tall) >= 3:
        bt = float(np.median([st[i, 1] for i in tall])); bb = float(np.median([st[i, 1] + st[i, 3] for i in tall]))
        band = max(1.0, bb - bt)
        for i in range(1, n):
            if keep[i] and st[i, 3] < 0.3 * band and st[i, 2] > 1.5 * st[i, 3]:
                cy = st[i, 1] + st[i, 3] / 2
                if cy < bt + 0.08 * band or cy > bb - 0.02 * band:
                    keep[i] = False
    km = keep[lab].astype(np.uint8)
    km = cv2.dilate(km, np.ones((3, 3), np.uint8), iterations=2)
    return dk * km


def _blur_from_edges(g, ink, contrast):
    """Gaussian sigma from edge steepness: max gradient of a blurred step = contrast / (sigma * sqrt(2*pi))."""
    import cv2
    if contrast < 15 or ink.sum() < 10:
        return 0.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    mag = np.hypot(gx, gy)
    near = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if near.sum() < 10:
        return 0.0
    gmax = float(np.percentile(mag[near], 95))
    sig = contrast / (2.5066 * max(gmax, 1e-3))
    return float(np.clip(math.sqrt(max(sig ** 2 - 0.5 ** 2, 0)), 0, 4))


def _estimate_blur(obs_dk, text, spec, size, sx):
    import cv2
    if not text.strip():
        return 0.0
    a, _ = render_text_alpha(text, spec, size, ss=4)
    a = cv2.resize(a, (max(1, int(round(a.shape[1] * sx))), a.shape[0]))
    rb = _ink_box(a); ob = _ink_box(obs_dk, 0.4)
    if rb is None or ob is None:
        return 0.0
    o = obs_dk[ob[1]:ob[3], ob[0]:ob[2]]
    best = (1e9, 0.0)
    for s in (0.0, 0.4, 0.7, 1.0, 1.4, 1.9, 2.5):
        r = cv2.GaussianBlur(a, (0, 0), s) if s else a
        r = r[rb[1]:rb[3], rb[0]:rb[2]]
        r = _resize(r, o.shape[1], o.shape[0])
        # compare edge sharpness distributions (robust to small misalignment)
        go = np.abs(np.diff(o, axis=1)).mean() + np.abs(np.diff(o, axis=0)).mean()
        gr = np.abs(np.diff(r, axis=1)).mean() + np.abs(np.diff(r, axis=0)).mean()
        e = abs(go - gr)
        if e < best[0]:
            best = (e, s)
    return best[1]


# ------------------------------------------------------------------ signatures
def _hpick(seed, items):
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return items[h % len(items)]


def signature_fonts():
    return [p for p in (_fc(f) for f in SIGNATURE_FAMILIES) if p]


def replace_signature(img, bbox, name, seed, rng, jpeg_q=None, quad=None, exclude=None, target_box=None):
    """Erase the strokes inside bbox (restricted to the rotated `quad` when given) and draw a different signature for
    `name` at the same slant, extent, stroke width and ink colour. Returns (mask, info)."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1].copy()
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if g.size == 0 or min(g.shape) < 6:
        return None, {"error": "region_too_small"}
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12) > 0
    # keep printed rules/lines (long straight runs) out of the stroke mask
    rules = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(25, (x1 - x0) // 3)), np.uint8)) > 0
    # thick printed bars/boxes and long rules (in either direction) are form furniture, not strokes
    dark = (g < 110).astype(np.uint8)
    bars = (cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, max(40, (x1 - x0) // 4)), np.uint8)) |
            cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(40, (y1 - y0) // 2), 1), np.uint8)) |
            cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))) > 0
    keep = cv2.dilate((rules | bars).astype(np.uint8), np.ones((5, 3), np.uint8)).astype(bool)
    if exclude is not None:
        keep |= exclude[y0:y1, x0:x1]
    strokes = ink & ~keep
    strokes = cv2.morphologyEx(strokes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0 if strokes.sum() > 200 else strokes
    poly = None
    if quad is not None:
        poly = np.zeros(g.shape, np.uint8)
        cv2.fillPoly(poly, [np.round(np.asarray(quad, np.float32) - [x0, y0]).astype(np.int32)], 1)
        strokes &= poly.astype(bool); ink &= cv2.dilate(poly, np.ones((5, 5), np.uint8)).astype(bool)
    if strokes.sum() < 15:
        return None, {"error": "no_strokes"}
    ys, xs = np.where(strokes)
    sb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    qangle = None
    if quad is not None:
        q = np.asarray(quad, np.float32)
        ux = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1e-6); uy = np.array([-ux[1], ux[0]])
        qangle = float(np.degrees(np.arctan2(ux[1], ux[0])))
        P = np.column_stack([xs, ys]).astype(np.float32)
        pu, pv = P @ ux, P @ uy
        lo_u, hi_u, lo_v, hi_v = np.percentile(pu, 1), np.percentile(pu, 99), np.percentile(pv, 2), np.percentile(pv, 98)
        cu, cv_ = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2
        cxy = cu * ux + cv_ * uy
        sb = (cxy[0] - (hi_u - lo_u) / 2, cxy[1] - (hi_v - lo_v) / 2, cxy[0] + (hi_u - lo_u) / 2, cxy[1] + (hi_v - lo_v) / 2)
    cs_ = crop[strokes].astype(np.float32)
    ink_rgb = np.median(cs_[cs_.mean(axis=1) <= np.percentile(cs_.mean(axis=1), 40)], axis=0)  # pen colour, not AA edge
    dt = cv2.distanceTransform(strokes.astype(np.uint8), cv2.DIST_L2, 3)
    stroke_w = float(2 * np.median(dt[dt > 0])) if (dt > 0).any() else 2.0
    stroke_w = max(stroke_w, 2.4)  # thinner than this does not survive scan-style re-thresholding
    # principal slant of the signature baseline
    pts = np.column_stack([xs, ys]).astype(np.float32)
    angle = 0.0
    if len(pts) > 20:
        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        angle = float(np.degrees(np.arctan2(vy, vx)))
        if abs(angle) > 25:
            angle = 0.0
    if qangle is not None:
        angle = qangle
    if target_box is not None and qangle is None:
        # (a rotated signature keeps its extent measured along its own baseline: an axis-aligned box of a tilted line
        # is far taller than the writing and would blow the new signature up)
        # full extent of the original signature incl. strokes that run over printed text (sizing only)
        sb = (target_box[0] - x0, target_box[1] - y0, target_box[2] - x0, target_box[3] - y0)
    fonts = signature_fonts()
    if not fonts:
        return None, {"error": "no_handwriting_font"}
    # scrawl vs. neat signature: a fast scrawl is compact (tall for its width) and sparse inside its box
    tw0, th0 = sb[2] - sb[0], sb[3] - sb[1]
    density = float(strokes.sum()) / max(1.0, tw0 * th0)
    scrawl = bool(th0 > 0 and (tw0 / th0 < 3.2 or density < 0.075))
    if scrawl:
        messy = [f for f in fonts if any(k in Path(f).stem for k in ("Zeyada", "MrDafoe", "HerrVon", "Cedarville", "LaBelle", "Dawning", "Meddon"))]
        fonts = messy or fonts
    fpath = _hpick(seed + "font", fonts)
    from .resolve import to_latin
    text = to_latin(name or "")  # handwriting fonts are Latin-only; a Greek/Cyrillic name is signed in transliteration
    text = re.sub(r"[^A-Za-z .'-]", "", text).strip().title() or "J. Doe"
    tw, th = sb[2] - sb[0], sb[3] - sb[1]
    f = ImageFont.truetype(fpath, 120)
    bb = f.getbbox(text)
    im = Image.new("L", (bb[2] - bb[0] + 40, bb[3] - bb[1] + 40), 0)
    ImageDraw.Draw(im).text((20 - bb[0], 20 - bb[1]), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    rbx = _ink_box(a)
    if rbx is None:
        return None, {"error": "signature_text_not_renderable"}
    a = a[rbx[1]:rbx[3], rbx[0]:rbx[2]]
    sc = min(tw / a.shape[1], th / a.shape[0])  # fit inside the original signature's own extent
    if scrawl:
        # fill the original's extent (a scrawl is squashed/stretched, not typeset), within limits
        sx_, sy_ = tw / a.shape[1], th / a.shape[0]
        g_ = math.sqrt(sx_ * sy_); sx_ = float(np.clip(sx_, g_ / 1.8, g_ * 1.8)); sy_ = float(np.clip(sy_, g_ / 1.8, g_ * 1.8))
        # the aspect clamp can push one axis past the original's extent: never larger than the original box
        f_ = min(1.0, tw * 1.03 / (a.shape[1] * sx_), th * 1.03 / (a.shape[0] * sy_)); sx_ *= f_; sy_ *= f_
        a = cv2.resize(a, (max(1, int(a.shape[1] * sx_)), max(1, int(a.shape[0] * sy_))), interpolation=cv2.INTER_AREA)
        # hand tremor / speed: smooth random displacement of the strokes
        rs = np.random.default_rng(int(hashlib.sha256((seed + "scrawl").encode()).hexdigest()[:8], 16))
        amp, sm = 0.07 * a.shape[0], max(3.0, a.shape[0] / 3)
        dx = cv2.GaussianBlur(rs.normal(0, 1, a.shape).astype(np.float32), (0, 0), sm); dy = cv2.GaussianBlur(rs.normal(0, 1, a.shape).astype(np.float32), (0, 0), sm)
        dx *= amp / (np.abs(dx).max() + 1e-6); dy *= amp / (np.abs(dy).max() + 1e-6)
        gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        a = np.pad(a, 4); dx = np.pad(dx, 4, mode="edge"); dy = np.pad(dy, 4, mode="edge")
        gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        a = cv2.remap(a, gx + dx, gy + dy, cv2.INTER_LINEAR)
    else:
        a = cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(a.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    # stroke width match
    b = a > 0.5
    if b.any():
        dtn = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 3)
        cur = float(2 * np.median(dtn[dtn > 0])) if (dtn > 0).any() else stroke_w
        k = int(round((stroke_w - cur) / 2))
        if k > 0:
            a = cv2.dilate(a, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        elif k < 0:
            a = cv2.erode(a, np.ones((2 * -k + 1, 2 * -k + 1), np.uint8))
    a = cv2.GaussianBlur(a, (0, 0), max(0.5, stroke_w * 0.25))
    if abs(angle) > 1:
        pd = int(np.hypot(*a.shape) / 2 - min(a.shape) / 2) + 2
        a = np.pad(a, pd)
        M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), -angle, 1.0)
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
    canvas = np.zeros(g.shape, np.float32)
    cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
    ox, oy = int(round(cx - a.shape[1] / 2)), int(round(cy - a.shape[0] / 2))
    ah, aw = a.shape
    yy0, xx0, yy1, xx1 = max(0, oy), max(0, ox), min(g.shape[0], oy + ah), min(g.shape[1], ox + aw)
    canvas[yy0:yy1, xx0:xx1] = a[yy0 - oy:yy1 - oy, xx0 - ox:xx1 - ox]
    canvas[keep & (g < 110)] = 0  # do not double-ink solid printed furniture
    erase = cv2.dilate(strokes.astype(np.uint8), np.ones((5, 5), np.uint8)) & ~keep.astype(np.uint8)
    sub = img[y0:y1, x0:x1]
    before = sub.copy()
    inpaint(sub, erase)
    bg = sub.astype(np.float32)
    bgm = np.median(before[~cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)], axis=0) if (~ink).any() else np.array([235, 235, 235.])
    ratio = np.clip((ink_rgb + 1) / (bgm + 1), 0, 1)
    out = bg * (1 - canvas[..., None] * (1 - ratio[None, None, :]))
    region = (erase > 0) | (canvas > 0.01)
    sub[region] = np.clip(out[region], 0, 255).astype(np.uint8)
    sig = noise_sigma(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), ~region)
    full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = region
    regrain(img, img, full, rng, sig, jpeg_q)
    sub[~region] = before[~region]
    efull = np.zeros((H, W), bool); efull[y0:y1, x0:x1] = erase > 0
    return full, {"font": Path(fpath).stem, "scrawl": scrawl, "stroke_density": round(density, 3), "_erase": efull,
                  "stroke_px": round(stroke_w, 2), "slant_deg": round(angle, 1),
                  "ink_rgb": [int(v) for v in ink_rgb], "inpaint": inpaint_method()}


# ------------------------------------------------------------------ faces
_FACE_BANK = None


def face_bank():
    """Synthetic faces (SFHQ sample grid, MIT/CC0). Returns list of (rgb, landmarks5) or []."""
    global _FACE_BANK
    if _FACE_BANK is not None:
        return _FACE_BANK
    import cv2
    from PIL import Image
    from .models import find_model
    _FACE_BANK = []
    src = os.environ.get("CASEPATCH_FACE_BANK") or find_model("sfhq_sample_4x8.jpg")
    if not src:
        p = Path.home() / ".cache" / "casepatch-real" / "url" / "sfhq_grid.jpg"
        src = p if p.exists() else None
    if not src:
        return _FACE_BANK
    grid = np.asarray(Image.open(src).convert("RGB"))
    H, W = grid.shape[:2]
    tw, th = W / 8, H / 4
    for r in range(4):
        for c in range(8):
            t = grid[int(r * th + th * 0.065):int((r + 1) * th) - 2, int(c * tw) + 2:int((c + 1) * tw) - 2].copy()
            f = detect_faces(t)
            if f:
                _FACE_BANK.append((t, f[0], r * 8 + c))
    return _FACE_BANK


def detect_faces(rgb, thr=0.6):
    import cv2
    from .models import find_model
    p = find_model("face_detection_yunet_2023mar.onnx")
    if p is None or not hasattr(cv2, "FaceDetectorYN"):
        return []
    h, w = rgb.shape[:2]
    det = cv2.FaceDetectorYN.create(str(p), "", (w, h), thr)
    _, faces = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return [] if faces is None else [f for f in faces]


def _face_ellipse(lm, shape, sx=1.0, sy=1.0):
    """Soft mask of the inner face (brows to chin, cheek to cheek) from 5 landmarks (re, le, nose, rm, lm)."""
    import cv2
    lm = np.asarray(lm, np.float32).reshape(5, 2)
    eyes = lm[:2].mean(0); mouth = lm[3:5].mean(0)
    d = float(np.linalg.norm(lm[0] - lm[1]))
    ang = float(np.degrees(np.arctan2(lm[1, 1] - lm[0, 1], lm[1, 0] - lm[0, 0])))
    if lm[1, 0] < lm[0, 0]:
        ang += 180
    c = eyes + (mouth - eyes) * 0.42
    m = np.zeros(shape[:2], np.float32)
    cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))), (max(2, int(d * 0.98 * sx)), max(2, int(d * 1.38 * sy))), ang, 0, 360, 1.0, -1)
    return m, d


# Hand-labelled attributes of the 32 SFHQ sample-grid tiles (row-major): gender f/m, age band, expression,
# realistic (not stylised/child). Used to pick a replacement that fits the mapped fake identity.
SFHQ_ATTRS = {
    0: ("f", "old", "smile", True), 1: ("f", "old", "neutral", True), 2: ("f", "young", "neutral", True), 3: ("m", "young", "neutral", True),
    4: ("f", "young", "neutral", True), 5: ("f", "old", "neutral", True), 6: ("f", "young", "smile", True), 7: ("m", "young", "neutral", True),
    8: ("m", "old", "neutral", True), 9: ("f", "young", "smile", True), 10: ("m", "mid", "smile", True), 11: ("m", "mid", "neutral", True),
    12: ("f", "young", "smile", True), 13: ("m", "mid", "neutral", True), 14: ("m", "old", "smile", True), 15: ("f", "young", "neutral", True),
    16: ("m", "young", "neutral", True), 17: ("m", "child", "neutral", False), 18: ("f", "young", "neutral", False), 19: ("m", "young", "neutral", True),
    20: ("m", "old", "smile", True), 21: ("m", "mid", "neutral", True), 22: ("f", "young", "smile", True), 23: ("f", "old", "neutral", True),
    24: ("f", "young", "smile", True), 25: ("f", "young", "neutral", False), 26: ("m", "young", "neutral", True), 27: ("m", "old", "neutral", True),
    28: ("m", "young", "neutral", True), 29: ("m", "mid", "neutral", True), 30: ("m", "old", "smile", True), 31: ("m", "mid", "neutral", True),
}


# Hand-labelled apparent ethnicity group of the same tiles (coarse; used only to keep a swap plausible for the photo).
SFHQ_ETH = {0: "white", 1: "east_asian", 2: "white", 3: "black", 4: "latin", 5: "east_asian", 6: "white", 7: "southeast_asian",
            8: "white", 9: "latin", 10: "white", 11: "white", 12: "white", 13: "east_asian", 14: "black", 15: "east_asian",
            16: "white", 17: "black", 18: "white", 19: "east_asian", 20: "white", 21: "white", 22: "mixed", 23: "east_asian",
            24: "white", 25: "white", 26: "south_asian", 27: "east_asian", 28: "black", 29: "white", 30: "mixed", 31: "white"}
ETH_NEAR = {"latin": ("latin", "mixed", "white"), "mixed": ("mixed", "latin"), "southeast_asian": ("southeast_asian", "east_asian"),
            "east_asian": ("east_asian", "southeast_asian"), "south_asian": ("south_asian", "latin", "mixed"),
            "middle_eastern": ("latin", "south_asian", "mixed", "white"), "black": ("black", "mixed"), "white": ("white",)}
AGE_NEAR = {"young": ("young",), "mid": ("mid", "young", "old"), "old": ("old", "mid")}


def _face_pools(profile):
    """Candidate-tile filters, strictest first: sex + age band + ethnicity, then relax age, then ethnicity to its
    neighbours, then age neighbours; sex is relaxed last. Unknown attributes do not filter."""
    p = profile or {}
    g = p.get("gender"); band = p.get("age_band") or age_band(p.get("age")); eth = p.get("ethnicity")
    def f(t, g_ok, ages, eths, neutral=True):
        at = SFHQ_ATTRS.get(t, (None, None, "neutral", True))
        return at[3] and (not neutral or at[2] == "neutral") and (not g_ok or g is None or at[0] == g) and \
            (ages is None or at[1] in ages) and (eths is None or SFHQ_ETH.get(t) in eths)
    ages0 = (band,) if band else ("young", "mid")
    ages1 = AGE_NEAR.get(band, ages0) if band else ("young", "mid")
    e0 = (eth,) if eth else None
    e1 = ETH_NEAR.get(eth, (eth,)) if eth else None
    return [lambda t: f(t, True, ages0, e0), lambda t: f(t, True, ages0, e0, False), lambda t: f(t, True, ages1, e0), lambda t: f(t, True, ages0, e1),
            lambda t: f(t, True, ages1, e1), lambda t: f(t, True, None, e1), lambda t: f(t, True, ages1, None),
            lambda t: f(t, False, None, e1), lambda t: f(t, False, None, None)]


def face_match(tile, profile):
    """Does the chosen tile fit the source photo's apparent attributes? -> {sex, age, ethnicity: bool|None}."""
    p = profile or {}; at = SFHQ_ATTRS.get(tile, (None, None)); band = p.get("age_band") or age_band(p.get("age"))
    return {"sex": None if not p.get("gender") else at[0] == p["gender"],
            "age": None if not band else at[1] in AGE_NEAR.get(band, (band,)),
            "ethnicity": None if not p.get("ethnicity") else SFHQ_ETH.get(tile) in ETH_NEAR.get(p["ethnicity"], (p["ethnicity"],))}


def age_band(age):
    if age is None:
        return None
    return "young" if age < 35 else "mid" if age < 55 else "old"


def face_candidates(profile, seed):
    """Eligible tiles for the identity, in a seed-stable order (first = what pick_face would choose)."""
    bank = face_bank()
    for ok in _face_pools(profile):
        pool = [t[2] for t in bank if ok(t[2])]
        if pool:
            first = _hpick(seed + "face", pool)
            rest = sorted((x for x in pool if x != first), key=lambda x: hashlib.sha256(f"{seed}face{x}".encode()).hexdigest())
            return [first] + rest
    return [t[2] for t in bank]


def choose_portrait_tile(img, boxes, profile, seed, rng_seed=0):
    """Several portraits of one person on a page (main + ghost): try eligible tiles in seed order on a scratch copy and
    keep the first whose main and ghost swaps both pass the identity checks (differs from original, ghost reads as the
    new main). Deterministic. Returns (tile, trials)."""
    trials = []
    for t in face_candidates(profile, seed):
        b = img.copy(); rng = np.random.default_rng(rng_seed)
        prof = dict(profile or {}, tile=t)
        r1, i1 = swap_portrait(b, boxes[0], seed, rng, None, prof)
        if r1 is None:
            trials.append({"tile": t, "ok": False}); continue
        e_main = i1.get("_emb")
        ok = bool(i1["identity_check"].get("differs_from_original"))
        for bb in boxes[1:]:
            r2, i2 = swap_portrait(b, bb, seed, rng, None, prof, secondary=True, primary_emb=e_main)
            ic = (i2 or {}).get("identity_check", {})
            ok = ok and r2 is not None and bool(ic.get("differs_from_original")) and (ic.get("cosine_to_main_portrait") or 0) >= SAME_PERSON_COSINE
        trials.append({"tile": t, "ok": ok})
        if ok:
            return t, trials
    return (trials[0]["tile"] if trials else None), trials


def pick_face(profile, seed):
    """Synthetic face for a mapped fake identity: same gender and age band, neutral expression, realistic; the same
    identity seed always gives the same face (so every portrait of one person on a page shows one synthetic person)."""
    bank = face_bank()
    if (profile or {}).get("tile") is not None:  # explicit choice (tests / reviewer)
        return next(t for t in bank if t[2] == profile["tile"])
    for ok in _face_pools(profile):  # unknown age: an ID-photo adult, not elderly
        pool = [t for t in bank if ok(t[2])]
        if pool:
            return _hpick(seed + "face", pool)
    return _hpick(seed + "face", bank)


def replace_face(img, bbox, seed, rng, jpeg_q=None, profile=None, source=None, ghost_scale=None, secondary=False):
    """Swap the inner face inside bbox for a synthetic one: aligned by 5 landmarks (incl. rotation), tone and blur
    matched, Poisson-blended inside a feathered face ellipse, so the original photo frame, background, hair outline,
    card pattern and lighting stay native. Returns (mask, info)."""
    import cv2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 16:
        return None, {"error": "region_too_small"}
    up = max(1.0, 160 / min(crop.shape[:2]))
    cu = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else crop
    faces = detect_faces(cu, 0.5)
    if not face_bank():
        return None, {"error": "no_synthetic_face_bank"}
    if not faces:
        return None, {"error": "no_face_landmarks"}
    f = max(faces, key=lambda z: z[2] * z[3])
    dst = (f[4:14].reshape(5, 2) / up).astype(np.float32)
    if source is not None:  # secondary/ghost portrait: copy of this card's own (already synthetic) main portrait
        syn, sf = source; tile = -1
    else:
        syn, sf, tile = pick_face(profile, seed)
    M, _ = cv2.estimateAffinePartial2D(sf[4:14].reshape(5, 2).astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        return None, {"error": "alignment_failed"}
    ch, cw = crop.shape[:2]
    warped = cv2.warpAffine(syn, M, (cw, ch), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    small = secondary or float(np.linalg.norm(dst[0] - dst[1])) < 22  # ghost/secondary portrait: swap face + hairline
    mask, d = _face_ellipse(dst, crop.shape, *(ghost_scale or ((1.3, 1.25) if small else (1.0, 1.0))))
    sel = mask > 0
    if sel.sum() < 30:
        return None, {"error": "face_too_small"}
    # tone match inside the face ellipse (LAB mean/std) -> handles grey/faded ghost portraits and colour casts
    lab_o = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    mono = max(lab_o[..., 1][sel].std(), lab_o[..., 2][sel].std()) < 4.0   # sepia / grey / faded print
    for k in range(3):
        mo, so = lab_o[..., k][sel].mean(), lab_o[..., k][sel].std() + 1e-3
        ms, ss_ = lab_s[..., k][sel].mean(), lab_s[..., k][sel].std() + 1e-3
        if k == 0 and mono:   # monochrome print luminance: quantile mapping (keeps the print's contrast curve, not just mean/std)
            qs = np.linspace(0, 100, 33)
            src_q = np.percentile(lab_s[..., 0][sel], qs); dst_q = np.percentile(lab_o[..., 0][sel], qs)
            lab_s[..., 0] = np.interp(lab_s[..., 0], np.maximum.accumulate(src_q + np.arange(33) * 1e-4), dst_q)
        elif mono:   # monochrome print: take its tint, drop the synthetic face's own colours (eyes, lips, skin)
            lab_s[..., k] = mo + (lab_s[..., k] - ms) * min(0.15, so / ss_)
        else:
            lab_s[..., k] = (lab_s[..., k] - ms) * min(1.6, so / ss_) + mo
    toned = cv2.cvtColor(np.clip(lab_s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    def sharp(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_32F)[sel].var())
    so_ = sharp(crop)
    sigma = 0.0
    for s_ in (0.5, 0.8, 1.1, 1.5, 2.0, 2.8, 3.6):
        if sharp(cv2.GaussianBlur(toned, (0, 0), s_)) <= so_ * 1.05:
            sigma = s_; break
    if sigma:
        toned = cv2.GaussianBlur(toned, (0, 0), sigma)
    before = crop.copy()
    blended = toned
    try:  # Poisson blend: keeps the original illumination gradient (glare, vignetting) across the face
        mu8 = (cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0).astype(np.uint8) * 255
        ys, xs = np.where(mu8 > 0)
        if len(xs) and xs.min() > 1 and ys.min() > 1 and xs.max() < cw - 2 and ys.max() < ch - 2:
            ctr = (int((xs.min() + xs.max() + 1) // 2), int((ys.min() + ys.max() + 1) // 2))
            blended = cv2.seamlessClone(toned, crop.copy(), mu8, ctr, cv2.NORMAL_CLONE)
    except cv2.error:
        blended = toned
    feather = max(1.5, d * 0.12)
    m = cv2.GaussianBlur(cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=max(1, int(feather))), (0, 0), feather)
    out = crop.astype(np.float32) * (1 - m[..., None]) + blended.astype(np.float32) * m[..., None]
    crop[:] = np.clip(out, 0, 255).astype(np.uint8)
    region = np.zeros((H, W), bool); region[y0:y1, x0:x1] = m > 0.01
    sig = noise_sigma(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY))
    regrain(img, img, region, rng, sig, jpeg_q)
    sub = img[y0:y1, x0:x1]; sub[~region[y0:y1, x0:x1]] = before[~region[y0:y1, x0:x1]]
    return region, {"synthetic_face_source": "SFHQ sample grid (MIT/CC0)", "synthetic_face_tile": int(tile),
                    "face_attrs": list(SFHQ_ATTRS.get(tile, ())[:3]), "profile": profile or {}, "small_portrait": bool(small), "blur_sigma": sigma,
                    "aligned_by": "5 YuNet landmarks (similarity transform)", "blend": "poisson inside feathered face ellipse"}


# ------------------------------------------------------------------ barcodes / QR
def decode_codes(rgb):
    try:
        import zxingcpp
    except ImportError:
        return []
    out = []
    for r in zxingcpp.read_barcodes(rgb):
        p = r.position
        fmt = getattr(r.format, "name", None) or str(r.format).split(".")[-1].replace(" ", "")
        out.append({"format": fmt, "text": r.text,
                    "quad": [(p.top_left.x, p.top_left.y), (p.top_right.x, p.top_right.y), (p.bottom_right.x, p.bottom_right.y), (p.bottom_left.x, p.bottom_left.y)]})
    return out


def replace_code(img, bbox, payload_map, seed, rng, jpeg_q=None):
    """Decode the symbol in bbox; if its payload contains mapped values, regenerate the same symbology at the same
    size/position with mapped payload. Returns (mask or None, info)."""
    import cv2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    codes = decode_codes(crop)
    if not codes:
        up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        codes = [dict(c, quad=[(a / 2, b / 2) for a, b in c["quad"]]) for c in decode_codes(up)]
    if not codes:
        return None, {"error": "undecodable_symbol"}
    c = codes[0]
    new = c["text"]
    for old, rep in sorted(payload_map.items(), key=lambda kv: -len(kv[0])):
        if old and old in new:
            new = new.replace(old, rep)
        digits = "".join(ch for ch in old if ch.isdigit())
        rdig = "".join(ch for ch in rep if ch.isdigit())
        if len(digits) >= 6 and digits in new and len(rdig) == len(digits):
            new = new.replace(digits, rdig)
    if new == c["text"]:
        return None, {"format": c["format"], "kept": "payload has no mapped values"}
    sym = _render_symbol(c["format"], new)
    if sym is None:
        return None, {"error": f"cannot_generate_{c['format']}"}
    q = np.array(c["quad"], np.float32)
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    dark = np.percentile(g, 5); light = np.percentile(g, 95)
    dark_rgb = np.median(crop[g <= dark + 10], axis=0); light_rgb = np.median(crop[g >= light - 10], axis=0)
    sh, sw = sym.shape
    if c["format"] in ("QRCode", "MicroQRCode", "DataMatrix", "Aztec"):
        srcq = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
        dstq = q
    else:  # 1D: zxing gives a line across the bars; use the region's bar extent
        xs = q[:, 0]; ink = (g < (dark + light) / 2)
        cols = np.where(ink.mean(axis=0) > 0.15)[0]; rows = np.where(ink[:, cols].mean(axis=1) > 0.2)[0] if len(cols) else []
        if not len(cols) or not len(rows):
            return None, {"error": "bar_extent"}
        bx0, bx1, by0, by1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
        srcq = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
        dstq = np.float32([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
    M = cv2.getPerspectiveTransform(srcq, dstq)
    ink = cv2.warpPerspective(sym.astype(np.float32), M, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_AREA)
    cover = cv2.warpPerspective(np.ones_like(sym, np.float32), M, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_NEAREST) > 0
    cover = cv2.dilate(cover.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    before = crop.copy()
    out = light_rgb[None, None] * (1 - ink[..., None]) + dark_rgb[None, None] * ink[..., None]
    sig = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 0.5)
    crop[cover] = np.clip(sig[cover], 0, 255).astype(np.uint8)
    region = np.zeros((H, W), bool); region[y0:y1, x0:x1] = cover
    regrain(img, img, region, rng, noise_sigma(g), jpeg_q)
    sub = img[y0:y1, x0:x1]; sub[~cover] = before[~cover]
    return region, {"format": c["format"], "payload_changed": True, "payload_len": len(new)}


def _render_symbol(fmt, text):
    """Return float 0..1 ink map (1 = dark module)."""
    try:
        if fmt in ("QRCode", "MicroQRCode"):
            import segno
            q = segno.make(text, micro=(fmt == "MicroQRCode"), error="m")
            m = np.array(q.matrix, np.float32)
            return np.kron(m, np.ones((8, 8), np.float32))
        import zxingcpp
        f = getattr(zxingcpp.BarcodeFormat, fmt)
        if hasattr(zxingcpp, "create_barcode"):
            b = zxingcpp.create_barcode(text, f)
            im = np.asarray(b.to_image(scale=4, add_quiet_zones=False), np.float32)
        else:
            im = np.asarray(zxingcpp.write_barcode(f, text, width=0, height=0, quiet_zone=0), np.float32)
        return 1.0 - im / 255.0
    except Exception:
        return None


def swap_portrait(img, bb, seed, rng, jpeg_q=None, profile=None, scales=((1.3, 1.25), (1.55, 1.5), (1.8, 1.7), (2.1, 2.0), (2.4, 2.3)),
                  secondary=False, primary_emb=None):
    """replace_face plus the identity check. Secondary/ghost portraits: widen the swap until the result no longer
    reads as the original person AND reads as the page's new main portrait (margins on both SFace thresholds);
    otherwise keep the attempt that best satisfies both."""
    x0, y0, x1, y1 = [int(round(v)) for v in bb]
    H, W = img.shape[:2]
    sl = (slice(max(0, y0), y1), slice(max(0, x0), x1))
    pad = (slice(max(0, y0 - 6), min(H, y1 + 6)), slice(max(0, x0 - 6), min(W, x1 + 6)))  # the scorer's crop
    before_pad = img[pad].copy()
    e0 = face_embedding(before_pad)
    best = None
    for gs in (scales if secondary else ((1.0, 1.0), (1.15, 1.12), (1.3, 1.25))):
        img[pad] = before_pad
        region, info = replace_face(img, bb, seed, rng, jpeg_q, profile=profile, ghost_scale=gs, secondary=secondary)
        if region is None:
            return None, info
        info["ghost_scale" if secondary else "face_scale"] = list(gs)
        if not secondary and not info.get("small_portrait"):
            so = face_similarity(e0, face_embedding(img[pad]))
            bad = 0.0 if so is None else max(0.0, so - (SAME_PERSON_COSINE - 0.12))
            if best is None or bad < best[0]:
                best = (bad, img[pad].copy(), region, info)
            if bad == 0:
                break
            continue
        e1 = face_embedding(img[pad])
        so, sp = face_similarity(e0, e1), face_similarity(primary_emb, e1) if primary_emb is not None else None
        if so is None:
            continue
        # badness: how far each criterion is from its margin
        bad = max(0.0, so - (SAME_PERSON_COSINE - 0.12)) + (max(0.0, (SAME_PERSON_COSINE + 0.06) - sp) if sp is not None else 0.0)
        if best is None or bad < best[0]:
            best = (bad, img[pad].copy(), region, info)
        if bad == 0:
            break
    if best is None:
        img[pad] = before_pad
        return None, {"error": "no_face_after_swap"}
    _, px, region, info = best
    img[pad] = px
    e1 = face_embedding(img[pad])
    sim = face_similarity(e0, e1)
    info["identity_check"] = {"face_detected_after": e1 is not None, "cosine_to_original": None if sim is None else round(sim, 3),
                              "differs_from_original": None if sim is None else sim < SAME_PERSON_COSINE}
    if primary_emb is not None:
        sp = face_similarity(primary_emb, e1)
        info["identity_check"]["cosine_to_main_portrait"] = None if sp is None else round(sp, 3)
    info["_emb"] = e1
    return region, info


# ------------------------------------------------------------------ face identity check (OpenCV SFace, Apache-2.0)
_SFACE = None


def face_embedding(rgb, face_row=None):
    """128-d SFace embedding of the largest (or given) YuNet face in rgb, or None when no model/face."""
    global _SFACE
    import cv2
    from .models import find_model
    if _SFACE is None:
        p = find_model("face_recognition_sface_2021dec.onnx")
        if not p:
            return None
        _SFACE = cv2.FaceRecognizerSF.create(str(p), "")
    if face_row is None:
        up = max(1.0, 200 / min(rgb.shape[:2]))
        big = cv2.resize(rgb, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else rgb
        fs = detect_faces(big, 0.5)
        if not fs:
            return None
        face_row = max(fs, key=lambda z: z[2] * z[3]); rgb = big
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    al = _SFACE.alignCrop(bgr, np.asarray(face_row, np.float32).reshape(1, -1))
    return _SFACE.feature(al).ravel().copy()


def face_similarity(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


SAME_PERSON_COSINE = 0.363  # OpenCV SFace published threshold


def _thin_to_mass(alpha, gain):
    """Less ink without greying: raise the alpha floor so stroke edges recede while stroke cores keep full ink, until
    the ink mass is `gain` times the original (multiplying alpha by gain drew black print as grey, D11a/A10c)."""
    total = float(alpha.sum())
    if total <= 0:
        return alpha
    target, lo, hi = gain * total, 0.0, 0.9
    for _ in range(18):
        t = (lo + hi) / 2
        if float(np.clip((alpha - t) / (1 - t), 0, 1).sum()) > target:
            lo = t
        else:
            hi = t
    t = (lo + hi) / 2
    return np.clip((alpha - t) / (1 - t), 0, 1).astype(alpha.dtype)
