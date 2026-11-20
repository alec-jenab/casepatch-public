"""Per-document state + the tool implementations the agent calls. Every pixel edit goes through a tool, so the set of
changed regions is known exactly (used by the verifier's facts-unchanged check) and each edit can be undone."""
from __future__ import annotations

import base64, difflib, io, json, re, threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
VIEW_MAX = 1568


_FOLD = {**{c: '"' for c in '“”„″«»'}, **{c: "'" for c in '‘’‚′`'},
         **{c: "-" for c in '–—‐‑−'}}
_GROUP = {'"': '"“”„″«»', "'": "'‘’‚′`", "-": "-–—‐‑−"}


def _fold(ch):
    return _FOLD.get(ch, ch)


def _char_re(ch):
    g = _GROUP.get(_fold(ch))
    return "[" + re.escape(g) + "]" if g else re.escape(ch)


def _match_quotes(got, rep):
    """Typed straight quotes in a replacement take the curly forms the PDF itself uses (open after space/start/bracket)."""
    out = []
    for k, ch in enumerate(rep):
        f = _fold(ch)
        if f in ('"', "'") and ch in ('"', "'"):
            opening = k == 0 or rep[k - 1].isspace() or rep[k - 1] in "([{"
            o, c = ("\u201c", "\u201d") if f == '"' else ("\u2018", "\u2019")
            want = o if opening else c
            if want in got:
                ch = want
        out.append(ch)
    return "".join(out)


_FB_CACHE = {}


def _fallback_font(serif, bold, italic):
    """Metric-compatible full-coverage face (Liberation Serif/Sans = Times/Arial metrics) as bytes; None -> base-14."""
    key = (serif, bold, italic)
    if key not in _FB_CACHE:
        from ..docs.fonts import BUNDLED, find
        style = ("Bold " if bold else "") + ("Italic" if italic else "") or "Regular"
        fam = "Liberation Serif" if serif else "Liberation Sans"
        try:
            f = find(fam, style.strip())
            if not f and not italic:  # bundled upright faces on systems without Liberation
                f = str(BUNDLED / f"Liberation{'Serif' if serif else 'Sans'}-{'Bold' if bold else 'Regular'}.ttf")
            _FB_CACHE[key] = Path(f).read_bytes() if f and "liberation" in f.lower() else None
        except Exception:
            _FB_CACHE[key] = None
    return _FB_CACHE[key]


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _bbox_of_chars(chars):
    from ..docs.model import quad_bbox
    bs = [quad_bbox(c.quad) for c in chars if c.c.strip()] or [quad_bbox(c.quad) for c in chars]
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def find_span(text, value):
    """(start, end) of value in text: exact, case-insensitive, else best fuzzy window (ratio >= 0.75)."""
    i = text.find(value)
    if i >= 0:
        return i, i + len(value)
    i = text.lower().find(value.lower())
    if i >= 0:
        return i, i + len(value)
    n, best = len(value), (0.0, None)
    for L in range(max(1, n - 3), n + 4):
        for k in range(0, max(1, len(text) - L + 1)):
            r = difflib.SequenceMatcher(None, value.lower(), text[k:k + L].lower()).ratio()
            if r > best[0]:
                best = (r, (k, k + L))
    return best[1] if best[0] >= 0.75 else None


_FONT_LOCK = threading.RLock()
HAND_FONTS = ("Caveat", "Kalam", "Patrick Hand", "Indie Flower", "Reenie Beanie", "Shadows Into Light", "Nothing You Could Do")
FONT_CLASSES = ("typewriter", "mono", "serif", "sans", "sans-narrow", "hand", "thermal")


class _restrict_fonts:
    """When the agent names the type class it sees, the font matcher only considers families of that class (handwriting
    detection inside the editor still uses its own hand candidates). Global patch, so serialised with a lock."""

    def __init__(self, cls):
        self.cls = cls if cls in FONT_CLASSES else None

    def __enter__(self):
        if self.cls:
            from ..docs import native as N
            _FONT_LOCK.acquire()
            self.saved = N.font_candidates
            allowed = {"typewriter": ("typewriter", "mono"), "mono": ("mono", "typewriter")}.get(self.cls, (self.cls,))
            full = self.saved(True)
            keep = tuple(c for c in full if dict(c)["class"] in allowed) or full
            N.font_candidates = lambda include_hand=True, _k=keep: _k
        return self

    def __exit__(self, *a):
        if self.cls:
            from ..docs import native as N
            N.font_candidates = self.saved
            _FONT_LOCK.release()


@dataclass
class Edit:
    id: int
    kind: str
    page: int
    bbox: list
    original: str
    replacement: str
    mask: object = None  # bool HxW (raster) or None (pdf text edit)
    note: str = ""
    undone: bool = False
    meta: dict = None


@dataclass
class DocState:
    """One input document. Raster pages are held as RGB arrays (orig + current); PDFs with a text layer are edited
    natively with PyMuPDF and raster-only pages fall back to page images."""
    path: Path
    name: str
    kind: str                                   # "image" | "pdf"
    orig: list = field(default_factory=list)    # per page RGB uint8
    cur: list = field(default_factory=list)
    ocr: dict = field(default_factory=dict)     # page -> [Line]
    pdf: object = None                          # fitz.Document being edited (pdf kind)
    native_text: list = field(default_factory=list)  # per page: does the page carry a real text layer?
    edits: list = field(default_factory=list)
    flags: list = field(default_factory=list)   # review items recorded by tools (e.g. faces kept by policy)
    faces: str = "swap"                          # "swap" | "keep"
    lock: object = field(default_factory=threading.Lock)

    @classmethod
    def load(cls, path, dpi=200):
        from PIL import Image, ImageOps
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            import fitz
            d = fitz.open(path)
            st = cls(path, path.name, "pdf", pdf=d)
            for pg in d:
                pix = pg.get_pixmap(dpi=dpi)
                a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
                st.orig.append(a); st.cur.append(a.copy())
                st.native_text.append(len(pg.get_text("text").strip()) > 20)
            st.scale = dpi / 72.0
            return st
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        a = np.array(im)
        st = cls(path, path.name, "image", orig=[a], cur=[a.copy()], native_text=[False])
        st.scale = 1.0
        return st

    # ------------------------------------------------------------------ reading
    def lines(self, page=0):
        with self.lock:
            if page not in self.ocr:
                from ..docs.extract import ocr_lines
                self.ocr[page] = [l for l in ocr_lines(self.orig[page], f"p{page}:") if l.text.strip()]
            return self.ocr[page]

    def line(self, page, line_id):
        for l in self.lines(page):
            if l.id == line_id or l.id.split(":")[-1] == str(line_id):
                return l
        return None

    def ocr_boxes(self, page=0, contains=None):
        from ..docs.render import _sample_colors
        out = []
        for l in self.lines(page):
            if contains and contains.lower() not in l.text.lower():
                continue
            bb = [round(v) for v in _bbox_of_chars(l.chars)] if l.chars else None
            hs = [abs(c.quad[3][1] - c.quad[0][1]) for c in l.chars if c.c.strip()]
            _, ink, _ = _sample_colors(self.orig[page], bb) if bb else (None, np.zeros(3), 0)
            out.append({"id": l.id.split(":")[-1], "text": l.text, "bbox": bb, "h": round(float(np.median(hs)), 1) if hs else None,
                        "ink": "#%02x%02x%02x" % tuple(int(x) for x in ink), "conf": round(l.conf, 2)})
        return out

    def view(self, page=0, bbox=None, which="current", max_side=VIEW_MAX):
        from PIL import Image
        a = (self.cur if which != "original" else self.orig)[page]
        H, W = a.shape[:2]
        info = f"{self.name} page {page} ({which}) {W}x{H}px"
        if bbox:
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("empty crop")
            a = a[y0:y1, x0:x1]; info += f", crop [{x0},{y0},{x1},{y1}]"
        im = Image.fromarray(a)
        s = min(1.0, min(int(max_side or VIEW_MAX), 2000) / max(im.size))
        if s < 1:
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            info += f", shown at scale {s:.3f} (multiply shown px by {1 / s:.2f} for page px)"
        b = io.BytesIO(); im.save(b, "PNG")
        return info, base64.b64encode(b.getvalue()).decode()

    # ------------------------------------------------------------------ editing
    def _record(self, kind, page, bbox, original, replacement, mask, note=""):
        e = Edit(len(self.edits) + 1, kind, page, [int(round(v)) for v in bbox], original, replacement, mask, note)
        self.edits.append(e)
        return e

    def style_cache(self, page, font_class=None):
        c = self.__dict__.setdefault("_styles", {}).setdefault((page, font_class), {})
        if font_class and "_class_votes" not in c:
            # the agent saw the page: seed the page-level style prior with its read of the type class
            c["_class_votes"] = {font_class: 10}
            if font_class in ("typewriter", "mono"):
                c["_typewriter"] = True
        return c

    def replace_text(self, page, original, replacement, line_id=None, bbox=None, seed=0, font_class=None):
        """Font-matched in-place replacement. Either line_id (+ original as it appears in that OCR line) or bbox
        (tight box around exactly the original text). Mask-only composite: pixels outside the edit mask never change."""
        from ..docs import native as N
        from ..docs.model import quad_from_rect
        if self.kind == "pdf" and self.native_text[page]:
            return self.pdf_replace(page, original, replacement)
        img = self.cur[page]
        if bbox:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            quad, text, s, e = quad_from_rect(x0, y0, x1, y1), original, 0, len(original)
        else:
            l = self.line(page, line_id) if line_id is not None else next((l for l in self.lines(page) if find_span(l.text, original)), None)
            if l is None:
                return {"ok": False, "error": "line not found; pass line_id from ocr_boxes or a tight bbox around the text"}
            sp = find_span(l.text, original)
            if sp is None:
                return {"ok": False, "error": f"'{original}' not found in line text '{l.text}'; pass bbox instead"}
            s, e = sp
            if len(l.chars) == len(l.text):  # per-char quads: edit from the span's start to the line end (keeps the right
                # context so the editor does not grow into the next word), span indices relative to that substring
                quad, text, e, s = quad_from_rect(*_bbox_of_chars(l.chars[s:])), l.text[s:], e - s, 0
            else:
                text = l.text; quad = quad_from_rect(*_bbox_of_chars(l.chars))
        work = img.copy()
        try:
            with _restrict_fonts(font_class):
                mask, info = N.replace_text(work, np.array(quad, np.float32), text, s, e, replacement, np.random.default_rng(seed + len(self.edits)),
                                            style_cache=self.style_cache(page, font_class), orig_img=self.orig[page])
        except Exception as ex:  # editor failure is a tool result
            return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        if mask is None:
            return {"ok": False, "error": info.get("error", "edit failed")}
        img[mask] = work[mask]
        ys, xs = np.nonzero(mask)
        bb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        ed = self._record("replace", page, bb, original, replacement, mask, json.dumps({k: str(v)[:40] for k, v in (info or {}).items()})[:200])
        return {"ok": True, "edit_id": ed.id, "changed_bbox": bb, "font": str((info or {}).get("font", ""))[:60]}

    def draw_text(self, page, bbox, original, text, font, size_px=None, ink=None, condense_to_fit=False, seed=0, fit_width=True, rotation=0):
        """Erase the ink inside bbox (tight box around the ORIGINAL text) and typeset `text` in a named font, sized so the
        original text in that font would have the observed ink height, left/baseline aligned, ink colour sampled, softened
        and grained like the scan. Composited multiplicatively over the paper (paper texture survives; no white boxes)."""
        if rotation:
            return self._upright(page, rotation, bbox, lambda bb: self.draw_text(page, bb, original, text, font, size_px, ink, condense_to_fit, seed, fit_width, 0))
        if self.kind == "pdf" and self.native_text[page] and original and not original.startswith("("):
            r = self.pdf_replace(page, original, text)   # native text layer: keep the page as text, not a raster
            if r.get("ok"):
                return r
            import fitz
            x0, y0, x1, y1 = [float(v) / self.scale for v in bbox]
            layer = self.pdf[page].get_text("text", clip=fitz.Rect(x0, y0, x1, y1)).strip()
            if layer:
                return {"ok": False, "text_layer": layer[:400], "error": r.get("error", "") + ". This box is on the PDF text "
                        "layer: a pixel edit would turn the whole page into an image. Call again with `original` copied from "
                        "`text_layer` (the PDF's own characters; OCR may differ), one line at a time."}
        import cv2
        from PIL import Image, ImageDraw
        from ..docs import native as N
        from ..docs.render import _sample_colors
        img = self.cur[page]; H, W = img.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {"ok": False, "error": "bbox too small"}
        path = N._fc(font, "Bold" if font.endswith(" Bold") else "Regular") if font else None
        fam = font[:-5] if font and font.endswith(" Bold") else font
        if not path:
            path = N._fc(fam, "Bold" if font.endswith(" Bold") else "Regular") if fam else None
        if not path:
            return {"ok": False, "error": f"font '{font}' not installed; pick one from the list in the tool description"}
        g = cv2.cvtColor(self.orig[page][y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        bgv = cv2.medianBlur(g, min(31, (min(g.shape) // 2) * 2 - 1) if min(g.shape) > 4 else 3)
        inkm = (bgv.astype(np.int16) - g.astype(np.int16)) > 25
        rows = np.nonzero(inkm.mean(axis=1) > 0.03)[0]; cols = np.nonzero(inkm.mean(axis=0) > 0.03)[0]
        if not len(rows):
            return {"ok": False, "error": "no ink in bbox; is bbox on the original text?"}
        oy0, oy1, ox0, ox1 = y0 + rows.min(), y0 + rows.max() + 1, x0 + cols.min(), x0 + cols.max() + 1
        if ink is None:
            _, ink_c, _ = _sample_colors(self.orig[page], [x0, y0, x1, y1]); ink_c = np.array(ink_c, np.float32)
            ink_c = np.minimum(ink_c, np.percentile(self.orig[page][oy0:oy1, ox0:ox1].reshape(-1, 3), 3, axis=0))
        else:
            ink_c = np.array([int(ink.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
        from PIL import ImageFont
        ss = 4

        def render(t, size, track=0.0):
            f = ImageFont.truetype(path, max(4, int(round(size * ss))))
            l, t_, r, b = f.getbbox(t)
            extra = int(max(0.0, track) * ss * max(0, len(t) - 1))
            im = Image.new("L", (r - l + 8 * ss + extra, b - t_ + 8 * ss), 0)
            d = ImageDraw.Draw(im)
            if not track:
                d.text((4 * ss - l, 4 * ss - t_), t, font=f, fill=255)
            else:  # letter-spacing: advance each glyph by its own width plus the tracking
                x = 4 * ss - l
                for ch in t:
                    d.text((x, 4 * ss - t_), ch, font=f, fill=255)
                    x += f.getlength(ch) + track * ss
            return np.array(im), t_ / ss, l / ss
        if size_px is None:
            a0, _, _ = render(original or text, 100)
            r0 = np.nonzero((a0 > 80).any(axis=1))[0]; c0 = np.nonzero((a0 > 80).any(axis=0))[0]
            by_h = 100 * (oy1 - oy0) / max(1.0, (r0.max() - r0.min() + 1) / ss)
            by_w = 100 * (ox1 - ox0) / max(1.0, (c0.max() - c0.min() + 1) / ss)
            # width of the same text is the more robust cue (speckle inflates height); height bounds it
            size_px = by_w if original and len(original) >= 3 else by_h
            size_px = float(np.clip(size_px, 0.75 * by_h, 1.25 * by_h))
        size_px = float(size_px)
        # baseline: the original's top-of-ink offset in this font at this size anchors the replacement
        f = ImageFont.truetype(path, max(4, int(round(size_px * ss))))
        top_o = f.getbbox(original or text)[1] / ss
        top_r = f.getbbox(text)[1] / ss
        def down(arr):
            return cv2.resize(arr, (max(1, arr.shape[1] // ss), max(1, arr.shape[0] // ss)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        hand = fam in self.SIGNATURE_FONTS or fam in HAND_FONTS
        a = down(render(text, size_px)[0])
        # stroke weight: thicken until the original text in this font is as inky as the scan's original (heavy
        # typewriter ribbons, bold stamps); never thinner than the font
        obs_d = float(inkm[rows.min():rows.max() + 1, cols.min():cols.max() + 1].mean())
        ref = down(render(original or text, size_px)[0]) > 0.5
        best_k, best_err = 0, 1e9
        for k in range(0, 5):
            rr = cv2.dilate(ref.astype(np.uint8), np.ones((k + 1, k + 1), np.uint8)) if k else ref.astype(np.uint8)
            ys_, xs_ = np.nonzero(rr)
            dd = float(rr[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1].mean()) if len(ys_) else 0
            if abs(dd - obs_d) < best_err:
                best_k, best_err = k, abs(dd - obs_d)
        def ink_w(arr):
            c = np.nonzero((arr > 0.3).any(axis=0))[0]
            return (c.max() - c.min() + 1) if len(c) else arr.shape[1]
        # length fit: the fake should occupy the original's width (no 'WONG YEE   ,' gaps, no overrun into the next
        # word). Typed text: letter-spacing (tracking) first, then a mild horizontal scale. Hand/signature fonts: scale only
        # (tracking breaks joined script).
        target = float(ox1 - ox0)
        if condense_to_fit:
            target = min(target, float(x1 - ox0))
        wr = float(ink_w(a)); fit = {"from_width": int(wr), "target_width": int(target)}
        if fit_width and wr > 0 and target > 0 and abs(target / wr - 1) > 0.06:
            if not hand and target > wr and len(text) > 1:
                track = min((target - wr) / (len(text) - 1), 0.35 * size_px)
                a = down(render(text, size_px, track)[0]); fit["tracking_px"] = round(track, 1)
            elif not hand and target < wr and len(text) > 1:
                track = max((target - wr) / (len(text) - 1), -0.06 * size_px)
                a = down(render(text, size_px, track)[0]) if track < -0.3 else a; fit["tracking_px"] = round(track, 1)
            wr2 = float(ink_w(a))
            sx = float(np.clip(target / max(1.0, wr2), 0.75 if hand else 0.85, 1.3 if hand else 1.0))
            if abs(sx - 1) > 0.02:
                a = cv2.resize(a, (max(1, int(round(a.shape[1] * sx))), a.shape[0]), interpolation=cv2.INTER_AREA); fit["scale_x"] = round(sx, 3)
            fit["final_width"] = int(ink_w(a))
        if best_k:
            a = cv2.dilate(a, np.ones((best_k + 1, best_k + 1), np.uint8))
        pad = 4
        rng = np.random.default_rng(seed + len(self.edits))
        sig = float(np.clip((oy1 - oy0) / 45, 0.4, 1.2))
        a = cv2.GaussianBlur(a, (0, 0), sig)
        a = np.clip(a * (0.85 + 0.25 * rng.random(a.shape)), 0, 1)
        # erase first (ink only, dilated), then composite
        pd = max(2, int((oy1 - oy0) * 0.12))  # OCR boxes clip the first/last stroke: erase a little wider
        e = self.erase(page, [x0 - pd, y0, x1 + pd, y1], ink_only=True)
        er = self.edits.pop() if e.get("ok") else None
        ty = int(round(oy0 + (top_r - top_o))) - pad; tx = int(round(ox0)) - pad
        Hh, Ww = a.shape
        X0, Y0, X1, Y1 = max(0, tx), max(0, ty), min(W, tx + Ww), min(H, ty + Hh)
        if X1 <= X0 or Y1 <= Y0:
            return {"ok": False, "error": "text falls outside the page"}
        al = a[Y0 - ty:Y1 - ty, X0 - tx:X1 - tx][..., None]
        reg = img[Y0:Y1, X0:X1].astype(np.float32)
        al = np.where(al > 0.02, al, 0.0)
        img[Y0:Y1, X0:X1] = np.where(al > 0, np.clip(reg * (1 - al) + ink_c * al, 0, 255), reg).astype(np.uint8)
        mask = er.mask.copy() if er is not None else np.zeros((H, W), bool)
        mask[Y0:Y1, X0:X1] |= al[..., 0] > 0.02
        ys, xs = np.nonzero(mask)
        bb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        ed = self._record("draw", page, bb, original, text, mask, f"{font} {size_px:.1f}px")
        overflow = int(X0 + (np.nonzero((al[..., 0] > 0.3).any(axis=0))[0].max() if (al > 0.3).any() else 0) - x1)
        miss = fit.get("final_width", wr) / max(1.0, target) - 1
        return {"ok": True, "edit_id": ed.id, "changed_bbox": bb, "size_px": round(size_px, 1), "stroke_boost": best_k, "fit": fit,
                **({"length_warning": f"the fake is {abs(miss):.0%} {'shorter' if miss < 0 else 'longer'} than the original even after spacing; "
                                      f"pick a fake of closer length"} if abs(miss) > 0.12 else {}),
                **({"warning": f"replacement runs {overflow}px past the original's right edge; check it doesn't touch the next word"} if overflow > 6 else {})}

    def erase(self, page, bbox, ink_only=True, grow=True, rotation=0):
        """Remove ink inside bbox (inpaint from the surrounding paper). ink_only keeps paper texture between glyphs.
        grow: also remove strokes of the same ink blobs that stick out of the box (a handwritten capital's loop, a
        descender) -- whole connected strokes that are mostly inside the box -- but never form rules or neighbouring words."""
        if rotation:
            return self._upright(page, rotation, bbox, lambda bb: self.erase(page, bb, ink_only, grow, 0))
        if self.kind == "pdf" and self.native_text[page]:
            r = self.pdf_erase(page, bbox)
            if r is not None:
                return r
        import cv2
        from ..docs import native as N
        img = self.cur[page]; H, W = img.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return {"ok": False, "error": "empty bbox"}
        m = np.zeros((H, W), np.uint8)
        if ink_only:
            g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            k = max(3, (min(g.shape) // 2) * 2 - 1); k = min(k, 31)
            bg = cv2.medianBlur(g, k if k % 2 else k + 1)
            ink = (bg.astype(np.int16) - g.astype(np.int16)) > 12
            ink = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8))
            m[y0:y1, x0:x1] = ink
            if grow:
                m |= self._leftover_strokes(page, [x0, y0, x1, y1])
        else:
            m[y0:y1, x0:x1] = 1
        before = img.copy()
        # classic diffusion fill from the surrounding paper (per bbox crop); learned inpainting brightened scanned paper
        cpad = 16
        CX0, CY0, CX1, CY1 = max(0, x0 - cpad), max(0, y0 - cpad), min(W, x1 + cpad), min(H, y1 + cpad)
        img[CY0:CY1, CX0:CX1] = cv2.inpaint(np.ascontiguousarray(img[CY0:CY1, CX0:CX1]), m[CY0:CY1, CX0:CX1], 7, cv2.INPAINT_TELEA)
        # match the paper: inpainting comes out smoother/brighter than scanned paper, which reads as a pale ghost of
        # the erased strokes. Shift the filled pixels to the local paper level and add the scan's grain back.
        sel = m > 0
        if sel.any():
            pad = 12
            X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
            ring = np.zeros_like(sel); ring[Y0:Y1, X0:X1] = True; ring &= ~cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool)
            src = self.orig[page]
            if ring.sum() > 50:
                paper = np.median(src[ring].astype(np.float32), axis=0)
                filled = img[sel].astype(np.float32)
                img[sel] = np.clip(filled - np.median(filled, axis=0) + paper, 0, 255).astype(np.uint8)
                # faint-ghost cleanup: diffusion keeps a low-contrast copy of the strokes (darker halo pixels feed the fill).
                # Lift every filled pixel that is still darker than the local paper back to the paper level (the paper
                # level is a smooth estimate from the ring, so stains/gradients across the box survive), then regrain.
                crop = img[Y0:Y1, X0:X1].astype(np.float32)
                ringc = ring[Y0:Y1, X0:X1]
                bgc = crop.copy(); bgc[~ringc] = 0
                wsum = cv2.GaussianBlur(ringc.astype(np.float32), (0, 0), max(4.0, (y1 - y0) * 0.6))
                local = cv2.GaussianBlur(bgc, (0, 0), max(4.0, (y1 - y0) * 0.6)) / np.maximum(wsum[..., None], 1e-3)
                local[wsum < 1e-3] = paper
                selc = sel[Y0:Y1, X0:X1]
                lum = crop.mean(axis=2); lum_bg = local.mean(axis=2)
                lift = np.where(selc & (lum < lum_bg - 2), lum_bg - lum, 0)[..., None]
                img[Y0:Y1, X0:X1] = np.clip(crop + lift, 0, 255).astype(np.uint8)
                gs = cv2.cvtColor(src[Y0:Y1, X0:X1], cv2.COLOR_RGB2GRAY)
                sig = N.noise_sigma(gs, ring[Y0:Y1, X0:X1])
                N.regrain(img, src, sel, np.random.default_rng(len(self.edits) + 11), sig)
        mask = (img != before).any(axis=2) | (m > 0)
        ed = self._record("erase", page, [x0, y0, x1, y1], "", "", mask)
        return {"ok": True, "edit_id": ed.id, "changed_bbox": [x0, y0, x1, y1]}

    def _leftover_strokes(self, page, bbox):
        """uint8 mask of original-ink connected components that intersect bbox, lie mostly (>=50%) inside it with their middle inside its rows, and are not
        long thin rules; dilated like the main erase mask."""
        import cv2
        src = self.orig[page]; H, W = src.shape[:2]
        x0, y0, x1, y1 = bbox; h = max(4, y1 - y0)
        p = int(0.8 * h)
        X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
        g = cv2.cvtColor(src[Y0:Y1, X0:X1], cv2.COLOR_RGB2GRAY)
        k = min(31, max(3, (min(g.shape) // 2) * 2 - 1))
        ink = ((cv2.medianBlur(g, k).astype(np.int16) - g.astype(np.int16)) > 18).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        out = np.zeros((H, W), np.uint8)
        inside = np.zeros_like(ink, bool); inside[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
        for i in range(1, n):
            cx, cy, cw, ch, area = st[i]
            if area < 6 or (cw > 3 * h and ch < 0.25 * h) or (ch > 3 * h and cw < 0.25 * h):  # rules, box edges
                continue
            if cx <= X0 - X0 and cx + cw >= X1 - X0:  # spans the whole window: a form line or a long joined scrawl
                continue
            comp = lab == i
            frac = (comp & inside).sum() / max(1, area)
            cyc = cy + ch / 2  # the blob's middle must sit inside the box's rows: never take the line above/below
            if frac >= 0.5 and frac < 1.0 and (y0 - Y0) <= cyc <= (y1 - Y0):
                out[Y0:Y1, X0:X1] |= comp.astype(np.uint8)
        return cv2.dilate(out, np.ones((7, 7), np.uint8)) if out.any() else out

    def _upright(self, page, rotation, bbox, fn):
        """Run an edit on an upright view of sideways/upside-down text. rotation = clockwise degrees the text is turned on
        the page (90: reads top-to-bottom, 270: reads bottom-to-top, 180: upside down)."""
        k = {90: 1, 180: 2, 270: 3}.get(int(rotation) % 360)
        if not k:
            return fn(bbox)
        H, W = self.cur[page].shape[:2]
        cur0, orig0 = self.cur[page], self.orig[page]
        m = np.zeros((H, W), bool); x0, y0, x1, y1 = [int(round(v)) for v in bbox]; m[max(0, y0):y1, max(0, x0):x1] = True
        mr = np.rot90(m, k); ys, xs = np.nonzero(mr)
        if not len(xs):
            return {"ok": False, "error": "empty bbox"}
        bb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        self.cur[page] = np.ascontiguousarray(np.rot90(cur0, k)); self.orig[page] = np.ascontiguousarray(np.rot90(orig0, k))
        n = len(self.edits)
        try:
            r = fn(bb)
            self.cur[page] = np.ascontiguousarray(np.rot90(self.cur[page], -k))
        except Exception:
            self.cur[page] = cur0
            raise
        finally:
            self.orig[page] = orig0
        for e in self.edits[n:]:
            if e.mask is not None:
                e.mask = np.ascontiguousarray(np.rot90(e.mask, -k)); ys, xs = np.nonzero(e.mask)
                if len(xs):
                    e.bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if isinstance(r, dict) and r.get("ok") and n < len(self.edits):
            r["changed_bbox"] = self.edits[-1].bbox; r["rotation"] = rotation
        return r

    # ------------------------------------------------------------------ faces
    def replace_face(self, page, bbox, seed=0, apparent=None):
        """Swap the face in a photo for a synthetic one (SFHQ bank, landmark-aligned, tone/blur matched, Poisson-blended
        inside the face ellipse) and check with SFace that the result no longer matches the original person.
        faces='keep' (policy for uploads where swapping is off): nothing is changed; a review flag is recorded."""
        import hashlib
        from ..docs import native as N
        if self.faces == "keep":
            self.flags.append({"check": "face", "page": page, "bbox": [int(v) for v in bbox], "detail": "face kept and flagged (face swap is off for this upload)"})
            return {"ok": True, "flagged": True, "note": "face swap is off for these documents; the face is flagged for review instead"}
        img = self.cur[page]; before = img.copy()
        fseed = f"agent:{self.name}:{seed}:"  # string seed, as the pipeline passes it
        a = apparent or {}
        sx = {"female": "f", "f": "f", "woman": "f", "male": "m", "m": "m", "man": "m"}.get(str(a.get("sex", "")).lower())
        band = a.get("age_band") if a.get("age_band") in ("young", "mid", "old") else None
        eth = a.get("ethnicity") if a.get("ethnicity") in N.ETH_NEAR else None
        profile = {k: v for k, v in (("gender", sx), ("age_band", band), ("ethnicity", eth)) if v}
        try:
            region, info = N.swap_portrait(img, bbox, fseed, np.random.default_rng(int(hashlib.sha256(fseed.encode()).hexdigest()[:8], 16)), None, profile)
        except Exception as ex:
            img[:] = before
            return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        if region is None:
            img[:] = before
            return {"ok": False, "error": (info or {}).get("error", "face swap failed")}
        mask = (img != before).any(axis=2)
        if not mask.any():
            return {"ok": False, "error": "face swap changed nothing"}
        ys, xs = np.nonzero(mask)
        ed = self._record("face", page, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "(face)", "(synthetic face)", mask)
        tile = (info or {}).get("synthetic_face_tile")
        ed.meta = {"apparent": profile, "tile": tile, "tile_attrs": {"sex": N.SFHQ_ATTRS.get(tile, ("?",))[0], "age_band": N.SFHQ_ATTRS.get(tile, (0, "?"))[1],
                   "ethnicity": N.SFHQ_ETH.get(tile)}, "match": N.face_match(tile, profile)}
        ic = (info or {}).get("identity_check", {})
        return {"ok": True, "edit_id": ed.id, "changed_bbox": ed.bbox, "identity_check": ic, "replacement_face": ed.meta["tile_attrs"], "match": ed.meta["match"],
                **({"warning": "SFace still matches the original person; try a larger bbox around the face"} if ic.get("differs_from_original") is False else {})}

    SIGNATURE_FONTS = ("Homemade Apple", "Herr Von Muellerhoff", "Alex Brush", "Allura", "Great Vibes", "Sacramento", "La Belle Aurore",
                       "Cedarville Cursive", "Zeyada", "Dawning of a New Day", "Yellowtail", "Mrs Saint Delafield")

    def replace_signature(self, page, bbox, fake_name, font=None, seed=0, rotation=0):
        """Erase every stroke in bbox and sign fake_name in a signature hand spanning the same width, in the darkest
        ink of the original strokes (paper-matched fill, mask-only composite)."""
        import hashlib
        from ..docs import native as N
        fonts = [font] if font else [f for f in self.SIGNATURE_FONTS if N._fc(f, "Regular")]
        if not fonts:
            return {"ok": False, "error": "no signature font installed"}
        f = fonts[int(hashlib.sha256(fake_name.encode()).hexdigest(), 16) % len(fonts)]
        r = self.draw_text(page, bbox, fake_name, fake_name, f, condense_to_fit=True, seed=seed, rotation=rotation)
        if r.get("ok"):
            self.edits[-1].kind = "signature"; self.edits[-1].original = "(signature)"
            r["font"] = f
        return r

    def undo(self, edit_id):
        e = next((e for e in self.edits if e.id == int(edit_id)), None)
        if e is None or e.undone:
            return {"ok": False, "error": "no such active edit"}
        if e.mask is None:
            return {"ok": False, "error": "native PDF text edits cannot be undone; replace again instead"}
        self.cur[e.page][e.mask] = self.orig[e.page][e.mask]
        e.undone = True
        for o in self.edits:  # re-apply later edits that overlapped (their pixels were just reset)
            if o.id > e.id and not o.undone and o.mask is not None and (o.mask & e.mask).any():
                return {"ok": True, "warning": f"edit {o.id} overlapped and was partly reset; redo it"}
        return {"ok": True}

    # ------------------------------------------------------------------ native PDF text
    @staticmethod
    def _pdf_chars(pg):
        import fitz
        chars = []  # (char, bbox, origin, span); None bbox = line break
        for b in pg.get_text("rawdict")["blocks"]:
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    for c in sp["chars"]:
                        chars.append((c["c"], fitz.Rect(c["bbox"]), c["origin"], sp))
                chars.append(("\n", None, None, None))
        return chars

    def pdf_erase(self, page, bbox):
        """Erase on a text-layer page: remove the text-layer glyphs whose centre is inside bbox (the page stays text).
        None when the box holds no text-layer glyphs (then it is a pixel erase of image content)."""
        import fitz
        pg = self.pdf[page]
        R = fitz.Rect(*[float(v) / self.scale for v in bbox])
        hit = [c for c in self._pdf_chars(pg) if c[1] is not None and R.contains(fitz.Point((c[1].x0 + c[1].x1) / 2, (c[1].y0 + c[1].y1) / 2))]
        if not hit:
            return None
        for ch, r, o, sp in hit:
            cx = (r.x0 + r.x1) / 2; hw = max(0.3, (r.x1 - r.x0) * 0.3); sz = sp["size"]
            pg.add_redact_annot(fitz.Rect(cx - hw, o[1] - sz * 0.75, cx + hw, o[1] + sz * 0.22), fill=None)
        pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
        txt = "".join(c[0] for c in hit)
        r_all = fitz.Rect(hit[0][1])
        for c in hit[1:]:
            r_all |= c[1]
        ed = self._record("pdf_text", page, [v * self.scale for v in r_all], txt, "", None, None)
        pix = pg.get_pixmap(dpi=round(72 * self.scale))
        self.cur[page] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
        return {"ok": True, "edit_id": getattr(ed, "id", None), "removed_text": txt[:200], "note": "text-layer glyphs removed; page kept as text"}

    def pdf_replace(self, page, original, replacement, near=None):
        """Native text-layer edit. Only the words that differ are touched: the original is located in the page's
        character stream (search hits split across spans count as one match), shared leading/trailing words are kept, the
        differing glyphs are removed with tight per-glyph redactions (one redaction pass per match, so the rest of the line
        never shifts), and the new text is set at the first changed glyph's baseline in the span's own embedded font when it
        covers every glyph (else the closest base-14 face). If the new text is wider or narrower, the rest of that line is
        re-set right after it, so no gap or overlap is left."""
        import fitz
        pg = self.pdf[page]
        o_core = re.sub(r"\s+", " ", original).strip()
        if not o_core:
            return {"ok": False, "error": "empty original"}
        pat = re.compile(r"\s*".join(_char_re(ch) for ch in o_core if ch != " "))
        NR = fitz.Rect(*[float(v) / self.scale for v in near]) + (-2, -2, 2, 2) if near else None

        def find():
            chars = self._pdf_chars(pg)
            hs = list(pat.finditer("".join(c[0] for c in chars)))
            if NR is not None:   # one specific occurrence (re-applying a recorded edit)
                hs = [m for m in hs if any(chars[k][1] is not None and chars[k][1].intersects(NR) for k in range(m.start(), m.end()))][:1]
            return chars, hs
        chars, hits = find()
        if not hits and o_core[:1] in "•·-–*":  # bullets often sit in a symbol-font run of their own
            o2 = o_core[1:].strip(); r2 = replacement.strip()
            r2 = r2[1:].strip() if r2[:1] == o_core[:1] else r2
            return self.pdf_replace(page, o2, r2, near) if o2 else {"ok": False, "error": "empty original"}
        if not hits:
            return {"ok": False, "error": f"'{original}' not in the PDF text layer of page {page}; use bbox for pixel edit"}
        total = 1 if NR is not None else len(hits); n = 0
        for _ in range(total):
            if not hits:
                break
            m = hits[0]
            seg = chars[m.start():m.end()]
            got = "".join(c[0] for c in seg)
            replacement = _match_quotes(got, replacement)
            i = 0
            while i < min(len(got), len(replacement)) and _fold(got[i]) == _fold(replacement[i]):
                i += 1
            j = 0
            while j < min(len(got), len(replacement)) - i and _fold(got[len(got) - 1 - j]) == _fold(replacement[len(replacement) - 1 - j]):
                j += 1
            while i > 0 and not got[i - 1].isspace():   # keep shared text only up to a word boundary
                i -= 1
            while j > 0 and not got[len(got) - j].isspace() and got[len(got) - j] not in ",.;:)":
                j -= 1
            old_idx = [k for k in range(m.start() + i, m.end() - j) if chars[k][1] is not None]
            new_txt = replacement[i:len(replacement) - j]
            if not old_idx and not new_txt:
                break
            a_i = old_idx[0] if old_idx else (m.start() + i - 1 if i else m.start())
            anchor = chars[a_i]; sp = anchor[3]
            size = sp["size"]; col = sp["color"]
            rgb = ((col >> 16) & 255) / 255, ((col >> 8) & 255) / 255, (col & 255) / 255
            x_start = chars[old_idx[0]][1].x0 if old_idx else (anchor[1].x1 if i else anchor[1].x0)
            free = (chars[old_idx[-1]][1].x1 - x_start) if old_idx else 0
            fname = sp["font"].split("+")[-1].lower()
            bold = "bold" in fname; italic = "italic" in fname or "oblique" in fname
            serif = any(t in fname for t in ("times", "serif", "roman", "georgia", "garamond", "cambria", "book"))
            base = ("tibi" if bold and italic else "tibo" if bold else "tiit" if italic else "tiro") if serif else \
                   ("hebi" if bold and italic else "hebo" if bold else "heit" if italic else "helv")
            fb = _fallback_font(serif, bold, italic)
            fb_tuple = ("CPfb" + ("S" if serif else "H") + ("b" if bold else "") + ("i" if italic else ""), fitz.Font(fontbuffer=fb), fb) if fb \
                else (base, fitz.Font(base), None)
            font, fobj, fbuf = fb_tuple
            try:  # the document's own embedded font
                xref = next((f[0] for f in pg.get_fonts() if f[3].split("+")[-1] == sp["font"].split("+")[-1] and f[1] in ("ttf", "cff", "otf", "type1c")), None)
                if xref:
                    buf = self.pdf.extract_font(xref)[3]
                    if buf:
                        font, fobj, fbuf = f"F{xref}", fitz.Font(fontbuffer=buf), buf
            except Exception:
                font, fobj, fbuf = fb_tuple
            # width change -> take the rest of the line along
            kill = list(old_idx)
            if old_idx and new_txt and abs(fobj.text_length(new_txt, fontsize=size) - free) > 0.04 * free + 0.8:
                k = old_idx[-1] + 1; rest = []
                while k < len(chars) and chars[k][1] is not None:
                    rest.append(k); k += 1
                if rest:
                    new_txt = new_txt + "".join(chars[k][0] for k in rest).rstrip()
                    kill += rest; free = chars[rest[-1]][1].x1 - x_start; j = 0
            if fbuf and not all(fobj.has_glyph(ord(ch)) for ch in new_txt if not ch.isspace()):
                font, fobj, fbuf = fb_tuple   # subset lacks a glyph: full-coverage metric-compatible face
            for k in kill:  # centre band of each glyph only: neighbours and other lines untouched
                r = chars[k][1]; y = chars[k][2][1]; cx = (r.x0 + r.x1) / 2; hw = max(0.3, (r.x1 - r.x0) * 0.3)
                pg.add_redact_annot(fitz.Rect(cx - hw, y - size * 0.75, cx + hw, y + size * 0.22), fill=None)
            if kill:
                pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
            fs = size
            if new_txt and free:
                w = fobj.text_length(new_txt, fontsize=size)
                tail = chars[m.end() - j] if j else None
                room = (tail[1].x0 - x_start) if tail and tail[1] is not None else free * 1.15
                if w > room > 0:
                    fs = max(size * 0.8, size * room / w)
            if new_txt:
                if fbuf:
                    try:
                        pg.insert_font(fontname=font, fontbuffer=fbuf)
                    except Exception:
                        font = base
                pg.insert_text((x_start, anchor[2][1]), new_txt, fontsize=fs, fontname=font, color=rgb)
            r_all = fitz.Rect(chars[kill[0]][1]) if kill else fitz.Rect(x_start, anchor[2][1] - size, x_start + 1, anchor[2][1])
            for k in kill[1:]:
                r_all |= chars[k][1]
            self._record("pdf_text", page, [v * self.scale for v in r_all], original, replacement, None, font)
            n += 1
            chars, hits = find()
        pix = pg.get_pixmap(dpi=round(72 * self.scale))
        self.cur[page] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
        return {"ok": True, "occurrences": n, "note": "PDF text layer edited natively"}

    # ------------------------------------------------------------------ export
    def save(self, out_dir):
        """Write the edited document. Images: re-encoded with no metadata (EXIF/XMP dropped). PDFs: metadata and XMP
        cleared, hidden content scrubbed, raster-edited pages replaced by their edited image."""
        from PIL import Image
        out = Path(out_dir) / self.name
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.kind == "image":
            im = Image.fromarray(self.cur[0])
            if self.path.suffix.lower() in (".jpg", ".jpeg"):
                im.save(out, "JPEG", quality=92)
            else:
                im.save(out)
            return out
        import fitz
        d = fitz.open("pdf", self.pdf.tobytes())  # a copy: save() runs once per verify round and must not stack page swaps
        raster_pages = {e.page for e in self.edits if e.mask is not None and not e.undone}
        for p in sorted(raster_pages):
            pg = d[p]; r = pg.rect
            b = io.BytesIO(); Image.fromarray(self.cur[p]).save(b, "JPEG", quality=92)
            pg.add_redact_annot(r); pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)  # drop old text+images
            pg.insert_image(r, stream=b.getvalue())
        d.set_metadata({}); d.del_xml_metadata()
        try:
            d.scrub(metadata=True, xml_metadata=True, attached_files=True, embedded_files=True, javascript=True, hidden_text=True,
                    thumbnails=True, reset_fields=False, reset_responses=True, redactions=False, remove_links=False)
        except TypeError:
            d.scrub()
        d.save(out, garbage=4, deflate=True, clean=True)
        return out


class CaseMap:
    """Shared replacement map for a case (thread-safe). Same original -> same replacement everywhere."""

    def __init__(self):
        self.entries = []; self.lock = threading.Lock()

    def write(self, entries, file):
        notes = []
        with self.lock:
            for x in entries or []:
                o, r = str(x.get("original", "")).strip(), str(x.get("replacement", "")).strip()
                kind = x.get("kind", "")
                if not o or not r:
                    continue
                prior = next((e for e in self.entries if _norm(e["original"]) == _norm(o) and _norm(e["replacement"]) != _norm(r)), None)
                if prior:
                    notes.append(f"'{o}' is already mapped to '{prior['replacement']}' (in {prior['file']}); use that for consistency")
                    r = prior["replacement"]
                f = x.get("file") or file
                if not any(e["file"] == f and _norm(e["original"]) == _norm(o) for e in self.entries):
                    self.entries.append({"file": f, "original": o, "replacement": r, "kind": kind})
        return notes

    def correct(self, original, replacement, file, reason=""):
        """Change the fake for an original everywhere (e.g. a wrong first choice, an original mapped to itself). Keeps the
        superseded fake on the entry so the verifier can find edits that still show it. -> (ok, note)"""
        o, r = str(original or "").strip(), str(replacement or "").strip()
        if not o or not r:
            return False, "original and replacement are required"
        if _norm(o) == _norm(r):
            return False, "the replacement must differ from the original"
        with self.lock:
            hit = [e for e in self.entries if _norm(e["original"]) == _norm(o)]
            if not hit:
                return False, f"'{o}' is not in the map; use write_map"
            clash = next((e for e in self.entries if _norm(e["replacement"]) == _norm(r) and _norm(e["original"]) != _norm(o)), None)
            if clash:
                return False, f"'{r}' is already the fake for a different original ('{clash['original']}'); pick another"
            users = sorted({e["file"] for e in hit})
            for e in hit:
                if _norm(e["replacement"]) != _norm(r):
                    e.setdefault("superseded", []).append(e["replacement"])
                    e["replacement"] = r; e["corrected_by"] = file; e["reason"] = str(reason)[:200]
        return True, f"'{o}' now maps to '{r}' in {', '.join(users)}" + ("; other documents must redo their edits" if set(users) - {file} else "")

    def text(self):
        with self.lock:
            return json.dumps(self.entries, ensure_ascii=False)
