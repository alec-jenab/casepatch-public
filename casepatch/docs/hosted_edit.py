"""Live hosted image edits on a reviewed case.

Uses the customer's own Fal key, off by default, never via CasePatch servers. Each document
must be opted in and an estimated cost is shown before anything runs. The local editor stays the default
until the reviewer picks a hosted candidate. Outputs are pasted back only inside the change mask.

This module does not construct the HTTP transport — callers must inject one. Tests inject a fake. This
work never makes a real Fal call. A direct Ideogram API key is refused (use a Fal key). Eval-key
fingerprints are refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import editor_pick as E
from .byok import ByokRefused, CaseBudget, is_eval_key


NOTICE = (
    "Hosted image edits use your own Fal API key, off by default, and never go through CasePatch servers. "
    "Each document must be opted in. An estimated cost is shown before anything runs. "
    "The local editor stays selected until you pick a hosted candidate. "
    "Outputs are pasted back only inside the change mask."
)

# Margin around the occurrence box, same rule as the public edit benchmark (2× height left/right, 1.5× top/bottom).
PAD_X, PAD_Y = 2.0, 1.5

HOSTED_EDITORS = [e["id"] for e in E.EDITORS if e["kind"] == "hosted"]

HOSTED_EDITS_NAME = "hosted-edits.json"
EDITS_DIRNAME = "edits"


class HostedEditError(Exception):
    """User-facing refusal or missing gate. Never includes a key."""


class NeedOptIn(HostedEditError):
    pass


class NeedAck(HostedEditError):
    pass


class NeedEstimate(HostedEditError):
    """Run was refused because the reviewer has not accepted the current estimate."""

    def __init__(self, estimate, msg="Accept the estimated cost before running hosted editors."):
        super().__init__(msg)
        self.estimate = estimate


class TransportRequired(HostedEditError):
    pass


def edit_instruction(value, replacement):
    return (
        f'Photo-edit this document crop. Inside the masked area only, replace the text "{value}" with '
        f'"{replacement}" in the same font, size, weight, ink colour, paper texture and noise, so it looks '
        "original. Do not change anything outside the mask."
    )


def crop_box(bbox, page_w, page_h, pad_x=PAD_X, pad_y=PAD_Y):
    """Inclusive-style box → integer crop [x0,y0,x1,y1] and a mask box relative to the crop, clipped to the page."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if max(x0, y0, x1, y1) <= 1.5 and page_w and page_h:
        x0, x1 = x0 * page_w, x1 * page_w
        y0, y1 = y0 * page_h, y1 * page_h
    hgt = max(8.0, y1 - y0)
    cx0 = max(0, int(x0 - pad_x * hgt))
    cy0 = max(0, int(y0 - pad_y * hgt))
    cx1 = min(int(page_w), int(x1 + pad_x * hgt)) if page_w else int(x1 + pad_x * hgt)
    cy1 = min(int(page_h), int(y1 + pad_y * hgt)) if page_h else int(y1 + pad_y * hgt)
    if cx1 <= cx0:
        cx1 = cx0 + 8
    if cy1 <= cy0:
        cy1 = cy0 + 8
    mx0 = int(x0) - cx0
    my0 = int(y0) - cy0
    mx1 = int(np.ceil(x1)) - cx0
    my1 = int(np.ceil(y1)) - cy0
    return [cx0, cy0, cx1, cy1], [mx0, my0, mx1, my1]


def crop_and_mask(page_rgb, bbox, pad_x=PAD_X, pad_y=PAD_Y):
    H, W = page_rgb.shape[:2]
    crop_xy, mask_xy = crop_box(bbox, W, H, pad_x, pad_y)
    x0, y0, x1, y1 = crop_xy
    crop = page_rgb[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    a, b, c, d = mask_xy
    mask[max(0, b - 1):max(0, d + 1), max(0, a - 1):max(0, c + 1)] = 1
    if not mask.any():
        mask[:, :] = 1
    return crop, mask, crop_xy


def estimate_one(crop_w, crop_h, model):
    """Worst-case Fal USD for one editor on a crop of this size. No network."""
    from casepatch.hosted import faledit as F
    from casepatch.hosted import pricing as P
    h, w = int(crop_h), int(crop_w)
    dims = F.billed_dims(model, h, w) if model in F.SPECS else (w, h)
    return float(P.worst_case_usd("fal", model, 120, [dims], 0, output_images=1))


def _occ_doc(o):
    return o.get("doc") or o.get("file")


def _occ_page(o):
    return o.get("page") if o.get("page") is not None else o.get("page_index")


def _editable(occ):
    if not occ.get("id"):
        return False
    if occ.get("text") is None and occ.get("value") is None:
        return False
    if occ.get("bbox") is None and occ.get("crop") is None:
        return False
    return True


def estimate_for_occurrences(occurrences, *, page_size=None, models=None, documents=None, page_loader=None):
    """Cost estimate for hosted editors on opted-in documents. No network, no key.

    `page_size` is an optional (doc, page) -> (w, h). `page_loader` is an optional (doc, page) -> RGB array
    (used only for dimensions). If neither is given, the occurrence bbox (plus pad) is used as the crop size.
    """
    models = list(models or HOSTED_EDITORS)
    docs = set(documents) if documents else None
    items, total = [], 0.0
    for occ in occurrences:
        if not _editable(occ):
            continue
        if docs is not None and _occ_doc(occ) not in docs:
            continue
        bbox = occ.get("bbox") or occ.get("crop")
        w = h = None
        if page_loader is not None:
            page = page_loader(_occ_doc(occ), _occ_page(occ))
            if page is not None:
                h, w = page.shape[:2]
        if (w is None or h is None) and page_size is not None:
            size = page_size(_occ_doc(occ), _occ_page(occ)) if callable(page_size) else page_size
            if size:
                w, h = size
        if w and h:
            crop_xy, _ = crop_box(bbox, w, h)
            cw, ch = crop_xy[2] - crop_xy[0], crop_xy[3] - crop_xy[1]
        else:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            hgt = max(8.0, y1 - y0)
            cw = max(8, int((x1 - x0) + 2 * PAD_X * hgt))
            ch = max(8, int((y1 - y0) + 2 * PAD_Y * hgt))
        per = [{"editor": m, "usd": estimate_one(cw, ch, m)} for m in models]
        usd = round(sum(p["usd"] for p in per), 6)
        total += usd
        items.append({
            "id": occ["id"], "doc": _occ_doc(occ), "page": _occ_page(occ),
            "crop_w": int(cw), "crop_h": int(ch), "per_editor": per, "usd": usd,
        })
    return {
        "usd": round(total, 6),
        "n_changes": len(items),
        "n_editors": len(models),
        "n_calls": len(items) * len(models),
        "editors": models,
        "documents": sorted(docs) if docs is not None else sorted({i["doc"] for i in items if i.get("doc")}),
        "items": items,
        "notice": NOTICE,
        "currency": "USD",
    }


def _check_gates(key, *, ack, documents, env=None):
    if key is None or not str(key).strip():
        raise ByokRefused("Hosted image edits are off until you add your own Fal API key.")
    E.check_hosted_edit_key(key, env)
    if not ack:
        raise NeedAck(NOTICE)
    if not documents:
        raise NeedOptIn("Select at least one document to run hosted editors on.")


def make_edit_client(key, max_usd, transport, env=None):
    """HostedClient for Fal image edits. `transport` is required; HttpTransport is never built here."""
    if transport is None:
        raise TransportRequired(
            "Hosted image edits require an injected transport. This module does not construct the HTTP client."
        )
    E.check_hosted_edit_key(key, env)
    from casepatch.hosted.client import HostedClient
    from casepatch.hosted.config import HostedConfig
    cap = float(max_usd)
    cfg = HostedConfig(
        provider="fal",
        cap_usd=cap,
        ledger_path=Path(os.devnull),
        max_retries=0,
        timeout_s=60.0,
        max_images_per_call=2,
        max_calls_per_case=200,
        pool="other",
        pool_cap_usd=cap,
        _key=key,
    )
    return HostedClient(cfg, transport=transport, ledger=CaseBudget(cap))


def _norm(s):
    return " ".join(str(s or "").split()).casefold()


def _contains(needle, hay):
    return bool(needle) and _norm(needle) in _norm(hay)


def _fuzzy(needle, hay, thresh=0.8):
    if _contains(needle, hay):
        return True
    if not needle or not hay:
        return False
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _norm(needle), _norm(hay)).ratio() >= thresh


def _gray(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _ring(mask):
    m = mask > 0
    p = np.pad(m, 1, constant_values=False)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    outer = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return (m & ~inner) | (outer & ~m)


def style_scores(orig, edited, mask):
    """Seam (mean |grad| on the mask border, edited minus original) and the shipped outside-mask change fraction."""
    outm = mask == 0
    diff = np.abs(edited.astype(np.int16) - orig.astype(np.int16)).max(axis=2)
    outside = float((diff[outm] > 25).mean()) if outm.any() else 0.0
    ring = _ring(mask)
    if ring.any():
        def grad(g):
            dx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
            dy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
            return dx + dy
        ga, gb = _gray(orig), _gray(edited)
        seam = float(grad(gb)[ring].mean() - grad(ga)[ring].mean())
    else:
        seam = 0.0
    return {"seam": round(seam, 2), "outside_mask": round(outside, 4), "raw_outside_mask": None}


def score_edit(orig, edited, mask, value, replacement, *, ocr_fn=None, raw=None):
    """Build the badge row for one hosted candidate. `ocr_fn(img) -> str` is optional (tests inject a stub)."""
    txt = ocr_fn(edited) if ocr_fn is not None else None
    exact = _contains(replacement, txt) if txt is not None else None
    fuzzy = _fuzzy(replacement, txt) if txt is not None else None
    residual = bool(txt is not None and (_contains(value, txt) or _fuzzy(value, txt)))
    style = style_scores(orig, edited, mask)
    if raw is not None and raw.shape[:2] == orig.shape[:2]:
        outm = mask == 0
        rd = np.abs(raw.astype(np.int16) - orig.astype(np.int16)).max(axis=2)
        style["raw_outside_mask"] = round(float((rd[outm] > 25).mean()), 3) if outm.any() else 0.0
    return {
        "exact": exact, "fuzzy": fuzzy, "residual": residual, "ghost": False, "page_leak": False,
        "seam": style["seam"], "out_of_mask_frac": style["outside_mask"],
        "raw_out_of_mask_frac": style["raw_outside_mask"], "ocr": (txt[:160] if isinstance(txt, str) else txt),
    }


def _png_bytes(arr):
    from PIL import Image
    import io
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def persist_run(review_dir, result):
    """Write review/hosted-edits.json and PNG crops under review/edits/. Source values are not stored."""
    review_dir = Path(review_dir)
    edits = review_dir / EDITS_DIRNAME
    edits.mkdir(parents=True, exist_ok=True)
    slim_changes = {}
    for cid, ch in (result.get("changes") or {}).items():
        cands = {}
        for eid, cand in (ch.get("candidates") or {}).items():
            rec = {k: v for k, v in cand.items() if k != "image"}
            # badges.ocr can contain the original or replacement string; do not persist it
            if isinstance(rec.get("badges"), dict):
                rec["badges"] = {k: v for k, v in rec["badges"].items() if k != "ocr"}
            img = cand.get("image")
            if img is not None:
                safe = eid.replace("/", "__")
                rel = f"{EDITS_DIRNAME}/{cid}/{safe}.png"
                dest = review_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_png_bytes(img))
                rec["src"] = rel
            cands[eid] = rec
        slim_changes[cid] = {
            "id": cid,
            "doc": ch.get("doc"),
            "page": ch.get("page"),
            "crop": ch.get("crop"),
            "candidates": cands,
        }
    payload = {
        "ack": True,
        "documents": result.get("documents") or [],
        "estimated_usd": result.get("estimated_usd"),
        "spent_usd": result.get("spent_usd"),
        "editors": result.get("editors") or [],
        "changes": slim_changes,
        "notice": NOTICE,
    }
    (review_dir / HOSTED_EDITS_NAME).write_text(json.dumps(payload, indent=2))
    return payload


def load_run(review_dir):
    p = Path(review_dir) / HOSTED_EDITS_NAME
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def merge_live_candidates(model, persisted, *, byok=False):
    """Attach persisted hosted candidates onto a live compare model. Local remains present."""
    if not persisted:
        return model
    stored = persisted.get("changes") or {}
    any_hosted = False
    for ch in model["changes"]:
        rec = stored.get(ch["id"]) or stored.get(ch.get("occurrence_id"))
        if not rec:
            continue
        for eid, cand in (rec.get("candidates") or {}).items():
            if eid == E.LOCAL_EDITOR:
                continue
            ch["candidates"][eid] = dict(cand)
            any_hosted = True
    if any_hosted or byok:
        model["editors"] = E.visible_editors(byok=True, precomputed=bool(model.get("precomputed")))
        # keep editor order stable even if some changes lack a candidate
        seen = {e["id"] for e in model["editors"]}
        for e in E.EDITORS:
            if e["id"] not in seen and any(e["id"] in c.get("candidates", {}) for c in model["changes"]):
                model["editors"].append(e)
    return model


def run_edits(occurrences, *, key, transport, ack, documents, accepted_usd, page_loader,
              models=None, max_usd=5.0, env=None, ocr_fn=None, review_dir=None, case_id="case"):
    """Run hosted editors on opted-in documents. `transport` is required (tests inject a fake).

    `accepted_usd` must be the estimate the reviewer saw; a higher fresh estimate is refused.
    `page_loader(doc, page) -> RGB array`. No HttpTransport is constructed here.
    """
    models = list(models or HOSTED_EDITORS)
    _check_gates(key, ack=ack, documents=documents, env=env)
    est = estimate_for_occurrences(occurrences, models=models, documents=documents, page_loader=page_loader)
    if accepted_usd is None:
        raise NeedEstimate(est)
    try:
        accepted = float(accepted_usd)
    except (TypeError, ValueError) as e:
        raise NeedEstimate(est, "Accept the estimated cost before running hosted editors.") from e
    if est["usd"] > accepted * 1.05 + 0.01:
        raise NeedEstimate(est, f"Estimated cost is now ${est['usd']:.4f}, above the accepted ${accepted:.4f}.")
    client = make_edit_client(key, max_usd, transport, env=env)
    from casepatch.hosted.client import ImageEditor, composite_inside_mask

    docs = set(documents)
    changes = {}
    errors = []
    spent0 = client.ledger.spent_estimate()
    for occ in occurrences:
        if not _editable(occ) or _occ_doc(occ) not in docs:
            continue
        value = occ.get("text") if occ.get("text") is not None else occ.get("value")
        replacement = occ.get("replacement") or ""
        page = page_loader(_occ_doc(occ), _occ_page(occ))
        if page is None:
            errors.append({"id": occ["id"], "error": "page not available"})
            continue
        crop, mask, crop_xy = crop_and_mask(page, occ.get("bbox") or occ.get("crop"))
        rec = {
            "id": occ["id"], "doc": _occ_doc(occ), "page": _occ_page(occ), "crop": crop_xy,
            "candidates": {},
        }
        instruction = edit_instruction(value, replacement)
        for mid in models:
            editor = ImageEditor(client, model=mid, ocr_check=None)
            before = client.ledger.spent_estimate()
            try:
                # ocr_check left off so a weak readback still reaches the picker (badges decide selectability)
                out = editor.edit(crop, mask, instruction, None, f"{case_id}:{occ['id']}:{mid}")
            except Exception as exc:
                errors.append({"id": occ["id"], "editor": mid, "error": str(exc)})
                rec["candidates"][mid] = {
                    "badges": E.badge({"exact": False, "fuzzy": False, "residual": False, "ghost": False,
                                       "page_leak": False, "seam": None, "out_of_mask_frac": None,
                                       "raw_out_of_mask_frac": None, "ocr": None}),
                    "src": None, "cost_usd": None, "latency_s": None, "error": str(exc),
                }
                rec["candidates"][mid]["badges"]["selectable"] = False
                continue
            if out is None:
                rec["candidates"][mid] = {
                    "badges": E.badge({"exact": False, "fuzzy": False, "residual": False, "ghost": False,
                                       "page_leak": False, "seam": None, "out_of_mask_frac": None,
                                       "raw_out_of_mask_frac": None, "ocr": None}),
                    "src": None, "cost_usd": None, "latency_s": None, "error": "no image returned",
                }
                rec["candidates"][mid]["badges"]["selectable"] = False
                continue
            raw = getattr(editor, "last_raw", None)
            # ImageEditor already composited inside the mask; re-assert so a leaky raw cannot ship outside it
            shipped = composite_inside_mask(crop, mask, out if raw is None else raw, None, None)
            row = score_edit(crop, shipped, mask, value, replacement, ocr_fn=ocr_fn, raw=raw)
            rec["candidates"][mid] = {
                "badges": E.badge(row),
                "src": None,
                "cost_usd": round(max(0.0, client.ledger.spent_estimate() - before), 6),
                "latency_s": None,
                "image": shipped,
            }
        changes[occ["id"]] = rec
    result = {
        "documents": sorted(docs),
        "editors": models,
        "estimated_usd": est["usd"],
        "spent_usd": round(max(0.0, client.ledger.spent_estimate() - spent0), 6),
        "changes": changes,
        "errors": errors,
        "notice": NOTICE,
    }
    if review_dir is not None:
        persist_run(review_dir, result)
        # drop numpy arrays from the returned payload after persist (callers load src paths)
        for ch in result["changes"].values():
            for cand in ch["candidates"].values():
                cand.pop("image", None)
    return result
