"""Live-case editor picker and BYOK hosted-edit gates for Studio.

The HTTP transport is never constructed here. Tests inject a fake on Studio(edit_transport=...).
This work does not make real Fal calls. Pack matches (public samples) may fill hosted candidates
without a live run. Local is always present and is the default pick.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from casepatch.docs import editor_pick as E
from casepatch.docs import hosted_edit as H

from . import compare_api as CMP

PICKS_NAME = "editor-picks.json"
SAFE_EDITOR = re.compile(r"^[A-Za-z0-9._-]+$")
NOTICE_VERSION = 1


def editor_token(eid: str) -> str:
    return eid.replace("/", "__")


def editor_from_token(token: str) -> str | None:
    if token in ("original", "local"):
        return token
    for e in E.EDITORS:
        if editor_token(e["id"]) == token:
            return e["id"]
    return None


def _png(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def _load_rgb(path: Path):
    if not path.is_file():
        return None
    try:
        return np.array(Image.open(path).convert("RGB"), copy=True)
    except Exception:
        return None


def preview_path(c, doc, page, side="before"):
    prev = Path(c.dir) / "previews"
    if side == "before":
        return prev / f"{doc}-p{page}-before.png"
    return prev / f"{doc}-p{page}-after-{c.run}.png"


def page_loader_for(c):
    def load(doc, page):
        return _load_rgb(preview_path(c, doc, page, "before"))
    return load


def load_picks(c, change_ids) -> E.Picks:
    p = E.Picks(change_ids)
    f = Path(c.dir) / "review" / PICKS_NAME
    if f.is_file():
        try:
            data = json.loads(f.read_text())
            p.chosen.update({k: v for k, v in (data.get("chosen") or {}).items() if k in p.change_ids})
            p.overrides = {(cid, ed) for row in (data.get("overrides") or [])
                           for cid, ed in [tuple(row.split("\t", 1)) if "\t" in row else (None, None)]
                           if cid in p.change_ids}
        except (ValueError, TypeError, KeyError):
            pass
    return p


def save_picks(c, picks: E.Picks):
    dest = Path(c.dir) / "review"
    dest.mkdir(parents=True, exist_ok=True)
    snap = picks.snapshot()
    (dest / PICKS_NAME).write_text(json.dumps(snap, indent=2))
    return snap


def _replaced(c, text):
    if text in (c.rejected_text or []):
        return False
    # model suggestions are only replaced after the reviewer accepts them
    return True


def compare_model(studio, c, *, include_kept=False):
    if c.status != "ready" or not c.meta:
        from .server import StudioError
        raise StudioError("NOT_ANALYZED")
    occs = studio.proposals(c).get("occurrences") or []
    pack = studio._compare_pack
    byok = bool(studio.hosted_edit_status().get("available"))
    model = E.compare_model_for_occurrences(occs, pack=pack, byok=byok, include_kept=include_kept)
    persisted = H.load_run(Path(c.dir) / "review")
    model = H.merge_live_candidates(model, persisted, byok=byok)
    # live crop URLs; pack hosted srcs are rewritten to a case-scoped pack file route
    prefix = f"/api/cases/{c.id}/compare"
    for ch in model["changes"]:
        ch["original_src"] = f"{prefix}/crop/{ch['id']}/original"
        for eid, cand in list((ch.get("candidates") or {}).items()):
            if eid == E.LOCAL_EDITOR:
                cand["src"] = f"{prefix}/crop/{ch['id']}/local"
            elif cand.get("src") and not str(cand["src"]).startswith("/"):
                # persisted relative path or pack-relative crop
                src = cand["src"]
                if src.startswith("edits/"):
                    cand["src"] = f"{prefix}/file/{src}"
                elif src.startswith("crops/"):
                    cand["src"] = f"{prefix}/pack/{src}"
                else:
                    cand["src"] = f"{prefix}/crop/{ch['id']}/{editor_token(eid)}"
            elif not cand.get("src"):
                cand["src"] = f"{prefix}/crop/{ch['id']}/{editor_token(eid)}"
    picks = load_picks(c, [ch["id"] for ch in model["changes"]])
    model["picks"] = dict(picks.chosen)
    model["overrides"] = [f"{a}\t{b}" for a, b in picks.overrides]
    model["live"] = True
    model["showFull"] = False
    docs = sorted({d["id"] for d in c.meta.get("documents") or [] if d.get("kind") != "json"})
    model["hosted_edit"] = {
        **studio.hosted_edit_status(),
        "notice": H.NOTICE,
        "notice_version": NOTICE_VERSION,
        "documents": docs,
        "opted_in": (persisted or {}).get("documents") or [],
        "estimated_usd": (persisted or {}).get("estimated_usd"),
        "spent_usd": (persisted or {}).get("spent_usd"),
    }
    return model, picks


def client_payload(studio, c):
    model, _ = compare_model(studio, c)
    return model


def badges_by_change(model):
    return {ch["id"]: {eid: (cand.get("badges") or {}) for eid, cand in (ch.get("candidates") or {}).items()}
            for ch in model.get("changes") or []}


def apply_pick(studio, c, body: dict):
    model, picks = compare_model(studio, c)
    badges = badges_by_change(model)
    if body.get("all"):
        applied, skipped = picks.select_all(body.get("editor"), badges, override_leaks=bool(body.get("override")))
        save_picks(c, picks)
        return {"chosen": picks.chosen, "overrides": sorted(f"{a}\t{b}" for a, b in picks.overrides),
                "applied": applied, "skipped": skipped}
    cid, editor = body.get("change_id"), body.get("editor")
    picks.select(cid, editor, badges.get(cid) or {}, override=bool(body.get("override")))
    save_picks(c, picks)
    return {"chosen": picks.chosen, "overrides": sorted(f"{a}\t{b}" for a, b in picks.overrides)}


def _crop_png_from_page(page, bbox, side_label="before"):
    crop, mask, xy = H.crop_and_mask(page, bbox)
    return _png(crop), crop, mask, xy


def crop_bytes(studio, c, occ_id, token):
    """PNG bytes for one crop card, or None."""
    model, _ = compare_model(studio, c)
    ch = next((x for x in model["changes"] if x["id"] == occ_id), None)
    if not ch:
        return None
    eid = editor_from_token(token)
    if eid is None:
        return None
    bbox = ch.get("crop")
    doc, page = ch.get("doc") or ch.get("file"), ch.get("page")
    if eid in ("original", E.LOCAL_EDITOR):
        side = "before" if eid == "original" else "after"
        # after: use the after preview if present, else before
        arr = _load_rgb(preview_path(c, doc, page, "after" if side == "after" else "before"))
        if arr is None and side == "after":
            arr = _load_rgb(preview_path(c, doc, page, "before"))
        if arr is None or not bbox:
            return None
        crop, _, _ = H.crop_and_mask(arr, bbox)
        return _png(crop)
    cand = (ch.get("candidates") or {}).get(eid) or {}
    src = cand.get("src") or ""
    # persisted file
    if "/compare/file/" in src:
        rel = src.split("/compare/file/", 1)[1].split("?", 1)[0]
        return file_bytes(c, rel)
    if "/compare/pack/" in src:
        rel = src.split("/compare/pack/", 1)[1].split("?", 1)[0]
        got = CMP.file_bytes(studio._compare_pack, rel) if studio._compare_pack else None
        return got[0] if got else None
    # last resort: load persisted path from hosted-edits.json
    rec = ((H.load_run(Path(c.dir) / "review") or {}).get("changes") or {}).get(occ_id) or {}
    psrc = ((rec.get("candidates") or {}).get(eid) or {}).get("src")
    if psrc:
        p = Path(c.dir) / "review" / psrc
        if p.is_file():
            return p.read_bytes()
    return None


def file_bytes(c, rel: str):
    if not rel or ".." in rel or rel.startswith("/"):
        return None
    if not re.fullmatch(r"edits/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.png", rel):
        return None
    p = (Path(c.dir) / "review" / rel).resolve()
    root = (Path(c.dir) / "review").resolve()
    if root not in p.parents:
        return None
    if p.is_file():
        return p.read_bytes()
    return None


def estimate(studio, c, body: dict):
    from .server import StudioError
    if c.status != "ready" or not c.meta:
        raise StudioError("NOT_ANALYZED")
    documents = body.get("documents") or []
    if not isinstance(documents, list) or not all(isinstance(d, str) for d in documents):
        raise StudioError("BAD_DECISIONS")
    occs = [o for o in studio.proposals(c).get("occurrences") or [] if o.get("text") not in (c.rejected_text or [])]
    return H.estimate_for_occurrences(occs, documents=documents, page_loader=page_loader_for(c))


def run(studio, c, body: dict):
    from .server import StudioError
    from casepatch.docs.byok import ByokRefused
    if c.status != "ready" or not c.meta:
        raise StudioError("NOT_ANALYZED")
    if studio._edit_transport is None:
        raise StudioError("EDIT_TRANSPORT_REQUIRED")
    documents = body.get("documents") or []
    if not isinstance(documents, list) or not documents or not all(isinstance(d, str) for d in documents):
        raise StudioError("HOSTED_EDIT_OPT_IN")
    if body.get("ack") is not True or body.get("notice_version") != NOTICE_VERSION:
        raise StudioError("HOSTED_EDIT_NOTICE_REQUIRED")
    key = studio._fal_key
    try:
        result = H.run_edits(
            [o for o in studio.proposals(c).get("occurrences") or [] if o.get("text") not in (c.rejected_text or [])],
            key=key, transport=studio._edit_transport, ack=True, documents=documents,
            accepted_usd=body.get("accepted_usd"), page_loader=page_loader_for(c),
            max_usd=float(body.get("max_usd") or 5.0), env=studio.env,
            review_dir=Path(c.dir) / "review", case_id=c.id,
        )
    except ByokRefused:
        raise StudioError("BAD_EDIT_KEY")
    except H.NeedOptIn:
        raise StudioError("HOSTED_EDIT_OPT_IN")
    except H.NeedAck:
        raise StudioError("HOSTED_EDIT_NOTICE_REQUIRED")
    except H.NeedEstimate as e:
        # include the fresh estimate so the UI can re-prompt
        raise StudioError("HOSTED_EDIT_ESTIMATE_REQUIRED", extra={"estimate": e.estimate})
    except H.TransportRequired:
        raise StudioError("EDIT_TRANSPORT_REQUIRED")
    model, _ = compare_model(studio, c)
    return {"ok": True, "spent_usd": result.get("spent_usd"), "estimated_usd": result.get("estimated_usd"),
            "errors": result.get("errors") or [], "compare": model}


def _resize(arr, hw):
    h, w = hw
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.LANCZOS))


def _candidate_image(studio, c, ch, editor):
    """RGB crop for a chosen editor, resized to the live crop box."""
    bbox = ch.get("crop")
    doc, page = ch.get("doc"), ch.get("page")
    if editor == E.LOCAL_EDITOR:
        arr = _load_rgb(preview_path(c, doc, page, "after")) or _load_rgb(preview_path(c, doc, page, "before"))
        if arr is None or not bbox:
            return None, None, None
        crop, mask, xy = H.crop_and_mask(arr, bbox)
        return crop, mask, xy
    raw = crop_bytes(studio, c, ch["id"], editor_token(editor))
    if not raw:
        return None, None, None
    cand = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    page_arr = _load_rgb(preview_path(c, doc, page, "before"))
    if page_arr is None or not bbox:
        return None, None, None
    orig, mask, xy = H.crop_and_mask(page_arr, bbox)
    cand = _resize(cand, orig.shape[:2])
    return cand, mask, xy


def apply_picks_to_outputs(studio, c, picks: E.Picks, model) -> dict[str, bytes]:
    """Return {output filename: bytes} for documents that have a non-local pick, mask-only paste.

    Image outputs are rewritten in place of the file contents. PDFs get the chosen crop stamped as an
    image annotation at the crop box (the rest of the page is not flattened).
    """
    out = {}
    by_doc = {}
    for ch in model.get("changes") or []:
        if picks.get(ch["id"]) == E.LOCAL_EDITOR:
            continue
        by_doc.setdefault((ch.get("doc"), ch.get("page")), []).append(ch)
    if not by_doc:
        return out
    out_dir = c.out_dir()
    for f in out_dir.iterdir():
        from .server import OUTPUT_NAME
        if not OUTPUT_NAME.fullmatch(f.name):
            continue
        doc, ext = f.stem, f.suffix.lower()
        pages_needed = [p for (d, p), chs in by_doc.items() if d == doc]
        if not pages_needed:
            continue
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
            page = _load_rgb(f)
            if page is None:
                continue
            page = np.array(page, copy=True)
            for ch in by_doc.get((doc, 0), []):
                editor = picks.get(ch["id"])
                crop, mask, xy = _candidate_image(studio, c, ch, editor)
                if crop is None:
                    continue
                x0, y0, x1, y1 = xy
                region = page[y0:y1, x0:x1]
                if region.shape[:2] != crop.shape[:2]:
                    crop = _resize(crop, region.shape[:2])
                    mask = _resize((mask > 0).astype(np.uint8) * 255, region.shape[:2])
                    mask = (mask > 127).astype(np.uint8)
                page[y0:y1, x0:x1] = E.paste_mask_only(region, crop, mask)
            out[f.name] = _png(page)
        elif ext == ".pdf":
            try:
                import pymupdf
            except ImportError:
                continue
            docpdf = pymupdf.open(f)
            try:
                changed = False
                for (d, pno), chs in by_doc.items():
                    if d != doc:
                        continue
                    if pno is None or pno < 0 or pno >= docpdf.page_count:
                        continue
                    page = docpdf[pno]
                    pw, ph = page.rect.width, page.rect.height
                    # preview pixels -> PDF points
                    prev = _load_rgb(preview_path(c, doc, pno, "before"))
                    sx = pw / prev.shape[1] if prev is not None else 1.0
                    sy = ph / prev.shape[0] if prev is not None else 1.0
                    for ch in chs:
                        editor = picks.get(ch["id"])
                        crop, mask, xy = _candidate_image(studio, c, ch, editor)
                        if crop is None:
                            continue
                        x0, y0, x1, y1 = xy
                        # paste mask-only onto the original crop, then stamp that crop
                        orig_page = prev
                        if orig_page is not None:
                            region = orig_page[y0:y1, x0:x1]
                            if region.shape[:2] == crop.shape[:2]:
                                crop = E.paste_mask_only(region, crop, mask)
                        rect = pymupdf.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                        page.insert_image(rect, stream=_png(crop))
                        changed = True
                if changed:
                    buf = io.BytesIO()
                    docpdf.save(buf, deflate=True)
                    out[f.name] = buf.getvalue()
            finally:
                docpdf.close()
    return out


def export_review_fields(studio, c):
    """Counts-only editor log for review.json. No source values."""
    try:
        model, picks = compare_model(studio, c)
    except Exception:
        return {}
    if not model.get("changes"):
        return {}
    log = E.review_log(picks, badges_by_change(model), include_values=False)
    return {"editors_chosen": log["editors_chosen"], "editor_default": log["default"],
            "editor_changes": [{"id": r["id"], "editor": r["editor"], "readback": r["readback"],
                                "leak": r["leak"], "leak_override": r["leak_override"],
                                "seam": r["seam"], "outside_mask": r["outside_mask"]}
                               for r in log["changes"]]}
