"""Studio helpers for the precomputed per-change editor comparison. No live model calls."""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

from casepatch.docs import editor_pick as E

# default pack, built from public samples and shipped with the demo
DEFAULT_PACK = Path(__file__).resolve().parents[1] / "demo" / "cases" / "compare"
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_PATH = re.compile(r"^(crops|pages)/[A-Za-z0-9._-]+$")


def pack_dir(explicit=None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if (p / "pack.json").exists() else None
    if (DEFAULT_PACK / "pack.json").exists():
        return DEFAULT_PACK
    return None


def load(explicit=None):
    d = pack_dir(explicit)
    if not d:
        return None
    return E.load_pack(d)


def new_picks(pack) -> E.Picks:
    return E.pack_picks(pack)


def apply_pick(picks: E.Picks, pack, body: dict):
    badges = E.pack_badges(pack)
    if body.get("all"):
        editor = body.get("editor")
        applied, skipped = picks.select_all(editor, badges, override_leaks=bool(body.get("override")))
        return {"chosen": picks.chosen, "overrides": sorted(f"{c}\t{e}" for c, e in picks.overrides),
                "applied": applied, "skipped": skipped}
    cid, editor = body.get("change_id"), body.get("editor")
    picks.select(cid, editor, badges.get(cid) or {}, override=bool(body.get("override")))
    return {"chosen": picks.chosen, "overrides": sorted(f"{c}\t{e}" for c, e in picks.overrides)}


def client_payload(pack, picks: E.Picks, prefix="/api/compare/file/"):
    data = E.pack_for_client(pack, image_prefix=prefix)
    # after-page URLs (composed from current picks; cache-busted by the client)
    for p in data["pages"]:
        p["after_src"] = f"/api/compare/page/{_page_token(p['id'])}/after"
    data["picks"] = dict(picks.chosen)
    data["overrides"] = [f"{c}\t{e}" for c, e in picks.overrides]
    data["precomputed"] = True
    return data


def _page_token(page_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in page_id)


def page_id_from_token(pack, token: str):
    for p in pack.get("pages") or []:
        if _page_token(p["id"]) == token:
            return p["id"]
    return None


def file_bytes(pack, rel: str) -> tuple[bytes, str] | None:
    if not SAFE_PATH.match(rel or ""):
        return None
    path = Path(pack["_dir"]) / rel
    try:
        path = path.resolve()
        root = Path(pack["_dir"]).resolve()
        if root not in path.parents and path != root:
            return None
    except OSError:
        return None
    if not path.is_file():
        return None
    ctype = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return path.read_bytes(), ctype


def after_page_png(pack, page_id: str, picks: E.Picks) -> bytes:
    from PIL import Image
    arr = E.compose_pack_page(pack, page_id, picks)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def export_zip(pack, picks: E.Picks) -> bytes:
    if not E.export_applies(picks, E.pack_badges(pack)):
        raise E.LeakBlocked("A selected candidate still shows the original text. Override it explicitly or pick another editor.")
    log = E.review_log(picks, E.pack_badges(pack))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pack.get("pages") or []:
            if not p.get("src"):
                continue
            token = _page_token(p["id"])
            before = Path(pack["_dir"]) / p["src"]
            if before.exists():
                z.write(before, f"pages/{token}-before.jpg")
            z.writestr(f"pages/{token}-after.png", after_page_png(pack, p["id"], picks))
        z.writestr("review.json", json.dumps({
            "casepatch_studio_compare": True,
            "precomputed_public_samples": True,
            "reviewed_by_user": True,
            "editors_chosen": log["editors_chosen"],
            "default_editor": log["default"],
            "changes": log["changes"],
            "note": "Which editor was chosen per change, with automatic badges. No source identity map or case key.",
        }, indent=2))
        z.writestr("README.txt", (
            "CasePatch editor-comparison export (precomputed public samples)\n\n"
            "pages/   Before JPEGs and after PNGs with your chosen editor applied to each change,\n"
            "         pasted strictly inside the change mask.\n"
            "review.json  Which editor was selected for each change, plus OCR / leak / seam badges.\n"
            "             No identity map or case key is included.\n\n"
            "These pages were built from precomputed outputs on public sample documents. No live\n"
            "model call was made for this export. CasePatch does not certify anonymity.\n"
        ))
    return buf.getvalue()
