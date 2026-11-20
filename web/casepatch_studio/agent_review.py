"""Agent-mode results in the Studio review screen: the agent's open review items and every agent edit as a picker row
(agent edit vs original, with before/after crops and verifier badges). Agent mode runs only with the user's own
Anthropic key (BYOK) after an explicit opt-in; nothing goes through CasePatch servers.

Layout under the case dir:  agent/files/<input name>  agent/edits/<input name>/<edit id>.png  agent/agent-report.json
                            agent/review-issues.json   agent/picks.json"""
from __future__ import annotations

import io, json, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

NOTICE = ("Agent mode sends page images and their text to Anthropic with your own API key, for this case only. A model reads "
          "the documents and edits them with CasePatch's local tools; CasePatch's checks then verify every edit. Off by default.")
NOTICE_VERSION = "agent-2026-11-13"
BLOCKING = {"leak", "ghost", "hidden"}   # an edit with one of these cannot be the final pick without an override


def adir(c):
    return Path(c.dir) / "agent"


def status(c):
    d = adir(c)
    if (d / "agent-report.json").exists():
        r = json.loads((d / "agent-report.json").read_text())
        return {"state": "done", "spent_usd": r.get("spent_usd"), "wall_s": r.get("wall_s"), "open_items": len(r.get("review_issues") or [])}
    if (d / "running").exists():
        return {"state": "running"}
    return {"state": "none"}


def _report(c):
    p = adir(c) / "agent-report.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_picks(c):
    p = adir(c) / "picks.json"
    v = json.loads(p.read_text()) if p.exists() else {}
    return {"chosen": v.get("chosen") or {}, "overrides": v.get("overrides") or [], "checked": v.get("checked") or []}


def _key(doc, edit_id):
    return f"{doc}#{edit_id}"


def _ov(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0


def payload(c):
    """Review items + picker rows. Values (originals/fakes) stay on this computer; this is the local UI."""
    r = _report(c)
    if r is None:
        return {"status": status(c), "items": [], "changes": [], "overrides": [], "notice": NOTICE, "notice_version": NOTICE_VERSION}
    picks = load_picks(c)
    labels = {f["name"]: f.get("original") or f["name"] for f in c.files}
    rows, items = [], []
    for d in r["documents"]:
        doc = d["file"]
        issues = d.get("open_issues") or []
        raster_pages = {e["page"] for e in d.get("edits") or [] if e.get("kind") != "pdf_text" and not e.get("undone")}
        for e in d.get("edits") or []:
            if e.get("undone"):
                continue
            k = _key(doc, e["id"])
            own = [i for i in issues if (i.get("edit_id") == e["id"]) or (i.get("bbox") and i.get("page", 0) == e["page"] and _ov(i["bbox"], e["bbox"]))]
            blocked = sorted({i["check"] for i in own if i["check"] in BLOCKING})
            rows.append({"id": k, "doc": doc, "doc_label": labels.get(doc, doc), "page": e["page"], "bbox": e["bbox"], "kind": e["kind"], "original": e["original"],
                         "replacement": e["replacement"], "issues": [i.get("detail") or i["check"] for i in own],
                         "badges": {"agent": {"selectable": not blocked, "blocked_by": blocked},
                                    "original": {"selectable": _revertible(doc, e, raster_pages)}},
                         "pick": picks["chosen"].get(k, "agent"),
                         "before_src": f"/api/cases/{c.id}/agent/crop/{k.replace('#', '~')}/before",
                         "after_src": f"/api/cases/{c.id}/agent/crop/{k.replace('#', '~')}/after"})
        for n, i in enumerate(issues):
            items.append({"id": f"{doc}#{n}", "doc": doc, "doc_label": labels.get(doc, doc), "check": i.get("check"),
                          "must_check": i.get("check") in BLOCKING, "checked": f"{doc}#{n}" in set(picks.get("checked") or []), "detail": i.get("detail") or (f"{i.get('check')}: {i.get('value', '')}"),
                          "page": i.get("page", 0), "bbox": i.get("bbox"),
                          "src": f"/api/cases/{c.id}/agent/issue/{doc}/{n}" if i.get("bbox") else None})
    return {"status": status(c), "items": items, "changes": rows, "overrides": picks["overrides"], "notice": NOTICE, "notice_version": NOTICE_VERSION}


def _revertible(doc, e, raster_pages):
    """Images: every edit (mask revert). PDFs: text-layer edits on pages kept as text (re-applied from the original);
    edits on a page the agent turned into an image cannot be reverted one by one."""
    if not doc.lower().endswith(".pdf"):
        return True
    return e.get("kind") == "pdf_text" and e["page"] not in raster_pages


def _rebuild_pdf(c, d, picks):
    """Original PDF + every kept text-layer edit re-applied at its own position (each recorded edit is one occurrence);
    pages the agent rasterised are taken from the agent output as they are."""
    import tempfile
    import pymupdf
    from casepatch.agent.tools import DocState
    doc = d["file"]
    st = DocState.load(Path(c.input) / doc)
    raster_pages = {e["page"] for e in d["edits"] if e.get("kind") != "pdf_text" and not e.get("undone")}
    failed = []
    for e in d["edits"]:
        if e.get("undone") or e.get("kind") != "pdf_text" or e["page"] in raster_pages:
            continue
        if picks.get(_key(doc, e["id"])) == "original":
            continue
        if e.get("replacement") == "" and e.get("original"):
            r = st.pdf_erase(e["page"], e["bbox"]) or {"ok": False}
        else:
            r = st.pdf_replace(e["page"], e["original"], e["replacement"], near=e["bbox"])
        if not r.get("ok"):
            failed.append(e["id"])
    with tempfile.TemporaryDirectory() as t:
        out = st.save(t)
        with pymupdf.open(out) as pd:
            if raster_pages:
                with pymupdf.open(adir(c) / "files" / doc) as ag:
                    for pno in sorted(raster_pages):
                        pd.delete_page(pno); pd.insert_pdf(ag, from_page=pno, to_page=pno, start_at=pno)
            pd.set_metadata({}); pd.del_xml_metadata()
            data = pd.tobytes(garbage=3, deflate=True)
    if failed:
        raise RuntimeError(f"{doc}: kept edits {failed} could not be re-applied; export stopped so nothing slips through")
    return data


def _load(p, page=0):
    if str(p).lower().endswith(".pdf"):   # same 200 dpi raster the agent's boxes are in
        import pymupdf
        with pymupdf.open(p) as d:
            pm = d[page].get_pixmap(dpi=200)
            return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[..., :3].copy()
    return np.asarray(Image.open(p).convert("RGB"))


def _png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG"); return b.getvalue()


def _safe_doc(c, doc):
    names = {f["name"] for f in c.files}
    return doc if doc in names else None


def _crop(arr, bbox, pad=24):
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return arr[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]


def crop_bytes(c, token, side):
    doc, _, eid = token.replace("~", "#").partition("#")
    doc = _safe_doc(c, doc)
    r = _report(c)
    if not doc or r is None or side not in ("before", "after"):
        return None
    e = next((e for d in r["documents"] if d["file"] == doc for e in d["edits"] if str(e["id"]) == eid), None)
    if not e:
        return None
    arr = _load((Path(c.input) if side == "before" else adir(c) / "files") / doc, e["page"])
    return _png(_crop(arr, e["bbox"]))


def issue_bytes(c, doc, n):
    doc = _safe_doc(c, doc); r = _report(c)
    if not doc or r is None:
        return None
    d = next((d for d in r["documents"] if d["file"] == doc), None)
    iss = (d or {}).get("open_issues") or []
    if not (0 <= int(n) < len(iss)) or not iss[int(n)].get("bbox"):
        return None
    return _png(_crop(_load(adir(c) / "files" / doc, iss[int(n)].get("page", 0)), iss[int(n)]["bbox"]))


def save_picks(c, body):
    r = _report(c)
    if r is None:
        raise ValueError("no agent run")
    model = {x["id"]: x for x in payload(c)["changes"]}
    cur = load_picks(c)
    for k, v in (body.get("chosen") or {}).items():
        if k not in model or v not in ("agent", "original") or not model[k]["badges"][v]["selectable"] and v == "original":
            raise ValueError("bad pick")
        cur["chosen"][k] = v
    ov = set(cur["overrides"]) | {k for k in body.get("override") or [] if k in model}
    for k, v in cur["chosen"].items():  # a blocked agent edit can only stay picked with an explicit override
        if v == "agent" and not model[k]["badges"]["agent"]["selectable"] and k not in ov:
            raise PermissionError(k)
    cur["overrides"] = sorted(ov)
    items = {i["id"] for i in payload(c)["items"]}
    for k in body.get("checked") or []:
        if k not in items:
            raise ValueError("bad item")
    cur["checked"] = sorted(set(cur["checked"]) | set(body.get("checked") or []))
    (adir(c) / "picks.json").write_text(json.dumps(cur))
    return cur


def apply_picks(c):
    """{input name: bytes} of the final files: agent output, with 'original' picks reverted inside their edit masks."""
    r = _report(c); picks = load_picks(c)["chosen"]
    out = {}
    for d in r["documents"]:
        doc = d["file"]; f = adir(c) / "files" / doc
        reverts = [e for e in d["edits"] if picks.get(_key(doc, e["id"])) == "original" and not e.get("undone")]
        if not reverts:
            out[doc] = f.read_bytes(); continue
        if doc.lower().endswith(".pdf"):
            out[doc] = _rebuild_pdf(c, d, picks); continue
        a = _load(f).copy(); o = _load(Path(c.input) / doc)
        for e in reverts:
            mp = adir(c) / "edits" / doc / f"{e['id']}.png"
            if mp.exists():
                m = np.asarray(Image.open(mp)) > 0
                a[m] = o[m]
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG" if doc.lower().endswith((".jpg", ".jpeg")) else "PNG", quality=92)
        out[doc] = b.getvalue()
    return out


def unresolved(c):
    """Agent edits a check blocked (leak/ghost/hidden) still picked without an override, and leak/ghost/hidden review
    items nobody has marked as checked (a value the agent never edited has no edit row to block)."""
    p = payload(c); ov = set(p.get("overrides") or [])
    return [r["id"] for r in p["changes"] if r["pick"] == "agent" and not r["badges"]["agent"]["selectable"] and r["id"] not in ov] + \
           [i["id"] for i in p["items"] if i["must_check"] and not i["checked"]]


def export_zip(c):
    bad = unresolved(c)
    if bad:
        raise PermissionError(bad)
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in apply_picks(c).items():
            orig = next((f["original"] for f in c.files if f["name"] == name), name)
            zf.writestr(orig, data)
    return z.getvalue()
