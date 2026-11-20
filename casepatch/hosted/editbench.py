"""Bounded crop-edit benchmark (bench/models/EDIT_BENCHMARK.md): hosted image models vs the local native editor on
the SAME crops of the public sample cases.

    python -m casepatch.hosted.editbench local  --out DIR            # $0: local glyph-matched editor (native.replace_text)
    python -m casepatch.hosted.editbench hosted --out DIR --model M  # PAID, capped by the shared ledger (OpenAI only)
    python -m casepatch.hosted.editbench score  --out DIR            # $0: metrics for every editor in DIR -> JSON + table

Per target (same deterministic targets as `modeleval edit`: 4 raster values per case, crop = value box + margins,
mask = the value's character boxes):
  * control_readable   OCR reads the ORIGINAL value in the original crop (else readback is not meaningful)
  * exact              OCR of the edited crop contains the replacement string exactly (whitespace-normalised)
  * fuzzy              ... fuzzy >= 0.8 (the old readback_ok)
  * residual           OCR still reads the original value (must be False)
  * ghost              original value recovered from the residual-ink layer / contrast-enhanced crop (hidden content)
  * page_leak          after pasting the edited crop back, a full-page OCR still finds the original value anywhere
  * out_of_mask        pixels changed outside the value mask in the SHIPPED crop (max abs diff, fraction > 25);
                       hosted composites are 0 by construction, the local editor may legitimately reflow a line
  * raw_out_of_mask    hosted only: how much the model repainted outside the mask before compositing
  * seam               mean |gradient| on the mask border ring, edited minus original (0 = invisible boundary)
  * ink_delta / height_ratio  style consistency: median ink colour difference and ink-height ratio new/original text
Images are written to DIR (outside the repo; FUNSD/SROIE crops are research-only and must not be used in demos).
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import modeleval as ME


FAL_CREDIT_USD = 25.0  # fal.ai account credit used as an eval safety bound (no auto-recharge)


def targets(cases=None, work=ME.WORK, n=ME.EVAL_EDIT_TARGETS_PER_CASE, tids=None):
    ok, _ = ME.available_cases(cases)
    out = []
    for c in ok:
        pages = ME.prepare_case(c, work)
        for i, t in enumerate(ME.edit_targets(c, pages, ME.load_labels(c), n)):
            out.append(dict(t, tid=f"{c}-{i}"))
    if tids:
        out = [t for t in out if t["tid"] in set(tids)]
    return out


def _load(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


def _save(arr, p):
    from PIL import Image
    Image.fromarray(arr).save(p)


# ------------------------------------------------------------------------------------------------ editors
def run_local(out, cases=None, work=ME.WORK):
    from ..docs import native
    out = Path(out) / "local"; out.mkdir(parents=True, exist_ok=True)
    meta = []
    pages = {}
    for t in targets(cases, work):
        key = (t["case"], t["png"])
        if key not in pages:
            pages[key] = next(p for p in ME.prepare_case(t["case"], work) if p["png"] == t["png"])
        page = pages[key]
        img = _load(t["png"]).copy(); orig = img.copy()
        line = next(l for l in page["lines"] if t["value"] in l["text"] and len(l["chars"]) == len(l["text"]))
        x0, y0, x1, y1 = line["box"]
        quad = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        s = line["text"].find(t["value"])
        t0 = time.perf_counter()
        try:
            m, info = native.replace_text(img, quad, line["text"], s, s + len(t["value"]), t["replacement"],
                                          np.random.default_rng(7), style_cache={}, orig_img=orig)
            err = None if m is not None else info.get("error")
        except Exception as e:  # editor failure is a result, not a crash
            err = f"{type(e).__name__}: {e}"[:200]
        cx0, cy0, cx1, cy1 = t["crop"]
        _save(img[cy0:cy1, cx0:cx1], out / f"{t['tid']}.png")
        meta.append({"tid": t["tid"], "error": err, "latency_s": round(time.perf_counter() - t0, 2), "cost_usd": 0.0,
                     "replacement": t["replacement"]})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def dry_run(models, cases=None, work=ME.WORK):
    """$0, no network: worst-case and expected cost of running each model once on every edit target."""
    from . import pricing as P
    ts = targets(cases, work)
    rows = []
    for model in models:
        prov = "openai" if model.startswith("gpt-image") else "fal" if "/" in model else "gemini"
        worst = exp = 0.0
        for t in ts:
            x0, y0, x1, y1 = t["crop"]; w, h, _ = ME.fit_dims(x1 - x0, y1 - y0, ME.EVAL_MAX_IMAGE_SIDE)
            if prov == "fal":
                worst += P.worst_case_usd("fal", model, 120, [(w, h)], 0, output_images=1)
                exp += P.fal_usd(model, 120, [(w, h)], 0, 1, safety=1.0)
            else:
                worst += P.worst_case_usd(prov, model, 120, [(w, h)] * 2, 0 if prov == "openai" else 256, output_images=1, out_side=max(w, h))
        rows.append({"model": model, "provider": prov, "targets": len(ts), "worst_usd": round(worst, 4),
                     "expected_usd": round(exp, 4) if prov == "fal" else None})
    return rows


def run_hosted(out, model, cases=None, work=ME.WORK, env=None, transport_factory=None, tids=None):
    """PAID. Same gates as modeleval.run_hosted: hosted flag, total cap <= approved, pool cap, worst case of the whole
    batch must fit the remaining budget before the first call; every call reserves in the shared ledger first."""
    from . import client as C
    from . import pricing as P
    from .config import load_config, HostedDisabled
    from .ledger import CapExceeded, SpendLedger
    env = os.environ if env is None else env
    prov = "openai" if model.startswith("gpt-image") else "fal" if "/" in model else "gemini"
    total, subs = ME.eval_env_check(env, [prov])
    cfg = load_config(env, prov)
    cfg = dataclasses.replace(cfg, max_image_side=ME.EVAL_MAX_IMAGE_SIDE, max_calls_per_case=500, timeout_s=180.0)
    if prov == "fal":  # the fal account credit is a hard ceiling too: a fal sub-cap <= the credit is required
        if subs[prov][0] is None or subs[prov][0] > FAL_CREDIT_USD + 1e-9:
            raise HostedDisabled(f"fal needs CASEPATCH_PROVIDER_CAP_FAL_USD <= the ${FAL_CREDIT_USD:.0f} account credit")
        cfg = dataclasses.replace(cfg, max_retries=0)  # one attempt per edit: no retries of a failed fal call
    ledger = SpendLedger(cfg.ledger_path)
    client = C.HostedClient(cfg, transport=transport_factory(cfg) if transport_factory else None, ledger=ledger)
    editor = C.ImageEditor(client, model=model)
    ts = targets(cases, work, tids=tids)
    crops = [ME._crop_and_mask(t) for t in ts]
    worst = sum(editor.estimate(c) for c, _ in crops)
    sub_cap, pool, pool_cap = subs[prov]
    qt, at = ledger.pool_totals(pool), ledger.totals()
    remaining = min(pool_cap - qt["spent_usd"] - qt["reserved_usd"], total - at["spent_usd"] - at["reserved_usd"])
    if sub_cap is not None:
        pt = ledger.provider_totals(prov); remaining = min(remaining, sub_cap - pt["spent_usd"] - pt["reserved_usd"])
    if worst > remaining:
        raise HostedDisabled(f"worst case ${worst:.2f} exceeds remaining {pool} budget ${remaining:.2f}")
    d = Path(out) / model.replace("/", "__"); d.mkdir(parents=True, exist_ok=True)
    meta, stopped, fails = [], None, 0
    orig_decode = C.decode_image
    for t, (crop, mask) in zip(ts, crops):
        cap = {}

        def spy(data, provider):
            try:
                b = (base64.b64decode(data["data"][0]["b64_json"]) if provider == "openai"
                     else C.fal_image_bytes(data) if provider == "fal" else None)
                if b:
                    from PIL import Image
                    im = Image.open(io.BytesIO(b)); cap["mode"], cap["size"] = im.mode, im.size
                    (d / f"{t['tid']}.model.png" if prov == "fal" else d / f"{t['tid']}.raw.png").write_bytes(b)
            except Exception:
                pass
            r = orig_decode(data, provider); cap["raw"] = r
            return r
        C.decode_image = spy
        t0 = time.perf_counter(); err = None
        try:
            comp = editor.edit(crop, mask, ME.edit_instruction(t), None, f"editbench:{t['case']}:{prov}:{model}")
        except CapExceeded as e:
            stopped = f"cap: {e}"; err = stopped; comp = None
        except Exception as e:
            err = ME.redact(e, [cfg.key()])[:300]; comp = None
        finally:
            C.decode_image = orig_decode
        last = ledger.last(f"editbench:{t['case']}:{prov}:{model}")
        cost = last["cost_usd"] if last else None  # committed = provider usage; kept_estimate = full reservation kept
        if comp is not None:
            _save(comp, d / f"{t['tid']}.png")
        if prov == "fal" and getattr(editor, "last_raw", None) is not None:  # model image mapped back onto the crop
            from PIL import Image
            Image.fromarray(editor.last_raw).save(d / f"{t['tid']}.raw.png")
        if cap.get("raw") is not None:
            _save(cap["raw"], d / f"{t['tid']}.rawrgb.png")
        meta.append({"tid": t["tid"], "error": err, "latency_s": round(time.perf_counter() - t0, 2), "cost_usd": cost,
                     "raw_mode": cap.get("mode"), "raw_size": cap.get("size"), "replacement": t["replacement"]})
        if err:
            fails += 1
            if stopped or ME.is_fatal_billing_or_auth(err):
                stopped = stopped or "billing/auth error (no retries): " + err[:160]; break
            if fails >= 2 and not any(m["error"] is None for m in meta):
                stopped = "two failed edit calls and no success"; break
    ledger.close()
    prev = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() and tids else None
    if prev:  # subset runs accumulate: newest result per target wins
        keep = {m["tid"]: m for m in prev["edits"]}; keep.update({m["tid"]: m for m in meta})
        meta_all = list(keep.values()); worst += prev.get("worst_case_usd", 0)
    else:
        meta_all = meta
    (d / "meta.json").write_text(json.dumps({"model": model, "stopped": stopped, "worst_case_usd": round(worst, 4),
                                             "edits": meta_all}, indent=1))
    return meta, stopped


# ------------------------------------------------------------------------------------------------ scoring
_CTX = {}


def _context(t, work=ME.WORK):
    """Document context of a target's page (for the plausibility check): cached per case."""
    from ..docs import consistency as CC
    if t["case"] not in _CTX:
        pages = ME.prepare_case(t["case"], work)
        _, g, _ = CC.plan_labelled_case(t["case"], pages, ME.load_labels(t["case"]))
        _CTX[t["case"]] = g
    return _CTX[t["case"]].contexts.get(t["file"])


def consistency_issues(t, replacement):
    """Plausibility of the replacement an editor was asked to write, in its document's context (e.g. a birth year
    after the document date, an age mismatch, a date that is not a date, a name that is not a name)."""
    from ..docs import consistency as CC
    kind = t.get("kind") or CC.refine_kind(None, t["value"])
    try:
        ctx = _context(t)
    except Exception:
        ctx = None
    return CC.plausibility(kind or "ID", t["value"], replacement, ctx)


def post_edit_issues(t, ocr_text):
    """Post-edit verifier on the re-read crop: for dates/years, every date-shaped token read near the edit that is
    not the original must be plausible in context (catches an editor that wrote a different, absurd value)."""
    from ..docs import consistency as CC
    kind = t.get("kind")
    if kind != "DOB":
        return []
    try:
        ctx = _context(t)
    except Exception:
        ctx = None
    issues = []
    rx = r"(?<!\d)\d{4}(?!\d)" if re.fullmatch(r"\d{4}", t["value"]) else CC.DATE_RX.pattern
    for m in re.finditer(rx, ocr_text or ""):
        v = m.group()
        if v == t["value"]:
            continue
        issues += [i for i in CC.plausibility("DOB", t["value"], v, ctx) if i not in ("DATE_FORMAT_CHANGED", "UNCHANGED")]
    return sorted(set(issues))


def _ocr(img):
    import cv2
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return ME._ocr_text(big)


def _nrm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _exact(value, text):
    v, t = _nrm(value), _nrm(text)
    return bool(v) and (v in t or v.replace(" ", "") in t.replace(" ", ""))


def _ghost(orig, edited, mask, value):
    """Hidden content: is the original value recoverable from the edited crop (residual ink layer / CLAHE)?"""
    import cv2
    from .evalscore import fuzzy_in
    ga, gb = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
    k = 31 if min(gb.shape) > 31 else 3
    ink = cv2.adaptiveThreshold(ga, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10) > 0
    dark = np.clip(cv2.medianBlur(gb, k).astype(np.int16) - gb.astype(np.int16), 0, 255)
    dark0 = np.clip(cv2.medianBlur(ga, k).astype(np.int16) - ga.astype(np.int16), 1, 255)
    keep = ink & (dark >= 0.4 * dark0) & (dark >= 12) & (mask > 0)
    res = np.full_like(gb, 255); res[keep] = 0
    clahe = cv2.createCLAHE(3.0, (8, 8)).apply(gb)
    for name, im in (("residual_layer", res), ("enhanced", clahe)):
        txt = _ocr(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
        if (len(value) >= 5 and fuzzy_in(value, txt, 0.85)) or _exact(value, txt):
            return name
    # ink persistence inside the mask (letters only: digit swaps in the same font correlate by design)
    ys, xs = np.where(mask > 0)
    if sum(ch.isalpha() for ch in value) >= 4 and len(ys):
        ta = ga[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
        tb = gb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
        if ta.std() > 5 and tb.std() > 5:
            corr = float(np.corrcoef(ta.ravel(), tb.ravel())[0, 1])
            pers = float(keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1].sum()) / max(1.0, float((ink & (mask > 0)).sum()))
            if corr >= 0.75 and pers >= 0.6:
                return f"ink_persistence({corr:.2f})"
    return None


def _style(orig, edited, mask):
    import cv2
    ga, gb = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
    m = mask > 0
    ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) - cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8))
    ring = ring > 0
    grad = lambda g: np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1))
    seam = float(grad(gb)[ring].mean() - grad(ga)[ring].mean()) if ring.any() else 0.0

    def ink(img, g):
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10) > 0
        sel = thr & m
        if sel.sum() < 5:
            return None, None
        ys = np.where(sel.any(axis=1))[0]
        return np.median(img[sel], axis=0), float(ys.max() - ys.min() + 1)
    ca, ha = ink(orig, ga); cb, hb = ink(edited, gb)
    return {"seam": round(seam, 2),
            "ink_delta": None if ca is None or cb is None else round(float(np.abs(ca.astype(float) - cb.astype(float)).max()), 1),
            "height_ratio": None if not ha or not hb else round(hb / ha, 2)}


def score(out, cases=None, work=ME.WORK, page_check=True, tids=None):
    import cv2
    from .evalscore import fuzzy_in
    out = Path(out)
    ts = targets(cases, work, tids=tids)
    ctrl = {}
    editors = sorted(p.name for p in out.iterdir() if p.is_dir() and (p / "meta.json").exists())
    res = {"targets": len(ts), "editors": {}}
    page_txt_cache = {}
    for t in ts:
        crop, _ = ME._crop_and_mask(t)
        ctrl[t["tid"]] = _exact(t["value"], _ocr(crop)) or fuzzy_in(t["value"], _ocr(crop), 0.8)
    for e in editors:
        meta = json.loads((out / e / "meta.json").read_text())
        meta = meta["edits"] if isinstance(meta, dict) else meta
        mi = {m["tid"]: m for m in meta}
        rows = []
        for t in ts:
            crop, mask = ME._crop_and_mask(t)
            p = out / e / f"{t['tid']}.png"
            m = mi.get(t["tid"], {})
            # the replacement this editor was actually asked for: outputs made before context-aware planning carry no
            # 'replacement' in their meta and used the legacy character shuffle -> stale (needs regeneration)
            asked = m.get("replacement") or t.get("legacy_replacement") or t["replacement"]
            r = {"tid": t["tid"], "value": t["value"], "replacement": asked, "intended_replacement": t["replacement"],
                 "stale": asked != t["replacement"], "control_readable": ctrl[t["tid"]],
                 "cost_usd": m.get("cost_usd"), "latency_s": m.get("latency_s"), "error": m.get("error"),
                 "plausibility": consistency_issues(t, asked)}
            r["plausible"] = not r["plausibility"]
            if not p.exists():
                rows.append(dict(r, ok=False)); continue
            ed = _load(p)
            if ed.shape != crop.shape:
                rows.append(dict(r, ok=False, error="shape mismatch")); continue
            txt = _ocr(ed)
            diff = np.abs(ed.astype(int) - crop.astype(int)).max(axis=2)
            outm = mask == 0
            unchanged = int(diff.max()) == 0
            r.update(ok=not unchanged, ocr=txt[:160], exact=_exact(asked, txt),
                     fuzzy=fuzzy_in(asked, txt, 0.8), post_edit=post_edit_issues(t, txt),
                     residual=_exact(t["value"], txt) or fuzzy_in(t["value"], txt, 0.8),
                     ghost=_ghost(crop, ed, mask, t["value"]),
                     out_of_mask_max=int(diff[outm].max()) if outm.any() else 0,
                     out_of_mask_frac=round(float((diff[outm] > 25).mean()), 4) if outm.any() else 0.0, **_style(crop, ed, mask))
            rawp = out / e / f"{t['tid']}.raw.png"
            if rawp.exists():
                from PIL import Image
                from .client import flatten_over
                raw = flatten_over(np.asarray(Image.open(rawp).convert("RGBA")), crop)  # what the model painted
                rd = np.abs(raw.astype(int) - crop.astype(int)).max(axis=2)
                r.update(raw_out_of_mask_mean_abs=round(float(rd[outm].mean()), 1), raw_out_of_mask_frac=round(float((rd[outm] > 25).mean()), 3),
                         raw_mode=m.get("raw_mode"), raw_size=m.get("raw_size"))
            if page_check:  # paste back and OCR the whole page: the original must not appear anywhere
                page = _load(t["png"]).copy(); x0, y0, x1, y1 = t["crop"]; page[y0:y1, x0:x1] = ed
                pt = ME._ocr_text(page)
                before = page_txt_cache.setdefault(t["png"], ME._ocr_text(_load(t["png"])))
                n0 = _nrm(before).count(_nrm(t["value"]))
                r["page_leak"] = _nrm(pt).count(_nrm(t["value"])) >= max(1, n0)  # every original occurrence still read
            # an edit passes only if the text is exact, nothing leaks, and the value makes sense in its document
            r["pass"] = bool(r.get("exact") and not r.get("residual") and not r.get("ghost") and not r.get("page_leak")
                             and r["plausible"] and not r.get("post_edit") and not r["stale"])
            rows.append(r)
        done = [r for r in rows if r.get("ok")]
        ctl = [r for r in done if r["control_readable"]]
        f = lambda k, rs: sum(bool(r.get(k)) for r in rs)
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        agg = {"attempted": len([t for t in ts if t["tid"] in mi]), "returned": len(done),
               "exact": f"{f('exact', done)}/{len(done)}", "exact_on_readable_controls": f"{f('exact', ctl)}/{len(ctl)}",
               "fuzzy": f"{f('fuzzy', done)}/{len(done)}", "residual": f("residual", done), "ghost": f("ghost", done),
               "page_leak": f("page_leak", done),
               "plausible": f"{f('plausible', rows)}/{len(rows)}", "stale": f("stale", rows),
               "post_edit_violations": sum(bool(r.get("post_edit")) for r in done), "pass": f"{f('pass', rows)}/{len(rows)}",
               "out_of_mask_changed_targets": sum(r["out_of_mask_max"] > 0 for r in done),
               "out_of_mask_frac_mean": round(float(np.mean([r["out_of_mask_frac"] for r in done])), 4) if done else None,
               "raw_out_of_mask_frac_mean": round(float(np.mean([r["raw_out_of_mask_frac"] for r in done if "raw_out_of_mask_frac" in r])), 3)
               if any("raw_out_of_mask_frac" in r for r in done) else None,
               "seam_mean": round(float(np.mean([r["seam"] for r in done])), 2) if done else None,
               "ink_delta_median": float(np.median([r["ink_delta"] for r in done if r["ink_delta"] is not None])) if done else None,
               "height_ratio_median": float(np.median([r["height_ratio"] for r in done if r["height_ratio"] is not None])) if done else None,
               "usd_per_edit": round(sum(costs) / len(costs), 4) if costs else 0.0,
               "p50_latency_s": float(np.median([r["latency_s"] for r in rows if r.get("latency_s") is not None])) if rows else None}
        res["editors"][e] = {"aggregate": agg, "per_target": rows}
    res["controls_readable"] = f"{sum(ctrl.values())}/{len(ctrl)}"
    (out / "editbench.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return res


def recomposite(out, model, cases=None, work=ME.WORK):
    """$0. Rebuild composites from the saved raw model images with the current client compositing (alpha-aware), and
    keep the previous composites as '<model>-rgbdrop' (alpha dropped: the behaviour before the fix). Costs are re-read
    from the ledger rows of this benchmark (one row per target, in order)."""
    import shutil
    from PIL import Image
    from .client import composite_inside_mask
    from .config import load_config
    d = Path(out) / model.replace("/", "__"); old = Path(out) / f"{model.replace('/', '__')}-rgbdrop"
    if not old.exists():
        shutil.copytree(d, old, ignore=shutil.ignore_patterns("*.raw.png", "*.rawrgb.png"))
    meta = json.loads((d / "meta.json").read_text())
    led = os.environ.get("CASEPATCH_SPEND_LEDGER")
    rows = []
    if led and Path(led).exists():
        import sqlite3
        db = sqlite3.connect(led)
        rows = db.execute("SELECT case_id, cost_usd, status FROM calls WHERE case_id LIKE ? ORDER BY created, rowid",
                          (f"editbench:%:{model}",)).fetchall()
        db.close()
    ts = targets(cases, work)
    for i, (t, m) in enumerate(zip(ts, meta["edits"])):
        if len(rows) == len(meta["edits"]):
            m["cost_usd"], m["ledger_status"] = rows[i][1], rows[i][2]
        rp = d / f"{t['tid']}.raw.png"
        if rp.exists():
            crop, mask = ME._crop_and_mask(t)
            _save(composite_inside_mask(crop, mask, np.asarray(Image.open(rp).convert("RGBA"))), d / f"{t['tid']}.png")
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    om = json.loads((old / "meta.json").read_text()); om["edits"] = meta["edits"]; om["note"] = "alpha dropped (pre-fix)"
    (old / "meta.json").write_text(json.dumps(om, indent=1))
    return meta


def table(res):
    cols = ["returned", "pass", "plausible", "stale", "post_edit_violations", "exact", "exact_on_readable_controls", "fuzzy", "residual", "ghost", "page_leak",
            "out_of_mask_changed_targets", "out_of_mask_frac_mean", "raw_out_of_mask_frac_mean", "seam_mean",
            "ink_delta_median", "height_ratio_median", "usd_per_edit", "p50_latency_s"]
    lines = ["| Editor | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for e, v in res["editors"].items():
        a = v["aggregate"]
        lines.append(f"| {e} | " + " | ".join("-" if a.get(c) is None else str(a.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    from ..docs.byok import register_eval_keys
    register_eval_keys()  # eval keys are for public samples only: the product refuses them on user documents
    ap = argparse.ArgumentParser(prog="python -m casepatch.hosted.editbench")
    ap.add_argument("action", choices=["targets", "local", "hosted", "recomposite", "score", "dry-run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-image-2.6-sunburst")
    ap.add_argument("--models", nargs="+", default=None, help="dry-run: models to price")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--tids", nargs="+", default=None, help="hosted: only these target ids (e.g. a 3-edit pilot)")
    a = ap.parse_args(argv)
    if a.action == "targets":
        for t in targets(a.cases):
            print(t["tid"], repr(t["value"]), "->", repr(t["replacement"]), t["crop"])
    elif a.action == "dry-run":
        print(json.dumps(dry_run(a.models or [a.model], a.cases), indent=1))
    elif a.action == "local":
        print(json.dumps(run_local(a.out, a.cases), indent=1))
    elif a.action == "hosted":
        from .config import HostedDisabled
        try:
            meta, stopped = run_hosted(a.out, a.model, a.cases, tids=a.tids)
        except HostedDisabled as e:
            print("refused:", ME.redact(e)); return 2
        print(json.dumps({"stopped": stopped, "edits": len(meta), "errors": [m["error"] for m in meta if m["error"]]}))
    elif a.action == "recomposite":
        print(json.dumps(recomposite(a.out, a.model, a.cases)["edits"], indent=1)[:2000])
    else:
        res = score(a.out, a.cases, tids=a.tids)
        print(table(res)); print("controls readable:", res["controls_readable"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
