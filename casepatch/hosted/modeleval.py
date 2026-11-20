"""Multi-model eval harness: Kimi (Moonshot), OpenAI and Gemini vs CasePatch's local baselines, on the same pages.

    python -m casepatch.hosted.modeleval prepare   [--cases ...]          # local: render + OCR, cached ($0, no network)
    python -m casepatch.hosted.modeleval dry-run   [--cases ...] [--json] # exact request counts + worst-case $ ($0)
    python -m casepatch.hosted.modeleval baselines [--cases ...]          # score local rules / GLiNER / Qwen runs ($0)
    python -m casepatch.hosted.modeleval run --providers kimi openai gemini [--tasks detect edit]   # PAID, capped

Tasks (identical pages and labels for every system):
  detect  one request per page: page image + its OCR lines -> grounded sensitive spans (cite an OCR line), values the
          OCR missed (box-grounded, e.g. handwriting), and face / signature / handwriting region boxes.
  edit    region replacement on a crop: crop + mask + 'replace X with Y'; scored by OCR readback of Y, residual X, and
          out-of-mask change of the raw model output (CasePatch composites strictly inside the mask). Kimi: N/A.

Spend: every call goes through HostedClient -> the ONE shared SQLite ledger (CASEPATCH_SPEND_LEDGER), which enforces
the total CASEPATCH_SPEND_CAP_USD across all providers AND a per-provider sub-budget (CASEPATCH_PROVIDER_CAP_<P>_USD,
required for eval runs). Before a provider starts, the planned worst case must fit its remaining sub-budget and the
remaining total, else the run is refused. Nothing here touches the network unless `run` is called with
CASEPATCH_HOSTED=1, a cap, sub-caps and keys in the process environment.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import pricing as P
from .config import (GRAND_TOTAL_MAX_USD, KEY_ENV, POOL_CAP_ENV, POOL_MAX_USD, PROVIDER_CAP_ENV, HostedDisabled, parse_pool_cap,
                     pool_of)
from .config import HostedDisabled, load_config, parse_cap, parse_provider_cap, redact

ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = ROOT / "examples" / "real"
DEFAULT_CASES = ["case-real-01", "case-real-02", "case-real-03", "case-real-04"]
WORK = Path(os.environ.get("CASEPATCH_EVAL_WORK", Path.home() / ".cache" / "casepatch" / "modeleval"))
RESULTS = ROOT / "bench" / "models" / "results"
SUPPLEMENTARY = ROOT / "bench" / "models" / "supplementary_labels.json"

# Models under evaluation (ids verified on the official pages on 2026-11-11; see pricing.PRICING_DATE / SOURCES)
EVAL_MODELS = {  # quality ceiling FIRST per provider, cheaper tiers after (ids verified by list-models 2026-11-11)
    "openai": {"detect": ["gpt-6-astra", "gpt-6.1-terra", "gpt-6.1-sol"],
               "edit": ["gpt-image-2.6-sunburst", "gpt-image-2"]},
    "gemini": {"detect": ["gemini-4-carbon", "gemini-4-argon", "gemini-3.8-flash", "gemini-3.5-flash-lite"],
               "edit": ["gemini-3-pro-image", "gemini-nano-banana-2.1", "gemini-3.1-flash-image"]},
    "kimi": {"detect": ["kimi-k3.1-pro", "kimi-k3"], "edit": []},  # Moonshot has no image-edit model: N/A
    "anthropic": {"detect": ["claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
                  "edit": []},  # Claude has no image-edit model: N/A
    # fal.ai ('other' pool): endpoint ids live in the public catalog 2026-11-12. No Qwen-VL endpoint is hosted on fal
    # (catalog search 'qwen vl' / 'qwen3.5' = 0 hits); NVIDIA Nemotron vision endpoints are.
    "fal": {"detect": ["nvidia/nemotron-3-5-super/vision", "nvidia/nemotron-3-nano-omni/vision"],
            "edit": ["fal-ai/qwen-image-edit-2511", "fal-ai/qwen-image-edit/inpaint", "fal-ai/flux-kontext/dev",
                     "fal-ai/flux-pro/kontext", "fal-ai/flux-2/klein/4b/edit", "alibaba/qwen-image-3/edit"]},
}
CEILING = {"openai": {"detect": ["gpt-6-astra", "gpt-6.1-terra"], "edit": ["gpt-image-2.6-sunburst"]},
           "gemini": {"detect": ["gemini-4-carbon", "gemini-4-argon"], "edit": ["gemini-3-pro-image", "gemini-nano-banana-2.1"]},
           "kimi": {"detect": ["kimi-k3.1-pro"], "edit": []},
           "anthropic": {"detect": ["claude-fable-5-2", "claude-opus-5-5"], "edit": []},
           "fal": {"detect": ["nvidia/nemotron-3-5-super/vision"], "edit": ["fal-ai/qwen-image-edit-2511", "fal-ai/flux-pro/kontext"]}}
TOTAL_CAP_USD = GRAND_TOTAL_MAX_USD  # grand total maximum: $300 across all providers
POOL_CAPS = dict(POOL_MAX_USD)       # hard, separate: anthropic $200, openai $50, other (kimi+gemini+any) $50 combined
EVAL_MAX_OUTPUT_TOKENS = 4096   # thinking models bill reasoning as output; 2048 truncates JSON on dense pages
EVAL_MAX_IMAGE_SIDE = 1600      # small print on letter pages is unreadable at 1024
EVAL_EDIT_TARGETS_PER_CASE = 4
EXPECTED_OUTPUT_TOKENS = 1500   # typical (not worst) output incl. low-effort reasoning, for the 'expected' column

KINDS = ["PERSON", "ORG", "ADDRESS", "ZIP", "PHONE", "EMAIL", "CARD", "CARD_LAST4", "ACCOUNT", "ID", "DOB", "BIRTHPLACE"]
EVAL_PROMPT = (
    "You are auditing one page of a document case so personal data can be replaced with fictional values. You get the "
    "page image and its OCR lines (Lnn: text). Return JSON only, exactly this shape:\n"
    "{\"items\":[{\"line\":\"L3\",\"text\":\"...\",\"kind\":\"PERSON\",\"person_id\":\"P1\",\"handwritten\":false}],\n"
    " \"unread\":[{\"text\":\"...\",\"kind\":\"DOB\",\"box\":[x0,y0,x1,y1],\"handwritten\":true}],\n"
    " \"regions\":[{\"kind\":\"face|signature|handwriting\",\"box\":[x0,y0,x1,y1]}]}\n"
    "Rules: items[].text must be copied EXACTLY from the cited OCR line (a substring). Use unread[] only for personal "
    "values you can see in the image that no OCR line contains (e.g. handwriting); read them as written. Boxes are "
    "integers 0-1000 relative to image width/height. kind is one of " + "|".join(KINDS) + ". Personal data = names of "
    "private people (any form), private addresses, phone numbers, emails, card/account/ID/licence/serial numbers, dates "
    "of birth, places of birth. Do NOT report amounts, ordinary dates, totals, merchants' or government offices' names "
    "unless they identify a private person. Same person_id = same human on this page. Report every face photo "
    "(including small ghost portraits) and every signature as a region.")

NON_LATIN = re.compile(r"[\u0370-\u03ff\u0400-\u04ff\u0590-\u06ff\u4e00-\u9fff]")
ID_KINDS = {"ID", "CARD", "CARD_LAST4", "ACCOUNT", "PHONE", "ZIP", "EMAIL"}


# ------------------------------------------------------------------------------------------------ page preparation
def _line_box(line, to_px):
    pts = [(x * to_px, y * to_px) for c in line.chars for (x, y) in c.quad]
    if not pts:
        return None
    xs, ys = zip(*pts)
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def _char_boxes(line, to_px):
    out = []
    for c in line.chars:
        xs, ys = zip(*c.quad)
        out.append([min(xs) * to_px, min(ys) * to_px, max(xs) * to_px, max(ys) * to_px])
    return out


def prepare_case(case, work=WORK, force=False):
    """Render + OCR every PDF/image page once (local, $0) and cache PNG + lines with pixel boxes. JSON traces are text
    only and are not part of the page-image eval."""
    cdir = CASES_DIR / case
    out = Path(work) / "pages" / case
    idx = out / "index.json"
    if idx.exists() and not force:
        return json.loads(idx.read_text())
    from PIL import Image
    from ..docs.extract import extract_dir
    out.mkdir(parents=True, exist_ok=True)
    docs = extract_dir(cdir)
    files = sorted(f.name for f in cdir.iterdir() if f.is_file() and f.name != "labels.json"
                   and f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".json"})
    pages = []
    for d, fname in zip(docs, files):
        if d.kind == "json":
            continue
        for p in d.pages:
            to_px = 1.0 if p.raster else p.scale
            png = out / f"{Path(fname).stem}_p{p.index}.png"
            Image.fromarray(p.image).save(png)
            lines = []
            for i, l in enumerate(x for x in p.lines if x.chars and x.text.strip()):
                lines.append({"id": f"L{i}", "text": l.text, "source": l.source, "box": _line_box(l, to_px),
                              "chars": [[round(v, 1) for v in b] for b in _char_boxes(l, to_px)]})
            pages.append({"case": case, "file": fname, "doc": d.id, "page": p.index, "png": str(png),
                          "w": int(p.image.shape[1]), "h": int(p.image.shape[0]), "raster": bool(p.raster),
                          "scale": float(p.scale), "to_px": to_px, "lines": lines})
    idx.write_text(json.dumps(pages, ensure_ascii=False))
    return pages


def available_cases(cases=None):
    """Cases with a labels.json (a case without labels cannot be scored and is reported as pending)."""
    ok, pending = [], []
    for c in cases or DEFAULT_CASES:
        if not re.fullmatch(r"case-[a-z0-9-]+", str(c)):  # a folder name inside examples/real (bundled public samples) only
            raise ValueError(f"not a bundled public sample case: {c!r}")
        (ok if (CASES_DIR / c / "labels.json").exists() else pending).append(c)
    return ok, pending


def load_labels(case):
    return json.loads((CASES_DIR / case / "labels.json").read_text())


GEMINI_BOX_NOTE = ("\nBox format for you: [ymin, xmin, ymax, xmax] integers 0-1000 (your native box_2d order); this "
                   "overrides the [x0,y0,x1,y1] order above.")


def page_prompt(page, provider=None):
    return EVAL_PROMPT + (GEMINI_BOX_NOTE if provider == "gemini" else "") + "\nOCR lines:\n" + "\n".join(f"{l['id']}: {l['text']}" for l in page["lines"])


def fit_dims(w, h, side):
    s = min(1.0, side / max(w, h))
    return max(1, int(w * s)), max(1, int(h * s)), s


def text_token_estimate(text):
    # ~3 chars/token for Latin text; non-Latin scripts tokenise worse -> count them double. pricing adds x1.5.
    nl = len(NON_LATIN.findall(text))
    return (len(text) + nl) // 3 + 50


# ------------------------------------------------------------------------------------------------ edit targets
def _fake_like(value, seed):
    h = hashlib.sha256((seed + value).encode()).digest()
    out, k = [], 0
    for ch in value:
        b = h[k % len(h)]; k += 1
        if ch.isdigit():
            out.append(str((int(ch) + 1 + b % 8) % 10))
        elif ch.isalpha() and ch.isascii():
            letter = "ABCDEFGHJKLMNPRSTUVWZ"[b % 21]
            out.append(letter if ch.isupper() else letter.lower())
        else:
            out.append(ch)
    s = "".join(out)
    return s if s != value else s[::-1]


# Per-sample overrides for the public demo/bench samples only (not planner logic). Each pin must itself pass the
# plausibility checker, otherwise the planned value is used. The planner picked 1891 for the
# Cole Porter draft card (age 25 allows only 1891 or 1893), but 1891 is his real birth year, which reads like a leak.
DEMO_PINS = {("case-real-04", "01_handwritten_draft_card.png", "1892"): "1893"}


def pinned(case, file, value, planned_value, kind=None, ctx=None):
    """The demo pin for this sample value if one exists and is plausible in context, else the planned value."""
    pin = DEMO_PINS.get((case, file, value))
    if not pin or pin == planned_value:
        return planned_value
    from ..docs import consistency as CC
    return planned_value if CC.plausibility(kind or "DOB", value, pin, ctx) else pin


def edit_targets(case, pages, labels, n=EVAL_EDIT_TARGETS_PER_CASE):
    """Deterministic list of (page, value, crop box, mask char span, replacement) on raster pages, first n by order.
    The replacement is planned in context (docs/consistency.py: typed, same era, consistent across the case), so a
    1892 birth year on a 1917 draft card becomes 1891/1893, not '6217'. `legacy_replacement` is the old
    character-shuffle value that the 2026-11-12 hosted outputs were made with (used to mark them stale)."""
    from ..docs import consistency as CC
    planned, graph, _ = CC.plan_labelled_case(case, pages, labels)
    out = []
    for p in pages:
        if not p["raster"]:
            continue
        for v in labels["occurrences"].get(p["file"], []):
            for l in p["lines"]:
                i = l["text"].find(v)
                if i < 0 or len(l["chars"]) != len(l["text"]) or len(v) < 3:
                    continue
                cb = l["chars"][i:i + len(v)]
                mx0, my0 = min(b[0] for b in cb), min(b[1] for b in cb)
                mx1, my1 = max(b[2] for b in cb), max(b[3] for b in cb)
                hgt = max(8.0, my1 - my0)
                kind = next((o.kind for o in graph.occ.values() if o.doc == p["file"] and o.text == v), None)
                x0, y0 = max(0, int(mx0 - 2 * hgt)), max(0, int(my0 - 1.5 * hgt))
                x1, y1 = min(p["w"], int(mx1 + 2 * hgt)), min(p["h"], int(my1 + 1.5 * hgt))
                planned_v = pinned(case, p["file"], v, planned.get((p["file"], v)), kind, graph.contexts.get(p["file"]))
                out.append({"case": case, "file": p["file"], "page": p["page"], "png": p["png"], "value": v,
                            "replacement": planned_v or _fake_like(v, case + p["file"]),
                            "legacy_replacement": _fake_like(v, case + p["file"]), "kind": kind,
                            "crop": [x0, y0, x1, y1], "mask": [int(mx0) - x0, int(my0) - y0, int(math.ceil(mx1)) - x0, int(math.ceil(my1)) - y0]})
                break
            if len(out) >= n:
                return out
    return out


def edit_instruction(t):
    return (f"Photo-edit this document crop. Inside the masked area only, replace the text \"{t['value']}\" with "
            f"\"{t['replacement']}\" in the same font, size, weight, ink colour, paper texture and noise, so it looks "
            "original. Do not change anything outside the mask.")


# ------------------------------------------------------------------------------------------------ dry run
def plan(cases=None, providers=("openai", "gemini", "kimi", "anthropic"), tasks=("detect", "edit"), models=None,
         max_output_tokens=EVAL_MAX_OUTPUT_TOKENS, max_image_side=EVAL_MAX_IMAGE_SIDE, max_retries=1,
         edit_n=EVAL_EDIT_TARGETS_PER_CASE, work=WORK):
    """Exact request counts and token/image estimates, worst case (every attempt incl. retries, full max output, x1.5
    on input estimates) and an expected case (no retries, typical output). $0: no network, prepares pages locally."""
    ok, pending = available_cases(cases)
    rows, per_provider = [], {}
    pages_by_case = {c: prepare_case(c, work) for c in ok}
    for prov in providers:
        pm = (models or {}).get(prov) or EVAL_MODELS[prov]
        for task in tasks:
            for model in pm.get(task, []):
                r = {"provider": prov, "model": model, "task": task, "requests": 0, "attempts_worst": 0,
                     "text_tokens_est": 0, "image_inputs": 0, "image_input_tokens_est": 0, "output_images": 0,
                     "worst_usd": 0.0, "expected_usd": 0.0, "per_case": {}}
                for c in ok:
                    pages = pages_by_case[c]
                    if task == "detect":
                        calls = []
                        for p in pages:
                            w, h, _ = fit_dims(p["w"], p["h"], max_image_side)
                            tt = text_token_estimate(page_prompt(p, prov))
                            calls.append((tt, [(w, h)], max_output_tokens, 0, 1024))
                    else:
                        calls = []
                        for t in edit_targets(c, pages, load_labels(c), edit_n):
                            x0, y0, x1, y1 = t["crop"]; w, h, _ = fit_dims(x1 - x0, y1 - y0, max_image_side)
                            calls.append((120, [(w, h)] * (1 if prov == "fal" else 2), 0 if prov in ("openai", "fal") else 256, 1, max(w, h)))
                    cw = ce = 0.0
                    for tt, imgs, mo, oi, side in calls:
                        wc = P.worst_case_usd(prov, model, tt, imgs, mo, output_images=oi, out_side=side)
                        cw += wc * (max_retries + 1)
                        if prov == "fal":  # output-unit billing: expected = unit price without the safety factor
                            ce += P.fal_usd(model, tt, imgs, min(mo, EXPECTED_OUTPUT_TOKENS) if mo else 0, oi, safety=1.0)
                            r["text_tokens_est"] += tt; r["image_inputs"] += len(imgs); r["output_images"] += oi
                            continue
                        p_ = P.price(prov, model)
                        img_tok = sum(P.image_input_tokens(prov, a, b, model) for a, b in imgs) / P.SAFETY
                        exp_out = min(mo, EXPECTED_OUTPUT_TOKENS) if mo else 0
                        e = (tt * p_["text_in"] + img_tok * p_["image_in"] + exp_out * p_["out"]) / 1e6
                        if oi:
                            e += oi * P.image_output_tokens(prov, side) / (P.SAFETY if prov == "openai" else 1) * p_.get("image_out", p_["out"]) / 1e6
                        ce += e
                        r["text_tokens_est"] += tt; r["image_inputs"] += len(imgs); r["output_images"] += oi
                        r["image_input_tokens_est"] += int(img_tok)
                    r["requests"] += len(calls); r["attempts_worst"] += len(calls) * (max_retries + 1)
                    r["per_case"][c] = {"requests": len(calls), "worst_usd": round(cw, 4), "expected_usd": round(ce, 4)}
                    r["worst_usd"] += cw; r["expected_usd"] += ce
                r["worst_usd"] = round(r["worst_usd"], 4); r["expected_usd"] = round(r["expected_usd"], 4)
                rows.append(r)
                pp = per_provider.setdefault(prov, {"worst_usd": 0.0, "expected_usd": 0.0, "requests": 0})
                pp["worst_usd"] = round(pp["worst_usd"] + r["worst_usd"], 4)
                pp["expected_usd"] = round(pp["expected_usd"] + r["expected_usd"], 4)
                pp["requests"] += r["requests"]
    total_w = round(sum(v["worst_usd"] for v in per_provider.values()), 4)
    total_e = round(sum(v["expected_usd"] for v in per_provider.values()), 4)
    return {"pricing_date": P.PRICING_DATE, "sources": P.SOURCES, "cases": ok, "cases_pending_labels": pending,
            "pages": {c: len(v) for c, v in pages_by_case.items()},
            "settings": {"max_output_tokens": max_output_tokens, "max_image_side": max_image_side, "max_retries": max_retries,
                         "edit_targets_per_case": edit_n, "safety_multiplier_on_input_estimates": P.SAFETY},
            "rows": rows, "per_provider": per_provider, "total_worst_usd": total_w, "total_expected_usd": total_e,
            "per_pool": per_pool(per_provider), "caps": {"total": TOTAL_CAP_USD, "pools": POOL_CAPS}}


def per_pool(per_provider):
    out = {}
    for prov, v in per_provider.items():
        q = out.setdefault(pool_of(prov), {"worst_usd": 0.0, "expected_usd": 0.0, "cap_usd": POOL_CAPS[pool_of(prov)]})
        q["worst_usd"] = round(q["worst_usd"] + v["worst_usd"], 4); q["expected_usd"] = round(q["expected_usd"] + v["expected_usd"], 4)
    return out


def plan_table(pl):
    lines = [f"Dry run (pricing read {pl['pricing_date']}; cases {', '.join(pl['cases'])}; pending labels: "
             f"{', '.join(pl['cases_pending_labels']) or 'none'}; pages {pl['pages']})",
             f"{'provider':8} {'model':24} {'task':6} {'req':>4} {'attempts':>8} {'worst $':>9} {'expected $':>10}"]
    for r in pl["rows"]:
        lines.append(f"{r['provider']:8} {r['model']:24} {r['task']:6} {r['requests']:>4} {r['attempts_worst']:>8} "
                     f"{r['worst_usd']:>9.4f} {r['expected_usd']:>10.4f}")
    for p, v in pl["per_provider"].items():
        lines.append(f"{p:8} TOTAL{'':19} {'':6} {v['requests']:>4} {'':>8} {v['worst_usd']:>9.4f} {v['expected_usd']:>10.4f}")
    for q, v in per_pool(pl["per_provider"]).items():
        lines.append(f"pool {q:9} worst ${v['worst_usd']:.2f}  expected ${v['expected_usd']:.2f}  cap ${v['cap_usd']:.0f}"
                     + ("  (worst exceeds cap: run needs --allow-partial or fewer models)" if v["worst_usd"] > v["cap_usd"] else ""))
    lines.append(f"ALL providers worst ${pl['total_worst_usd']:.2f}  expected ${pl['total_expected_usd']:.2f}  "
                 f"(grand total cap ${TOTAL_CAP_USD:.0f})")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ hosted run (PAID)
def _salvage(text):
    """Malformed JSON (seen on cheap tiers: a missing bracket) -> recover each complete flat object; classify by keys."""
    d = {"items": [], "unread": [], "regions": [], "_salvaged": True}
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        frag = m.group(0)
        frag = re.sub(r"(\[[^\[\]]*?)\}$", r"\1]}", frag)  # '[1, 2, 3}' -> '[1, 2, 3]}'
        try:
            o = json.loads(frag)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        if "line" in o and "text" in o:
            d["items"].append(o)
        elif "text" in o and "box" in o:
            d["unread"].append(o)
        elif "box" in o and "kind" in o:
            d["regions"].append(o)
    return d


def _parse_detect(text):
    try:
        d = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except Exception:
            d = None
        if d is None:
            d = _salvage(text)
    return d if isinstance(d, dict) else {}


def ground_page_output(page, d, yx=False):
    """Keep only items whose text occurs in the cited OCR line (exact, then case/space-insensitive); unread items need
    a valid box; boxes 0-1000 -> page pixels."""
    lines = {l["id"]: l["text"] for l in page["lines"]}
    dets, rejected = [], 0
    for it in d.get("items", []) if isinstance(d.get("items"), list) else []:
        if not isinstance(it, dict):
            continue
        t, src = str(it.get("text", "")), lines.get(str(it.get("line", "")))
        if not t.strip() or src is None:
            rejected += 1; continue
        if t not in src:
            m = re.search(r"\s+".join(re.escape(x) for x in t.split()), src, re.I)
            if not m:
                rejected += 1; continue
            t = m.group(0)
        dets.append({"file": page["file"], "page": page["page"], "text": t, "kind": str(it.get("kind", "")),
                     "person_id": it.get("person_id"), "grounding": "ocr", "handwritten": bool(it.get("handwritten"))})

    def box(b):
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            return None
        x0, y0, x1, y1 = (max(0.0, min(1000.0, float(v))) for v in b)
        if yx:  # Gemini native order [ymin, xmin, ymax, xmax]
            y0, x0, y1, x1 = x0, y0, x1, y1
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0 * page["w"] / 1000, y0 * page["h"] / 1000, x1 * page["w"] / 1000, y1 * page["h"] / 1000]
    for it in d.get("unread", []) if isinstance(d.get("unread"), list) else []:
        if isinstance(it, dict) and str(it.get("text", "")).strip() and box(it.get("box")):
            dets.append({"file": page["file"], "page": page["page"], "text": str(it["text"]), "kind": str(it.get("kind", "")),
                         "person_id": it.get("person_id"), "grounding": "box", "box": box(it["box"]),
                         "handwritten": bool(it.get("handwritten"))})
        else:
            rejected += 1
    regs = []
    for r in d.get("regions", []) if isinstance(d.get("regions"), list) else []:
        if isinstance(r, dict) and box(r.get("box")):
            regs.append({"file": page["file"], "page": page["page"], "kind": str(r.get("kind", "")).lower(), "bbox": box(r["box"])})
    return dets, regs, rejected


def eval_env_check(env, providers):
    """Eval gate on top of load_config: total <= approved grand total; each provider's budget = min(optional provider
    sub-budget, its hard pool cap). Returns (total, {provider: (sub_or_None, pool, pool_cap)})."""
    total = parse_cap(env.get("CASEPATCH_SPEND_CAP_USD"))
    if total > TOTAL_CAP_USD + 1e-9:
        raise HostedDisabled(f"eval total cap ${total:.2f} exceeds the approved ${TOTAL_CAP_USD:.0f}")
    subs = {}
    for p in providers:
        subs[p] = (parse_provider_cap(env, p, total), pool_of(p), parse_pool_cap(env, pool_of(p), total))
    return total, subs


_FATAL = ("credit balance", "billing", "insufficient", "payment", "balance is too low", "quota exceeded for this account",
          "http 401", "http 402", "http 403", "authentication_error", "permission_error", "suspended")


def is_fatal_billing_or_auth(msg):
    """Billing / zero-balance / auth rejections are never retried and stop the provider for the rest of the run."""
    m = msg.lower()
    if "http 429" in m or "gave up after" in m:  # rate limits (even ones mentioning billing) use the backoff path
        return "credit balance" in m or "balance is too low" in m or "insufficient_quota" in m or "billing_hard_limit" in m
    return any(k in m for k in _FATAL)


def run_hosted(providers, cases=None, tasks=("detect",), models=None, env=None, transport_factory=None, work=WORK,
               max_output_tokens=EVAL_MAX_OUTPUT_TOKENS, max_image_side=EVAL_MAX_IMAGE_SIDE, edit_n=EVAL_EDIT_TARGETS_PER_CASE,
               allow_partial=False, ocr_fn=None, out_dir=None, min_interval_s=0.0, rate_limit_backoff_s=60.0,
               sleep=time.sleep, timeout_s=180.0, gemini_thinking=None):
    """PAID when a real transport is used. Refuses unless every gate passes. Returns {system_name: result}."""
    from .client import HostedClient, ImageEditor, _fit, _png_b64, build_vision_request, response_text
    from .ledger import CapExceeded, SpendLedger
    env = os.environ if env is None else env
    total, subs = eval_env_check(env, providers)
    cfgs = {p: load_config(env, p) for p in providers}  # raises HostedDisabled: flag / cap / key missing
    ok, pending = available_cases(cases)
    pl = plan(ok, providers, tasks, models, max_output_tokens, max_image_side, max(c.max_retries for c in cfgs.values()), edit_n, work)
    ledger = SpendLedger(next(iter(cfgs.values())).ledger_path)  # ONE shared ledger for all providers
    t_all = ledger.totals()
    results = {}
    last_call = [0.0]
    for prov in providers:
        cfg = dataclasses.replace(cfgs[prov], max_output_tokens=max_output_tokens, max_image_side=max_image_side,
                                  max_calls_per_case=500, timeout_s=timeout_s)
        pt = ledger.provider_totals(prov)
        sub_cap, pool, pool_cap = subs[prov]
        t_all = ledger.totals()
        qt = ledger.pool_totals(pool)
        remaining = min(pool_cap - qt["spent_usd"] - qt["reserved_usd"], total - t_all["spent_usd"] - t_all["reserved_usd"])
        if sub_cap is not None:
            remaining = min(remaining, sub_cap - pt["spent_usd"] - pt["reserved_usd"])
        need = pl["per_provider"].get(prov, {}).get("worst_usd", 0.0)
        if need > remaining and not allow_partial:
            raise HostedDisabled(f"{prov}: planned worst case ${need:.2f} exceeds remaining budget ${remaining:.2f} "
                                 "(lower --models / --edit-targets, raise the sub-budget, or pass --allow-partial)")
        transport = transport_factory(cfg) if transport_factory else None
        client = HostedClient(cfg, transport=transport, ledger=ledger)
        pm = (models or {}).get(prov) or EVAL_MODELS[prov]
        provider_fatal = None
        for model in pm.get("detect", []) if "detect" in tasks else []:
            name = f"{prov}:{model}"
            if provider_fatal:  # billing / auth failure: never touch this provider again in this run
                results[name] = {"system": name, "provider": prov, "model": model, "task": "detect", "calls": [], "detections": [],
                                 "regions": [], "errors": [], "stopped": f"skipped: {provider_fatal}", "pages_work": str(work)}
                continue
            res = {"system": name, "provider": prov, "model": model, "task": "detect", "box_order": "yx" if prov == "gemini" else "xy", "pages_work": str(work),
                   "calls": [], "detections": [],
                   "regions": [], "rejected_ungrounded": 0, "errors": [], "stopped": None}
            consecutive_err = 0
            for c in ok:
                for page in prepare_case(c, work):
                    from PIL import Image
                    img, _ = _fit(np.asarray(Image.open(page["png"]).convert("RGB")), cfg.max_image_side)
                    b64, prompt = _png_b64(img), page_prompt(page, prov)
                    est = P.worst_case_usd(prov, model, text_token_estimate(prompt), [img.shape[1::-1]], cfg.max_output_tokens)
                    data, backed_off = None, False
                    while True:  # pacing + one backoff on 429; a second 429 = quota/rate exhausted -> stop
                        if min_interval_s and last_call[0]:
                            wait = min_interval_s - (time.monotonic() - last_call[0])
                            if wait > 0:
                                sleep(wait)
                        last_call[0] = time.monotonic()
                        t0 = time.perf_counter()
                        try:
                            data = client.call("vision", model, lambda: build_vision_request(prov, model, prompt, b64, cfg.max_output_tokens, gemini_thinking),
                                               est, f"eval:{c}:{prov}:{model}")
                            consecutive_err = 0
                            break
                        except CapExceeded as e:
                            res["stopped"] = f"cap: {e}"; break
                        except Exception as e:
                            msg = redact(e, [cfg.key()])[:300]
                            if is_fatal_billing_or_auth(msg):  # stop at once: no backoff, no retry, no next page/model
                                res["errors"].append({"case": c, "file": page["file"], "page": page["page"], "error": msg})
                                res["stopped"] = provider_fatal = "billing/auth error (no retries): " + msg[:160]
                                break
                            transient = ("http 429" in msg or re.search(r"http 5\d\d", msg) is not None
                                         or "timeout" in msg.lower() or "timed out" in msg.lower())
                            if transient and not backed_off:
                                backed_off = True; res["backoffs"] = res.get("backoffs", 0) + 1
                                sleep(rate_limit_backoff_s); continue
                            res["errors"].append({"case": c, "file": page["file"], "page": page["page"], "error": msg})
                            consecutive_err += 1
                            if "http 429" in msg:
                                res["stopped"] = "rate limit / quota exhausted (429 after backoff)"
                            elif transient and consecutive_err >= 2:
                                res["stopped"] = "provider unavailable (5xx/timeout after backoff on two pages)"
                            elif consecutive_err >= 2:
                                res["stopped"] = "two consecutive failed calls: stopped to avoid hammering"
                            break
                    if res["stopped"]:
                        break
                    if data is None:
                        continue
                    lat = time.perf_counter() - t0
                    usage = data.get("usageMetadata") if prov == "gemini" else data.get("usage")
                    try:
                        cost = P.actual_usd(prov, model, data if prov == "fal" else usage)
                    except Exception:
                        cost = None
                    try:
                        raw_text = response_text(data, prov)
                    except Exception:
                        raw_text = ""
                    d = _parse_detect(raw_text)
                    dets, regs, rej = ground_page_output(page, d, yx=(prov == "gemini"))
                    res["malformed_json_pages"] = res.get("malformed_json_pages", 0) + bool(d.get("_salvaged"))
                    res.setdefault("raw", []).append({"case": c, "file": page["file"], "page": page["page"], "text": raw_text[:20000]})
                    for x in dets + regs:
                        x["case"] = c
                    res["detections"] += dets; res["regions"] += regs; res["rejected_ungrounded"] += rej
                    res["calls"].append({"case": c, "file": page["file"], "page": page["page"], "latency_s": round(lat, 2),
                                         "cost_usd": cost, "reserved_usd": est, "usage": usage})
                if res["stopped"]:
                    break
            results[name] = res
        for model in pm.get("edit", []) if "edit" in tasks else []:
            name = f"{prov}:{model}"
            res = {"system": name, "provider": prov, "model": model, "task": "edit", "edits": [], "errors": [], "stopped": None}
            if provider_fatal:
                res["stopped"] = f"skipped: {provider_fatal}"; results[name] = res; continue
            editor = ImageEditor(client, model=model)
            for c in ok:
                for t in edit_targets(c, prepare_case(c, work), load_labels(c), edit_n):
                    crop, mask = _crop_and_mask(t)
                    captured = {}
                    t0 = time.perf_counter()
                    try:
                        raw = _edit_raw(editor, crop, mask, edit_instruction(t), f"eval:{c}:{prov}:{model}", captured)
                    except CapExceeded as e:
                        res["stopped"] = f"cap: {e}"; break
                    except Exception as e:
                        msg = redact(e, [cfg.key()])[:300]
                        res["errors"].append({"case": c, "file": t["file"], "error": msg})
                        if is_fatal_billing_or_auth(msg):
                            res["stopped"] = provider_fatal = "billing/auth error (no retries): " + msg[:160]; break
                        if len(res["errors"]) >= 2 and not res["edits"]:
                            res["stopped"] = "two failed edit calls and no success: stopped to avoid hammering"; break
                        continue
                    res["edits"].append(score_edit(t, crop, mask, raw, ocr_fn) | {"latency_s": round(time.perf_counter() - t0, 2),
                                                                                    "cost_usd": captured.get("cost")})
                if res["stopped"]:
                    break
            results[name] = res
    ledger.close()
    out_dir = Path(out_dir or (Path(work) / "hosted"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        (out_dir / (name.replace(":", "__") + ".json")).write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    return results


def probe_access(providers, model_ids, env=None, transport_factory=None):
    """ONE minimal text-only request per model (max 16 output tokens, no retries) through the capped client, to learn
    whether this key/tier can use the model at all. A rejected request (e.g. free tier 'limit: 0') is not billed;
    the ledger still reserves the tiny worst case and keeps it on any non-200."""
    from .client import HostedClient
    from .ledger import SpendLedger
    env = dict(os.environ if env is None else env)
    env["CASEPATCH_MAX_RETRIES"] = "0"
    eval_env_check(env, providers)
    out = []
    for prov in providers:
        cfg = load_config(env, prov)
        cfg = dataclasses.replace(cfg, max_output_tokens=16)
        client = HostedClient(cfg, transport=transport_factory(cfg) if transport_factory else None, ledger=SpendLedger(cfg.ledger_path))
        for model in [m for t in ("detect", "edit") for m in EVAL_MODELS[prov][t] if m in model_ids]:
            if prov == "fal":  # every fal call is a billed output: no free probe exists
                out.append({"provider": prov, "model": model, "status": "skipped (fal has no unbilled probe)"}); continue
            if prov == "openai" and model.startswith("gpt-image"):
                out.append({"provider": prov, "model": model, "status": "skipped (image endpoint has no text-only probe)"}); continue
            if prov == "gemini":
                body = {"contents": [{"role": "user", "parts": [{"text": "Reply with OK."}]}],
                        "generationConfig": {"maxOutputTokens": 16}}
                url = ENDPOINT_GEMINI.format(model=model)
            elif prov == "anthropic":
                body = {"model": model, "max_tokens": 16, "output_config": {"effort": "low"},
                        "messages": [{"role": "user", "content": "Reply with OK."}]}
                url = "https://api.anthropic.com/v1/messages"
            elif prov == "kimi":
                body = {"model": model, "max_completion_tokens": 16, "reasoning_effort": "low",
                        "messages": [{"role": "user", "content": "Reply with OK."}]}
                url = "https://api.moonshot.ai/v1/chat/completions"
            else:
                body = {"model": model, "max_output_tokens": 16, "store": False, "input": "Reply with OK."}
                url = "https://api.openai.com/v1/responses"
            est = P.worst_case_usd(prov, model, 20, [], 16)
            try:
                client.call("probe", model, lambda: (url, json.dumps(body).encode(), "application/json"), est, f"probe:{prov}:{model}")
                out.append({"provider": prov, "model": model, "status": "ok"})
            except Exception as e:
                out.append({"provider": prov, "model": model, "status": redact(e, [cfg.key()])[:300]})
        client.ledger.close()
    return out


def ledger_summary(path):
    import sqlite3
    if not Path(path).exists():
        return {}
    db = sqlite3.connect(str(path))
    out = {"by_provider": {}, "by_model": []}
    for prov, n, spent, reserved in db.execute(
            "SELECT provider, COUNT(*), COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0),"
            " COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0) FROM calls GROUP BY provider"):
        out["by_provider"][prov] = {"attempts": n, "ledger_spent_usd": round(spent, 4), "in_flight_reserved_usd": round(reserved, 4)}
    for row in db.execute("SELECT provider, model, status, COUNT(*), ROUND(COALESCE(SUM(cost_usd),0),4) FROM calls GROUP BY provider, model, status"):
        out["by_model"].append(dict(zip(("provider", "model", "status", "attempts", "usd"), row)))
    db.close()
    return out


def write_report(hosted_dir, date="2026-11-11", hosted_pages_work=None):
    from .evalscore import results_markdown, run_baselines, score_hosted_dir
    base = run_baselines()
    hosted = score_hosted_dir(hosted_dir, work=hosted_pages_work)
    sets = [{x["case"] for x in s["per_case"]} for n, s in base["systems"].items() if n in ("rules+propagation", "gliner+propagation")]
    sets += [{x["case"] for x in s["per_case"]} for s in hosted["systems"].values() if s["task"] == "detect" and s["aggregate"].get("calls")]
    common = sorted(set.intersection(*sets)) if sets else []
    base_c = run_baselines(common) if common else base
    hosted_c = score_hosted_dir(hosted_dir, common, work=hosted_pages_work) if common else hosted
    led = ledger_summary(os.environ.get("CASEPATCH_SPEND_LEDGER", str(Path.home() / ".casepatch" / f"eval-{date}.sqlite")))
    dry = json.loads((WORK / "dryrun.json").read_text()) if (WORK / "dryrun.json").exists() else None
    probes = [json.loads(l) for l in (WORK / "probe_gemini.jsonl").read_text().splitlines() if l.strip()] if (WORK / "probe_gemini.jsonl").exists() else []
    res = {"date": date, "common_cases": common, "baselines_all": base, "hosted_all": hosted, "baselines_common": base_c,
           "hosted_common": hosted_c, "ledger": led, "dry_run": dry, "gemini_access_probe": probes}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{date}-results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    md = [f"# Model eval results ({date})", "",
          f"Same pages and labels for every system. Main table = cases present for every system: {', '.join(common) or 'none'}.",
          "Gemini rows ran on the **free tier** ($0 billed); Anthropic and OpenAI rows are paid (capped). `$ / page` is the "
          "paid-tier price from provider usage for every hosted row.",
          "Precision counts any span not in the labels as a false positive, so it is a lower bound (models also flag unlabeled personal data such as personal numbers and birthplaces).",
          "", "## Common cases", "", results_markdown(base_c, hosted_c), "", "## All cases each system ran", "",
          results_markdown(base, hosted), "", "## Spend (shared ledger, CasePatch estimate at paid prices)", "",
          "```", json.dumps(led, indent=1), "```"]
    if dry:
        md += ["", "## Dry run (worst case for the full plan)", "", "```",
               "\n".join(f"{r['provider']:7} {r['model']:24} {r['task']:6} req {r['requests']:>3}  worst ${r['worst_usd']:.2f}  expected ${r['expected_usd']:.2f}" for r in dry["rows"]),
               f"TOTAL worst ${dry['total_worst_usd']:.2f} expected ${dry['total_expected_usd']:.2f}", "```"]
    am = RESULTS / f"{date}-anthropic-models.json"
    if am.exists():
        a_ = json.loads(am.read_text())
        md += ["", f"## Anthropic (Claude): models and prices", "",
               f"Read from the official models/pricing docs at {a_['read_at']}. Detection only; Claude has no image-edit endpoint (edit N/A).",
               f"Hard Anthropic pool cap ${POOL_CAPS['anthropic']:.0f} inside the ${TOTAL_CAP_USD:.0f} grand total. List-models check: "
               + ("done" if a_.get("verified_by_list_models") else "not run (" + a_.get("why_not_verified", "") + ")"), "",
               "| Model | Class | $/MTok in | $/MTok out |", "|---|---|---|---|"] + \
              [f"| `{x['id']}` | {x['class']} | {x['input_usd_per_mtok']:g} | {x['output_usd_per_mtok']:g} |" for x in a_["models_top_first"]]
    rep = WORK / "hosted-sonnet-run2"
    if rep.exists() and any(rep.glob("*.json")):  # same model, same pages, second run: run-to-run variance
        rh = score_hosted_dir(rep, common or None, work=hosted_pages_work)
        md += ["", "## Repeat run (variance): same model, same pages, second independent run", "",
               results_markdown({"systems": {}}, {"systems": {k + " (run 2)": v for k, v in rh["systems"].items()}})]
        res["repeat_runs"] = rh
        (RESULTS / f"{date}-results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    eb = RESULTS / f"{date}-editbench.json"
    if eb.exists():
        from .editbench import table as eb_table
        e_ = json.loads(eb.read_text())
        md += ["", "## Edit benchmark: bounded crop edits, hosted vs local (same 16 crops)", "",
               e_.get("note", ""), "", eb_table(e_), "", f"OCR reads the original value in {e_['controls_readable']} original crops "
               "(`exact_on_readable_controls` counts only those)."] + e_.get("findings", [])
    if probes:
        md += ["", "## Gemini access probe (one minimal request each, free-tier key)", ""] + \
              [f"* `{p['model']}`: {p['status'].splitlines()[0][:120]}" for p in probes]
    (RESULTS / "RESULTS.md").write_text("\n".join(md) + "\n")
    splice_local_section(os.environ.get("CASEPATCH_LOCALVLM_DIR", str(WORK / "localvlm")))
    return str(RESULTS / "RESULTS.md")


LOCAL_MARK = ("<!-- local-vlm:start -->", "<!-- local-vlm:end -->")


def local_vlm_markdown(local_dir, cases=None):
    """Open-weight VLMs run on this box's CPU through llama.cpp (casepatch.hosted.localvlm): same prompt, grounding and
    scoring as the hosted rows; $0; latency is wall time per page on a SHARED 8-core box (other jobs were running)."""
    from .evalscore import results_markdown, score_hosted_dir
    local_dir = Path(local_dir)
    if not local_dir.exists() or not any(local_dir.glob("local-vlm__*.json")):
        return ""
    sc = score_hosted_dir(local_dir, cases)
    lines = [LOCAL_MARK[0], "## Local open-weight VLMs (CPU only, llama.cpp GGUF, $0)", "",
             "Same page images, OCR lines, prompt, grounding and labels as the hosted rows. Latency = wall time per page "
             "on this shared 8-core / 16 GB box while other jobs ran (load average far above 8), so it is an upper bound.",
             "Candidates, licences and rented-GPU estimates for models too big for this box: `bench/models/OPEN_WEIGHT.md`.", "",
             results_markdown({"systems": {}}, sc), "", "| System | Pages done | Errors | Malformed JSON pages | Stopped | Settings |",
             "|---|---|---|---|---|---|"]
    for f in sorted(local_dir.glob("local-vlm__*.json")):
        r = json.loads(f.read_text()); st = r.get("settings", {})
        lines.append(f"| {r['system']} | {len(r['calls'])} | {len(r['errors'])} | {r.get('malformed_json_pages', 0)} | {r.get('stopped') or '-'} | "
                     f"{st.get('model_file')} + {st.get('mmproj_file')}, ctx {st.get('ctx')}, image side {st.get('max_image_side')}, "
                     f"{st.get('threads')} threads |")
    lines.append(LOCAL_MARK[1])
    return "\n".join(lines)


def splice_local_section(local_dir, cases=None, md_path=None):
    """Replace (or append) the local-VLM section of RESULTS.md without re-running the paid/hosted report."""
    md_path = Path(md_path or (RESULTS / "RESULTS.md"))
    txt = md_path.read_text() if md_path.exists() else ""
    sec = local_vlm_markdown(local_dir, cases)
    if LOCAL_MARK[0] in txt:
        pre, rest = txt.split(LOCAL_MARK[0], 1)
        post = rest.split(LOCAL_MARK[1], 1)[1] if LOCAL_MARK[1] in rest else ""
        txt = pre.rstrip("\n") + ("\n\n" + sec if sec else "") + post
    elif sec:
        txt = txt.rstrip("\n") + "\n\n" + sec + "\n"
    md_path.write_text(txt)
    return str(md_path)


ENDPOINT_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _crop_and_mask(t):
    from PIL import Image
    im = np.asarray(Image.open(t["png"]).convert("RGB"))
    x0, y0, x1, y1 = t["crop"]
    crop = im[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    a, b, c, d = t["mask"]
    mask[max(0, b - 1):d + 1, max(0, a - 1):c + 1] = 1
    return crop, mask


def _edit_raw(editor, crop, mask, instruction, case_id, captured):
    """Run ImageEditor but also return the RAW model image (before mask compositing) for out-of-mask scoring."""
    from . import client as C
    orig = C.composite_inside_mask
    def spy(crop_rgb, m, edited, ocr_check=None, expect_text=None):
        captured["raw"] = edited
        return orig(crop_rgb, m, edited, None, None)
    C.composite_inside_mask = spy
    try:
        out = editor.edit(crop, mask, instruction, None, case_id)
    finally:
        C.composite_inside_mask = orig
    last = editor.c.ledger.last(case_id)  # this call's row (other runs may share the ledger concurrently)
    if last and last["status"] == "committed":
        captured["cost"] = last["cost_usd"]
    return {"composite": out, "raw": captured.get("raw")}


def _ocr_text(img, ocr_fn=None):
    if ocr_fn is not None:
        return ocr_fn(img)
    from ..docs.extract import ocr_lines
    return " ".join(l.text for l in ocr_lines(np.ascontiguousarray(img), "e:"))


def score_edit(t, crop, mask, raw, ocr_fn=None):
    import cv2
    from .evalscore import fuzzy_in
    comp, model_img = raw.get("composite"), raw.get("raw")
    if comp is None or model_img is None:
        return {"case": t["case"], "file": t["file"], "value": t["value"], "ok": False, "reason": "no image returned"}
    from .client import flatten_over
    m = flatten_over(model_img, crop)
    out = mask == 0
    diff = np.abs(m.astype(int) - crop.astype(int)).max(axis=2)
    comp_diff = np.abs(comp.astype(int) - crop.astype(int)).max(axis=2)
    txt = _ocr_text(comp, ocr_fn)
    return {"case": t["case"], "file": t["file"], "value": t["value"], "replacement": t["replacement"], "ok": True,
            "readback_ok": fuzzy_in(t["replacement"], txt, 0.8), "residual_original": fuzzy_in(t["value"], txt, 0.8),
            "raw_out_of_mask_mean_abs": round(float(diff[out].mean()), 2) if out.any() else 0.0,
            "raw_out_of_mask_changed_frac": round(float((diff[out] > 25).mean()), 4) if out.any() else 0.0,
            "composite_out_of_mask_max": int(comp_diff[out].max()) if out.any() else 0, "ocr_text": txt[:200]}


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    from ..docs.byok import register_eval_keys
    register_eval_keys()  # eval keys are for public samples only: the product refuses them on user documents
    ap = argparse.ArgumentParser(prog="python -m casepatch.hosted.modeleval")
    ap.add_argument("action", choices=["prepare", "dry-run", "baselines", "run", "score", "probe", "report", "report-local"])
    ap.add_argument("--hosted-dir", default=None)
    ap.add_argument("--hosted-pages-work", default=None, help="pages cache the hosted run used (if not recorded in its JSON)")
    ap.add_argument("--ceiling", action="store_true", help="restrict to the quality-ceiling models per provider")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--providers", nargs="+", default=["openai", "gemini", "kimi", "anthropic"],
                    choices=["openai", "gemini", "kimi", "anthropic", "fal"])  # fal: opt-in only, never a default
    ap.add_argument("--tasks", nargs="+", default=["detect", "edit"], choices=["detect", "edit"])
    ap.add_argument("--models", nargs="+", default=None, help="restrict to these model ids")
    ap.add_argument("--max-output-tokens", type=int, default=EVAL_MAX_OUTPUT_TOKENS)
    ap.add_argument("--max-image-side", type=int, default=EVAL_MAX_IMAGE_SIDE)
    ap.add_argument("--edit-targets", type=int, default=EVAL_EDIT_TARGETS_PER_CASE)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--gemini-thinking", choices=["low", "high"], default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between calls (free-tier pacing)")
    ap.add_argument("--backoff", type=float, default=60.0, help="seconds to wait once after a 429 before stopping")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not 64 <= a.max_output_tokens <= 8192 or not 128 <= a.max_image_side <= 2048 or not 0 <= a.edit_targets <= 20:
        ap.error("bounds: max-output-tokens 64..8192, max-image-side 128..2048, edit-targets 0..20")
    models = None
    if a.ceiling:
        models = CEILING
    if a.models:
        models = {p: {t: [m for m in EVAL_MODELS[p][t] if m in a.models] for t in ("detect", "edit")} for p in EVAL_MODELS}
    if a.action == "report":
        print(write_report(a.hosted_dir or str(WORK / "hosted-v2"), hosted_pages_work=a.hosted_pages_work))
        return 0
    if a.action == "probe":
        try:
            for r in probe_access(a.providers, a.models or [m for p in a.providers for t in ("detect", "edit") for m in EVAL_MODELS[p][t]]):
                print(json.dumps(r))
        except HostedDisabled as e:
            print("refused:", redact(e)); return 2
        return 0
    if a.action == "prepare":
        ok, pending = available_cases(a.cases)
        for c in ok:
            print(c, len(prepare_case(c, force=True)), "pages")
        print("pending labels:", pending)
        return 0
    if a.action == "dry-run":
        pl = plan(a.cases, a.providers, a.tasks, models, a.max_output_tokens, a.max_image_side, 1, a.edit_targets)
        print(plan_table(pl))
        if a.json:
            Path(a.json).write_text(json.dumps(pl, indent=1))
        return 0
    if a.action == "baselines":
        from .evalscore import run_baselines
        res = run_baselines(a.cases)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return 0
    if a.action == "report-local":  # $0: splice the local open-weight VLM section into RESULTS.md
        print(splice_local_section(os.environ.get("CASEPATCH_LOCALVLM_DIR", str(WORK / "localvlm")), a.cases))
        return 0
    if a.action == "score":
        from .evalscore import score_hosted_dir
        res = score_hosted_dir(WORK / "hosted", a.cases)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return 0
    try:
        run_hosted(a.providers, a.cases, a.tasks, models, max_output_tokens=a.max_output_tokens,
                   max_image_side=a.max_image_side, edit_n=a.edit_targets, allow_partial=a.allow_partial,
                   min_interval_s=a.min_interval, rate_limit_backoff_s=a.backoff, gemini_thinking=a.gemini_thinking,
                   out_dir=a.out_dir)
    except HostedDisabled as e:
        print("refused:", redact(e)); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
