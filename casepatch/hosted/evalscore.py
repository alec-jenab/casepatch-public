"""Scoring for the multi-model eval (shared by hosted models and the local baselines; $0, no network).

Per system and case, on the PDF/image pages only (JSON traces are excluded for every system):
  recall (raw = found on that file; propagated = also counted when the same value was detected anywhere in the case,
  which is what CasePatch's case-wide propagation does), leaked values, precision / false-positive proposals,
  preserved-fact hits (over-redaction of amounts, places, evidence), per-category recall, identity consistency,
  face / signature region recall, and eval-only supplementary handwriting labels.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

from .modeleval import CASES_DIR, NON_LATIN, ID_KINDS, RESULTS, SUPPLEMENTARY, WORK, available_cases, load_labels, prepare_case


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def fuzzy_in(value, text, thr=0.9):
    v, t = norm(value), norm(text)
    if not v:
        return False
    if v in t:
        return True
    if len(v) < 5:
        return False
    n = len(v)
    return any(difflib.SequenceMatcher(None, v, t[i:i + n]).ratio() >= thr for i in range(max(1, len(t) - n + 1)))


def _match(value, det):
    v, t = norm(value), norm(det["text"])
    if not t:
        return False
    if v in t or (len(t) >= 4 and t in v):
        return True
    if det.get("grounding") == "box":  # model read pixels itself: allow small reading differences
        return difflib.SequenceMatcher(None, v, t).ratio() >= 0.8 or fuzzy_in(v, t, 0.8) or fuzzy_in(t, v, 0.8)
    return False


def _category(entity_kind, value, file, case, supp):
    cats = []
    k = entity_kind
    cats.append("names" if k in ("PERSON", "ORG") else "ids_numbers" if k in ID_KINDS else "address" if k == "ADDRESS"
                else "dob" if k == "DOB" else "other")
    if NON_LATIN.search(value):
        cats.append("non_latin")
    if value in supp.get("handwritten_in_main_labels", {}).get(case, {}).get(file, []):
        cats.append("handwriting")
    return cats


def _iou_cov(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter), inter / max(1e-9, aa)


REGION_KIND = {"face": {"face", "photo", "portrait"}, "signature": {"signature", "handwriting"}}


def score_case(case, detections, regions, pages=None, linked_by="person_id"):
    labels = load_labels(case)
    pages = pages or prepare_case(case)
    supp = json.loads(SUPPLEMENTARY.read_text()) if SUPPLEMENTARY.exists() else {}
    page_files = {p["file"] for p in pages}
    ent_of = {}
    for e in labels["entities"]:
        for v in e["values"]:
            ent_of.setdefault(norm(v), e)
    dets = [d for d in detections if d["file"] in page_files]
    by_file = {}
    for d in dets:
        by_file.setdefault(d["file"], []).append(d)
    total = found_raw = found_prop = 0
    leaks, cats = [], {}
    for f, values in labels["occurrences"].items():
        if f not in page_files:
            continue
        for v in values:
            total += 1
            e = ent_of.get(norm(v)) or next((ent_of[k] for k in ent_of if norm(v) in k or k in norm(v)), {"kind": "OTHER", "id": "?"})
            raw = any(_match(v, d) for d in by_file.get(f, []))
            prop = raw or any(_match(v, d) and d.get("grounding") != "box" for d in dets)
            found_raw += raw; found_prop += prop
            if not prop:
                leaks.append({"file": f, "value": v})
            for c in _category(e["kind"], v, f, case, supp):
                cc = cats.setdefault(c, {"total": 0, "found": 0})
                cc["total"] += 1; cc["found"] += prop
    allv = sorted({norm(v) for e in labels["entities"] for v in e["values"]} |
                  {norm(v) for vs in labels["occurrences"].values() for v in vs})
    fps = [d for d in dets if not any(_match(v, d) for v in allv)]
    facts = [(None, x) for x in labels.get("facts", {}).get("preserved_values", [])] + \
            [(ev["file"], ev["must_contain"]) for ev in labels.get("evidence", [])]
    fact_hits = sorted({x for f, x in facts if any((f is None or d["file"] == f) and len(norm(d["text"])) >= 3 and
                                                   (norm(x) in norm(d["text"]) or norm(d["text"]) in norm(x)) for d in dets)})
    # identity consistency: pairs of matched detections on the same page (hosted) / case (baselines with entity ids)
    matched = []
    for d in dets:
        if d.get(linked_by) in (None, ""):
            continue
        e = next((ent_of[v] for v in ent_of if _match(v, d) and ent_of[v]["kind"] == "PERSON"), None)
        if e:
            matched.append((d, e["id"]))
    same_ok = same_n = merge_err = diff_n = 0
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            (a, ea), (b, eb) = matched[i], matched[j]
            if linked_by == "person_id" and (a["file"], a["page"]) != (b["file"], b["page"]):
                continue
            if norm(a["text"]) == norm(b["text"]):
                continue
            if ea == eb:
                same_n += 1; same_ok += a[linked_by] == b[linked_by]
            else:
                diff_n += 1; merge_err += a[linked_by] == b[linked_by]
    # regions
    scale_of = {(p["file"], p["page"]): p["to_px"] if not p["raster"] else p["scale"] for p in pages}
    reg_res = {}
    for f, rl in labels.get("regions", {}).items():
        if f not in page_files:
            continue
        for r in rl:
            kind = r["kind"]
            if kind not in REGION_KIND:
                continue
            bb = r.get("bbox") or [v * scale_of.get((f, r.get("page", 0)), 1.0) for v in r["bbox_pt"]]
            hit = any(g["file"] == f and g["kind"] in REGION_KIND[kind] and
                      (lambda iou_cov: iou_cov[0] >= 0.3 or iou_cov[1] >= 0.5)(_iou_cov(bb, g["bbox"])) for g in regions)
            key = kind + (":" + r["role"] if r.get("role") else "")
            rr = reg_res.setdefault(key, {"total": 0, "found": 0}); rr["total"] += 1; rr["found"] += hit
    # supplementary handwriting labels
    sup = supp.get("cases", {}).get(case, {})
    s_total = s_found = 0; s_missed = []
    for f, items in sup.items():
        for it in items:
            s_total += 1
            ok = any(_match(it["value"], d) for d in by_file.get(f, []))
            s_found += ok
            if not ok:
                s_missed.append(it["value"])
    return {"case": case, "labeled_occurrences": total, "found_raw": found_raw, "found_propagated": found_prop,
            "recall_raw": round(found_raw / max(total, 1), 4), "recall_propagated": round(found_prop / max(total, 1), 4),
            "leaked": leaks, "proposals": len(dets), "false_positive_proposals": len(fps),
            "precision": round((len(dets) - len(fps)) / max(len(dets), 1), 4),
            "false_positive_examples": sorted({d["kind"] + ":" + d["text"] for d in fps})[:15],
            "preserved_fact_hits": fact_hits, "categories": cats,
            "identity": {"same_person_pairs": same_n, "same_person_linked": same_ok, "different_person_pairs": diff_n,
                         "wrongly_merged": merge_err},
            "regions": reg_res, "box_grounded_detections": sum(d.get("grounding") == "box" for d in dets),
            "handwriting_supplementary": {"total": s_total, "found": s_found, "missed": s_missed}}


def aggregate(per_case):
    agg = {"labeled_occurrences": 0, "found_raw": 0, "found_propagated": 0, "proposals": 0, "false_positive_proposals": 0,
           "leaked_values": 0, "preserved_fact_hits": 0, "categories": {}, "regions": {},
           "identity": {"same_person_pairs": 0, "same_person_linked": 0, "different_person_pairs": 0, "wrongly_merged": 0},
           "handwriting_supplementary": {"total": 0, "found": 0}}
    for r in per_case:
        for k in ("labeled_occurrences", "found_raw", "found_propagated", "proposals", "false_positive_proposals"):
            agg[k] += r[k]
        agg["leaked_values"] += len(r["leaked"]); agg["preserved_fact_hits"] += len(r["preserved_fact_hits"])
        for grp in ("categories", "regions"):
            for c, v in r[grp].items():
                a = agg[grp].setdefault(c, {"total": 0, "found": 0}); a["total"] += v["total"]; a["found"] += v["found"]
        for k in agg["identity"]:
            agg["identity"][k] += r["identity"][k]
        for k in ("total", "found"):
            agg["handwriting_supplementary"][k] += r["handwriting_supplementary"][k]
    n = max(agg["labeled_occurrences"], 1)
    agg["recall_raw"] = round(agg["found_raw"] / n, 4); agg["recall_propagated"] = round(agg["found_propagated"] / n, 4)
    agg["precision"] = round((agg["proposals"] - agg["false_positive_proposals"]) / max(agg["proposals"], 1), 4)
    return agg


# ------------------------------------------------------------------------------------------------ baselines
def baseline_detections(case, review_dir, pages):
    fmap = json.loads((Path(review_dir) / "file-map.json").read_text())
    props = json.loads((Path(review_dir) / "proposals.json").read_text())
    pinfo = {(p["file"], p["page"]): p for p in pages}
    dets, regs = [], []
    for o in props["occurrences"]:
        f = fmap.get(o["doc"])
        dets.append({"file": f, "page": o.get("page", 0), "text": o["text"], "kind": o["kind"], "entity": o.get("entity"),
                     "grounding": "ocr", "case": case})
    for r in props["regions"]:
        f = fmap.get(r["doc"]); p = pinfo.get((f, r.get("page", 0)))
        k = p["to_px"] if p and not p["raster"] else 1.0
        regs.append({"file": f, "page": r.get("page", 0), "kind": r["kind"], "bbox": [v * k for v in r["bbox"]]})
    return dets, regs


BASELINE_NAMES = {"rules": "local rules (no propagation)", "rules+propagation": "local rules + propagation",
                  "gliner": "local rules+GLiNER (no propagation)", "gliner+propagation": "local rules+GLiNER + propagation",
                  "llm": "local rules+Qwen2.5-3B Q4 (no propagation)", "llm+propagation": "local rules+Qwen2.5-3B Q4 + propagation"}


def run_baselines(cases=None, work=WORK):
    """Score the local baseline runs produced by `python -m casepatch.bench.detectors` (see README) for each case."""
    ok, pending = available_cases(cases)
    out = {"cases": ok, "pending": pending, "systems": {}}
    for c in ok:
        pages = prepare_case(c, work)
        bj = Path(work) / f"{c}-baselines.json"
        binfo = json.loads(bj.read_text())["runs"] if bj.exists() else {}
        for name in BASELINE_NAMES:
            rd = Path(work) / c / (name + ".review")
            if not ((rd / "proposals.json").exists() and (rd / "file-map.json").exists()) or "error" in binfo.get(name, {}):
                continue  # missing, in progress or failed run
            dets, regs = baseline_detections(c, rd, pages)
            sc = score_case(c, dets, regs, pages, linked_by="entity")
            info = binfo.get(name, {})
            sc["seconds_case"] = info.get("seconds"); sc["peak_rss_mb"] = info.get("peak_rss_mb")
            out["systems"].setdefault(name, {"label": BASELINE_NAMES[name], "per_case": []})["per_case"].append(sc)
    for s in out["systems"].values():
        s["aggregate"] = aggregate(s["per_case"])
        s["aggregate"]["seconds_total"] = round(sum(x.get("seconds_case") or 0 for x in s["per_case"]), 1)
        s["aggregate"]["cost_usd"] = 0.0
    return out


def _fix_legacy_gemini_boxes(r, pages_by_case):
    """Runs made before box_order existed stored Gemini boxes read as [x0,y0,x1,y1] although Gemini answers in its
    native [ymin,xmin,ymax,xmax]; undo the page scaling and swap (exact, no information lost)."""
    if r.get("box_order") or r.get("provider") != "gemini":
        return r
    for x in r.get("regions", []) + [d for d in r.get("detections", []) if d.get("box")]:
        p = next((q for q in pages_by_case.get(x["case"], []) if q["file"] == x["file"] and q["page"] == x["page"]), None)
        if not p:
            continue
        key = "bbox" if "bbox" in x else "box"
        a, b, c, e = x[key]
        r0, r1, r2, r3 = a * 1000 / p["w"], b * 1000 / p["h"], c * 1000 / p["w"], e * 1000 / p["h"]
        x[key] = [r1 * p["w"] / 1000, r0 * p["h"] / 1000, r3 * p["w"] / 1000, r2 * p["h"] / 1000]
    r["box_order"] = "yx (fixed post hoc)"
    return r


def score_hosted_dir(d, cases=None, work=None):
    ok, _ = available_cases(cases)
    out = {"systems": {}}
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        work = Path(r.get("pages_work") or work or WORK)  # score against the exact pages/OCR lines the model was shown
        if r.get("task") == "detect":
            run_cases = [c for c in ok if any(x["case"] == c for x in r["calls"] + r["errors"])]
            if r.get("raw") and r.get("box_order"):  # re-ground from the stored raw answers with the current parser
                from .modeleval import _parse_detect, ground_page_output
                pmap = {(c, p["file"], p["page"]): p for c in run_cases for p in prepare_case(c, work)}
                r["detections"], r["regions"], r["rejected_ungrounded"], r["malformed_json_pages"] = [], [], 0, 0
                for raw in r["raw"]:
                    pg = pmap.get((raw["case"], raw["file"], raw["page"]))
                    if not pg:
                        continue
                    dd = _parse_detect(raw["text"])
                    r["malformed_json_pages"] += bool(dd.get("_salvaged"))
                    dets, regs, rej = ground_page_output(pg, dd, yx=r["box_order"].startswith("yx"))
                    for x in dets + regs:
                        x["case"] = raw["case"]
                    r["detections"] += dets; r["regions"] += regs; r["rejected_ungrounded"] += rej
            r = _fix_legacy_gemini_boxes(r, {c: prepare_case(c, work) for c in run_cases})
            per = []
            for c in run_cases:
                pages = prepare_case(c, work)
                per.append(score_case(c, [x for x in r["detections"] if x.get("case") == c],
                                      [x for x in r["regions"] if x.get("case") == c], pages))
            agg = aggregate(per)
            calls = r["calls"]
            costs = [x["cost_usd"] for x in calls if x.get("cost_usd") is not None]
            agg.update({"calls": len(calls), "errors": len(r["errors"]), "stopped": r.get("stopped"),
                        "rejected_ungrounded": r.get("rejected_ungrounded", 0), "malformed_json_pages": r.get("malformed_json_pages", 0),
                        "cost_usd": round(sum(costs), 4), "cost_unknown_calls": len(calls) - len(costs),
                        "cost_per_page_usd": round(sum(costs) / max(len(costs), 1), 5),
                        "latency_p50_s": sorted(x["latency_s"] for x in calls)[len(calls) // 2] if calls else None})
            out["systems"][r["system"]] = {"task": "detect", "per_case": per, "aggregate": agg}
        elif r.get("task") == "edit":
            e = [x for x in r["edits"] if x.get("ok")]
            out["systems"][r["system"]] = {"task": "edit", "aggregate": {
                "edits": len(r["edits"]), "returned_image": len(e), "errors": len(r["errors"]),
                "readback_ok": sum(x["readback_ok"] for x in e), "residual_original": sum(x["residual_original"] for x in e),
                "raw_out_of_mask_mean_abs": round(sum(x["raw_out_of_mask_mean_abs"] for x in e) / max(len(e), 1), 2),
                "composite_out_of_mask_max": max([x["composite_out_of_mask_max"] for x in e] or [0]),
                "cost_usd": round(sum(x.get("cost_usd") or 0 for x in e), 4),
                "latency_p50_s": sorted(x["latency_s"] for x in r["edits"] if "latency_s" in x)[len(r["edits"]) // 2] if r["edits"] else None},
                "edits": r["edits"]}
    return out


def _rf(d, k):
    v = d.get(k) or {"total": 0, "found": 0}
    return f"{v['found']}/{v['total']}" if v["total"] else "-"


def results_markdown(baselines, hosted, cases_note=""):
    hdr = ("| System | Cases | Recall raw | Recall propagated | Leaked | Precision | FP proposals | Fact hits | Names | IDs/numbers | "
           "Non-Latin | Handwriting (main) | Handwriting (supp.) | Face primary | Face ghost | Signature | Wrong merges | $/page | p50 latency |")
    rows = [hdr, "|" + "---|" * 19]
    def row(name, s, cost, lat):
        a = s["aggregate"]; cs = ",".join(x["case"][-2:] for x in s["per_case"])
        hs = a["handwriting_supplementary"]
        rows.append(f"| {name} | {cs} | {a['recall_raw']:.2f} | {a['recall_propagated']:.2f} | {a['leaked_values']} | {a['precision']:.2f} | "
                    f"{a['false_positive_proposals']} | {a['preserved_fact_hits']} | {_rf(a['categories'], 'names')} | "
                    f"{_rf(a['categories'], 'ids_numbers')} | {_rf(a['categories'], 'non_latin')} | {_rf(a['categories'], 'handwriting')} | "
                    f"{hs['found']}/{hs['total']} | {_rf(a['regions'], 'face:primary')} | {_rf(a['regions'], 'face:ghost')} | "
                    f"{_rf(a['regions'], 'signature')} | {a['identity']['wrongly_merged']}/{a['identity']['different_person_pairs']} | {cost} | {lat} |")
    for name, s in baselines.get("systems", {}).items():
        a = s["aggregate"]
        row(s["label"], s, "$0 (local)", f"{a.get('seconds_total', 0) / max(1, sum(1 for _ in s['per_case'])):.0f} s/case (whole pipeline)")
    for name, s in hosted.get("systems", {}).items():
        if s["task"] != "detect":
            continue
        a = s["aggregate"]
        row(name, s, "$0 (local CPU)" if name.startswith("local-vlm:") else f"${a['cost_per_page_usd']:.4f} est.", f"{a['latency_p50_s']} s/page" if a.get("latency_p50_s") is not None else "-")
    return "\n".join(rows) + ("\n\n" + cases_note if cases_note else "")
