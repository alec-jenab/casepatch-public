"""Orchestrates extract -> regions -> detect -> resolve -> review -> render -> verify for one case folder."""
from __future__ import annotations

import json

import numpy as np
import os
import platform
import re
import time
from pathlib import Path

from . import resolve as R
from .detect import RulesDetector, Unit, Chunk, chunks_for, make_detector
from .extract import extract_dir
from .model import quad_bbox, span_quad
from .render import render_document
from .verify import verify_case
from .visual import find_regions


def _mem():
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":"); out[k] = round(int(v.split()[0]) / 1024)
    except OSError:
        pass
    return out


def hardware():
    mem = 0
    try:
        mem = round(int(open("/proc/meminfo").readline().split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return {"cpu_count": os.cpu_count(), "machine": platform.machine(), "python": platform.python_version(), "ram_gb": mem, "gpu": None}


def run(input_dir, out_dir, detector="rules", review_dir=None, decisions_path=None, verify=True, propagate=True,
        categories=True, category_policy=None, hosted=None, relation_client=None, progress=None, source_uploaded=None,
        keep=None):
    """`hosted`: an opt-in casepatch.docs.hosted_detect.HostedPageDetector (None = local only, the default). Its
    proposals are cached in the review folder on the first run and reused afterwards (no second call), and values
    only the model found are not applied until the reviewer accepts them (decisions `_accept_text`).
    `source_uploaded`: True when the source files were uploaded to a service that runs this engine (report says so);
    None reads CASEPATCH_SOURCE_UPLOADED (set by the hosted service for its worker).
    `relation_client`: optional BYOK relation proposer for the consistency planner (None = local rules only).
    `keep`: terms never to replace (list of strings, newline-separated string or casepatch.keep.KeepList): a detected
    value equal to a kept term, or a whole-word part of one, is left unchanged, not listed, and not flagged as residual.
    `progress`: optional callable(event, detail) for coarse progress: ("start", None), (<stage name>, None) when a
    stage finishes, ("render_doc", (done, total)) per rendered document. Stage names and counts only, never content."""
    from ..keep import as_keep_list
    keep = as_keep_list(keep)  # invalid lists fail before any work
    out_dir = Path(out_dir)
    review_dir = Path(review_dir) if review_dir else out_dir.with_name(out_dir.name + ".review")
    if out_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    stages = {}
    def stage(name, t0):
        stages[name] = {"seconds": round(time.perf_counter() - t0, 2), **_mem()}
        _tick(name, None)
    def _tick(event, detail):
        if progress:
            try:
                progress(event, detail)
            except Exception:  # progress reporting must never break a run
                pass
    _tick("start", None)
    from . import decisions as DEC
    given_raw = json.loads(Path(decisions_path).read_text()) if decisions_path else None
    if decisions_path:
        DEC.parse(given_raw)  # malformed files fail before any work
    R.REJECTED.clear(); R.KEPT_VALUES.clear()
    t = time.perf_counter(); docs = extract_dir(input_dir); stage("extract_native_and_ocr", t)
    t = time.perf_counter()
    regions = []
    for d in docs:
        for p in d.pages:
            if p.image is not None:
                regions += find_regions(p, d.id)
    stage("visual_regions", t)
    chunks = chunks_for(docs)
    proposals, det_info = [], []
    names = ["rules"] + ([detector] if detector != "rules" else [])
    for name in names:
        t = time.perf_counter()
        det = make_detector(name)
        stage(f"load_{name}", t)
        t = time.perf_counter()
        for c in chunks:
            proposals += det.detect(c)
        stage(f"detect_{name}", t)
        info = {"name": det.name, "learned": det.learned, "pages": sum(1 for d in docs for _ in d.pages)}
        if hasattr(det, "rejected"):
            info["ungrounded_rejected"] = len(det.rejected)
        if hasattr(det, "calls"):
            info["llm_calls"] = det.calls
        det_info.append(info)
        if hasattr(det, "close"):
            det.close()
        del det
    # region text cleanup: OCR garbage inside a withheld region is not text to rewrite
    def in_region(doc_id, page, bbox):
        for r in regions:
            if r.id.startswith(f"{doc_id}:p{page}:"):
                rb = quad_bbox(r.quad)
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                    return True
        return False
    given, specials = DEC.parse(given_raw) if decisions_path else ({}, {})
    additions = specials.get("_add", []) or []
    reject_text = set(specials.get("_reject_text", []) or [])
    accept_text = set(specials.get("_accept_text", []) or [])
    replace_overrides = specials.get("_replace", []) or []
    region_kind_actions = specials.get("_region_kind", {}) or {}
    only_text = specials.get("_only_text", None)  # re-render: keep the reviewed set
    t = time.perf_counter()
    from .forms import form_field_proposals
    ff = form_field_proposals(docs)
    proposals += ff
    stage("detect_form_fields", t)
    det_info.append({"name": "form-fields", "learned": False, "proposals": len(ff)})
    from .cues import cue_proposals
    from .iddoc import id_document_proposals
    t = time.perf_counter()
    idp = id_document_proposals(docs, regions, proposals)
    cu = cue_proposals(docs)
    from .legal import legal_name_proposals, legal_gliner_proposals, filter_legal_structure_proposals
    lg = legal_name_proposals(docs)
    from .letters import letter_name_proposals
    lt = letter_name_proposals(docs)
    from .addressblock import address_block_proposals
    ab = address_block_proposals(docs)
    proposals += idp + cu + lg + lt + ab
    det_info.append({"name": "legal-names", "learned": False, "proposals": len(lg)})
    det_info.append({"name": "letters", "learned": False, "proposals": len(lt)})
    det_info.append({"name": "address-block", "learned": False, "proposals": len(ab)})
    t_gl = time.perf_counter()
    gl = legal_gliner_proposals(docs)
    covered = {(pr.line_id, pr.start, pr.end) for pr in proposals if pr.kind in ("PERSON", "SURNAME")}
    added = 0
    for pr in gl:
        if any(lid == pr.line_id and s <= pr.start and pr.end <= e for lid, s, e in covered):
            continue
        proposals.append(pr)
        covered.add((pr.line_id, pr.start, pr.end))
        added += 1
    stage("detect_legal_gliner", t_gl)
    det_info.append({"name": "legal-gliner", "learned": True, "proposals": added})
    det_info.append({"name": "cues", "learned": False, "proposals": len(cu)})
    stage("detect_id_layout", t)
    det_info.append({"name": "id-layout", "learned": False, "proposals": len(idp)})
    if categories:
        t = time.perf_counter()
        from .categories import hook, last_detector
        cp = hook(chunks, R, policy=category_policy)
        proposals += cp
        stage("detect_categories", t)
        det_info.append({"name": "categories", "learned": False, "proposals": len(cp), "flagged": len(last_detector().flags)})
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    hosted_info = None
    from . import hosted_detect as HD
    cached = HD.load_cache(review_dir)
    if cached is not None:  # model output from this case's first run: reused, never re-requested
        hp, hosted_info = cached
        proposals += hp
        det_info.append(hosted_info)
    elif hosted is not None:
        t = time.perf_counter()
        hp = hosted.detect_docs(docs)
        HD.save_cache(review_dir, hosted, hp)
        proposals += hp
        hosted_info = hosted.info()
        stage("detect_hosted", t)
        det_info.append(hosted_info)
    proposals += reviewer_additions(docs, additions)
    proposals = filter_legal_structure_proposals(docs, proposals)
    from .gazetteer import filter_places_and_businesses
    proposals, kept_as_facts = filter_places_and_businesses(docs, proposals)  # D4/A7: places and merchants are facts
    det_info.append({"name": "places-businesses", "learned": False, "dropped": len(kept_as_facts),
                     "kept_as_facts": sorted({(x["text"], x["reason"]) for x in kept_as_facts})})
    kept_by_user = sorted({pr.text for pr in proposals if keep.covers(pr.text)})
    if keep:  # the user's keep list wins over every detector (and reviewer additions)
        n_before = len(proposals)
        proposals = [pr for pr in proposals if not keep.covers(pr.text)]
        det_info.append({"name": "keep-list", "learned": False, "terms": len(keep), "dropped": n_before - len(proposals)})
    t = time.perf_counter()
    from .gazetteer import merchant_texts
    case_merchants = merchant_texts(docs)  # merchants named like people ('Corwin Hollow'): facts, words not propagated
    occs = R.build_occurrences(docs, proposals, propagate=propagate,
                               kept={x["text"] for x in kept_as_facts} | case_merchants)
    if keep:  # propagated variants of other values that fall on a kept term
        kept_by_user = sorted(set(kept_by_user) | {o.text for o in occs if keep.covers(o.text)})
        occs = [o for o in occs if not keep.covers(o.text)]
    digests = DEC.doc_digests(docs)
    DEC.assign_occurrence_ids(occs, digests)  # content-bound: independent of file order and of other proposals
    model_only = HD.model_only_texts(occs) if hosted_info else set()
    for o in occs:
        if o.text in model_only and HD.REVIEW_FLAG not in o.review:
            o.review.append(HD.REVIEW_FLAG)
    key_file = review_dir / "case.key"
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = key_file.read_bytes() if key_file.exists() else None
    cmap = R.CaseMap(key)
    if categories:
        from .categories import attach
        attach(cmap)
    if key is None:
        key_file.write_bytes(cmap.key); os.chmod(key_file, 0o600)
    for o in occs:
        p = R.parse_person(o.text) if o.kind == "PERSON" else None
        if p:
            cmap.sources.update(x.lower() for x in (p["given"], p["surname"]) if x)
    # persons first so email local parts can reuse the person token map
    for o in occs:  # honorific next to a name: keep 'Mr./Mrs./Ms. <fake>' coherent
        if o.kind in ("PERSON", "GIVEN"):
            ln = next((l for d in docs if d.id == o.doc for p in d.pages for l in p.lines if l.id == o.line_id), None)
            m = re.search(r"\b(Mr|Mrs|Ms|Miss|Sir|Madam)\.?\s*$", ln.text[:o.start]) if ln is not None else None
            pp = R.parse_person(o.text) if o.kind == "PERSON" else {"given": o.text}
            if m and pp and pp.get("given") and len(pp["given"]) > 1:
                cmap.style_hint[pp["given"].lower()] = "m" if m.group(1) in ("Mr", "Sir") else "f"
    # consistency planning: case fact graph (entities, variants, typed relations across and within documents), then
    # constraint-aware replacements (persons first, MRZ last: it reuses names/DOB/numbers), then the plan check
    from . import consistency as CC
    graph = CC.build_graph(CC.lines_of(docs), occs)
    CC.propose_relations(graph, relation_client)
    planned_violations = CC.plan(cmap, graph, occs) + CC.gender_violations(graph, cmap)
    if replace_overrides:
        apply_replacement_overrides(docs, occs, replace_overrides, digests)
    regions += signatures_matching_names(docs, occs, regions)
    DEC.assign_region_ids(regions, digests)
    stage("resolve_and_map", t)
    lines = {l.id: (d, p, l) for d in docs for p in d.pages for l in p.lines}
    def bbox_of(o):
        d, p, l = lines[o.line_id]
        if not l.chars:
            return None
        return [round(v, 1) for v in quad_bbox(span_quad(l.chars, o.start, o.end))]
    proposals_out = {"occurrences": [], "regions": []}
    for o in occs:
        bb = bbox_of(o)
        sad = bool(getattr(o, "meta", {}).get("sad"))  # PCI sensitive authentication data: source never written to disk
        proposals_out["occurrences"].append({"id": o.id, "doc": o.doc, "page": o.page, "line": o.line_id, "bbox": bb, "source": o.source,
                                             "kind": o.kind, "text": None if sad else o.text,
                                             "replacement": None if sad else o.replacement, "entity": f"{o.kind.lower()}:hidden" if sad else o.entity,
                                             "detectors": sorted(o.detectors), "confidence": round(o.confidence, 3), "review": o.review})
    for r in regions:
        doc_id, pg = r.id.split(":")[0], int(r.id.split(":")[1][1:])
        proposals_out["regions"].append({"id": r.id, "doc": doc_id, "page": pg, "kind": r.kind, "bbox": [round(v, 1) for v in quad_bbox(r.quad)],
                                         "reason": r.reason, "qr_payload_present": bool(r.payload)})
    defaults = {o["id"]: ("reject" if o["text"] in model_only else "accept") for o in proposals_out["occurrences"]}
    if model_only:
        proposals_out["model_suggestions"] = sorted(model_only)
    defaults.update({r["id"]: ("replace" if r["kind"] in REPLACEABLE else "keep") for r in proposals_out["regions"]})
    for r in proposals_out["regions"]:  # reviewer action by document and kind (e.g. {"document-03:signature": "erase"})
        act = region_kind_actions.get(f'{r["doc"]}:{r["kind"]}')
        if act in ("keep", "replace", "erase"):
            defaults[r["id"]] = act
    proposals_out["consistency"] = {"graph": graph.summary(), "violations": planned_violations}
    decisions = dict(defaults)
    decision_info = {"format": DEC.FORMAT, "proposals_digest": DEC.proposals_digest(proposals_out), "given": 0}
    if decisions_path:
        # every decision must match a current proposal (id bound to content, recorded kind/text equal); otherwise the
        # whole set is refused before anything is written. Unmentioned proposals keep their default (replace).
        applied = DEC.validate(given, {o["id"]: o for o in proposals_out["occurrences"]}, {r["id"]: r for r in proposals_out["regions"]})
        decisions.update(applied)
        decision_info.update(given=len(given), changed_from_default=sum(1 for k, v in applied.items() if defaults[k] != v),
                             proposals_without_decision=len(set(defaults) - set(applied)))
        for o in proposals_out["occurrences"]:
            if o["text"] in accept_text and o["text"] in model_only:
                decisions[o["id"]] = "accept"  # reviewer accepted a model suggestion
            if o["text"] in reject_text or (only_text is not None and o["text"] not in set(only_text)):
                decisions[o["id"]] = "reject"
    if decisions_path:
        nested = protect_inside_rejected(occs, decisions, digests)
        for o in nested:
            proposals_out["occurrences"].append({"id": o.id, "doc": o.doc, "page": o.page, "line": o.line_id, "bbox": bbox_of(o),
                                                 "source": o.source, "kind": o.kind, "text": o.text, "replacement": o.replacement,
                                                 "entity": o.entity, "detectors": sorted(o.detectors), "confidence": round(o.confidence, 3),
                                                 "review": o.review})
            decisions[o.id] = "accept"
        decision_info["values_replaced_inside_rejected_spans"] = len(nested)
        acc_texts = [o.text for o in occs if decisions.get(o.id) == "accept"]
        missing = [{"kind": a.get("kind"), "reason": "NOT_FOUND_OR_NOT_APPLIED"} for a in additions
                   if isinstance(a, dict) and not any(_squash(a.get("text", "")) in _squash(t) or _squash(t) in _squash(a.get("text", ""))
                                                      for t in acc_texts if t)]
        decision_info["additions"] = len(additions)
        decision_info["additions_not_applied"] = missing
    # a name or street word linked to a full value stays as printed when none of the values it came from is replaced
    # (the reviewer kept the person or the address)
    replaced_texts = {o.text for o in occs if decisions.get(o.id) == "accept"}
    for o in occs:
        src = (getattr(o, "meta", None) or {}).get("part_from")
        if src and decisions.get(o.id) == "accept" and not replaced_texts & set(src):
            decisions[o.id] = "reject"
    (review_dir / "proposals.json").write_text(json.dumps(proposals_out, indent=1, ensure_ascii=False))
    if not decisions_path:
        (review_dir / "decisions.json").write_text(json.dumps(DEC.file_entries(proposals_out, defaults), indent=1, ensure_ascii=False))
    from .review_html import build
    build(docs, proposals_out, {"_format": DEC.FORMAT, "_proposals": decision_info["proposals_digest"], **decisions}, review_dir / "review.html")
    # regions to edit (replace / erase); text edits inside them are suppressed (the region edit covers them)
    withheld = [r for r in regions if decisions[r.id] in ("replace", "erase")]
    for r in withheld:
        r.decision = decisions[r.id]
    accepted = []
    for o in occs:
        if decisions[o.id] != "accept":
            continue
        bb = bbox_of(o)
        if bb and _inside_any(o, bb, withheld):
            continue
        accepted.append(o)
    # render
    t = time.perf_counter()
    out_dir.mkdir(parents=True, mode=0o700)
    outputs, logs = {}, {}
    payload_map = {}
    for o in accepted:
        payload_map[o.text] = o.replacement
    # synthetic-face choice is a pure function of the case's input bytes (same case -> same faces, run after run)
    import hashlib as _hl
    case_digest = _hl.sha256(b"".join(_hl.sha256(Path(d.path).read_bytes()).digest() for d in sorted(docs, key=lambda d: d.id))).hexdigest()[:16]
    for d in docs:
        out_path = out_dir / f"{d.id}{d.ext}"
        ctx = {"payload_map": payload_map, "seed": cmap._h("region-seed")[:16], "signer": signers(d, regions, accepted, cmap),
               "default_signer": default_signer(d, accepted, cmap), "face_profile": face_profiles(d, accepted, cmap),
               "face_seed": case_digest + d.id}
        logs[d.id] = render_document(d, [o for o in accepted if o.doc == d.id], [r for r in withheld if r.id.startswith(d.id + ":")], out_path, ctx=ctx)
        outputs[d.id] = out_path
        _tick("render_doc", (len(outputs), len(docs)))
    stage("render_fresh_outputs", t)
    # values deliberately left unchanged: never claim the files are fully synthetic when any were kept
    kept_text, kept_regions = {}, {}
    for o in occs:
        if decisions[o.id] != "accept":
            why = "model_suggestion_not_accepted" if o.text in model_only and o.text not in accept_text and o.text not in reject_text \
                and given.get(o.id, {}).get("action") != "reject" else "reviewer_rejected"
            kept_text.setdefault(o.kind, {}).setdefault(why, 0)
            kept_text[o.kind][why] += 1
    for r in regions:
        if decisions[r.id] == "keep":
            kept_regions[r.kind] = kept_regions.get(r.kind, 0) + 1
    n_kept = sum(sum(v.values()) for v in kept_text.values())
    # TRUST1/A6c: never claim the files are synthetic. Only what CasePatch replaced is synthetic; anything its detectors
    # did not find is unchanged original content, and a run cannot know what it did not find.
    undetected = ("Values CasePatch did not detect are unchanged ORIGINAL content; review the files before sharing them.")
    if n_kept or kept_regions:
        statement = (f"NOT fully synthetic: {n_kept} detected text value(s) and {sum(kept_regions.values())} region(s) were kept "
                     "unchanged by decision and are ORIGINAL content (counts by kind below; values are not listed here). The "
                     "replaced values, signatures, faces and symbols are synthetic. " + undetected + " The documents are "
                     "transformed copies for evaluation use, not originals. No visible watermark is added; this report is the marker.")
    else:
        statement = ("The replaced values, signatures, faces and symbols are synthetic. " + undetected + " The documents are "
                     "transformed copies for evaluation use, not originals. No visible watermark is added; this report is the marker.")
    from .. import __version__ as _cp_version
    uploaded = source_uploaded if source_uploaded is not None else bool(os.environ.get("CASEPATCH_SOURCE_UPLOADED"))
    report = {"casepatch": _cp_version, "engine": "native-edit", "status": "TRANSFORMED_REVIEW_REQUIRED", "external_use": "NOT_CLEARED",
              "synthetic_content": {"is_synthetic": False,
                                    "is_synthetic_note": "never claimed: values the detectors missed stay original",
                                    "replaced_values_are_synthetic": True,
                                    "kept_by_decision": bool(n_kept or kept_regions), "statement": statement,
                                    "kept_original_text_by_kind": {k: kept_text[k] for k in sorted(kept_text)},
                                    "kept_regions_by_kind": dict(sorted(kept_regions.items())),
                                    "inpainting": inpaint_label(), "reviewer_additions": len(additions)},
              "decisions": decision_info,
              "privacy_certification": "NONE. Verification below reduces risk; it does not certify that no personal data remains.",
              "source_sent_to_cloud": bool(uploaded or (hosted_info and hosted_info.get("pages_sent"))),
              "processed_by": "casepatch hosted service" if uploaded else "local engine",
              "hardware": hardware(),
              "detectors": det_info, "documents": {}}
    if hosted_info:
        report["hosted_detection"] = {"provider": hosted_info.get("provider"), "model": hosted_info.get("model"),
                                      "pages_sent": hosted_info.get("pages_sent", 0), "error": hosted_info.get("error"),
                                      "model_only_values": len(model_only),
                                      "model_only_values_accepted": len(model_only & accept_text),
                                      "policy": "model suggestions are applied only after the reviewer accepts them"}
    if categories:
        from .categories import last_detector
        flags = last_detector().flags
        report["category_flags"] = [{k: v for k, v in f.items() if k != "text"} for f in flags]
        # local review folder only (like proposals.json): flagged values with their text and box, so the review screen
        # can show what was kept on purpose. PCI SAD never has text here (categories.py stores None).
        cf = []
        for f in flags:
            d_, p_, l_ = lines.get(f["line_id"], (None, None, None))
            bb = [round(v, 1) for v in quad_bbox(span_quad(l_.chars, f["start"], f["end"]))] if l_ is not None and l_.chars else None
            cf.append({"doc": f["doc"], "page": f["page"], "kind": f["kind"], "regulation": f["regulation"], "text": f["text"],
                       "bbox": bb, "confidence": f["confidence"], "reason": f.get("reason")})
        (review_dir / "category-flags.json").write_text(json.dumps(cf, indent=1, ensure_ascii=False))
    hidden_counts = {}
    rules = RulesDetector()
    for d in docs:
        texts = []
        for k, v in d.hidden.items():
            texts += [json.dumps(v, ensure_ascii=False)] if v else []
        for r in regions:
            if r.id.startswith(d.id + ":") and r.payload:
                texts.append(r.payload)
        units = [Unit(f"h{i}", t_, 0, t_, "hidden") for i, t_ in enumerate(texts)]
        found = rules.detect(Chunk(d.id, 0, units)) if units else []
        known = {o.text for o in occs}
        hidden_counts[d.id] = {"pii_like_values_in_stripped_layers": len(found) + sum(1 for t_ in texts for v in known if v in t_)}
    unresolved = []
    for o in occs:
        if o.review or decisions[o.id] == "reject":
            unresolved.append({"id": o.id, "doc": o.doc, "page": o.page, "kind": o.kind, "reasons": o.review,
                               "decision": decisions[o.id], "bbox": bbox_of(o)})
    for d in docs:
        dl = logs[d.id]
        counts = {}
        for o in accepted:
            if o.doc == d.id:
                counts[o.kind] = counts.get(o.kind, 0) + 1
        notes = [n for p in d.pages for n in p.notes]
        report["documents"][d.id] = {
            "type": d.kind, "pages": len(d.pages), "output": f"{d.id}{d.ext}",
            **({"format": d.textual.get("format"), "characters": d.textual.get("chars"),
                "billable_pages": _trace_pages(d.textual.get("chars") or 0),
                "pages_note": "a trace is one document page here; hosted accounting counts ceil(characters / 10,000) pages"}
               if d.textual else {}),
            "replacements_by_kind": counts,
            "edited_regions": [{"id": r.id.split(":", 1)[1], "kind": r.kind, "decision": decisions[r.id], "bbox": [round(v, 1) for v in quad_bbox(r.quad)],
                                "reason": r.reason, "method": next((e.get("method") for e in dl if e.get("region") == r.id), None)}
                               for r in withheld if r.id.startswith(d.id + ":")],
            "kept_regions_by_reviewer": [r.id for r in regions if r.id.startswith(d.id + ":") and decisions[r.id] == "keep"],
            "extraction_notes": [{k: v for k, v in n.items() if k != "line"} | ({"line": n["line"].split(":", 1)[1]} if "line" in n else {}) for n in notes],
            "stripped_layers": hidden_counts[d.id], "edit_methods": _count([e["method"] for e in dl]),
            "timings": d.timings}
    for d in docs:
        for e in logs[d.id]:
            if e.get("needs_review") or str(e.get("method", "")).startswith("raster_patch_rewrite_fallback"):
                unresolved.append({"id": e.get("region") or e.get("occurrence"), "doc": d.id, "page": e.get("page"), "kind": e.get("kind"),
                                   "reasons": [e.get("method"), e.get("reason") or e.get("fallback_reason")]})
    for r in regions:
        if decisions[r.id] == "keep" and r.kind in ("photo", "graphic", "handwriting", "face", "signature", "qr", "barcode"):
            unresolved.append({"id": r.id, "doc": r.id.split(":")[0], "page": int(r.id.split(":")[1][1:]), "kind": r.kind,
                               "reasons": ["REGION_KEPT_UNCHANGED_CONFIRM_NOT_SENSITIVE"]})
    for a in decision_info.get("additions_not_applied", []):
        unresolved.append({"id": None, "doc": None, "kind": a["kind"], "reasons": ["REVIEWER_ADDITION_NOT_APPLIED"]})
    report["needs_review"] = unresolved
    report["identity"] = {"entities": len({o.entity for o in accepted if o.kind == "PERSON"}),
                          "ambiguous_mentions": sum(1 for o in accepted if o.entity.startswith("ambiguous:")),
                          "policy": "one case-scoped map across all documents and traces; different given names => different people"}
    report["consistency"] = {"relations": graph.summary()["relations"], "entities": len(graph.entities),
                             "planned_violations": _count([v["code"] for v in planned_violations]),
                             "relations_from_model": sum(1 for r in graph.relations if r.source == "model"),
                             "policy": "relations are checked on original vs replacement; a mismatch already in the source is preserved"}
    consistency_detail = {"planned": planned_violations}
    report["rejected_proposals"] = _count([f"{x['detector']}:{x['reason']}" for x in R.REJECTED])
    if keep:  # counts only: the terms are the user's own and stay out of the shareable report
        report["keep_list"] = {"terms": len(keep), "detected_values_kept": len(kept_by_user)}
    if verify:
        t = time.perf_counter()
        all_repl = {o.replacement for o in occs}
        out_texts = {}
        kept_facts = set(R.KEPT_VALUES) | {x["text"] for x in kept_as_facts} | case_merchants
        vrep, vdetail = verify_case(docs, outputs, accepted, regions, logs, all_repl, texts=out_texts, kept=kept_facts,
                                    keep=keep)
        post = CC.verify_outputs(graph, out_texts, {o.id: o.replacement for o in accepted})
        report["consistency"]["post_edit"] = {"checked": post["checked"], "violations": _count([v["code"] for v in post["violations"]]),
                                              "replacements_not_read_back": len(post["not_found"])}
        consistency_detail["post_edit"] = post
        for v in post["violations"]:
            unresolved.append({"id": (v["occs"] or [None])[0], "doc": v["detail"].get("doc") if isinstance(v.get("detail"), dict) else None,
                               "kind": "CONSISTENCY", "reasons": ["POST_EDIT:" + v["code"]]})
        stage("verify_independent_rescan", t)
        report["verification"] = vrep
        for _id, _dl in logs.items():  # per-edit style parameters (local review folder only; no original values)
            if _id in vdetail and isinstance(vdetail[_id], dict):
                vdetail[_id]["edits"] = [{k: v for k, v in e.items() if k not in ("text", "original")} for e in _dl]
        (review_dir / "verification-detail.json").write_text(json.dumps(vdetail, indent=1, ensure_ascii=False, default=_jsonable))
    # likely misses (A6d/D3e/TRUST2): what a run can say about values it may have missed, without listing values
    handled = {o.text for o in occs} | {x["text"] for x in kept_as_facts} | set(kept_by_user) | set(keep.terms)  # proposed, kept as a fact, or on the keep list
    misses = likely_misses(docs, accepted, report.get("verification") or {}, handled)
    report["likely_misses"] = misses
    for m in misses:
        unresolved.append({"id": None, "doc": m["doc"], "page": m.get("page"), "kind": m.get("kind"), "reasons": ["LIKELY_MISS:" + m["code"]]})
    report["verification_scope"] = ("residual_original_values counts values CasePatch DETECTED, and name or street words from them "
                                    "standing alone, that are still readable in the output (text layer, objects, OCR of pixels). Values it never detected cannot be checked; "
                                    "likely_misses lists structural signs of them.")
    vr = report.get("verification") or {}
    report["gate"] = {"verified": bool(verify),
                      "residual_original_values": sum(int(v.get("residual_original_values") or 0) for v in vr.values()),
                      "pii_like_text_not_replacements": sum(int(v.get("rescan_pii_like_not_replacements") or 0) for v in vr.values()),
                      "likely_misses": len(misses), "needs_review": len(unresolved)}
    report["gate"]["pass"] = bool(verify) and not misses and report["gate"]["residual_original_values"] == 0
    CC.save(review_dir, graph, consistency_detail, cmap)
    report["stages"] = stages
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # local-only table of source filenames -> neutral output names (filenames can identify people)
    (review_dir / "file-map.json").write_text(json.dumps({d.id: Path(d.path).name for d in docs}, indent=1))
    return report, docs, occs, regions, decisions


def _override_doc(e, digests):
    """Document scope of a `_replace` entry, by a STABLE identity: `doc_sha256` (the source file's SHA-256, or a prefix
    of at least 12 hex digits) or `file` (the source file name). A bare positional `doc` ('document-02') follows file
    order and is refused unless it agrees with `doc_sha256`."""
    from .decisions import DecisionsRefused
    sha, fname, doc = e.get("doc_sha256"), e.get("file"), e.get("doc")
    if sha is None and fname is None:
        if doc is None:
            return None
        raise DecisionsRefused([{"id": f"_replace:{doc}", "reason": "POSITIONAL_DOC_SCOPE"}], 1)
    hits = set()
    if sha is not None:
        sha = str(sha).lower()
        if len(sha) < 12 or not re.fullmatch(r"[0-9a-f]+", sha):
            raise ValueError("INVALID_REPLACE_OVERRIDE")
        hits = {d for d, h in (digests or {}).items() if h.split("#")[0].startswith(sha)}
    if fname is not None:
        names = {d for d, n in e.get("_names", {}).items() if n == fname}
        hits = names if sha is None else hits & names
    if len(hits) != 1 or (doc is not None and doc not in hits):
        raise DecisionsRefused([{"id": f"_replace:{sha or fname}", "reason": "NO_MATCHING_PROPOSAL" if not hits else "AMBIGUOUS_DUPLICATE_PROPOSAL"
                                 if len(hits) > 1 else "DOC_SCOPE_MISMATCH", "doc": doc}], 1)
    return next(iter(hits))


def apply_replacement_overrides(docs, occs, entries, digests=None):
    """Reviewer-chosen replacement values (decisions `_replace`), applied after consistency planning so they win.
    Each entry: {"text": original, "value": new} plus an optional scope: one document by "doc_sha256" (source file
    SHA-256 or a >=12-digit prefix) or "file" (source file name) -- a positional "doc" id alone is refused --, "page" and "near"
    ([x, y] in page units) or "near_rel" ([fx, fy], fractions of the page) to pick the single closest occurrence on that
    page. Entries are applied in order (later entries win). Returns the number of occurrences changed."""
    pages = {(d.id, p.index): p for d in docs for p in d.pages}
    lines = {l.id: l for d in docs for p in d.pages for l in p.lines}
    def center(o):
        l = lines.get(o.line_id)
        if l is None or not l.chars:
            return None
        b = quad_bbox(span_quad(l.chars, o.start, o.end))
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    changed = 0
    for e in entries or []:
        if not isinstance(e, dict) or not isinstance(e.get("text"), str) or not isinstance(e.get("value"), str):
            raise ValueError("INVALID_REPLACE_OVERRIDE")
        value = e["value"].strip()
        if not value or len(value) > 200:
            raise ValueError("INVALID_REPLACE_OVERRIDE")
        if e.get("doc_sha256") is not None and digests is None:
            from .decisions import doc_digests
            digests = doc_digests(docs)
        scope = _override_doc(dict(e, _names={d.id: Path(d.path).name for d in docs}), digests)
        hits = [o for o in occs if o.text == e["text"] and (scope is None or o.doc == scope)
                and (e.get("page") is None or o.page == e["page"])]
        near = e.get("near")
        if near is None and e.get("near_rel") is not None and hits:
            pg = pages.get((hits[0].doc, hits[0].page))
            if pg is not None:
                near = (e["near_rel"][0] * pg.width, e["near_rel"][1] * pg.height)
        if near is not None and len(hits) > 1:
            scored = [(c, o) for o in hits for c in [center(o)] if c is not None]
            if scored:
                hits = [min(scored, key=lambda co: (co[0][0] - near[0]) ** 2 + (co[0][1] - near[1]) ** 2)[1]]
        for o in hits:
            o.replacement = value
            changed += 1
    return changed


def decisions_for_region(o, withheld):
    return any(r.id.startswith(f"{o.doc}:p{o.page}:") for r in withheld)


def _count(items):
    out = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


REPLACEABLE = ("face", "signature", "qr", "barcode")


def inpaint_label():
    from .native import inpaint_method
    return inpaint_method()


NESTABLE = ("PERSON", "GIVEN", "SURNAME", "EMAIL", "PHONE", "ACCOUNT", "ID", "DOB", "ADDRESS", "CARD", "SSN_LAST4")


def protect_inside_rejected(occs, decisions, digests):
    """A rejected span (e.g. a mis-bounded 'is Jane Doe' PERSON) must not keep an identity value that the case replaces
    elsewhere. Accepted values that occur whole-word inside a rejected span get their own occurrence (same entity and
    replacement); the rest of the span stays as the reviewer chose. Returns the new occurrences (already in `occs`)."""
    from dataclasses import replace as _dc_replace
    from . import decisions as DEC
    accepted = {}
    for o in occs:
        if decisions.get(o.id) == "accept" and o.kind in NESTABLE and len(o.text) >= 3:
            accepted.setdefault(o.text, o)
    rejected_texts = {o.text for o in occs if decisions.get(o.id) == "reject"}
    taken = {(o.line_id, o.start, o.end) for o in occs}
    new = []
    for o in [o for o in occs if decisions.get(o.id) == "reject"]:
        for text, twin in sorted(accepted.items(), key=lambda kv: -len(kv[0])):
            if text == o.text or text in rejected_texts:
                continue
            for m in re.finditer(r"(?<!\w)" + re.escape(text) + r"(?!\w)", o.text):
                s0, e0 = o.start + m.start(), o.start + m.end()
                if (o.line_id, s0, e0) in taken or any(n.line_id == o.line_id and n.start < e0 and s0 < n.end for n in new):
                    continue
                n = _dc_replace(twin, id="", doc=o.doc, page=o.page, line_id=o.line_id, start=s0, end=e0, source=o.source,
                                detectors=set(twin.detectors) | {"inside_rejected_span"}, review=list(o.review))
                for attr in ("hints", "meta"):
                    if hasattr(twin, attr):
                        setattr(n, attr, getattr(twin, attr))
                new.append(n); taken.add((o.line_id, s0, e0))
    DEC.assign_occurrence_ids(new, digests)
    occs.extend(new)
    return new


def likely_misses(docs, accepted, verification, handled=()):
    """Structural signs that a document still holds a personal value the detectors missed (no values listed):
    - NO_PERSON_WITH_ADDRESS: an address was replaced but no name in that document (statements, letters, labels);
    - LABELED_NAME_UNCHANGED: a person label ('Cardholder:', 'Patient name:', 'Dear') whose following name no detector
      proposed (names a reviewer kept, and places/businesses kept as facts, are decisions, not misses);
    - RESIDUAL_ORIGINAL: a detected value is still readable in the output;
    - PII_LIKE_TEXT: the output rescan found PII-shaped text that is not one of the replacements."""
    from .detect import LABELED_NAME, NAME_CUE, LABEL_ORG_HINT
    out = []
    names = {"PERSON", "GIVEN", "SURNAME"}
    by_doc = {}
    for o in accepted:
        by_doc.setdefault(o.doc, []).append(o)
    for d in docs:
        acc = by_doc.get(d.id, [])
        kinds = {o.kind for o in acc}
        if "ADDRESS" in kinds and not kinds & names:
            out.append({"doc": d.id, "code": "NO_PERSON_WITH_ADDRESS", "kind": "PERSON"})
        spans, known = {}, set()
        compact = lambda t: re.sub(r"[\W_]+", "", t).lower()
        known |= {compact(t) for t in handled if len(compact(t)) >= 4}
        for o in acc:
            if o.kind in names:
                spans.setdefault(o.line_id, []).append((o.start, o.end))
                known.add(compact(o.text))
        for p in d.pages:
            for l in p.lines:
                hit = False
                for rx in (LABELED_NAME, NAME_CUE):
                    for m in rx.finditer(l.text):
                        s_, e_ = m.span(1)
                        if LABEL_ORG_HINT.search(m.group(1)):
                            continue
                        # the same name replaced elsewhere in the document (e.g. a second OCR reading of the line): the
                        # residual check of the output pixels covers it
                        if any(k and k in compact(m.group(1)) for k in known):
                            continue
                        if not any(a < e_ and s_ < b for a, b in spans.get(l.id, [])):
                            hit = True
                if hit:
                    out.append({"doc": d.id, "page": p.index, "code": "LABELED_NAME_UNCHANGED", "kind": "PERSON"})
        v = verification.get(d.id) or {}
        if int(v.get("residual_original_values") or 0):
            out.append({"doc": d.id, "code": "RESIDUAL_ORIGINAL", "kind": None, "count": int(v["residual_original_values"])})
        if int(v.get("rescan_pii_like_not_replacements") or 0):
            out.append({"doc": d.id, "code": "PII_LIKE_TEXT", "kind": None, "count": int(v["rescan_pii_like_not_replacements"])})
    return out


def _squash(t):
    return re.sub(r"\s+", "", str(t or "")).casefold()


def reviewer_additions(docs, additions):
    """Reviewer-added spans (values the detectors missed). Each must be found verbatim in an extracted line; it is
    then propagated case-wide like any detector proposal. Shape: [{"text": str, "kind": str, "doc": optional id}]."""
    from .detect import Proposal
    out = []
    for a in additions:
        text, kind = a.get("text", ""), a.get("kind", "ID")
        if not text or not isinstance(text, str):
            raise ValueError("INVALID_ADDITION")
        if kind == "SSN" and re.fullmatch(r"\d{4}", text.strip()):
            kind = "SSN_LAST4"  # a last-4 must stay a last-4 (a full fake SSN in 'XXX-XX-4096' breaks the format)
        if kind == "PERSON" and len(text.split()) == 1:
            kind = "GIVEN"  # single-token person names are dropped by the PERSON shape rules; the reviewer asked for it
        hit = False
        for d in docs:
            if a.get("doc") and a["doc"] != d.id:
                continue
            for p in d.pages:
                for l in p.lines:
                    i = l.text.find(text)
                    while i >= 0:
                        out.append(Proposal(l.id, i, i + len(text), kind, text, "reviewer", 1.0)); hit = True
                        i = l.text.find(text, i + 1)
        if not hit:  # OCR spacing/case differs from what the reviewer typed: match ignoring case and spaces
            chars = [c for c in text if not c.isspace()]
            rx = re.compile(r"(?<!\w)" + r"\s*".join(map(re.escape, chars)) + r"(?!\w)", re.I) if len(chars) >= 3 else None
            for d in docs if rx else []:
                if a.get("doc") and a["doc"] != d.id:
                    continue
                for p in d.pages:
                    for l in p.lines:
                        for m in rx.finditer(l.text):
                            out.append(Proposal(l.id, m.start(), m.end(), kind, m.group(), "reviewer", 1.0)); hit = True
        if not hit:
            R.REJECTED.append({"detector": "reviewer", "kind": kind, "reason": "ADDITION_NOT_FOUND_IN_ANY_LINE"})
    return out


def _center(bb):
    return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)


def signers(doc, regions, accepted, cmap):
    """Fake signer name per signature region: the replacement of the nearest person mention on the same page
    (a signature belongs to a nearby printed name), so the new signature reads as the same fake identity."""
    from .model import quad_bbox, span_quad
    lines = {l.id: l for p in doc.pages for l in p.lines}
    out = {}
    for r in regions:
        if not r.id.startswith(doc.id + ":") or r.kind != "signature":
            continue
        page = int(r.id.split(":")[1][1:])
        rc = _center(quad_bbox(r.quad))
        best = None
        for o in accepted:
            if o.doc != doc.id or o.page != page or o.kind not in ("PERSON", "GIVEN", "SURNAME"):
                continue
            l = lines.get(o.line_id)
            if not l or not l.chars:
                continue
            oc = _center(quad_bbox(span_quad(l.chars, o.start, o.end)))
            dist = abs(oc[0] - rc[0]) * 0.5 + abs(oc[1] - rc[1])
            if best is None or dist < best[0]:
                best = (dist, o)
        if best and best[1].kind == "PERSON":
            out[r.id] = best[1].replacement
        elif best:
            out[r.id] = default_signer(doc, accepted, cmap)
    return out


def _trace_pages(chars):
    from .textual import pages_for
    return pages_for(chars)


def _jsonable(v):
    """numpy scalars/arrays in edit logs."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (set, tuple)):
        return list(v)
    return str(v)


def face_profiles(doc, accepted, cmap):
    """Per page: the mapped fake identity a portrait must fit (gender styling of the fake given name, rough age at the
    document's date from the DOB on the page and the latest other year printed there)."""
    import re as _re
    out = {}
    for p in doc.pages:
        mine = [o for o in accepted if o.doc == doc.id and o.page == p.index]
        given = next((o for o in mine if o.kind == "GIVEN"), None) or next((o for o in mine if o.kind == "PERSON"), None)
        g = None
        if given:
            first = given.replacement.split()[0].strip(",.").lower() if given.replacement else ""
            g = cmap.given_style.get(first)
        age = None
        dob = next((o for o in mine if o.kind == "DOB"), None)
        if dob:
            m = _re.search(r"(19|20)\d\d", dob.text) or _re.search(r"\b(\d\d)\s*$", dob.text)
            if m:
                by = int(m.group(0)) if len(m.group(0)) == 4 else 1900 + int(m.group(1))
                text = " ".join(l.text for l in p.lines)
                years = [int(y) for y in _re.findall(r"\b(?:19|20)\d\d\b", text) if by < int(y) <= 2026]
                ref = max(years) if years else 2026
                age = ref - by if 0 < ref - by < 110 else None
        out[f"p{p.index}"] = {"gender": g, "age": age, "identity": (given.replacement if given else f"p{p.index}")}
    return out


def default_signer(doc, accepted, cmap):
    given = next((o.replacement for o in accepted if o.doc == doc.id and o.kind == "GIVEN"), None)
    sur = next((o.replacement for o in accepted if o.doc == doc.id and o.kind == "SURNAME"), None)
    if given and sur:
        return f"{given.title()} {sur.title()}"
    per = next((o.replacement for o in accepted if o.doc == doc.id and o.kind == "PERSON"), None)
    return per.title() if per and per.isupper() else (per or cmap.person("Alex Morgan") or "A. Morgan")


def _inside_any(o, bbox, regs):
    from .model import quad_bbox
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for r in regs:
        if r.id.startswith(f"{o.doc}:p{o.page}:"):
            rb = quad_bbox(r.quad)
            if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                return True
    return False


def signatures_matching_names(docs, occs, regions):
    """Handwritten signatures that OCR read as a garbled name: a low-confidence raster line that fuzzily matches a
    person in this case is a signature of that person."""
    import difflib
    from .model import Region, quad_bbox, quad_from_rect
    names = set()
    given = [o.text for o in occs if o.kind == "GIVEN"]; sur = [o.text for o in occs if o.kind == "SURNAME"]
    names |= {o.text for o in occs if o.kind == "PERSON"}
    names |= {f"{g} {s}" for g in given for s in sur}
    names = {n.lower() for n in names if len(n) >= 5}
    out, dropped = [], []
    taken = {o.line_id for o in occs}
    for d in docs:
        for p in d.pages:
            if not p.raster:
                continue
            for l in p.lines:
                if not l.chars or l.conf >= 0.9 or l.id in taken or len(l.text) < 5:
                    continue
                t = l.text.lower()
                best = max((difflib.SequenceMatcher(None, t, n).ratio() for n in names), default=0)
                if best < 0.55:
                    continue
                from .native import _expand_quad
                q0 = (l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3])
                h = float(np.linalg.norm(np.subtract(q0[3], q0[0])))
                qx = [tuple(map(float, v)) for v in _expand_quad(q0, h * 0.35, h * 0.45)]
                bb = quad_bbox(qx)
                if any(r.id.startswith(d.id + f":p{p.index}:") and r.kind == "signature" and _ov(quad_bbox(r.quad), bb) > 0.3 for r in regions):
                    continue
                # the OCR'd name is stronger evidence than the label-proximity search: a label-based signature box on
                # this page that does not overlap the named handwriting was most likely mislocated -> keep it out
                for r in [r for r in regions if r.id.startswith(d.id + f":p{p.index}:") and r.kind == "signature"
                          and "printed label" in r.reason]:
                    regions.remove(r)
                    dropped.append(r.id)
                n = 1 + max([int(r.id.rsplit(":R", 1)[1]) for r in regions + out + [type("X", (), {"id": x})() for x in dropped]
                             if r.id.startswith(d.id + f":p{p.index}:")] or [-1])
                out.append(Region(f"{d.id}:p{p.index}:R{n}", "signature", tuple(qx),
                                  f"handwriting OCR'd as a garbled case name (similarity {best:.2f})", None, rotated=True))
    return out


def _ov(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
