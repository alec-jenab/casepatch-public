"""Recall of the regulated-category detectors on a labelled real-sample case (no rendering, no network).

    python -m casepatch.bench.categories_eval --input examples/real/case-real-05-health [--json out.json]

For every labelled value in labels.json["category_labels"], every extracted line containing it is one occurrence.
An occurrence is DETECTED when the union of spans proposed (replace) or flagged (kept for review) on that line
covers all of its alphanumeric characters; POLICY-CORRECT when it was handled the way the label expects
('replace' -> proposed for replacement, 'flag' -> flagged and kept, 'either' -> both count). Unlabelled category hits
are listed as candidate false positives (they include generic medical words in boilerplate).
Systems compared: the existing engine (rules + form fields), categories alone, and both. A final pass runs the
proposals through resolve.build_occurrences to show what survives normalisation (protected-fact and shape checks).
Extraction uses the engine's own extract_dir (PyMuPDF + RapidOCR)."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path


def _covered(value, line_text, spans):
    out = []
    i = line_text.find(value)
    while i >= 0:
        need = {k for k in range(i, i + len(value)) if line_text[k].isalnum()}
        have = set()
        for s, e in spans:
            have.update(range(s, e))
        out.append(need <= have)
        i = line_text.find(value, i + 1)
    return out


def pipeline_system(input_dir):
    """The shipped pipeline end to end (rules + form fields + categories + case-wide propagation + normalisation):
    accepted occurrences = replaced; report category_flags = kept and flagged. Local only, no verification pass."""
    import shutil
    import tempfile
    from ..docs.pipeline import run
    tmp = Path(tempfile.mkdtemp())
    try:
        rep, _docs, occs, _regions, dec = run(input_dir, tmp / "out", verify=False, review_dir=tmp / "rv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    replace = [o for o in occs if dec.get(o.id) == "accept"]
    return {"replace": replace, "flag": rep.get("category_flags", [])}


def evaluate(input_dir, docs=None, with_pipeline=False):
    from ..docs import resolve as R
    from ..docs.categories import CategoryDetector, install
    from ..docs.detect import RulesDetector, chunks_for
    from ..docs.extract import extract_dir
    from ..docs.forms import form_field_proposals
    input_dir = Path(input_dir)
    labels = json.loads((input_dir / "labels.json").read_text())
    t0 = time.perf_counter()
    docs = docs if docs is not None else extract_dir(input_dir)
    t_extract = time.perf_counter() - t0
    file_of = {d.id: Path(d.path).name for d in docs}
    lines = {l.id: (d, l) for d in docs for p in d.pages for l in p.lines}
    chunks = chunks_for(docs)
    t1 = time.perf_counter()
    rules = [p for c in chunks for p in RulesDetector().detect(c)] + form_field_proposals(docs)
    det = CategoryDetector().prime(chunks)
    cat = [p for c in chunks for p in det.detect(c)]
    t_detect = time.perf_counter() - t1
    flags = det.flags
    systems = {
        "existing_engine": {"replace": rules, "flag": []},
        "categories": {"replace": cat, "flag": flags},
        "engine_plus_categories": {"replace": rules + cat, "flag": flags},
    }
    if with_pipeline:
        systems["full_pipeline"] = pipeline_system(input_dir)

    def spans(items, lid):
        out = []
        for x in items:
            if isinstance(x, dict):
                if x["line_id"] == lid:
                    out.append((x["start"], x["end"]))
            elif x.line_id == lid:
                out.append((x.start, x.end))
        return out

    result = {"case": labels.get("case"), "extract_seconds": round(t_extract, 1), "detect_seconds": round(t_detect, 2), "systems": {}}
    for sname, sysd in systems.items():
        rows, per_cat = [], {}
        for fname, labs in labels["category_labels"].items():
            for lab in labs:
                for lid, (d, l) in lines.items():
                    if file_of[d.id] != fname or lab["value"] not in l.text:
                        continue
                    rep = _covered(lab["value"], l.text, spans(sysd["replace"], lid))
                    flg = _covered(lab["value"], l.text, spans(sysd["flag"], lid))
                    for r_, f_ in zip(rep, flg):
                        detected = r_ or f_
                        pol = lab["policy"]
                        correct = (pol == "either" and detected) or (pol == "replace" and r_) or (pol == "flag" and f_ and not r_)
                        rows.append({"file": fname, "line": lid.split(":", 1)[1], "value": lab["value"], "kind": lab["kind"],
                                     "category": lab["category"], "policy": pol, "detected": detected, "policy_correct": correct})
                        c = per_cat.setdefault(lab["kind"], {"occurrences": 0, "detected": 0, "policy_correct": 0})
                        c["occurrences"] += 1; c["detected"] += detected; c["policy_correct"] += correct
        n = len(rows)
        result["systems"][sname] = {"occurrences": n, "detected": sum(r["detected"] for r in rows),
                                    "policy_correct": sum(r["policy_correct"] for r in rows),
                                    "recall": round(sum(r["detected"] for r in rows) / max(n, 1), 3),
                                    "by_kind": per_cat, "misses": [r for r in rows if not r["detected"]]}
    # unlabelled category hits (candidate false positives)
    labelled = {(fname, lab["value"]) for fname, labs in labels["category_labels"].items() for lab in labs}
    fps = []
    for x in cat + flags:
        lid = x["line_id"] if isinstance(x, dict) else x.line_id
        s, e = (x["start"], x["end"]) if isinstance(x, dict) else (x.start, x.end)
        kind = x["kind"] if isinstance(x, dict) else x.kind
        d, l = lines[lid]
        txt = l.text[s:e]
        if not any(fn == file_of[d.id] and (txt in v or v in txt) and v in l.text for fn, v in labelled):
            fps.append({"file": file_of[d.id], "kind": kind, "text": txt, "handled": "flag" if isinstance(x, dict) else "replace"})
    result["unlabelled_category_hits"] = fps
    # what survives the engine's normalisation (resolve.build_occurrences) with and without install()
    R.REJECTED.clear()
    before = {(o.kind, o.text) for o in R.build_occurrences(docs, list(cat), propagate=False)}
    rej_before = len(R.REJECTED)
    install(R)
    R.REJECTED.clear()
    after = {(o.kind, o.text) for o in R.build_occurrences(docs, list(cat), propagate=False)}
    result["engine_normalisation"] = {"category_proposals": len(cat), "distinct_after_without_install": len(before),
                                      "rejected_without_install": rej_before, "distinct_after_with_install": len(after),
                                      "rejected_with_install": len(R.REJECTED)}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--json")
    ap.add_argument("--pipeline", action="store_true", help="also score the full shipped pipeline (adds one local run)")
    ap.add_argument("--docs-cache", help="pickle of extracted docs (dev speed-up; local only)")
    a = ap.parse_args(argv)
    docs = None
    if a.docs_cache and Path(a.docs_cache).exists():
        docs = pickle.loads(Path(a.docs_cache).read_bytes())
    elif a.docs_cache:
        from ..docs.extract import extract_dir
        docs = extract_dir(a.input)
        Path(a.docs_cache).write_bytes(pickle.dumps(docs))
    res = evaluate(a.input, docs, with_pipeline=a.pipeline)
    out = json.dumps(res, indent=1, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(out)
    for s, v in res["systems"].items():
        print(f"{s:24s} recall {v['detected']}/{v['occurrences']} = {v['recall']:.3f}   policy-correct {v['policy_correct']}/{v['occurrences']}")
    print("unlabelled category hits:", len(res["unlabelled_category_hits"]))
    print("normalisation:", res["engine_normalisation"])


if __name__ == "__main__":
    main()
