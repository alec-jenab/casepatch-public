"""Re-plan the intended replacements of a precomputed editor-comparison pack in context, with NO model calls.

The draft-card sample showed 1892 -> 6217 as a birth year. For every change whose stored replacement
fails the plausibility checker (docs/consistency.py), the pack gets the context-planned replacement instead:
  * the local candidate is swapped for the local editor's output for the new value (a $0 local bench run), with
    fresh badges, when that output is available;
  * hosted candidates made for the old value are kept for reference but marked stale / needs-regeneration and are
    not selectable (no Fal or other paid calls are made here).
Changes whose stored replacement is already plausible are left untouched (their hosted outputs stay valid).

    python -m casepatch.docs.pack_refresh --pack web/demo/cases/compare --bench path/to/bench-dir
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import consistency as CC


def _case_plan(case, pages_root, labels_root, cache):
    if case not in cache:
        pages = json.loads((Path(pages_root) / case / "index.json").read_text())
        labels = json.loads((Path(labels_root) / case / "labels.json").read_text())
        cache[case] = CC.plan_labelled_case(case, pages, labels)
    return cache[case]


def _bench_rows(bench_dir):
    """{editor_id: {tid: scored row}} from a scored edit-bench directory (editbench.json)."""
    from .editor_pick import _normalize_editor_id
    if not bench_dir or not (Path(bench_dir) / "editbench.json").exists():
        return {}
    eb = json.loads((Path(bench_dir) / "editbench.json").read_text())
    return {_normalize_editor_id(name): {r["tid"]: r for r in blk.get("per_target") or []}
            for name, blk in (eb.get("editors") or {}).items()}


def _regen(ch, eid, tid, new, rows, bench_dir, pack_dir, write, badge):
    """Swap one candidate for the bench output that editor made for `new` (local $0 run, or an approved hosted
    regeneration), with fresh badges. True if swapped; a stale candidate stays stale otherwise."""
    from .editor_pick import _FOLDER
    row, cand = (rows.get(eid) or {}).get(tid), ch["candidates"].get(eid)
    folder = _FOLDER.get(eid, eid.replace("/", "__"))
    src = Path(bench_dir) / folder / f"{tid}.png" if row else None
    if not (cand is not None and row and row.get("replacement") == new and src.exists() and not row.get("stale")):
        return False
    dest = Path(pack_dir) / "crops" / f"{tid}.{folder}.png"
    if write:
        shutil.copy(src, dest)
    b = badge(row)
    b["plausibility"] = row.get("plausibility") or []
    for k in ("stale", "needs_regeneration", "made_for"):
        cand.pop(k, None)
    cand.update(badges=b, ocr=row.get("ocr"), latency_s=row.get("latency_s"),
                cost_usd=0.0 if eid == "local" else row.get("cost_usd"), src=f"crops/{dest.name}", regenerated_for=new)
    return True


def _adopt(ch, tid, rows, bench_dir, pack_dir, write, badge):
    """Bring every candidate of a re-planned change up to date with the bench, where the bench has the output."""
    done = []
    for eid, cand in ch["candidates"].items():
        if (cand.get("stale") or cand.get("regenerated_for", ch["replacement"]) != ch["replacement"]
                or (eid == "local" and cand.get("regenerated_for") != ch["replacement"])):
            if _regen(ch, eid, tid, ch["replacement"], rows, bench_dir, pack_dir, write, badge):
                done.append(eid)
    return done


def _entry(ch, tid, regenerated, prior=None):
    e = dict(prior or {"id": tid, "kept": False})
    e.update(replacement=ch["replacement"], previous=ch["previous_replacement"],
             previous_issues=ch.get("previous_issues"),
             regenerated=sorted(set(e.get("regenerated") or []) | set(regenerated)),
             stale_hosted=sorted(x for x, c in ch["candidates"].items() if c.get("stale")))
    e.pop("local_regenerated", None)
    if "local" in e["regenerated"] or ch["candidates"].get("local", {}).get("regenerated_for") == ch["replacement"]:
        e["local_regenerated"] = True
    return e


def refresh(pack_dir, bench_dir=None, pages_root=None, labels_root=None, write=True):
    from .editor_pick import badge
    root = Path(__file__).resolve().parents[2]
    pack_dir = Path(pack_dir)
    pages_root = Path(pages_root or (Path.home() / ".cache" / "casepatch" / "modeleval" / "pages"))
    labels_root = Path(labels_root or (root / "examples" / "real"))
    man = json.loads((pack_dir / "pack.json").read_text())
    rows = _bench_rows(bench_dir)
    cache, report = {}, []
    prior = {r["id"]: r for r in (man.get("replanned") or {}).get("changes", []) if r.get("replacement")}
    for ch in man["changes"]:
        tid, case = ch["id"], ch["id"].rsplit("-", 1)[0]
        planned, g, _ = _case_plan(case, pages_root, labels_root, cache)
        ctx = g.contexts.get(ch.get("file"))
        kind = next((o.kind for o in g.occ.values() if o.doc == ch.get("file") and o.text == ch["value"]), None) or "ID"
        ch["kind"] = kind
        current = ch.get("previous_replacement") and ch["replacement"] or ch["replacement"]
        issues = CC.plausibility(kind, ch["value"], ch.get("previous_replacement") or current, ctx)
        from ..hosted.modeleval import pinned
        new = pinned(case, ch.get("file"), ch["value"], planned.get((ch.get("file"), ch["value"])), kind, ctx)
        if ch.get("previous_replacement"):  # refreshed before: follow a changed target (a demo pin), adopt new outputs
            if new and new != ch["replacement"]:
                ch["replacement"] = new
                for eid, cand in ch["candidates"].items():  # outputs made for the earlier planned value are stale too
                    if eid != "local" and not cand.get("stale") and cand.get("regenerated_for") not in (None, new):
                        cand.update(stale=True, needs_regeneration=True, made_for=cand["regenerated_for"])
                        cand["badges"] = dict(cand.get("badges") or {}, stale=True, made_for=cand["made_for"], selectable=False)
            ch["plausibility"] = CC.plausibility(kind, ch["value"], ch["replacement"], ctx)
            report.append(_entry(ch, tid, _adopt(ch, tid, rows, bench_dir, pack_dir, write, badge), prior.get(tid)))
            continue
        if not issues or not new or new == ch["replacement"]:
            ch["plausibility"] = issues
            report.append({"id": tid, "kept": True, "replacement": ch["replacement"]})
            continue
        old = ch["replacement"]
        ch.update(previous_replacement=old, previous_issues=issues, replacement=new,
                  plausibility=CC.plausibility(kind, ch["value"], new, ctx))
        for eid, cand in ch["candidates"].items():
            b = dict(cand.get("badges") or {})
            b.update(stale=True, made_for=old, selectable=False)
            cand.update(badges=b, stale=True, needs_regeneration=True, made_for=old)
        report.append(_entry(ch, tid, _adopt(ch, tid, rows, bench_dir, pack_dir, write, badge)))
    man["replanned"] = {"note": "Intended replacements re-planned in context (docs/consistency.py). Candidates are taken "
                                "from edit-bench outputs made for the new value; any still made for a replaced value "
                                "are stale and need regeneration. This step makes no model calls.",
                        "changes": report}
    if write:
        (pack_dir / "pack.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m casepatch.docs.pack_refresh")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    man = refresh(a.pack, a.bench, write=not a.dry_run)
    print(json.dumps(man["replanned"]["changes"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
