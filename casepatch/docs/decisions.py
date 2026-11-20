"""Review decisions bound to proposal content (decisions format 2).

Proposal ids are content-bound: an occurrence id is a hash of the source document's bytes, the line's position inside
that document, the span, the kind and the text, so it does not depend on the order of input files or on other
proposals (reviewer additions, new detectors). Region ids keep their `document-NN:pN:` prefix but end in a hash of the
document bytes, page, kind and box. A decision therefore can only ever land on the proposal it was made for.

On a rerun every decision is checked against the current proposal set before anything is rendered. Decisions that
point to no current proposal, whose recorded kind/text no longer match, that use positional ids from older versions,
or that carry an invalid action are refused as a group (DecisionsRefused, nothing is written). Proposals without a
decision get the default (replace).
"""
from __future__ import annotations

import hashlib
import re

FORMAT = "casepatch-decisions/2"
TEXT_ACTIONS = ("accept", "reject")
REGION_ACTIONS = ("keep", "replace", "erase", "withhold")
SPECIAL = ("_add", "_reject_text", "_accept_text", "_replace", "_region_kind", "_only_text")
META = ("_comment", "_format", "_proposals", "_help")
LEGACY_OCC = re.compile(r"^t\d{4,}$")
LEGACY_REGION = re.compile(r"^document-\d+:p\d+:R\d+$")


class DecisionsRefused(ValueError):
    """Decisions that cannot be applied safely. `problems` is a list of dicts (id, reason, kind?, doc?)."""

    def __init__(self, problems, total):
        self.problems, self.total = problems, total
        super().__init__(format_problems(problems, total))


def _h(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8", "surrogatepass")).hexdigest()


def doc_digests(docs) -> dict:
    """doc id -> content digest (sha256 of the file bytes, plus a copy ordinal for byte-identical files)."""
    from pathlib import Path
    raw = {d.id: hashlib.sha256(Path(d.path).read_bytes()).hexdigest() for d in docs}
    seen, out = {}, {}
    for did in sorted(raw):
        n = seen.get(raw[did], 0); seen[raw[did]] = n + 1
        out[did] = raw[did] if n == 0 else f"{raw[did]}#{n}"
    return out


def _local(line_id: str) -> str:
    return line_id.split(":", 1)[1] if ":" in line_id else line_id


def assign_occurrence_ids(occs, digests) -> None:
    """Replace positional ids with content-bound ids (stable across file order and added proposals)."""
    used = {}
    for o in occs:
        base = "o-" + _h("occ", digests[o.doc], o.page, _local(o.line_id), o.start, o.end, o.kind, o.text)[:16]
        n = used.get(base, 0); used[base] = n + 1
        o.id = base if n == 0 else f"{base}~{n}"


def assign_region_ids(regions, digests) -> None:
    from .model import quad_bbox
    used = {}
    for r in regions:
        doc, page = r.id.split(":")[0], r.id.split(":")[1]
        bb = [int(round(v)) for v in quad_bbox(r.quad)]
        base = f"{doc}:{page}:r-" + _h("region", digests[doc], page, r.kind, bb)[:12]
        n = used.get(base, 0); used[base] = n + 1
        r.id = base if n == 0 else f"{base}~{n}"


def proposals_digest(proposals_out) -> str:
    items = sorted([(o["id"], o["kind"]) for o in proposals_out["occurrences"]] + [(r["id"], r["kind"]) for r in proposals_out["regions"]])
    return "sha256:" + _h(*(f"{a}={b}" for a, b in items))


def parse(raw):
    """Split a decisions file into (entries, specials). Entries map id -> {"action", optional "kind", "text", "doc"}.
    Accepts format 2 (dict entries with recorded kind/text) and flat {id: action} maps of current ids."""
    if not isinstance(raw, dict):
        raise ValueError("INVALID_DECISIONS_FILE")
    fmt = raw.get("_format")
    if fmt is not None and fmt != FORMAT:
        raise ValueError("UNSUPPORTED_DECISIONS_FORMAT")
    specials = {k: raw[k] for k in SPECIAL if k in raw}
    entries = {}
    for k, v in raw.items():
        if k in SPECIAL or k in META:
            continue
        if isinstance(v, str):
            entries[k] = {"action": v}
        elif isinstance(v, dict) and isinstance(v.get("action"), str):
            entries[k] = dict(v)
        else:
            entries[k] = {"action": None}
    return entries, specials


def validate(entries, occ_by_id, region_by_id):
    """Check every decision against the current proposals. Returns {id: action}; raises DecisionsRefused."""
    problems, out = [], {}
    dup = {i.split("~")[0] for i in occ_by_id if "~" in i}
    for k, e in entries.items():
        act = e.get("action")
        if k in occ_by_id:
            o = occ_by_id[k]
            if act not in TEXT_ACTIONS:
                problems.append({"id": k, "reason": "INVALID_ACTION_FOR_TEXT", "kind": o["kind"], "doc": o["doc"]}); continue
            if k.split("~")[0] in dup:
                problems.append({"id": k, "reason": "AMBIGUOUS_DUPLICATE_PROPOSAL", "kind": o["kind"], "doc": o["doc"]}); continue
            # the recorded document id is informational: ids are bound to document bytes, not to file order
            if ("kind" in e and e["kind"] != o["kind"]) or ("text" in e and e["text"] is not None and e["text"] != o["text"]):
                problems.append({"id": k, "reason": "RECORDED_KIND_OR_TEXT_DIFFERS", "kind": e.get("kind"), "doc": o["doc"]}); continue
            out[k] = act
        elif k in region_by_id:
            r = region_by_id[k]
            if act not in REGION_ACTIONS:
                problems.append({"id": k, "reason": "INVALID_ACTION_FOR_REGION", "kind": r["kind"], "doc": r["doc"]}); continue
            if "kind" in e and e["kind"] != r["kind"]:
                problems.append({"id": k, "reason": "RECORDED_KIND_OR_TEXT_DIFFERS", "kind": e.get("kind"), "doc": r["doc"]}); continue
            out[k] = "erase" if act == "withhold" else act
        elif LEGACY_OCC.match(k) or LEGACY_REGION.match(k):
            problems.append({"id": k, "reason": "POSITIONAL_ID_FROM_OLDER_VERSION", "kind": e.get("kind"), "doc": e.get("doc")})
        else:
            problems.append({"id": k, "reason": "NO_MATCHING_PROPOSAL", "kind": e.get("kind"), "doc": e.get("doc")})
    if problems:
        raise DecisionsRefused(problems, len(entries))
    return out


HINT = {
    "POSITIONAL_ID_FROM_OLDER_VERSION": "positional id from casepatch <= 0.5.0; it can point at a different value now",
    "NO_MATCHING_PROPOSAL": "no current proposal has this id (the input, detectors or additions changed it)",
    "RECORDED_KIND_OR_TEXT_DIFFERS": "the decision was made for a different kind or text than the current proposal",
    "AMBIGUOUS_DUPLICATE_PROPOSAL": "the id matches more than one identical proposal",
    "POSITIONAL_DOC_SCOPE": "_replace scoped by a positional document id ('document-02' follows file order); add doc_sha256 or file",
    "DOC_SCOPE_MISMATCH": "_replace doc id and doc_sha256/file point at different documents",
    "INVALID_ACTION_FOR_TEXT": "text proposals take accept or reject",
    "INVALID_ACTION_FOR_REGION": "regions take replace, keep or erase",
}


def format_problems(problems, total) -> str:
    """Refusal message. Lists ids, reasons, kinds and documents, never original values."""
    lines = [f"DECISIONS_REFUSED: {len(problems)} of {total} decision(s) do not match the current proposals; "
             "nothing was applied and no output was written."]
    by = {}
    for p in problems:
        by.setdefault(p["reason"], []).append(p)
    for reason, ps in by.items():
        lines.append(f"  {reason} ({len(ps)}): {HINT.get(reason, '')}")
        for p in ps[:50]:
            extra = ", ".join(x for x in (p.get("kind"), p.get("doc")) if x)
            lines.append(f"    - {p['id']}" + (f" ({extra})" if extra else ""))
        if len(ps) > 50:
            lines.append(f"    ... {len(ps) - 50} more")
    lines.append("  Fix: run `casepatch docs` again without --decisions (into a new --out and --review-dir), redo the review in the new "
                 "review.html / decisions.json, then rerun with those decisions. Value-level entries (_reject_text, _add) stay valid across runs.")
    return "\n".join(lines)


def file_entries(proposals_out, actions) -> dict:
    """decisions.json body (format 2): every proposal with its default action and the kind/text it was made for."""
    body = {"_format": FORMAT, "_proposals": proposals_digest(proposals_out),
            "_help": "Change action to reject (text) or keep/erase (regions). Ids are bound to content; kind/text are checked on rerun."}
    for o in proposals_out["occurrences"]:
        e = {"action": actions[o["id"]], "kind": o["kind"], "doc": o["doc"]}
        if o["text"] is not None:
            e["text"] = o["text"]
        body[o["id"]] = e
    for r in proposals_out["regions"]:
        body[r["id"]] = {"action": actions[r["id"]], "kind": r["kind"], "doc": r["doc"]}
    return body


def actions_of(raw) -> dict:
    """Flat {id: action} view of a decisions file (either format)."""
    if not isinstance(raw, dict):
        return {}
    return {k: (v["action"] if isinstance(v, dict) else v) for k, v in raw.items()
            if not k.startswith("_") and (isinstance(v, str) or (isinstance(v, dict) and "action" in v))}
