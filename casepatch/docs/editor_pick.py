"""Per-change editor comparison: badges, selection, mask-only export, review log.

The local editor is always a candidate and is the default pick. Hosted editors (Ideogram 4.5, Nano Banana Pro,
Qwen inpaint) are off by default and appear only when the user has enabled bring-your-own-key editing with their
own key, or when a precomputed public-sample pack is loaded for local review / the static demo (no live calls).

A candidate that still reads as the original value is a leak: it cannot be selected without an explicit override.
Every shipped composite is pasted strictly inside the change mask.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Product-facing editor ids (slash form). Folder names in the fal bench dir use '__' instead of '/'.
LOCAL_EDITOR = "local"
HOSTED_EDITORS = (
    "ideogram/v4.5/edit",
    "fal-ai/nano-banana-pro/edit",
    "fal-ai/qwen-image-edit/inpaint",
)
DEFAULT_EDITOR = LOCAL_EDITOR

EDITORS = (
    {"id": LOCAL_EDITOR, "label": "Local editor", "kind": "local", "default": True},
    {"id": "ideogram/v4.5/edit", "label": "Ideogram 4.5", "kind": "hosted", "provider": "ideogram", "via": "fal"},
    {"id": "fal-ai/nano-banana-pro/edit", "label": "Nano Banana Pro", "kind": "hosted", "provider": "fal"},
    {"id": "fal-ai/qwen-image-edit/inpaint", "label": "Qwen Image Edit (inpaint)", "kind": "hosted", "provider": "fal"},
)
EDITOR_IDS = tuple(e["id"] for e in EDITORS)
_FOLDER = {e["id"]: e["id"].replace("/", "__") for e in EDITORS}
_FOLDER[LOCAL_EDITOR] = "local"
_ID_FROM_FOLDER = {v: k for k, v in _FOLDER.items()}

# Public-sample tids that may appear in the static demo or committed screenshots.
# FUNSD (case-real-01 fax) and SROIE (case-real-02 invoice) are research-only and must stay out.
PUBLIC_DEMO_TIDS = {
    "case-real-01-0": {
        "source": "CORD v2 receipt (test row 88), Park et al.",
        "license": "CC BY 4.0",
        "file": "02_receipt_photo.jpg",
        "page_label": "Receipt photo",
    },
    "case-real-03-0": {
        "source": "WWII draft card of John List (Wikimedia Commons)",
        "license": "Public domain (US government form)",
        "file": "02_draft_card_scan.png",
        "page_label": "Draft card",
    },
    "case-real-03-1": {
        "source": "WWII draft card of John List (Wikimedia Commons)",
        "license": "Public domain (US government form)",
        "file": "02_draft_card_scan.png",
        "page_label": "Draft card",
    },
    "case-real-03-2": {
        "source": "WWII draft card of John List (Wikimedia Commons)",
        "license": "Public domain (US government form)",
        "file": "02_draft_card_scan.png",
        "page_label": "Draft card",
    },
    "case-real-03-3": {
        "source": "WWII draft card of John List (Wikimedia Commons)",
        "license": "Public domain (US government form)",
        "file": "02_draft_card_scan.png",
        "page_label": "Draft card",
    },
    "case-real-04-0": {
        "source": "WWI draft registration card for Cole Porter (US National Archives)",
        "license": "Public domain (US federal record)",
        "file": "01_handwritten_draft_card.png",
        "page_label": "Handwritten draft card",
    },
    "case-real-04-1": {
        "source": "Russian diplomatic passport 2004 specimen (Wikimedia Commons)",
        "license": "Public domain (official specimen)",
        "file": "02_passport_specimen_cyrillic.jpg",
        "page_label": "Passport specimen",
    },
    "case-real-04-2": {
        "source": "Russian diplomatic passport 2004 specimen (Wikimedia Commons)",
        "license": "Public domain (official specimen)",
        "file": "02_passport_specimen_cyrillic.jpg",
        "page_label": "Passport specimen",
    },
    "case-real-04-3": {
        "source": "Russian diplomatic passport 2004 specimen (Wikimedia Commons)",
        "license": "Public domain (official specimen)",
        "file": "02_passport_specimen_cyrillic.jpg",
        "page_label": "Passport specimen",
    },
}

# Fal user keys look like id:secret. The earlier broken card was only the id (no colon).
_FAL_KEY_SHAPE = re.compile(r"^[A-Za-z0-9_-]{8,128}:[A-Za-z0-9_-]{16,256}$")
BYOK_FAL_ENV = "CASEPATCH_BYOK_FAL_KEY"
USER_FAL_ENV = "FAL_KEY"  # accepted only if it is not an evaluation-key fingerprint
KEYCHAIN_USER_FAL = "fal-api-key"

# Direct Ideogram API keys are not accepted yet: Fal is the only hosted-edit key type.
IDEOGRAM_DIRECT_STUB = True
IDEOGRAM_DIRECT_REASON = (
    "A direct Ideogram API key is not accepted yet. Use your own Fal API key (id:secret) for Ideogram 4.5 "
    "and the other hosted editors."
)


class LeakBlocked(Exception):
    """Raised when a leaking candidate is selected without an explicit override."""


class PickError(Exception):
    """Invalid editor, change, or key. Never includes a key value."""


# ------------------------------------------------------------------------------------------------ badges
def badge(row: dict) -> dict:
    """Automatic badges for one editor's output on one change.

    readback: 'exact' | 'fuzzy' | 'fail' against the intended replacement.
    leak: original text still readable (residual OCR, page leak, or residual ink / ghost). Hard fail.
    seam: mean |gradient| on the mask border, edited minus original (0 = invisible; may be negative).
    outside_mask: fraction of shipped (mask-composited) pixels outside the mask that changed. Should be 0.
    raw_outside_mask: how much the raw model image repainted outside the mask, before compositing (informational).
    """
    residual = bool(row.get("residual") or row.get("page_leak"))
    ghost = row.get("ghost")
    leak = residual or (ghost not in (None, False, 0, "0", ""))
    if row.get("exact"):
        readback = "exact"
    elif row.get("fuzzy"):
        readback = "fuzzy"
    else:
        readback = "fail"
    seam = row.get("seam")
    try:
        seam = None if seam is None else float(seam)
    except (TypeError, ValueError):
        seam = None
    outside = row.get("out_of_mask_frac")
    try:
        outside = 0.0 if outside is None else float(outside)
    except (TypeError, ValueError):
        outside = None
    raw = row.get("raw_out_of_mask_frac")
    try:
        raw = None if raw is None else float(raw)
    except (TypeError, ValueError):
        raw = None
    return {
        "readback": readback,
        "leak": leak,
        "seam": seam,
        "outside_mask": outside,
        "raw_outside_mask": raw,
        "selectable": not leak,
        "ocr": row.get("ocr"),
    }


def can_select(badges: dict, override: bool = False) -> bool:
    if badges.get("stale"):
        return False  # made for an older (implausible) replacement: regenerate, never ship
    if not badges.get("leak"):
        return True
    return bool(override)


# ------------------------------------------------------------------------------------------------ selection
@dataclass
class Picks:
    """Per-change editor choices. Local is the default. Leak picks require an explicit override."""

    change_ids: list
    default: str = DEFAULT_EDITOR
    chosen: dict = field(default_factory=dict)
    overrides: set = field(default_factory=set)  # {(change_id, editor_id), ...}

    def __post_init__(self):
        if self.default not in EDITOR_IDS:
            raise PickError(f"unknown editor: {self.default}")
        self.change_ids = list(self.change_ids)
        if not self.chosen:
            self.chosen = {cid: self.default for cid in self.change_ids}
        else:
            self.chosen = dict(self.chosen)
            for cid in self.change_ids:
                self.chosen.setdefault(cid, self.default)
        self.overrides = {(c, e) for c, e in self.overrides}

    def get(self, change_id) -> str:
        if change_id not in self.chosen:
            raise PickError(f"unknown change: {change_id}")
        return self.chosen[change_id]

    def select(self, change_id, editor, badges_for_change: dict, override: bool = False):
        if change_id not in self.chosen:
            raise PickError(f"unknown change: {change_id}")
        if editor not in EDITOR_IDS:
            raise PickError(f"unknown editor: {editor}")
        if editor not in badges_for_change:
            raise PickError(f"no candidate from {editor} for {change_id}")
        b = badges_for_change[editor]
        if b.get("leak") and not override:
            raise LeakBlocked("This candidate still shows the original text. It cannot be selected without an explicit override.")
        if b.get("leak") and override:
            self.overrides.add((change_id, editor))
        else:
            self.overrides.discard((change_id, editor))
        self.chosen[change_id] = editor
        return self.chosen[change_id]

    def select_all(self, editor, badges_by_change: dict, override_leaks: bool = False):
        """Apply one editor to every change that has a candidate. Leaks are skipped unless override_leaks is true."""
        if editor not in EDITOR_IDS:
            raise PickError(f"unknown editor: {editor}")
        applied, skipped = [], []
        for cid in self.change_ids:
            badges = badges_by_change.get(cid) or {}
            if editor not in badges:
                skipped.append({"id": cid, "reason": "no_candidate"})
                continue
            if badges[editor].get("leak") and not override_leaks:
                skipped.append({"id": cid, "reason": "leak"})
                continue
            self.select(cid, editor, badges, override=bool(badges[editor].get("leak") and override_leaks))
            applied.append(cid)
        return applied, skipped

    def snapshot(self):
        return {"chosen": dict(self.chosen), "overrides": sorted([f"{c}\t{e}" for c, e in self.overrides])}


# ------------------------------------------------------------------------------------------------ mask-only paste + export
def paste_mask_only(original, candidate, mask):
    """Return a copy of `original` with `candidate` pasted where `mask` is non-zero. Shapes must match."""
    import numpy as np
    if original.shape != candidate.shape:
        raise PickError("candidate and original crops must have the same shape (resize before pasting)")
    if mask.shape[:2] != original.shape[:2]:
        raise PickError("mask must match the crop height and width")
    out = np.array(original, copy=True)
    m = np.asarray(mask) > 0
    out[m] = np.asarray(candidate)[m]
    return out


def compose_page(page, changes, picks: Picks, crops: dict, masks: dict):
    """Paste each chosen crop into `page` (HxWx3), strictly inside that change's mask.

    `crops` is {change_id: {editor_id: array}}; `masks` is {change_id: mask array}.
    Each change is a dict with `id` and `crop` [x0, y0, x1, y1] in page pixels.
    """
    import numpy as np
    out = np.array(page, copy=True)
    for ch in changes:
        cid = ch["id"]
        editor = picks.get(cid)
        crop = (crops.get(cid) or {}).get(editor)
        mask = masks.get(cid)
        if crop is None or mask is None:
            raise PickError(f"missing crop or mask for {cid} / {editor}")
        x0, y0, x1, y1 = (int(v) for v in ch["crop"])
        region = out[y0:y1, x0:x1]
        if region.shape[:2] != crop.shape[:2]:
            raise PickError(f"crop size does not match the box for {cid}")
        out[y0:y1, x0:x1] = paste_mask_only(region, crop, mask)
    return out


def review_log(picks: Picks, badges_by_change: dict, *, include_values=False, changes=None):
    """Records which editor was chosen per change. Source values are omitted unless include_values is true
    (they must never go in the exported ZIP)."""
    rows = []
    counts = {}
    for cid in picks.change_ids:
        editor = picks.get(cid)
        b = (badges_by_change.get(cid) or {}).get(editor) or {}
        row = {
            "id": cid,
            "editor": editor,
            "readback": b.get("readback"),
            "leak": bool(b.get("leak")),
            "leak_override": (cid, editor) in picks.overrides,
            "seam": b.get("seam"),
            "outside_mask": b.get("outside_mask"),
        }
        if include_values and changes:
            ch = next((c for c in changes if c["id"] == cid), None)
            if ch:
                row["value"] = ch.get("value")
                row["replacement"] = ch.get("replacement")
        rows.append(row)
        counts[editor] = counts.get(editor, 0) + 1
    return {"editors_chosen": counts, "default": picks.default, "changes": rows}


def export_applies(picks: Picks, badges_by_change: dict):
    """True if every chosen candidate is allowed to ship (no un-overridden leak)."""
    for cid in picks.change_ids:
        editor = picks.get(cid)
        b = (badges_by_change.get(cid) or {}).get(editor) or {}
        if b.get("leak") and (cid, editor) not in picks.overrides:
            return False
    return True


# ------------------------------------------------------------------------------------------------ BYOK (hosted editors)
def valid_fal_key_shape(key) -> bool:
    return isinstance(key, str) and bool(_FAL_KEY_SHAPE.fullmatch(key.strip()))


def check_hosted_edit_key(key, env=None):
    """Accept a user Fal key; refuse evaluation-key fingerprints. Never logs or returns the key."""
    from .byok import ByokRefused, is_eval_key
    if not valid_fal_key_shape(key):
        raise ByokRefused("That does not look like a Fal API key (it should be id:secret, with a colon).")
    if is_eval_key(key, env):
        raise ByokRefused("This key is registered as a CasePatch evaluation key, which may only be used on public "
                          "samples. Use your own Fal key.")
    return key


def resolve_hosted_edit_key(env=None, user_key=None):
    """(key, source). Order: a key typed in Studio, CASEPATCH_BYOK_FAL_KEY, then FAL_KEY if it is not an eval key."""
    from .byok import ByokRefused
    env = env if env is not None else __import__("os").environ
    if user_key:
        return check_hosted_edit_key(user_key, env), "studio-settings"
    if env.get(BYOK_FAL_ENV):
        return check_hosted_edit_key(env[BYOK_FAL_ENV], env), "environment"
    if env.get(USER_FAL_ENV):
        return check_hosted_edit_key(env[USER_FAL_ENV], env), "environment"
    raise ByokRefused("No Fal API key of your own is set. Enter your key in settings, or set CASEPATCH_BYOK_FAL_KEY.")


def hosted_editors_available(env=None, user_key=None, allowed=False):
    """Off unless the app allows hosted editing AND a non-eval user key is present."""
    if not allowed:
        return {"allowed": False, "available": False, "reason": "Hosted editors are off. Restart with --allow-hosted-edit and add your own Fal key.",
                "editors": [e for e in EDITORS if e["kind"] == "hosted"]}
    try:
        _, source = resolve_hosted_edit_key(env, user_key)
        return {"allowed": True, "available": True, "key_source": source, "reason": None,
                "editors": [e for e in EDITORS if e["kind"] == "hosted"]}
    except Exception as e:
        return {"allowed": True, "available": False, "reason": str(e),
                "editors": [e for e in EDITORS if e["kind"] == "hosted"]}


def visible_editors(*, byok=False, precomputed=False):
    """Local is always visible. Hosted editors require BYOK or a precomputed pack (dev / static demo, no live calls)."""
    out = [e for e in EDITORS if e["kind"] == "local"]
    if byok or precomputed:
        out.extend(e for e in EDITORS if e["kind"] == "hosted")
    return out


def check_ideogram_direct_key(key, env=None):
    """Stub: a direct Ideogram key is recognised as the wrong type and refused. Never logs the key."""
    from .byok import ByokRefused
    raise ByokRefused(IDEOGRAM_DIRECT_REASON)


# ------------------------------------------------------------------------------------------------ occurrence matching
def norm_text(s):
    if s is None:
        return ""
    return " ".join(str(s).split()).casefold()


def _occ_page(o):
    return o.get("page") if o.get("page") is not None else o.get("page_index")


def _change_page(ch):
    return ch.get("page") if ch.get("page") is not None else ch.get("page_index")


def match_changes_to_occurrences(occurrences, changes):
    """Map occurrence id -> pack/live change. Each change is used at most once.

    Prefer an exact (normalised) text match. When the same original text appears more than once, prefer a
    change whose page (and file, when present) agrees with the occurrence; otherwise the first unused match.
    Occurrences with no text (e.g. card security data that is never stored) are not matched.
    """
    unused = list(changes)
    out = {}
    for occ in occurrences:
        oid = occ.get("id")
        if not oid:
            continue
        want = norm_text(occ.get("text") or occ.get("value"))
        if not want:
            continue
        page, fn = _occ_page(occ), occ.get("file") or occ.get("doc")
        candidates = [c for c in unused if norm_text(c.get("value") or c.get("text")) == want]
        if not candidates:
            continue
        def score(ch):
            s = 0
            if _change_page(ch) is not None and page is not None and int(_change_page(ch)) == int(page):
                s += 2
            ch_file = ch.get("file") or ch.get("doc") or (ch.get("page_id") or "")
            if fn and ch_file and str(fn) in str(ch_file):
                s += 1
            return s
        best = sorted(candidates, key=score, reverse=True)[0]
        unused.remove(best)
        out[oid] = best
    return out


def change_from_occurrence(occ, pack_change=None, *, editors=None):
    """A compare-change record keyed by the live occurrence. bbox/crop come from the occurrence.

    Hosted candidates are copied from a matched pack_change when present. The local editor is always listed
    (the live after-crop is filled in later by Studio).
    """
    editors = editors or [e for e in EDITORS if e["kind"] == "local"]
    bbox = occ.get("bbox") or occ.get("crop")
    rec = {
        "id": occ["id"],
        "occurrence_id": occ["id"],
        "doc": occ.get("doc"),
        "page": _occ_page(occ),
        "file": occ.get("file"),
        "kind": occ.get("kind"),
        "value": occ.get("text") if occ.get("text") is not None else occ.get("value"),
        "replacement": occ.get("replacement"),
        "crop": list(bbox) if bbox else None,
        "matched_pack": pack_change["id"] if pack_change else None,
        "public": bool(pack_change and pack_change.get("public")),
        "attribution": (pack_change or {}).get("attribution"),
        "candidates": {},
    }
    # always a local slot (src filled by the case previews)
    rec["candidates"][LOCAL_EDITOR] = {
        "badges": {"readback": "exact", "leak": False, "seam": None, "outside_mask": None,
                   "raw_outside_mask": None, "selectable": True, "ocr": None, "source": "local-engine"},
        "src": None, "cost_usd": 0, "latency_s": None,
    }
    if pack_change:
        for eid, cand in (pack_change.get("candidates") or {}).items():
            if eid == LOCAL_EDITOR:
                # keep live local as the default slot; copy pack badges only as a hint when no live after exists
                rec["candidates"][LOCAL_EDITOR] = dict(cand)
                rec["candidates"][LOCAL_EDITOR].setdefault("badges", {})
                rec["candidates"][LOCAL_EDITOR]["badges"] = dict(rec["candidates"][LOCAL_EDITOR]["badges"])
                rec["candidates"][LOCAL_EDITOR]["badges"]["source"] = rec["candidates"][LOCAL_EDITOR]["badges"].get("source") or "precomputed-local"
                continue
            rec["candidates"][eid] = dict(cand)
    rec["page_id"] = occ.get("page_id") or (
        f"{occ.get('doc')}:p{rec['page']}" if occ.get("doc") is not None and rec["page"] is not None else None
    )
    return rec


def compare_model_for_occurrences(occurrences, *, pack=None, byok=False, include_kept=False, replaced=None):
    """Build the picker model for a live case. Local is always present and is the default pick.

    `replaced` is an optional callable text -> bool (False = the reviewer kept this value; those are omitted
    unless include_kept is true).
    """
    changes_in = (pack or {}).get("changes") or []
    matched = match_changes_to_occurrences(occurrences, changes_in) if changes_in else {}
    editors = visible_editors(byok=byok, precomputed=bool(matched))
    # if any occurrence matched a pack change that has hosted candidates, show hosted editors
    if any((matched.get(o["id"]) or {}).get("candidates") for o in occurrences if o.get("id")):
        editors = visible_editors(byok=byok, precomputed=True)
    out = []
    for occ in occurrences:
        if not occ.get("id"):
            continue
        if occ.get("text") is None and occ.get("value") is None:
            continue  # not stored (e.g. PCI SAD): no picker row
        if replaced is not None and not include_kept and not replaced(occ.get("text")):
            continue
        if occ.get("bbox") is None and occ.get("crop") is None and occ.get("id") not in matched:
            continue  # no crop to show (e.g. a JSON-trace field) and no pack match
        out.append(change_from_occurrence(occ, matched.get(occ["id"]), editors=editors))
    pages = (pack or {}).get("pages") or []
    return {
        "editors": editors,
        "changes": out,
        "pages": pages,
        "precomputed": bool(matched),
        "live": True,
        "note": "Local editor is the default. Hosted candidates appear after a precomputed match or a BYOK run.",
    }


# ------------------------------------------------------------------------------------------------ precomputed pack
def _normalize_editor_id(name: str) -> str:
    if name in EDITOR_IDS:
        return name
    return _ID_FROM_FOLDER.get(name, name.replace("__", "/"))


def badges_from_results(results: dict, editor_ids=EDITOR_IDS):
    """{tid: {editor_id: badge}} from a 2026-11-12-fal-*.json document."""
    editors = results.get("editors") or {}
    out = {}
    for raw_name, block in editors.items():
        eid = _normalize_editor_id(raw_name)
        if eid not in editor_ids:
            continue
        for row in block.get("per_target") or []:
            tid = row.get("tid")
            if not tid:
                continue
            out.setdefault(tid, {})[eid] = badge(row)
            out[tid][eid]["_row"] = row
    return out


def load_results(path: Path):
    return json.loads(Path(path).read_text())


def change_records(results: dict, tids=None, public_only=False):
    """One record per target, with per-editor badges and the intended replacement (from the results file)."""
    badges = badges_from_results(results)
    # use local rows as the canonical target list (every full run includes local)
    local_block = None
    for name, block in (results.get("editors") or {}).items():
        if _normalize_editor_id(name) == LOCAL_EDITOR:
            local_block = block
            break
    if not local_block:
        raise PickError("results file has no local editor rows")
    want = set(tids) if tids is not None else None
    changes = []
    for row in local_block.get("per_target") or []:
        tid = row["tid"]
        if want is not None and tid not in want:
            continue
        if public_only and tid not in PUBLIC_DEMO_TIDS:
            continue
        rec = {
            "id": tid,
            "value": row.get("value"),
            "replacement": row.get("replacement"),
            "public": tid in PUBLIC_DEMO_TIDS,
            "attribution": PUBLIC_DEMO_TIDS.get(tid),
            "candidates": {},
        }
        for eid, b in (badges.get(tid) or {}).items():
            row_e = b.get("_row") or {}
            rec["candidates"][eid] = {
                "badges": {k: v for k, v in b.items() if k != "_row"},
                "cost_usd": row_e.get("cost_usd"),
                "latency_s": row_e.get("latency_s"),
                "ocr": row_e.get("ocr"),
            }
        changes.append(rec)
    return changes


def default_image_dir():
    # Outside the repo: research crops live here; the public pack copies only license-safe ones.
    import os
    return Path(os.environ.get("CASEPATCH_FAL_BENCH_DIR") or (Path.home() / ".cache" / "casepatch" / "bench-fal"))


def default_results_path():
    root = Path(__file__).resolve().parents[2]
    return root / "bench" / "models" / "results" / "2026-11-12-fal-full-editbench.json"


def image_paths(image_dir: Path, tid: str, editor_id: str):
    folder = Path(image_dir) / _FOLDER[editor_id]
    return {
        "composite": folder / f"{tid}.png",
        "raw": folder / f"{tid}.raw.png",
        "model": folder / f"{tid}.model.png",
    }


def crop_targets_from_pages(pages, labels, case, n=4):
    """Reproduce the eval crop-box rule so we can recover the original crop and mask without a paid call.

    Replacements are not recomputed here (they already live in the results file).
    """
    import math
    occs = labels.get("occurrences") or {}
    out = []
    for p in pages:
        if not p.get("raster"):
            continue
        values = occs.get(p["file"], [])
        texts = []
        for v in values:
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, dict):
                texts.append(v.get("text") or v.get("value") or "")
        for v in texts:
            if len(v) < 3:
                continue
            for l in p.get("lines") or []:
                i = l["text"].find(v)
                if i < 0 or len(l.get("chars") or []) != len(l["text"]):
                    continue
                cb = l["chars"][i:i + len(v)]
                mx0, my0 = min(b[0] for b in cb), min(b[1] for b in cb)
                mx1, my1 = max(b[2] for b in cb), max(b[3] for b in cb)
                hgt = max(8.0, my1 - my0)
                x0, y0 = max(0, int(mx0 - 2 * hgt)), max(0, int(my0 - 1.5 * hgt))
                x1, y1 = min(p["w"], int(mx1 + 2 * hgt)), min(p["h"], int(my1 + 1.5 * hgt))
                out.append({
                    "case": case, "file": p["file"], "page": p["page"], "png": p["png"], "value": v,
                    "crop": [x0, y0, x1, y1],
                    "mask": [int(mx0) - x0, int(my0) - y0, int(math.ceil(mx1)) - x0, int(math.ceil(my1)) - y0],
                })
                break
            if len(out) >= n:
                return out
    return out


def original_crop_and_mask(target):
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(target["png"]).convert("RGB"))
    x0, y0, x1, y1 = target["crop"]
    crop = im[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    a, b, c, d = target["mask"]
    mask[max(0, b - 1):d + 1, max(0, a - 1):c + 1] = 1
    return crop, mask


def build_pack(out_dir: Path, *, results_path=None, image_dir=None, pages_root=None, labels_root=None,
               public_only=True, tids=None):
    """Build a self-contained comparison pack (JSON + PNG crops) from precomputed fal outputs. No network, no paid calls.

    public_only=True copies only license-safe public samples (no FUNSD / SROIE).
    """
    import numpy as np
    from PIL import Image

    results_path = Path(results_path or default_results_path())
    image_dir = Path(image_dir or default_image_dir())
    root = Path(__file__).resolve().parents[2]
    labels_root = Path(labels_root or (root / "examples" / "real"))
    pages_root = Path(pages_root or (Path.home() / ".cache" / "casepatch" / "modeleval" / "pages"))
    results = load_results(results_path)
    changes = change_records(results, tids=tids, public_only=public_only)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)

    # recover crop boxes per case from the local page cache (already prepared; $0)
    targets_by_tid = {}
    cases = sorted({ch["id"].rsplit("-", 1)[0] for ch in changes})
    for case in cases:
        idx = pages_root / case / "index.json"
        lab = labels_root / case / "labels.json"
        if not idx.exists() or not lab.exists():
            continue
        pages = json.loads(idx.read_text())
        labels = json.loads(lab.read_text())
        for i, t in enumerate(crop_targets_from_pages(pages, labels, case, n=4)):
            targets_by_tid[f"{case}-{i}"] = t

    manifest_changes = []
    pages_meta = {}
    for ch in changes:
        tid = ch["id"]
        t = targets_by_tid.get(tid)
        if t:
            orig, mask = original_crop_and_mask(t)
            Image.fromarray(orig).save(out_dir / "crops" / f"{tid}.original.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "crops" / f"{tid}.mask.png")
            ch = dict(ch)
            ch["crop"] = t["crop"]
            ch["mask_box"] = t["mask"]
            ch["file"] = t["file"]
            ch["page"] = t["page"]
            ch["page_png"] = t["png"]
            ch["page_size"] = None
            try:
                im = Image.open(t["png"])
                ch["page_size"] = [im.size[0], im.size[1]]
            except OSError:
                pass
            page_key = f"{ch.get('attribution', {}).get('page_label') or t['file']}"
            pages_meta.setdefault(t["png"], {
                "id": f"{tid.rsplit('-', 1)[0]}:{t['file']}",
                "file": t["file"], "label": (ch.get("attribution") or {}).get("page_label") or t["file"],
                "credit": "{source}. {license}.".format(**(ch.get("attribution") or {"source": t["file"], "license": ""})),
                "src_png": t["png"], "w": (ch["page_size"] or [0, 0])[0], "h": (ch["page_size"] or [0, 0])[1],
            })
        for eid in list(ch["candidates"]):
            src = image_paths(image_dir, tid, eid)["composite"]
            if src.exists():
                dest = out_dir / "crops" / f"{tid}.{_FOLDER[eid]}.png"
                dest.write_bytes(src.read_bytes())
                ch["candidates"][eid]["src"] = f"crops/{dest.name}"
        if t:
            ch["original_src"] = f"crops/{tid}.original.png"
            ch["mask_src"] = f"crops/{tid}.mask.png"
            ch["page_id"] = f"{tid.rsplit('-', 1)[0]}:{t['file']}"
            ch.pop("page_png", None)
        manifest_changes.append(ch)

    (out_dir / "pages").mkdir(exist_ok=True)
    pages_out = []
    for meta in pages_meta.values():
        src = Path(meta["src_png"])
        page_id = meta["id"]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in page_id)
        safe = re.sub(r"\.(jpg|jpeg|png|pdf|gif|webp)$", "", safe, flags=re.I)
        dest = out_dir / "pages" / f"{safe}.jpg"
        try:
            im = Image.open(src).convert("RGB")
            w, h = im.size
            max_side = 1200
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                # scale crop boxes later via scale factors stored on the page
                meta["scale"] = scale
                meta["w"], meta["h"] = im.size
            else:
                meta["scale"] = 1.0
            im.save(dest, "JPEG", quality=85)
            meta["src"] = f"pages/{dest.name}"
        except OSError:
            meta["src"] = None
            meta["scale"] = 1.0
        meta.pop("src_png", None)
        pages_out.append(meta)
    # scale crop boxes to the (possibly downscaled) page image
    page_scale = {p["id"]: p.get("scale") or 1.0 for p in pages_out}
    for ch in manifest_changes:
        s = page_scale.get(ch.get("page_id") or "", 1.0)
        if ch.get("crop") and s != 1.0:
            ch["crop_original"] = list(ch["crop"])
            ch["crop"] = [int(round(v * s)) for v in ch["crop"]]
            if ch.get("page_size"):
                ch["page_size"] = [int(round(ch["page_size"][0] * s)), int(round(ch["page_size"][1] * s))]

    manifest = {
        "generated_from": str(results_path.name),
        "note": "Precomputed public-sample editor outputs. No live model calls. Hosted images were produced earlier on public samples only.",
        "editors": [e for e in EDITORS],
        "changes": manifest_changes,
        "pages": pages_out,
        "public_only": public_only,
    }
    (out_dir / "pack.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def load_pack(pack_dir: Path):
    pack_dir = Path(pack_dir)
    man = json.loads((pack_dir / "pack.json").read_text())
    man["_dir"] = pack_dir
    return man


def pack_badges(pack: dict):
    return {ch["id"]: {eid: cand["badges"] for eid, cand in ch["candidates"].items()} for ch in pack["changes"]}


def pack_picks(pack: dict) -> Picks:
    return Picks([ch["id"] for ch in pack["changes"]])


def _open_rgb(path):
    from PIL import Image
    import numpy as np
    return np.asarray(Image.open(path).convert("RGB"))


def _open_mask(path):
    from PIL import Image
    import numpy as np
    im = np.asarray(Image.open(path).convert("L"))
    return (im > 0).astype(np.uint8)


def compose_pack_page(pack, page_id, picks: Picks):
    """Compose the after-image of one pack page from the chosen crops. Mask-only paste."""
    import numpy as np
    from PIL import Image
    pack_dir = Path(pack["_dir"])
    page = next((p for p in pack["pages"] if p["id"] == page_id), None)
    if not page or not page.get("src"):
        raise PickError(f"no page image for {page_id}")
    full = _open_rgb(pack_dir / page["src"])
    changes, crops, masks = [], {}, {}
    for ch in pack["changes"]:
        if ch.get("page_id") != page_id or not ch.get("crop") or not ch.get("original_src"):
            continue
        changes.append(ch)
        masks[ch["id"]] = _open_mask(pack_dir / ch["mask_src"])
        crops[ch["id"]] = {}
        # Use the original crop size; candidate PNGs are that size. If the page was
        # downscaled, scale the candidate/mask to the (scaled) crop box.
        x0, y0, x1, y1 = (int(v) for v in ch["crop"])
        tw, th = x1 - x0, y1 - y0
        for eid, cand in ch["candidates"].items():
            src = cand.get("src")
            if not src:
                continue
            arr = _open_rgb(pack_dir / src)
            if arr.shape[0] != th or arr.shape[1] != tw:
                arr = np.asarray(Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS))
            crops[ch["id"]][eid] = arr
        m = masks[ch["id"]]
        if m.shape[0] != th or m.shape[1] != tw:
            masks[ch["id"]] = np.asarray(Image.fromarray(m * 255).resize((tw, th), Image.Resampling.NEAREST)) > 0
            masks[ch["id"]] = masks[ch["id"]].astype(np.uint8)
    return compose_page(full, changes, picks, crops, masks)


def pack_for_client(pack, image_prefix=""):
    """JSON-safe view of a pack: relative image paths, no filesystem locations, no OCR dumps of source values
    beyond the already-public replacement strings (needed so the reviewer can see the intended text)."""
    editors = pack.get("editors") or list(EDITORS)
    pages = []
    for p in pack.get("pages") or []:
        pages.append({k: p[k] for k in ("id", "file", "label", "credit", "src", "w", "h") if k in p})
        if image_prefix and pages[-1].get("src"):
            pages[-1]["src"] = image_prefix + pages[-1]["src"]
    changes = []
    for ch in pack.get("changes") or []:
        rec = {k: ch[k] for k in ("id", "value", "replacement", "public", "attribution", "crop", "file", "page",
                                  "page_id", "page_size", "original_src", "mask_src", "kind", "previous_replacement") if k in ch}
        # context-check results in plain words (consistency.explain); codes stay in pack.json
        from .consistency import explain
        for src_k, dst_k in (("plausibility", "context_issues"), ("previous_issues", "previous_issues")):
            if ch.get(src_k):
                rec[dst_k] = [explain(c) for c in ch[src_k]]
        rec["candidates"] = {}
        for eid, cand in ch.get("candidates", {}).items():
            rec["candidates"][eid] = {
                "badges": {k: v for k, v in cand["badges"].items() if k != "plausibility"},
                "cost_usd": cand.get("cost_usd"),
                "latency_s": cand.get("latency_s"),
                "src": ((image_prefix + cand["src"]) if (image_prefix and cand.get("src")) else cand.get("src")),
            }
        if image_prefix:
            if rec.get("original_src"):
                rec["original_src"] = image_prefix + rec["original_src"]
            if rec.get("mask_src"):
                rec["mask_src"] = image_prefix + rec["mask_src"]
        changes.append(rec)
    return {
        "note": pack.get("note"),
        "generated_from": pack.get("generated_from"),
        "public_only": pack.get("public_only", True),
        "precomputed": True,
        "editors": editors,
        "pages": pages,
        "changes": changes,
    }
