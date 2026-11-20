"""Agent loop: one model session per document (documents in parallel), CasePatch tools, deterministic verifier after
`finish`, failures fed back for up to N fix rounds, escalation to a stronger model only when the verifier still fails,
hard per-case budget. Two key modes, never mixed:

  product (default)  BYOK: the customer's own Anthropic key (casepatch.docs.byok rules: off by default, refuses keys
                     registered as CasePatch evaluation keys, in-memory per-case budget, calls go straight from the
                     customer's machine to the provider, no CasePatch server involved).
  eval               bench only: CasePatch eval key from the environment, public samples under examples/real only,
                     every call reserved in the shared spend ledger first (experiment cap + pool cap).

    python -m casepatch.agent INPUT_DIR --out OUT_DIR [--budget 3] [--model claude-sonnet-5-5]   (needs --enable-agent)
"""
from __future__ import annotations

import copy, json, os, re, threading, time, urllib.error, urllib.request

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..tls import ssl_context
from .tools import CaseMap, DocState, IMG_EXT, _bbox_of_chars, find_span
from . import verify as V

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL, ESCALATE_MODEL = "claude-sonnet-5-5", "claude-opus-5-5"
ANTHROPIC_VERSION = "2026-10-21"
EVAL_LEDGER = Path(os.environ.get("CASEPATCH_SPEND_LEDGER", str(Path.home() / ".casepatch" / "eval-2026-11-11.sqlite")))
EVAL_EXPERIMENT_PREFIX, EVAL_EXPERIMENT_CAP, EVAL_POOL_CAP = "cpagent:", 30.0, 200.0
KEEP_IMAGES = 3
DOC_EXT = IMG_EXT | {".pdf"}

FONTS = ("typewriter: Courier Prime, Special Elite, Cutive Mono | mono: Liberation Mono, Cousine, Nimbus Mono PS | "
         "serif: Liberation Serif, Tinos, PT Serif, Libre Baskerville, Nimbus Roman, Merriweather | "
         "sans: Liberation Sans, Arimo, Roboto, Open Sans, Noto Sans, Nimbus Sans, PT Sans | narrow: Roboto Condensed, Archivo Narrow, Oswald | "
         "handwriting: Caveat, Kalam, Patrick Hand, Indie Flower, Reenie Beanie, Shadows Into Light, Nothing You Could Do | "
         "signatures: Homemade Apple, Mrs Saint Delafield, Herr Von Muellerhoff, Alex Brush, Allura, Great Vibes, Sacramento, "
         "La Belle Aurore, Cedarville Cursive, Zeyada, Dawning of a New Day, Yellowtail. Append ' Bold' for bold where it exists.")

SYSTEM = """You are CasePatch's document de-identification agent. You edit ONE document (other documents of the same case are being edited in parallel by other agents that share the replacement map with you).

Goal: replace every piece of personal information with a realistic, consistent fake, edited in place so the document looks untouched (no boxes, matching font/size/ink), and keep everything else exactly as it is.
Personal information = people's names (every mention: full, surname-only, initials, signatures, stamps, letterheads naming a person, handwritten), dates of birth, personal addresses, personal ID/account/phone/email numbers, and anything else that identifies a private individual. Officials and clerks are people too. Every handwritten signature of a person and every face photo of a person must be replaced, including sideways signatures and small secondary photos. Keep verbatim: document dates, places, institutions/companies, case/file numbers, amounts, ages, and all ordinary text, unless a value itself identifies a person.
Fakes: same person -> same fake everywhere (check get_map first and reuse); plausible for the era, region, ethnicity and gender of the original; about the same number of characters (±1) so it fills the same width without gaps; dates/ages must stay mutually consistent (birth year consistent with stated age and document date); a fake date keeps the original's era: birth years within ±10 years, other dates within ±3, never later than the document's own date (write_map refuses others); never use a real famous person's name.

Workflow (be efficient: you have a limited budget, so do not re-view things you have already seen and batch tool calls):
1. ocr_boxes(page) gives every OCR line with id, text, bbox, glyph height and ink colour; view the page once (view, low max_side) to see layout, handwriting, signatures and stamps. OCR can misread or miss handwriting; trust your eyes and use view crops of the original to read hard parts.
2. write_map with every original -> fake (with kind) BEFORE editing.
3. Edit each occurrence: draw_text (you choose the font family from the list; give line_id + the original as it appears in that OCR line, or a tight bbox around exactly the original ink) is the default; replace_text (automatic font matching) is an alternative. For handwriting use a handwriting font; for signatures use replace_signature (rotation for sideways ones); for face photos use replace_face. Use erase_region only for marks that must disappear (and are personal). On PDF pages with a real text layer, draw_text/replace_text edit the text layer itself (only the changed words; the rest of the line is re-set if the width changes) - never erase_region or redraw text there, that turns the page into an image. The original text must be fully covered by the bbox/span.
4. View a crop of each edited area once (current) to check it looks native and has no leftover strokes; undo + redo if it doesn't.
5. finish. CasePatch then runs automatic checks (OCR re-read for leftover originals incl. surname-only mentions, ghost ink, edits that touched non-personal text, date/age/identity consistency, hidden layers) and sends you any failures to fix.

Fonts available: """ + FONTS


def _tools(pages, hosted=False):
    page = {"type": "integer", "description": f"0-based page (document has {pages} page(s))"}
    bbox = {"type": "array", "items": {"type": "number"}, "description": "[x0,y0,x1,y1] in full-resolution page pixels"}
    rot = {"type": "integer", "enum": [0, 90, 180, 270], "description": "clockwise degrees the text/signature is turned on the page: 90 = reads top-to-bottom, 270 = reads bottom-to-top, 180 = upside down"}
    t = [
        {"name": "ocr_boxes", "description": "OCR lines of the ORIGINAL page: id, text, bbox, h (glyph height px), ink colour, conf. Optional `contains` filters lines (case-insensitive substring).",
         "input_schema": {"type": "object", "properties": {"page": page, "contains": {"type": "string"}}}},
        {"name": "view", "description": "Look at the page (whole or a crop). which='current' (edited) or 'original'. max_side default 1200 (use small values for overviews, crops for detail).",
         "input_schema": {"type": "object", "properties": {"page": page, "bbox": bbox, "which": {"type": "string", "enum": ["current", "original"]}, "max_side": {"type": "integer"}}}},
        {"name": "draw_text", "description": "Replace text: erases the original ink and typesets `text` in `font`, auto-sized from the original's ink extent, baseline/left aligned, ink colour and stroke weight matched, grain added. Locate the original by line_id (+ `original` exactly as in that OCR line) OR by a tight bbox around the original ink (+ `original` = what is written there). Optional size_px, ink '#rrggbb', condense_to_fit, rotation (sideways text). The fake is fitted to the original's width (letter-spacing/scale); a length_warning means pick a fake of closer length and redo.",
         "input_schema": {"type": "object", "properties": {"page": page, "original": {"type": "string"}, "text": {"type": "string"}, "font": {"type": "string"}, "line_id": {"type": "string"}, "bbox": bbox,
                                                           "size_px": {"type": "number"}, "ink": {"type": "string"}, "condense_to_fit": {"type": "boolean"}, "rotation": rot}, "required": ["original", "text", "font"]}},
        {"name": "replace_text", "description": "Replace text with CasePatch's automatic font-matching editor (line_id + original as in that OCR line, or tight bbox). Optional font_class hint: typewriter|mono|serif|sans|sans-narrow|hand. Native PDF text layers are edited as text automatically.",
         "input_schema": {"type": "object", "properties": {"page": page, "original": {"type": "string"}, "replacement": {"type": "string"}, "line_id": {"type": "string"}, "bbox": bbox, "font_class": {"type": "string"}}, "required": ["original", "replacement"]}},
        {"name": "replace_signature", "description": "CasePatch's signature editor: erases ALL ink inside bbox (cover every stroke of the signature, including descenders and flourishes) and signs fake_name in a signature hand across the same width in the original ink colour. Optional font (a signature font name). If the signature overlaps printed text, that text is erased too: retype it afterwards with draw_text (bbox + same text).",
         "input_schema": {"type": "object", "properties": {"page": page, "bbox": bbox, "fake_name": {"type": "string"}, "font": {"type": "string"}, "rotation": rot}, "required": ["bbox", "fake_name"]}},
        {"name": "erase_region", "description": "Remove the ink inside bbox, filling from the surrounding paper (strokes of the same blobs that stick out of the box are removed too).", "input_schema": {"type": "object", "properties": {"page": page, "bbox": bbox, "rotation": rot}, "required": ["bbox"]}},
        {"name": "replace_face", "description": "Replace the face in an ID/portrait photo with a synthetic face (aligned, tone-matched, blended inside the photo; frame and background kept) and check it no longer matches the original person. bbox = the photo or the face with some margin. Use it for every photo of a real person's face, including small ghost/secondary portraits. Give your estimate of the ORIGINAL face's apparent sex, age band and ethnicity group so the synthetic face is plausible for the photo (same person type, different identity).",
         "input_schema": {"type": "object", "properties": {"page": page, "bbox": bbox, "apparent": {"type": "object", "properties": {
             "sex": {"type": "string", "enum": ["female", "male", "unknown"]}, "age_band": {"type": "string", "enum": ["young", "mid", "old", "unknown"], "description": "young <35, mid 35-55, old >55"},
             "ethnicity": {"type": "string", "enum": ["white", "black", "east_asian", "southeast_asian", "south_asian", "middle_eastern", "latin", "mixed", "unknown"]}}, "required": ["sex", "age_band", "ethnicity"]}},
             "required": ["bbox", "apparent"]}},
        {"name": "undo", "description": "Undo one edit by edit_id (restores original pixels of that edit).", "input_schema": {"type": "object", "properties": {"edit_id": {"type": "integer"}}, "required": ["edit_id"]}},
        {"name": "write_map", "description": "Add replacements to the case-wide map: entries [{original, replacement, kind}] (kind: PERSON|DOB|DATE|ADDRESS|ID|PHONE|EMAIL|ORG|OTHER). If an original is already mapped, the existing fake is returned and must be used.",
         "input_schema": {"type": "object", "properties": {"entries": {"type": "array", "items": {"type": "object", "properties": {"original": {"type": "string"}, "replacement": {"type": "string"}, "kind": {"type": "string"}}, "required": ["original", "replacement"]}}}, "required": ["entries"]}},
        {"name": "fix_map", "description": "Correct the fake for an original that is already in the map (a wrong first choice, an original mapped to itself, an implausible date). Applies to every document; the consistency check is re-run and returned. Then redo the edits that show the old fake.",
         "input_schema": {"type": "object", "properties": {"original": {"type": "string"}, "replacement": {"type": "string"}, "reason": {"type": "string"}}, "required": ["original", "replacement"]}},
        {"name": "get_map", "description": "The case-wide replacement map so far (all documents).", "input_schema": {"type": "object", "properties": {}}},
        {"name": "finish", "description": "Done with this document; triggers CasePatch's checks.", "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
    ]
    if hosted:
        t.insert(4, {"name": "hosted_patch_edit", "description": "Customer-enabled hosted image edit of a small area (bbox) with an instruction; only the masked area is pasted back.",
                     "input_schema": {"type": "object", "properties": {"page": page, "bbox": bbox, "original": {"type": "string"}, "instruction": {"type": "string"}}, "required": ["bbox", "instruction"]}})
    return t


class BudgetExceeded(Exception):
    pass


class Abort(Exception):
    pass


class Spend:
    """Thread-safe per-case spend. eval: SpendLedger rows (fail closed); product: byok.CaseBudget in memory."""

    def __init__(self, mode, case_tag, budget, doc_budget=None):
        from ..hosted import pricing as P
        self.P, self.mode, self.tag, self.budget = P, mode, case_tag, float(budget)
        self.doc_budget = float(doc_budget) if doc_budget else None; self.doc_spent = {}
        self.docs, self.done = set(), set()   # budget flow: unused allowance of other documents can be borrowed
        self.lock = threading.Lock(); self.cv = threading.Condition(self.lock); self.wait_s = 900.0; self.waiting = set()
        self.spent = 0.0; self.list_usd = 0.0; self.calls = 0; self.by_model = {}
        if mode == "eval":
            self._tl = threading.local()
        else:
            from ..docs.byok import CaseBudget
            self.ledger = CaseBudget(self.budget)

    @property
    def ledger(self):
        if self.mode != "eval":
            return self._budget
        if not hasattr(self._tl, "l"):  # sqlite connections are per thread
            from ..hosted.ledger import SpendLedger
            self._tl.l = SpendLedger(EVAL_LEDGER)
        return self._tl.l

    @ledger.setter
    def ledger(self, v):
        self._budget = v

    def reserve(self, model, est, doc):
        with self.lock:
            if self.spent + est > self.budget:
                raise BudgetExceeded(f"case budget ${self.budget:.2f} reached (spent ${self.spent:.3f}, next call est ${est:.3f})")
            ds = self.doc_spent.get(doc, 0.0)
            cap = self._doc_cap(doc)
            t_end = time.time() + self.wait_s
            # out of its own allowance while other documents still run: wait for one to finish and release money
            # (only while some other document is still working, not waiting itself: no mutual waits)
            while cap is not None and ds + est > cap and (self.docs - self.done - self.waiting - {doc}) and self.spent + est <= self.budget \
                    and time.time() < t_end:
                self.waiting.add(doc); self.cv.notify_all()
                self.cv.wait(timeout=min(30.0, max(1.0, t_end - time.time())))
                self.waiting.discard(doc)
                ds = self.doc_spent.get(doc, 0.0); cap = self._doc_cap(doc)
            if cap is not None and ds + est > cap:
                raise BudgetExceeded(f"document budget ${cap:.2f} reached (base ${self.doc_budget:.2f} + borrowed; spent ${ds:.3f}, next call est ${est:.3f})")
            self._pending = getattr(self, "_pending", {}); 
            if self.mode == "eval":
                used = self.ledger.db.execute("SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd ELSE cost_usd END),0) FROM calls "
                                              "WHERE provider='anthropic' AND case_id LIKE ?", (EVAL_EXPERIMENT_PREFIX + "%",)).fetchone()[0]
                if used + est > EVAL_EXPERIMENT_CAP:
                    raise BudgetExceeded(f"agent experiment cap ${EVAL_EXPERIMENT_CAP:.0f} reached")
                rid = self.ledger.reserve(est, 300.0, case_id=f"{self.tag}:{doc}", provider="anthropic", model=model, kind="agent",
                                          pool="anthropic", pool_cap_usd=EVAL_POOL_CAP)
            else:
                rid = self.ledger.reserve(est)
            self._pending[rid] = doc
            return rid

    def register(self, docs):
        with self.lock:
            self.docs |= set(docs)

    def mark_done(self, doc):
        with self.lock:
            self.done.add(doc); self.cv.notify_all()

    def _doc_cap(self, doc):
        """Base per-document allowance plus what the case can spare: case money not spent and not earmarked for the base
        allowance of documents still running (finished documents release theirs). Never more than 3x the base."""
        if not self.doc_budget:
            return None
        if doc not in self.docs:
            return self.doc_budget
        others = sum(max(0.0, self.doc_budget - self.doc_spent.get(d, 0.0)) for d in self.docs - self.done - {doc})
        mine = max(0.0, self.doc_budget - self.doc_spent.get(doc, 0.0))
        spare = max(0.0, self.budget - self.spent - others - mine)
        return min(3 * self.doc_budget, self.doc_budget + spare)

    def doc_left(self, doc):
        with self.lock:
            left = self.budget - self.spent
            cap = self._doc_cap(doc)
            if cap is not None:
                left = min(left, cap - self.doc_spent.get(doc, 0.0))
            return left

    def commit(self, rid, model, usage):
        pr = self.P.price("anthropic", model)
        list_usd = (usage.get("input_tokens", 0) * pr["text_in"] + usage.get("cache_creation_input_tokens", 0) * pr["cache_write"]
                    + usage.get("cache_read_input_tokens", 0) * pr["text_in"] * 0.1 + usage.get("output_tokens", 0) * pr["out"]) / 1e6
        with self.lock:
            try:
                c = self.P.actual_usd("anthropic", model, usage)
            except Exception:
                c = list_usd * 2
            self.ledger.commit(rid, c)
            d = self._pending.pop(rid, None); self.doc_spent[d] = self.doc_spent.get(d, 0.0) + c
            self.spent += c; self.list_usd += list_usd; self.calls += 1
            m = self.by_model.setdefault(model, {"calls": 0, "usd": 0.0, "list_usd": 0.0}); m["calls"] += 1; m["usd"] += c; m["list_usd"] += list_usd
        return c

    def keep(self, rid, note):
        with self.lock:
            if self.mode == "eval":
                c = self.ledger.db.execute("SELECT reserved_usd FROM calls WHERE id=?", (rid,)).fetchone()[0]
            else:
                c = self.ledger.rows[rid]["reserved"]
            self.ledger.keep(rid, note); self.spent += c; self.list_usd += c
            d = self._pending.pop(rid, None); self.doc_spent[d] = self.doc_spent.get(d, 0.0) + c


def _http(url, headers, body, timeout=600):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:2000].decode(errors="replace")}
    except (urllib.error.URLError, TimeoutError) as e:
        return 599, {"error": str(e)[:300]}


def _elide(msgs, keep):
    seen = 0
    for m in reversed(msgs):
        if m["role"] != "user":
            continue
        for b in m["content"]:
            if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                for i, x in enumerate(b["content"]):
                    if x.get("type") == "image":
                        seen += 1
                        if seen > keep:
                            b["content"][i] = {"type": "text", "text": "[older image removed; view again if needed]"}


_YEAR = re.compile(r"(?<!\d)(1[6-9]\d\d|20\d\d)(?!\d)")
_YY = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2})(?!\d)")


def doc_year(st):
    """The document's own date, estimated as the latest year (not in the future) that it mentions; None if none."""
    if getattr(st, "_doc_year", 0) != 0:
        return st._doc_year
    now = time.localtime().tm_year
    ys = []
    for p in range(len(st.orig)):
        try:
            ys += [int(y) for l in st.lines(p) for y in _YEAR.findall(l.text)]
        except Exception:
            pass
    ys = [y for y in ys if y <= now]
    st._doc_year = max(ys) if ys else None
    return st._doc_year


def _years(t):
    ys = [int(y) for y in _YEAR.findall(t)]
    return ys


_PARTICLES = {"van", "von", "der", "den", "del", "della", "des", "the", "bin", "bint", "ibn", "abu", "mac", "and", "jr.", "sr."}


def era_filter(st, entries):
    """Planner constraint: a fake date stays plausible for the document's era. A replacement year after the document's
    own date, or a birth year moved more than 10 years (other dates 3), is refused with the allowed range; identity
    mappings (original == fake) are refused too."""
    dy = doc_year(st); keep, notes = [], []
    for x in entries:
        o, r = str(x.get("original", "")), str(x.get("replacement", ""))
        kind = str(x.get("kind", "")).upper()
        if o.strip() and " ".join(o.lower().split()) == " ".join(r.lower().split()):
            notes.append(f"'{o}' -> itself refused: if it is personal give it a fake, otherwise do not map it")
            continue
        if kind in ("PERSON", "NAME", "GIVEN", "SURNAME", ""):
            ot = {t.lower() for t in re.findall(r"[^\W\d_]{3,}", o)} - _PARTICLES
            shared = sorted(ot & {t.lower() for t in re.findall(r"[^\W\d_]{3,}", r)})
            if shared and not o.startswith("("):
                notes.append(f"'{o}' -> '{r}' refused: the fake reuses '{', '.join(shared)}' from the original (a kept name part "
                             f"still identifies the person and shows through as a ghost); replace every part, keeping family "
                             f"surnames consistent via the same fake surname")
                continue
            import difflib
            look = sorted({f"{a}~{b}" for a in ot if len(a) >= 4 for b in {t.lower() for t in re.findall(r"[^\W\d_]{4,}", r)}
                           if difflib.SequenceMatcher(None, a, b).ratio() >= 0.75})
            if look and not o.startswith("("):
                notes.append(f"'{o}' -> '{r}' refused: look-alike name part ({', '.join(look)}); a one-letter change still "
                             f"identifies the person and OCR/fuzzy search finds it. Pick an unrelated name of similar length")
                continue
        if kind in ("ID", "PHONE") or kind.startswith("ID"):
            do, dr = re.sub(r"\D", "", o), re.sub(r"\D", "", r)
            if len(do) >= 6 and len(do) == len(dr):
                changed = sum(a_ != b for a_, b in zip(do, dr))
                if changed < max(4, round(0.4 * len(do))):
                    notes.append(f"'{o}' -> '{r}' refused: only {changed} of {len(do)} digits differ; change at least "
                                 f"{max(4, round(0.4 * len(do)))} (a fixed prefix such as an IMEI TAC may stay, the serial part must change)")
                    continue
        yo, yr = _years(o), _years(r)
        if yo and yr and (kind in ("DOB", "DATE", "") or kind.startswith("DATE")):
            span = 10 if kind == "DOB" else 3
            bad = [b for a_, b in zip(yo, yr) if abs(b - a_) > span or (dy and b > dy)]
            if bad:
                lo = yo[0] - span; hi = min(yo[0] + span, dy) if dy else yo[0] + span
                notes.append(f"'{o}' -> '{r}' refused: keep the year within {lo}-{hi} (document dated {dy or 'unknown'}; "
                             f"{'births' if kind == 'DOB' else 'dates'} must stay plausible for the era)")
                continue
        keep.append(x)
    return keep, notes


class DocAgent:
    def __init__(self, case, st: DocState, cmap: CaseMap, spend: Spend, key, log_dir, hosted_patch=None, effort="high", max_turns=40, replay=None):
        self.case, self.st, self.cmap, self.spend, self.key = case, st, cmap, spend, key
        self.replay = replay   # recorded assistant turns (list) -> no API call, no key, no spend
        self.hosted_patch, self.effort, self.max_turns = hosted_patch, effort, max_turns
        self.log = open(Path(log_dir) / f"{st.name}.transcript.jsonl", "a")
        self.turns = 0; self.tool_calls = {}; self.finished = False

    def note(self, **kw):
        self.log.write(json.dumps({"t": round(time.time(), 2), **kw}, ensure_ascii=False, default=str) + "\n"); self.log.flush()

    # ------------------------------------------------------------------ tools
    def tool(self, name, a):
        st = self.st; pg = int(a.get("page") or 0)
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        try:
            if pg >= len(st.orig):
                return f"error: page {pg} does not exist", None
            if name == "ocr_boxes":
                return json.dumps(st.ocr_boxes(pg, a.get("contains")), ensure_ascii=False), None
            if name == "view":
                info, b64 = st.view(pg, a.get("bbox"), a.get("which", "current"), a.get("max_side") or 1200)
                return info, b64
            if name == "draw_text":
                bbox = a.get("bbox")
                if not bbox:
                    l = st.line(pg, a.get("line_id")) if a.get("line_id") else next((l for l in st.lines(pg) if find_span(l.text, a["original"])), None)
                    if l is None:
                        return "error: line not found; pass line_id or bbox", None
                    sp = find_span(l.text, a["original"])
                    if sp is None:
                        return f"error: '{a['original']}' not in line '{l.text}'; pass a bbox", None
                    chs = l.chars[sp[0]:sp[1]] if len(l.chars) == len(l.text) else l.chars
                    bbox = _bbox_of_chars(chs)
                return json.dumps(st.draw_text(pg, bbox, a["original"], a["text"], a["font"], a.get("size_px"), a.get("ink"), bool(a.get("condense_to_fit")),
                                               rotation=int(a.get("rotation") or 0))), None
            if name == "replace_text":
                return json.dumps(st.replace_text(pg, a["original"], a["replacement"], a.get("line_id"), a.get("bbox"), font_class=a.get("font_class"))), None
            if name == "replace_signature":
                return json.dumps(st.replace_signature(pg, a["bbox"], a["fake_name"], a.get("font"), rotation=int(a.get("rotation") or 0))), None
            if name == "erase_region":
                return json.dumps(st.erase(pg, a["bbox"], rotation=int(a.get("rotation") or 0))), None
            if name == "replace_face":
                return json.dumps(st.replace_face(pg, a["bbox"], apparent=a.get("apparent")), default=str), None
            if name == "undo":
                return json.dumps(st.undo(a["edit_id"])), None
            if name == "hosted_patch_edit":
                if not self.hosted_patch:
                    return "error: hosted edits are not enabled for this case", None
                return json.dumps(self.hosted_patch(st, pg, a)), None
            if name == "write_map":
                entries, notes = era_filter(st, a.get("entries") or [])
                notes += self.cmap.write(entries, st.name)
                for x in a.get("entries") or []:  # length planning: fakes should fill the same space
                    o, r_ = str(x.get("original", "")), str(x.get("replacement", ""))
                    if o and r_ and not o.startswith("(") and len(o) >= 4 and abs(len(r_) - len(o)) > max(2, 0.2 * len(o)):
                        notes.append(f"'{r_}' has {len(r_)} characters vs {len(o)} in '{o}': choose a fake of about the same length so it fits without gaps")
                return ("ok" + ("; " + " | ".join(notes) if notes else "")), None
            if name == "fix_map":
                ok, note = self.cmap.correct(a.get("original"), a.get("replacement"), st.name, a.get("reason", ""))
                if not ok:
                    return "error: " + note, None
                from .verify import consistency_issues
                cons = consistency_issues(st, json.loads(self.cmap.text()))
                return json.dumps({"ok": True, "note": note + "; now redo the edits that show the old fake (undo + draw_text)",
                                   "consistency_after": [c["detail"] for c in cons][:6]}), None
            if name == "get_map":
                return self.cmap.text(), None
            if name == "finish":
                self.finished = True
                return "ok", None
            return f"unknown tool {name}", None
        except (BudgetExceeded, Abort):
            raise
        except Exception as e:
            return f"error: {type(e).__name__}: {str(e)[:300]}", None

    def _call(self, model, raw, est):
        """One paid API call (ledger reservation first). Returns the response dict, or None to retry."""
        rid = self.spend.reserve(model, round(est, 6), self.st.name)
        t0 = time.time()
        st_, data = _http("https://api.anthropic.com/v1/messages", {"x-api-key": self.key, "anthropic-version": ANTHROPIC_VERSION,
                                                                      "content-type": "application/json"}, raw)
        if st_ != 200:
            self.spend.keep(rid, f"http {st_}")
            msg = str(data.get("error"))[:400].replace(self.key, "[key]")
            self.note(kind="http_error", status=st_, error=msg)
            if st_ in (401, 402, 403) or "credit" in msg.lower() or "billing" in msg.lower():
                raise Abort(f"auth/billing error http {st_}: {msg}")
            if st_ in (429, 500, 502, 503, 529, 599):
                time.sleep(20); return None
            raise Abort(f"http {st_}: {msg}")
        data["_cost"] = self.spend.commit(rid, model, data.get("usage", {}))
        data["_secs"] = round(time.time() - t0, 1)
        return data

    # ------------------------------------------------------------------ one session (until finish / turn cap)
    def session(self, model, msgs):
        from ..hosted import pricing as P
        tools = _tools(len(self.st.orig), bool(self.hosted_patch))
        self.finished = False; nudged = False; turns0 = self.turns
        while self.turns - turns0 < self.max_turns and not self.finished:
            self.turns += 1
            _elide(msgs, KEEP_IMAGES)
            send = copy.deepcopy(msgs)
            send[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            max_tokens = 4000
            body = {"model": model, "max_tokens": max_tokens, "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    "tools": tools, "messages": send, "output_config": {"effort": self.effort}}
            raw = json.dumps(body).encode()
            n_img = raw.count(b'"type": "image"')
            img_chars = sum(len(x.get("source", {}).get("data", "")) for m in send for b in m["content"] if isinstance(b, dict)
                            for x in ([b] + (b.get("content") if isinstance(b.get("content"), list) else [])) if x.get("type") == "image")
            pr = P.price("anthropic", model)
            est = (((len(raw) - img_chars) / 3.2 + n_img * 1700) * pr["cache_write"] + max_tokens * pr["out"]) / 1e6
            if self.replay is not None:
                if not self.replay:
                    break
                rec = self.replay.pop(0)
                content = [{"type": "text", "text": b.get("text", "")} if b["type"] == "text" else
                           {"type": "tool_use", "id": f"rp{self.turns}_{k}", "name": b["name"], "input": _loads_or_none(b.get("input"))}
                           for k, b in enumerate(rec.get("blocks") or []) if b["type"] in ("text", "tool_use")]
                content = [b for b in content if b["type"] == "text" or b["input"] is not None]
                self.note(kind="assistant", model=rec.get("model", model), turn=self.turns, secs=0, usage=None, cost=0, replay=True,
                          blocks=rec.get("blocks"))
                data = {"content": content, "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"}
            else:
                data = self._call(model, raw, est)
                if data is None:
                    continue
            content = data.get("content", [])
            if self.replay is None:
                self.note(kind="assistant", model=model, turn=self.turns, secs=data.pop("_secs", 0), usage=data.get("usage"), cost=data.pop("_cost", 0),
                          blocks=[{"type": b["type"], **({"text": b.get("text", "")[:3000]} if b["type"] == "text" else {}),
                                   **({"name": b["name"], "input": json.dumps(b.get("input"))[:20000]} if b["type"] == "tool_use" else {})} for b in content])
            msgs.append({"role": "assistant", "content": content})
            uses = [b for b in content if b.get("type") == "tool_use"]
            if not uses:
                if data.get("stop_reason") == "max_tokens":
                    msgs.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]}); continue
                if nudged:
                    break
                nudged = True
                msgs.append({"role": "user", "content": [{"type": "text", "text": "Continue with the edits and call finish when done."}]}); continue
            res = []
            for b in uses:
                text, img = self.tool(b["name"], b.get("input") or {})
                self.note(kind="tool", name=b["name"], result=text[:2000], image=bool(img))
                cc = [{"type": "text", "text": text}]
                if img:
                    cc.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}})
                res.append({"type": "tool_result", "tool_use_id": b["id"], "content": cc})
            if self.spend.doc_left(self.st.name) < 0.2 * (self.spend.doc_budget or self.spend.budget):
                res.append({"type": "text", "text": "[CasePatch] This document's budget is nearly used: edit the remaining personal values now (no more polishing), then call finish."})
            msgs.append({"role": "user", "content": res})
        return msgs


def _loads_or_none(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except ValueError:
        return None


def load_replay(log_dir):
    """{document name: [recorded assistant turns]} from a run's logs/*.transcript.jsonl (replay: no API, no key, no spend)."""
    out = {}
    for f in Path(log_dir).glob("*.transcript.jsonl"):
        turns = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        out[f.name[:-len(".transcript.jsonl")]] = [t for t in turns if t.get("kind") == "assistant"]
    return out


def _replay_for(replays, name):
    """Recorded turns for a document; uploads may carry a different 'NN_' order prefix than the recording."""
    if name in replays:
        return list(replays[name])
    bare = re.sub(r"^(\d+_)+", "", name)
    hit = [k for k in replays if re.sub(r"^(\d+_)+", "", k) == bare]
    return list(replays[hit[0]]) if len(hit) == 1 else []


def _issue_text(issues):
    return "\n".join("- " + json.dumps(i, ensure_ascii=False) for i in issues[:40])


BLOCKING = ("leak", "ghost", "facts", "consistency", "hidden", "signature", "face")


def run_document(case, path, out_dir, cmap, spend, key, log_dir, fix_rounds=2, model=DEFAULT_MODEL, escalate=ESCALATE_MODEL,
                 hosted_patch=None, effort="high", faces="swap", replay=None):
    t0 = time.time()
    st = DocState.load(path); st.faces = faces
    st.lines(0)  # OCR up front (parallel across documents)
    ag = DocAgent(case, st, cmap, spend, key, log_dir, hosted_patch, effort, replay=replay)
    rounds, issues, models_used, stop = [], [], [model], None
    msgs = [{"role": "user", "content": [{"type": "text", "text": f"Document: {st.name} ({len(st.orig)} page(s), {st.orig[0].shape[1]}x{st.orig[0].shape[0]} px). Case documents being edited in parallel: {', '.join(sorted(p.name for p in Path(path).parent.iterdir() if p.suffix.lower() in DOC_EXT))}. Start."}]}]
    try:
        msgs = ag.session(model, msgs)
        for r in range(fix_rounds + 1):
            out = st.save(out_dir)
            issues = V.check_doc(st, cmap.entries, out)
            rounds.append({"round": r, "model": models_used[-1], "issues": len(issues), "codes": sorted({i["check"] for i in issues})})
            ag.note(kind="verify", round=r, issues=issues)
            if not issues or r == fix_rounds:
                break
            fb = ("CasePatch checks found problems in this document. Fix them (view the area, then draw_text/undo/write_map as needed), "
                  "and call finish again. If a flagged item is genuinely not personal information or is a false alarm, say so in finish's summary.\n" + _issue_text(issues))
            if r == fix_rounds - 1 and escalate and escalate != model and spend.doc_left(st.name) > 0.35:
                # last round goes to the stronger model with a fresh, compact context (current edits are kept)
                models_used.append(escalate)
                msgs = [{"role": "user", "content": [{"type": "text", "text": f"Document: {st.name}. Another agent already edited it (edits are in place; view 'current'). Case map so far: {cmap.text()}\n\n" + fb}]}]
                msgs = ag.session(escalate, msgs)
            else:
                msgs.append({"role": "user", "content": [{"type": "text", "text": fb}]})
                msgs = ag.session(models_used[-1], msgs)
    except BudgetExceeded as e:
        stop = f"budget: {e}"
        out = st.save(out_dir); issues = V.check_doc(st, cmap.entries, out)
        issues.append({"check": "budget", "detail": f"stopped: {e}. Edits so far are saved; the items above (and anything the agent had not reached) need review."})
    except Abort as e:
        stop = f"abort: {e}"
        out = st.save(out_dir); issues = V.check_doc(st, cmap.entries, out)
    issues = issues + list(st.flags)
    # per-edit masks for the review screen (revert an edit = put the original pixels back inside its mask)
    from PIL import Image as _I
    ed_dir = Path(out_dir).parent / "edits" / st.name; ed_dir.mkdir(parents=True, exist_ok=True)
    for e in st.edits:
        if e.mask is not None and not e.undone:
            _I.fromarray(e.mask.astype(np.uint8) * 255).save(ed_dir / f"{e.id}.png", optimize=True)
    spend.mark_done(st.name)
    return {"file": st.name, "doc_usd": round(spend.doc_spent.get(st.name, 0.0), 4), "wall_s": round(time.time() - t0, 1), "turns": ag.turns, "tool_calls": ag.tool_calls, "models": models_used,
            "rounds": rounds, "stop": stop, "open_issues": issues,
            "edits": [{"id": e.id, "kind": e.kind, "page": e.page, "bbox": e.bbox, "original": e.original, "replacement": e.replacement, "undone": e.undone, **({"meta": e.meta} if e.meta else {})}
                      for e in st.edits]}


def run_case(input_dir, out_dir, *, mode="product", budget=6.0, doc_budget=1.5, faces="swap", model=DEFAULT_MODEL, escalate=ESCALATE_MODEL, fix_rounds=2, workers=4,
             user_key=None, run_tag="run", effort="high", hosted_patch=None, only=None, replay_dir=None):
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    replays = None
    if mode == "replay":   # recorded transcripts re-executed against the local tools: no key, no network, $0
        replays = load_replay(replay_dir); key = ""; hosted_patch = None
    elif mode == "eval":
        real = (ROOT / "examples" / "real").resolve()
        if not str(input_dir.resolve()).startswith(str(real)):
            raise SystemExit("eval mode runs on public samples under examples/real only")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise SystemExit("no eval key in the environment")
        hosted_patch = None  # no hosted image edits in eval runs of this tool
    else:
        from ..docs import byok
        key, _src = byok.resolve_key(user_key=user_key)   # refuses CasePatch eval keys (fingerprint registry)
        model = byok.model_from({"CASEPATCH_BYOK_MODEL": model}) if model else byok.DEFAULT_MODEL
        if escalate:
            escalate = byok.model_from({"CASEPATCH_BYOK_MODEL": escalate})
        if budget > byok.MAX_USD_PER_CASE_LIMIT:
            raise SystemExit("budget too high")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = out_dir / "files"; files.mkdir(exist_ok=True)
    logs = out_dir / "logs"; logs.mkdir(exist_ok=True)
    case = input_dir.name
    spend = Spend(mode, f"{EVAL_EXPERIMENT_PREFIX}{run_tag}:{case}" if mode == "eval" else case, budget, doc_budget)
    cmap = CaseMap()
    docs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in DOC_EXT and (not only or p.name in only))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(docs)))) as ex:
        spend.register(p.name for p in docs)

        def one(p):
            try:
                return run_document(case, p, files, cmap, spend, key, logs, fix_rounds, model, escalate, hosted_patch, effort, faces,
                                    replay=_replay_for(replays, p.name) if replays is not None else None)
            except Exception as e:  # one broken document must not lose the rest of the case
                spend.mark_done(p.name)
                import shutil
                shutil.copyfile(p, files / p.name) if not (files / p.name).exists() else None
                return {"file": p.name, "doc_usd": round(spend.doc_spent.get(p.name, 0.0), 4), "wall_s": 0, "turns": 0, "tool_calls": 0, "models": [],
                        "rounds": 0, "stop": "error", "edits": [],
                        "open_issues": [{"check": "error", "detail": f"agent failed on this document ({type(e).__name__}); review it by hand"}]}
        res = list(ex.map(one, docs))
    # final cross-document pass: every map original re-checked on every output (a name mapped late by another document)
    final = {}
    for r in res:
        st = DocState.load(files / r["file"])
        st.orig = st.cur  # output only: leak + hidden checks
        lines = [V.ocr_text_lines(img) for img in st.cur]
        final[r["file"]] = [i for i in V.check_doc(st, cmap.entries, files / r["file"], lines_out=lines) if i["check"] in ("leak", "hidden")]
    (out_dir / "replacements.json").write_text(json.dumps(cmap.entries, indent=1, ensure_ascii=False))
    review = [dict(i, file=r["file"]) for r in res for i in r["open_issues"]] + \
             [dict(i, file=f, stage="final") for f, iss in final.items() for i in iss if not any(j.get("value") == i.get("value") and j["check"] == i["check"] for j in next(r for r in res if r["file"] == f)["open_issues"])]
    from .verify import compact_review
    review_all = review
    review = compact_review(review)
    report = {"case": case, "mode": mode, "wall_s": round(time.time() - t0, 1), "budget_usd": budget, "doc_budget_usd": doc_budget, "spent_usd": round(spend.spent, 4),
              "list_usd": round(spend.list_usd, 4), "model_calls": spend.calls, "by_model": spend.by_model, "documents": res, "review_issues": review, "review_issues_raw": len(review_all)}
    (out_dir / "agent-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    (out_dir / "review-issues.json").write_text(json.dumps(review, indent=1, ensure_ascii=False, default=str))
    return report
