"""Identity-document layout detector: personal values on ID cards, passports and licences whose printed field labels
are too small or blurred to read (a card photographed on a table), so the label-based form detector finds nothing.

It uses the structure ID documents share (ICAO 9303 visual zone and national cards modelled on it), not any one
document: a portrait, an issuing-state header, then the holder's surname and given names as the first short
name-shaped value lines, a document / personal number (a long run of digits, maybe with letters) and several full
dates of which the earliest, well before the others, is the date of birth. Everything it proposes is grounded in an
OCR line and carries a reason; layout guesses are proposed below full confidence so review flags them.
Pages qualify only when they look like an identity document (a detected face plus ID wording or several dates)."""
from __future__ import annotations

import math
import re

from .detect import Proposal
from .forms import NOT_A_NAME, label_kind
from .model import quad_bbox

ID_WORDS = re.compile(r"(?i)(?:\bidentit|\bidentification\b|\bid\b|\bpass(?:port|eport)\b|\bpasaport|\breisepass|"
                      r"personalausweis|\blicen[cs]e\b|\bdriv(?:er|ing)\b|leternjoftim|\bcarte\b|\bausweis|\bdni\b|"
                      r"c[ée]dula|\bpermit\b|residen(?:ce|t)\s+(?:card|permit))")
MONTHS = r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
DATE = re.compile(r"(?<![\d.])(\d{1,2})\s?[./ -]\s?(\d{1,2}|(?:" + MONTHS + r")[A-Z]*)\s?[./ -]\s?((?:19|20)\d{2})(?![\d])", re.I)
DATE_YMD = re.compile(r"(?<![\d.])((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})(?![\d])")
DOC_NO = re.compile(r"(?<![\w<])(?=(?:[A-Z]*\d){6})[A-Z]{0,3}\d[A-Z0-9]{5,13}(?![\w<])")
NAME_LINE = re.compile(r"[^\W\d_]{2,}(?:[ '’-][^\W\d_]{2,}){0,2}")
# words on ID documents that are field contents but never the holder's name
FURNITURE = re.compile(r"(?i)\b(?:specimen|sample|signature|holder|nationality|male|female|sex|type|code|valid|until|"
                       r"expiry|expires|issue|issued|authority|place|birth|date|card|number|no|class|categor\w*)\b")


def _bb(l):
    return quad_bbox((l.chars[0].quad[0], l.chars[-1].quad[1], l.chars[-1].quad[2], l.chars[0].quad[3])) if l.chars else None


def _angle(lines):
    a = []
    for l in lines:
        if len(l.text) >= 5 and l.chars:
            (x0, y0), (x1, y1) = l.chars[0].quad[0], l.chars[-1].quad[1]
            a.append(math.atan2(y1 - y0, x1 - x0))
    a.sort()
    return a[len(a) // 2] if a else 0.0


def _in(bb, rb, slack=0.0):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return rb[0] - slack <= cx <= rb[2] + slack and rb[1] - slack <= cy <= rb[3] + slack


def _dates(text):
    out = []
    for m in DATE.finditer(text):
        out.append((int(m.group(3)), m.start(), m.end()))
    for m in DATE_YMD.finditer(text):
        out.append((int(m.group(1)), m.start(), m.end()))
    return out


def is_name_line(l):
    t = l.text.strip()
    if l.conf < 0.9 or not NAME_LINE.fullmatch(t) or not t[:1].isupper():
        return False
    if NOT_A_NAME.search(t) or FURNITURE.search(t) or ID_WORDS.search(t) or label_kind(t):
        return False
    letters = re.sub(r"[^\w]", "", t)
    if len(letters) < 3 or re.fullmatch(r"(?i)(?:" + MONTHS + r")", t):
        return False
    return True


def mrz_fields(lines):
    """Holder fields read from the page's machine-readable zone (TD1/TD2/TD3): surname words, given-name words,
    document number, date of birth (date). Empty when there is no MRZ."""
    import datetime as dt
    from .mrz import _yymmdd_to_date, is_mrz
    out = {"surname": [], "given": [], "number": None, "dob": None}
    for l in lines:
        t = re.sub(r"\s+", "", l.text)
        if not is_mrz(t):
            continue
        name = None
        if re.match(r"^[PVIACT][A-Z<][A-Z<]{3}", t) and "<<" in t and not re.search(r"\d", t[5:]):
            name = t[5:]
        elif re.fullmatch(r"[A-Z<]+", t) and "<<" in t:
            name = t
        if name:
            sur, _, giv = name.partition("<<")
            out["surname"] += [w for w in sur.split("<") if len(w) >= 2]
            out["given"] += [w for w in giv.split("<") if len(w) >= 2]
            continue
        m = re.match(r"^([A-Z0-9<]{9})\d[A-Z<]{3}(\d{6})\d[MFX<]\d{6}", t)          # TD3/TD2 line 2
        if m:
            out["number"] = m.group(1).strip("<") or None
            out["dob"] = _yymmdd_to_date(m.group(2))
            continue
        m = re.match(r"^[A-Z][A-Z<][A-Z<]{3}([A-Z0-9<]{9})\d", t)                      # TD1 line 1
        if m:
            out["number"] = m.group(1).strip("<") or None
            continue
        m = re.match(r"^(\d{6})\d[MFX<]\d{6}", t)                                     # TD1 line 2
        if m:
            out["dob"] = _yymmdd_to_date(m.group(1))
    return out


def _date_value(text):
    """dates in a line -> [(date, start, end)] for numeric day-month-year forms."""
    import datetime as dt
    out = []
    for m in DATE.finditer(text):
        try:
            mon = int(m.group(2)) if m.group(2).isdigit() else \
                ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"].index(m.group(2)[:3].upper()) + 1
            out.append((dt.date(int(m.group(3)), mon, int(m.group(1))), m.start(), m.end()))
        except ValueError:
            pass
    return out


def id_page(page, regions_on_page):
    lines = [l for l in page.lines if l.chars and l.source != "json" and l.text.strip()]
    if not lines or len(lines) > 70:
        return False
    mz = mrz_fields(lines)
    if mz["surname"] or mz["number"]:
        return True
    has_face = any(r.kind == "face" for r in regions_on_page)
    words = any(ID_WORDS.search(l.text) or NOT_A_NAME.search(l.text) for l in lines)
    ndates = sum(len(_dates(l.text)) for l in lines)
    return (has_face and (words or ndates >= 2)) or (words and ndates >= 2 and page.raster)


def id_document_proposals(docs, regions, existing=()):
    """Proposals for pages that look like identity documents. `existing`: proposals from other detectors (a page
    whose names were already found through readable labels is left to them)."""
    out = []
    named_lines = {p.line_id for p in existing if p.kind in ("PERSON", "GIVEN", "SURNAME")}
    for d in docs:
        for p in d.pages:
            regs = [r for r in regions if r.id.startswith(f"{d.id}:p{p.index}:")]
            if not id_page(p, regs):
                continue
            def add(l, s, e, kind, why, conf=0.75):
                if e - s >= 2:
                    out.append(Proposal(l.id, s, e, kind, l.text[s:e], "id-layout", conf, None, {"why": why}))
            avoid = [quad_bbox(r.quad) for r in regs if r.kind in ("face", "signature")]
            lines = [l for l in p.lines if l.chars and l.source != "json" and l.text.strip()]
            lines = [l for l in lines if not any(_in(_bb(l), rb) for rb in avoid)]
            # --- document / personal numbers
            for l in lines:
                if l.text.count("<") >= 2:
                    continue  # MRZ lines are rewritten as a whole by the rules detector
                for m in DOC_NO.finditer(l.text):
                    # a bare 6-digit run is as often a form edition or month/year ('092019') as a number
                    if not _dates(m.group()) and (re.search(r"[A-Z]", m.group()) or len(m.group()) >= 7):
                        add(l, m.start(), m.end(), "ID", "ID document: document/personal number")
            # --- the machine-readable zone names the holder: find the same values in the visual zone
            mz = mrz_fields(lines)
            from .mrz import is_mrz
            for l in lines:
                if is_mrz(l.text):
                    continue
                for kind, words in (("SURNAME", mz["surname"]), ("GIVEN", mz["given"])):
                    for w in words:
                        for m in re.finditer(r"(?<![^\W\d_])" + re.escape(w) + r"(?![^\W\d_])", l.text, re.I):
                            add(l, m.start(), m.end(), kind, "ID document: name also in the MRZ", 0.9)
                if mz["number"] and len(mz["number"]) >= 5:
                    for m in re.finditer(re.escape(mz["number"]), l.text):
                        add(l, m.start(), m.end(), "ID", "ID document: number also in the MRZ", 0.9)
                if mz["dob"]:
                    for d0, s0, e0 in _date_value(l.text):
                        if d0 == mz["dob"]:
                            add(l, s0, e0, "DOB", "ID document: date of birth also in the MRZ", 0.9)
            # --- date of birth: the earliest full date, at least 10 years before another date on the document
            ds = [(y, l, s, e) for l in lines for y, s, e in _dates(l.text)]
            if len(ds) >= 2:
                y0, l0, s0, e0 = min(ds, key=lambda x: x[0])
                if max(y for y, *_ in ds) - y0 >= 10 and sum(1 for y, *_ in ds if y == y0) == 1:
                    add(l0, s0, e0, "DOB", "ID document: earliest date, well before issue/expiry", 0.8)
            # --- holder name: first run of name-shaped lines below the header, read in the document's own rotation
            if any(l.id in named_lines for l in lines):
                continue
            a = _angle(lines)
            ca, sa = math.cos(-a), math.sin(-a)
            def rot(l):
                b = _bb(l); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                return (cx * ca - cy * sa, cx * sa + cy * ca)
            order = sorted(lines, key=lambda l: rot(l)[1])
            hs = sorted(math.hypot(l.chars[0].quad[3][0] - l.chars[0].quad[0][0], l.chars[0].quad[3][1] - l.chars[0].quad[0][1]) for l in lines)
            H = hs[len(hs) // 2] or 1.0
            header_seen = False
            run = []
            for l in order:
                if ID_WORDS.search(l.text) or NOT_A_NAME.search(l.text):
                    header_seen = True
                    if run:
                        break
                    continue
                if not header_seen:
                    continue
                if is_name_line(l):
                    if run and (abs(rot(l)[1] - rot(run[-1])[1]) > 3.5 * H or abs(_bb_x(l, rot) - _bb_x(run[-1], rot)) > 4 * H):
                        break
                    run.append(l)
                    if len(run) == 3:
                        break
                    continue
                if l.conf < 0.85 and not _dates(l.text) and not DOC_NO.search(l.text):
                    continue  # a half-read field label between values ('e sadu s' for 'emri/given name')
                if run:
                    break
            if not run:
                continue
            if len(run) == 1:
                t = run[0].text.strip()
                if len(t.split()) >= 2:
                    add(run[0], 0, len(run[0].text.rstrip()), "PERSON", "ID document: holder name line")
                continue
            kinds = ["SURNAME"] + ["GIVEN"] * (len(run) - 1)   # ICAO 9303 order: primary identifier first
            for l, k in zip(run, kinds):
                st = len(l.text) - len(l.text.lstrip())
                add(l, st, len(l.text.rstrip()), k, f"ID document: holder name ({'surname' if k == 'SURNAME' else 'given names'} line)")
    return out


def _bb_x(l, rot):
    """left edge in the document's own rotation (value lines of one field column start at the same x)."""
    a = l.chars[0].quad[0]
    c = rot(l)
    b = _bb(l)
    return c[0] - (b[2] - b[0]) / 2 if a else c[0]
