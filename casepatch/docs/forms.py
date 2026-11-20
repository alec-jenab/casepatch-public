"""Form-field detector: personal values located by their printed LABEL, using page geometry.

Rule/NER detectors read lines in isolation, so a typed 'John' under '1. NAME (Print)' or '8319' under
'4. TELEPHONE' is invisible to them. This detector pairs label lines with value lines (same row to the right, the
row below, or the line above a parenthesised caption like '(Last)'), tolerating OCR misspellings of the label.
It also understands the numbered-field layout of EU/ISO driving licences (1 surname, 2 given names, 3 birth date,
4d/5 numbers, 7 signature). Every proposal is grounded in an extracted line, like any other detector."""
from __future__ import annotations

import difflib
import math
import re

from .detect import Proposal

# label phrase -> kind. Order matters: longer/more specific phrases first.
LABELS = [
    ("date of birth", "DOB"), ("birth date", "DOB"), ("born on", "DOB"), ("dob", "DOB"), ("date de naissance", "DOB"),
    ("geburtsdatum", "DOB"), ("fecha de nacimiento", "DOB"), ("дата рождения", "DOB"),
    ("given names", "GIVEN"), ("given name", "GIVEN"), ("first name", "GIVEN"), ("forename", "GIVEN"), ("prénom", "GIVEN"),
    ("vorname", "GIVEN"), ("имя", "GIVEN"), ("отчество", "GIVEN"),
    ("surname", "SURNAME"), ("last name", "SURNAME"), ("family name", "SURNAME"), ("nachname", "SURNAME"), ("фамилия", "SURNAME"),
    ("name and address", "PERSON+ADDRESS"), ("full name", "PERSON"),
    ("signature of", "PERSON"), ("signed by", "PERSON"), ("signature", "PERSON"),
    ("wife", "PERSON"), ("husband", "PERSON"), ("spouse", "PERSON"),
    ("children", "PERSON"), ("child", "PERSON"), ("daughter", "PERSON"), ("son", "PERSON"),
    ("witnesses", "PERSON"), ("witness", "PERSON"),
    ("applicant", "PERSON"), ("declarant", "PERSON"), ("petitioner", "PERSON"),
    ("name", "PERSON"), ("nom", "PERSON"), ("nombre", "PERSON"),
    ("telephone", "PHONE"), ("phone", "PHONE"), ("tel", "PHONE"), ("телефон", "PHONE"),
    ("place of residence", "ADDRESS"), ("home address", "ADDRESS"), ("residence", "ADDRESS"), ("address", "ADDRESS"),
    ("adresse", "ADDRESS"), ("domicile", "ADDRESS"), ("адрес", "ADDRESS"),
    ("passport no", "ID"), ("passport number", "ID"), ("document no", "ID"), ("licence no", "ID"), ("license no", "ID"),
    ("id no", "ID"), ("serial number", "ID"), ("social security", "ID"),
    ("card no", "ID"), ("personal no", "ID"), ("personal number", "ID"),
    # Albanian ID card (MIDV-500 specimen): mbiemri = surname, emri = given name, datëlindja = date of birth
    ("mbiemri", "SURNAME"), ("emri", "GIVEN"), ("datelindja", "DOB"), ("datëlindja", "DOB"),
]
# whole-label matches only (too short or too common to match fuzzily inside longer labels): US licence abbreviations
# ('LN', 'FN', 'ADDR') and the person a statement, EOB or claim is about ('Member:' but not 'Member ID:' or
# 'Member services')
EXACT_LABELS = {"ln": "SURNAME", "fn": "GIVEN", "addr": "ADDRESS", "member": "PERSON", "patient": "PERSON",
                "subscriber": "PERSON", "insured": "PERSON", "cardholder": "PERSON", "card holder": "PERSON",
                "account holder": "PERSON", "guarantor": "PERSON", "member name": "PERSON", "patient name": "PERSON"}
# labels about organisations or places, never personal values
NOT_PERSONAL = re.compile(r"employer|company|business|firm|bank|place of birth|birthplace|country|nationality|issued|authority|"
                          r"office|board|place of employment|occupation|race|color|height|weight|sex|"
                          r"printed name and title|name and title", re.I)
CAPTIONS = [("given name", "GIVEN"), ("given names", "GIVEN"), ("first name", "GIVEN"), ("family name", "SURNAME"), ("last name", "SURNAME"),
            ("first", "GIVEN"), ("given", "GIVEN"), ("christian", "GIVEN"), ("middle", "GIVEN"), ("last", "SURNAME"),
            ("surname", "SURNAME"), ("family", "SURNAME"), ("number and street", "ADDRESS"), ("street", "ADDRESS"),
            # date-of-birth fields split into captioned boxes on old forms: '1892' above '(Year)'
            ("year", "DOB"), ("month", "DOB"), ("day", "DOB")]
NUMBERING = re.compile(r"^\s*(?:\d{1,2}[a-z]?|[a-z])\s*[.)]\s*", re.I)
EU_MARK = re.compile(r"^\s*(1|2|3|4a|4b|4c|4d|5|7|8|9)\s*\.\s*(.*)$", re.I)


def _bb(l, scale=1.0, rot=0.0):
    """Line box; with `rot` (radians) in the frame of a page photographed at that angle, so rows and columns of a
    tilted card line up the way they do on the document itself."""
    pts = [p for c in l.chars for p in c.quad]
    if rot:
        ca, sa = math.cos(-rot), math.sin(-rot)
        pts = [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale)


def page_angle(lines):
    """Median text baseline angle of a raster page (0 when it is level within 3 degrees)."""
    a = sorted(math.atan2(l.chars[-1].quad[1][1] - l.chars[0].quad[0][1], l.chars[-1].quad[1][0] - l.chars[0].quad[0][0])
               for l in lines if l.chars and len(l.text.strip()) >= 5)
    m = a[len(a) // 2] if a else 0.0
    return m if abs(m) > math.radians(3) else 0.0


def _norm(t):
    t = NUMBERING.sub("", t.strip())
    t = re.sub(r"\((?:print|type|printed|in full|if any)\)", "", t, flags=re.I)
    return re.sub(r"[^\w\s']", " ", t.lower()).split()


def _fuzzy_has(words, phrase, cut=0.8):
    """phrase occurs in words, word-by-word fuzzy (OCR: 'BITH' ~ 'BIRTH', 'Firnt' ~ 'First')."""
    p = phrase.split()
    for i in range(len(words) - len(p) + 1):
        ok, short_miss = True, 0
        for j, (a, b) in enumerate(zip(words[i:i + len(p)], p)):
            if a == b:
                continue
            if i + j == len(words) - 1 and j == len(p) - 1 and (len(p) > 1 or len(a) >= 5) and \
                    difflib.SequenceMatcher(None, a, b[:len(a)]).ratio() >= 0.8:
                continue  # the label's last word cut off by the card edge / OCR box ('given m', 'datélind')
            if len(b) <= 3 and len(p) >= 3 and short_miss == 0 and a and difflib.SequenceMatcher(None, a, b).ratio() >= 0.66:
                short_miss += 1  # one clipped short word in a 3+ word label ('Det o bith' ~ 'date of birth')
                continue
            if len(b) <= 3 or difflib.SequenceMatcher(None, a, b).ratio() < cut:
                ok = False; break
        if ok:
            return True
        if len(p) >= 3:
            # handwritten-era labels OCR badly word by word ('Det o bith'): a 3+ word phrase also matches when every
            # word is at least half right and the phrase as a whole is mostly right
            rs = [1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(words[i:i + len(p)], p)]
            if min(rs) >= 0.5 and sum(rs) / len(rs) >= 0.7:
                return True
    return False


def label_kind(text):
    """Kind for a label line, or None. Label lines are short, mostly capitals / end with ':' / are numbered."""
    raw = re.sub(r"№\s*[oо0]?", "No", text.strip())
    if "/" in raw and not raw.startswith("("):
        # bilingual labels on ID documents: 'Дата рождения / Date of birth', 'Номер паспорта / Passport No.'
        parts = [x.strip() for x in raw.split("/") if x.strip()]
        if len(parts) >= 2 and all(re.search(r"[^\W\d_]{2}", x) for x in parts):  # not 'DOB: 03/07/1984' (a date)
            for part in parts:
                k = label_kind(part)
                if k:
                    return k
            return None
    # trailing OCR debris after a handwritten label ('Det o bith 4' where the pen stroke of the value was read as '4')
    raw = re.sub(r"(?:\s+(?:[^\s\w]|\d){1,2})+$", "", raw) or raw  # (X has no spaces: no backtracking blow-up)
    words = _norm(raw)
    if not words or len(words) > 12 or NOT_PERSONAL.search(raw):
        return None
    if re.search(r"\d", NUMBERING.sub("", raw)):
        return None
    letters = [c for c in raw if c.isalpha()]
    labelish = raw.endswith(":") or NUMBERING.match(raw) or (letters and sum(c.isupper() for c in letters) / len(letters) > 0.7) or len(words) <= 3
    if not labelish:
        return None
    if len(words) > 6 and not NUMBERING.match(raw):
        return None
    exact = EXACT_LABELS.get(" ".join(words))
    if exact:
        return exact
    for phrase, kind in LABELS:
        if _fuzzy_has(words, phrase):
            return kind
    return None


def is_printed_label(text):
    """Any numbered, mostly-capitals form label (including non-personal ones like 'EMPLOYER'S NAME')."""
    raw = text.strip()
    letters = [c for c in raw if c.isalpha()]
    return bool(NUMBERING.match(raw) and len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7)


def caption_kind(text):
    t = text.strip()
    if not (t.startswith("(") and t.rstrip(")").count("(") == 1):
        return None
    words = _norm(t.strip("() "))
    if not words or len(words) > 4:
        return None
    for phrase, kind in CAPTIONS:
        if _fuzzy_has(words, phrase) and (len(words) == len(phrase.split()) or kind == "ADDRESS"):
            return kind
    return None


DATE_SHAPE = re.compile(r"\b\d{1,4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b|\b\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}\b|"
                        r"\b[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}\b|\b(?:19|20)\d{6}\b")
OWN_LABEL = re.compile(r"^\s*[A-Za-z][A-Za-z .#'/-]{0,30}:\s*\S")  # 'Member ID: …', 'MRN:0048…'


def _value_kind(kind, text):
    """Refine compound/address labels by the value's shape; None = not a personal value."""
    t = text.strip()
    if len(re.sub(r"\W", "", t)) < 2:
        return None
    if kind == "ADDRESS":
        return "ADDRESS" if re.search(r"[^\W\d_]{2}", t) and re.search(r"\d", t) and len(t) > 6 else None
    if kind == "PERSON+ADDRESS":
        if re.match(r"^\d{1,6}\s", t):
            return "ADDRESS"
        return "PERSON" if re.search(r"[A-Za-z]{2}", t) and not re.search(r"\d", t) else None
    if kind == "PHONE":
        return "PHONE" if len(re.sub(r"\D", "", t)) >= 4 else None
    if kind == "DOB":
        # a date (or a lone year/month/day box on old forms), never another field's value ('MRN:00483921',
        # 'Member ID: HLX884210937') that happens to sit under a date-of-birth label
        if OWN_LABEL.match(t):
            return None
        return "DOB" if DATE_SHAPE.search(t) or re.fullmatch(r"\d{1,4}", t) else None
    if OWN_LABEL.match(t):
        return None  # 'Phone: (415) 555-0142' carries its own label: it is not the value of a label above it
    if kind == "ID":
        return "ID" if re.search(r"\d{3,}", t) else None
    if kind in ("PERSON", "GIVEN", "SURNAME"):
        toks = t.replace("/", " ").split()
        # a name is 1-4 word-like tokens; OCR of cursive ('h W hill .tt g Hel') is not, and must not be retyped
        # a bare middle initial ('JOHN L SMITH') is allowed between two name words
        # (OCR splits a long hyphenated name: 'OKONKW O-BRANDTHAUSER')
        ok = 1 <= len(toks) <= 4 and all(re.fullmatch(r"[^\W\d_]{2,}[.'’-]?[^\W\d_]*\.?|[^\W\d_]\.|[^\W\d_]+(?:[-'’][^\W\d_]+)+", x)
                                         or (0 < i < len(toks) - 1 and re.fullmatch(r"[A-Z]", x)) for i, x in enumerate(toks))
        if ok and (NOT_A_NAME.search(t) or (("/" in t) and t[:1].islower()) or label_kind(t)):
            ok = False  # document furniture ('REPUBLIC OF ALBANIA'), a nationality, or the next field's label
        return kind if ok else None
    return None


# words printed on ID documents that are never a holder's name (issuing state, nationality, card titles)
NOT_A_NAME = re.compile(r"(?i)\b(?:republi[ck]a?|republika|kingdom|federation|federa[tc]i[oy]a?|"
                        r"identity|identification|passport|pasaport\w*|nationality|citizenship|shqip\w*|albani\w*|"
                        r"ministry|government|driving|licen[cs]e|printed name and title|name and title)\b")


def _person_span(text):
    """Strip honorifics / OCR dots: 'Mra..J. F..List' -> span of 'List' (the surname) + its kind."""
    m = re.search(r"([A-Z][a-z'’-]{1,})\s*$", text)
    return (m.start(1), m.end(1)) if m else None


def eu_licence_fields(lines):
    """[(field, line, bbox)] when the page uses the EU numbered licence layout (>= 4 distinct markers incl. 4a-4d)."""
    marks = []
    for l, bb in lines:
        m = EU_MARK.match(l.text)
        if m:
            marks.append((m.group(1).lower(), l, bb, m.start(2)))
    keys = {k for k, *_ in marks}
    if len(keys) < 4 or not keys & {"4a", "4b", "4c", "4d"}:
        return []
    return sorted(marks, key=lambda x: (x[2][1] + x[2][3]) / 2)


def eu_field_values(lines, marks):
    """field -> [(line, start)] lines belonging to each numbered field (rest of the marker line + following rows)."""
    out = {}
    for i, (k, l, bb, st) in enumerate(marks):
        vals = []
        if l.text[st:].strip():
            vals.append((l, st))
        y_end = (marks[i + 1][2][1] + marks[i + 1][2][3]) / 2 if i + 1 < len(marks) else bb[3] + 2 * (bb[3] - bb[1])
        h = bb[3] - bb[1]
        cy0 = (bb[1] + bb[3]) / 2 - 0.6 * h
        for l2, b2 in lines:
            if l2 is l or EU_MARK.match(l2.text):
                continue
            cy = (b2[1] + b2[3]) / 2
            if b2[0] >= bb[0] - h * 0.5 and b2[0] < bb[2] + 12 * h and cy0 <= cy < y_end - 0.15 * h:
                vals.append((l2, 0))
        out.setdefault(k, []).extend(vals)
    return out


EU_KIND = {"1": "SURNAME", "2": "GIVEN", "3": "DOB", "4d": "ID", "5": "ID"}


def form_field_proposals(docs):
    out = []
    for d in docs:
        for p in d.pages:
            sc = 1.0
            rot = page_angle([l for l in p.lines if l.source in ("ocr", "ocr_pixel_only")]) if getattr(p, "raster", False) else 0.0
            lines = [(l, _bb(l, sc, rot)) for l in p.lines if l.chars and l.source not in ("json",) and l.text.strip()]
            if not lines:
                continue
            def add(l, s, e, kind, why, conf=0.85):
                while e > s and l.text[e - 1] in " /|":  # 'СМИРНОВА /': the separator before the second script
                    e -= 1
                if e - s >= 2:
                    out.append(Proposal(l.id, s, e, kind, l.text[s:e], "form-fields", conf, None, {"why": why}))
            # --- EU licence numbered fields
            marks = eu_licence_fields(lines)
            if marks:
                for k, vals in eu_field_values(lines, marks).items():
                    kind = EU_KIND.get(k)
                    for l, st in vals if kind else []:
                        v = l.text[st:]
                        if kind == "DOB":
                            m = re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", v)
                            m and add(l, st + m.start(), st + m.end(), "DOB", f"licence field {k}")
                        elif kind == "ID":
                            m = re.search(r"[A-Z0-9]*\d{5,}[A-Z0-9]*", v)
                            if m:
                                mm = re.match(r"\d{5,}", m.group()) or m
                                add(l, st + m.start(), st + m.start() + len(mm.group()), "ID", f"licence field {k}")
                        else:
                            lead = len(v) - len(v.lstrip())
                            core = v.strip()
                            if re.search(r"[^\W\d_]{2}", core.replace(" ", "")) and not re.search(r"\d", core):
                                add(l, st + lead, st + lead + len(core), kind, f"licence field {k}")
                continue
            hs = sorted(b[3] - b[1] for _, b in lines)
            H = hs[len(hs) // 2]
            labels = [(l, b, label_kind(l.text)) for l, b in lines]
            labels = [(l, b, k) for l, b, k in labels if k and not caption_kind(l.text)]
            label_ids = {l.id for l, _, _ in labels} | {l.id for l, _ in lines if is_printed_label(l.text)}
            caps = [(l, b, caption_kind(l.text)) for l, b in lines]
            cap_lines = {l.id for l, b, k in caps if l.text.strip().startswith("(")}
            used, used_bb = set(), []
            # --- captions under values: '(Last)' -> the value right above
            for cl, cb, ck in caps:
                if not ck:
                    continue
                ccx = (cb[0] + cb[2]) / 2
                best = None
                for l, b in lines:
                    if l.id in label_ids or l.id in cap_lines or l is cl:
                        continue
                    if b[3] < cb[1] - 1.0 * H or b[1] > cb[1] - 0.2 * H:
                        continue
                    if not (b[0] - 1.5 * H <= ccx <= b[2] + 1.5 * H):
                        continue
                    dist = abs(cb[1] - b[3]) + abs((b[0] + b[2]) / 2 - ccx) * 0.2
                    if best is None or dist < best[0]:
                        best = (dist, l, b)
                if best:
                    l = best[1]
                    if l.id in used:
                        continue
                    m = re.match(r"^(?:\d+\s+)+", l.text)  # a stamped number in front of the written value
                    st = m.end() if m else 0
                    val = l.text[st:]
                    kk = "PERSON" if ck in ("GIVEN", "SURNAME") and len(val.split()) >= 2 else ck
                    k = _value_kind(kk, val)
                    if k:
                        add(l, st, len(l.text), k, f"caption {cl.text.strip()!r}"); used.add(l.id); used_bb.append(best[2])
            # --- labels: value to the right on the same row, else the first row below within the label's column
            for ll, lb, lk in labels:
                lh = lb[3] - lb[1]
                if any(abs((b[1] + b[3]) / 2 - (lb[1] + lb[3]) / 2) < max(lh, H) and b[2] > lb[0] for b in used_bb):
                    continue  # this label's value was already found through its captions
                row_labels = [b for l, b, _ in labels if l is not ll and abs((b[1] + b[3]) / 2 - (lb[1] + lb[3]) / 2) < 0.6 * lh and b[0] > lb[0]]
                col_end = min([b[0] for b in row_labels] + [lb[2] + 14 * H])
                cand = []
                for l, b in lines:
                    if l is ll or l.id in label_ids or l.id in cap_lines or len(re.sub(r"\W", "", l.text)) < 2 or getattr(l, "conf", 1.0) < 0.78:
                        continue  # punctuation-only OCR fragments (a dropped non-Latin word leaves '/') are not values
                    cy = (b[1] + b[3]) / 2
                    # two-column forms align values in one column: a short label ('Member:') sits far from its value
                    right = abs(cy - (lb[1] + lb[3]) / 2) < 0.5 * max(lh, H) and lb[2] - 2 <= b[0] < lb[2] + 12 * H
                    below = lb[3] - 0.3 * lh <= b[1] <= lb[3] + 1.6 * max(H, lh) and b[2] > lb[0] - 3 * H and b[0] < col_end
                    if below and not right and any(l2 is not ll and abs((b2[1] + b2[3]) / 2 - cy) < 0.6 * max(lh, H) and b2[2] <= b[0] + 2
                                                   for l2, b2, _ in labels):
                        below = False  # that line is the value of the label printed on its own row ('FN' under 'LN')
                    if right or below:
                        cand.append((0 if right else 1, b[1], l, b))
                # only lines that can be this label's value compete (a neighbouring caption in another script on the
                # same row, 'Учетная запись', must not shadow the date printed under 'Date of birth')
                vk = "GIVEN" if lk == "PERSON" else lk
                cand = [c for c in cand if _value_kind(vk, c[2].text) or (lk == "PERSON" and _value_kind("PERSON", c[2].text))]
                if not cand:
                    continue
                cand.sort(key=lambda x: (x[0], x[1]))
                first = cand[0]
                row = [c for c in cand if c[0] == first[0] and abs(c[3][1] - first[3][1]) < 0.6 * H]
                if lk == "PERSON+ADDRESS":
                    row = [c for c in cand if c[0] == first[0] and c[3][1] < first[3][1] + 1.6 * H]
                elif "/" in ll.text and first[0] == 1 and lk in ("SURNAME", "GIVEN"):
                    # bilingual ID field: the value is printed twice, one row per script ('СМИРНОВА /' then 'SMIRNOVA')
                    nxt = [c for c in cand if c[0] == 1 and first[3][1] + 0.6 * H <= c[3][1] < first[3][3] + 1.2 * H
                           and abs(c[3][0] - first[3][0]) < 2 * H]
                    from .resolve import _script
                    # only the same value in the other script ('Shqiptare' under 'Monika' is the next field, not a twin)
                    nxt = [c for c in nxt if _script(c[2].text) != _script(first[2].text)]
                    row = row + nxt[:1]
                row.sort(key=lambda c: c[3][0])
                if lk == "PERSON" and len(row) >= 2:
                    # name split over several boxes on one row (first / middle / last): last box is the surname
                    for i, (_, _, l, b) in enumerate(row):
                        if l.id in used:
                            continue
                        k = _value_kind("GIVEN" if i < len(row) - 1 else "SURNAME", l.text)
                        k and add(l, 0, len(l.text), k, f"label {ll.text.strip()!r}")
                    continue
                for i, (_, _, l, b) in enumerate(row):
                    if l.id in used:
                        continue
                    k = _value_kind(lk, l.text)
                    if not k:
                        continue
                    if k == "PERSON" and lk == "PERSON+ADDRESS":
                        if i > 0:
                            continue
                        sp = _person_span(l.text)
                        if sp and len(l.text.split()) > 1 and not re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}", l.text.strip()):
                            add(l, sp[0], sp[1], "SURNAME", f"label {ll.text.strip()!r}"); continue
                    add(l, 0, len(l.text), k, f"label {ll.text.strip()!r}")
    return out
