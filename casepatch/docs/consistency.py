"""Case-level consistency planning: a fact graph over every document in a case, a constraint-aware replacement
planner, a plausibility checker for planned replacements, and a post-edit verifier over the re-read outputs.

Why: a format-only fake can be absurd in context (a birth year 1892 -> 6217 on a 1917 draft card, a date
04.08.1980 -> 57.11.3337, a DOB after the passport's issue date, a card whose last four no longer match the
statement). Every rule below is local and deterministic (keyed by the case key); no model is needed.

Graph
  entities   person/org/account/card/address/... with every occurrence and surface variant
             (full name, initials, Last-First, ALL CAPS, possessive, MRZ name, email local part, ...)
  relations  typed edges between occurrences and unreplaced document facts:
             DOB_DOC (DOB before the document date, never in the future), AGE_DOB (stated age vs DOB vs document
             date, or the date implied by them), DOB_ISSUE, ISSUE_EXPIRY, SAME_DATE (one birth date in every
             format, page, document and the MRZ), MRZ_CHECK, MRZ_VISUAL, CARD_LUHN, CARD_NETWORK, CARD_LAST4,
             ACCOUNT_LAST4, ZIP_STATE, PHONE_AREA, EMAIL_NAME, GENDER_NAME, AMOUNT_SUM, SAME_VALUE, DISTINCT_VALUE
Constraint semantics
  A relation is evaluated on the ORIGINAL values and on the replacements. It is a violation only when it held on
  the original and fails on the replacement, so an intentional or historical mismatch in the source (a typo'd
  DOB, a card stated age that is one off, two different signers) is preserved, never "fixed".
Planner
  Dates/years are planned per date VALUE (not per text), so 'June 9 1892', '09.06.1892', '1892' boxed alone and
  the MRZ '920609' stay one fictional date. Candidates come from a keyed stream (deterministic per case key); the
  first one satisfying every hard constraint (and, if possible, the same rendered length) wins.
Post-edit verifier
  Re-reads output text (OCR / text layer), finds each planned replacement and re-checks every relation on the
  values actually printed, plus MRZ check digits and amount sums. Violations go to review and fail the edit bench.
Model step
  `propose_relations(graph, client)` merges relation proposals from a BYOK client (off by default; typed allowlist;
  never required). Tests use a fake client; there are no network calls here.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field

TODAY = dt.date.today()
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MAX_YEAR_SHIFT = 4          # a birth year moves by 1..4 years: same era, plausibly the same generation
DAY_SHIFT = (20, 300)       # a full birth date moves by 20..300 days (keyed), before the constraint filter

# ------------------------------------------------------------------------------------------------ typed date values
_TOK = re.compile(r"\d+|[^\W\d_]+|[^\w]+|_", re.U)


@dataclass
class DateVal:
    text: str
    toks: list
    date: dt.date | None          # None for year precision
    year: int
    precision: str                # 'day' | 'year'
    roles: dict                   # role -> token index ('d', 'm', 'y')
    two_digit_year: bool = False

    @property
    def key(self):
        return ("d", self.date.isoformat()) if self.precision == "day" else ("y", self.year)


def _month_index(word):
    w = word.lower().rstrip(".")
    if len(w) < 3:
        return None
    for i, m in enumerate(MONTHS):
        if m.lower().startswith(w[:3]) and (len(w) <= len(m) and m.lower().startswith(w) or w == "sept"):
            return i
    ru = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    for i, m in enumerate(ru):
        if w.startswith(m[:3]):
            return i
    return None


def _century(yy, ref_year):
    ref = ref_year or TODAY.year
    for c in (2000, 1900, 1800):
        if c + yy <= ref:
            return c + yy
    return 1900 + yy


def parse_date(text, ref_year=None, loose=False):
    """A date-of-birth-like value -> DateVal, or None. Day precision: month-name, d/m/y, m/d/y, y-m-d (2- or 4-digit
    years). Year precision: a bare 4-digit year 1800..2099 (old forms box the year separately). `loose` accepts any
    4-digit year (to judge a planned replacement such as '6217')."""
    if text is None:
        return None
    toks = _TOK.findall(text)
    nums = [(i, t) for i, t in enumerate(toks) if t.isdigit()]
    # an ordinal day ('July 9th, 1951', '9th of July 1951'): the suffix and 'of' are layout, not a month name
    ords = [i for i, t in enumerate(toks) if t.lower() in ("st", "nd", "rd", "th") and i and toks[i - 1].isdigit()
            and len(toks[i - 1]) <= 2]
    alphas = [(i, t) for i, t in enumerate(toks) if t.isalpha() and i not in ords and not (t.lower() == "of" and ords)]
    if len(ords) > 1:
        return None
    if len(nums) == 1 and not alphas and len(nums[0][1]) == 4 and (loose or 1800 <= int(nums[0][1]) <= 2099):
        return DateVal(text, toks, None, int(nums[0][1]), "year", {"y": nums[0][0]})
    roles = {}
    if len(alphas) == 1 and len(nums) == 2 and _month_index(alphas[0][1]) is not None:
        mi = _month_index(alphas[0][1])
        (ia, a), (ib, b) = nums
        if ords and ords[0] - 1 not in (ia,):
            return None
        if len(b) in (2, 4) and len(a) <= 2:
            roles = {"m": alphas[0][0], "d": ia, "y": ib}
        elif len(a) == 4 and len(b) <= 2:
            roles = {"m": alphas[0][0], "d": ib, "y": ia}
        else:
            return None
        y, d, m = toks[roles["y"]], int(toks[roles["d"]]), mi + 1
    elif len(nums) == 3 and not alphas and not ords:
        (i0, a), (i1, b), (i2, c) = nums
        seps = "".join(t for t in toks if not t.isdigit()).strip()
        if len(a) == 4:
            roles = {"y": i0, "m": i1, "d": i2}
        elif len(c) in (2, 4) and len(a) <= 2 and len(b) <= 2:
            if int(a) > 12:
                roles = {"d": i0, "m": i1, "y": i2}
            elif int(b) > 12:
                roles = {"m": i0, "d": i1, "y": i2}
            elif "/" in seps and "." not in seps:
                roles = {"m": i0, "d": i1, "y": i2}   # US default for slashes
            else:
                roles = {"d": i0, "m": i1, "y": i2}
        else:
            return None
        y, d, m = toks[roles["y"]], int(toks[roles["d"]]), int(toks[roles["m"]])
    else:
        return None
    two = len(y) == 2
    year = _century(int(y), ref_year) if two else int(y)
    try:
        date = dt.date(year, m, d)
    except ValueError:
        return None
    if ords:
        roles["ord"] = ords[0]
    return DateVal(text, toks, date, year, "day", roles, two)


def _ordinal(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def render_date(dv: DateVal, new):
    """Write `new` (a date, or an int year for year precision) in the source layout: separators, zero padding,
    month-name abbreviation and case, 2- vs 4-digit year."""
    toks = list(dv.toks)
    year = new if isinstance(new, int) else new.year
    ty = toks[dv.roles["y"]]
    toks[dv.roles["y"]] = f"{year % 100:02d}" if len(ty) == 2 else str(year)
    if dv.precision == "day":
        td, tm = toks[dv.roles["d"]], toks[dv.roles["m"]]
        numeric = toks[dv.roles["m"]].isdigit()
        toks[dv.roles["d"]] = str(new.day).zfill(len(td)) if td.startswith("0") or (numeric and len(td) == 2) else str(new.day)
        if tm.isdigit():
            toks[dv.roles["m"]] = str(new.month).zfill(len(tm)) if tm.startswith("0") or len(tm) == 2 else str(new.month)
        else:
            name = MONTHS[new.month - 1]
            core = tm.rstrip(".")
            if core.lower() == "sept":
                name = "Sept" if new.month == 9 else name[:3]
            elif len(core) <= 4 and len(core) < len(MONTHS[dv.date.month - 1]) or len(core) == 3 and core.isupper():
                # 'MAY' in an upper-case '29 MAY 1947' ID layout is the 3-letter style, not the full name
                name = name[:3]
            if _month_index(core) is not None and not core.isascii():
                name = name[:3]
            name = name.upper() if core.isupper() else name.lower() if core.islower() else name
            toks[dv.roles["m"]] = name + ("." if tm.endswith(".") else "")
        if "ord" in dv.roles:
            o = toks[dv.roles["ord"]]
            suf = _ordinal(new.day)
            toks[dv.roles["ord"]] = suf.upper() if o.isupper() else suf
    return "".join(toks)


def age_at(born: dt.date, on: dt.date):
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


# ------------------------------------------------------------------------------------------------ document context
@dataclass
class L:
    id: str
    doc: str
    page: int
    text: str
    box: tuple | None = None


def lines_of(docs):
    """Pipeline Documents -> flat L list (box from the char quads when present)."""
    from .model import quad_bbox
    out = []
    for d in docs:
        for p in d.pages:
            for l in p.lines:
                if l.source == "native_hidden":
                    continue
                box = None
                if l.chars:
                    try:
                        bs = [quad_bbox(c.quad) for c in l.chars]
                        box = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
                    except Exception:
                        box = None
                out.append(L(l.id, d.id, p.index, l.text, box))
    return out


def lines_of_pages(pages, doc=None):
    """modeleval page index (dicts with lines/text/box) -> L list."""
    out = []
    for p in pages:
        for l in p.get("lines") or []:
            out.append(L(l.get("id") or f"{p.get('file')}:{len(out)}", doc or p.get("file"), p.get("page", 0), l["text"],
                         tuple(l["box"]) if l.get("box") else None))
    return out


DATE_RX = re.compile(r"(?<![\d])(?:\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}|"
                     r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})(?![\d])")
LBL_DOC = re.compile(r"\b(?:date\s+of\s+issue|issued?(?:\s+on)?|issue\s+date|registration|registered|statement\s+(?:date|period)|"
                     r"dated|date\s+signed|signed\s+on|date\s+of\s+service|service\s+date|filed|invoice\s+date|pay\s+date|"
                     r"date\s+of\s+notice|notice\s+date|date)\b|дата\s+выдачи|délivr", re.I)
LBL_ISSUE = re.compile(r"\b(?:date\s+of\s+issue|issued?|issue\s+date)\b|дата\s+выдачи|délivr|ausstell", re.I)
LBL_EXPIRY = re.compile(r"\b(?:expir\w*|exp\.?|valid\s+(?:until|thru|through)|date\s+of\s+expiry)\b|срок|действителен", re.I)
LBL_DOB = re.compile(r"\b(?:birth|born|d\.?o\.?b\.?|date\s+of\s+birth|naissance|geburt|nacimiento)\b|рождения", re.I)
LBL_AGE = re.compile(r"\b(?:age|aged)\b", re.I)
SEX_RX = re.compile(r"\b(?:sex|gender|пол)\b[^A-Za-zА-Яа-я]{0,6}(?:[^/\s]*/\s*)?\b([MFЖМ]|male|female)\b", re.I)
MONEY_AMT = re.compile(r"(?<![\d.])\$?\s?(\d{1,3}(?:,\d{3})*\.\d{2})(?![\d])")
STATE_NAMES = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
               "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
               "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
               "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
               "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
               "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
               "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
               "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
               "wisconsin": "WI", "wyoming": "WY"}
STATE_NAME_RX = re.compile(r"\b(" + "|".join(sorted(STATE_NAMES, key=len, reverse=True)) + r")\b,?\s+(?=\d{5}\b)", re.I)
STATE_OLD_RX = re.compile(r"\b([A-Z][a-z])\.\s+(?=\d{5}\b)")   # 'Austin, Tx. 78701'
STATE_RX = re.compile(r"\b(A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[AT]|W[AIVY])\b")


@dataclass
class DocContext:
    doc: str
    doc_date: dt.date | None = None        # issue / registration / statement / dated
    doc_date_source: str = ""             # 'labelled' | 'latest' | 'implied' | ''
    issue: dt.date | None = None
    expiry: dt.date | None = None
    era_year: int | None = None
    ages: list = field(default_factory=list)      # [(line_id, age:int)]
    sex: str | None = None                # 'f' | 'm'
    states: list = field(default_factory=list)    # [(line_id, 'NY')]

    def summary(self):
        return {"doc": self.doc, "doc_date": self.doc_date and self.doc_date.isoformat(), "doc_date_source": self.doc_date_source,
                "issue": self.issue and self.issue.isoformat(), "expiry": self.expiry and self.expiry.isoformat(),
                "era_year": self.era_year, "ages": [a for _, a in self.ages], "sex": self.sex}


def _near(a: L, b: L):
    """b is the value box for label a: same row to the right, or just below with horizontal overlap."""
    if a.box is None or b.box is None:
        return False
    ax0, ay0, ax1, ay1 = a.box; bx0, by0, bx1, by1 = b.box
    h = max(4.0, ay1 - ay0)
    same_row = abs((ay0 + ay1) / 2 - (by0 + by1) / 2) < h and bx0 >= ax1 - h and bx0 - ax1 < 8 * h
    below = -h < by0 - ay1 < 4 * h and by1 > ay1 and min(ax1, bx1) - max(ax0, bx0) > -h  # handwriting may overlap the label
    return same_row or below


def detect_context(doc_lines, dob_texts=()):
    """One document's unreplaced facts: document date, issue/expiry, stated ages, stated sex, US states."""
    if not doc_lines:
        return DocContext("")
    ctx = DocContext(doc_lines[0].doc)
    dobs = {t.strip() for t in dob_texts}
    labelled, plain, years = [], [], []
    for i, l in enumerate(doc_lines):
        prev = doc_lines[i - 1].text if i else ""
        for m in DATE_RX.finditer(l.text):
            v = m.group().strip()
            dv = parse_date(v)
            if dv is None or v in dobs:
                continue
            before = l.text[:m.start()]
            lbl = before if before.strip() else prev
            if LBL_DOB.search(lbl):
                continue
            if LBL_EXPIRY.search(lbl):
                ctx.expiry = ctx.expiry or dv.date
                continue
            if LBL_ISSUE.search(lbl):
                ctx.issue = ctx.issue or dv.date
            (labelled if LBL_DOC.search(lbl) else plain).append(dv.date)
            years.append(dv.date.year)
        for m in re.finditer(r"\b(1[89]\d\d|20\d\d)\b", l.text):
            if m.group() not in dobs:
                years.append(int(m.group()))
        m = LBL_AGE.search(l.text)
        if m:
            tail = re.search(r"\b(\d{1,3})\b", l.text[m.end():])
            if tail and 0 < int(tail.group(1)) < 125:
                ctx.ages.append((l.id, int(tail.group(1))))
            else:  # 'Age, in yrs.' label with the handwritten value in its own box
                for b in doc_lines[i + 1:i + 14]:
                    if re.fullmatch(r"\s*\d{1,3}\s*", b.text) and 0 < int(b.text) < 125 and _near(l, b):
                        ctx.ages.append((b.id, int(b.text))); break
        m = SEX_RX.search(l.text)
        if m and ctx.sex is None:
            v = m.group(1).lower()
            ctx.sex = "f" if v in ("f", "female", "ж") else "m"
        for s in STATE_RX.finditer(l.text):
            if re.search(r"\b\d{5}(?:-\d{4})?\b", l.text[s.end():s.end() + 12]):
                ctx.states.append((l.id, s.group()))
        for s in STATE_NAME_RX.finditer(l.text):
            ctx.states.append((l.id, STATE_NAMES[s.group(1).lower()]))
        for s in STATE_OLD_RX.finditer(l.text):
            if s.group(1).upper() in ZIP3:
                ctx.states.append((l.id, s.group(1).upper()))
    pool = labelled or [d for d in plain if d <= TODAY and (ctx.expiry is None or d < ctx.expiry)]
    if ctx.issue:
        ctx.doc_date, ctx.doc_date_source = ctx.issue, "labelled"
    elif labelled:
        ctx.doc_date, ctx.doc_date_source = max(d for d in labelled if d <= TODAY) if any(d <= TODAY for d in labelled) else None, "labelled"
    elif pool:
        ctx.doc_date, ctx.doc_date_source = max(pool), "latest"
    ys = sorted(y for y in years if y <= TODAY.year)
    ctx.era_year = ctx.doc_date.year if ctx.doc_date else (ys[len(ys) // 2] if ys else None)
    return ctx


# ------------------------------------------------------------------------------------------------ small validators
def luhn_ok(digits):
    ds = [int(c) for c in digits if c.isdigit()]
    if len(ds) < 12:
        return False
    tot = 0
    for i, x in enumerate(reversed(ds)):
        tot += sum(divmod(2 * x, 10)) if i % 2 else x
    return tot % 10 == 0


def card_network(digits):
    d = re.sub(r"\D", "", digits)
    if re.match(r"3[47]", d): return "amex"
    if re.match(r"4", d): return "visa"
    if re.match(r"5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)", d): return "mastercard"
    if re.match(r"6(?:011|5|4[4-9])", d): return "discover"
    if re.match(r"35", d): return "jcb"
    if re.match(r"3(?:0[0-5]|[68])", d): return "diners"
    return "other:" + d[:1]


ZIP3 = {"AL": [(350, 369)], "AK": [(995, 999)], "AZ": [(850, 865)], "AR": [(716, 729)], "CA": [(900, 961)], "CO": [(800, 816)],
        "CT": [(60, 69)], "DE": [(197, 199)], "DC": [(200, 205)], "FL": [(320, 349)], "GA": [(300, 319), (398, 399)],
        "HI": [(967, 968)], "ID": [(832, 838)], "IL": [(600, 629)], "IN": [(460, 479)], "IA": [(500, 528)], "KS": [(660, 679)],
        "KY": [(400, 427)], "LA": [(700, 714)], "ME": [(39, 49)], "MD": [(206, 219)], "MA": [(10, 27), (55, 55)],
        "MI": [(480, 499)], "MN": [(550, 567)], "MS": [(386, 397)], "MO": [(630, 658)], "MT": [(590, 599)], "NE": [(680, 693)],
        "NV": [(889, 898)], "NH": [(30, 38)], "NJ": [(70, 89)], "NM": [(870, 884)], "NY": [(5, 5), (100, 149)],
        "NC": [(270, 289)], "ND": [(580, 588)], "OH": [(430, 459)], "OK": [(730, 749)], "OR": [(970, 979)], "PA": [(150, 196)],
        "RI": [(28, 29)], "SC": [(290, 299)], "SD": [(570, 577)], "TN": [(370, 385)], "TX": [(750, 799), (885, 885)],
        "UT": [(840, 847)], "VT": [(50, 59)], "VA": [(201, 201), (220, 246)], "WA": [(980, 994)], "WV": [(247, 268)],
        "WI": [(530, 549)], "WY": [(820, 831)]}


def zip_in_state(z, st):
    rng = ZIP3.get(st)
    if not rng or not re.match(r"\d{3}", z or ""):
        return True
    p = int(z[:3])
    return any(a <= p <= b for a, b in rng)


def nanp_area_ok(phone):
    ds = re.sub(r"\D", "", phone)
    if len(ds) == 11 and ds[0] == "1":
        ds = ds[1:]
    if len(ds) != 10:
        return True  # not a NANP number: nothing to check
    return ds[0] in "23456789" and ds[1:3] != "11" and ds[3] in "23456789"


def name_like(text):
    """A replacement personal/organisation name must look like a name: pronounceable words, not 'JNFRHWAE'."""
    for w in re.findall(r"[^\W\d_]+", text):
        if not w.isascii() or len(w) <= 2 or (w.isupper() and len(w) <= 4 and not re.search(r"[AEIOUY]", w)):
            continue  # initials and abbreviations ('LLC', 'SDN', 'BHD', 'PLC') are not words to pronounce
        lw = w.lower()
        vow = sum(c in "aeiouy" for c in lw)
        if vow == 0 or vow / len(lw) < 0.2 or re.search(r"[^aeiouy]{5,}", lw) or re.search(r"(.)\1\1", lw) or re.match(r"([^aeiouy])\1", lw):
            return False
    return True


# ------------------------------------------------------------------------------------------------ plausibility (single value)
def plausibility(kind, original, replacement, ctx: DocContext | None = None, partner_dob=None):
    """Context checks on ONE planned replacement -> list of issue codes (empty = plausible). Used by the edit bench and
    the review screen; the graph checker covers relations between values."""
    issues = []
    if replacement is None or original is None:
        return issues
    if replacement == original:
        issues.append("UNCHANGED")
    if kind != "DOB":
        od = parse_date(original)
        if od is not None and od.precision == "day" and parse_date(replacement, loose=True) is None:
            issues.append("DATE_NOT_A_DATE")
    if kind == "DOB":
        o, n = parse_date(original, ctx and ctx.era_year), parse_date(replacement, ctx and ctx.era_year, loose=True)
        if o is not None and n is None:
            issues.append("DATE_NOT_A_DATE")
            return issues
        if o is None or n is None:
            return issues
        if (n.precision, n.two_digit_year) != (o.precision, o.two_digit_year) or len(replacement) != len(original) and o.precision == "year":
            issues.append("DATE_FORMAT_CHANGED")
        if re.sub(r"\d", "0", replacement) != re.sub(r"\d", "0", original) and o.toks[o.roles.get("m", 0)].isdigit():
            if [t for t in n.toks if not t.isdigit()] != [t for t in o.toks if not t.isdigit()]:
                issues.append("DATE_FORMAT_CHANGED")
        if n.year > TODAY.year or (n.date and n.date > TODAY):
            issues.append("DOB_IN_FUTURE")
        if abs(n.year - o.year) > 10:
            issues.append("DATE_ERA_JUMP")
        if ctx is not None:
            ref = ctx.doc_date
            if ref and (n.date or dt.date(n.year, 1, 1)) > ref and (o.date or dt.date(o.year, 1, 1)) <= ref:
                issues.append("DOB_AFTER_DOC_DATE")
            if ctx.issue and (n.date or dt.date(n.year, 1, 1)) >= ctx.issue and (o.date or dt.date(o.year, 1, 1)) < ctx.issue:
                issues.append("DOB_AFTER_ISSUE")
            for _, age in ctx.ages:
                if _age_ok(o, age, ctx) and not _age_ok(n, age, ctx, implied_from=o):
                    issues.append("AGE_MISMATCH"); break
    elif kind in ("PERSON", "GIVEN", "SURNAME", "ORG"):
        if not name_like(replacement):
            issues.append("NAME_NOT_PLAUSIBLE")
    elif kind == "CARD":
        if luhn_ok(original) and not luhn_ok(replacement):
            issues.append("CARD_LUHN")
        if card_network(original) != card_network(replacement):
            issues.append("CARD_NETWORK_CHANGED")
    elif kind == "PHONE":
        if nanp_area_ok(original) and not nanp_area_ok(replacement):
            issues.append("PHONE_AREA_INVALID")
    elif kind == "ZIP" and ctx is not None:
        for _, st in ctx.states:
            if zip_in_state(original, st) and not zip_in_state(replacement, st):
                issues.append("ZIP_STATE_MISMATCH"); break
    elif kind == "ADDRESS":
        if not name_like(re.sub(r"\b\d+\w*\b", "", replacement)) or not re.match(r"\d", replacement) and re.match(r"\d", original):
            issues.append("ADDRESS_NOT_PLAUSIBLE")
    if kind not in ("EMAIL", "MRZ") and re.sub(r"\w", "a", replacement) != re.sub(r"\w", "a", original) and kind in ("ID", "ACCOUNT", "CARD", "PHONE", "ZIP"):
        issues.append("FORMAT_CHANGED")
    return issues


def _age_window(dv: DateVal, age):
    """Document dates on which a person born `dv` is `age` years old: [lo, hi)."""
    if dv.precision == "day":
        def bday(y):  # 29 Feb: the birthday is 29 Feb in leap years, 1 Mar otherwise (same rule as age_at)
            try:
                return dv.date.replace(year=y)
            except ValueError:
                return dt.date(y, 3, 1)
        return bday(dv.date.year + age), bday(dv.date.year + age + 1)
    return dt.date(dv.year + age, 1, 1), dt.date(dv.year + age + 2, 1, 1)


def _age_ok(dv, age, ctx, implied_from=None):
    """Stated age consistent with DOB and the document date. Undated document: the date implied by the ORIGINAL DOB
    and age stays fixed (a 1917 card stays a 1917 card), so the replacement must give the same age then."""
    if dv is None:
        return True
    lo, hi = _age_window(dv, age)
    if ctx.doc_date:
        return lo <= ctx.doc_date < hi
    if implied_from is None:
        return True
    plo, phi = _age_window(implied_from, age)
    return lo < phi and plo < hi


# ------------------------------------------------------------------------------------------------ the fact graph
@dataclass
class Rel:
    type: str
    occs: tuple                    # occurrence ids
    data: dict = field(default_factory=dict)
    source: str = "rules"          # 'rules' | 'model'


@dataclass
class FactGraph:
    contexts: dict                 # doc -> DocContext
    entities: dict                 # entity id -> {"kind", "occs": [ids], "variants": [texts]}
    relations: list
    occ: dict                      # id -> Occ
    line_doc: dict = field(default_factory=dict)

    def summary(self):
        by = {}
        for r in self.relations:
            by[r.type] = by.get(r.type, 0) + 1
        return {"documents": {d: c.summary() for d, c in self.contexts.items()}, "entities": len(self.entities),
                "relations": by}


MODEL_RELATION_TYPES = {"SAME_VALUE", "AGE_DOB", "DOB_DOC", "EMAIL_NAME", "GENDER_NAME", "CARD_LAST4", "ACCOUNT_LAST4"}


def _person_tokens(o):
    return [w.lower() for w in re.findall(r"[^\W\d_]+", o.text) if len(w) > 1]


def build_graph(lines, occs):
    """lines: L list for every document in the case; occs: resolved occurrences (resolve.Occ)."""
    by_doc = {}
    for l in lines:
        by_doc.setdefault(l.doc, []).append(l)
    occ = {o.id: o for o in occs}
    dob_texts = [o.text for o in occs if o.kind == "DOB"]
    contexts = {d: detect_context(ls, dob_texts) for d, ls in by_doc.items()}
    for o in occs:
        contexts.setdefault(o.doc, DocContext(o.doc))
    ents = {}
    for o in occs:
        e = ents.setdefault(o.entity or f"{o.kind.lower()}:{o.text.lower()}", {"kind": o.kind, "occs": [], "variants": []})
        e["occs"].append(o.id)
        if o.text not in e["variants"]:
            e["variants"].append(o.text)
    rels = []
    # identity: same source value -> same fake; distinct source values of one kind stay distinct (mismatches survive)
    groups = {}
    for o in occs:
        groups.setdefault((o.kind, o.text), []).append(o.id)
    for (k, t), ids in groups.items():
        if len(ids) > 1:
            rels.append(Rel("SAME_VALUE", tuple(ids), {"kind": k}))
    for k in {o.kind for o in occs if o.kind in ("PERSON", "GIVEN", "SURNAME", "DOB", "ID", "ACCOUNT", "CARD", "EMAIL", "PHONE", "ZIP")}:
        reps = [ids[0] for (kk, _), ids in groups.items() if kk == k]
        if len(reps) > 1:
            rels.append(Rel("DISTINCT_VALUE", tuple(reps), {"kind": k}))
    # one person's name variants (full, Last-First, ALL CAPS, initials, surname alone, MRZ name) share one fake surname
    for eid, e in ents.items():
        ids = [i for i in e["occs"] if occ[i].kind in ("PERSON", "SURNAME")]
        if len(ids) > 1 and len({occ[i].text for i in ids}) > 1:
            rels.append(Rel("NAME_VARIANTS", tuple(ids), {"entity": eid}))
    # dates
    dobs = [o for o in occs if o.kind == "DOB"]
    by_key = {}
    for o in dobs:
        dv = parse_date(o.text, contexts[o.doc].era_year)
        if dv is None:
            continue
        by_key.setdefault(dv.key, []).append(o.id)
        c = contexts[o.doc]
        rels.append(Rel("DOB_DOC", (o.id,), {"doc": o.doc}))
        if c.issue:
            rels.append(Rel("DOB_ISSUE", (o.id,), {"issue": c.issue.isoformat()}))
        for lid, age in c.ages:
            rels.append(Rel("AGE_DOB", (o.id,), {"age": age, "age_line": lid}))
    for key, ids in by_key.items():
        if len(ids) > 1:
            rels.append(Rel("SAME_DATE", tuple(ids), {"key": list(key)}))
    for d, c in contexts.items():
        if c.issue and c.expiry:
            rels.append(Rel("ISSUE_EXPIRY", (), {"doc": d, "issue": c.issue.isoformat(), "expiry": c.expiry.isoformat()}))
    # MRZ
    for o in occs:
        if o.kind != "MRZ":
            continue
        rels.append(Rel("MRZ_CHECK", (o.id,)))
        same_doc_dobs = [x.id for x in dobs if x.doc == o.doc]
        if same_doc_dobs and re.search(r"\d{6}\d[MFX<]\d{6}\d", re.sub(r"\s", "", o.text)):
            rels.append(Rel("MRZ_VISUAL", (o.id,) + tuple(same_doc_dobs)))
    # cards / accounts
    cards = [o for o in occs if o.kind == "CARD"]
    for o in cards:
        rels.append(Rel("CARD_LUHN", (o.id,))); rels.append(Rel("CARD_NETWORK", (o.id,)))
        for x in occs:
            if x.kind == "CARD_LAST4" and x.text.strip() == re.sub(r"\D", "", o.text)[-4:]:
                rels.append(Rel("CARD_LAST4", (o.id, x.id)))
    accts = [o for o in occs if o.kind == "ACCOUNT"]
    for a in accts:
        da = re.sub(r"\D", "", a.text)
        for b in accts:
            db = re.sub(r"\D", "", b.text)
            if a.id < b.id and da != db and len(min(da, db, key=len)) >= 4 and (da.endswith(db[-4:]) and (len(db) <= 4 or len(da) <= 4)):
                rels.append(Rel("ACCOUNT_LAST4", (a.id, b.id)))
    # addresses / phones
    for o in occs:
        c = contexts[o.doc]
        if o.kind == "ZIP":
            st = next((s for lid, s in c.states if lid == o.line_id), None)
            if st:
                rels.append(Rel("ZIP_STATE", (o.id,), {"state": st}))
        elif o.kind == "PHONE" and len(re.sub(r"\D", "", o.text)) >= 10:
            rels.append(Rel("PHONE_AREA", (o.id,)))
    # email local part derived from a name in the case
    names = [o for o in occs if o.kind in ("PERSON", "GIVEN", "SURNAME")]
    for e in (o for o in occs if o.kind == "EMAIL"):
        toks = set(re.findall(r"[a-z]+", e.text.partition("@")[0].lower()))
        for n in names:
            hit = [t for t in _person_tokens(n) if t in toks]
            if hit:
                rels.append(Rel("EMAIL_NAME", (e.id, n.id), {"tokens": hit}))
    # stated sex on the document -> given-name style
    for o in names:
        c = contexts[o.doc]
        if c.sex and o.kind in ("PERSON", "GIVEN"):
            rels.append(Rel("GENDER_NAME", (o.id,), {"sex": c.sex}))
    # amounts summing to a total (amounts are never replaced; the relation is re-read after the edit)
    for d, ls in by_doc.items():
        amts = []
        for l in ls:
            for m in MONEY_AMT.finditer(l.text):
                amts.append((l, float(m.group(1).replace(",", "")), m.group(1)))
        for i, (l, v, s) in enumerate(amts):
            if not re.search(r"\btotal\b", l.text, re.I) or i < 2:
                continue
            for j in range(i - 1, max(-1, i - 25), -1):
                run = amts[j:i]
                if len(run) >= 2 and abs(sum(x[1] for x in run) - v) < 0.005:
                    rels.append(Rel("AMOUNT_SUM", (), {"doc": d, "total": s, "items": [x[2] for x in run]})); break
    return FactGraph(contexts, ents, rels, occ, {l.id: l.doc for l in lines})


def propose_relations(graph: FactGraph, client=None):
    """Optional BYOK model step (off by default). `client.propose_relations(summary)` returns
    [{"type": T, "occs": [ids], "data": {...}}]; unknown types or ids are dropped. Local rules never depend on it."""
    if client is None:
        return []
    summary = {"relations": graph.summary()["relations"],
               "occurrences": [{"id": o.id, "doc": o.doc, "kind": o.kind} for o in graph.occ.values()]}  # no source values
    out = []
    for r in client.propose_relations(summary) or []:
        ids = tuple(i for i in r.get("occs", []) if i in graph.occ)
        if r.get("type") in MODEL_RELATION_TYPES and ids:
            out.append(Rel(r["type"], ids, dict(r.get("data") or {}), source="model"))
    graph.relations += out
    return out


# ------------------------------------------------------------------------------------------------ planner
class DatePlanner:
    """Per-case date plan attached to a CaseMap (cmap.dates). Keyed by date value, so every format of one birth date
    gets the same fictional date; constraints come from every document the value appears in."""

    def __init__(self, cmap):
        self.cmap = cmap
        self.constraints = {}     # key -> list of dicts (doc_date, issue, ages, era, implied)
        self.plan = {}            # key -> date | int year
        self.texts = {}           # key -> set of source texts (for the same-length preference)
        self.notes = {}           # key -> note for the review screen
        self.text_ctx = {}        # source text -> DocContext (era for 2-digit years)

    def add(self, dv: DateVal, ctx: DocContext | None):
        self.texts.setdefault(dv.key, set()).add(dv.text)
        self.text_ctx[dv.text] = ctx
        if ctx is not None:
            self.constraints.setdefault(dv.key, []).append(ctx)

    def _ok(self, dv: DateVal, cand, strict=True):
        n = dv.date if dv.precision == "day" else None
        ny = cand if isinstance(cand, int) else cand.year
        nd = cand if not isinstance(cand, int) else dt.date(ny, 1, 1)
        if ny > TODAY.year or (not isinstance(cand, int) and cand > TODAY):
            return False
        if abs(ny - dv.year) > MAX_YEAR_SHIFT:
            return False
        new_dv = DateVal(dv.text, dv.toks, None if isinstance(cand, int) else cand, ny, dv.precision, dv.roles, dv.two_digit_year)
        orig_d = dv.date or dt.date(dv.year, 1, 1)
        for c in self.constraints.get(dv.key, []):
            if c.doc_date and orig_d <= c.doc_date and (ny > c.doc_date.year if isinstance(cand, int) else nd > c.doc_date):
                return False
            if c.issue and orig_d < c.issue and nd >= c.issue:
                return False
            for _, age in c.ages:
                if _age_ok(dv, age, c) and not _age_ok(new_dv, age, c, implied_from=dv):
                    return False
        if strict:  # same rendered length everywhere (layout fidelity, no reflow)
            for t in self.texts.get(dv.key, {dv.text}):
                src = parse_date(t)
                if src is not None and len(render_date(src, cand)) != len(t):
                    return False
        return True

    def fake(self, dv: DateVal):
        if dv.key in self.plan:
            return self.plan[dv.key]
        cands = []
        base = self.cmap._n("dob-plan", repr(dv.key))
        if dv.precision == "year":
            # a full DOB with the same year elsewhere in the case: the boxed year follows that date's plan
            full = [k for k in self.plan if k[0] == "d" and int(k[1][:4]) == dv.year]
            if full:
                y = self.plan[full[0]].year
                if self._ok(dv, y, strict=False):
                    self.plan[dv.key] = y
                    return y
            ks = [k for k in range(-MAX_YEAR_SHIFT, MAX_YEAR_SHIFT + 1) if k]
            ks = sorted(ks, key=lambda k: (self.cmap._n("dob-year", str(dv.year), str(k)) + base) % 9973)
            cands = [dv.year + k for k in ks]
        else:
            lo, hi = DAY_SHIFT
            for s in range(240):
                n = self.cmap._n("dob-day", dv.date.isoformat(), str(s))
                shift = lo + n % (hi - lo)
                cands.append(dv.date + dt.timedelta(days=shift if (n >> 20) % 2 else -shift))
            # small shifts last: they keep a stated age when the document date is close to the birthday
            cands += [dv.date + dt.timedelta(days=k) for k in (7, -7, 3, -3, 11, -11, 1, -1)]
        pick = next((c for c in cands if self._ok(dv, c, True)), None) or next((c for c in cands if self._ok(dv, c, False)), None)
        if pick is None:
            # nothing satisfies every constraint: keep the year (and age) and move the day/month, else nearest year
            if dv.precision == "day":
                pick = next((c for c in cands if c.year == dv.year and c <= TODAY), cands[0])
            else:
                pick = cands[0] if cands[0] <= TODAY.year else dv.year - 1
            self.notes[dv.key] = "NO_FULLY_CONSISTENT_CANDIDATE"
        self.plan[dv.key] = pick
        return pick

    def replace(self, text, ctx=None):
        ctx = ctx or self.text_ctx.get(text)
        dv = parse_date(text, ctx.era_year if ctx else None)
        if dv is None:
            return None
        return render_date(dv, self.fake(dv))

    def fake_date(self, d: dt.date):
        """A birth date read from an MRZ (YYMMDD) -> the same fictional date as the visual zone."""
        key = ("d", d.isoformat())
        if key in self.plan:
            return self.plan[key]
        dv = DateVal(d.strftime("%d.%m.%Y"), _TOK.findall(d.strftime("%d.%m.%Y")), d, d.year, "day", {"d": 0, "m": 2, "y": 4})
        return self.fake(dv)


def dates(cmap):
    if not hasattr(cmap, "_dates"):
        cmap._dates = DatePlanner(cmap)
    return cmap._dates


def plan(cmap, graph: FactGraph, occs, order=None):
    """Register constraints, assign every occurrence's replacement through the CaseMap, apply the cross-value fix-ups
    (account last four), then check. Returns the violation list (also appended to each occurrence's review)."""
    dp = dates(cmap)
    for o in occs:
        if o.kind == "DOB":
            c = graph.contexts.get(o.doc)
            dv = parse_date(o.text, c.era_year if c else None)
            if dv is not None:
                dp.add(dv, c)
    for rel in graph.relations:  # stated sex -> given-name pool styling
        if rel.type == "GENDER_NAME":
            o = graph.occ[rel.occs[0]]
            from .resolve import parse_person
            p = parse_person(o.text) if o.kind == "PERSON" else {"given": o.text.split()[0] if o.text.split() else ""}
            if p and p.get("given") and len(p["given"]) > 1:
                cmap.style_hint[p["given"].lower()] = rel.data["sex"]
    # full dates before boxed years (a boxed year follows its full date's plan), MRZ after DOBs and names
    seq = order or sorted(occs, key=lambda o: (o.kind != "PERSON", o.kind == "MRZ",
                                                 o.kind == "DOB" and (parse_date(o.text) or DateVal("", [], None, 0, "day", {})).precision == "year"))
    for o in seq:
        o.replacement = cmap.replace(o.kind, o.text)
        if o.kind not in ("DOB", "MRZ"):
            od = parse_date(o.text)
            if od is not None and od.precision == "day" and parse_date(o.replacement, loose=True) is None:
                o.replacement = dp.replace(o.text) or o.replacement  # an ID-typed date stays a real, same-era date
    for rel in graph.relations:
        if rel.type == "ACCOUNT_LAST4":
            a, b = (graph.occ[i] for i in rel.occs)
            full, short = (a, b) if len(re.sub(r"\D", "", a.text)) > len(re.sub(r"\D", "", b.text)) else (b, a)
            last = re.sub(r"\D", "", full.replacement)[-4:]
            it = iter(last[-len(re.sub(r"\D", "", short.text)[-4:]):])
            n_short = len(re.sub(r"\D", "", short.text))
            ds = re.sub(r"\D", "", short.replacement)
            new = ds[:max(0, n_short - 4)] + last[-min(4, n_short):]
            it = iter(new)
            short.replacement = "".join(next(it) if ch.isdigit() else ch for ch in short.text)
    # SSN last four: the same fake as the last four of the case's full SSN (when the full number is in the case)
    full_ssn = {}
    for o in occs:
        ds = re.sub(r"\D", "", o.text)
        if o.kind in ("SSN", "ID") and re.fullmatch(r"\d{3}[- ]?\d{2}[- ]?\d{4}", o.text.strip()) and o.replacement:
            full_ssn.setdefault(ds[-4:], re.sub(r"\D", "", o.replacement)[-4:])
    for o in occs:
        if o.kind == "SSN_LAST4" and o.text in full_ssn and len(full_ssn[o.text]) == 4:
            o.replacement = full_ssn[o.text]
    link_partial_identifiers(occs)
    vs = check(graph, {o.id: o.replacement for o in occs})
    for v in vs:
        for i in v["occs"]:
            o = graph.occ.get(i)
            if o is not None:
                tag = "CONSISTENCY:" + v["code"]
                if tag not in o.review:
                    o.review.append(tag)
    return vs


_PARTIAL_KINDS = {"CARD_LAST4", "SSN_LAST4", "ACCOUNT", "ID", "MEMBER_ID", "PHONE"}
_CONTEXT_KIND = (("CARD", r"card|visa|mastercard|amex|discover|debit|credit"), ("PHONE", r"phone|mobile|cell|tel|number ending|text"),
                 ("ACCOUNT", r"account|acct|checking|savings"), ("SSN", r"ssn|social"))


def link_partial_identifiers(occs, line_text=None):
    """A partial identifier (the last 2-6 digits of a card, account, phone, member ID...) gets the matching digits of
    its full value's replacement, so '**** 0187', 'phone_last4: "0187"' and '(541) 555-0187' stay one number in every
    document, trace, tool call and expected answer. Full values: 7+ digits, any identifier kind. When several full
    values end the same way with different fakes, the one of the occurrence's own kind wins; otherwise it is left as
    planned (independent fake) and flagged for review."""
    from .resolve import SUFFIX_SOURCE_KINDS
    full = {}
    for o in occs:
        ds = re.sub(r"\D", "", o.text)
        if o.kind in SUFFIX_SOURCE_KINDS and len(ds) >= 7 and o.replacement:
            rd = re.sub(r"\D", "", o.replacement)
            if len(rd) == len(ds):
                full.setdefault(ds, (o.kind, rd))
    if not full:
        return
    for o in occs:
        if o.kind not in _PARTIAL_KINDS:
            continue
        core = re.sub(r"[\s*xX\u2022#\-.]", "", o.text)
        if not re.fullmatch(r"\d{2,6}", core) or len(re.sub(r"\D", "", o.text)) >= 7:
            continue
        cands = {(k, rd[-len(core):]) for ds, (k, rd) in full.items() if ds.endswith(core) and ds != core}
        if not cands:
            continue
        fakes = {f for _, f in cands}
        if len(fakes) > 1:
            own = {f for k, f in cands if k == o.kind}
            fakes = own if len(own) == 1 else fakes
        if len(fakes) != 1:
            if "PARTIAL_ID_AMBIGUOUS" not in o.review:
                o.review.append("PARTIAL_ID_AMBIGUOUS")
            continue
        fake = fakes.pop()
        if fake == core:
            continue  # the full fake happens to keep these digits: keep the planned independent fake rather than the original
        it = iter(fake)
        o.replacement = "".join(next(it) if ch.isdigit() else ch for ch in o.text)


# ------------------------------------------------------------------------------------------------ checker
def _rep(repl, o):
    return repl.get(o.id, getattr(o, "replacement", None))


def check(graph: FactGraph, repl: dict):
    """Every relation on original vs replacement values -> violations [{code, occs, detail}]. A relation that did
    not hold in the source is never reported (intentional mismatches are preserved, not repaired)."""
    out = []
    occ = graph.occ

    def bad(code, ids, **detail):
        out.append({"code": code, "occs": list(ids), "detail": detail})

    for o in occ.values():  # single-value plausibility in context
        r = _rep(repl, o)
        if r is None or getattr(o, "meta", {}).get("sad"):
            continue
        for code in plausibility(o.kind, o.text, r, graph.contexts.get(o.doc)):
            if code in ("AGE_MISMATCH", "DOB_AFTER_DOC_DATE", "DOB_AFTER_ISSUE", "DOB_IN_FUTURE"):
                continue  # reported by the relation checks below with the right occurrences
            bad(code, [o.id], kind=o.kind)
    for rel in graph.relations:
        t, ids = rel.type, rel.occs
        os_ = [occ[i] for i in ids if i in occ]
        rs = [_rep(repl, o) for o in os_]
        if t == "SAME_VALUE":
            if len({r for r in rs if r is not None}) > 1:
                bad("SAME_VALUE_SPLIT", ids, kind=rel.data.get("kind"))
        elif t == "DISTINCT_VALUE":
            seen = {}
            for o, r in zip(os_, rs):
                if r is None:
                    continue
                if r in seen and seen[r].text != o.text:
                    bad("MISMATCH_COLLAPSED", [seen[r].id, o.id], kind=o.kind)
                seen.setdefault(r, o)
        elif t in ("DOB_DOC", "DOB_ISSUE", "AGE_DOB"):
            o, r = os_[0], rs[0]
            c = graph.contexts.get(o.doc)
            dv0, dv1 = parse_date(o.text, c.era_year), parse_date(r or "", c.era_year, loose=True)
            if dv0 is None or dv1 is None:
                continue
            d0, d1 = dv0.date or dt.date(dv0.year, 1, 1), dv1.date or dt.date(dv1.year, 1, 1)
            if t == "DOB_DOC":
                if d1 > TODAY or dv1.year > TODAY.year:
                    bad("DOB_IN_FUTURE", ids, new=r)
                if c.doc_date and d0 <= c.doc_date and (d1 > c.doc_date or dv1.year > c.doc_date.year):
                    bad("DOB_AFTER_DOC_DATE", ids, doc_date=c.doc_date.isoformat(), new=r)
            elif t == "DOB_ISSUE":
                iss = dt.date.fromisoformat(rel.data["issue"])
                if d0 < iss <= d1:
                    bad("DOB_AFTER_ISSUE", ids, issue=rel.data["issue"], new=r)
            else:
                age = rel.data["age"]
                if _age_ok(dv0, age, c) and not _age_ok(dv1, age, c, implied_from=dv0):
                    bad("AGE_MISMATCH", ids, age=age, new=r, doc_date=c.doc_date and c.doc_date.isoformat())
        elif t == "SAME_DATE":
            keys = set()
            for o, r in zip(os_, rs):
                dv = parse_date(r or "", graph.contexts[o.doc].era_year)
                keys.add(dv.key if dv else None)
            if len(keys) > 1:
                bad("SAME_DATE_SPLIT", ids)
        elif t == "ISSUE_EXPIRY":
            pass  # both are unreplaced document facts; re-checked on the output by verify_outputs
        elif t == "MRZ_CHECK":
            from .mrz import check_digit
            if rs[0] and not mrz_checks_ok(rs[0]) and mrz_checks_ok(os_[0].text):
                bad("MRZ_CHECK_DIGIT", ids)
        elif t == "MRZ_VISUAL":
            m0, mr = os_[0].text, rs[0]
            for o, r in zip(os_[1:], rs[1:]):
                if _mrz_dob_matches(m0, o.text) and not _mrz_dob_matches(mr or "", r or ""):
                    bad("MRZ_VISUAL_MISMATCH", [os_[0].id, o.id], visual=r)
        elif t == "CARD_LUHN":
            if luhn_ok(os_[0].text) and rs[0] and not luhn_ok(rs[0]):
                bad("CARD_LUHN", ids)
        elif t == "CARD_NETWORK":
            if rs[0] and card_network(os_[0].text) != card_network(rs[0]):
                bad("CARD_NETWORK_CHANGED", ids)
        elif t in ("CARD_LAST4", "ACCOUNT_LAST4"):
            a, b = rs
            if a and b:
                da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
                k = min(4, len(da), len(db))
                if da[-k:] != db[-k:]:
                    bad(t + "_MISMATCH", ids)
        elif t == "ZIP_STATE":
            if rs[0] and zip_in_state(os_[0].text, rel.data["state"]) and not zip_in_state(rs[0], rel.data["state"]):
                bad("ZIP_STATE_MISMATCH", ids, state=rel.data["state"])
        elif t == "PHONE_AREA":
            if rs[0] and nanp_area_ok(os_[0].text) and not nanp_area_ok(rs[0]):
                bad("PHONE_AREA_INVALID", ids)
        elif t == "EMAIL_NAME":
            e, n = os_
            re_, rn = rs
            if re_ and rn:
                fake_toks = {w.lower() for w in re.findall(r"[^\W\d_]+", rn)}
                local = re_.partition("@")[0].lower()
                src_toks = _person_tokens(n)
                # every name token the source email used must come back as the fake name's token
                pos = [src_toks.index(tk) for tk in rel.data["tokens"] if tk in src_toks]
                fk = [w.lower() for w in re.findall(r"[^\W\d_]+", rn)]
                if any(i < len(fk) and fk[i] not in local for i in pos):
                    bad("EMAIL_NAME_MISMATCH", ids)
        elif t == "GENDER_NAME":
            pass  # checked below with the CaseMap's pool styling when available
        elif t == "NAME_VARIANTS":
            from .resolve import parse_person
            seen = {}
            for o, r in zip(os_, rs):
                if not r:
                    continue
                po, pr = (parse_person(o.text) if o.kind == "PERSON" else {"surname": o.text}), \
                         (parse_person(r) if o.kind == "PERSON" else {"surname": r})
                if not po or not pr:
                    continue
                k = po["surname"].lower()
                if k in seen and seen[k] != pr["surname"].lower():
                    bad("NAME_VARIANT_SPLIT", ids, surname_form=o.kind); break
                seen.setdefault(k, pr["surname"].lower())
    return out


def mrz_checks_ok(text):
    from .mrz import check_digit
    t = re.sub(r"\s", "", text)
    m = re.match(r"^([A-Z0-9<]{9})(\d)[A-Z<]{3}(\d{6})(\d)[MFX<](\d{6})(\d)", t)
    if m:
        return all(check_digit(a) == b for a, b in ((m.group(1), m.group(2)), (m.group(3), m.group(4)), (m.group(5), m.group(6))))
    m = re.match(r"^(\d{6})(\d)[MFX<](\d{6})(\d)", t)
    if m:
        return check_digit(m.group(1)) == m.group(2) and check_digit(m.group(3)) == m.group(4)
    return True


def _mrz_dob6(text):
    t = re.sub(r"\s", "", text)
    m = re.match(r"^[A-Z0-9<]{9}\d[A-Z<]{3}(\d{6})\d", t) or re.match(r"^(\d{6})\d[MFX<]", t)
    return m.group(1) if m else None


def _mrz_dob_matches(mrz_text, visual):
    s = _mrz_dob6(mrz_text)
    dv = parse_date(visual)
    if not s or dv is None or dv.precision != "day":
        return False
    return dv.date.strftime("%y%m%d") == s


def gender_violations(graph: FactGraph, cmap):
    out = []
    for rel in graph.relations:
        if rel.type != "GENDER_NAME":
            continue
        o = graph.occ[rel.occs[0]]
        r = o.replacement or ""
        w = re.findall(r"[^\W\d_]+", r)
        if not w:
            continue
        first = w[1] if o.kind == "PERSON" and "," in r and len(w) > 1 else w[0]
        style = cmap.given_style.get(first.lower())
        if style and style != rel.data["sex"]:
            out.append({"code": "GENDER_NAME_MISMATCH", "occs": [o.id], "detail": {"sex": rel.data["sex"]}})
    return out


# ------------------------------------------------------------------------------------------------ post-edit verifier
def _fuzzy_find(value, text, thr=0.85):
    import difflib
    v = re.sub(r"\s+", " ", value.strip().lower())
    t = re.sub(r"\s+", " ", text.lower())
    if not v:
        return None
    if v in t:
        return value
    n = len(v)
    best = (0, None)
    for i in range(0, max(1, len(t) - n + 1)):
        w = t[i:i + n]
        r = difflib.SequenceMatcher(None, v, w).ratio()
        if r > best[0]:
            best = (r, text[i:i + n] if len(text) == len(t) else w)
    return best[1] if best[0] >= thr else None


def verify_outputs(graph: FactGraph, out_text: dict, repl: dict | None = None):
    """out_text: {doc: re-read text of the OUTPUT (OCR of pixels + text layer)}. Finds each replacement as printed,
    then re-checks every relation on what was actually read; also MRZ check digits of every MRZ line read back,
    issue < expiry, and amount sums. -> {"checked": n, "violations": [...], "not_found": [ids]}"""
    repl = repl or {o.id: o.replacement for o in graph.occ.values()}
    read, not_found = {}, []
    for o in graph.occ.values():
        r = repl.get(o.id)
        if not r or getattr(o, "meta", {}).get("sad"):
            continue
        txt = out_text.get(o.doc, "")
        if o.kind == "MRZ":
            t = re.sub(r"\s", "", r)
            lines = [re.sub(r"\s", "", x) for x in txt.splitlines()]
            got = max(lines, key=lambda x: _sim(x, t), default="")
            if got and _sim(got, t) >= 0.8:
                read[o.id] = got
            else:
                not_found.append(o.id)
            continue
        got = _fuzzy_find(r, txt)
        if got is None:
            not_found.append(o.id)
        else:
            read[o.id] = got if o.kind == "DOB" and parse_date(got, loose=True) else r
    vs = [v for v in check(graph, read) if all(i in read for i in v["occs"])]
    for d, c in graph.contexts.items():
        txt = out_text.get(d)
        if txt is None:
            continue
        for line in txt.splitlines():
            t = re.sub(r"\s", "", line)
            if re.fullmatch(r"[A-Z0-9<]{28,46}", t) and re.search(r"\d{6}\d[MFX<]\d{6}\d", t) and not mrz_checks_ok(t):
                vs.append({"code": "MRZ_CHECK_DIGIT_READBACK", "occs": [], "detail": {"doc": d}})
    for rel in graph.relations:
        if rel.type == "ISSUE_EXPIRY":
            txt = out_text.get(rel.data["doc"], "")
            iss, exp = dt.date.fromisoformat(rel.data["issue"]), dt.date.fromisoformat(rel.data["expiry"])
            found = [parse_date(m.group()) for m in DATE_RX.finditer(txt)]
            ds = {f.date for f in found if f and f.date}
            if iss < exp and not (iss in ds and exp in ds):
                vs.append({"code": "ISSUE_OR_EXPIRY_NOT_PRESERVED", "occs": [], "detail": rel.data})
        elif rel.type == "AMOUNT_SUM":
            txt = out_text.get(rel.data["doc"], "")
            missing = [a for a in [rel.data["total"]] + rel.data["items"] if a not in txt]
            if missing:
                vs.append({"code": "AMOUNT_SUM_NOT_PRESERVED", "occs": [], "detail": {"doc": rel.data["doc"], "missing": len(missing)}})
    return {"checked": len(read), "violations": vs, "not_found": not_found}


def _sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------------------------------------ persistence / report
def identity_map(graph: FactGraph):
    """Local review folder only (holds source values, like proposals.json): entity -> {variant: replacement}."""
    out = {}
    for eid, e in graph.entities.items():
        if all(getattr(graph.occ[i], "meta", {}).get("sad") for i in e["occs"]):
            continue  # PCI sensitive authentication data: never written, not even as an entity id
        m = {}
        for i in e["occs"]:
            o = graph.occ[i]
            if getattr(o, "meta", {}).get("sad"):
                continue
            m[o.text] = o.replacement
        out[eid] = {"kind": e["kind"], "variants": m}
    return out


def save(review_dir, graph: FactGraph, violations, cmap=None):
    from pathlib import Path
    p = Path(review_dir)
    data = {"graph": graph.summary(), "violations": violations,
            "relations": [{"type": r.type, "occs": list(r.occs), "source": r.source} for r in graph.relations],
            "date_notes": {repr(k): v for k, v in (dates(cmap).notes.items() if cmap is not None else [])}}
    (p / "consistency.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    (p / "identity-map.json").write_text(json.dumps(identity_map(graph), indent=1, ensure_ascii=False))
    import os
    os.chmod(p / "identity-map.json", 0o600)
    return data


# ------------------------------------------------------------------------------------------------ labelled public cases (edit bench / compare pack)
def label_kind_of(value, labels):
    """Kind of a labelled value from the case's entity list. A bare 4-digit year inside a DOB value ('1892' of
    'June 9 1892') is a DOB year; a single name token is GIVEN if it starts a full name value, else SURNAME."""
    best = None
    for e in labels.get("entities") or []:
        vals = e.get("values") or []
        k = e.get("kind")
        if value in vals:
            if k == "PERSON" and len(value.split()) == 1:
                full = [v for v in vals if len(v.split()) > 1]
                if full and any(v.split()[0].lower() == value.lower() for v in full):
                    return "GIVEN", e["id"]
                if full:
                    return "SURNAME", e["id"]
                # no full form (ID documents list SURNAME, its transliteration, then given names): the first value and
                # anything that transliterates to it are the surname, the rest given names; a lone value is a given name
                if len(vals) == 1:
                    return ("GIVEN" if _is_given(value) else "SURNAME"), e["id"]
                from .resolve import to_latin
                import difflib
                first = re.sub(r"[^A-Z]", "", to_latin(vals[0]).upper())
                me = re.sub(r"[^A-Z]", "", to_latin(value).upper())
                return ("SURNAME" if difflib.SequenceMatcher(None, first, me).ratio() >= 0.75 else "GIVEN"), e["id"]
            return k, e["id"]
        if k == "DOB" and any(value in v for v in vals) and re.fullmatch(r"\d{4}", value):
            best = best or ("DOB", e["id"])
        if k == "ID" and any(re.sub(r"\D", "", value) and re.sub(r"\D", "", value) in re.sub(r"\D", "", v) for v in vals):
            best = best or ("ID", e["id"])
    return best or ("ID" if re.search(r"\d", value) else "PERSON", None)


def _is_given(tok):
    try:
        from .legal import given_names
        return tok.upper() in given_names()
    except Exception:
        return False


ORG_WORDS = re.compile(r"\b(?:PERNIAGAAN|SYARIKAT|KEDAI|SDN|BHD|BERHAD|ENTERPRISES?|TRADING|INC|LLC|LLP|LTD|GMBH|CORP|COMPANY|"
                       r"ENGINEERING|SERVICES)\b", re.I)
STREET_SFX = r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way|Place|Pl|(?-i:Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Trail|Trl|Bend|Crescent|Plaza|Ridge|Alley|Hollow|Commons))\.?"


def refine_kind(kind, value, line_text=""):
    """Shape and line context beat a generic label: a date is a date, a phone is a phone, '78701' after 'TX' is a ZIP,
    'PERNIAGAAN ZHENG HUI' is a business."""
    if kind == "PERSON" and ORG_WORDS.search(value):
        return "ORG"
    if kind in ("ID", "ACCOUNT", None):
        dv = parse_date(value)
        if dv is not None and dv.precision == "day":
            return "DOB"
        if re.fullmatch(r"\(?\d{3}\)?[\s.-]?\d{3}-\d{4}", value.strip()):
            return "PHONE"
        if re.fullmatch(r"\d{3}-\d{4}", value.strip()) and re.search(r"\b(?:phone|tel|fax|telephone)\b", line_text, re.I):
            return "PHONE"
        before = line_text[:max(0, line_text.find(value))] + " "
        if re.fullmatch(r"\d{5}(?:-\d{4})?", value.strip()) and (STATE_RX.search(before) or STATE_NAME_RX.search(before + value) or STATE_OLD_RX.search(before + value)):
            return "ZIP"
        if re.match(r"\d+[A-Za-z]?\s+(?:[NSEW]\.?\s+)?[A-Za-z]", value) and re.search(r"\b" + STREET_SFX + r"(?:\s|$)", value, re.I):
            return "ADDRESS"
    return kind


def plan_labelled_case(case, pages, labels, key=None):
    """Plan context-aware replacements for every labelled value of a public sample case (deterministic per case).
    -> (mapping {(file, value): replacement}, graph, violations). $0, local only."""
    import hashlib
    from .resolve import CaseMap
    lines, occs = [], []
    for p in pages:
        for l in p.get("lines") or []:
            lid = f"{p['file']}:{p.get('page', 0)}:{l.get('id') or len(lines)}"
            lines.append(L(lid, p["file"], p.get("page", 0), l["text"], tuple(l["box"]) if l.get("box") else None))
    by_doc = {}
    for l in lines:
        by_doc.setdefault(l.doc, []).append(l)
    for f, vals in (labels.get("occurrences") or {}).items():
        for v in vals:
            v = v if isinstance(v, str) else (v.get("text") or v.get("value") or "")
            if not v:
                continue
            kind, ent = label_kind_of(v, labels)
            line = next((l for l in by_doc.get(f, []) if v in l.text), None)
            kind = refine_kind(kind, v, line.text if line else "")
            o = __import__("casepatch.docs.resolve", fromlist=["Occ"]).Occ(
                f"{f}#{len(occs)}", f, line.page if line else 0, line.id if line else f"{f}:?", 0, len(v), kind, v, "labels")
            o.entity = ent or f"{kind.lower()}:{v.lower()}"
            occs.append(o)
    cmap = CaseMap(key or hashlib.sha256(b"casepatch-editbench\x1f" + case.encode()).digest())
    g = build_graph(lines, occs)
    vs = plan(cmap, g, occs) + gender_violations(g, cmap)
    return {(o.doc, o.text): o.replacement for o in occs}, g, vs


CODE_TEXT = {
    "DOB_IN_FUTURE": "birth date is in the future",
    "DOB_AFTER_DOC_DATE": "birth date is after the document date",
    "DOB_AFTER_ISSUE": "birth date is after the issue date",
    "AGE_MISMATCH": "stated age no longer matches the birth date",
    "DATE_ERA_JUMP": "date moved out of the document's era",
    "DATE_NOT_A_DATE": "replacement is not a real date",
    "DATE_FORMAT_CHANGED": "date format changed",
    "SAME_DATE_SPLIT": "one birth date became different dates",
    "MRZ_CHECK_DIGIT": "MRZ check digit is wrong",
    "MRZ_VISUAL_MISMATCH": "MRZ birth date differs from the printed one",
    "CARD_LUHN": "card number fails the Luhn check",
    "CARD_NETWORK_CHANGED": "card network (BIN type) changed",
    "CARD_LAST4_MISMATCH": "card last four differ between documents",
    "ACCOUNT_LAST4_MISMATCH": "account last four differ from the masked form",
    "ZIP_STATE_MISMATCH": "ZIP code is not in the stated state",
    "PHONE_AREA_INVALID": "phone area code is not valid",
    "EMAIL_NAME_MISMATCH": "email no longer matches the person's fake name",
    "GENDER_NAME_MISMATCH": "fake given name does not match the stated sex",
    "NAME_NOT_PLAUSIBLE": "replacement does not look like a name",
    "ADDRESS_NOT_PLAUSIBLE": "replacement does not look like an address",
    "NAME_VARIANT_SPLIT": "name variants of one person got different surnames",
    "SAME_VALUE_SPLIT": "one value got different replacements",
    "MISMATCH_COLLAPSED": "two different source values got the same replacement",
    "FORMAT_CHANGED": "format changed",
    "UNCHANGED": "replacement equals the original",
}


def explain(code):
    c = code.split(":", 1)[-1]
    return CODE_TEXT.get(c, c.replace("_", " ").lower())
