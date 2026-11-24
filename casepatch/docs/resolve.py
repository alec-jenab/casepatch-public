"""One case-scoped identity map shared by every document and trace in the case.

- Proposals from any detector are normalized, merged per line, then PROPAGATED: a value found in one
  document is searched in every document (all surface forms of a person name).
- Persons are resolved by (given name, surname). Different given names are different people even with a
  shared surname (keeps a genuine signer mismatch). Initial-only mentions ('J. Doe') link when unique;
  otherwise they are flagged — and stay coherent anyway because fake given names preserve initial classes.
- Replacements are format-preserving and keyed by an HMAC with a per-case random key; the key and the
  table stay in the local review folder, never in the output."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field

from .detect import Proposal

HONORIFIC = r"(?:Mr|Ms|Mrs|Miss|Dr|Ofc|Officer|Det)\.?"


def split_role_title(text):
    """'Captain Velloran' -> ('Captain ', 'Velloran'); 'Chief Judge Ada Brill' -> ('Chief Judge ', 'Ada Brill'); a text
    without a leading role title (or only a title) -> ('', text)."""
    from .legal import ROLE_TITLES
    t = text or ""
    i = 0
    for w in re.finditer(r"([A-Za-z]+)\.?\s+", t):
        if w.start() != i or w.group(1).upper() not in ROLE_TITLES:
            break
        i = w.end()
    rest = t[i:]
    if not i or not re.match(r"[A-Za-z]", rest):
        return "", t
    return t[:i], rest


# Keep in sync with casepatch.docs.legal.LINE_PREFIX. Used so an OCR-mangled
# 'D-I DAVID BRIAN STONE' still parses as the person name, not as unparsed text.
_DOCKET_PREFIX = re.compile(
    r"^(?:D[\-.][\w]+|D\d+|#[\w]+[.)]?|\d+[.)]|\[[\w]+\]|\([\w]+\))\s+",
    re.I,
)


def _split_docket_prefix(text):
    raw = text or ""
    stripped = raw.lstrip()
    lead = len(raw) - len(stripped)
    m = _DOCKET_PREFIX.match(stripped)
    if not m:
        return "", raw
    end = lead + m.end()
    return raw[:end], raw[end:]

GIVEN = {
    "A": ["Alma", "Avery", "Anton", "Ada"], "B": ["Bianca", "Boris", "Bea", "Bram"], "C": ["Clara", "Cyrus", "Celia", "Cole"],
    "D": ["Dana", "Dmitri", "Delia", "Dax"], "E": ["Elena", "Emil", "Esme", "Ezra"], "F": ["Freya", "Felix", "Fern", "Flynn"],
    "G": ["Greta", "Gideon", "Gwen", "Gus"], "H": ["Hazel", "Hugo", "Hana", "Hal"], "I": ["Ingrid", "Ivo", "Isla", "Ike"],
    "J": ["Jolene", "Jasper", "Juno", "Joel"], "K": ["Kira", "Kurt", "Kaia", "Knox"], "L": ["Lydia", "Leon", "Lark", "Lev"],
    "M": ["Maren", "Milo", "Mabel", "Moss"], "N": ["Nadia", "Nolan", "Nell", "Niko"], "O": ["Odette", "Otto", "Opal", "Orrin"],
    "P": ["Petra", "Pavel", "Pia", "Pike"], "Q": ["Quinn", "Quilla", "Quentin", "Quade"], "R": ["Rosa", "Ronan", "Rhea", "Rafe"],
    "S": ["Selma", "Silas", "Sage", "Soren"], "T": ["Talia", "Tobias", "Tess", "Tamsin"], "U": ["Uma", "Ulric", "Una", "Uri"],
    "V": ["Vera", "Viggo", "Vida", "Vaughn"], "W": ["Wren", "Wade", "Willa", "Wolf"], "X": ["Xenia", "Xavi", "Xara", "Xan"],
    "Y": ["Yara", "Yusuf", "Yvette", "Yann"], "Z": ["Zara", "Zane", "Zelda", "Zev"]}
SURNAMES = ["Roe", "Fox", "Ley", "Kim", "Orr", "Poe", "Ash", "Bell", "Cole", "Dunn", "Hale", "Lund", "Moss", "Pratt", "Brooks", "Calloway", "Draper", "Ellison", "Fairbanks", "Garrow", "Holloway", "Ingram", "Jessup", "Kessler",
            "Lindqvist", "Marchetti", "Northcott", "Okafor", "Pemberton", "Quarles", "Rasmussen", "Sorensen", "Thackeray",
            "Underhill", "Vasquez", "Whitlock", "Yardley", "Zielinski", "Ashdown", "Birch", "Corrigan", "Delacroix"]
ORG_POOL = ["Arvo", "Belmar", "Corvex", "Daltrin", "Everline", "Fenwick", "Granite", "Halcyon", "Ironleaf", "Junova", "Kestrel",
            "Lumen", "Meridian", "Northgate", "Orchard", "Pelican", "Quarry", "Redwood", "Sterling", "Tidewater", "Unison", "Vantage",
            "Westbrook", "Yarrow", "Zenith", "Brightwave", "Copperfield", "Driftwood"]
STREETS = ["Elm Park", "Oak Glen", "Fern Hill", "Briar Ridge", "Willow Bend", "Cedar Hollow", "Larkspur", "Juniper Hill", "Foxglove", "Harbor View", "Aspen Crest", "Stonebrook"]
STREET_LINE = re.compile(r"(\d+[A-Za-z]?)\s+((?:[A-Za-z][A-Za-z.'’-]*\s+)+?)(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|"
                         r"Boulevard|Blvd|Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Trail|Trl|"
                         r"Bend|Crescent|Plaza|Ridge|Alley|Hollow|Commons)\b(.*)", re.I)


@dataclass
class Occ:
    id: str
    doc: str
    page: int
    line_id: str
    start: int
    end: int
    kind: str
    text: str
    source: str
    detectors: set = field(default_factory=set)
    confidence: float = 0.0
    entity: str = ""
    review: list = field(default_factory=list)   # reasons this needs a human look
    replacement: str = ""


MONEY = re.compile(r"^[-+]?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\$")
DATEISH = re.compile(r"^(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4})(?:\s*-\s*.*)?$")
SHAPE = {"ACCOUNT": lambda t: len(re.sub(r"\D", "", t)) >= 6, "ID": lambda t: len(re.sub(r"\D", "", t)) >= 3 and len(t) <= 40,
         "CARD": lambda t: len(re.sub(r"\D", "", t)) >= 12, "CARD_LAST4": lambda t: re.fullmatch(r"\d{4}", t) is not None and len(set(t)) > 1,  # 0000/1111 = form placeholders
         "SSN_LAST4": lambda t: re.fullmatch(r"\d{4}", t) is not None,
         "PHONE": lambda t: len(re.sub(r"\D", "", t)) >= 10, "EMAIL": lambda t: "@" in t, "ZIP": lambda t: re.fullmatch(r"\d{5}(?:-\d{4})?", t) is not None,
         "PERSON": lambda t: not re.search(r"\d", t) and len(t) <= 60, "ADDRESS": lambda t: re.match(r"\d", t) is not None,
         "DOB": lambda t: re.search(r"\d", t) is not None}
REJECTED = []  # (kind, reason) counters for the report; no source text
KEPT_VALUES = set()  # values deliberately kept as facts (organisation contact lines); local only, for the verifier rescan


def protected_fact(kind, text):
    """Amounts and ordinary dates are evidence: never replaced (DOB is the only date kind)."""
    t = text.strip()
    if kind != "DOB" and (MONEY.search(t) or DATEISH.match(t)):
        return True
    return False


def normalize(p: Proposal, value: str, card_last4=frozenset()):
    """`card_last4`: last four digits of full identifiers (card, account, phone, ID...) detected in this case. A last
    four that repeats one digit ('1111') is normally a form placeholder, but not when a full number in the case ends with it."""
    n = _normalize(p, value)
    if n is None:
        return None
    if protected_fact(n.kind, n.text):
        REJECTED.append({"detector": p.detector, "kind": n.kind, "reason": "PROTECTED_FACT_AMOUNT_OR_DATE"}); return None
    labelled_short_phone = n.kind == "PHONE" and ((p.detector == "form-fields" and len(re.sub(r"\D", "", n.text)) >= 4) or
                                                  ((p.meta or {}).get("labelled") and len(re.sub(r"\D", "", n.text)) >= 7))
    case_last4 = n.kind == "CARD_LAST4" and n.text in card_last4
    if not labelled_short_phone and not case_last4 and not SHAPE.get(n.kind, lambda t: True)(n.text):
        REJECTED.append({"detector": p.detector, "kind": n.kind, "reason": "FAILS_TYPE_SHAPE_CHECK"}); return None
    return n


def _person_label_prefix():
    from .detect import PERSON_LABELS
    return re.compile(r"(?i:(?:" + PERSON_LABELS + r"|Name|Holder|Payee|Payer|Sender|Debtor|Creditor|Owner|Signer|Signatory)"
                      r"(?:'s)?(?:\s+name)?)\s*[:\-\u2013]?\s+(?=[A-Z])")


PERSON_LABEL_PREFIX = _person_label_prefix()
# a self-introduction in front of a name ("I'm Anna Lindqvist", 'this is Jane Roe'): text, not part of the name
SELF_INTRO_PREFIX = re.compile(r"(?:I['\u2019]?m|I am|[Tt]his is|[Ii]t['\u2019]?s|[Mm]y name is|[Mm]y name['\u2019]s|[Nn]ame['\u2019]s)\s+(?=[A-Z])")


def _ssn_masked(value, s, e):
    """The value value[s:e] ends an SSN mask ('XXX-XX-4096', '***-**-4096'); a masked card ('************6467') or an
    SSN elsewhere on the line does not count."""
    w0 = max(0, s - 16)
    return any(w0 + m.end() == e for m in _SSN_MASK.finditer(value[w0:e + 1]))


# exactly an SSN mask: a longer mask run ('************6467', a masked card) is not one
_SSN_MASK = re.compile(r"(?<![*Xx\d])(?:XXX|xxx|\*{3})[- ]?(?:XX|xx|\*{2})[- ]?\d{4}(?!\d)")


def _normalize(p: Proposal, value: str):
    """Trim detector-specific noise into canonical spans (honorifics, 'City, ST 12345' -> ZIP, 4-digit CARD -> LAST4)."""
    s, e, kind = p.start, p.end, p.kind
    text = value[s:e]
    if kind == "PERSON":
        m = re.match(HONORIFIC + r"\s+", text)
        if m:
            s += m.end()
        lab = PERSON_LABEL_PREFIX.match(value[s:e]) or SELF_INTRO_PREFIX.match(value[s:e])
        if lab and len(value[s + lab.end():e].split()) >= 2:  # 'Cardholder Jane Doe' -> 'Jane Doe' (the label stays)
            s += lab.end()
        text = value[s:e].rstrip(" ,.;:") if not re.search(r"\b[A-Z]\.$", value[s:e]) else value[s:e]
        e = s + len(text)
        s, e = _trim_person_words(value, s, e)
        text = value[s:e]
        if len(text.split()) < 2 and not re.fullmatch(r"[A-Z][A-Z'’-]+", text):
            return None  # single tokens are handled only via honorific+surname propagation
    elif kind == "ZIP":
        m = re.search(r"\b\d{5}(?:-\d{4})?\b", text)
        if not m:
            return None
        s, e = s + m.start(), s + m.end()
    elif kind == "CARD" and re.fullmatch(r"\d{4}", text):
        kind = "CARD_LAST4"
    elif kind in ("ID", "ACCOUNT", "SSN", "CARD_LAST4") and _ssn_masked(value, s, e) and \
            re.search(r"\d{4}$", text):
        # 'XXX-XX-4096' (or the 'XX-4096' tail an ID rule picks up): only the last four digits are a value, an SSN's
        kind = "SSN_LAST4"
        s = e - 4
    elif kind in ("ADDRESS",) and not re.search(r"\d", text):
        return None
    if e <= s:
        return None
    return Proposal(p.line_id, s, e, kind, value[s:e], p.detector, p.confidence, p.person_hint, p.meta)


# given names / surnames that are also everyday English words: propagated alone only in greeting/honorific context
COMMON_WORD_NAMES = {"will", "may", "june", "april", "august", "grace", "hope", "faith", "joy", "rose", "lily", "ivy", "dawn",
                     "summer", "autumn", "april", "mark", "bill", "pat", "sue", "art", "jack", "frank", "rich", "rob", "don",
                     "ray", "guy", "max", "earl", "dean", "hunter", "mason", "carter", "cook", "baker", "smith", "brown",
                     "green", "white", "black", "young", "king", "hill", "wood", "stone", "rock", "bell", "lane", "page",
                     "price", "long", "short", "little", "fox", "wolf", "case", "bank", "card", "chase", "best", "love",
                     "star", "sky", "river", "dale", "glen", "cliff", "bud", "sunny", "honey", "penny", "ruby", "amber"}

# document vocabulary that a layout heuristic can mistake for a name ('PATIENT INTAKE'); never propagated alone
DOC_WORDS = {"patient", "member", "customer", "client", "account", "statement", "intake", "form", "notice", "claim", "policy",
             "summary", "report", "invoice", "receipt", "letter", "visit", "note", "office", "plan", "protection", "service",
             "services", "electronics", "dispute", "card", "visa", "credit", "debit", "union", "bank", "medical", "medicine",
             "family", "clinic", "health", "insurance", "benefits", "explanation", "total", "payment", "balance", "date",
             "name", "address", "phone", "email", "number", "signature", "signed", "dear", "sincerely", "regards", "license",
             "driver", "specimen", "department", "city", "state", "county", "street", "avenue", "road", "north", "south",
             "east", "west", "new", "york", "san", "los", "las", "saint", "fort", "port", "mount", "lake", "river"}

_NAME_PARTICLES = {"van", "von", "de", "del", "della", "der", "den", "da", "di", "du", "la", "le", "bin", "al", "el", "abu",
                   "ibn", "bint", "das", "dos", "ter", "ten", "do", "y"}
# particles that also start a surname when capitalised ('De Silva', 'Van Houten', 'Della Rocca', 'Le Blanc')
_TITLE_PARTICLES = {"van", "von", "de", "del", "della", "der", "den", "da", "di", "du", "la", "le", "das", "dos", "ter"}


def _trim_person_words(value, s, e):
    """A person span never starts or ends with a lowercase function word ('this is Jane Doe' -> 'Jane Doe', 'Jane Doe
    said' -> 'Jane Doe'); name particles inside a name ('Ludwig van Beethoven') are kept. All-lowercase spans are left
    as they are (some sources write names in lowercase)."""
    seg = value[s:e]
    if not re.search(r"[A-Z]", seg):
        return s, e
    toks = [(m.start() + s, m.end() + s, m.group()) for m in re.finditer(r"\S+", seg)]
    def low(t):
        w = t.strip(".,;:!?\"'()")
        return bool(w) and w[0].islower()
    while len(toks) > 1 and low(toks[0][2]) and toks[0][2].strip(".,") not in _NAME_PARTICLES:
        toks.pop(0)
    while len(toks) > 1 and low(toks[-1][2]) and not re.fullmatch(r"(?:al|el|ad|ar|as|at|ash|az)-[A-Z][\w'’-]*", toks[-1][2].rstrip(".,;:!?")):
        toks.pop()  # ('Idris al-Haddawi' keeps its surname)
    return (toks[0][0], toks[-1][1]) if toks else (s, e)


def _particle_start(toks, upper=False):
    """Index of the first particle of a 'Given [M.] particle... Surname' name ('Ottilie van der Brakel' -> 1), or None.
    Particles are lowercase (any case when the whole name is in capitals); one to three of them, then a capitalised word."""
    if not 3 <= len(toks) <= 6 or not re.fullmatch(r"[A-Za-z'’-]+", toks[0]) or \
            not re.fullmatch(r"[A-Z][A-Za-z'’]+(?:-[A-Za-z'’]+)*", toks[-1]):
        return None
    for i in (1, 2):
        run = toks[i:-1]
        if run and len(run) <= 3 and all(x.lower() in _NAME_PARTICLES and (upper or x.islower()) for x in run):
            return i
        # Title-case particles ('Ana De Silva', 'Marco Van Houten', 'Rosa De La Cruz'): the given name is not a particle
        if run and len(run) <= 3 and not upper and toks[0].lower() not in _NAME_PARTICLES and \
                all(x.lower() in _TITLE_PARTICLES and x[:1].isupper() for x in run):
            return i
    return None


def particle_surname(surname):
    """'van der Brakel' -> ('van der', 'Brakel'); a surname without particles -> None."""
    toks = (surname or "").split()
    if len(toks) < 2 or len(toks) > 4 or not all(x.lower() in _NAME_PARTICLES for x in toks[:-1]) or \
            toks[-1].lower() in _NAME_PARTICLES:
        return None
    return " ".join(toks[:-1]), toks[-1]


def parse_person(text):
    """-> dict(given, middle, surname, form, case) or None."""
    t = re.sub(r"^" + HONORIFIC + r"\s+", "", text.strip())
    _, t = _split_docket_prefix(t)
    t = t.strip()
    case = "upper" if t.isupper() else "title"
    m = re.fullmatch(r"([A-Za-z'’-]+),\s*([A-Za-z'’-]+)(?:\s+([A-Za-z])\.?)?", t)
    if m:
        return {"surname": m.group(1), "given": m.group(2), "middle": m.group(3), "form": "last_first", "case": case}
    toks = t.split()
    if len(toks) == 2 and re.fullmatch(r"[A-Z]\.", toks[0]):
        return {"surname": toks[1], "given": toks[0][0], "middle": None, "form": "initial", "case": case}
    if len(toks) == 3 and re.fullmatch(r"[A-Z]\.?", toks[1]):
        return {"surname": toks[2], "given": toks[0], "middle": toks[1][0], "form": "first_middle_last", "case": case}
    if particle_surname(t) and (toks[0].islower() or len(toks) >= 3 and (case == "upper" or
                                                                       all(x.islower() for x in toks[1:-1]))):
        # 'van der Brakel' / 'Van der Brakel' / 'VAN DER BRAKEL' alone: a surname, not 'Van' + 'Brakel'
        return {"surname": t, "given": None, "middle": None, "form": "surname", "case": case}
    pi = _particle_start(toks, case == "upper")
    if pi is not None:  # 'Ottilie van der Brakel', 'Ana M. de la Cruz', 'OTTILIE VAN DER BRAKEL': the surname is 'van der Brakel'
        sur = " ".join(toks[pi:])
        if pi == 1:
            return {"surname": sur, "given": toks[0], "middle": None, "form": "first_last", "case": case}
        if re.fullmatch(r"[A-Z]\.?", toks[1]):
            return {"surname": sur, "given": toks[0], "middle": toks[1][0], "form": "first_middle_last", "case": case}
    if 2 <= len(toks) <= 3 and all(re.fullmatch(r"[A-Za-z'’-]+", x) for x in toks):
        return {"surname": toks[-1], "given": toks[0], "middle": None, "form": "first_last", "case": case}
    if len(toks) == 1 and re.fullmatch(r"[A-Za-z'’-]+", toks[0]):
        return {"surname": toks[0], "given": None, "middle": None, "form": "surname", "case": case}
    return None


def middle_name(text):
    """The spelled-out middle name of a 'Given Middle Surname' name ('Marguerite' in 'Odalys Marguerite Brennecke-Vance'),
    or None (initials, particles such as 'de', other forms)."""
    t = re.sub(r"^" + HONORIFIC + r"\s+", "", (text or "").strip())
    _, t = _split_docket_prefix(t)
    p = parse_person(t)
    toks = t.strip().split()
    if not p or p["form"] != "first_last" or len(toks) != 3:
        return None
    m = toks[1]
    if len(m) < 2 or not m[0].isupper() or m.lower() in _NAME_PARTICLES or not re.fullmatch(r"[A-Za-z']+", m):
        return None
    return m


MASC_EXCEPTIONS = {"mike", "joe", "dave", "steve", "jose", "george", "randy", "andy", "roy", "jeremy", "tony", "harry", "larry",
                   "gary", "terry", "jerry", "bobby", "billy", "timothy", "anthony", "henry", "rocky", "rudy", "eddie", "freddie",
                   "johnny", "jimmy", "danny", "tommy", "luke", "pete", "mike", "wayne", "bruce", "lee", "jesse", "dale", "kyle",
                   "lance", "vince", "clyde", "duane", "shane", "dwayne", "andre", "rene", "mickey", "stanley", "bradley", "ricky"}


EMAIL_ROLE_WORDS = {"support", "info", "billing", "help", "hello", "contact", "admin", "office", "team", "sales",
                    "noreply", "no", "reply", "disputes", "claims", "service", "services", "care", "mail", "accounts"}

class CaseMap:
    def __init__(self, key: bytes | None = None):
        self.key = key or secrets.token_bytes(32)
        self._given, self._surname, self._cache = {}, {}, {}
        self.given_style = {}  # fake given name -> "f"/"m" pool styling (used to pick a matching synthetic portrait)
        self.style_hint = {}  # given name -> "m"/"f" from an adjacent honorific (Mr./Mrs./Ms.) so 'Mr. <name>' stays coherent
        letters = [chr(c) for c in range(65, 91)]
        perm = sorted(letters, key=lambda c: self._h("letter", c))
        # derangement: rotate the shuffled order by one so no letter maps to itself
        self.letter = {perm[i]: perm[(i + 1) % 26] for i in range(26)}
        self.sources = set()

    def _h(self, *parts):
        return hmac.new(self.key, "\x1f".join(parts).encode(), hashlib.sha256).hexdigest()

    def _n(self, *parts):
        return int(self._h(*parts)[:12], 16)

    def _closest_len(self, pool, n, *salt):
        """Prefer replacements of similar length (layout fidelity), tie-broken by the keyed hash."""
        best = min(abs(len(x) - n) for x in pool)
        near = [x for x in pool if abs(len(x) - n) <= best + 1]
        return near[self._n(*salt) % len(near)]

    # ---- persons
    def given(self, name):
        k = name.lower()
        if k not in self._given:
            # crude style heuristic only (pools alternate feminine/masculine-sounding names); not a gender inference
            fem = k.endswith(("a", "e", "y", "i", "elle", "lyn")) and k not in MASC_EXCEPTIONS
            if k in self.style_hint:
                fem = self.style_hint[k] == "f"
            pool = [g for i, g in enumerate(GIVEN[self.letter[name[0].upper()]]) if (i % 2 == 0) == fem]
            pool = [g for g in pool if g.lower() != k and g not in self._given.values()] or \
                   [g for g in GIVEN[self.letter[name[0].upper()]] if g.lower() != k and g not in self._given.values()] or \
                   [g for g in GIVEN[self.letter[name[0].upper()]] if g.lower() != k]  # many people share one initial: reuse
            self._given[k] = self._closest_len(pool, len(name), "given", k)
            self.given_style[self._given[k].lower()] = "f" if fem else "m"
        return self._given[k]

    def alias_surname(self, part, full):
        """One person's surname printed in part ('Hale' for 'Quintero-Hale', on a signature line or in prose) gets
        the fake of the full surname, so every variant of that person reads as the same synthetic person."""
        if part.lower() != full.lower():
            self.__dict__.setdefault("_surname_alias", {})[part.lower()] = full

    def surname(self, name):
        k = name.lower()
        alias = self.__dict__.get("_surname_alias", {}).get(k)
        if alias is not None and k not in self._surname:
            self._surname[k] = self.surname(alias)
        aliases = self.__dict__.get("_surname_alias", {})
        parts = [x for x in k.split("-") if x]
        if alias is None and k not in self._surname and len(parts) > 1 and any(x in aliases for x in parts) \
                and k not in {v.lower() for v in aliases.values()}:
            # a hyphenated form with a part of a known compound ('Vance-Brennecke' or 'Ostrander-Vance' for
            # 'Brennecke-Vance'): the parts of that surname read as its fake, other parts get their own
            fulls = {aliases[x].lower() for x in parts if x in aliases}
            if len(fulls) == 1 and all(x in aliases or _part_of(x, next(iter(fulls))) for x in parts):
                self._surname[k] = self.surname(next(iter(fulls)))
            else:
                out = []
                for x in parts:
                    f = self.surname(x)
                    if f not in out:
                        out.append(f)
                self._surname[k] = "-".join(out)
        if k not in self._surname:
            pool = [s for s in SURNAMES if s.lower() != k and s not in self._surname.values() and s.lower() not in self.sources] or \
                   [s for s in SURNAMES if s.lower() != k and s.lower() not in self.sources]  # pool exhausted: reuse
            self._surname[k] = self._closest_len(pool, len(name), "surname", k)
        ps = particle_surname(k)
        if ps:  # 'brakel' / 'vanderbrakel' of 'van der brakel' (an email's local part) read as the same fake
            for x in (ps[1], re.sub(r"\s+", "", k)):
                self._surname.setdefault(x, self._surname[k])
        if len(parts) > 1 and all(re.fullmatch(r"[a-z']+", x) for x in parts):
            # 'pennickhale' (an email's local part) for 'pennick-hale': the same fake
            self._surname.setdefault("".join(parts), self._surname[k])
        return self._surname[k]

    def person(self, text):
        prefix, core = _split_docket_prefix((text or "").strip())
        title, rest = split_role_title(core)
        if title:  # 'Captain Velloran' -> 'Captain <fake surname>': the title is kept, it is not a given name
            fake = self.person(rest)
            return f"{prefix}{title}{fake}" if fake else None
        p = parse_person(core)
        if not p:
            return None
        sur = self.surname(p["surname"])
        given = self.given(p["given"]) if p["given"] and len(p["given"]) > 1 else None
        ini = self.letter[p["given"][0].upper()] if p["given"] else None
        mid = self.letter[p["middle"].upper()] if p["middle"] else None
        f = p["form"]
        middle = middle_name(core) if given else None  # a spelled-out middle name gets a fake given name of its own
        if middle:
            given = f"{given} {self.given(middle)}"
        out = {"last_first": f"{sur}, {given}" + (f" {mid}" if mid else ""), "initial": f"{ini}. {sur}",
               "first_middle_last": f"{given} {mid}. {sur}", "first_last": f"{given} {sur}", "surname": sur}[f]
        if "middle" in text and False:
            pass
        if p["form"] == "first_middle_last" and not re.search(r"\b[A-Z]\.", text):
            out = f"{given} {mid} {sur}"
        body = out.upper() if p["case"] == "upper" else out
        return f"{prefix}{body}" if prefix else body

    # ---- format preserving digits/letters
    def digits(self, text, kind, keep_prefix=0):
        key = re.sub(r"\D", "", text)
        stream = self._h("digits", kind, key) + self._h("digits2", kind, key)
        out, i, n = [], 0, 0
        for ch in text:
            if ch.isdigit():
                if n < keep_prefix:
                    out.append(ch)
                else:
                    out.append(str(int(stream[i], 16) % 10)); i += 1
                n += 1
            else:
                out.append(ch)
        res = "".join(out)
        if re.sub(r"\D", "", res) == key:  # never equal to the source
            last = max((i for i, ch in enumerate(res) if ch.isdigit()), default=None)
            if last is not None:  # bump the last digit in place: appending would change the length ('7KAZCWTXH' -> '7KAZCWTXH0',
                # which an MRZ field then truncates back to the real number)
                res = res[:last] + str((int(res[last]) + 1) % 10) + res[last + 1:]
            else:
                res = res + "0"
        return res

    def last4(self, d):
        return self.digits(d, "last4")

    def card(self, text):
        ds = re.sub(r"\D", "", text)
        last = self.last4(ds[-4:])
        from .consistency import card_network
        keep = 4 if card_network(ds).startswith(("discover", "other", "diners")) else 2  # issuer network (BIN type) kept
        # Every 4-digit window of the real number is forbidden anywhere in the fake body, including the real
        # BIN digits reappearing after the kept network prefix (e.g. real 4532... giving fake 4551 2210 4532 ...).
        windows = {ds[i:i + 4] for i in range(len(ds) - 3)}
        first = 1 if keep >= 4 else 0  # with a 4-digit kept prefix, window 0 is the kept prefix itself
        for salt in range(50):
            # The fake must not reproduce any 4-digit window of the real number past the kept network prefix
            # (e.g. a fake card that happens to start with the real last four).
            body = self.digits(ds[:-4], "card%d" % salt if salt else "card", keep_prefix=keep) + last
            nums = [int(c) for c in body]
            for fix in range(10):  # make it Luhn-valid by adjusting a middle digit
                nums[len(nums) // 2] = fix
                tot = sum(sum(divmod(2 * x, 10)) if (len(nums) - i) % 2 == 0 else x for i, x in enumerate(nums))
                if tot % 10 == 0:
                    break
            fake = "".join(map(str, nums))
            # (the fake last4 is fixed by last4() for cross-document consistency, so only windows touching the body count)
            if not any(fake[i:i + 4] in windows for i in range(first, len(fake) - 4)) and ds[-4:] not in fake[:-4]:
                break
        it = iter(fake)
        return "".join(next(it) if c.isdigit() else c for c in text)

    def phone(self, text):
        ds = re.sub(r"\D", "", text)
        if len(ds) < 7:  # local exchange number on old forms ('8319'): different digits, same layout
            return self.digits(text, "phone")
        new = ds[:-7] + "555" + "01" + self.digits(ds[-2:], "phone" + ds)  # keep area code; 555-01xx is reserved fictional
        if new == ds:
            new = new[:-1] + str((int(new[-1]) + 1) % 10)
        it = iter(new)
        return "".join(next(it) if c.isdigit() else c for c in text)

    def email(self, text):
        """Realistic fake address: name parts map to the case's fake person, initials to the case letter map,
        other words to a fake surname of similar length; mailbox role words (support, billing, ...) are kept."""
        local, _, domain = text.partition("@")
        def known(k):
            if k in self._given:
                return self._given[k].lower()
            if k in self._surname:
                return self._surname[k].lower()
            return None
        def initials(w):
            return "".join(self.letter.get(c.upper(), c.upper()).lower() for c in w)
        def tok(m):
            w = m.group()
            k = w.lower()
            hit = known(k)
            if hit:
                return hit
            if k in EMAIL_ROLE_WORDS:
                return k
            if len(k) <= 2:
                return initials(k)
            # a known name glued to initials or another known name: 'mquintero', 'marisolq', 'quinterohale'
            for i in range(1, len(k)):
                a, b = k[:i], k[i:]
                ka, kb = known(a), known(b)
                if ka and kb:
                    return ka + kb
                if kb and len(a) <= 2:
                    return initials(a) + kb
                if ka and len(b) <= 2:
                    return ka + initials(b)
            pool = [x.lower() for x in SURNAMES if x.lower() != k and x.lower() not in self.sources]
            return self._closest_len(pool, len(k), "emailtok", k)
        new_local = re.sub(r"[A-Za-z]+", tok, local)
        new_local = re.sub(r"\d+", lambda m: self.digits(m.group(), "email"), new_local)
        if new_local.lower() == local.lower():  # only role words: still a different mailbox, same shape
            new_local = new_local + self.digits("%02d" % (self._n("emailrole", local.lower()) % 100), "emailrole")
        dom = domain if re.search(r"(^|\.)example\.(com|net|org)$", domain) or domain.endswith(".invalid") else "example.com"
        return new_local + "@" + dom

    def zip(self, z):
        """Same 3-digit prefix (state/region kept). Per case, the 5-digit part is one-to-one: two different real ZIPs
        never collapse into one fake (only 100 endings per prefix, so plain hashing collides), the same 5-digit ZIP
        maps the same with or without a +4, and a fake never reuses a real ZIP already seen in the case."""
        m = re.fullmatch(r"(\s*)(\d{5})(.*)", z, re.S)
        if not m:
            return self.digits(z, "zip", keep_prefix=3)
        lead, five, tail = m.groups()
        fwd = self.__dict__.setdefault("_zip_fwd", {})
        rev = self.__dict__.setdefault("_zip_rev", {})
        if five not in fwd:
            out = None
            for i in range(200):
                c = self.digits(five, "zip" if i == 0 else f"zip{i}", keep_prefix=3)
                if c != five and c not in rev and c not in fwd:
                    out = c; break
            out = out or self.digits(five, "zip", keep_prefix=3)
            fwd[five], rev[out] = out, five
        tail = re.sub(r"\d+", lambda d: self.digits(d.group(), "zip4"), tail)
        return lead + fwd[five] + tail

    def _street_name(self, name):
        real = set(name.strip().lower().split())  # never keep the real street name or a word of it ('Elm' -> not 'Elm Park')
        pool = [x for x in STREETS if not real & set(x.lower().split())] or [x for x in STREETS if x.lower() != name.strip().lower()]
        return self._closest_len(pool, len(name.strip()), "street", name.strip().lower())

    def alias_street(self, word, name):
        """A street's name word printed on its own ('the Wrenfield house' for '2208 Wrenfield Drive') gets the fake
        street name of the full address line."""
        self.__dict__.setdefault("_street_alias", {})[word.lower()] = name

    def address(self, text):
        alias = self.__dict__.get("_street_alias", {}).get(text.strip().lower())
        if alias is not None:
            return self._case(text.strip(), self._street_name(alias))
        m = STREET_LINE.match(text)
        # (a value already known to be an address: the suffix may be in any case, 'JUNIPER TER' on a licence)
        if not m:
            return self.digits(text, "address")
        num, name, suffix, rest = m.groups()
        street = self._street_name(name)
        rest = re.sub(r"(\bApt\.?|\bSuite|\bUnit|#)(\s*)(\d+)", lambda a: a.group(1) + a.group(2) + self._unit(a.group(3)), rest, flags=re.I)
        rest = re.sub(r"\b\d{5}(?:-\d{4})?\b", lambda z: self.zip(z.group()), rest)
        house = self.digits(num, "house")
        if house[0] == "0":  # no '0001 Main St'
            house = str(1 + int(self._h("house0", num)[:4], 16) % 9) + house[1:]
        if name.strip().isupper():
            street = street.upper()
        return f"{house} {street} {suffix}{rest}"

    def _unit(self, num):
        out = self.digits(num, "unit")
        return out if out[0] != "0" else str(1 + int(self._h("unit0", num)[:4], 16) % 9) + out[1:]

    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    def dob(self, text):
        """Context-aware birth date: planned per date value by consistency.DatePlanner (same era, never in the future,
        before the document/issue date, stated age kept, same fictional date in every format and the MRZ)."""
        from .consistency import dates
        core = text.strip()
        out = dates(self).replace(core)
        if out is None:
            return self.digits(text, "dob")
        i = text.find(core)
        return text[:i] + out + text[i + len(core):]

    def _case(self, src, out):
        return out.upper() if src.isupper() else out

    def _linked(self, kind, text):
        """OCR variants of one name in one case (a letter-spaced OCR line, the Greek line and its Latin transliteration) share a key."""
        import difflib
        lat = re.sub(r"[^A-Z]", "", to_latin(_ocr_script_fix(text)).upper())
        seen = self.__dict__.setdefault("_lat_seen", {})
        best = max(((difflib.SequenceMatcher(None, lat, k).ratio(), k) for k in seen.get(kind, [])), default=(0, None))
        key = best[1] if best[0] >= 0.75 else (lat.capitalize() if lat else text.strip())
        if best[0] < 0.75 and lat:
            seen.setdefault(kind, []).append(lat)
        return key if lat else text.strip()

    def _wordwise(self, text, one):
        """'ВАЛЕНТИНА ФЕДОРОВНА' (given name + patronymic) or 'MARIA JOSE': each word is its own name, so the same
        word printed alone elsewhere ('VALENTINA' on the Latin line, in the MRZ) gets the same fake."""
        words = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", text)
        if len(words) <= 1:
            return one(text)
        return re.sub(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", lambda m: one(m.group()), text)

    def given_only(self, text):
        def one(t):
            plain = re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", t.strip()) is not None
            key = t.strip() if plain else self._linked("GIVEN", t) if _script(t) != "latin" or t.isupper() else t.strip()
            return _in_script(t, lambda x: self._case(x, self.given(key)))
        return self._wordwise(text, one)

    def surname_only(self, text):
        if particle_surname(text.strip()):  # 'Van der Brakel' / 'VAN DER BRAKEL': one surname, one fake
            return self._case(text.strip(), self.surname(text.strip()))
        if len(text.split()) > 1 and text.strip().lower() in self.__dict__.get("_surname_alias", {}):
            # 'Pennick Hale' printed for 'Pennick-Hale': the fake of that one surname, not word by word
            return self._case(text.strip(), self.surname(text.strip()))
        def one(t):
            plain = re.fullmatch(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", t.strip()) is not None
            key = t.strip() if plain else self._linked("SURNAME", t) if _script(t) != "latin" or t.isupper() else t.strip()
            return _in_script(t, lambda x: self._case(x, self.surname(key)))
        return self._wordwise(text, one)

    def org(self, text):
        """Fictional organisation name: each distinctive word -> a pool word of similar length; legal suffixes kept."""
        keep = re.compile(r"^(?:\([A-Z]\)|SDN|BHD|BERHAD|INC\.?|LLC|LLP|PLC|LTD\.?|LIMITED|GMBH|AG|KG|PERNIAGAAN|SYARIKAT|KEDAI|CO\.?|CORP\.?|COMPANY|&|AND|OF|THE|ENTERPRISE|TRADING|ENGINEERING|SERVICES?|GROUP)$", re.I)
        out = []
        for i, w in enumerate(text.split(" ")):
            if not w or keep.match(w) or not re.search(r"[A-Za-z]", w):
                out.append(w); continue
            pool = [x for x in ORG_POOL if x.lower() != w.lower()]
            out.append(self._case(w, self._closest_len(pool, len(w), "org", w.lower(), str(i))))
        res = " ".join(out)
        return res if res != text else res + " GROUP"

    def mrz(self, text):
        from .mrz import rewrite
        return rewrite(self, text)

    def _person_fallback(self, text):
        """Last-resort person replacement that does not keep source name tokens."""
        prefix, core = _split_docket_prefix((text or "").strip())
        toks = [w for w in core.split() if re.search(r"[A-Za-z]", w)]
        if len(toks) >= 2:
            body = self.person(" ".join((toks[0], toks[-1])))
            if body:
                # person() may re-add a prefix if toks[0] looks like one; use body as-is if it already has prefix
                return body if body.startswith(prefix) else f"{prefix}{body}"
        if toks:
            fake = self.surname(re.sub(r"[^A-Za-z'’-]", "", toks[-1]) or "Name")
            body = fake.upper() if (core.isupper() and core) else fake
            return f"{prefix}{body}" if prefix else body
        return (prefix + "Roe") if prefix else "Roe"

    def replace(self, kind, text):
        k = (kind, text)
        if k not in self._cache and kind not in NOT_IDENTIFIER_KINDS and re.search(r"[^0-9A-Za-z]", text.strip()) and \
                len(re.sub(r"\D", "", text)) >= 5:
            # an identifier printed with spaces/dashes gets the fake of its compact form, re-printed in this layout
            # ('4417009322 81' and '441700932281' are one account)
            compact = re.sub(r"[^0-9A-Za-z]", "", text)
            rc = re.sub(r"[^0-9A-Za-z]", "", self.replace(kind, compact))
            if len(rc) == len(compact) and rc != compact:
                it = iter(rc)
                self._cache[k] = "".join(next(it) if ch.isalnum() else ch for ch in text)
        if k not in self._cache:
            f = {"GIVEN": self.given_only, "SURNAME": self.surname_only, "ORG": self.org, "PERSON": self.person, "CARD": self.card, "CARD_LAST4": self.last4, "PHONE": self.phone, "EMAIL": self.email,
                 "ZIP": self.zip, "ADDRESS": self.address, "DOB": self.dob, "MRZ": self.mrz}.get(kind)
            out = f(text) if f else self.digits(text, kind)
            if out is None:
                # never fall through to digits() for a person: that appends '0' to the
                # original text when there are no digits, leaking the source name.
                out = self._person_fallback(text) if kind == "PERSON" else self.digits(text, kind)
            self._cache[k] = out
        return self._cache[k]


# Non-Latin names: map to a Latin key (so a Greek name and its printed Latin transliteration share one fictional
# name), pick the fictional Latin name, then write it back in the source script.
_GR2LAT = [("ΟΥ", "OU"), ("ΑΙ", "AI"), ("ΕΙ", "EI"), ("ΜΠ", "B"), ("ΝΤ", "D"), ("Θ", "TH"), ("Χ", "CH"), ("Ψ", "PS"), ("Φ", "F")] + \
    list(zip("ΑΒΓΔΕΖΗΙΚΛΜΝΞΟΠΡΣΤΥΩ", ["A", "V", "G", "D", "E", "Z", "I", "I", "K", "L", "M", "N", "X", "O", "P", "R", "S", "T", "Y", "O"]))
_LAT2GR = [("TH", "Θ"), ("CH", "Χ"), ("PS", "Ψ"), ("OU", "ΟΥ"), ("PH", "Φ")] + list(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ΑΒΚΔΕΦΓΗΙΖΚΛΜΝΟΠΚΡΣΤΟΥΒΩΞΥΖ"))
_CY2LAT = list(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", ["A", "B", "V", "G", "D", "E", "E", "ZH", "Z", "I", "Y", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "F", "KH", "TS", "CH", "SH", "SHCH", "", "Y", "", "E", "YU", "YA"]))
_LAT2CY = [("SHCH", "Щ"), ("ZH", "Ж"), ("KH", "Х"), ("TS", "Ц"), ("CH", "Ч"), ("SH", "Ш"), ("YU", "Ю"), ("YA", "Я")] + \
    list(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "АБКДЕФГХИЖКЛМНОПКРСТУВВКЙЗ"))


def _ocr_script_fix(text):
    """OCR artefacts in non-Latin names: CJK corner bracket for Γ, letter-spaced capitals."""
    t = text.replace("「", "Γ").replace("Г", "Г")
    if re.fullmatch(r"(?:\S ){2,}\S", t.strip()):
        t = t.replace(" ", "")
    return t


def _script(text):
    gr = sum(1 for ch in text if "\u0370" <= ch <= "\u03ff")
    cy = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters and gr >= 1 and gr >= cy:
        return "greek"
    if letters and cy >= 1:
        return "cyrillic"
    return "latin"


def _subst(text, table):
    out, i, up = "", 0, text.upper()
    while i < len(up):
        for a, b in table:
            if up.startswith(a, i):
                out += b; i += len(a); break
        else:
            out += up[i]; i += 1
    return out


def to_latin(text):
    sc = _script(text)
    if sc == "latin":
        return text
    # OCR mixes look-alike Latin capitals into Greek/Cyrillic words (an Omega among Latin I, A, N, O, Y): normalise those first
    look = {"greek": dict(zip("ABEZHIKMNOPTYX", "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ")), "cyrillic": dict(zip("ABEKMHOPCTYX", "АВЕКМНОРСТУХ"))}[sc]
    t = "".join(look.get(ch, ch) for ch in text.upper())
    return _subst(t, _GR2LAT if sc == "greek" else _CY2LAT)


def _in_script(text, fn):
    sc = _script(text)
    if sc == "latin":
        return fn(text)
    lat = fn(to_latin(text).strip().upper())
    out = _subst(lat.upper(), _LAT2GR if sc == "greek" else _LAT2CY)
    return out if text.isupper() or not any(c.islower() for c in text) else out.capitalize()


PRIORITY = {"CARD": 0, "ADDRESS": 1, "EMAIL": 1, "PERSON": 2, "ACCOUNT": 3, "ID": 3, "PHONE": 3, "DOB": 4, "CARD_LAST4": 5, "SSN_LAST4": 5, "ZIP": 5}


ORG_WORDS = re.compile(r"\b(?:ELECTRONICS|BANK|STORES?|MARKET|MART|INC|LLC|LTD|CORP(?:ORATION)?|CO|COMPANY|PHARMACY|DRUG|"
                       r"RESTAURANT|CAFE|COFFEE|GRILL|DINER|HOTEL|MOTEL|AUTO|MOTORS|SERVICES?|SUPPLY|HARDWARE|GROCERY|FOODS?|"
                       r"DEPARTMENT|POLICE|SHERIFF|CREDIT UNION|FEDERAL|HOSPITAL|CLINIC|OUTLET|BOUTIQUE|GAS|FUEL|STATION|"
                       r"MEDICINE|MEDICAL|HEALTH|HEALTHCARE|DENTAL|PEDIATRICS|PHYSICIANS|URGENT CARE|IMAGING|LABS?|LABORATORY|"
                       r"FAMILY PRACTICE|CENTER|CENTRE|INSURANCE|MUTUAL|ASSURANCE|UNIVERSITY|COLLEGE|SCHOOL)\b\.?", re.I)


def org_header(prev):
    """A street line directly under a business/agency name (receipt or letterhead) is that organization's address:
    evidence to preserve, not personal data. Person-cue lines never qualify."""
    if not prev or re.search(r"\d", prev) or len(prev) > 60:
        return False
    if re.search(r"\b(?:Name|Cardholder|Customer|Dear|Attn|To)\b", prev, re.I):
        return False
    return ORG_WORDS.search(prev) is not None


# a help desk named after its audience ('Member services', 'Customer care') is the organisation's contact, not a person cue
PERSON_CUE_LINE = re.compile(r"\b(?:Name|Cardholder|Customer|Patient|Member|Dear|Attn|To|Bill to|Ship to|Insured|Subscriber)\b"
                             r"(?!\s+(?:services?|support|care|relations|help\s*desk|helpline|hotline)\b)", re.I)


def org_contact_lines(page_lines):
    """Line ids of an organisation's contact block: the lines right under a business/agency name (receipt header,
    letterhead, clinic header) that carry its address, store number or phone, up to three, until a line without
    digits or a person cue ('Cardholder:', 'Patient:'). Their ADDRESS / ZIP / PHONE values are the organisation's,
    evidence to keep (A10a: the clinic ZIP was replaced and overprinted its phone)."""
    out, block = set(), 0
    for l in page_lines:
        t = l.text.strip()
        if not t:
            continue
        if org_header(t):
            block = 3
            continue
        if block and re.search(r"\d", t) and not PERSON_CUE_LINE.search(t) and len(t) <= 80:
            out.add(l.id); block -= 1
        else:
            block = 0
    return out


def build_occurrences(docs, proposals, propagate=True, kept=()):
    """Normalize + merge proposals per line, propagate values across the case, resolve persons. `kept`: texts kept as
    facts (places, businesses); a name or street word inside one is not propagated there."""
    lines = {l.id: (d, p, l) for d in docs for p in d.pages for l in p.lines}
    prev_line = {}
    org_lines = set()
    for d in docs:
        for p in d.pages:
            last = None
            org_lines |= org_contact_lines([l for l in p.lines if getattr(l, "source", "") != "json"])
            for l in p.lines:
                if l.text.strip():
                    prev_line[l.id] = last
                    last = l.text
    card_last4 = set()  # last fours of full identifiers in this case (a repeated-digit '1111' is then not a placeholder)
    for pr in proposals:
        if pr.kind in SUFFIX_SOURCE_KINDS and pr.line_id in lines:
            dg = re.sub(r"\D", "", lines[pr.line_id][2].text[pr.start:pr.end])
            if len(dg) >= (13 if pr.kind == "CARD" else 7):
                card_last4.add(dg[-4:])
    norm = []
    for pr in proposals:
        if pr.line_id in lines:
            n = normalize(pr, lines[pr.line_id][2].text, card_last4)
            if n and n.kind == "ADDRESS" and org_header(prev_line.get(n.line_id)):
                REJECTED.append({"detector": n.detector, "kind": n.kind, "reason": "ORGANIZATION_ADDRESS_PRESERVED"}); KEPT_VALUES.add(n.text)
                continue
            if n and n.kind in ("ADDRESS", "ZIP", "PHONE") and n.line_id in org_lines and n.detector != "reviewer":
                REJECTED.append({"detector": n.detector, "kind": n.kind, "reason": "ORGANIZATION_CONTACT_PRESERVED"}); KEPT_VALUES.add(n.text)
                continue
            if n:
                norm.append(n)
    by_line = {}
    for n in norm:
        by_line.setdefault(n.line_id, []).append(n)
    if propagate:
        _propagate(lines, norm, by_line, kept)
        # a value seen on a person's line elsewhere in the case stays as printed in an organisation's contact block
        for lid in org_lines & set(by_line):
            keep = [x for x in by_line[lid] if x.kind in ("ADDRESS", "ZIP", "PHONE") and x.detector != "reviewer"]
            for x in keep:
                REJECTED.append({"detector": x.detector, "kind": x.kind, "reason": "ORGANIZATION_CONTACT_PRESERVED"}); KEPT_VALUES.add(x.text)
            by_line[lid] = [x for x in by_line[lid] if x not in keep]
    _vote_name_kinds(lines, norm, by_line)
    _particle_surname_mentions(lines, norm, by_line)
    if propagate:
        _propagate_parts(lines, norm, by_line, set(kept) | KEPT_VALUES)
    occs = []
    for lid, props in by_line.items():
        d, p, l = lines[lid]
        merged = {}
        for pr in props:
            merged.setdefault((pr.start, pr.end, pr.kind), []).append(pr)
        chosen = []
        for (s, e, kind), group in sorted(merged.items(), key=lambda kv: (-(kv[0][1] - kv[0][0]), PRIORITY.get(kv[0][2], 9))):
            if any(s < ce and e > cs for cs, ce, _ in chosen):
                # contained span: credit its detectors to the containing span
                for cs, ce, occ in chosen:
                    if cs <= s and e <= ce:
                        occ.detectors.update(x.detector for x in group)
                continue
            occ = Occ("", d.id, p.index, lid, s, e, kind, l.text[s:e], l.source, {x.detector for x in group},
                      max(x.confidence for x in group))
            occ.hints = {x.person_hint for x in group if x.person_hint}
            occ.meta = {k: v for x in group for k, v in x.meta.items()}
            chosen.append((s, e, occ))
        occs.extend(o for _, _, o in chosen)
    occs.sort(key=lambda o: (o.doc, o.page, o.line_id, o.start))
    unify_identifier_kinds(occs)
    for i, o in enumerate(occs):
        o.id = f"t{i:04d}"
        if o.source in ("ocr", "ocr_pixel_only") and lines[o.line_id][2].conf < 0.85:
            o.review.append("LOW_OCR_CONFIDENCE")
        if o.confidence < 0.6:
            o.review.append("LOW_DETECTOR_CONFIDENCE")
        if o.meta.get("standalone_last4_line"):
            o.review.append("LAST4_MATCHED_ON_ITS_OWN_LINE")
        if o.meta.get("line_citation_wrong"):
            o.review.append("LLM_CITED_WRONG_LINE")
        if o.detectors == {"propagation"}:
            o.review.append("FOUND_ONLY_BY_CASE_PROPAGATION") if o.kind != "PERSON" else None
    resolve_people(occs)
    return occs


GENERIC_ID_KINDS = ("ID", "ACCOUNT")
NOT_IDENTIFIER_KINDS = {"PERSON", "GIVEN", "SURNAME", "ORG", "ADDRESS", "EMAIL", "PHONE", "ZIP", "CARD", "CARD_LAST4",
                        "SSN_LAST4", "MRZ", "DOB"}
_DATE_LIKE = re.compile(r"\d{1,4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}")


def identifier_key(text):
    """Entity key of an account / member / record number: letters and digits only, so '4417009322 81',
    '441700932281' and '4417-0093-2281' are one value."""
    return re.sub(r"[^0-9A-Za-z]", "", text).upper()


def unify_identifier_kinds(occs):
    """One identifier, one kind, across the case. Detectors label the same number differently ('ACCOUNT' on the
    statement, 'ID' in the trace, 'MEMBER_ID' from the category pack, or even 'DOB' from a mis-read form label), and the
    kind picks the replacement generator, so mixed kinds gave one value several fakes. Each group of occurrences
    with the same identifier key takes the most specific kind seen (a category kind such as MEMBER_ID or NPI over
    ACCOUNT over ID). A 'DOB' whose text is not a date is an identifier."""
    for o in occs:
        if o.kind == "DOB" and not _DATE_LIKE.search(o.text) and re.search(r"\d{5,}", o.text) and \
                not re.fullmatch(r"(?:19|20)\d{6}", re.sub(r"\D", "", o.text)):
            o.kind = "ID"
            o.review.append("KIND_CORRECTED_DOB_TO_ID")
    groups = {}
    for o in occs:
        if o.kind in NOT_IDENTIFIER_KINDS:
            continue
        k = identifier_key(o.text)
        if len(re.sub(r"\D", "", k)) >= 5:
            groups.setdefault(k, []).append(o)
    for k, os_ in groups.items():
        kinds = {o.kind for o in os_}
        if len(kinds) == 1:
            continue
        specific = sorted(kinds - set(GENERIC_ID_KINDS), key=lambda x: (-sum(o.kind == x for o in os_), x))
        win = specific[0] if specific else ("ACCOUNT" if "ACCOUNT" in kinds else "ID")
        for o in os_:
            o.kind = win


def _vote_name_kinds(lines, norm, by_line):
    """One word is one name part across the case: 'Monika' labelled given name in the agent trace and guessed as a
    surname from a garbled ID label must get the same kind (and so the same fake) everywhere. Structured keys (JSON)
    outrank layout guesses."""
    from dataclasses import replace as _dc_replace
    score = {}
    for n in norm:
        if n.kind in ("GIVEN", "SURNAME"):
            src = getattr(lines[n.line_id][2], "source", "")
            sc = n.confidence + (0.2 if src == "json" else 0.0)
            k = n.text.strip().lower()
            score.setdefault(k, {}).setdefault(n.kind, 0.0)
            score[k][n.kind] = max(score[k][n.kind], sc)
    win = {k: max(v, key=v.get) for k, v in score.items() if len(v) > 1}
    if not win:
        return
    for lid, props in by_line.items():
        for i, pr in enumerate(props):
            if pr.kind in ("GIVEN", "SURNAME") and win.get(pr.text.strip().lower(), pr.kind) != pr.kind:
                props[i] = _dc_replace(pr, kind=win[pr.text.strip().lower()])


def _variants(text):
    p = parse_person(text)
    if not p or not p["given"] or len(p["given"]) < 2:
        return []
    g, m, s = p["given"], p["middle"], p["surname"]
    forms = [f"{g} {s}", f"{s}, {g}", f"{g[0]}. {s}"]
    if m:
        forms += [f"{g} {m}. {s}", f"{g} {m} {s}", f"{s}, {g} {m}"]
    return forms


def _role_titles():
    from .legal import ROLE_TITLES
    return ROLE_TITLES


# words in front of a KNOWN card's last four: 'ending in', 'ends with', 'last four digits:', 'last 4 of the card is'
LAST4_CUE = (r"(?:ending(?: in| with)?\s+|ends? (?:in|with)\s+|last ?(?:4|four)(?: digits)?(?: (?:of|on|for) (?:the |my |your |this |her |his |their )?"
             r"[A-Za-z]+(?: (?:number|no\.?|#))?)?(?: (?:is|are))?[:\s]\s*)")
# a masked number in front of a known value's last four: '**** 1111', '***-***-0187', 'XXX-XX-6789', '\u2022\u2022\u2022\u2022 4242'
LAST4_MASK = r"(?:[*xX\u2022#]{3,}[\s*xX\u2022#\-]*)"
# full identifiers whose last four can stand in for them (the suffix gets the last four of the value's replacement)
SUFFIX_SOURCE_KINDS = {"CARD", "ACCOUNT", "ID", "MEMBER_ID", "PHONE", "SSN", "PASSPORT", "LICENSE", "IBAN", "POLICY"}


def _propagate(lines, norm, by_line, kept=()):
    values = {}
    for n in norm:
        if n.confidence >= 0.5:
            values.setdefault(n.kind, {}).setdefault(n.text, max(n.confidence, values.get(n.kind, {}).get(n.text, 0)))
    patterns = []
    surnames = set()
    suffixes = {re.sub(r"\D", "", v)[-4:] for k, vs in values.items() if k in SUFFIX_SOURCE_KINDS for v in vs
                if len(re.sub(r"\D", "", v)) >= 7}
    for kind, vals in values.items():
        for v, conf in vals.items():
            if kind == "PERSON":
                p = parse_person(v)
                # 'Captain Hutaree' is a title+handle, not a source of the surname Hutaree
                if p and p["surname"] and len(p["surname"]) > 1:
                    given_u = (p.get("given") or "").upper()
                    first = (v.split() or [""])[0].strip(".,").upper()
                    if given_u not in _role_titles() and first not in _role_titles():
                        surnames.add(p["surname"])
                for f in dict.fromkeys([v] + (_variants(v) or [])):  # (the full form too: 'Mary Ellen Felps')
                    patterns.append((re.compile(r"(?<![\w.])" + re.escape(f).replace(r"\ ", r"\s+") + r"(?![\w])", re.I), kind, conf))
            elif kind == "CARD_LAST4":
                patterns.append((re.compile(r"(?:[*xX\u2022]{3,}[\s*xX\u2022-]*|" + LAST4_CUE + r"|(?:card|account|acct|checking|savings)[^\n]{0,20}?"
                                            r"|\b(?:VISA|MASTERCARD|MASTER|MC|AMEX|DISCOVER|DEBIT|CREDIT)\s*[:#]?\s*)(" + re.escape(v) + r")\b", re.I), kind, conf))
            elif kind == "CARD":
                last4 = re.sub(r"\D", "", v)[-4:]
                patterns.append((re.compile(r"(?<!\d)" + re.escape(v) + r"(?!\d)"), kind, conf))
                patterns.append((re.compile(r"(?:" + LAST4_MASK + r"|" + LAST4_CUE + r")(" + last4 + r")\b", re.I), "CARD_LAST4", conf))
            else:
                patterns.append((re.compile(r"(?<![\w])" + re.escape(v) + r"(?![\w])"), kind, conf))
                dv = re.sub(r"\D", "", v)
                if kind in SUFFIX_SOURCE_KINDS and len(dv) >= 7:
                    # the same identifier by its last four ('phone ending in 0187', 'ID ***1904'); linked to the full
                    # value's replacement when the case is planned
                    patterns.append((re.compile(r"(?:" + LAST4_MASK + r"|" + LAST4_CUE + r")(" + dv[-4:] + r")\b", re.I),
                                     "CARD_LAST4", min(conf, 0.85)))
                if kind in ("ID", "ACCOUNT") and re.search(r"[IO]", v) and re.search(r"\d", v):
                    # OCR reads a printed I/O as 1/0 (and back): match the confusable forms of an ID from the trace
                    conf_rx = "".join("[I1l|]" if ch == "I" else "[O0]" if ch == "O" else "[1Il|]" if ch == "1" else "[0O]" if ch == "0"
                                      else re.escape(ch) for ch in v)
                    patterns.append((re.compile(r"(?<![\w])" + conf_rx + r"(?![\w])"), kind, conf))
                if kind == "ADDRESS" and "," in v:  # the street line alone
                    street = re.match(r"[^,]+(?:,\s*(?:Apt|Suite|Unit)\.?\s*\w+)?", v).group()
                    patterns.append((re.compile(r"(?<![\w])" + re.escape(street) + r"(?![\w])"), kind, conf))
    # a given name or surname of a known person printed alone on its own line (ID cards stack surname / given name)
    parts = {}
    for v in values.get("PERSON", {}):
        pp = parse_person(v)
        if pp and pp.get("given") and pp.get("surname") and len(pp["given"]) > 1 and len(pp["surname"]) > 1:
            parts[pp["given"].lower()] = "GIVEN"; parts[pp["surname"].lower()] = "SURNAME"
    for lid, (d, p, l) in lines.items():
        t = l.text.strip()
        k = parts.get(t.lower())
        if k and len(t) >= 3:
            s0 = l.text.index(t)
            by_line.setdefault(lid, []).append(Proposal(lid, s0, s0 + len(t), k, t, "propagation", 0.85))
    # after an honorific, a person's surname or one part of a compound surname ('Ms. Okafor' for Ferreira-Okafor)
    hon = set(surnames) | {x.capitalize() for s in surnames for x in surname_parts(s) if len(x) >= 3 and x not in DOC_WORDS}
    for s in sorted(hon, key=len, reverse=True):
        patterns.append((re.compile(r"\b" + HONORIFIC + r"\s+(" + re.escape(s) + r")\b"), "PERSON_SURNAME", 0.85))
    # a known person's given name or surname used alone in prose ('Thanks, Marisol.', 'Okoye's card'): same identity.
    # Title-case or all-caps whole words only; names that are also ordinary words ('Will', 'Grace', 'Brown') only next
    # to a greeting/sign-off or an honorific, so ordinary prose is not rewritten.
    strong = {}
    for v, conf in values.get("PERSON", {}).items():
        pp = parse_person(v)
        first = (v.split() or [""])[0].strip(".,").upper()
        if pp and pp.get("given") and pp.get("surname") and len(pp["given"]) > 1 and \
                pp["given"].upper() not in _role_titles() and first not in _role_titles():   # not 'Captain Hutaree'
            strong[pp["given"].lower()] = "GIVEN"; strong[pp["surname"].lower()] = "SURNAME"
    for word, k in strong.items():
        if len(word) < 3 or word in DOC_WORDS:
            continue
        forms = {word.capitalize(), word.upper(), "-".join(x.capitalize() for x in word.split("-"))}
        alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
        if word in COMMON_WORD_NAMES:
            rx = re.compile(r"(?:\b(?:Hi|Hello|Hey|Dear|Thanks|Thank you|Sincerely|Regards|Cheers|Bye|Goodbye)\b[,!]?\s+|" + HONORIFIC + r"\s+)(" + alt + r")(?![\w'’-])")
        else:
            rx = re.compile(r"(?<![\w'’-])(" + alt + r")(?![\w’-])")
        patterns.append((rx, k, 0.8))
    # full-span: a given name (and optional middle or initial) in front of a known surname, so
    # 'Narges WITT' is replaced as a whole rather than leaving ghost ink around a bare 'WITT'.
    from .legal import (STOP as _LEGAL_STOP, PARTICLES as _PARTICLES, given_names as _given_names,
                        SUFFIX as _SUFFIX, ROLE_TITLES as _ROLE_TITLES, DETERMINERS as _DETERMINERS)
    _G = _given_names()
    _fence = {x.upper() for x in _LEGAL_STOP} | set(_ROLE_TITLES) | set(_DETERMINERS) | {
        "AGAINST", "BETWEEN", "INCLUDING", "NAMED", "CALLED", "SEE", "BEFORE", "AFTER", "UNDER", "OVER", "INTO",
        "UNTO", "UPON", "WITHIN", "WITHOUT", "ABOUT", "ABOVE", "DURING", "THROUGH", "FOLLOWING", "PURSUANT",
        "HEREIN", "THEREOF", "COUNTY", "CITY", "TOWN", "STATE", "COUNTRY", "VERSUS", "VS",
    }
    for lid, (d, p, l) in lines.items():
        for sn in surnames:
            if len(sn) < 3:
                continue
            rx = re.compile(r"(?<![\w'’-])((?:[A-Z][A-Za-z'’\-]{1,}\s+){1,2}(?:[A-Z]\.\s+)?)(" +
                            re.escape(sn) + r")(?![\w])", re.I)
            for m in rx.finditer(l.text):
                head = m.group(1)
                s0 = m.start()
                # a greeting / cue word in front ('Dear Rosalind Achterberg', 'Thanks Odile ...') is not part of the
                # name, even when it is also listed as a given name somewhere ('Dear', 'Will')
                lead = re.match(r"(?i)(?:dear|hi|hello|hey|thanks|thank|attn|to|from|cc|signed|patient|member|customer|"
                                r"dr|mr|mrs|ms|miss|mx|prof)\b\.?[,:]?\s+", head) or PERSON_LABEL_PREFIX.match(head) or \
                    SELF_INTRO_PREFIX.match(head)
                if lead:
                    head = head[lead.end():]; s0 += lead.end()
                    if not head.strip():
                        continue
                toks = re.findall(r"[A-Za-z'’\-]+|\b[A-Z]\b", head)
                ok = True
                for tok in toks:
                    cu = tok.strip(".").upper()
                    if re.fullmatch(r"[A-Z]\.", tok + ("." if len(tok) == 1 else "")) or re.fullmatch(r"[A-Z]", tok):
                        continue
                    if tok.lower() in _PARTICLES or cu in _SUFFIX or cu in _G:
                        continue
                    if cu in _fence or len(cu) < 2:
                        ok = False
                        break
                if ok:
                    s, e = s0, m.end()
                    s, e = _trim_person_words(l.text, s, e)
                    by_line.setdefault(lid, []).append(Proposal(lid, s, e, "PERSON", l.text[s:e], "propagation", 0.85,
                                                                None, {"why": "full-span-known-surname"}))
    from .verify import _ORG_NEXT
    kept = sorted({k for k in kept if k and len(k) >= 3}, key=len, reverse=True)
    for lid, (d, p, l) in lines.items():
        in_kept = [(m.start(), m.end()) for k in kept for m in re.finditer(re.escape(k), l.text)] if kept else []
        for rx, kind, conf in patterns:
            for m in rx.finditer(l.text):
                s, e = (m.span(1) if rx.groups else m.span())
                if kind in ("GIVEN", "SURNAME") and _ORG_NEXT.match(l.text, e):
                    continue  # a business named with the word ('HOLLOWAY AUTO', 'Vance Services LLC') is a kept fact
                if kind in ("GIVEN", "SURNAME") and any(s < ke and e > ks for ks, ke in in_kept):
                    continue  # a word inside a kept merchant name ('Rosalind Thistle' next to 'Rosalind Kellerman')
                k = "PERSON" if kind == "PERSON_SURNAME" else kind
                by_line.setdefault(lid, []).append(Proposal(lid, s, e, k, l.text[s:e], "propagation", min(conf, 0.9)))
        # JSON leaf whose key says card_last4 is handled by rules; also last4 anywhere in a json leaf equal to the value
        leaf = l.text.strip()
        key = (getattr(l, "path", None) or ("",))[-1]
        if getattr(l, "source", None) == "json" and re.fullmatch(r"\d{4}", leaf) and leaf in suffixes and \
                re.search(r"last|suffix|ending|four|_4|4$", str(key), re.I) and \
                leaf not in values.get("CARD_LAST4", {}):
            # a JSON field such as 'phone_last4' / 'id_suffix' holding the last four of a value known in the case
            s0 = l.text.index(leaf)
            by_line.setdefault(lid, []).append(Proposal(lid, s0, s0 + 4, "CARD_LAST4", leaf, "propagation", 0.9))
        if "CARD_LAST4" in values and l.text.strip() in values["CARD_LAST4"]:
            s0 = l.text.index(l.text.strip())
            pr = Proposal(lid, s0, s0 + len(l.text.strip()), "CARD_LAST4", l.text.strip(), "propagation", 0.9 if l.source == "json" else 0.7)
            if l.source != "json":
                pr.meta["standalone_last4_line"] = True
            by_line.setdefault(lid, []).append(pr)


PART_KIND = {"GIVEN": "GIVEN", "MIDDLE": "GIVEN", "SURNAME": "SURNAME", "STREET": "ADDRESS"}
# a hyphenated surname written as Title or ALL-CAPS words ('Vance-Brennecke', 'OSTRANDER-VANCE')
# (no OCR-glued 'MarisolQuintero-Hale': a part is Title case or capitals)
_HYPHENATED = re.compile(r"(?<![A-Za-z'’-])[A-Z](?:[a-z']+|[A-Z']+)(?:-[A-Z](?:[a-z']+|[A-Z']+))+(?![A-Za-z'’-])")


def _particle_surname_mentions(lines, norm, by_line):
    """A person found as 'De Silva' / 'Ms. Van Houten' / 'DE SILVA' when the case also has 'Ana De Silva' is that
    person's surname, not a given name 'De' + 'Silva': the span (honorific left out) becomes a SURNAME linked to the full
    name, so it gets the person's one fake surname. Likewise a hyphenated surname printed with a space instead of the
    hyphen ('Pennick Hale' / 'PENNICK HALE' when the case has 'Marisol Pennick-Hale') is that surname, not a second
    person with given name 'Pennick'."""
    from dataclasses import replace as _dc_replace
    known, spaced = {}, {}
    for n in norm:
        if n.kind != "PERSON" or n.confidence < 0.5:
            continue
        p = parse_person(n.text)
        if p and p.get("given") and len(p["given"]) > 1 and particle_surname(p["surname"]):
            known.setdefault(p["surname"].lower(), set()).add(n.text)
        elif p and p.get("given") and len(p["given"]) > 1 and len(surname_parts(p["surname"])) == 2:
            spaced.setdefault(" ".join(surname_parts(p["surname"])), {}).setdefault(p["surname"], set()).add(n.text)
    spaced = {k: v for k, v in spaced.items() if len({x.lower() for x in v}) == 1}
    if not known and not spaced:
        return
    swap = {}
    for lid, props in by_line.items():
        text = (lines[lid][2].text or "") if lid in lines else ""
        for i, pr in enumerate(props):
            if pr.kind != "PERSON":
                continue
            span = text[pr.start:pr.end]
            m = re.match(r"\s*(?:" + HONORIFIC + r"\s+)?", span)
            core = span[m.end():].rstrip()
            sp = spaced.get(" ".join(core.lower().split())) if len(core.split()) == 2 else None
            if sp and core.lower() not in known:
                full, srcs = next(iter(sp.items()))
                s0 = pr.start + m.end()
                new = _dc_replace(pr, start=s0, end=s0 + len(core), kind="SURNAME", text=core,
                                  meta={**pr.meta, "part_of": full, "part_from": sorted(srcs)})
                swap[id(pr)] = new
                props[i] = new
                continue
            if not core or core.lower() not in known or len(core.split()) < 2:
                continue
            p = parse_person(core)
            if p and p.get("given") and p["given"].lower() not in _NAME_PARTICLES:
                continue  # 'Ana De Silva' itself, or a different given name
            s0 = pr.start + m.end()
            new = _dc_replace(pr, start=s0, end=s0 + len(core), kind="SURNAME", text=core,
                              meta={**pr.meta, "part_of": core, "part_from": sorted(known[core.lower()])})
            swap[id(pr)] = new
            props[i] = new
    if swap:
        norm[:] = [swap.get(id(n), n) for n in norm]


def _propagate_parts(lines, norm, by_line, kept):
    """A given name, surname part or street name from a detected name / street line printed on its own elsewhere in
    the case ('Pike called back' after 'Ines Halvorsen-Pike', 'the Wrenfield house' after '2208 Wrenfield Drive') is
    the same value: proposed there, linked to the full value (meta part_of) so it gets the same fake. The words are the
    ones the output check looks for (verify.identifying_part_sources: Title or ALL-CAPS whole words, no city, street
    type, unit or everyday-word name); a word from two different values, a middle name, a word inside a kept fact or a
    business name, and spans another proposal already covers are left alone."""
    from .verify import identifying_part_sources, standalone_spans, titled_only_words, after_determiner, _norm, _ORG_NEXT
    vals = [(n.kind, n.text) for n in norm if n.confidence >= 0.5]
    full = {_norm(t) for _, t in vals}
    # a name known only after a title ('a.k.a. "Captain Velloran"') can also be a group's name ('the VELLORAN'):
    # a use after a determiner is left alone (the output check skips it too)
    group_ok = titled_only_words(vals)
    srcs = {}
    for kind, text in vals:
        for w, roles in identifying_part_sources([(kind, text)]).items():
            srcs.setdefault(w, {"roles": set(), "from": set()})
            srcs[w]["roles"] |= {(r, (sv or "").lower() or None) for r, sv in roles}
            srcs[w]["from"].add(text)
            srcs[w].setdefault("name", {}).update({(r, (sv or "").lower()): sv for r, sv in roles if sv})
    # a person found only by a surname already known in the case ('Quenneville's husband, Anselm Quenneville') is a
    # full name too: its words not yet known from another value ('Anselm called') are linked to it the same way
    extra = sorted({pr.text for props in by_line.values() for pr in props
                    if pr.kind == "PERSON" and pr.meta.get("why") == "full-span-known-surname"} - {t for _, t in vals})
    for text in extra:
        for w, roles in identifying_part_sources([("PERSON", text)]).items():
            if w in srcs or _norm(w) in full:
                continue
            srcs[w] = {"roles": {(r, (sv or "").lower() or None) for r, sv in roles}, "from": {text},
                       "name": {(r, (sv or "").lower()): sv for r, sv in roles if sv}}
    full |= {_norm(t) for t in extra}
    kept = sorted({k for k in kept if k and len(k) >= 3}, key=len, reverse=True)
    # a street name of two or more words printed on its own ('the Copper Fennel house' after '4517 Copper Fennel
    # Road') is one span with one fake, proposed before its words so they are not replaced one by one
    phrases = {}
    for w, info in srcs.items():
        if _norm(w) in full or len(info["roles"]) != 1:
            continue
        role, key = next(iter(info["roles"]))
        if role == "STREET" and key and len(key.split()) > 1 and _norm(key) not in full:
            phrases[key] = (info["name"][(role, key)], info["from"])
    for name, frm in sorted(phrases.values(), key=lambda x: -len(x[0])):
        rxs = [re.compile(r"(?<![A-Za-z'’-])" + r"[ \t]+".join(map(re.escape, f.split())) + r"(?![A-Za-z'’-])")
               for f in {name, name.upper()}]
        for lid, (d, p, l) in lines.items():
            text = l.text or ""
            masked = [(m.start(), m.end()) for k in kept for m in re.finditer(re.escape(k), text)]
            masked += [(pr.start, pr.end) for pr in by_line.get(lid, [])]
            for m in sorted((m for rx in rxs for m in rx.finditer(text)), key=lambda m: m.start()):
                s0, e0 = m.span()
                if any(s0 < me and e0 > ms for ms, me in masked) or _ORG_NEXT.match(text, e0):
                    continue  # covered already, inside a kept fact, or a business ('Copper Fennel Bakery')
                by_line.setdefault(lid, []).append(Proposal(lid, s0, e0, "ADDRESS", text[s0:e0], "propagation", 0.8,
                                                            None, {"part_of": name, "part_from": sorted(frm)}))
                masked.append((s0, e0))
    # a surname with particles printed without the given name ('Van der Brakel will call', 'VAN DER BRAKEL signed',
    # 'Ms. van der Brakel' after 'Ottilie van der Brakel') is one span with the fake surname, particles included
    psur = {}
    for w, info in srcs.items():
        if len(info["roles"]) != 1:
            continue
        role, key = next(iter(info["roles"]))
        if role == "SURNAME" and key and particle_surname(key):
            psur[key] = (info["name"][(role, key)], info["from"])
    for name, frm in sorted(psur.values(), key=lambda x: -len(x[0])):
        parts, core = particle_surname(name)
        rx = re.compile(r"(?<![A-Za-z'’-])(?i:" + r"[ \t]+".join(map(re.escape, parts.split())) + r")[ \t]+(?:" +
                        "|".join(re.escape(f) for f in {core, core.upper(), "-".join(x.capitalize() for x in core.split("-"))}) +
                        r")(?![A-Za-z'’-])")  # a name known only in capitals is also looked for in Title case
        for lid, (d, p, l) in lines.items():
            text = l.text or ""
            masked = [(m.start(), m.end()) for k in kept for m in re.finditer(re.escape(k), text)]
            masked += [(pr.start, pr.end) for pr in by_line.get(lid, [])]
            for m in rx.finditer(text):
                s0, e0 = m.span()
                if any(s0 < me and e0 > ms for ms, me in masked) or _ORG_NEXT.match(text, e0):
                    continue
                by_line.setdefault(lid, []).append(Proposal(lid, s0, e0, "SURNAME", text[s0:e0], "propagation", 0.8,
                                                            None, {"part_of": name, "part_from": sorted(frm)}))
                masked.append((s0, e0))
    for w, info in sorted(srcs.items()):
        if _norm(w) in full or len(info["roles"]) != 1:
            continue
        role, key = next(iter(info["roles"]))
        if role is None or key is None:
            continue
        source = info["name"][(role, key)]
        if role in ("GIVEN", "MIDDLE") and source != w:
            continue
        for lid, (d, p, l) in lines.items():
            text = l.text or ""
            if w.lower() not in text.lower():
                continue
            masked = [(m.start(), m.end()) for k in kept for m in re.finditer(re.escape(k), text)]
            masked += [(pr.start, pr.end) for pr in by_line.get(lid, [])]
            spans = standalone_spans(w, text, compound=False)
            if role == "SURNAME":  # the part inside another hyphenated form ('Vance-Brennecke', 'Ostrander-Vance')
                spans += [(m.start(), m.end()) for m in _HYPHENATED.finditer(text)
                          if any(x in (w, w.upper()) for x in m.group().split("-")) and not _ORG_NEXT.match(text, m.end())]
            for s0, e0 in spans:
                if any(s0 < me and e0 > ms for ms, me in masked):
                    continue
                if w in group_ok and after_determiner(text, s0):
                    continue
                by_line.setdefault(lid, []).append(Proposal(lid, s0, e0, PART_KIND[role], text[s0:e0], "propagation", 0.8,
                                                            None, {"part_of": source, "part_from": sorted(info["from"])}))
                masked.append((s0, e0))


def surname_parts(surname):
    """'Quintero-Hale' -> ['quintero', 'hale']; a single surname -> []."""
    parts = [x for x in re.split(r"[-‐‑–]", (surname or "").lower()) if x]
    return parts if len(parts) > 1 else []


def _part_of(short, full):
    return short != full and short in surname_parts(full)


def surname_aliases(occs):
    """{surname part: full compound surname} for resolved people printed with only part of their compound surname
    (after resolve_people). A part that is also the own surname of another entity in the case is left alone."""
    by_ent = {}
    for o in occs:
        p = parse_person(o.text) if o.kind == "PERSON" else None
        if p and p.get("surname"):
            by_ent.setdefault(o.entity, set()).add(p["surname"].lower())
    owners = {}
    for e, ss in by_ent.items():
        for s in ss:
            owners.setdefault(s, set()).add(e)
    out = {}
    for e, ss in by_ent.items():
        for s in ss:
            fulls = sorted(f for f in ss if _part_of(s, f))
            if len(fulls) == 1 and owners[s] == {e}:
                out[s] = fulls[0] if out.get(s, fulls[0]) == fulls[0] else None
    return {k: v for k, v in out.items() if v}


def resolve_people(occs):
    people = {}  # (given, surname) -> entity id
    pending = []
    for o in occs:
        if o.kind != "PERSON":
            o.entity = f"{o.kind.lower()}:{re.sub(r'[^0-9a-z]', '', to_latin(o.text).lower()) or o.text.strip().lower()}"
            continue
        p = parse_person(o.text)
        if not p:
            o.entity = "person:unparsed:" + re.sub(r"\W+", "", o.text.lower()); o.review.append("UNPARSED_NAME"); continue
        if p["given"] and len(p["given"]) > 1:
            key = (p["given"].lower(), p["surname"].lower())
            o.entity = people.setdefault(key, f"person:{len(people) + 1}")
        else:
            pending.append((o, p))
    # one part of a person's compound surname with the same given name ('Marisol Q. Hale' next to 'Marisol
    # Quintero-Hale') is that person, when exactly one compound fits
    own = {s for (g, s) in people}
    merged = {}
    for (g, s), e in people.items():
        if any(_part_of(s, s2) for s2 in own):
            fulls = sorted({e2 for (g2, s2), e2 in people.items() if g2 == g and _part_of(s, s2)})
            if len(fulls) == 1:
                merged[e] = fulls[0]
    if merged:
        for o in occs:
            if o.kind == "PERSON" and o.entity in merged:
                o.entity = merged[o.entity]
        people = {k: merged.get(e, e) for k, e in people.items()}
    for o, p in pending:
        if p["form"] == "surname":
            cands = {e for (g, s), e in people.items() if s == p["surname"].lower()} or \
                    {e for (g, s), e in people.items() if _part_of(p["surname"].lower(), s)}
            o.entity = next(iter(cands)) if len(cands) == 1 else "surname:" + p["surname"].lower()
            if len(cands) > 1:
                o.review.append("SURNAME_ONLY_SHARED_BY_MULTIPLE_PEOPLE (replacement coherent: shared fake surname)")
            continue
        cands = sorted({e for (g, s), e in people.items() if s == p["surname"].lower() and g[0] == p["given"].lower()}) or \
            sorted({e for (g, s), e in people.items() if _part_of(p["surname"].lower(), s) and g[0] == p["given"].lower()})
        if len(cands) == 1:
            o.entity = cands[0]
        else:
            o.entity = "ambiguous:" + o.text.lower()
            o.candidates = cands
            o.review.append("AMBIGUOUS_INITIAL (replacement coherent: candidates share a fake initial)")
