"""Person names in court and legal documents (indictments, complaints, affidavits, subpoenas), including image-only
scans where the only signal is OCR text. Legal documents name people in fixed ways the general rules miss:

- party captions and defendant lists in capitals ('D-3 JOSHUA MATTHEW STONE, a.k.a. ...', 'JONATHAN TOEBBE and');
- given name + surname in prose and signature blocks ('Robert W. Trumble, United States Magistrate Judge');
- several initials + surname ('H. W. Lowe', 'C. N. Hildreth Jr.');
- docket stamps 'Judge: Roberts, Victoria A';
- quoted or unquoted aliases after 'a.k.a.' / 'a/k/a' / 'also known as' ('a.k.a. "Jason Z. Charles"', 'a/k/a Narges WITT');
- ALL-CAPS or title-case name lines in a caption block after 'v.' / 'plaintiff', without requiring a US given name;
- the bare surname of a person named above, when printed in capitals ('defendant BACA') or after a role word.

A name in running prose is taken when its first token is a given name from the gazetteers in docs/data/
(US SSA public-domain list plus multilingual CC0 lists; see data/LICENSES.md) or it is initials + surname,
and no token is a legal / place / organisation word. Caption / a.k.a. / defendant-list cues can also take a
name whose given token is not in those lists, as long as the tokens look like a person name and are not a
heading or a common-English phrase ('Red Fox'). Only documents that read as legal documents are scanned.
Every proposal is grounded in its line and records the rule in meta['why'].

GLiNER is a second PERSON detector on legal documents only (never on the default non-legal path).
The same court / legal structure filters apply to its proposals and to PERSON spans from the general rules detector.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .detect import NOT_NAME_TOKENS, Proposal
from .gazetteer import BUSINESS as _BUSINESS

LEGAL_CUES = re.compile(r"(?i)\b(?:district court|superior court|circuit court|court of appeals|indictment|grand jury|"
                        r"criminal complaint|complaint|affidavit|subpoena|summons|plaintiffs?|defendants?|petitioners?|"
                        r"respondents?|case no|docket|magistrate|attorney|v\.|vs\.|in re)\b")
# Distinctive court-instrument language. Caption rules (unknown given names, typographic
# name-list lines) run only on documents that match this, not on litigation-support forms
# that merely mention a complaint or an attorney.
COURT_FILING_CUES = re.compile(
    r"(?i)\b(?:united states district court|district court|superior court|circuit court|"
    r"court of appeals|indictment|superseding indictment|grand jury|"
    r"criminal complaint|affidavit|subpoena|magistrate judge|"
    r"the grand jury charges|count one)\b")

# tokens that are never part of a person's name in these documents
STOP = NOT_NAME_TOKENS | {
    "THE", "OF", "AND", "FOR", "IN", "ON", "AT", "TO", "BY", "A", "AN", "OR", "AS", "WITH", "FROM", "ALL", "ANY", "THAT",
    "GRAND", "JURY", "INDICTMENT", "COMPLAINT", "CRIMINAL", "CIVIL", "AFFIDAVIT", "SUBPOENA", "SUMMONS", "WARRANT",
    "COUNT", "COUNTS", "CHARGES", "VIOLATION", "VIOLATIONS", "SECTION", "TITLE", "CODE", "CASE", "NO", "PLAINTIFF",
    "PLAINTIFFS", "DEFENDANT", "DEFENDANTS", "PETITIONER", "RESPONDENT", "JUDGE", "JUSTICE", "MAGISTRATE", "CLERK",
    "MARSHAL", "SHERIFF", "AGENT", "SPECIAL", "ATTORNEY", "ATTORNEYS", "COUNSEL", "GENERAL", "DIVISION", "CIRCUIT",
    "NORTHERN", "SOUTHERN", "EASTERN", "WESTERN", "CENTRAL", "MIDDLE", "NORTH", "SOUTH", "EAST", "WEST", "FEDERAL",
    "NATIONAL", "BUREAU", "INVESTIGATION", "POLICE", "AGENCY", "ADMINISTRATION", "GOVERNMENT", "COMMONWEALTH",
    "PEOPLE", "PRESIDENT", "SECRET", "RESTRICTED", "DATA", "NAVY", "ARMY", "AIR", "FORCE", "MARINE", "FIELD", "TASK",
    "STREET", "AVENUE", "ROAD", "BOULEVARD", "DRIVE", "LANE", "PLAZA", "SQUARE", "PARK", "BEACH", "RIVER", "LAKE",
    "MOUNTAIN", "VALLEY", "ISLAND", "BRIDGE", "HIGHWAY", "STATION", "AIRPORT", "CENTER", "CENTRE", "HALL", "BUILDING",
    "UNIVERSITY", "COLLEGE", "SCHOOL", "ACADEMY", "HOSPITAL", "CHURCH", "COUNCIL", "COMMITTEE", "COMMISSION", "BOARD",
    "PRISON", "JAIL", "FACILITY", "FACILITIES", "DETENTION", "CORRECTIONAL", "TOBACCO", "FOUNDATION", "INSTITUTE",
    "MEMORIAL", "BUREAU", "PLAINTIFF'S", "DEFENDANT'S", "SUPERSEDING", "SEALED", "MATTER", "FILED", "RECEIVED",
    "ORIGINAL", "COPY", "PAGE", "DOCUMENT", "EXHIBIT", "ORDER", "MOTION", "NOTICE", "SERVICE", "PROCESS", "LAW",
    "UNITED", "STATES", "AMERICA", "REPUBLIC", "ISLAMIC", "TERM", "VERSUS", "VS", "DOE",  # 'John Doe' placeholders
    "DEPARTMENT", "CUSTOMS", "IMMIGRATION", "NATURALIZATION", "COMMERCE", "TREASURY", "INTERIOR",
    "GOODS", "MERCHANDISE",
    "CLAIMANT", "COMMENTS", "STATUTORY", "TRANSMITTED", "ENCLOSURE", "ENCLOSURES", "ROUTING",
}
ORG_AFTER = re.compile(r"^\W*(?:\S+\s+){0,2}?(?:COMPANY|CO\.|CORP(?:ORATION)?|INC\b|LLC|LTD|BANK|TOBACCO|ASSOCIATES|"
                       r"PARTNERS|GROUP|& SONS|FOUNDATION|UNIVERSITY|COLLEGE|HOSPITAL)", re.I)
# given names that are also places/words: accepted only with a middle part (initial or middle name)
AMBIGUOUS_GIVEN = {"MAY", "WILL", "MARK", "BILL", "GRANT", "MAJOR", "RICH", "FAITH", "HOPE", "JOY", "JUNE", "APRIL",
                   "AUGUST", "PAGE", "DEAN", "CHASE", "GRACE", "ART", "GUY", "ROSE", "SUNNY", "ROYAL", "PRICE", "YOUNG",
                   "VIRGINIA", "GEORGIA", "CAROLINA", "INDIANA", "DAKOTA", "FLORIDA", "WASHINGTON", "AUSTIN", "JACKSON",
                   "LINCOLN", "MADISON", "HOUSTON", "CHARLOTTE", "PHOENIX", "COLUMBIA", "DALLAS", "TYLER", "TROY", "ORLANDO",
                   "ALEXANDRIA", "RICHMOND", "NORFOLK",
                   "JORDAN", "PARIS", "SAVANNAH", "CHINA", "INDIA", "ASIA", "JUSTICE", "KING", "SHERIFF", "COLONEL", "MILES",
                   "LOUIS", "SAINT", "SANTA", "HUNTER", "FISHER", "BAKER", "CARTER", "GIBSON", "LOGAN", "WARREN", "MARSHALL",
                   "MASON", "SPENCER", "TAYLOR", "PARKER", "COOPER", "CHRISTIAN", "LEE", "SAN", "ST", "SPECIAL", "CLERK",
                   "SAID"}  # 'SAID' is a Persian/Arabic given name and also legal jargon ('the said defendant')
PARTICLES = {"van", "von", "de", "del", "della", "der", "den", "da", "di", "du", "la", "le", "st.", "mc", "bin", "al", "el",
             "abu", "ibn", "bint", "das", "dos", "los", "las", "ter", "ten", "da", "do"}
ROLE_BEFORE = re.compile(r"(?:\b(?:defendants?|co-?defendants?|co-?conspirators?|Mr|Ms|Mrs|Dr|Agent|SA|Judge|Justice|Sheriff|"
                         r"Undersheriff|Captain|Capt|Lieutenant|Lt|Sergeant|Sgt|Deputy|Officer|Detective|Det|Inspector|"
                         r"Chief|Commander|Warden|witness|Witness)\.?\s+)$")
TOKEN = r"[A-ZÀ-ÖØ-ÞĀ-Ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]*\.?"
# Latin letters including diacritics that OCR keeps ('ÉDWARD', 'José', 'Moholy-Nagy')
_LATIN_U = r"A-ZÀ-ÖØ-ÞĀ-ŽǺ-ǾẀ-Ỳ"
_LATIN_L = r"A-Za-zÀ-ÖØ-öø-ÿĀ-žǺ-ǿẀ-ỳ"
CAP_WORD = rf"[{_LATIN_U}][{_LATIN_L}'’-]*"
CAP_WORD_DOT = CAP_WORD + r"\.?"
WORD_CHARS = rf"[{_LATIN_L}'’-]"
# end of a caption / defendant list (the heading itself is not a name)
CAPTION_END = re.compile(r"(?i)^\s*(?:defendants?\.?|the grand jury charges.*|indictment|superseding indictment|"
                         r"criminal complaint|general allegations|count one|information)\s*$")
CAPTION_START = re.compile(r"(?i)\b(?:v\.|vs\.|-vs-)\b|^\s*plaintiffs?\b")
# opens a typographic name list outside a 'v.' caption (affidavits, petitions, styled matters)
NAME_LIST_START = re.compile(
    r"(?i)\b(?:affidavit of|petition of|complaint (?:of|against)|application of|"
    r"in the matter of|in re\b|regarding|on behalf of|styled)\b"
)
# federal / state citation fragments often OCR'd as if they were names
CITATION = re.compile(
    r"(?i)\b(?:fed\.?\s*)?r\.?\s*(?:crim|civ|app|evid|bankr)\.?(?:\s*p\.?|\s*proc\.?)?\b|"
    r"\b(?:crim|civ|app|evid)\.?\s*p\.?\b"
)
# 'Nevada Bar', 'Eastern Division' — a court unit after a place word
COURT_UNITS = {"BAR", "DIVISION", "DISTRICT", "COUNTY", "CIRCUIT", "PARISH", "BOROUGH", "PRECINCT"}
US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT",
    "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
    "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "HAMPSHIRE",
    "JERSEY", "MEXICO", "YORK", "CAROLINA", "DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "ISLAND", "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA",
    "WASHINGTON", "WISCONSIN", "WYOMING",
}
# abbreviations only in a court-unit pattern, never as a free surname
US_STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY",
}
# apostrophe particles that OCR often splits from the rest of the name: "N' Djimbi", "O' Brien"
APOS_PARTICLE = re.compile(r"^[A-Z]['’]$")
# Defendant-list markers, including OCR that reads D-1 as D-I or D-l.
LINE_PREFIX = re.compile(r"^(?:D[\-.][\w]+|D\d+|#[\w]+[.)]?|\d+[.)]|\[[\w]+\]|\([\w]+\))\s+", re.I)
DOCKET_TOKEN = re.compile(r"^(?:D[\-.][\w]+|D\d+|#[\w]+)$", re.I)
AKA_MARK = re.compile(r"(?i)(?:a\.?\s*/?\s*k\.?\s*/?\s*a\.?|also known as)")
# common English words: two-word a.k.a. values made only of these are handles, not personal names ('Red Fox').
COMMON_ENGLISH = {
    "RED", "BLUE", "GREEN", "BLACK", "WHITE", "GOLD", "SILVER", "GREY", "GRAY", "PINK", "BROWN", "ORANGE", "PURPLE",
    "FOX", "WOLF", "BEAR", "DOG", "CAT", "BIRD", "HAWK", "EAGLE", "LION", "TIGER", "MOUSE", "RAT", "FISH", "SNAKE",
    "STAR", "MOON", "SUN", "SKY", "STONE", "ROCK", "TREE", "WOOD", "FIRE", "ICE", "SNOW", "RAIN", "WIND", "STORM",
    "LUCKY", "HAPPY", "CRAZY", "DARK", "LIGHT", "FAST", "SLOW", "BIG", "LITTLE", "OLD", "NEW", "FREE", "TRUE", "FAKE",
    "KING", "QUEEN", "PRINCE", "BOSS", "CHIEF", "MASTER", "GHOST", "SHADOW", "NIGHT", "DAY", "MAN", "BOY", "GIRL",
    "LADY", "TOP", "BEST", "ACE", "ONE", "TWO", "THREE", "FOX", "RED", "PEG",
    "PALE", "HORSE", "COLT", "MARE", "PONY", "STAG", "FAWN", "BUCK", "RAM", "BULL", "LAMB", "MUSTANG",
    "HILL", "WOODS", "BROOK", "CREEK", "FANCY", "GOODS", "MERCHANDISE", "TRADE", "BUSINESS",
    "IMPORT", "EXPORT", "WHOLESALE", "RETAIL",
    "OFFENSE", "DESCRIPTION", "ASSIGN", "ASSIGNMENT", "FILE", "UNDER", "SEAL",
    "COMMIT", "COMPUTER", "COMPUTERS", "INTRUSION", "INTRUSIONS", "OBTAINING",
    "INFORMATION", "PROTECTED", "SOLDIER", "SOLDIERS",
    "AGGRAVATED", "IDENTITY", "THEFT", "CONSPIRACY", "SERVED", "SERVICE",
}

SUFFIX = {"JR", "SR", "II", "III", "IV", "ESQ"}
# role / courtesy titles are not given names, even on a caption line ('Captain Hutaree')
ROLE_TITLES = {
    "CAPTAIN", "CAPT", "LIEUTENANT", "LT", "SERGEANT", "SGT", "DEPUTY", "OFFICER",
    "DETECTIVE", "DET", "INSPECTOR", "CHIEF", "COMMANDER", "WARDEN", "JUDGE",
    "JUSTICE", "MAGISTRATE", "CLERK", "AGENT", "SPECIAL", "HON", "HONORABLE",
    "MR", "MS", "MRS", "DR", "PROF",
}
# form / docket labels that OCR types like a two-word name
HEADING_LABELS = {"DATE", "TIME", "FILED", "ENTERED", "ASSIGN", "ASSIGNED", "PAGE", "DOCKET"}
# typographic form / transmittal furniture that caption rules otherwise treat as a name list
FORM_HEADER_PHRASES = {
    "EXECUTIVE OFFICES", "PAPERS TRANSMITTED", "REG MAIL", "MOTOR CARRIER",
    "FORM PREPARED BY", "FORM PREPARED-BY", "TEL CALL PLACED", "TELEPHONE CALL PLACED",
    "CERTIFIED MAIL", "REGISTERED MAIL", "RETURN RECEIPT",
}
FORM_HEADER_TOKENS = {
    "CLAIMANT", "STATUTORY", "COMMENTS", "TRANSMITTED", "ENCLOSURES", "ENCLOSURE",
    "ROUTING", "OFFICES", "PREPARED",
}
DETERMINERS = {"SOME", "EACH", "EVERY", "SUCH", "THESE", "THOSE", "THIS", "THAT"}
COURT_UNIT_STEMS = ("DISTRICT", "DIVISION", "CIRCUIT", "COURT", "COUNTY", "PARISH",
                    "BOROUGH", "PRECINCT", "BAR")
# leading office on a GLiNER / caption span: 'Chief Judge Beryl A. Howell'
TITLE_PREFIX = re.compile(
    r"^(?:(?:Chief|Special|Assistant|Deputy|Acting)\s+)?(?:Magistrate\s+)?"
    r"(?:Judge|Justice|Clerk|Agent|Officer|Captain|Detective|Inspector|"
    r"Sergeant|Lieutenant|Warden|Commander|Hon(?:orable)?)\s+",
    re.I,
)
# a command or process line that introduces a person ('WE COMMAND YOU TO SUMMON X')
NAME_CUE_BEFORE = re.compile(
    r"(?i)\b(?:summon(?:s)?|subpoena(?:ed)?|within named|named defendants?|witness the hon\.?|hon\.)\s+$"
)


@lru_cache(maxsize=1)
def given_names():
    d = Path(__file__).with_name("data")
    names = set()
    for p in sorted(d.glob("given_names_*.txt")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip() and not l.startswith("#"):
                names.add(l.strip().upper())
    return names


def is_legal(text):
    hits = {m.group(0).lower() for m in LEGAL_CUES.finditer(text)}
    return len(hits) >= 2


def is_court_filing(text):
    """True for charging instruments, affidavits, subpoenas, and similar court filings.

    Caption rules that accept unknown given names stay off on litigation-support or
    office forms that only share a couple of legal cue words (attorney, complaint).
    """
    return bool(is_legal(text) and COURT_FILING_CUES.search(text))


def _clean(tok):
    """Uppercase, strip punctuation, and fold diacritics so 'ÉDWARD' matches the given-name EDWARD."""
    t = tok.strip(".,;:'’\"“”()").upper()
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _is_heading_tokens(parts):
    return all(_clean(p) in STOP or _clean(p) in AMBIGUOUS_GIVEN or _clean(p) in SUFFIX for p in parts)


def _is_state_like(c):
    if c in US_STATES or c in US_STATE_ABBR:
        return True
    # typographic truncation of a longer state name (VIRG, VIRGIN), not a one-off typo list
    if len(c) >= 4:
        return any(st.startswith(c) for st in US_STATES if len(st) >= len(c) + 2)
    return False


def _is_court_unit_like(c):
    """DISTRICT, typographic 'DISTRICTI' / 'DISTRICE', or another court-unit stem."""
    if c in COURT_UNITS or c in {"COURT", "COURTS"}:
        return True
    if len(c) >= 5:
        for stem in COURT_UNIT_STEMS:
            if c.startswith(stem[:max(5, len(stem) - 1)]) or (len(c) >= 5 and stem.startswith(c)):
                return True
    return False


def _prepare_name_text(t):
    """Flatten dotted/underscore form-fills and insert a space in glued initials, with an index map back to t.

    'ANGELAD.CAESAR' / 'STEVEN W.MYHRE' / '...Romona..Grubt...' / 'defendantx_Ramona_Grubt'.
    Same-length substitutions keep original indexes; inserted spaces map to the following original index.
    """
    # glued initials first, so 'ANGELAD.CAESAR' becomes 'ANGELA D. CAESAR' and is not flattened to 'ANGELAD CAESAR'
    insert_at = set()
    for m in re.finditer(r"[A-Z]\.(?=[A-Z][A-Za-z])", t):
        insert_at.add(m.end())
    # require a real initial ('D.CAESAR'), not an ellipsis ('SUMMON...')
    for m in re.finditer(r"[A-Za-z]{3,}(?=[A-Z]\.[A-Za-z])", t):
        insert_at.add(m.end())
    out, mp = [], []
    for i, ch in enumerate(t):
        if i in insert_at:
            out.append(" ")
            mp.append(i)
        out.append(ch)
        mp.append(i)
    mid, mid_mp = "".join(out), mp
    # same-length: runs of dots/underscores, and a mid-word dot before a single initial ('Jemes.H')
    chars, mp2 = list(mid), list(mid_mp)
    mid2 = re.sub(r"\.{2,}", lambda m: " " * len(m.group()), mid)
    mid2 = re.sub(r"_+", lambda m: " " * len(m.group()), mid2)
    mid2 = re.sub(r"(?<=[A-Za-z]{2})\.(?=[A-Z](?:[\s.,;:]|$))", " ", mid2)
    # substitutions above are same length, so mid_mp still applies
    return mid2, mid_mp


def _orig_span(mp, s, e):
    if not mp or s >= len(mp):
        return s, e
    a = mp[s]
    b = mp[min(e, len(mp)) - 1] + 1 if e else a
    return a, b


def _joined_is_stop(parts):
    """Adjacent ALL-CAPS fragments that concatenate to a legal stop word (JUD + GE -> JUDGE)."""
    cores = [_clean(p) for p in parts if _clean(p)]
    for i in range(len(cores) - 1):
        j = cores[i] + cores[i + 1]
        if j in STOP or j in AMBIGUOUS_GIVEN:
            return True
    if len(cores) >= 2 and ("".join(cores) in STOP or "".join(cores) in AMBIGUOUS_GIVEN):
        return True
    return False


def _is_form_header(t, parts=None):
    """Office / transmittal / form-header furniture, not a person's name."""
    cores = [_clean(p) for p in (parts if parts is not None else re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+", t)) if _clean(p)]
    if not cores:
        return False
    joined = " ".join(cores)
    phrases = FORM_HEADER_PHRASES | {p.replace("-", " ") for p in FORM_HEADER_PHRASES}
    if joined in phrases or joined.replace("-", " ") in phrases:
        return True
    if any(c in FORM_HEADER_TOKENS for c in cores) and not any(
            c in given_names() and c not in AMBIGUOUS_GIVEN for c in cores):
        return True
    return False


def _span_is_legal_fp(t, s, e, parts=None):
    """Court / legal structure that looks typographically like a person name."""
    span = t[s:e].strip()
    if not span:
        return True
    if CITATION.search(span):
        return True
    window = t[max(0, s - 8):min(len(t), e + 8)]
    if CITATION.search(window) and re.search(r"(?i)\b(?:fed|r|crim|civ|app|evid|proc|bankr)\b", span):
        return True
    tokens = parts if parts is not None else re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+", span)
    cores = [_clean(p) for p in tokens if _clean(p)]
    if _joined_is_stop(tokens):
        return True
    if len(cores) >= 2 and cores[-1] in COURT_UNITS and (
            _is_state_like(cores[-2]) or cores[-2] in STOP or cores[-2] in AMBIGUOUS_GIVEN):
        return True
    # 'JURG. WV', 'Martinsburg, WV' — a trailing USPS state abbreviation is a place, not a surname
    if len(cores) >= 2 and cores[-1] in US_STATE_ABBR:
        return True
    if re.search(r"(?i)\bbar\s+no\.?\b", t[s:min(len(t), e + 12)]):
        return True
    # all-common-English handle ('Pale Horse'): apply to caption names, not only a/k/a
    if cores and cores[0] not in given_names():
        if all(c in COMMON_ENGLISH or c in STOP for c in cores):
            return True
    # 'Assign. Date', 'Filed Date' — a form / docket label, not a person
    if cores and cores[-1] in HEADING_LABELS:
        return True
    if _is_form_header(span, tokens):
        return True
    # OCR junk typography: several bare letters treated as initials ('ONG S G G NI')
    if sum(len(_clean(p)) == 1 and not str(p).endswith(".") for p in tokens) >= 3:
        return True
    if re.match(r"\s*:\s*\d", t[e:]):
        return True
    # typographic court venue: 'U.S. DISTRICTI', 'BAST RN DISTRICE OB MICHIG'
    if any(_is_court_unit_like(c) for c in cores) and not any(c in given_names() for c in cores):
        return True
    # 'ALEXANDRIA VIRE' (truncated VIRGINIA) / 'John Doe' placeholders
    if cores and cores[-1] == "DOE":
        return True
    if len(cores) >= 2 and _is_state_like(cores[-1]) and not any(
            c in given_names() and c not in AMBIGUOUS_GIVEN for c in cores[:-1]):
        return True
    # 'Las Vegas, Nevada 89101' — a city followed by a state and/or ZIP is a place
    m = re.match(r",\s*([A-Z][A-Za-z]+)(?:\s+\d{5})?", t[e:])
    if m and _is_state_like(_clean(m.group(1))):
        return True
    if re.match(r",?\s*\d{5}(?:-\d{4})?\b", t[e:]):
        return True
    return False


def _is_name_list_line(t):
    """Short typographic name / a.k.a. line, used to keep caption mode across a boundary."""
    t = t.strip().rstrip(",")
    if not t or len(t) > 90:
        return False
    if CAPTION_END.match(t):
        return False
    if AKA_MARK.search(t):
        return True
    words = [w for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+", t) if re.search(r"[A-Za-z]", w)]
    if not (2 <= len(words) <= 5 and all(w[:1].isupper() for w in words)):
        return False
    cores = [_clean(w) for w in words]
    if all(c in STOP or c in AMBIGUOUS_GIVEN or c in SUFFIX for c in cores):
        return False
    if all(len(c) <= 1 for c in cores):
        return False
    if _joined_is_stop(words):
        return False
    if _is_form_header(t, words):
        return False
    if any(_is_court_unit_like(c) or _is_state_like(c) or c in HEADING_LABELS for c in cores):
        return False
    if re.search(r":\s*\d", t):
        return False
    if re.search(r",\s*[A-Z][A-Za-z]+\s+\d{5}", t):
        return False
    if (not t.isupper()) and re.search(
            r"\b(?:the|of|to|for|in|on|at|by|with|from|that|this|was|were|his|her)\b", t):
        return False
    return True


def _name_like_run(t, start, G, allow_unknown_given):
    """Longest person-name run starting at character offset start in line t. Returns (s, e, why) or None."""
    toks = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\s,;:()“”\"]+", t[start:], flags=0)]
    toks = [(start + s, start + e, w) for s, e, w in toks]
    if not toks or toks[0][0] != start and t[start:toks[0][0]].strip():
        # allow only leading punctuation/space already skipped by caller
        pass
    if not toks:
        return None
    # skip a leading defendant-list marker so the span is the name, not 'D-I DAVID BRIAN STONE'
    while toks and DOCKET_TOKEN.match(toks[0][2].strip(".,")):
        toks = toks[1:]
    # 'Captain John Smith' / 'Chief Judge Beryl A. Howell' — the title is not a given name
    while toks and _clean(toks[0][2]) in ROLE_TITLES:
        toks = toks[1:]
    if not toks:
        return None
    s0, e0, w0 = toks[0]
    c0 = _clean(w0)
    if c0 in STOP or c0 in DETERMINERS or not re.fullmatch(CAP_WORD, w0.strip(".,")):
        return None
    caps = w0.isupper()
    j, parts, middle = 0, [w0], False
    k = 1
    while k < len(toks) and len(parts) < 6:
        w = toks[k][2]
        if t[toks[k - 1][1]:toks[k][0]].strip():
            break
        cw = _clean(w)
        # spaced hyphen / apostrophe between name tokens: "Moholy - Nagy", "N ' Djimbi"
        if re.fullmatch(r"[-'’]", w) and k + 1 < len(toks):
            nxt = toks[k + 1][2]
            if re.fullmatch(CAP_WORD_DOT, nxt) and _clean(nxt) not in STOP:
                parts.append(w); k += 1
                continue
            break
        # O' / N' / D' / L' particle split from the following token
        if APOS_PARTICLE.match(w) and k + 1 < len(toks):
            nxt = toks[k + 1][2]
            if re.fullmatch(CAP_WORD_DOT, nxt) and _clean(nxt) not in STOP:
                parts.append(w); k += 1
                continue
            break
        if re.fullmatch(r"[A-Z]\.?", w):
            middle = True
        elif w.lower() in PARTICLES and k + 1 < len(toks):
            pass
        elif re.fullmatch(CAP_WORD_DOT, w) and (w.rstrip(".").isupper() == caps or not caps) and cw not in STOP:
            pass
        else:
            break
        parts.append(w); k += 1
    while len(parts) > 1 and (re.fullmatch(r"[A-Z]\.?", parts[-1]) or parts[-1].lower() in PARTICLES or
                              re.fullmatch(r"[-'’]", parts[-1]) or APOS_PARTICLE.match(parts[-1]) or
                              _clean(parts[-1]) in SUFFIX):
        parts.pop()
    # 'Romona Grubt Miami Florida' — a trailing state, and the city before it, is a place
    if len(parts) >= 2 and _is_state_like(_clean(parts[-1])):
        parts = parts[:-1]
        leftover = [p for p in parts if not re.fullmatch(r"[A-Z]\.?", p) and p.lower() not in PARTICLES
                    and not re.fullmatch(r"[-'’]", p) and not APOS_PARTICLE.match(p)]
        if len(leftover) >= 2:
            parts = parts[:-1]
    full = [p for p in parts[1:] if not re.fullmatch(r"[A-Z]\.?", p) and p.lower() not in PARTICLES
            and not re.fullmatch(r"[-'’]", p) and not APOS_PARTICLE.match(p)]
    known = c0 in G
    if not full or len(full[-1].rstrip(".")) < 2:
        return None
    if _is_heading_tokens(parts):
        return None
    if not allow_unknown_given and NAME_CUE_BEFORE.search(t[:s0]):
        allow_unknown_given = True
    if not (known or allow_unknown_given):
        return None
    if c0 in AMBIGUOUS_GIVEN and not (len(parts) >= 3 or middle):
        return None
    e = toks[len(parts) - 1][1]
    while e > s0 and t[e - 1] in ".,'’":
        e -= 1
    if ORG_AFTER.match(t[e:]):
        return None
    if _span_is_legal_fp(t, s0, e, parts):
        return None
    why = "given-name" if known else "caption-name"
    return (s0, e, why)


def _aka_spans(t, G):
    out = []
    # quoted alias
    for m in re.finditer(r"(?:a\.?\s*/?\s*k\.?\s*/?\s*a\.?|also known as)\s*[“\"]([^”\"]{3,40})[”\"]", t, re.I):
        a = m.group(1).strip(" ,")
        out.extend(_alias_if_name(t, m.start(1) + m.group(1).index(a) if a in m.group(1) else m.start(1), a, G))
    # unquoted alias (stops at comma, semicolon, 'and', or end)
    for m in re.finditer(r"(?:a\.?\s*/?\s*k\.?\s*/?\s*a\.?|also known as)\s+(?!“|\"|”)([A-ZÀ-ÖØ-ÞĀ-Ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’.\-]*(?:\s+[A-ZÀ-ÖØ-ÞĀ-Ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’.\-]*){1,3})", t, re.I):
        a = m.group(1).strip(" ,")
        out.extend(_alias_if_name(t, m.start(1), a, G))
    return out


def _alias_if_name(t, s, a, G):
    ws = a.split()
    if not (2 <= len(ws) <= 4 and all(re.fullmatch(TOKEN, w) or w.lower() in PARTICLES or _clean(w) in SUFFIX for w in ws)):
        return []
    if any(_clean(w) in STOP for w in ws if w.lower() not in PARTICLES and _clean(w) not in SUFFIX):
        return []
    cores = [w for w in ws if w.lower() not in PARTICLES and _clean(w) not in SUFFIX and not re.fullmatch(r"[A-Z]\.", w)]
    if len(cores) < 2:
        return []
    # handles made of ordinary English words ('Red Fox') stay unlabelled; a known given name is always kept
    if _clean(ws[0]) not in G and not any(re.fullmatch(r"[A-Z]\.", w) for w in ws) and \
            all(_clean(w) in COMMON_ENGLISH or _clean(w) in STOP for w in cores):
        return []
    e = s + len(a)
    while e > s and t[s:e][-1:] in ".,'’":
        e -= 1
    return [(s, e, "aka-alias")]


def _caption_spans(t, G):
    """Name runs on a caption / defendant-list line, including names whose given token is not in the gazetteer."""
    out = []
    rest, offset = t, 0
    m = LINE_PREFIX.match(t.lstrip())
    if m:
        # prefix may follow leading space
        lead = len(t) - len(t.lstrip())
        offset = lead + m.end()
        rest = t[offset:]
    # split around a.k.a. markers; the left part is the primary name, rights are aliases (handled globally)
    parts = AKA_MARK.split(rest)
    primary = parts[0] if parts else rest
    # one or two name phrases separated by 'and' or comma
    for piece in re.split(r"\s+\band\b|,", primary, flags=re.I):
        piece = piece.strip()
        if not piece:
            continue
        # skip leftover role words
        if re.fullmatch(r"(?i)(?:plaintiffs?|defendants?|petitioners?|respondents?|et al\.?)", piece):
            continue
        s = t.find(piece, offset)
        if s < 0:
            continue
        got = _name_like_run(t, s, G, allow_unknown_given=True)
        if got:
            out.append(got)
    return out


def _name_spans(t, in_caption=False):
    """(start, end, why) person-name spans in one line of a legal document."""
    out = []
    G = given_names()
    nt, mp = _prepare_name_text(t)

    def mapped(s, e, why):
        a, b = _orig_span(mp, s, e)
        return (a, b, why)

    toks = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\s,;:()“”\"]+", nt)]
    i = 0
    while i < len(toks):
        s0 = toks[i][0]
        got = _name_like_run(nt, s0, G, allow_unknown_given=False)
        if got:
            out.append(mapped(*got))
            while i < len(toks) and toks[i][0] < got[1]:
                i += 1
            continue
        i += 1
    # initials + surname: 'H. W. Lowe', 'C. N. Hildreth Jr.', 'STEVEN W. MYHRE' after ungluing
    for m in re.finditer(r"(?<![\w.])((?:[A-Z]\.\s?){2,3}[A-ZÀ-ÖØ-ÞĀ-Ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]{2,})(?![\w])", nt):
        if _clean(m.group(1).split()[-1].split(".")[-1]) not in STOP and not ORG_AFTER.match(nt[m.end():]):
            out.append(mapped(m.start(1), m.end(1), "initials+surname"))
    # docket stamp on the original line: 'Judge: Roberts, Victoria A'
    for m in re.finditer(r"\b(?:Judge|MJ|Magistrate(?: Judge)?|Clerk|Assigned to)\s*:\s*([A-Z][A-Za-z'’-]+,\s*[A-Z][A-Za-z'’-]+(?:\s[A-Z]\b\.?)?)", t):
        out.append((m.start(1), m.end(1), "docket-stamp"))
    out.extend(_aka_spans(t, G))
    if in_caption:
        for sp in _caption_spans(nt, G):
            msp = mapped(*sp)
            if not any(s <= msp[0] and msp[1] <= e for s, e, _ in out):
                out.append(msp)
    return [sp for sp in out if not _span_is_legal_fp(t, sp[0], sp[1])]


def _caption_flags(lines, court_filing=True):
    """True on party-caption lines and typographic name-list lines (affidavits, post-heading names).

    Caption mode (unknown given names, typographic name lists) is limited to court filings.
    """
    in_cap = False
    flags = []
    for l in lines:
        t = l.text.strip()
        if court_filing and CAPTION_START.search(t):
            in_cap = True
        # a 'v.' block stays on until the end heading; typographic / affidavit lines are flagged
        # even after that heading or without a 'v.' cue (caption-boundary and standalone lists)
        flags.append(bool(court_filing and (in_cap or NAME_LIST_START.search(t) or _is_name_list_line(t))))
        if in_cap and CAPTION_END.match(t):
            in_cap = False
    return flags


def _is_stacked_name_line(l):
    """A short typographic line that may be one wrapped token of a longer caption name."""
    t = l.text.strip()
    if not t or len(t) > 40 or AKA_MARK.search(t) or re.search(r"\d{2,}", t):
        return False
    words = [w for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+", t) if re.search(r"[A-Za-z]", w)]
    cores = [_clean(w) for w in words]
    # one token only: two-word heading lines ('Computer Intrusions') must not join into a fake name
    if not (len(words) == 1 and words[0][:1].isupper()):
        return False
    if all(c in STOP or c in SUFFIX or c in ROLE_TITLES or c in COMMON_ENGLISH or c in HEADING_LABELS for c in cores):
        return False
    if _is_heading_tokens(words) or any(_is_court_unit_like(c) or _is_state_like(c) for c in cores):
        return False
    return True


def _wrapped_name_groups(lines):
    """Consecutive short capitalized lines that typography stacked from one name."""
    groups, cur = [], []
    for l in lines:
        if _is_stacked_name_line(l):
            cur.append(l)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = []
    if len(cur) >= 2:
        groups.append(cur)
    return groups


def legal_name_proposals(docs):
    out = []
    for d in docs:
        lines = [l for p in d.pages for l in p.lines if l.source != "json" and l.text.strip()]
        body = "\n".join(l.text for l in lines)
        if not lines or not is_legal(body):
            continue
        court = is_court_filing(body)
        flags = _caption_flags(lines, court_filing=court)
        found = []  # (text, why)
        for l, in_cap in zip(lines, flags):
            for s, e, why in _name_spans(l.text, in_caption=in_cap):
                found.append((l.text[s:e], why))
                out.append(Proposal(l.id, s, e, "PERSON", l.text[s:e], "legal-names", 0.8, None, {"why": why}))
        # typographic wrap: one long name stacked as one token per line (court captions only)
        covered = {(pr.line_id, pr.start, pr.end) for pr in out}
        for group in _wrapped_name_groups(lines) if court else []:
            parts, segs = [], []
            for i, l in enumerate(group):
                if i:
                    parts.append(" ")
                s = sum(len(x) for x in parts)
                piece = l.text.strip()
                parts.append(piece)
                lead = len(l.text) - len(l.text.lstrip())
                segs.append((s, s + len(piece), l, lead))
            joined = "".join(parts)
            for s, e, why in _name_spans(joined, in_caption=True):
                for vs, ve, l, lead in segs:
                    a, b = max(s, vs), min(e, ve)
                    if a >= b:
                        continue
                    ls, le = lead + (a - vs), lead + (b - vs)
                    if (l.id, ls, le) in covered:
                        continue
                    span = l.text[ls:le]
                    found.append((span, why))
                    out.append(Proposal(l.id, ls, le, "PERSON", span, "legal-names", 0.8, None,
                                        {"why": "wrapped-" + why}))
                    covered.add((l.id, ls, le))
        # bare surname of a person named in this document: in capitals, or after a role word
        surnames = set()
        for v, why in found:
            # a role-title alias ('Captain Hutaree') must not seed a document-wide surname
            if why == "aka-alias":
                first = _clean((v.split() or [""])[0])
                if first in ROLE_TITLES:
                    continue
            # a shop name ('MARLOWE CYCLES') is a kept fact: its last word is not a surname to look for
            if _BUSINESS.search(v):
                continue
            v2 = re.sub(r",.*$", "", v) if "," in v else v.split()[-1]  # 'Roberts, Victoria A' -> Roberts
            v2 = v2.strip(".,")
            if len(v2) >= 3 and _clean(v2) not in STOP and _clean(v2) not in given_names():
                surnames.add(v2)
        for l in lines:
            for sn in surnames:
                for m in re.finditer(r"(?<![\w'’-])(" + re.escape(sn) + r")(?![\w-])", l.text, re.I):
                    w = m.group(1)
                    # 'the HUTAREE' is the group, not the person whose alias ended in that word
                    if re.search(r"(?i)\bthe\s+$", l.text[:m.start(1)]):
                        continue
                    caps_in_prose = w.isupper() and not l.text.isupper()
                    if caps_in_prose or (w[:1].isupper() and ROLE_BEFORE.search(l.text[:m.start(1)])):
                        if _span_is_legal_fp(l.text, m.start(1), m.end(1)):
                            continue
                        out.append(Proposal(l.id, m.start(1), m.end(1), "PERSON" if w.isupper() else "SURNAME", w, "legal-names", 0.75, None,
                                            {"why": "surname-of-named-person"}))
    return out


def filter_legal_structure_proposals(docs, proposals):
    """Drop PERSON/SURNAME proposals on legal documents that are court or form structure, not people.

    The general rules detector can still propose citations like 'R. Crim'; this applies the same
    structure filter used by legal-names, on legal documents only.
    """
    legal_lines = {}
    for d in docs:
        lines = [l for p in d.pages for l in p.lines if getattr(l, "source", "ocr") != "json" and l.text.strip()]
        if not lines or not is_legal("\n".join(l.text for l in lines)):
            continue
        for l in lines:
            legal_lines[l.id] = l.text
    if not legal_lines:
        return proposals
    out = []
    for pr in proposals:
        if pr.kind in ("PERSON", "SURNAME") and pr.line_id in legal_lines:
            if _span_is_legal_fp(legal_lines[pr.line_id], pr.start, pr.end):
                continue
        out.append(pr)
    return out


# GLiNER is a second PERSON detector on legal documents only. It is not run on the
# default non-legal path. Structure false positives are dropped with the same rules
# as legal-names. The model is loaded once, offline, and only after a legal doc is seen.
_GLINER_MODEL = None


def _gliner_person_model():
    global _GLINER_MODEL
    if _GLINER_MODEL is False:
        return None
    if _GLINER_MODEL is None:
        try:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from gliner import GLiNER
            _GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        except Exception:
            _GLINER_MODEL = False
            return None
    return _GLINER_MODEL


def legal_gliner_proposals(docs):
    """PERSON proposals from GLiNER on legal documents only, filtered by court/legal structure rules."""
    out = []
    model = None
    for d in docs:
        lines = [l for p in d.pages for l in p.lines if l.source != "json" and l.text.strip()]
        if not lines or not is_legal("\n".join(l.text for l in lines)):
            continue
        if model is None:
            model = _gliner_person_model()
            if model is None:
                return []
        win, size = [], 0

        def run(window):
            text, spans = "", []
            for l in window:
                spans.append((len(text), l))
                text += l.text + "\n"
            try:
                ents = model.predict_entities(text, ["person"], threshold=0.4)
            except Exception:
                return
            for ent in ents:
                s, e = ent["start"], ent["end"]
                base, line = max((b, l) for b, l in spans if b <= s)
                vs, ve = s - base, e - base
                if vs < 0 or ve > len(line.text):
                    continue
                span = line.text[vs:ve]
                mtitle = TITLE_PREFIX.match(span)
                if mtitle:
                    vs += mtitle.end()
                    span = line.text[vs:ve]
                if len(span.strip()) < 3 or _span_is_legal_fp(line.text, vs, ve):
                    continue
                toks = [w for w in span.split() if _clean(w)]
                if toks and all(_clean(w) in STOP or _clean(w) in SUFFIX for w in toks):
                    continue
                # a lone common noun or uncapitalized token is not a person name
                if len(toks) == 1 and (_clean(toks[0]) in COMMON_ENGLISH or not re.match(r"[A-Z]", span.strip())):
                    continue
                out.append(Proposal(line.id, vs, ve, "PERSON", span, "legal-gliner",
                                    float(ent.get("score", 0.4)), None, {"why": "gliner-person"}))

        for l in lines:
            n = max(1, len(l.text.split()))
            if size + n > 180 and win:
                run(win)
                win, size = [], 0
            win.append(l)
            size += n
        if win:
            run(win)
    return out
