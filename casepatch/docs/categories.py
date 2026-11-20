"""Regulated-category detectors (HIPAA Safe Harbor identifiers + PHI health content, PCI DSS account data,
GLBA NPI, GDPR Art. 9/10 special categories) as a plug-in for the documents engine.

    from casepatch.docs.categories import CategoryDetector
    det = CategoryDetector(policy={"DIAGNOSIS": "replace"})   # optional per-kind overrides
    proposals = [p for c in chunks for p in det.detect(c)]    # same Proposal type as detect.py
    det.flags                                                 # kept-but-flagged items for the review screen

Per-kind policy (DEFAULT_POLICY, user-configurable):
- "replace": emit a Proposal -> replaced in place by a realistic, case-consistent, format/checksum-valid fake;
- "flag":    keep the original (decision-relevant by default) and record it in `self.flags` for review;
- "keep":    ignore silently.
Health content and GDPR special categories default to "flag": they are often the very facts an eval checks (a
diagnosis on a claim, a medication on a pharmacy dispute). When a user marks them non-decision-relevant they are
replaced by a DIFFERENT term of the SAME group (condition -> condition in the same body system, drug -> drug in the
same class), never blanked. Card verification codes, PINs and track data are sensitive authentication data under
PCI DSS: they always get fresh random digits (not a keyed mapping) and their source text is never put in flags.

No network, no models. Everything is regex + checksum + a small CC0 vocabulary (category_vocab.py).
"""
from __future__ import annotations

import re

from .category_fakes import (CategoryFaker, aba_ok, dea_ok, iban_ok, luhn_ok,
                             mbi_ok, MBI_RX, npi_ok, ssn_ok, vin_ok)
from .category_vocab import CONDITIONS, LAB_TESTS, MEDICATIONS, PROCEDURES, SPECIAL_VALUES
from .detect import Proposal

# kind -> (regulation tags, default policy). Existing engine kinds (PERSON, ORG, ACCOUNT, EMAIL...) are reused where the
# engine already has a consistent replacement for them.
CATEGORY_KINDS = {
    "SSN": ("HIPAA(G) GLBA", "replace"), "TIN": ("GLBA IRS", "replace"),
    "IBAN": ("GLBA PSD2", "replace"), "ROUTING": ("GLBA", "replace"),
    "CARD_EXPIRY": ("PCI-CHD", "replace"), "CVV": ("PCI-SAD", "replace"), "PIN": ("PCI-SAD", "replace"),
    "TRACK_DATA": ("PCI-SAD", "replace"),
    "MRN": ("HIPAA(H)", "replace"), "MEMBER_ID": ("HIPAA(I)", "replace"), "NPI": ("HIPAA-provider", "replace"),
    "DEA": ("HIPAA-provider", "replace"),
    "PASSPORT": ("HIPAA(K) GDPR", "replace"), "DRIVER_LICENSE": ("HIPAA(K)", "replace"),
    "VIN": ("HIPAA(L)", "replace"), "PLATE": ("HIPAA(L)", "replace"),
    "DEVICE_ID": ("HIPAA(M)", "replace"), "IMEI": ("HIPAA(M)", "replace"), "MAC": ("HIPAA(M)", "replace"),
    "IP": ("HIPAA(O) GDPR", "replace"), "URL": ("HIPAA(N)", "flag"),
    "GEO": ("HIPAA(B) GDPR", "flag"), "AGE_90": ("HIPAA(C)", "replace"),
    "DIAGNOSIS": ("PHI GDPR-9", "flag"), "DIAGNOSIS_CODE": ("PHI GDPR-9", "flag"), "MEDICATION": ("PHI GDPR-9", "flag"),
    "PROCEDURE": ("PHI GDPR-9", "flag"), "PROCEDURE_CODE": ("PHI GDPR-9", "flag"), "LAB_RESULT": ("PHI GDPR-9", "flag"),
    "SPECIAL": ("GDPR-9/10", "flag"), "INCOME": ("GLBA", "flag"), "CREDIT_SCORE": ("GLBA FCRA", "flag"),
    "MRZ": ("ICAO-9303", "keep"),  # the rules detector rewrites MRZ lines with valid check digits (docs/mrz.py)
    # existing engine kinds this module may also emit
    "PERSON": ("HIPAA(A)", "replace"), "ORG": ("HIPAA(A)-employer", "replace"), "ACCOUNT": ("HIPAA(J) GLBA", "replace"),
    "FACILITY": ("PHI", "flag"),
}
DEFAULT_POLICY = {k: v[1] for k, v in CATEGORY_KINDS.items()}

# Content categories a user may choose to KEEP (e.g. keep diagnoses for a medical-review eval, still replace names).
# Identifiers (names, member / provider IDs, SSN, card and account data) are not in this list: they are always replaced.
# 'keep' = kept unchanged and flagged on the review screen; 'replace' = a different term of the same group.
POLICY_GROUPS = [
    ("diagnoses", "Diagnoses and conditions (incl. ICD codes)", ("DIAGNOSIS", "DIAGNOSIS_CODE")),
    ("medications", "Medications", ("MEDICATION",)),
    ("procedures", "Procedures (incl. CPT / HCPCS codes)", ("PROCEDURE", "PROCEDURE_CODE")),
    ("lab_results", "Lab results (value only; the test name stays)", ("LAB_RESULT",)),
    ("facilities", "Treating facilities", ("FACILITY",)),
    ("special", "Other sensitive categories (religion, union membership, criminal record, ...)", ("SPECIAL",)),
    ("financial_profile", "Income and credit score", ("INCOME", "CREDIT_SCORE")),
]
GROUP_CHOICES = ("replace", "keep")


def policy_from_groups(choices=None, default="replace"):
    """{group: 'replace'|'keep'} (missing groups -> `default`) -> per-kind detector policy. Raises ValueError on unknown
    groups or values, so a UI can never switch off replacement of identifiers through this path."""
    choices = dict(choices or {})
    known = {g for g, _, _ in POLICY_GROUPS}
    bad = [k for k in choices if k not in known] + [v for v in choices.values() if v not in GROUP_CHOICES]
    if bad or default not in GROUP_CHOICES:
        raise ValueError(f"unknown category group or choice: {bad or default}")
    out = {}
    for g, _, kinds in POLICY_GROUPS:
        for k in kinds:
            out[k] = "flag" if choices.get(g, default) == "keep" else "replace"
    return out


def group_choices(choices=None, default="replace"):
    """Effective {group: choice} for display."""
    choices = dict(choices or {})
    return {g: choices.get(g, default) for g, _, _ in POLICY_GROUPS}
SAD_KINDS = {"CVV", "PIN", "TRACK_DATA"}
SUBSTITUTE_KINDS = {"DIAGNOSIS", "MEDICATION", "PROCEDURE", "DIAGNOSIS_CODE"}
# relative priority when spans tie in resolve.build_occurrences (lower wins); generic PHONE/ID must not win over a
# validated NPI / routing / MBI on the same digits
CATEGORY_PRIORITY = {"TRACK_DATA": 0, "IBAN": 0, "VIN": 1, "MBI": 1, "SSN": 2, "TIN": 2, "NPI": 2, "ROUTING": 2, "DEA": 2,
                     "MEMBER_ID": 2, "MRN": 2, "IMEI": 2, "PASSPORT": 2, "DRIVER_LICENSE": 2, "CVV": 2, "CARD_EXPIRY": 2,
                     "PIN": 2, "IP": 2, "MAC": 2, "DEVICE_ID": 2, "PLATE": 3, "AGE_90": 3, "GEO": 3, "URL": 4,
                     "DIAGNOSIS": 4, "DIAGNOSIS_CODE": 4, "MEDICATION": 4, "PROCEDURE": 4, "PROCEDURE_CODE": 4,
                     "LAB_RESULT": 4, "SPECIAL": 4}

V = r"[:#.\s]*"  # separator between a label and its value
NUM_SUFFIX = r"(?:\s*(?:No\.?|Number|Num\.?|#|ID))?"

# (label regex, kind, value regex). Value is searched right after the label on the same line, or as the whole next line.
LABELS = [
    (r"\b(?:SSN|SS#|S\.S\.N\.?|Social Security" + NUM_SUFFIX + r")", "SSN", r"\d{3}[- ]?\d{2}[- ]?\d{4}"),
    (r"\b(?:TIN|EIN|ITIN|Tax ?ID|Taxpayer ID|Federal Tax I\.?D\.?" + NUM_SUFFIX + r"|Employer Identification Number)",
     "TIN", r"\d{2}-\d{7}|\d{3}-\d{2}-\d{4}|\d{9}"),
    (r"\b(?:IBAN)", "IBAN", r"[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}"),
    (r"\b(?:Routing" + NUM_SUFFIX + r"|ABA" + NUM_SUFFIX + r"|RTN|Transit(?: No\.?| Number)?)", "ROUTING", r"\d{9}"),
    (r"\b(?:Account|Acct\.?)" + NUM_SUFFIX, "ACCOUNT", r"[0-9][0-9 -]{4,20}[0-9]"),
    (r"\b(?:MRN|Medical Record" + NUM_SUFFIX + r"|Chart" + NUM_SUFFIX + r"|Patient ID|Patient" + NUM_SUFFIX + r")",
     "MRN", r"[A-Z]{0,4}-?\d[A-Z0-9-]{3,19}"),
    (r"\b(?:Member ID|Member" + NUM_SUFFIX + r"|Subscriber ID|Subscriber" + NUM_SUFFIX + r"|Policy" + NUM_SUFFIX +
     r"|Group" + NUM_SUFFIX + r"|Insured'?s I\.?D\.?" + NUM_SUFFIX + r"|Medicare" + NUM_SUFFIX +
     r"|Medicare Beneficiary Identifier|Medicaid" + NUM_SUFFIX + r"|MBI|HICN|Beneficiary" + NUM_SUFFIX +
     r"|Health Plan ID|Plan ID|Member Number)", "MEMBER_ID", r"[A-Z0-9][A-Z0-9-]{4,19}"),
    (r"\bNPI\b", "NPI", r"\d{10}"),
    (r"\bDEA" + NUM_SUFFIX, "DEA", r"[A-Z]{2}\d{7}"),
    (r"\bPassport" + NUM_SUFFIX, "PASSPORT", r"[A-Z0-9]{6,9}"),
    (r"\b(?:Driver'?s? Licen[cs]e" + NUM_SUFFIX + r"|DL" + NUM_SUFFIX + r"|D\.L\.|Licen[cs]e" + NUM_SUFFIX + r")",
     "DRIVER_LICENSE", r"[A-Z0-9][A-Z0-9-]{4,17}"),
    (r"\b(?:License Plate|Licen[cs]e Plate" + NUM_SUFFIX + r"|Plate" + NUM_SUFFIX + r"|Tag" + NUM_SUFFIX + r"|Reg(?:istration)?\.? (?:No\.?|Number|Plate))",
     "PLATE", r"[A-Z0-9]{1,4}[ -]?[A-Z0-9]{1,5}"),
    (r"\bVIN\b", "VIN", r"[A-HJ-NPR-Z0-9]{17}"),
    (r"\b(?:Serial" + NUM_SUFFIX + r"|S/N|SN|Device ID|Device Serial|UDI|Implant ID)", "DEVICE_ID", r"[A-Z0-9][A-Z0-9-]{4,30}"),
    (r"\bIMEI\b", "IMEI", r"\d{15}|\d{2}-\d{6}-\d{6}-\d"),
    (r"\b(?:MAC(?: Address)?)", "MAC", r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}"),
    (r"\b(?:Exp(?:\.|iry|iration)?(?: Date)?|Valid (?:Thru|Through)|Good Thru|EXP)", "CARD_EXPIRY",
     r"(?:0?[1-9]|1[0-2])\s?[/-]\s?(?:\d{4}|\d{2})(?!\d)"),
    (r"\b(?:CVV2?|CVC2?|CID|CSC|CAV2|CVN2?|Security Code|Card Verification(?: Code| Value)?)", "CVV", r"\d{3,4}"),
    (r"\bPIN\b", "PIN", r"\d{4,6}"),
    (r"\b(?:Employer|Employer'?s Name|Employed (?:by|at)|Place of Employment|Company Name|Occupation and Employer)", "ORG",
     r"[A-Z][A-Za-z0-9&.,'’ -]{2,60}"),
    (r"\b(?:Annual Income|Household Income|Gross (?:Pay|Income)|Net Pay|Salary|Wages|Monthly Income|Income)", "INCOME",
     r"\$?\s?\d[\d,]*(?:\.\d{2})?"),
    (r"\b(?:Credit Score|FICO(?: Score)?)", "CREDIT_SCORE", r"[3-8]\d{2}"),
    (r"\b(?:Religion|Religious (?:Affiliation|Preference)|Faith|Denomination)", "SPECIAL:religion", "WORDS"),
    (r"\b(?:Race|Ethnicity|Ethnic (?:Origin|Group|Background)|Race/Ethnicity)", "SPECIAL:racial_ethnic_origin", "WORDS"),
    (r"\b(?:Political (?:Party|Affiliation|Opinions?|Views)|Party Affiliation|Voter Registration Party)", "SPECIAL:political_opinion", "WORDS"),
    (r"\b(?:Union (?:Membership|Member|Affiliation|Local)|Trade Union)", "SPECIAL:trade_union", "WORDS"),
    (r"\b(?:Sexual Orientation|Sex Life)", "SPECIAL:sex_life_orientation", "WORDS"),
    (r"\b(?:Criminal (?:Record|History|Convictions?)|Convictions?|Offen[cs]es?|Arrest Record)", "SPECIAL:criminal", "WORDS"),
    (r"\b(?:Genetic (?:Test(?: Result)?|Information|Testing)|Genotype)", "SPECIAL:genetic", "WORDS"),
    (r"\b(?:Biometric(?: Data| Identifier| Template)?|Fingerprint(?: ID)?|Voiceprint)", "SPECIAL:biometric", "WORDS"),
    (r"\b(?:Diagnosis|Diagnoses|Dx|Condition|Nature of Illness|Chief Complaint|Impression|Problem List)", "DIAGNOSIS", "WORDS"),
    (r"\b(?:Medications?|Current Meds|Rx|Prescription|Prescribed)", "MEDICATION", "WORDS"),
    (r"\b(?:Attending|Physician|Provider|Prescriber|Referring (?:Physician|Provider)|Treating (?:Physician|Doctor)|Doctor)", "PERSON", "NAME"),
    (r"\b(?:Facility|Hospital|Clinic|Service Facility)", "FACILITY", r"[A-Z][A-Za-z0-9&.,'’ -]{2,60}"),
]
LABELS = [(re.compile(l, re.I), k, v) for l, k, v in LABELS]

# unlabeled patterns: strong formats / checksums
SSN_DASHED = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
IBAN_RX = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
VIN_RX = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![\w:])|(?<![\w:])(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4}:?){1,6}(?![\w:])")
MAC_RX = re.compile(r"(?<![\w:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:-])")
URL_RX = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"')\]]+[^\s<>\"').,;\]]")
TRACK1 = re.compile(r"%B\d{12,19}\^[^^]{2,26}\^\d{4}\d{3}[^?]*\?")
TRACK2 = re.compile(r";\d{12,19}=\d{4}\d{3}\d*\?")
GEO_RX = re.compile(r"(?<![\d.])[-+]?(?:[1-8]?\d)\.\d{4,}\s*,\s*[-+]?(?:1[0-7]\d|[1-9]?\d)\.\d{4,}(?![\d.])")
AGE_RX = re.compile(r"(?i)\b(?:age[d:]?\s*|aged\s+)(9\d|1[01]\d)\b|\b(9\d|1[01]\d)[- ](?:year|yr)s?[- ]old\b")
MRZ_RX = re.compile(r"^[A-Z0-9<]{30,44}$")
ICD10_RX = re.compile(r"(?<![A-Za-z0-9.])[A-TV-Z]\d[0-9A-Z](?:\.[0-9A-Z]{1,4})?(?![A-Za-z0-9])")
ICD9_RX = re.compile(r"(?<![\d.$])(?:V\d{2}|\d{3})\.\d{1,2}(?![\d.])")
CPT_RX = re.compile(r"(?<![\d.$A-Za-z])(?:\d{5}|\d{4}[FTU]|[A-V]\d{4})(?![\dA-Za-z])(?!\.\d)")
CRIMINAL_RX = re.compile(r"(?i)\b(?:convicted of|conviction for|pleaded guilty to|pled guilty to|arrested for|charged with|"
                         r"sentenced to|on probation for|on parole for)\s+[^.;\n]{3,60}")
GENETIC_RX = re.compile(r"(?i)\bBRCA[12]\b(?:[ -](?:positive|negative|mutation|variant|carrier))?|\b(?:genetic|DNA) test(?:ing)?\s+"
                        r"(?:result|showed|revealed|confirmed|was positive|was negative)[^.;\n]{0,50}|\bcarrier of (?:the )?[A-Z][\w-]+ "
                        r"(?:gene|mutation|variant)\b")
ORIENTATION_RX = re.compile(r"(?i)\b(?:identifies as|came out as|is openly)\s+(?:gay|lesbian|bisexual|queer|asexual|pansexual|transgender)\b|"
                            r"\bsame-sex (?:partner|spouse|relationship)\b")
UNION_RX = re.compile(r"\b(?:member of|belongs to|joined)\s+(?:the\s+)?(?:[A-Z][\w&.'-]*\s+){0,6}(?:Union|Teamsters|Local \d+|AFL-CIO|SEIU|UAW)\b")
HEALTH_CUE = re.compile(r"(?i)(?:\b(?:diagnosed with|history of|hx of|suffers? from|treated for|positive for|presents with|"
                        r"admitted for|takes|taking|prescribed|on)\s+$|\b(?:diagnos[ie]s|dx|medications?|meds|rx|condition|"
                        r"procedure|impression|assessment)\b\s*[:#-]\s*(?:[^:]{0,80}[,;/]\s*)?$)")
PROVIDER_RX = re.compile(r"\b((?:Dr\.?\s+)?[A-Z][a-zA-Z'’-]+(?:\s+(?:[A-Z]\.|[A-Z][a-zA-Z'’-]+)){1,2}),?\s+"
                         r"(?:M\.D\.|MD|D\.O\.|DO|NP|PA-C|RN|DDS|DMD|DPM|OD|O\.D\.|PharmD|LCSW|PsyD|Ph\.D\.)(?![A-Za-z])")
DR_RX = re.compile(r"\bDr\.?\s+([A-Z][a-zA-Z'’-]+(?:\s+(?:[A-Z]\.|[A-Z][a-zA-Z'’-]+)){1,2})")
FACILITY_RX = re.compile(r"\b((?:[A-Z][\w'’&.-]*\s+){0,5}(?:Hospital|Medical Center|Health Center|Health System|Clinic|"
                         r"(?:Eye|Cancer|Surgical|Surgery|Dialysis|Imaging|Wellness|Treatment|Rehabilitation|Heart|Children'?s|"
                         r"Behavioral Health|Fertility|Women'?s|Kidney|Urgent Care)\s+(?:Center|Centre|Institute|Associates|Group)|"
                         r"Pharmacy|Laboratory|Laboratories|Hospice|Nursing Home|Urgent Care)(?:\s+(?:PA|PC|LLC|PLLC|Inc\.?|LLP))?)\b")
DESC_CODE_RX = re.compile(r"^\s*([A-Z][a-z][A-Za-z ,'/-]{6,80}?)\s+\((?:\d{5}|\d{4}[FTU]|[A-V]\d{4})\)")
DOSE_AFTER = re.compile(r"(?i)^\s*\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|units?|IU|%)\b")


def _vocab_index():
    out = []
    for kind, pool in (("DIAGNOSIS", CONDITIONS), ("MEDICATION", MEDICATIONS), ("PROCEDURE", PROCEDURES)):
        for e in pool:
            for form, words in (("term", [e["term"]]), ("syn", e["syn"]), ("abbr", e["abbr"])):
                for w in words:
                    out.append((w, kind, e, form))
    out.sort(key=lambda x: -len(x[0]))
    return out


VOCAB = _vocab_index()
VOCAB_RX = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(w) for w, *_ in VOCAB) + r")(?![\w-])", re.I)
VOCAB_KEY = {}
for _w, _k, _e, _f in VOCAB:
    VOCAB_KEY.setdefault(_w.lower(), (_k, _e, _f))
LAB_RX = re.compile(r"(?<![\w-])(" + "|".join(re.escape(t) for t in sorted(LAB_TESTS, key=len, reverse=True)) +
                    r")\b[\s:=]*(?:was|of|is|at)?\s*(\d{1,4}(?:\.\d+)?(?:/\d{2,3})?\s*(?:%|mg/dL|mmol/L|mL/min(?:/1\.73\s?m2)?|g/dL|"
                    r"ng/mL|mIU/L|U/L|cells/mm3|copies/mL|mmHg|K/uL|kg/m2)?)", re.I)
SPECIAL_VALUE_RX = {k: re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(v) for v in sorted(vs, key=len, reverse=True)) + r")(?![\w-])", re.I)
                    for k, vs in SPECIAL_VALUES.items()}


def vocab_lookup(kind, text):
    """(entry, surface form, pool) for a detected health term, used by the substitution generator."""
    t = text.strip().lower()
    if kind == "DIAGNOSIS_CODE":
        for e in CONDITIONS:
            if t.upper() == (e["icd10"] or "").upper():
                return e, "icd10", CONDITIONS
            if t == (e["icd9"] or ""):
                return e, "icd9", CONDITIONS
        return None
    hit = VOCAB_KEY.get(t)
    if not hit:
        return None
    k, e, f = hit
    pool = {"DIAGNOSIS": CONDITIONS, "MEDICATION": MEDICATIONS, "PROCEDURE": PROCEDURES}[k]
    return e, f, pool


def _page_context(chunk):
    text = " ".join(u.text for u in chunk.units)
    return {
        "cpt": bool(re.search(r"\b(?:CPT|HCPCS|PROCEDURES?,? SERVICES|Billing Code|Procedure Code)\b", text, re.I)),
        "icd": bool(re.search(r"\b(?:ICD|DIAGNOSIS|Diagnoses|Dx|NATURE OF ILLNESS)\b", text, re.I)),
        "health": bool(re.search(r"\b(?:patient|diagnos|medical|clinic|physician|prescri|Medicare|Medicaid|health plan|claim)\w*", text, re.I)),
    }


def _label_value(kind, vrx, rest, next_line=False):
    """Find the value right after a label. Returns (start, end) relative to `rest` or None.
    Free-text and name values ('Diagnosis', 'Physician', 'Facility') need an explicit ':' / '#' / '-' after the label
    on the same line ('for diagnosis and treatment' and 'Physician Quality Reporting' are prose, not fields)."""
    if vrx in ("WORDS", "NAME") or kind == "FACILITY":
        if not next_line and not re.match(r"\s*(?:\([^)]{0,30}\)\s*)?[:#-]", rest):
            return None
    m = re.match(V + r"(?:-\s*)?(?:\([^)]{0,30}\)\s*)?", rest)  # skip separators and '(For Program in Item 1)'-style captions
    i = m.end() if m else 0
    body = rest[i:]
    if vrx == "WORDS":
        mm = re.match(r"[A-Za-z][^\n;|]{1,60}?(?=\s{2,}|[;|]|$)", body)
        if not mm:
            return None
        val = mm.group().rstrip(" .,")
        return (i, i + len(val)) if len(val) >= 2 else None
    if vrx == "NAME":
        mm = re.match(r"(?:Dr\.?\s+)?([A-Z][a-zA-Z'’-]+(?:\s+(?:[A-Z]\.|[A-Z][a-zA-Z'’-]+)){1,2})", body)
        return (i + mm.start(1), i + mm.end(1)) if mm else None
    mm = re.match(r"(?:" + vrx + r")(?![A-Za-z0-9])", body, re.I if kind not in ("PASSPORT", "DRIVER_LICENSE", "MEMBER_ID", "MRN") else 0)
    if not mm and kind in ("PASSPORT", "DRIVER_LICENSE", "MEMBER_ID", "MRN", "DEVICE_ID", "PLATE"):
        mm = re.match(r"(?:" + vrx + r")(?![A-Za-z0-9])", body, re.I)
    if not mm:
        return None
    val = mm.group().rstrip(" -")
    if kind in ("MEMBER_ID", "MRN", "DEVICE_ID", "DRIVER_LICENSE", "PASSPORT", "PLATE") and not re.search(r"\d", val):
        return None  # an id has digits; avoids 'Policy Holder', 'Group Health'
    return i, i + len(val)


def _is_label_only(line):
    t = line.strip()
    if not t or len(t) > 60:
        return None
    for rx, kind, vrx in LABELS:
        m = rx.search(t)
        if m and m.start() <= 6:
            tail = t[m.end():]
            if re.fullmatch(r"[\s:#.]*(?:/[^/]{0,40})?(?:\([^)]{0,40}\))?[\s:#.]*", tail):
                return kind, vrx
    return None


class CategoryDetector:
    name = "categories"
    learned = False

    def __init__(self, policy=None):
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.flags = []

    # -------------------------------------------------------------- core
    def prime(self, chunks):
        """Page-level context (a 'CPT/HCPCS' header in one chunk applies to the codes in the next chunk of that page)."""
        self._ctx = {}
        for c in chunks:
            k = (c.doc, c.page)
            cur = _page_context(c)
            prev = self._ctx.get(k, {})
            self._ctx[k] = {key: cur[key] or prev.get(key, False) for key in cur}
        return self

    def candidates(self, chunk):
        """Every category hit (kind, unit index, start, end, confidence, meta) BEFORE policy is applied."""
        ctx = getattr(self, "_ctx", {}).get((chunk.doc, chunk.page)) or _page_context(chunk)
        out = []
        for i, u in enumerate(chunk.units):
            t, off = u.text, u.offset
            def add(s, e, kind, conf, **meta):
                s2, e2 = s - off, e - off
                if s2 < 0 or e2 > len(u.value) or e2 <= s2:
                    return
                out.append((i, s2, e2, kind, conf, meta))
            # -- labelled values on the same line
            for rx, kind, vrx in LABELS:
                for m in rx.finditer(t):
                    if m.start() > 0 and t[m.start() - 1].isalnum():
                        continue
                    span = _label_value(kind, vrx, t[m.end():])
                    if span:
                        s, e = m.end() + span[0], m.end() + span[1]
                        self._typed(add, kind, t[s:e], s, e, 0.85, label=True)
            # -- value on the line after a label-only line ('Medicare Number' / '1A23BC4DE56')
            if i:
                lab = _is_label_only(chunk.units[i - 1].text)
                if lab and t.strip():
                    kind, vrx = lab
                    s0 = len(t) - len(t.lstrip())
                    span = _label_value(kind, vrx, t[s0:], next_line=True)
                    if span and (span[1] - span[0]) >= len(t.strip()) * 0.6:
                        self._typed(add, kind, t[s0 + span[0]:s0 + span[1]], s0 + span[0], s0 + span[1], 0.8, label="previous_line")
            # -- json keys as labels ('ssn: 123-45-6789')
            if u.source == "json" and u.path:
                key = next((k for k in reversed(u.path) if isinstance(k, str)), "")
                lab = _is_label_only(re.sub(r"[_-]", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", key)))
                if lab:
                    kind, vrx = lab
                    span = _label_value(kind, vrx, u.value, next_line=True)
                    if span:
                        self._typed(add, kind, u.value[span[0]:span[1]], off + span[0], off + span[1], 0.9, label="json_key")
            # -- strong unlabeled formats
            for m in SSN_DASHED.finditer(t):
                if ssn_ok(m.group()):
                    add(m.start(), m.end(), "SSN", 0.7, unlabeled=True)
            for m in IBAN_RX.finditer(t):
                if iban_ok(m.group()):
                    add(m.start(), m.end(), "IBAN", 0.95, checksum="mod97")
            for m in VIN_RX.finditer(t):
                if vin_ok(m.group()) and re.search(r"\d", m.group()) and re.search(r"[A-Z]", m.group()):
                    add(m.start(), m.end(), "VIN", 0.9, checksum="vin")
            for m in MBI_RX.finditer(t.upper()):
                add(m.start(), m.end(), "MEMBER_ID", 0.85, format="MBI")
            for m in IPV4.finditer(t):
                pre = t[max(0, m.start() - 12):m.start()].lower()
                if not re.search(r"v(?:ersion)?\s*$|\bver\.?\s*$", pre):
                    add(m.start(), m.end(), "IP", 0.8)
            for m in IPV6.finditer(t):
                if m.group().count(":") >= 3:
                    add(m.start(), m.end(), "IP", 0.8)
            for m in MAC_RX.finditer(t):
                add(m.start(), m.end(), "MAC", 0.85)
            for m in URL_RX.finditer(t):
                host_path = re.sub(r"(?i)^(?:https?://)?(?:www\.)?", "", m.group())
                host, _, path = host_path.partition("/")
                if re.search(r"(?i)\.(?:gov|mil|edu|int)$|^(?:nucc|ama-assn|who|cdc)\.org$", host) and not re.search(r"\d{3,}|[?=]", path):
                    continue  # an institution's public page, not an identifier of a person
                add(m.start(), m.end(), "URL", 0.9)
            for rx in (TRACK1, TRACK2):
                for m in rx.finditer(t):
                    add(m.start(), m.end(), "TRACK_DATA", 0.95)
            for m in GEO_RX.finditer(t):
                add(m.start(), m.end(), "GEO", 0.8)
            for m in AGE_RX.finditer(t):
                g = 1 if m.group(1) else 2
                add(m.start(g), m.end(g), "AGE_90", 0.85)
            if MRZ_RX.match(t.strip()) and t.count("<") >= 3:
                s0 = len(t) - len(t.lstrip())
                add(s0, s0 + len(t.strip()), "MRZ", 0.9)
            # -- NPI / routing / DEA by checksum when the page talks about them
            for m in re.finditer(r"(?<!\d)\d{10}(?!\d)", t):
                if npi_ok(m.group()) and re.search(r"\bNPI\b", " ".join(x.text for x in chunk.units), re.I):
                    add(m.start(), m.end(), "NPI", 0.75, checksum="npi")
            for m in re.finditer(r"(?<!\d)\d{9}(?!\d)", t):
                if aba_ok(m.group()) and re.search(r"\b(?:routing|ABA|RTN)\b", " ".join(x.text for x in chunk.units), re.I):
                    add(m.start(), m.end(), "ROUTING", 0.7, checksum="aba")
            for m in re.finditer(r"\b[A-Z]{2}\d{7}\b", t):
                if dea_ok(m.group()) and re.search(r"\bDEA\b", " ".join(x.text for x in chunk.units)):
                    add(m.start(), m.end(), "DEA", 0.75, checksum="dea")
            # -- health content
            for m in VOCAB_RX.finditer(t):
                w = m.group()
                kind, e, form = VOCAB_KEY[w.lower()]
                if form == "abbr" and not (w.isupper() or w in e["abbr"]):
                    continue  # 'ms' / 'ed' / 'ra' in running text are not abbreviations
                if form == "abbr" and len(w) <= 2 and not ctx["health"]:
                    continue
                if kind == "MEDICATION" and w.lower() in ("lithium", "insulin", "testosterone", "progesterone") and not \
                        (DOSE_AFTER.match(t[m.end():]) or HEALTH_CUE.search(t[:m.start()])):
                    continue
                cue = bool(HEALTH_CUE.search(t[:m.start()]) or (kind == "MEDICATION" and DOSE_AFTER.match(t[m.end():])))
                add(m.start(), m.end(), kind, 0.8 if cue else 0.6, group=e["group"], form=form)
            if ctx["icd"]:
                for m in ICD10_RX.finditer(t):
                    if re.search(r"\.", m.group()) or re.fullmatch(r"[A-TV-Z]\d{2}", m.group()) and \
                            re.search(r"\b(?:ICD|Dx|diagnos)", t, re.I):
                        add(m.start(), m.end(), "DIAGNOSIS_CODE", 0.7, system="ICD-10")
                for m in ICD9_RX.finditer(t):
                    add(m.start(), m.end(), "DIAGNOSIS_CODE", 0.6, system="ICD-9")
            if ctx["cpt"] or re.search(r"\b(?:CPT|HCPCS)\b", t):
                for m in CPT_RX.finditer(t):
                    if re.fullmatch(r"\d{5}", m.group()) and re.search(r"\b(?:ZIP|[A-Z]{2})\s*$", t[:m.start()]):
                        continue  # 'NC 28712' is a ZIP code
                    add(m.start(), m.end(), "PROCEDURE_CODE", 0.6)
            for m in re.finditer(r"\((\d{5}|\d{4}[FTU]|[A-V]\d{4})\)", t):  # 'Destruction of ... (17000)'
                if re.search(r"[a-z]{4,}", t[:m.start()]):
                    add(m.start(1), m.end(1), "PROCEDURE_CODE", 0.55)
            for m in LAB_RX.finditer(t):
                if re.search(r"\d", m.group(2)):
                    add(m.start(), m.end(), "LAB_RESULT", 0.7)
            # -- provider names by credential ('Craig I. Secosan, M.D.') and treating facilities by their generic noun
            for rx in (PROVIDER_RX, DR_RX):
                for m in rx.finditer(t):
                    s_, e_ = m.span(1)
                    nm = re.sub(r"^Dr\.?\s+", "", t[s_:e_]); s_ = e_ - len(nm)
                    add(s_, e_, "PERSON", 0.8, role="provider")
            for m in FACILITY_RX.finditer(t):
                if len(m.group(1).split()) >= 2:
                    add(m.start(1), m.end(1), "FACILITY", 0.65)
            # -- a service description followed by its billing code on the same line ('Destruction of ... (17000)')
            m = DESC_CODE_RX.match(t)
            if m and not re.search(r"(?i)\b(?:total|subtotal|claim|page)\b", m.group(1)):
                add(m.start(1), m.end(1), "PROCEDURE", 0.6, structure="description_before_code")
            # -- GDPR special-category phrases without labels
            for rx, sub in ((CRIMINAL_RX, "criminal"), (GENETIC_RX, "genetic"), (ORIENTATION_RX, "sex_life_orientation"),
                            (UNION_RX, "trade_union")):
                for m in rx.finditer(t):
                    add(m.start(), m.end(), "SPECIAL", 0.7, subtype=sub)
        return out

    def _typed(self, add, kind, val, s, e, conf, label):
        """Apply the per-kind validation for a labelled value and pick the most specific kind."""
        sub = None
        if kind.startswith("SPECIAL:"):
            kind, sub = "SPECIAL", kind.split(":", 1)[1]
        d = re.sub(r"\D", "", val)
        meta = {"label": label}
        if sub:
            meta["subtype"] = sub
        if kind == "SSN" and len(d) != 9:
            return
        if kind == "NPI":
            meta["checksum_valid"] = npi_ok(d)
            conf = conf if npi_ok(d) else conf - 0.15  # labelled but fails Luhn: still an identifier slot
        if kind == "ROUTING" and not aba_ok(d):
            conf -= 0.2
        if kind == "MEMBER_ID" and mbi_ok(val):
            meta["format"] = "MBI"
        if kind == "VIN" and not vin_ok(val):
            conf -= 0.2
        if kind == "IMEI" and not luhn_ok(d):
            conf -= 0.2
        if kind == "MRN" and not re.search(r"\d{3}", val):
            return
        if kind == "SPECIAL" and sub in SPECIAL_VALUE_RX:
            # a labelled special-category value: if it names a known value, take just that, else the whole value
            mm = SPECIAL_VALUE_RX[sub].search(val)
            if mm:
                s, e, conf = s + mm.start(), s + mm.end(), conf + 0.05
        if kind in ("DIAGNOSIS", "MEDICATION"):
            if val.strip().lower() in ("yes", "no", "none", "n/a", "unknown"):
                return
            if VOCAB_RX.search(val):
                return  # known terms inside the value are proposed one by one by the vocabulary pass
            cut = re.split(r"[,;(]", val)[0].rstrip()
            if len(cut) < 3 or re.fullmatch(r"[\d\s./-]+", cut):
                return
            e, conf = s + len(cut), conf - 0.2  # unknown free text under a health label: lower confidence
            meta["free_text"] = True
        add(s, e, kind, conf, **meta)

    def detect(self, chunk):
        cands = self.candidates(chunk)
        # longest span wins on overlap within a line (a labelled MBI beats a bare 'ID'-shaped substring)
        cands.sort(key=lambda c: (c[0], -(c[2] - c[1]), -c[4]))
        taken, props = {}, []
        for i, s, e, kind, conf, meta in cands:
            spans = taken.setdefault(i, [])
            if any(s < te and e > ts for ts, te in spans):
                continue
            spans.append((s, e))
            u = chunk.units[i]
            pol = self.policy.get(kind, "flag")
            if kind == "SPECIAL":
                pol = self.policy.get("SPECIAL:" + meta.get("subtype", ""), pol)
            if pol == "replace" and not meta.get("free_text") and kind in ("DIAGNOSIS", "MEDICATION", "PROCEDURE") and \
                    vocab_lookup(kind, u.value[s:e]) is None:
                meta = dict(meta, free_text=True)  # (e.g. a service description before its code: no vocabulary entry)
            if pol == "replace" and meta.get("free_text"):
                # free text under a health / special-category label that is not in the vocabulary: no realistic
                # same-group substitute exists, so it stays unchanged and flagged for the reviewer
                pol = "flag"
                meta = dict(meta, replace_requested=True)
            if pol == "keep":
                continue
            if pol == "flag":
                self.flags.append({"doc": chunk.doc, "page": chunk.page, "line_id": u.line_id, "start": s, "end": e, "kind": kind,
                                   "confidence": round(conf, 2), "meta": meta, "regulation": CATEGORY_KINDS.get(kind, ("", ""))[0],
                                   "text": None if kind in SAD_KINDS else u.value[s:e],
                                   "reason": "NO_REALISTIC_SUBSTITUTE_KEPT_FOR_REVIEW" if meta.get("replace_requested")
                                   else "KEPT_DECISION_RELEVANT_BY_DEFAULT"})
                continue
            m = dict(meta, regulation=CATEGORY_KINDS.get(kind, ("", ""))[0])
            if kind in SAD_KINDS:
                m["sad"] = True  # PCI sensitive authentication data: original must not be written to review files
            props.append(Proposal(u.line_id, s, e, kind, u.value[s:e], self.name, round(conf, 2), None, m))
        return props


# ------------------------------------------------------------------ replacement routing

def category_replacement(faker: CategoryFaker, kind: str, text: str):
    """Replacement for a category kind, or None when the kind belongs to the existing engine (PERSON, ORG, ...)."""
    f = {
        "SSN": faker.ssn, "TIN": faker.tin, "IBAN": faker.iban, "ROUTING": faker.routing, "CARD_EXPIRY": faker.card_expiry,
        "NPI": faker.npi, "DEA": faker.dea, "VIN": faker.vin, "IMEI": faker.imei, "IP": faker.ip, "MAC": faker.mac,
        "URL": faker.url, "GEO": faker.geo, "AGE_90": faker.age_90, "INCOME": faker.income,
    }.get(kind)
    if f:
        return f(text)
    if kind == "TRACK_DATA":
        return faker.track(text)
    if kind in SAD_KINDS:
        return faker.sad_digits(text, kind)
    if kind == "MEMBER_ID":
        return faker.mbi(text) if mbi_ok(text) else faker.shape(text, kind)
    if kind in ("MRN", "PASSPORT", "DRIVER_LICENSE", "PLATE", "DEVICE_ID", "CREDIT_SCORE", "MRZ", "LAB_RESULT", "PROCEDURE_CODE"):
        if kind == "LAB_RESULT":  # keep the test name, change only the value
            m = LAB_RX.match(text)
            if not m:
                return faker.shape(text, kind)
            return text[:m.start(2)] + faker.shape(m.group(2), kind) + text[m.end(2):]
        if kind == "PROCEDURE_CODE":
            return faker.shape(text, kind, keep_prefix=1)
        return faker.shape(text, kind)
    if kind in SUBSTITUTE_KINDS:
        return faker.substitute(kind, text, vocab_lookup)
    if kind == "SPECIAL":
        for sub, rx in SPECIAL_VALUE_RX.items():
            if rx.fullmatch(text.strip()):
                pool = [v for v in SPECIAL_VALUES[sub] if v.lower() != text.strip().lower()]
                return pool[faker.rng("special", sub, text.lower()).randrange(len(pool))]
        return None  # free-text special category: no safe realistic rewrite; stays flagged
    return None


def attach(cmap):
    """Route category kinds through format/checksum-preserving fakes on this case's CaseMap (same key -> consistent)."""
    faker = CategoryFaker(cmap)
    base = cmap.replace

    def replace(kind, text):
        if kind == "FACILITY":
            return cmap.org(text)
        if kind in CATEGORY_KINDS and kind not in ("PERSON", "ORG", "ACCOUNT", "MRZ"):
            k = ("cat", kind, text)
            if kind in SAD_KINDS:
                return category_replacement(faker, kind, text)  # never cached
            if k not in cmap._cache:
                out = category_replacement(faker, kind, text)
                cmap._cache[k] = out if out is not None else base(kind, text)
            return cmap._cache[k]
        return base(kind, text)

    cmap.replace = replace
    cmap.category_faker = faker
    return cmap


def install(resolve_module):
    """Make resolve.build_occurrences category-aware without editing resolve.py (idempotent):
    - category kinds win ties against generic PHONE/ID on the same digits;
    - a card expiry ('03/27') and a dotted diagnosis code ('250.00') are not mistaken for a protected date/amount
      when THIS detector proposed them with a label/code context (other detectors are unaffected)."""
    R = resolve_module
    R.PRIORITY.update({k: v for k, v in CATEGORY_PRIORITY.items() if k not in R.PRIORITY})
    if getattr(R.protected_fact, "_categories", False):
        return
    orig = R.protected_fact

    def protected_fact(kind, text):
        if kind in ("CARD_EXPIRY", "DIAGNOSIS_CODE", "PROCEDURE_CODE", "AGE_90", "GEO"):
            return False
        return orig(kind, text)

    protected_fact._categories = True
    R.protected_fact = protected_fact


_LAST = {"detector": None}


def hook(chunks, resolve_module=None, policy=None):
    """One-call integration for pipeline.run: returns proposals; flags stay on `last_detector().flags`."""
    if resolve_module is not None:
        install(resolve_module)
    det = CategoryDetector(policy).prime(chunks)
    _LAST["detector"] = det
    return [p for c in chunks for p in det.detect(c)]


def last_detector():
    return _LAST["detector"]
