"""Pluggable detectors producing GROUNDED proposals.

Every detector receives page chunks (numbered lines with provenance) and returns proposals that must cite
a line and an exact substring of it. `ground()` rejects anything that does not exist in the source
(ungrounded LLM output is reported, never applied)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ..core import PATTERNS
from ..review import STREET_ADDRESS, field_kind

KINDS = ("PERSON", "ORG", "ADDRESS", "ZIP", "PHONE", "EMAIL", "CARD", "CARD_LAST4", "ACCOUNT", "ID", "DOB", "MRZ")


@dataclass
class Unit:
    """One line of detector input: text the detector sees plus where it came from."""
    line_id: str
    text: str            # what detectors see (json: 'key: value')
    offset: int          # where the source value starts inside `text`
    value: str           # source line text (spans are grounded against this)
    source: str
    path: tuple = ()


@dataclass
class Chunk:
    doc: str
    page: int
    units: list

    def prompt_text(self):
        return "\n".join(f"L{i}: {u.text}" for i, u in enumerate(self.units))


@dataclass
class Proposal:
    line_id: str
    start: int
    end: int
    kind: str
    text: str
    detector: str
    confidence: float = 1.0
    person_hint: str | None = None   # detector's own cluster label (LLM)
    meta: dict = field(default_factory=dict)


def chunks_for(docs, max_units=60):
    out = []
    for d in docs:
        for p in d.pages:
            units = []
            for l in p.lines:
                if l.source == "json":
                    key = next((k for k in reversed(l.path) if isinstance(k, str)), "")
                    prefix = f"{key}: " if key else ""
                    units.append(Unit(l.id, prefix + l.text, len(prefix), l.text, l.source, l.path))
                else:
                    units.append(Unit(l.id, l.text, 0, l.text, l.source))
            for i in range(0, len(units), max_units):
                out.append(Chunk(d.id, p.index, units[i:i + max_units]))
    return out


def ground(chunk, line_index, text, kind, detector, confidence=1.0, hint=None):
    """Locate `text` in the cited line (exact, then case/whitespace-insensitive). Returns proposals or []."""
    if kind not in KINDS or not isinstance(text, str) or len(text.strip()) < 2:
        return []
    cands = range(len(chunk.units)) if line_index is None else [line_index] if 0 <= line_index < len(chunk.units) else []
    out = []
    for li in cands:
        u = chunk.units[li]
        hits = [m.start() for m in re.finditer(re.escape(text), u.value)]
        if not hits:
            pat = r"\s+".join(re.escape(t) for t in text.split())
            hits = [m.start() for m in re.finditer(pat, u.value, re.I)]
            if hits:
                m = re.search(pat, u.value[hits[0]:], re.I)
                text = u.value[hits[0]:hits[0] + m.end()]
        for h in hits:
            out.append(Proposal(u.line_id, h, h + len(text), kind, u.value[h:h + len(text)], detector, confidence, hint))
    return out


# ------------------------------------------------------------------ rules
NAME = r"[A-Z][a-zA-Z'’-]+"
# lowercase surname particles inside a name ('Ottilie van der Brakel', 'Ana de la Cruz', 'Omar al-Rashid'): a particle
# run only counts when a capitalised word follows it; in capitals too ('OTTILIE VAN DER BRAKEL')
NAME_PARTICLE = r"(?i:van|von|de|del|della|der|den|da|di|du|la|le|bin|binti|bint|ibn|abu|al|el|das|dos|do|ter|ten|y)"
NAME_NEXT = r"(?:[A-Z]\.|(?:" + NAME_PARTICLE + r"\s+){0,3}" + NAME + r"|(?:al|el)-" + NAME + r")"
NAME_CUE = re.compile(r"(?:\b(?:Name|Cardholder|Reporting party|Customer|Signed by|signed by|Dear|Reporting officer|sister,|brother,|"
                      r"Notice for|Statement for|Prepared for|Patient|Patient name|Beneficiary|Member name|Insured)[:,]?|\bcc:)\s+"
                      r"((?:(?:Mr|Ms|Mrs|Dr|Ofc)\.\s+)?" + NAME + r"(?:\s+" + NAME_NEXT + r"){1,2})")
# a person introducing themselves or named as a message's author in prose and chat traces (D3f): 'Hi, this is Corbin
# Ashgrove', 'My name is ...', 'I'm ...', 'Second note from ...'. Two or three capitalised words; the first must be a
# known given name (so 'This is Customer Service' / 'a note from Harbor Grocery' are not people)
SELF_INTRO = re.compile(r"\b(?:[Tt]his is|[Ii]t'?s|I'?m|I am|[Mm]y name is|[Mm]y name's|[Nn]ame's|(?:[Nn]ote|[Mm]essage|[Ee]mail|[Ll]etter|[Cc]all|[Cc]omplaint|[Rr]equest)\s+from|[Ff]rom)\s+(?:(?:Mr|Ms|Mrs|Dr)\.?\s+)?"
                        r"(" + NAME + r"(?:\s+" + NAME_NEXT + r"){1,2})")
# the payee / payer of a person-to-person payment on a statement row ('ZELLE TO Teodoro Villanueva', 'Venmo from ...',
# 'Payment to ...'); the first word must be a known given name (so 'PAYMENT TO CHASE CARD' is not a person)
P2P_NAME = re.compile(r"(?i:\b(?:zelle|venmo|paypal|cash\s?app|wire|transfer|xfer|payment|sent|received|refund)(?:\s+(?:payment|transfer))?\s+(?:to|from))\s+"
                      r"(" + NAME + r"(?:\s+" + NAME_NEXT + r"){1,2})")
# labelled person fields on forms, claims, statements and records ('Policyholder: Desmond Arkwright', 'Account holder name:
# Ana Ruiz'). A colon is required: without it these words start ordinary phrases ('Member Services', 'Client Relations')
PERSON_LABELS = (r"Policy\s?holder|Named insured|Insured(?: person| party)?|Claimant|Beneficiary|Account\s?holder|Card\s?holder|"
                 r"Patient|Member|Subscriber|Client|Customer|Applicant|Co-?applicant|Borrower|Co-?borrower|Guarantor|Tenant|"
                 r"Lessee|Employee|Insured driver|Driver|Policy owner|Account owner|Recipient|Addressee|Guardian|Spouse|"
                 r"Dependent|Resident|Student|Passenger|Traveller|Traveler|Guest|Caller|Complainant|Requester|Requestor|"
                 r"Decedent|Deceased|Contact person|Primary contact|Emergency contact|Next of kin|Victim|Witness|Suspect|"
                 r"Employee record|Employee name|Patient name|Manager|Supervisor|Assignee|Reporter|Representative|Physician|"
                 r"Doctor|Nurse|Attorney|Author|Prepared by|Submitted by|Reviewed by|Approved by|Signed by|Referred by")
LABELED_NAME = re.compile(r"(?<![\w-])(?i:" + PERSON_LABELS + r")(?:'s)?(?:\s+(?i:full\s+)?(?i:name))?\s*:\s*"
                          r"((?:(?:Mr|Ms|Mrs|Dr)\.?\s+)?(?:[A-Z]\.\s+" + NAME + r"|" + NAME + r"(?:\s+" + NAME_NEXT + r"){1,2}))")
# a labelled value that is a business, not a person ('Customer: Harbor Grocery', 'Client: Tallow Freight')
LABEL_ORG_HINT = re.compile(r"\b(?:Bank|Banking|Store|Stores|Market|Mart|Grocery|Pharmacy|Hospital|Clinic|Insurance|Mutual|Assurance|"
                            r"Services?|Group|Freight|Logistics|Holdings|Partners|Associates|Solutions|Systems|Technologies|"
                            r"Foundation|Trust|Fund|Agency|Department|University|College|School|Church|Hotel|Restaurant|Cafe|"
                            r"Motors|Auto|Company|Enterprises?|Industries|International|Global|Media|Labs?|Electronics|Coffee|Marketplace|"
                            r"Corporation|Corp|Inc|LLC|Ltd|Airlines?|Medicine|Medical|Health|Dental|Wireless|Telecom)\b")
LAST_FIRST = re.compile(r"\b([A-Z][A-Z'’-]+, [A-Z][A-Z'’-]+(?: [A-Z]\b)?)")
INITIAL = re.compile(r"(?<![\w.])(?<![A-Za-z] )([A-Z]\. " + NAME + r")\b")
STOP = {"Badge", "DOB", "Phone", "Address", "Date", "Dispute", "Analyst", "Card", "Account"}
# 'LORILLARD, INC' / 'COMPANY, ET AL' / 'YORK, NEW' look like 'SURNAME, GIVEN' but are organisations or places
NOT_NAME_TOKENS = {"INC", "CORP", "CO", "LLC", "LLP", "LTD", "PLC", "COMPANY", "CORPORATION", "ET", "AL", "BANK", "TRUST",
                   "NEW", "YORK", "JERSEY", "CITY", "COUNTY", "STATE", "STATES", "UNITED", "AMERICA", "USA", "SDN", "BHD",
                   "DEPARTMENT", "DEPT", "OFFICE", "COURT", "DISTRICT", "SERVICE", "SERVICES", "SYSTEM", "GROUP", "HOLDINGS"}
TITLE_TOKENS = {"ESQ", "JR", "SR", "MD", "PHD", "II", "III"}
RX = [
    ("EMAIL", PATTERNS["EMAIL"]), ("PHONE", PATTERNS["PHONE"]), ("CARD", PATTERNS["PAN_LIKE"]),
    ("ACCOUNT", re.compile(r"(?<![\d-])\d{4}-\d{3}-\d{4}(?![\d-])")),
    ("ID", re.compile(r"\b(?:[A-Z]{2,4}-\d{2,}(?:-\d+)*|DL \d{4}-\d{4}-\d{2})\b")),
    ("ADDRESS", STREET_ADDRESS),
    ("ZIP", re.compile(r"(?<=\b[A-Z]{2} )\d{5}(?:-\d{4})?\b")),
]
# ---- business entities (invoices/receipts): a private counterparty's name, registration/tax numbers, phone lines
# documents that record a private transaction; only there is a company name treated as identifying (court filings and
# statements name public companies and merchants as evidence)
TRANSACTION_DOC = re.compile(r"(?i)\b(?:(?:tax |simplified tax |commercial )?invoice|receipt|re[cç]u|facture|factura|rechnung|quittung|"
                             r"kassenbon|kassenzettel|beleg|cash (?:bill|sale)|guest bill|hotel bill|nota fiscal|fattura|ricevuta)\b")
_ORG_WORD = r"[A-Z0-9][A-Za-z0-9&'’.+-]*"
LEGAL_SUFFIX = (r"(?:SDN\.?\s*BHD\.?|Sdn\.?\s*Bhd\.?|BERHAD|Berhad|L\.?L\.?C\.?|LLP|INC\.?|Inc\.?|CORP\.?|Corp\.?|CORPORATION|"
                r"Corporation|LTD\.?|Ltd\.?|LIMITED|Limited|PLC|GmbH|GMBH|AG|KG|OHG|e\.K\.|S\.A\.?R?\.?L?\.?|S\.p\.A\.|B\.V\.|N\.V\.|"
                r"S\.?A\.?S\.?|Pty\.?\s*Ltd\.?|PTE\.?\s*LTD\.?|Pte\.?\s*Ltd\.?|K\.K\.|Oy|AB|A/S|ApS|SRL|S\.R\.L\.)")
# 'GOGIANT ENGINEERING (M) SDN BHD', 'Acme Widgets LLC', 'Muster GmbH & Co. KG'; no comma before the suffix
ORG_SUFFIX = re.compile(r"(?<![\w&])((?:" + _ORG_WORD + r"\s+){1,5}(?:\([A-Z]\)\s+)?" + LEGAL_SUFFIX +
                        r"(?:\s*&\s*Co\.?\s*(?:KG|OHG))?)(?![\w])")
# business-name prefixes for sole proprietorships ('PERNIAGAAN ZHENG HUI', 'SYARIKAT ...', 'KEDAI ...')
ORG_PREFIX = re.compile(r"\b((?:PERNIAGAAN|SYARIKAT|KEDAI|ENTERPRISE|TOKO|PT\.|CV\.)\s+(?:[A-Z][A-Z0-9&'’.-]*\s*){1,4})")
# company registration numbers: Malaysian SSM '(123456-X)'/'JM0325955-V'; labelled numbers below
COMPANY_REG = re.compile(r"(?<![\w-])\(?([A-Z]{0,3}\d{5,9}-[A-Z])\)?(?![\w-])")
TAX_LABEL = re.compile(r"(?i:\b(?:GST|SST|VAT|TVA|IVA|MWST|MwSt|UID|UStId|USt-?IdNr|USt\.?\s*-?\s*ID|St\.?\s*-?\s*Nr|Steuer-?\s*Nr|"
                       r"Steuernummer|TIN|EIN|ABN|ACN|BRN|CRN|SIRET|SIREN|KvK|Company\s*(?:Reg(?:istration)?\.?\s*)?No|"
                       r"Co\.?\s*Reg(?:istration)?\.?\s*No|Reg(?:istration)?\.?\s*No|HRB|HRA)\b\.?\s*(?:(?:REG(?:ISTRATION)?\.?\s*)?"
                       r"(?:NO|Nr|ID|#|number)\b\.?)?\s*[:.#]?\s*)"
                       r"((?:[A-Z]{2,3}[\s-]?)?\d[\d\s./-]{4,18}\d(?:\s?[A-Z]{2,3})?|[A-Z]{2}\d{8,12})")
PHONE_LABEL = re.compile(r"(?i:\b(?:tel(?:ephone|efon|éfono)?|t[ée]l|phone|ph|fax|telefax|mob(?:ile)?|hp|handphone|cell|natel)\b\.?\s*"
                         r"(?:no\.?\s*)?[:.]?\s*)(\+?\(?\d[\d ()/.-]{5,17}\d)")
US_STATE = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR "
            "PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
# 'Austin, Tx. 78701', 'New York, N.Y. 10016', 'Austin, TX 78701-1234' (the plain 'XX 12345' form is in RX)
US_ZIP_ADDR = re.compile(r",\s*(?i:" + "|".join(a + r"\.?" + b + r"\.?" for a, b in US_STATE) + r")\s+(\d{5}(?:-\d{4})?)(?!\d)")
# a payslip's employee number printed in brackets on its own line under the header ('(021)')
PAYSLIP_DOC = re.compile(r"(?i)\b(?:pay\s?slip|pay stub|payroll statement|earnings statement|lohnabrechnung|gehaltsabrechnung|"
                         r"bulletin de (?:paie|salaire)|n[oó]mina|расч[её]тный лист(?:ок)?|платіжна відомість)\b")
PAREN_NUMBER_LINE = re.compile(r"^\s*\((\d{2,6})\)\s*$")
FORM_NUMBER = re.compile(r"(?:CMS|HCFA|SF|DD|OMB|FORM|ISO|IEC|RFC|PCI|HHS|SSA|IRS|DS|GSA)-\d", re.I)
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
# 'DOB: 06/19/1984', 'Date of birth: 1984-06-19', 'Date of Birth: June 19, 1984', 'D.O.B. 19 Jun 1984', 'DOB 19.06.1984'
# prose too (A3b): 'My date of birth is 03/07/1984', 'I was born on March 7, 1984'
# a short qualifier may sit between the label and the value: 'Date of birth (for verification): 03/14/1986',
# 'DOB of the insured: ...', 'Birth date - ...'
DOB = re.compile(r"(?i:\b(?:DOB|Date of birth|Birth\s?date|Birthday|D\.O\.B\.?)(?:\s*\([^()\n\d]{1,40}\))?(?:\s+of\s+(?:the\s+)?[a-z]+(?:\s+[a-z]+)?)?"
                 r"(?:\s+(?:on file|on record))?(?:\s+(?:is|was))?|\bborn(?:\s+on)?)\s*[:\-\u2013]?\s*("
                 r"\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}|\d{1,2}[.-]\d{1,2}[.-](?:\d{4}|\d{2})|"
                 + _MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+" + _MONTH + r",?\s+\d{4})(?!\d)")
# the last four of a card, account, phone or any other identifier: '**** 1111', '***-***-0187', 'ending in 1904',
# 'last four of the ID: 1904', 'my card ends in 1111' ('ends in' only next to an identifier word: 'the lease ends in 2027'
# is not one). The case links each one to its full value (consistency.plan), whatever kind that value is.
_ID_NOUN = r"(?:card|visa|mastercard|amex|discover|debit|credit|account|acct|phone|number|no\.?|ID|policy|member|ssn|social|licen[cs]e|passport|routing|iban)"
LAST4 = re.compile(r"(?:\*{3,}[\s*\-]*|ending(?: in| with)?\s+|x{2,}[\s\-]*|last ?(?:4|four)(?: digits)?(?: (?:of|on|for) (?:the |my |your |this |her |his |their )?"
                   r"[A-Za-z]+(?: (?:number|no\.?|#))?)?(?: (?:is|are))?:?\s*|\b" + _ID_NOUN + r"\b[^\n\d]{0,25}?\bends? (?:in|with)\s+)(\d{4})\b", re.I)
LABEL_LINE = {"NAME": "PERSON", "DOB": "DOB", "ADDRESS": "ADDRESS", "LIC NO": "ID", "DL NO": "ID"}
_LABEL_KEYS = {re.sub(r"\s+", "", k) for k in LABEL_LINE}


def _is_field_label(v):
    """A line that is itself a field label ('NAME', OCR'd twice or letter-spaced 'AD D R E SS') is not the value of
    the label above it."""
    return re.sub(r"[\s:]+", "", v.upper()) in _LABEL_KEYS


JSON_FIELD = {"PERSON": "PERSON", "EMAIL": "EMAIL", "PHONE": "PHONE", "CARD": "CARD", "ID": "ID", "ADDRESS": "ADDRESS", "DATE": "DOB"}
# agent-trace keys that hold a person / a document number (tool results quote documents field by field)
JSON_PERSON_KEY = re.compile(r"(?:^|_)(?:full_?name|holder(?:_name)?|customer_name|patient_name|member_name|insured_name|"
                             r"cardholder|signer|signed_by|sender|recipient|from|to|cc|bcc|escalated_to|assignee|"
                             r"addressee|attention|attn|applicant|registrant|claimant|beneficiary|attorney|counsel|lawyer|prepared_by|"
                             r"served_by|notary|customer|patient|member|client|subscriber|insured|policyholder|guarantor|caller|"
                             r"contact_name|employee_name|tenant|borrower|account_holder|reporter|complainant)$")
# keys whose value is a person even when it is one word ('customer_name': 'Maria')
JSON_NAME_ONLY_KEY = re.compile(r"(?:^|_)(?:customer_name|holder_name|full_?name|patient_name|member_name|insured_name)$")
JSON_SURNAME_KEY = re.compile(r"(?:^|_)(?:surname|last_?name|family_?name)$")
JSON_GIVEN_KEY = re.compile(r"(?:^|_)(?:given_?names?|first_?name|forename)$")
JSON_ID_KEY = re.compile(r"(?:^|_)(?:card_no|card_number|id_no|id_number|personal_no|personal_number|passport_no|passport_number|"
                         r"licen[cs]e_no|licen[cs]e_number|document_no|document_number|national_id|ssn|case_no|case_number|docket_no|"
                         r"claim_no|claim_number|policy_no|policy_number|member_id|member_number|subscriber_id|mrn|medical_record_number|patient_id|patient_number|"
                         r"policy_id|insurance_id|employee_id|student_id|tax_id)$")
# keys holding a private counterparty organisation (an invoice's seller/buyer); 'merchant' on a statement is evidence
JSON_ORG_KEY = re.compile(r"(?:^|_)(?:seller|buyer|vendor|supplier|bill_to|sold_to|employer|company_name|business_name)$")
JSON_SSN_KEY = re.compile(r"(?:^|_)(?:ssn|social_security(?:_number|_no)?|sin|tin|tax_id)$")
JSON_ACCOUNT_KEY = re.compile(r"(?:^|_)(?:account|account_number|account_no|acct|acct_no|acct_number|iban|routing_number)$")
# "key": "value" inside a JSON document that is itself a string value (escaped quotes are already decoded)
EMBEDDED_KV = re.compile(r'"([A-Za-z_][\w .-]{0,40})"\s*:\s*"((?:[^"\\]){1,200})"')
# SSN last four: masked form or stated in words ('XXX-XX-4096', 'last four digits of my SSN are 4096', 'SSN ending 4096')
SSN4 = [re.compile(r"(?<![\w*-])(?:XXX|xxx|\*{3})[- ]?(?:XX|xx|\*{2})[- ]?(\d{4})(?![\d-])"),  # not a longer mask run (a card)
        re.compile(r"(?i)\blast\s+(?:four|4)(?:\s+digits)?\s+of\s+(?:my\s+|the\s+|your\s+|his\s+|her\s+|their\s+)?"
                   r"(?:SSN|social\s+security(?:\s+number)?)\s*(?:is|are|:|=|-)?\s*(\d{4})(?![\d-])"),
        re.compile(r"(?i)\b(?:SSN|social\s+security(?:\s+number)?|SS#)\s*(?:ending(?:\s+in)?|last\s+(?:four|4)(?:\s+digits)?)\s*"
                   r"(?:is|are|:|=|-)?\s*(\d{4})(?![\d-])")]
JSON_DOB_KEY = re.compile(r"(?:^|_)(?:date_of_birth|dob|birth_date|birthdate)$")
PERSONISH = re.compile(r"(?:(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+)?([^\W\d_][^\W\d_'’.-]*(?:[\s'’-][^\W\d_][^\W\d_'’.-]*){0,3})")


def _looks_like_person(text):
    from .legal import given_names
    toks = text.split()
    return toks[0].upper() in given_names() and not any(x.strip(".,").upper() in NOT_NAME_TOKENS for x in toks)


class RulesDetector:
    name = "rules"
    learned = False

    def detect(self, chunk):
        out = []
        for i, u in enumerate(chunk.units):
            v = u.value
            def add(s, e, kind, conf=0.9, meta=None):
                out.append(Proposal(u.line_id, s, e, kind, v[s:e], self.name, conf, None, meta or {}))
            if u.source == "json":
                if self._json_field(u.path, v, 0, add):
                    continue
                # a JSON document serialized inside a string (tool-call arguments, tool results): its key/value pairs
                # are classified like real fields, at their offsets inside the string
                if v.lstrip()[:1] in "{[":
                    for m in EMBEDDED_KV.finditer(v):
                        self._json_field(tuple(u.path) + (m.group(1),), m.group(2), m.start(2), add)
            if u.source != "json":
                from .mrz import is_mrz
                if is_mrz(v):
                    st = len(v) - len(v.lstrip())
                    add(st, len(v.rstrip()), "MRZ", 0.95); continue
            for kind, rx in RX:
                for m in rx.finditer(v):
                    if kind == "CARD" and len(re.sub(r"\D", "", m.group())) < 13:
                        continue
                    if kind == "PHONE" and re.search(r"\b8(?:00|88|77|66|55)\b", m.group()):
                        continue  # toll-free numbers belong to organizations
                    if kind == "ID" and FORM_NUMBER.match(m.group()):
                        continue  # a form's own number ('CMS-1500', 'SF-86') names the form, not a person
                    add(m.start(), m.end(), kind)
            for m in DOB.finditer(v):
                add(m.start(1), m.end(1), "DOB")
            ssn4 = set()
            for rx in SSN4:
                for m in rx.finditer(v):
                    add(m.start(1), m.end(1), "SSN_LAST4", 0.9); ssn4.add(m.start(1))
            for m in LAST4.finditer(v):
                if m.start(1) not in ssn4:
                    add(m.start(1), m.end(1), "CARD_LAST4", 0.85)
            for rx in (NAME_CUE, LABELED_NAME, LAST_FIRST, INITIAL, SELF_INTRO, P2P_NAME):
                for m in rx.finditer(v):
                    if rx in (NAME_CUE, LABELED_NAME, SELF_INTRO, P2P_NAME) and LABEL_ORG_HINT.search(m.group(1)):
                        continue
                    if rx in (SELF_INTRO, P2P_NAME) and not _looks_like_person(m.group(1)):
                        continue
                    s, e = m.span(1)
                    t = re.sub(r"^(?:Mr|Ms|Mrs|Dr|Ofc)\.?\s+", "", v[s:e]); s = e - len(t)
                    toks = t.split()
                    while len(toks) > 1 and (toks[-1] in STOP or toks[-1].strip(".,").upper() in TITLE_TOKENS):
                        toks.pop()
                    if any(x.strip(".,").upper() in NOT_NAME_TOKENS for x in toks):
                        continue
                    t = " ".join(toks); e = s + len(t)
                    if len(toks) < 2:
                        continue
                    add(s, e, "PERSON", 0.75)
            if u.source != "json":
                self._business(chunk, i, u, v, add)
            prev = chunk.units[i - 1].value.strip().upper() if i else ""
            if prev in LABEL_LINE and v.strip() and not _is_field_label(v):
                add(0, len(v), LABEL_LINE[prev], 0.8)
            if prev.rstrip(",") in ("SINCERELY", "REGARDS", "THANK YOU") and re.fullmatch(NAME + r"(?:\s+(?:[A-Z]\.|" + NAME + r")){1,2}", v.strip()):
                add(0, len(v), "PERSON", 0.8)
        return out

    @staticmethod
    def _json_field(path, v, base, add0):
        """Classify one JSON string value by its key. `base` = offset of `v` inside the unit text. True = the value is
        fully handled (no other rule runs on it)."""
        def add(s_, e_, kind, conf=0.9, meta=None):
            add0(base + s_, base + e_, kind, conf, meta)
        fk = field_kind(path)
        key = next((k for k in reversed(path) if isinstance(k, str)), "").lower()
        kn0 = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
        if fk in ("ID",) and re.fullmatch(r"\d{4}-\d{3}-\d{4}", v):
            fk = "ACCOUNT"
        if "last4" in kn0.replace("_", "") and re.fullmatch(r"\d{4}", v):
            add(0, len(v), "SSN_LAST4" if "ssn" in kn0 or "social" in kn0 else "CARD_LAST4", 0.95); return True
        if JSON_SSN_KEY.search(kn0) and re.fullmatch(r"\d{3}[- ]?\d{2}[- ]?\d{4}", v.strip()):
            st = len(v) - len(v.lstrip()); add(st, st + len(v.strip()), "SSN", 0.95); return True
        if JSON_ACCOUNT_KEY.search(kn0) and re.fullmatch(r"[0-9][0-9 -]{4,}[0-9]", v.strip()) and len(re.sub(r"\D", "", v)) >= 6:
            st = len(v) - len(v.lstrip()); add(st, st + len(v.strip()), "ACCOUNT", 0.95); return True
        if fk == "PERSON" and isinstance(v, str) and JSON_NAME_ONLY_KEY.search(kn0) and \
                re.fullmatch(r"[^\W\d_][^\W\d_'’-]+", v.strip()) and v.strip()[:1].isupper():
            st = len(v) - len(v.lstrip())
            add(st, st + len(v.strip()), "GIVEN", 0.9)  # one-word holder name: the given name used alone
        elif fk in JSON_FIELD:
            add(0, len(v), JSON_FIELD[fk], 0.95)
        elif isinstance(v, str) and v.strip():
            kn = kn0
            m = PERSONISH.fullmatch(v.strip())
            st = len(v) - len(v.lstrip())
            if m and (JSON_SURNAME_KEY.search(kn) or JSON_GIVEN_KEY.search(kn)) and len(m.group(1).split()) <= 3:
                add(st + m.start(1), st + m.end(1), "SURNAME" if JSON_SURNAME_KEY.search(kn) else "GIVEN", 0.9)
            elif m and JSON_PERSON_KEY.search(kn) and 2 <= len(m.group(1).split()) <= 4 and \
                    not any(x.strip(".,").upper() in NOT_NAME_TOKENS for x in m.group(1).split()) and \
                    not LABEL_ORG_HINT.search(m.group(1)):
                add(st + m.start(1), st + m.end(1), "PERSON", 0.9)
            elif m and kn == "name" and 2 <= len(m.group(1).split()) <= 4 and _looks_like_person(m.group(1)):
                add(st + m.start(1), st + m.end(1), "PERSON", 0.85)  # a bare 'name' that starts with a known given name
            elif m and JSON_NAME_ONLY_KEY.search(kn) and len(m.group(1).split()) == 1 and m.group(1)[:1].isupper() and \
                    len(m.group(1)) >= 2 and m.group(1).upper() not in NOT_NAME_TOKENS:
                add(st + m.start(1), st + m.end(1), "GIVEN", 0.85)
            elif JSON_ID_KEY.search(kn) and re.fullmatch(r"[A-Z0-9][A-Z0-9 ,./-]{3,}", v.strip()) and \
                    len(re.sub(r"\D", "", v)) >= 5 and (re.search(r"\d{4}", v) or re.fullmatch(r"\d{1,3}(?:,\d{3})+", v.strip())):
                add(st, len(v.rstrip()), "ID", 0.9)
            elif JSON_DOB_KEY.search(kn) and re.search(r"\d", v):
                add(st, len(v.rstrip()), "DOB", 0.9)
            elif JSON_ORG_KEY.search(kn) and re.search(r"[A-Za-z]{3}", v):
                add(st, len(v.rstrip()), "ORG", 0.85)
        return False

    @staticmethod
    def _business(chunk, i, u, v, add):
        """Generic business-entity rules: labelled tax/registration numbers and phone numbers (any document); company
        names (legal suffix, proprietorship prefix, the name line above a registration number) only in transaction
        documents; US ZIP after 'City, St.'; a payslip's bracketed employee number."""
        for m in TAX_LABEL.finditer(v):
            if len(re.sub(r"\D", "", m.group(1))) >= 6:
                add(m.start(1), m.end(1) - (len(m.group(1)) - len(m.group(1).rstrip())), "ID", 0.85)
        for m in PHONE_LABEL.finditer(v):
            num = m.group(1).rstrip(" (/.-")
            if len(re.sub(r"\D", "", num)) >= 7 and not re.match(r"\(?(?:\+?1[ .-]?)?\(?8(?:00|88|77|66|55)\b", num):
                out_s = m.start(1)
                add(out_s, out_s + len(num), "PHONE", 0.85, {"labelled": True})
        for m in US_ZIP_ADDR.finditer(v):
            add(m.start(1), m.end(1), "ZIP", 0.85)
        texts = getattr(chunk, "_doc_text", None)
        if texts is None:
            texts = chunk._doc_text = "\n".join(x.value for x in chunk.units if x.source != "json")
        if PAYSLIP_DOC.search(texts) and i < 12:
            m = PAREN_NUMBER_LINE.match(v)
            if m:
                add(m.start(1), m.end(1), "ID", 0.8)
        if not TRANSACTION_DOC.search(texts):
            return
        for m in COMPANY_REG.finditer(v):
            add(m.start(1), m.end(1), "ID", 0.85)
        for rx in (ORG_SUFFIX, ORG_PREFIX):
            for m in rx.finditer(v):
                t = m.group(1).rstrip(" ,.;:")
                if not re.search(r"[A-Za-z]{3}", t):
                    continue
                add(m.start(1), m.start(1) + len(t), "ORG", 0.8)
        # the business-name line directly above its registration number (receipt headers)
        nxt = chunk.units[i + 1].value.strip() if i + 1 < len(chunk.units) else ""
        if COMPANY_REG.fullmatch(nxt) and re.fullmatch(r"[A-Z][A-Z0-9&'’.() -]{3,60}", v.strip()) and len(v.split()) >= 2:
            st = len(v) - len(v.lstrip())
            add(st, len(v.rstrip()), "ORG", 0.8)


# ------------------------------------------------------------------ GLiNER
GLINER_LABELS = {"person": "PERSON", "street address": "ADDRESS", "phone number": "PHONE", "email": "EMAIL",
                 "credit card number": "CARD", "bank account number": "ACCOUNT", "identification number": "ID",
                 "date of birth": "DOB", "postal code": "ZIP"}


class GlinerDetector:
    learned = True

    def __init__(self, model="urchade/gliner_multi_pii-v1", threshold=0.4):
        try:
            from gliner import GLiNER
        except ImportError:
            raise RuntimeError("GLINER_NOT_INSTALLED") from None
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download during a case run
        self.model = GLiNER.from_pretrained(model)
        self.name = "gliner:" + model.split("/")[-1]
        self.threshold = threshold

    def detect(self, chunk):
        out = []
        # windows of lines (GLiNER max ~384 tokens); offsets mapped back to units
        win, size = [], 0
        def run(win):
            text, spans = "", []
            for i, u in win:
                spans.append((len(text), i)); text += u.text + "\n"
            for ent in self.model.predict_entities(text, list(GLINER_LABELS), threshold=self.threshold):
                s, e = ent["start"], ent["end"]
                base, li = max((b, i) for b, i in spans if b <= s)
                u = chunk.units[li]
                vs, ve = s - base - u.offset, e - base - u.offset
                if vs < 0 or ve > len(u.value):
                    continue
                out.append(Proposal(u.line_id, vs, ve, GLINER_LABELS[ent["label"]], u.value[vs:ve], self.name, float(ent["score"])))
        for i, u in enumerate(chunk.units):
            n = len(u.text.split())
            if size + n > 180 and win:
                run(win); win, size = [], 0
            win.append((i, u)); size += n
        if win:
            run(win)
        return out


# ------------------------------------------------------------------ local LLM (llama.cpp)
LLM_SYSTEM = ("You find personal data in document text so it can be replaced with fictional values. "
              "Only report text copied EXACTLY from a numbered line. Do not report amounts, ordinary dates, merchants, banks, "
              "products, police departments or other organizations. Types: PERSON (any form: 'Jane A. Doe', 'DOE, JANE', 'J. Doe'), "
              "ADDRESS (street line), ZIP, PHONE, EMAIL, CARD (full card number), CARD_LAST4 (last 4 digits of a card), "
              "ACCOUNT (bank account number), ID (customer, case, report or license number that identifies a person or their case), DOB (date of birth). "
              "Do not report transaction, reference, authorization or confirmation numbers, or the addresses and phone numbers of businesses. "
              "For each PERSON give person_id: the same id for mentions of the same human, different ids for different humans "
              "(e.g. relatives sharing a surname are different people).")
LLM_SCHEMA = {"type": "object", "properties": {"entities": {"type": "array", "items": {"type": "object", "properties": {
    "line": {"type": "integer"}, "text": {"type": "string"}, "type": {"type": "string", "enum": list(KINDS)},
    "person_id": {"type": "string"}}, "required": ["line", "text", "type"]}}}, "required": ["entities"]}


class LlamaCppDetector:
    learned = True

    def __init__(self, model_file="qwen2.5-3b-instruct-q4_k_m.gguf", n_ctx=4096, threads=8):
        from .models import find_model
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("LLAMA_CPP_NOT_INSTALLED") from None
        path = find_model(model_file)
        if path is None:
            raise RuntimeError("LOCAL_LLM_MODEL_NOT_FOUND")
        self.llm = Llama(str(path), n_ctx=n_ctx, n_threads=threads, verbose=False, seed=0)
        self.name = "llama.cpp:" + model_file
        self.rejected = []
        self.calls = []

    def _chat(self, user, schema):
        t0 = time.perf_counter()
        r = self.llm.create_chat_completion(messages=[{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": user}],
                                            response_format={"type": "json_object", "schema": schema}, temperature=0, max_tokens=1500)
        self.calls.append({"seconds": round(time.perf_counter() - t0, 2), "prompt_tokens": r["usage"]["prompt_tokens"],
                           "completion_tokens": r["usage"]["completion_tokens"]})
        try:
            return json.loads(r["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError):
            return {}

    def detect(self, chunk):
        data = self._chat("Document lines:\n" + chunk.prompt_text() + "\n\nReturn JSON {\"entities\": [...]}.", LLM_SCHEMA)
        out = []
        for e in data.get("entities", []) if isinstance(data, dict) else []:
            if not isinstance(e, dict):
                continue
            g = ground(chunk, e.get("line"), e.get("text"), e.get("type"), self.name, 0.8,
                       f"{chunk.doc}:{e.get('person_id')}" if e.get("person_id") else None)
            if not g:  # cited line wrong? accept only if the exact text exists elsewhere on the page, else reject
                g = ground(chunk, None, e.get("text"), e.get("type"), self.name, 0.6)
                for p in g:
                    p.meta["line_citation_wrong"] = True
            if g:
                out.extend(g)
            else:
                self.rejected.append({"doc": chunk.doc, "page": chunk.page, "type": e.get("type"), "reason": "UNGROUNDED_TEXT_NOT_IN_SOURCE"})
        return out

    def resolve_people(self, mentions):
        """Case-level entity resolution over person mentions (strings + short contexts). Returns {mention: cluster}."""
        lines = "\n".join(f"M{i}: \"{m}\" — context: {c}" for i, (m, c) in enumerate(mentions))
        schema = {"type": "object", "properties": {"clusters": {"type": "array", "items": {"type": "object", "properties": {
            "mention": {"type": "integer"}, "person": {"type": "string"}}, "required": ["mention", "person"]}}}, "required": ["clusters"]}
        data = self._chat("Assign every person mention below to a person label (P1, P2, ...). Same human => same label. "
                          "Different first names are different humans even with the same surname. An initial like 'J. Doe' "
                          "should be linked using context.\n" + lines + "\nReturn JSON {\"clusters\": [...]}.", schema)
        out = {}
        for c in data.get("clusters", []) if isinstance(data, dict) else []:
            if isinstance(c, dict) and isinstance(c.get("mention"), int) and 0 <= c["mention"] < len(mentions):
                out[mentions[c["mention"]][0]] = str(c.get("person"))
        return out

    def close(self):
        self.llm.close() if hasattr(self.llm, "close") else None
        del self.llm


# ------------------------------------------------------------------ hosted (stub only)
class HostedLLMDetector:
    """NOT CALLED. Placeholder for an opt-in hosted model (vision or text). It would send page text and/or page
    images of sensitive documents to a third party, which needs explicit customer consent, a DPA, retention terms
    and a key. CasePatch ships with this disabled; no key, no network call."""
    name = "hosted-llm-stub"
    learned = True

    def __init__(self, provider="unconfigured"):
        raise RuntimeError("HOSTED_DETECTOR_DISABLED: no consent, contract or key configured; source never leaves the machine")


def make_detector(name):
    if name == "rules":
        return RulesDetector()
    if name == "gliner":
        return GlinerDetector()
    if name == "llm":
        return LlamaCppDetector()
    if name == "hosted":
        return HostedLLMDetector()
    raise ValueError("UNKNOWN_DETECTOR")
