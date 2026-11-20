"""Person names in letters and correspondence.

Letters name people in fixed ways the general rules miss:

- salutations ('Dear Miss Calder', 'My dear Mr. Henrik Voss', 'To Mrs. Ana Silva');
- courtesy titles in the body or an addressee block ('Miss Calder', 'Mr. Baehr');
- the signature block after a complimentary close ('Sincerely yours,' then 'Anna Mayer-Schmidt');
- short surname-first romanized names in those same positions ('LIN YUTANG', 'WANG WEI').

Proposals are grounded in a line and record the rule in meta['why']. Generic greetings
('Dear Sir', 'To whom it may concern') are not names.
"""
from __future__ import annotations

import re

from .detect import NOT_NAME_TOKENS, Proposal

HON = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Hon(?:orable)?|Messrs|Mme|Mlle|Herr|Frau|Sr|Sra|Srta)\.?"
NAME_TOKEN = r"[A-ZÀ-ÖØ-ÞĀ-Ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’.-]*"
# 1-4 name tokens, allowing a bare or dotted middle initial and a hyphenated surname
NAME_RUN = rf"{NAME_TOKEN}(?:\s+(?:[A-ZÀ-ÖØ-ÞĀ-Ž]\.(?=\s|$)|{NAME_TOKEN})){{0,3}}"
GENERIC = re.compile(
    r"(?i)^(sir|madam|madame|sirs|mesdames|gentlemen|friends?|sir or madam|"
    r"to whom it may concern)(?:\s|[/,]|$)"
)
SALUTATION = re.compile(
    rf"(?i)\b(?:(?:my\s+)?dear|attn:?|attention:?)(?:\s+{HON})?\s+({NAME_RUN})"
)
# 'To Mrs. Ana Silva' at the start of a line (not 'to find the defendant')
TO_LINE = re.compile(rf"(?i)^(?:to|for the attention of)\s+(?:{HON}\s+)?({NAME_RUN})\s*$")
HON_NAME = re.compile(rf"(?<![\w])((?:{HON})\s+{NAME_RUN})")
CLOSE = re.compile(
    r"(?i)^\s*(?:sincerely(?:\s+yours)?|yours\s+(?:truly|sincerely|faithfully)|"
    r"very\s+truly\s+yours|respectfully(?:\s+yours)?|cordially(?:\s+yours)?|"
    r"(?:best|kind)\s+regards|regards|thank\s+you(?:\s+again)?|"
    r"faithfully\s+yours|i\s+remain|your\s+obedient\s+servant|"
    r"i\s+am(?:,|\s+sir|\s+madam)?)\s*,?\s*$"
)
LETTER_CUE = re.compile(
    r"(?i)\b(?:dear|sincerely|yours\s+truly|respectfully|regards|letterhead|"
    r"commissioner of|inspector in charge)\b"
)
ORGISH = re.compile(
    r"(?i)\b(?:department|service|office|bureau|company|co\.|corp|inc|llc|ltd|"
    r"university|hospital|court|government|ministry|agency|commission|"
    r"inspector|commissioner|clerk|agent|director|officer|immigrant|"
    r"insurance|mutual|assurance|bank|bancorp|financial|credit union|group|holdings|partners|associates|"
    r"foundation|clinic|medical|health|pharmacy|store|market|solutions|systems|technologies|labs|freight|"
    r"logistics|airlines?|wireless|telecom|energy|utilities|community|services|support|team)\b"
)
# romanized surname-first: two or three short typographic tokens, no English function words
ROMANIZED = re.compile(
    rf"^({NAME_TOKEN}\s+{NAME_TOKEN}(?:\s+{NAME_TOKEN})?)$"
)
FUNCTION = NOT_NAME_TOKENS | {
    "THE", "OF", "AND", "FOR", "IN", "ON", "AT", "TO", "BY", "A", "AN", "OR", "AS",
    "DEAR", "SINCERELY", "YOURS", "TRULY", "RESPECTFULLY", "REGARDS", "THANK",
}


def _clean(tok):
    return tok.strip(".,;:'’\"“”()").upper()


def _ok_name(text):
    t = text.strip(" ,;:")
    if not t or GENERIC.match(t):
        return False
    parts = [p for p in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’.-]+", t) if re.search(r"[A-Za-z]", p)]
    if not parts:
        return False
    if HON and re.fullmatch(HON, parts[0], re.I) and len(parts) == 1:
        return False
    cores = [_clean(p) for p in parts if not re.fullmatch(r"[A-Z]\.?", p)]
    if any(c in FUNCTION for c in cores):
        return False
    if ORGISH.search(t):
        return False
    # a name has at least one token of 2+ letters after any honorific
    rest = parts[1:] if re.fullmatch(HON, parts[0], re.I) else parts
    return any(len(_clean(p)) >= 2 for p in rest)


def _add(out, line, s, e, why):
    span = line.text[s:e].strip(" ,;:")
    # trim to the matched span inside the line
    inner = line.text[s:s + len(span)] if line.text[s:s + len(span)] == span else line.text[s:e]
    end = s + len(inner.rstrip(" ,;:"))
    cue = re.match(r"(?i)b?cc:\s*", line.text[s:end])
    if cue:  # a 'cc:' recipient line names the person after the cue
        s += cue.end()
    text = line.text[s:end]
    head = text.split(",", 1)[0].rstrip()
    if head != text and len(head.split()) >= 2 and _ok_name(head):
        # 'Marcus Webb, Dispute Analyst' / 'Ana Ruiz, Esq.': the name is before the comma; the role or credential after it
        # stays as written, so the name links to the same person elsewhere in the case ('DOE, JANE' is one token before
        # the comma and stays whole)
        end = s + len(head)
        text = head
    if end - s >= 2 and _ok_name(text):
        out.append(Proposal(line.id, s, end, "PERSON", text, "letters", 0.8, None, {"why": why}))


def _is_letter(docs):
    text = "\n".join(l.text for d in docs for p in d.pages for l in p.lines)
    return bool(LETTER_CUE.search(text))


def _romanized_span(t):
    """Surname-first romanized name typography ('LIN YUTANG', 'WANG WEI')."""
    s = t.strip()
    m = ROMANIZED.match(s)
    if not m:
        return None
    parts = s.split()
    if not (2 <= len(parts) <= 3):
        return None
    if any(len(_clean(p)) < 2 or len(_clean(p)) > 8 or _clean(p) in FUNCTION for p in parts):
        return None
    if not all(p.isupper() or p[0].isupper() for p in parts):
        return None
    # reject obvious place / org typography
    if ORGISH.search(s) or any(_clean(p) in {"STREET", "AVENUE", "ROAD", "CITY", "STATE", "COUNTY"} for p in parts):
        return None
    # an all-caps heading ('PATIENT INTAKE', 'CLAIM DETAILS') is two ordinary words: a romanized name has at least one
    # token that is a known given name ('LIN YUTANG', 'WANG WEI') (D4f)
    from .legal import given_names
    if not any(_clean(p) in given_names() for p in parts):
        return None
    lead = len(t) - len(t.lstrip())
    return lead, lead + len(s)


def _given():
    from .legal import given_names
    return given_names()


def letter_name_proposals(docs):
    out = []
    letter = _is_letter(docs)
    for d in docs:
        for p in d.pages:
            lines = [l for l in p.lines if getattr(l, "source", "ocr") != "json" and l.text.strip()]
            for i, l in enumerate(lines):
                t = l.text
                for m in SALUTATION.finditer(t):
                    if _ok_name(m.group(1)):
                        _add(out, l, m.start(1), m.end(1), "salutation")
                for m in TO_LINE.finditer(t.strip()):
                    # match against stripped text: map back
                    s = t.find(m.group(1))
                    if s >= 0 and _ok_name(m.group(1)):
                        _add(out, l, s, s + len(m.group(1)), "addressee")
                for m in HON_NAME.finditer(t):
                    if _ok_name(m.group(1)):
                        _add(out, l, m.start(1), m.end(1), "honorific")
                if letter and i + 1 < len(lines) and CLOSE.match(t):
                    # signature block: up to three following short name-like lines
                    for j in range(1, 4):
                        if i + j >= len(lines):
                            break
                        nxt = lines[i + j]
                        nt = nxt.text.strip()
                        if not nt or CLOSE.match(nt) or ORGISH.search(nt) or len(nt) > 60:
                            break
                        if re.search(r"\d{3,}", nt) and not HON_NAME.search(nt):
                            break
                        if _ok_name(nt) or _romanized_span(nt):
                            s = nxt.text.find(nt) if nt in nxt.text else len(nxt.text) - len(nxt.text.lstrip())
                            _add(out, nxt, s, s + len(nt), "signature")
                        elif not re.fullmatch(rf"{HON}", nt, re.I):
                            # a title-only line ('Inspector in Charge') ends the block
                            if re.search(r"(?i)\b(?:inspector|commissioner|clerk|agent|director|officer)\b", nt):
                                break
                if letter:
                    rs = _romanized_span(t)
                    if rs:
                        _add(out, l, rs[0], rs[1], "romanized-surname-first")
            # addressee block: honorific / romanized / short name lines before the first salutation
            if letter:
                sal_i = next((k for k, l in enumerate(lines) if SALUTATION.search(l.text) or TO_LINE.match(l.text.strip())),
                             min(12, len(lines)))
                for l in lines[:sal_i]:
                    t = l.text.strip()
                    if not t or ORGISH.search(t) or re.search(r"\d{3,}", t):
                        continue
                    if HON_NAME.search(l.text) or SALUTATION.search(l.text):
                        continue  # already added
                    rs = _romanized_span(t)
                    if rs:
                        continue  # already added on the letter path
                    if 2 <= len(t.split()) <= 4 and _ok_name(t) and re.match(rf"{NAME_RUN}$", t) and \
                            (not t.isupper() or _romanized_span(t) is not None or
                             any(_clean(x) in _given() for x in t.split())):  # not an all-caps heading
                        s = l.text.find(t)
                        _add(out, l, s, s + len(t), "addressee-block")
    return out
