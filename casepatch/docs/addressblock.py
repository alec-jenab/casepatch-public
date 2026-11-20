"""Person names from mailing-address blocks (D3/A3).

Statements, bills and letters print the recipient as a block:

    Darnell Okoye            <- the name: 2-4 name tokens on their own line
    88 Fulton St             <- a street address line
    Brooklyn, NY 11201       <- 'City, ST ZIP'

The general rules only see 'Darnell Okoye' as two capitalised words. When the line right below a short name-shaped
line is a street address, and the line below that is a city/state/ZIP line (or the street line carries it), the
name-shaped line is the addressee. Organisation names ('Harborline Credit Union', 'PO Box ...') are excluded by the
same org hints the other name cues use. Proposals record meta['why'] = 'address-block'.
"""
from __future__ import annotations

import re

from ..review import STREET_ADDRESS
from .detect import LABEL_ORG_HINT, NOT_NAME_TOKENS, Proposal
from .letters import ORGISH

NAME_TOKEN = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*"
NAME_LINE = re.compile(rf"^\s*(?:(?:Mr|Mrs|Ms|Miss|Dr|Mx)\.?\s+)?({NAME_TOKEN}(?:\s+(?:[A-Z]\.|{NAME_TOKEN})){{1,3}})\s*$")
CITY_LINE = re.compile(r"^\s*[A-Za-z][A-Za-z .'’-]{1,40},?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\s*$")
PO_BOX = re.compile(r"(?i)\bP\.?\s*O\.?\s*Box\b")
NOT_NAMES = {t.upper() for t in NOT_NAME_TOKENS} | {
    "STATEMENT", "ACCOUNT", "SUMMARY", "MONTHLY", "PERIOD", "PAYMENT", "BALANCE", "CUSTOMER", "MEMBER", "SERVICES",
    "REWARDS", "PLATINUM", "VISA", "MASTERCARD", "CARD", "NOTICE", "INVOICE", "RECEIPT", "PATIENT", "INTAKE", "CLAIM",
    "EXPLANATION", "BENEFITS", "TOTAL", "DATE", "AMOUNT", "DESCRIPTION", "RE", "SUBJECT", "TO", "FROM", "ATTN"}


def _y(line):
    q = line.chars[0].quad if line.chars else None
    return (q[0][1] + q[3][1]) / 2 if q else None


def _ordered(page):
    lines = [l for l in page.lines if l.text.strip() and l.source != "json"]
    ys = [_y(l) for l in lines]
    if all(y is not None for y in ys):
        x = lambda l: l.chars[0].quad[0][0]
        return sorted(lines, key=lambda l: (round(_y(l) / 3), x(l)))
    return lines


def _name_ok(text):
    toks = [t.strip(".,").upper() for t in text.split()]
    if any(t in NOT_NAMES for t in toks):
        return False
    if ORGISH.search(text) or LABEL_ORG_HINT.search(text):
        return False
    return sum(1 for t in toks if len(t) >= 2) >= 2


def _below(ls, i):
    """The lines under ls[i] in the same column (left edges aligned), nearest first: statements print the address
    block next to a column of account details, so the next line in reading order is often the other column."""
    l = ls[i]
    if not l.chars:
        return ls[i + 1:i + 3]
    x0, y0 = l.chars[0].quad[0][0], _y(l)
    h = abs(l.chars[0].quad[3][1] - l.chars[0].quad[0][1]) or 10
    col = [m for m in ls if m.chars and _y(m) > y0 + 0.3 * h and abs(m.chars[0].quad[0][0] - x0) <= 1.5 * h]
    col.sort(key=_y)
    near = [m for m in col if _y(m) - y0 <= 4.5 * h * (1 + col.index(m))]
    return near[:2]


def address_block_proposals(docs):
    out = []
    for d in docs:
        for p in d.pages:
            ls = _ordered(p)
            for i, l in enumerate(ls[:-1]):
                m = NAME_LINE.match(l.text)
                if not m or not _name_ok(m.group(1)):
                    continue
                nxt = _below(ls, i)
                if not nxt:
                    continue
                street = nxt[0].text
                sm = STREET_ADDRESS.search(street)
                if not sm or PO_BOX.search(street) or sm.start() > 2:
                    continue
                has_city = bool(re.search(r",\s*[A-Z]{2}\s+\d{5}", street[sm.start():])) or \
                    (len(nxt) > 1 and bool(CITY_LINE.match(nxt[1].text)))
                if not has_city:
                    continue
                s, e = m.span(1)
                out.append(Proposal(l.id, s, e, "PERSON", l.text[s:e], "address-block", 0.85, None, {"why": "address-block"}))
    return out
