"""Inline cue detector for receipts: a payment card's last four digits announced by the card brand on the same line
('VISA 8945', 'MASTERCARD **** 1234'). The rules detector's last-4 rule needs a mask or 'ending in'; a receipt prints
the brand and the digits. A year after the word 'card' ('Credit card 2019') is not taken without a mask.
Every proposal is grounded in its line and says which cue found it."""
from __future__ import annotations

import re

from .detect import Proposal

CARD_CUE = re.compile(r"\b(?:VISA|MASTERCARD|MASTER CARD|MC|AMEX|AMERICAN EXPRESS|DISCOVER|MAESTRO|DEBIT|CREDIT|CARD)"
                      r"(?:\s+(?:CARD|NO\.?|#))?\s*[:#]?\s*[*xX•.]*\s*(\d{4})(?![\d/.,:-])", re.I)


def cue_proposals(docs):
    out = []
    for d in docs:
        for p in d.pages:
            for l in p.lines:
                if l.source == "json" or not l.text.strip():
                    continue
                t = l.text
                for m in CARD_CUE.finditer(t):
                    if not re.search(r"(?:19|20)\d\d", m.group(1)) or re.search(r"[*xX•]", m.group()):
                        out.append(Proposal(l.id, m.start(1), m.end(1), "CARD_LAST4", m.group(1), "cues", 0.85, None,
                                            {"why": "cue " + t[m.start():m.start(1)].strip()[:20]}))
    return out
