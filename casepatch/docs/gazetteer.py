"""Places and businesses are facts, not people (D4/A7).

A dispute or claim case needs its merchant ('Velocity Electronics'), the city of a transaction ('OAKLAND, CA') and the
product line ('PROTECTION PLAN') kept exactly: they are what the case is about. Name rules see two capitalised words,
or a 'SURNAME, GIVEN' shape, and cannot tell. This module drops person proposals that are

- a place: 'City, ST', 'City, ST 94610', a US state name, or a city that the same case prints as 'City, ST' somewhere
  (a per-case gazetteer, so no city list has to be shipped or kept current);
- a business or product: a word such as Electronics, Coffee, Plan, Medicine in any case, or the merchant descriptor of
  a transaction row (a line with a date and an amount), unless the row is a person-to-person payment (Zelle, Venmo...);

Reviewer additions are never dropped (the reviewer asked for them). Dropped proposals are listed in the run report with
the reason, so a real person whose surname happens to be a city can be added back.
"""
from __future__ import annotations

import re

from .detect import US_STATE

STATE_NAMES = {s.upper() for s in (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana "
    "Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska "
    "Nevada Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah Vermont Virginia Washington Wisconsin Wyoming").split()} | {
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "RHODE ISLAND",
    "SOUTH CAROLINA", "SOUTH DAKOTA", "WEST VIRGINIA", "DISTRICT OF COLUMBIA"}
# names that are also common given names or surnames: never dropped on their own ('Georgia', 'Virginia', 'Washington')
STATE_PERSONS = {"GEORGIA", "VIRGINIA", "WASHINGTON", "CAROLINA", "INDIANA", "MONTANA", "DAKOTA", "NEVADA", "FLORIDA"}
_ST = r"(?:" + "|".join(a + r"\.?" + b + r"\.?" for a, b in US_STATE) + r")"  # upper case only: 'in', 'or', 'me' are words
# state codes that are also credentials after a name ('Anand Okafor, MD', 'PA', 'ND' naturopath, 'MA', 'DC' chiropractor):
# they only make a place with a ZIP after them
CREDENTIAL_STATES = {"MD", "PA", "ND", "MA", "DC", "DE", "ME"}
_ZIP = r"\d{5}(?:-\d{4})?"
_STATE_NAME = r"(?i:" + "|".join(sorted(STATE_NAMES, key=len, reverse=True)) + r")"
PLACE = re.compile(r"^\s*([A-Za-z][A-Za-z .'’-]{1,40}?)(?:,\s*(" + _ST + r"|" + _STATE_NAME + r")(?:\s+(" + _ZIP + r"))?|\s+(" + _ST + r")\s+(" + _ZIP + r"))\s*$")
# 'Oakland, CA 94610' / 'BROOKLYN NY 11201' / 'Oakland, CA' anywhere in a line: the per-case city list
_CITY = r"([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){0,2})"
CITY_IN_LINE = re.compile(r"(?:^|(?<=[\s,|]))" + _CITY + r"(?:,\s*(" + _ST + r")(?:\s+" + _ZIP + r")?|\s+(" + _ST + r")\s+" + _ZIP + r")(?=$|[\s,|.])")
# trade words that end a shop or service name ('MARLOWE CYCLES', 'Fenwick Florists'); none of them is a common given
# name or surname, so a capitalised name run ending in one is a business, not a person. Shared with verify (part rescans).
TRADE_WORDS = ("Cycles?", "Cyclery", "Bikes", "Bicycles", "Hardware", "Supply", "Supplies", "Lumber", "Audio", "Outfitters",
               "Florists?", "Flowers", "Floral", "Tailors", "Cleaners", "Laundry", "Laundromat", "Jewel(?:l)?ers", "Jewelry",
               "Optical", "Opticians", "Eyewear", "Plumbing", "Roofing", "Electrical", "Garage", "Tavern", "Pub", "Brewing",
               "Brewery", "Distillery", "Winery", "Wines", "Vineyards?", "Cellars", "Delicatessen", "Creamery", "Bagels",
               "Donuts", "Taqueria", "Eatery", "Bakeshop", "Barbershop", "Bookstore", "Bookshop", "Furniture", "Appliances",
               "Pets", "Toys", "Sporting", "Sports", "Hobbies", "Crafts", "Antiques", "Imports", "Printing", "Repairs?",
               "Locksmith", "Pawn", "Liquors?", "Nursery", "Landscaping", "Movers", "Storage", "Dispensary", "Thrift", "Carwash")
BUSINESS = re.compile(r"(?i)\b(?:" + "|".join(TRADE_WORDS) + r"|Electronics|Coffee|Cafe|Café|Bakery|Grill|Kitchen|Pizza|Burger|Restaurant|Bistro|Diner|"
                      r"Market|Mart|Supermarket|Grocery|Groceries|Foods?|Wholesale|Depot|Outlet|Store|Stores|Shop|Boutique|"
                      r"Pharmacy|Drugstore|Clinic|Hospital|Medical|Medicine|Health|Healthcare|Dental|Labs?|Laboratory|Imaging|Urgent|"
                      r"Bank|Banking|Credit|Union|Savings|Financial|Insurance|Mutual|Assurance|Capital|Lending|"
                      r"Airlines?|Airways|Hotel|Hotels|Inn|Suites|Resort|Motel|Rentals?|Travel|Tours|"
                      r"Motors|Auto|Automotive|Tire|Gasoline|Fuel|Station|Parking|Transit|Taxi|Rideshare|"
                      r"Wireless|Mobile|Telecom|Internet|Cable|Energy|Electric|Utilities|"
                      r"Fitness|Gym|Salon|Spa|Studio|Cinema|Theatre|Theater|Books|Music|Games|Gaming|Streaming|Software|"
                      r"Services?|Solutions|Systems|Technologies|Tech|Digital|Online|Subscriptions?|Membership|"
                      r"Plan|Plans|Protection|Warranty|Coverage|Premium|Rewards|Giftcard|Visa|Mastercard|Amex|Discover|"
                      r"Inc|LLC|Ltd|Corp|Corporation|Company|Group|Holdings|Partners|Associates|Enterprises?|Industries|"
                      r"Center|Centre|Family\s+Medicine|Family\s+Practice|Pediatrics|Associates|Department|University|College|"
                      r"School|Church|Foundation|Trust|Fund|Agency|Office|Federal|County|City|State|National|International|Global)\b")
TXN_DATE = re.compile(r"(?:^|\s)(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2})(?=\s)",
                      re.I)
TXN_AMOUNT = re.compile(r"(?:^|\s)-?\$?\d{1,3}(?:,\d{3})*\.\d{2}(?:\s*(?:CR|DR))?\s*$")
P2P = re.compile(r"(?i)\b(?:zelle|venmo|paypal|cash\s?app|transfer|xfer|payment\s+(?:to|from)|sent\s+to|from|to)\b")
PERSON_KINDS = ("PERSON", "GIVEN", "SURNAME")


def case_places(docs):
    """City names this case prints as 'City, ST' (any document): upper-cased."""
    out = set()
    for d in docs:
        for p in d.pages:
            for l in p.lines:
                for m in CITY_IN_LINE.finditer(l.text):
                    st = (m.group(2) or m.group(3) or "").replace(".", "").upper()
                    if st in CREDENTIAL_STATES and m.group(3) is None and not re.search(_ZIP + r"\s*$", m.group()):
                        continue  # 'Anand Okafor, MD' is a doctor, not a city
                    c = re.sub(r"\s+", " ", m.group(1)).strip(" .").upper()
                    if c and c not in STATE_NAMES:
                        out.add(c)
    return out


def merchant_rows(docs):
    """line_id -> True for transaction rows: a date, a descriptor, an amount at the end, not a person-to-person payment."""
    out = {}
    for d in docs:
        for p in d.pages:
            for l in p.lines:
                t = l.text
                if TXN_DATE.search(t) and TXN_AMOUNT.search(t) and not P2P.search(t):
                    out[l.id] = True
    return out


# a merchant named in a label or a purchase phrase ('Merchant: Delphine Lumen', 'the charge of $54.20 at Theo Marlow',
# 'Purchase from Corwin Hollow', 'my order with Sadie Rose'): two to four capitalised words
_MNAME = r"([A-Z][A-Za-z'’&.-]*(?:\s+(?:&\s+)?[A-Z][A-Za-z'’&.-]*){1,3})"
MERCHANT_LABEL = re.compile(r"(?i:\b(?:merchant|merchant\s+name|merchant\s+descriptor|retailer|seller|vendor|store\s+name)\s*:)"
                            r"[ \t]*" + _MNAME)
MERCHANT_PHRASE = re.compile(r"(?i:\b(?:charges?|purchases?|transactions?|orders?|subscriptions?)"
                             r"(?:\s+of\s+\$?\d[\d,]*(?:\.\d{2})?)?\s+(?:at|from|with))\s+" + _MNAME)
_MTRAIL = re.compile(r"(?i)\s+(?:on|in|for|of|and|was|is|has|had|were|are|to|by|at)\b.*$")


def _norm_name(t):
    t = re.sub(r"[’]", "'", re.sub(r"\s+", " ", t)).strip(" .,;:")
    return re.sub(r"'s$", "", t).upper()


def merchant_texts(docs, rows=None):
    """Merchant names as the case prints them: the whole descriptor of a transaction row (a date, the descriptor, an
    amount; not a person-to-person payment), a 'Merchant:' / 'Seller:' label value, or the name after 'charge/purchase/
    order ... at/from/with'. Two or more words; names a person label introduces elsewhere in the case are left out."""
    rows = merchant_rows(docs) if rows is None else rows
    out = set()
    for d in docs:
        for p in d.pages:
            for l in p.lines:
                t = l.text
                if rows.get(l.id):
                    dm = list(TXN_DATE.finditer(t))
                    am = TXN_AMOUNT.search(t)
                    if dm and am and dm[-1].end() < am.start():
                        desc = re.sub(r"\s+", " ", t[dm[-1].end():am.start()]).strip(" \t-|")
                        if 1 <= len(desc.split()) <= 5 and re.search(r"[A-Za-z]", desc) and \
                                all(re.match(r"[A-Z0-9&#*'’]", w) for w in desc.split()):  # not prose ('Kellerman paid')
                            out.add(desc)
                for rx in (MERCHANT_LABEL, MERCHANT_PHRASE):
                    for m in rx.finditer(t):
                        out.add(_MTRAIL.sub("", m.group(1)).strip(" .,;:"))
    strong = _strong_person_texts(docs)
    strong_words = {w for x in strong if len(x.split()) >= 2 for w in re.split(r"[\s-]+", x.split()[-1]) if len(w) > 2}
    # a name a person label introduces, or a text holding that person's surname, is never a merchant here
    return {x for x in out if len(x.split()) >= 2 and _norm_name(x) not in strong
            and not strong_words & set(re.split(r"[\s-]+", _norm_name(x)))}


def case_merchants(docs, rows=None):
    """Upper-cased merchant names of the case (merchant_texts), used to keep a merchant named like a person
    ('Corwin Hollow', 'Rosalind Thistle') as a fact."""
    return {_norm_name(x) for x in merchant_texts(docs, rows)}


def _strong_person_texts(docs):
    """Names a person label or cue introduces somewhere in the case ('Cardholder: ...', 'Name: ...', 'Signed by ...'):
    never treated as a merchant even when the same name appears on a transaction row."""
    from .detect import NAME_CUE, LABELED_NAME
    out = set()
    for d in docs:
        for p in d.pages:
            for l in p.lines:
                for rx in (NAME_CUE, LABELED_NAME):
                    for m in rx.finditer(l.text):
                        out.add(_norm_name(re.sub(r"^(?:Mr|Ms|Mrs|Dr|Ofc)\.?\s+", "", m.group(1))))
    return out


def _given_first(t):
    from .legal import given_names
    toks = t.replace(",", " ").split()
    return bool(toks) and toks[0].upper() in given_names()


def why_not_person(text, places, line_is_txn=False, merchants=()):
    """Reason string when `text` (a proposed person) is a place or a business, else None. `merchants`: upper-cased
    names this case prints as a merchant (case_merchants)."""
    t = re.sub(r"\s+", " ", text).strip(" .,")
    u = t.upper()
    m = PLACE.match(t)
    if m:
        st, zip1, zip2 = m.group(2), m.group(3), m.group(5)
        if zip1 or zip2 or (st or "").replace(".", "").upper() not in CREDENTIAL_STATES:
            return "place:city-state"
    if u in STATE_NAMES and u not in STATE_PERSONS:
        return "place:state"
    if u in places:
        return "place:case-city"
    toks = u.replace(",", " ").split()
    if len(toks) >= 2 and toks[-1] in places and not _given_first(t):  # 'BLUE BOTTLE COFFEE OAKLAND': a descriptor ending in a case city
        return "place:descriptor-city"
    if BUSINESS.search(t):
        return "business:word"
    if line_is_txn and not _given_first(t):
        return "business:transaction-row"
    if merchants and _norm_name(t) in merchants:
        return "business:case-merchant"
    return None


def filter_places_and_businesses(docs, proposals):
    """Drop person proposals that are places or businesses. Returns (kept, dropped) where dropped lists
    {text, kind, detector, reason} (no line ids: the report may be shared)."""
    places = case_places(docs)
    rows = merchant_rows(docs)
    merchants = case_merchants(docs, rows)
    kept, dropped = [], []
    for pr in proposals:
        if pr.kind in PERSON_KINDS and pr.detector != "reviewer":
            r = why_not_person(pr.text, places, rows.get(pr.line_id, False), merchants)
            if r:
                dropped.append({"text": pr.text, "kind": pr.kind, "detector": pr.detector, "reason": r})
                continue
        kept.append(pr)
    return kept, dropped
