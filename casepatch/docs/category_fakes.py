"""Format- and checksum-preserving fakes for the regulated categories (HIPAA / PCI DSS / GLBA / GDPR).

Every generator is a pure function of (case key, kind, source text): the same value maps to the same fake across every
document and trace in a case, and different cases get different fakes. Fakes are *valid-looking* (they pass the public
checksum or structure rule a validator would apply) but are generated, never looked up, and never equal the source.

Checksums implemented here (all public algorithms):
- Luhn (cards, IMEI) and NPI (Luhn over the '80840' prefix + 9 digits, per CMS NPI check-digit rule)
- ABA routing number (3-7-1 weights, mod 10)
- IBAN (ISO 13616 mod-97 = 1)
- VIN (ISO 3779 / 49 CFR 565 position-9 check digit, weights 8765432X098765432)
- DEA registration number (sum rule)
- Medicare Beneficiary Identifier (CMS MBI character-class layout; no check digit exists)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import random
import re
import secrets

# ------------------------------------------------------------------ checksums / validators

def luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in digits]
    tot = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            d -= 9 if d > 9 else 0
        tot += d
    return len(ds) > 1 and tot % 10 == 0


def luhn_digit(payload: str) -> str:
    for c in "0123456789":
        if luhn_ok(payload + c):
            return c
    raise ValueError("LUHN")


def npi_ok(npi: str) -> bool:
    return bool(re.fullmatch(r"[12]\d{9}", npi)) and luhn_ok("80840" + npi)


def aba_ok(r: str) -> bool:
    if not re.fullmatch(r"\d{9}", r):
        return False
    d = [int(c) for c in r]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


def aba_digit(first8: str) -> str:
    for c in "0123456789":
        if aba_ok(first8 + c):
            return c
    raise ValueError("ABA")


# IBAN lengths by country (ISO 13616 registry, common SEPA + GB/CH/NO etc.)
IBAN_LEN = {"AD": 24, "AT": 20, "BE": 16, "BG": 22, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "EE": 20, "ES": 24,
            "FI": 18, "FR": 27, "GB": 22, "GI": 23, "GR": 27, "HR": 21, "HU": 28, "IE": 22, "IS": 26, "IT": 27, "LI": 21,
            "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MT": 31, "NL": 18, "NO": 15, "PL": 28, "PT": 25, "RO": 24, "SE": 24,
            "SI": 19, "SK": 24, "SM": 27, "TR": 26, "AE": 23, "SA": 24, "IL": 23, "BR": 29, "UA": 29}
# characters of the BBAN that identify the BANK (an organisation: evidence, kept), per country
IBAN_BANK_CHARS = {"DE": 8, "GB": 4, "IE": 4, "FR": 5, "ES": 4, "IT": 6, "NL": 4, "BE": 3, "CH": 5, "AT": 5, "PT": 4, "PL": 3}


def _iban_num(s: str) -> int:
    return int("".join(str(int(c, 36)) for c in s))


def iban_ok(iban: str) -> bool:
    s = re.sub(r"\s", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s) or IBAN_LEN.get(s[:2], len(s)) != len(s):
        return False
    return _iban_num(s[4:] + s[:4]) % 97 == 1


def iban_check(country: str, bban: str) -> str:
    return "%02d" % (98 - _iban_num(bban + country + "00") % 97)


VIN_MAP = {**{str(i): i for i in range(10)}, **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMN", range(1, 6))),
           "P": 7, "R": 9, **dict(zip("STUVWXYZ", range(2, 10)))}
VIN_W = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def vin_digit(vin: str) -> str:
    s = sum(VIN_MAP[c] * w for c, w in zip(vin.upper(), VIN_W)) % 11
    return "X" if s == 10 else str(s)


def vin_ok(vin: str) -> bool:
    v = vin.upper()
    return bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", v)) and vin_digit(v) == v[8]


def dea_ok(dea: str) -> bool:
    m = re.fullmatch(r"[A-Z][A-Z9](\d{7})", dea.upper())
    if not m:
        return False
    d = [int(c) for c in m.group(1)]
    return (d[0] + d[2] + d[4] + 2 * (d[1] + d[3] + d[5])) % 10 == d[6]


MBI_ALPHA = "ACDEFGHJKMNPQRTUVWXY"   # CMS MBI: letters exclude S, L, O, I, B, Z
MBI_LAYOUT = "CAN" "NAN" "NAA" "NN"  # C=1-9, A=MBI letter, N=0-9 ; positions 3 and 6 are alphanumeric
MBI_RX = re.compile(r"(?<![A-Z0-9])[1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d-?[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d-?"
                    r"[AC-HJKMNP-RT-Y]{2}\d{2}(?![A-Z0-9])")


def mbi_ok(text: str) -> bool:
    return bool(MBI_RX.fullmatch(text.upper()))


def ssn_ok(ssn: str) -> bool:
    d = re.sub(r"\D", "", ssn)
    if len(d) != 9:
        return False
    a, g, s = d[:3], d[3:5], d[5:]
    return a not in ("000", "666") and a[0] != "9" and g != "00" and s != "0000"


ITIN_GROUPS = [*range(50, 66), *range(70, 89), 90, 91, 92, *range(94, 100)]


def itin_ok(t: str) -> bool:
    d = re.sub(r"\D", "", t)
    return len(d) == 9 and d[0] == "9" and int(d[3:5]) in ITIN_GROUPS


EIN_PREFIXES = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, *range(30, 49), *range(50, 69),
                71, 72, 73, 74, 75, 76, 77, *range(80, 89), *range(90, 96), 98, 99]


def ein_ok(t: str) -> bool:
    d = re.sub(r"\D", "", t)
    return len(d) == 9 and int(d[:2]) in EIN_PREFIXES


# ------------------------------------------------------------------ generator

def _relayout(template: str, chars: str) -> str:
    """Write `chars` into the alphanumeric positions of `template`, keeping separators and spacing."""
    it = iter(chars)
    return "".join(next(it) if c.isalnum() else c for c in template)


def _case_like(src: str, out: str) -> str:
    if src.isupper():
        return out.upper()
    if src.islower():
        return out.lower()
    if src[:1].isupper():
        return out[:1].upper() + out[1:]
    return out


class CategoryFaker:
    """Keyed, case-consistent fakes. Pass the case's resolve.CaseMap (shares its key, org/person pools) or a raw key."""

    def __init__(self, cmap=None, key: bytes | None = None):
        self.cmap = cmap
        self.key = cmap.key if cmap is not None else (key or secrets.token_bytes(32))
        self._cache = {}

    def rng(self, *parts) -> random.Random:
        h = hmac.new(self.key, "\x1f".join(("category",) + parts).encode(), hashlib.sha256).hexdigest()
        return random.Random(int(h, 16))

    # -- generic shape-preserving replacement (digits->digits, letters->letters, separators kept)
    def shape(self, text: str, kind: str, keep_prefix: int = 0, letters=None) -> str:
        src_al = [c for c in text if c.isalnum()]
        for attempt in range(40):
            r = self.rng("shape", kind, text, str(attempt))
            out, n = [], 0
            for c in text:
                if not c.isalnum():
                    out.append(c); continue
                if n < keep_prefix:
                    out.append(c)
                elif c.isdigit():
                    out.append(str(r.randrange(10)))
                else:
                    pool = letters or "ABCDEFGHJKLMNPRSTUVWXYZ"
                    ch = r.choice(pool)
                    out.append(ch if c.isupper() else ch.lower())
                n += 1
            res = "".join(out)
            new_al = [c for c in res if c.isalnum()]
            core = "".join(src_al[keep_prefix:])
            windows = {core[i:i + 4] for i in range(max(0, len(core) - 3))}
            new_core = "".join(new_al[keep_prefix:])
            if res != text and not (len(core) >= 6 and any(new_core[i:i + 4] in windows for i in range(len(new_core) - 3))):
                return res
        return res

    # -- national / tax identifiers
    def ssn(self, text):
        d = re.sub(r"\D", "", text)
        for i in range(50):
            r = self.rng("ssn", d, str(i))
            area = r.choice([a for a in range(1, 900) if a != 666])
            new = "%03d%02d%04d" % (area, r.randrange(1, 100), r.randrange(1, 10000))
            if new != d and not any(new[j:j + 4] in d for j in range(6)):
                break
        return _relayout(text, new) if len(re.sub(r"\W", "", text)) == 9 else new[:3] + "-" + new[3:5] + "-" + new[5:]

    def tin(self, text):
        d = re.sub(r"\D", "", text)
        r = self.rng("tin", d)
        if itin_ok(d):
            new = "9%02d%02d%04d" % (r.randrange(100), r.choice(ITIN_GROUPS), r.randrange(10000))
        elif re.fullmatch(r"\d{2}-\d{7}", text.strip()) or not ssn_ok(d):
            new = "%02d%07d" % (r.choice(EIN_PREFIXES), r.randrange(10 ** 7))
        else:
            return self.ssn(text)
        if new == d:
            new = new[:-1] + str((int(new[-1]) + 1) % 10)
        return _relayout(text, new)

    # -- financial
    def iban(self, text):
        s = re.sub(r"\s", "", text).upper()
        cc, bban = s[:2], s[4:]
        keep = IBAN_BANK_CHARS.get(cc, 4)
        for i in range(30):
            body = self.shape(bban, "iban%d" % i, keep_prefix=keep)
            out = cc + iban_check(cc, body) + body
            if out != s:
                break
        # keep the source's grouping (print format groups of 4, or electronic run-together)
        if " " in text.strip():
            return " ".join(out[i:i + 4] for i in range(0, len(out), 4))
        return out

    def routing(self, text):
        d = re.sub(r"\D", "", text)
        r = self.rng("aba", d)
        for i in range(30):
            first8 = d[:2] + "".join(str(r.randrange(10)) for _ in range(6))  # keep Federal Reserve district prefix
            new = first8 + aba_digit(first8)
            if new != d:
                break
        return _relayout(text, new)

    def card_expiry(self, text):
        m = re.fullmatch(r"\s*(\d{1,2})(\s?[/-]\s?)(\d{2}|\d{4})\s*", text)
        if not m:
            return self.shape(text, "exp")
        mm, sep, yy = int(m.group(1)), m.group(2), m.group(3)
        r = self.rng("exp", text)
        shift = r.choice([-1, 1]) * r.randrange(3, 20)  # months; keeps the card plausibly current/expired as before
        y = int(yy) if len(yy) == 4 else 2000 + int(yy)
        t = y * 12 + (mm - 1) + shift
        ny, nm = divmod(t, 12)
        yy_out = str(ny) if len(yy) == 4 else "%02d" % (ny % 100)
        mm_out = ("%02d" if len(m.group(1)) == 2 else "%d") % (nm + 1)
        return text.replace(m.group(0).strip(), mm_out + sep + yy_out)

    def sad_digits(self, text, kind):
        """Sensitive authentication data (CVV/CVC, PIN, track data): replaced with fresh random digits, not a keyed
        mapping, and never cached, so nothing derived from the original persists anywhere."""
        out = []
        for c in text:
            out.append(str(secrets.randbelow(10)) if c.isdigit() else c)
        res = "".join(out)
        return res if res != text else self.sad_digits(text, kind)

    # -- health identifiers
    def npi(self, text):
        d = re.sub(r"\D", "", text)
        r = self.rng("npi", d)
        for i in range(30):
            body = (d[0] if d[:1] in ("1", "2") else "1") + "".join(str(r.randrange(10)) for _ in range(8))
            new = body + luhn_digit("80840" + body)
            if new != d:
                break
        return _relayout(text, new)

    def dea(self, text):
        t = text.strip()
        r = self.rng("dea", t)
        first = r.choice("ABFGMPR")  # registrant-type letters
        d = [r.randrange(10) for _ in range(6)]
        d.append((d[0] + d[2] + d[4] + 2 * (d[1] + d[3] + d[5])) % 10)
        second = r.choice([c for c in "ABCDEFGHJKLMNPRSTUVWY" if c != t[1].upper()])  # registrant surname initial: not kept
        return text.replace(t, first + second + "".join(map(str, d)))

    def mbi(self, text):
        src = re.sub(r"-", "", text.upper())
        for i in range(30):
            r = self.rng("mbi", src, str(i))
            out = []
            for k, cls in enumerate(MBI_LAYOUT):
                if k in (2, 5):
                    out.append(r.choice(MBI_ALPHA + "0123456789"))
                elif cls == "C":
                    out.append(str(r.randrange(1, 10)))
                elif cls == "A":
                    out.append(r.choice(MBI_ALPHA))
                else:
                    out.append(str(r.randrange(10)))
            new = "".join(out)
            if new != src:
                break
        return _relayout(text, new) if len(re.sub(r"\W", "", text)) == 11 else new

    def vin(self, text):
        v = text.strip().upper()
        if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", v):
            return self.shape(text, "vin")
        # keep WMI (1-3), vehicle descriptor (4-8) and model year (10): make/model/year are case evidence;
        # replace plant (11) + serial (12-17), recompute the check digit
        r = self.rng("vin", v)
        for i in range(30):
            tail = r.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") + "".join(str(r.randrange(10)) for _ in range(6))
            cand = v[:8] + "0" + v[9] + tail
            cand = cand[:8] + vin_digit(cand) + cand[9:]
            if cand != v:
                break
        return text.replace(text.strip(), cand if text.strip().isupper() or text.strip().isdigit() else cand.lower())

    def imei(self, text):
        d = re.sub(r"\D", "", text)
        r = self.rng("imei", d)
        body = d[:8] + "".join(str(r.randrange(10)) for _ in range(6))  # keep TAC (device model), new serial
        new = body + luhn_digit(body)
        return _relayout(text, new)

    # -- network
    def ip(self, text):
        t = text.strip()
        r = self.rng("ip", t)
        if ":" in t:  # IPv6 -> documentation prefix 2001:db8::/32 (RFC 3849)
            return text.replace(t, "2001:db8:%x:%x::%x" % (r.randrange(65536), r.randrange(65536), r.randrange(1, 65536)))
        o = [int(x) for x in t.split(".")]
        if o[0] == 10:
            new = [10, r.randrange(256), r.randrange(256), r.randrange(1, 255)]
        elif o[0] == 192 and o[1] == 168:
            new = [192, 168, r.randrange(256), r.randrange(1, 255)]
        elif o[0] == 172 and 16 <= o[1] <= 31:
            new = [172, r.randrange(16, 32), r.randrange(256), r.randrange(1, 255)]
        else:  # public -> one of the RFC 5737 documentation ranges (never routable, never someone's address)
            new = list(r.choice([(192, 0, 2), (198, 51, 100), (203, 0, 113)])) + [r.randrange(1, 255)]
        if new == o:
            new[-1] = (new[-1] % 253) + 1
        return text.replace(t, ".".join(map(str, new)))

    def mac(self, text):
        return self.shape(text, "mac", keep_prefix=6, letters="ABCDEF")  # keep OUI (vendor), new NIC part

    def url(self, text):
        m = re.match(r"(?i)((?:https?://)?)((?:www\.)?)([^/\s?#]+)(.*)", text)
        if not m:
            return self.shape(text, "url")
        scheme, www, host, rest = m.groups()
        word = self.cmap.org(host.split(".")[0]).split()[0].lower() if self.cmap is not None else \
            self.rng("urlhost", host).choice(["arvo", "belmar", "corvex", "daltrin", "everline", "fenwick", "halcyon"])
        rest = re.sub(r"[0-9a-fA-F]{6,}|\d{3,}", lambda x: self.shape(x.group(), "urlpath"), rest)
        return f"{scheme}{www}{word}.example.com{rest}"

    # -- location / age
    def geo(self, text):
        def mv(mm):
            v = mm.group()
            dec = len(v.split(".")[1]) if "." in v else 0
            r = self.rng("geo", text, v)
            f = float(v)
            new = int(f) + (abs(f) % 1 + r.uniform(0.15, 0.6)) % 1 * (1 if f >= 0 else -1)  # same whole degree
            return f"{new:.{dec}f}"
        return re.sub(r"[-+]?\d{1,3}\.\d+", mv, text)

    def age_90(self, text):
        m = re.search(r"\d+", text)
        r = self.rng("age", text)
        new = str(r.choice([a for a in range(90, 100) if str(a) != m.group()]))
        return text[:m.start()] + new + text[m.end():]

    # -- category-preserving substitution (health, GDPR special categories)
    def substitute(self, kind, text, vocab_lookup):
        """Replace a term with a DIFFERENT term of the same category group (condition->condition in the same body
        system, drug->drug in the same class). Surface form follows the source (abbreviation stays an abbreviation)."""
        hit = vocab_lookup(kind, text)
        if not hit:
            return self.shape(text, kind)
        entry, form, pool = hit
        cands = [e for e in pool if e is not entry and e.get("group") == entry.get("group")] or [e for e in pool if e is not entry]
        # keyed on the canonical entry (not the surface form or kind): 'Diabetes Mellitus', 'DM' and '250.00' all map to
        # the same substitute condition, so names and codes stay coherent across the case
        r = self.rng("subst", entry.get("group", ""), entry["term"])
        new = cands[r.randrange(len(cands))]
        if form == "abbr":
            out = (new.get("abbr") or ["".join(w[0] for w in new["term"].split()).upper()])[0]
            return out
        if form == "icd10":
            return new.get("icd10") or self.shape(text, kind, keep_prefix=1)
        if form == "icd9":
            return new.get("icd9") or self.shape(text, kind)
        out = new["term"]
        return _case_like(text, out.title() if text.istitle() else out)

    def track(self, text):
        """Magnetic-stripe track data (PCI SAD). PAN and cardholder name follow the case map (same fake card / person as
        everywhere else); expiry, service code and discretionary data get fresh random digits."""
        m = re.fullmatch(r"%B(\d{12,19})\^([^^]*)\^(\d*)\?", text.strip())
        m2 = re.fullmatch(r";(\d{12,19})=(\d*)\?", text.strip())
        def pan(p):
            if self.cmap is not None:
                return self.cmap.card(p)
            body = "".join(str(secrets.randbelow(10)) for _ in range(len(p) - 1))
            return body + luhn_digit(body)
        if m:
            name = m.group(2)
            if self.cmap is not None and "/" in name:
                sur, _, given = name.partition("/")
                fake = self.cmap.person(f"{given.strip().title()} {sur.strip().title()}") or "Alex Morgan"
                g, s_ = fake.split()[0], fake.split()[-1]
                name = f"{s_.upper()}/{g.upper()}".ljust(len(m.group(2)))[:max(len(m.group(2)), 2 + len(s_) + len(g))]
            return text.replace(m.group(0), f"%B{pan(m.group(1))}^{name}^{self.sad_digits(m.group(3), 'track') if m.group(3) else ''}?")
        if m2:
            return text.replace(m2.group(0), f";{pan(m2.group(1))}={self.sad_digits(m2.group(2), 'track') if m2.group(2) else ''}?")
        return self.sad_digits(text, "track")

    def income(self, text):
        """Only used when the user marks an amount non-decision-relevant: scale by a keyed factor, keep formatting."""
        m = re.search(r"\d[\d,]*(?:\.\d{2})?", text)
        if not m:
            return text
        v = float(m.group().replace(",", ""))
        f = self.rng("income", text).uniform(0.8, 1.25)
        nv = round(v * f, 2 if "." in m.group() else 0)
        s = f"{nv:,.2f}" if "." in m.group() else f"{int(nv):,}"
        if "," not in m.group():
            s = s.replace(",", "")
        return text[:m.start()] + s + text[m.end():]
