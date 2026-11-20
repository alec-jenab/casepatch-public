"""Machine-readable zones (ICAO 9303 TD1/TD2/TD3): detect the lines, rewrite them from the case's fictional values
with valid check digits, at the original length. A name/number replaced in the visual zone but left in the MRZ is a
leak; a MRZ with a broken check digit is an obvious tell. Expiry, nationality, sex and document type are kept."""
from __future__ import annotations

import datetime as dt
import re

MRZ_LINE = re.compile(r"^[A-Z0-9<]{28,46}$")
_W = (7, 3, 1)


def check_digit(s):
    tot = 0
    for i, ch in enumerate(s):
        v = int(ch) if ch.isdigit() else ord(ch) - 55 if "A" <= ch <= "Z" else 0
        tot += v * _W[i % 3]
    return str(tot % 10)


def is_mrz(text):
    t = re.sub(r"\s+", "", text)
    return bool(MRZ_LINE.match(t)) and t.count("<") >= 2 and (t.count("<") >= 4 or re.search(r"\d{6}", t))


def _name_field(cmap, field, n):
    """'SURNAME<<GIVEN<NAMES<<<' -> fictional equivalent padded with '<' to n."""
    sur, _, giv = field.partition("<<")
    sur_t = [w for w in sur.split("<") if w]
    giv_t = [w for w in giv.split("<") if w]
    fs = [re.sub(r"[^A-Z]", "", cmap.replace("SURNAME", w).upper()) for w in sur_t]
    fg = [re.sub(r"[^A-Z]", "", cmap.replace("GIVEN", w).upper()) for w in giv_t]
    out = "<".join(x for x in fs if x)
    if fg:
        out += "<<" + "<".join(x for x in fg if x)
    return (out + "<" * n)[:n]


def _yymmdd_to_date(s, past=True):
    try:
        yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    except ValueError:
        return None
    y = 2000 + yy if (not past or 2000 + yy <= dt.date.today().year) else 1900 + yy
    try:
        return dt.date(y, mm, dd)
    except ValueError:
        return None


def _fake_dob6(cmap, s):
    d = _yymmdd_to_date(s)
    if d is None:
        return cmap.digits(s, "mrz-dob")
    # the same fictional date as the printed date of birth on the card: planned per date value (consistency.py)
    from .consistency import dates
    return dates(cmap).fake_date(d).strftime("%y%m%d")


def _fake_number(cmap, s):
    """Document/personal number: digits mapped like the printed number (same CaseMap key), letters kept."""
    if not re.search(r"\d", s):
        return s
    return cmap.digits(s, "ID")


def rewrite(cmap, text):
    raw = text
    t = re.sub(r"\s+", "", text)
    n = len(t)
    out = None
    if re.match(r"^[PVIACT][A-Z<][A-Z<]{3}", t) and "<<" in t and not re.search(r"\d", t[5:]):
        out = t[:5] + _name_field(cmap, t[5:], n - 5)                    # TD3/TD2 line 1
    elif re.fullmatch(r"[A-Z<]+", t) and "<<" in t:
        out = _name_field(cmap, t, n)                                    # TD1 line 3
    elif n >= 43 and re.match(r"^[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d[MFX<]\d{6}\d", t):
        doc, nat, dob, sex, exp, opt = t[0:9], t[10:13], t[13:19], t[20], t[21:27], t[28:42]
        doc2 = _fake_number(cmap, doc.rstrip("<")).ljust(9, "<")[:9]
        opt2 = _fake_number(cmap, opt) if re.search(r"\d", opt) else opt
        dob2 = _fake_dob6(cmap, dob)
        c_doc, c_dob, c_exp = check_digit(doc2), check_digit(dob2), check_digit(exp)
        c_opt = check_digit(opt2) if opt2.strip("<") else ("<" if t[42] == "<" else "0")
        comp = check_digit(doc2 + c_doc + dob2 + c_dob + exp + c_exp + opt2 + c_opt)
        out = doc2 + c_doc + nat + dob2 + c_dob + sex + exp + c_exp + opt2 + c_opt + comp + t[44:]
    elif n >= 30 and re.match(r"^[A-Z][A-Z<][A-Z<]{3}[A-Z0-9<]{9}\d", t):  # TD1 line 1
        doc = t[5:14]
        doc2 = _fake_number(cmap, doc.rstrip("<")).ljust(9, "<")[:9]
        opt = t[15:30]
        opt2 = _fake_number(cmap, opt) if re.search(r"\d", opt) else opt
        out = t[:5] + doc2 + check_digit(doc2) + opt2 + t[30:]
    elif n >= 30 and re.match(r"^\d{6}\d[MFX<]\d{6}\d", t):          # TD1 line 2 (composite left as is when unknown)
        dob2 = _fake_dob6(cmap, t[0:6])
        out = dob2 + check_digit(dob2) + t[7:]
    if out is None:
        out = re.sub(r"\d+", lambda m: cmap.digits(m.group(), "mrz"), t)
    out = (out + "<" * n)[:n]
    if out == t:  # never identical to the source
        out = re.sub(r"\d", lambda m: str((int(m.group()) + 1) % 10), out, count=1)
    # keep the OCR line's own spacing (rare: OCR split the MRZ into words)
    if raw != t:
        it = iter(out)
        out = "".join(ch if ch.isspace() else next(it, "<") for ch in raw)
    return out
