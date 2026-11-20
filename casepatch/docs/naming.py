"""Output file names that do not carry a person's name (D3g/D14e).

Callers that keep a source-like output name ('statement_01_okoye.pdf') pass the case's replacements here: every token of
the name that matches a token of an original personal value is swapped for the matching token of its replacement
('statement_01_lind.pdf'), so names stay readable and consistent with the files' contents. A token with no counterpart
is dropped; if nothing readable is left the name falls back to 'document-NN'. Pure function, no I/O."""
from __future__ import annotations

import re
from pathlib import PurePath

_SPLIT = re.compile(r"([^A-Za-z0-9]+)")
_PERSONAL_KINDS = {"PERSON", "GIVEN", "SURNAME", "EMAIL", "ADDRESS", "ACCOUNT", "ID", "MRN", "MEMBER_ID", "PHONE", "SSN",
                   "SSN4", "DOB", "CARD", "CARD_LAST4", "POLICY", "CLAIM", "LICENSE", "PASSPORT", "PLATE", "VIN", "IBAN"}


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", s or "") if t]


def token_map(values) -> dict:
    """{original token (lower): replacement token or ''} from (original, replacement, kind) triples. Tokens of 3+
    characters only; amounts/dates are facts and are not passed in by callers."""
    out = {}
    for orig, repl, kind in values:
        if kind and kind.upper() not in _PERSONAL_KINDS:
            continue
        ot, rt = _tokens(orig), _tokens(repl or "")
        for i, t in enumerate(ot):
            if len(t) < 3 or t.isdigit() and len(t) < 4:
                continue
            r = rt[i] if len(rt) == len(ot) else (rt[-1] if rt and i == len(ot) - 1 else "")
            if r.lower() == t.lower():  # shared word ('example', 'ave'): not personal
                continue
            out.setdefault(t.lower(), r)
    return out


def neutral_name(source_name: str, values, index: int = 1) -> str:
    """`values`: iterable of (original, replacement, kind). Returns a file name with personal tokens replaced."""
    p = PurePath(source_name)
    stem, ext = p.stem, p.suffix.lower()
    tm = token_map(values)
    parts = _SPLIT.split(stem)
    out, changed = [], False
    for part in parts:
        key = part.lower()
        if part and not _SPLIT.fullmatch(part) and key in tm:
            r = tm[key]
            changed = True
            if r:
                out.append(r.lower() if part.islower() else r.upper() if part.isupper() else r.capitalize())
            continue
        out.append(part)
    name = re.sub(r"([^A-Za-z0-9])\1+", r"\1", "".join(out)).strip("_-. ")
    if not changed:
        return stem + ext
    if not re.search(r"[A-Za-z]{2,}", name):
        return f"document-{index:02d}{ext}"
    return name + ext


def has_personal_token(source_name: str, values) -> bool:
    return neutral_name(source_name, values) != PurePath(source_name).stem + PurePath(source_name).suffix.lower()
