"""Keep list: terms that must never be replaced (your own bank or merchant name, a product, a city, a support address).

Matching is case-insensitive and by whole phrase: a detected value that equals a kept term, or is a whole-word part of
one ("Acme" for the kept term "Acme Bank"), is left unchanged everywhere in the case, is not listed as a change, and is
not reported as a residual value by the verifier. A detected value that is LONGER than a kept term (a full address that
contains a kept city) is still replaced as a whole."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

MAX_TERMS = 200
MAX_CHARS = 200

KeepInput = Union[None, str, Iterable[str]]


class KeepListError(ValueError):
    """A keep list that cannot be used (too many terms, a term too long, control characters, wrong type)."""


def _norm(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip().casefold()
    return re.sub(r"^[^\w@]+|[^\w]+$", "", t)  # edge punctuation ('Acme Bank,' or '"Acme"') does not change the term


def parse_keep(value: KeepInput, max_terms: int = MAX_TERMS, max_chars: int = MAX_CHARS) -> list[str]:
    """A keep list from a list of strings or one newline-separated string. Blank lines are ignored and duplicates
    (ignoring case) dropped; raises KeepListError on anything else that is unusable."""
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray)):
        raise KeepListError("keep must be text, not bytes.")
    items = [value] if isinstance(value, str) else value
    try:
        items = list(items)
    except TypeError:
        raise KeepListError("keep must be a list of strings or a newline-separated string.") from None
    out, seen = [], set()
    for item in items:
        if not isinstance(item, str):
            raise KeepListError("keep must be a list of strings or a newline-separated string.")
        for line in item.splitlines():
            term = re.sub(r"\s+", " ", line).strip()
            if not term:
                continue
            if any(ord(c) < 32 or ord(c) == 127 for c in term):
                raise KeepListError("keep terms cannot contain control characters.")
            if len(term) > max_chars:
                raise KeepListError(f"Each keep term can be at most {max_chars} characters (one is {len(term)}).")
            n = _norm(term)
            if not n:
                raise KeepListError("keep terms must contain at least one letter or digit.")
            if n in seen:
                continue
            seen.add(n)
            out.append(term)
            if len(out) > max_terms:
                raise KeepListError(f"keep can have at most {max_terms} terms.")
    return out


def read_keep_file(path: Union[str, Path]) -> list[str]:
    """One term per line (UTF-8); blank lines and lines starting with '#' are ignored."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise KeepListError(f"{path}: the keep file must be UTF-8 text.") from None
    return [l for l in raw.splitlines() if not l.lstrip().startswith("#")]


class KeepList:
    """Compiled keep list. `covers(text)`: True when `text` equals a kept term or is a whole-word part of one."""

    def __init__(self, terms: KeepInput = None):
        self.terms = parse_keep(terms)
        self._n = [_norm(t) for t in self.terms]

    def __bool__(self) -> bool:
        return bool(self._n)

    def __len__(self) -> int:
        return len(self._n)

    def covers(self, text: Optional[str]) -> bool:
        if not self._n or not text:
            return False
        n = _norm(text)
        if not n:
            return False
        rx = re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)")
        return any(n == k or rx.search(k) for k in self._n)


def as_keep_list(keep) -> KeepList:
    return keep if isinstance(keep, KeepList) else KeepList(keep)
