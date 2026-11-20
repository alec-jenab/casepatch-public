"""Agent traces and attachments as case documents: JSON, JSON Lines and plain UTF-8 text.

One validator for every entry point (local engine, SDK, hosted probe), standard library only, so the same file is
accepted or refused the same way everywhere. Content is never echoed in errors: codes, file positions and limits only.

Formats (by file extension; content must match):
* .json            one JSON value. Rewritten in place: same keys, nesting, arrays and value types; only strings change.
* .jsonl / .ndjson one JSON value per line (blank lines kept). Written back one record per line, same order.
* .txt             UTF-8 text (chat transcripts, notes, emails). Rewritten line by line; line endings kept.

Bounds: TEXT_MAX_BYTES per file, JSONL_MAX_RECORDS records, TEXT_MAX_LINES lines, JSON_MAX_STRINGS string leaves and
JSON_MAX_DEPTH nesting. Hosted accounting: a textual document counts ceil(characters / CHARS_PER_PAGE) pages (at least 1).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

TEXT_EXT = {".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".txt": "text"}
OUT_EXT = {"json": ".json", "jsonl": ".jsonl", "text": ".txt"}
TEXT_MAX_BYTES = 5 * 1024 * 1024
JSONL_MAX_RECORDS = 20_000
TEXT_MAX_LINES = 100_000
JSON_MAX_STRINGS = 100_000
JSON_MAX_DEPTH = 64
CHARS_PER_PAGE = 10_000


class TextualError(ValueError):
    """`code` is stable (API/CLI); `where` is a position such as 'line 12' (never content)."""

    def __init__(self, code: str, where: str = ""):
        super().__init__(code + (f" ({where})" if where else ""))
        self.code, self.where = code, where


def fmt_of(ext: str):
    return TEXT_EXT.get(ext.lower())


def _decode(raw: bytes) -> str:
    if len(raw) > TEXT_MAX_BYTES:
        raise TextualError("TEXT_FILE_TOO_LARGE", f"limit {TEXT_MAX_BYTES // 1048576} MB")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:4096]:
        raise TextualError("TEXT_NOT_UTF8", "UTF-16 or binary data")
    try:
        text = raw.decode("utf-8-sig")  # a UTF-8 byte-order mark is accepted and dropped
    except UnicodeDecodeError as e:
        raise TextualError("TEXT_NOT_UTF8", f"byte {e.start}") from None
    if not text.strip():
        raise TextualError("TEXT_EMPTY")
    return text


def _utf8_ok(s: str, where: str) -> None:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:  # an unpaired \ud800 escape: valid JSON syntax, but not text that can be written back
        raise TextualError("JSON_INVALID_UNICODE", where) from None


def _check_json(value: Any, where: str, budget: list) -> None:
    stack = [(value, 1)]
    while stack:
        v, depth = stack.pop()
        if depth > JSON_MAX_DEPTH:
            raise TextualError("JSON_TOO_DEEP", where)
        if isinstance(v, dict):
            for k in v:
                _utf8_ok(k, where)
            stack.extend((x, depth + 1) for x in v.values())
        elif isinstance(v, list):
            stack.extend((x, depth + 1) for x in v)
        elif isinstance(v, str):
            _utf8_ok(v, where)
            budget[0] += 1
            if budget[0] > JSON_MAX_STRINGS:
                raise TextualError("JSON_TOO_MANY_STRINGS", f"limit {JSON_MAX_STRINGS}")


class _Dup(ValueError):
    pass


def _loads(s: str, where: str):
    """Strict JSON: NaN/Infinity, numbers that overflow to infinity and duplicate keys are refused, because writing them
    back would change the data (a duplicate key would silently lose one of its values)."""
    def no_constants(c):
        raise ValueError(c)

    def finite(x):
        v = float(x)
        if math.isinf(v):
            raise ValueError("overflow")
        if v == 0.0 and re.search(r"[1-9]", re.split(r"[eE]", x)[0]):
            raise ValueError("underflow")  # 1e-400 would be written back as 0.0 (an eval score silently becomes zero)
        return v

    def no_dups(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise _Dup()
            d[k] = v
        return d
    try:
        return json.loads(s, parse_constant=no_constants, parse_float=finite, object_pairs_hook=no_dups)
    except _Dup:
        raise TextualError("JSON_DUPLICATE_KEY", where) from None
    except json.JSONDecodeError as e:  # a position, never the text around it
        raise TextualError("JSON_MALFORMED", where or f"line {e.lineno}") from None
    except (ValueError, RecursionError):
        raise TextualError("JSON_MALFORMED", where) from None


def parse(raw: bytes, fmt: str) -> dict:
    """Validate and parse. Returns {"format", "data", "chars", ...}: data is the JSON value (json), the list of records
    (jsonl; "blank" lists the indexes of blank lines, which are None in data, distinct from a `null` record) or the list
    of lines without their endings (text)."""
    text = _decode(raw)
    budget = [0]
    if fmt == "json":
        data = _loads(text, "")
        _check_json(data, "", budget)
        return {"format": fmt, "data": data, "chars": len(text)}
    if fmt == "jsonl":
        rows = text.split("\n")
        if rows and rows[-1] == "":
            rows.pop()
        records, blank, n = [], [], 0
        for i, row in enumerate(rows, 1):
            row = row.rstrip("\r")
            if not row.strip():
                records.append(None); blank.append(i - 1)  # a blank line, not a JSON null record
                continue
            n += 1
            if n > JSONL_MAX_RECORDS:
                raise TextualError("JSONL_TOO_MANY_RECORDS", f"limit {JSONL_MAX_RECORDS}")
            rec = _loads(row, f"line {i}")
            _check_json(rec, f"line {i}", budget)
            records.append(rec)
        if not n:
            raise TextualError("TEXT_EMPTY")
        return {"format": fmt, "data": records, "blank": blank, "chars": len(text), "crlf": "\r\n" in text}
    if fmt == "text":
        parts = text.split("\n")
        if len(parts) > TEXT_MAX_LINES:
            raise TextualError("TEXT_TOO_MANY_LINES", f"limit {TEXT_MAX_LINES}")
        lines, ends = [], []
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            if p.endswith("\r"):
                lines.append(p[:-1]); ends.append("\r\n")
            else:
                lines.append(p); ends.append("" if last else "\n")
        return {"format": fmt, "data": lines, "ends": ends, "chars": len(text)}
    raise TextualError("UNSUPPORTED_TEXT_FORMAT", fmt)


def serialize(fmt: str, data: Any, meta: dict) -> str:
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if fmt == "jsonl":
        nl = "\r\n" if meta.get("crlf") else "\n"
        blank = set(meta.get("blank") or ())
        return "".join(("" if i in blank else json.dumps(r, ensure_ascii=False)) + nl for i, r in enumerate(data))
    if fmt == "text":
        ends = meta.get("ends") or ["\n"] * (len(data) - 1) + [""]
        return "".join(line + end for line, end in zip(data, ends))
    raise TextualError("UNSUPPORTED_TEXT_FORMAT", fmt)


def pages_for(chars: int) -> int:
    return max(1, math.ceil(chars / CHARS_PER_PAGE))


def probe(raw: bytes, ext: str) -> dict:
    """{"ok": True, "pages", "type", "format"} or {"ok": False, "error": CODE, "where"} (for the sandboxed hosted probe)."""
    fmt = fmt_of(ext)
    if fmt is None:
        return {"ok": False, "error": "TYPE"}
    try:
        p = parse(raw, fmt)
    except TextualError as e:
        return {"ok": False, "error": e.code, "where": e.where}
    return {"ok": True, "pages": pages_for(p["chars"]), "type": "text", "format": fmt}
