"""Offline, explicitly annotated transformation. No learned PII detector is included."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Any


class PolicyError(ValueError):
    """Messages contain policy codes only, never source values."""


@dataclass
class Result:
    candidate: dict | None
    report: dict


PATTERNS = {
    "EMAIL": re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?![\w-])"),
    "SSN": re.compile(r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)"),
    "PAN_LIKE": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "PHONE": re.compile(r"(?<!\d)(?:\+1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "IPV4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "SECRET": re.compile(r"(?i)\b(?:sk-[a-z0-9_-]{12,}|Bearer\s+[a-z0-9._-]{12,})"),
    "ISO_DATE": re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}))?\b"),
}
FORMS = {
    "PERSON": {"full", "initial"}, "EMAIL": {"default"},
    "ID": {"default"}, "ADDRESS": {"default"},
    "PHONE": {"default"}, "CARD": {"default"}, "SSN": {"default"},
}
GIVEN = ("Nora", "Eli", "Mina", "Theo", "Iris", "Owen", "Lena", "Arlo")
FAMILY = ("Chen", "Reed", "Patel", "Brooks", "Kim", "Rivera", "Shah", "Bell")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(key: bytes, *parts: str) -> str:
    return hmac.new(key, canonical(list(parts)), hashlib.sha256).hexdigest()


def walk(value: Any, path=()):
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise PolicyError("NON_STRING_KEY")
            yield from walk(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, path + (i,))
    elif isinstance(value, str):
        yield path, value
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise PolicyError("UNSUPPORTED_VALUE")


def get(value, path):
    if not isinstance(path, list) or not path:
        raise PolicyError("INVALID_PATH")
    try:
        for p in path:
            if not isinstance(p, (str, int)) or isinstance(p, bool):
                raise PolicyError("INVALID_PATH")
            if isinstance(value, list) and (not isinstance(p, int) or p < 0):
                raise PolicyError("INVALID_PATH")
            value = value[p]
        return value
    except (KeyError, IndexError, TypeError):
        raise PolicyError("MISSING_PATH") from None


def parse_date(value):
    if not isinstance(value, str):
        raise PolicyError("DATE_NOT_STRING")
    try:
        if "T" in value:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise PolicyError("DATE_NEEDS_TIMEZONE")
            return parsed
        return dt.date.fromisoformat(value)
    except ValueError:
        raise PolicyError("INVALID_DATE") from None


def shifted(value, days):
    parsed = parse_date(value) + dt.timedelta(days=days)
    out = parsed.isoformat()
    return out.replace("+00:00", "Z") if value.endswith("Z") else out


def evaluate(inv, records):
    op = inv.get("op")
    if op == "unchanged":
        return get(records, inv.get("path"))
    if op == "equal":
        return get(records, inv.get("left")) == get(records, inv.get("right"))
    if op == "seconds_between":
        try:
            return (parse_date(get(records, inv.get("right"))) - parse_date(get(records, inv.get("left")))).total_seconds()
        except TypeError:
            raise PolicyError("INCOMPATIBLE_DATES") from None
    raise PolicyError("UNKNOWN_INVARIANT")


def boundary_pattern(aliases):
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")(?!\w)")


def build_mapping(bundle, key):
    mapping, owners, used, ids = {}, {}, set(), set()
    entities = bundle.get("entities")
    if not isinstance(entities, list) or not entities:
        raise PolicyError("ENTITIES_REQUIRED")
    source_aliases = set()
    for entity in entities:
        aliases = entity.get("aliases", [])
        if not isinstance(aliases, list) or not aliases:
            raise PolicyError("ALIASES_REQUIRED")
        for alias in aliases:
            value = alias.get("value")
            if not isinstance(value, str) or len(value) < 3:
                raise PolicyError("SHORT_OR_INVALID_ALIAS")
            source_aliases.add(value)
    for entity in entities:
        eid, kind = entity.get("id"), entity.get("kind")
        if not isinstance(eid, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", eid):
            raise PolicyError("INVALID_ENTITY_ID")
        if eid in ids or kind not in FORMS:
            raise PolicyError("DUPLICATE_ENTITY_OR_UNKNOWN_KIND")
        ids.add(eid)
        for attempt in range(512):
            token = digest(key, bundle["scope"], kind, eid, str(attempt))
            n = int(token[:12], 16)
            if kind == "PERSON":
                first, last = GIVEN[n % len(GIVEN)], FAMILY[(n // len(GIVEN)) % len(FAMILY)]
                forms = {"full": f"{first} {last}", "initial": f"{first[0]}. {last}"}
            elif kind == "EMAIL":
                forms = {"default": f"case-{token[:16]}@example.invalid"}
            elif kind == "ADDRESS":
                # Explicitly fictional location; geographical fidelity is out of scope.
                forms = {"default": f"{100 + n % 800} Example Lane, Testville, ZZ 00000"}
            else:
                forms = {"default": f"{kind}TEST-{token[:16]}"}
            proposed = set(forms.values())
            if not proposed & (used | source_aliases):
                break
        else:
            raise PolicyError("REPLACEMENT_SPACE_EXHAUSTED")
        used.update(proposed)
        for alias in entity["aliases"]:
            value, form = alias["value"], alias.get("form", "default")
            if form not in FORMS[kind]:
                raise PolicyError("UNKNOWN_ALIAS_FORM")
            if value in mapping and owners[value] != eid:
                raise PolicyError("AMBIGUOUS_ALIAS")
            if value in mapping and mapping[value] != forms[form]:
                raise PolicyError("INCONSISTENT_ALIAS")
            mapping[value], owners[value] = forms[form], eid
    return mapping


def transform(bundle: dict, key: bytes) -> Result:
    if not isinstance(key, bytes) or len(key) < 32:
        raise PolicyError("KEY_TOO_SHORT")
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise PolicyError("SCHEMA_VERSION")
    if bundle.get("data_class") != "synthetic":
        raise PolicyError("FIXTURE_ONLY")
    scope = bundle.get("scope")
    if not isinstance(scope, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", scope):
        raise PolicyError("INVALID_SCOPE")
    records = bundle.get("records")
    if not isinstance(records, list) or not records:
        raise PolicyError("RECORDS_REQUIRED")
    canonical(records)  # Reject NaN/infinity before any write.
    mapping = build_mapping(bundle, key)
    days = 7 * (1 + int(digest(key, scope, "date-shift")[:8], 16) % 26)
    date_paths = bundle.get("date_paths", [])
    if not isinstance(date_paths, list):
        raise PolicyError("INVALID_DATE_PATHS")
    for path in date_paths:
        value = get(records, path)
        replacement = shifted(value, days)
        if value in mapping and mapping[value] != replacement:
            raise PolicyError("CONFLICTING_DATE_ALIAS")
        mapping[value] = replacement
    pattern = boundary_pattern(mapping)
    issues = []
    # Keys cannot be transformed without changing schema: stop if they contain
    # registered aliases or detector hits. Keys and paths are not echoed in reports.
    def check_keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if pattern.search(k) or any(p.search(k) for p in PATTERNS.values()):
                    issues.append({"code": "SENSITIVE_KEY"})
                check_keys(v)
        elif isinstance(node, list):
            for v in node:
                check_keys(v)
    check_keys(records)
    # Every regex hit must be contained inside an explicitly registered span.
    # This is a tripwire, not evidence that unknown names/addresses were found.
    aliases_seen = set()
    for _, text in walk(records):
        spans = list(pattern.finditer(text))
        aliases_seen.update(m.group() for m in spans)
        for kind, detector in PATTERNS.items():
            for m in detector.finditer(text):
                if not any(a.start() <= m.start() and a.end() >= m.end() for a in spans):
                    issues.append({"code": "UNREGISTERED_PATTERN", "kind": kind})
    if set(mapping) - aliases_seen:
        issues.append({"code": "UNUSED_ALIAS"})
    invariants = bundle.get("invariants", [])
    if not isinstance(invariants, list) or not invariants:
        raise PolicyError("INVARIANTS_REQUIRED")
    inv_ids = set()
    for inv in invariants:
        iid = inv.get("id")
        if not isinstance(iid, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", iid) or iid in inv_ids:
            raise PolicyError("INVALID_INVARIANT_ID")
        inv_ids.add(iid)
    before = [evaluate(inv, records) for inv in invariants]
    candidate = copy.deepcopy(records)
    replacements = 0
    def rewrite(node):
        nonlocal replacements
        if isinstance(node, dict):
            return {k: rewrite(v) for k, v in node.items()}
        if isinstance(node, list):
            return [rewrite(v) for v in node]
        if isinstance(node, str):
            out, count = pattern.subn(lambda m: mapping[m.group()], node)
            replacements += count
            return out
        return node
    candidate = rewrite(candidate)
    checks = []
    for inv, original in zip(invariants, before):
        passed = evaluate(inv, candidate) == original
        checks.append({"id": inv["id"], "passed": passed})
        if not passed:
            issues.append({"code": "INVARIANT_CHANGED", "check": inv["id"]})
    for _, text in walk(candidate):
        if pattern.search(text):
            issues.append({"code": "SOURCE_ALIAS_REMAINS"})
    # Report has no source text, source identifiers, replacement table, or key.
    report = {
        "version": "0.1.0", "fixture_only": True,
        "status": "QUARANTINED" if issues else "TRANSFORMED_REVIEW_REQUIRED",
        "external_use": "NOT_CLEARED", "record_count": len(records),
        "replacement_count": replacements, "checks": checks, "issues": issues,
        "input_commitment": digest(key, scope, canonical(bundle).decode()),
        "candidate_commitment": digest(key, scope, canonical(candidate).decode()),
        "limits": ["Explicit aliases only; no general name or address detection.",
                   "No PDF, image, audio, OCR, metadata, or archive processing.",
                   "No privacy certification or agent-outcome equivalence test.",
                   "Date shifts preserve elapsed time and weekday, not holidays or legal deadlines."],
    }
    output = None if issues else {"schema_version": 1, "data_class": "synthetic_transformed", "records": candidate}
    return Result(output, report)
