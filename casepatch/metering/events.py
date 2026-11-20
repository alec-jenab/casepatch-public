"""Local-first metering and opt-in event log.

Design rules (enforced in code, covered by tests/test_metering.py):
* Meter: numeric counters only (cases, documents, pages, reviewed exports) in a local SQLite file. Always on, never
  leaves the machine, visible and resettable in the app. Used for local usage display and future quotas.
* EventLog: written only after the user opts in. Every event is checked against a strict schema: an allowlisted
  event name, allowlisted property names, and values that are ints, bools or members of small enums. Free text, file
  names, document text, replacement values and paths cannot be represented, so they cannot be logged.
* Identity: a random install id (not derived from hardware, user name or content) and, only if the user later
  verifies through an approved service, a random participant token issued by that service. A random case_ref
  (not derived from content) lets exports be de-duplicated per case.
* Upload: Uploader refuses unless telemetry is opted in, CASEPATCH_TELEMETRY_ENDPOINT is set AND that exact URL is in
  APPROVED_ENDPOINTS (empty in this build). The default transport refuses too, so this build makes no network calls.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

SCHEMA_VERSION = 1
APPROVED_ENDPOINTS: tuple = ()  # add the exact https URL here in a reviewed commit before uploads are enabled

_ENUMS = {
    "mode": {"local", "hosted"},
    "detector": {"rules", "gliner", "llm", "hosted", "rules+hosted"},
    "reason": {"not_reviewed", "no_run", "stale_run", "model_suggestions_pending", "pipeline_error", "unsupported_file", "too_large"},
    "source": {"upload", "sample"},
    "os": {"linux", "darwin", "windows", "other"},
}
_INT = "int"; _BOOL = "bool"
SCHEMA = {
    "app_started": {"mode": "mode", "os": "os"},
    "telemetry_opted_in": {},
    "case_created": {"source": "source"},
    "case_analyzed": {"documents": _INT, "pages": _INT, "pdf_pages": _INT, "image_pages": _INT, "json_documents": _INT,
                      "proposals": _INT, "regions": _INT, "seconds": _INT, "detector": "detector", "source": "source",
                      "model_suggestions": _INT},
    "review_applied": {"accepted": _INT, "rejected": _INT, "added": _INT, "regions_replace": _INT, "regions_keep": _INT,
                       "regions_erase": _INT, "changed_from_default": _INT, "seconds": _INT},
    "export_completed": {"documents": _INT, "pages": _INT, "reviewed": _BOOL, "review_seconds": _INT,
                         "needs_review_items": _INT, "verification_ran": _BOOL, "source": "source"},
    "export_refused": {"reason": "reason"},
    "analysis_failed": {"reason": "reason"},
}
_ID_FIELDS = {"case_ref"}


class EventRejected(ValueError):
    pass


class TelemetryDisabled(RuntimeError):
    pass


def casepatch_home() -> Path:
    from ..paths import home as _home
    home = _home()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    return home


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def is_internal() -> bool:
    """Internal test runs are tagged so they can be excluded from active-user counts."""
    return os.environ.get("CASEPATCH_INTERNAL", "") == "1"


class Settings:
    def __init__(self, home: Path | None = None):
        self.path = (home or casepatch_home()) / "settings.json"
        self._lock = threading.Lock()

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, data: dict):
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.path)

    @property
    def telemetry_opt_in(self) -> bool:
        return self.load().get("telemetry_opt_in") is True

    def set_telemetry(self, opt_in: bool):
        data = self.load()
        data["telemetry_opt_in"] = bool(opt_in)
        if opt_in and not data.get("install_id"):
            data["install_id"] = secrets.token_hex(16)
        data["telemetry_changed_at"] = _utc_now().isoformat()
        self.save(data)

    @property
    def install_id(self):
        return self.load().get("install_id")

    @property
    def participant(self):
        """Random token from an approved verification service (none exists yet)."""
        tok = self.load().get("participant")
        return tok if isinstance(tok, str) and len(tok) == 32 and all(c in "0123456789abcdef" for c in tok) else None


def validate(event: str, props: dict) -> dict:
    if event not in SCHEMA:
        raise EventRejected(f"unknown event {event!r}")
    allowed = SCHEMA[event]
    out = {}
    for k, v in (props or {}).items():
        if k not in allowed:
            raise EventRejected(f"property {k!r} not allowed for {event}")
        t = allowed[k]
        if t == _INT:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 10**9:
                raise EventRejected(f"{k} must be a non-negative int")
        elif t == _BOOL:
            if not isinstance(v, bool):
                raise EventRejected(f"{k} must be bool")
        elif v not in _ENUMS[t]:
            raise EventRejected(f"{k} must be one of {sorted(_ENUMS[t])}")
        out[k] = v
    if event == "export_completed" and out.get("reviewed") is not True:
        raise EventRejected("an export event must be a reviewed export")
    return out


class EventLog:
    """Append-only JSONL with a hash chain (detects local edits/deletions; not tamper-proof)."""

    def __init__(self, home: Path | None = None, settings: Settings | None = None, app_version: str = "0"):
        self.home = home or casepatch_home()
        self.settings = settings or Settings(self.home)
        self.path = self.home / "events.jsonl"
        self.app_version = app_version
        self._lock = threading.Lock()

    def _last(self):
        if not self.path.exists():
            return 0, "0" * 64
        last = None
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    last = line
        if last is None:
            return 0, "0" * 64
        e = json.loads(last)
        return e["seq"], e["hash"]

    def record(self, event: str, props: dict | None = None, case_ref: str | None = None):
        """Returns the stored event, or None when the user has not opted in (nothing is written)."""
        clean = validate(event, props or {})
        if not self.settings.telemetry_opt_in:
            return None
        if case_ref is not None and not (isinstance(case_ref, str) and len(case_ref) == 32 and all(c in "0123456789abcdef" for c in case_ref)):
            raise EventRejected("case_ref must be a random 32-hex token")
        with self._lock:
            seq, prev = self._last()
            e = {"v": SCHEMA_VERSION, "seq": seq + 1, "event": event, "ts": _utc_now().isoformat(),
                 "install": self.settings.install_id, "participant": self.settings.participant, "internal": is_internal(),
                 "app": self.app_version, "case_ref": case_ref, "props": clean, "prev": prev}
            e["hash"] = hashlib.sha256(json.dumps({k: v for k, v in e.items()}, sort_keys=True).encode()).hexdigest()
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(json.dumps(e, sort_keys=True) + "\n")
            return e

    def read(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for i, e in enumerate(self.read(), 1):
            h = e.pop("hash")
            if e["seq"] != i or e["prev"] != prev:
                return False
            if hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() != h:
                return False
            prev = h
        return True

    def clear(self):
        with self._lock:
            if self.path.exists():
                self.path.unlink()


def _refuse_transport(url, body):
    raise TelemetryDisabled("no network transport in this build")


class Uploader:
    """Builds exactly the batch that would be sent. Sending is refused unless every gate is open."""

    def __init__(self, log: EventLog, transport=None, endpoint: str | None = None):
        self.log = log
        self.transport = transport or _refuse_transport
        self.endpoint = endpoint if endpoint is not None else os.environ.get("CASEPATCH_TELEMETRY_ENDPOINT", "")
        self.cursor_path = log.home / "events.sent"

    def status(self) -> dict:
        return {"opted_in": self.log.settings.telemetry_opt_in, "endpoint_configured": bool(self.endpoint),
                "endpoint_approved": self.endpoint in APPROVED_ENDPOINTS, "approved_endpoints": list(APPROVED_ENDPOINTS),
                "sending_enabled": self._gate() is None}

    def _gate(self):
        if not self.log.settings.telemetry_opt_in:
            return "user has not opted in"
        if not self.endpoint:
            return "no telemetry endpoint configured"
        if self.endpoint not in APPROVED_ENDPOINTS or not self.endpoint.startswith("https://"):
            return "endpoint is not in APPROVED_ENDPOINTS"
        return None

    def pending(self) -> list:
        sent = int(self.cursor_path.read_text() or 0) if self.cursor_path.exists() else 0
        return [e for e in self.log.read() if e["seq"] > sent]

    def build_batch(self) -> dict:
        return {"schema": SCHEMA_VERSION, "events": self.pending()}

    def send(self) -> int:
        reason = self._gate()
        if reason:
            raise TelemetryDisabled(reason)
        batch = self.build_batch()
        if not batch["events"]:
            return 0
        self.transport(self.endpoint, json.dumps(batch).encode())
        self.cursor_path.write_text(str(batch["events"][-1]["seq"]))
        return len(batch["events"])


class Meter:
    """Local numeric usage counters by UTC day. Never uploaded."""
    METRICS = ("cases_analyzed", "documents_processed", "pages_processed", "pdf_pages", "image_pages", "json_documents",
               "review_runs", "reviewed_exports", "pages_exported", "processing_seconds")

    def __init__(self, home: Path | None = None):
        self.path = (home or casepatch_home()) / "usage.sqlite"
        with closing(self._db()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS usage(day TEXT, metric TEXT, value INTEGER, PRIMARY KEY(day, metric))")
            db.commit()
        os.chmod(self.path, 0o600)

    def _db(self):
        return sqlite3.connect(str(self.path), timeout=10)

    def add(self, **counts):
        day = _utc_now().date().isoformat()
        with closing(self._db()) as db:
            db.execute("BEGIN IMMEDIATE")
            for k, v in counts.items():
                if k not in self.METRICS or isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    db.rollback()
                    raise ValueError(f"bad metric {k}={v!r}")
                db.execute("INSERT INTO usage VALUES(?,?,?) ON CONFLICT(day,metric) DO UPDATE SET value=value+excluded.value",
                           (day, k, v))
            db.commit()

    def totals(self, since: str | None = None) -> dict:
        with closing(self._db()) as db:
            q = "SELECT metric, SUM(value) FROM usage" + (" WHERE day >= ?" if since else "") + " GROUP BY metric"
            rows = db.execute(q, (since,) if since else ()).fetchall()
        out = {m: 0 for m in self.METRICS}
        out.update({m: int(v) for m, v in rows})
        return out

    def month(self) -> dict:
        return self.totals(_utc_now().date().replace(day=1).isoformat())

    def reset(self):
        with closing(self._db()) as db:
            db.execute("DELETE FROM usage"); db.commit()
