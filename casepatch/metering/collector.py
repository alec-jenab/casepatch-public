"""Reference collector + participant verification for activation metrics. NOT DEPLOYED; nothing here sends mail or
opens a public socket. It specifies and tests the server side of a "verified external active user" count
so a host can be wired up later.

What it does
* Participant verification: start_verification(email) creates a single-use, 15-minute magic code and hands it to an
  injected mail transport (the default refuses, so no mail can go out). confirm(code) issues a random 32-hex
  participant token. The server keeps only HMAC(secret, normalized email) as the person key and, for work domains,
  HMAC(secret, domain) as the org key; free-mail domains get no org key. Re-verifying the same email maps the new
  token to the SAME person, so one person with several installs/tokens counts once.
* Ingest: every event is re-validated against the client schema (casepatch.metering.events.validate), the envelope
  is checked field by field, unknown participant tokens are stripped (the event still counts toward installs, never
  toward users), and (install, seq) duplicates are ignored.
* Metrics: activation_metrics() over person keys (not tokens), with internal people (test emails registered
  via mark_internal) and internal-tagged events excluded. evidence() lists each counted person pseudonymously with
  first/last reviewed-export time, so the headline number can be audited row by row.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from .active import activation_metrics
from .events import SCHEMA_VERSION, EventRejected, validate

FREE_MAIL = frozenset({"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "yahoo.com", "icloud.com",
                       "me.com", "aol.com", "proton.me", "protonmail.com", "gmx.com", "mail.com", "yandex.com", "zoho.com",
                       "fastmail.com", "hey.com", "pm.me"})
CODE_TTL = dt.timedelta(minutes=15)
_ENVELOPE = {"v", "seq", "event", "ts", "install", "participant", "internal", "app", "case_ref", "props", "prev", "hash"}


def _hex32(x):
    return isinstance(x, str) and len(x) == 32 and all(c in "0123456789abcdef" for c in x)


def _refuse_mail(email, code):
    raise RuntimeError("no mail transport in this build")


def normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if e.count("@") != 1:
        raise ValueError("not an email address")
    local, domain = e.split("@")
    if not local or "." not in domain:
        raise ValueError("not an email address")
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


class Collector:
    def __init__(self, db: Path, secret: bytes, mail_transport=_refuse_mail, now=None):
        if len(secret) < 32:
            raise ValueError("secret must be at least 32 bytes")
        self.db, self.secret, self.mail, self._now = Path(db), secret, mail_transport, now
        with closing(self._c()) as c, c:
            c.executescript("""
            create table if not exists codes(code text primary key, person text, org text, expires text, used int default 0);
            create table if not exists tokens(token text primary key, person text not null, issued text);
            create table if not exists people(person text primary key, org text, internal int default 0, first_seen text);
            create table if not exists events(install text, seq int, body text, primary key(install, seq));
            """)

    def _c(self):
        return sqlite3.connect(self.db)

    def now(self):
        return self._now() if self._now else dt.datetime.now(dt.timezone.utc)

    def _h(self, kind, value):
        return hmac.new(self.secret, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:32]

    def _keys(self, email):
        e = normalize_email(email)
        domain = e.split("@")[1]
        return self._h("person", e), (None if domain in FREE_MAIL else self._h("org", domain))

    # -- verification --------------------------------------------------------------------------------------------
    def start_verification(self, email: str) -> None:
        person, org = self._keys(email)
        code = secrets.token_urlsafe(24)
        with closing(self._c()) as c, c:
            c.execute("insert into codes values(?,?,?,?,0)", (code, person, org, (self.now() + CODE_TTL).isoformat()))
        self.mail(email, code)  # raises with the default transport: nothing is sent from this build

    def confirm(self, code: str) -> str:
        with closing(self._c()) as c, c:
            row = c.execute("select person, org, expires, used from codes where code=?", (code,)).fetchone()
            if not row or row[3] or dt.datetime.fromisoformat(row[2]) < self.now():
                raise PermissionError("code invalid, used or expired")
            person, org = row[0], row[1]
            c.execute("update codes set used=1 where code=?", (code,))
            c.execute("insert or ignore into people(person, org, first_seen) values(?,?,?)", (person, org, self.now().isoformat()))
            token = secrets.token_hex(16)
            c.execute("insert into tokens values(?,?,?)", (token, person, self.now().isoformat()))
        return token

    def mark_internal(self, email: str) -> None:
        person, org = self._keys(email)
        with closing(self._c()) as c, c:
            c.execute("insert into people(person, org, internal, first_seen) values(?,?,1,?) "
                      "on conflict(person) do update set internal=1", (person, org, self.now().isoformat()))

    # -- ingest --------------------------------------------------------------------------------------------------
    def ingest(self, batch: list) -> dict:
        accepted = dup = stripped = 0
        rejected = []
        with closing(self._c()) as c, c:
            known = {t for (t,) in c.execute("select token from tokens")}
            for i, e in enumerate(batch):
                try:
                    if not isinstance(e, dict) or set(e) - _ENVELOPE:
                        raise EventRejected("unexpected envelope fields")
                    if e.get("v") != SCHEMA_VERSION or not _hex32(e.get("install")):
                        raise EventRejected("bad version or install id")
                    if not isinstance(e.get("seq"), int) or isinstance(e.get("seq"), bool) or e["seq"] < 1:
                        raise EventRejected("bad seq")
                    if e.get("case_ref") is not None and not _hex32(e["case_ref"]):
                        raise EventRejected("bad case_ref")
                    if not isinstance(e.get("internal"), bool):
                        raise EventRejected("internal must be bool")
                    ts = dt.datetime.fromisoformat(e["ts"])
                    if ts.tzinfo is None or ts > self.now() + dt.timedelta(minutes=5):
                        raise EventRejected("bad timestamp")
                    props = validate(e["event"], e.get("props") or {})
                except (EventRejected, KeyError, TypeError, ValueError) as err:
                    rejected.append({"index": i, "reason": str(err)})
                    continue
                p = e.get("participant")
                if p is not None and p not in known:
                    p = None; stripped += 1
                body = {"event": e["event"], "ts": ts.isoformat(), "install": e["install"], "participant": p,
                        "internal": e["internal"], "case_ref": e.get("case_ref"), "props": props}
                cur = c.execute("insert or ignore into events values(?,?,?)", (e["install"], e["seq"], json.dumps(body)))
                if cur.rowcount:
                    accepted += 1
                else:
                    dup += 1
        return {"accepted": accepted, "duplicates": dup, "unverified_participant_stripped": stripped, "rejected": rejected}

    # -- metrics -------------------------------------------------------------------------------------------------
    def _person_events(self):
        with closing(self._c()) as c:
            tok = dict(c.execute("select token, person from tokens"))
            people = {p: (o, bool(i)) for p, o, i in c.execute("select person, org, internal from people")}
            evs = [json.loads(b) for (b,) in c.execute("select body from events")]
        for e in evs:
            e["participant"] = tok.get(e["participant"]) if e["participant"] else None
        return evs, people

    def metrics(self, as_of=None) -> dict:
        evs, people = self._person_events()
        internal = frozenset(p for p, (_, i) in people.items() if i)
        org_of = {p: o for p, (o, _) in people.items() if o}
        m = activation_metrics(evs, as_of=as_of or self.now(), internal_participants=internal, org_of=org_of)
        m["counting_unit"] = "verified person (HMAC of normalized email), not install or token"
        return m

    def evidence(self, as_of=None) -> list:
        """One row per counted person in the trailing 7 days; the length equals verified_external_active_users_7d."""
        as_of = as_of or self.now()
        evs, people = self._person_events()
        rows = {}
        for e in evs:
            p = e["participant"]
            ts = dt.datetime.fromisoformat(e["ts"])
            if (e["event"] != "export_completed" or e["internal"] or not p or people.get(p, (None, True))[1]
                    or not as_of - dt.timedelta(days=7) < ts <= as_of):
                continue
            r = rows.setdefault(p, {"person": p[:8], "org": (people[p][0] or "")[:8] or None, "reviewed_exports": 0,
                                    "cases": set(), "first": ts, "last": ts})
            r["reviewed_exports"] += 1; r["cases"].add(e["case_ref"])
            r["first"], r["last"] = min(r["first"], ts), max(r["last"], ts)
        return [{**r, "cases": len(r["cases"]), "first": r["first"].isoformat(), "last": r["last"].isoformat()}
                for r in sorted(rows.values(), key=lambda r: r["first"])]
