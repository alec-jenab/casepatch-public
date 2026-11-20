"""Local SQLite spend ledger: reserve (atomic, BEGIN IMMEDIATE) -> commit actual / keep estimate.

* Every reservation is a row; it counts against the cap until committed with the provider-reported cost.
* Reservations never expire: a crash, timeout or unknown outcome leaves them counted (fail closed), also across
  restarts and across processes sharing the file.
* Enforces our estimate of spend for calls made through CasePatch only; not the provider's authoritative bill.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time
import uuid


class CapExceeded(Exception):
    pass


SCHEMA = """CREATE TABLE IF NOT EXISTS calls(
  id TEXT PRIMARY KEY, case_id TEXT, provider TEXT, model TEXT, kind TEXT, attempt INTEGER,
  reserved_usd REAL NOT NULL, cost_usd REAL, status TEXT NOT NULL, note TEXT, created REAL, updated REAL)"""


class SpendLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600); os.close(fd)
        self.db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(SCHEMA)

    def close(self):
        self.db.close()

    def totals(self):
        r = self.db.execute("SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0),"
                            " COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0) FROM calls").fetchone()
        return {"reserved_usd": round(r[0], 6), "spent_usd": round(r[1], 6)}

    def provider_totals(self, provider):
        r = self.db.execute("SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0),"
                            " COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0) FROM calls WHERE provider=?",
                            (provider,)).fetchone()
        return {"reserved_usd": round(r[0], 6), "spent_usd": round(r[1], 6)}

    def pool_totals(self, pool):
        from .config import pool_filter
        op, members = pool_filter(pool)
        q = ",".join("?" * len(members))
        r = self.db.execute("SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0),"
                            " COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0) FROM calls"
                            f" WHERE provider {'IN' if op == 'in' else 'NOT IN'} ({q})", members).fetchone()
        return {"reserved_usd": round(r[0], 6), "spent_usd": round(r[1], 6)}

    def reserve(self, amount_usd, cap_usd, *, case_id="", provider="", model="", kind="", attempt=0, max_calls_per_case=None,
                provider_cap_usd=None, pool=None, pool_cap_usd=None):
        if not (amount_usd > 0):
            raise CapExceeded("refusing a call without a positive cost estimate")
        rid = uuid.uuid4().hex
        self.db.execute("BEGIN IMMEDIATE")  # write lock: check-and-insert is atomic across processes
        try:
            reserved, spent = self.db.execute(
                "SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0),"
                " COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0) FROM calls").fetchone()
            if reserved + spent + amount_usd > cap_usd + 1e-12:
                raise CapExceeded(f"cap ${cap_usd:.2f} would be exceeded: spent ${spent:.4f} + reserved ${reserved:.4f} + this call ${amount_usd:.4f}")
            if provider_cap_usd is not None:  # sub-budget for this provider inside the shared total cap
                pr, ps = self.db.execute(
                    "SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0),"
                    " COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0) FROM calls WHERE provider=?",
                    (provider,)).fetchone()
                if pr + ps + amount_usd > provider_cap_usd + 1e-12:
                    raise CapExceeded(f"{provider} sub-budget ${provider_cap_usd:.2f} would be exceeded: spent ${ps:.4f} + reserved ${pr:.4f} + this call ${amount_usd:.4f}")
            if pool_cap_usd is not None:  # hard pool cap (anthropic / openai / other = every remaining provider combined)
                from .config import pool_filter
                op, members = pool_filter(pool or "other")
                q = ",".join("?" * len(members))
                qr, qs = self.db.execute(
                    "SELECT COALESCE(SUM(CASE WHEN status='reserved' THEN reserved_usd END),0),"
                    " COALESCE(SUM(CASE WHEN status!='reserved' THEN cost_usd END),0) FROM calls"
                    f" WHERE provider {'IN' if op == 'in' else 'NOT IN'} ({q})", members).fetchone()
                if qr + qs + amount_usd > pool_cap_usd + 1e-12:
                    raise CapExceeded(f"{pool or 'other'} pool cap ${pool_cap_usd:.2f} would be exceeded: spent ${qs:.4f} + reserved ${qr:.4f} + this call ${amount_usd:.4f}")
            if max_calls_per_case is not None and case_id:
                n = self.db.execute("SELECT COUNT(*) FROM calls WHERE case_id=?", (case_id,)).fetchone()[0]
                if n >= max_calls_per_case:
                    raise CapExceeded(f"per-case call bound reached ({max_calls_per_case})")
            now = time.time()
            self.db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (rid, case_id, provider, model, kind, attempt, amount_usd, None, "reserved", None, now, now))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return rid

    def commit(self, rid, actual_usd):
        self.db.execute("UPDATE calls SET cost_usd=?, status='committed', updated=? WHERE id=? AND status='reserved'",
                        (float(actual_usd), time.time(), rid))

    def keep(self, rid, note):
        """Unknown/failed outcome: the full reservation stays as spent."""
        self.db.execute("UPDATE calls SET cost_usd=reserved_usd, status='kept_estimate', note=?, updated=? WHERE id=? AND status='reserved'",
                        (note[:200], time.time(), rid))

    def last(self, case_id):
        """Most recent row for one case_id (a call's own row even when other processes share the ledger)."""
        r = self.db.execute("SELECT id, reserved_usd, cost_usd, status, note FROM calls WHERE case_id=? ORDER BY created DESC, rowid DESC LIMIT 1",
                            (case_id,)).fetchone()
        return dict(zip(("id", "reserved_usd", "cost_usd", "status", "note"), r)) if r else None

    def rows(self):
        return self.db.execute("SELECT id, case_id, kind, attempt, reserved_usd, cost_usd, status, note FROM calls ORDER BY created").fetchall()
