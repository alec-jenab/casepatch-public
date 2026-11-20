"""Server-side activation metrics, computed from collected events. Runs on synthetic events today.

Definitions:
* Verified external active user: a distinct *participant* (random token issued by a
  verification service after the person verifies once, e.g. an email magic link or GitHub sign-in) with at least one
  export_completed event (reviewed=True by schema) within the trailing 7 days, excluding internal participants
  (allowlist) and events tagged internal.
* Anonymous installs with a reviewed export: same window, counted by install id. This is an UPPER BOUND on people
  (one person can have several installs; one install can be shared) and is never reported as users.
* 30-day repeat: participants with reviewed exports on >= 2 distinct UTC days within the trailing 30 days.
* Distinct organizations: count of distinct org keys among active participants. Org keys are assigned by the
  verification service (keyed hash of a verified work-email domain); free-mail domains get no org key.
"""
from __future__ import annotations

import datetime as dt


def _ts(e):
    return dt.datetime.fromisoformat(e["ts"])


def activation_metrics(events, as_of=None, internal_participants=frozenset(), internal_installs=frozenset(), org_of=None):
    as_of = as_of or dt.datetime.now(dt.timezone.utc)
    org_of = org_of or {}
    w7, w30 = as_of - dt.timedelta(days=7), as_of - dt.timedelta(days=30)
    exports = [e for e in events if e.get("event") == "export_completed" and e.get("props", {}).get("reviewed") is True
               and not e.get("internal") and _ts(e) <= as_of]
    def external_p(e):
        p = e.get("participant")
        return p and p not in internal_participants
    seen_case = set()
    unique = []
    for e in sorted(exports, key=_ts):  # one export per (participant/install, case) per day counts once
        k = (e.get("participant") or e.get("install"), e.get("case_ref"), _ts(e).date())
        if k not in seen_case:
            seen_case.add(k); unique.append(e)
    active = {e["participant"] for e in unique if external_p(e) and _ts(e) > w7}
    days = {}
    for e in unique:
        if external_p(e) and _ts(e) > w30:
            days.setdefault(e["participant"], set()).add(_ts(e).date())
    installs = {e["install"] for e in unique if e.get("install") and e["install"] not in internal_installs and _ts(e) > w7}
    orgs = {org_of[p] for p in active if org_of.get(p)}
    return {"as_of": as_of.isoformat(), "verified_external_active_users_7d": len(active),
            "repeat_users_30d": sum(1 for d in days.values() if len(d) >= 2),
            "distinct_orgs_among_active": len(orgs),
            "anonymous_installs_with_reviewed_export_7d_upper_bound_not_users": len(installs),
            "reviewed_exports_7d": sum(1 for e in unique if _ts(e) > w7)}
