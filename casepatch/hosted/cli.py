"""`python -m casepatch hosted status`: provider, key present (yes/no only), cap, spent and reserved."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..tls import ssl_context
from .config import HostedDisabled, key_present, ledger_path, parse_cap, provider_from, redact


def status(env=None):
    env = os.environ if env is None else env
    lines = []
    try:
        provider = provider_from(env)
    except HostedDisabled as e:
        return f"hosted: invalid provider ({e})"
    lines.append(f"provider: {provider}")
    lines.append(f"key present: {'yes' if key_present(env, provider) else 'no'}")
    enabled = env.get("CASEPATCH_HOSTED") == "1"
    try:
        cap = parse_cap(env.get("CASEPATCH_SPEND_CAP_USD")); cap_s = f"${cap:.2f}"
    except HostedDisabled as e:
        cap, cap_s = None, f"not usable ({e})"
    lines.append(f"cap: {cap_s}")
    lp = ledger_path(env)
    if lp.exists():
        from .ledger import SpendLedger
        led = SpendLedger(lp); t = led.totals(); led.close()
        lines.append(f"spent (CasePatch estimate/usage): ${t['spent_usd']:.4f}")
        lines.append(f"reserved (in flight or unknown): ${t['reserved_usd']:.4f}")
    else:
        lines.append("spent: $0.0000 (no ledger yet)")
        lines.append("reserved: $0.0000")
    lines.append(f"ledger: {lp}")
    on = enabled and cap is not None and key_present(env, provider)
    lines.append("hosted calls: " + ("ENABLED (capped)" if on else "disabled"))
    return redact("\n".join(lines))


# ---------------------------------------------------------------------------------------------- verify (multi-provider)
AUTH_CHECK = {  # free, non-generating endpoints only
    "openai": ("https://api.openai.com/v1/models", "bearer"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", "goog"),
    "kimi": ("https://api.moonshot.ai/v1/models", "bearer"),
    "anthropic": ("https://api.anthropic.com/v1/models?limit=1000", "anthropic"),
}
KIMI_BALANCE = "https://api.moonshot.ai/v1/users/me/balance"


def _http_get(url, headers, timeout=20):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read()[:4000].decode(errors="replace"))
        except Exception:
            body = {}
        return e.code, body
    except Exception as e:  # network/DNS/timeout
        return 0, {"error": type(e).__name__}


def verify(env=None, auth_check=False, getter=None):
    """Key presence (yes/no only) for every provider, total cap, sub-budgets and ledger location. With auth_check=True
    (explicit operator request only) it also calls each provider's FREE list-models endpoint (plus Kimi's balance
    endpoint) once; nothing generating, nothing reserved in the ledger, no retries."""
    from .config import KEY_ENV, PROVIDER_CAP_ENV, parse_provider_cap
    from .modeleval import EVAL_MODELS
    env = os.environ if env is None else env
    getter = getter or _http_get
    if env is os.environ:  # the eval harness: keep the product from ever using these keys on user documents
        from ..docs.byok import register_eval_keys
        register_eval_keys(env)
    out = [f"hosted flag CASEPATCH_HOSTED=1: {'yes' if env.get('CASEPATCH_HOSTED') == '1' else 'no'}"]
    try:
        cap = parse_cap(env.get("CASEPATCH_SPEND_CAP_USD")); out.append(f"total cap (shared by all providers): ${cap:.2f}")
    except HostedDisabled as e:
        cap = None; out.append(f"total cap: not usable ({e})")
    lp = ledger_path(env)
    repo = Path(__file__).resolve().parents[2]
    inside = str(lp.resolve()).startswith(str(repo))
    out.append(f"ledger: {lp} ({'INSIDE THE REPO: refuse' if inside else 'outside the repo'})")
    if lp.exists():
        from .ledger import SpendLedger
        led = SpendLedger(lp)
    else:
        led = None
    for p in ("openai", "gemini", "kimi", "anthropic"):
        line = f"{p}: key present {'yes' if key_present(env, p) else 'no'} ({'/'.join(KEY_ENV[p])})"
        try:
            sub = parse_provider_cap(env, p, cap) if cap else None
            line += f"; sub-budget {('$%.2f' % sub) if sub else 'not set'} ({PROVIDER_CAP_ENV[p]})"
        except HostedDisabled as e:
            line += f"; sub-budget invalid ({e})"
        if led:
            t = led.provider_totals(p); line += f"; spent ${t['spent_usd']:.4f} reserved ${t['reserved_usd']:.4f}"
        out.append(line)
        if auth_check and key_present(env, p):
            key = next(env[n] for n in KEY_ENV[p] if env.get(n))
            url, kind = AUTH_CHECK[p]
            if kind == "bearer":
                h = {"Authorization": "Bearer " + key}
            elif kind == "anthropic":
                from .client import ANTHROPIC_VERSION
                h = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
            else:
                h = {"x-goog-api-key": key}
            st, body = getter(url, h)
            ids = [m.get("id") or m.get("name", "").split("/")[-1] for m in (body.get("data") or body.get("models") or [])
                   if isinstance(m, dict)]
            want = EVAL_MODELS[p]["detect"] + EVAL_MODELS[p]["edit"]
            msg = f"  auth check: http {st}" + (f"; {len(ids)} models listed; eval models available: "
                                                 + ", ".join(f"{m}={'yes' if m in ids else 'no'}" for m in want) if st == 200 else
                                                 f"; error {str((body.get('error') or {}).get('message', body.get('error', '')) if isinstance(body.get('error'), dict) else body.get('error', ''))[:160]}")
            em = json.dumps(body)[:2000]
            if st == 200:
                msg += "\n  auth: OK"
            elif st == 403 and "api.model.read" in em:
                msg += "\n  auth: OK (key accepted; restricted key has no Models-read scope, as recommended)"
            elif st in (401,):
                msg += "\n  auth: FAILED (key rejected)"
            else:
                msg += "\n  auth: UNKNOWN"
            out.append(redact(msg, [key]))
            if p == "kimi" and st == 200:
                bs, bb = getter(KIMI_BALANCE, h)
                d = bb.get("data") or {}
                out.append(redact(f"  kimi balance: http {bs}; available_balance={d.get('available_balance')} "
                                  f"cash={d.get('cash_balance')} voucher={d.get('voucher_balance')}", [key]))
    if led:
        led.close()
    return redact("\n".join(out))
