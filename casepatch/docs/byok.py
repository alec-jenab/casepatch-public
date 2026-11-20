"""Bring-your-own-key (BYOK) for optional hosted detection on USER documents.

Rules this module enforces (fail closed):
* The key is the user's OWN provider key: ANTHROPIC_API_KEY in the user's environment, or a key the user typed into
  local Studio settings (held in memory, or in the OS keychain when the user ticks "remember" and `keyring` is
  installed). Calls go directly from the user's machine to the provider. No CasePatch server is involved.
* Keys are never written to job files, logs, reports, caches or exports, and never echoed back by the API.
* The CasePatch evaluation setup is NOT used: no shared spend ledger (a per-case estimate limit is enforced in memory
  instead), no CASEPATCH_HOSTED / CASEPATCH_SPEND_CAP_USD switches, and any key registered as a CasePatch
  evaluation key (public-sample evals only) is refused. The registry stores only SHA-256 fingerprints, never keys.
* Model: claude-sonnet-5-5 by default; only models on ALLOWED_MODELS. Fable and Mythos are excluded (not eligible
  for zero-data-retention arrangements at the provider, so they are never offered).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5-5"
ALLOWED_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5")
EXCLUDED_MARKERS = ("fable", "mythos")
USER_KEY_ENV = "ANTHROPIC_API_KEY"                   # the user's own key in their own environment
HANDOFF_ENV = "CASEPATCH_BYOK_ANTHROPIC_KEY"         # Studio server -> its own local worker process only
MODEL_ENV = "CASEPATCH_BYOK_MODEL"
MAX_USD_ENV = "CASEPATCH_BYOK_MAX_USD_PER_CASE"
DEFAULT_MAX_USD_PER_CASE = 2.0
MAX_USD_PER_CASE_LIMIT = 50.0
EVAL_KEYS_ENV = "CASEPATCH_EVAL_KEYS_FILE"
EVAL_ENV_MARKERS = ("CASEPATCH_HOSTED", "CASEPATCH_SPEND_CAP_USD", "CASEPATCH_SPEND_LEDGER")
_KEY_SHAPE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,300}")
KEYCHAIN_SERVICE, KEYCHAIN_USER = "casepatch", "anthropic-api-key"


class ByokRefused(Exception):
    """args[0] is a user-facing reason; never contains a key."""


# ------------------------------------------------------------------------------------------------ eval-key registry
def eval_keys_path(env=None):
    env = os.environ if env is None else env
    return Path(env.get(EVAL_KEYS_ENV) or (Path.home() / ".casepatch" / "eval-key-fingerprints.json")).expanduser()


def fingerprint(key: str) -> str:
    return hashlib.sha256(("casepatch-eval-key:" + key).encode()).hexdigest()[:32]


def register_eval_keys(env=None):
    """Called by the evaluation harness (modeleval, editbench, `hosted verify`): records fingerprints of every provider
    key in the eval environment so the product refuses them on user documents. Returns how many are registered."""
    from ..hosted.config import KEY_ENV
    env = os.environ if env is None else env
    keys = [env[n] for names in KEY_ENV.values() for n in names if env.get(n)]
    if not keys:
        return 0
    p = eval_keys_path(env)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        have = set(json.loads(p.read_text()).get("fingerprints", []))
    except (OSError, ValueError, AttributeError):
        have = set()
    have |= {fingerprint(k) for k in keys}
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"note": "SHA-256 fingerprints of CasePatch evaluation keys (public samples only); never the keys",
                   "fingerprints": sorted(have)}, f)
    os.replace(tmp, p)
    return len(have)


PACKAGED_EVAL_KEYS = Path(__file__).with_name("data") / "eval_key_fingerprints.json"   # shipped: a clean install refuses them too


def is_eval_key(key: str, env=None) -> bool:
    fp = fingerprint(key)
    for p in (PACKAGED_EVAL_KEYS, eval_keys_path(env)):
        try:
            if fp in set(json.loads(p.read_text()).get("fingerprints", [])):
                return True
        except (OSError, ValueError, AttributeError):
            pass
    return False


# ------------------------------------------------------------------------------------------------ key + settings
def valid_key_shape(key) -> bool:
    return isinstance(key, str) and bool(_KEY_SHAPE.fullmatch(key))


def check_key(key, env=None):
    if not valid_key_shape(key):
        raise ByokRefused("That does not look like an Anthropic API key (it should start with sk-ant-).")
    if is_eval_key(key, env):
        raise ByokRefused("This key is registered as a CasePatch evaluation key, which may only be used on public "
                          "samples. Use your own Anthropic key.")
    return key


def resolve_key(env=None, user_key=None):
    """(key, source). Order: a key the user entered in Studio settings, the Studio handoff variable (worker only),
    then the user's own ANTHROPIC_API_KEY. Raises ByokRefused."""
    env = os.environ if env is None else env
    if user_key:
        return check_key(user_key, env), "studio-settings"
    if env.get(HANDOFF_ENV):
        return check_key(env[HANDOFF_ENV], env), "studio-settings"
    if env.get(USER_KEY_ENV):
        return check_key(env[USER_KEY_ENV], env), "environment"
    raise ByokRefused("No Anthropic API key of your own is set. Enter your key in settings, or set ANTHROPIC_API_KEY.")


def model_from(env=None):
    env = os.environ if env is None else env
    m = (env.get(MODEL_ENV) or DEFAULT_MODEL).strip().lower()
    if any(x in m for x in EXCLUDED_MARKERS):
        raise ByokRefused(f"{m} is not offered for user documents (not eligible for zero-data-retention terms).")
    if m not in ALLOWED_MODELS:
        raise ByokRefused(f"{MODEL_ENV} must be one of: {', '.join(ALLOWED_MODELS)}")
    return m


def max_usd_from(env=None):
    env = os.environ if env is None else env
    raw = env.get(MAX_USD_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_MAX_USD_PER_CASE
    try:
        v = float(str(raw).strip())
    except ValueError:
        raise ByokRefused(f"{MAX_USD_ENV} must be a number")
    if not math.isfinite(v) or v <= 0 or v > MAX_USD_PER_CASE_LIMIT:
        raise ByokRefused(f"{MAX_USD_ENV} must be between 0 and {MAX_USD_PER_CASE_LIMIT:.0f}")
    return v


def eval_markers(env=None):
    env = os.environ if env is None else env
    return [m for m in EVAL_ENV_MARKERS if env.get(m)]


# ------------------------------------------------------------------------------------------------ keychain (optional)
def key_file(env=None):
    """Fallback when no OS keychain backend exists (headless Linux): a 0600 file in the user's CasePatch folder."""
    env = os.environ if env is None else env
    from ..paths import home
    return home(env) / "anthropic-key"


def store_key(key, env=None):
    """Remember the user's own key on this computer only: OS keychain if available, else a 0600 file. Returns where."""
    check_key(key, env)
    if keychain_available():
        try:
            keychain_save(key); return "keychain"
        except Exception:
            pass
    p = key_file(env); p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    os.replace(tmp, p)
    return "file"


def load_stored_key(env=None):
    k = keychain_load()
    if not k:
        try:
            p = key_file(env)
            k = p.read_text().strip() if p.exists() and (p.stat().st_mode & 0o077) == 0 else None
        except OSError:
            k = None
    if not valid_key_shape(k) or is_eval_key(k, env):
        return None
    return k


def forget_stored_key(env=None):
    keychain_forget()
    key_file(env).unlink(missing_ok=True)


def keychain_available():
    try:
        import keyring  # noqa: F401
        return True
    except Exception:
        return False


def keychain_load():
    try:
        import keyring
        k = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_USER)
        return k if valid_key_shape(k) else None
    except Exception:
        return None


def keychain_save(key):
    import keyring
    keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_USER, key)


def keychain_forget():
    try:
        import keyring
        keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_USER)
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ per-case budget
class CaseBudget:
    """In-memory replacement for the eval SpendLedger: one case, one process, worst-case reservations against a
    per-case estimate limit. It protects the user's own bill from runaway calls; the provider's bill is authoritative."""

    def __init__(self, max_usd):
        self.max_usd = float(max_usd)
        self.rows = {}

    def _used(self):
        return sum(r["cost"] if r["cost"] is not None else r["reserved"] for r in self.rows.values())

    def reserve(self, amount_usd, cap_usd=None, *, max_calls_per_case=None, **_):
        from ..hosted.ledger import CapExceeded
        if not (amount_usd > 0):
            raise CapExceeded("refusing a call without a positive cost estimate")
        if self._used() + amount_usd > self.max_usd + 1e-12:
            raise CapExceeded(f"per-case limit ${self.max_usd:.2f} would be exceeded (estimate ${self._used() + amount_usd:.4f})")
        if max_calls_per_case is not None and len(self.rows) >= max_calls_per_case:
            raise CapExceeded(f"per-case call bound reached ({max_calls_per_case})")
        rid = uuid.uuid4().hex
        self.rows[rid] = {"reserved": float(amount_usd), "cost": None, "t": time.time()}
        return rid

    def commit(self, rid, actual_usd):
        if rid in self.rows:
            self.rows[rid]["cost"] = float(actual_usd)

    def keep(self, rid, note):
        if rid in self.rows:
            self.rows[rid]["cost"] = self.rows[rid]["reserved"]

    def spent_estimate(self):
        return round(self._used(), 6)


def make_client(key, model, max_usd, transport=None, max_calls=40):
    """HostedClient wired to the user's key and an in-memory CaseBudget (never the eval ledger)."""
    from ..hosted.client import HostedClient
    from ..hosted.config import HostedConfig
    cfg = HostedConfig(provider=PROVIDER, cap_usd=max_usd, ledger_path=Path(os.devnull), max_output_tokens=4096,
                       max_images_per_call=1, max_image_side=1600, max_calls_per_case=max_calls, max_retries=0,
                       pool="anthropic", pool_cap_usd=max_usd, _key=key)
    return HostedClient(cfg, transport=transport, ledger=CaseBudget(max_usd))
