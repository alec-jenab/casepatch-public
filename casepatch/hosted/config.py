"""Environment-only configuration. Keys are read ONLY from OPENAI_API_KEY / GEMINI_API_KEY (GOOGLE_API_KEY as the
Gemini fallback) / MOONSHOT_API_KEY (Kimi) / ANTHROPIC_API_KEY (Claude) / FAL_KEY (fal.ai, in the shared 'other' pool); never from files, never logged, never written to reports."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re

ABSURD_CAP_USD = 1000.0
# Hard caps: separate pools plus a grand total. All fail closed.
GRAND_TOTAL_MAX_USD = 300.0
POOL_MAX_USD = {"anthropic": 200.0, "openai": 50.0, "other": 50.0}  # 'other' = Kimi + Gemini + any other provider, COMBINED
POOL_CAP_ENV = {"anthropic": "CASEPATCH_POOL_CAP_ANTHROPIC_USD", "openai": "CASEPATCH_POOL_CAP_OPENAI_USD",
                "other": "CASEPATCH_POOL_CAP_OTHER_USD"}
NAMED_POOLS = ("anthropic", "openai")


def pool_of(provider):
    return provider if provider in NAMED_POOLS else "other"


def pool_filter(pool):
    """(op, providers) for the ledger: named pools match their provider; 'other' matches everything else (fail closed)."""
    return ("in", (pool,)) if pool in NAMED_POOLS else ("not in", NAMED_POOLS)
KEY_ENV = {"openai": ("OPENAI_API_KEY",), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "kimi": ("MOONSHOT_API_KEY",),
           "anthropic": ("ANTHROPIC_API_KEY",), "fal": ("FAL_KEY",)}
PROVIDER_CAP_ENV = {"openai": "CASEPATCH_PROVIDER_CAP_OPENAI_USD", "gemini": "CASEPATCH_PROVIDER_CAP_GEMINI_USD",
                    "kimi": "CASEPATCH_PROVIDER_CAP_KIMI_USD", "anthropic": "CASEPATCH_PROVIDER_CAP_ANTHROPIC_USD",
                    "fal": "CASEPATCH_PROVIDER_CAP_FAL_USD"}
_KEY_RX = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|msk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|(?i:bearer)\s+[A-Za-z0-9._\-]{8,}|"
                     r"(?i:\bkey)\s+[A-Za-z0-9:._\-]{16,}|"
                     r"(?i:x-goog-api-key|api[_-]?key|authorization)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{8,})")


class HostedDisabled(Exception):
    """Hosted calls are off; .args[0] says why (never contains a key)."""


def redact(text, extra=()):
    s = str(text)
    for k in extra:
        if k:
            s = s.replace(k, "[REDACTED_KEY]")
    return _KEY_RX.sub("[REDACTED_KEY]", s)


def _bounded_int(env, name, default, lo, hi):
    v = env.get(name, "")
    if v == "":
        return default
    try:
        n = int(v)
    except ValueError:
        raise HostedDisabled(f"{name} must be an integer")
    if not lo <= n <= hi:
        raise HostedDisabled(f"{name} must be between {lo} and {hi}")
    return n


@dataclass(frozen=True)
class HostedConfig:
    provider: str
    cap_usd: float
    ledger_path: Path
    max_output_tokens: int = 2048
    max_images_per_call: int = 2
    max_image_side: int = 1024
    max_calls_per_case: int = 40
    max_retries: int = 1
    timeout_s: float = 60.0
    provider_cap_usd: float | None = None  # optional sub-budget for this provider inside its pool
    pool: str = "other"
    pool_cap_usd: float | None = None      # hard cap for the provider's pool (anthropic / openai / other)
    _key: str = field(default="", repr=False, compare=False)

    def key(self):
        return self._key

    def __repr__(self):  # never show the key
        return (f"HostedConfig(provider={self.provider!r}, cap_usd={self.cap_usd}, ledger={str(self.ledger_path)!r}, "
                f"max_output_tokens={self.max_output_tokens}, max_images_per_call={self.max_images_per_call}, "
                f"max_image_side={self.max_image_side}, max_calls_per_case={self.max_calls_per_case}, max_retries={self.max_retries})")


def parse_cap(raw):
    if raw is None or str(raw).strip() == "":
        raise HostedDisabled("CASEPATCH_SPEND_CAP_USD is not set (no default cap: hosted calls refused)")
    try:
        cap = float(str(raw).strip())
    except ValueError:
        raise HostedDisabled("CASEPATCH_SPEND_CAP_USD is not a number")
    if not math.isfinite(cap) or cap <= 0:
        raise HostedDisabled("CASEPATCH_SPEND_CAP_USD must be a positive finite number")
    if cap > GRAND_TOTAL_MAX_USD:
        raise HostedDisabled(f"CASEPATCH_SPEND_CAP_USD above the approved grand total ${GRAND_TOTAL_MAX_USD:.0f} is refused")
    return cap


def parse_pool_cap(env, pool, total_cap):
    """Hard pool cap. Default = the approved maximum (bounded by the total); present but invalid or above max => refused."""
    name = POOL_CAP_ENV[pool]; mx = POOL_MAX_USD[pool]
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return min(mx, total_cap)
    try:
        v = float(str(raw).strip())
    except ValueError:
        raise HostedDisabled(f"{name} is not a number")
    if not math.isfinite(v) or v <= 0:
        raise HostedDisabled(f"{name} must be a positive finite number")
    if v > mx + 1e-9:
        raise HostedDisabled(f"{name} above the approved ${mx:.0f} for the {pool} pool is refused")
    if v > total_cap:
        raise HostedDisabled(f"{name} exceeds the total CASEPATCH_SPEND_CAP_USD")
    return v


def parse_provider_cap(env, provider, total_cap):
    """Optional per-provider sub-budget inside the shared total cap. Present but invalid => refused."""
    raw = env.get(PROVIDER_CAP_ENV[provider])
    if raw is None or str(raw).strip() == "":
        return None
    try:
        v = float(str(raw).strip())
    except ValueError:
        raise HostedDisabled(f"{PROVIDER_CAP_ENV[provider]} is not a number")
    if not math.isfinite(v) or v <= 0:
        raise HostedDisabled(f"{PROVIDER_CAP_ENV[provider]} must be a positive finite number")
    if v > total_cap:
        raise HostedDisabled(f"{PROVIDER_CAP_ENV[provider]} exceeds the total CASEPATCH_SPEND_CAP_USD")
    return v


def provider_from(env):
    p = (env.get("CASEPATCH_HOSTED_PROVIDER") or "openai").strip().lower()
    if p not in KEY_ENV:
        raise HostedDisabled("CASEPATCH_HOSTED_PROVIDER must be 'openai', 'gemini', 'kimi', 'anthropic' or 'fal'")
    return p


def key_present(env, provider):
    return any(env.get(n) for n in KEY_ENV[provider])


def ledger_path(env):
    from ..paths import home
    return Path(env.get("CASEPATCH_SPEND_LEDGER") or (home(env) / "spend.sqlite")).expanduser()


def load_config(env=None, provider=None):
    """Return a HostedConfig, or raise HostedDisabled. Disabled unless CASEPATCH_HOSTED=1 AND a valid cap AND a key."""
    env = os.environ if env is None else env
    if env.get("CASEPATCH_HOSTED") != "1":
        raise HostedDisabled("hosted calls are disabled (set CASEPATCH_HOSTED=1 to enable)")
    cap = parse_cap(env.get("CASEPATCH_SPEND_CAP_USD"))
    provider = provider_from(env) if provider is None else provider_from({"CASEPATCH_HOSTED_PROVIDER": provider})
    key = next((env[n] for n in KEY_ENV[provider] if env.get(n)), "")
    if not key:
        raise HostedDisabled(f"no API key in the environment for {provider} ({' or '.join(KEY_ENV[provider])})")
    return HostedConfig(
        provider=provider, cap_usd=cap, ledger_path=ledger_path(env),
        max_output_tokens=_bounded_int(env, "CASEPATCH_MAX_OUTPUT_TOKENS", 2048, 64, 8192),
        max_images_per_call=_bounded_int(env, "CASEPATCH_MAX_IMAGES_PER_CALL", 2, 1, 4),
        max_image_side=_bounded_int(env, "CASEPATCH_MAX_IMAGE_SIDE", 1024, 128, 2048),
        max_calls_per_case=_bounded_int(env, "CASEPATCH_MAX_CALLS_PER_CASE", 40, 1, 500),
        max_retries=_bounded_int(env, "CASEPATCH_MAX_RETRIES", 1, 0, 3),
        provider_cap_usd=parse_provider_cap(env, provider, cap),
        pool=pool_of(provider), pool_cap_usd=parse_pool_cap(env, pool_of(provider), cap),
        _key=key)
