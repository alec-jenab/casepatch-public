"""Hosted frontier-model adapters (OpenAI / Gemini): built, disabled by default, spend-capped in code.

Nothing here makes a network call unless CASEPATCH_HOSTED=1, a valid CASEPATCH_SPEND_CAP_USD is set, the provider key
is present in the environment, and every call fits under the cap in the local SQLite spend ledger. See
docs/HOSTED_SETUP.md. The cap is enforced against CasePatch's own conservative estimates for calls made through
CasePatch only; it is not the provider's authoritative bill.
"""
from .config import HostedDisabled, HostedConfig, load_config, redact  # noqa: F401
from .ledger import CapExceeded, SpendLedger  # noqa: F401
