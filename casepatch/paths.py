"""Where CasePatch keeps files on this computer. One place, so every entry point agrees.

* home (`CASEPATCH_HOME`, default ~/.casepatch): settings, Studio cases (deleted after use), the numeric usage
  counters (usage.sqlite), opt-in event log, the hosted spend ledger, an optional stored BYOK key.
* cache (`CASEPATCH_CACHE_DIR`, else `$XDG_CACHE_HOME/casepatch`, else ~/.cache/casepatch): re-creatable files such as
  instanced fonts.
* models (`CASEPATCH_MODEL_DIR`, else <cache>/models): the pinned model files `casepatch setup-local` downloads.
  Models already in the legacy ~/.cache/casepatch/models are still found, so moving the cache never forces a
  re-download.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def home(env: Optional[Mapping[str, str]] = None) -> Path:
    return Path(_env(env).get("CASEPATCH_HOME") or Path.home() / ".casepatch").expanduser()


def cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    e = _env(env)
    if e.get("CASEPATCH_CACHE_DIR"):
        return Path(e["CASEPATCH_CACHE_DIR"]).expanduser()
    xdg = e.get("XDG_CACHE_HOME")
    if xdg and Path(xdg).expanduser().is_absolute():  # the XDG spec ignores relative values
        return Path(xdg).expanduser() / "casepatch"
    return Path.home() / ".cache" / "casepatch"


def model_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    e = _env(env)
    return Path(e["CASEPATCH_MODEL_DIR"]).expanduser() if e.get("CASEPATCH_MODEL_DIR") else cache_dir(e) / "models"


LEGACY_MODEL_DIR = Path("~/.cache/casepatch/models")


def describe(env: Optional[Mapping[str, str]] = None) -> dict:
    """The resolved locations (for setup-local output and the docs)."""
    return {"home": str(home(env)), "usage_counters": str(home(env) / "usage.sqlite"), "cache": str(cache_dir(env)),
            "models": str(model_dir(env))}
