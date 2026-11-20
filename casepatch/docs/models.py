"""Local model file lookup. Models are never downloaded during a case run."""
from __future__ import annotations

import os
from pathlib import Path


def model_dirs():
    from ..paths import LEGACY_MODEL_DIR, model_dir
    dirs = []
    if os.environ.get("CASEPATCH_MODEL_DIR"):
        dirs.append(Path(os.environ["CASEPATCH_MODEL_DIR"]).expanduser())
    dirs += [Path(__file__).resolve().parents[3] / "models", model_dir(), LEGACY_MODEL_DIR.expanduser(),
             Path(os.environ.get("CASEPATCH_RAW", Path.home() / ".cache" / "casepatch-real")) / "models"]
    return list(dict.fromkeys(dirs))


def find_model(name):
    for d in model_dirs():
        if (d / name).exists():
            return d / name
    return None
