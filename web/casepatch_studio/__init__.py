"""CasePatch Studio: self-serve local web app for document case bundles (PDF, scans, photos, JSON/JSONL/TXT traces).
Ships in the casepatch wheel and reports the casepatch package version (one version in reports and exports)."""
try:
    from casepatch import __version__
except ImportError:  # a source checkout run without the repo root on sys.path
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from casepatch import __version__  # noqa: E402
