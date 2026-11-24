"""In-process local engine backend: the file never leaves this machine and no API key is used.
Needs the local extra (pip install "casepatch[local]") and the model files (`casepatch setup-local`, one-time)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .core import CASE_SPLIT_REFUSED, CasePatchError, Result, copy_out, report_fields

HOSTED_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MOONSHOT_API_KEY",
              "CASEPATCH_BYOK_ANTHROPIC_KEY")


def derive_case_key(case_key: str) -> bytes:
    """Engine key for a caller's case_key (local mode): same case_key + same input -> same fakes, on any machine."""
    import hashlib, hmac
    ck = str(case_key)
    if not (16 <= len(ck) <= 200) or not ck.isprintable():
        raise CasePatchError("case_key must be 16-200 printable characters (use a long random string and keep it secret).")
    return hmac.new(b"casepatch-case-key/1", ("local\x1f" + ck).encode(), hashlib.sha256).digest()


def missing_engine_module() -> str | None:
    """The first module of the local engine (pip install "casepatch[local]") that cannot be imported, or None."""
    try:
        from casepatch.docs import pipeline  # noqa: F401
        import cv2, pymupdf, rapidocr  # noqa: F401
    except ImportError as e:
        return e.name or "casepatch[local]"
    return None


class LocalBackend:
    def __init__(self, verify: bool = True, keep=None):
        self.verify = verify
        self.keep = list(keep or [])  # terms never replaced (casepatch.keep)
        missing = missing_engine_module()
        if missing:
            raise CasePatchError(f'Local mode needs the local engine: pip install "casepatch[local]" (missing: {missing}).')

    def anonymize(self, src: Path, dst: Path, case_key: bytes | None = None) -> Result:
        """One file as its own case. `case_key`: reuse a case key (same replacements as other files of that case)."""
        with _engine_env():
            tmp = Path(tempfile.mkdtemp(prefix="casepatch-"))
            try:
                os.chmod(tmp, 0o700)
                inp = tmp / "input"; inp.mkdir()
                shutil.copyfile(src, inp / ("01_document" + src.suffix.lower()))
                if case_key:
                    (tmp / "review").mkdir(mode=0o700)
                    (tmp / "review" / "case.key").write_bytes(case_key)
                report, docs, occs = self._run(inp, tmp, {"01_document" + src.suffix.lower(): src.name})
                outs = sorted(p for p in (tmp / "out").iterdir() if p.name.startswith("document-"))
                if len(outs) != 1:
                    raise CasePatchError("Local engine produced no output document.")
                copy_out(outs[0], dst)
                return _result(src, dst, report, [d.id for d in docs], occs)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    def anonymize_case(self, items: list[tuple[Path, Path]], case_key: bytes | None = None) -> dict[Path, Result]:
        """Several files as ONE case (one identity map; values found in one file are searched in all). If the case
        cannot run as a whole, every file reports the error and nothing is written (never a silent per-file split)."""
        with _engine_env():
            tmp = Path(tempfile.mkdtemp(prefix="casepatch-"))
            try:
                os.chmod(tmp, 0o700)
                inp = tmp / "input"; inp.mkdir()
                staged = {}
                for i, (src, _) in enumerate(items):
                    name = f"{i:04d}_document{src.suffix.lower()}"  # keeps the folder order; no source names on disk
                    shutil.copyfile(src, inp / name)
                    staged[name] = src
                if case_key:
                    (tmp / "review").mkdir(mode=0o700)
                    (tmp / "review" / "case.key").write_bytes(case_key)
                try:
                    report, docs, occs = self._run(inp, tmp, {n: s.name for n, s in staged.items()})
                except CasePatchError as e:
                    msg = CASE_SPLIT_REFUSED.format(err=e.message)
                    return {src: Result(source=str(src), output=None, mode="local", status="error", error=msg) for src, _ in items}
                fmap = json.loads((tmp / "review" / "file-map.json").read_text())
                by_src = {staged[name]: did for did, name in fmap.items()}
                out = {}
                for src, dst in items:
                    did = by_src.get(src)
                    o = next((p for p in (tmp / "out").iterdir() if did and p.stem == did), None)
                    if o is None:
                        out[src] = Result(source=str(src), output=None, mode="local", status="error", error="Local engine produced no output document.")
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    copy_out(o, dst)
                    out[src] = _result(src, dst, report, [did], occs)
                return out
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    def _run(self, inp, tmp, names):
        """`names`: staged file name -> the user's file name, so an engine error names the user's file and never the
        private staging folder."""
        from casepatch.docs import pipeline
        try:
            report, docs, occs, regions, _ = pipeline.run(inp, tmp / "out", detector="rules", hosted=None,
                                                          review_dir=tmp / "review", verify=self.verify,
                                                          keep=self.keep)
        except Exception as e:  # engine errors carry codes, not content
            raise CasePatchError(engine_error(e, tmp, names)) from None
        return report, docs, occs


def engine_error(e: BaseException, tmp: Path, names: dict) -> str:
    """A plain one-line message for an engine failure: which of the user's files (when known) and the engine's reason,
    without exception class names or internal temporary paths."""
    import re
    detail = str(e)[:300]
    which = None
    for staged, real in names.items():
        if staged in detail:
            which = real
    root = re.escape(str(tmp))
    detail = re.sub(r"(filename=)?['\"]?" + root + r"[^\s'\"]*['\"]?", "the file", detail)
    for staged, real in names.items():
        detail = detail.replace(staged, real)
    detail = re.sub(r"\s+", " ", detail).strip().rstrip(".")
    if which is None and len(names) == 1:
        which = next(iter(names.values()))
    head = f"{which} could not be processed" if which else "The local engine could not process the files"
    return f"{head}: {detail}." if detail else f"{head}."


class _engine_env:
    """Local rules engine only: never calls a hosted model, whatever keys are in the environment."""
    def __enter__(self):
        self.prev = os.environ.get("CASEPATCH_HOSTED")
        os.environ["CASEPATCH_HOSTED"] = "0"
        self.saved = {k: os.environ.pop(k) for k in HOSTED_ENV if k in os.environ}

    def __exit__(self, *exc):
        os.environ.update(self.saved)


def _result(src, dst, report, doc_ids, occs) -> Result:
    summary: dict[str, int] = {}
    pages = 0
    for did in doc_ids:
        d = (report.get("documents") or {}).get(did) or {}
        pages += int(d.get("pages") or 0)
        for k, n in (d.get("replacements_by_kind") or {}).items():
            summary[k] = summary.get(k, 0) + int(n)
    changes, seen = [], set()
    for o in occs:
        key = (o.kind, o.text, o.replacement)
        if o.doc in doc_ids and o.replacement and key not in seen:
            seen.add(key)
            changes.append({"id": f"f{len(changes)}", "type": "text", "kind": o.kind, "original": o.text,
                            "replacement": o.replacement, "entity": getattr(o, "entity", None)})
    fields = report_fields(report, docs=doc_ids)
    fields["verification"] = {"residual_original_values": fields["gate"].get("residual_original_values"),
                              "pii_like_text_not_replacements": fields["gate"].get("pii_like_text_not_replacements"),
                              "scope": "counts detected values only; a value CasePatch never detected is not counted"}
    return Result(source=str(src), output=str(dst), mode="local", pages=pages, summary=summary, changes=changes, **fields)


# what each optional item that `doctor()` can report missing does, and how to add it (shown by `casepatch setup-local`)
OPTIONAL_NOTES = {
    "torch": 'optional: LaMa inpainting under replaced text on scans/photos (big-lama.pt) runs on torch; without it the '
             'engine uses OpenCV inpainting. To add it: pip install "casepatch[local-hq]"',
    "fc_match": "optional: fontconfig's fc-match is not installed; replacement text uses the platform's font folders and "
                "the bundled fonts instead",
}


def setup(quiet: bool = False) -> dict:
    """Download the pinned model files (inpainting + face models, ~210 MB, SHA-256 checked) to the model folder
    (CASEPATCH_MODEL_DIR, else $XDG_CACHE_HOME/casepatch/models, else ~/.cache/casepatch/models; see casepatch.paths).
    Checks the local engine first, so nothing is downloaded on an install that could not use the files."""
    missing = missing_engine_module()
    if missing:
        raise CasePatchError(f'setup-local needs the local engine first: pip install "casepatch[local]" (missing: {missing}). '
                             "Nothing was downloaded.")
    from casepatch_studio.assets import DownloadError, doctor, fetch
    try:
        fetch(quiet=quiet)
    except DownloadError as e:
        raise CasePatchError(f"setup-local: {e}") from None
    report = doctor()
    notes = {p: OPTIONAL_NOTES.get(p, f'needed by the local engine: pip install "casepatch[local]" (missing: {p})')
             for p in report["missing_packages"]}
    if not report["fc_match"]:
        notes["fc_match"] = OPTIONAL_NOTES["fc_match"]
    report["local_engine_ready"] = not report["missing_assets"] and all(p in OPTIONAL_NOTES for p in report["missing_packages"])
    report["notes"] = notes
    from ..paths import describe
    report["locations"] = describe()
    return report
