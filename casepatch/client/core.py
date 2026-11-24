from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]
Mode = Literal["auto", "local", "cloud"]
DOCUMENTS = (".pdf", ".png", ".jpg", ".jpeg")
TRACES = (".json", ".jsonl", ".ndjson", ".txt")  # agent traces and attachments (casepatch.docs.textual)
SUPPORTED = DOCUMENTS + TRACES
CASE_SPLIT_REFUSED = ("The files could not run as ONE case (one identity map), so none was written: {err} "
                      "Fix or remove the failing file and run the folder again.")
DEFAULT_BASE_URL = "https://casepatch.com"


class CasePatchError(Exception):
    """An API or engine error. `status` is the HTTP status for cloud calls (0 for local/engine errors)."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Result:
    """One anonymized document. `changes` lists each replaced value (original -> replacement, kind, entity);
    `summary` counts replacements by kind. Keep `changes` private: it contains the original values."""
    source: str
    output: Optional[str]
    mode: str
    status: str = "done"
    pages: int = 0
    summary: dict[str, int] = field(default_factory=dict)
    changes: list[dict[str, Any]] = field(default_factory=list)
    job_id: Optional[str] = None
    error: Optional[str] = None
    # what the engine's report says about this output (report.json / hosted job JSON). Codes and counts, never values.
    engine_version: Optional[str] = None
    is_synthetic: bool = False             # never claimed: values the detectors missed stay original
    source_sent_to_cloud: bool = False     # True when the source file was uploaded (cloud mode)
    likely_misses: list[dict[str, Any]] = field(default_factory=list)  # structural signs a value may still be original
    verification: dict[str, Any] = field(default_factory=dict)        # counts of DETECTED values still readable
    needs_review: int = 0
    gate: dict[str, Any] = field(default_factory=dict)  # verified / pass / counts; pass is not a privacy certification

    @property
    def ok(self) -> bool:
        return self.status == "done"

    @property
    def review_required(self) -> bool:
        """True unless verification ran and found no residual detected value, no likely miss and nothing to review.
        Even then, values CasePatch never detected are not counted: review outputs before sharing them."""
        g = self.gate
        clean = (g.get("verified") is True and g.get("pass") is True and not g.get("residual_original_values")
                 and not self.likely_misses and not g.get("likely_misses") and not self.needs_review and not g.get("needs_review"))
        return not clean

    def review_line(self) -> str:
        """One line for people: what to check before sharing (no values)."""
        if not self.ok:
            return ""
        if not self.gate:
            return "review before sharing: this engine did not report a verification gate"
        if not self.gate.get("verified"):
            return "review before sharing: verification did not run (undetected values are never counted)"
        parts = [f"{self.gate.get('residual_original_values', 0)} detected value(s) still readable",
                 f"{len(self.likely_misses)} likely miss(es)", f"{self.needs_review} item(s) to review"]
        head = "verification found nothing to fix" if not self.review_required else "check before sharing"
        return f"{head}: " + ", ".join(parts) + " (values CasePatch never detected are not counted)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def local_available() -> bool:
    """True when the local engine and its dependencies (pip install "casepatch[local]") are importable."""
    try:
        import cv2, numpy, pymupdf, rapidocr  # noqa: F401
        from casepatch.docs import pipeline  # noqa: F401
        return True
    except Exception:
        return False


def local_partly_installed() -> bool:
    """True when part of the local engine is installed (RapidOCR, which only the `local` extra brings in) while the
    engine itself cannot load: auto mode then refuses instead of quietly uploading to the hosted API."""
    import importlib.util
    try:
        return importlib.util.find_spec("rapidocr") is not None
    except (ImportError, ValueError):
        return False


def report_fields(rep: dict, docs: Optional[Iterable[str]] = None, sent: bool = False) -> dict[str, Any]:
    """Result fields from an engine report.json (local) or a hosted job JSON. `docs`: keep only these documents'
    likely misses (one file of a case); the gate is the case's."""
    wanted = set(docs) if docs is not None else None
    misses = [m for m in rep.get("likely_misses") or []
              if wanted is None or (m.get("doc") or m.get("document")) in wanted]
    gate = dict(rep.get("gate") or {})
    if not gate and "verification" in rep and "needs_review" in rep:  # hosted job JSON without a gate block
        v = rep.get("verification") or {}
        gate = {"verified": v.get("residual_original_values") is not None,
                "residual_original_values": v.get("residual_original_values"),
                "likely_misses": len(rep.get("likely_misses") or []), "needs_review": rep.get("needs_review")}
        gate["pass"] = bool(gate["verified"] and not gate["residual_original_values"] and not gate["likely_misses"])
    nr = rep.get("needs_review")
    return {"engine_version": rep.get("casepatch") or rep.get("engine_version"),
            "is_synthetic": False,
            "source_sent_to_cloud": bool(sent or rep.get("source_sent_to_cloud")),
            "likely_misses": misses,
            "verification": dict(rep.get("verification") or {}) if sent else {},  # local: set from the gate by the caller
            "needs_review": len(nr) if isinstance(nr, list) else int(nr or gate.get("needs_review") or 0),
            "gate": gate}


def precheck(src: Path, deep: bool = False) -> Optional[str]:
    """A reason the file cannot join a case (type/content), or None. Cheap and local: textual files are fully validated
    (UTF-8, JSON/JSONL syntax, size and record bounds), documents by their signature; `deep` (local mode) also opens
    the document the way the engine will."""
    ext = src.suffix.lower()
    if ext not in SUPPORTED:
        return f"Unsupported file type {ext!r}: use PDF, PNG, JPG, JSON, JSONL or TXT."
    if ext in TRACES:
        from ..docs.textual import TextualError, fmt_of, parse
        try:
            parse(src.read_bytes(), fmt_of(ext))
        except TextualError as e:
            return f"{src.name}: {e.code}" + (f" ({e.where})" if e.where else "")
        return None
    if src.stat().st_size == 0:
        return f"{src.name} is empty (0 bytes)."
    with open(src, "rb") as f:
        head = f.read(8)
    want = {".pdf": (b"%PDF-",), ".png": (b"\x89PNG\r\n\x1a\n",), ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",)}[ext]
    if not head.startswith(want):
        return f"{src.name}: the contents are not a {ext[1:].upper()} file."
    return _unreadable(src, ext) if deep else None


def _unreadable(src: Path, ext: str) -> Optional[str]:
    """Open the document the way the engine will, when the engine's libraries are installed (local mode); the cloud-only
    install has no PDF/image libraries, and the hosted API checks uploads itself."""
    if ext == ".pdf":
        try:
            import pymupdf
        except ImportError:
            return None
        try:
            pymupdf.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass
        try:
            with pymupdf.open(src, filetype="pdf") as d:
                if d.needs_pass:
                    return f"{src.name} is password-protected; remove the password and try again."
                if d.page_count < 1:
                    return f"{src.name} could not be read as a PDF (it may be corrupt or empty)."
                d.load_page(0)
        except Exception:
            return f"{src.name} could not be read as a PDF (it may be corrupt or empty)."
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(src) as im:
            im.load()
    except Image.DecompressionBombError:  # very large but readable: the engine reports the size limit
        return None
    except Exception:
        return f"{src.name} could not be read as a {ext[1:].upper()} image (it may be corrupt or truncated)."
    return None


def default_out(src: Path) -> Path:
    return src.with_name(f"document.anonymized{src.suffix.lower()}")


class CasePatch:
    """mode="local": run the CasePatch engine in this process (no upload, no API key, nothing leaves the machine).
    mode="cloud": send the file to the hosted API with an API key (CASEPATCH_API_KEY).
    mode="auto" (default, or the CASEPATCH_MODE environment variable): local when the local engine is installed,
    otherwise cloud. When the local engine is only partly installed, auto mode raises instead of uploading."""

    def __init__(self, api_key: Optional[str] = None, mode: Optional[Mode] = None, base_url: Optional[str] = None,
                 timeout: float = 900.0, poll_interval: float = 1.5, transport: Optional[Callable[..., Any]] = None,
                 verify: bool = True, case_key: Optional[str] = None, keep: Optional[Iterable[str] | str] = None):
        mode = mode or (os.environ.get("CASEPATCH_MODE") or "").strip().lower() or "auto"  # type: ignore[assignment]
        if mode not in ("auto", "local", "cloud"):
            raise ValueError('mode must be "auto", "local" or "cloud" (also for CASEPATCH_MODE)')
        self.api_key = api_key or os.environ.get("CASEPATCH_API_KEY") or None
        self.case_key = case_key or os.environ.get("CASEPATCH_CASE_KEY") or None
        from ..keep import KeepListError, parse_keep
        try:  # terms never replaced: your own bank/merchant name, a product, a city, a support address
            self.keep: list[str] = parse_keep(keep)
        except KeepListError as e:
            raise CasePatchError(str(e)) from None
        if mode == "auto":
            if local_available():
                mode = "local"
            elif local_partly_installed():  # the user meant to run locally: never fall back to an upload
                raise CasePatchError('The local engine is installed but cannot load, so auto mode will not upload your files. '
                                     'Reinstall it with: pip install --force-reinstall "casepatch[local]", or pass mode="cloud" '
                                     '(CLI: --cloud) to use the hosted API on purpose.')
            else:
                mode = "cloud"
        self.mode: str = mode
        if mode == "cloud":
            from .cloud import CloudBackend
            if not self.api_key:
                raise CasePatchError("Cloud mode needs an API key: pass api_key=... or set CASEPATCH_API_KEY. "
                                     'For local mode (no key, nothing uploaded) install "casepatch[local]".')
            self._backend: Any = CloudBackend(self.api_key, (base_url or os.environ.get("CASEPATCH_BASE_URL") or DEFAULT_BASE_URL),
                                              timeout=timeout, poll_interval=poll_interval, transport=transport)
            self._backend.keep = list(self.keep)
        else:
            from .local import LocalBackend
            self._backend = LocalBackend(verify=verify, keep=self.keep)
        self._engine_key = None
        if self.case_key is not None and self.mode == "local":
            from .local import derive_case_key
            self._engine_key = derive_case_key(self.case_key)

    # ---- one file
    def anonymize(self, path: PathLike, out: Optional[PathLike] = None, *, overwrite: bool = False) -> Result:
        """Anonymize one PDF/PNG/JPG document or JSON/JSONL/TXT trace and write the result to `out`
        (default: document.anonymized.<ext> next to it; explicit output paths are kept)."""
        src = Path(path)
        if not src.is_file():
            if src.is_dir():
                raise IsADirectoryError(f"{src} is a folder: use anonymize_folder (CLI: the same command handles folders).")
            raise FileNotFoundError(f"File not found: {src}")
        bad = precheck(src, deep=getattr(self, "mode", None) == "local")
        if bad:
            raise CasePatchError(bad)
        dst = Path(out) if out is not None else default_out(src)
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} exists (use --overwrite, or overwrite=True in Python, to replace it)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "local":
            return self._backend.anonymize(src, dst, case_key=self._engine_key)
        return self._backend.anonymize(src, dst, case_key=self.case_key)

    # ---- a folder
    def anonymize_folder(self, folder: PathLike, out_dir: PathLike, *, recursive: bool = False, overwrite: bool = False,
                         report: Optional[PathLike] = None, on_result: Optional[Callable[[Result], None]] = None) -> list[Result]:
        """Anonymize every document (PDF/PNG/JPG) and trace (JSON/JSONL/TXT) in `folder` into `out_dir` (neutral numbered
        file names, without source folder names). The folder is ONE case in both modes: one identity map across all files (same person / account / card /
        date -> same replacement in every document and trace, including JSON fields such as expected outputs and tool
        arguments). Cloud mode sends the folder as one case job (a ZIP), so `entity` ids are shared by every file.
        A file that fails the type/content check is reported (status "error") and left out; the others still run as
        one case. If the case itself fails, every file reports that error: a case is never silently split into files
        with separate identity maps. `report`: optional JSONL file, one line per document."""
        src, dst = Path(folder), Path(out_dir)
        if not src.is_dir():
            if src.exists():
                raise NotADirectoryError(f"{src} is a file, not a folder: use anonymize for one file.")
            raise FileNotFoundError(f"Folder not found: {src}")
        results: list[Result] = []
        files = list(iter_documents(src, recursive))
        case = getattr(self._backend, "anonymize_case", None)
        if case is not None and len(files) > 1:
            # local mode: the folder is ONE case (one identity map: the same person, account, card or date gets the
            # same replacement in every file, and a value found in one file is searched in all of them)
            todo, pre = [], {}
            for i, f in enumerate(files, 1):
                target = dst / f"document-{i:02d}{f.suffix.lower()}"
                bad = precheck(f, deep=getattr(self, "mode", None) == "local")
                if bad:
                    pre[f] = Result(source=str(f), output=None, mode=self.mode, status="error",
                                    error=f"{bad} Not included in the case.")
                elif target.exists() and not overwrite:
                    pre[f] = Result(source=str(f), output=None, mode=self.mode, status="error",
                                    error=f"{target} exists (use --overwrite / overwrite=True to replace it)")
                else:
                    todo.append((f, target))
            done = case(todo, case_key=self._engine_key if self.mode == "local" else self.case_key) if todo else {}
            for f in files:
                r = pre.get(f) or done[f]
                results.append(r)
                if on_result:
                    on_result(r)
        else:
            for i, f in enumerate(files, 1):
                target = dst / f"document-{i:02d}{f.suffix.lower()}"
                try:
                    r = self.anonymize(f, target, overwrite=overwrite)
                except (CasePatchError, FileExistsError, OSError) as e:
                    r = Result(source=str(f), output=None, mode=self.mode, status="error", error=str(e))
                results.append(r)
                if on_result:
                    on_result(r)
        if report is not None:
            Path(report).parent.mkdir(parents=True, exist_ok=True)
            with open(report, "w", encoding="utf-8") as fh:
                for r in results:
                    fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return results

    # ---- cloud-only extras
    def usage(self) -> dict[str, Any]:
        """Free pages left this month and credit balance (cloud mode)."""
        return self._cloud().usage()

    def submit(self, path: PathLike) -> dict[str, Any]:
        """Start a cloud job without waiting; returns the job (id, status)."""
        return self._cloud().submit(Path(path))

    def job(self, job_id: str) -> dict[str, Any]:
        return self._cloud().job(job_id)

    def download(self, job_id: str, out: PathLike) -> Path:
        return self._cloud().download(job_id, Path(out))

    def delete(self, job_id: str) -> None:
        self._cloud().delete(job_id)

    def _cloud(self):
        if self.mode != "cloud":
            raise CasePatchError("This call needs cloud mode (CasePatch(mode='cloud', api_key=...)).")
        return self._backend


def iter_documents(folder: Path, recursive: bool = False) -> Iterable[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."))


def copy_out(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
