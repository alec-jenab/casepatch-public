"""`casepatch anonymize PATH --out OUT [--local|--cloud]` and `casepatch setup-local`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import SUPPORTED, CasePatch, CasePatchError

SUPPORTED_TEXT = "supported: PDF, PNG, JPG/JPEG documents and JSON, JSONL/NDJSON, TXT traces"


def unsupported_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Files in the folder that `anonymize` does not read (hidden files are ignored)."""
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and not p.name.startswith(".") and p.suffix.lower() not in SUPPORTED)


def skipped_note(folder: Path, files: list[Path], limit: int = 10) -> str:
    names = [str(p.relative_to(folder)) for p in files]
    shown = ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")
    return f"skipped {len(names)} file(s) of unsupported types, not part of the case: {shown} ({SUPPORTED_TEXT})"


def add_parsers(commands) -> None:
    a = commands.add_parser("anonymize", help="Anonymize a case: PDF/PNG/JPG documents and JSON/JSONL/TXT agent traces, "
                                              "one file or a folder (one identity map across the folder; local engine or hosted API)")
    a.add_argument("path", help="A file or a case folder (documents and traces together)")
    a.add_argument("--out", required=True, help="Output file (for one file) or folder (for a folder)")
    m = a.add_mutually_exclusive_group()
    m.add_argument("--local", dest="mode", action="store_const", const="local", help="Run on this machine (no upload, no key)")
    m.add_argument("--cloud", dest="mode", action="store_const", const="cloud", help="Use the hosted API (CASEPATCH_API_KEY)")
    a.add_argument("--api-key", help="API key for --cloud (default: $CASEPATCH_API_KEY)")
    a.add_argument("--base-url", help=argparse.SUPPRESS)
    a.add_argument("--case-key", help="Reproducible fakes: same input + same case key -> same replacements "
                                      "(default: $CASEPATCH_CASE_KEY; 16+ characters, keep it secret)")
    a.add_argument("--keep", action="append", metavar="TERM", default=[],
                   help="Never replace this term (case-insensitive, whole phrase), e.g. your own bank name; repeatable")
    a.add_argument("--keep-file", metavar="PATH", help="Terms never to replace, one per line ('#' starts a comment)")
    a.add_argument("--recursive", action="store_true", help="Include subfolders")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--report", help="Write one JSON line per document (includes original values: keep it private)")
    commands.add_parser("setup-local", help="Download the local engine's model files (~210 MB, one time)")


def run(args) -> int:
    if args.command == "setup-local":
        from .local import setup
        try:
            print(json.dumps(setup(), indent=1))
        except (CasePatchError, OSError) as e:  # one line with the fix, never a traceback
            print(f"casepatch: {e}", file=sys.stderr)
            return 2
        return 0
    if not Path(args.path).exists():
        print(f"casepatch: File or folder not found: {args.path}", file=sys.stderr)
        return 2
    try:
        keep = list(getattr(args, "keep", None) or [])
        if getattr(args, "keep_file", None):
            from ..keep import KeepListError, read_keep_file
            try:
                keep += read_keep_file(args.keep_file)
            except (KeepListError, OSError) as e:
                raise CasePatchError(f"Could not read the keep file: {e}") from None
        cp = CasePatch(api_key=args.api_key, mode=args.mode, base_url=args.base_url,
                       case_key=getattr(args, "case_key", None), keep=keep)
        src = Path(args.path)
        print(f"mode: {cp.mode}" + (" (files are uploaded to the CasePatch API)" if cp.mode == "cloud" else " (nothing is uploaded)"),
              file=sys.stderr)
        if src.is_dir():
            def show(r):
                print(f"{'ok ' if r.ok else 'ERR'} {r.source} -> {r.output or r.error}  {r.summary if r.ok else ''}")
                if r.ok:
                    print(f"    {r.review_line()}")
            skipped = unsupported_files(src, args.recursive)
            if skipped:
                print(skipped_note(src, skipped), file=sys.stderr)
            rs = cp.anonymize_folder(src, args.out, recursive=args.recursive, overwrite=args.overwrite,
                                     report=args.report, on_result=show)
            bad = sum(not r.ok for r in rs)
            if not rs:
                print(f"casepatch: no supported files in {args.path} ({SUPPORTED_TEXT}).", file=sys.stderr)
                return 2
            print(f"{len(rs) - bad}/{len(rs)} documents anonymized into {args.out}")
            return 1 if bad else 0
        r = cp.anonymize(src, args.out, overwrite=args.overwrite)
        if args.report:
            Path(args.report).write_text(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        print(f"{r.output}  {r.summary}")
        if r.ok:
            print(r.review_line())
        return 0 if r.ok else 1
    except (CasePatchError, FileNotFoundError, FileExistsError, NotADirectoryError, IsADirectoryError) as e:
        print(f"casepatch: {e}", file=sys.stderr)
        return 2
