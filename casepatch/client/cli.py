"""`casepatch anonymize PATH --out OUT [--local|--cloud]` and `casepatch setup-local`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import CasePatch, CasePatchError


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
    try:
        cp = CasePatch(api_key=args.api_key, mode=args.mode, base_url=args.base_url,
                       case_key=getattr(args, "case_key", None))
        src = Path(args.path)
        print(f"mode: {cp.mode}" + (" (files are uploaded to the CasePatch API)" if cp.mode == "cloud" else " (nothing is uploaded)"),
              file=sys.stderr)
        if src.is_dir():
            def show(r):
                print(f"{'ok ' if r.ok else 'ERR'} {r.source} -> {r.output or r.error}  {r.summary if r.ok else ''}")
                if r.ok:
                    print(f"    {r.review_line()}")
            rs = cp.anonymize_folder(src, args.out, recursive=args.recursive, overwrite=args.overwrite,
                                     report=args.report, on_result=show)
            bad = sum(not r.ok for r in rs)
            print(f"{len(rs) - bad}/{len(rs)} documents anonymized into {args.out}")
            from .core import SUPPORTED
            it = src.rglob("*") if args.recursive else src.iterdir()
            other = sorted({p.suffix.lower() or "(no extension)" for p in it if p.is_file() and not p.name.startswith(".")
                            and p.suffix.lower() not in SUPPORTED})
            if other:
                n = sum(1 for p in (src.rglob("*") if args.recursive else src.iterdir()) if p.is_file()
                        and not p.name.startswith(".") and p.suffix.lower() not in SUPPORTED)
                print(f"skipped {n} file(s) of unsupported types ({', '.join(other)}): not part of the case", file=sys.stderr)
            return 1 if bad else 0
        r = cp.anonymize(src, args.out, overwrite=args.overwrite)
        if args.report:
            Path(args.report).write_text(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        print(f"{r.output}  {r.summary}")
        if r.ok:
            print(r.review_line())
        return 0 if r.ok else 1
    except (CasePatchError, FileNotFoundError, FileExistsError, NotADirectoryError) as e:
        print(f"casepatch: {e}", file=sys.stderr)
        return 2
