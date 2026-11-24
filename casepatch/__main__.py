import argparse
import json
import os
import secrets
import sys
from importlib.resources import files
from pathlib import Path

from .core import PolicyError, transform
from .pricing import estimate


LOCAL_MODEL = "en_core_web_sm"
BUNDLE_SHAPE = ('a JSON case bundle {"schema_version": 1, "data_class": "synthetic", "scope": "<id>", "entities": [...], '
                '"records": [...]} (see `casepatch demo` and casepatch/fixtures/dispute.json)')
POLICY_HINTS = {
    "SCHEMA_VERSION": "transform expects " + BUNDLE_SHAPE + ".",
    "FIXTURE_ONLY": "transform is the structured-record engine for synthetic fixtures (data_class \"synthetic\"). "
                    "For documents and agent traces use: casepatch anonymize CASE_FOLDER --out OUT",
    "RECORDS_MUST_BE_LIST": "estimate expects a JSON list of records, or " + BUNDLE_SHAPE + ".",
    "RECORDS_REQUIRED": "the bundle needs a non-empty \"records\" list.",
    "LOCAL_MODEL_NOT_INSTALLED": f"The local name model (spaCy {LOCAL_MODEL}) is not installed. Install it once: "
                                 f"pip install \"casepatch[ner]\" && python -m spacy download {LOCAL_MODEL}",
}


def policy_error(exc, command=None):
    """POLICY_ERROR line on stderr, then a one-line fix when there is one."""
    print("POLICY_ERROR: " + str(exc), file=sys.stderr)
    hint = POLICY_HINTS.get(str(exc))
    if hint and command == "app" and str(exc) == "LOCAL_MODEL_NOT_INSTALLED":
        hint += " - or start without it in limited mode: casepatch app --rules-only"
    if hint:
        print(hint, file=sys.stderr)


def exclusive_json(path, value):
    with open(path, "x", encoding="utf-8", opener=lambda p, f: os.open(p, f, 0o600)) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(prog="casepatch")
    from . import __version__
    parser.add_argument("--version", action="version", version=f"casepatch {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("keygen", help="Create a 32-byte random key file for `transform` (keep it private)")
    gen.add_argument("--out", required=True, help="A NEW key file; existing paths are refused")
    run = commands.add_parser("transform", help="Structured-record engine: transform a JSON case bundle with a key file "
                                                "(for documents and traces use `anonymize`)")
    run.add_argument("input", help="JSON case bundle: " + BUNDLE_SHAPE)
    run.add_argument("--key-file", required=True, help="Key from `casepatch keygen` (same key + same bundle -> same output)")
    run.add_argument("--out", required=True, help="A NEW output directory; existing paths are refused")
    demo = commands.add_parser("demo", help="Run bundled fictional records without setup or an account")
    demo.add_argument("--out", required=True, help="A NEW output directory")
    app = commands.add_parser("app", help="Open the local import, review and export product")
    app.add_argument("--port", type=int, default=0, help="Local port; 0 selects a free port")
    app.add_argument("--open", action="store_true", dest="open_browser", help="Open the browser after binding the local port")
    app.add_argument("--rules-only", action="store_true", help="Limited mode without learned name detection")
    quote = commands.add_parser("estimate", help="Preview a proposed usage price locally; no actual billing")
    quote.add_argument("input", help="A JSON list of records, or " + BUNDLE_SHAPE)
    quote.add_argument("--monthly-used-chars", type=int, default=0, help="Assumed prior monthly account usage; not fetched")
    docs = commands.add_parser("docs", help="Transform a case folder of PDFs, scans, photos and images plus JSON / JSONL / TXT agent traces as one case",
                               epilog="A file that cannot be read (damaged or encrypted PDF, unreadable image, invalid trace) is "
                                      "skipped and named on stderr (SKIPPED); the other files are still processed as one case. "
                                      "Exit codes: 0 done (also when some files were skipped), 2 nothing processed (every file "
                                      "skipped, output folder exists, or an error), 3 decisions refused, 4 --fail-on-residual, "
                                      "5 --fail-on-review.")
    docs.add_argument("input_dir")
    docs.add_argument("--out", required=True, help="Output directory (transformed files + report.json, no identity map). "
                                                   "An existing one is refused unless you pass --overwrite")
    docs.add_argument("--overwrite", action="store_true",
                      help="Replace an existing --out folder from an earlier run (it must hold a CasePatch report.json, or be empty). "
                           "The review folder and its case key are kept, so the same input gives the same fakes")
    docs.add_argument("--detector", choices=["rules", "gliner", "llm", "hosted"], default="rules",
                      help="Learned detector added to the rules (hosted is a disabled stub)")
    docs.add_argument("--review-dir", help="Local review folder (proposals, review.html, decisions.json, case key). Default: <out>.review")
    docs.add_argument("--decisions", help="decisions.json from review (accept/reject per text proposal, replace/keep/erase per region; optional _add list of {text, kind} reviewer additions). "
                           "Ids are bound to content: decisions that no longer match a current proposal (or positional ids from 0.5.0 and older) are refused with exit code 3")
    docs.add_argument("--no-verify", action="store_true")
    docs.add_argument("--keep", action="append", metavar="TERM", default=[],
                      help="Never replace this term (case-insensitive, whole phrase), e.g. your own bank name; repeatable")
    docs.add_argument("--keep-file", metavar="PATH", help="Terms never to replace, one per line ('#' starts a comment)")
    docs.add_argument("--fail-on-residual", action="store_true",
                      help="Exit 4 when verification finds a detected original value still readable in an output (or verification was skipped)")
    docs.add_argument("--fail-on-review", action="store_true",
                      help="Exit 5 when any item needs review (including likely misses such as a labelled name left unchanged)")
    docs.add_argument("--hosted-detect", action="store_true",
                      help="OPT-IN, bring your own key: also send each page image + its OCR text directly from this computer to "
                           "Anthropic (Claude Sonnet 5.5) for detection, using YOUR ANTHROPIC_API_KEY under your agreement "
                           "with Anthropic. Per-case estimate limit $2 (CASEPATCH_BYOK_MAX_USD_PER_CASE). "
                           "Model-only suggestions are applied only after you accept them in review (decisions _accept_text)")
    docs.add_argument("--accept-hosted-notice", action="store_true",
                      help="Confirm you read the hosted-detection notice (page images leave this computer)")
    from .client.cli import add_parsers as add_client_parsers
    add_client_parsers(commands)
    hosted = commands.add_parser("hosted", help="Hosted frontier adapters (disabled by default; spend-capped)")
    hosted.add_argument("action", choices=["status", "verify"])
    hosted.add_argument("--auth-check", action="store_true",
                        help="verify only: also call each provider's free list-models endpoint (no paid call)")
    args = parser.parse_args()
    if args.command in ("anonymize", "setup-local"):
        from .client.cli import run as run_client
        return run_client(args)
    if args.command == "hosted":
        from .hosted.cli import status, verify
        print(verify(auth_check=args.auth_check) if args.action == "verify" else status())
        return 0
    if args.command == "docs":
        return docs_command(args)
    try:
        if args.command == "app":
            from .server import serve
            serve(args.port, args.rules_only, args.open_browser)
            return 0
        if args.command == "keygen":
            with open(args.out, "xb", opener=lambda p, f: os.open(p, f, 0o600)) as handle:
                handle.write(secrets.token_bytes(32))
            print("KEY_CREATED")
            return 0
        if args.command == "estimate":
            value = json.loads(Path(args.input).read_text())
            records = value.get("records") if isinstance(value, dict) else value
            print(json.dumps(estimate(records, args.monthly_used_chars), indent=2))
            return 0
        if args.command == "demo":
            bundle = json.loads(files("casepatch").joinpath("fixtures/dispute.json").read_text())
            key = secrets.token_bytes(32)
        else:
            bundle = json.loads(Path(args.input).read_text())
            key = Path(args.key_file).read_bytes()
        # Validate everything before creating the output directory.
        result = transform(bundle, key)
        destination = Path(args.out)
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        exclusive_json(destination / "report.json", result.report)
        if result.candidate is not None:
            exclusive_json(destination / "candidate.json", result.candidate)
        print(result.report["status"])
        return 2 if result.candidate is None else 0
    except PolicyError as exc:
        policy_error(exc, args.command)
        return 2
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        print("INPUT_OR_OUTPUT_ERROR: inspect local files; no source text logged", file=sys.stderr)
        return 2


def output_refusal(out: Path, input_dir: Path, overwrite: bool):
    """Why an existing --out folder must not be replaced, or None. --overwrite deletes only an earlier CasePatch output
    (a folder with report.json, or an empty one), never the input folder or a folder holding it."""
    if not overwrite:
        return f"OUTPUT_DIRECTORY_EXISTS: {out} already exists. Choose a new --out, or pass --overwrite to replace it."
    if not out.is_dir() or out.is_symlink():
        return f"OVERWRITE_REFUSED: {out} is not a folder."
    o, i = out.resolve(), input_dir.resolve()
    if o == i or o in i.parents:
        return f"OVERWRITE_REFUSED: {out} is (or contains) the input folder."
    if any(out.iterdir()):
        try:
            rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
            ours = isinstance(rep, dict) and "casepatch" in rep and "documents" in rep
        except (OSError, ValueError):
            ours = False
        if not ours:
            return (f"OVERWRITE_REFUSED: {out} has no CasePatch report.json, so it does not look like an earlier CasePatch "
                    "output. Not deleting it.")
    return None


def unreadable_reason(f: Path):
    """A short code (and position) when the file cannot join the case, or None. Never file content."""
    ext = f.suffix.lower()
    try:
        from .docs.textual import TextualError, fmt_of, parse
        if fmt_of(ext):
            try:
                parse(f.read_bytes(), fmt_of(ext))
            except TextualError as e:
                return e.code + (f" ({e.where})" if e.where else "")
            return None
        if ext == ".pdf":
            import pymupdf
            try:
                with pymupdf.open(f) as d:
                    if d.needs_pass:
                        return "PDF_ENCRYPTED"
                    if d.page_count == 0:
                        return "PDF_EMPTY"
                    d.load_page(0)
            except Exception:
                return "PDF_UNREADABLE"
            return None
        from PIL import Image
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception:
            return "IMAGE_UNREADABLE"
        return None
    except OSError:
        return "FILE_UNREADABLE"


def docs_command(args):
    try:
        from .docs.pipeline import run
    except ImportError:
        print("DOCS_EXTRAS_MISSING: pip install -e '.[docs]'", file=sys.stderr)
        return 2
    hosted = None
    if args.hosted_detect:
        from .docs.hosted_detect import NOTICE, HostedNotAllowed, HostedPageDetector
        if not args.accept_hosted_notice:
            print("HOSTED_NOTICE: " + NOTICE + "\nRe-run with --accept-hosted-notice to proceed.", file=sys.stderr)
            return 2
        try:
            hosted = HostedPageDetector.from_user(True)
        except HostedNotAllowed as exc:
            print("HOSTED_DISABLED: " + str(exc), file=sys.stderr); return 2
    from .docs.decisions import DecisionsRefused
    from .keep import KeepList, KeepListError, read_keep_file
    try:
        keep = KeepList(list(getattr(args, "keep", None) or []) +
                        (read_keep_file(args.keep_file) if getattr(args, "keep_file", None) else []))
    except (KeepListError, OSError) as exc:
        print(f"KEEP_LIST_INVALID: {exc}", file=sys.stderr); return 2
    out = Path(args.out)
    if out.exists():
        refused = output_refusal(out, Path(args.input_dir), getattr(args, "overwrite", False))
        if refused:
            print(refused, file=sys.stderr); return 2
    skipped, staged = [], None
    if Path(args.input_dir).is_dir():
        from .docs.extract import case_files
        found = case_files(args.input_dir)
        good = []
        for f in found:
            reason = unreadable_reason(f)
            if reason:
                skipped.append((f.name, reason))
                print(f"SKIPPED {f.name}: {reason}", file=sys.stderr)
            else:
                good.append(f)
        if found and not good:
            print(f"NO_VALID_FILES: all {len(found)} file(s) were skipped; nothing was processed", file=sys.stderr)
            return 2
        if skipped:  # the readable files, as one case, from a private staging folder (same names, same order)
            import shutil
            import tempfile
            staged = Path(tempfile.mkdtemp(prefix="casepatch-docs-"))
            for f in good:
                shutil.copy2(f, staged / f.name)
    if out.exists():  # --overwrite, checked above: only now, once the run is going ahead
        import shutil
        shutil.rmtree(out)
    try:
        report, *_ = run(str(staged) if staged else args.input_dir, args.out, args.detector, args.review_dir, args.decisions,
                         not args.no_verify, hosted=hosted, keep=keep)
    except DecisionsRefused as exc:
        print(str(exc), file=sys.stderr); return 3
    except FileNotFoundError:
        if Path(args.input_dir).exists():
            raise
        print(f"INPUT_NOT_FOUND: File or folder not found: {args.input_dir}", file=sys.stderr); return 2
    except FileExistsError:
        print(f"OUTPUT_DIRECTORY_EXISTS: {args.out} already exists. Choose a new --out, or pass --overwrite to replace it.",
              file=sys.stderr); return 2
    except (RuntimeError, ValueError) as exc:
        print("DOCS_ERROR: " + str(exc).split(":")[0], file=sys.stderr); return 2
    finally:
        if staged:
            import shutil
            shutil.rmtree(staged, ignore_errors=True)
    if skipped:
        report["skipped_inputs"] = [{"reason": r.split(" ")[0]} for _n, r in skipped]  # codes only: names can identify someone
        try:
            rp = out / "report.json"
            on_disk = json.loads(rp.read_text(encoding="utf-8"))
            on_disk["skipped_inputs"] = report["skipped_inputs"]
            rp.write_text(json.dumps(on_disk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError):
            pass
    ver = report.get("verification", {})
    for doc_id, d in report["documents"].items():
        v = ver.get(doc_id, {})
        print(f"{doc_id} {d['type']:5} replaced={sum(d['replacements_by_kind'].values()):3} regions_edited={len(d['edited_regions'])} "
              f"residual={v.get('residual_original_values', '-')} facts_missing={len(v.get('facts_missing', [])) if v else '-'} "
              f"pixel_diff={','.join(p['verdict'] for p in v.get('pixel_diff', [])) or '-'}")
    h = report.get("hosted_detection")
    if h:
        print(f"hosted={h['provider']}:{h['model']} pages_sent={h['pages_sent']} model_only_suggestions={h['model_only_values']} "
              f"accepted={h['model_only_values_accepted']} error={h['error'] or '-'}")
    miss = (report.get("decisions") or {}).get("additions_not_applied") or []
    if miss:
        print(f"ADDITIONS_NOT_APPLIED: {len(miss)} reviewer addition(s) were not found or not applied (kinds: "
              + ", ".join(sorted({str(m['kind']) for m in miss})) + "); see needs_review", file=sys.stderr)
    sc = report.get("synthetic_content", {})
    if sc.get("kept_by_decision"):
        kept = {k: sum(v.values()) for k, v in sc.get("kept_original_text_by_kind", {}).items()}
        print("KEPT_ORIGINALS (not synthetic): text " + (", ".join(f"{k}={n}" for k, n in kept.items()) or "none")
              + "; regions " + (", ".join(f"{k}={n}" for k, n in sc.get("kept_regions_by_kind", {}).items()) or "none"))
    lm = report.get("likely_misses") or []
    if lm:
        codes = {}
        for m in lm:
            codes[m["code"]] = codes.get(m["code"], 0) + 1
        print("LIKELY_MISSES: " + ", ".join(f"{k}={n}" for k, n in sorted(codes.items()))
              + " (a value may still be original; see needs_review)", file=sys.stderr)
    g = report.get("gate") or {}
    print("Replaced values are synthetic; values CasePatch did not detect are unchanged originals. Review before sharing.")
    print(f"needs_review={len(report['needs_review'])} status={report['status']} report={args.out}/report.json"
          + (f" skipped={len(skipped)}" if skipped else ""))
    if getattr(args, "fail_on_residual", False):
        if not g.get("verified"):
            print("FAIL_ON_RESIDUAL: verification was skipped (--no-verify); nothing was checked", file=sys.stderr)
            return 4
        if g.get("residual_original_values") or any(m["code"] == "RESIDUAL_ORIGINAL" for m in lm):
            print(f"FAIL_ON_RESIDUAL: {g.get('residual_original_values')} detected original value(s) still readable in the output",
                  file=sys.stderr)
            return 4
    if getattr(args, "fail_on_review", False) and report["needs_review"]:
        print(f"FAIL_ON_REVIEW: {len(report['needs_review'])} item(s) need review", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
