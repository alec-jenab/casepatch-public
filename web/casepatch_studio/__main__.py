import argparse
import sys


def engine_missing():
    """One-line install hint when the local engine is not installed (a base `pip install casepatch`), else None.
    Without it Studio would start, serve its page, then fail on the first API call."""
    try:
        from casepatch.client.local import missing_engine_module
    except ImportError as e:
        return f"casepatch-studio: cannot import casepatch ({e.name})."
    missing = missing_engine_module()
    if missing:
        return f'casepatch-studio needs the local engine: pip install "casepatch[studio]" (missing: {missing}).'
    return None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        if not {"-h", "--help"} & set(sys.argv[2:]) and engine_missing():
            print(engine_missing(), file=sys.stderr); return 2
        from .setup import main as setup_main
        return setup_main(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="casepatch-studio", description="CasePatch Studio: local, self-serve case-bundle transformation")
    ap.add_argument("--port", type=int, default=0, help="local port (default: any free port)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("--fast", action="store_true", help="skip the independent verification rescan (faster, less evidence)")
    ap.add_argument("--allow-hosted", action="store_true",
                    help="offer the opt-in hosted detection toggle, bring your own key (Claude Sonnet 5.5; page images go directly "
                         "from this computer to Anthropic when you turn it on for a case). Uses your ANTHROPIC_API_KEY or a "
                         "key you enter in the app's settings")
    ap.add_argument("--fetch-models", action="store_true", help="download the pinned model files (~250 MB) and exit")
    ap.add_argument("--allow-hosted-edit", action="store_true",
                    help="offer the opt-in hosted image-editor toggle, bring your own Fal key (id:secret). "
                         "Off by default. Evaluation keys are refused. This build does not make live edit calls; "
                         "the comparison UI uses precomputed public-sample outputs.")
    ap.add_argument("--agent-replay", metavar="LOGS_DIR",
                    help="demo/test: agent mode re-executes a recorded run's transcripts (logs/*.transcript.jsonl) on the local "
                         "tools instead of calling Anthropic. No key, no network, $0.")
    ap.add_argument("--compare-pack", help="directory of a precomputed editor-comparison pack (pack.json + crops/)")
    a = ap.parse_args()
    hint = engine_missing()
    if hint:
        print(hint, file=sys.stderr); return 2
    from . import setup as SU
    cfg = SU.load()
    if cfg is None and sys.stdin.isatty() and not a.fetch_models:
        print("First run: a short setup (models, optional agent mode). Skip any step; run `casepatch-studio setup` later.\n")
        SU.main([]); cfg = SU.load()
    if cfg and cfg.get("agent_mode"):
        a.allow_hosted = True   # chosen in setup: offer agent mode / hosted detection with the user's own key
    from .assets import ASSETS, DownloadError, _find, fetch
    if a.fetch_models:
        try:
            fetch()
        except DownloadError as e:
            print(f"casepatch-studio: {e}", file=sys.stderr); return 2
        print("models installed"); return 0
    missing = [n for n in ASSETS if not _find(n)]
    if missing:
        print("WARNING: model files missing (" + ", ".join(missing) + "). Faces/inpainting are degraded. "
              "Run `casepatch-studio --fetch-models` once.", file=sys.stderr, flush=True)
    from .server import serve
    serve(a.port, not a.no_open, verify=not a.fast, allow_hosted=a.allow_hosted, allow_hosted_edit=a.allow_hosted_edit, compare_pack=a.compare_pack, agent_replay=a.agent_replay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
