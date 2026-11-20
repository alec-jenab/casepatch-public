"""First-run walkthrough: `casepatch-studio setup` (also offered automatically on the first interactive start).

1. what CasePatch does and what stays on this computer   2. package and font check   3. model files (one download,
pinned by SHA-256)   4. optional agent mode with YOUR OWN Anthropic key (refuses CasePatch evaluation keys; stored on
this computer only: OS keychain, else a 0600 file; never printed, never sent to CasePatch)   5. how to start.
Writes ~/.casepatch/setup.json (no secrets in it)."""
from __future__ import annotations

import argparse, getpass, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path


def home(env=None) -> Path:
    from casepatch.paths import home as _home
    return _home(env)


def load(env=None) -> dict | None:
    p = home(env) / "setup.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _ask(q, default, yes):
    if yes is not None:
        return yes
    if not sys.stdin.isatty():
        return default
    a = input(f"{q} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not a else a.startswith("y")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="casepatch-studio setup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="accept the model download without asking")
    ap.add_argument("--no-models", action="store_true", help="skip the model download (faces/inpainting degraded)")
    ap.add_argument("--agent", choices=["ask", "on", "off"], default="ask", help="agent mode with your own Anthropic key")
    ap.add_argument("--key-stdin", action="store_true", help="read your Anthropic key from stdin (scripts); never echoed")
    ap.add_argument("--remember", choices=["ask", "yes", "no"], default="ask")
    a = ap.parse_args(argv)
    from casepatch.docs import byok
    from .assets import ASSETS, PACKAGES, DownloadError, _find, fetch
    print("CasePatch Studio setup\n"
          "  CasePatch replaces names, numbers, addresses, faces and signatures in a case's documents with consistent\n"
          "  synthetic values, checks the result, and lets you review every edit before export.\n"
          "  Your files are processed on this computer. Nothing is sent anywhere unless you turn on agent mode for a\n"
          "  case, which then calls Anthropic directly from this computer with your own key.\n", flush=True)
    print("1/4 Packages")
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    print("   ok" if not missing else "   missing: " + ", ".join(missing) + "  (reinstall with the install command)")
    fonts = shutil.which("fc-match") is not None
    print("   fonts: " + ("fontconfig found" if fonts else "fontconfig not found: text replacement falls back to built-in fonts"))
    print("2/4 Model files")
    need = [n for n in ASSETS if not _find(n)]
    models = not need
    if need and not a.no_models:
        mb = sum(ASSETS[n][2] for n in need)
        print(f"   {len(need)} pinned public model files to download (~{mb:.0f} MB): " + ", ".join(need))
        if _ask("   Download now?", True, True if a.yes else None):
            try:
                fetch(); models = True
            except DownloadError as e:
                print(f"   {e}")
    print("   " + ("installed" if models else "skipped (run `casepatch-studio setup` again later)"))
    print("3/4 Agent mode (optional, off by default)\n"
          "   Uses your own Anthropic API key. Page images go from this computer straight to Anthropic, per case, only\n"
          "   after you accept a notice in the app. CasePatch never sees your key or your documents.")
    agent = {"on": True, "off": False}.get(a.agent)
    if agent is None:
        agent = _ask("   Enable agent mode?", False, None)
    stored = None
    if agent:
        key = sys.stdin.readline().strip() if a.key_stdin else (getpass.getpass("   Anthropic API key (sk-ant-…, input hidden; Enter to skip): ").strip()
                                                               if sys.stdin.isatty() else "")
        if key:
            try:
                byok.check_key(key)
            except byok.ByokRefused as e:
                print("   refused: " + str(e)); key = ""
        if key:
            rem = {"yes": True, "no": False}.get(a.remember)
            if rem is None:
                rem = _ask("   Remember the key on this computer?", True, None)
            if rem:
                stored = byok.store_key(key)
                print("   saved " + ("in your OS keychain" if stored == "keychain" else f"in {byok.key_file()} (readable only by you)"))
            else:
                print("   not saved: set ANTHROPIC_API_KEY or enter the key in the app each time")
        elif not os.environ.get(byok.USER_KEY_ENV):
            print("   no key yet: enter it later in the app (Hosted detection → key) or set ANTHROPIC_API_KEY")
    h = home(); h.mkdir(parents=True, exist_ok=True, mode=0o700)
    (h / "setup.json").write_text(json.dumps({"version": 1, "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": models,
                                              "agent_mode": bool(agent), "key_stored": stored or None}, indent=1))
    print("4/4 Done. Start with:  casepatch-studio      (opens your browser on a local address)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
