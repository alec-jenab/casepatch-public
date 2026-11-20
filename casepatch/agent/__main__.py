import argparse, json, os, sys

from .run import run_case, DEFAULT_MODEL, ESCALATE_MODEL

ap = argparse.ArgumentParser(prog="python -m casepatch.agent", description="CasePatch agent mode (BYOK; off by default)")
ap.add_argument("input"); ap.add_argument("--out", required=True)
ap.add_argument("--enable-agent", action="store_true", help="agent mode is off unless this flag (or CASEPATCH_AGENT=1) is given")
ap.add_argument("--eval", action="store_true", help="bench only: CasePatch eval key, public samples, spend ledger")
ap.add_argument("--budget", type=float, default=6.0, help="case ceiling (USD)"); ap.add_argument("--doc-budget", type=float, default=1.5); ap.add_argument("--faces", choices=["swap", "keep"], default="swap"); ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--escalate", default=ESCALATE_MODEL)
ap.add_argument("--fix-rounds", type=int, default=2); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--run", default="run")
ap.add_argument("--effort", default="high"); ap.add_argument("--only", nargs="*")
ap.add_argument("--replay", metavar="LOGS_DIR", help="re-execute a recorded run's transcripts with the local tools: no key, no network, $0")
a = ap.parse_args()
if a.replay:
    r = run_case(a.input, a.out, mode="replay", replay_dir=a.replay, budget=a.budget, doc_budget=a.doc_budget, faces=a.faces, fix_rounds=a.fix_rounds, workers=a.workers)
    print(json.dumps({k: r[k] for k in ("case", "wall_s", "spent_usd", "model_calls")} | {"open_issues": len(r["review_issues"]), "mode": "replay"}))
    sys.exit(0)
if not (a.enable_agent or os.environ.get("CASEPATCH_AGENT") == "1"):
    sys.exit("Agent mode is off by default. It sends document pages to your own Anthropic account; pass --enable-agent to opt in.")
r = run_case(a.input, a.out, mode="eval" if a.eval else "product", budget=a.budget, doc_budget=a.doc_budget, faces=a.faces, model=a.model, escalate=a.escalate or None,
             fix_rounds=a.fix_rounds, workers=a.workers, run_tag=a.run, effort=a.effort, only=a.only)
print(json.dumps({k: r[k] for k in ("case", "wall_s", "spent_usd", "list_usd", "model_calls")} | {"open_issues": len(r["review_issues"])}))
