"""Run the CasePatch pipeline once per detector (separate processes, so peak RAM is per detector), then report
detection recall/precision against labels (with and without case-level propagation) and the full bench score.

    python -m casepatch.bench.detectors --input examples/documents/case-dispute-01 --work /tmp/bench --detectors rules gliner llm
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def _n(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def detection_metrics(input_dir, review_dir):
    labels = json.loads((Path(input_dir) / "labels.json").read_text())
    fmap = json.loads((Path(review_dir) / "file-map.json").read_text())
    props = json.loads((Path(review_dir) / "proposals.json").read_text())
    by_doc = {}
    for o in props["occurrences"]:
        by_doc.setdefault(fmap[o["doc"]], []).append(o)
    found, missed = 0, []
    total = 0
    for f, values in labels["occurrences"].items():
        texts = [_n(o["text"]) for o in by_doc.get(f, [])]
        for v in values:
            total += 1
            if any(_n(v) in t or (len(t) >= 4 and t in _n(v)) for t in texts):
                found += 1
            else:
                missed.append({"file": f, "value": v})
    allv = {_n(v) for e in labels["entities"] for v in e["values"]}
    fps = [o for o in props["occurrences"] if not any(_n(o["text"]) in v or v in _n(o["text"]) for v in allv)]
    return {"labeled_occurrences": total, "detected": found, "recall": round(found / max(total, 1), 4), "missed": missed,
            "proposals": len(props["occurrences"]), "false_positive_proposals": len(fps),
            "false_positive_examples": sorted({o["kind"] + ":" + o["text"] for o in fps})[:20],
            "regions": {r["kind"]: sum(1 for x in props["regions"] if x["kind"] == r["kind"]) for r in props["regions"]}}


RUNNER = r'''
import json, sys, time, resource
sys.path.insert(0, sys.argv[1])
from casepatch.docs.pipeline import run
t = time.perf_counter()
rep, *_ = run(sys.argv[2], sys.argv[3], sys.argv[4], verify=False, propagate=sys.argv[5] == "1")
print("RESULT " + json.dumps({"seconds": round(time.perf_counter() - t, 2), "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                               "stages": rep["stages"], "detectors": rep["detectors"]}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--detectors", nargs="+", default=["rules", "gliner", "llm"])
    ap.add_argument("--score", action="store_true", help="also run the full bench scorer on each propagated output")
    ap.add_argument("--json")
    a = ap.parse_args()
    root = str(Path(__file__).resolve().parents[2])
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    out = {"hardware": None, "runs": {}}
    from ..docs.pipeline import hardware
    out["hardware"] = hardware()
    for det in a.detectors:
        for prop in ("0", "1"):
            name = f"{det}{'+propagation' if prop == '1' else ''}"
            od = work / name
            r = subprocess.run([sys.executable, "-c", RUNNER, root, a.input, str(od), det, prop], capture_output=True, text=True)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
            if not line:
                out["runs"][name] = {"error": (r.stderr.strip().splitlines() or ["unknown"])[-1][:200]}
                continue
            info = json.loads(line[7:])
            info["detection"] = detection_metrics(a.input, str(od) + ".review")
            if a.score and prop == "1":
                from .score import score
                info["bench"] = score(a.input, od, "casepatch-" + name)
            out["runs"][name] = info
            print(name, info["seconds"], info["peak_rss_mb"], info["detection"]["recall"], info["detection"]["false_positive_proposals"], flush=True)
    s = json.dumps(out, indent=2, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(s)


if __name__ == "__main__":
    main()
