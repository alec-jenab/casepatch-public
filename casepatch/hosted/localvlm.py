"""Local open-weight VLM adapter for the model eval (detect task only, $0, no network beyond 127.0.0.1).

Runs a GGUF vision-language model through llama.cpp's `llama-server` (mtmd: model + mmproj) on this machine and sends
exactly the same per-page request as the hosted adapters: the page image fitted to EVAL_MAX_IMAGE_SIDE plus
`modeleval.page_prompt(page)` (EVAL_PROMPT + OCR lines). Output is written in the hosted-result schema
(`<out>/local-vlm__<name>.json`), so `evalscore.score_hosted_dir` grounds and scores it with the same labels, the same
OCR-substring grounding and the same metrics as Claude / Gemini / OpenAI.

    python -m casepatch.hosted.localvlm run --name qwen3.5-4b-q4km --model M.gguf --mmproj P.gguf \
        --server /path/llama-server --out DIR [--cases ...] [--threads 8] [--ctx 16384] [--max-image-side 1600]

Resumable: pages already answered in the output JSON are skipped. No API keys, no ledger (cost $0, CPU time only).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import modeleval as ME


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class LlamaServer:
    def __init__(self, server, model, mmproj, threads=8, ctx=16384, extra=(), log=None):
        self.port = _free_port()
        env = dict(os.environ, LD_LIBRARY_PATH=str(Path(server).parent) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        for k in list(env):  # the local server never needs provider keys
            if k.endswith("_API_KEY"):
                env.pop(k)
        cmd = [str(server), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1", "--port", str(self.port),
               "-t", str(threads), "-c", str(ctx), "--no-webui", "-np", "1", *extra]
        self.log = open(log or os.devnull, "ab")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} (see {log})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(2)
        raise RuntimeError("llama-server did not become healthy in 900 s")

    def chat(self, prompt, png_b64, max_tokens, timeout=3600, json_mode=True, thinking=False):
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64}},
                    {"type": "text", "text": prompt}]}],
                "temperature": 0, "seed": 0, "max_tokens": max_tokens, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def run(name, model, mmproj, server, out, cases=None, work=ME.WORK, threads=8, ctx=16384, max_image_side=ME.EVAL_MAX_IMAGE_SIDE,
        max_tokens=ME.EVAL_MAX_OUTPUT_TOKENS, json_mode=True, extra=(), box_order="xy", meta=None, page_limit=None):
    from PIL import Image
    from .client import _fit, _png_b64
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"local-vlm__{name}.json"
    res = json.loads(f.read_text()) if f.exists() else {
        "system": f"local-vlm:{name}", "provider": "local-vlm", "model": name, "task": "detect", "box_order": box_order,
        "pages_work": str(work), "calls": [], "detections": [], "regions": [], "rejected_ungrounded": 0, "errors": [],
        "stopped": None, "raw": [], "settings": {}}
    res["settings"] = {"model_file": Path(model).name, "mmproj_file": Path(mmproj).name, "threads": threads, "ctx": ctx,
                       "max_image_side": max_image_side, "max_tokens": max_tokens, "json_mode": json_mode,
                       "server_build": meta or "", "extra": list(extra)}
    done = {(x["case"], x["file"], x["page"]) for x in res["raw"]}
    ok, _ = ME.available_cases(cases)
    srv = LlamaServer(server, model, mmproj, threads, ctx, extra, log=out / f"{name}.server.log")
    n = 0
    try:
        for c in ok:
            for page in ME.prepare_case(c, work):
                key = (c, page["file"], page["page"])
                if key in done:
                    continue
                if page_limit is not None and n >= page_limit:
                    raise StopIteration
                img, _ = _fit(np.asarray(Image.open(page["png"]).convert("RGB")), max_image_side)
                t0 = time.perf_counter()
                try:
                    data = srv.chat(ME.page_prompt(page), _png_b64(img), max_tokens, json_mode=json_mode)
                    text = data["choices"][0]["message"].get("content") or ""
                    usage = data.get("usage"); timings = data.get("timings")
                except Exception as e:
                    res["errors"].append({"case": c, "file": page["file"], "page": page["page"], "error": str(e)[:300]})
                    f.write_text(json.dumps(res, indent=1, ensure_ascii=False)); n += 1
                    continue
                lat = round(time.perf_counter() - t0, 2)
                d = ME._parse_detect(text)
                dets, regs, rej = ME.ground_page_output(page, d, yx=box_order.startswith("yx"))
                for x in dets + regs:
                    x["case"] = c
                res["detections"] += dets; res["regions"] += regs; res["rejected_ungrounded"] += rej
                res["malformed_json_pages"] = res.get("malformed_json_pages", 0) + bool(d.get("_salvaged"))
                res["raw"].append({"case": c, "file": page["file"], "page": page["page"], "text": text[:20000]})
                res["calls"].append({"case": c, "file": page["file"], "page": page["page"], "latency_s": lat, "cost_usd": 0.0,
                                     "reserved_usd": 0.0, "usage": usage, "timings": timings, "image_px": list(img.shape[1::-1])})
                f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
                print(f"{name} {c} {page['file']} p{page['page']}: {lat}s, {len(dets)} items, {len(regs)} regions", flush=True)
                n += 1
    except StopIteration:
        pass
    finally:
        srv.close()
    f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m casepatch.hosted.localvlm")
    ap.add_argument("action", choices=["run"])
    ap.add_argument("--name", required=True); ap.add_argument("--model", required=True); ap.add_argument("--mmproj", required=True)
    ap.add_argument("--server", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-image-side", type=int, default=ME.EVAL_MAX_IMAGE_SIDE)
    ap.add_argument("--max-tokens", type=int, default=ME.EVAL_MAX_OUTPUT_TOKENS)
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--box-order", default="xy", choices=["xy", "yx"])
    ap.add_argument("--page-limit", type=int, default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra llama-server args (after everything else)")
    a = ap.parse_args(argv)
    ver = ""
    try:
        ver = subprocess.run([a.server, "--version"], capture_output=True, text=True, timeout=30,
                             env=dict(os.environ, LD_LIBRARY_PATH=str(Path(a.server).parent))).stderr.strip().splitlines()[0]
    except Exception:
        pass
    run(a.name, a.model, a.mmproj, a.server, a.out, a.cases, threads=a.threads, ctx=a.ctx, max_image_side=a.max_image_side,
        max_tokens=a.max_tokens, json_mode=not a.no_json_mode, extra=a.extra, box_order=a.box_order, meta=ver, page_limit=a.page_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
