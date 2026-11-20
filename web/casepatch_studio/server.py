"""CasePatch Studio local server (stdlib only; binds 127.0.0.1).

Flow: create case -> upload files (raw PUT per file) or load the example case -> analyze (child process runs the
CasePatch pipeline with default decisions) -> review proposals with before/after page previews -> apply decisions
(re-render, same case key so replacements stay consistent) -> view every output page -> confirm -> export ZIP.

Local mode guarantees: no outbound network calls from this server or the worker, unless the app was started with
--allow-hosted AND the user turns on hosted detection for a case after reading the notice (then the worker sends that
case's page images + OCR text directly to Anthropic with the USER'S OWN key (bring your own key: ANTHROPIC_API_KEY or
a key entered in settings, kept in memory or the OS keychain; never written to job files, logs or exports; never a
CasePatch evaluation key or ledger), once, under a per-case estimate limit; model-only suggestions are applied only
after the reviewer accepts them, and export is refused until every suggestion was accepted or declined). Case files live under ~/.casepatch/studio/<case> (0700) and are deleted on Delete, after 2 h idle, and at startup.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from casepatch.metering.events import EventLog, Meter, Settings, Uploader, casepatch_home  # noqa: E402

STATIC = Path(__file__).with_name("static")
SAMPLE = Path(__file__).with_name("sample")
# engine output names: one explicit allowlist (a length rule like \.[a-z]{3,4} silently dropped .jsonl traces from exports)
OUTPUT_NAME = re.compile(r"document-\d{2}\.(?:pdf|png|jpe?g|webp|tiff?|json|jsonl|txt)")
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".json", ".jsonl", ".ndjson", ".txt"}
TRACE_EXT = {".json", ".jsonl", ".ndjson", ".txt"}  # validated by casepatch.docs.textual
MAX_FILE = 30 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024
MAX_FILES = 20
MAX_PAGES = 40
IDLE_TTL = 2 * 3600
JOB_TIMEOUT = 1800
RUN_LOCK = threading.Semaphore(1)  # one pipeline pass at a time (memory: up to ~3 GB per pass)

MESSAGES = {
    "NO_FILES": "Add at least one PDF, image or JSON file first.",
    "TOO_MANY_FILES": f"A case can have at most {MAX_FILES} files.",
    "FILE_TOO_LARGE": "Each file must be 30 MB or less (100 MB per case).",
    "UNSUPPORTED_FILE": "Supported files: PDF, PNG, JPG, WEBP and TIFF documents, and JSON, JSONL and TXT traces.",
    "FILE_CONTENT_MISMATCH": "That file's contents do not match its extension.",
    "TRACE_INVALID": "That trace could not be read: JSON, JSONL and TXT files must be valid UTF-8 (and valid JSON / one JSON record per line), at most 5 MB.",
    "TOO_MANY_PAGES": f"A case can have at most {MAX_PAGES} pages in this local build.",
    "CASE_NOT_FOUND": "This case was deleted or expired. Start a new case.",
    "BUSY": "This case is still processing. Wait for it to finish.",
    "NOT_ANALYZED": "Analyze the case first.",
    "AGENT_NOTICE_REQUIRED": "Read and accept the agent-mode notice first.",
    "AGENT_KEY_REQUIRED": "Agent mode needs your own Anthropic API key.",
    "AGENT_NOT_RUN": "Run agent mode first.",
    "AGENT_UNRESOLVED": "Some agent edits were flagged by the checks (an original may still be visible). For each, keep the original or record an override before exporting.",
    "AGENT_REAPPLY_FAILED": "A kept PDF text edit could not be re-applied, so the export was stopped. Keep the agent result for that file or review it by hand.",
    "VIEW_ALL_PAGES": "Open every output page in the review screen before exporting.",
    "CONFIRM_REVIEW": "Tick the confirmation that you reviewed every output page.",
    "BAD_DECISIONS": "Some review choices were not recognised. Reload the case.",
    "PIPELINE_FAILED": "Processing failed. No output was produced. See the case log on your computer for details.",
    "LOCKED": "Upload is closed after analysis. Start a new case to add files.",
    "HOSTED_NOT_AVAILABLE": "Hosted detection is not available in this app. Analyze locally, or restart with --allow-hosted and add your own Anthropic API key.",
    "BAD_KEY": "That key was not accepted. Use your own Anthropic API key (it starts with sk-ant-).",
    "KEYCHAIN_UNAVAILABLE": "No OS keychain is available, so the key is kept only until the app stops.",
    "HOSTED_NOTICE_REQUIRED": "Confirm that you understand page images are sent to Anthropic before using hosted detection.",
    "MODEL_SUGGESTIONS_PENDING": "Accept or decline every model suggestion before exporting.",
    "BAD_CATEGORIES": "A category choice was not recognised. Each category can be 'replace' or 'keep'.",
    "NO_COMPARE_PACK": "No precomputed editor-comparison pack is available.",
    "LEAK_BLOCKED": "This candidate still shows the original text. It cannot be selected without an explicit override.",
    "BAD_PICK": "That editor choice was not recognised.",
    "BAD_EDIT_KEY": "That key was not accepted. Use your own Fal API key (it should be id:secret).",
    "IDEOGRAM_KEY_STUB": "A direct Ideogram API key is not accepted yet. Use your own Fal API key (id:secret) for Ideogram 4.5 and the other hosted editors.",
    "EDIT_TRANSPORT_REQUIRED": "Hosted image edits are wired but this process has no edit transport. Tests inject a fake; this work does not construct the HTTP client or make a real Fal call.",
    "HOSTED_EDIT_OPT_IN": "Select at least one document to run hosted editors on.",
    "HOSTED_EDIT_NOTICE_REQUIRED": "Confirm that hosted image edits use your own Fal key, never go through CasePatch servers, and only run on the documents you opt in.",
    "HOSTED_EDIT_ESTIMATE_REQUIRED": "Review the estimated cost and accept it before running hosted editors.",
}

def _agent_status(c):
    from .agent_review import status
    return status(c)


def category_choices(choices):
    from casepatch.docs.categories import group_choices
    return group_choices(choices)


def category_group_list():
    from casepatch.docs.categories import POLICY_GROUPS
    return [{"id": g, "label": label, "default": "replace"} for g, label, _ in POLICY_GROUPS]


def clean_categories(body_value):
    """Validated {group: 'replace'|'keep'} from a request body (None = unchanged)."""
    if body_value is None:
        return None
    from casepatch.docs.categories import policy_from_groups
    if not isinstance(body_value, dict):
        raise StudioError("BAD_CATEGORIES")
    try:
        policy_from_groups(body_value)
    except ValueError:
        raise StudioError("BAD_CATEGORIES")
    return dict(body_value)


HOSTED_KEY_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MOONSHOT_API_KEY",
                  "CASEPATCH_BYOK_ANTHROPIC_KEY")
# evaluation-harness switches: never passed to a worker (the product does not use the eval ledger or its caps)
EVAL_ENV = ("CASEPATCH_HOSTED", "CASEPATCH_SPEND_CAP_USD", "CASEPATCH_SPEND_LEDGER", "CASEPATCH_HOSTED_PROVIDER",
            "CASEPATCH_POOL_CAP_ANTHROPIC_USD", "CASEPATCH_POOL_CAP_OPENAI_USD", "CASEPATCH_POOL_CAP_OTHER_USD",
            "CASEPATCH_PROVIDER_CAP_ANTHROPIC_USD", "CASEPATCH_PROVIDER_CAP_OPENAI_USD", "CASEPATCH_PROVIDER_CAP_GEMINI_USD",
            "CASEPATCH_PROVIDER_CAP_KIMI_USD")


class StudioError(Exception):
    def __init__(self, code, status=400, extra=None):
        super().__init__(code); self.code = code; self.status = status; self.extra = extra or {}


def _sniff(name: str, head: bytes) -> bool:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return head.startswith(b"%PDF")
    if ext == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in (".jpg", ".jpeg"):
        return head.startswith(b"\xff\xd8\xff")
    if ext == ".webp":
        return head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if ext in (".tif", ".tiff"):
        return head[:4] in (b"II*\x00", b"MM\x00*")
    if ext in TRACE_EXT:
        return True  # validated in full by add_file (UTF-8, JSON/JSONL syntax, bounds)
    return False


def safe_name(name: str, index: int) -> str:
    base = Path(name.replace("\\", "/")).name
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:60] or "file"
    return f"{index:02d}_{stem}{ext.lower()}"


class Case:
    def __init__(self, root: Path, source="upload"):
        self.id = secrets.token_urlsafe(18)
        self.ref = secrets.token_hex(16)  # random, never derived from content
        self.dir = root / self.id
        self.input = self.dir / "input"
        self.input.mkdir(parents=True, mode=0o700)
        os.chmod(self.dir, 0o700)
        self.source = source
        self.files = []  # [{name, original, size, pages}]
        self.status = "empty"  # empty | queued | running | ready | error
        self.error = None
        self.run = 0
        self.meta = None
        self.started = None
        self.ready_at = None
        self.analyzed_at = None
        self.decisions = None
        self.defaults = None
        self.additions = []
        self.rejected_text = []
        self.hints = None  # example case only: reviewer additions Studio pre-fills (shown, removable)
        self.viewed = set()
        self.touched = time.time()
        self.exports = 0
        self.hosted = False          # user opted in to hosted detection for this case (after the notice)
        self.accept_text = []        # model suggestions the reviewer accepted (applied in the current run)
        self.categories = {}         # {policy group: 'replace'|'keep'}; missing groups are replaced (default)
        self.lock = threading.Lock()

    # --- geometry / previews ------------------------------------------------
    def pages(self):
        if not self.meta:
            return []
        return [(d["id"], p["index"]) for d in self.meta["documents"] if d["kind"] != "json" for p in d["pages"]] + \
               [(d["id"], "json") for d in self.meta["documents"] if d["kind"] == "json"]

    def out_dir(self, run=None):
        return self.dir / f"out-{run or self.run}"

    def snapshot(self):
        s = {"id": self.id, "status": self.status, "error": self.error, "run": self.run, "source": self.source,
             "files": [{"name": f["original"], "size": f["size"], "pages": f["pages"]} for f in self.files],
             "elapsed": round(time.time() - self.started, 1) if self.started and self.status in ("queued", "running") else None,
             "exports": self.exports, "hosted": self.hosted, "categories": category_choices(self.categories)}
        if self.status == "ready" and self.meta:
            total = self.pages()
            s["pages_total"] = len(total)
            s["pages_viewed"] = len([p for p in total if p in self.viewed])
            s["seconds"] = self.meta.get("seconds")
        return s


class Studio(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port=0, home: Path | None = None, verify=True, worker_cmd=None, allow_hosted=False,
                 allow_hosted_edit=False, compare_pack=None, env=None, edit_transport=None, agent_replay=None):
        self.allow_hosted = bool(allow_hosted)
        self.agent_replay = str(agent_replay) if agent_replay else None   # demo/test: recorded agent run, no API, $0
        self.allow_hosted_edit = bool(allow_hosted_edit)
        self.env = os.environ if env is None else env
        self._user_key = None  # the user's own Anthropic key entered in settings: memory only (or the OS keychain)
        self._fal_key = None   # the user's own Fal key: memory only, never logged or echoed
        # Injected Fal transport (tests pass a fake). HttpTransport is never constructed in this work.
        self._edit_transport = edit_transport
        if self.allow_hosted:
            from casepatch.docs.byok import load_stored_key
            self._user_key = load_stored_key(self.env)
        from . import compare_api as CMP
        self._compare_dir = CMP.pack_dir(compare_pack)
        self._compare_pack = CMP.load(compare_pack) if self._compare_dir else None
        self._compare_picks = CMP.new_picks(self._compare_pack) if self._compare_pack else None
        self.home = home or casepatch_home()
        self.root = self.home / "studio"
        if self.root.exists():  # nothing survives a restart: the server forgot these cases
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, mode=0o700)
        self.settings = Settings(self.home)
        self.events = EventLog(self.home, self.settings, __version__)
        self.meter = Meter(self.home)
        self.uploader = Uploader(self.events)
        self.verify = verify
        self.worker_cmd = worker_cmd or [sys.executable, "-m", "casepatch_studio.worker"]
        self.cases: dict[str, Case] = {}
        from .assets import ASSETS, _find
        self.missing_assets = [{"name": n, "effect": ASSETS[n][3]} for n in ASSETS if not _find(n)]
        self.csrf = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.events.record("app_started", {"mode": "local", "os": {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(sys.platform, "other")})

    # --- case registry --------------------------------------------------------
    def purge(self):
        now = time.time()
        for cid, c in list(self.cases.items()):
            if now - c.touched > IDLE_TTL and c.status not in ("queued", "running"):
                self.delete(cid)

    def get(self, cid) -> Case:
        self.purge()
        c = self.cases.get(cid)
        if c is None:
            raise StudioError("CASE_NOT_FOUND", 404)
        c.touched = time.time()
        return c

    def delete(self, cid):
        c = self.cases.pop(cid, None)
        if c:
            shutil.rmtree(c.dir, ignore_errors=True)

    # --- uploads ----------------------------------------------------------------
    def add_file(self, c: Case, name: str, data: bytes):
        if c.status != "empty":
            raise StudioError("LOCKED")
        if len(c.files) >= MAX_FILES:
            raise StudioError("TOO_MANY_FILES")
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise StudioError("UNSUPPORTED_FILE")
        if not data or len(data) > MAX_FILE or sum(f["size"] for f in c.files) + len(data) > MAX_TOTAL:
            raise StudioError("FILE_TOO_LARGE", 413)
        if not _sniff(name, data[:16]):
            raise StudioError("FILE_CONTENT_MISMATCH")
        pages = 1
        if ext == ".pdf":
            import pymupdf
            try:
                with pymupdf.open(stream=data, filetype="pdf") as d:
                    pages = d.page_count
            except Exception:
                raise StudioError("FILE_CONTENT_MISMATCH")
        elif ext in TRACE_EXT:
            from casepatch.docs.textual import TextualError, fmt_of, pages_for, parse
            try:
                pages = pages_for(parse(data, fmt_of(ext))["chars"])
            except TextualError as e:
                raise StudioError("TRACE_INVALID", extra={"detail": e.code + (f" ({e.where})" if e.where else "")})
        if sum(f["pages"] for f in c.files) + pages > MAX_PAGES:
            raise StudioError("TOO_MANY_PAGES")
        stored = safe_name(name, len(c.files) + 1)
        if stored.endswith("labels.json"):
            stored = stored.replace("labels.json", "labels_.json")
        fd = os.open(c.input / stored, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        c.files.append({"name": stored, "original": Path(name.replace("\\", "/")).name[:120], "size": len(data), "pages": pages})

    def load_sample(self, c: Case):
        # case files are NN_name.ext; ATTRIBUTION.md and review_hints.json are not part of the case
        for f in sorted(SAMPLE.iterdir()):
            if f.suffix.lower() in ALLOWED_EXT and re.match(r"\d\d_", f.name):
                self.add_file(c, f.name, f.read_bytes())
        c.source = "sample"
        hints = SAMPLE / "review_hints.json"
        if hints.exists():
            h = json.loads(hints.read_text())
            c.hints = {"note": str(h.get("note", "")), "add": [{"text": a["text"], "kind": a["kind"]} for a in h.get("add", [])],
                       "regions": [{"doc": r["doc"], "kind": r["kind"], "action": r["action"]} for r in h.get("regions", [])
                                   if r.get("action") in REGION_VALID]}

    # --- pipeline runs ------------------------------------------------------------
    def category_policy(self, c):
        from casepatch.docs.categories import policy_from_groups
        return policy_from_groups(c.categories)

    def hosted_status(self):
        from casepatch.docs.hosted_detect import status
        return status(self.env, allowed=self.allow_hosted, user_key=self._user_key)

    def set_user_key(self, key, remember=False):
        from casepatch.docs import byok
        if not self.allow_hosted:
            raise StudioError("HOSTED_NOT_AVAILABLE")
        try:
            byok.check_key(key, self.env)
        except byok.ByokRefused:
            raise StudioError("BAD_KEY")
        self._user_key = key
        if remember:
            byok.store_key(key, self.env)   # OS keychain, else a 0600 file under ~/.casepatch; never sent anywhere

    def forget_user_key(self):
        from casepatch.docs import byok
        self._user_key = None
        byok.forget_stored_key(self.env)

    def hosted_edit_status(self):
        from casepatch.docs.editor_pick import hosted_editors_available
        from casepatch.docs.hosted_edit import NOTICE
        s = hosted_editors_available(self.env, user_key=self._fal_key, allowed=self.allow_hosted_edit)
        s["live"] = self._edit_transport is not None
        s["notice"] = NOTICE
        s["notice_version"] = 1
        return s

    def set_fal_key(self, key, remember=False, provider="fal"):
        from casepatch.docs.byok import ByokRefused
        from casepatch.docs.editor_pick import check_hosted_edit_key, check_ideogram_direct_key
        if not self.allow_hosted_edit:
            raise StudioError("HOSTED_NOT_AVAILABLE")
        if (provider or "fal") == "ideogram":
            try:
                check_ideogram_direct_key(key, self.env)
            except ByokRefused:
                raise StudioError("IDEOGRAM_KEY_STUB")
        try:
            self._fal_key = check_hosted_edit_key(key, self.env)
        except ByokRefused:
            raise StudioError("BAD_EDIT_KEY")
        # memory only: hosted image editing does not write keys to the keychain in this build

    def forget_fal_key(self):
        self._fal_key = None

    def compare_status(self):
        p = self._compare_pack
        if not p:
            return {"available": False}
        return {"available": True, "precomputed": True, "public_only": p.get("public_only", True),
                "changes": len(p.get("changes") or []), "note": p.get("note")}

    def start_run(self, c: Case, decisions: dict | None, hosted=None, categories=None):
        with c.lock:
            if c.status in ("queued", "running"):
                raise StudioError("BUSY", 409)
            if not c.files:
                raise StudioError("NO_FILES")
            if decisions is not None and c.meta is None:
                raise StudioError("NOT_ANALYZED")
            if hosted is not None:
                from casepatch.docs.hosted_detect import NOTICE_VERSION
                if not hosted.get("on"):
                    if c.meta is None:
                        c.hosted = False
                elif c.meta is not None:
                    raise StudioError("LOCKED")
                elif not self.hosted_status().get("available"):
                    raise StudioError("HOSTED_NOT_AVAILABLE")
                elif hosted.get("notice_version") != NOTICE_VERSION:
                    raise StudioError("HOSTED_NOTICE_REQUIRED")
                else:
                    c.hosted = True
            if categories is not None:
                c.categories = categories
            c.status, c.error, c.started = "queued", None, time.time()
        threading.Thread(target=self._run, args=(c, decisions), daemon=True).start()

    def _run(self, c: Case, decisions):
        with RUN_LOCK:
            c.status = "running"
            run = c.run + 1
            job = {"input": str(c.input), "out": str(c.out_dir(run)), "review": str(c.dir / "review"),
                   "decisions": None, "detector": "rules", "verify": self.verify, "previews": str(c.dir / "previews"),
                   "run": run, "meta": str(c.dir / f"meta-{run}.json"),
                   "hosted": bool(c.hosted and run == 1), "case_ref": c.ref,  # later runs reuse the cached model output
                   "category_policy": self.category_policy(c)}
            if decisions is not None:
                p = c.dir / f"decisions-{run}.json"
                p.write_text(json.dumps(decisions)); os.chmod(p, 0o600)
                job["decisions"] = str(p)
            if run == 1:
                self.seed_case_key(c)
            jp = c.dir / f"job-{run}.json"
            jp.write_text(json.dumps(job))
            env = dict(self.env, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "web")]))
            for k in HOSTED_KEY_ENV + EVAL_ENV:  # no provider key and no eval switch reaches the worker by default
                env.pop(k, None)
            env["CASEPATCH_HOSTED"] = "0"
            if job["hosted"]:  # BYOK: only the user's own key, handed over in the child's environment (never on disk)
                from casepatch.docs.byok import HANDOFF_ENV, ByokRefused, resolve_key
                try:
                    env[HANDOFF_ENV] = resolve_key(self.env, self._user_key)[0]
                except ByokRefused:
                    c.status, c.error = ("ready" if c.meta else "error"), "HOSTED_NOT_AVAILABLE"
                    return
            try:
                with open(c.dir / f"worker-{run}.log", "wb") as log:
                    rc = subprocess.run(self.worker_cmd + [str(jp)], cwd=str(ROOT), env=env, stdout=log, stderr=log,
                                        timeout=JOB_TIMEOUT).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            if rc != 0 or not Path(job["meta"]).exists():
                c.status, c.error = ("ready" if c.meta else "error"), "PIPELINE_FAILED"
                self.events.record("analysis_failed", {"reason": "pipeline_error"}, c.ref)
                return
            meta = json.loads(Path(job["meta"]).read_text())
            first = c.meta is None
            c.meta, c.run, c.viewed = meta, run, set()
            c.decisions = meta["decisions"]
            c.ready_at = time.time()
            if first:
                c.analyzed_at = c.ready_at
                c.defaults = dict(meta["decisions"])
            else:
                c.additions = list(decisions.get("_add", []))
                c.rejected_text = list(decisions.get("_reject_text", []))
                c.accept_text = list(decisions.get("_accept_text", []))
            c.status = "ready"
            docs = meta["documents"]
            pdf_pages = sum(len(d["pages"]) for d in docs if d["kind"] == "pdf")
            img_pages = sum(len(d["pages"]) for d in docs if d["kind"] == "image")
            jdocs = sum(1 for d in docs if d["kind"] == "json")
            pages = pdf_pages + img_pages + jdocs
            secs = int(meta.get("seconds") or 0)
            if first:
                prop = self.proposals(c)
                self.meter.add(cases_analyzed=1, documents_processed=len(docs), pages_processed=pages, pdf_pages=pdf_pages,
                               image_pages=img_pages, json_documents=jdocs, processing_seconds=secs)
                self.events.record("case_analyzed", {"documents": len(docs), "pages": pages, "pdf_pages": pdf_pages,
                                                     "image_pages": img_pages, "json_documents": jdocs,
                                                     "proposals": len(prop["occurrences"]), "regions": len(prop["regions"]),
                                                     "seconds": secs, "detector": "rules+hosted" if c.hosted else "rules",
                                                     "model_suggestions": len(prop.get("model_suggestions", [])),
                                                     "source": c.source}, c.ref)
            else:
                self.meter.add(review_runs=1, pages_processed=pages, processing_seconds=secs)
                vals = [v for k, v in c.decisions.items() if not k.startswith("_")]
                changed = sum(1 for k, v in c.decisions.items() if c.defaults.get(k) not in (None, v))
                self.events.record("review_applied", {
                    "accepted": vals.count("accept"), "rejected": vals.count("reject"), "added": len(c.additions),
                    "regions_replace": vals.count("replace"), "regions_keep": vals.count("keep"),
                    "regions_erase": vals.count("erase"), "changed_from_default": changed, "seconds": secs}, c.ref)

    def start_agent(self, c: Case, body: dict):
        """Agent mode (BYOK, opt-in per case): `python -m casepatch.agent` in a worker process with the user's own key in
        its environment only (never on disk); no eval switches or other provider keys reach it."""
        from casepatch.docs.byok import HANDOFF_ENV, ByokRefused, resolve_key
        from . import agent_review as AR
        if c.status != "ready" or not c.meta:
            raise StudioError("NOT_ANALYZED")
        key = None
        if not self.agent_replay:
            try:
                key = resolve_key(self.env, self._user_key)[0]
            except ByokRefused:
                raise StudioError("AGENT_KEY_REQUIRED")
        d = AR.adir(c)
        if (d / "running").exists():
            raise StudioError("BUSY", 409)
        shutil.rmtree(d, ignore_errors=True); d.mkdir(mode=0o700)
        (d / "running").write_text("1")
        budget = min(float(body.get("case_budget_usd") or 6.0), 50.0); doc_budget = min(float(body.get("doc_budget_usd") or 1.5), budget)
        env = dict(self.env, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "web")]))
        for k in HOSTED_KEY_ENV + EVAL_ENV:
            env.pop(k, None)
        faces = "swap" if body.get("faces") == "swap" else "keep"   # uploads: faces flagged unless the user turns swap on
        cmd = [sys.executable, "-m", "casepatch.agent", str(c.input), "--out", str(d), "--budget", str(budget), "--doc-budget", str(doc_budget), "--faces", faces]
        if self.agent_replay:
            cmd += ["--replay", self.agent_replay]
        else:
            env[HANDOFF_ENV] = key; cmd.append("--enable-agent")

        def go():
            try:
                with open(d / "worker.log", "wb") as log:
                    subprocess.run(cmd,
                                   cwd=str(ROOT), env=env, stdout=log, stderr=log, timeout=JOB_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
            finally:
                (d / "running").unlink(missing_ok=True)
        threading.Thread(target=go, daemon=True).start()

    def seed_case_key(self, c: Case):
        """Deterministic replacements per case: the same files analyzed on this computer get the same fake identities.

        The case key is HMAC(install secret, sorted SHA-256 of the case files). The install secret never leaves this
        computer, so the key cannot be recomputed from the documents alone (and is never exported)."""
        review = c.dir / "review"; review.mkdir(parents=True, exist_ok=True, mode=0o700)
        kf = review / "case.key"
        if kf.exists():
            return
        sec = self.root / "install.secret"
        if not sec.exists():
            fd = os.open(sec, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        digests = sorted(hashlib.sha256((c.input / f["name"]).read_bytes()).hexdigest() for f in c.files)
        key = hmac.new(sec.read_bytes(), ("casepatch-case-key\x1f" + "\x1f".join(digests)).encode(), hashlib.sha256).digest()
        fd = os.open(kf, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)

    def proposals(self, c: Case):
        p = json.loads((c.dir / "review" / "proposals.json").read_text())
        return p

    def category_flags(self, c: Case):
        f = c.dir / "review" / "category-flags.json"
        return json.loads(f.read_text()) if f.exists() else []

    def report(self, c: Case):
        return json.loads((c.out_dir() / "report.json").read_text())

    # --- export ---------------------------------------------------------------------
    def export(self, c: Case, confirmed: bool, suggestions_reviewed=None) -> bytes:
        if c.status != "ready" or not c.meta:
            self.events.record("export_refused", {"reason": "no_run"}, c.ref)
            raise StudioError("NOT_ANALYZED")
        missing = [p for p in c.pages() if p not in c.viewed]
        if missing:
            self.events.record("export_refused", {"reason": "not_reviewed"}, c.ref)
            raise StudioError("VIEW_ALL_PAGES")
        if confirmed is not True:
            self.events.record("export_refused", {"reason": "not_reviewed"}, c.ref)
            raise StudioError("CONFIRM_REVIEW")
        pending = set(self.proposals(c).get("model_suggestions", [])) - set(suggestions_reviewed or [])
        if pending:
            self.events.record("export_refused", {"reason": "model_suggestions_pending"}, c.ref)
            raise StudioError("MODEL_SUGGESTIONS_PENDING")
        out = c.out_dir()
        report = self.report(c)
        buf = io.BytesIO()
        names = []
        from . import live_compare as LC
        try:
            editor_fields = LC.export_review_fields(self, c)
        except Exception:
            editor_fields = {}
        overlays = {}
        try:
            model, picks = LC.compare_model(self, c)
            if model.get("changes"):
                overlays = LC.apply_picks_to_outputs(self, c, picks, model)
        except Exception:
            overlays = {}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                if OUTPUT_NAME.fullmatch(f.name):
                    if f.name in overlays:
                        z.writestr("documents/" + f.name, overlays[f.name])
                    else:
                        z.write(f, "documents/" + f.name)
                    names.append(f.name)
            z.writestr("report.json", json.dumps(report, indent=2, ensure_ascii=False))
            vals = [v for k, v in (c.decisions or {}).items() if not k.startswith("_")]
            review = {
                "casepatch_studio": __version__, "run": c.run, "reviewed_by_user": True,
                "hosted_detection": bool(c.hosted), "model_suggestions_accepted": len(c.accept_text),
                "output_pages_viewed": len(c.viewed), "output_pages_total": len(c.pages()),
                "decision_counts": {k: vals.count(k) for k in sorted(set(vals))},
                "note": "Counts only. Source values, the identity map and the case key are not exported."}
            review.update(editor_fields)
            z.writestr("review.json", json.dumps(review, indent=2))
            z.writestr("README.txt", README_TXT)
        pages = sum(len(d["pages"]) for d in c.meta["documents"])
        c.exports += 1
        self.meter.add(reviewed_exports=1, pages_exported=pages)
        self.events.record("export_completed", {"documents": len(names), "pages": pages, "reviewed": True,
                                                "review_seconds": int(time.time() - (c.analyzed_at or time.time())),
                                                "needs_review_items": len(report.get("needs_review", [])),
                                                "verification_ran": "verification" in report, "source": c.source}, c.ref)
        return buf.getvalue()


README_TXT = """CasePatch export (synthetic, review required)

documents/   Transformed copies of the case files. Names, numbers, addresses, faces, signatures and codes that were
             accepted in review were replaced in place with synthetic values. Amounts, dates and other facts were
             left unchanged unless you selected them.
report.json  Verification report: independent rescan of the outputs, residual-value checks, pixel diff outside
             edited regions, legibility checks, and items that still need human review.
review.json  Counts of your review decisions. No source values, identity map or case key are included.

These files are synthetic copies for evaluation and testing. They are not originals. CasePatch does not certify
anonymity or regulatory de-identification (e.g. HIPAA, PCI); detection can miss data. Review before sharing.
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CasePatchStudio"
    sys_version = ""
    server: Studio

    def log_message(self, *_):
        pass  # no URLs, file names or payloads in logs

    def send(self, status, data, ctype="application/json; charset=utf-8", filename=None):
        if not isinstance(data, (bytes, bytearray)):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
                         "font-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def err(self, e: StudioError):
        payload = {"error": MESSAGES.get(e.code, e.code), "code": e.code}
        if getattr(e, "extra", None):
            payload.update(e.extra)
        return self.send(e.status, payload)

    def host_ok(self):
        from casepatch.loopback import single
        return single(self.headers, "Host") == self.server.origin.removeprefix("http://")

    def write_ok(self):
        from casepatch.loopback import single
        return (self.host_ok() and single(self.headers, "Origin") == self.server.origin
                and secrets.compare_digest(single(self.headers, "X-CasePatch-Token") or "", self.server.csrf))

    def body(self, limit=1_000_000):
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0 or n > limit:
            raise StudioError("FILE_TOO_LARGE", 413)
        return self.rfile.read(n) if n else b""

    def json_body(self):
        raw = self.body()
        v = json.loads(raw or b"{}")
        if not isinstance(v, dict):
            raise StudioError("BAD_DECISIONS")
        return v

    # ---- GET -----------------------------------------------------------------------
    def do_GET(self):
        if not self.host_ok():
            from casepatch.loopback import redirect_target, send_redirect
            from casepatch.loopback import single
            to = redirect_target(single(self.headers, "Host"), self.path, self.server.origin)
            if to:
                return send_redirect(self, to)
            return self.send(403, {"error": f"Open the app from its local address: {self.server.origin}"})
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if url.path in ("/", "/index.html"):
                return self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[:1] == ["static"] and len(parts) == 2 and re.fullmatch(r"[a-z0-9_-]+\.(js|css|svg)", parts[1]):
                f = STATIC / parts[1]
                if f.exists():
                    ct = {"js": "text/javascript", "css": "text/css", "svg": "image/svg+xml"}[parts[1].rsplit(".", 1)[1]]
                    return self.send(200, f.read_bytes(), ct + "; charset=utf-8")
                return self.send(404, {"error": "Not found."})
            if parts[:2] == ["static", "fonts"] and len(parts) == 3 and re.fullmatch(r"[A-Za-z0-9._-]+\.(woff2|txt|md)", parts[2], re.I):
                root = (STATIC / "fonts").resolve()
                f = (root / parts[2]).resolve()
                if f.is_file() and (f.parent == root):
                    ext = parts[2].rsplit(".", 1)[1]
                    ct = {"woff2": "font/woff2", "txt": "text/plain; charset=utf-8", "md": "text/markdown; charset=utf-8"}[ext]
                    return self.send(200, f.read_bytes(), ct)
                return self.send(404, {"error": "Not found."})
            if url.path == "/api/config":
                s = self.server
                return self.send(200, {"csrf": s.csrf, "version": __version__, "mode": "local", "sample": SAMPLE.exists(),
                                       "limits": {"files": MAX_FILES, "file_mb": MAX_FILE >> 20, "pages": MAX_PAGES},
                                       "missing_assets": s.missing_assets, "hosted": s.hosted_status(), "hosted_edit": s.hosted_edit_status(), "compare": s.compare_status(), "category_groups": category_group_list(), "telemetry": s.uploader.status(), "usage": {"all_time": s.meter.totals(), "this_month": s.meter.month(), "file": str(s.meter.path), "events_file": str(s.home / "events.jsonl")}})
            if url.path == "/api/events":
                return self.send(200, {"opted_in": self.server.settings.telemetry_opt_in, "events": self.server.events.read()[-200:],
                                       "would_send": self.server.uploader.build_batch(), "upload": self.server.uploader.status()})
            if parts[:2] == ["api", "compare"]:
                from . import compare_api as CMP
                from casepatch.docs.editor_pick import PickError, LeakBlocked
                s = self.server
                pack, picks = s._compare_pack, s._compare_picks
                if not pack or picks is None:
                    raise StudioError("NO_COMPARE_PACK")
                if len(parts) == 2:
                    return self.send(200, CMP.client_payload(pack, picks))
                if parts[2] == "file" and len(parts) >= 5:
                    rel = "/".join(parts[3:])
                    got = CMP.file_bytes(pack, rel)
                    if not got:
                        return self.send(404, {"error": "Not found."})
                    data, ctype = got
                    return self.send(200, data, ctype)
                if parts[2] == "page" and len(parts) == 5 and parts[4] == "after":
                    pid = CMP.page_id_from_token(pack, parts[3])
                    if not pid:
                        return self.send(404, {"error": "Not found."})
                    try:
                        return self.send(200, CMP.after_page_png(pack, pid, picks), "image/png")
                    except (PickError, LeakBlocked) as e:
                        raise StudioError("BAD_PICK")
            if len(parts) >= 3 and parts[:2] == ["api", "cases"]:
                c = self.server.get(parts[2])
                if len(parts) == 3:
                    return self.send(200, c.snapshot())
                if parts[3] == "review" and len(parts) == 4:
                    if c.status != "ready" or not c.meta:
                        raise StudioError("NOT_ANALYZED")
                    rep = self.server.report(c)
                    return self.send(200, {"case": c.snapshot(), "documents": c.meta["documents"], "files": [f["original"] for f in c.files],
                                           "proposals": self.server.proposals(c), "decisions": c.decisions,
                                           "additions": c.additions, "rejected_text": c.rejected_text,
                                           "additions_not_found": [a["text"] for a in c.additions if a["text"] not in
                                                                   {o["text"] for o in self.server.proposals(c)["occurrences"]}],
                                           "needs_review": rep.get("needs_review", []), "verification": rep.get("verification", {}),
                                           "identity": rep.get("identity"), "viewed": sorted(f"{d}:{p}" for d, p in c.viewed),
                                           "accept_text": c.accept_text, "hosted": rep.get("hosted_detection"),
                                           "category_flags": self.server.category_flags(c),
                                           "source_sent_to_cloud": bool(rep.get("source_sent_to_cloud")),
                                           "suggested": c.hints if (c.hints and c.run == 1) else None,
                                           "agent": _agent_status(c)})
                if parts[3] == "preview" and len(parts) == 6:
                    return self.preview(c, parts[4], parts[5])
                if parts[3] == "agent":
                    from . import agent_review as AR
                    if len(parts) == 4:
                        rp = bool(self.server.agent_replay)
                        return self.send(200, AR.payload(c) | {"available": rp or bool(self.server.hosted_status().get("available")), "replay": rp})
                    data = None
                    if len(parts) == 7 and parts[4] == "crop":
                        data = AR.crop_bytes(c, parts[5], parts[6])
                    elif len(parts) == 7 and parts[4] == "issue" and parts[6].isdigit():
                        data = AR.issue_bytes(c, parts[5], int(parts[6]))
                    if not data:
                        return self.send(404, {"error": "Not found."})
                    return self.send(200, data, "image/png")
                if parts[3] == "compare":
                    from . import live_compare as LC
                    from . import compare_api as CMP
                    from casepatch.docs.editor_pick import LeakBlocked, PickError
                    if len(parts) == 4:
                        return self.send(200, LC.client_payload(self.server, c))
                    if len(parts) >= 6 and parts[4] == "crop":
                        data = LC.crop_bytes(self.server, c, parts[5], parts[6] if len(parts) > 6 else "")
                        if not data:
                            return self.send(404, {"error": "Not found."})
                        return self.send(200, data, "image/png")
                    if len(parts) >= 6 and parts[4] == "file":
                        rel = "/".join(parts[5:])
                        data = LC.file_bytes(c, rel)
                        if not data:
                            return self.send(404, {"error": "Not found."})
                        return self.send(200, data, "image/png")
                    if len(parts) >= 6 and parts[4] == "pack":
                        rel = "/".join(parts[5:])
                        got = CMP.file_bytes(self.server._compare_pack, rel) if self.server._compare_pack else None
                        if not got:
                            return self.send(404, {"error": "Not found."})
                        data, ctype = got
                        return self.send(200, data, ctype)
            return self.send(404, {"error": "Not found."})
        except StudioError as e:
            return self.err(e)

    def preview(self, c: Case, doc, name):
        if not re.fullmatch(r"document-\d{2}", doc) or c.status != "ready":
            return self.send(404, {"error": "Not found."})
        m = re.fullmatch(r"(p(\d+)|json)-(before|after|diff)", name)
        if not m:
            return self.send(404, {"error": "Not found."})
        prev = c.dir / "previews"
        side = m.group(3)
        if m.group(1) == "json":
            f = prev / (f"{doc}-before.json" if side == "before" else f"{doc}-after-{c.run}.json")
            if side == "diff":
                return self.send(404, {"error": "Not found."})
            key, ctype = (doc, "json"), "application/json; charset=utf-8"
        else:
            pg = int(m.group(2))
            f = prev / {"before": f"{doc}-p{pg}-before.png", "after": f"{doc}-p{pg}-after-{c.run}.png",
                        "diff": f"{doc}-p{pg}-diff-{c.run}.png"}[side]
            key, ctype = (doc, pg), "image/png"
        if not f.exists():
            return self.send(404, {"error": "Not found."})
        if side == "after":
            c.viewed.add(key)
        return self.send(200, f.read_bytes(), ctype)

    # ---- writes ------------------------------------------------------------------------
    def do_PUT(self):
        if not self.write_ok():
            return self.send(403, {"error": f"Open the app from its local address: {self.server.origin}"})
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if len(parts) == 4 and parts[:2] == ["api", "cases"] and parts[3] == "files":
                c = self.server.get(parts[2])
                name = (parse_qs(url.query).get("name") or [""])[0]
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_FILE:
                    raise StudioError("FILE_TOO_LARGE", 413)
                self.server.add_file(c, name, self.body(MAX_FILE))
                return self.send(200, c.snapshot())
            return self.send(404, {"error": "Not found."})
        except StudioError as e:
            return self.err(e)

    def do_POST(self):
        if not self.write_ok():
            return self.send(403, {"error": f"Open the app from its local address: {self.server.origin}"})
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        s = self.server
        try:
            if parts == ["api", "cases"]:
                s.purge()
                body = self.json_body()
                c = Case(s.root)
                s.cases[c.id] = c
                if body.get("sample"):
                    s.load_sample(c)
                s.events.record("case_created", {"source": c.source}, c.ref)
                return self.send(200, c.snapshot())
            if parts == ["api", "telemetry"]:
                body = self.json_body()
                opt = body.get("opt_in") is True
                was = s.settings.telemetry_opt_in
                s.settings.set_telemetry(opt)
                if opt and not was:
                    s.events.record("telemetry_opted_in")
                if body.get("clear_log"):
                    s.events.clear()
                return self.send(200, s.uploader.status())
            if parts == ["api", "hosted-key"]:  # the user's own key: stored in memory (or the keychain), never echoed
                body = self.json_body()
                if body.get("forget") is True:
                    s.forget_user_key()
                else:
                    s.set_user_key(body.get("key"), remember=body.get("remember") is True)
                return self.send(200, {"hosted": s.hosted_status()})
            if parts == ["api", "hosted-edit-key"]:
                body = self.json_body()
                if body.get("forget") is True:
                    s.forget_fal_key()
                elif body.get("provider") == "ideogram":
                    s.set_fal_key(body.get("key"), remember=False, provider="ideogram")
                else:
                    s.set_fal_key(body.get("key"), remember=body.get("remember") is True)
                return self.send(200, {"hosted_edit": s.hosted_edit_status()})
            if parts == ["api", "compare", "picks"]:
                from . import compare_api as CMP
                from casepatch.docs.editor_pick import PickError, LeakBlocked
                if not s._compare_pack or s._compare_picks is None:
                    raise StudioError("NO_COMPARE_PACK")
                body = self.json_body()
                try:
                    return self.send(200, CMP.apply_pick(s._compare_picks, s._compare_pack, body))
                except LeakBlocked:
                    raise StudioError("LEAK_BLOCKED")
                except PickError:
                    raise StudioError("BAD_PICK")
            if parts == ["api", "compare", "export"]:
                from . import compare_api as CMP
                from casepatch.docs.editor_pick import LeakBlocked
                if not s._compare_pack or s._compare_picks is None:
                    raise StudioError("NO_COMPARE_PACK")
                body = self.json_body()
                if body.get("review_confirmed") is not True:
                    raise StudioError("CONFIRM_REVIEW")
                try:
                    data = CMP.export_zip(s._compare_pack, s._compare_picks)
                except LeakBlocked:
                    raise StudioError("LEAK_BLOCKED")
                return self.send(200, data, "application/zip", "casepatch-compare-export.zip")
            if parts == ["api", "usage", "reset"]:
                s.meter.reset()
                return self.send(200, {"usage": s.meter.totals()})
            if len(parts) >= 5 and parts[:2] == ["api", "cases"]:
                from . import live_compare as LC
                from casepatch.docs.editor_pick import LeakBlocked, PickError
                c = s.get(parts[2])
                if parts[3] == "compare" and len(parts) == 5 and parts[4] == "picks":
                    body = self.json_body()
                    try:
                        return self.send(200, LC.apply_pick(s, c, body))
                    except LeakBlocked:
                        raise StudioError("LEAK_BLOCKED")
                    except PickError:
                        raise StudioError("BAD_PICK")
                if parts[3] == "agent" and len(parts) == 5:
                    from . import agent_review as AR
                    body = self.json_body()
                    if parts[4] == "run":
                        if body.get("notice_version") != AR.NOTICE_VERSION:
                            raise StudioError("AGENT_NOTICE_REQUIRED")
                        s.start_agent(c, body)
                        return self.send(202, {"status": AR.status(c)})
                    if parts[4] == "picks":
                        try:
                            return self.send(200, AR.save_picks(c, body))
                        except PermissionError:
                            raise StudioError("LEAK_BLOCKED")
                        except ValueError:
                            raise StudioError("BAD_PICK")
                    if parts[4] == "export":
                        if body.get("review_confirmed") is not True:
                            raise StudioError("BAD_PICK")
                        if AR.status(c)["state"] != "done":
                            raise StudioError("AGENT_NOT_RUN")
                        try:
                            data = AR.export_zip(c)
                        except PermissionError:
                            raise StudioError("AGENT_UNRESOLVED", 409)
                        except RuntimeError:
                            raise StudioError("AGENT_REAPPLY_FAILED", 500)
                        return self.send(200, data, "application/zip", "casepatch-agent-export.zip")
                if parts[3] == "hosted-edit" and len(parts) == 5 and parts[4] == "estimate":
                    return self.send(200, LC.estimate(s, c, self.json_body()))
                if parts[3] == "hosted-edit" and len(parts) == 5 and parts[4] == "run":
                    return self.send(200, LC.run(s, c, self.json_body()))
            if len(parts) == 4 and parts[:2] == ["api", "cases"]:
                c = s.get(parts[2])
                action = parts[3]
                if action == "analyze":
                    body = self.json_body()
                    s.start_run(c, None, hosted={"on": body.get("hosted") is True, "notice_version": body.get("hosted_notice")},
                                categories=clean_categories(body.get("categories")))
                    return self.send(202, c.snapshot())
                if action == "apply":
                    body = self.json_body()
                    if not c.meta:
                        raise StudioError("NOT_ANALYZED")
                    prop = s.proposals(c)
                    dec = clean_review(body, {r["id"] for r in prop["regions"]},
                                       {o["text"] for o in prop["occurrences"]} | {a["text"] for a in c.additions},
                                       set(prop.get("model_suggestions", [])))
                    s.start_run(c, dec, categories=clean_categories(body.get("categories")))
                    return self.send(202, c.snapshot())
                if action == "export":
                    body = self.json_body()
                    sr = body.get("suggestions_reviewed") or []
                    if not isinstance(sr, list) or not all(isinstance(x, str) for x in sr):
                        raise StudioError("BAD_DECISIONS")
                    data = s.export(c, body.get("review_confirmed"), sr)
                    return self.send(200, data, "application/zip", "casepatch-export-reviewed.zip")
                if action == "delete":
                    s.delete(c.id)
                    return self.send(200, {"deleted": True})
            return self.send(404, {"error": "Not found."})
        except StudioError as e:
            return self.err(e)
        except (ValueError, TypeError, KeyError):
            return self.send(400, {"error": "Invalid request."})


REGION_VALID = {"replace", "keep", "erase"}
KINDS = {"PERSON", "GIVEN", "SURNAME", "ADDRESS", "PHONE", "EMAIL", "ID", "ACCOUNT", "CARD", "DOB", "ZIP", "ORG"}


def clean_review(body: dict, known_regions: set, known_texts: set, suggestions: set = frozenset()) -> dict:
    """Translate the review screen's state into pipeline decisions.

    Text decisions are value-level (every occurrence of a value is accepted or rejected together, matching the
    case-wide mapping) and are passed as `_reject_text`, so they stay valid when reviewer additions renumber
    occurrence ids in the next pass. Region decisions use region ids, which do not depend on text additions.
    """
    regions, reject, add = body.get("regions", {}), body.get("reject_text", []), body.get("add", [])
    accept = body.get("accept_text", [])
    if not isinstance(regions, dict) or not isinstance(reject, list) or not isinstance(add, list) or len(add) > 100 \
            or not isinstance(accept, list):
        raise StudioError("BAD_DECISIONS")
    if any(not isinstance(t, str) or t not in suggestions for t in accept):  # only model suggestions can be accepted
        raise StudioError("BAD_DECISIONS")
    out = {}
    for k, v in regions.items():
        if k not in known_regions or v not in REGION_VALID:
            raise StudioError("BAD_DECISIONS")
        out[k] = v
    rej = []
    for t in reject:
        if not isinstance(t, str) or t not in known_texts:
            raise StudioError("BAD_DECISIONS")
        rej.append(t)
    adds = []
    for a in add:
        if not isinstance(a, dict):
            raise StudioError("BAD_DECISIONS")
        text = a.get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 200 or a.get("kind") not in KINDS:
            raise StudioError("BAD_DECISIONS")
        adds.append({"text": text.strip(), "kind": a["kind"]})
    out["_reject_text"] = sorted(set(rej))
    out["_accept_text"] = sorted(set(accept))
    out["_add"] = adds
    return out


def serve(port=0, open_browser=False, verify=True, allow_hosted=False, allow_hosted_edit=False, compare_pack=None, agent_replay=None):
    app = Studio(port, verify=verify, allow_hosted=allow_hosted, allow_hosted_edit=allow_hosted_edit,
                 compare_pack=compare_pack, agent_replay=agent_replay)
    print(f"CasePatch Studio (local mode): {app.origin}", flush=True)
    if agent_replay:
        print(f"Agent mode is in REPLAY: a recorded run ({agent_replay}) is re-executed on local tools; no API calls, no key, $0.", flush=True)
    if allow_hosted:
        h = app.hosted_status()
        print("Hosted detection (your own Anthropic key) can be turned on per case; page images go directly to Anthropic: "
              + ("available" if h["available"] else "not available: " + h.get("reason", "")), flush=True)
    if app._compare_pack:
        print(f"Precomputed editor comparison: {len(app._compare_pack['changes'])} public-sample changes (no live calls).", flush=True)
    if allow_hosted_edit:
        h = app.hosted_edit_status()
        live = "live path ready (injected transport)" if h.get("live") else "live path tests-only: no HTTP transport is constructed; inject a fake to run"
        print("Hosted image editors (your own Fal key, off by default, never via CasePatch servers): "
              + ("key accepted" if h["available"] else "not available: " + h.get("reason", ""))
              + "; " + live, flush=True)
    print("Files stay on this computer. Press Ctrl+C to stop; open cases are deleted on exit.", flush=True)
    if open_browser:
        import webbrowser
        threading.Thread(target=webbrowser.open, args=(app.origin,), daemon=True).start()
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for cid in list(app.cases):
            app.delete(cid)
        app.server_close()
