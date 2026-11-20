"""Loopback-only local review app. No source files or case maps are written to disk."""
from __future__ import annotations

from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
import io
import json
import secrets
import time
import zipfile

from .core import PolicyError
from .review import ReviewCase, RulesDetector, SpacyDetector

MAX_REQUEST = 4_000_000
CASE_TTL = 1800
MAX_CASES = 8


def archive(result):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for index, doc in enumerate(result["documents"]):
            suffix = "txt" if doc["format"] == "text" else "json"
            data = doc["content"] if suffix == "txt" else json.dumps(doc["content"], ensure_ascii=False, indent=2, allow_nan=False)
            bundle.writestr(f"document-{index+1:03}.{suffix}", data)
        bundle.writestr("report.json", json.dumps(result["report"], indent=2))
    return buffer.getvalue()


class LocalApp(HTTPServer):
    def __init__(self, port=0, detector=None):
        self.detector = detector if detector is not None else SpacyDetector()
        self.csrf = secrets.token_urlsafe(32)
        self.cases = OrderedDict()
        super().__init__(("127.0.0.1", port), Handler)
        self.origin = "http://127.0.0.1:" + str(self.server_port)

    def purge(self):
        now = time.monotonic()
        for cid in list(self.cases):
            if now - self.cases[cid]["at"] > CASE_TTL:
                del self.cases[cid]

    def case(self, cid):
        self.purge()
        if not isinstance(cid, str) or cid not in self.cases:
            raise PolicyError("CASE_EXPIRED_IMPORT_AGAIN")
        self.cases[cid]["at"] = time.monotonic()
        return self.cases[cid]

    def service_actions(self):
        self.purge()


ERRORS = {
    "LOCAL_MODEL_NOT_INSTALLED": "The local name model (spaCy en_core_web_sm) is not installed. Install it once: "
                                 "pip install \"casepatch[ner]\" && python -m spacy download en_core_web_sm - or restart "
                                 "in limited mode: casepatch app --rules-only",
    "SENSITIVE_JSON_KEYS_UNSUPPORTED": "A JSON key contains detected sensitive data. Rename the key locally before importing.",
    "LINK_OR_KEEP_AMBIGUOUS_INITIAL": "Link the ambiguous initial to the correct person, or choose Keep for that occurrence.",
    "LINK_OR_KEEP_AMBIGUOUS_NAME": "Link the ambiguous name part to the correct person, or choose Keep for that occurrence.",
    "CONFIRM_OR_SPLIT_IDENTITY": "Different customer or patient IDs share this name. Confirm one identity, split the occurrences, or choose Keep.",
    "FIRST_AND_LAST_NAME_MUST_BE_TEXT_PAIR": "First and last name fields must both contain text. Use a full_name field for an unpaired name.",
    "REPLACEMENT_SPACE_EXHAUSTED": "This case has more names than this preview can replace. Use a smaller case.",
    "USE_ONE_TO_250000_TEXT_CHARACTERS": "Import between 1 and 250,000 text characters.",
    "CASE_EXPIRED_IMPORT_AGAIN": "This case expired. Import it again.",
    "UNSUPPORTED_DATE_FOR_SHIFT": "This date cannot be shifted. Keep it or edit the source before importing.",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "CasePatchLocal"
    sys_version = ""
    timeout = 10

    def log_message(self, *_):
        pass  # No URL, payload, source text, request header or traceback logs.

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def finish(self):
        try:super().finish()
        except (BrokenPipeError, ConnectionResetError):pass

    def send(self, status, data, content_type="application/json; charset=utf-8", download=False):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        if download:self.send_header("Content-Disposition", 'attachment; filename="casepatch-review-required.zip"')
        self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass

    def valid_host(self):
        from .loopback import single
        return single(self.headers, "Host") == self.server.origin.removeprefix("http://")

    def do_GET(self):
        if not self.valid_host():
            from .loopback import redirect_target, send_redirect
            from .loopback import single
            to = redirect_target(single(self.headers, "Host"), self.path, self.server.origin)
            if to:
                return send_redirect(self, to)
            return self.send(403, {"error": f"Open the app from its local address: {self.server.origin}"})
        if self.path == "/api/config":
            return self.send(200, {"csrf":self.server.csrf,"detector":self.server.detector.name,
                                   "learned_detector":self.server.detector.learned})
        if self.path == "/api/example":
            fixture=json.loads(files("casepatch").joinpath("fixtures/dispute.json").read_text())
            return self.send(200,{"documents":[{"format":"json","content":fixture["records"][:3]},
                                             {"format":"text","content":fixture["records"][3]["transcript"]}]})
        assets={"/":"index.html","/app.js":"app.js","/json-import.js":"json-import.js","/styles.css":"styles.css"}
        types={"/":"text/html; charset=utf-8","/app.js":"text/javascript; charset=utf-8","/json-import.js":"text/javascript; charset=utf-8","/styles.css":"text/css; charset=utf-8"}
        if self.path in assets:
            return self.send(200,files("casepatch").joinpath("web",assets[self.path]).read_bytes(),types[self.path])
        return self.send(404,{"error":"Not found."})

    def do_POST(self):
        from .loopback import single
        if not self.valid_host() or single(self.headers, "Origin") != self.server.origin or not secrets.compare_digest(single(self.headers, "X-CasePatch-Token") or "",self.server.csrf):
            return self.send(403,{"error":f"Open the app from its local address: {self.server.origin}"})
        if self.headers.get("Content-Type","").split(";")[0] != "application/json":
            return self.send(415,{"error":"JSON requests required."})
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0 < length <= MAX_REQUEST:
                return self.send(413,{"error":"This import is too large."})
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise PolicyError("REQUEST_SCHEMA")
            if self.path=="/api/analyze":
                self.server.purge()
                if len(self.server.cases)>=MAX_CASES:
                    return self.send(429,{"error":"Clear a case before importing another."})
                case=ReviewCase(body.get("documents"),self.server.detector,body.get("manual",[]))
                cid=secrets.token_urlsafe(24)
                self.server.cases[cid]={"case":case,"at":time.monotonic(),"result":None}
                return self.send(200,{"case_id":cid,**case.snapshot()})
            if self.path=="/api/transform":
                entry=self.server.case(body.get("case_id"))
                result=entry["case"].transform(body.get("decisions"))
                entry["result"]=result
                return self.send(200,result)
            if self.path=="/api/export":
                entry=self.server.case(body.get("case_id"))
                if entry["result"] is None:raise PolicyError("PREVIEW_BEFORE_EXPORT")
                if body.get("review_confirmed") is not True:raise PolicyError("CONFIRM_REVIEW_BEFORE_EXPORT")
                return self.send(200,archive(entry["result"]),"application/zip",True)
            if self.path=="/api/clear":
                cid=body.get("case_id")
                if isinstance(cid,str):self.server.cases.pop(cid,None)
                return self.send(200,{"cleared":True})
            return self.send(404,{"error":"Not found."})
        except PolicyError as exc:
            code=str(exc)
            return self.send(400,{"error":ERRORS.get(code,"Check the import or review choices. Code: "+code),"code":code})
        except (ValueError,TypeError,KeyError,RecursionError,OverflowError):
            return self.send(400,{"error":"Invalid input. Check the supported format and try again."})
        except Exception:
            return self.send(500,{"error":"Processing stopped. No output was exported."})


def serve(port=0, rules_only=False, open_browser=False):
    app=LocalApp(port,RulesDetector() if rules_only else None)
    print("CasePatch local review: "+app.origin,flush=True)
    print("No cloud processing. No source files or maps are stored. Press Ctrl+C to stop.",flush=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Thread(target=webbrowser.open,args=(app.origin,),daemon=True).start()
    try:app.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        app.cases.clear()
        app.server_close()
