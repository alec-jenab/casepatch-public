"""Hosted API backend (standard library only)."""
from __future__ import annotations

import io
import json
import time
import zipfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from .. import __version__
from ..tls import describe, network_hint, ssl_context
from .core import CASE_SPLIT_REFUSED, CasePatchError, Result, report_fields

TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".json": "application/json", ".jsonl": "application/jsonl", ".ndjson": "application/jsonl",
         ".txt": "text/plain; charset=utf-8"}
# transport(method, url, headers, body, timeout) -> (status, headers dict, body bytes); replaceable in tests
Transport = Callable[[str, str, dict, Optional[bytes], float], "tuple[int, dict, bytes]"]


def urllib_transport(method: str, url: str, headers: dict, body: Optional[bytes], timeout: float):
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise CasePatchError(" ".join(filter(None, [f"Could not reach the CasePatch API: {describe(e)}.", network_hint(e)]))) from None


class CloudBackend:
    def __init__(self, api_key: str, base_url: str, timeout: float = 900.0, poll_interval: float = 1.5,
                 transport: Optional[Transport] = None, request_timeout: float = 120.0, max_retries: int = 4):
        self.key, self.base = api_key, base_url.rstrip("/")
        self.timeout, self.poll = timeout, poll_interval
        self.http: Transport = transport or urllib_transport
        self.request_timeout, self.max_retries = request_timeout, max_retries

    def _call(self, method: str, path: str, body: Optional[bytes] = None, ctype: Optional[str] = None, raw: bool = False,
              extra: Optional[dict] = None):
        headers = {"Authorization": f"Bearer {self.key}", "User-Agent": f"casepatch-python/{__version__}",
                   "Accept": "application/json", **(extra or {})}
        if ctype:
            headers["Content-Type"] = ctype
        for attempt in range(self.max_retries + 1):
            status, hdrs, data = self.http(method, self.base + path, headers, body, self.request_timeout)
            if status in (429, 503) and attempt < self.max_retries and "limit for this API key" not in data.decode("utf-8", "replace"):
                time.sleep(min(30.0, float({k.lower(): v for k, v in hdrs.items()}.get("retry-after") or 2 ** (attempt + 1))))
                continue
            break
        if status >= 400:
            try:
                msg = json.loads(data).get("error") or f"HTTP {status}"
            except ValueError:
                msg = f"HTTP {status}"
            raise CasePatchError(msg, status)
        return data if raw else json.loads(data or b"{}")

    def usage(self) -> dict[str, Any]:
        return self._call("GET", "/api/v1/usage")

    def submit(self, src: Path, case_key: Optional[str] = None) -> dict[str, Any]:
        return self._call("POST", "/api/v1/anonymize", src.read_bytes(), TYPES.get(src.suffix.lower(), "application/octet-stream"),
                          extra=_case_header(case_key))

    def submit_case(self, files: list[tuple[str, Path]], case_key: Optional[str] = None) -> dict[str, Any]:
        """One case job: every (name, path) in one ZIP, so all documents share one identity map."""
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
            for name, path in files:
                z.write(path, name)
        return self._call("POST", "/api/v1/anonymize", b.getvalue(), "application/zip", extra=_case_header(case_key))

    def job(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/jobs/{job_id}")

    def wait(self, job_id: str) -> dict[str, Any]:
        end = time.monotonic() + self.timeout
        while True:
            j = self.job(job_id)
            if j["status"] in ("done", "error"):
                return j
            if time.monotonic() > end:
                raise CasePatchError(f"Timed out waiting for job {job_id}")
            time.sleep(self.poll)

    def download(self, job_id: str, out: Path, fmt: str = "file") -> Path:
        data = self._call("GET", f"/api/v1/jobs/{job_id}/result" + ("?format=zip" if fmt == "zip" else ""), raw=True)
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(out)
        return out

    def delete(self, job_id: str) -> None:
        self._call("DELETE", f"/api/v1/jobs/{job_id}")

    def anonymize(self, src: Path, dst: Path, case_key: Optional[str] = None) -> Result:
        j = self.submit(src, case_key)
        jid = j["id"]
        try:
            j = self.wait(jid)
            if j["status"] != "done":
                return Result(source=str(src), output=None, mode="cloud", status="error", pages=j.get("pages", 0),
                              job_id=jid, error=j.get("error") or "Processing failed.")
            if len(j.get("files") or []) > 1:
                dst = dst.with_suffix(".zip")
            self.download(jid, dst)
            return Result(source=str(src), output=str(dst), mode="cloud", pages=j.get("pages", 0), job_id=jid,
                          summary=j.get("summary") or {}, changes=j.get("changes") or [], **report_fields(j, sent=True))
        finally:
            try:
                self.delete(jid)  # don't wait for the 15-minute expiry
            except CasePatchError:
                pass

    def anonymize_case(self, items: list[tuple[Path, Path]], case_key: Optional[str] = None) -> dict[Path, Result]:
        """Several files as ONE case job (one identity map, shared `entity` ids). Each output goes to its target."""
        names, seen = [], set()
        for i, (src, _) in enumerate(items):
            n = src.name if src.name not in seen else f"{i:03d}_{src.name}"
            seen.add(n); names.append(n)
        # the case runs as ONE job or not at all: splitting it into per-file jobs would give the files separate identity
        # maps (a name found in one file would not be searched in the others)
        def all_failed(msg, account=False):
            msg = msg if account else CASE_SPLIT_REFUSED.format(err=msg)
            return {src: Result(source=str(src), output=None, mode="cloud", status="error", error=msg) for src, _ in items}
        try:
            j = self.submit_case(list(zip(names, [s for s, _ in items])), case_key)
        except CasePatchError as e:
            return all_failed(e.message, account=e.status in (401, 402, 403, 429))
        jid = j["id"]
        try:
            j = self.wait(jid)
            if j["status"] != "done":
                self._quiet_delete(jid); jid = None
                return all_failed(j.get("error") or "Processing failed.")
            z = zipfile.ZipFile(io.BytesIO(self._call("GET", f"/api/v1/jobs/{jid}/result?format=zip", raw=True)))
            back = {v: k for k, v in (j.get("sources") or json.loads(z.read("sources.json"))).items()}
            out = {}
            for (src, dst), name in zip(items, names):
                member = back.get(name)
                if not member:
                    out[src] = Result(source=str(src), output=None, mode="cloud", status="error", job_id=jid,
                                      error="The case job returned no output for this file.")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(z.read(member)); tmp.replace(dst)
                ch = [c for c in (j.get("changes") or []) if name in (c.get("documents") or [])]
                summ: dict[str, int] = {}
                for c in ch:
                    summ[c.get("kind") or "OTHER"] = summ.get(c.get("kind") or "OTHER", 0) + 1
                out[src] = Result(source=str(src), output=str(dst), mode="cloud", job_id=jid, summary=summ, changes=ch,
                                  **report_fields(j, docs=[name], sent=True))
            return out
        finally:
            if jid:
                self._quiet_delete(jid)

    def _quiet_delete(self, jid):
        try:
            self.delete(jid)
        except CasePatchError:
            pass


def _case_header(case_key: Optional[str]) -> dict:
    # a header, not a query parameter, so the key never lands in URL logs
    return {"X-CasePatch-Case-Key": str(case_key)} if case_key else {}
