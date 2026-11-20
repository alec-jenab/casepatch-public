"""Opening a local UI as http://localhost:PORT (what many tools and people type) instead of http://127.0.0.1:PORT.

The local servers bind 127.0.0.1 only and accept exactly one origin, http://127.0.0.1:PORT, for every API call (Host,
Origin and the CSRF token are all checked). A GET whose Host is exactly `localhost:PORT` (same port) is redirected to
the same path on that origin. The target is built from the server's own origin, never from the request, so the
redirect cannot point anywhere else; any other Host (a rebinding domain, another port) keeps getting 403.
"""
from __future__ import annotations

from typing import Optional


def redirect_target(host_header: Optional[str], path: str, origin: str) -> Optional[str]:
    """The URL to redirect a GET to, or None when the request must be refused as before."""
    port = origin.rsplit(":", 1)[1]
    if (host_header or "").lower() != f"localhost:{port}":
        return None
    if not path.startswith("/") or path.startswith("//") or "\\" in path or any(c in path for c in "\r\n"):
        path = "/"
    return origin + path


def single(headers, name: str):
    """The header's value only when it appears exactly once; a repeated Host/Origin is refused, never 'first wins'."""
    vals = headers.get_all(name) or []
    return vals[0] if len(vals) == 1 else None


def send_redirect(handler, location: str) -> None:
    handler.send_response(307)
    handler.send_header("Location", location)
    handler.send_header("Content-Length", "0")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Referrer-Policy", "no-referrer")
    handler.end_headers()
