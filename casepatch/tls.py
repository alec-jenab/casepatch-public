"""Certificate trust for CasePatch's own HTTPS calls (cloud API, model downloads, hosted adapters). Standard library only.

Verification is always on (CERT_REQUIRED + hostname check); this module only decides WHICH trusted roots to load:
1. SSL_CERT_FILE / SSL_CERT_DIR set: exactly what they name (custom or corporate CA bundles).
2. The interpreter's default store when it has CAs (Linux, Windows, Homebrew/system Python, python.org Python after
   "Install Certificates.command").
3. Otherwise (python.org macOS Python out of the box: an empty store, every HTTPS call fails with
   CERTIFICATE_VERIFY_FAILED): the OS trust store via `truststore` if installed, else `certifi`'s bundle if installed,
   else macOS's built-in system roots (exported by /usr/bin/security from SystemRootCertificates.keychain).
"""
from __future__ import annotations

import os
import ssl
import subprocess
import sys
from typing import Optional

MACOS_ROOTS = "/System/Library/Keychains/SystemRootCertificates.keychain"
_cache: dict = {}


def _default_store_usable(ctx: ssl.SSLContext) -> bool:
    if ctx.cert_store_stats().get("x509_ca", 0):
        return True
    p = ssl.get_default_verify_paths()  # a CA directory (Linux) is loaded lazily, so the stats above can read 0
    return bool((p.cafile and os.path.isfile(p.cafile)) or (p.capath and os.path.isdir(p.capath) and os.listdir(p.capath)))


def _macos_system_roots() -> Optional[str]:
    if sys.platform != "darwin" or not os.path.exists("/usr/bin/security") or not os.path.exists(MACOS_ROOTS):
        return None
    try:
        out = subprocess.run(["/usr/bin/security", "find-certificate", "-a", "-p", MACOS_ROOTS],
                             capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out if "BEGIN CERTIFICATE" in out else None


def _build() -> "tuple[ssl.SSLContext, str]":
    ctx = ssl.create_default_context()  # honours SSL_CERT_FILE / SSL_CERT_DIR
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return ctx, "SSL_CERT_FILE/SSL_CERT_DIR"
    if _default_store_usable(ctx):
        return ctx, "default"
    try:
        import truststore  # optional: the real OS trust store, with the OS's own trust evaluation
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), "truststore"
    except ImportError:
        pass
    try:
        import certifi  # optional
        ctx.load_verify_locations(cafile=certifi.where())
        return ctx, "certifi"
    except ImportError:
        pass
    pem = _macos_system_roots()
    if pem:
        ctx.load_verify_locations(cadata=pem)
        return ctx, "macos-system-roots"
    return ctx, "default (empty)"


def ssl_context() -> ssl.SSLContext:
    """A verifying client context for CasePatch's HTTPS calls (cached per SSL_CERT_FILE/SSL_CERT_DIR value)."""
    return _context_and_source()[0]


def trust_source() -> str:
    """Which roots `ssl_context()` uses (for diagnostics; never a secret)."""
    return _context_and_source()[1]


def _context_and_source():
    k = (os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR"))
    if k not in _cache:
        _cache.clear()
        _cache[k] = _build()
    return _cache[k]


def network_hint(err: BaseException) -> str:
    """One short sentence on how to fix a failed HTTPS call, or '' when there is no specific advice."""
    reason = getattr(err, "reason", err)
    if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
        return ("The server's TLS certificate could not be verified with this Python's trusted roots. Behind a proxy "
                "or with a custom CA, set SSL_CERT_FILE to your CA bundle; on python.org macOS Python you can also run "
                "'Install Certificates.command' or pip install truststore.")
    if isinstance(reason, (TimeoutError, OSError)) or "timed out" in str(reason):
        return "Check the network connection (and any proxy or firewall), then retry."
    return ""


def describe(err: BaseException) -> str:
    """The short reason of a urllib/socket error, without a traceback."""
    reason = getattr(err, "reason", None) or err
    return str(reason) or type(err).__name__
