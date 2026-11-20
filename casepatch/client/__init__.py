"""CasePatch Python client: anonymize documents locally (in-process engine, nothing leaves the machine) or with the
hosted API.

    from casepatch import CasePatch
    cp = CasePatch(mode="local")                    # needs: pip install "casepatch[local]"
    cp = CasePatch(api_key="cpk_...")               # cloud; or set CASEPATCH_API_KEY
    result = cp.anonymize("file.pdf", out="anon.pdf")
"""
from .core import CasePatch, CasePatchError, Result, local_available

__all__ = ["CasePatch", "CasePatchError", "Result", "local_available"]
