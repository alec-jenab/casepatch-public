"""CasePatch: anonymize a case (documents plus agent traces) for AI agent evals, locally or via the hosted API.
`from casepatch import CasePatch` is the SDK; `casepatch` is the CLI."""

__version__ = "0.6.2"


def __getattr__(name):  # `from casepatch import CasePatch` without importing the client on every engine import
    if name in ("CasePatch", "CasePatchError", "Result", "local_available"):
        from . import client
        return getattr(client, name)
    raise AttributeError(name)
