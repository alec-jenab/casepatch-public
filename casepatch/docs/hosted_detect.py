"""Optional hosted detection on the user's documents: Claude reads each page image + its OCR lines and proposes
personal values. BRING-YOUR-OWN-KEY ONLY (see casepatch.docs.byok).

OFF by default. It runs only when ALL of these hold:
  1. the caller passes an explicit opt-in (`consent=True`, from `--hosted-detect --accept-hosted-notice` / the Studio
     toggle, after the user was shown NOTICE);
  2. the user supplied their OWN Anthropic key (ANTHROPIC_API_KEY in their environment, or entered in local Studio
     settings). Keys registered as CasePatch evaluation keys are refused; the eval spend ledger is never used;
  3. every call reserves its worst-case cost against an in-memory per-case limit first (default $2, user-adjustable
     with CASEPATCH_BYOK_MAX_USD_PER_CASE up to $50).
Requests go directly from the user's machine to api.anthropic.com. No CasePatch server is involved.

What is sent: the page image (long side <= 1600 px) and the page's OCR text lines. Nothing else (no file names, no
other pages, no case key). JSON traces are never sent.

What comes back is only a SUGGESTION: every item must be a verbatim substring of an OCR line it cites (else dropped),
then it is merged with the local rules and case-wide propagation like any detector. A value found ONLY by the model
(or only by propagating a model value) is marked HOSTED_MODEL_SUGGESTION and is NOT applied until the reviewer
accepts it on the review screen (`_accept_text`). Model output is cached in the case's local review folder, so applying
review decisions never calls the model again.

Model choice: `claude-sonnet-5-5` by default (best measured recall on the four public cases: 0.89 vs 0.62 local; see
bench/models/results). Fable and Mythos are never offered for user documents. The prompt is the one that produced
that measurement (casepatch.hosted.modeleval.EVAL_PROMPT).
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from .detect import Proposal

PROVIDER = "anthropic"
MODEL = "claude-sonnet-5-5"
NAME = f"hosted:{PROVIDER}:{MODEL}"
NOTICE_VERSION = "2026-11-12-byok"
NOTICE = ("Optional detection sends each page image of this case, and the text read from it, directly from this computer "
          "to Anthropic's API (Claude Sonnet 5.5) using your own Anthropic API key. Anthropic processes it under your "
          "agreement with Anthropic, including any data-retention terms you have with them; CasePatch servers are not "
          "involved and CasePatch does not control the provider's retention. Use it only for documents you are allowed "
          "to share with Anthropic. Model suggestions are not applied until you accept each one in review. Calls are "
          "billed to your Anthropic account; CasePatch stops a case at its per-case estimate limit.")
MAX_OUTPUT_TOKENS = 4096
MAX_IMAGE_SIDE = 1600
REVIEW_FLAG = "HOSTED_MODEL_SUGGESTION"
# model kinds -> engine kinds (BIRTHPLACE has no consistent fake yet: reported, never applied)
KIND_MAP = {"PERSON": "PERSON", "ORG": "ORG", "ADDRESS": "ADDRESS", "ZIP": "ZIP", "PHONE": "PHONE", "EMAIL": "EMAIL",
            "CARD": "CARD", "CARD_LAST4": "CARD_LAST4", "ACCOUNT": "ACCOUNT", "ID": "ID", "DOB": "DOB",
            "GIVEN": "GIVEN", "SURNAME": "SURNAME"}
CACHE = "hosted-proposals.json"


class HostedNotAllowed(Exception):
    """Hosted detection refused; args[0] is a user-facing reason (never contains a key)."""


def status(env=None, allowed=True, user_key=None):
    """What the UI may show: whether the mode can be offered, and why not. Never makes a network call, never returns
    the key (only where it came from)."""
    from . import byok
    out = {"provider": PROVIDER, "model": MODEL, "notice": NOTICE, "notice_version": NOTICE_VERSION, "available": False,
           "allowed": bool(allowed), "byok": True, "key_source": None, "keychain_available": byok.keychain_available()}
    if not allowed:
        out["reason"] = "Hosted detection is not enabled for this app (start it with --allow-hosted)."
        return out
    try:
        out["model"] = byok.model_from(env)
        out["max_usd_per_case"] = byok.max_usd_from(env)
        _key, out["key_source"] = byok.resolve_key(env, user_key)
    except byok.ByokRefused as e:
        out["reason"] = str(e.args[0]) if e.args else "hosted detection is not configured"
        return out
    out["available"] = True
    return out


class HostedPageDetector:
    name = NAME
    learned = True

    def __init__(self, client, model=MODEL, case_id="case"):
        self.c, self.model, self.case_id = client, model, case_id
        self.calls = 0
        self.pages_sent = 0
        self.rejected = 0          # items not found verbatim in the cited OCR line
        self.unsupported = 0       # kinds the engine cannot replace consistently (e.g. BIRTHPLACE)
        self.unread = 0            # values the model saw that no OCR line contains (reviewer adds them by hand)
        self.error = None          # first refusal / billing / cap error: no further calls in this run
        self.estimated_worst_usd = 0.0
        self.max_usd_per_case = None

    @classmethod
    def from_user(cls, consent, env=None, user_key=None, transport=None, case_id="case"):
        """BYOK only: the user's own key (see casepatch.docs.byok), an in-memory per-case limit, never the eval ledger."""
        if consent is not True:
            raise HostedNotAllowed("Hosted detection needs an explicit opt-in for this case.")
        from . import byok
        try:
            key, _src = byok.resolve_key(env, user_key)
            model = byok.model_from(env)
            max_usd = byok.max_usd_from(env)
        except byok.ByokRefused as e:
            raise HostedNotAllowed(str(e.args[0]) if e.args else "hosted detection is not configured") from None
        det = cls(byok.make_client(key, model, max_usd, transport=transport), model=model, case_id=case_id)
        det.max_usd_per_case = max_usd
        return det

    from_env = from_user  # old name; same BYOK rules

    # ---- page payloads
    @staticmethod
    def pages(docs):
        for d in docs:
            if d.kind == "json":
                continue
            for p in d.pages:
                if p.image is None:
                    continue
                lines = [l for l in p.lines if l.chars and l.text.strip() and l.source != "native_hidden"]
                yield d, p, [(f"L{i}", l) for i, l in enumerate(lines)]

    def _prompt(self, listing):
        from ..hosted.modeleval import page_prompt
        return page_prompt({"lines": [{"id": i, "text": l.text} for i, l in listing]}, PROVIDER)

    def estimate(self, docs):
        """Worst-case USD for the whole case (what the ledger will reserve, call by call)."""
        from ..hosted import pricing as P
        from ..hosted.client import _fit
        from ..hosted.modeleval import text_token_estimate
        tot = 0.0
        for _d, p, listing in self.pages(docs):
            img, _ = _fit(p.image, self.c.cfg.max_image_side)
            tot += P.worst_case_usd(PROVIDER, self.model, text_token_estimate(self._prompt(listing)), [img.shape[1::-1]],
                                    self.c.cfg.max_output_tokens)
        return round(tot, 4)

    def detect_docs(self, docs):
        from ..hosted import pricing as P
        from ..hosted.client import _fit, _png_b64, build_vision_request, response_text
        from ..hosted.ledger import CapExceeded
        from ..hosted.modeleval import _parse_detect, text_token_estimate
        self.estimated_worst_usd = self.estimate(docs)
        out = []
        for d, p, listing in self.pages(docs):
            if self.error:
                break
            img, _ = _fit(p.image, self.c.cfg.max_image_side)
            prompt = self._prompt(listing)
            est = P.worst_case_usd(PROVIDER, self.model, text_token_estimate(prompt), [img.shape[1::-1]], self.c.cfg.max_output_tokens)
            b64 = _png_b64(img)
            try:
                data = self.c.call("vision", self.model,
                                   lambda: build_vision_request(PROVIDER, self.model, prompt, b64, self.c.cfg.max_output_tokens),
                                   est, self.case_id)
            except CapExceeded as e:
                self.error = "CASE_LIMIT_REACHED: " + str(e)[:200]; break
            except Exception as e:  # refusal, billing, auth, timeout: stop (no retries, no further pages)
                self.error = "HOSTED_CALL_FAILED: " + str(e)[:200]; break
            self.calls += 1; self.pages_sent += 1
            out += self.ground(d, p, listing, _parse_detect(response_text(data, PROVIDER)))
        return out

    def ground(self, d, p, listing, parsed):
        by_id = dict(listing)
        props = []
        self.unread += len(parsed.get("unread") or []) if isinstance(parsed.get("unread"), list) else 0
        for it in parsed.get("items") or [] if isinstance(parsed.get("items"), list) else []:
            if not isinstance(it, dict):
                continue
            kind = KIND_MAP.get(str(it.get("kind", "")).upper())
            line = by_id.get(str(it.get("line", it.get("line_id", ""))))
            t = str(it.get("text", ""))
            if kind is None:
                self.unsupported += 1; continue
            if line is None or len(t.strip()) < 2:
                self.rejected += 1; continue
            hits = [(m.start(), m.end()) for m in re.finditer(re.escape(t), line.text)]
            if not hits:
                hits = [(m.start(), m.end()) for m in re.finditer(r"\s+".join(re.escape(x) for x in t.split()), line.text, re.I)]
            if not hits:
                self.rejected += 1; continue
            pid = it.get("person_id")
            hint = f"hosted:{d.id}:{p.index}:{pid}" if pid else None
            for s, e in hits:
                props.append(Proposal(line.id, s, e, kind, line.text[s:e], "hosted", 0.9, hint, {"hosted_model": self.model}))
        return props

    def info(self):
        return {"name": self.name, "learned": True, "provider": PROVIDER, "model": self.model, "calls": self.calls,
                "pages_sent": self.pages_sent, "ungrounded_rejected": self.rejected, "unsupported_kind_items": self.unsupported,
                "values_not_in_ocr": self.unread, "estimated_worst_case_usd": self.estimated_worst_usd,
                "error": self.error, "notice_version": NOTICE_VERSION, "byok": True,
                "max_usd_per_case": self.max_usd_per_case}


def save_cache(review_dir: Path, det: HostedPageDetector, props):
    f = Path(review_dir) / CACHE
    data = {"model": det.model, "info": det.info(),
            "proposals": [dataclasses.asdict(p) for p in props]}
    f.write_text(json.dumps(data, ensure_ascii=False))
    f.chmod(0o600)


def load_cache(review_dir: Path):
    f = Path(review_dir) / CACHE
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    return [Proposal(**p) for p in data["proposals"]], dict(data["info"], cached=True)


def model_only_texts(occs):
    """Values for which no local detector agreed: every occurrence came from the model or from propagating it."""
    by_text = {}
    for o in occs:
        by_text.setdefault(o.text, []).append(o)
    out = set()
    for t, group in by_text.items():
        dets = set().union(*(o.detectors for o in group))
        if "hosted" in dets and not (dets - {"hosted", "propagation"}):
            out.add(t)
    return out
