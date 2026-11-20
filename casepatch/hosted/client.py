"""Transport, capped call loop, and the two adapters (vision detector/linker, image editor)."""
from __future__ import annotations

import base64
import io
import json
import logging
import uuid

import numpy as np

from ..tls import ssl_context
from . import pricing as P
from .config import POOL_MAX_USD, HostedConfig, HostedDisabled, redact
from .ledger import SpendLedger

log = logging.getLogger("casepatch.hosted")

ENDPOINTS = {
    ("openai", "vision"): "https://api.openai.com/v1/responses",
    ("openai", "edit"): "https://api.openai.com/v1/images/edits",
    ("gemini", "vision"): "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    ("gemini", "edit"): "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    ("kimi", "vision"): "https://api.moonshot.ai/v1/chat/completions",  # OpenAI-compatible; no image-edit endpoint
    ("anthropic", "vision"): "https://api.anthropic.com/v1/messages",   # Messages API; no image-edit endpoint
    ("fal", "vision"): "https://fal.run/{model}",  # synchronous model API; model = fal endpoint id
    ("fal", "edit"): "https://fal.run/{model}",
}
FAL_MASK_ENDPOINTS = {"fal-ai/qwen-image-edit/inpaint"}           # take image_url + mask_url (white = repaint)
FAL_SINGLE_IMAGE = {"fal-ai/flux-kontext/dev", "fal-ai/flux-pro/kontext"}  # image_url; others take image_urls[]
FAL_EDIT_SEED = 7


ANTHROPIC_VERSION = "2026-10-21"  # as shown in the current official API examples (read 2026-11-11)
ANTHROPIC_EFFORT = "medium"


class TransportTimeout(Exception):
    pass


class HttpTransport:
    """The only class that touches the network. Constructed only from a validated HostedConfig."""

    def __init__(self, cfg: HostedConfig):
        if not isinstance(cfg, HostedConfig):
            raise HostedDisabled("HttpTransport requires a validated HostedConfig")
        self.cfg = cfg

    def send(self, url, headers, body, timeout):
        import socket
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as r:
                return r.status, json.loads(r.read().decode())
        except (socket.timeout, TimeoutError) as e:
            raise TransportTimeout(redact(e, [self.cfg.key()]))
        except urllib.error.HTTPError as e:
            return e.code, {"error": redact(e.read()[:2000].decode(errors="replace"), [self.cfg.key()])}


def _png_b64(arr):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fname, data, ctype) in files.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
        out.write(data); out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


class HostedClient:
    """Every attempt (first try, each retry, each image edit) is its own reservation in the ledger."""

    def __init__(self, cfg: HostedConfig, transport=None, ledger=None):
        self.cfg = cfg
        self.transport = transport or HttpTransport(cfg)
        self.ledger = ledger or SpendLedger(cfg.ledger_path)

    def _headers(self, kind, ctype="application/json"):
        if self.cfg.provider in ("openai", "kimi"):
            return {"Authorization": "Bearer " + self.cfg.key(), "Content-Type": ctype}
        if self.cfg.provider == "fal":
            return {"Authorization": "Key " + self.cfg.key(), "Content-Type": ctype}
        if self.cfg.provider == "anthropic":
            return {"x-api-key": self.cfg.key(), "anthropic-version": ANTHROPIC_VERSION, "Content-Type": ctype}
        return {"x-goog-api-key": self.cfg.key(), "Content-Type": ctype}

    def call(self, kind, model, build, estimate_usd, case_id):
        """build() -> (url, body bytes, content-type). Returns parsed JSON or raises. Never logs/returns the key."""
        P.price(self.cfg.provider, model)  # fail closed before anything else
        last = None
        for attempt in range(self.cfg.max_retries + 1):
            rid = self.ledger.reserve(estimate_usd, self.cfg.cap_usd, case_id=case_id, provider=self.cfg.provider, model=model,
                                      kind=kind, attempt=attempt, max_calls_per_case=self.cfg.max_calls_per_case,
                                      provider_cap_usd=self.cfg.provider_cap_usd, pool=self.cfg.pool,
                                      pool_cap_usd=self.cfg.pool_cap_usd if self.cfg.pool_cap_usd is not None
                                      else min(POOL_MAX_USD[self.cfg.pool], self.cfg.cap_usd))
            try:
                url, body, ctype = build()
                status, data = self.transport.send(url, self._headers(kind, ctype), body, self.cfg.timeout_s)
            except TransportTimeout as e:
                self.ledger.keep(rid, "timeout: outcome unknown"); last = redact(e, [self.cfg.key()])
                log.warning("hosted %s timeout (reservation kept): %s", kind, last)
                continue
            except Exception as e:  # unknown outcome -> keep reservation, do not retry blindly
                self.ledger.keep(rid, "error: outcome unknown")
                raise RuntimeError(redact(f"hosted call failed: {e}", [self.cfg.key()])) from None
            if status == 429 or status >= 500:
                self.ledger.keep(rid, f"http {status}")  # may or may not have been billed: keep
                err = (data or {}).get("error", "")
                tag = f"[{err.get('code') or ''}/{err.get('type') or ''}/{err.get('status') or ''}] " if isinstance(err, dict) else ""
                last = redact(f"http {status} {tag}{str(err)[:240]}", [self.cfg.key()]); continue
            if status != 200:
                self.ledger.keep(rid, f"http {status}")
                raise RuntimeError(redact(f"hosted call rejected: http {status} {data.get('error', '')}", [self.cfg.key()]))
            usage = (data.get("usageMetadata") if self.cfg.provider == "gemini"
                     else data if self.cfg.provider == "fal" else data.get("usage"))  # fal bills output units
            try:
                cost = P.actual_usd(self.cfg.provider, model, usage)
                self.ledger.commit(rid, cost)
            except (KeyError, TypeError, ValueError):
                self.ledger.keep(rid, "response lacks usage fields: estimate kept")
            return data
        raise RuntimeError(redact(f"hosted call gave up after {self.cfg.max_retries + 1} attempts: {last}", [self.cfg.key()]))


def _fit(arr, side):
    import cv2
    h, w = arr.shape[:2]
    s = min(1.0, side / max(h, w))
    return (cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), s) if s < 1 else (arr, 1.0)


DETECT_PROMPT = ("You receive one document page image and its OCR lines (id: text). Return JSON only: "
                 "{\"items\":[{\"line_id\":str,\"text\":str,\"kind\":\"PERSON|GIVEN|SURNAME|ORG|ADDRESS|ZIP|PHONE|EMAIL|CARD|CARD_LAST4|ACCOUNT|ID|DOB\","
                 "\"person_id\":str|null}]} . Every text must be copied verbatim from the given line. Same person_id = same human. "
                 "Do not include amounts or ordinary dates.")


def build_vision_request(provider, model, prompt_text, b64_png, max_output_tokens, gemini_thinking=None):
    """(url, body bytes, content-type) for one page-image + text request. Shared by the detector and the eval."""
    if provider == "openai":
        body = {"model": model, "max_output_tokens": max_output_tokens, "store": False,
                "text": {"format": {"type": "json_object"}},
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text},
                                                       {"type": "input_image", "image_url": "data:image/png;base64," + b64_png,
                                                        "detail": "high"}]}]}
        return ENDPOINTS[("openai", "vision")], json.dumps(body).encode(), "application/json"
    if provider == "anthropic":
        # adaptive thinking is billed as output and counted inside max_tokens (hard limit on thinking + text)
        body = {"model": model, "max_tokens": max_output_tokens, "output_config": {"effort": ANTHROPIC_EFFORT},
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_png}},
                    {"type": "text", "text": prompt_text}]}]}
        return ENDPOINTS[("anthropic", "vision")], json.dumps(body).encode(), "application/json"
    if provider == "fal":  # NVIDIA Nemotron vision on fal: one image URL (data URI) + prompt -> {"output": text}
        body = {"prompt": prompt_text, "image_url": "data:image/png;base64," + b64_png, "max_tokens": max_output_tokens,
                "reasoning_mode": "no_think"}
        return ENDPOINTS[("fal", "vision")].format(model=model), json.dumps(body).encode(), "application/json"
    if provider == "kimi":
        # thinking is always on for K3.x and is billed as completion tokens: bound it with max_completion_tokens and
        # low reasoning effort; content must be an array of parts; JSON mode keeps the final content parseable
        body = {"model": model, "max_completion_tokens": max_output_tokens, "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_png}},
                    {"type": "text", "text": prompt_text}]}]}
        return ENDPOINTS[("kimi", "vision")], json.dumps(body).encode(), "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt_text},
                                                    {"inline_data": {"mime_type": "image/png", "data": b64_png}}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens, "responseMimeType": "application/json"}}
    if gemini_thinking:  # Gemini 3+: bound thinking latency/cost ("low" | "high")
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": gemini_thinking}
    return ENDPOINTS[("gemini", "vision")].format(model=model), json.dumps(body).encode(), "application/json"


class VisionDetector:
    """Hosted vision-LLM detector + identity linker. Output is grounded: an item survives only if its text occurs
    verbatim in the OCR line it cites."""

    def __init__(self, client: HostedClient, model=None):
        self.c = client; self.model = model or P.DEFAULT_MODELS[client.cfg.provider]["vision"]

    def estimate(self, page_rgb, lines):
        img, _ = _fit(page_rgb, self.c.cfg.max_image_side)
        text_tokens = (len(DETECT_PROMPT) + sum(len(t) + 12 for _, t in lines)) // 3 + 50
        return P.worst_case_usd(self.c.cfg.provider, self.model, text_tokens, [img.shape[1::-1]], self.c.cfg.max_output_tokens)

    def detect(self, page_rgb, lines, case_id):
        cfg = self.c.cfg
        img, _ = _fit(page_rgb, cfg.max_image_side)
        listing = "\n".join(f"{i}: {t}" for i, t in lines)
        b64 = _png_b64(img)

        def build():
            if cfg.provider in ("openai", "kimi", "anthropic", "fal"):
                return build_vision_request(cfg.provider, self.model, DETECT_PROMPT + "\n" + listing, b64, cfg.max_output_tokens)
            if cfg.provider == "openai":
                body = {"model": self.model, "max_output_tokens": cfg.max_output_tokens, "store": False,
                        "text": {"format": {"type": "json_object"}},
                        "input": [{"role": "user", "content": [{"type": "input_text", "text": DETECT_PROMPT + "\n" + listing},
                                                               {"type": "input_image", "image_url": "data:image/png;base64," + b64}]}]}
                return ENDPOINTS[("openai", "vision")], json.dumps(body).encode(), "application/json"
            body = {"contents": [{"role": "user", "parts": [{"text": DETECT_PROMPT + "\n" + listing},
                                                            {"inline_data": {"mime_type": "image/png", "data": b64}}]}],
                    "generationConfig": {"maxOutputTokens": cfg.max_output_tokens, "responseMimeType": "application/json"}}
            return ENDPOINTS[("gemini", "vision")].format(model=self.model), json.dumps(body).encode(), "application/json"
        data = self.c.call("vision", self.model, build, self.estimate(page_rgb, lines), case_id)
        return ground(parse_json_text(data, cfg.provider), dict(lines))


def response_text(data, provider):
    if provider == "openai":
        return data.get("output_text") or "".join(c.get("text", "") for o in data.get("output", []) for c in o.get("content", [])
                                                  if c.get("type", "output_text") == "output_text")
    if provider == "anthropic":  # only text blocks, never thinking blocks
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    if provider == "fal":  # only the answer, never the reasoning trace
        return data.get("output") or ""
    if provider == "kimi":  # only the final content, never reasoning_content
        return data["choices"][0]["message"].get("content") or ""
    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"] if not p.get("thought"))


def parse_json_text(data, provider):
    try:
        if provider in ("openai", "kimi", "anthropic", "fal"):
            txt = response_text(data, provider)
        else:
            txt = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        return json.loads(txt).get("items", [])
    except Exception:
        return []


def ground(items, lines):
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        lid, t = str(it.get("line_id", "")), str(it.get("text", ""))
        src = lines.get(lid)
        if not t or src is None or t not in src:
            continue  # ungrounded: dropped
        s = src.index(t)
        out.append({"line_id": lid, "start": s, "end": s + len(t), "text": t, "kind": str(it.get("kind", "")),
                    "person_id": it.get("person_id"), "detector": "hosted_vision"})
    return out


class ImageEditor:
    """Hosted image edit: sends ONLY the crop and its mask plus a short instruction (no page, no OCR text, no case
    data beyond the replacement string); composites the result strictly inside the mask; then an OCR check must read
    the intended replacement, else None is returned and the local engine result is kept."""

    def __init__(self, client: HostedClient, model=None, ocr_check=None):
        if client.cfg.provider in ("kimi", "anthropic"):
            raise HostedDisabled(f"{client.cfg.provider} has no image-edit endpoint: image edit is N/A")
        self.c = client; self.model = model or P.DEFAULT_MODELS[client.cfg.provider]["edit"]
        self.ocr_check = ocr_check

    def estimate(self, crop):
        side = max(crop.shape[:2])
        if self.c.cfg.provider == "fal":
            from . import faledit
            dims = faledit.billed_dims(self.model, *crop.shape[:2]) if self.model in faledit.SPECS else crop.shape[1::-1]
            return P.worst_case_usd("fal", self.model, 120, [dims], 0, output_images=1)
        n_in = 2 if self.c.cfg.provider == "openai" else 2
        return P.worst_case_usd(self.c.cfg.provider, self.model, 120, [crop.shape[1::-1]] * n_in, 0 if self.c.cfg.provider == "openai" else 256,
                                output_images=1, out_side=side)

    def edit(self, crop_rgb, mask, instruction, expect_text, case_id):
        cfg = self.c.cfg
        if 2 > cfg.max_images_per_call:
            raise HostedDisabled("image edit needs 2 images (crop + mask) but CASEPATCH_MAX_IMAGES_PER_CALL < 2")
        crop, s = _fit(crop_rgb, cfg.max_image_side)
        m, _ = _fit((mask > 0).astype(np.uint8) * 255, cfg.max_image_side)
        rgba_mask = np.dstack([np.zeros(m.shape + (3,), np.uint8), 255 - m])  # OpenAI: transparent = edit here
        self.last_raw = None
        tf = {}

        def build():
            if cfg.provider == "openai":
                from PIL import Image
                def png(a):
                    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG"); return b.getvalue()
                body, ctype = _multipart({"model": self.model, "prompt": instruction, "n": "1"},
                                         {"image": ("crop.png", png(crop), "image/png"), "mask": ("mask.png", png(rgba_mask), "image/png")})
                return ENDPOINTS[("openai", "edit")], body, ctype
            if cfg.provider == "fal":
                from . import faledit
                if self.model in faledit.SPECS:  # padded / scaled canvas, mapped back after the call
                    url, body, ctype, t = faledit.build(self.model, crop, m, instruction)
                    tf.update(t); return url, body, ctype
                return build_fal_edit(self.model, crop, m, instruction)
            body = {"contents": [{"role": "user", "parts": [
                {"text": instruction + " Edit only inside the white area of the second image; keep everything else identical."},
                {"inline_data": {"mime_type": "image/png", "data": _png_b64(crop)}},
                {"inline_data": {"mime_type": "image/png", "data": _png_b64(np.dstack([m] * 3))}}]}],
                "generationConfig": {"responseModalities": ["IMAGE"], "maxOutputTokens": 256}}
            return ENDPOINTS[("gemini", "edit")].format(model=self.model), json.dumps(body).encode(), "application/json"
        data = self.c.call("edit", self.model, build, self.estimate(crop), case_id)
        img = decode_image(data, cfg.provider)
        if img is None:
            return None
        if tf:
            from . import faledit
            img = faledit.unprep(img, tf)
        self.last_raw = img  # model image aligned with the (fitted) crop, before compositing
        return composite_inside_mask(crop_rgb, mask, img, self.ocr_check, expect_text)


def build_fal_edit(model, crop, m, instruction):
    """fal edit request. Images go inline as PNG data URIs (no upload); sync_mode returns the result as a data URI so
    nothing is stored on fal's CDN; fixed seed; PNG out; size = input size. Only the mask-native inpaint endpoint gets
    the mask; for the others the instruction names the exact string and the result is composited inside our mask."""
    uri = lambda a: "data:image/png;base64," + _png_b64(a)
    if model not in FAL_MASK_ENDPOINTS:
        instruction += " No mask image is given: change only that text and keep every other pixel identical."
    body = {"prompt": instruction, "sync_mode": True, "num_images": 1, "output_format": "png", "seed": FAL_EDIT_SEED}
    if model in FAL_MASK_ENDPOINTS:
        body |= {"image_url": uri(crop), "mask_url": uri(np.dstack([m] * 3)),
                 "image_size": {"width": int(crop.shape[1]), "height": int(crop.shape[0])}}
    elif model in FAL_SINGLE_IMAGE:
        body["image_url"] = uri(crop)
        if model == "fal-ai/flux-kontext/dev":
            body["resolution_mode"] = "match_input"
    else:
        body["image_urls"] = [uri(crop)]
        if model != "alibaba/qwen-image-3/edit":
            body["image_size"] = {"width": int(crop.shape[1]), "height": int(crop.shape[0])}
        else:
            body["enable_prompt_expansion"] = False
    return ENDPOINTS[("fal", "edit")].format(model=model), json.dumps(body).encode(), "application/json"


def fal_image_bytes(data):
    """First output image of a fal response as bytes (sync_mode data URI). A plain https URL is NOT fetched (the
    transport only POSTs to the model endpoint) -> None."""
    url = data["images"][0]["url"]
    if not url.startswith("data:"):
        return None
    return base64.b64decode(url.split(",", 1)[1])


def decode_image(data, provider):
    from PIL import Image
    try:
        if provider == "fal":
            b = fal_image_bytes(data)
            if b is None:
                return None
        elif provider == "openai":
            b = base64.b64decode(data["data"][0]["b64_json"])
        else:
            part = next(p for p in data["candidates"][0]["content"]["parts"] if "inlineData" in p or "inline_data" in p)
            b = base64.b64decode((part.get("inlineData") or part.get("inline_data"))["data"])
        im = Image.open(io.BytesIO(b))
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            return np.asarray(im.convert("RGBA"))  # gpt-image edits can return (partly) transparent pixels
        return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def flatten_over(edited, crop_rgb):
    """Model image -> RGB at the crop's size. Transparent / semi-transparent pixels (OpenAI image edits often return
    RGBA with alpha < 255, even inside the mask) are alpha-blended over the ORIGINAL crop instead of being dropped to
    whatever RGB sits under alpha 0 (black), which used to paste black or stale pixels into the mask."""
    import cv2
    h, w = crop_rgb.shape[:2]
    ed = cv2.resize(edited, (w, h), interpolation=cv2.INTER_AREA) if edited.shape[:2] != (h, w) else edited
    if ed.ndim == 3 and ed.shape[2] == 4:
        a = ed[..., 3:4].astype(np.float32) / 255.0
        ed = np.clip(ed[..., :3].astype(np.float32) * a + crop_rgb.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
    return ed


def composite_inside_mask(crop_rgb, mask, edited, ocr_check=None, expect_text=None):
    ed = flatten_over(edited, crop_rgb)
    sel = mask > 0
    out = crop_rgb.copy()
    out[sel] = ed[sel]
    if ocr_check is not None and expect_text is not None and not ocr_check(out, expect_text):
        return None
    return out
