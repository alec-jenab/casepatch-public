"""Pricing table used for worst-case reservations (USD per 1M tokens).

Sources (all re-read 2026-11-11, America/Los_Angeles):
  OpenAI   https://platform.openai.com/docs/pricing  (Standard tier; image models: "Image" / "Text" modality rows)
           image-input tokens: https://platform.openai.com/docs/guides/images-vision (32px patches x model multiplier)
  Google   https://ai.google.dev/gemini-api/docs/pricing  (Paid tier; where a price rises on a known date the HIGHER
           price is used so reservations stay conservative)
  Moonshot https://platform.kimi.ai/docs/pricing/chat  (international USD billing via api.moonshot.ai; cache-miss input
           price used; Kimi bills image tokens as input tokens, count is dynamic, so we use a conservative patch bound)
  Anthropic https://platform.claude.com/docs/en/about-claude/pricing (base, global routing; cache writes billed at the
           5-minute write price, cache reads conservatively at the full input price) and
           https://platform.claude.com/docs/en/build-with-claude/vision (28x28 px patches; high-res tier long edge 2576 px,
           <= 4784 visual tokens per image). Haiku 5.5 doubles price above 100k prompt tokens (higher price applied).
  fal      https://fal.ai/models/<endpoint_id>/llms.txt (each model page's "Pricing" line, read 2026-11-12) and
           https://fal.ai/docs/documentation/model-apis/pricing (billed per successful output; per-megapixel prices
           round each output image UP to whole megapixels). Endpoint ids checked live in the public catalog
           GET https://api.fal.ai/v1/models (status 'active', 2026-11-12).
A model missing from this table is refused (fail closed). Update PRICING_DATE when the table is refreshed.
"""
from __future__ import annotations

import math

PRICING_DATE = "2026-11-12"
SOURCES = {"openai": "https://platform.openai.com/docs/pricing", "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
           "kimi": "https://platform.kimi.ai/docs/pricing/chat",
           "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
           "fal": "https://fal.ai/models/{endpoint_id}/llms.txt"}

# fal.ai bills per OUTPUT unit, not per token: unit = megapixel (rounded up per image), image, or 1k tokens (input +
# output). Values from each model page (2026-11-12). weights: licence of the open weights the endpoint serves.
FAL_PRICES = {
    # image edit
    "fal-ai/flux-kontext/dev": {"unit": "megapixel", "usd": 0.025, "weights": "FLUX.1 [dev] Non-Commercial License"},
    "fal-ai/flux-pro/kontext": {"unit": "image", "usd": 0.04, "weights": "closed (API only)"},
    "fal-ai/qwen-image-edit-2511": {"unit": "megapixel", "usd": 0.03, "weights": "Apache-2.0"},
    "fal-ai/qwen-image-edit/inpaint": {"unit": "megapixel", "usd": 0.03, "weights": "Apache-2.0"},
    "fal-ai/flux-2/klein/4b/edit": {"unit": "megapixel", "usd": 0.01, "weights": "Apache-2.0"},
    "alibaba/qwen-image-3/edit": {"unit": "image", "usd": 0.075, "weights": "closed (API only)",  # $0.04 at 1K, $0.075 at 2K: worst used
                                  "tiers": [(1_200_000, 0.04), (10**12, 0.075)], "note": "$0.04/image at 1K, $0.075 at 2K"},
    # image edit, added 2026-11-12 (funded run): ids from GET https://api.fal.ai/v1/models (status active) and prices
    # from each https://fal.ai/models/<id>/llms.txt "Pricing" line + https://fal.ai/pricing.md, read 2026-11-12.
    # "tiers": [(max output pixels, usd per image), ...] -> priced from the returned image size.
    "google/nano-banana-2.1/edit": {"unit": "image", "usd": 0.10, "weights": "closed (API only)", "estimate_only": True,
                                    "note": "token-billed: $1.764/M input, $8.823/M text out, $35.294/M image out; response "
                                            "carries no usage, so a conservative flat $0.10 per 1K image is reserved and kept"},
    "fal-ai/nano-banana-pro/edit": {"unit": "image", "usd": 0.15, "weights": "closed (API only)", "note": "4K = 2x"},
    "bytedance/seedream/v5.5/edit": {"unit": "image", "usd": 0.0525, "weights": "closed (API only)",
                                     "tiers": [(2_610_000, 0.042), (10**12, 0.0525)],
                                     "note": "$0.03 per output <= 2.61 MP ($0.0405 above) + $0.012 per input image"},
    "blackforestlabs/flux-3/edit-image": {"unit": "megapixel", "usd": 0.048, "weights": "closed (API only)"},
    "blackforestlabs/flux-3/klein/edit-image": {"unit": "image", "usd": 0.036, "weights": "FLUX.3 Klein (see BFL licence)",
                                                "tiers": [(600_000, 0.008), (1_100_000, 0.009), (2_400_000, 0.011),
                                                          (4_300_000, 0.014), (10**12, 0.036)],
                                                "note": "$0.008 768sq, $0.009 1K, $0.011 1.5K, $0.014 2K, $0.036 4K"},
    "fal-ai/flux-pro/v1/fill": {"unit": "megapixel", "usd": 0.05, "weights": "closed (API only)"},
    "ideogram/v4.5/edit": {"unit": "image", "usd": 0.06, "weights": "closed (API only)",
                           "note": "quality medium ($0.008 very low, $0.03 low, $0.06 medium, $0.22 high)"},
    # vision (detect task)
    "nvidia/nemotron-3-5-super/vision": {"unit": "ktokens", "usd": 0.01, "min_usd": 0.01, "weights": "NVIDIA Open Model License"},
    "nvidia/nemotron-3-nano-omni/vision": {"unit": "ktokens", "usd": 0.006, "weights": "NVIDIA Open Model License"},
}
FAL_IMAGE_TOKENS_MAX = 3328  # Nemotron VL tiling bound used for fal token estimates: 12 x 256-token tiles + thumbnail

# per 1M tokens
PRICES = {
    # OpenAI vision / grounded-JSON detector (Responses API)
    ("openai", "gpt-5.4"): {"text_in": 2.50, "image_in": 2.50, "out": 15.00},
    ("openai", "gpt-5.5"): {"text_in": 5.00, "image_in": 5.00, "out": 30.00},
    ("openai", "gpt-5.4-mini"): {"text_in": 0.75, "image_in": 0.75, "out": 4.50},
    ("openai", "gpt-6-astra"): {"text_in": 10.00, "image_in": 10.00, "out": 50.00},
    ("openai", "gpt-6.1-sol"): {"text_in": 2.00, "image_in": 2.00, "out": 10.00},
    ("openai", "gpt-6.1-terra"): {"text_in": 1.00, "image_in": 1.00, "out": 5.00},
    ("openai", "gpt-6-luna"): {"text_in": 0.10, "image_in": 0.10, "out": 0.50},
    # OpenAI image edit (Images API /v1/images/edits)
    ("openai", "gpt-image-2"): {"text_in": 5.00, "image_in": 8.00, "out": 30.00, "image_out": 30.00},
    ("openai", "gpt-image-2.6-sunburst"): {"text_in": 5.00, "image_in": 8.00, "out": 30.00, "image_out": 30.00},
    ("openai", "gpt-image-2.6-flare"): {"text_in": 5.00, "image_in": 8.00, "out": 30.00, "image_out": 30.00},
    ("openai", "gpt-image-1-mini"): {"text_in": 2.00, "image_in": 2.50, "out": 8.00, "image_out": 8.00},
    # Gemini vision detector (generateContent); Carbon uses the price effective 2026-12-01 ($4 / $20)
    ("gemini", "gemini-4-carbon"): {"text_in": 4.00, "image_in": 4.00, "out": 20.00},
    ("gemini", "gemini-4-argon"): {"text_in": 4.00, "image_in": 4.00, "out": 20.00},
    ("gemini", "gemini-3.8-flash"): {"text_in": 1.50, "image_in": 1.50, "out": 7.50},
    ("gemini", "gemini-3.5-flash-lite"): {"text_in": 0.30, "image_in": 0.30, "out": 2.50},
    # Gemini image edit model (generateContent with image output)
    ("gemini", "gemini-3.1-flash-image"): {"text_in": 0.50, "image_in": 0.50, "out": 3.00, "image_out": 60.00},
    ("gemini", "gemini-3-pro-image"): {"text_in": 2.00, "image_in": 2.00, "out": 12.00, "image_out": 120.00},  # Nano Banana Pro
    ("gemini", "gemini-nano-banana-2.1"): {"text_in": 1.50, "image_in": 1.50, "out": 7.50, "image_out": 30.00},
    # Moonshot Kimi (OpenAI-compatible chat completions; vision input; thinking always on, billed as output; no image edit)
    ("kimi", "kimi-k3.1-pro"): {"text_in": 2.00, "image_in": 2.00, "out": 10.00},
    ("kimi", "kimi-k3.1-pro-highspeed"): {"text_in": 4.00, "image_in": 4.00, "out": 20.00},
    ("kimi", "kimi-k3"): {"text_in": 3.00, "image_in": 3.00, "out": 15.00},
    # Anthropic Claude (Messages API; adaptive thinking billed as output; no image edit). cache_write = 5-min write price
    ("anthropic", "claude-fable-5-2"): {"text_in": 8.00, "image_in": 8.00, "out": 40.00, "cache_write": 10.00},
    ("anthropic", "claude-opus-5-5"): {"text_in": 4.00, "image_in": 4.00, "out": 20.00, "cache_write": 5.00},
    ("anthropic", "claude-sonnet-5-5"): {"text_in": 2.00, "image_in": 2.00, "out": 10.00, "cache_write": 2.50},
    ("anthropic", "claude-haiku-5-5"): {"text_in": 0.10, "image_in": 0.10, "out": 0.50, "cache_write": 0.125,
                                        "long_threshold": 100000,
                                        "long": {"text_in": 0.50, "image_in": 0.50, "out": 2.50, "cache_write": 0.625}},
}
PRICES.update({("fal", k): {"text_in": 0.0, "image_in": 0.0, "out": 0.0, "fal": v} for k, v in FAL_PRICES.items()})

# OpenAI patch-based image tokenization multipliers (images-vision guide); a model not listed uses the highest
# documented general-purpose multiplier we would plausibly hit (1.62) so the estimate stays conservative.
OPENAI_PATCH_MULTIPLIER = {"gpt-6-astra": 1.2, "gpt-5.6-sol": 1.2, "gpt-5.6-terra": 1.2, "gpt-5.6-luna": 1.2, "gpt-5.5": 1.2,
                           "gpt-5.4": 1.2, "gpt-5.4-mini": 1.2, "gpt-5.4-nano": 1.2, "gpt-5.2": 1.2}
OPENAI_UNLISTED_MULTIPLIER = 1.62

DEFAULT_MODELS = {"openai": {"vision": "gpt-5.4", "edit": "gpt-image-2"},
                  "gemini": {"vision": "gemini-4-carbon", "edit": "gemini-3.1-flash-image"},
                  "kimi": {"vision": "kimi-k3.1-pro", "edit": None},
                  "anthropic": {"vision": "claude-opus-5-5", "edit": None},
                  "fal": {"vision": "nvidia/nemotron-3-5-super/vision", "edit": "fal-ai/qwen-image-edit-2511"}}  # Claude has no image-edit endpoint  # Kimi has no image-edit endpoint

SAFETY = 1.5  # multiplier on token estimates we cannot know exactly before the call


class NoPricing(Exception):
    pass


def price(provider, model):
    p = PRICES.get((provider, model))
    if p is None:
        raise NoPricing(f"no pricing entry for {provider}/{model}: refusing (fail closed)")
    return p


def image_input_tokens(provider, w, h, model=None):
    """Conservative image-input token estimate (x SAFETY). Takes the max of the legacy tile rule and the patch rule
    for OpenAI so either tokenizer is covered."""
    if provider == "anthropic":  # 28x28 px patches after fitting the high-res tier (long edge 2576, <= 4784 tokens)
        s = min(1.0, 2576 / max(w, h)); w2, h2 = w * s, h * s
        t = math.ceil(w2 / 28) * math.ceil(h2 / 28)
        if t > 4784:
            s2 = math.sqrt(4784 * 28 * 28 / (w2 * h2)); t = math.ceil(w2 * s2 / 28) * math.ceil(h2 * s2 / 28)
        return int(min(max(t, 1), 4784 + 200) * SAFETY)
    if provider == "kimi":  # dynamic; bound by 28px patches (14px ViT patch, 2x2 merge) at full resolution
        t = max(1, math.ceil(w / 28)) * max(1, math.ceil(h / 28)) + 16
        return int(t * SAFETY)
    if provider == "openai" and model is not None and not str(model).startswith("gpt-image"):
        tile = image_input_tokens(provider, w, h)
        mult = OPENAI_PATCH_MULTIPLIER.get(model, OPENAI_UNLISTED_MULTIPLIER)
        patches = math.ceil(w / 32) * math.ceil(h / 32)
        if patches > 2500:  # 'high' detail budget used by the adapters (images are pre-fit to <= 2048 px)
            s = math.sqrt(32 * 32 * 2500 / (w * h)); patches = math.ceil(w * s / 32) * math.ceil(h * s / 32)
        return max(tile, int(math.ceil(patches * mult) * SAFETY))
    if provider == "openai":  # high detail: 85 base + 170 per 512px tile after fitting into 2048 / shortest side 768
        s = min(1.0, 2048 / max(w, h)); w2, h2 = w * s, h * s
        s2 = min(1.0, 768 / min(w2, h2)); w2, h2 = w2 * s2, h2 * s2
        t = 85 + 170 * math.ceil(w2 / 512) * math.ceil(h2 / 512)
    else:  # Gemini: 258 tokens per 768x768 tile (legacy); Gemini 3+ media_resolution up to 2240 tokens/image: take max
        t = max(2240, 258 * max(1, math.ceil(w / 768)) * max(1, math.ceil(h / 768)))
    return int(t * SAFETY)


def image_output_tokens(provider, side, model=None):
    """Output image tokens for a square-ish image of `side` px (OpenAI: conservative high-quality bound;
    Gemini: documented 1120 tokens at 1K, 1680 at 2K)."""
    if provider == "openai":
        return int((6240 if side > 1024 else 4160) * SAFETY)
    return 1120 if side <= 1024 else 1680


def _tiered(p, prompt_tokens):
    if "long_threshold" in p and prompt_tokens > p["long_threshold"]:
        return p["long"]
    return p


def _fal_megapixels(w, h):
    return max(1, math.ceil(w * h / (1024 * 1024) - 1e-9))


def _wh(i):
    """(width, height) of one fal output image: the response fields, else the sync_mode data URI's header."""
    if i.get("width") and i.get("height"):
        return int(i["width"]), int(i["height"])
    u = i.get("url") or ""
    if not u.startswith("data:"):
        raise KeyError("width")
    import base64, io
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).size


def _fal_tier(f, pixels):
    """Per-image price for a resolution-tiered fal endpoint (flat price when untiered)."""
    for mx, usd in f.get("tiers", ()):
        if pixels <= mx:
            return usd
    return f["usd"]


def fal_usd(model, text_tokens, images, out_tokens, output_images=0, out_dims=None, safety=SAFETY):
    """fal cost for one call. Image units: megapixels of each OUTPUT image (rounded up; out_dims or, by default, the
    first input image size, with `safety` applied to the pixel count because endpoints may snap/upscale), or a flat
    per-image price. Token units: (text + image input + output tokens) / 1000 x price, with the per-request minimum."""
    f = price("fal", model)["fal"]
    if f["unit"] in ("megapixel", "image"):
        n = max(1, output_images)
        w, h = out_dims or (images[0] if images else (1024, 1024))
        if f["unit"] == "image":
            return round(n * _fal_tier(f, w * safety * h), 6)
        return round(n * _fal_megapixels(w * safety, h) * f["usd"], 6)
    img = sum(max(min(math.ceil(w / 28) * math.ceil(h / 28), FAL_IMAGE_TOKENS_MAX), 256) for w, h in images)
    toks = (text_tokens + img) * safety + out_tokens
    return round(max(f.get("min_usd", 0.0), math.ceil(toks) / 1000 * f["usd"]), 6)


def worst_case_usd(provider, model, text_tokens, images, max_output_tokens, output_images=0, out_side=1024):
    p = price(provider, model)
    if provider == "fal":
        return fal_usd(model, text_tokens, images, max_output_tokens, output_images)
    img_in = sum(image_input_tokens(provider, w, h, model) for w, h in images)
    p = _tiered(p, int(text_tokens * SAFETY) + img_in)
    usd = (int(text_tokens * SAFETY) * p["text_in"] + img_in * p["image_in"] + max_output_tokens * p["out"]) / 1e6
    if output_images:
        usd += output_images * image_output_tokens(provider, out_side) * p.get("image_out", p["out"]) / 1e6
    return round(usd, 6)


def actual_usd(provider, model, usage):
    """Cost from the provider's usage fields; raises KeyError/TypeError when they are missing (caller keeps the
    full reservation)."""
    p = price(provider, model)
    if provider == "fal":  # `usage` is the whole fal response: images[] carry width/height; LLM endpoints carry usage
        f = p["fal"]
        if f.get("estimate_only"):  # no billable units in the response: the caller keeps the full reservation
            raise KeyError("estimate_only")
        if f["unit"] == "image":
            ims = usage["images"]
            if "tiers" in f:
                if not ims:
                    raise KeyError("images")
                return round(sum(_fal_tier(f, _wh(i)[0] * _wh(i)[1]) for i in ims), 6)
            return round(len(ims) * f["usd"], 6)
        if f["unit"] == "megapixel":
            ims = usage["images"]
            if not ims:
                raise KeyError("images")
            return round(sum(_fal_megapixels(*_wh(i)) for i in ims) * f["usd"], 6)
        u = usage["usage"]
        toks = int(u["total_tokens"]) if "total_tokens" in u else int(u.get("prompt_tokens", u.get("input_tokens"))) + int(u.get("completion_tokens", u.get("output_tokens")))
        return round(max(f.get("min_usd", 0.0), toks / 1000 * f["usd"]), 6)
    if provider == "openai":
        it = int(usage["input_tokens"]); ot = int(usage["output_tokens"])
        det = usage.get("input_tokens_details") or {}
        img_in = int(det.get("image_tokens", 0)); txt_in = it - img_in
        odet = usage.get("output_tokens_details") or {}
        img_out = int(odet.get("image_tokens", 0)) if "image_out" in p else 0
        if "image_out" in p and not odet:
            img_out, ot_text = ot, 0  # Images API reports image output tokens as output_tokens
        else:
            ot_text = ot - img_out
        return round((txt_in * p["text_in"] + img_in * p["image_in"] + ot_text * p["out"] + img_out * p.get("image_out", 0)) / 1e6, 6)
    if provider == "anthropic":  # output_tokens include thinking; cache reads counted at full input price (conservative)
        it = int(usage["input_tokens"]); ot = int(usage["output_tokens"])
        cw = int(usage.get("cache_creation_input_tokens") or 0); cr = int(usage.get("cache_read_input_tokens") or 0)
        q = _tiered(p, it + cw + cr)
        return math.ceil(((it + cr) * q["text_in"] + cw * q["cache_write"] + ot * q["out"]) - 1e-9) / 1e6  # round UP
    if provider == "kimi":  # OpenAI chat-completions usage; completion_tokens include reasoning tokens
        it = int(usage["prompt_tokens"]); ot = int(usage["completion_tokens"])
        return round((it * p["text_in"] + ot * p["out"]) / 1e6, 6)
    um = usage
    it = int(um["promptTokenCount"]); ot = int(um.get("candidatesTokenCount", 0)) + int(um.get("thoughtsTokenCount", 0))
    if "candidatesTokenCount" not in um:
        raise KeyError("candidatesTokenCount")
    img_out = sum(int(d.get("tokenCount", 0)) for d in um.get("candidatesTokensDetails", []) if d.get("modality") == "IMAGE")
    return round((it * p["text_in"] + (ot - img_out) * p["out"] + img_out * p.get("image_out", p["out"])) / 1e6, 6)
