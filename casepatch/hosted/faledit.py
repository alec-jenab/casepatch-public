"""fal.ai image-edit request specs for in-place document text edits (crop + mask contract, see EDIT_BENCHMARK.md).

Most current edit endpoints reject or distort tiny, very wide crops (min 256 px per side, min ~0.92 MP total, a fixed
set of aspect ratios). So the crop is
  1. padded with flat paper colour (median of the crop border; never page content) to the nearest aspect ratio the
     endpoint accepts (free-ratio endpoints: no padding), then
  2. scaled to ~1 MP (multiples of 16), which also makes small print legible to the model;
the model output is mapped back exactly (resize to the padded canvas, cut the padding, resize to the crop) and then
composited strictly inside OUR mask by client.composite_inside_mask. Only the crop (and, for mask-native endpoints,
its mask) is sent, inline as a data URI; sync_mode keeps the result out of fal's request history.
Endpoint ids and parameter names: https://fal.ai/models/<id>/llms.txt (read 2026-11-12).
"""
from __future__ import annotations

import base64
import io
import json
import math

import numpy as np

NANO_RATIOS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "4:1", "1:4", "8:1", "1:8"]
FLUX3_RATIOS = ["21:9", "2:1", "16:9", "3:2", "7:5", "4:3", "5:4", "1:1", "4:5", "3:4", "5:7", "2:3", "9:16", "1:2"]

# mask: None = instruction-only (we composite inside our mask); "white" = white repaints; "black" = black repaints.
# size: "aspect" -> aspect_ratio enum; "image_size" -> explicit {"width","height"}; None -> output follows the input.
SPECS = {
    "google/nano-banana-2.1/edit": dict(img="urls", mask=None, ratios=NANO_RATIOS, size="aspect",
                                        params={"resolution": "1K", "limit_generations": True, "seed": 7}),
    "fal-ai/nano-banana-pro/edit": dict(img="urls", mask=None, ratios=NANO_RATIOS, size="aspect",
                                        params={"resolution": "1K", "limit_generations": True, "seed": 7}),
    "bytedance/seedream/v5.5/edit": dict(img="urls", mask=None, ratios=None, size="image_size", min_area=921_600,
                                         params={"watermark": False, "background": "opaque"}),
    "blackforestlabs/flux-3/edit-image": dict(img="urls", mask=None, ratios=FLUX3_RATIOS, size="aspect",
                                              params={"resolution": "1k", "enable_prompt_expansion": False}),
    "blackforestlabs/flux-3/klein/edit-image": dict(img="urls", mask=None, ratios=None, size="image_size",
                                                    params={"enable_prompt_expansion": False, "seed": 7}),
    "alibaba/qwen-image-3/edit": dict(img="urls", mask=None, ratios=None, size="image_size",
                                      params={"enable_prompt_expansion": False, "seed": 7}),
    "fal-ai/qwen-image-edit-2511": dict(img="urls", mask=None, ratios=None, size="image_size", params={"seed": 7}),
    "fal-ai/qwen-image-edit/inpaint": dict(img="url", mask="white", ratios=None, size="image_size", params={"seed": 7}),
    "fal-ai/flux-pro/v1/fill": dict(img="url", mask="white", ratios=None, size=None, params={"seed": 7}),
    "ideogram/v4.5/edit": dict(img="url", mask="black", ratios=None, size=None, min_side=256,
                               params={"edit_precision": "high", "context_window": "auto", "quality": "medium", "seed": 7}),
}
TARGET_AREA = 1_000_000  # <= 1 MP (1024*1024) so per-megapixel endpoints bill one unit
MULT = 16


def _ratio(s):
    a, b = s.split(":"); return int(a) / int(b)


def canvas(model, h, w):
    """(pad_w, pad_h, ratio_name, out_w, out_h): padded crop size at crop scale, and the size sent to the model."""
    spec = SPECS[model]
    r = w / h; name = None; pw, ph = w, h
    if spec["ratios"]:
        name = min(spec["ratios"], key=lambda n: abs(math.log(_ratio(n) / r)))
        R = _ratio(name)
        if R >= r:
            pw = max(w, round(h * R))
        else:
            ph = max(h, round(w / R))
    area = max(TARGET_AREA, spec.get("min_area", 0))
    k = math.sqrt(area / (pw * ph))
    ow = max(MULT, int(pw * k) // MULT * MULT); oh = max(MULT, int(ph * k) // MULT * MULT)
    while ow * oh < spec.get("min_area", 0):
        ow += MULT; oh = max(MULT, round(ow * ph / pw / MULT) * MULT)
    ms = spec.get("min_side", 0)
    if min(ow, oh) < ms:
        s = ms / min(ow, oh); ow, oh = math.ceil(ow * s / MULT) * MULT, math.ceil(oh * s / MULT) * MULT
    return pw, ph, name, ow, oh


def paper_colour(crop):
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    return np.median(border, axis=0).astype(np.uint8)


def prep(model, crop, mask):
    """crop RGB uint8 (h,w,3), mask (h,w) >0 = edit -> (model image, model mask or None, transform)."""
    import cv2
    h, w = crop.shape[:2]
    pw, ph, name, ow, oh = canvas(model, h, w)
    x0, y0 = (pw - w) // 2, (ph - h) // 2
    can = np.empty((ph, pw, 3), np.uint8); can[:] = paper_colour(crop)
    can[y0:y0 + h, x0:x0 + w] = crop
    m = np.zeros((ph, pw), np.uint8); m[y0:y0 + h, x0:x0 + w] = (mask > 0) * 255
    img = cv2.resize(can, (ow, oh), interpolation=cv2.INTER_CUBIC)
    mm = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
    mm = cv2.dilate(mm, np.ones((3, 3), np.uint8))  # cover resampling at the border; composite uses OUR mask anyway
    tf = {"x0": x0, "y0": y0, "w": w, "h": h, "pw": pw, "ph": ph, "ow": ow, "oh": oh, "ratio": name}
    kind = SPECS[model]["mask"]
    mimg = None if kind is None else np.dstack([mm if kind == "white" else 255 - mm] * 3)
    return img, mimg, tf


def unprep(out, tf):
    """Model output (any size, RGB or RGBA) -> array at the crop size, aligned with the crop."""
    import cv2
    pw, ph = tf["pw"], tf["ph"]
    if out.shape[1] > 4 * pw:  # box-filter to 4x first, then cubic: plain INTER_AREA over ~8x softens thermal print
        out = cv2.resize(out, (4 * pw, 4 * ph), interpolation=cv2.INTER_AREA)
    a = cv2.resize(out, (pw, ph), interpolation=cv2.INTER_CUBIC if out.shape[1] > pw else cv2.INTER_AREA)
    return np.ascontiguousarray(a[tf["y0"]:tf["y0"] + tf["h"], tf["x0"]:tf["x0"] + tf["w"]])


def instruction_for(model, instruction):
    """The shared edit instruction, reworded for endpoints that get no mask image."""
    if SPECS[model]["mask"] is not None:
        return instruction
    return (instruction.replace("Inside the masked area only, replace", "Replace")
            .replace("Do not change anything outside the mask.",
                     "Change nothing else: every other character, line, mark and pixel must stay exactly as it is; "
                     "the plain margins are padding and stay blank. Do not crop, zoom, rotate or re-frame."))


def _uri(a):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def build(model, crop, mask, instruction):
    """-> (url, body bytes, content type, transform)."""
    spec = SPECS[model]
    img, mimg, tf = prep(model, crop, mask)
    body = {"prompt": instruction_for(model, instruction), "sync_mode": True, "num_images": 1, "output_format": "png"}
    body.update(spec["params"])
    if spec["img"] == "urls":
        body["image_urls"] = [_uri(img)]
    else:
        body["image_url"] = _uri(img)
    if mimg is not None:
        body["mask_url"] = _uri(mimg)
    if spec["size"] == "aspect":
        body["aspect_ratio"] = tf["ratio"]
    elif spec["size"] == "image_size":
        body["image_size"] = {"width": tf["ow"], "height": tf["oh"]}
    return f"https://fal.run/{model}", json.dumps(body).encode(), "application/json", tf


def billed_dims(model, h, w):
    """Output size used for the cost estimate (aspect endpoints at '1K' land near 1 MP)."""
    _, _, _, ow, oh = canvas(model, h, w)
    return ow, oh
