"""Cost-per-page model for pricing hypotheses. Pure arithmetic: no network, no provider calls.

Inputs:
* Measured local runtimes: report.json stage timings from three full CasePatch runs on this box (8 vCPU x86_64,
  15.6 GB RAM, no GPU), 2026-11-11/12. Re-measure with `python -m casepatch.metering.cost_model REPORT.json ...`.
* Compute price: AWS c7i.2xlarge (8 vCPU, 16 GiB) Linux on-demand us-east-1 $0.357/h, observed 2026-11-09 on
  https://ec2calc.cloud/instances/c7i.2xlarge (third-party price mirror; verify on aws.amazon.com before relying on it).
* Hosted frontier worst case: casepatch/hosted/pricing.py (dated 2026-11-11 provider pricing pages, x1.5 safety on
  estimated input tokens, max output tokens always charged). These are reservation bounds, not measured bills.
Everything this module prints is an ESTIMATE / HYPOTHESIS.
"""
from __future__ import annotations

import json
import sys

from ..hosted.pricing import DEFAULT_MODELS, worst_case_usd

VCPU_HOUR_USD = 0.357 / 8  # c7i.2xlarge on-demand / 8 vCPU
BOX_VCPUS = 8

# Measured runs (stage seconds; pages incl. 1 JSON trace "page" per case; raster/image edits counted from edit logs)
MEASURED = [
    {"case": "case-real-01 (CFPB, CORD, FUNSD, MIDV-500 + trace)", "pages": 5, "seconds": 46.3, "raster_edits": 19, "verify_seconds": 24.0},
    {"case": "case-real-02 (Dallas Fed, SROIE, FUNSD, MIDV-500 + trace)", "pages": 6, "seconds": 67.5, "raster_edits": 25, "verify_seconds": 27.0},
    {"case": "case-dispute-01 (fictional fixture)", "pages": 6, "seconds": 67.6, "raster_edits": 22, "verify_seconds": 46.2},
]


def measured_summary():
    pages = sum(m["pages"] for m in MEASURED)
    secs = sum(m["seconds"] for m in MEASURED)
    edits = sum(m["raster_edits"] for m in MEASURED)
    return {"pages": pages, "seconds": round(secs, 1), "seconds_per_page": round(secs / pages, 1),
            "raster_edits_per_page": round(edits / pages, 2),
            "compute_usd_per_page_on_demand": round(secs / pages * BOX_VCPUS * VCPU_HOUR_USD / 3600, 5)}


def frontier_page_usd(provider, edits_per_page, page_px=(1024, 1325), ocr_text_tokens=1500, max_out=2048, crop_px=512):
    """Worst-case reservation for one page: one vision detection call + one image-edit call per raster edit."""
    m = DEFAULT_MODELS[provider]
    vision = worst_case_usd(provider, m["vision"], ocr_text_tokens, [page_px], max_out)
    if not m.get("edit"):  # provider without an image-edit model: detection-only
        return {"vision_call": vision, "edit_call": None, "detect_only": round(vision, 4), "detect_plus_edits": None}
    edit = worst_case_usd(provider, m["edit"], 200, [(crop_px, crop_px), (crop_px, crop_px)], 256, output_images=1, out_side=crop_px)
    return {"vision_call": vision, "edit_call": edit, "detect_only": round(vision, 4),
            "detect_plus_edits": round(vision + edits_per_page * edit, 4)}


def model(edits_per_page=None):
    s = measured_summary()
    e = s["raster_edits_per_page"] if edits_per_page is None else edits_per_page
    return {"measured_local": s, "frontier_worst_case_per_page": {p: frontier_page_usd(p, e) for p in DEFAULT_MODELS},
            "assumptions": {"vcpu_hour_usd": round(VCPU_HOUR_USD, 5), "status": "ESTIMATE - hypotheses for pricing only"}}


def margin(price, cost):
    return {"price": price, "cost": round(cost, 4), "gross_margin_pct": round(100 * (price - cost) / price, 1) if price else None}


def from_reports(paths):
    rows = []
    for p in paths:
        r = json.loads(open(p).read())
        secs = sum(v["seconds"] for v in r["stages"].values())
        pages = sum(d["pages"] for d in r["documents"].values())
        rows.append({"report": p, "pages": pages, "seconds": round(secs, 1), "seconds_per_page": round(secs / pages, 1)})
    return rows


# ---------------------------------------------------------------- pricing hypotheses
UTILIZATION = 0.25          # assumed average utilization of hosted CPU workers (idle capacity is paid for)
CARD_FEE_PCT = 0.029        # assumed card processing fee on revenue (plus a fixed fee per invoice, ignored per page)
CANDIDATES = [
    {"id": "A", "name": "Standard (hosted local engine)", "unit": "page", "price": 0.10, "frontier": None},
    {"id": "B", "name": "Frontier-assisted detection", "unit": "page", "price": 0.25, "frontier": "detect_only"},
    {"id": "C", "name": "Per case, frontier detection + generative edits, up to 15 pages", "unit": "case", "price": 12.00,
     "frontier": "detect_plus_edits", "pages": 15},
]


def pricing_candidates():
    m = model()
    cpu = m["measured_local"]["compute_usd_per_page_on_demand"] / UTILIZATION
    rows = []
    for c in CANDIDATES:
        pages = c.get("pages", 1)
        for prov in (["openai", "gemini"] if c["frontier"] else [None]):
            fr = m["frontier_worst_case_per_page"][prov][c["frontier"]] if prov else 0.0
            cost = pages * (cpu + fr) + CARD_FEE_PCT * c["price"]
            rows.append({"candidate": c["id"], "name": c["name"], "unit": c["unit"], "price_usd": c["price"],
                         "provider": prov or "none", "cost_usd": round(cost, 4),
                         "gross_margin_pct": round(100 * (c["price"] - cost) / c["price"], 1)})
    # typical measured case size (5-6 pages) for C
    typical = 6
    for prov in ("openai", "gemini"):
        fr = m["frontier_worst_case_per_page"][prov]["detect_plus_edits"]
        cost = typical * (cpu + fr) + CARD_FEE_PCT * 12.0
        rows.append({"candidate": "C@6pages", "name": "C on a typical 6-page case", "unit": "case", "price_usd": 12.0,
                     "provider": prov, "cost_usd": round(cost, 4), "gross_margin_pct": round(100 * (12.0 - cost) / 12.0, 1)})
    return {"cpu_usd_per_page_at_utilization": round(cpu, 5), "utilization": UTILIZATION, "card_fee_pct": CARD_FEE_PCT,
            "rows": rows, "status": "HYPOTHESES - worst-case frontier reservations, not measured bills"}


if __name__ == "__main__":
    if sys.argv[1:] == ["pricing"]:
        print(json.dumps(pricing_candidates(), indent=2))
    else:
        print(json.dumps(from_reports(sys.argv[1:]) if sys.argv[1:] else model(), indent=2))

