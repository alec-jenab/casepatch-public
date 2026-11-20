"""Runs one CasePatch pipeline pass in a child process and writes previews.

Usage: python -m casepatch_studio.worker JOB.json
JOB = {input, out, review, decisions|null, detector, verify, previews, run, meta}
Writes meta (document/page geometry, timings) atomically; exit code 0 = success.
A child process keeps model memory and the pipeline's module-level state out of the web server.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

MAX_PREVIEW = 1100      # PDF page width in preview pixels
MAX_IMAGE_PREVIEW = 2200  # photos/scans keep detail so the review viewer can zoom into a card
NAMES = Path(__file__).with_name("names.json")  # curated replacement-name pool (see curate_names.py)


def _pages(path: Path):
    from PIL import Image
    if path.suffix.lower() == ".pdf":
        import pymupdf
        out = []
        with pymupdf.open(path) as doc:
            for pg in doc:
                zoom = min(2.0, MAX_PREVIEW / max(1.0, pg.rect.width))
                pm = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                out.append(Image.frombytes("RGB", (pm.width, pm.height), pm.samples))
        return out
    im = Image.open(path)
    im.load()
    im = im.convert("RGB")
    im.thumbnail((MAX_IMAGE_PREVIEW, MAX_IMAGE_PREVIEW))
    return [im]


def _pts(page):
    for ln in page.lines:
        for ch in ln.chars or []:
            yield from ch.quad
    for r in page.regions:
        yield from r.quad


def focus_box(page, regions=()):
    """Where the document is on a photo (e.g. a card on a desk), in page coordinates, or None to show the full page.

    Text lines and detected regions say where the content is; a contour around them (the card edge) is preferred so the
    crop shows the whole document, else their padded union. Only raster pages, and only when it saves real space."""
    if not page.raster or page.image is None:
        return None
    pts = list(_pts(page)) + [q for r in regions for q in r]
    if len(pts) < 8:
        return None
    W, H = float(page.width), float(page.height)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ux0, uy0, ux1, uy1 = min(xs), min(ys), max(xs), max(ys)
    box = None
    try:
        import cv2
        import numpy as np
        img = page.image; s = float(page.scale or 1.0)
        g = cv2.GaussianBlur(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY), (5, 5), 0)
        e = cv2.dilate(cv2.Canny(g, 30, 90), np.ones((5, 5), np.uint8), iterations=2)
        cs, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            x0, y0, x1, y1 = x / s, y / s, (x + w) / s, (y + h) / s
            area = (x1 - x0) * (y1 - y0)
            if area < 0.08 * W * H or area > 0.9 * W * H:
                continue
            inside = sum(1 for px, py in zip(xs, ys) if x0 <= px <= x1 and y0 <= py <= y1) / len(xs)
            if inside >= 0.85 and (best is None or area < best[0]):
                best = (area, (x0, y0, x1, y1))
        if best:
            box = best[1]
    except Exception:
        box = None
    if box is None:
        pad = 0.08 * max(ux1 - ux0, uy1 - uy0)
        box = (ux0 - pad, uy0 - pad, ux1 + pad, uy1 + pad)
    m = 0.03 * max(box[2] - box[0], box[3] - box[1])
    x0, y0, x1, y1 = max(0, box[0] - m), max(0, box[1] - m), min(W, box[2] + m), min(H, box[3] + m)
    if (x1 - x0) * (y1 - y0) > 0.7 * W * H:
        return None
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def diff_overlay(before, after):
    """Transparent PNG marking every pixel that changed (dilated so thin strokes are visible): a truthful 'what changed'."""
    import numpy as np
    from PIL import Image, ImageFilter
    if before.size != after.size:
        after = after.resize(before.size)
    a, b = np.asarray(before, dtype=np.int16), np.asarray(after, dtype=np.int16)
    m = (np.abs(a - b).max(axis=2) > 24).astype(np.uint8) * 255
    mask = Image.fromarray(m, "L").filter(ImageFilter.MaxFilter(5))
    rgba = Image.new("RGBA", before.size, (255, 0, 170, 0))
    rgba.putalpha(mask.point(lambda v: 80 if v else 0))
    return rgba


def write_previews(docs, sources: dict, out_dir: Path, previews: Path, run: int):
    previews.mkdir(parents=True, exist_ok=True, mode=0o700)
    for d in docs:
        if d.kind == "json":  # traces: JSON as a value; JSONL / text as their text (the UI shows text, never HTML)
            raw_src = Path(sources[d.id]).read_text(encoding="utf-8-sig")
            raw_dst = (out_dir / f"{d.id}{d.ext}").read_text(encoding="utf-8")
            json_doc = (d.textual or {}).get("format", "json") == "json"
            src = json.loads(raw_src) if json_doc else raw_src
            dst = json.loads(raw_dst) if json_doc else raw_dst
            (previews / f"{d.id}-before.json").write_text(json.dumps(src, indent=2, ensure_ascii=False))
            (previews / f"{d.id}-after-{run}.json").write_text(json.dumps(dst, indent=2, ensure_ascii=False))
            continue
        before = previews / f"{d.id}-p0-before.png"
        if not before.exists():
            for i, im in enumerate(_pages(Path(sources[d.id]))):
                im.save(previews / f"{d.id}-p{i}-before.png", optimize=True)
        from PIL import Image
        for i, im in enumerate(_pages(out_dir / f"{d.id}{d.ext}")):
            im.save(previews / f"{d.id}-p{i}-after-{run}.png", optimize=True)
            b = previews / f"{d.id}-p{i}-before.png"
            if b.exists():
                diff_overlay(Image.open(b).convert("RGB"), im).save(previews / f"{d.id}-p{i}-diff-{run}.png", optimize=True)


def _region_quads(regions, doc_id, page_index):
    pre = f"{doc_id}:p{page_index}:"
    return [{"quad": r.quad} for r in regions or [] if str(getattr(r, "id", "")).startswith(pre)]


def use_curated_names():
    """Studio replaces names from a curated pool whose glyphs survive the photo restyle (OCR readback checked by
    curate_names.py). Same structure as the engine's pool: GIVEN[letter] alternates feminine/masculine-styled names."""
    if not NAMES.exists():
        return
    from casepatch.docs import resolve as R
    pool = json.loads(NAMES.read_text())
    given, surnames = pool.get("given") or {}, pool.get("surnames") or []
    if all(len(given.get(chr(c), [])) >= 2 for c in range(65, 91)) and len(surnames) >= 20:
        R.GIVEN = {k: list(v) for k, v in given.items()}
        R.SURNAMES = list(surnames)


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from casepatch.docs import pipeline
    use_curated_names()
    hosted = None
    if job.get("hosted"):  # the user opted in for this case after the notice (server checked availability)
        from casepatch.docs.hosted_detect import HostedPageDetector
        from casepatch.docs.byok import HANDOFF_ENV
        user_key = os.environ.pop(HANDOFF_ENV, None)  # the user's own key, handed over in memory by the local server
        hosted = HostedPageDetector.from_user(True, env={k: v for k, v in os.environ.items()}, user_key=user_key,
                                              case_id="studio-" + str(job.get("case_ref", "")))
    t0 = time.perf_counter()
    report, docs, occs, regions, decisions = pipeline.run(job["input"], job["out"], detector=job.get("detector", "rules"), hosted=hosted,
                                                          review_dir=job["review"], decisions_path=job.get("decisions"),
                                                          verify=job.get("verify", True),
                                                          category_policy=job.get("category_policy"))
    seconds = time.perf_counter() - t0
    sources = {d.id: d.path for d in docs}
    write_previews(docs, sources, Path(job["out"]), Path(job["previews"]), job["run"])
    meta = {"run": job["run"], "seconds": round(seconds, 1),
            "documents": [{"id": d.id, "kind": d.kind, "ext": d.ext,
                           "pages": [{"index": p.index, "width": p.width, "height": p.height, "raster": p.raster,
                                      "focus": focus_box(p, [r["quad"] for r in _region_quads(regions, d.id, p.index)])}
                                     for p in d.pages]}
                          for d in docs],
            "decisions": decisions}
    tmp = Path(job["meta"] + ".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, job["meta"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1]))
    except SystemExit:
        raise
    except BaseException as exc:  # message only; no document contents are printed by our code
        sys.stderr.write("WORKER_FAILED " + type(exc).__name__ + ": " + str(exc)[:300] + "\n")
        sys.stderr.write(traceback.format_exc()[-2000:])
        raise SystemExit(3)
