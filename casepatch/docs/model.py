"""Unified representation: documents -> pages -> lines -> characters with geometry and provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Quad = tuple  # ((x0,y0),(x1,y1),(x2,y2),(x3,y3)) clockwise from top-left, page/image pixel or PDF point coords


@dataclass
class Char:
    c: str
    quad: Quad


@dataclass
class Line:
    """One run of text. `source` is provenance: native (PDF text layer), native_hidden (invisible/occluded
    text layer), ocr (raster page), ocr_pixel_only (text present only in pixels of a native page), json (a string
    leaf of a JSON / JSONL document; `path` locates it), text (a line of a plain-text document; path = (line,))."""
    id: str
    text: str
    chars: list  # list[Char]; empty for json
    source: str
    conf: float = 1.0
    style: dict = field(default_factory=dict)  # font, size, color, origin (native)
    path: tuple = ()  # json path


@dataclass
class Region:
    """A non-text visual region (signature, barcode, qr, face, photo, unclassified graphic)."""
    id: str
    kind: str
    quad: Quad
    reason: str
    payload: str | None = None  # decoded QR text (kept local, never written to output)
    rotated: bool = False  # quad is a true (possibly rotated) outline rather than an axis-aligned box


@dataclass
class Page:
    index: int
    width: float
    height: float
    raster: bool                    # True: pixels are the document (scan/photo/image)
    scale: float = 1.0              # pixels per coordinate unit of `image`
    image: Any = None               # numpy RGB array of the page render (scale applied)
    lines: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    notes: list = field(default_factory=list)


@dataclass
class Document:
    id: str                         # neutral id, e.g. document-01; never the source filename
    path: str
    kind: str                       # pdf | image | json (json = every textual format: .json, .jsonl, .txt)
    ext: str
    pages: list = field(default_factory=list)
    json: Any = None                # json: the value; jsonl: records (None = blank line); text: lines
    textual: dict = field(default_factory=dict)  # textual documents: {"format": json|jsonl|text, line endings}
    hidden: dict = field(default_factory=dict)  # metadata, annotations, embedded files, exif (local only)
    timings: dict = field(default_factory=dict)


def quad_from_rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def quad_bbox(q):
    xs = [p[0] for p in q]; ys = [p[1] for p in q]
    return (min(xs), min(ys), max(xs), max(ys))


def span_quad(chars, start, end):
    """Quad covering chars[start:end] following the line's slant/perspective."""
    a, b = chars[start].quad, chars[end - 1].quad
    return (a[0], b[1], b[2], a[3])


def rect_inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def rect_overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area
