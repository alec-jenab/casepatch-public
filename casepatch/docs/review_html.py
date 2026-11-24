"""Review artifact: one self-contained HTML page per case with page images, overlaid proposal boxes, and
accept/reject (text) or replace/keep/erase (regions) toggles. 'Save decisions' downloads decisions.json, which
`casepatch docs ... --decisions decisions.json` consumes. Local file only; contains ORIGINAL values."""
from __future__ import annotations

import base64
import html

HIDDEN = '(card security data: not stored)'
import io
import json

KIND_COLOR = {"PERSON": "#e6194b", "ADDRESS": "#f58231", "ZIP": "#f58231", "PHONE": "#911eb4", "EMAIL": "#911eb4", "CARD": "#4363d8",
              "CARD_LAST4": "#4363d8", "ACCOUNT": "#3cb44b", "ID": "#008080", "DOB": "#9a6324"}


def _png_b64(img, max_w=1000):
    from PIL import Image
    im = Image.fromarray(img)
    s = min(1.0, max_w / im.width)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)))
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode(), s


def build(docs, proposals, decisions, path):
    parts = ["<!doctype html><meta charset=utf-8><title>CasePatch review</title><style>",
             "body{font:14px system-ui;margin:20px;background:#f4f4f4}.pg{position:relative;display:inline-block;margin:8px 0;box-shadow:0 1px 4px #0004}",
             ".bx{position:absolute;border:2px solid;cursor:pointer;box-sizing:border-box}.bx.off{border-style:dashed;opacity:.45}",
             ".rg{background:#0003}.rv{outline:3px dotted #ff0}table{border-collapse:collapse;background:#fff}td,th{padding:3px 6px;border-bottom:1px solid #ddd;font-size:12px}",
             "tr.off{color:#999;text-decoration:line-through}.badge{font-size:11px;background:#fe6;padding:1px 4px;border-radius:3px}</style>",
             "<h1>CasePatch review (local only — contains original values)</h1>",
             "<p>Click a box or a row to toggle. Yellow dotted = needs review. Grey = non-text region (withheld unless you choose keep). ",
             "<button onclick=save()>Save decisions.json</button></p>"]
    cons = (proposals.get("consistency") or {}).get("violations") or []
    if cons:  # plausibility / relation problems in the planned replacements (consistency.py)
        from .consistency import explain
        occ_text = {o["id"]: (o["kind"], o["text"] or HIDDEN, o["replacement"] or HIDDEN) for o in proposals["occurrences"]}
        lines = {}  # one line per distinct problem; the same span repeated across pages/documents is counted, not relisted
        for v in cons:
            ex = "; ".join(dict.fromkeys(f"{k}: {html.escape(t)} &rarr; {html.escape(r)}"
                                         for k, t, r in (occ_text.get(i, ("?", "?", "?")) for i in v["occs"])))
            key = (v["code"], ex)
            lines[key] = lines.get(key, 0) + 1
        parts.append("<div style='background:#fff3cd;border:1px solid #e0b000;padding:8px;margin:8px 0'><b>Consistency check: "
                     f"{len(lines)} planned replacement problem(s)</b><ul>")
        for (code, ex), n in lines.items():
            ex = "; ".join(ex.split("; ")[:3])
            times = f" <small>(&times;{n})</small>" if n > 1 else ""
            parts.append(f"<li>{html.escape(explain(code))} <small>({html.escape(code)})</small> &mdash; {ex}{times}</li>")
        parts.append("</ul></div>")
    occ_by_doc = {}
    for o in proposals["occurrences"]:
        occ_by_doc.setdefault(o["doc"], []).append(o)
    reg_by_doc = {}
    for r in proposals["regions"]:
        reg_by_doc.setdefault(r["doc"], []).append(r)
    for d in docs:
        parts.append(f"<h2>{html.escape(d.id)} <small>({html.escape(d.kind)})</small></h2>")
        if d.kind == "json":
            parts.append("<p>JSON trace: occurrences listed below.</p>")
        for p in d.pages:
            if p.image is None:
                continue
            b64, s = _png_b64(p.image)
            k = s * (1 if p.raster else p.scale)
            parts.append(f"<div class=pg><img src='data:image/png;base64,{b64}'>")
            for o in occ_by_doc.get(d.id, []):
                if o["page"] != p.index or not o.get("bbox"):
                    continue
                x0, y0, x1, y1 = [v * k for v in o["bbox"]]
                c = KIND_COLOR.get(o["kind"], "#000")
                parts.append(f"<div class='bx{' rv' if o['review'] else ''}' data-id={o['id']} title='{html.escape(o['kind'] + ': ' + (o['text'] or HIDDEN) + ' -> ' + (o['replacement'] or HIDDEN))}' "
                             f"style='left:{x0:.0f}px;top:{y0:.0f}px;width:{x1 - x0:.0f}px;height:{y1 - y0:.0f}px;border-color:{c}' onclick=tog('{o['id']}')></div>")
            for r in reg_by_doc.get(d.id, []):
                if r["page"] != p.index:
                    continue
                x0, y0, x1, y1 = [v * k for v in r["bbox"]]
                parts.append(f"<div class='bx rg' data-id={r['id']} title='{html.escape(r['kind'] + ': ' + r['reason'])}' style='left:{x0:.0f}px;top:{y0:.0f}px;"
                             f"width:{x1 - x0:.0f}px;height:{y1 - y0:.0f}px;border-color:#555' onclick=tog('{r['id']}')></div>")
            parts.append("</div>")
        parts.append("<table><tr><th>id</th><th>kind</th><th>original</th><th>replacement</th><th>source</th><th>detectors</th><th>conf</th><th>review</th></tr>")
        for o in occ_by_doc.get(d.id, []):
            parts.append(f"<tr data-id={o['id']} onclick=tog('{o['id']}')><td>{o['id']}</td><td>{o['kind']}</td><td>{html.escape(o['text'] or HIDDEN)}</td>"
                         f"<td>{html.escape(o['replacement'] or HIDDEN)}</td><td>{o['source']}</td><td>{html.escape(', '.join(o['detectors']))}</td>"
                         f"<td>{o['confidence']:.2f}</td><td>{''.join(f'<span class=badge>{html.escape(x)}</span> ' for x in o['review'])}</td></tr>")
        for r in reg_by_doc.get(d.id, []):
            parts.append(f"<tr data-id={r['id']} onclick=tog('{r['id']}')><td>{r['id'].split(':')[-1]}</td><td>{r['kind']}</td><td colspan=2>{html.escape(r['reason'])}</td>"
                         f"<td>pixels</td><td>visual</td><td></td><td><span class=badge>NON_TEXT_REGION</span></td></tr>")
        parts.append("</table>")
    parts.append("<script>const D=" + json.dumps(decisions) + ";")
    parts.append("""function paint(){document.querySelectorAll('[data-id]').forEach(e=>{const v=D[e.dataset.id];e.classList.toggle('off',v==='reject'||v==='keep')})}
function tog(id){const v=D[id];D[id]=v==='accept'?'reject':v==='reject'?'accept':v==='replace'?'keep':v==='keep'?'erase':'replace';paint()}
function save(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(D,null,1)],{type:'application/json'}));a.download='decisions.json';a.click()}
paint()</script>""")
    open(path, "w").write("".join(parts))
