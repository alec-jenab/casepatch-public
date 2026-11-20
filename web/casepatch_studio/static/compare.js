/* Per-change editor comparison. Renders into a root node from a data model; makes no network calls of its own. */
"use strict";
window.CasePatchCompare = (() => {
  const el = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else if (v === true) e.setAttribute(k, "");
      else if (v !== false && v != null) e.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) e.append(k);
    return e;
  };
  const $ = (root, sel) => root.querySelector(sel);

  function readbackLabel(b) {
    if (!b) return { text: "—", cls: "bdg" };
    if (b.stale) return { text: "Stale: made for “" + (b.made_for || "an older value") + "”, needs regeneration", cls: "bdg bdg-warn" };
    if (b.leak) return { text: "Leak: original still readable", cls: "bdg bdg-leak" };
    if (b.readback === "exact") return { text: "OCR exact", cls: "bdg bdg-ok" };
    if (b.readback === "fuzzy") return { text: "OCR close (not exact)", cls: "bdg bdg-warn" };
    return { text: "OCR miss", cls: "bdg bdg-fail" };
  }
  function seamLabel(b) {
    if (!b || b.seam == null) return "Seam —";
    const s = Number(b.seam);
    const abs = Math.abs(s);
    if (abs < 2) return "Seam " + s.toFixed(2) + " (blends in)";
    if (abs < 8) return "Seam " + s.toFixed(2);
    return "Seam " + s.toFixed(1) + (s > 0 ? " (visible edge)" : "");
  }
  function outsideLabel(b) {
    if (!b || b.outside_mask == null) return "Outside mask —";
    const f = Number(b.outside_mask);
    if (f === 0) return "Outside mask 0% (pasted mask-only)";
    return "Outside mask " + (100 * f).toFixed(2) + "%";
  }

  function render(root, model, handlers) {
    handlers = handlers || {};
    const editors = model.editors || [];
    const changes = model.changes || [];
    const pages = model.pages || [];
    const picks = model.picks || {};
    const overrides = new Set(model.overrides || []);
    const currentId = model.current || (changes[0] && changes[0].id);
    const showBoxes = model.showBoxes !== false;
    const showFull = model.showFull !== false;
    const current = changes.find((c) => c.id === currentId) || changes[0];

    root.textContent = "";
    root.classList.add("cmp-root");

    const summary = el("div", { class: "cmp-summary" });
    const counts = {};
    for (const c of changes) counts[picks[c.id] || "local"] = (counts[picks[c.id] || "local"] || 0) + 1;
    const countTxt = editors.map((e) => (counts[e.id] ? `${counts[e.id]} ${e.label}` : null)).filter(Boolean).join(", ");
    summary.append(
      el("strong", { text: changes.length + " change" + (changes.length === 1 ? "" : "s") }),
      el("span", { text: " · " + (countTxt || "local editor (default)") + " selected" }),
      model.precomputed ? el("span", { class: "muted small", text: " · precomputed public samples, no live model calls" }) : "",
      model.live ? el("span", { class: "muted small", text: " · local editor is the default; hosted candidates appear after a pack match or a BYOK run" }) : ""
    );
    root.append(summary);

    const tools = el("div", { class: "cmp-tools" });
    const boxToggle = el("label", { class: "toggle" }, el("input", { type: "checkbox", ...(showBoxes ? { checked: true } : {}), onchange: (e) => handlers.onToggle && handlers.onToggle({ showBoxes: e.target.checked, showFull }) }), " Boxes around changes");
    const fullToggle = el("label", { class: "toggle" }, el("input", { type: "checkbox", ...(showFull ? { checked: true } : {}), onchange: (e) => handlers.onToggle && handlers.onToggle({ showBoxes, showFull: e.target.checked }) }), " Full-page before/after");
    tools.append(boxToggle, fullToggle,
      el("span", { class: "muted small cmp-keys", text: "Keys: 1–4 pick an editor · J / K previous / next change · Shift+1–4 use that editor for all" }));
    const allBtns = el("div", { class: "cmp-all" });
    allBtns.append(el("span", { class: "muted small", text: "Use for all:" }));
    for (const ed of editors) {
      allBtns.append(el("button", { class: "button tiny", type: "button", text: ed.label,
        title: "Select " + ed.label + " on every change that is not a leak",
        onclick: () => handlers.onSelectAll && handlers.onSelectAll(ed.id) }));
    }
    tools.append(allBtns);
    root.append(tools);

    if (showFull && pages.length) {
      const wrap = el("div", { class: "cmp-pages" });
      for (const page of pages) {
        const onPage = changes.filter((c) => c.page_id === page.id);
        if (!onPage.length) continue;
        wrap.append(pagePair(page, onPage, picks, editors, showBoxes, current && current.id, handlers));
      }
      root.append(wrap);
    }

    if (!current) {
      root.append(el("p", { class: "muted", text: "No comparison changes in this pack." }));
      return { root, current: null };
    }

    const nav = el("div", { class: "cmp-nav" });
    const idx = changes.indexOf(current);
    nav.append(
      el("button", { class: "button tiny", type: "button", text: "Previous", disabled: idx <= 0, onclick: () => handlers.onCurrent && handlers.onCurrent(changes[idx - 1].id) }),
      el("strong", { text: (idx + 1) + " / " + changes.length }),
      el("button", { class: "button tiny", type: "button", text: "Next", disabled: idx >= changes.length - 1, onclick: () => handlers.onCurrent && handlers.onCurrent(changes[idx + 1].id) }),
      el("span", { class: "cmp-intent" },
        el("span", { class: "muted", text: "Replace " }),
        el("code", { text: current.value || "" }),
        el("span", { class: "muted", text: " with " }),
        el("code", { class: "repl", text: current.replacement || "" }))
    );
    if (current.attribution) {
      nav.append(el("div", { class: "muted small", text: (current.attribution.page_label || "") + " · " + current.attribution.source + " · " + current.attribution.license }));
    }
    if (current.previous_replacement) {
      nav.append(el("div", { class: "small warn cmp-replan", text: "Replacement planned in context: the earlier value “" + current.previous_replacement + "” was rejected (" +
        (current.previous_issues || []).join(", ") + ")." + (Object.values(current.candidates || {}).some((c) => c && c.badges && c.badges.stale)
          ? " Hosted outputs made for it are stale." : " Every editor was re-run for the new value.") }));
    }
    if (current.context_issues && current.context_issues.length) {
      nav.append(el("div", { class: "small bdg bdg-fail", text: "Replacement makes no sense here: " + current.context_issues.join(", ") }));
    }
    root.append(nav);

    const row = el("div", { class: "cmp-cands" });
    row.append(candCard({
      id: "original", label: "Original", kind: "original",
      src: current.original_src, badges: null, crop: current.crop, mask: true, chosen: false, locked: true
    }, current, null, handlers));
    for (let i = 0; i < editors.length; i++) {
      const ed = editors[i];
      const cand = (current.candidates || {})[ed.id];
      if (!cand) continue;
      const chosen = (picks[current.id] || "local") === ed.id;
      const over = overrides.has(current.id + "\t" + ed.id) || overrides.has(current.id + ":" + ed.id);
      row.append(candCard({ ...ed, src: cand.src, badges: cand.badges, key: String(i + 1), chosen, override: over }, current, cand, handlers));
    }
    root.append(row);
    return { root, current, changes, editors };
  }

  function candCard(ed, change, cand, handlers) {
    const leak = !!(ed.badges && ed.badges.leak);
    const stale = !!(ed.badges && ed.badges.stale);
    const card = el("div", { class: "cmp-card" + (ed.chosen ? " chosen" : "") + (leak ? " leak" : "") + (stale ? " stale" : "") + (ed.kind === "original" ? " orig" : "") });
    const head = el("div", { class: "cmp-card-h" });
    if (ed.key) head.append(el("span", { class: "cmp-key", text: ed.key }));
    head.append(el("strong", { text: ed.label }));
    if (ed.chosen) head.append(el("span", { class: "pill", text: "Selected" }));
    if (ed.kind === "local" && ed.default !== false) head.append(el("span", { class: "muted small", text: "default" }));
    card.append(head);
    const fig = el("figure", { class: "cmp-fig" });
    if (ed.src) {
      const img = el("img", { src: ed.src, alt: ed.label + " crop", draggable: "false" });
      fig.append(img);
      // box on the original crop around the mask / change area: the crop IS the change, so a full-frame box
      if (ed.kind === "original") fig.append(el("div", { class: "cmp-crop-box", title: "Change area" }));
    } else {
      fig.append(el("div", { class: "muted", text: "No image" }));
    }
    card.append(fig);
    if (ed.badges) {
      const rb = readbackLabel(ed.badges);
      const badges = el("div", { class: "cmp-badges" },
        el("span", { class: rb.cls, text: rb.text }),
        el("span", { class: "bdg", text: seamLabel(ed.badges) }),
        el("span", { class: "bdg", text: outsideLabel(ed.badges) })
      );
      card.append(badges);
    } else {
      card.append(el("div", { class: "muted small", text: "Before the edit, with the change boxed." }));
    }
    if (ed.kind !== "original") {
      if (stale) {
        card.append(el("button", { class: "button tiny", type: "button", disabled: true, text: "Needs regeneration (no live call made)" }));
      } else if (leak) {
        const wrap = el("div", { class: "cmp-override" });
        const id = "ovr-" + change.id + "-" + ed.id.replace(/[^a-z0-9]+/gi, "-");
        const cb = el("input", { type: "checkbox", id, ...(ed.override ? { checked: true } : {}) });
        wrap.append(cb, el("label", { for: id, text: "Override: I can still read the original and I still want this output" }));
        card.append(wrap);
        card.append(el("button", { class: "button tiny", type: "button", text: ed.chosen ? "Selected (override)" : "Select with override",
          onclick: () => {
            if (!cb.checked) { cb.focus(); return; }
            handlers.onSelect && handlers.onSelect(change.id, ed.id, true);
          } }));
      } else {
        card.append(el("button", { class: "button tiny" + (ed.chosen ? " primary" : ""), type: "button",
          text: ed.chosen ? "Selected" : "Select",
          onclick: () => handlers.onSelect && handlers.onSelect(change.id, ed.id, false) }));
      }
    }
    return card;
  }

  function pagePair(page, onPage, picks, editors, showBoxes, currentId, handlers) {
    const fig = el("div", { class: "cmp-pagepair" });
    fig.append(el("h3", { text: page.label || page.id }));
    if (page.credit) fig.append(el("p", { class: "muted small", text: page.credit }));
    const pair = el("div", { class: "cmp-pair" + (showBoxes ? "" : " noboxes") });
    pair.append(pageFrame("Before", page.src, page, onPage, null, currentId, handlers, true));
    // After uses the same original plus chosen crops composited by the server when an after_src is provided
    const afterSrc = page.after_src || page.src;
    pair.append(pageFrame("After (your picks)", afterSrc, page, onPage, picks, currentId, handlers, false));
    fig.append(pair);
    return fig;
  }

  function pageFrame(caption, src, page, onPage, picks, currentId, handlers, isBefore) {
    const figure = el("figure");
    figure.append(el("figcaption", { text: caption }));
    const frame = el("div", { class: "cmp-frame" });
    if (src) frame.append(el("img", { src, alt: caption, width: String(page.w || ""), height: String(page.h || "") }));
    const ov = el("div", { class: "overlay" });
    const w = page.w || 1, h = page.h || 1;
    for (const c of onPage) {
      const box = c.crop;
      if (!box) continue;
      const editor = picks ? (picks[c.id] || "local") : null;
      const leak = editor && c.candidates && c.candidates[editor] && c.candidates[editor].badges && c.candidates[editor].badges.leak;
      const cls = "box" + (c.id === currentId ? " on" : "") + (leak ? " leak" : " repl");
      const title = (c.value || c.id) + (editor ? " → " + (c.replacement || "") + " (" + editor + ")" : "");
      const b = el("div", { class: cls, title, onclick: () => handlers.onCurrent && handlers.onCurrent(c.id) });
      b.style.left = (100 * box[0] / w) + "%";
      b.style.top = (100 * box[1] / h) + "%";
      b.style.width = (100 * (box[2] - box[0]) / w) + "%";
      b.style.height = (100 * (box[3] - box[1]) / h) + "%";
      ov.append(b);
    }
    frame.append(ov);
    figure.append(frame);
    return figure;
  }

  function keydown(ev, model, handlers) {
    if (!model || !model.changes || !model.changes.length) return false;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return false;
    const changes = model.changes;
    const idx = Math.max(0, changes.findIndex((c) => c.id === model.current));
    const editors = model.editors || [];
    const key = ev.key;
    if (key === "j" || key === "J" || key === "ArrowDown" || key === "ArrowRight") {
      if (idx < changes.length - 1) handlers.onCurrent && handlers.onCurrent(changes[idx + 1].id);
      return true;
    }
    if (key === "k" || key === "K" || key === "ArrowUp" || key === "ArrowLeft") {
      if (idx > 0) handlers.onCurrent && handlers.onCurrent(changes[idx - 1].id);
      return true;
    }
    if (/^[1-4]$/.test(key)) {
      const ed = editors[Number(key) - 1];
      if (!ed) return false;
      if (ev.shiftKey) handlers.onSelectAll && handlers.onSelectAll(ed.id);
      else if (model.current) handlers.onSelect && handlers.onSelect(model.current, ed.id, false);
      return true;
    }
    return false;
  }

  return { render, keydown, readbackLabel, seamLabel, outsideLabel };
})();
