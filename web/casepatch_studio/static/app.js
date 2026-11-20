"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) e.append(k);
    return e;
  };
  let cfg = null, caseId = null, review = null, current = null, state = null, poll = null;
  let cats = {};  // {category group: "replace" | "keep"}; groups not listed are replaced (default)

  let cfgResolve; const cfgReady = new Promise((r) => { cfgResolve = r; });
  async function api(method, path, body, raw) {
    if (method !== "GET" && !cfg) await cfgReady;   // a write before the token arrived was rejected (403) silently
    const headers = { "X-CasePatch-Token": cfg ? cfg.csrf : "" };
    let data = body;
    if (body !== undefined && !raw) { headers["Content-Type"] = "application/json"; data = JSON.stringify(body); }
    const r = await fetch(path, { method, headers, body: data });
    const ct = r.headers.get("Content-Type") || "";
    if (!r.ok) {
      const j = ct.includes("json") ? await r.json() : {};
      const err = new Error(j.error || ("Request failed (" + r.status + ")"));
      err.code = j.code; err.estimate = j.estimate;
      throw err;
    }
    return ct.includes("json") ? r.json() : r.blob();
  }
  function showError(e) { const p = $("error"); p.textContent = e ? (e.message || String(e)) : ""; p.classList.toggle("hidden", !e); }
  function screen(name) {
    for (const s of ["add", "run", "review", "export", "compare"]) { const n = $("screen-" + s); if (n) n.classList.toggle("hidden", s !== name); }
    for (const li of $("steps").children) {
      const order = ["add", "run", "review", "export"];
      li.className = li.dataset.step === name ? "on" : (order.indexOf(li.dataset.step) < order.indexOf(name) ? "done" : "");
    }
    showError(null);
  }

  // ---------- step 1: add files ----------
  async function ensureCase(sample) {
    if (caseId && !sample) return;
    const c = await api("POST", "/api/cases", { sample: !!sample });
    caseId = c.id; renderFiles(c);
  }
  function renderFiles(c) {
    const ul = $("files"); ul.textContent = "";
    for (const f of c.files) ul.append(el("li", {}, el("span", { text: f.name }), el("span", { class: "muted", text: ` ${f.pages} page${f.pages === 1 ? "" : "s"} · ${(f.size / 1024).toFixed(0)} KB` })));
    $("file-list").classList.toggle("hidden", !c.files.length);
  }
  async function addFiles(list) {
    try {
      await ensureCase(false);
      let snap = null;
      for (const f of list) snap = await api("PUT", `/api/cases/${caseId}/files?name=${encodeURIComponent(f.name)}`, f, true);
      if (snap) renderFiles(snap);
    } catch (e) { showError(e); }
  }
  $("file-input").addEventListener("change", (ev) => { addFiles([...ev.target.files]); ev.target.value = ""; });
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles([...e.dataTransfer.files]); });
  $("btn-sample").addEventListener("click", async () => {
    try { if (caseId) await api("POST", `/api/cases/${caseId}/delete`, {}); caseId = null; await ensureCase(true); } catch (e) { showError(e); }
  });
  $("btn-reset").addEventListener("click", async () => { await resetCase(); });
  async function resetCase() {
    if (caseId) { try { await api("POST", `/api/cases/${caseId}/delete`, {}); } catch (_) { } }
    caseId = null; review = null; $("files").textContent = ""; $("file-list").classList.add("hidden");
    $("hosted-on").checked = false; $("hosted-ack").checked = false; renderHosted();
    cats = {}; renderCats("cat-rows", "cat-summary");
    $("exported").classList.add("hidden"); $("confirm").checked = false; screen("add");
  }
  // ---------- per-category keep / replace (default: replace) ----------
  function catChoice(g) { return cats[g] === "keep" ? "keep" : "replace"; }
  const GROUP_REGIMES = {
    diagnoses: ["HIPAA", "GDPR"], medications: ["HIPAA", "GDPR"], procedures: ["HIPAA", "GDPR"],
    lab_results: ["HIPAA", "GDPR"], facilities: ["HIPAA"], special: ["GDPR"], financial_profile: ["GLBA"],
  };
  const KIND_REGIMES = {
    DIAGNOSIS: ["HIPAA", "GDPR"], DIAGNOSIS_CODE: ["HIPAA", "GDPR"], MEDICATION: ["HIPAA", "GDPR"],
    PROCEDURE: ["HIPAA", "GDPR"], PROCEDURE_CODE: ["HIPAA", "GDPR"], LAB_RESULT: ["HIPAA", "GDPR"],
    FACILITY: ["HIPAA"], SPECIAL: ["GDPR"], INCOME: ["GLBA"], CREDIT_SCORE: ["GLBA"],
    CVV: ["PCI"], PIN: ["PCI"], TRACK_DATA: ["PCI"], CARD_EXPIRY: ["PCI"],
  };
  function regimeBadges(names) {
    const wrap = el("span", { class: "regimes" });
    for (const n of names || []) wrap.append(el("span", { class: "bdg bdg-" + n.toLowerCase(), text: n, title: n + " category (highlighted, not a compliance claim)" }));
    return wrap;
  }
  function renderCats(rowsId, summaryId, onChange) {
    const groups = (cfg && cfg.category_groups) || [], rows = $(rowsId);
    rows.textContent = "";
    for (const g of groups) {
      const sel = el("select", { "data-group": g.id, "aria-label": g.label },
        el("option", { value: "replace", text: "Replace" }), el("option", { value: "keep", text: "Keep (flag for review)" }));
      sel.value = catChoice(g.id);
      sel.addEventListener("change", () => {
        cats[g.id] = sel.value; renderCatSummary(summaryId); row.classList.toggle("kept", sel.value === "keep");
        if (onChange) onChange();
      });
      const label = el("span", {}, el("span", { text: g.label }), regimeBadges(GROUP_REGIMES[g.id]));
      const row = el("div", { class: "cat-row" + (sel.value === "keep" ? " kept" : "") }, label, sel);
      rows.append(row);
    }
    renderCatSummary(summaryId);
  }
  function renderCatSummary(id) {
    const groups = (cfg && cfg.category_groups) || [];
    const kept = groups.filter((g) => catChoice(g.id) === "keep");
    $(id).textContent = kept.length ? ` · keeping ${kept.map((g) => g.label.split(" (")[0].toLowerCase()).join(", ")}; everything else replaced`
      : " · all replaced (default)";
  }
  function catBody() { const o = {}; for (const g of (cfg && cfg.category_groups) || []) o[g.id] = catChoice(g.id); return o; }

  // ---------- optional hosted detection (off by default; explicit notice per case) ----------
  function hostedChosen() { return !!(cfg && cfg.hosted && cfg.hosted.available && $("hosted-on").checked); }
  function renderHosted() {
    const h = cfg && cfg.hosted, box = $("hosted-box");
    if (!h || !h.allowed) { box.classList.add("hidden"); setMode(false); return; }  // local-only app: no hosted UI at all
    box.classList.remove("hidden");
    $("hosted-model").textContent = "Claude Sonnet 5.5 by Anthropic, with your own key";
    $("hosted-on").disabled = !h.available;
    $("hosted-reason").textContent = h.available
      ? `Uses your Anthropic key (${h.key_source === "studio-settings" ? "entered here" : "from ANTHROPIC_API_KEY"}) under your agreement with Anthropic; requests go straight from this computer to Anthropic. Finds more names and numbers than local detection alone on our public test cases. Its suggestions are not applied until you accept each one. Per-case limit: $${(h.max_usd_per_case || 2).toFixed(2)}.`
      : `Not available: ${h.reason || "not configured"}`;
    $("hosted-remember-wrap").classList.toggle("hidden", !h.keychain_available);
    $("btn-hosted-forget").classList.toggle("hidden", h.key_source !== "studio-settings");
    $("hosted-notice-text").textContent = h.notice;
    const on = hostedChosen();
    $("hosted-notice").classList.toggle("hidden", !on);
    if (!on) $("hosted-ack").checked = false;
    $("btn-analyze").disabled = on && !$("hosted-ack").checked;
    setMode(on);
  }
  function setMode(cloud) {
    const m = $("mode"); m.classList.toggle("cloud", !!cloud);
    m.textContent = cloud ? "Hosted detection on" : "Local mode";
    m.title = cloud ? "Hosted detection on: page images go to Anthropic for this case." : "Local mode: the app and all processing run on this computer. No cloud calls.";
    m.title = cloud ? "This case sends page images and their text to Anthropic for detection. Everything else stays on this computer." : "Local mode: the app and all processing run on this computer. No cloud calls.";
  }
  async function sendHostedKey(body) {
    try { const r = await api("POST", "/api/hosted-key", body); cfg.hosted = r.hosted; }
    catch (e) { showError(e); }
    finally { $("hosted-key-input").value = ""; renderHosted(); }
  }
  $("btn-hosted-key").addEventListener("click", () => sendHostedKey({ key: $("hosted-key-input").value.trim(), remember: $("hosted-remember").checked }));
  $("btn-hosted-forget").addEventListener("click", () => sendHostedKey({ forget: true }));
  $("hosted-on").addEventListener("change", renderHosted);
  $("hosted-ack").addEventListener("change", renderHosted);
  $("btn-analyze").addEventListener("click", async () => {
    try {
      const body = hostedChosen() ? { hosted: true, hosted_notice: $("hosted-ack").checked ? cfg.hosted.notice_version : null } : {};
      body.categories = catBody();
      await api("POST", `/api/cases/${caseId}/analyze`, body);
      waitRun(body.hosted ? "Analyzing your case (local + hosted detection)…" : "Analyzing your case…");
    } catch (e) { showError(e); }
  });

  // ---------- step 2: wait ----------
  function waitRun(title) {
    screen("run"); $("run-title").textContent = title;
    clearInterval(poll);
    poll = setInterval(async () => {
      try {
        const s = await api("GET", `/api/cases/${caseId}`);
        $("run-elapsed").textContent = s.elapsed != null ? `${Math.round(s.elapsed)} s elapsed` : "";
        if (s.status === "ready" || s.status === "error") {
          clearInterval(poll);
          if (s.status === "error") { screen("add"); showError(new Error("Processing failed. No output was produced. Check that the files open normally, then try again.")); return; }
          await loadReview();
          if (s.error) showError(new Error("Re-render failed; showing the previous result. Your changes were not applied."));
        }
      } catch (e) { clearInterval(poll); showError(e); }
    }, 1000);
  }

  // ---------- step 3: review ----------
  async function loadReview() {
    review = await api("GET", `/api/cases/${caseId}/review`);
    const regions = {};
    for (const r of review.proposals.regions) regions[r.id] = review.decisions[r.id] || "keep";
    const sugg = new Set(review.proposals.model_suggestions || []);
    state = { regions, reject: new Set(review.rejected_text || []), add: (review.additions || []).slice(), dirty: false,
              sugg, accept: new Set(review.accept_text || []), reviewed: new Set(review.accept_text || []) };
    setMode(review.source_sent_to_cloud || (review.case && review.case.hosted));
    cats = Object.assign({}, (review.case && review.case.categories) || {});
    renderCats("cat-rows-review", "cat-summary-review", () => setDirty(true));
    const sug = review.suggested;
    const prefill = !!(sug && !state.add.length && sug.add.length);
    if (prefill) {
      state.add = sug.add.map((a) => ({ text: a.text, kind: a.kind }));
      for (const h of sug.regions || []) for (const r of review.proposals.regions) if (r.doc === h.doc && r.kind === h.kind) state.regions[r.id] = h.action;
    }
    const pages = [];
    for (const d of review.documents) {
      if (d.kind === "json") pages.push({ doc: d.id, kind: "json", label: `${d.id} · ${{ ".jsonl": "JSONL trace", ".txt": "text" }[d.ext] || "JSON trace"}`, key: `${d.id}:json` });
      else for (const p of d.pages) pages.push({ doc: d.id, kind: d.kind, page: p.index, w: p.width, h: p.height, focus: p.focus || null, label: `${d.id} · page ${p.index + 1}`, key: `${d.id}:${p.index}` });
    }
    review.pages = pages;
    review.viewedSet = new Set(review.viewed);
    screen("review");
    renderSummary(); renderPageList(); renderAdds(); renderCaseMap();
    showPage(current && pages.find((p) => p.key === current.key) || pages[0]);
    setDirty(prefill);
    loadLiveCompare();
    loadAgent();
    const hint = $("add-hint");
    hint.classList.toggle("hidden", !prefill);
    if (prefill) { hint.textContent = "Example case: " + sug.note + " We pre-filled the choices a reviewer would make; change any you disagree with, then click Apply."; $("add-box").open = true; }
  }
  function fileLabel(docId) {
    const i = parseInt(docId.split("-")[1], 10) - 1; const n = review.files[i];
    return n ? `${docId} (${n})` : docId;
  }
  function replaced(text) {
    if (state.reject.has(text)) return false;
    return state.sugg.has(text) ? state.accept.has(text) : true;
  }
  function pendingSuggestions() { return [...state.sugg].filter((t) => !state.reviewed.has(t)); }
  function renderSummary() {
    const occ = review.proposals.occurrences, values = new Set(occ.map((o) => o.text));
    const nrep = occ.filter((o) => replaced(o.text)).length;
    $("summary").textContent = "";
    $("summary").append(
      el("strong", { text: `${values.size} sensitive values` }), ` · ${nrep} of ${occ.length} occurrences replaced · `,
      el("strong", { text: `${review.proposals.regions.length} visual regions` }), ` · ${review.needs_review.length} items flagged for a human look · run ${review.case.run}`,
      ...(state.sugg.size ? [el("span", { class: "sugg-state", text: ` · ${state.sugg.size} model suggestions, ${pendingSuggestions().length} still to accept or decline` })] : []));
  }
  function renderPageList() {
    const ul = $("page-list"); ul.textContent = "";
    for (const p of review.pages) {
      const n = review.proposals.occurrences.filter((o) => o.doc === p.doc && (p.kind === "json" || o.page === p.page)).length;
      const li = el("li", { class: (current && current.key === p.key ? "on " : "") + (review.viewedSet.has(p.key) ? "seen" : ""), title: fileLabel(p.doc), onclick: () => showPage(p) },
        el("span", { text: p.label }), el("span", { class: "count", text: String(n) }));
      ul.append(li);
    }
    const seen = review.pages.filter((p) => review.viewedSet.has(p.key)).length;
    $("viewed").textContent = `Viewed ${seen} of ${review.pages.length} output pages. Export opens after you have viewed them all.`;
  }
  function boxEl(bb, p, cls, title, id) {
    const b = el("div", { class: "box " + cls, title, "data-id": id });
    b.style.left = (100 * bb[0] / p.w) + "%"; b.style.top = (100 * bb[1] / p.h) + "%";
    b.style.width = (100 * (bb[2] - bb[0]) / p.w) + "%"; b.style.height = (100 * (bb[3] - bb[1]) / p.h) + "%";
    return b;
  }
  // ---------- viewer: crop to the document, synchronized zoom/pan, magnifier, change highlighting ----------
  const V = { s: 1, x: 0, y: 0, p: null, flick: null, flickOrig: false };
  function frames() { return [$("before"), $("after")]; }
  function applyView() {
    if (!V.p) return;
    for (const f of frames()) {
      const st = f.querySelector(".stage"); if (!st) continue;
      st.style.width = (V.p.w * V.s) + "px";
      st.style.transform = `translate(${-V.x * V.s}px, ${-V.y * V.s}px)`;
    }
    $("zoom-level").textContent = Math.round(100 * V.s * V.p.w / (V.p.nat || V.p.w)) + "%";
  }
  function fitRect(r) {
    const f = $("after"), fw = f.clientWidth, fh = f.clientHeight;
    const rw = r[2] - r[0], rh = r[3] - r[1];
    V.s = Math.min(fw / rw, fh / rh);
    V.x = r[0] - (fw / V.s - rw) / 2; V.y = r[1] - (fh / V.s - rh) / 2;
    applyView();
  }
  function fit() {
    if (!V.p) return;
    const crop = $("crop-doc").checked && V.p.focus;
    const r = crop ? V.p.focus : [0, 0, V.p.w, V.p.h], rw = r[2] - r[0], rh = r[3] - r[1];
    // pick the layout that shows the document largest: side by side, or before above after (each frame as tall as
    // the document needs, both inside the window); a landscape photo of a card usually wins stacked
    const pair = $("pair"), W = pair.clientWidth || 800;
    const top = pair.getBoundingClientRect().top + window.scrollY;
    const room = Math.max(220, (window.innerHeight - Math.min(top, window.innerHeight * 0.45) - 60) / 2);
    const sbs = Math.min((W - 10) / 2 / rw, window.innerHeight * 0.72 / rh), st = Math.min(W / rw, room / rh);
    const stacked = st > 1.05 * sbs;
    pair.classList.toggle("stacked", stacked);
    const scale = stacked ? st : sbs;
    for (const f of frames()) f.style.height = Math.round(Math.max(220, rh * scale)) + "px";
    fitRect(r);
  }
  function zoomAt(factor, fx, fy) {  // fx, fy: point in frame pixels that stays put
    const px = V.x + fx / V.s, py = V.y + fy / V.s;
    V.s = Math.max(0.05, Math.min(V.s * factor, 40));
    V.x = px - fx / V.s; V.y = py - fy / V.s; applyView();
  }
  function framePoint(f, ev) { const b = f.getBoundingClientRect(); return [ev.clientX - b.left, ev.clientY - b.top]; }
  function lens(ev, f) {
    const on = $("magnifier").checked && V.p;
    for (const g of frames()) {
      const L = g.querySelector(".lens"); if (!L) continue;
      if (!on || !ev) { L.classList.add("hidden"); continue; }
      const [mx, my] = framePoint(f, ev), Z = 3, R = 90;
      const img = g.querySelector(".stage img.page:not(.hidden)") || g.querySelector(".stage img.page");
      L.classList.remove("hidden");
      L.style.left = (mx - R) + "px"; L.style.top = (my - R) + "px";
      L.style.backgroundImage = `url("${img.src}")`;
      const W = V.p.w * V.s * Z, H = V.p.h * V.s * Z;
      L.style.backgroundSize = `${W}px ${H}px`;
      const px = (V.x * V.s + mx) * Z, py = (V.y * V.s + my) * Z;
      L.style.backgroundPosition = `${-(px - R)}px ${-(py - R)}px`;
    }
  }
  function wireFrame(f) {
    let drag = null, moved = false;
    f.addEventListener("wheel", (ev) => { if (!V.p) return; ev.preventDefault(); const [x, y] = framePoint(f, ev); zoomAt(ev.deltaY < 0 ? 1.2 : 1 / 1.2, x, y); lens(ev, f); }, { passive: false });
    f.addEventListener("mousedown", (ev) => { if (!V.p) return; drag = [ev.clientX, ev.clientY, V.x, V.y]; moved = false; ev.preventDefault(); });
    window.addEventListener("mousemove", (ev) => {
      if (drag) {
        const dx = ev.clientX - drag[0], dy = ev.clientY - drag[1];
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        V.x = drag[2] - dx / V.s; V.y = drag[3] - dy / V.s; applyView();
      }
    });
    f.addEventListener("mousemove", (ev) => lens(ev, f));
    f.addEventListener("mouseleave", () => lens(null, f));
    window.addEventListener("mouseup", (ev) => {
      if (drag && !moved && f.contains(ev.target)) { const [x, y] = framePoint(f, ev); zoomAt(ev.shiftKey ? 0.5 : 2, x, y); }
      drag = null;
    });
  }
  frames().forEach(wireFrame);
  $("zoom-in").addEventListener("click", () => { const f = $("after"); zoomAt(1.5, f.clientWidth / 2, f.clientHeight / 2); });
  $("zoom-out").addEventListener("click", () => { const f = $("after"); zoomAt(1 / 1.5, f.clientWidth / 2, f.clientHeight / 2); });
  $("zoom-fit").addEventListener("click", fit);
  $("crop-doc").addEventListener("change", fit);
  $("view-mode").addEventListener("change", applyMode);
  function applyMode() {
    const mode = $("view-mode").value;
    $("pair").classList.toggle("changes", mode === "changes");
    if (V.flick) { clearInterval(V.flick); V.flick = null; }
    const a = $("after"), cap = $("after-cap");
    const imA = a.querySelector("img.page.out"), imB = a.querySelector("img.page.orig");
    const show = (orig) => { V.flickOrig = orig; if (imA) imA.classList.toggle("hidden", orig); if (imB) imB.classList.toggle("hidden", !orig); cap.textContent = orig ? "Flicker: showing BEFORE (your file)" : (mode === "flicker" ? "Flicker: showing AFTER (CasePatch output)" : "After (CasePatch output)"); };
    show(false);
    if (mode === "flicker" && imA && imB) V.flick = setInterval(() => show(!V.flickOrig), 900);
  }

  async function showPage(p) {
    current = p;
    $("page-title").textContent = `${fileLabel(p.doc)}${p.kind === "json" ? "" : " · page " + (p.page + 1)}`;
    const before = $("before"), after = $("after"); before.textContent = ""; after.textContent = "";
    const t = Date.now();
    $("view-tools").classList.toggle("hidden", p.kind === "json");
    $("pair").classList.toggle("zoomable", p.kind !== "json");
    if (p.kind === "json") {
      V.p = null; $("pair").classList.remove("stacked"); for (const f of frames()) f.style.height = "";
      const [a, b] = await Promise.all([api("GET", `/api/cases/${caseId}/preview/${p.doc}/json-before`), api("GET", `/api/cases/${caseId}/preview/${p.doc}/json-after`)]);
      const show = (v) => (typeof v === "string" ? v : JSON.stringify(v, null, 2));  // JSONL / TXT traces arrive as text
      before.append(el("pre", { class: "json", text: show(a) }));
      after.append(el("pre", { class: "json", text: show(b) }));
    } else {
      V.p = p;
      const base = `/api/cases/${caseId}/preview/${p.doc}/p${p.page}`;
      const imgB = el("img", { class: "page", src: `${base}-before?t=${t}`, alt: "original page", draggable: "false" });
      const imgA = el("img", { class: "page out", src: `${base}-after?t=${t}`, alt: "transformed page", draggable: "false" });
      const imgB2 = el("img", { class: "page orig hidden", src: `${base}-before?t=${t}`, alt: "original page (flicker)", draggable: "false" });
      const diff = el("img", { class: "diff", src: `${base}-diff?t=${t}`, alt: "", draggable: "false" });
      const ov = el("div", { class: "overlay" }), ov2 = el("div", { class: "overlay" });
      for (const o of review.proposals.occurrences.filter((o) => o.doc === p.doc && o.page === p.page && o.bbox)) {
        const cls = state.sugg.has(o.text) && !state.accept.has(o.text) ? "sugg" : (!replaced(o.text) ? "keep" : (o.review && o.review.length ? "flag" : "repl"));
        ov.append(boxEl(o.bbox, p, cls, `${o.kind}: ${o.text ?? "(not stored)"}`, o.id)); ov2.append(boxEl(o.bbox, p, cls, `${o.kind} → ${o.replacement ?? "random digits"}`, o.id));
      }
      for (const f of (review.category_flags || []).filter((f) => f.doc === p.doc && f.page === p.page && f.bbox)) {
        ov.append(boxEl(f.bbox, p, "catflag", `${f.kind} (kept, flagged): ${f.text ?? ""}`, "")); ov2.append(boxEl(f.bbox, p, "catflag", `${f.kind}: kept unchanged`, ""));
      }
      for (const r of review.proposals.regions.filter((r) => r.doc === p.doc && r.page === p.page)) {
        const cls = "region " + state.regions[r.id];
        ov.append(boxEl(r.bbox, p, cls, `${r.kind}: ${state.regions[r.id]}`, r.id)); ov2.append(boxEl(r.bbox, p, cls, `${r.kind}: ${state.regions[r.id]}`, r.id));
      }
      before.append(el("div", { class: "stage" }, imgB, ov), el("div", { class: "lens hidden" }));
      after.append(el("div", { class: "stage" }, imgA, imgB2, diff, ov2), el("div", { class: "lens hidden" }));
      $("crop-doc").disabled = !p.focus;
      $("crop-doc").parentElement.title = p.focus ? "Zoom to the document found on this photo" : "This page is already the whole document";
      const onReady = () => { review.viewedSet.add(p.key); renderPageList(); p.nat = imgA.naturalWidth; fit(); };
      if (imgA.complete) requestAnimationFrame(onReady);
      else imgA.addEventListener("load", onReady);
      requestAnimationFrame(() => requestAnimationFrame(fit));
    }
    if (p.kind === "json") { review.viewedSet.add(p.key); }
    applyBoxToggle(); applyMode(); renderItems(); renderPageList();
    syncLiveComparePage();
  }
  window.addEventListener("resize", () => { if (V.p) fit(); });
  function applyBoxToggle() { $("pair").classList.toggle("noboxes", !$("show-boxes").checked); }
  $("show-boxes").addEventListener("change", applyBoxToggle);

  // ---------- case summary: one identity map across the documents ----------
  function shortName(docId) {
    const i = parseInt(docId.split("-")[1], 10) - 1, n = review.files[i] || docId;
    return n.replace(/^\d+_/, "").replace(/\.[a-z0-9]+$/i, "").replace(/_/g, " ");
  }
  const KIND_LABEL = { GIVEN: "given name", SURNAME: "surname", PERSON: "full name" };
  function renderCaseMap() {
    const body = $("case-map-body"); body.textContent = "";
    const occ = review.proposals.occurrences, groups = new Map();
    for (const o of occ) {
      const k = o.entity || `${o.kind}:${o.text}`;
      if (!groups.has(k)) groups.set(k, { key: k, kind: o.kind, texts: new Map(), docs: new Map(), parts: [] });
      const g = groups.get(k);
      g.texts.set(o.text ?? "(card security data, not stored)", o.replacement ?? "random digits");
      const d = g.docs.get(o.doc) || { n: 0, kinds: new Set() }; d.n++; d.kinds.add(o.kind); g.docs.set(o.doc, d);
    }
    // attach single-token names (an ID card prints surname and given name on separate lines) to the full name they belong to
    const persons = [...groups.values()].filter((g) => g.kind === "PERSON");
    for (const g of [...groups.values()].filter((g) => g.kind === "GIVEN" || g.kind === "SURNAME")) {
      const tok = [...g.texts.keys()][0].toLowerCase();
      const owner = persons.find((p) => [...p.texts.keys()].some((t) => t.toLowerCase().split(/[\s,]+/).includes(tok)));
      if (owner) { owner.parts.push(g); groups.delete(g.key); }
    }
    const order = ["PERSON", "GIVEN", "SURNAME", "ORG", "ID", "ACCOUNT", "CARD", "CARD_LAST4", "SSN", "ADDRESS", "ZIP", "PHONE", "EMAIL", "DOB", "DATE"];
    const rows = [...groups.values()].sort((a, b) => (order.indexOf(a.kind) + 99) % 99 - (order.indexOf(b.kind) + 99) % 99);
    const tbl = el("table", { class: "casemap" }, el("tr", {}, el("th", { text: "Kind" }), el("th", { text: "Original" }), el("th", { text: "Becomes" }), el("th", { text: "Appears in" })));
    for (const g of rows) {
      const kept = [...g.texts.keys()].every((t) => !replaced(t));
      const docs = new Map([...g.docs]);
      for (const pt of g.parts) for (const [d, v] of pt.docs) { const x = docs.get(d) || { n: 0, kinds: new Set() }; x.n += v.n; v.kinds.forEach((k) => x.kinds.add(k)); docs.set(d, x); }
      const where = el("td", {});
      for (const [d, v] of docs) {
        const pg = review.pages.find((p) => p.doc === d);
        const kinds = [...v.kinds].filter((k) => k !== g.kind).map((k) => KIND_LABEL[k] || k.toLowerCase());
        where.append(el("button", { class: "chip", title: `Open ${fileLabel(d)}`, onclick: () => pg && showPage(pg), text: `${shortName(d)} ×${v.n}${kinds.length ? " (" + kinds.join(", ") + ")" : ""}` }));
      }
      const orig = el("td", { class: "mono" }, [...g.texts.keys()].join(" / "));
      const repl = el("td", { class: "mono repl" }, kept ? "(kept unchanged)" : [...new Set(g.texts.values())].join(" / "));
      if (g.parts.length) {
        const full = [...g.texts.values()][0] || "";
        const bad = g.parts.filter((pt) => !full.split(/[\s,]+/).includes([...pt.texts.values()][0]));
        orig.append(el("div", { class: "muted small", text: "also as " + g.parts.map((pt) => `${KIND_LABEL[pt.kind]} “${[...pt.texts.keys()][0]}”`).join(", ") }));
        repl.append(el("div", { class: bad.length ? "warn small" : "muted small", text: (bad.length ? "⚠ not the same fake name: " : "→ ") + g.parts.map((pt) => `“${[...pt.texts.values()][0]}”`).join(", ") }));
      }
      tbl.append(el("tr", { class: kept ? "kept" : "" }, el("td", { text: (KIND_LABEL[g.kind] || g.kind).toString() }), orig, repl, where));
    }
    const nDocs = new Set(occ.map((o) => o.doc)).size;
    $("case-map-count").textContent = ` · ${rows.length} values across ${nDocs} documents`;
    body.append(rows.length ? tbl : el("p", { class: "muted", text: "Nothing detected yet." }));
  }

  function flagText(o) {
    const CONS = { DOB_IN_FUTURE: "birth date in the future", DOB_AFTER_DOC_DATE: "birth date after the document date",
      DOB_AFTER_ISSUE: "birth date after the issue date", AGE_MISMATCH: "age no longer matches the birth date",
      DATE_ERA_JUMP: "date moved out of the document's era", DATE_NOT_A_DATE: "not a real date", SAME_DATE_SPLIT: "one birth date split",
      MRZ_VISUAL_MISMATCH: "MRZ date differs from the printed date", MRZ_CHECK_DIGIT: "MRZ check digit wrong",
      CARD_LAST4_MISMATCH: "card last four differ", ACCOUNT_LAST4_MISMATCH: "account last four differ", ZIP_STATE_MISMATCH: "ZIP not in the state",
      EMAIL_NAME_MISMATCH: "email does not match the fake name", GENDER_NAME_MISMATCH: "name does not match the stated sex" };
    const f = (o.review || []).map((r) => r.startsWith("CONSISTENCY:") ? "⚠ " + (CONS[r.slice(12)] || r.slice(12).replace(/_/g, " ").toLowerCase())
      : r.split(" ")[0].replace(/_/g, " ").toLowerCase());
    return f.length ? f.join("; ") : "";
  }
  function renderItems() {
    const box = $("items"); box.textContent = "";
    const p = current;
    const occ = review.proposals.occurrences.filter((o) => o.doc === p.doc && (p.kind === "json" || o.page === p.page));
    const byText = new Map();
    for (const o of occ) { if (!byText.has(o.text)) byText.set(o.text, o); }
    if (!byText.size && !review.proposals.regions.some((r) => r.doc === p.doc && r.page === p.page)) box.append(el("p", { class: "muted", text: "Nothing detected on this page. Check the page yourself and add anything missed." }));
    for (const [text, o] of byText) {
      const total = review.proposals.occurrences.filter((x) => x.text === text).length;
      const replace = replaced(text);
      if (state.sugg.has(text)) { box.append(suggestionRow(text, o, total)); continue; }
      if (text === null) {  // PCI sensitive authentication data: value is never stored, always replaced
        box.append(el("div", { class: "item" }, el("div", { class: "kind", text: o.kind }, " ", regimeBadges(["PCI"])),
          el("div", { class: "vals" }, el("span", { class: "orig", text: "(card security data, not stored)" }), el("span", { class: "arrow", text: " → " }), el("span", { class: "repl", text: "random digits" })),
          el("div", { class: "muted small", text: "Always replaced. CasePatch never writes this value to disk." })));
        continue;
      }
      const flag = flagText(o);
      const row = el("div", { class: "item" + (replace ? "" : " kept") + (flag ? " flagged" : "") },
        el("div", { class: "kind", text: o.kind }),
        el("div", { class: "vals" }, el("span", { class: "orig", text }), el("span", { class: "arrow", text: " → " }), el("span", { class: "repl", text: replace ? o.replacement : "(kept unchanged)" })),
        flag ? el("div", { class: "flagtext", text: "Check: " + flag }) : null,
        el("div", { class: "row-actions" },
          el("label", {}, el("input", { type: "checkbox", ...(replace ? { checked: "" } : {}), onchange: (e) => { if (e.target.checked) state.reject.delete(text); else state.reject.add(text); setDirty(true); renderItems(); renderSummary(); renderCaseMap(); } }), " Replace"),
          el("span", { class: "muted", text: total > 1 ? ` · ${total} places in the case` : "" })));
      box.append(row);
    }
    const CAT_LABEL = { DIAGNOSIS: "Medical condition", DIAGNOSIS_CODE: "Diagnosis code", MEDICATION: "Medication", PROCEDURE: "Procedure",
      PROCEDURE_CODE: "Procedure code", LAB_RESULT: "Lab result", FACILITY: "Treating facility", SPECIAL: "Sensitive category (GDPR Art. 9/10)",
      INCOME: "Income", CREDIT_SCORE: "Credit score", URL: "Personal URL", GEO: "GPS coordinates" };
    const flags = (review.category_flags || []).filter((f) => f.doc === p.doc && (p.kind === "json" || f.page === p.page));
    const seenF = new Set();
    for (const f of flags) {
      const k = f.kind + "|" + f.text; if (seenF.has(k)) continue; seenF.add(k);
      box.append(el("div", { class: "item catflag-item" },
        el("div", { class: "kind", text: (CAT_LABEL[f.kind] || f.kind) }, " ", regimeBadges(KIND_REGIMES[f.kind])),
        el("div", { class: "vals" }, el("span", { class: "orig", text: f.text ?? "(not stored)" }), el("span", { class: "arrow", text: " · " }), el("span", { class: "muted", text: "kept unchanged" })),
        el("div", { class: "muted small", text: f.reason === "NO_REALISTIC_SUBSTITUTE_KEPT_FOR_REVIEW"
          ? "Free text with no realistic same-kind substitute, so it stays unchanged. Confirm it may stay in the export, or edit the source."
          : "Kept because this case keeps this category (Health and sensitive details). Confirm it may stay in the export." })));
    }
    for (const r of review.proposals.regions.filter((r) => r.doc === p.doc && r.page === p.page)) {
      const sel = el("select", { onchange: (e) => { state.regions[r.id] = e.target.value; setDirty(true); } });
      for (const [v, label] of [["replace", "Replace with synthetic"], ["erase", "Erase (fill with background)"], ["keep", "Keep unchanged"]]) {
        const opt = el("option", { value: v, text: label }); if (state.regions[r.id] === v) opt.selected = true; sel.append(opt);
      }
      box.append(el("div", { class: "item region-item" }, el("div", { class: "kind", text: r.kind.toUpperCase() }),
        el("div", { class: "muted small", text: r.reason }), sel));
    }
  }
  function suggestionRow(text, o, total) {
    const acc = state.accept.has(text), seen = state.reviewed.has(text);
    const decide = (yes) => {
      const was = state.accept.has(text);
      if (yes) state.accept.add(text); else state.accept.delete(text);
      state.reviewed.add(text);
      if (was !== yes) setDirty(true);
      renderItems(); renderSummary(); renderCaseMap(); showBoxesOnly();
    };
    return el("div", { class: "item suggestion" + (acc ? "" : " kept") },
      el("div", { class: "badge-model", text: "Model suggestion · not found by local detection" }),
      el("div", { class: "kind", text: o.kind }),
      el("div", { class: "vals" }, el("span", { class: "orig", text }), el("span", { class: "arrow", text: " → " }), el("span", { class: "repl", text: acc ? o.replacement : "(not applied unless you accept)" })),
      el("div", { class: "row-actions sugg-actions" },
        el("button", { class: "button tiny" + (acc ? " primary" : ""), text: acc ? "Accepted" : "Accept", onclick: () => decide(true) }),
        el("button", { class: "button tiny" + (seen && !acc ? " primary" : ""), text: seen && !acc ? "Declined" : "Decline", onclick: () => decide(false) }),
        el("span", { class: "muted", text: total > 1 ? ` · ${total} places in the case` : "" })));
  }
  function showBoxesOnly() {
    if (!current || current.kind === "json") return;
    for (const b of document.querySelectorAll(".overlay .box[data-id]")) {
      const o = review.proposals.occurrences.find((x) => x.id === b.dataset.id); if (!o) continue;
      b.classList.remove("sugg", "keep", "flag", "repl");
      b.classList.add(state.sugg.has(o.text) && !state.accept.has(o.text) ? "sugg" : (!replaced(o.text) ? "keep" : (o.review && o.review.length ? "flag" : "repl")));
    }
  }
  function renderAdds() {
    const ul = $("add-list"); ul.textContent = "";
    for (const [i, a] of state.add.entries()) {
      const nf = (review.additions_not_found || []).includes(a.text);
      ul.append(el("li", {}, `${a.kind}: ${a.text}`, nf ? el("span", { class: "warn", text: " — not found in any file; check spelling" }) : null,
        el("button", { class: "link", text: "remove", onclick: (ev) => { ev.preventDefault(); state.add.splice(i, 1); setDirty(true); renderAdds(); } })));
    }
  }
  $("btn-add").addEventListener("click", (ev) => {
    ev.preventDefault();
    const t = $("add-text").value.trim(); if (!t) return;
    if ($("add-kind").value === "PERSON" && !/\s/.test(t)) { showError(new Error("A single word is not a full name. Choose “First name only” or “Surname only”.")); return; }
    state.add.push({ text: t, kind: $("add-kind").value }); $("add-text").value = ""; setDirty(true); renderAdds();
  });
  function setDirty(d) {
    state.dirty = d; $("btn-apply").disabled = !d; $("dirty").classList.toggle("hidden", !d);
  }
  $("btn-apply").addEventListener("click", async () => {
    try {
      await api("POST", `/api/cases/${caseId}/apply`, { regions: state.regions, reject_text: [...state.reject], add: state.add, accept_text: [...state.accept],
                                                         categories: catBody() });
      waitRun("Re-rendering with your choices…");
    } catch (e) { showError(e); }
  });
  $("btn-to-export").addEventListener("click", () => {
    if (state.dirty) { showError(new Error("Apply your changes (or reload) before exporting, so the export matches what you reviewed.")); return; }
    const pend = pendingSuggestions();
    if (pend.length) {
      const o = review.proposals.occurrences.find((x) => x.text === pend[0]);
      showError(new Error(`Accept or decline every model suggestion first: ${pend.length} left.`));
      const pg = o && review.pages.find((p) => p.doc === o.doc && (p.kind === "json" || p.page === o.page)); if (pg) showPage(pg);
      return;
    }
    const missing = review.pages.filter((p) => !review.viewedSet.has(p.key));
    if (missing.length) { showError(new Error(`Open every output page first: ${missing.length} not viewed yet (${missing.map((p) => p.label).join(", ")}).`)); showPage(missing[0]); return; }
    renderExport(); screen("export");
  });

  // ---------- step 4: export ----------
  function renderExport() {
    const t = $("verify"); t.textContent = "";
    t.append(el("tr", {}, ...["Document", "Original values left", "Rescan hits (not replacements)", "Changes outside edited areas", "Replacement legibility", "Metadata"].map((h) => el("th", { text: h }))));
    for (const [doc, v] of Object.entries(review.verification || {})) {
      if (typeof v !== "object" || !v) continue;
      const pd = (v.pixel_diff || []).map((x) => x.verdict).join(", ") || (v.kind === "json" ? "n/a" : "—");
      const leg = v.legibility ? `${Math.round(100 * (v.legibility.mean_similarity || 0))}% OCR match` : "—";
      const fmt = (x) => (x && typeof x === "object") ? (Object.values(x).every((y) => !y) ? "none" : Object.entries(x).filter(([, y]) => y).map(([k, y]) => `${k}: ${y}`).join(", ")) : x;
      const meta = v.object_inspection ? Object.entries(v.object_inspection).map(([k, x]) => `${k.replace(/_/g, " ")}: ${fmt(x)}`).join(" · ") : "—";
      t.append(el("tr", {}, el("td", { text: fileLabel(doc) }), el("td", { class: v.residual_original_values ? "bad" : "ok", text: String(v.residual_original_values ?? "—") }),
        el("td", { text: String(v.rescan_pii_like_not_replacements ?? "—") }), el("td", { text: pd }), el("td", { text: leg }), el("td", { class: "small", text: meta })));
    }
    if (!Object.keys(review.verification || {}).length) t.append(el("tr", {}, el("td", { text: "Verification was skipped (--fast)." })));
    const ul = $("needs-review"); ul.textContent = "";
    $("nr-count").textContent = `(${review.needs_review.length})`;
    for (const n of review.needs_review.slice(0, 60)) {
      let reasons = (n.reasons || []).filter(Boolean).map((r) => String(r).split(" ")[0].replace(/_/g, " ").toLowerCase()).join("; ");
      if (n.decision === "reject") reasons = (reasons ? reasons + "; " : "") + "you chose to keep this value unchanged";
      if (!reasons) reasons = "check this value";
      ul.append(el("li", {}, el("strong", { text: (n.kind || "").toString() }), ` in ${fileLabel(n.doc)}${n.page != null ? ", page " + (n.page + 1) : ""}: ${reasons}`));
    }
    $("confirm").checked = false; $("btn-export").disabled = true; $("exported").classList.add("hidden");
  }
  $("confirm").addEventListener("change", (e) => { $("btn-export").disabled = !e.target.checked; });
  $("btn-back").addEventListener("click", () => screen("review"));
  $("btn-export").addEventListener("click", async () => {
    try {
      const blob = await api("POST", `/api/cases/${caseId}/export`, { review_confirmed: $("confirm").checked, suggestions_reviewed: [...state.reviewed] });
      const a = el("a", { href: URL.createObjectURL(blob), download: "casepatch-export-reviewed.zip" });
      document.body.append(a); a.click(); a.remove();
      $("exported").classList.remove("hidden");
    } catch (e) { showError(e); }
  });
  $("btn-new").addEventListener("click", () => resetCase());
  $("btn-delete").addEventListener("click", async () => { await resetCase(); });

  // ---------- usage & privacy ----------
  async function openUsage() {
    cfg = Object.assign(cfg, await api("GET", "/api/config"));
    const t = $("usage"); t.textContent = "";
    const all = cfg.usage.all_time, mo = cfg.usage.this_month;
    if (cfg.usage.file) $("usage-file").textContent = cfg.usage.file;
    if (cfg.usage.events_file) $("events-file").textContent = cfg.usage.events_file;
    t.append(el("tr", {}, el("th", { text: "" }), el("th", { text: "This month" }), el("th", { text: "All time" })));
    for (const [k, label] of [["cases_analyzed", "Cases analyzed"], ["pages_processed", "Pages processed (incl. re-renders)"], ["review_runs", "Re-renders after review"], ["reviewed_exports", "Reviewed exports"], ["pages_exported", "Pages exported"]])
      t.append(el("tr", {}, el("td", { text: label }), el("td", { text: String(mo[k]) }), el("td", { text: String(all[k]) })));
    $("tele").checked = cfg.telemetry.opted_in;
    $("tele-state").textContent = cfg.telemetry.opted_in ? "recording locally · sending disabled" : "off";
    const ev = await api("GET", "/api/events");
    $("events").textContent = ev.events.length ? ev.events.map((e) => JSON.stringify(e)).join("\n") : "(nothing recorded)";
    $("dlg-usage").showModal();
  }
  $("btn-usage").addEventListener("click", () => openUsage().catch(showError));
  $("tele").addEventListener("change", async (e) => { await api("POST", "/api/telemetry", { opt_in: e.target.checked }); await openUsage(); });
  $("btn-reset-usage").addEventListener("click", async (e) => { e.preventDefault(); await api("POST", "/api/usage/reset", {}); await openUsage(); });
  $("btn-help").addEventListener("click", () => $("dlg-help").showModal());
  // first-run walkthrough: the how-it-works steps open once on a fresh install (remembered in this browser only)
  try {
    if (localStorage.getItem("casepatch-help-seen") !== "1") { $("dlg-help").showModal(); localStorage.setItem("casepatch-help-seen", "1"); }
    $("help-again").checked = localStorage.getItem("casepatch-help-seen") !== "1";
    $("help-again").addEventListener("change", (e) => localStorage.setItem("casepatch-help-seen", e.target.checked ? "0" : "1"));
  } catch (e) { /* storage blocked: no auto-open */ }


  // ---------- precomputed editor comparison + optional hosted-edit BYOK (no live calls in this build) ----------
  const CMP = { model: null, showBoxes: true, showFull: true };

  function renderHostedEdit() {
    const box = $("hosted-edit-box"), h = cfg && cfg.hosted_edit;
    if (!box) return;
    if (!h || !h.allowed) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    $("hosted-edit-on").disabled = !h.available;
    $("hosted-edit-reason").textContent = h.available
      ? "Your Fal key is set and stays on this computer only. On the review screen, opt in per document and accept the estimated cost before hosted editors run. They never go through CasePatch servers. The local editor stays the default."
      : ("Not available: " + (h.reason || "not configured"));
    $("btn-hosted-edit-forget").classList.toggle("hidden", !(h.key_source === "studio-settings"));
    if (!$("hosted-edit-on").checked) $("hosted-edit-on").checked = false;
  }
  async function sendFalKey(body) {
    try { const r = await api("POST", "/api/hosted-edit-key", body); cfg.hosted_edit = r.hosted_edit; }
    catch (e) { showError(e); }
    finally { $("hosted-edit-key-input").value = ""; renderHostedEdit(); }
  }
  $("btn-hosted-edit-key").addEventListener("click", () => sendFalKey({ key: $("hosted-edit-key-input").value.trim() }));
  $("btn-hosted-edit-forget").addEventListener("click", () => sendFalKey({ forget: true }));

  function showCompareEntry() {
    const on = !!(cfg && cfg.compare && cfg.compare.available);
    $("compare-entry").classList.toggle("hidden", !on);
  }
  function currentModel() {
    const m = CMP.model;
    if (!m) return null;
    return Object.assign({}, m, { showBoxes: CMP.showBoxes, showFull: CMP.showFull, current: CMP.current });
  }
  function paintCompare() {
    if (!CMP.model || !window.CasePatchCompare) return;
    const handlers = {
      onSelect: (id, editor, override) => pick({ change_id: id, editor, override: !!override }),
      onSelectAll: (editor) => pick({ editor, all: true }),
      onCurrent: (id) => { CMP.current = id; paintCompare(); },
      onToggle: (t) => { CMP.showBoxes = t.showBoxes; CMP.showFull = t.showFull; paintCompare(); }
    };
    window.CasePatchCompare.render($("cmp-mount"), currentModel(), handlers);
  }
  async function loadCompare() {
    const m = await api("GET", "/api/compare");
    // cache-bust composed after images whenever we reload
    const t = Date.now();
    for (const p of m.pages || []) if (p.after_src) p.after_src += (p.after_src.includes("?") ? "&" : "?") + "t=" + t;
    CMP.model = m;
    CMP.current = CMP.current && (m.changes || []).some((c) => c.id === CMP.current) ? CMP.current : (m.changes[0] && m.changes[0].id);
    paintCompare();
  }
  async function pick(body) {
    try {
      await api("POST", "/api/compare/picks", body);
      await loadCompare();
    } catch (e) { showError(e); }
  }
  $("btn-open-compare").addEventListener("click", async () => {
    try { await loadCompare(); screen("compare"); } catch (e) { showError(e); }
  });
  $("btn-cmp-back").addEventListener("click", () => screen("add"));
  $("btn-cmp-export").addEventListener("click", async () => {
    try {
      const blob = await api("POST", "/api/compare/export", { review_confirmed: true });
      const a = el("a", { href: URL.createObjectURL(blob), download: "casepatch-compare-export.zip" });
      document.body.append(a); a.click(); a.remove();
      $("cmp-exported").classList.remove("hidden");
    } catch (e) { showError(e); }
  });
  window.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (!$("screen-compare").classList.contains("hidden") && CMP.model && window.CasePatchCompare) {
      if (window.CasePatchCompare.keydown(ev, currentModel(), {
        onSelect: (id, editor, override) => pick({ change_id: id, editor, override: !!override }),
        onSelectAll: (editor) => pick({ editor, all: true }),
        onCurrent: (id) => { CMP.current = id; paintCompare(); }
      })) ev.preventDefault();
      return;
    }
    if (!$("screen-review").classList.contains("hidden") && LIVE.model && window.CasePatchCompare) {
      if (window.CasePatchCompare.keydown(ev, liveModel(), {
        onSelect: (id, editor, override) => livePick({ change_id: id, editor, override: !!override }),
        onSelectAll: (editor) => livePick({ editor, all: true }),
        onCurrent: (id) => { LIVE.current = id; paintLiveCompare(); }
      })) ev.preventDefault();
    }
  });

  // ---------- live review picker (local default; hosted after opt-in + estimate, fake transport in tests) ----------
  const LIVE = { model: null, current: null, showBoxes: true, estimate: null };

  function liveModel() {
    if (!LIVE.model) return null;
    return Object.assign({}, LIVE.model, { showBoxes: LIVE.showBoxes, showFull: false, current: LIVE.current, live: true });
  }
  function paintLiveCompare() {
    const root = $("review-compare");
    if (!root || !window.CasePatchCompare || !LIVE.model) { if (root) root.classList.add("hidden"); return; }
    root.classList.remove("hidden");
    const handlers = {
      onSelect: (id, editor, override) => livePick({ change_id: id, editor, override: !!override }),
      onSelectAll: (editor) => livePick({ editor, all: true }),
      onCurrent: (id) => { LIVE.current = id; paintLiveCompare(); },
      onToggle: (t) => { LIVE.showBoxes = t.showBoxes; paintLiveCompare(); }
    };
    window.CasePatchCompare.render(root, liveModel(), handlers);
  }
  function syncLiveComparePage() {
    if (!LIVE.model || !current) return;
    const onPage = (LIVE.model.changes || []).find((c) => c.doc === current.doc && (current.kind === "json" || c.page === current.page));
    if (onPage) LIVE.current = onPage.id;
    paintLiveCompare();
  }
  async function loadLiveCompare() {
    if (!caseId) return;
    try {
      const m = await api("GET", `/api/cases/${caseId}/compare`);
      const t = Date.now();
      for (const ch of m.changes || []) {
        if (ch.original_src) ch.original_src += (ch.original_src.includes("?") ? "&" : "?") + "t=" + t;
        for (const cand of Object.values(ch.candidates || {})) {
          if (cand.src) cand.src += (cand.src.includes("?") ? "&" : "?") + "t=" + t;
        }
      }
      LIVE.model = m;
      LIVE.current = LIVE.current && (m.changes || []).some((c) => c.id === LIVE.current) ? LIVE.current : (m.changes[0] && m.changes[0].id);
      renderReviewHosted(m.hosted_edit);
      syncLiveComparePage();
    } catch (e) {
      const root = $("review-compare");
      if (root) root.classList.add("hidden");
    }
  }
  async function livePick(body) {
    try {
      await api("POST", `/api/cases/${caseId}/compare/picks`, body);
      await loadLiveCompare();
    } catch (e) { showError(e); }
  }
  function renderReviewHosted(h) {
    const box = $("review-hosted-edit");
    if (!box) return;
    if (!h || !h.allowed) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    $("review-hosted-notice").textContent = h.notice || "";
    const docs = $("review-hosted-docs"); docs.textContent = "";
    for (const d of h.documents || []) {
      const id = "opt-in-" + d;
      const lab = el("label", { class: "toggle" },
        el("input", { type: "checkbox", id, value: d, ...( (h.opted_in || []).includes(d) ? { checked: true } : {}) }),
        " " + d);
      docs.append(lab);
    }
    const ready = !!(h.available && h.live);
    $("btn-hosted-estimate").disabled = !h.available;
    $("btn-hosted-run").disabled = true;
    $("review-hosted-status").textContent = h.available
      ? (h.live
          ? "A transport is configured for this session. Hosted editors still need your opt-in and an accepted estimate."
          : "Your key is set. Hosted editors run in this work only with a test (fake) transport; no real Fal call is made.")
      : ("Add your Fal key on the first screen first. " + (h.reason || ""));
    if (LIVE.estimate) {
      $("review-hosted-estimate").textContent = "Estimated cost: $" + Number(LIVE.estimate.usd).toFixed(4) +
        " for " + LIVE.estimate.n_calls + " calls (" + LIVE.estimate.n_changes + " changes × " + LIVE.estimate.n_editors + " editors).";
      $("btn-hosted-run").disabled = !($("review-hosted-ack").checked && ready && optedDocs().length);
    } else {
      $("review-hosted-estimate").textContent = "";
    }
  }
  function optedDocs() {
    return [...document.querySelectorAll("#review-hosted-docs input[type=checkbox]:checked")].map((i) => i.value);
  }
  $("btn-hosted-estimate").addEventListener("click", async () => {
    try {
      if (!$("review-hosted-ack").checked) { showError(new Error("Confirm the hosted-edit notice first.")); return; }
      const docs = optedDocs();
      if (!docs.length) { showError(new Error("Select at least one document.")); return; }
      LIVE.estimate = await api("POST", `/api/cases/${caseId}/hosted-edit/estimate`, { documents: docs });
      $("review-hosted-estimate").textContent = "Estimated cost: $" + Number(LIVE.estimate.usd).toFixed(4) +
        " for " + LIVE.estimate.n_calls + " calls. Review this amount before running.";
      $("btn-hosted-run").disabled = false;
    } catch (e) { showError(e); }
  });
  $("btn-hosted-run").addEventListener("click", async () => {
    try {
      if (!$("review-hosted-ack").checked) { showError(new Error("Confirm the hosted-edit notice first.")); return; }
      const docs = optedDocs();
      if (!docs.length || !LIVE.estimate) { showError(new Error("Estimate the cost first.")); return; }
      $("btn-hosted-run").disabled = true;
      $("review-hosted-status").textContent = "Running hosted editors…";
      const r = await api("POST", `/api/cases/${caseId}/hosted-edit/run`, {
        documents: docs, ack: true, notice_version: (LIVE.model && LIVE.model.hosted_edit && LIVE.model.hosted_edit.notice_version) || 1,
        accepted_usd: LIVE.estimate.usd
      });
      $("review-hosted-status").textContent = "Done. Spent $" + Number(r.spent_usd || 0).toFixed(4) + " of the accepted estimate. Local stays selected until you pick a hosted candidate.";
      await loadLiveCompare();
    } catch (e) {
      if (e && e.code === "HOSTED_EDIT_ESTIMATE_REQUIRED" && e.estimate) {
        LIVE.estimate = e.estimate;
        $("review-hosted-estimate").textContent = "Estimated cost changed to $" + Number(e.estimate.usd).toFixed(4) + ". Accept the new amount to run.";
      }
      showError(e);
      $("btn-hosted-run").disabled = false;
    }
  });
  $("review-hosted-ack").addEventListener("change", () => { $("btn-hosted-run").disabled = !($("review-hosted-ack").checked && LIVE.estimate && optedDocs().length); });

  (async () => {
    try {
      cfg = await api("GET", "/api/config"); cfgResolve();
      $("limits").textContent = `PDF, PNG, JPG, WEBP, TIFF documents and JSON, JSONL, TXT traces · up to ${cfg.limits.files} files, ${cfg.limits.file_mb} MB each, ${cfg.limits.pages} pages per case`;
      if (!cfg.sample) $("btn-sample").disabled = true;
      renderHosted(); renderHostedEdit(); showCompareEntry(); renderCats("cat-rows", "cat-summary");
      if (cfg.missing_assets && cfg.missing_assets.length) {
        const w = $("asset-warning"); w.classList.remove("hidden");
        w.textContent = "Some local model files are missing, so parts of the engine are degraded: " +
          cfg.missing_assets.map((a) => `${a.name} — ${a.effect}`).join("; ") + ". Quit and run `casepatch-studio --fetch-models` once to install them.";
      }
      screen("add");
    } catch (e) { showError(e); }
  })();
  // ---------- agent mode (BYOK, opt-in): open review items + picker (agent edit vs original) ----------
  const AG = { model: null, timer: null, open: new Set() };
  async function loadAgent() {
    const root = $("review-agent");
    if (!root || !caseId) return;
    let m;
    try { m = await api("GET", `/api/cases/${caseId}/agent`); } catch (e) { root.classList.add("hidden"); return; }
    AG.model = m; root.classList.remove("hidden");
    const st = m.status || {};
    $("agent-notice").textContent = (m.replay ? "REPLAY: this Studio re-runs a recorded agent run on your computer (no API calls, no key, $0). " : "") +
      (m.notice || "Agent mode sends page images to Anthropic with your own key.");
    $("agent-state").textContent = st.state === "done" ? (m.replay ? `· replay done in ${Math.round(st.wall_s || 0)} s, no API calls` : `· done in ${Math.round(st.wall_s || 0)} s, $${(st.spent_usd || 0).toFixed(2)} on your key`) :
      st.state === "running" ? "· running…" : (m.available ? "" : "· add your Anthropic key under Hosted detection to enable");
    $("agent-start").classList.toggle("hidden", st.state === "running");
    $("btn-agent-run").disabled = !(m.available && $("agent-ack").checked) || st.state === "running";
    const items = $("agent-items"); items.textContent = "";
    $("agent-item-count").textContent = `(${(m.items || []).length})`;
    for (const it of m.items || []) {
      const chk = it.must_check ? el("label", { class: "toggle small" }, Object.assign(el("input", { type: "checkbox" }), { checked: !!it.checked, disabled: !!it.checked,
        onchange: async () => { try { await api("POST", `/api/cases/${caseId}/agent/picks`, { checked: [it.id] }); } catch (e) { showError(e); } await loadAgent(); } }),
        " I checked this area (required before export)") : null;
      items.append(el("li", { class: it.must_check && !it.checked ? "bad" : "" }, el("strong", { text: (it.check || "").toString() }), ` in ${it.doc_label || it.doc}, page ${(it.page || 0) + 1}: ${it.detail || ""}`,
        it.src ? el("img", { src: it.src + "?t=" + Date.now(), class: "agent-crop", alt: "review item crop" }) : null, chk));
    }
    const box = $("agent-changes"); box.textContent = "";
    const groups = new Map();
    for (const ch of m.changes || []) { if (!groups.has(ch.doc)) groups.set(ch.doc, []); groups.get(ch.doc).push(ch); }
    if (!AG.open.size && groups.size) AG.open.add([...groups.keys()][0]);
    const holders = new Map();
    for (const [doc, list] of groups) {
      const flagged = list.filter((c) => !c.badges.agent.selectable).length, kept = list.filter((c) => c.pick === "original").length;
      const det = el("details", { class: "agent-doc" }, el("summary", {}, el("strong", { text: list[0].doc_label || doc }),
        ` · ${list.length} edit${list.length === 1 ? "" : "s"}` + (flagged ? ` · ${flagged} flagged` : "") + (kept ? ` · ${kept} kept original` : "")));
      det.open = AG.open.has(doc) || flagged > 0;
      det.addEventListener("toggle", () => { if (det.open) AG.open.add(doc); else AG.open.delete(doc); });
      box.append(det); holders.set(doc, det);
    }
    for (const ch of m.changes || []) {
      const blocked = !ch.badges.agent.selectable, ov = (m.overrides || []).includes(ch.id);
      // a blocked edit has no pick until the reviewer makes one (a pre-checked radio could never fire the override)
      const radio = (val, label, extra) => el("label", { class: "toggle" }, Object.assign(el("input", { type: "radio", name: "ag-" + ch.id, value: val }), { checked: ch.pick === val && !(val === "agent" && blocked && !ov),
        onchange: () => agentPick(ch.id, val, extra) }), " " + label);
      const same = (m.changes || []).filter((o) => o.id !== ch.id && o.original === ch.original && o.pick !== "original").length;
      const radio2 = (val, label) => el("label", { class: "toggle" }, Object.assign(el("input", { type: "radio", name: "ag-" + ch.id, value: val }), { checked: ch.pick === val,
        onchange: () => { if (same && !confirm(`“${ch.original}” is still replaced in ${same} other place(s). Keeping the original here undoes the replacement only at this spot. Continue?`)) { loadAgent(); return; } agentPick(ch.id, val, false); } }), " " + label);
      holders.get(ch.doc).append(el("div", { class: "agent-change" + (blocked ? " bad" : "") },
        el("div", { class: "small" }, el("strong", { text: ch.kind }), ` · ${ch.doc_label || ch.doc} p${ch.page + 1}: “${ch.original}” → “${ch.replacement}”`),
        el("div", { class: "agent-pair" }, el("img", { src: ch.before_src, class: "agent-crop", alt: "before" }), el("img", { src: ch.after_src, class: "agent-crop", alt: "after" })),
        ch.issues.length ? el("div", { class: "muted small", text: "Checks: " + ch.issues.join("; ") }) : null,
        blocked ? el("div", { class: "bad small", text: `Blocked by ${ch.badges.agent.blocked_by.join(", ")} check${ov ? " (override recorded)" : ""}` }) : null,
        radio("agent", "Agent edit" + (blocked && !ov ? " (needs override)" : ""), blocked && !ov), ch.badges.original.selectable ? radio2("original", "Keep original (undo this edit)") :
          el("span", { class: "muted small", text: " Keep original: not available (the agent turned this PDF page into an image)" })));
    }
    const open = (m.items || []).filter((i) => i.must_check && !i.checked).length +
      (m.changes || []).filter((c) => c.pick === "agent" && !c.badges.agent.selectable && !(m.overrides || []).includes(c.id)).length;
    $("btn-agent-export").disabled = st.state !== "done" || open > 0;
    $("btn-agent-export").title = open ? `${open} flagged item(s) still need a decision` : "";
    clearTimeout(AG.timer);
    if (st.state === "running") AG.timer = setTimeout(loadAgent, 4000);
  }
  async function agentPick(id, val, needsOverride) {
    try {
      const body = { chosen: { [id]: val } };
      if (needsOverride) {
        if (!confirm("The checks flagged this agent edit (an original value may still be visible). Keep it anyway?")) { await loadAgent(); return; }
        body.override = [id];
      }
      await api("POST", `/api/cases/${caseId}/agent/picks`, body);
    } catch (e) { showError(e); }
    await loadAgent();
  }
  if ($("agent-ack")) {
    $("agent-ack").addEventListener("change", () => { $("btn-agent-run").disabled = !($("agent-ack").checked && AG.model && AG.model.available); });
    $("btn-agent-run").addEventListener("click", async () => {
      if (AG.model && AG.model.status && AG.model.status.state === "done" &&
          !confirm("Run agent mode again? The current agent result and your picks for it are replaced.")) return;
      try {
        await api("POST", `/api/cases/${caseId}/agent/run`, { notice_version: AG.model.notice_version || "agent-2026-11-13", faces: $("agent-faces").checked ? "swap" : "keep" });
      } catch (e) { showError(e); }
      await loadAgent();
    });
    $("btn-agent-export").addEventListener("click", async () => {
      if (!confirm("Export the agent result with your picks? Check every review item first.")) return;
      try {
        const blob = await api("POST", `/api/cases/${caseId}/agent/export`, { review_confirmed: true });
        const a = el("a", { href: URL.createObjectURL(blob), download: "casepatch-agent-export.zip" });
        document.body.append(a); a.click(); a.remove();
      } catch (e) { showError(e); }
    });
  }
  window.CasePatchStudio = { reloadReview: () => loadReview() };
})();
