# casepatch.agent — agent mode (BYOK, off by default)

A frontier model (default `claude-sonnet-5-5`) reads each document and decides what to replace; CasePatch's own pieces
are its tools; CasePatch's deterministic checks verify the result and send failures back.

* **Tools** (`tools.py`): `ocr_boxes` (lines + bbox + glyph height + ink colour), `view` (page/crop, original or current),
  `draw_text` (erase + typeset in a named font, auto size/baseline/ink/stroke weight, paper grain, mask-only composite),
  `replace_text` (CasePatch's automatic font-matching editor; native PDF text-layer edit when the page has text),
  `replace_signature`, `erase_region` (paper-matched fill), `undo`, `write_map`/`get_map` (case-wide, one fake per original),
  optional `hosted_patch_edit` (only when a customer-enabled callable is injected; not wired to a provider yet).
  Export strips image metadata; PDFs get metadata/XMP cleared, hidden text/JS/attachments scrubbed, raster-edited pages replaced.
* **Verifier** (`verify.py`, $0): OCR leak of every mapped original and each capitalised name token; ghost ink on every
  edited area (residual layer, enhanced crop, glyph-persistence correlation); edit scope (edits may only touch mapped text,
  no pixel changes outside recorded masks); `docs.consistency` on a fact graph built from the map; hidden layers/metadata.
* **Loop** (`run.py`): one session per document, documents in parallel (shared map), verifier after `finish`, up to
  2 fix rounds; the last round escalates to `claude-opus-5-5`; hard per-case budget (default $3) across all documents.
  Remaining issues -> `review-issues.json` (for the review screen) + `agent-report.json`.
* **Keys**: product mode = BYOK via `docs.byok` (customer's own key, refuses CasePatch eval-key fingerprints, in-memory
  budget, direct to the provider). `--eval` = bench only (eval key, `examples/real` only, shared spend ledger, $30 experiment cap).

    python -m casepatch.agent INPUT_DIR --out OUT --enable-agent [--budget 3]

Results: `bench/agentic/RESULTS.md` (agent mode section).

## Round 2 additions
- **Budgets:** per-document budget (`--doc-budget`, default $1.50) under a case ceiling (`--budget`, default $6). A document
  that runs out stops gracefully with a `budget` review item; Opus escalation only happens with >35% of the doc budget left.
- **Faces:** `replace_face` swaps a portrait with an SFHQ bank face and checks identity with SFace (cosine < 0.363).
  `--faces keep` (the Studio default for uploads) flags photos for review instead of swapping.
- **Signatures / sideways text:** `replace_signature`, `draw_text` and `erase_region` take `rot` (90/180/270); the verifier
  reports any detected signature/photo region that is unchanged.
- **Length fit:** `draw_text` fits the original's width with tracking or horizontal scale and warns when the fake is >12% off;
  `write_map` notes length mismatches so the agent picks a similar-width fake.
- **Leftover strokes:** erase grows the mask to connected ink components mostly inside the box (handwriting tails).
- **Studio:** review screen → "Agent mode" panel (own Anthropic key, notice + opt-in). Shows open review items with crops and
  each agent edit as a picker row (agent edit vs keep original; verifier-blocked edits need an explicit override).
  Masks are saved under `edits/<file>/<id>.png` so a revert restores only that edit's pixels.
