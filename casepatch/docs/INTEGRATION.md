# Regulated-category detectors in the pipeline

Files: `categories.py` (detector, policy, replacement routing), `category_fakes.py` (checksum-valid fakes),
`category_vocab.py` (CC0 vocabulary), tests in `tests/test_categories.py` and `tests/test_categories_pipeline.py`,
recall bench `python -m casepatch.bench.categories_eval --input examples/real/case-real-05-health --pipeline`.

## How it runs (integrated 2026-11-12)

`pipeline.run(..., categories=True, category_policy=None)` (the default) does, after the form-field detector:

- `hook(chunks, R, policy)` returns `detect.Proposal`s (detector `categories`), merged with rules, form fields,
  reviewer additions and (if opted in) hosted suggestions, then propagated case-wide like any detector. `install(R)`
  (idempotent) adds category kinds to `R.PRIORITY` and exempts `CARD_EXPIRY`, `DIAGNOSIS_CODE`, `PROCEDURE_CODE`,
  `AGE_90`, `GEO` from `protected_fact`.
- `attach(cmap)` right after the case key is loaded routes category kinds through `CategoryFaker` (same key, so one
  source value gets one fake across files and the trace). `PERSON`, `ORG`, `ACCOUNT` and `MRZ` stay on the engine's
  own CaseMap code (MRZ lines are rewritten with valid ICAO check digits by `docs/mrz.py`).
- `report["category_flags"]`: kept-but-flagged items (`doc/page/line_id/start/end/kind/regulation/confidence`), never
  the source text.
- PCI sensitive authentication data (`CVV`, `PIN`, `TRACK_DATA`, proposal `meta["sad"]`): fresh random digits, never
  cached; `proposals.json` / `review.html` record `text: null`, `replacement: null`, entity `<kind>:hidden`.

Per-kind policy: `category_policy={"DIAGNOSIS": "replace", "MEDICATION": "replace"}` ('replace' | 'flag' | 'keep').
The Studio review screen lists flagged items through `needs_review`; a per-kind "replace with a same-category value"
toggle in Studio is not built yet (CLI/API only).

## Remaining engine gaps

- Multi-line service descriptions (`Eye and medical examination / for diagnosis and treatment, / ... (92014)`) are
  not joined; only one-line `description (code)` rows are recognised (2 misses on case-real-05-health).
