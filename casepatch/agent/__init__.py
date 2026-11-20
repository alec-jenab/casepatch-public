"""CasePatch agent mode (2026-11-12 redesign, bench/agentic/RESULTS.md): a frontier model does the reading and decides
what to replace; CasePatch's own working pieces are exposed to it as tools (OCR with boxes, font-matched in-place text
replacement, erase, native PDF text edits), and CasePatch's deterministic checks verify the result and feed failures
back for a bounded number of fix rounds. Product use is BYOK only (the customer's own key, off by default)."""
