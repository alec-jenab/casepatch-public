"""CasePatch document pipeline (v0.3.0 spike): PDFs, scans, photos, images and JSON traces.

Layers: extract (native text + OCR, reconciled) -> detect (pluggable, grounded proposals)
-> visual (faces/signatures/codes to replace in place) -> resolve (one case-scoped identity map)
-> review (HTML + JSON decisions) -> render (constrained edits into fresh files) -> verify.
Heavy dependencies are imported lazily; install with `pip install -e .[docs]`.
"""
