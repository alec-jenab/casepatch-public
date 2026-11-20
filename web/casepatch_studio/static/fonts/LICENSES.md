# Vendored UI fonts

These files are bundled with CasePatch Studio and the demo so the UI can load a
modern typeface with no network request. The demo Content-Security-Policy
blocks all external resources (`default-src 'none'`), so a CDN font would fail
to load and is not used.

## Inter

- Source: [rsms/inter](https://github.com/rsms/inter)
- License: SIL Open Font License 1.1 (see `OFL.txt` in this directory)
- Copyright: 2016 The Inter Project Authors (https://github.com/rsms/inter)
- Files: `inter-latin-var.woff2`, `inter-latin-var-italic.woff2`
- Subset: Latin (`U+0000-00FF` plus common punctuation and typographic
  symbols), variable `opsz` and `wght` axes, roman and italic, as WOFF2.
  Generated with `pyftsubset` from the upstream variable TTFs. No fonts are
  fetched at runtime.

Icons in the UI are inline SVG or CSS-only; no icon font is bundled.
