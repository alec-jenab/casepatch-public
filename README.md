# casepatch

Production data anonymization for AI agent evals. CasePatch replaces the people, accounts, addresses and other
identifiers in a case (PDFs, scans and photos, plus the agent's JSON / JSONL / text traces) with realistic fake
values, in place, and keeps them consistent across every file of the case.

```sh
pip install casepatch              # cloud client, no dependencies
pip install "casepatch[local]"     # local engine: files never leave your machine
```

Two modes, one interface:

| | `mode="local"` | `mode="cloud"` |
|---|---|---|
| Where it runs | In this Python process | casepatch.com API |
| Upload | None: files never leave the machine | Files sent over HTTPS, deleted within 15 minutes |
| Key | None | `CASEPATCH_API_KEY` |
| Install | `pip install "casepatch[local]"` + `casepatch setup-local` (models, ~210 MB) | `pip install casepatch` (no dependencies) |
| Cost | Free | Free monthly pages, then prepaid credits |

`mode="auto"` (the default, or set `CASEPATCH_MODE`) uses local when the local engine is installed, otherwise cloud.
Local only? Pass `mode="local"` (CLI: `--local`): it never uploads and stops with an error if the engine can't run.
Auto mode also stops, rather than uploading, when the local engine is installed but can't load. There is no
self-hosted HTTP server: on your own servers, use this package or the CLI.

Worked example with downloadable fictional files (statement, dispute letter, JSONL agent trace, expected answers):
https://casepatch.com/guides/anonymize-production-data-for-ai-agent-evals.html

## One case folder

Put everything that belongs to one case in one folder: documents (PDF, PNG, JPG, TIFF, WEBP) and traces (`.json`, `.jsonl`,
`.txt`: transcripts, tool calls, expected outputs). The folder is ONE case: the same person, account, card or date
gets the same replacement in every document and every trace, including JSON fields such as tool-call arguments and
expected outputs. JSON keeps its keys, nesting and value types; JSONL keeps one record per line; text keeps its lines.

```python
from casepatch import CasePatch

cp = CasePatch(mode="local")              # or CasePatch(mode="cloud", api_key="cpk_...")  /  CASEPATCH_API_KEY
results = cp.anonymize_folder("./case", "./anon", report="./anon/changes.jsonl")
for r in results:
    print(r.source, r.ok, r.summary)      # {'PERSON': 3, 'CARD_LAST4': 2, ...}
    print(r.review_line())                # what to check before sharing (never claims zero leaks)
```

```sh
casepatch anonymize ./case --out ./anon --local
casepatch anonymize ./case --out ./anon --cloud          # uses CASEPATCH_API_KEY; the folder is sent as one case
casepatch anonymize statement.pdf --out anon.pdf          # one file
casepatch anonymize ./case --out ./anon --local --keep "Acme Bank"
```

`CasePatch(keep=["Acme Bank"])` (or `--keep` / `--keep-file`) leaves those terms unchanged in every file of the case.

A file that fails its type/content check (for example invalid JSON, with the line number) is reported and left out;
the other files still run as one case. If the case itself cannot run, every file reports the error: a case is never
split into files with separate identity maps. Traces must be UTF-8, at most 5 MB (JSONL: 20,000 records).

## What the result says

Each `Result` has `output`, `summary`, `changes` (original -> replacement, keep this private) and what the engine's
report says about it: `engine_version`, `is_synthetic` (always `False`: values CasePatch did not detect stay
original), `source_sent_to_cloud`, `likely_misses` (structural signs a value may still be original), `verification`
and `gate`. `review_required` is `False` only when verification ran and found nothing to check; values CasePatch never
detected are not counted, so review every output before you share it. This is not a compliance certification or a
formal de-identification determination.

## Where files go

| Folder | Default | Override |
|---|---|---|
| Settings, numeric usage counters (`usage.sqlite`: per-day counts only, never sent), Studio cases | `~/.casepatch` | `CASEPATCH_HOME` |
| Cache (fonts) | `~/.cache/casepatch` | `CASEPATCH_CACHE_DIR`, or `$XDG_CACHE_HOME/casepatch` |
| Model files from `casepatch setup-local` | `<cache>/models` | `CASEPATCH_MODEL_DIR` |

HTTPS uses this Python's trusted certificates; on a python.org macOS Python without them it uses the system roots.
Behind a proxy or with a custom CA, set `SSL_CERT_FILE`.

## API

The cloud mode wraps the REST API described at https://casepatch.com/api/v1/openapi.json:
`POST /api/v1/anonymize` (one document, one trace, or a ZIP of a whole case), `GET /api/v1/jobs/{id}`,
`GET /api/v1/jobs/{id}/result`, `DELETE /api/v1/jobs/{id}`, `GET /api/v1/usage`. Authenticate with
`Authorization: Bearer <key>`.

Support: support@casepatch.com
