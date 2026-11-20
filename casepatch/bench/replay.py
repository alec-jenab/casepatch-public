"""Agent replay: does a downstream agent reach the SAME outcome on the original case and on the CasePatch output?

Design: bench/models/EDIT_BENCHMARK.md ("Agent replay"). For a case with an agent trace (`*agent_trace.json`, holding
`task`, tool steps, `expected_decision`), the same agent is run twice with the same prompt:
  (a) original documents + original trace tool results, (b) scrubbed documents + scrubbed trace tool results.
The final summary step and the answer keys (`expected_decision`, `evidence`) are removed before the agent sees the
trace. Each run returns the same fields; the check passes when every decision-relevant field agrees between (a) and (b)
while the identities in (b) are replaced (no original holder / ID name left) and stay consistent across files.

Judges (no paid calls anywhere in this module):
  local-rules   deterministic reference agent: reads the tool results, re-checks them against the document text
                (holder on the statement, ID name on the ID, expiry date read from the ID, amounts present, duplicate
                and receipt-vs-statement differences) and applies a fixed decision policy.
  local-qwen    Qwen2.5-3B-Instruct (Q4_K_M, llama.cpp, CPU) answering a JSON schema; $0, slow, noisy.
  transport     any chat model behind an injected `transport(messages, schema) -> dict`; tests use a fake transport.
                Hosted providers are intentionally NOT wired here: a paid judge needs its own approval and cap.

    python -m casepatch.bench.replay --input examples/real/case-real-01 --output <casepatch out dir> \\
        --judges local-rules [local-qwen] --json result.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

SCHEMA_ID = "casepatch-replay/v1"
DECISIONS = ["escalate_identity_mismatch_and_expired_id", "refund_duplicate_and_overcharge_after_identity_check",
             "escalate_identity_mismatch", "approve_refund", "deny_claim", "request_more_information"]
# fields every judge returns; all except the free-text ones must agree between original and scrubbed
FIELDS = ["decision", "id_matches_account_holder", "id_expired", "key_amounts", "escalate"]
RULE_FIELDS = FIELDS + ["holder_on_statement", "id_name_on_id", "charge_on_statement", "duplicate_confirmed", "overcharge_confirmed"]
ANSWER_SCHEMA = {"type": "object", "properties": {
    "decision": {"type": "string", "enum": DECISIONS},
    "id_matches_account_holder": {"type": "boolean"}, "id_expired": {"type": "boolean"},
    "key_amounts": {"type": "array", "items": {"type": "string"}}, "escalate": {"type": "boolean"}},
    "required": FIELDS}
DATE_RX = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
AMOUNT_KEYS = re.compile(r"amount|total|balance|difference", re.I)


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).lower()


def _amount_key(a):
    return re.sub(r"[^\d]", "", str(a)).lstrip("0")


# ------------------------------------------------------------------ inputs
def find_trace(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() == ".json" and p.name != "labels.json":
            try:
                d = json.loads(p.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(d, dict) and "steps" in d:
                return p, d
    return None, None


def agent_view(trace):
    """What the replayed agent sees: task + tool calls/results, without the final summary or the answer keys."""
    steps = [s for s in trace.get("steps", []) if s.get("tool") not in ("llm_summarize",)]
    return {"task": trace.get("task"), "steps": steps}


class CaseReader:
    """Original-file-name -> text (text layer + OCR) for the original folder or a CasePatch output folder."""

    def __init__(self, reader=None):
        self.reader = reader
        self._cache = {}

    def texts(self, input_dir: Path, folder: Path, mapping=None):
        from .score import Reader, auto_map
        self.reader = self.reader or Reader()
        names = sorted(f.name for f in input_dir.iterdir() if f.is_file() and f.name != "labels.json")
        m = mapping or ({n: n for n in names} if folder == input_dir else auto_map(input_dir, folder))
        out = {}
        for orig, mapped in m.items():
            p = folder / mapped
            if not p.exists() or p.suffix.lower() == ".json":
                continue
            if p not in self._cache:
                lay = self.reader.layers(p)
                self._cache[p] = (lay.get("text") or "") + "\n" + (lay.get("ocr") or "")
            out[orig] = self._cache[p]
        trace = None
        for orig, mapped in m.items():
            if mapped.endswith(".json") and (folder / mapped).exists():
                d = json.loads((folder / mapped).read_text())
                if isinstance(d, dict) and "steps" in d:
                    trace = d
        return out, trace


# ------------------------------------------------------------------ judges
def _results(view):
    flat = {}
    for s in view["steps"]:
        r = s.get("result")
        if isinstance(r, dict):
            for k, v in r.items():
                flat.setdefault(k, v)
    return flat


def _role(texts, *words):
    for name, t in texts.items():
        if any(w in name.lower() for w in words):
            return t
    return ""


def _name_tokens(s):
    return {t for t in re.findall(r"[^\W\d_]{2,}", str(s or "").lower())}


class RulesJudge:
    """Deterministic reference agent (no model)."""
    name = "local-rules"

    def __init__(self, today=None):
        self.today = today or dt.date.today()

    def judge(self, view, texts):
        r = _results(view)
        stmt, idt = _role(texts, "statement"), _role(texts, "id_")
        alltext = _norm("\n".join(texts.values()))
        holder = r.get("customer_name")
        id_name = " ".join(x for x in (r.get("id_given_name"), r.get("id_surname")) if x)
        out = {"holder_on_statement": bool(holder) and _norm(holder) in _norm(stmt),
               "id_name_on_id": bool(id_name) and all(t in _norm(idt) for t in _name_tokens(id_name))}
        out["id_matches_account_holder"] = bool(holder and id_name and _name_tokens(holder) & _name_tokens(id_name))
        dates = []
        for d_, m_, y_ in DATE_RX.findall(idt):
            try:
                dates.append(dt.date(int(y_), int(m_), int(d_)))
            except ValueError:
                pass
        out["id_expired"] = bool(len(dates) >= 2 and max(dates) < self.today)
        amounts = sorted({str(v) for k, v in r.items() if AMOUNT_KEYS.search(k) and re.search(r"\d", str(v))})
        digits = re.sub(r"[^\d\n]", "", alltext)
        out["key_amounts"] = sorted(a for a in amounts if _amount_key(a) and _amount_key(a) in digits)
        sd = re.sub(r"[^\d\n]", "", _norm(stmt))
        rec_total = r.get("total")
        out["charge_on_statement"] = bool(rec_total) and _amount_key(rec_total) in sd
        dup = next((s.get("result") for s in view["steps"] if s.get("tool") == "find_duplicates"), None)
        out["duplicate_confirmed"] = bool(dup and int(dup.get("count", 0)) >= 2 and
                                          sd.count(_amount_key(dup.get("amount", ""))) >= 2)
        cmp_ = next((s.get("result") for s in view["steps"] if s.get("tool") == "compare_receipt"), None)
        out["overcharge_confirmed"] = bool(cmp_ and _amount_key(cmp_.get("statement_amount")) in sd and
                                           _amount_key(cmp_.get("statement_amount")) != _amount_key(cmp_.get("receipt_total")))
        if out["duplicate_confirmed"] or out["overcharge_confirmed"]:
            d = "refund_duplicate_and_overcharge_after_identity_check"
        elif not out["id_matches_account_holder"] and out["id_expired"]:
            d = "escalate_identity_mismatch_and_expired_id"
        elif not out["id_matches_account_holder"]:
            d = "escalate_identity_mismatch"
        else:
            d = "approve_refund" if out["charge_on_statement"] else "request_more_information"
        out["decision"] = d
        out["escalate"] = d.startswith("escalate") or not out["id_matches_account_holder"]
        return out


PROMPT = ("You are a careful payments-operations agent. Decide the case using ONLY the tool results and document text "
          "below. Answer with JSON: decision (one of: {decisions}), id_matches_account_holder (does the ID card name "
          "match the account holder?), id_expired (is the ID's expiry date in the past? today is {today}), key_amounts "
          "(the disputed amounts exactly as printed), escalate (must a human team review it?).\n\nTask: {task}\n\n"
          "Tool results:\n{tools}\n\nDocuments:\n{docs}")


def build_prompt(view, texts, today, per_doc=1800):
    docs = "\n".join(f"=== {n} ===\n{t.strip()[:per_doc]}" for n, t in sorted(texts.items()))
    return PROMPT.format(decisions=", ".join(DECISIONS), today=today.isoformat(), task=view["task"],
                         tools=json.dumps(view["steps"], ensure_ascii=False), docs=docs)


class TransportJudge:
    """A chat model behind `transport(messages, schema) -> dict` (tests: a fake transport; never a paid call here)."""

    def __init__(self, transport, name="transport", today=None):
        self.transport, self.name, self.today = transport, name, today or dt.date.today()

    def judge(self, view, texts):
        msgs = [{"role": "user", "content": build_prompt(view, texts, self.today)}]
        ans = self.transport(msgs, ANSWER_SCHEMA) or {}
        return {k: ans.get(k) for k in FIELDS}


def qwen_transport(model_file="qwen2.5-3b-instruct-q4_k_m.gguf", n_ctx=8192, threads=8):
    from llama_cpp import Llama
    from ..docs.models import find_model
    path = find_model(model_file)
    if path is None:
        raise RuntimeError("LOCAL_LLM_MODEL_NOT_FOUND")
    llm = Llama(str(path), n_ctx=n_ctx, n_threads=threads, verbose=False, seed=0)

    def call(messages, schema):
        r = llm.create_chat_completion(messages=messages, response_format={"type": "json_object", "schema": schema},
                                       temperature=0, max_tokens=400)
        try:
            return json.loads(r["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError):
            return {}
    return call


def make_judge(name, today=None):
    if name == "local-rules":
        return RulesJudge(today)
    if name == "local-qwen":
        return TransportJudge(qwen_transport(), "local-qwen2.5-3b", today)
    raise SystemExit(f"judge {name!r} is not available here (paid hosted judges are not wired into the replay)")


# ------------------------------------------------------------------ scoring
def _same(a, b, field):
    if field == "key_amounts":
        return sorted(_amount_key(x) for x in (a or [])) == sorted(_amount_key(x) for x in (b or []))
    return a == b


def compare(orig, scrub, fields, expected=None):
    per = {f: {"original": orig.get(f), "scrubbed": scrub.get(f), "same": _same(orig.get(f), scrub.get(f), f)} for f in fields}
    res = {"fields": per, "all_same": all(v["same"] for v in per.values()),
           "decision_same": per.get("decision", {}).get("same")}
    if expected:
        res["original_matches_expected"] = orig.get("decision") == expected
        res["scrubbed_matches_expected"] = scrub.get("decision") == expected
    return res


def identity_check(orig_view, scrub_view):
    """Identities in the scrubbed trace must be replaced (no original holder / ID name left)."""
    ro, rs = _results(orig_view), _results(scrub_view)
    keys = [k for k in ("customer_name", "id_surname", "id_given_name", "id_card_no", "attention", "claimant_attorney", "from", "to")
            if ro.get(k)]
    kept = [k for k in keys if str(rs.get(k, "")).strip().lower() == str(ro[k]).strip().lower()]
    return {"identity_fields": len(keys), "left_unchanged": kept, "replaced": not kept}


def replay(input_dir, output_dir, judges=("local-rules",), today=None, reader=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    tp, trace = find_trace(input_dir)
    if trace is None:
        return {"schema": SCHEMA_ID, "case": input_dir.name, "skipped": "no agent trace in this case"}
    cr = CaseReader(reader)
    t_orig, _ = cr.texts(input_dir, input_dir)
    t_scrub, trace_s = cr.texts(input_dir, output_dir)
    if trace_s is None:
        return {"schema": SCHEMA_ID, "case": input_dir.name, "error": "scrubbed trace not found in output"}
    vo, vs = agent_view(trace), agent_view(trace_s)
    res = {"schema": SCHEMA_ID, "case": input_dir.name, "task": trace.get("task"), "expected_decision": trace.get("expected_decision"),
           "identity": identity_check(vo, vs), "judges": {}}
    for j in judges:
        jd = make_judge(j, today) if isinstance(j, str) else j
        a, b = jd.judge(vo, t_orig), jd.judge(vs, t_scrub)
        fields = RULE_FIELDS if isinstance(jd, RulesJudge) else FIELDS
        res["judges"][jd.name] = compare(a, b, fields, trace.get("expected_decision"))
    res["passed"] = bool(res["identity"]["replaced"] and all(v["all_same"] for v in res["judges"].values()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--judges", nargs="+", default=["local-rules"]); ap.add_argument("--json")
    a = ap.parse_args()
    r = replay(a.input, a.output, a.judges)
    s = json.dumps(r, indent=1, ensure_ascii=False, default=str)
    if a.json:
        Path(a.json).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
