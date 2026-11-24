"""Local, span-based review engine. Source data never enters a hosted service."""
from __future__ import annotations

import copy
import re
import secrets
from dataclasses import dataclass

from .core import PATTERNS, PolicyError, canonical, digest, shifted, walk

KINDS = ("PERSON", "EMAIL", "PHONE", "SSN", "CARD", "ID", "ADDRESS", "LOCATION", "ORGANIZATION", "SECRET", "DATE")
FIELDS = {
    "customer_name": "PERSON", "claimant_name": "PERSON", "signer_name": "PERSON",
    "full_name": "PERSON", "patient_name": "PERSON", "employee_name": "PERSON",
    "email": "EMAIL", "email_address": "EMAIL", "phone": "PHONE", "phone_number": "PHONE",
    "ssn": "SSN", "social_security_number": "SSN", "card": "CARD", "card_number": "CARD",
    "account": "ID", "account_id": "ID", "account_number": "ID", "customer_id": "ID",
    "patient_id": "ID", "record_id": "ID", "address": "ADDRESS", "street_address": "ADDRESS",
    "medical_record_number": "ID", "member_id": "ID", "claim_id": "ID",
    "api_token": "SECRET", "access_token": "SECRET", "api_key": "SECRET",
    "client_secret": "SECRET", "password": "SECRET",
    "date_of_birth": "DATE", "birth_date": "DATE", "dob": "DATE",
}
NORMAL_FIELDS = {re.sub(r"[^a-z0-9]", "", key): value for key, value in FIELDS.items()}
PERSON_CONTEXTS = {"customer", "claimant", "client", "patient", "employee", "cardholder", "person", "user"}


def field_kind(path):
    if not path or not isinstance(path[-1], str):return None
    name = re.sub(r"[^a-z0-9]", "", path[-1].lower())
    if name == "name" and len(path) > 1 and isinstance(path[-2], str) and path[-2].lower() in PERSON_CONTEXTS:
        return "PERSON"
    return NORMAL_FIELDS.get(name)


def object_nodes(node, path=()):
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():yield from object_nodes(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):yield from object_nodes(value, path + (index,))


def conflicting_person_names(documents):
    """Flag evidence needing review; distinct IDs do not prove distinct people."""
    evidence = {}
    for document in documents:
        for path, node in object_nodes(document['content']):
            fields = {re.sub(r'[^a-z0-9]', '', key.lower()): value for key, value in node.items()}
            full = fields.get('fullname')
            if not full and isinstance(fields.get('firstname'), str) and isinstance(fields.get('lastname'), str):
                full = fields['firstname'] + ' ' + fields['lastname']
            for role in ('customer', 'patient'):
                name = fields.get(role + 'name', full)
                if path and isinstance(path[-1], str) and path[-1].lower() == role:
                    name = fields.get('name', name)
                identifier = fields.get(role + 'id')
                if isinstance(name, str) and name.strip() and isinstance(identifier, str) and identifier.strip():
                    evidence.setdefault((role, name), set()).add(identifier.strip())
    return {name for (_, name), identifiers in evidence.items() if len(identifiers) > 1}


MODEL_KINDS = {"PERSON": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION", "ORG": "ORGANIZATION"}
# the number must not be the year or day of a date in front of it (a statement row '11/07/2026  Main Street Deli', '2026-11-07 Corwin Hollow', 'Nov 07 Oak Park Grill')
_NOT_AFTER_DATE = (r"(?<!\d/\d/)(?<!\d/\d\d/)(?<!\d\.\d\d\.)(?<!\d\d-\d\d-)"
                   r"(?<!\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) )(?<!\b(?:jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\. )(?<!\bsept )(?<!\bsept\. )")
STREET_ADDRESS = re.compile(_NOT_AFTER_DATE + r"(?<!\w)\d{1,6}[A-Za-z]?\s+(?:[A-Za-z][A-Za-z.\'’-]*\s+){1,6}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way|Place|Pl|(?-i:Terrace|Ter|Circle|Cir|Parkway|Pkwy|Highway|Hwy|Trail|Trl|Bend|Crescent|Plaza|Ridge|Alley|Hollow|Commons))\b(?:,?\s+(?:Apt|Suite|Unit)\.?\s+[A-Za-z0-9-]+)?(?:,\s+[A-Za-z][A-Za-z .\'’-]{0,60},?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?", re.IGNORECASE)
RULE_KINDS = {"PAN_LIKE": "CARD", "IPV4": "ID", "ISO_DATE": "DATE"}
LIMITS = {"documents": 20, "characters": 250_000, "spans": 4000}


class RulesDetector:
    name = "structured-fields-and-patterns"
    learned = False

    def detect(self, text):
        return []


class SpacyDetector:
    """Load an installed local pipeline; never download a model during a case run."""
    learned = True

    def __init__(self, model="en_core_web_sm"):
        try:
            import spacy
            self.nlp = spacy.load(model, exclude=["parser", "tagger", "lemmatizer", "attribute_ruler"])
            if "ner" not in self.nlp.pipe_names:
                raise PolicyError("LOCAL_MODEL_HAS_NO_NER")
        except (ImportError, OSError):
            raise PolicyError("LOCAL_MODEL_NOT_INSTALLED") from None
        self.name = "spacy-" + model + "-" + self.nlp.meta.get("version", "unknown")

    def detect(self, text):
        return [(ent.start_char, ent.end_char, MODEL_KINDS[ent.label_])
                for ent in self.nlp(text).ents if ent.label_ in MODEL_KINDS]


def text_at(documents, doc, path):
    value = documents[doc]["content"]
    for part in path:
        value = value[part]
    return value


def leaf_texts(documents):
    for index, doc in enumerate(documents):
        for path, text in walk(doc["content"]):
            yield index, path, text


def validate_documents(documents):
    if not isinstance(documents, list) or not 1 <= len(documents) <= LIMITS["documents"]:
        raise PolicyError("USE_ONE_TO_TWENTY_DOCUMENTS")
    size = 0
    for doc in documents:
        if not isinstance(doc, dict) or set(doc) != {"format", "content"}:
            raise PolicyError("DOCUMENT_SCHEMA")
        if doc["format"] == "text":
            if not isinstance(doc["content"], str):
                raise PolicyError("TEXT_CONTENT_REQUIRED")
        elif doc["format"] == "json":
            if not isinstance(doc["content"], (dict, list)):
                raise PolicyError("JSON_OBJECT_OR_LIST_REQUIRED")
        else:
            raise PolicyError("SUPPORTED_FORMATS_TEXT_AND_JSON")
        try:
            canonical(doc)
            size += sum(len(text) for _, text in walk(doc["content"]))
        except (ValueError, RecursionError):
            raise PolicyError("INVALID_JSON_VALUE") from None
    if size == 0 or size > LIMITS["characters"]:
        raise PolicyError("USE_ONE_TO_250000_TEXT_CHARACTERS")


@dataclass
class Span:
    id: str
    document: int
    path: tuple
    start: int
    end: int
    kind: str
    value: str
    source: str
    group: str = ""
    selected: bool = True
    form: str = 'full'
    identity_review: bool = False


class ReviewCase:
    def __init__(self, documents, detector=None, manual=None, key=None):
        validate_documents(documents)
        self.documents = copy.deepcopy(documents)
        self.detector = detector or RulesDetector()
        self.key = key or secrets.token_bytes(32)
        if not isinstance(self.key, bytes) or len(self.key) < 32:
            raise PolicyError("KEY_TOO_SHORT")
        self.scope = secrets.token_hex(16)
        self.spans = []
        self.issues = []
        self.manual = manual or []
        if not isinstance(self.manual, list) or len(self.manual) > 100:
            raise PolicyError("TOO_MANY_MANUAL_VALUES")
        self._discover()

    def _discover(self):
        leaves = list(leaf_texts(self.documents))
        if len(leaves) > 5000:
            raise PolicyError("TOO_MANY_TEXT_FIELDS")
        # Higher priority whole-value anchors prevent fragment matches inside IDs/emails.
        seeds = {}
        for _, path, text in leaves:
            kind = field_kind(path)
            if kind and text.strip():
                if len(text) > 500:
                    raise PolicyError("STRUCTURED_VALUE_TOO_LONG")
                self._seed(seeds, text, kind, "field", 0)
        # Pair explicit first/last fields before suggesting any fragment links.
        # Fragments shared by multiple full names require a review choice.
        name_parts = {}
        for document in self.documents:
            for _, node in object_nodes(document['content']):
                fields = {re.sub(r"[^a-z0-9]", "", key.lower()):value for key,value in node.items()}
                first, last = fields.get('firstname'), fields.get('lastname')
                if first is None and last is None:continue
                if not isinstance(first,str) or not isinstance(last,str) or not first.strip() or not last.strip():
                    raise PolicyError('FIRST_AND_LAST_NAME_MUST_BE_TEXT_PAIR')
                if len(first)>200 or len(last)>200:raise PolicyError('STRUCTURED_VALUE_TOO_LONG')
                full = first + ' ' + last
                self._seed(seeds,full,'PERSON','field',0)
                name_parts.setdefault(first,{'given':set(),'family':set()})['given'].add(full)
                name_parts.setdefault(last,{'given':set(),'family':set()})['family'].add(full)
        for item in self.manual:
            if not isinstance(item, dict) or set(item) != {"value", "kind"}:
                raise PolicyError("MANUAL_VALUE_SCHEMA")
            value, kind = item["value"], item["kind"]
            if not isinstance(value, str) or not 2 <= len(value) <= 500 or kind not in KINDS:
                raise PolicyError("INVALID_MANUAL_VALUE")
            if not any(value in text for _, _, text in leaves):
                raise PolicyError("MANUAL_VALUE_NOT_FOUND")
            self._seed(seeds, value, kind, "manual", -1)
        for _, _, text in leaves:
            for match in STREET_ADDRESS.finditer(text):
                self._seed(seeds, match.group(), "ADDRESS", "street-pattern", 1)
            for rule, pattern in PATTERNS.items():
                kind = RULE_KINDS.get(rule, rule)
                for match in pattern.finditer(text):
                    self._seed(seeds, match.group(), kind, "pattern", 1)
            for start, end, kind in self.detector.detect(text):
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or kind not in KINDS:
                    raise PolicyError("INVALID_DETECTOR_SPAN")
                self._seed(seeds, text[start:end], kind, "model", 2)
        # Initials are linked only when they identify one known full name.
        initials = {}
        for value, (kind, _, _) in list(seeds.items()):
            parts = value.split()
            if kind == "PERSON" and len(parts) == 2 and len(parts[0]) > 1 and not parts[0].endswith("."):
                initials.setdefault(parts[0][0] + ". " + parts[1], []).append(value)
        self.alias_parent = {}
        for fragment, roles in name_parts.items():
            owners = roles['given'] | roles['family']
            # A full-name owner discovered elsewhere also makes a shared fragment ambiguous.
            for value,(kind,_,_) in seeds.items():
                if kind=='PERSON' and value!=fragment and re.search(r'(?<!\w)'+re.escape(fragment)+r'(?!\w)',value):owners.add(value)
            unique_role = 'given' if roles['given'] and not roles['family'] else 'family' if roles['family'] and not roles['given'] else None
            if len(owners)==1 and unique_role:
                self._seed(seeds,fragment,'PERSON','unique-'+unique_role,0)
                self.alias_parent[fragment]=next(iter(owners))
            else:
                self._seed(seeds,fragment,'PERSON','ambiguous-name-part',0)
                self.issues.append('AMBIGUOUS_NAME_PART_REQUIRES_REVIEW')
        for alias, owners in initials.items():
            if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) for _, _, text in leaves):
                if len(owners) == 1:
                    self._seed(seeds, alias, "PERSON", "unique-initial", 1)
                    self.alias_parent[alias] = owners[0]
                else:
                    self._seed(seeds, alias, "PERSON", "ambiguous-initial", 1)
                    self.issues.append("AMBIGUOUS_INITIAL_REQUIRES_REVIEW")
        compiled_seeds = [(re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)"), value, kind, source, priority)
                          for value, (kind, source, priority) in seeds.items()]
        groups = {}
        for doc, path, text in leaves:
            candidates = []
            for pattern, value, kind, source, priority in compiled_seeds:
                # Exact matching deliberately avoids unreviewed fuzzy/OCR identity merges.
                for match in pattern.finditer(text):
                    candidates.append((priority, -(match.end()-match.start()), match.start(), match.end(), kind, value, source))
            accepted = []
            for _, _, start, end, kind, value, source in sorted(candidates):
                if any(start < b and end > a for a, b in accepted):
                    continue
                accepted.append((start, end))
                identity = (kind, self.alias_parent.get(value, value))
                group = groups.setdefault(identity, "g" + str(len(groups)))
                name_field = re.sub(r'[^a-z0-9]','',path[-1].lower()) if path and isinstance(path[-1],str) else ''
                form = {'firstname':'given','lastname':'family'}.get(name_field, {'unique-given':'given','unique-family':'family'}.get(source,'full')) if kind=='PERSON' else 'full'
                self.spans.append(Span("s" + str(len(self.spans)), doc, path, start, end, kind, value, source, group,
                                       kind not in {"DATE", "LOCATION", "ORGANIZATION"},form))
        if len(self.spans) > LIMITS["spans"]:
            raise PolicyError("TOO_MANY_SENSITIVE_SPANS")
        conflicting_names = conflicting_person_names(self.documents)
        for span in self.spans:
            if span.kind == 'PERSON' and self.alias_parent.get(span.value, span.value) in conflicting_names:
                span.identity_review = True
        if any(span.identity_review for span in self.spans):
            self.issues.append('CONFLICTING_PERSON_IDS_REQUIRES_REVIEW')
        def keys(node,path=()):
            if isinstance(node, dict):
                for k, v in node.items():
                    if field_kind(path+(k,)) and v is not None and not isinstance(v,str):
                        raise PolicyError("SENSITIVE_FIELDS_MUST_BE_TEXT")
                    # Known schema keys are labels, not identifier-bearing values.
                    # Their values still pass the checks above and detection below.
                    known_key = field_kind(path+(k,)) or re.sub(r'[^a-z0-9]','',k.lower()) in {'firstname','lastname','records','note','notes'}
                    if any(p.search(k) for p in PATTERNS.values()) or any(pattern.search(k) for pattern, *_ in compiled_seeds) or (not known_key and any(kind in {"PERSON", "LOCATION"} for _, _, kind in self.detector.detect(k))):
                        raise PolicyError("SENSITIVE_JSON_KEYS_UNSUPPORTED")
                    keys(v,path+(k,))
            elif isinstance(node, list):
                for index,v in enumerate(node):keys(v,path+(index,))
        for document in self.documents:
            keys(document["content"])

    @staticmethod
    def _seed(seeds, value, kind, source, priority):
        if value not in seeds and len(seeds) >= 1000:
            raise PolicyError("TOO_MANY_DISTINCT_VALUES")
        if value not in seeds or priority < seeds[value][2]:
            seeds[value] = (kind, source, priority)

    def snapshot(self):
        groups = {}
        for span in self.spans:
            group = groups.setdefault(span.group, {"id": span.group, "kind": span.kind, "values": [], "occurrences": []})
            if span.value not in group["values"]:group["values"].append(span.value)
            text = text_at(self.documents, span.document, span.path)
            group["occurrences"].append({"id": span.id, "document": span.document, "path": list(span.path),
                 "context": text[max(0,span.start-35):min(len(text),span.end+35)], "source": span.source,
                 "selected": span.selected, "identity_review": span.identity_review})
        return {"groups": list(groups.values()), "detector": self.detector.name, "learned_detector": self.detector.learned,
                "characters": sum(len(text) for _, _, text in leaf_texts(self.documents)), "issues": self.issues,
                "documents": len(self.documents), "source_sent_to_cloud": False}

    def transform(self, decisions):
        if not isinstance(decisions, list) or len(decisions) != len(self.spans):
            raise PolicyError("REVIEW_ALL_OCCURRENCES")
        by_id = {}
        group_kinds = {}
        for decision in decisions:
            if not isinstance(decision, dict) or set(decision) != {"id", "group", "selected"}:
                raise PolicyError("REVIEW_SCHEMA")
            sid, group, selected = decision["id"], decision["group"], decision["selected"]
            if not isinstance(sid,str) or sid in by_id or not isinstance(group,str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}",group) or type(selected) is not bool:
                raise PolicyError("REVIEW_SCHEMA")
            by_id[sid] = decision
        if set(by_id) != {s.id for s in self.spans}:
            raise PolicyError("UNKNOWN_REVIEW_OCCURRENCE")
        for span in self.spans:
            decision = by_id[span.id]
            if not decision["selected"]:continue
            if span.identity_review and decision['group'] == span.group:
                raise PolicyError('CONFIRM_OR_SPLIT_IDENTITY')
            group = decision["group"]
            if group in group_kinds and group_kinds[group] != span.kind:
                raise PolicyError("CANNOT_LINK_DIFFERENT_TYPES")
            group_kinds[group] = span.kind
        if any(issue in self.issues for issue in ["AMBIGUOUS_INITIAL_REQUIRES_REVIEW","AMBIGUOUS_NAME_PART_REQUIRES_REVIEW"]):
            ambiguous = [s for s in self.spans if s.source in {"ambiguous-initial","ambiguous-name-part"}]
            for span in ambiguous:
                if by_id[span.id]["selected"] and by_id[span.id]["group"] == span.group:
                    raise PolicyError("LINK_OR_KEEP_AMBIGUOUS_INITIAL" if span.source=='ambiguous-initial' else "LINK_OR_KEEP_AMBIGUOUS_NAME")
        replacements = {}
        sources = {s.value for s in self.spans}
        used = set()
        days = 7 * (1 + int(digest(self.key, self.scope, "date")[:8], 16) % 26)
        for group, kind in sorted(group_kinds.items()):
            if kind == "DATE":continue
            for attempt in range(512):
                token = digest(self.key, self.scope, group, str(attempt))
                n = int(token[:12], 16)
                if kind == "PERSON":
                    from .core import GIVEN, FAMILY
                    first,last=GIVEN[n % len(GIVEN)], FAMILY[(n // len(GIVEN)) % len(FAMILY)]
                    forms={"full":first+" "+last,"initial":first[0]+". "+last,"given":first,"family":last}
                elif kind == "EMAIL":forms={"full":"case-"+token[:16]+"@example.invalid"}
                elif kind == "ADDRESS":forms={"full":str(100+n%800)+" Example Lane, Testville, ZZ 00000"}
                else:forms={"full":kind+"TEST-"+token[:16]}
                source_in_form = any(re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", form) for source in sources for form in forms.values())
                if not set(forms.values()) & used and not source_in_form:break
            else:raise PolicyError("REPLACEMENT_SPACE_EXHAUSTED")
            replacements[group]=forms
            used.update(value for form,value in forms.items() if form in {'full','initial'})
        result=copy.deepcopy(self.documents)
        edits={}
        for span in self.spans:
            decision=by_id[span.id]
            if not decision["selected"]:continue
            if span.kind=="DATE":
                try:replacement=shifted(span.value,days)
                except (ValueError,OverflowError):raise PolicyError("UNSUPPORTED_DATE_FOR_SHIFT") from None
            else:
                form="initial" if span.source in {"unique-initial","ambiguous-initial"} else span.form
                replacement=replacements[decision["group"]].get(form,replacements[decision["group"]]["full"])
            edits.setdefault((span.document,span.path),[]).append((span.start,span.end,replacement))
        for (doc,path),changes in edits.items():
            text=text_at(self.documents,doc,path)
            for start,end,replacement in sorted(changes,reverse=True):text=text[:start]+replacement+text[end:]
            target=result[doc]
            if path:
                target=target["content"]
                for part in path[:-1]:target=target[part]
                target[path[-1]]=text
            else:target["content"]=text
        # Every edit is limited to reviewed string spans. Values, keys, arrays and order outside them are untouched.
        selected=sum(d["selected"] for d in decisions)
        report={"version":"0.2.2-alpha","status":"TRANSFORMED_REVIEW_REQUIRED","external_use":"NOT_CLEARED",
                "document_count":len(result),"source_characters":sum(len(t) for _,_,t in leaf_texts(self.documents)),
                "replacement_count":selected,"kept_detected_occurrences":len(decisions)-selected,
                "detector":self.detector.name,"source_sent_to_cloud":False,
                "input_commitment":digest(self.key,self.scope,canonical(self.documents).decode()),
                "output_commitment":digest(self.key,self.scope,canonical(result).decode()),
                "checks":{"edits_limited_to_reviewed_spans":True,"source_structure_preserved":True},
                "limits":["Detectors can miss sensitive data. Review all source and output documents.",
                          "Equal source text is grouped by default. Split occurrences for different people with the same name.",
                          "Dates, locations and organizations stay unchanged unless selected. They can identify people.",
                          "Date shifts preserve elapsed time, not holidays or policy deadlines.",
                          "No original binary documents, privacy certification or agent-outcome equivalence test."]}
        return {"documents":result,"report":report}
