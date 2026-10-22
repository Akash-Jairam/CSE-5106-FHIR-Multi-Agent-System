"""Question-text router for the V10 finishers (docs/diagnosis_v8/design_v10.txt).

The finishers are keyed by question template, but an agent never sees a
question id or the family manifest. `classify_template` recovers the template
from the question wording alone, for the seven templates that have a finisher
(T001, T051, T052, T054, T055, T064, T073), and returns None for every other
question. tests/test_template_router.py checks it against
manifests/question_families.csv: on all 2,931 benchmark questions it must
return the manifest template for every question of those seven templates and
None for every other question. No drug, specimen or organism vocabulary is
used, only the template's own wording.
"""
import re

FIRST_LAST = r"(?:first|last|earliest|latest|initial|final|most recent)"
_FIRST_LAST_RE = re.compile(rf"\b{FIRST_LAST}\b", re.I)
_WHEN_RE = re.compile(r"\bwhen\b|\bwhat time\b|\bwhat (?:was|is) the (?:\w+ )?time\b|^\s*(?:first|last) time\b(?!,)"
                      r"|\b(?:test|examination|prescription) time\b", re.I)
_DRUG_RE = re.compile(r"\b(?:drugs?|medications?|medicines?|meds?|prescriptions?)\b", re.I)
_COUNT_RE = re.compile(r"\b(?:how many|count|number)\b", re.I)
_DISTINCT_RE = re.compile(r"\b(?:distinct|unique|different)\b", re.I)
_DOSE_RE = re.compile(r"\b(?:dose|doses|dosage|amount|volume|how much)\b", re.I)
# T064 asks about any microbiology test; T059 names a specimen right before "microbiology test"
_ANY_MICRO_RE = re.compile(r"\b(?:the|a|an|any|his|her|their|first|last|earliest|latest|given)\s+microbiolog\w*\b", re.I)
_MICRO_RE = re.compile(r"\bmicrobiolog\w*\b", re.I)
_SPECIMEN_RE = re.compile(r"\bspecimens?\b", re.I)
_ORGANISM_RE = re.compile(r"\borganisms?(?:'s)?\b", re.I)
_EXISTS_RE = re.compile(r"^\s*(?:has|have|had|did|was|were|is|are|does|do)\b", re.I)


def _route(question):
    return bool(re.search(r"\broute\b|\b(?:via|through|using)\b(?!\s+(?:the|their|his|her)\s+(?:first|last)\b)", question, re.I))


def classify_template(question):
    """The finisher template of `question` (bare question text), or None."""
    q = question.strip()
    if not q or _EXISTS_RE.match(q):
        return None
    if _ORGANISM_RE.search(q):
        return "T055" if _FIRST_LAST_RE.search(q) else None
    if _WHEN_RE.search(q):
        if _MICRO_RE.search(q):
            return "T064" if _FIRST_LAST_RE.search(q) and _ANY_MICRO_RE.search(q) else None
        if _DRUG_RE.search(q) and _route(q):
            return "T073" if _FIRST_LAST_RE.search(q) else None
        return None
    if _DRUG_RE.search(q):
        if _COUNT_RE.search(q) and _DISTINCT_RE.search(q):
            return "T001"
        if _DOSE_RE.search(q) or _COUNT_RE.search(q):
            return None
        if _FIRST_LAST_RE.search(q):
            return "T052" if _route(q) else "T051"
        return None
    if _SPECIMEN_RE.search(q) or re.search(r"\bculture\b", q, re.I):
        return "T054" if _FIRST_LAST_RE.search(q) else None
    return None
