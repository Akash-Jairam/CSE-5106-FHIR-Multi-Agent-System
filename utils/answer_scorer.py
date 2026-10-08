"""Deterministic typed answer scorer.

The reference answer (an EHRSQL result such as "[[1]]", "[['M']]",
"[['2133-12-31 02:00:00']]", "[]", or "[[None]]") fixes the answer type:

  boolean    yes/no questions ("Has ...", "Is ..."), reference [[0]] or [[1]]
  timestamp  compared at the precision the question asks for: day if it asks
             for a date/day, otherwise minute; time zone offsets are ignored
             because references are local wall-clock times
  number     counts must match exactly; other values within
             max(ABS_TOL, REL_TOL * |reference|); stated units are converted
             to the unit the question implies and incompatible units are wrong
  category   normalized string match (e.g. "M" matches "male")
  set        several reference rows: every reference value must be present;
             a structured (list) answer must also have no extra values
  empty      reference "[]" or "[[None]]": correct only if the agent gives no
             value (abstains, says none/no, or answers 0 to a numeric question)

Abstention is wrong whenever the reference is non-empty. Answers can be free
text (the current agents) or structured values (bool, number, str, list, or
None), which skip text extraction.
"""
import ast
import math
import re
from datetime import datetime

REL_TOL = 0.01
ABS_TOL = 0.01

DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?")
SLASH_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}/\d{4}\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
NUMBER_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|-?\.\d+)(?![\d.]*\d)\s*(%|°\s?[CF]|[A-Za-z][A-Za-z/]*)?")

ABSTAIN_RE = re.compile(
    r"\b(cannot|can ?not|can't|could ?not|couldn't|unable|not enough|insufficient|unknown|"
    r"not (?:be )?determin\w*|undetermin\w*|not available|unavailable|n/a|no answer)\b", re.I)
NEGATIVE_RE = re.compile(
    r"^\W*(?:no|none|never|false|zero)\b|"
    r"\b(?:has not|have not|hasn't|haven't|had not|hadn't|did not|didn't|was not|were not|wasn't|weren't|"
    r"is not|isn't|not been|never|not (?:been )?(?:found|recorded|documented|observed))\b|"
    r"\bno\s+(?:[\w()/,-]+\s+){0,6}?(?:found|recorded|observed|prescribed|documented|record|records|"
    r"result|results|data|evidence|visit|visits|admission|admissions|measurement|measurements)\b", re.I)
AFFIRMATIVE_RE = re.compile(r"^\W*(?:yes|true)\b|\b(?:has been|have been|had|was|were|did|received|underwent)\b", re.I)
# A bare "No." / "Yes," answer; "No heparin was found" is handled as a negative statement
YES_NO_RE = re.compile(r"^\W*(yes|no|true|false)\b(?=\s*(?:[.,;:!-]|$))", re.I)
DECREASE_RE = re.compile(r"\b(decrease[ds]?|dropp?(?:ed|s)?|fell|fall|lower|down|reduc\w*|declin\w*|less)\b", re.I)
FINAL_ANSWER_RE = re.compile(r"^\s*the final answer is\s*:?\s*", re.I)

# unit -> (dimension, factor to the dimension's base unit)
UNITS = {
    "min": ("time", 1 / 60), "mins": ("time", 1 / 60), "minute": ("time", 1 / 60), "minutes": ("time", 1 / 60),
    "h": ("time", 1), "hr": ("time", 1), "hrs": ("time", 1), "hour": ("time", 1), "hours": ("time", 1),
    "d": ("time", 24), "day": ("time", 24), "days": ("time", 24),
    "week": ("time", 168), "weeks": ("time", 168),
    "kg": ("mass", 1), "kgs": ("mass", 1), "kilogram": ("mass", 1), "kilograms": ("mass", 1),
    "g": ("mass", 1e-3), "gram": ("mass", 1e-3), "grams": ("mass", 1e-3),
    "lb": ("mass", 0.45359237), "lbs": ("mass", 0.45359237), "pound": ("mass", 0.45359237),
    "pounds": ("mass", 0.45359237),
    "ml": ("volume", 1), "cc": ("volume", 1), "l": ("volume", 1000), "liter": ("volume", 1000),
    "liters": ("volume", 1000),
    "cm": ("length", 1), "mm": ("length", 0.1), "m": ("length", 100), "in": ("length", 2.54),
    "inch": ("length", 2.54), "inches": ("length", 2.54),
}
GENDER = {"m": "male", "f": "female"}


# ---------------------------------------------------------------------------
# Reference parsing and answer types
# ---------------------------------------------------------------------------

def parse_reference(true_answer):
    """Parse an EHRSQL reference into a list of row values ([] if empty)."""
    if isinstance(true_answer, list):
        rows = true_answer
    else:
        text = str(true_answer).strip()
        if text in ("", "nan", "[]"):
            return []
        # Multi-row references are numpy reprs: "[['a']\n ['b']]" or "[[67.]\n [42.]]"
        text = re.sub(r"\]\s*\n\s*\[", "], [", text)
        rows = ast.literal_eval(text)
    return [row[0] if isinstance(row, (list, tuple)) else row for row in rows]


def _is_timestamp(value):
    return isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", value.strip()))


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def value_type(values, template=""):
    """Type of the non-null reference values."""
    if template.startswith(("Has_verb", "Is the")):
        return "boolean"
    if all(_is_timestamp(v) for v in values):
        return "timestamp"
    if all(_as_number(v) is not None for v in values):
        return "number"
    return "category"


def answer_type(true_answer, template=""):
    values = parse_reference(true_answer)
    if not values or all(v is None for v in values):
        return "empty"
    kind = value_type(values, template)
    return "set" if len(values) > 1 else kind


def _is_count(question, template):
    text = f"{template} {question}".lower()
    if "have passed" in text or "[unit_count]" in template:
        return False
    return template.startswith("Count") or bool(re.search(r"\b(how many times|number of|count)\b", text))


def expected_unit(question, template):
    """Unit a numeric answer should be in, if the question implies one."""
    text = question.lower()
    if "[unit_count]" in template or "have passed" in template:
        match = re.search(r"\b(minutes?|hours?|days?|weeks?)\b", text)
        return match.group(1) if match else None
    if "length of" in template and "stay" in template:
        return "days"
    if "weight" in template:
        return "kg"
    if "height" in template:
        return "cm"
    if "output" in template and ("volume" in template or "total" in template):
        return "ml"
    return None


def timestamp_precision(question):
    text = question.lower()
    if re.search(r"\b(date|day)\b", text) and not re.search(r"\b(time|hour|when)\b", text):
        return "day"
    return "minute"


# ---------------------------------------------------------------------------
# Extraction from free-text answers
# ---------------------------------------------------------------------------

def _clean(answer):
    return FINAL_ANSWER_RE.sub("", str(answer)).strip()


def _parse_datetime(text):
    text = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", text.strip()).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt), fmt != "%Y-%m-%d"
        except ValueError:
            continue
    return None, False


def extract_timestamps(text, question=""):
    """(datetime, has_time) pairs in order of appearance, skipping ones quoted from the question."""
    echoed = {_parse_datetime(m.group(0))[0] for m in DATETIME_RE.finditer(question)}
    found = [_parse_datetime(m.group(0)) for m in DATETIME_RE.finditer(text)]
    return [(dt, has_time) for dt, has_time in found if dt and dt not in echoed]


def _echoed_numbers(question):
    """Numbers in the question that an answer is likely to repeat as context:
    thresholds with decimals, patient IDs, and (shifted) years."""
    echoed = set()
    for n in re.findall(r"\d+(?:\.\d+)?", DATETIME_RE.sub(" ", question)):
        value = float(n)
        if "." in n or len(n) >= 5 or (len(n) == 4 and 2100 <= value <= 2220):
            echoed.add(value)
    return echoed


def extract_numbers(text, question=""):
    """(value, unit) pairs, skipping dates, IDs, and values echoed from the question."""
    stripped = UUID_RE.sub(" ", text)
    stripped = DATETIME_RE.sub(" ", stripped)
    stripped = SLASH_DATE_RE.sub(" ", stripped)
    echoed = _echoed_numbers(question)
    numbers = []
    for match in NUMBER_RE.finditer(stripped):
        value = float(match.group(1).replace(",", ""))
        if value in echoed:
            continue
        unit = (match.group(2) or "").lower().rstrip(".")
        numbers.append((value, unit))
    return numbers


def _normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w%.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def _contains(text, value):
    norm_value = _normalize(value)
    if not norm_value:
        return False
    candidates = {norm_value, GENDER.get(norm_value, norm_value)}
    norm_text = f" {_normalize(text)} "
    return any(f" {c} " in norm_text for c in candidates)


def is_abstention(answer):
    if answer is None:
        return True
    if isinstance(answer, (list, tuple)) and not answer:
        return True
    text = _clean(answer) if isinstance(answer, str) else ""
    return isinstance(answer, str) and (not text or bool(ABSTAIN_RE.search(text)))


def is_negative(text):
    return bool(NEGATIVE_RE.search(text))


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------

def numbers_match(agent, reference, exact=False):
    if exact:
        return math.isclose(agent, reference, abs_tol=1e-9)
    return abs(agent - reference) <= max(ABS_TOL, REL_TOL * abs(reference))


def _convert(value, unit, target):
    """Convert `value` in `unit` to `target`; None if the units are incompatible."""
    if not unit or not target or unit not in UNITS:
        return value
    dim, factor = UNITS[unit]
    target_dim, target_factor = UNITS[target]
    if dim != target_dim:
        return None
    return value * factor / target_factor


def timestamps_match(agent, reference, precision):
    agent_dt, has_time = agent
    ref_dt, _ = _parse_datetime(reference)
    if precision == "day":
        return agent_dt.date() == ref_dt.date()
    return has_time and agent_dt.replace(second=0, microsecond=0) == ref_dt.replace(second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _result(correct, kind, reason, parsed=None):
    return {"typed_correct": int(correct), "answer_type": kind, "typed_reason": reason, "typed_parsed": parsed}


def _score_boolean(answer, reference, text):
    ref = bool(_as_number(reference))
    if isinstance(answer, bool) or (not isinstance(answer, str) and _as_number(answer) is not None):
        agent = bool(answer if isinstance(answer, bool) else _as_number(answer))
    else:
        lead = YES_NO_RE.match(text)
        if lead:
            agent = lead.group(1).lower() in ("yes", "true")
        elif ABSTAIN_RE.search(text):
            return _result(0, "boolean", "abstained")
        elif is_negative(text):
            agent = False
        elif AFFIRMATIVE_RE.search(text):
            agent = True
        else:
            return _result(0, "boolean", "unparsed")
    return _result(agent == ref, "boolean", "match" if agent == ref else "mismatch", agent)


def _score_number(answer, reference, text, question, template):
    ref = _as_number(reference)
    exact = _is_count(question, template) and float(ref).is_integer()
    if not isinstance(answer, str):
        agent = _as_number(answer)
        if agent is None:
            return _result(0, "number", "unparsed")
        return _result(numbers_match(agent, ref, exact), "number", "match" if numbers_match(agent, ref, exact) else "mismatch", agent)
    numbers = extract_numbers(text, question)
    if not numbers:
        if ref == 0 and is_negative(text) and not ABSTAIN_RE.search(text):
            return _result(1, "number", "negative statement for 0", 0)
        return _result(0, "number", "abstained" if ABSTAIN_RE.search(text) else "unparsed")
    value, unit = numbers[0]
    target = expected_unit(question, template)
    converted = _convert(value, unit, target)
    if converted is None:
        return _result(0, "number", f"unit mismatch ({unit} vs {target})", value)
    if numbers_match(converted, ref, exact):
        return _result(1, "number", "match", converted)
    # "dropped by 38" for a change of -38
    if ref < 0 and numbers_match(-converted, ref, exact) and DECREASE_RE.search(text):
        return _result(1, "number", "match (decrease stated in words)", -converted)
    return _result(0, "number", "mismatch", converted)


def _score_timestamp(answer, reference, text, question):
    precision = timestamp_precision(question)
    found = extract_timestamps(text if isinstance(answer, str) else str(answer), question)
    if not found:
        return _result(0, "timestamp", "abstained" if ABSTAIN_RE.search(text) or is_negative(text) else "unparsed")
    ok = timestamps_match(found[0], reference, precision)
    reason = "match" if ok else ("missing time" if precision == "minute" and not found[0][1] else "mismatch")
    return _result(ok, "timestamp", f"{reason} ({precision} precision)", found[0][0].isoformat())


def _score_category(answer, reference, text):
    if not isinstance(answer, str):
        ok = _normalize(answer) in {_normalize(reference), GENDER.get(_normalize(reference))}
    else:
        ok = _contains(text, reference)
    if not ok and ABSTAIN_RE.search(text):
        return _result(0, "category", "abstained")
    return _result(ok, "category", "match" if ok else "mismatch")


def _score_set(answer, values, text, question, template):
    kind = value_type(values, template)
    if isinstance(answer, (list, tuple)):
        if kind == "number":
            agent = sorted(_as_number(a) for a in answer)
            ref = sorted(_as_number(v) for v in values)
            ok = len(agent) == len(ref) and all(a is not None and numbers_match(a, r) for a, r in zip(agent, ref))
        else:
            ok = {_normalize(a) for a in answer} == {_normalize(v) for v in values}
        return _result(ok, "set", "match" if ok else "mismatch", list(answer))
    if kind == "number":
        found = [v for v, _ in extract_numbers(text, question)]
        missing = [v for v in values if not any(numbers_match(f, _as_number(v)) for f in found)]
    elif kind == "timestamp":
        found = extract_timestamps(text, question)
        precision = timestamp_precision(question)
        missing = [v for v in values if not any(timestamps_match(f, v, precision) for f in found)]
    else:
        missing = [v for v in values if not _contains(text, v)]
    if missing and ABSTAIN_RE.search(text):
        return _result(0, "set", "abstained")
    reason = "all reference values present" if not missing else f"missing {len(missing)} of {len(values)}"
    return _result(not missing, "set", reason)


def _score_empty(answer, values, text, question, template):
    """Reference has no value: correct only if the agent gives none."""
    kind = value_type([v for v in values if v is not None], template) if any(v is not None for v in values) else None
    if is_abstention(answer):
        return _result(1, "empty", "abstained")
    if not isinstance(answer, str):
        ok = _as_number(answer) == 0
        return _result(ok, "empty", "zero" if ok else "gave a value", answer)
    if template.startswith(("Has_verb", "Is the")):
        # A yes/no answer to a question whose reference is NULL is still an answer
        if YES_NO_RE.match(text):
            return _result(0, "empty", "gave yes/no")
    if extract_timestamps(text, question) and not is_negative(text):
        return _result(0, "empty", "gave a timestamp")
    numbers = [v for v, _ in extract_numbers(text, question)]
    if numbers and any(n != 0 for n in numbers) and not is_negative(text):
        return _result(0, "empty", "gave a value", numbers[0])
    if is_negative(text) or (numbers and all(n == 0 for n in numbers)):
        return _result(1, "empty", "said none / zero")
    return _result(0, "empty", "gave a value")


def score_answer(answer, true_answer, question="", template=""):
    """Score one answer against its reference. Returns a dict with
    `typed_correct` (0/1), `answer_type`, `typed_reason`, and `typed_parsed`."""
    values = parse_reference(true_answer)
    text = _clean(answer) if isinstance(answer, str) else ""
    if not values or all(v is None for v in values):
        return _score_empty(answer, values, text, question, template)
    if is_abstention(answer) and (not isinstance(answer, str) or not text or
                                   not _has_value(answer, values, text, question, template)):
        kind = "set" if len(values) > 1 else value_type(values, template)
        return _result(0, kind, "abstained (reference is non-empty)")
    if len(values) > 1:
        return _score_set(answer, values, text, question, template)
    kind = value_type(values, template)
    if kind == "boolean":
        return _score_boolean(answer, values[0], text)
    if kind == "number":
        return _score_number(answer, values[0], text, question, template)
    if kind == "timestamp":
        return _score_timestamp(answer, values[0], text, question)
    return _score_category(answer, values[0], text)


def _has_value(answer, values, text, question, template):
    """Whether a hedged free-text answer still commits to a value of the expected type."""
    kind = value_type(values, template)
    if kind == "boolean":
        return bool(YES_NO_RE.match(text))
    if kind == "number":
        return bool(extract_numbers(text, question))
    if kind == "timestamp":
        return bool(extract_timestamps(text, question))
    return any(_contains(text, v) for v in values)
