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
  set        several reference rows: values are matched without regard to
             order; numeric/timestamp lists must have no missing or extra values
  empty      reference "[]" or "[[None]]": correct only if the agent gives no
             value (abstains, says none/no, or answers 0 to a numeric question)

Abstention is wrong whenever the reference is non-empty. Answers can be free
text (the current agents) or structured values (bool, number, str, list, or
None), which skip text extraction.
"""
import ast
import json
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
    r"^\W*(?:no|none|never|false|zero)\b|\bnone of\b|"
    r"\b(?:has not|have not|hasn't|haven't|had not|hadn't|did not|didn't|was not|were not|wasn't|weren't|"
    r"is not|isn't|not been|never|not (?:been )?(?:found|recorded|documented|observed))\b|"
    r"\bno\s+(?:[\w()/,-]+\s+){0,6}?(?:found|recorded|observed|prescribed|documented|record|records|"
    r"result|results|data|evidence|visit|visits|admission|admissions|measurement|measurements)\b", re.I)
AFFIRMATIVE_RE = re.compile(r"^\W*(?:yes|true)\b|\b(?:has been|have been|had|was|were|did|received|underwent)\b", re.I)
# A bare "No." / "Yes," answer; "No heparin was found" is handled as a negative statement
YES_NO_RE = re.compile(r"^\W*(yes|no|true|false)\b(?=\s*(?:[.,;:!-]|$))", re.I)
# A verdict that closes the answer: "... **Answer:** No." or a last line that is just "Yes."
CLOSING_YES_NO_RE = re.compile(r"(?:^|\banswer\b)[\W_]*(yes|no|true|false)[\W_]*\Z", re.I | re.M)
DECREASE_RE = re.compile(r"\b(decrease[ds]?|dropp?(?:ed|s)?|fell|fall|lower|down|reduc\w*|declin\w*|less)\b", re.I)
FINAL_ANSWER_RE = re.compile(r"^\s*the final answer is\s*:?\s*", re.I)

# unit -> (dimension, factor to the dimension's base unit)
UNITS = {
    "s": ("time", 1 / 3600), "sec": ("time", 1 / 3600), "second": ("time", 1 / 3600), "seconds": ("time", 1 / 3600),
    "min": ("time", 1 / 60), "mins": ("time", 1 / 60), "minute": ("time", 1 / 60), "minutes": ("time", 1 / 60),
    "h": ("time", 1), "hr": ("time", 1), "hrs": ("time", 1), "hour": ("time", 1), "hours": ("time", 1),
    "d": ("time", 24), "day": ("time", 24), "days": ("time", 24),
    "week": ("time", 168), "weeks": ("time", 168),
    "kg": ("mass", 1), "kgs": ("mass", 1), "kilogram": ("mass", 1), "kilograms": ("mass", 1),
    "g": ("mass", 1e-3), "gram": ("mass", 1e-3), "grams": ("mass", 1e-3),
    "mg": ("mass", 1e-6), "mcg": ("mass", 1e-9), "ug": ("mass", 1e-9),
    "lb": ("mass", 0.45359237), "lbs": ("mass", 0.45359237), "pound": ("mass", 0.45359237),
    "pounds": ("mass", 0.45359237),
    "ml": ("volume", 1), "cc": ("volume", 1), "l": ("volume", 1000), "liter": ("volume", 1000),
    "liters": ("volume", 1000),
    "cm": ("length", 1), "mm": ("length", 0.1), "m": ("length", 100), "in": ("length", 2.54),
    "inch": ("length", 2.54), "inches": ("length", 2.54),
}
GENDER = {"m": "male", "f": "female"}
COUNT_LABELS = {"time", "times", "record", "records", "admission", "admissions", "visit", "visits",
                "encounter", "encounters", "drug", "drugs", "medication", "medications", "procedure", "procedures",
                "measurement", "measurements", "test", "tests", "prescription", "prescriptions", "hospital", "icu"}
CONTEXT_WORDS = {"and", "or", "at", "on", "for", "from", "to", "with", "because", "which", "as",
                 "was", "were", "is", "are", "based", "recorded", "measured"}
CATEGORY_PREFIX_RE = re.compile(
    r"^(?:(?:the )?(?:final )?answer (?:is|was)|"
    r"(?:the )?patient (?:is|was)|"
    r"(?:the )?(?:patient(?: s)? )?(?:sex|gender|marital status|admission type|care ?unit|"
    r"drug|medication|specimen|organism(?: name)?) (?:is|was))\s+")
EMPTY_STATEMENT_RE = re.compile(
    r"(?:no|none|nothing|never|no answer)|"
    r"no (?:(?:matching|qualifying|relevant|available|recorded|documented|such|hospital|icu|lab) )*"
    r"(?:records?|data|results?|measurements?|values?|visits?|admissions?|encounters?|medications?|drugs?|"
    r"procedures?|diagnoses|observations?|organisms?)"
    r"(?: (?:was|were|is|are|has been|have been))?(?: (?:found|available|recorded|documented|observed|prescribed))?")


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
    try:
        number = float(value) if isinstance(value, (int, float)) else float(str(value).replace(",", ""))
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
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
    return template.lower().startswith("count") or bool(re.search(r"\b(how many|number of|count (?:the|of))\b", text))


def _expects_number(question, template):
    if template.startswith(("Has_verb", "Is the")):
        return False
    text = f"{template} {question}".lower()
    return _is_count(question, template) or bool(re.search(
        r"\b(value|weight|height|dose|volume|amount|length of|how many|change in)\b", text))


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
    if re.search(r"\bseconds?\b", text):
        return "second"
    if re.search(r"\b(dates?|days?)\b", text) and not re.search(r"\b(times?|hours?|when)\b", text):
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
    if DATETIME_RE.fullmatch(text.strip()):
        dt, has_time = _parse_datetime(text)
        return [(dt, has_time)] if dt else []
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
    echoed = set() if NUMBER_RE.fullmatch(text.strip()) else _echoed_numbers(question)
    numbers = []
    for match in NUMBER_RE.finditer(stripped):
        value = float(match.group(1).replace(",", ""))
        if value in echoed:
            continue
        unit = (match.group(2) or "").lower().rstrip(".")
        if unit in CONTEXT_WORDS:
            unit = ""
        numbers.append((value, unit))
    return numbers


def _normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w%.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def _category_value(value):
    normalized = CATEGORY_PREFIX_RE.sub("", _normalize(value))
    return GENDER.get(normalized, normalized)


def is_abstention(answer):
    if answer is None:
        return True
    if isinstance(answer, float) and math.isnan(answer):
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
        return agent == reference
    return abs(agent - reference) <= max(ABS_TOL, REL_TOL * abs(reference))


def _convert(value, unit, target):
    """Convert `value` in `unit` to `target`; None if the units are incompatible."""
    if not unit or not target:
        return value
    if unit not in UNITS or target not in UNITS:
        return None
    dim, factor = UNITS[unit]
    target_dim, target_factor = UNITS[target]
    if dim != target_dim:
        return None
    return value * factor / target_factor


def timestamps_match(agent, reference, precision):
    agent_dt, has_time = agent
    ref_dt, _ = _parse_datetime(reference)
    if agent_dt is None or ref_dt is None:
        return False
    if precision == "day":
        return agent_dt.date() == ref_dt.date()
    if precision == "second":
        return has_time and agent_dt.replace(microsecond=0) == ref_dt.replace(microsecond=0)
    return has_time and agent_dt.replace(second=0, microsecond=0) == ref_dt.replace(second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _result(correct, kind, reason, parsed=None):
    return {"typed_correct": int(correct), "answer_type": kind, "typed_reason": reason, "typed_parsed": parsed}


def _score_boolean(answer, reference, text):
    ref = bool(_as_number(reference))
    if isinstance(answer, bool) or (not isinstance(answer, str) and _as_number(answer) is not None):
        value = answer if isinstance(answer, bool) else _as_number(answer)
        if value not in (0, 1):
            return _result(0, "boolean", "unparsed")
        agent = bool(value)
    else:
        lead = YES_NO_RE.match(text) or CLOSING_YES_NO_RE.search(text)
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
        if (ref == 0 and _is_count(question, template) and
                (EMPTY_STATEMENT_RE.fullmatch(_normalize(text)) or _normalize(text) == "zero")):
            return _result(1, "number", "negative statement for 0", 0)
        return _result(0, "number", "abstained" if ABSTAIN_RE.search(text) else "unparsed")
    if len(numbers) != 1:
        return _result(0, "number", "unparsed (multiple numeric values)")
    value, unit = numbers[0]
    if _is_count(question, template) and unit and unit not in COUNT_LABELS:
        return _result(0, "number", f"unit mismatch ({unit} on a count)", value)
    target = expected_unit(question, template)
    if unit and not target and not _is_count(question, template):
        return _result(0, "number", f"unparsed (reference unit unknown for {unit})", value)
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
    if len(found) != 1:
        return _result(0, "timestamp", "unparsed (multiple timestamps)")
    ok = timestamps_match(found[0], reference, precision)
    reason = "match" if ok else ("missing time" if precision == "minute" and not found[0][1] else "mismatch")
    return _result(ok, "timestamp", f"{reason} ({precision} precision)", found[0][0].isoformat())


def _score_category(answer, reference, text):
    ok = _category_value(answer) == _category_value(reference)
    if not ok and ABSTAIN_RE.search(text):
        return _result(0, "category", "abstained")
    return _result(ok, "category", "match" if ok else "mismatch")


def _score_set(answer, values, text, question, template):
    kind = value_type(values, template)
    if isinstance(answer, (list, tuple)):
        if kind == "number":
            agent = [_as_number(a) for a in answer]
            if any(a is None for a in agent):
                return _result(0, "set", "unparsed")
            agent = [(a, "") for a in agent]
        elif kind == "timestamp":
            agent = [_parse_datetime(str(a)) for a in answer]
        else:
            ok = {_category_value(a) for a in answer} == {_category_value(v) for v in values}
            return _result(ok, "set", "match" if ok else "mismatch", list(answer))
    elif kind == "number":
        agent = extract_numbers(text, question)
    elif kind == "timestamp":
        agent = extract_timestamps(text, question)
    else:
        # A delimited list is unambiguous; prose membership is not an exact set.
        items = [item.strip() for item in re.split(r"[,;\n]|\s+and\s+", text, flags=re.I)]
        if len(items) < 2 or any(not item for item in items):
            return _result(0, "set", "unparsed")
        ok = {_category_value(item) for item in items} == {_category_value(v) for v in values}
        return _result(ok, "set", "match" if ok else "mismatch")

    if kind == "number":
        if _is_count(question, template) and any(unit and unit not in COUNT_LABELS for _, unit in agent):
            return _result(0, "set", "unit mismatch (unsupported unit on a count)")
        target = expected_unit(question, template)
        if not target and not _is_count(question, template) and any(unit for _, unit in agent):
            return _result(0, "set", "unparsed (reference unit unknown)")
        converted = [_convert(v, unit, target) for v, unit in agent]
        if any(v is None for v in converted):
            return _result(0, "set", "unit mismatch")
        reference = sorted(_as_number(v) for v in values)
        exact = _is_count(question, template)
        ok = len(converted) == len(reference) and all(
            numbers_match(a, r, exact) for a, r in zip(sorted(converted), reference))
    else:
        precision = timestamp_precision(question)
        reference = sorted(values, key=lambda v: _parse_datetime(v)[0])
        if any(dt is None for dt, _ in agent):
            return _result(0, "set", "unparsed")
        ok = len(agent) == len(reference) and all(
            timestamps_match(a, r, precision) for a, r in zip(sorted(agent), reference))
    return _result(ok, "set", "match" if ok else "mismatch")


def _score_empty(answer, values, text, question, template):
    """Reference has no value: correct only if the agent gives none."""
    if not isinstance(answer, str):
        if is_abstention(answer):
            return _result(1, "empty", "abstained")
        ok = _expects_number(question, template) and _as_number(answer) == 0
        return _result(ok, "empty", "zero" if ok else "gave a value", answer)
    if template.startswith(("Has_verb", "Is the")):
        # A yes/no answer to a question whose reference is NULL is still an answer
        if YES_NO_RE.match(text):
            return _result(0, "empty", "gave yes/no")
    if extract_timestamps(text, question):
        return _result(0, "empty", "gave a timestamp")
    numbers = [v for v, _ in extract_numbers(text, question)]
    if numbers and any(n != 0 for n in numbers):
        return _result(0, "empty", "gave a value", numbers[0])
    if is_abstention(answer):
        if re.search(r";|[.!?]\s+\S|\b(?:but|however)\b", text, re.I):
            return _result(0, "empty", "unparsed (mixed abstention and claims)")
        return _result(1, "empty", "abstained")
    if (EMPTY_STATEMENT_RE.fullmatch(_normalize(text)) or
            (_expects_number(question, template) and
             (_normalize(text) == "zero" or (numbers and all(n == 0 for n in numbers))))):
        return _result(1, "empty", "said none / zero")
    return _result(0, "empty", "gave a value")


def score_answer(answer, true_answer, question="", template=""):
    """Score one answer against its reference. Returns a dict with
    `typed_correct` (0/1), `answer_type`, `typed_reason`, and `typed_parsed`."""
    question = question if isinstance(question, str) else ""
    template = template if isinstance(template, str) else ""
    # CSV output serializes structured lists; restore them before comparing sets.
    if isinstance(answer, str) and answer.strip().startswith("["):
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(answer)
            except (ValueError, SyntaxError):
                parsed = answer
        if isinstance(parsed, (list, tuple)):
            answer = parsed
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
        return bool(YES_NO_RE.match(text) or CLOSING_YES_NO_RE.search(text))
    if kind == "number":
        return bool(extract_numbers(text, question))
    if kind == "timestamp":
        return bool(extract_timestamps(text, question))
    return any(_category_value(text) == _category_value(v) for v in values)
