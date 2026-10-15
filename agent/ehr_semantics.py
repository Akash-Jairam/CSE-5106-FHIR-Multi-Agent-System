"""Gold-free interpretation guidance shared by the competing systems.

These rules describe the released FHIR data representation and question
language, not reference answers or per-question benchmark annotations.
"""

CONVENTION_GUIDANCE = """
Data-representation conventions (how released fields map to question words):
- Coded answers are the released display string verbatim, with its original
  spelling and capitalisation; do not paraphrase, normalise or translate codes.
- "Specimen" or "specimen name" of a microbiology test is the Specimen
  resource's type display (the test's specimen reference), not the test name
  in Observation.code. "Organism(s)" are the child Observations linked to the
  test through hasMember/derivedFrom; fetch them before answering or counting.
  The time of a microbiology test is the parent test's effectiveDateTime.
- "Admission type" of a hospital visit is the Encounter priority coding, not
  Encounter.class or the encounter type list.
- First/last/earliest/latest selection by time returns every distinct value
  that shares the selected timestamp exactly: if two drugs (or routes, items)
  start at the same minute, the answer lists both. Ties at a selected extreme
  value are broken by the earliest time.
- A drug or route named in the question matches the released name/route string
  exactly (e.g. "Furosemide" does not include "Furosemide 10mg/mL"); when the
  question quotes a name, filter on equality after inspecting the actual strings.
- "How many different drugs/items" counts distinct released name strings over
  every scoped record. Search previews are bounded: enumerate the complete scoped
  set (paging or the patient-wide Medication closure) before counting; a count
  equal to a preview size is a retrieval gap, not an answer.
- A "when" answer is the full released timestamp of the selected record
  (prescription validityPeriod.start, measurement effectiveDateTime, encounter
  period boundary) at its stored precision, not a date or a related field.
- Existence questions ("has/was there any ...") are answered within the
  question's time window and scope only; an event outside the window is no.
- Per-period aggregates (monthly/daily/yearly min, max, mean) are answered as a
  plain list of the numeric values in period order, one number per period,
  without labels or text.
"""

SEMANTIC_GUIDANCE = """
Interpret the question before computing:
- Use the assumed current time as the upper bound of every search. Compare
  deidentified timestamps as wall-clock values (ignore timezone offsets).
- Observation measurement time is effectiveDateTime, not issued. A drug
  prescription starts at dispenseRequest.validityPeriod.start, not authoredOn.
- Hospital Encounters have an identifier system ending in encounter-hosp;
  ICU Encounters end in encounter-icu and link to the hospital through partOf.
  Exclude other encounter classes when selecting a hospital visit or ICU stay.
  First/last visit is ordered by period.start. For last/latest visit, select the
  latest completed encounter (period.end present and no later than the assumed
  current time). Explicit current/ongoing-stay questions use the active encounter.
  Do not confuse first/last visit
  with first/last qualifying measurement. Events may reference either an ICU
  encounter or its parent hospital; resolve partOf when matching hospital scope.
- "During the last hospital visit" limits events to that hospital encounter;
  "since" a visit/date is a time lower bound, not automatically an encounter filter.
  For years/months/days use the question's reference time, not today's real date.
  Relative "since N days/months/years ago" starts at midnight of the computed
  reference date. Calendar windows are [start, next day/month/year), bounded
  by the assumed current time. Month/year subtraction follows SQLite's
  day-of-month overflow rule rather than clipping: December 31 minus one
  month gives December 1 (November 31 rolls forward).
  Honor strict above/below versus inclusive at-least/at-most thresholds.
- Inspect actual code.coding/code.text labels and systems before selecting a
  clinical concept. MIMIC local codes are not necessarily LOINC. Different
  chart and laboratory concepts can have similar names. Do not infer an empty
  result from a failed standard-code lookup or a small starting packet.
- MedicationRequest can use medicationReference: fetch its linked Medication
  before interpreting names. Medication/Location are shared metadata, not events.
  Do not replace a prescription start with the medication metadata timestamp.
  A Medication mixture can reference component drugs through ingredient.itemReference;
  resolve those when the question names a component rather than the mixture.
  One patient-wide Medication retrieval gathers referenced products and their
  ingredient dependencies in bounded batches. Inspect the returned closure and
  completeness metadata before spending calls on individual medication IDs.
- Microbiology Observations can be linked test, organism and susceptibility
  records. Resolve their relationships and codes before counting tests or
  reporting organisms; child findings are not separate specimen collections.
- Fetch Encounters before interpreting undated Conditions and resolve their
  encounter links. Diagnoses, procedures and observations are different types;
  inspect the appropriate type when a concept is absent from the first search.
- Count actual scoped events or distinct visits according to the question;
  duplicate concept labels are not automatically duplicate events. Sort full
  scoped records before selecting the first/last. Compute extrema/means over
  all qualifying measurements, then cite the selected fields and scope records.
- For duration questions subtract the scoped period endpoints in Python in the
  requested units. For elapsed time in an explicitly current/ongoing stay use
  the question's reference time; do not invent a missing end for completed-stay length.
  For paired comparisons or differences independently compute each requested
  window/visit statistic, then apply the requested comparison/subtraction.
  A citation in an explicit paired measurement question may include scope_index
  0 or 1 for its operand in textual order; the source checker derives each
  operand's boundaries from the question. Operand membership alone does not
  establish the requested measurement rank, concept or arithmetic direction.
- Distinguish established absence (null for no measurement, false for existence,
  zero for a count) from missing retrieval, failed tools or uncertain interpretation.
  A negative answer needs a completed search of the appropriate resource type.
""" + CONVENTION_GUIDANCE