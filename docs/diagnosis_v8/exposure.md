# Development exposure recorded by the V8 diagnosis

`manifests/development_exposure.json` is hash-pinned by `manifests/split.json`
(`development_exposure_sha256`), so this record lives here instead of mutating
the manifest. All patients below are development-split patients already listed
as exposed by earlier phases; this makes the V8 inspection explicit.

- **What was inspected**: gold SQL, reference answers and typed-scorer reasons
  for the 45 both-wrong-core questions and 4 asymmetric-win questions
  (`root_causes.csv`). Agent raw answers were not available (gitignored).
- **Patients**: 36 `subject_id`s in `audited_subject_ids.json`.
- **What changed because of it**: `agent/ehr_semantics.CONVENTION_GUIDANCE`
  (appended to `SEMANTIC_GUIDANCE` and to the direct-LLM prompt). The rules
  state how released FHIR fields map to question words; they contain no
  question ids, patient ids, reference values or per-question logic
  (`tests/test_ehr_semantics.py`).
- **Fairness**: the same text reaches direct, single and every team role, so
  no arm benefits unilaterally. It changes the code fingerprint; every arm in
  Phase B/C is re-run under the new fingerprint and prior numbers are not
  compared across fingerprints.
- **Status of the 300 dev questions**: exposed for the eighth time. Every
  Phase B/C number on them is exploratory; only the protected-set confirmation
  run (gate in `decision.md` §4) is confirmatory.

## Live check calibration (hand-off §3a)

- **What was inspected**: the same 49 questions and 36 patients. On live FHIR,
  `scripts/validate_convention_checks_live.py` compared the reference answers
  and `true_fhir_ids` with each patient's released resources. It ran the V8
  checks on a reference-shaped draft and on the audit's typical wrong draft.
  The script writes check names, booleans and counts only, never values.
- **What it settled**: T054 reference = cited `Specimen.type.coding[0].display`
  (11/11). T055 organisms are `hasMember` children (3/3). T043 references occur
  nowhere in the cited Encounter (4/4), confirming `label_issue`. T051/T052 tie
  sets are reproduced by mixture-expanded names at minute precision within the
  cited encounter (6/7 and 2/2; the T052 tie is restricted to the cited route).
  Some released MedicationRequests have no `validityPeriod.start`, and their
  `authoredOn` differs from the reference `starttime` by hours.
- **What changed because of it** (`agent/convention_checks.py`; question
  gates and field handling only, with no question ids or values):
  `DISTINCT_COUNT_RE` covers "count/number of distinct/unique" (gate coverage
  1/6 → 6/6). `WHEN_RE` covers "And what was the last time". The tie check
  skips dose, count and time answers and restricts route questions to the
  cited route. The prescription timestamp check skips records without
  `validityPeriod.start`. Reference-shaped false positives fell from 12 to 4.
  All 4 remaining are `admission_type_field` on T043, where the reference is
  not representable.
