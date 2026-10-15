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
