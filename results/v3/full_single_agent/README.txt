Matched completed cohort N=2929: Single Agent and V3

Use summary.json and full_single_analysis.txt for the matched analysis. The main
comparison uses exactly the same 2929 question/patient pairs in both arms. This
posthoc shared-availability cohort differs from the separate complete V3 N=2931
study. It does not declare the entire baseline/full benchmark complete. Missing
questions are never imputed incorrect.

The unchanged typed scorer, aggregate definitions and patient-cluster bootstrap
are used: 10000 resamples, seed5106, 95% percentile intervals. Paired differences
in this directory are Single Agent minus V3 (left minus right). Accuracy values
are fractions; multiply differences by100 to express percentage points.

per_question_measurements_and_scores.{jsonl,csv}: Single Agent numeric records,
correctness/stage flags and availability. matched_v3_measurements_and_scores.*:
V3 records restricted to the exact same IDs. *_summary.json, matched_arm_summaries
and metric_summary.csv report total/mean/median/p95/sampleSD/min/max/missingness.
by_* tables contain both arms; paired_by_* tables contain matched differences.
phase_usage preserves available explicit phase labels only; absent phase metadata
is unavailable, not zero work. call_ordinal_usage is usage by call index, not an
inferred phase or agent role. mechanisms and draft_final_scope_reconciliation
explain the different denominator for draft-to-final transition counts.

Empty/no-value flags are syntactic. They can include correct empty-reference
answers; consult correctness_empty_flag_crosstab.json before interpreting them.
Evidence validation/precision/recall were not newly evaluated and remain unknown
where unavailable. All17 frozen measurement fields are retained with missingness.

Costs are estimates at the frozen official OpenAI schedule, adopted by the user
as the Azure proxy: input0.10,cached0.01,output0.50 USD per million tokens. Actual
billing is unknown. Cached tokens are a subset of prompt tokens. Aggregate
usage_available flags do not establish cached-field presence: the completed
Single Agent cached-field provenance checkpoint covers891 accepted questions;
a full baseline cached-field audit was not performed. Repeated_prompt_tokens is
the existing prompt-overlap estimate, not a billed cache measurement.

Exposure labels in the frozen source split are historical. The later300-question
development run includes58 patients all labeled held_out_patient_descriptive in
that original split. Original35 dev/pilot plus later58 cover93/94 matched patients.
exploratory_development_exposure_* documents those overlaps without changing
frozen labels or claiming the remaining group is untouched.

Portable detailed results are in raw/patient_chunks/*.jsonl.xz, indexed by
raw/question_index.jsonl and described in raw/ARCHIVE_MANIFEST.json. Every line
has exactly two fields: canonical (all original parsed result fields) and
audit_json (the exact original decompressed UTF8 audit payload as a JSON string).
Together they retain answers,evidence,usage,tool calls,traces,retrieved resources
and revision messages without duplicating rich and raw audit archives. Files are
ordered by patient then question ID; source_row_index recovers original order.
Original canonical whitespace and original gzip-container bytes/headers are not
recreated. Original canonical,compressed-audit,and uncompressed-payload hashes
remain available for provenance. Parsed-row and payload hashes are checked on
full readback by the packager. The data can be read using Python standard-library
lzma.open, json.loads, and hashlib; no project code or provider setup is needed.

To restore large fields after reading an envelope, parse audit_json, inspect
canonical.audit_artifacts.fields, and copy each declared payload field back to
the canonical object. The special field revision.messages goes inside revision;
trace and retrieved_fhir_resources go at the top level. The payload _audit_meta
records question ownership and whether audit_artifacts originally existed.
The original descriptor itself remains in the preserved canonical envelope.

Operational status,excluded-attempt records and first-clean replay provenance
are under internal/. They never contribute to accepted primary measurements.
No provider calls,model inference,source/runtime changes,or completion-gate
changes were made while preparing this package. Inference implementation source files
are not copied into this results directory; identities and hashes are retained.
The offline analysis extractor is preserved separately in reproduce/sourceanalysis/.
