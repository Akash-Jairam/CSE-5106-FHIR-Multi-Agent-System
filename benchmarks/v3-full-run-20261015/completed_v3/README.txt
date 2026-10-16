COMPLETE V3 RESULTS; BASELINES DEFERRED BY USER

Start with REPORT.txt. This directory contains all 2,931 accepted per-question
CSV/JSONL records, overall and subgroup statistics, phase/revision summaries,
frozen pricing, selection ledger, provenance and independent verification.

The frozen summary.json and original receipt retain V3_COMPLETE_CONTROLS_PENDING
and benchmark_complete=false because the original matched three-arm benchmark
was not completed. V3 itself is complete. On 16 October 2026 the user asked to
finalize and push V3, stop the task, and wait for instructions before baseline
work. All owned processes have stopped and the heartbeat is paused.
Partial baseline attempts remain saved in the local run and are not analyzed
or compared here. No baseline retry or classifier amendment was applied.

REPORT.txt explains the syntactic empty/no-value flag. Correct empty-reference
answers are included in that flag; do not equate it with refusal or failure.
Primary latency, usage and estimated cost include accepted attempts only.
Estimated USD uses the user-approved official OpenAI schedule for Azure; actual
provider billing was not measured. The full cohort includes development exposure.

per_question.csv contains JSON strings in structured columns; some cells are
large. With Python csv, use csv.field_size_limit(sys.maxsize). JSONL preserves
rich diagnostics directly. Raw trace pointers and recorded absolute paths refer
to the original local artifacts, not paths inside this copied result directory.

Original export: output/experiments/gpt-6-luna/v3_complete_2931_20261016
Original run and retained raw attempts/audits/diagnostics:
output/experiments/gpt-6-luna/v3_full_2931_20261015
The local export snapshot can contain hardlinks to immutable run data; do not
edit those snapshot files or change their permissions. Use these copied tables.

original_delivery_artifact_hashes.json and verified_provenance.json describe
the original export and its full local audit inventory. They do not claim that
all raw traces are copied into this Git directory. ARTIFACT_SHA256SUMS.json
provides sizes and hashes for the files actually packaged here.
