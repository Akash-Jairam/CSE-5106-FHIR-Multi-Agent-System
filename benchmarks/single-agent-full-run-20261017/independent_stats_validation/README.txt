Independent completed-pair statistical QA (private preparation)

Run only AFTER the reviewed exporter creates a complete matched export with
V3_SINGLE_COMPLETE_DIRECT_PENDING.json. Both arms must contain the exact frozen
2,931-question cohort. This utility refuses missing/partial receipts and cohorts.

From the repository root:
  PYTHONPATH='.local-setup/phase2-dependencies:.' .venv/bin/python \
    /private/tmp/v3-single-independent-stats/reconcile_final_pair.py \
    --export-dir PATH_TO_COMPLETED_PAIRED_EXPORT \
    --output /private/tmp/NEW_COMPLETE_PAIR_QA.json

The QA output must be new and outside the immutable export. No source/benchmark
files are written. No provider, launcher, analyzer or answer-scorer code is called.
Full first-clean classifier/raw-audit replay remains owned by the reviewed export
gate; this utility verifies its bound receipt, canonical/ledger/input identities,
all delivery artifact hashes, and 17 independently recomputed measurement fields.
Correctness and empty/no-value flags are frozen scorer outputs; their counts,
cross-tabs, paired wins/losses and patient-cluster intervals are reconciled without
rescoring answers. The empty/no-value flag is not a synonym for model refusal.

Scalar measurements: n/observed/missing, total/mean/median/p95/sampleSD/min/max and
observed-only equivalents. Costs charge uncached+cached input exactly once and
output separately using input=.10, cached=.01, output=.50 USD per million tokens.
Actual billed costs are preserved when available and otherwise remain unknown.
All six group tables and phase aggregates are checked. Per-question CSV records
must exactly match JSONL records. Bootstrap intervals independently aggregate
patients, use 10,000 multinomial replicates, seed5106, a question-weighted mean,
linear percentiles and paired left-minus-right signs.

Tests (synthetic ONLY; no actual incomplete baseline scoring):
  .venv/bin/python /private/tmp/v3-single-independent-stats/test_stats_core.py
  PYTHONPATH='.local-setup/phase2-dependencies:.' .venv/bin/python \
    /private/tmp/v3-single-independent-stats/test_full_export_synthetic.py

The full harness is an isolated copy of the archived provider-amendment harness.
It creates temporary synthetic 2,931-row arms, invokes the unchanged frozen
analyzer only on those synthetic arms, and deletes the fixture afterward. It also
checks incomplete cohort/receipt refusal and rejects measurement, cost and paired
sign corruption. It does not modify the original helper or harness files.
