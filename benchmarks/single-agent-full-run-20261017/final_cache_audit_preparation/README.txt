PREPARATION ONLY: byte-bound final Single Agent cache audit

This replaces the initial unbound preparation while preserving it under
initial_preparation/. The complete audit has NOT been run. Both current live
incomplete CLI attempts refused before canonical parsing or verifier execution;
no output directories were created. The final verifier was mocked only inside
synthetic metadata tests. Fifteen focused synthetic regressions passed in author,
independent reviewer and root runs. They are not a real completed-run validation.

Completion is mandatory: exactly 2,931 unique first-clean selected outcomes,
completed selected-arm state, no parked pending questions or active batches,
and the unchanged original three-arm benchmark_complete=false scope. A pending
provider-rejected question keeps this gate closed. No accepted query is rerun.

AFTER actual completion, run from the repository root using NEW directories
outside the live run. Keep all four code files in this folder together.

1. Capture the unchanged reviewed final monitor and stable byte/inventory bindings:
   PYTHONPATH='.local-setup/phase2-dependencies:.' .venv/bin/python \
     /private/tmp/cache-provenance-single-final-bound/capture_final_proof.py \
     --project-root . \
     --run-dir output/experiments/gpt-6-luna/v3_full_2931_20261015 \
     --output-dir /private/tmp/NEW_FINAL_SINGLE_MONITOR_CAPTURE

   This invokes the existing monitor verify --final-arm, without its optional
   --append-validation flag. The permanent monitor source remains unchanged at
   daa88848d68f0dbd9fbe586ecb4cea3fdd65bb8d574770f0bb42c943a88e9047.
   proof.json preserves exact successful stdout bytes; binding.json binds that
   digest to identical before/after snapshots. The snapshots cover canonical and
   identity bytes, the WHOLE acceptance ledger, supervisor state, manifest,
   monitor/public-source/input pins, and every current round file plus the exact
   sealed batch/result/identity/shard inventory. Changes during verification
   refuse capture and produce no proof directory.

2. Use only that newly bound proof for the measurement-only cache audit:
   PYTHONPATH='.local-setup/phase2-dependencies:.' .venv/bin/python \
     /private/tmp/cache-provenance-single-final-bound/audit_complete_single_cache.py \
     --project-root . \
     --run-dir output/experiments/gpt-6-luna/v3_full_2931_20261015 \
     --completion-proof /private/tmp/NEW_FINAL_SINGLE_MONITOR_CAPTURE/proof.json \
     --completion-binding /private/tmp/NEW_FINAL_SINGLE_MONITOR_CAPTURE/binding.json \
     --output-dir /private/tmp/NEW_COMPLETE_SINGLE_CACHE_AUDIT

   The gate requires proof digest, capture helper hashes, and exact current byte/
   inventory equality with BOTH saved snapshots before canonical parsing. Extra
   sealed rounds, changed first-clean choices/counts/pointers, canonical/identity
   bytes, completion timestamps, shards or sources invalidate the old receipt.
   Selected per-call usage_available must be True, matching the reviewed original
   guard. The final audit rechecks all byte bindings AND the exact round inventory.

The helper never changes run records, inference settings, prices or metrics and
never launches inference, scoring, provider requests or a replacement benchmark.
Standard JSON parsing is followed only by measurement/ID accesses; actual answer,
reference, correctness and tool-argument values remain unexamined.

Diagnostic coverage is a separate result from outcome-cohort completeness.
Missing journals, ambiguous/no-sequence matches and nullable cache detail are
preserved and reported. Missing detail is never relabeled an explicit zero;
cache count null/zero/positive values retain their meaning. Visible errors and
counter gaps remain matching barriers. Observed event totals are not automatically
accepted-cohort call totals, including interrupted work and unmatched starts.
Matches are reconstructed P/C-only contiguous token sequences in retained worker
journals, not a persisted question/request-ID join. Best-effort logging can omit
an event before incrementing a counter, so apparent coverage cannot establish
unconditional logging completeness.

Malformed/truncated/invalid journals, source mutation or duration-validation
failures can abort AFTER the measurement output directory or partial files exist.
Only a valid final safe_evidence.json establishes a completed diagnostic coverage
report. Directory existence, projections or copied journal fragments alone do NOT
establish one. Do not infer complete cache availability from a full outcome cohort.

The six original numerical helper functions and P/C matching block are preserved.
The bounded revision adds trusted stable monitor capture/current bindings, rejects
stale receipt inventories and restores the True-only guard. adaptation.diff and
adaptation_summary.json record the changes. The original helper/gate plus blocking
review and reproductions remain archived; their historical hashes are unchanged.
No original monitor, guarded source, live state or completed export was edited.
