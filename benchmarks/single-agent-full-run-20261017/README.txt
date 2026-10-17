Matched Single Agent benchmark: evidence navigation

The authorized 2,931-question Single Agent baseline stopped incomplete at
2026-10-17 13:52:28 UTC: 2,929 runtime-clean outcomes are accepted and two
provider-rejected questions remain pending review. No inference jobs remain
active. V3's complete result is unchanged. Direct LLM remains deferred; the
original three-arm benchmark is incomplete. Do not evaluate partial baseline
accuracy or describe this selected arm as complete.

The scheduled monitoring heartbeat is paused because inference has stopped and
the remaining cases require external provider review. monitor_pause_receipt.json
records the applied status and preserved schedule/prompt. This does not pause or
complete the goal, authorize another request or change any benchmark data.

Saved checkpoint: sealed_round30_first_clean_validation.json records 2,859 of
2,931 first-clean outcomes. This is a historical checkpoint, not live progress
or a completion claim. Later sealed proofs may supersede this checkpoint.
The final 70-question round is reconciled in runtime_metadata_round31/.
terminal_attention_capture/ preserves the actual incomplete stop, independent
native absence, stable source/file bindings and the unchanged monitor's exact
attention refusal. That refusal is not a successful completion-verifier receipt.

Authoritative local run directory, relative to the repository root:
  output/experiments/gpt-6-luna/v3_full_2931_20261015
It retains canonical results, the acceptance ledger, sealed source attempts,
compressed audits, diagnostic journals and supervisor state. Saved evidence
packages supplement that run; they do not schedule or resume inference.

For current answer-blind observation only, run from the repository root:
  PYTHONPATH='.local-setup/phase2-dependencies:.' .venv/bin/python \
    benchmarks/single-agent-full-run-20261017/monitor_single_agent.py observe \
    --run-dir output/experiments/gpt-6-luna/v3_full_2931_20261015 --skip-native
This records terminal attention without treating stopped processes as live;
native absence is preserved separately in the terminal package. An observation
failure is not a reason to restart inference or duplicate jobs. Provider review
and a reviewed legitimate recovery amendment are needed for the parked cases.

Start with these files in this directory:
- protocol.txt: authorized scope, matched settings, first-clean selection,
  bounded recovery, provider parking, interpretation and publication limits.
- analysis_when_complete.txt: gated full-cohort verification/export procedure
  and independent reconciliation. Pending questions mean the arm is incomplete.
- metric_interpretation.txt: repeated-prompt heuristic, normalized cache counts,
  cache-detail availability and the limits of token-based price estimates.

Execution and recovery evidence:
- observation_traces/: closed read-only observation windows and source hashes;
  these are sampled monitoring records, not whole-run completion proofs.
- reused_batch_validation.json: reuse of the earlier ten clean outcomes.
- sealed_round*_first_clean_validation.json: saved boundary proofs of selection,
  immutable attempts/audits, retry queues and preservation of unselected arms.
- launch_receipt.json, resume_validation.json and monitor_validation.json:
  operational setup/scope evidence; retained process details are historical.
- provider_recovery/: reviewed operational classifier amendment, answer-blind
  replay, bounded unchanged-request recovery and validation sources.
- runtime_metadata_round*/: safe boundary records of exclusions and recoveries.
- terminal_attention_capture/: reviewed metadata capture and independent QA of
  the incomplete 2,929-outcome stop; all full-cohort analysis gates remain closed.
- transport_health_round23/: timeout-gap and same-client recovery evidence,
  with live snapshot versus post-seal journal bindings and explicit limits.
- provider_review_packet/: safe provider-review metadata for parked requests.
  Provider rejection remains pending, never an incorrect benchmark answer.

Interim measurement packages:
- measurement_checkpoint_497/ and measurement_checkpoint_891/: accepted-only
  numeric projections, source bindings, first-clean proofs and independent QA.
  They contain no partial accuracy comparison and are not final cohort results.
- cache_provenance_checkpoint_891/: inferred numeric-sequence associations for
  that checkpoint, coverage summaries and independent reconstruction.
  This coverage does not establish availability for future baseline rows.
- metric_definition_reviews/: source evidence supporting metric interpretation.
- final_cache_audit_preparation/: reviewed full-baseline metadata helper,
  bound final-verifier capture, synthetic validation and stale-receipt checks.
  This is preparation only; no full-baseline cache audit has run yet.
- independent_stats_validation/, paired_export_validation.json and
  paired_export_independent_review.json: synthetic/offline validation of the
  eventual complete-cohort analysis path.

Completed V3 evidence, relative to this directory:
  ../v3-full-run-20261015/completed_v3/ : original complete report and tables.
  ../v3-full-run-20261015/cache_metadata_reconciliation_20261017/ : separate
  full-V3 cache-metadata supplement. Its inferred-identity and best-effort
  logging limits remain explicit; it does not establish baseline coverage.

Use package receipts and ARTIFACT_SHA256SUMS.json/artifact_hashes.json to verify
saved bytes and source bindings. Preserve original immutable reports and hashes.
Evidence/test sources retain original absolute repository and private-temp paths;
those paths record provenance and are not portable execution instructions.
Do not rerun fixed-count capture scripts against a later live checkpoint.

Dollar costs are frozen-rate usage estimates; actual provider billing is unknown.
Cached counts and numeric balances alone do not certify cache-detail completeness.
Detailed public GitHub publication still awaits the user's explicit approval;
no automatic public push or external provider message is authorized here.
