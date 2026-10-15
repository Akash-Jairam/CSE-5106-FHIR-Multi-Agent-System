V6 posthoc mechanism helper — prepared before completion; no actual results opened

Scope
-----
This is analysis-only. It does not alter the runtime, run models, refetch FHIR,
execute generated programs, or change scoring. Do not invoke run/main until both
root-owned jobs are actually terminal, the separately pinned V6 completion guard
passes all 600 first attempts, and the unchanged official comparison is frozen.
The external terminal flag is a caller assertion, not proof of process state.

The helper reuses the V5 analysis gate structure and the independently reviewed
V6 guard. It hashes the frozen source/runtime/ZIPs/public binding/input metadata,
restores all 600 owned gzip audits, checks exact cohorts/identities/raw snapshots,
and binds the official report BEFORE loading the unchanged typed scorer. It
loads gold/reference values only from the already completed, SHA-bound fixed300
raw first-attempt rows. The full 2931 CSV is byte-hashed, never parsed for answers
or question text. Every saved error remains in the 300-question denominators.

A-only reconstruction
---------------------
* revision.primary is the actual Base-normalized released A (answer, evidence,
  limitation). primary_revision.final is a computed candidate before source
  withholding and is never substituted for A.
* A primary fatal outcome is incorrect, including an empty reference. If the
  primary is ineligible/prose-only and available, its error is the same row.error
  copied by the controller. For an eligible A, the actual final execution,
  workspace, rendered answer and original automatic source check must bind;
  then a later auxiliary fatal error does not retroactively make A fail.
* Missing primary/revision/common-output metadata is unavailable. It is never
  filled with None, 0, or an abstention. The report gives an availability count
  and bounds; any paired A-only comparison states its available denominator.
* A-only is the recorded 175k/27 natural Single policy with its actual 30-iteration
  limit. It is not a new run or a counterfactual 200k/30 refreshed control.
* Source-feedback attempts, completed primary-source revisions, selected B
  switches, and total auxiliary-stage effects have different counters. Generic
  revision.revised is informational only. Source revision effects use initial
  draft -> released A, not draft -> pre-withholding computed candidate.
* B eligibility, exact JSON agreement/disagreement, B typed correctness, actual
  selected B help/harm/neutral, source withholding, known budget stops, unknown
  usage and non-switch auxiliary losses remain separate.
* The unchanged scorer can credit actual non-error None on an empty reference;
  those credits are labeled. Typed correctness does not imply a justified
  search, source completeness, semantic correctness or independent evidence.
* All comparisons are posthoc development diagnostics, not new acceptance gates.
  The secondary patient-cluster comparison is Team minus available recorded A,
  10,000 draws, seed 5106, using unchanged experiment_stats.paired_difference.

Stable reviewed dependency
--------------------------
/tmp guard packet (not modified here):
/private/tmp/fhir-v6-integrity-bfjeklza/v6_guard.py
SHA256 49ec8be98372cbeaf9c9ab917d02fdf0bbbdce06471a8a0745a02aae9c485c18
Sibling reviewed_v4_guard.py must remain beside it, SHA256
98d272f8ed8e42914b5ed1776908cf6e25e209e5726d5386b4c168a270a499e8.
A copied formal guard must retain these exact bytes and sibling layout.

Synthetic test invocation (no real results)
------------------------------------------
cd '/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System'
.venv/bin/python -I /private/tmp/fhir-v6-posthoc-sj4zhs4u/test_v6_posthoc.py
32 synthetic tests pass. Tests import only public guard/helper sources and own
synthetic fixtures. They never invoke main on the actual run.

Post-completion invocation — ROOT MUST SUPPLY THE FOUR REVIEWED EXTERNAL PINS
--------------------------------------------------------------------------
.venv/bin/python -I /private/tmp/fhir-v6-posthoc-sj4zhs4u/v6_posthoc.py \
  --root '/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System' \
  --completion-report '<formal-record>/completion_protocol_audit.json' \
  --completion-report-sha256 '<frozen-completion-report-SHA256>' \
  --official-report '<formal-record>/results.json' \
  --official-report-sha256 '<frozen-official-results-SHA256>' \
  --guard-module /private/tmp/fhir-v6-integrity-bfjeklza/v6_guard.py \
  --guard-sha256 49ec8be98372cbeaf9c9ab917d02fdf0bbbdce06471a8a0745a02aae9c485c18 \
  --guard-pins '<actual-prelaunch-reviewed-pins-path>' \
  --guard-pins-sha256 '<actual-prelaunch-prespecified-pins-SHA256>' \
  --single-label v6_single --team-label v6_team \
  --output '<formal-record>/diagnostics' \
  --confirm-both-owned-jobs-terminal

<formal-record> is the absolute benchmarks/design-v6-20261015 directory.
The output must be fresh and must be that exact formal diagnostics directory or
an external /private/tmp directory. No secret configuration/cache/pickle is read
or archived. Actual report/raw SHA checks are repeated during analysis before
writing. Failure writes no diagnostics directory through the normal run entry.
The helper's SHA is recorded in the report; root should preserve/review this
analysis packet before invoking it. All generated records are analysis metadata
and may contain completed development answers; keep them in the formal record
only when root explicitly chooses that publication scope.
