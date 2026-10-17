Complete V3 cache-metadata provenance audit

The immutable V3 canonical SHA is 1c8c974ccff99aae802f698ad043f4d959bb1808b3abee0f9e27c9c3ffa8097c.
All 2,931 first-clean outcomes and their selected sealed raw measurement projections
validate against the original complete receipt/ledger/source inventory. No baseline
canonical or state is read or changed. Measurement projections contain IDs, timing,
usage and cache/call counts only; answers/references/correctness remain unexamined.

All 33 sealed rounds have retained diagnostic journals: 64 journals, 18,004 starts and
18,004 ends, including 17,923 successes and 81 errors. Every round's raw per-call count
reconciles to its retained end-event count and known-usage/success count. There are
no unmatched start/end keys, duplicate keys or recorded counter gaps. Observed
successful cache fields are null 0, explicit zero 6,876 and positive 11,047.

All 2,931 accepted question sequences match exactly one contiguous worker-journal
sequence, using input/output token counts independently of cache values. All 17,715
matched event keys are distinct; 50 individually ambiguous signatures resolve with
the full sequence. Matched cache fields are null 0, explicit zero 6,763, positive 10,952.
Every matched cache count equals the frozen canonical count; both cached-token sums
are recorded in coverage_summary.json. SDK duration sums fit accepted task timing.
The 208 other successful events and 81 errors remain outside accepted measurements.

These are unique reconstructed observed-metadata associations, not a persisted
question/request-ID join. Logging is best-effort; a failed start before incrementing
a counter may leave no visible gap. Count reconciliation and sequence uniqueness
within retained journals support observed coverage but do not prove unconditional
logging completeness. Do not extend this V3 finding to a future provider run or to
other baseline questions. Omitted/null cache detail is never inferred to mean zero.
No frozen metrics/prices/settings were changed and no accepted query was rerun.

The original 891 audit is preserved. The adapted audit streams the frozen public
result reader, which imports no scorer/provider modules, and verifies canonical,
ledger, source receipt and raw/journal hashes again after associations. Metadata-only
raw journals are byte-copied in raw_diagnostics/; input_provenance/ preserves the
small public receipts and ledger. first_pass/ preserves the initial equivalent
association pass before adding all-attempt call-count reconciliation.
