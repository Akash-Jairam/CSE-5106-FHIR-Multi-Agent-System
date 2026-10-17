Cache provenance at the immutable 891-row checkpoint

IN_PROGRESS_NOT_FINAL. Audit uses the preserved 891-row measurement capture and
its exact round 10 proof, not the later live canonical. Only numeric usage, IDs,
PID/session/counters/request identifiers and timestamps were inspected. No answer,
reference, prompt, correctness or tool-argument content was inspected or scored.

The retained 20 journals contain 5,944 starts and 5,942 ends: 5,921 successes and 21 errors.
Successful event cache detail is explicit zero 55, positive 5,866, null 0. Two final
round 1 starts lack ends; no recorded counters are missing and no ends lack starts.
The extra 63 successful events are outside the accepted 891-row call associations.
They are not added to accepted measurements or billed-cost estimates.

All 891 accepted question token sequences have exactly one contiguous match in a
worker stream from the same sealed round. All 5,858 matched event keys are distinct.
Twenty individual input/output token signatures were ambiguous alone; the full
question sequences resolve them. Matched cache detail is explicit zero 50, positive
5,808, null 0. Every matched diagnostic cache count agrees with the frozen normalized
canonical count. Cache was not used to locate these matches. Every matched SDK
duration sum also fits within the accepted question latency.

This proves unique observed-metadata associations in these retained journals. It
is not a persisted question/request-ID join. Logging is best-effort; a failed start
before counter increment may leave no visible gap. Token-sequence uniqueness is
conditional on the retained events and serial per-worker question-call sequence.
Do not claim unconditional cache completeness or extend this 891-row finding to
V3, later baseline questions or missing logs. Omitted/null cache detail is never
inferred to mean zero, and no frozen metric/price/query was altered or rerun.

Raw metadata-only journals were copied byte-identically under raw_diagnostics/.
Source hashes are verified before/after capture. audit_cache_provenance.py retains
the reproducible procedure; it refuses overwriting evidence and binds all inputs
to the recorded 891-row/proof hashes. No repository/live files were modified.
