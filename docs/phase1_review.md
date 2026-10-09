# Phase 1 Review

Reviewed on 2026-10-09 against the project proposal and PRs #33-#42.
The correction branch starts at the assembled Phase 1 head
`ad25258bfcd4f9ba5d6051e5d7c8c68e9d1db5ba` (`p1-10-budget-controls`).
These changes repair Phase 1 contracts; they do not implement the planned
multi-agent system or run the final benchmark.

## Findings and Corrections

| Original PR | Reproduced problem | Correction |
|---|---|---|
| #33 Environment | A partial demo import could pass readiness; generated Python ran in the host process without an enforced deadline. | Require 100 demo patients; isolate code with copied task data, stripped environment, no host records/network, and a real timeout. |
| #34 Families | Paired differences, comparisons, and durations were classified as temporal selection. | Keep the audit rows but mark 459 out of scope; sample only the three proposed operations. |
| #35 Split | 26 previously exposed patients appeared in the held-out pool. | Record exposure provenance; reserve all 32 known exposed patients for development. |
| #36 Packets | Gold recall selected the cap; pruning removed answer-bearing fields and linked metadata. | Gold-free construction/sweeps, post-freeze diagnostics only, exact rendered resources, typed dependencies, and a regenerated version-2 freeze. |
| #37 Typed scoring | Extra/contradictory values earned credit; malformed sets crashed; units and nulls were mishandled. | Complete normalized answers, exact counts, requested timestamp precision, explicit absence, and verifiable units. |
| #38 Evidence | Bare IDs collided across types; precision omitted wrong resource types; input metadata could replace gold labels. | Typed identities, structural NDJSON/gzip parsing, exact field values, authoritative labels, and intact evaluation denominators. |
| #39 Output | Malformed evidence crashed validation; failed attempts and truncated traces could look successful. | Enforce the common contract at runtime; retain usage and full traces; explicit retryable failures. |
| #40 Direct baseline | Incomplete JSON was treated as an answer. | One call, no tools, and explicit failure for incomplete JSON. |
| #41 Single agent | Revised answers bypassed checks; mutable working records passed as original evidence; full hidden fields gave unequal starting information. | Validate original patient/time/encounter evidence, recheck the single revision, and preload only rendered packet fields. |
| #42 Budgets | Completions, automatic checks, and retries bypassed ceilings. | Cap completions, count checks, disable hidden bounded-call retries, preserve attempts, and refuse unaccounted usage or reported overruns. |

Resumption now binds results to question IDs, input content, source code,
model, settings, budgets, and frozen split. Failed attempts retain their
usage in `attempt_history`. Empty and completed inputs avoid unnecessary
credentials or process pools. Unknown provider pricing is null, not free.

## Data Freeze

There are 2,472 eligible questions among 2,931 audited source rows.
The frozen selection is 20 pilot, 50 dev, and 300 held-out questions from
3, 32, and 59 disjoint patient pools. Empty and null cases remain eligible.
All 370 version-2 packets fit the common 8,000-token ceiling. The manifest
records exact packet-text hashes, dataset/split/retriever hashes, and the
checksums of 24 public FHIR gzip files verified against the official release.
See [the data review](phase1_data_review.md) for regeneration and exposure
provenance. No held-out answer inspection or model experiment was used to
tune these changes.

## Verification

From the repository with its Python environment and dependencies installed:

```sh
python -m unittest discover -s tests -p 'test_phase1_*.py' -v
git diff --check
python scripts/check_setup.py --model <configured-model>
```

The integrated suite passed **117 tests**, including all 2,931 released
reference round-trips and an offline evaluator CLI check of all six answer
types. The Python worker's real macOS tests cover arithmetic, copied-data
isolation, host-file denial, environment stripping, network denial,
`SystemExit`, timeout termination, and unavailable-sandbox refusal.
Three independent development packet rebuilds (one per family) match the
frozen text hashes, including simulated store ordering and metadata removal.

The live readiness check passed against the configured store: 100 Patients,
637 Encounters, 17,552 MedicationRequests, and nonzero Observations.
The Observation search total is an estimate, not an import checksum.
Seven additional mocked local Azure adapter tests passed; that adapter and
all credentials remain local and are not part of this contribution.

No paid model calls were made. The optional GPT-6 Luna development smoke
test awaits explicit approval to send the public demo questions and packets
to the configured model endpoint. This review makes no model-accuracy claim.

## Remaining Limits

- The original gold-guided cap selection cannot be erased. Disclose it as
  a historical protocol deviation; do not describe the original run as a
  clean patient-disjoint final evaluation.
- Provider prompt framing differs from local tokenization. A reserve and
  completion cap limit exposure; reported overruns are logged and refused,
  rather than concealed. Calls without usage cannot continue under a ceiling.
- Python requires `sandbox-exec` on macOS or `bubblewrap` on Linux and fails
  closed otherwise. The Linux backend was not exercised on this macOS host.
- Field checks verify cited source values and scope, not the completeness
  of retrieval or the correctness of derived arithmetic. Those remain
  evaluation questions and planned verifier responsibilities.

## Integration

Akash reviews and performs the final merge. The correction PR targets
`p1-10-budget-controls`, which already assembles the original Phase 1 stacks.
Integrate the corrections into that assembled branch before final merging.
Use merge commits if continuing the stacked workflow so ancestry is kept;
do not assume an earlier PR's branch already contains these corrections.
The original checkout, Akash's branches, credentials, downloaded source data,
and private planning notes are not modified or published by this contribution.
