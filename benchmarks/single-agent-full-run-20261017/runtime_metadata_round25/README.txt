Answer-blind Single Agent round25 boundary audit

Final round25: 100 runtime-clean terminal results, no runtime failures or new
provider-policy rejections. All 100 results were adopted from round25 in the
first-clean acceptance ledger. Round24 had no generic runtime recoveries to
carry into this batch. These are runtime classifications, not correctness.

safe_review.json binds immutable batch/result/identity/shard input hashes and
all 100 restored audit artifacts. classification_review.py uses the frozen
manifest-selected classifier and answer-presence-preserving safe projection;
no answer/reference/correctness content is output or scored. Source hashes
bind the original runner, monitor, audit store, and manifest at capture.

The original active-checkpoint projection was a snapshot. The captured result
bytes were subsequently checked against the natural final seal. Original
batch.seal hashes cover batch/result/identity, not acceptance.json or native
journal files. Ledger observations are dated snapshots of a continuing run;
the normal seal-to-ledger transition is retained in ledger_observations.jsonl.
No provider journal was copied or claimed immutable in this packet.

parked_provider_status.json independently verifies the two reviewed provider
questions have exactly two sealed runtime policy rejections each, no third
attempt through round25, no round25 scheduling, and no accepted selection.
Earlier unattempted placeholders are explicitly not counted as requests or
rejections. State and acceptance hashes in that review are snapshots; prior
attempt hashes and audits are checked against their immutable seals.

All source code and artifact hashes belong to this private review packet.
No inference/scorer/provider calls, configuration reads, process actions,
run/source/settings mutations, clinical output or credential output occurred.
