Preparation for the reviewed retry monitor

These four helpers retain the original complete-cohort and byte-binding gates.
The sole source adaptation updates bound_completion.MONITOR_SHA to the reviewed
two-question retry monitor (1e091082d844fcabafb0e6a558d8b41634a26e2f476790b30b31c61033eba3dc).
The original preparation package and its manifests remain unchanged.

Use this directory's capture_final_proof.py only after exactly 2,931 selected
first-clean outcomes, completed selected-arm state, no active batch and no parked
question. Save to a NEW output directory outside the run. Then pass that new
proof.json and binding.json to this directory's audit_complete_single_cache.py.
Keep all four helpers together. The original analysis_when_complete.txt gives
the command structure; use these reviewed_provider_retry/final_cache_audit paths
for this amended run. The selected scope still excludes Direct LLM.

Source preparation is not evidence that the complete cache audit has run.
Actual execution and any coverage/missingness results require separate receipts.
Provider cache-detail identity is inferred from numeric sequences in best-effort
journals; it is not a recorded direct question/request-ID join. Actual provider
billing remains unknown. No prompts, filters, measurement rules or raw results
are changed by these helpers.
