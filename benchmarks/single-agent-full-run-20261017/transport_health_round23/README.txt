Round23 transport-health evidence (metadata only)

safe_progress_live_snapshot_1133.json is a byte-identical preservation of the
11:33:28 UTC live snapshot. It documents the fresh heartbeat, advancing raw
checkpoint, original ReadTimeout and three ConnectTimeouts then observed,
open SDK/HTTP clients, and FD counts 17-18. Its diagnostic_source_hashes refer
to LIVE JOURNAL BYTES AT THAT SNAPSHOT, not immutable or final journal hashes.

safe_transport_review.json separately summarizes stable journal bytes observed
after the round23 batch seal: 635 successful requests and six timeout errors.
Both original worker sessions/clients later succeeded (23 and 22 successes
after their respective final timeout). Final FD range was 17-19 for one worker
and 17-18 for the other; closed flags stayed false. This supports recovery
without an audit-triggered restart or setting change. It does not establish
which local network, route or provider component caused the connection gap.

The batch seal hashes the batch, result and identity, NOT the diagnostic
journals. Post-batch-seal stable journal hashes are labeled separately from
the earlier live snapshot hashes. Raw journal files remain at their original
local run paths; this compact package contains safe aggregates and bindings.
No direct question/request-ID association is claimed by this health packet.

The original capture was executed as inline Python. Its code is reconstructed
from that tool cell in capture_snapshot_source_reconstructed.py and was saved
afterward; no before-capture source-file checksum is claimed. It was not rerun.
review_final_transport.py is the independently saved post-seal aggregation code.
The README/code/JSON checksums are in artifact_hashes.json. No raw provider
messages, clinical content, answers/references, credentials or endpoint strings
are included, and no model/provider/scorer requests or live changes were made.
