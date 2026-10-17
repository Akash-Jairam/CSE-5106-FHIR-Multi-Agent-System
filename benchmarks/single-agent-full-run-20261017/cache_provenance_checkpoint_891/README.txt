IN_PROGRESS_NOT_FINAL: 891-question cache-metadata reconstruction.

This directory preserves the metadata view of the full private package. The
20 byte-preserved raw diagnostic journals remain in the retained local run at
cache_provenance_checkpoints/accepted-00891/raw_diagnostics. The original
package manifest and README are preserved with explicit ORIGINAL/original
names. This view has its own checksum inventory after independent review; it
is not a standalone portable raw-journal archive. Scripts retain original
absolute paths. No frozen metric values or inference settings were changed.

Associations match full ordered prompt/completion-count sequences, excluding
cache counts from match keys. They are reconstructed observed metadata links,
not persisted question/request-ID identities. Best-effort logging limits
remain. No baseline accuracy, comparison or extrapolation is included.
