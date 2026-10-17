INTERNAL NONSELECTED V3 ATTEMPT RAW ARCHIVE

This separate archive preserves all 81 nonselected sealed source rows from the
completed V3 execution. The 2,931 selected rows are preserved in the primary
full_v3 archive. No scoring or classification was rerun to select these rows;
source-result/selection-ledger origin metadata determines the partition.

Each UTF-8 JSONL envelope in chunks/*.jsonl.xz contains:
  origin: question/patient ID, round, original source file and row index
  attempt: every original parsed nonselected source-row field
  audit_json: original owned gzip-decompressed audit text (or null if absent)
The attempt index records row semantic hashes and original compressed and
uncompressed audit hashes/sizes. Source batch/result/identity/shard/seal files
are byte-bound by the manifest. All parsed attempts and byte-exact payloads
were read back and checked, and all original source hashes were reobserved.
All compressed files are below 90MB. This archive is outside primary accepted
measurements and is retained for internal reproducibility.

Read using Python standard-library lzma.open(..., 'rt') and json.loads(line).
Original canonical whitespace, source file containers and gzip headers are
not recreated. The exact owned payload text and parsed row values are retained.
Packaging sources are historical provenance, with original temporary paths;
they are not portable launch instructions and make no model/provider calls.
No original input, source, canonical result or attempt file was modified.
