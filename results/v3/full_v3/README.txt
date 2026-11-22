COMPLETE V3 RESULTS AND PORTABLE RAW DATA

This directory preserves the original full V3 run: 2,931 accepted questions
across 94 patients. It contains the saved full-cohort numeric evidence and the
complete parsed canonical outcomes plus their owned audit payloads. No answers
were rescored or benchmark implementation changed during consolidation.

numeric_evidence/ copies the original tables, report, pricing and independent
numeric reconciliation verbatim from:
  benchmarks/v3-full-run-20261015/completed_v3/
SOURCE_HASHES.json binds each copied file to its original path, bytes and SHA256.
These describe the original complete V3 cohort. Separately declared matched
comparisons use their own explicit cohort/sample count.

data_archive/ARCHIVE_MANIFEST.json and question_index.jsonl describe the portable
patient-grouped XZ envelopes. Each envelope stores one original parsed canonical
row and one owned original audit text payload, avoiding duplicate rich/gzip/full
JSON copies. All parsed fields and decompressed audit bytes are preserved;
original canonical whitespace and gzip container bytes are not recreated.

The accepted archive contains 311 XZ parts (286,469,212 bytes) preserving
12,271,848,432 decompressed original audit bytes. Complete readback and the
delivered portable verifier both passed; PORTABLE_VERIFICATION.json records it.

Use archive_tools.py to verify without the original project/runtime, or rebuild
canonical JSON and its audit/ tree into a NEW directory. See RECONSTRUCTION.txt.
Original source and payload hashes, source-row ordering and per-file hashes are
retained. The archive includes detailed original question/record/answer content;
the numeric index and source bindings are supplements, not replacements.

All original run files remain unchanged. The original source directory is:
  output/experiments/gpt-6-luna/v3_full_2931_20261015
Original absolute paths in receipts are historical provenance, not requirements
for portable archive verification or reconstruction.
