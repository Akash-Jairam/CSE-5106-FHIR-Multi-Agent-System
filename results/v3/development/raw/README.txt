Frozen development raw archive

Three ordered parts reconstruct the original completed-development ZIP exactly. Each Git blob is below 90,000,000 bytes. No archive member or compressed byte was changed.

From this directory:
cat query_contract_v3d_300q_completed.zip.part001 query_contract_v3d_300q_completed.zip.part002 query_contract_v3d_300q_completed.zip.part003 > /private/tmp/query_contract_v3d_300q_completed.zip
shasum -a 256 /private/tmp/query_contract_v3d_300q_completed.zip

Expected SHA256: d6144775c811e387e9201273db5c2ae4faec45647bfecd020ec7ef49e8db1738
Expected bytes: 241257127

The destination must be a new file. The archive contains original benchmark outputs and detailed audits; new numeric exports omit clinical/prediction/reference content. See the original raw_archive_manifest.json for provenance.
