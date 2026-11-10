# New question sources

This build log covers the question sources added after the multi-agent v2 results: extra pools drawn from questions the split left unsampled, FHIR-native generated questions, finisher paraphrases, multi-record perturbations, and graded demo questions. Only the multi-record perturbations touch the runtime: `tools/perturbation.py` now also loads version-3 manifests. Because `tools/` is part of the code fingerprint, result files written before that change show as stale, and `manifests/perturbations.json` was rebuilt for the new omitter hash. Everything else lives in `scripts/` and `demo/`.

Each source is a split manifest with the same patient pools as `manifests/split.json`, so it runs through the unchanged runner. `scripts/run_experiments.py` enforces the 900/200/300 run sizes only for `manifests/split.json`.

```bash
python scripts/run_experiments.py run --arms ordinary \
    --split_manifest manifests/pools/held_out_v2.json \
    --output_dir output/experiments/<model>/held_out_v2 \
    --max_total_tokens 200000 --max_tool_calls 30

# A generated bank also needs its own input and family manifest
python scripts/run_experiments.py run --arms ordinary \
    --input final_dataset/generated/fhir_native.csv \
    --split_manifest manifests/generated/fhir_native.split.json \
    --families manifests/generated/fhir_native.families.csv \
    --output_dir output/experiments/<model>/fhir_native \
    --max_total_tokens 200000 --max_tool_calls 30
```

Only `ordinary` arms apply. The recovery and variability arms are tied to the frozen split.

## Pools of unsampled released questions

`scripts/build_question_pools.py` writes `manifests/pools/<name>.json`. In each one, pilot and dev match `split.json`, and held_out asks a different set of the held-out patients' released questions. No pool reuses one of the 300 frozen held-out questions. Pools may overlap each other.

| Pool | Questions | Patients | Templates | Selection |
| --- | --- | --- | --- | --- |
| `held_out_v2` | 300 | 58 | 54 | Family-stratified, same family mix as the 300 (92 / 49 / 159) |
| `template_balanced` | 272 | 56 | 63 | Up to 5 per in-scope template |
| `out_of_scope` | 213 | 53 | 11 | Every duration / paired-difference / paired-comparison question |
| `zero_finisher` | 300 | 58 | 45 | `classify_template(question) is None` |
| `truncation` | 161 | 49 | 26 | Truncated 8,000-token packet that omits at least one cited non-context record |
| `disagreement` | 99 | 45 | 13 | Non-integer aggregation answers, or lookups/counts naming a multi-word concept |

The truncation and disagreement selections read gold citations and answers. This is offline construction; agents never see the criteria.

The planning notes listed the in-scope templates missing from held_out incorrectly. The 13 in-scope templates absent from the frozen 300 are T004, T005, T006, T008, T011, T014, T015, T032, T034, T063, T066, T067 and T072. All except T008 have 2–6 unsampled questions on held-out patients, and `template_balanced` includes them.

The `out_of_scope` pool asks question types the proposal excluded. Report it separately from the proposal's three operations.

## Full-record oracle

`scripts/fhir_oracle.py` (`RecordQuery`) extends `scripts/packet_query.py` as a subclass, because `packet_query.py` is pinned by hash in the frozen perturbation manifest. It evaluates a released EHRSQL query over the patient's complete FHIR record instead of a packet, and traces citations. Each main-table row is attributed to the resource that produced it, using the difference in projected rows with and without that resource. The query is then rewritten to select that source column:

- selections cite the selected rows;
- counts, existence checks and aggregates cite every row they reduce;
- paired differences and comparisons cite each subquery's rows;
- linked Medication, Specimen and parent microbiology resources are added to the citations.

`packet_query.py` does not trace resource IDs and has no `diagnoses_icd` relation, contrary to the planning notes. Diagnosis questions therefore remain unsupported.

`python scripts/fhir_oracle.py` checks the oracle against all 2,931 released labels:

- **Supported:** 2,392 of 2,931 rows. The rest are unsupported, mostly because microbiology hospital membership is absent from FHIR (192), mixed prescription doses cannot be attributed to individual drugs (145), or the query uses an unprojected relation such as diagnoses or transfers (89).
- **Agreement:** 95.1% of supported rows reproduce the SQL label.
- **Citations:** mean gold-ID recall is 86.5%.
- **Divergent templates:** T001 (73%), T051 (79%), T003, T006 and T066 (86%), and the durations T023, T024, T025 and T027 (29–43%). There, the FHIR projection and the raw MIMIC SQL disagree, for example on medication names inside mixes and on stay and prescription end times.

## FHIR-native generated questions (plan methods 2 and 3)

`scripts/generate_fhir_questions.py` re-instantiates released questions for other patients. It does not need the raw MIMIC SQLite database or the EHRSQL parameter file, which are not in the repository. It uses the 2,931 released rows as skeletons (wording, SQL and slots). For each target patient it:

1. swaps in the patient's subject ID and their most common released current-time anchor;
2. redraws slot values from the patient's own record: lab, vital, output, drug (including the three-drug template), route, procedure, specimen, care unit, admission route, and a vital threshold at the patient's median value;
3. evaluates the rewritten SQL with `RecordQuery`, which gives the answer and citations from the FHIR fields.

Safeguards:

- A skeleton is used only if the oracle reproduces its own label, and its template reaches at least 90% agreement (`--min-agreement`). This excludes the divergent templates above.
- Skeletons with absolute dates ("in 11/2161") are skipped.
- Per (template, patient), at most one question with a non-empty answer is kept, plus one whose answer is empty, 0 or False for a concept present elsewhere in the patient's record. These scope tests are the positive/negative pairing from the plan.
- Generated questions that repeat a released question are dropped.

Outputs:

- `final_dataset/generated/fhir_native.csv`: the released columns plus `source_question_id`, `polarity`, and `features`.
- `manifests/generated/fhir_native.families.csv`: the family manifest.
- `manifests/generated/fhir_native.split.json`: per patient pool, a family-stratified sample of 20 / 50 / 300 questions, plus each template's oracle agreement.

`features` tags the reference hops each answer needs (`medication_mix`, `specimen_link`, `micro_organism_chain`, `icu_to_hospital`), so multi-hop subsets (plan 4B) can be cut from the bank. Questions about susceptibility and ICU-to-hospital diagnoses are not generated, because the projection has no relations for them.

Of the 2,931 released rows, 1,498 serve as skeletons, covering 57 of the 69 templates the oracle supports. The bank holds 7,340 questions over all 94 patients:

| Family | Positive | Negative | Total |
| --- | --- | --- | --- |
| lookup_counting | 1,445 | 1,108 | 2,553 |
| numerical_aggregation | 552 | 670 | 1,222 |
| temporal_selection | 1,394 | 1,253 | 2,647 |
| out_of_scope | 485 | 433 | 918 |
| **All** | **3,876** | **3,464** | **7,340** |

Feature tags: `icu_to_hospital` 2,446, `specimen_link` 1,199, `medication_mix` 207, `micro_organism_chain` 6. Organism chains are rare because most organism questions need microbiology hospital membership, which the FHIR record does not carry.

The split manifest samples the three in-scope families only (`--split-families` to change); out-of-scope rows stay in the bank.

- **Held out:** 300 questions (119 / 57 / 124), 153 positive and 147 negative; answer kinds 208 value, 62 empty, 30 null value. Filter on `polarity` for a positive-only set.
- **Pilot / dev:** 20 and 50 questions.

Procedure slots draw only ICD-coded procedures. The projection also files ED and ICU procedure events under `procedures_icd`, and those names are not EHRSQL procedures. `--resplit` rebuilds the split manifest from an existing bank without regenerating.

## Finisher paraphrases (plan 6A)

`scripts/build_paraphrases.py` rewrites every released question of the seven finisher templates from its structured slots, keeping the patient, current time, answer and citations:

- `routed`: synonyms the router still matches ("earliest", "latest"), so the finisher still fires.
- `unrouted`: no router cue words ("oldest / newest … start time", with "first / last" visit scopes rewritten the same way), so no finisher fires.

The router's verdict on every rewrite is checked. All 994 rewrites of 497 questions pass. The split manifest asks the rewrites of each split's sampled questions; held_out asks 134 rewrites (67 × 2) of the 67 frozen held-out finisher questions. Comparing original, routed and unrouted accuracy on the same 67 questions separates the finishers from the analysis convention guidance.

## Multi-record perturbations (plan 5)

`scripts/build_multi_perturbations.py` writes `manifests/perturbations_multi.json` (version 3) over the 300 frozen held-out questions. It uses the same visible-query proof as the one-record builder: the original query reproduces the reference, and the omitted packet's result differs and scores wrong.

| Kind | Eligible | Selected | Description |
| --- | --- | --- | --- |
| `link_target` | 22 | 19 | A Medication or Specimen that a visible main-table record links to |
| `tie_peer` | 1 | 1 | A tie set keeps its earliest cited record; the rest are removed |
| `k_record` | 18 | 18 | The smallest k (2–5) most recent cited records that changes a count, existence check or aggregate |

The runtime keys omissions by question, so each question gets at most one kind, assigned rarest kind first. That leaves 38 entries.

`--verify-manifest` rebuilds every entry from local sources.

Only 17 frozen held-out questions have multi-value answers, and most of them (T051) fail the visible-query reproduction check. That is why `tie_peer` yields a single question.

`omit_from_packet` also redacts references to an omitted resource, so a link-target omission hides the link as well as the name.

`tools/perturbation.py` accepts both protocols: version 2 (exactly 50 entries, one `removed` record) and version 3 (any size, a list of `removed` records). `scripts/revisions.py` counts a multi-record omission as retrieved, cited, used or recovered only if every removed record is. Run the recovery arms with the new manifest and a separate output directory:

```bash
python scripts/run_experiments.py run --arms recovery \
    --perturbations manifests/perturbations_multi.json \
    --output_dir output/experiments/<model>/held_out_multi \
    --max_total_tokens 200000 --max_tool_calls 30
```

This runtime change alters the code fingerprint, so result files written before it now show as stale. `manifests/perturbations.json` was rebuilt for the new omitter hash; its 50 entries are unchanged.

## Graded demo questions (plan 6B)

`demo/app.py --generated <csv>` (default `final_dataset/generated/fhir_native.csv`) adds a "Generated (graded)" question mode for pilot and dev patients. After a run, `demo.backend.grade` scores the released answer with the typed scorer and shows the FHIR-derived reference. Generated questions are never listed or graded for held-out patients.

## Rebuilding

```bash
python scripts/fhir_oracle.py                       # oracle agreement with the released labels (~3 min)
python scripts/build_question_pools.py              # manifests/pools/*.json (~2 min)
python scripts/generate_fhir_questions.py           # final_dataset/generated/fhir_native.csv (~25-45 min)
python scripts/build_paraphrases.py                 # final_dataset/generated/paraphrases.csv
python scripts/build_multi_perturbations.py         # manifests/perturbations_multi.json (~4 min)
python scripts/build_multi_perturbations.py --verify-manifest manifests/perturbations_multi.json
```

All builders read the local NDJSON under `physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir` and need no FHIR store access. Starting packets for a pool can be frozen with `python scripts/build_starting_packets.py --split <pool manifest> --output <path>`.
