# New question sources

This build log covers the question sources added after the multi-agent v2 results: extra pools drawn from questions the split left unsampled, FHIR-native generated questions, finisher paraphrases, multi-record perturbations, and graded demo questions.

Only the multi-record perturbations touch the runtime: `tools/perturbation.py` now also loads version-3 manifests. Because `tools/` is part of the code fingerprint, result files written before that change show as stale. `scripts/restamp_run_identities.py` moves them to the new fingerprint (see [Re-stamping](#re-stamping-saved-results)). `manifests/perturbations.json` was rebuilt for the new omitter and oracle hashes, with identical entries. Everything else lives in `scripts/` and `demo/`.

Each source is a split manifest with the same patient pools as `manifests/split.json`, so it runs through the unchanged runner. `scripts/run_experiments.py` enforces the 900/200/300 run sizes only for `manifests/split.json`.

```bash
python scripts/run_experiments.py run --arms ordinary \
    --split_manifest manifests/pools/held_out_v2.json \
    --output_dir output/experiments/<model>/suite_pool_held_out_v2 \
    --max_total_tokens 200000 --max_tool_calls 30

# A generated bank also needs its own input and family manifest
python scripts/run_experiments.py run --arms ordinary \
    --input final_dataset/generated/fhir_native.csv \
    --split_manifest manifests/generated/fhir_native.split.json \
    --families manifests/generated/fhir_native.families.csv \
    --output_dir output/experiments/<model>/suite_generated \
    --max_total_tokens 200000 --max_tool_calls 30
```

Only `ordinary` arms apply; the recovery and variability arms are tied to the frozen split. Name output directories `suite_<name>`: `held_out_v2` is already the directory of the frozen multi-agent v2 results. `scripts/submit_question_suites.sh` prints (or, with `SUBMIT=1`, submits) every suite for one model on Compute2, one job at a time.

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

`scripts/fhir_oracle.py` (`RecordQuery`) extends `scripts/packet_query.py` as a subclass, through class attributes and three hooks. Those hooks were a behaviour-preserving refactor of `packet_query.py`; the rebuilt perturbation manifests have identical entries. The oracle evaluates a released EHRSQL query over the patient's complete FHIR record instead of a packet, and traces citations.

Each main-table row is attributed to the resource that produced it, using the difference in projected rows with and without that resource. The query is then rewritten to select that source column:

- selections cite the selected rows;
- counts, existence checks and aggregates cite every row they reduce;
- paired differences and comparisons cite each subquery's rows;
- linked Medication, Specimen and parent microbiology resources are added, and for transfers the Location of each answered care unit.

Beyond the packet oracle, the record projection adds:

- **`diagnoses_icd` and `d_icd_diagnoses`**, from Condition resources (T008, T018: 35 released questions);
- **`transfers`**, from hospital Encounter location periods named by Location (T026, T042);
- **microbiology `test_name`**, and, when a query asks for them, antibiotic susceptibility columns: `ab_name`, `dilution_comparison`, `dilution_value`, and `interpretation` (the FHIR display, such as "Resistant"). These come from the test → organism → susceptibility Observation chain;
- **prescription corrections.** 2,327 provider order entries (`medication-request-poe`) are MedicationRequests but not prescriptions, so they are left out. 651 pharmacy prescriptions have no `dispenseRequest.validityPeriod`; they start at `authoredOn`, truncated to the minute. These two causes, not mix naming or stay semantics, explained the earlier T001 and T051 gaps. The packet oracle still includes the order entries, which keeps the frozen perturbation proofs unchanged. Agents are not told about either kind, which may cost them prescription questions.

**Stay semantics follow the agents' guidance** (`agent/ehr_semantics.py`): a stay still open at the assumed current time is the current stay, with no end, so it is not the "last" (completed) visit. The released labels were computed on the static database, where that stay already has its discharge. `RecordQuery(..., static=True)` evaluates that reading. A row is semantics-sensitive if the two readings differ. All the low T023–T027 agreement came from this: the static reading reproduces all 139 labels, which are all empty.

`python scripts/fhir_oracle.py` checks the oracle against all 2,931 released labels:

- **Supported:** 2,540 of 2,931. Unsupported rows: microbiology hospital membership absent from FHIR (192), mixed prescription doses (145), and admission type (47).
- **Left out:** 11 selections that order by an approximated prescription start.
- **Agreement:** the oracle reproduces 96.1% of the remaining labels. Counting the static reading for semantics-sensitive rows (3.0%), it explains 99.2%.
- **Citations:** mean gold-ID recall is 97.5%.
- **Templates below 90% explained:** T003 and T066 (86%). T001 is now 100% and T051 94%.

The planning notes' fixes for T001/T051 (keep discharge times; emit parent and component mix names) were measured and rejected. Neither changes T001 or T051 agreement, and emitting parent names drops T001 to 26%.

## FHIR-native generated questions (plan methods 2 and 3)

`scripts/generate_fhir_questions.py` re-instantiates released questions for other patients. It does not need the raw MIMIC SQLite database or the EHRSQL parameter file, which are not in the repository. It uses the 2,931 released rows as skeletons (wording, SQL and slots). For each target patient it:

1. swaps in the patient's subject ID and their most common released current-time anchor;
2. redraws slot values from the patient's own record: lab, vital, output, drug (including the three-drug template), route, ICD procedure, diagnosis, specimen, antibiotic, care unit, admission route, and a vital threshold at the patient's median value. Slots of one table, such as specimen and antibiotic, are drawn from the same row, so they co-occur;
3. evaluates the rewritten SQL with `RecordQuery`, which gives the answer and citations from the FHIR fields.

Three **authored templates** cover multi-hop questions no released template asks (plan 4B):

| Template | Question | Hops |
| --- | --- | --- |
| T076 | Susceptibility of the organism in the last {specimen} culture with susceptibility results to {antibiotic} | Culture → organism → susceptibility |
| T077 | Antibiotics that organism was resistant to | Culture → organism → susceptibility |
| T078 | Diagnosis {name} during the hospital stay of the last ICU visit | ICU stay → hospital admission → diagnosis |

They have no released labels, so they skip the agreement gate. `tests/test_question_generation.py` checks them on a synthetic record.

Safeguards:

- **Agreement gate.** A skeleton is used only if the oracle explains its own label, and its template has at least 90% of supported labels explained (`--min-agreement`).
- **No absolute dates.** Skeletons with absolute dates ("in 11/2161") are skipped.
- **Approximated starts.** Selections that order by an approximated prescription start are not generated.
- **Polarity pairing.** Per (template, patient), at most one question with a non-empty answer is kept, plus one whose answer is empty, 0 or False for a concept present elsewhere in the patient's record.
- **No repeats.** Generated questions that repeat a released question are dropped.
- **Answers follow the agents' stay semantics.** `semantics_sensitive` marks the 149 questions whose static-database answer differs. All are out-of-scope durations; none is in the in-scope samples.

Outputs:

- `final_dataset/generated/fhir_native.csv`: the released columns plus `source_question_id`, `polarity`, `features` and `semantics_sensitive`;
- `manifests/generated/fhir_native.families.csv`;
- `manifests/generated/fhir_native.split.json`: per patient pool, a family-stratified sample of 20 / 50 / 300 in-scope questions (`--split-families` to change), plus each template's agreement;
- `manifests/generated/fhir_native.positive.split.json`: the same pools, positive questions only, taken one template at a time (held out: 300 questions over 60 templates, at most a handful each).

Of the 2,931 released rows, 1,798 serve as skeletons, covering 70 of the 74 supported templates. The bank holds 9,264 questions over all 94 patients and 73 templates:

| Family | Positive | Negative | Total |
| --- | --- | --- | --- |
| lookup_counting | 1,933 | 1,377 | 3,310 |
| numerical_aggregation | 568 | 674 | 1,242 |
| temporal_selection | 1,744 | 1,521 | 3,265 |
| out_of_scope | 639 | 808 | 1,447 |
| **All** | **4,884** | **4,380** | **9,264** |

That includes 163 T001, 144 T051, 93–124 per duration template (T023–T027), and 347 authored questions (77 T076, 118 T077, 152 T078).

Feature tags, for cutting multi-hop subsets:

- `icu_to_hospital` 2,544
- `specimen_link` 1,465
- `medication_mix` 316
- `susceptibility_chain` 195
- `icu_to_diagnosis` 152
- `micro_organism_chain` 109

The balanced held-out sample has 300 questions (127 / 48 / 125), 161 positive and 139 negative.

Procedure slots draw only ICD-coded procedures. The projection also files ED and ICU procedure events under `procedures_icd`, and those names are not EHRSQL procedures. `--resplit` rebuilds both split manifests from an existing bank without regenerating.

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
| `tie_peer` | 2 | 2 | A tie set (T052, T055) keeps its earliest shown tied record; the other shown ones are removed |
| `k_record` | 18 | 18 | The smallest k (2–5) most recent cited records that changes a count, existence check or aggregate |

The runtime keys omissions by question, so each question gets at most one kind, assigned rarest kind first. That leaves 39 entries.

`--verify-manifest` rebuilds every entry from local sources.

Tie-set references are checked over the full record with `RecordQuery`, not over the truncated packet. Each tie entry records `intact_packet_reproduces`. Tied organisms may be removed even though their test references them.

The frozen held-out split still yields only 2 tie sets:

- only 17 of its questions have multi-value answers;
- 4 are min/max/avg ties, not selections;
- 3 fail the full-record check and 2 need microbiology hospital membership;
- the rest show fewer than two tied records in the packet.

More would need perturbations outside the frozen 300, which `load_perturbations` does not allow.

`omit_from_packet` also redacts references to an omitted resource, so a link-target omission hides the link as well as the name.

`tools/perturbation.py` accepts both protocols: version 2 (exactly 50 entries, one `removed` record) and version 3 (any size, a list of `removed` records). `scripts/revisions.py` counts a multi-record omission as retrieved, cited, used or recovered only if every removed record is. Run the recovery arms with the new manifest and a separate output directory:

```bash
python scripts/run_experiments.py run --arms recovery \
    --perturbations manifests/perturbations_multi.json \
    --output_dir output/experiments/<model>/suite_multi_perturbations \
    --max_total_tokens 200000 --max_tool_calls 30
```

This runtime change alters the code fingerprint, so result files written before it now show as stale. `manifests/perturbations.json` was rebuilt for the new omitter and oracle hashes; its 50 entries are unchanged.

## Re-stamping saved results

`scripts/restamp_run_identities.py <output dirs>` moves saved run identities to the current code fingerprint. It prints a dry run unless `--apply` is given. It refuses unless the fingerprinted files differ from `4ecbe0d~1` only in `tools/perturbation.py`, and it re-stamps only identities whose `code_hash` equals that commit's fingerprint in the current environment (same package versions).

A recovery identity also needs its `perturbations_hash` to be the old manifest's, with identical entries in the new one. Each changed file keeps a `.bak` copy, and `restamp_log.json` records what changed.

The planning notes counted 1,400 stale baseline runs. Those are v1-era results produced by different agent code, so they are left alone. Only the multi-agent v2 `held_out_v2` results, whose code hash is `4ecbe0d~1`'s fingerprint, can be re-stamped honestly.

## Graded demo questions (plan 6B)

`demo/app.py --generated <csv>` (default `final_dataset/generated/fhir_native.csv`) adds a "Generated (graded)" question mode for pilot and dev patients. After a run, `demo.backend.grade` scores the released answer with the typed scorer and shows the FHIR-derived reference. Generated questions are never listed or graded for held-out patients.

## Rebuilding

```bash
python scripts/fhir_oracle.py                       # oracle agreement with the released labels (~5 min)
python scripts/build_question_pools.py              # manifests/pools/*.json (~2 min)
python scripts/generate_fhir_questions.py           # final_dataset/generated/fhir_native.csv (~60 min)
python scripts/build_paraphrases.py                 # final_dataset/generated/paraphrases.csv
python scripts/build_multi_perturbations.py         # manifests/perturbations_multi.json (~4 min)
python scripts/build_multi_perturbations.py --verify-manifest manifests/perturbations_multi.json
```

All builders read the local NDJSON under `physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir` and need no FHIR store access. Starting packets for a pool can be frozen with `python scripts/build_starting_packets.py --split <pool manifest> --output <path>`.
