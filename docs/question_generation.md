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

## Suite results (plan 4.2)

All ten suites ran on Compute2 for gpt-oss:120b (vLLM) and qwen3:32b (Ollama) at commit 9da4131, with 200,000 tokens and 30 tool calls per question. qwen3:14b, qwen3:8b and qwen3:4b ran later with the same runtime code (unchanged code fingerprint).

Ollama's `qwen3:4b` tag is not a smaller copy of the other three. It is Qwen3-4B-Thinking-2507, a later reasoning-only release with its own chat template, labelled `qwen3:4b-2507` below. It outscores qwen3:8b and qwen3:14b on most suites and ran 5–7 times slower per question than qwen3:8b. Read its rows as a fifth model, not as the bottom of a size ladder; the size comparison is qwen3:32b, 14b and 8b, which share one release and template. Every question was attempted. Failed runs count as wrong; they are rare except for single-agent on the smaller qwen3 models (see [Failed runs](#failed-runs)). The frozen `held_out_v2` results were re-stamped first (see [Re-stamping](#re-stamping-saved-results)).

Accuracy is the first attempt under the typed scorer. Intervals are 95% patient-clustered bootstraps from `run_experiments.py analyze`. p is the exact McNemar test from `scripts/paired_tests.py`, which treats questions as independent.

### Ordinary arms

| Suite | n | Model | Multi | Single | Direct | Multi − single | p | Multi − direct | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pool_held_out_v2 | 300 | gpt-oss | 75.0% | 68.0% | 63.7% | +7.0 [+1.8, +12.1] | 0.005 | +11.3 [+7.0, +15.5] | 3e-7 |
| | | qwen3:32b | 58.3% | 43.0% | 48.0% | +15.3 [+8.3, +22.2] | 2e-5 | +10.3 [+5.2, +15.3] | 2e-4 |
| | | qwen3:14b | 50.3% | 39.7% | 40.0% | +10.7 [+3.7, +17.3] | 0.004 | +10.3 [+4.1, +16.4] | 2e-4 |
| | | qwen3:8b | 45.7% | 32.0% | 44.3% | +13.7 [+6.6, +20.1] | 2e-4 | +1.3 [−5.2, +7.5] | 0.70 |
| | | qwen3:4b-2507 | 65.7% | 53.0% | 63.3% | +12.7 [+7.2, +18.1] | 2e-5 | +2.3 [−1.4, +6.0] | 0.28 |
| pool_template_balanced | 272 | gpt-oss | 78.3% | 71.0% | 69.9% | +7.4 [+3.8, +11.2] | 0.003 | +8.5 [+3.4, +13.4] | 0.001 |
| | | qwen3:32b | 62.5% | 50.7% | 51.5% | +11.8 [+6.2, +17.5] | 0.001 | +11.0 [+6.3, +15.6] | 1e-4 |
| | | qwen3:14b | 55.1% | 44.5% | 44.9% | +10.7 [+3.6, +17.7] | 0.005 | +10.3 [+3.8, +16.9] | 0.002 |
| | | qwen3:8b | 51.1% | 35.3% | 46.7% | +15.8 [+10.0, +21.8] | 9e-5 | +4.4 [−1.5, +10.5] | 0.17 |
| | | qwen3:4b-2507 | 72.8% | 57.4% | 69.1% | +15.4 [+9.4, +21.1] | 4e-6 | +3.7 [−1.5, +8.9] | 0.22 |
| pool_zero_finisher | 300 | gpt-oss | 82.7% | 78.0% | 75.3% | +4.7 [+0.3, +9.5] | 0.07 | +7.3 [+3.6, +11.0] | 3e-4 |
| | | qwen3:32b | 64.0% | 55.0% | 58.0% | +9.0 [+1.2, +16.9] | 0.01 | +6.0 [+1.1, +10.5] | 0.03 |
| | | qwen3:14b | 58.0% | 47.0% | 51.7% | +11.0 [+3.0, +19.3] | 0.004 | +6.3 [+1.1, +11.8] | 0.03 |
| | | qwen3:8b | 56.0% | 38.0% | 53.7% | +18.0 [+9.5, +27.2] | 1e-6 | +2.3 [−3.3, +7.8] | 0.48 |
| | | qwen3:4b-2507 | 72.0% | 63.0% | 72.0% | +9.0 [+3.7, +14.4] | 0.002 | +0.0 [−3.8, +4.0] | 1 |
| pool_truncation | 161 | gpt-oss | 57.8% | 49.7% | 40.4% | +8.1 [+1.4, +14.8] | 0.02 | +17.4 [+10.4, +23.9] | 6e-8 |
| | | qwen3:32b | 47.2% | 21.1% | 41.0% | +26.1 [+19.2, +34.2] | 5e-10 | +6.2 [+1.1, +11.9] | 0.05 |
| | | qwen3:14b | 46.6% | 18.0% | 40.4% | +28.6 [+19.4, +37.3] | 8e-10 | +6.2 [+0.0, +12.0] | 0.04 |
| | | qwen3:8b | 37.3% | 7.5% | 39.1% | +29.8 [+22.5, +37.9] | 3e-12 | −1.9 [−6.8, +3.4] | 0.65 |
| | | qwen3:4b-2507 | 43.5% | 28.0% | 39.8% | +15.5 [+8.9, +23.3] | 6e-8 | +3.7 [+0.6, +8.1] | 0.03 |
| pool_disagreement | 99 | gpt-oss | 77.8% | 70.7% | 74.7% | +7.1 [+1.9, +13.0] | 0.07 | +3.0 [−3.2, +9.9] | 0.58 |
| | | qwen3:32b | 59.6% | 40.4% | 66.7% | +19.2 [+8.5, +29.4] | 6e-4 | −7.1 [−21.6, +6.4] | 0.31 |
| | | qwen3:14b | 57.6% | 36.4% | 56.6% | +21.2 [+10.1, +33.3] | 5e-4 | +1.0 [−8.7, +10.6] | 1 |
| | | qwen3:8b | 49.5% | 23.2% | 67.7% | +26.3 [+14.1, +38.5] | 7e-5 | −18.2 [−30.4, −6.2] | 0.005 |
| | | qwen3:4b-2507 | 82.8% | 63.6% | 76.8% | +19.2 [+8.3, +30.1] | 9e-4 | +6.1 [−2.0, +14.3] | 0.29 |
| pool_out_of_scope | 213 | gpt-oss | 78.9% | 77.5% | 73.2% | +1.4 [−3.3, +6.9] | 0.68 | +5.6 [+0.9, +11.1] | 0.04 |
| | | qwen3:32b | 47.4% | 42.7% | 23.5% | +4.7 [−4.3, +14.8] | 0.26 | +23.9 [+17.3, +31.8] | 2e-9 |
| | | qwen3:14b | 45.1% | 37.6% | 12.7% | +7.5 [−5.1, +20.7] | 0.12 | +32.4 [+22.7, +42.7] | 8e-15 |
| | | qwen3:8b | 28.2% | 20.2% | 14.1% | +8.0 [−2.6, +19.6] | 0.05 | +14.1 [+8.0, +20.4] | 3e-6 |
| | | qwen3:4b-2507 | 64.3% | 58.7% | 66.7% | +5.6 [−4.9, +15.8] | 0.15 | −2.3 [−7.7, +3.5] | 0.47 |
| paraphrases | 134 | gpt-oss | 65.7% | 44.0% | 44.0% | +21.6 [+13.6, +29.0] | 5e-6 | +21.6 [+10.8, +32.4] | 5e-6 |
| | | qwen3:32b | 41.0% | 22.4% | 34.3% | +18.7 [+6.5, +30.1] | 5e-4 | +6.7 [+0.8, +13.5] | 0.05 |
| | | qwen3:14b | 30.6% | 21.6% | 35.1% | +9.0 [−0.8, +18.5] | 0.07 | −4.5 [−14.9, +6.6] | 0.39 |
| | | qwen3:8b | 31.3% | 20.1% | 35.8% | +11.2 [+2.6, +20.5] | 0.02 | −4.5 [−12.3, +3.3] | 0.31 |
| | | qwen3:4b-2507 | 52.2% | 27.6% | 40.3% | +24.6 [+17.2, +32.1] | 2e-9 | +11.9 [+6.2, +17.9] | 0.002 |
| generated | 300 | gpt-oss | 83.3% | 73.7% | 77.0% | +9.7 [+4.8, +14.8] | 2e-4 | +6.3 [+3.1, +9.6] | 0.004 |
| | | qwen3:32b | 58.0% | 56.0% | 51.3% | +2.0 [−4.8, +8.7] | 0.63 | +6.7 [+0.6, +12.7] | 0.03 |
| | | qwen3:14b | 58.0% | 55.0% | 51.0% | +3.0 [−3.8, +9.8] | 0.46 | +7.0 [+2.2, +12.3] | 0.008 |
| | | qwen3:8b | 54.7% | 41.0% | 48.0% | +13.7 [+5.2, +22.1] | 4e-4 | +6.7 [+1.4, +12.2] | 0.02 |
| | | qwen3:4b-2507 | 72.7% | 60.0% | 72.3% | +12.7 [+6.3, +19.4] | 8e-5 | +0.3 [−4.0, +4.6] | 1 |
| generated_positive | 300 | gpt-oss | 80.3% | 78.3% | 76.3% | +2.0 [−1.3, +5.4] | 0.42 | +4.0 [+0.3, +7.8] | 0.10 |
| | | qwen3:32b | 67.7% | 51.3% | 66.3% | +16.3 [+12.1, +20.5] | 7e-10 | +1.3 [−3.2, +5.9] | 0.71 |
| | | qwen3:14b | 63.3% | 48.7% | 58.0% | +14.7 [+9.2, +20.1] | 1e-5 | +5.3 [+0.0, +10.9] | 0.05 |
| | | qwen3:8b | 61.3% | 30.3% | 60.3% | +31.0 [+24.4, +38.0] | 1e-19 | +1.0 [−4.5, +6.6] | 0.81 |
| | | qwen3:4b-2507 | 71.0% | 62.0% | 68.3% | +9.0 [+4.0, +13.8] | 0.001 | +2.7 [−1.7, +7.0] | 0.30 |

- **Multi-agent wins on released questions it was not tuned on.** On the unsampled pools (held_out_v2, template_balanced, zero_finisher, truncation), multi-agent beats single-agent for every model, and every interval excludes zero. It also beats direct for gpt-oss, qwen3:32b and qwen3:14b; for qwen3:8b and qwen3:4b-2507 it mostly ties direct.
- **The gain does not come from the finishers.** On pool_zero_finisher, where no finisher template is asked, multi-agent still gains 4.7 (gpt-oss), 9.0 (qwen3:32b), 11.0 (qwen3:14b), 18.0 (qwen3:8b) and 9.0 (qwen3:4b-2507) over single-agent.
- **On qwen3:8b, single-agent collapses rather than multi-agent improving.** Its multi-agent lead over single-agent is +11.2 to +31.0 on every suite but out_of_scope, but multi-agent only ties direct on most suites. Part of the collapse is failed runs: 17% of qwen3:8b single-agent runs end in an empty reply or other failure, which count as wrong. qwen3:14b sits close to qwen3:32b.
- **Truncated packets favour multi-agent most.** Multi-agent fetches the record beyond the packet. Single-agent drops to 21% (qwen3:32b), 18% (14b), 7.5% (8b) and 28% (4b-2507) on these questions while multi-agent keeps 37–47%.
- **On the disagreement pool, qwen3 direct matches or beats multi-agent.** Direct scores 66.7% (32b) and 67.7% (8b), against 59.6% and 49.5% for multi-agent; on 14b and 4b-2507 the difference is not significant. qwen3:4b-2507 multi-agent scores 82.8% here, the highest of any model.
- **On out-of-scope questions, the agents tie.** Neither multi-agent difference from single-agent is significant. The intervals are wide because the 213 questions cluster in 53 patients.
- **On generated questions, gpt-oss and qwen3 disagree.** gpt-oss's multi-agent gain is on negative (empty-answer) questions; on the positive-only suite it shrinks to +2.0 (p=0.42). qwen3:32b, 14b and 8b show the reverse: a large gain on the positive-only suite (+14.7 to +31.0) and a loss on negative questions. qwen3:4b-2507 gains on both (+9.0 positive-only). See [Generated breakdown](#generated-breakdown).

### Paraphrase ablation

The 67 frozen held-out finisher questions, as released (from `held_out_v2`), as routed paraphrases (the finisher still fires) and as unrouted paraphrases (no finisher). The released column comes from an earlier batch, so the original-vs-paraphrase comparisons also include batch-to-batch variation; routed vs unrouted ran in the same batch. Only gpt-oss and qwen3:32b have a `held_out_v2` run, so the other models have no original column.

| Model | Arm | Original | Routed | Unrouted | Routed − unrouted (p) |
| --- | --- | --- | --- | --- | --- |
| gpt-oss | multi | 53.7% | 64.2% | 67.2% | −2 of 67 (0.79) |
| | single | 41.8% | 40.3% | 47.8% | −5 of 67 (0.30) |
| | direct | 37.3% | 41.8% | 46.3% | −3 of 67 (0.51) |
| qwen3:32b | multi | 41.8% | 43.3% | 38.8% | +3 of 67 (0.55) |
| | single | 16.4% | 17.9% | 26.9% | −6 of 67 (0.18) |
| | direct | 25.4% | 32.8% | 35.8% | −2 of 67 (0.69) |
| qwen3:14b | multi | — | 26.9% | 34.3% | −5 of 67 (0.36) |
| | single | — | 20.9% | 22.4% | −1 of 67 (1) |
| | direct | — | 31.3% | 38.8% | −5 of 67 (0.06) |
| qwen3:8b | multi | — | 31.3% | 31.3% | 0 of 67 (1) |
| | single | — | 17.9% | 22.4% | −3 of 67 (0.45) |
| | direct | — | 34.3% | 37.3% | −2 of 67 (0.69) |
| qwen3:4b-2507 | multi | — | 55.2% | 49.3% | +4 of 67 (0.45) |
| | single | — | 22.4% | 32.8% | −7 of 67 (0.07) |
| | direct | — | 34.3% | 46.3% | −8 of 67 (0.06) |

- **Removing the finisher cue does not hurt.** No arm loses accuracy when the finisher stops firing. The only significant change is qwen3:32b single-agent, which scores higher on unrouted than on original wording (+7 of 67, p=0.039).
- **The multi-agent lead survives without finishers.** Multi-agent minus single-agent on unrouted paraphrases is +13 of 67 for gpt-oss (p=0.004), +8 for qwen3:32b (p=0.08), +8 for qwen3:14b (p=0.13), +6 for qwen3:8b (p=0.24) and +11 for qwen3:4b-2507 (p=0.003). On routed paraphrases it is +16 (p=0.0009), +17 (p=0.005), +4 (p=0.45), +9 (p=0.06) and +22 (p=5e-7).
- **On qwen3:14b and qwen3:8b, multi-agent no longer beats direct on paraphrases.** Direct scores 35–36% on qwen3:14b and qwen3:8b, against 31% for multi-agent.
- **Wording alone moves accuracy by several points.** Original-vs-paraphrase differences of 5–10 points (none significant at 67 questions) are a floor on how much these results depend on phrasing.

### Generated breakdown

`generated` mixes 161 positive and 139 negative questions; `generated_positive` has 300 positive ones. No sampled question is semantics-sensitive.

| Model | Subset | n | Multi | Single | Direct | Multi − single (p) |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-oss | negative | 139 | 74.1% | 58.3% | 68.3% | +22 of 139 (7e-4) |
| | positive | 161 | 91.3% | 87.0% | 84.5% | +7 of 161 (0.14) |
| qwen3:32b | negative | 139 | 42.4% | 52.5% | 23.7% | −14 of 139 (0.12) |
| | positive | 161 | 71.4% | 59.0% | 75.2% | +20 of 161 (0.002) |
| qwen3:14b | negative | 139 | 40.3% | 55.4% | 30.2% | −21 of 139 (0.013) |
| | positive | 161 | 73.3% | 54.7% | 68.9% | +30 of 161 (4e-5) |
| qwen3:8b | negative | 139 | 40.3% | 46.8% | 22.3% | −9 of 139 (0.34) |
| | positive | 161 | 67.1% | 36.0% | 70.2% | +50 of 161 (1e-11) |
| qwen3:4b-2507 | negative | 139 | 64.0% | 56.1% | 66.9% | +11 of 139 (0.14) |
| | positive | 161 | 80.1% | 63.4% | 77.0% | +27 of 161 (4e-5) |

By relation feature (positive-only suite), multi / single / direct:

| Feature | n | gpt-oss | qwen3:32b | qwen3:14b | qwen3:8b | qwen3:4b-2507 |
| --- | --- | --- | --- | --- | --- | --- |
| none | 116 | 93.1 / 90.5 / 79.3 | 72.4 / 59.5 / 70.7 | 67.2 / 50.9 / 63.8 | 66.4 / 28.4 / 63.8 | 78.4 / 67.2 / 74.1 |
| icu_to_hospital | 71 | 76.1 / 74.6 / 76.1 | 78.9 / 53.5 / 67.6 | 64.8 / 60.6 / 49.3 | 63.4 / 39.4 / 59.2 | 69.0 / 69.0 / 69.0 |
| specimen_link | 69 | 76.8 / 78.3 / 81.2 | 69.6 / 56.5 / 62.3 | 65.2 / 55.1 / 63.8 | 59.4 / 29.0 / 59.4 | 75.4 / 68.1 / 68.1 |
| medication_mix | 17 | 52.9 / 52.9 / 47.1 | 35.3 / 35.3 / 47.1 | 58.8 / 23.5 / 41.2 | 41.2 / 23.5 / 41.2 | 47.1 / 17.6 / 47.1 |
| micro_organism_chain, specimen_link | 12 | 83.3 / 50.0 / 66.7 | 66.7 / 16.7 / 66.7 | 66.7 / 8.3 / 33.3 | 66.7 / 16.7 / 50.0 | 66.7 / 25.0 / 50.0 |
| … + susceptibility_chain | 10 | 20.0 / 80.0 / 60.0 | 0.0 / 0.0 / 50.0 | 0.0 / 10.0 / 50.0 | 10.0 / 40.0 / 60.0 | 10.0 / 60.0 / 50.0 |
| icu_to_diagnosis | 5 | 100 / 0 / 100 | 20 / 0 / 100 | 60 / 0 / 100 | 100 / 0 / 100 | 80 / 0 / 80 |

- **Susceptibility chains are a multi-agent weakness.** On gpt-oss, multi-agent gets 2 of 10 right where single-agent gets 8 (all 6 discordant favour single-agent, p=0.031). Multi-agent gets at most 1 of 10 on every qwen3 model, while direct gets 5 or 6.
- **ICU-to-diagnosis questions defeat single-agent.** Single-agent gets 0 of 5 with every model. The cells are small.
- **Medication mixes are hard for every arm** (18–59%).

### Multi-record perturbations

39 held-out questions, each run from the intact and from the omitted packet (`manifests/perturbations_multi.json`). Recovery flags come from `run_experiments.py revisions` without evidence fetching, so `cited` and `used` are lower bounds.

| Model | System | Intact | Omitted | Omitted − intact | Omitted correct given intact correct |
| --- | --- | --- | --- | --- | --- |
| gpt-oss | multi | 79.5% | 74.4% | −5.1 [−14.3, +5.3] | 28/31 (90%) |
| | single | 66.7% | 46.2% | −20.5 [−38.5, −2.7] | 15/26 (58%) |
| qwen3:32b | multi | 53.8% | 48.7% | −5.1 [−23.1, +11.1] | 15/21 (71%) |
| | single | 28.2% | 10.3% | −17.9 [−32.5, −2.9] | 3/11 (27%) |
| qwen3:14b | multi | 53.8% | 25.6% | −28.2 [−44.7, −12.5] | 7/21 (33%) |
| | single | 35.9% | 12.8% | −23.1 [−40.0, −8.3] | 5/14 (36%) |
| qwen3:8b | multi | 56.4% | 30.8% | −25.6 [−42.4, −10.3] | 10/22 (45%) |
| | single | 28.2% | 7.7% | −20.5 [−35.7, −7.7] | 2/11 (18%) |
| qwen3:4b-2507 | multi | 89.7% | 48.7% | −41.0 [−60.0, −22.2] | 18/35 (51%) |
| | single | 43.6% | 7.7% | −35.9 [−53.1, −19.4] | 3/17 (18%) |

By kind, omitted given intact correct, multi / single:

| Kind | n | gpt-oss | qwen3:32b | qwen3:14b | qwen3:8b | qwen3:4b-2507 |
| --- | --- | --- | --- | --- | --- | --- |
| k_record | 18 | 15/17 / 7/15 | 8/12 / 3/10 | 4/11 / 4/10 | 4/11 / 2/9 | 4/16 / 0/12 |
| link_target | 19 | 11/12 / 8/10 | 7/9 / 0/1 | 3/8 / 1/4 | 5/10 / 0/2 | 13/17 / 3/5 |
| tie_peer | 2 | 2/2 / 0/1 | 0/0 / 0/0 | 0/2 / 0/0 | 1/1 / 0/0 | 1/2 / 0/0 |

- **On the larger models, multi-agent tolerates missing records; single-agent does not.** gpt-oss and qwen3:32b multi-agent lose 5 points when several records are removed; single-agent loses 18–21. Omitted-packet multi − single is +28.2 [+12.5, +43.9] (gpt-oss) and +38.5 [+20.6, +56.5] (qwen3:32b).
- **On qwen3:14b, qwen3:8b and qwen3:4b-2507, multi-agent loses as much as single-agent.** qwen3:4b-2507 has the best intact-packet multi-agent score of any model (89.7%) and still loses 41 points; it keeps 13 of 17 link-target questions but only 4 of 16 k-record ones. For 14b and 8b: multi-agent drops 26–28 points, and keeps only a third to a half of the questions it got right from the intact packet. Its omitted-packet lead over single-agent (+12.8 [−8.3, +30.8] on 14b, +23.1 [+10.3, +37.5] on 8b) is no larger than its intact-packet lead (+17.9 and +28.2).
- **Removing the most recent k records hurts single-agent most.** Single-agent keeps 7 of 15 (gpt-oss) where multi-agent keeps 15 of 17.
- **The mechanism is the full-record fetch, and smaller models fetch without using it.** Multi-agent fetched the omitted records in 39 of 39 runs for every model; gpt-oss and qwen3:32b single-agent did in about half, qwen3:4b-2507 single-agent in 4 of 25 runs with the metadata. Multi-agent cites them less often (9/39 for gpt-oss, 3/39 for qwen3:32b and 4b-2507, 2/39 for 14b and 8b), so it often answers right without citing what it recovered. qwen3:14b, qwen3:8b and qwen3:4b-2507 fetch as reliably but answer right only 26–49% of the time afterwards.
- **Two tie sets are too few to conclude anything.**

### Failed runs

Failed first attempts count as wrong, and they fall mostly on single-agent and direct, so they add to multi-agent's lead. `python scripts/suite_breakdowns.py <model dir> --only errors` counts them by kind for every suite and arm.

| Model | Arm | Failed (all ordinary suites) | Main kinds |
| --- | --- | --- | --- |
| gpt-oss | multi | 7 of 2,079 | input too long 5, empty reply 2 |
| | single | 52 of 2,079 | invalid output 50 |
| | direct | 19 of 2,079 | invalid output |
| qwen3:32b | multi | 7 of 2,079 | unusable analysis draft 6 |
| | single | 96 of 2,079 | invalid output 60, empty reply 32 |
| | direct | 61 of 2,079 | invalid output |
| qwen3:14b | multi | 5 of 2,079 | empty reply 3, timeout 2 |
| | single | 52 of 2,079 | invalid output 35, empty reply 12 |
| | direct | 5 of 2,079 | timeout 4 |
| qwen3:8b | multi | 12 of 2,079 | empty reply 9 |
| | single | 362 of 2,079 | empty reply 324, invalid output 22, timeout 13 |
| | direct | 19 of 2,079 | invalid output 13 |
| qwen3:4b-2507 | multi | 0 of 2,079 | |
| | single | 20 of 2,079 | invalid output 15, empty reply 3 |
| | direct | 82 of 2,079 | invalid output |

- **"Invalid output" is a final answer that failed the output schema**, for example evidence without `resource_id` or a non-JSON reply.
- **An "empty reply" is usually a tool call that was not valid JSON.** Replaying the qwen3:8b single-agent failures through Ollama's own template, with raw generation, showed the model writing Python with double quotes inside the `"code"` string (`f'Encounter/{enc.get("id")}'`) without escaping them. Ollama's tool-call parser cannot read the call and returns an empty message, with no error, although it generated the tokens. `safe_llm_call` then asks again with "I got an empty response", which does not say what was wrong, and the model usually repeats the call until its retries run out. Ollama returns an HTTP error for gpt-oss in the same situation, which `_rejected_tool_call` already turns into a specific correction.
- **qwen3:8b single-agent is hit hardest.** Empty replies end 324 of its 2,079 runs (16%), against 32 for qwen3:32b, 12 for qwen3:14b and 9 for qwen3:8b multi-agent, so the effect is not simply proportional to model size. qwen3:4b-2507, with its different template and reasoning, has only 3. On qwen3:8b these failures are a sizeable part of single-agent's deficit; a fairer retry message would likely narrow multi-agent's lead there. The runtime was left unchanged so that every model runs the same code. Fixing the retry message would change the code fingerprint, so it belongs in a round that reruns every model.

### Cost

Mean per question, including retries, on pool_held_out_v2. `cost` is not recorded for local models, so GPU time is approximated by latency.

| Model | Arm | Tokens | Prompt share | Model calls | Tool calls | Median latency |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-oss | multi | 79.3k | 94% | 11.8 | 11.9 | 37 s |
| | single | 33.0k | 96% | 3.4 | 3.4 | 9.5 s |
| | direct | 6.6k | 90% | 1.0 | 0 | 4.4 s |
| qwen3:32b | multi | 75.1k | 97% | 10.1 | 9.8 | 142 s |
| | single | 44.2k | 98% | 4.1 | 4.1 | 42 s |
| | direct | 7.6k | 95% | 1.0 | 0 | 17 s |

- **Multi-agent costs 1.7–2.4× single-agent's tokens and 3–4× its latency.** The ratio holds across suites (2.1–3.5× tokens on gpt-oss). Truncation and paraphrase questions are the most expensive: 88–117k tokens for multi-agent.
- **Almost all of it is re-sent prompt.** Prefix caching or a smaller hand-off between agents would remove most of the overhead.

### Component ablations

Multi-agent with one component removed at a time, on gpt-oss and qwen3:32b, on 100 questions from the `generated` suite: 46 negative and 54 positive, from 51 patients, stratified by polarity with seed 5106 (`manifests/generated/fhir_native.ablation_subset.json`). The variants are subclasses in `scripts/run_ablations.py`, so the code fingerprint is unchanged and `full` is an ordinary multi-agent run. gpt-oss finished all 500 runs; qwen3:32b failed 3 (one each in full, no_prefetch and no_verifier), counted as wrong. Intervals are patient-clustered bootstraps; p values come from exact McNemar tests against `full`. Each cell gives accuracy, the difference from full with its interval, gained / lost questions and p.

| Variant | What is removed | gpt-oss | qwen3:32b | Mean tokens (gpt-oss / qwen3:32b) |
| --- | --- | --- | --- | --- |
| full | nothing | 87.0% [80.6, 92.7] | 69.0% [60.6, 77.0] | 67.1k / 64.1k |
| no_prefetch | retrieval's prefetch of the complete records | 86.0%; −1.0 [−4.3, +2.2]; 2 / 3; p=1 | 68.0%; −1.0 [−6.2, +4.2]; 4 / 5; p=1 | 75.3k / 68.2k |
| no_packet | the starting packet shown to analysis | 82.0%; −5.0 [−10.8, +0.0]; 2 / 7; p=0.18 | 64.0%; −5.0 [−11.4, +1.1]; 4 / 9; p=0.27 | 87.1k / 49.1k |
| no_guidance | search guidance, semantic guidance and worked examples in analysis's prompt (14.7k → 2.2k characters) | 81.0%; −6.0 [−12.6, +0.0]; 1 / 7; p=0.07 | 64.0%; −5.0 [−10.2, +0.0]; 1 / 6; p=0.12 | 60.6k / 66.1k |
| no_verifier | the verifier, so no repair and no verifier vote | 86.0%; −1.0 [−5.9, +4.3]; 3 / 4; p=1 | 68.0%; −1.0 [−5.6, +3.2]; 2 / 3; p=1 | 44.8k / 41.6k |

The release step records every candidate answer, so the draft can also be scored on its own from the same runs. An abstention (a null answer) is scored like any other answer, so it counts as right on a negative question whose answer is that nothing matches.

| Model | Questions | Released | Draft | Draft, else repaired | Released − draft, gained / lost (p) | Repairs | Verifier answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss | ablation subset (100) | 87.0% | 89.0% | 89.0% | −2.0, 0 / 2 (0.5) | 6 | 30 (26 right) |
| | whole `generated` suite (300) | 83.3% | 84.0% | 84.0% | −0.7, 3 / 5 (0.73) | 13 | 66 |
| qwen3:32b | ablation subset (100) | 69.0% | 77.0% | 81.0% | −8.0, 3 / 11 (0.057) | 30 (22 right) | 77 (56 right) |
| | whole `generated` suite (300) | 58.0% | 66.3% | 69.7% | −8.3, 9 / 34 (2e-4) | 97 (60 right) | 246 (155 right) |

Tracing the 43 questions of the qwen3:32b suite where the released answer and the draft disagree (`release.winner` and `release.finisher` in each run's revision):

- **31 of the 34 losses are negative questions where the draft correctly abstained and the packet reading answered anyway.** Abstentions do not vote, so the packet reading wins alone (29), or with the repaired answer (1), or the repaired answer wins alone (1). gpt-oss loses 5 questions the same way.
- **The other 3 losses are positive questions** where the repaired answer, with the verifier or the packet reading, outvoted a correct draft.
- **The 9 gains are 4 positive questions where the draft abstained and the packet reading was right, and 5 where the finisher replaced a wrong answer.** The finisher never turned a right answer wrong.
- Letting an abstention vote, so that the draft wins its tie with a lone packet reading, would have recovered about 30 of these questions and given up the 4 gains: about +26 (to roughly 67%) on qwen3:32b and +3 on gpt-oss. This is an estimate from the recorded candidates, not a run; the change would alter the code fingerprint, so it belongs in a round that reruns every model.

Findings:

- **Analysis's prompt and its packet carry the system on both models.** Removing the guidance and worked examples costs 5–6 points, and hiding the packet costs 5. None of these differences is significant with 100 questions, but most discordant questions go against the ablation on both models.
- **The verifier adds nothing measurable on generated questions.** Removing it costs 1 point on both models and saves a third of the tokens. It does not explain qwen3:32b's release loss either, because without the verifier the packet reading still outvotes an abstaining draft. The multi-record perturbations above are where verification might still matter, and they were not ablated.
- **The release vote costs qwen3:32b 8 points on generated questions**, almost entirely through abstentions that cannot vote (above). On gpt-oss, whose packet reading rarely answers a negative question, the vote costs under a point.
- **Prefetching the complete records makes no difference here.** Without it, retrieval asks for the records itself and uses more tokens.
- **Only differences of about 6 points or more are detectable at this size**, so "no difference" means no large one. These results are for generated questions only.

### Reproducing

On the cluster, from `~/fhir-ma-v2` (vLLM runs use `--base_url http://127.0.0.1:28431/v1`; Ollama runs use `:11434/v1` for qwen3:32b, `:11790/v1` for qwen3:14b, `:11816/v1` for qwen3:8b and `:11636/v1` for qwen3:4b; the URL is part of each saved run's identity):

```bash
python scripts/run_experiments.py analyze --arms ordinary --model qwen3:32b \
    --base_url http://127.0.0.1:11434/v1 --max_total_tokens 200000 --max_tool_calls 30 \
    --split_manifest manifests/pools/truncation.json \
    --output_dir output/experiments/qwen3:32b/suite_pool_truncation
python scripts/paired_tests.py output/experiments/qwen3:32b/suite_pool_truncation \
    --split_manifest manifests/pools/truncation.json
# Recovery flags for the multi-record suite
python scripts/run_experiments.py revisions --arms recovery \
    --perturbations manifests/perturbations_multi.json --model qwen3:32b \
    --base_url http://127.0.0.1:11434/v1 --max_total_tokens 200000 --max_tool_calls 30 \
    --output_dir output/experiments/qwen3:32b/suite_multi_perturbations
# Paraphrase, generated and perturbation-kind breakdowns
python scripts/suite_breakdowns.py output/experiments/qwen3:32b
```

The generated and paraphrase suites also take their `--input`, `--families` and `--split_manifest` (see `scripts/submit_question_suites.sh`).

The component ablations run as one job per model and are analyzed with the same flags:

```bash
RUNNER=scripts/run_ablations.py sbatch --job-name=ablations-gpt-oss scripts/run_experiments_vllm.sbatch
RUNNER=scripts/run_ablations.py OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
    sbatch --job-name=ablations-qwen3-32b scripts/run_experiments_qwen.sbatch
python scripts/run_ablations.py analyze --model gpt-oss:120b \
    --base_url http://127.0.0.1:28431/v1 --max_total_tokens 200000 --max_tool_calls 30
```

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
