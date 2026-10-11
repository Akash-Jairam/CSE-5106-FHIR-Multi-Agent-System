# Phase 1 run log: environment setup, smoke tests, and cost analysis

Issue: [#4 P1-01: Environment & FHIR store setup](https://github.com/Akash-Jairam/CSE-5106-FHIR-Multi-Agent-System/issues/4)
Runs performed: 2026-10-07 to 2026-10-08

## Summary

- The MIMIC-IV FHIR demo v2.1.0 is loaded into a GCP FHIR store: 928,935 resources and 100 patients.
- `scripts/check_setup.py` passes all checks.
- The single-turn resource agent runs end to end on 5 questions, but cannot answer most of the dataset. Its prompts are capped at 32k tokens, and 69% of questions need more than that.
- The multi-turn code agent keeps patient data out of the prompt, so every question can be attempted. With `gpt-5-nano` it scored **15/20 (75%)** on a random held-out set, at about **$0.005 per question**. A full 2,931-question run would cost about $15 at standard prices, or about $8 with the Batch API.
- An experimental `multi_turn_code_resource_v2` agent adds guidance about the data layout. It scored **14/20**, which is no measurable accuracy difference from the baseline, but it was about 35% cheaper.

## 1. Environment

| Item | Value |
|---|---|
| GCP project | `gen-lang-client-0053799010` |
| Location | `us-central1` |
| Dataset / FHIR store | `mimic-fhir` / `mimic-fhir` (R4, `--enable-update-create`) |
| Source bucket | `gs://akashj-mimic-fhir-demo/mimic-fhir/` (30 NDJSON files) |
| Source data | [MIMIC-IV Clinical Database Demo on FHIR v2.1.0](https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/) |
| Python | 3.12 venv with `requirements.txt` |

### FHIR import

| Metric | Value |
|---|---|
| Import operation | `18435717290939908097` |
| Started / finished (UTC) | 2026-10-08 11:51:47 / 12:06:29 (about 15 min) |
| Resources imported | 928,935 (0 failures) |

### `scripts/check_setup.py` result

| Check | Result |
|---|---|
| Config (API key and FHIR IDs, not placeholders) | ✅ |
| Google Application Default Credentials | ✅ |
| Patient resources | ✅ 100 (expected 100) |
| Smoke patient `dd2bf984-33c3-5874-8f68-84113327877e` | ✅ found |
| Encounter / Observation / MedicationRequest | ✅ 637 / ~536k–624k* / 17,552 |

\*The Observation total reported by the store varied between runs (536,165–624,459). The `total` field in Cloud Healthcare API search results is an estimate.

### Setup issues encountered

| Issue | Resolution |
|---|---|
| `wget` not installed on macOS | `setup_fhir_store.sh` downloads with `curl` |
| PhysioNet download timed out on the 547 MB chartevents file | `curl --retry 10 --retry-all-errors -C -` (resume) |
| First `CreateDataset` operation stalled for over 20 min in a freshly enabled project | Completed by itself overnight. No retry needed. |
| `gcloud storage rsync` re-uploaded everything and stalled | Replaced with `gcloud storage cp --no-clobber` |
| `Patient?_summary=count` is rejected by Cloud Healthcare API | `check_setup.py` uses `?_count=1` and reads `total` |
| `gemini-2.5-flash` is "no longer available to new users" | Use `gemini-3.8-flash` or `gemini-3.5-flash-lite` |
| Gemini key: prepaid credits depleted. OpenAI key: no credits, until credit was added. | Added OpenAI credit |
| `evaluation_metrics.py` stops at an interactive y/N prompt when results already exist | `run_smoke_test.sh` passes `--force` |

## 2. Smoke test: the first 5 questions

Command: `bash scripts/run_smoke_test.sh`, i.e. `--limit 5` on `final_dataset/questions_answers_sql_fhir.csv`. All runs were graded by `evaluation_metrics.py` with `o4-mini`.

| Agent | Model | Correct | Input tokens | Output tokens | Cost (litellm) |
|---|---|---|---|---|---|
| `single_turn_resource` | `o4-mini` | 0/5 | 20,746 | 3,232 | $0.0370 |
| `multi_turn_code_resource` | `gpt-5-nano` | 1/5 | 70,684 | 40,111 | $0.0175 |
| `multi_turn_code_resource` | `o4-mini` | 1/5 | 35,716 | 11,707 | $0.0666 |
| `multi_turn_code_resource_v2` | `gpt-5-nano` | 2/5 | 58,765 | 48,658 | $0.0203 |

Answers per question:

| # | Question | True answer | `single_turn_resource` / `o4-mini` | `multi_turn_code_resource` / `gpt-5-nano` | `multi_turn_code_resource` / `o4-mini` | `multi_turn_code_resource_v2` / `gpt-5-nano` |
|---|---|---|---|---|---|---|
| 1 | When was the first time the respiratory rate of patient 10018081 was measured to be less than 23.0 today? | `[['2133-12-31 02:00:00']]` | ❌ Input tokens exceeded: 7421703 > 32000 | ✅ 2133-12-31T02:00:00 | ✅ 2133-12-31T02:00:00-05:00 | ✅ The first time today that the respiratory rate was measured below 23.0… |
| 2 | Did patient 10029291 undergo a ultrasonography of superior vena cava, guidance treatment in the first hospital visit? | `[[1]]` | ❌ I’m sorry, but I can’t answer that. I was unable to find the requested… | ❌ No, there is no record of ultrasonography of the superior vena cava du… | ❌ No. The patient did not undergo an ultrasonography of superior vena ca… | ❌ No ultrasonography of the superior vena cava with guidance treatment w… |
| 3 | Has docusate sodium (liquid), glucose gel, or heparin been prescribed since 2157 to patient 10036156? | `[[1]]` | ❌ I looked through all MedicationRequest entries authored during 2157 (f… | ❌ No prescriptions for docusate sodium (liquid), glucose gel, or heparin… | ❌ None of docusate sodium (liquid), glucose gel, or heparin have been pr… | ❌ Docusate sodium (liquid): No records found since 2157. Glucose gel: No… |
| 4 | Show me patient 10014354's total void output amount since 08/11/2146. | `[[3075]]` | ❌ Input tokens exceeded: 6761882 > 32000 | ❌ 0.0 | ❌ 0.0 | ❌ 26060.0 |
| 5 | How frequently was milk of magnesia prescribed to patient 10005348 in 09/this year? | `[[1]]` | ❌ Input tokens exceeded: 62198 > 32000 | ❌ 0 prescriptions of milk of magnesia for the patient in September 2128. | ❌ 0 | ✅ Milk of Magnesia was prescribed 1 time to patient 3886cafb-65f4-5789-9… |

Notes:
- The earlier Gemini smoke runs failed at the API level (model retired, then credits depleted), so they produced no answers. They are not included above.
- An earlier version of the v2 prompt scored 3/5 (Q1, Q2 and Q3 correct). After the prompt was revised (see §4) it scored 2/5. These first 5 questions turned out to be much harder than average; compare the held-out results in §3.

### Why the single-turn agent fails

`single_turn_resource` pastes every retrieved resource into the prompt. `safe_llm_call()` refuses any prompt over 32,000 tokens, and a single patient's Observations can reach 7–16M tokens.

## 3. Held-out comparison: 20 random questions

The questions were sampled with `df.iloc[5:].sample(20, random_state=42)` from `final_dataset/questions_answers_sql_fhir.csv`, which excludes the 5 smoke questions. Both agents ran with `gpt-5-nano`, and answers were graded with `o4-mini`.

| | `multi_turn_code_resource` (baseline) | `multi_turn_code_resource_v2` |
|---|---|---|
| Correct | **15/20 (75%)** | 14/20 (70%) |
| Avg. model calls per question | 5.4 | 3.9 |
| Input / output tokens | 537,919 / 195,184 | 276,562 / 137,869 |
| Cost at list price ($0.05 in, $0.40 out per 1M tokens) | $0.105 ($0.0052/question) | $0.069 ($0.0034/question) |
| Cost reported by litellm | $0.087 | $0.060 |

Answers per question:

| # | Source table | Question | True answer | Baseline | v2 |
|---|---|---|---|---|---|
| 1 | icustays | How many icu visits did patient 10018845 have during the first hospital encounter? | `[[1]]` | ✅ 1 | ✅ 1 |
| 2 | admissions | Has patient 10026406 gone to an emergency room since 2129? | `[[1]]` | ❌ No, the patient has not gone to the emergency department in … | ❌ No. The patient has not had an Emergency Department (ER) vis… |
| 3 | chartevents | Tell me the average systolic blood pressure in patient 10021487 on 12/09/2116. | `[[102.56]]` | ✅ 102.56 mmHg | ✅ Average systolic blood pressure on 2116-12-09: 102.56 mmHg |
| 4 | chartevents | What is the body weight change of patient 10038933 second measured on the last hospital visit compared to the first value measured on the last hospital visit? | `[[0.5]]` | ❌ Not enough Daily Weight observations on the last hospital vi… | ❌ Could not calculate body weight change because fewer than tw… |
| 5 | microbiologyevents | For the first time since 07/2185, what specimen test did patient 10038992 get? | `[['BLOOD CULTURE']]` | ✅ The first specimen collected after 07/2185 for patient 10038… | ✅ {'collectedDateTime': '2185-11-02T20:55:00-04:00', 'testName… |
| 6 | chartevents | How much was the difference in weight of patient 10029291 measured at 2123-03-07 03:00:00 compared to the value measured at 2123-03-05 00:00:00? | `[[None]]` | ✅ Could not determine weight measurements for the specified ti… | ✅ Weight observations at the specified times not found. |
| 7 | icustays | Tell me the total icu visits of patient 10004733 since 2174. | `[[1]]` | ✅ 1 | ✅ 1 ICU visit since 2174-01-01. |
| 8 | outputevents | Gather the total output since 10/05/2167 for patient 10007058. | `[[7515]]` | ❌ 63332.62 | ✅ 7515.0 |
| 9 | patients | What is the recorded sex of patient 10022880? | `[['M']]` | ✅ male | ✅ male |
| 10 | chartevents | Can you tell me what patient 10007795's last heart rate was during the first icu visit? | `[[112.]]` | ✅ 112 bpm | ✅ Last heart rate during the first ICU visit: 112 bpm |
| 11 | prescriptions | How many unique medications were prescribed to patient 10004422 in the previous year? | `[[0]]` | ✅ 0 unique medications were prescribed to patient 10004422 in … | ✅ 0 unique medications were prescribed to patient 10004422 in … |
| 12 | chartevents | Tell me patient 10026354's first measured heart rate. | `[[99.]]` | ✅ 99 bpm | ✅ 99 bpm at 2119-10-26T08:44:00-04:00 |
| 13 | admissions | How many times was patient 10023239 in the hospital this year? | `[[1]]` | ✅ 1 | ✅ 1 |
| 14 | chartevents | Is the body temperature level of patient 10005817 second measured on the last icu visit greater than the first measurement on the last icu visit? | `[[None]]` | ✅ Cannot determine: fewer than 2 body temperature observations… | ❌ No. In the last ICU visit for patient 10005817, the first bo… |
| 15 | prescriptions | Has patient 10022880 acquired any medication in 2177? | `[[1]]` | ✅ Yes. In 2177, patient with FHIR ID 13df78e7-150e-5eb7-be5f-5… | ✅ Yes. The patient acquired medications in 2177. There are 215… |
| 16 | chartevents | What is the difference in patient 10001725's spo2 last measured on the first icu visit compared to the second to last value measured on the first icu visit? | `[[-1.]]` | ✅ The difference in SpO2 last measured on the first ICU visit … | ✅ -1.0 percentage points (the last SPO2 value during the first… |
| 17 | labevents | Is the lymphocytes level of patient 10037928 last measured on the last hospital visit less than the first level measured on the last hospital visit? | `[[0]]` | ❌ Yes. On the last hospital visit for patient fa5fbf9c-23e3-5e… | ❌ Yes. First lymphocyte value: 18.2 at 2183-08-03 20:40:00-04:… |
| 18 | chartevents | When was first time the respiratory rate of patient 10020740 was greater than 17.0 on 06/15/this year? | `[]` | ✅ No respiratory rate observation > 17.0 on 06/15/2150 was fou… | ✅ No respiratory rate >17.0 observed on 06/15/2150 for this pa… |
| 19 | prescriptions | What amount of fish oil (omega 3) has been prescribed to patient 10004457 for the last time during their last hospital encounter? | `[['1000']]` | ❌ No data found for patient 10004457: there were 0 Encounter r… | ❌ No omega-3 fish oil prescription found for the last hospital… |
| 20 | microbiologyevents | When was patient 10021666 last mrsa screen microbiologic test? | `[['2172-03-13 04:00:00']]` | ✅ 2172-03-13T04:00:00 | ❌ 2172-03-13T03:00:00 |

Observations:
- The two agents disagree on only 3 of the 20 questions (#8, #14, #20), and the smoke reruns gave different answers each time, so the accuracy difference is noise. Telling apart a gain of a few points needs about 100+ questions.
- #20: v2 was off by exactly one hour, most likely a timezone or daylight-saving conversion. The ground truth uses local timestamps.
- #19: both agents reported finding no encounters or prescriptions, even though the ground truth expects an answer of 1000. This is worth investigating.

Question IDs, for reproduction: `5ad1c0bd8c724486755abcd6`, `68002865c7084ae451a8b103`, `573ae8f84da94b97d13dd488`, `db1b480d2286ade3469fa410`, `39881273c6b26ba01d54e3ee`, `7fd1121ddb007a49e5ed2f4c`, `ea018f554b80d642628c416c`, `67c2dbeaea4791e0feb0a69b`, `7d7aee677bb35ba8d39e2a8e`, `9b39448454fb368c2b7fb229`, `78deb17fb4b0f07c1c1b0051`, `2e3384d7ef9a76c0237df519`, `750dbf95a8c1c5e4d75ecefe`, `cd68238c19445383f9469b6b`, `21b660c0148b20c1b1c2131d`, `941095e5c2a9df6746e4a383`, `10a09034b24ebc5ea1fcd470`, `6e40bf924c854c6bb7c3d749`, `54234a211c3fb7360ae4e161`, `db74cd1a2f794e17f35ddec4`

## 4. Failure analysis (smoke questions, multi-turn code agent)

In every case the model guessed the field names and values instead of inspecting the data first:

| Q | What the model searched for | What the data contains |
|---|---|---|
| 2 | exact match on "…superior vena cava, guidance treatment" | `Procedure.code.coding[].display` = "Ultrasonography of Superior Vena Cava, Guidance" |
| 3, 5 | drug name on `MedicationRequest` | `MedicationRequest.medicationReference` → `Medication.identifier[system=…/mimic-medication-name]`, e.g. "Milk of Magnesia" |
| 4 | Observation display "void output" | display "Void"; `category[].coding[].code` = "Output"; `valueQuantity` in ml |

`multi_turn_code_resource_v2` (`agent/multi_turn_code_resource_v2_agent.py`) adds general guidance on top of the baseline prompt:
- Inspect candidate values before filtering on them.
- Match names partially and case-insensitively.
- Explains where drug names are stored and the shapes of the main fields.

While testing it, I found that `execute_python_code` returns only the `answer` variable to the model. Anything the model `print()`s never reaches it, so the guidance tells the model to put candidate values in `answer`.

## 5. Cost analysis

Prices are list prices as of 2026-10-08 ([OpenAI](https://developers.openai.com/api/docs/pricing), [Gemini](https://ai.google.dev/gemini-api/docs/pricing)). Token counts use tiktoken `cl100k_base`; Gemini's tokenizer differs by roughly 10–20%. "Low" and "high" assume about 600 vs 3,000 output tokens on each answering call.

### 5.1 How many questions the single-turn agent can attempt (full dataset, measured)

Prompt sizes were measured for all 2,931 questions, after the Patient-lookup fix, with no LLM calls. Every patient in the dataset has at least 415 Observations, so every Observation question exceeds 100k tokens.

| Resource type | Questions | Fit in 32k | Median prompt |
|---|---|---|---|
| Observation | 1,625 | 0 | over the limit (median ≈ 2.0M tokens, max ≈ 16.4M) |
| MedicationRequest | 602 | 203 | 45.5k |
| Encounter | 485 | 485 | 2.6k |
| Procedure | 133 | 133 | 7.0k |
| Patient | 44 | 44 | 1.2k |
| Condition | 35 | 34 | 4.9k |
| Location | 7 | 7 | 4.6k |
| **Total** | **2,931** | **906 (31%)** | |

Observation sizes are each patient's Observation count × 337 tokens, the high end of the 283–337 tokens per resource measured on 18 patients.

### 5.2 Single-turn agent on `gemini-3.8-flash`: effect of caching (full run)

| Scenario | 2026 | 2027 (prices double) |
|---|---|---|
| Standard, no caching | $11–23 | $22–46 |
| Automatic caching, current prompt order | $11–23 (no hits; the shared prefix is ~790 tokens, below the 4,096 minimum) | $22–46 |
| Caching, patient data placed first in the prompt | $8–20 | $16–40 |
| Explicit caching, data first (+1h storage) | $8.50–20.50 | $17–41 |
| Batch API | $6–12 | $11–23 |
| Batch API plus data-first caching | $4–10 | $8–20 |

Only 321 questions would get a cache hit, although 86% of questions reuse a (patient, resource type) fetch: the 2,931 questions involve only 94 patients and 413 distinct fetches.

### 5.3 Single-turn agent: OpenAI vs Gemini, Batch API, full run

| Provider | Model | Context | Questions attemptable at full window | Batch cost at the 32k cap | Batch cost at full window |
|---|---|---|---|---|---|
| OpenAI | `gpt-5-nano` | 272k input | 1,304 (44%) | $0.45–1.09 | $1.35–2.18 |
| OpenAI | `gpt-4.1-nano` | 1M | 1,831 (62%) | $0.70–1.34 | $20–21 |
| OpenAI | `gpt-6-luna` | 1.05M | 1,831 (62%) | $0.75–1.55 | $37–38 |
| OpenAI | `gpt-5-mini` | 272k input | 1,304 (44%) | $2.24–5.44 | $7–11 |
| OpenAI | `o4-mini` | 200k | 1,284 (44%) | $7.70–14.74 | $24–33 |
| Gemini | `gemini-3.5-flash-lite` | 1M | 1,831 (62%) | $2.74–6.74 | $60–66 |
| Gemini | `gemini-3.8-flash` (2026) | 1M | 1,831 (62%) | $5.62–11.62 | $147–157 |
| Gemini | `gemini-3.8-flash` (2027) | 1M | 1,831 (62%) | $11.23–23.23 | $294–315 |

1,100 Observation questions exceed 1M tokens, so no model can attempt them with the single-turn agent.

### 5.4 Multi-turn code agent (measured, 20 held-out questions)

| Agent | Model | Per question | Full run, standard | Full run, Batch |
|---|---|---|---|---|
| `multi_turn_code_resource` | `gpt-5-nano` | $0.0052 | about $15 | about $8 |
| `multi_turn_code_resource_v2` | `gpt-5-nano` | $0.0034 | about $10 | about $5 |
| `multi_turn_code_resource` | `o4-mini` | about $0.013–0.018 (5 smoke questions only) | about $40–53 | about $20–27 |

Batch mode is not implemented in `run_agent.py`; it would need a separate submit-and-collect script.

### 5.5 Other notes

- A ChatGPT Pro subscription does not include API access. API usage is billed separately on platform.openai.com.
- Using a long context window needs a tokens-per-minute rate limit at least as large as the prompt. Check this at https://platform.openai.com/settings/organization/limits.

## 6. Code changes made during these runs

| Change | File(s) |
|---|---|
| Untracked `config.yml` and added `config.example.yml` and `.gitignore` | `.gitignore`, `config.example.yml` |
| Placeholder-aware config validation; `run_agent.py` exits when the config is invalid | `utils/core_utils.py`, `run_agent.py` |
| `--limit N`, and a default output path when `--output` is omitted | `run_agent.py` |
| One-command FHIR store setup | `scripts/setup_fhir_store.sh` |
| Setup verification | `scripts/check_setup.py` |
| 5-question smoke test (setup check → agent → evaluation) | `scripts/run_smoke_test.sh` |
| **Bug fix:** `Patient?patient=<id>` returned all 100 patients (41k tokens). It now uses `Patient?_id=<id>`. | `tools/resource_tools.py` |
| **Bug fix:** model code calling `exit()` / `raise SystemExit` killed the whole agent run | `tools/python_executor.py` |
| Experimental v2 agent and run script | `agent/multi_turn_code_resource_v2_agent.py`, `scripts/run_multi_turn_code_resource_v2_agent.sh` |

### Known issues not fixed

- `execute_python_code` accepts a `timeout` argument but never enforces it, so model code with an infinite loop hangs the run.
- Model-generated code runs unsandboxed via `exec` in the agent process.
- The multi-turn code agents tell the model that the data is `retrieved_fhir_resources[...]`, but the variable available to its code is `retrieved_resources`.
- `get_model_provider()` recognises only `gpt-5`, `gpt-4` and `o4` for OpenAI, so `gpt-6-*` models are rejected.
- When an LLM call fails, the error text is stored as `agent_answer` with `error = null`. Failed rows therefore look answered and are not retried on resume.
- `Location?patient=` and `Medication?patient=` ignore the patient filter and return the whole table. This is expected for these types (Medication is the shared drug catalog).

## 7. Reproducing

```bash
# One-time GCP setup
gcloud auth login && gcloud auth application-default login
PROJECT_ID=<project> LOCATION=us-central1 DATASET_ID=mimic-fhir STORE_ID=mimic-fhir \
BUCKET=<bucket> bash scripts/setup_fhir_store.sh
cp config.example.yml config.yml   # fill in OPENAI_API_KEY and FHIR_CONFIG

# Checks and smoke test
python scripts/check_setup.py
bash scripts/run_smoke_test.sh                                   # o4-mini, single-turn
MODEL=gpt-5-nano bash scripts/run_multi_turn_code_resource_agent.sh --limit 5

# Held-out comparison
python -c "import pandas as pd; d=pd.read_csv('final_dataset/questions_answers_sql_fhir.csv'); d.iloc[5:].sample(20, random_state=42).to_csv('heldout20.csv', index=False)"
python run_agent.py --agent_strategy multi_turn_code_resource    --model gpt-5-nano --input heldout20.csv --num_processes 1 --save_interval 1 --output output/compare/held20_baseline.json
python run_agent.py --agent_strategy multi_turn_code_resource_v2 --model gpt-5-nano --input heldout20.csv --num_processes 1 --save_interval 1 --output output/compare/held20_v2.json
python evaluation_metrics.py --input output/compare/held20_baseline.json --model o4-mini --num_processes 1 --force
```

Raw outputs are in `output/` and `output_eval/`. Both are git-ignored or untracked, so they are kept locally and not committed.

## 8. Direct LLM baseline: smoke test (#11)

First run on 2026-10-08 with `gpt-5-nano`, on the same first 5 questions as §2. See §8.1 for the re-run after the starting packet started excluding resources dated after the assumed current time. The agent makes one LLM call on the question plus the 8,000-token starting packet (#7), with no tools. Answers were scored with the typed scorer (#8), the LLM judge (`o4-mini`, secondary), and the evidence scorer (#9, resolved against the local NDJSON).

| # | Question | True answer | Answer | Typed | Evidence valid |
|---|---|---|---|---|---|
| 1 | When was the first time the respiratory rate of patient 10018081 was measured to be less than 23.0 today? | `[['2133-12-31 02:00:00']]` | `null` | ❌ abstained | 0/0 |
| 2 | Did patient 10029291 undergo a ultrasonography of superior vena cava, guidance treatment in the first hospital visit? | `[[1]]` | `true` | ✅ | 4/4 |
| 3 | Has docusate sodium (liquid), glucose gel, or heparin been prescribed since 2157 to patient 10036156? | `[[1]]` | `true` | ✅ | 0/3 (cited resources that don't exist) |
| 4 | Show me patient 10014354's total void output amount since 08/11/2146. | `[[3075]]` | `7225` | ❌ | 16/16 |
| 5 | How frequently was milk of magnesia prescribed to patient 10005348 in 09/this year? | `[[1]]` | `null` | ❌ abstained | 3/5 (2 bad paths) |

| Metric | Value |
|---|---|
| Typed accuracy | **2/5** (§2: multi-turn code agent 1/5, v2 2/5 on the same questions) |
| LLM judge accuracy | 2/5 (agrees with the typed scorer on all 5) |
| Packet retrieval recall / precision | 0.80 / 0.14 |
| Valid citations | 23/28; evidence precision 0.86, recall 0.60 |
| Tokens / cost | 52,794 tokens, $0.0081 total |
| Latency | 45–107 s per question, mostly the first fetch of each patient's full record (later questions hit the tool cache) |

Failure notes:
- **Q1:** the record has respiratory-rate readings dated after the question's assumed current time (2133-12-31). The packet keeps the earliest and latest readings of a label, so the 2134 readings were kept and the "today" readings in between were cut. The starting-packet retriever should drop resources after the assumed current time before selecting.
- **Q4:** the model summed outputs over the wrong date range, although all 16 cited values were correct.
- **Q5:** the model read "how frequently" as the dosing frequency rather than a count, and abstained.
- **Judge bug (fixed in #37):** the first judge run scored 0/5 because the judge crashed on non-string answers such as `true` and counted them wrong. Structured answers are now passed to the judge as JSON.

### 8.1 Re-run after the current-time cutoff

The Q1 failure led to a retriever change (#7): resources and encounters dated after the question's assumed current time are left out of the packet. The same 5 questions were re-run with `gpt-5-nano`.

| # | True answer | Answer | Typed | Judge | Evidence valid |
|---|---|---|---|---|---|
| 1 | `[['2133-12-31 02:00:00']]` | `"2133-12-31 02:00:00"` | ✅ | ✅ | 2/2 |
| 2 | `[[1]]` | `true` | ✅ | ✅ | 4/4 |
| 3 | `[[1]]` | `true` | ✅ | ✅ | 0/3 (cited resources that don't exist) |
| 4 | `[[3075]]` | `3075` | ✅ | ✅ | 7/7 |
| 5 | `[[1]]` | `null` | ❌ abstained | ❌ | 0/2 (bad paths) |

| Metric | Before cutoff (§8) | After cutoff |
|---|---|---|
| Typed / judge accuracy | 2/5 / 2/5 | **4/5 / 4/5** |
| Packet retrieval recall / precision | 0.80 / 0.14 | 1.00 / 0.30 |
| Valid citations | 23/28 | 13/18 |
| Evidence precision / recall | 0.86 / 0.60 | 1.00 / 0.80 |
| Tokens / cost | 52,794 / $0.0081 | 44,607 / $0.0069 |
| Latency per question | 45–107 s | 15–42 s (patient records now cached) |

- **Q1** now has the "today" readings in its packet and is answered correctly.
- **Q4**'s packet shrank from 7.9k to 2.1k tokens once later output readings were excluded, and the model summed the right range.
- **Q5** still reads "how frequently" as a dosing schedule and abstains.
- Five questions is a smoke test; model sampling alone can move single answers.

Reproduce:

```bash
python run_agent.py --agent_strategy direct_llm --model gpt-5-nano --limit 5 --num_processes 1 --output output/smoke_direct_llm_gpt-5-nano.json
python evaluation_metrics.py --input output/smoke_direct_llm_gpt-5-nano.json --model o4-mini --evidence_source ndjson --force   # needs the #8/#9 evaluation code
```

## 9. Tool-equipped single agent: smoke test (#12)

`single_agent` is the multi-turn code agent started from the same starting packet as the direct baseline (its resources preloaded for code), with the deterministic `check_fields` tool (`tools/field_check.py`) and one revision: its first JSON answer is checked automatically, and if the check finds problems the agent is told what they are and may revise once. Run on 2026-10-08 with `gpt-5-nano` (5 processes) on the same 5 questions; scored with the typed scorer, the `o4-mini` judge, and the evidence scorer (resolved against the FHIR store).

| # | True answer | Answer | Typed | Judge | Evidence valid | Tool calls | Check issues on draft | Revised |
|---|---|---|---|---|---|---|---|---|
| 1 | `[['2133-12-31 02:00:00']]` | `"2133-12-31 02:00:00"` | ✅ | ✅ | 2/2 | 0 | 0 | no |
| 2 | `[[1]]` | `true` | ✅ | ✅ | 3/3 | 0 | 0 | no |
| 3 | `[[1]]` | `true` | ✅ | ✅ | 3/6 (3 bad paths) | 3 | 6 | yes, answer unchanged |
| 4 | `[[3075]]` | `3075` | ✅ | ✅ | 14/14 | 3 | 0 | no |
| 5 | `[[1]]` | `1` | ✅ | ✅ | 2/2 | 0 | 1 | yes, answer unchanged |

| Metric | Direct LLM (§8.1) | Single agent |
|---|---|---|
| Typed / judge accuracy | 4/5 / 4/5 | **5/5 / 5/5** |
| Retrieval recall / precision | 1.00 / 0.30 | 1.00 / 0.05 (fetches whole resource types) |
| Valid citations | 13/18 | 24/27 |
| Evidence precision / recall | 1.00 / 0.80 | 0.90 / 0.90 |
| LLM calls per question | 1 | 1–4 |
| Tokens / cost | 44,607 / $0.0069 | 154,261 / $0.0226 |
| Latency per question | 15–42 s | 23–151 s |

Notes:
- Three questions were answered from the packet alone, without tool calls; the other two fetched more data.
- The automatic check fired on two drafts and the agent revised both; the answers stayed the same, but on Q3 three citations still had bad paths after the revision.
- The agent never called `check_fields` on its own before answering.
- Five questions is a smoke test, not a comparison.

Code changes made with this agent:
- `get_resources_by_resource_id` searched with `?id=`, which the Cloud Healthcare API ignores, so it returned every resource of the type (e.g. all 637 Encounters). It now uses `_id`, and its cache entries are versioned so old results are not reused. This changes what the original agents get when they look up a resource by id.
- The multi-turn code agent's tool handling moved into `_handle_tool_call()` so the single agent can extend it; a 1-question run of `multi_turn_code_resource` after the refactor behaved as before (`2133-12-31T02:00:00`, 3 tool calls).

Reproduce:

```bash
python run_agent.py --agent_strategy single_agent --model gpt-5-nano --limit 5 --num_processes 5 --output output/smoke_single_agent_gpt-5-nano.json
python evaluation_metrics.py --input output/smoke_single_agent_gpt-5-nano.json --model o4-mini --force
```
