# FHIR-AgentBench

This repository contains the code and dataset for FHIR-AgentBench.


## 📁 Project Structure

```
FHIR-AgentBench/
├── scripts/                          # Bash scripts for data setup, agent inference, and evaluation
├── agent/                            # Multiple agent implementations
├── tools/                            # Tools for agents
├── utils/                            # Utility modules
├── config.py                         # Configuration settings and constants
├── config.yml                        # YAML configuration file
├── create_db.py                      # Creates database for Q&A conversion to FHIR
├── create_question_answer_dataset.py # Creates Q&A dataset from EHRSQL
├── create_question_fhir_dataset.py   # Creates FHIR-compatible question dataset
├── evaluation_metrics.py             # Main evaluation script
├── fhir_client.py                    # FHIR client for Google Cloud Healthcare API
├── run_agent.py                      # Main script to run agents on datasets
├── question_fixes_complete.json      # Hard-coded question fixes
├── value_mapping_valid_natural.json  # Natural language value mappings 
├── requirements.txt                  # Python package dependencies
└── images/                           # Documentation images
```

## 🚀 Getting Started

### Prerequisites

- Install required packages:
  ```bash
  # Create a conda environment
  conda create -n fhir-agentbench python=3.11
  conda activate fhir-agentbench

  # Install dependencies
  pip install -r requirements.txt
  ```

### Data Preparation

#### 1. Upload the MIMIC-IV FHIR data to a GCP FHIR store

**Scripted option:** with the [gcloud CLI](https://cloud.google.com/sdk/docs/install-sdk) installed and authenticated (see step 2), this downloads the demo, uploads it to a bucket, creates the dataset/store, and runs the import:

```bash
PROJECT_ID=<your-project> LOCATION=us-central1 DATASET_ID=mimic-fhir STORE_ID=mimic-fhir \
BUCKET=<your-bucket> bash scripts/setup_fhir_store.sh
```

**Manual option (Cloud Console):**
- Download [MIMIC-IV Clinical Database Demo on FHIR](https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/) from PhysioNet and extract the .gz files.
- Create a GCP account, then in the [Google Cloud Console](https://console.cloud.google.com) search for FHIR Viewer.
- Click Browser on the left, then Create dataset.
![dataset creation](/images/create_dataset.png)
- Next, click Create data store to prepare for the data upload.
![datastore creation](/images/create_data_store.png)
- For Configure your FHIR store, select R4 as the FHIR Version. Keep other settings as default and click Create.
- Separately, in [Cloud Storage](https://console.cloud.google.com/storage), upload your unzipped folder containing the MIMIC-IV FHIR data (*.ndjson) to a bucket.
- Back in the FHIR store, click Actions in the upper right and choose Import.
![FHIR Data Store Import](/images/import_fhir.png)
- Select the folder you uploaded. Under FHIR Import Settings, choose Resource for Content Structure. Click Import and grant permissions if prompted.
![FHIR import settings](/images/fhir_import_settings.png)
- Open the Import operation to confirm success. It usually completes in about 10 minutes.

#### 2. Enable APIs and authenticate with gcloud

You can enable the required APIs and verify access using the [gcloud CLI](https://cloud.google.com/sdk/docs/install-sdk). This is often the fastest way to confirm your setup before running code.

0) Log in

   ```bash
   # Authenticate with your Google account
   gcloud auth login

   # Set up Application Default Credentials (ADC)
   gcloud auth application-default login --no-launch-browser
   ```

1) Check or set the current project and project number

   ```bash
   # List all available projects to find your PROJECT_ID
   gcloud projects list
   ```

   ```bash
   # Set the quota project for ADC (to handle billing and quotas)
   gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>

   # Set the default project for gcloud CLI
   gcloud config set project <YOUR_PROJECT_ID>
   ```

   ```bash
   # Get the current project ID and project number
   PROJECT_ID="$(gcloud config get-value project)"
   PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")"

   # Print them for confirmation
   echo "$PROJECT_ID"
   echo "$PROJECT_NUMBER"
   ```

2) Enable required APIs

   ```bash
   # Enable the Cloud Healthcare API (for FHIR, DICOM, HL7v2 resources)
   gcloud services enable healthcare.googleapis.com --project="$PROJECT_ID"

   # Enable the Cloud Asset API (needed for dataset and store discovery)
   gcloud services enable cloudasset.googleapis.com --project="$PROJECT_ID"

   # Enable the Cloud Resource Manager API (needed for project and resource management)
   gcloud services enable cloudresourcemanager.googleapis.com --project="$PROJECT_ID"

   # Enable the Service Usage API (needed to enable and check other APIs)
   gcloud services enable serviceusage.googleapis.com --project="$PROJECT_ID"
   ```

3) Automatically discover dataset, FHIR store, and location

   ```bash
   # Find the dataset ID and location
   read DATASET_ID LOCATION <<<$(gcloud asset search-all-resources \
   --scope="projects/$PROJECT_NUMBER" \
   --asset-types="healthcare.googleapis.com/Dataset" \
   --format="value(name.basename(), location)")

   echo "LOCATION=$LOCATION"
   echo "DATASET_ID=$DATASET_ID"

   # Find the FHIR store ID
   STORE_ID="$(gcloud healthcare fhir-stores list \
   --dataset="$DATASET_ID" --location="$LOCATION" --project="$PROJECT_ID" \
   --format="value(name.basename())")"

   echo "STORE_ID=$STORE_ID"
   ```

4) Grant IAM permissions to your user (if not already granted)

   ```bash
   # Get the current logged-in user
   USER="$(gcloud config get-value account)"

   # Grant FHIR resource read access
   gcloud healthcare datasets add-iam-policy-binding "$DATASET_ID" \
   --location="$LOCATION" --project="$PROJECT_ID" \
   --member="user:$USER" \
   --role="roles/healthcare.fhirResourceReader"

   # Grant FHIR store viewer access
   gcloud healthcare datasets add-iam-policy-binding "$DATASET_ID" \
   --location="$LOCATION" --project="$PROJECT_ID" \
   --member="user:$USER" \
   --role="roles/healthcare.fhirStoreViewer"
   ```

5) Project configuration

   Copy the template and fill in your values (`config.yml` is git-ignored so keys stay local):

   ```bash
   cp config.example.yml config.yml
   ```

   ```yaml
   OPENAI_API_KEY: "your-api-key"
   GEMINI_API_KEY: "your-api-key"
   FHIR_CONFIG:
      PROJECT_ID: "your-gcp-project-id"
      LOCATION: "your-fhir-dataset-location"
      DATASET_ID: "your-dataset-id"
      STORE_ID: "fhir-store-id (usually the same as dataset_id)"
   ```

   Environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `FHIR_PROJECT_ID`, `FHIR_LOCATION`, `FHIR_DATASET_ID`, `FHIR_STORE_ID`) take precedence over `config.yml`.

   Each collaborator chooses their own model and provider configuration. Keep personal launchers, provider adapters, deployment settings, and credential files local (for example, under the ignored `.local-setup/` directory); shared agents must not depend on them. Commit reusable code, tests, placeholder-only configuration templates, and project-wide reproducibility manifests. Virtual environments, caches, downloaded data, and run outputs remain local.

6) Quick setup check

   ```bash
   # Validates config.yml, credentials, and FHIR store contents
   python scripts/check_setup.py

   # Runs the single-turn resource agent on 5 questions and evaluates them
   bash scripts/run_smoke_test.sh            # MODEL=... NUM_QUESTIONS=... to override
   ```

#### 3. (Optional) Run the script to download and prepare the dataset:
If `final_dataset/questions_answers_sql_fhir.csv` already exists, you can skip this stage.

   ```bash
   bash scripts/setup_data.sh
   python create_question_answer_dataset.py
   python create_question_fhir_dataset.py
   ```

#### 4. Build the question-family manifest
Audits every question in `manifests/question_families.csv`. Eligible questions are tagged `lookup_counting`, `temporal_selection`, or `numerical_aggregation`; paired differences, comparisons, elapsed times, and stay durations are `out_of_scope` with a reason. Empty and null answers remain eligible and are marked in `answer_kind` (`empty` / `null_value`).

   ```bash
   python scripts/build_question_families.py
   ```

#### 5. Build the patient-disjoint split
Writes `manifests/split.json` with pilot (20), dev (50), and held-out (300) question sets drawn from disjoint patients, stratified by in-scope family. `manifests/development_exposure.json` reserves proposal examples, inherited few-shot examples, and documented prior runs for development: 32 dev patients, 3 fresh pilot patients, and 59 held-out patients. Tune on pilot/dev only; every question from a held-out patient is reserved for final evaluation.

   ```bash
   python scripts/build_split.py   # seed 5106 by default
   ```

#### 6. Build the starting packets
`tools/starting_packet.py` builds a deterministic, token-bounded excerpt (keyword-ranked resource labels plus encounters; a fixed common 8,000-token cap). Resources dated after the assumed current time are left out. Both baselines start with exactly the rendered fields, not hidden full records. `get_starting_packet(question, patient_fhir_id, current_time=parse_current_time(assumption))` reads the FHIR store. The offline builder accepts plain or gzip NDJSON and freezes the packet IDs, text hashes, and source hashes in `manifests/starting_packets.jsonl` and its metadata companion. It does not read gold IDs by default.

   ```bash
   python scripts/build_starting_packets.py
   python scripts/build_starting_packets.py --sweep 4000 8000 16000   # dev token/truncation statistics only
   ```

The original cap was selected using development gold recall. That historical deviation cannot be undone; it is disclosed in [the data review](docs/phase1_data_review.md). Do not use `--gold-diagnostics` to tune retrieval: it is only an optional post-freeze pilot/dev diagnostic, never a held-out analysis.

## 🤖 Agent Execution

The project includes several agent implementations:

```bash
# Single-turn agents
bash scripts/run_single_turn_request_agent.sh       # Single-turn FHIR RESTful API generation and retrieval → Natural language reasoning
bash scripts/run_single_turn_resource_agent.sh      # Single-turn FHIR resource retrieval → Natural language reasoning
bash scripts/run_single_turn_code_resource_agent.sh # Single-turn FHIR resource retrieval → Code-based reasoning

# Multi-turn agents
bash scripts/run_multi_turn_resource_agent.sh       # Multi-turn/iterative resource retrieval → Natural language reasoning
bash scripts/run_multi_turn_code_resource_agent.sh  # Multi-turn/iterative resource retrieval → Code-based reasoning

# Baseline without tools
bash scripts/run_direct_llm_agent.sh                # One LLM call on the question + starting packet (tools/starting_packet.py)

# Tool-equipped single agent
bash scripts/run_single_agent.sh                    # Multi-turn code agent started from the packet, with check_fields (tools/field_check.py) and one revision

# Multi-agent system
bash scripts/run_multi_agent.sh                     # Fixed controller: retrieval → analysis → verification, at most one repair
```

Extra arguments are passed through to `run_agent.py`, e.g. `MODEL=gemini/gemini-2.5-flash bash scripts/run_single_turn_resource_agent.sh --limit 5`.

Per-question budgets (both off by default):
- `--max_total_tokens N`: reserve the next prompt (including tool schemas/arguments and a framing margin), cap its completion to the remaining aggregate allowance, and stop if no allowance remains. Provider-reported overruns are recorded and the answer refused. Tokenizers differ, so the prompt reserve is an estimate rather than a promise of exact provider framing.
- `--max_tool_calls N`: tool calls beyond N are not run. Automatic first/final evidence checks count too; an unchecked answer is not released.

Each result records `tokens` (including `per_call` usage, `repeated_prompt`, the estimated prompt tokens re-sent from earlier turns, and `cached_prompt`, those served from the provider's prompt cache) and `budget` (the ceilings, tool calls run, and which ceiling stopped the run, if any). `run_agent.py` prints a usage summary at the end.

Select a frozen set with `--split_manifest manifests/split.json --split pilot` (or `dev` / `held_out`); `--limit` applies after split selection. An output's `.run.json` binds resumption to inputs, code, model, split, and budgets. Legacy or incompatible outputs need a new path. Failed attempts remain in `attempt_history`; unknown pricing is `cost: null`, not zero.

With `--split held_out --perturbations manifests/perturbations.json --condition omitted`, the run covers only the 50 perturbed questions and each system starts from the frozen packet with one answer-changing record left out (`--condition intact` runs the same questions from the full packet).

### Phase 3 experiment

`scripts/run_experiments.py` runs the full experiment: 900 ordinary runs (each system on the 300 held-out questions) and 200 recovery runs (the single and multi-agent systems on the 50 perturbed questions, intact and omitted). All systems share one model and budget, and the tool cache is on.

```bash
python scripts/run_experiments.py plan   --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30  # arms, counts, commands
python scripts/run_experiments.py run    --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30  # run or resume
python scripts/run_experiments.py status --model gpt-5-nano --max_total_tokens 200000 --max_tool_calls 30  # counts only
```

Rerunning `run` resumes from the saved results and retries failed questions. Use `--split pilot --limit 2` for a smoke run, and `--arms` to run a subset. See [the Phase 3 notes](docs/phase3_experiments.md).

Generated Python runs with copied retrieved resources, no repository credentials or network, and a real deadline. It requires `sandbox-exec` on macOS or `bubblewrap` (`bwrap`) on Linux and fails closed without a supported sandbox. See [the review and verification notes](docs/phase1_review.md).

To use open-source models locally with vLLM, start the vLLM server and set base_url to `http://localhost:<port>/v1`.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python -m vllm.entrypoints.openai.api_server --model <model> --load-format safetensors --max-model-len 32768 --tensor-parallel-size <num_gpus> --port <port> --enable-auto-tool-choice --tool-call-parser llama3_json
```

### Output contract

Every agent's `run()` returns the same JSON object (`agent/output_schema.py`), and `run_agent.py` saves its fields per question:

| Field | Saved as | Meaning |
|---|---|---|
| `answer` | `agent_answer` | `true`/`false`, a number, a `"YYYY-MM-DD HH:MM:SS"` time, a string, a list, or `null` when the agent abstains |
| `evidence` | `evidence` | `[{"resource_id": "Observation/<id>", "field_path": "valueQuantity.value", "value": 98.7}, ...]` |
| `limitation` | `limitation` | `null`, or what is missing or uncertain |
| `tokens` | `tokens` | `{"prompt", "completion", "total", "cost", "llm_calls", "cached_prompt", "repeated_prompt", "per_call"}` (also saved in the legacy `usage` format) |
| `tool_calls` | `tool_calls` | `[{"name", "arguments"}, ...]` |
| `latency` | `latency` | wall-clock seconds |
| `budget` | `budget` | `{"max_total_tokens", "max_tool_calls", "tool_calls_run", "stopped"}` |

New agents implement `_run()` and either return these fields directly or end with the JSON object described in `ANSWER_FORMAT_INSTRUCTIONS`. The original agents answer in free text, which is kept as the `answer` with empty `evidence`.

## 📊 Evaluation

Run the following command to normalize, evaluate answers, and visualize performance (FHIR resource retrieval recall/precision, answer correctness):

```bash
python evaluation_metrics.py --input <agent_output_json_file_path>
python evaluation_metrics.py --input <agent_output_json_file_path> --skip_llm_judge   # typed scorer only, no API calls
```

Answer correctness has two metrics:
- **Typed scorer (primary)**, `utils/answer_scorer.py`: deterministic. The reference fixes the type: exact counts, measurements within 1% (minimum absolute tolerance 0.01), requested timestamp precision, yes/no, normalized categories, and complete sets without extras. Known implied units convert; incompatible claims are wrong and claims whose target unit is unknown are unparsed. Abstaining is wrong for a non-empty reference; empty/null references accept explicit absence (or zero for numeric questions). Ambiguous prose is unparsed, not credited for containing a matching fragment. Results are in `typed_correct`, `answer_type`, and `typed_reason`.
- **LLM judge (secondary)**: the original `answer_correctness`, reported as `accuracy` in `output_eval/performance_summary.csv` next to `typed_accuracy`.

If the agent output has an `evidence` field, each citation `{"resource_id": "Observation/<id>", "field_path": "valueQuantity.value", "value": 98.7}` is resolved against the FHIR store (`utils/evidence_scorer.py`). A citation is valid when the resource exists, the path resolves, and the value matches. The evaluation reports the valid-evidence fraction, the reasons citations were invalid (`malformed`, `missing_resource`, `bad_path`, `value_mismatch`), and the precision/recall of the cited resource ids against `true_fhir_ids`. Use `--evidence_source ndjson` to resolve against the local NDJSON files instead of the store.


## Authorship
FHIR-AgentBench is a joint research effort between Verily Life Sciences, Korea Advanced Institute of Science & Technology (KAIST), and Massachusetts Institute of Technology (MIT).
