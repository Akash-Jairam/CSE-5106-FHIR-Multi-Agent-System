# Guided baselines: gpt-oss:120b and qwen3:32b

Nov 18, 2026 · Akash Jairam

## Design note

Written and committed before any held-out run; the results below are read against it.

**Question.** On the 300 held-out questions, multi-agent v2 leads the single agent by 9.0 points with gpt-oss:120b and 11.3 with qwen3:32b ([Multi-agent v2](phase3-qwen3-32b.md#multi-agent-v2)). Its analysis role's prompt has two blocks the baselines never see: the search guidance (`MultiTurnCodeResourceAgent.search_guidance`) and `SEMANTIC_GUIDANCE`, which ends with `CONVENTION_GUIDANCE`, rules that came from auditing errors on some of these same 300 questions. How much of the lead remains when the baselines get the same instructions?

**Arms.** Five, run together in one batch per model and question set (`scripts/run_guided_baselines.py`):

| Arm | System |
| --- | --- |
| `multi_agent` | multi-agent v2 as already run (release vote with abstentions not voting) |
| `single_agent` | the single agent as already run |
| `direct_llm` | the direct baseline as already run |
| `single_agent_guided` | the single agent with the search guidance and `SEMANTIC_GUIDANCE` added to its system prompt. It already has `WORKED_EXAMPLES`, so it has every shared block the analysis prompt has. |
| `direct_llm_guided` | the direct baseline with `SEMANTIC_GUIDANCE` added. It has no tools, so it does not get the search guidance, which is about filtering records in code. |

The blocks are imported, not copied, and added verbatim, including where they disagree (the search guidance names `authoredOn` as the prescription date and `SEMANTIC_GUIDANCE` says otherwise; the analysis role sees both). Nothing else about the baselines changes: tools, starting packet, automatic check and abstention, the one revision, budgets (200,000 tokens and 30 tool calls per question) and server settings stay as they are. The guided prompts are set on subclass instances, so the code fingerprint is unchanged and `multi_agent` is the system already run. The guided prompts are fixed by this definition and are not adjusted on any result.

**Models.** gpt-oss:120b on vLLM (`scripts/run_experiments_vllm.sbatch`, port 28431) and qwen3:32b on Ollama (`scripts/run_experiments_qwen.sbatch`, port 11434), one H100 each on RIS Compute2.

**Question sets and order.**

1. Preflight: the two guided arms on the 50 dev questions, both models, only to check that they run and return valid answers.
2. Stage 1: the frozen 300 held-out questions (`manifests/split.json`, `held_out`; 58 patients), both models.
3. Stage 2, only after Stage 1 is analyzed and written up: the fresh sample of 300 (`manifests/pools/held_out_v2.json`, `held_out`; 58 patients, no question shared with the frozen 300). These questions were not used to derive the guidance, so this is the cleaner test.

Each held-out set is run once. Every question must be attempted; a job that dies is resubmitted and resumes. A failed first attempt counts as wrong and a retry does not change that, so prompts and retry behaviour are not altered to rescue failures. Comparisons use only runs from the same batch, never the saved runs: baselines have moved by up to 3 points between earlier batches.

**Scoring.** As in the existing analysis: first attempt, typed scorer, failed runs count as wrong, 95% patient-clustered bootstrap intervals (10,000 resamples), exact McNemar tests as a supplement.

**Primary comparison:** `multi_agent − single_agent_guided`, for each model and question set.

- If the interval excludes zero, a lead remains that the instructions do not explain.
- If it includes zero, these data do not show a lead beyond the instructions. That is not proof that there is none.

**Also reported,** whatever the outcome: `single_agent_guided − single_agent` (what the instructions are worth to the single agent); `multi_agent − direct_llm_guided` and `direct_llm_guided − direct_llm`; `multi_agent − single_agent` and `multi_agent − direct_llm` in the same batch; each arm's accuracy overall and by question family; failed runs by kind; mean tokens per question.

**What still differs** between multi-agent v2 and the guided single agent, so the result does not isolate role separation alone:

- Retrieval prefetches the patient's complete records of every relevant type; the single agent starts from the packet and fetches what it asks for.
- The analysis prompt tells it to answer 0 or false for an established absence, and null only when the records establish nothing.
- The single agent abstains (answer null) when its evidence fails the automatic check after its one revision; the team never withholds an answer on the verifier's word.
- The team releases by a vote among the draft, its repair, the packet reading and the verifier's recomputation, and then applies deterministic finishers for seven templates.
- The team spends more model calls and tokens per question under the same per-question ceilings.

## Stage 1: the frozen 300

Pending.

## Stage 2: the fresh 300

Pending.

## Reproducing

From `~/fhir-ma-v2` on Compute2, at this branch. One job per model and question set; the vLLM job serves on a fixed port, so gpt-oss jobs share a job name with `--dependency=singleton`. A job that stops early is resubmitted with the same command and resumes.

```bash
# Preflight: the guided arms on dev
SPLIT=dev RUNNER=scripts/run_guided_baselines.py sbatch --job-name=guided-gpt-oss --dependency=singleton \
    --time=3:00:00 scripts/run_experiments_vllm.sbatch --arms ordinary_single_agent_guided ordinary_direct_llm_guided
SPLIT=dev RUNNER=scripts/run_guided_baselines.py OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
    sbatch --job-name=guided-qwen3-32b --dependency=singleton --time=6:00:00 scripts/run_experiments_qwen.sbatch \
    --arms ordinary_single_agent_guided ordinary_direct_llm_guided

# Stage 1: all five arms on the frozen 300 (output/experiments/<model>/guided_held_out)
RUNNER=scripts/run_guided_baselines.py sbatch --job-name=guided-gpt-oss --dependency=singleton \
    scripts/run_experiments_vllm.sbatch
RUNNER=scripts/run_guided_baselines.py OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
    sbatch --job-name=guided-qwen3-32b --dependency=singleton scripts/run_experiments_qwen.sbatch

# Stage 2: the fresh 300 (output/experiments/<model>/guided_held_out_v2)
RUNNER=scripts/run_guided_baselines.py sbatch --job-name=guided-gpt-oss --dependency=singleton \
    scripts/run_experiments_vllm.sbatch --split_manifest manifests/pools/held_out_v2.json
RUNNER=scripts/run_guided_baselines.py OLLAMA_MODELS=/scratch2/fs1/di2cap/$USER/ollama-models \
    sbatch --job-name=guided-qwen3-32b --dependency=singleton scripts/run_experiments_qwen.sbatch \
    --split_manifest manifests/pools/held_out_v2.json

# Analysis (same flags as the run; add --split_manifest manifests/pools/held_out_v2.json for Stage 2)
python scripts/run_guided_baselines.py analyze --model gpt-oss:120b --base_url http://127.0.0.1:28431/v1 \
    --max_total_tokens 200000 --max_tool_calls 30
python scripts/run_guided_baselines.py analyze --model qwen3:32b --base_url http://127.0.0.1:11434/v1 \
    --max_total_tokens 200000 --max_tool_calls 30
python -m unittest tests.test_guided_baselines
```
