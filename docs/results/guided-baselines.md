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

**Verdict.** With gpt-oss:120b, multi-agent v2 leads the guided single agent by 6.7 points [+2.5, +10.8]: a lead remains that the instructions do not explain. With qwen3:32b the lead is 6.0 points [−0.3, +13.4]: these data do not show a lead beyond the instructions, which is not proof that there is none.

**Runs.** All five arms ran in one batch per model at commit cdbc037 (code fingerprint `78f297746133`, the same as the saved v2 results), 18–19 November 2026: gpt-oss:120b on vLLM in 1 h 38 min (job 4600217), qwen3:32b on Ollama in 6 h 45 min (job 4600265). Every arm attempted all 300 questions; nothing was resubmitted. The preflight on dev (jobs 4600140 and 4600141) returned valid answers for both guided arms on both models, at the same answer rate as the existing baselines, and every saved trace carried the intended blocks; one dev question failed per guided arm on qwen3:32b. Dev results were not looked at for accuracy.

Accuracy over 300 questions (58 patients), with 95% patient-clustered CIs:

| Arm | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| multi_agent | **80.3%** [75.7, 84.8] | **62.0%** [56.7, 67.4] |
| single_agent_guided | 73.7% [69.0, 78.2] | 56.0% [49.8, 61.6] |
| single_agent | 71.0% [66.4, 75.0] | 49.7% [43.5, 55.5] |
| direct_llm_guided | 78.7% [74.6, 82.6] | 55.0% [49.4, 60.7] |
| direct_llm | 72.0% [67.6, 76.1] | 55.7% [49.7, 61.7] |

Paired differences in points, a − b on the same questions, with the exact McNemar test (questions only a / only b got right):

| Comparison | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| **multi_agent − single_agent_guided** (primary) | **+6.7 [+2.5, +10.8]**; 31 / 11, p = 0.003 | **+6.0 [−0.3, +13.4]**; 60 / 42, p = 0.09 |
| single_agent_guided − single_agent | +2.7 [−0.3, +5.7]; 14 / 6, p = 0.12 | +6.3 [+1.9, +10.5]; 31 / 12, p = 0.005 |
| multi_agent − direct_llm_guided | +1.7 [−2.0, +5.3]; 18 / 13, p = 0.47 | +7.0 [+1.9, +11.9]; 44 / 23, p = 0.014 |
| direct_llm_guided − direct_llm | +6.7 [+2.9, +10.5]; 25 / 5, p = 0.0003 | −0.7 [−2.9, +1.7]; 6 / 8, p = 0.79 |
| multi_agent − single_agent | +9.3 [+5.9, +13.0]; 37 / 9, p < 0.0001 | +12.3 [+5.7, +19.5]; 71 / 34, p = 0.0004 |
| multi_agent − direct_llm | +8.3 [+3.4, +13.4]; 38 / 13, p = 0.0006 | +6.3 [+1.2, +11.4]; 40 / 21, p = 0.02 |

Accuracy by question family (multi / single guided / single / direct guided / direct):

| Family | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| Counting (n=92) | 92.4 / 79.3 / 79.3 / 87.0 / 83.7 | 79.3 / 58.7 / 51.1 / 80.4 / 80.4 |
| Aggregation (n=49) | 79.6 / 79.6 / 75.5 / 85.7 / 79.6 | 59.2 / 57.1 / 57.1 / 49.0 / 49.0 |
| Time-based (n=159) | 73.6 / 68.6 / 64.8 / 71.7 / 62.9 | 52.8 / 54.1 / 46.5 / 42.1 / 43.4 |

Failed first attempts (counted as wrong) and cost per question:

| Arm | gpt-oss: failed | mean tokens | median latency | qwen3:32b: failed | mean tokens | median latency |
| --- | --- | --- | --- | --- | --- | --- |
| multi_agent | 1 (input too long) | 80,774 | 35.3 s | 0 | 69,114 | 110.4 s |
| single_agent_guided | 4 (invalid output) | 37,164 | 8.8 s | 12 (invalid output 9, empty reply 3) | 50,133 | 42.5 s |
| single_agent | 4 (invalid output) | 34,078 | 9.6 s | 10 (invalid output 7, empty reply 3) | 44,500 | 38.2 s |
| direct_llm_guided | 0 | 7,869 | 4.3 s | 4 (invalid output) | 8,896 | 14.0 s |
| direct_llm | 2 (invalid output) | 6,525 | 4.0 s | 4 (invalid output) | 7,634 | 13.0 s |

Findings:

- **The instructions explain part of the lead on both models, not all of it on gpt-oss.** The lead over the single agent falls from 9.3 to 6.7 points on gpt-oss:120b and from 12.3 to 6.0 on qwen3:32b. On gpt-oss:120b the remaining lead is significant (p = 0.003); on qwen3:32b it is not (p = 0.09), but the point estimate is the same size.
- **The instructions are worth more to the weaker single agent.** They add 6.3 points [+1.9, +10.5] to qwen3:32b's single agent, mostly on time-based (+7.6) and counting (+7.6) questions, and 2.7 [−0.3, +5.7] to gpt-oss:120b's. On qwen3:32b the guided single agent matches multi-agent on time-based questions (54.1 vs 52.8%); what remains of the lead is counting (79.3 vs 58.7%), where the single agent has trailed the direct baseline since Phase 3.
- **On gpt-oss:120b the guided direct baseline nearly closes the gap.** `SEMANTIC_GUIDANCE` adds 6.7 points [+2.9, +10.5] to the direct baseline (25 questions gained, 5 lost), and multi-agent leads it by only 1.7 [−2.0, +5.3]. On qwen3:32b the same guidance does nothing for the direct baseline (−0.7), and multi-agent keeps a 7.0-point lead over it [+1.9, +11.9].
- **The guidance is cheap and does not change failure rates.** It adds 1,300 tokens per question to the direct baseline and 3,100–5,600 to the single agent. Multi-agent still uses 2.2× (gpt-oss) and 1.4× (qwen3) the guided single agent's tokens and 2.6–4× its latency. Failed runs are within two of the unguided arms'.
- **These are the questions the conventions were audited on.** Some `CONVENTION_GUIDANCE` rules came from errors on these 300 questions, which favours every guided arm and multi-agent alike here; Stage 2 uses questions that played no part in it.

The unguided arms land where the saved v2 batch had them (gpt-oss:120b: single 72.0 → 71.0%, direct 71.7 → 72.0%, multi 81.0 → 80.3%; qwen3:32b: 49.7, 55.7 and 61.0 → 62.0%). These are context only; every comparison above is within this batch.

**Deviations from the design note:** none. The design note planned exact McNemar tests as a supplement and reports them; no multiple-comparison correction was planned and none is applied.

Result files on Compute2: `~/fhir-ma-v2/output/experiments/<model>/guided_held_out/` (one JSON file per arm, plus `analysis/` with `summary.json`, `accuracy.csv`, `comparisons.csv` with every comparison by family, and `latency.csv`). Logs: `logs/guided-gpt-oss-4600217.out`, `logs/guided-qwen3-32b-4600265.out`, `logs/analysis/guided_stage1_*.txt`.

## Stage 2: the fresh 300

**Verdict.** With gpt-oss:120b, multi-agent v2 leads the guided single agent by 2.3 points [−3.0, +7.4]: these data do not show a lead beyond the instructions, which is not proof that there is none. With qwen3:32b the lead is 6.3 points [+0.6, +12.0]: a lead remains that the instructions do not explain. This is the reverse of Stage 1.

**Runs.** All five arms ran in one batch per model at commit cdbc037 (code fingerprint `78f297746133`, no stale warning), 19 November 2026, after Stage 1 was written up: gpt-oss:120b on vLLM in 1 h 37 min (job 4606603), qwen3:32b on Ollama in 7 h 12 min (job 4606604). Every arm attempted all 300 questions; nothing was resubmitted and no failed question was retried.

Accuracy over 300 questions (58 patients), with 95% patient-clustered CIs:

| Arm | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| multi_agent | **72.7%** [67.6, 77.6] | **54.7%** [49.8, 59.5] |
| single_agent_guided | 70.3% [65.9, 74.7] | 48.3% [42.2, 54.5] |
| single_agent | 66.0% [61.9, 70.3] | 46.3% [40.4, 52.6] |
| direct_llm_guided | 68.3% [63.8, 73.1] | 50.0% [44.6, 55.3] |
| direct_llm | 64.0% [59.0, 69.3] | 48.0% [42.5, 53.5] |

Paired differences in points, with 95% patient-clustered CIs, over the same 300 questions, with the exact McNemar test (questions only a / only b got right):

| Comparison | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| **multi_agent − single_agent_guided** (primary) | **+2.3 [−3.0, +7.4]**; 27 / 20, p = 0.38 | **+6.3 [+0.6, +12.0]**; 56 / 37, p = 0.06 |
| single_agent_guided − single_agent | +4.3 [+1.4, +7.5]; 18 / 5, p = 0.011 | +2.0 [−2.3, +6.3]; 22 / 16, p = 0.42 |
| multi_agent − direct_llm_guided | +4.3 [+1.0, +7.6]; 25 / 12, p = 0.047 | +4.7 [−0.7, +9.8]; 39 / 25, p = 0.10 |
| direct_llm_guided − direct_llm | +4.3 [+1.1, +7.5]; 21 / 8, p = 0.024 | +2.0 [−0.4, +4.6]; 13 / 7, p = 0.26 |
| multi_agent − single_agent | +6.7 [+2.0, +11.0]; 34 / 14, p = 0.0055 | +8.3 [+2.7, +13.6]; 62 / 37, p = 0.015 |
| multi_agent − direct_llm | +8.7 [+4.1, +13.2]; 36 / 10, p = 0.0002 | +6.7 [+1.3, +11.8]; 42 / 22, p = 0.017 |

Accuracy by question family (multi / single guided / single / direct guided / direct):

| Family | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| Counting (n=92) | 81.5 / 71.7 / 69.6 / 76.1 / 73.9 | 70.7 / 48.9 / 47.8 / 63.0 / 64.1 |
| Aggregation (n=49) | 59.2 / 61.2 / 57.1 / 57.1 / 61.2 | 32.7 / 38.8 / 38.8 / 34.7 / 26.5 |
| Time-based (n=159) | 71.7 / 72.3 / 66.7 / 67.3 / 59.1 | 52.2 / 50.9 / 47.8 / 47.2 / 45.3 |

Failed first attempts (counted as wrong) and cost per question:

| Arm | gpt-oss: failed | mean tokens | median latency | qwen3:32b: failed | mean tokens | median latency |
| --- | --- | --- | --- | --- | --- | --- |
| multi_agent | 1 (input too long) | 81,729 | 35.0 s | 4 (analysis reply not a usable draft) | 75,081 | 124.5 s |
| single_agent_guided | 10 (invalid output) | 36,587 | 8.5 s | 17 (invalid output 12, empty reply 3, timeout 2) | 51,158 | 40.1 s |
| single_agent | 12 (invalid output 10, input too long 2) | 37,431 | 9.1 s | 15 (invalid output 8, empty reply 5, input too long 1, timeout 1) | 50,294 | 36.6 s |
| direct_llm_guided | 1 (invalid output) | 7,893 | 4.3 s | 6 (invalid output) | 8,907 | 14.8 s |
| direct_llm | 5 (invalid output) | 6,523 | 3.9 s | 10 (invalid output) | 7,645 | 14.7 s |

Findings:

- **On fresh questions the guided single agent comes within 2.3 points of multi-agent on gpt-oss:120b.** The lead over the single agent falls from 6.7 to 2.3 points, and the guided single agent matches multi-agent on time-based (72.3 vs 71.7%) and aggregation (61.2 vs 59.2%) questions. On qwen3:32b the lead falls from 8.3 to 6.3 points and stays significant on the interval, though the McNemar test is not (p = 0.06).
- **What remains of the lead is counting, on both models.** Multi-agent leads the guided single agent on counting by 9.8 points on gpt-oss:120b and 21.8 on qwen3:32b, and by at most 1.3 points on any other family (it trails on aggregation on both). Stage 1 showed the same pattern on qwen3:32b. On counting the direct baselines also beat both single-agent arms, so this looks like a weakness of the single agent's loop, not something the instructions address.
- **The instructions still help on questions they were not derived from.** They add 4.3 points to both gpt-oss:120b baselines (single [+1.4, +7.5], direct [+1.1, +7.5]), mostly on time-based questions, and 2.0 points to both qwen3:32b baselines, where neither interval excludes zero. On gpt-oss:120b the single agent gains more here than on the frozen 300 (+4.3 vs +2.7); on qwen3:32b it gains less (+2.0 vs +6.3).
- **Multi-agent keeps a lead over the guided direct baseline** of 4.3 points on gpt-oss:120b [+1.0, +7.6] and 4.7 on qwen3:32b [−0.7, +9.8], wider on gpt-oss than on the frozen 300 (+1.7).
- **Cost and failures look like Stage 1.** The guidance adds about 1,300 tokens per question to the direct baseline and changes the single agent's by under 900. Multi-agent uses 2.2× (gpt-oss) and 1.5× (qwen3) the guided single agent's tokens and 3.1–4.1× its latency. The guided arms failed less often than their unguided counterparts in three of the four pairs; qwen3:32b's multi-agent failed 4 times where it failed none on the frozen 300, each an analysis reply with no usable answer.

The unguided arms land near the saved suite batch on these questions (gpt-oss:120b: single 68.0 → 66.0%, direct 63.7 → 64.0%; qwen3:32b: single 43.0 → 46.3%, direct 48.0 → 48.0%), but multi-agent is lower than in that batch on both models (gpt-oss:120b 75.0 → 72.7%, qwen3:32b 58.3 → 54.7%), a larger batch-to-batch shift than the 3 points seen before. These are context only; every comparison above is within this batch.

**Deviations from the design note:** none. No multiple-comparison correction was planned and none is applied; with two models and two question sets there are four primary comparisons, and the qwen3:32b Stage 2 interval's lower end (+0.6) is close to zero.

Result files on Compute2: `~/fhir-ma-v2/output/experiments/<model>/guided_held_out_v2/` (one JSON file per arm, plus `analysis/`). Logs: `logs/guided-gpt-oss-4606603.out`, `logs/guided-qwen3-32b-4606604.out`, `logs/analysis/guided_stage2_*.txt`.

### Both stages

| multi_agent − single_agent_guided | gpt-oss:120b | qwen3:32b |
| --- | --- | --- |
| Stage 1, frozen 300 | +6.7 [+2.5, +10.8] | +6.0 [−0.3, +13.4] |
| Stage 2, fresh 300 | +2.3 [−3.0, +7.4] | +6.3 [+0.6, +12.0] |

Giving the baselines multi-agent's instructions shrinks its lead on every model and question set, from 6.7–12.3 points to 2.3–6.7. Each model has one stage whose interval excludes zero and one whose interval does not, so neither model shows a consistent lead beyond the instructions. The point estimates for qwen3:32b are steady at about 6 points; those for gpt-oss:120b fall to 2.3 on the questions the conventions were not audited on. In both stages and on both models, whatever lead remains comes mostly from counting questions.

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
