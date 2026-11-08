# Phase 3 Results: qwen3:32b

Nov 5, 2026 · Akash Jairam

## Summary

With qwen3:32b, the single agent matched the direct baseline (52.7% vs 55.3% on 300 held-out questions, difference not significant), and the multi-agent system trailed both at 28.7%.

The multi-agent gap comes mostly from its verification step, not its analysis. Verification withheld 61 correct answers to block 38 wrong ones. Releasing whatever answer it reached, a rule chosen in advance on the dev split, raises multi-agent accuracy to 46.7%, still 6.0 points below the single agent. That figure is a post-hoc what-if applied to the saved runs, not a new run.

Neither tool-using system reliably recovers a record that is missing from its starting packet: the single agent fell from 76% to 26% when one was removed.

A larger model, qwen3.5:122b, keeps the same ranking: direct_llm 62.0%, single_agent 57.3% (a tie) and multi_agent 22.0% as built, 47.7% releasing every answer. gpt-oss:120b reached 71.0% without tools, but Ollama rejected many of its tool calls, so its agent results are not comparable.

## Setup

All three systems ran on the same base model, budget and tool cache, as the Phase 3 protocol requires.

| Item | Value |
| --- | --- |
| Model | qwen3:32b (Q4_K_M), served by Ollama, thinking off |
| Context window | 32,768 tokens |
| Hardware | One NVIDIA H100 80GB on WashU RIS Compute2 (c2-gpu-009) |
| Budget per question | 200,000 tokens and 30 tool calls |
| Systems | direct_llm (one call, no tools), single_agent, multi_agent |
| Ordinary runs | 300 held-out questions per system, 58 patients (900 runs) |
| Recovery runs | 50 perturbed questions, intact and omitted packets, tool-using systems only (200 runs) |
| Variability runs | 20-question subset, 5 repeats per system (300 runs) |
| Scoring | Typed scorer; 95% CIs from a patient-clustered bootstrap |
| Code | Branch fix/boolean-closing-verdict at commit 42eb617 |

The run took four Slurm jobs on 4-5 November 2026: one 7 h 16 min job, then three short follow-ups that retried failed questions.

## Main comparison

The direct baseline and the single agent are statistically tied; the multi-agent system is about 25 points behind both.

| System | Accuracy (95% CI) | Median latency | p95 latency | Mean tokens / question | Failed runs |
| --- | --- | --- | --- | --- | --- |
| direct_llm | 55.3% [49.4, 61.4] | 14.0 s | 123.9 s | 7,618 | 3 |
| single_agent | 52.7% [46.6, 58.6] | 42.8 s | 194.3 s | 47,801 | 8 |
| multi_agent | 28.7% [23.0, 34.7] | 93.3 s | 302.7 s | 37,832 | 41 |

Paired differences on the same 300 questions (95% CI):

- multi_agent − single_agent: −24.0 points [−30.3, −17.8]
- multi_agent − direct_llm: −26.7 points [−35.1, −18.6]
- single_agent − direct_llm: −2.7 points [−10.4, +4.5]

Failed runs count as wrong. The single agent costs about 6 times the tokens of the baseline for no accuracy gain.

Accuracy by question type (95% CI):

| Question type | direct_llm | single_agent | multi_agent |
| --- | --- | --- | --- |
| Counting (n=92) | **80.4%** [72.4, 87.6] | 56.5% [46.1, 67.0] | 44.6% [34.8, 54.3] |
| Aggregation (n=49) | 49.0% [34.7, 63.0] | **57.1%** [40.4, 71.7] | 30.6% [14.0, 47.2] |
| Time-based (n=159) | 42.8% [34.8, 51.1] | **49.1%** [41.0, 57.8] | 18.9% [11.9, 26.2] |

The overall tie between the baseline and the single agent hides a split: the baseline is far ahead on counting, while tool use helps on time-based questions.

## Recovery from omitted records

Removing one needed record from the starting packet cut the single agent's accuracy by 50 points, so it rarely went back to the FHIR store to find what was missing.

| System | Intact packet | Record omitted | Omitted − intact (95% CI) |
| --- | --- | --- | --- |
| single_agent | 76.0% | 26.0% | −50.0 points [−64.6, −35.6] |
| multi_agent | 14.0% | 4.0% | −10.0 points [−19.1, −2.3] |

50 questions per condition, 27 patients. The multi-agent drop is smaller only because it starts near the floor; with the record omitted it is still 22 points below the single agent [−35.1, −8.3]. Omitted-record accuracy for the single agent held up best on counting questions (53%) and collapsed on time-based ones (5%).

## Repeatability

The direct baseline gave the same result in all 5 repeats; both agent systems varied, and the multi-agent system abstained or errored in 82% of its runs.

| System | Mean accuracy (range) | Abstention rate | Error rate | Always / never / sometimes correct | Identical answers in every repeat |
| --- | --- | --- | --- | --- | --- |
| direct_llm | 60% (60–60) | 6% | 0% | 12 / 8 / 0 | 95% |
| single_agent | 60% (50–65) | 41% | 1% | 9 / 7 / 4 | 45% |
| multi_agent | 28% (25–35) | 55% | 27% | 4 / 13 / 3 | 40% |

20 questions, 5 repeats each. On this subset the single agent ties the baseline on average but swings by 15 points between repeats.

## Why multi_agent underperforms

On first attempts, the verification agent withheld 190 of 300 answers, and most of the withheld answers were correct.

Each withheld result keeps the answer the system reached before the gate: the revised answer, or the draft when there was none. Scoring those with the same typed scorer, on first attempts as in the official analysis:

| | Ordinary (300) | Recovery, intact (50) |
| --- | --- | --- |
| Runs that failed outright (counted wrong) | 41 | 8 |
| Answers withheld (null) | 190 | 29 |
| Withheld with an answer before the gate | 99 | 23 |
| That answer was correct | 61 | 19 |
| That answer was wrong (verification right to block) | 38 | 4 |
| Accuracy as reported (strict rule) | 28.7% | 14.0% |
| Accuracy releasing every answer reached (always rule) | 46.7% | 52.0% |

The checks that most often fired on correct answers were recompute_mismatch (33), where the verifier's own Python result disagreed; missing_evidence (31), where the verifier judged needed records missing; and independent_check (25), where the verifier could not complete its own fetch and recomputation. All three rest on the verifier model's judgment, not the deterministic field check. One answer can trigger several checks.

The always rule gains less than the 61 correct answers suggest: some withheld answers were correct abstentions on questions whose true answer is empty, and releasing a draft there turns them wrong. The rule was chosen on dev before being applied to held-out. Of three rules fixed in advance, always scored best on the 50 dev questions (46.0%, against 42.0% for field_check and 32.0% for strict). Applied once to held-out it gives 46.7% [40.6, 52.8]: 6.0 points below the single agent [−12.4, +0.7] and 8.7 below the baseline [−16.1, −1.6].

## Larger models

A larger model lifts the direct baseline more than the agents, so the ranking holds: with qwen3.5:122b the single agent ties the baseline and the multi-agent system trails both, even after releasing every answer it reached.

Both models ran the same protocol as qwen3:32b (same 300 held-out questions, budget, tool cache, 32k context and typed scorer). qwen3.5:122b (81 GB, thinking off) ran on two H100s at commit 8503cf6, 5-8 November 2026. gpt-oss:120b (65 GB, reasoning effort medium) ran on one H100 at commit a5cd295, 7-8 November 2026.

| Held-out result | qwen3:32b | qwen3.5:122b | gpt-oss:120b (see below) |
| --- | --- | --- | --- |
| direct_llm | 55.3% | **62.0%** [57.0, 67.0] | 71.0% [66.4, 75.3] |
| single_agent | 52.7% | **57.3%** [51.7, 62.6] | 61.3% [55.7, 67.0] |
| multi_agent, strict rule (as built) | 28.7% | **22.0%** [16.2, 27.8] | 7.3% [4.3, 10.7] |
| multi_agent, always rule | 46.7% | **47.7%** [42.0, 53.1] | 17.3% [13.5, 21.5] |
| single_agent − direct_llm | −2.7 [−10.4, +4.5] | −4.7 [−10.3, +0.6] | −9.7 [−14.8, −4.6] |
| multi_agent (always) − single_agent | −6.0 [−12.4, +0.7] | −9.7 [−15.1, −4.0] | −44.0 [−50.5, −37.1] |
| single_agent recovery, intact → omitted | 76% → 26% | 74% → 30% | 76% → 20% |
| multi_agent failed first attempts (of 300) | 41 | 107 | 227 |
| single_agent median latency | 42.8 s | 205.7 s | 22.7 s |
| single_agent mean tokens / question | 47,801 | 102,001 | 29,891 |

Accuracy with 95% patient-clustered CIs; differences in points. Failed runs count as wrong.

**qwen3.5:122b, what changed.** The baseline gained 6.7 points over qwen3:32b and the single agent 4.6, so tool use still buys nothing on accuracy. It does cost 13 times the baseline's tokens and about 10 times its latency. The multi-agent system as built got worse (22.0%), because more of its runs failed outright: 51 first attempts broke the output format, 34 hit the turn limit and 17 got empty model replies.

**Release rule.** On the dev split, field_check and always tied at 54.0% (strict 30.0%). No tie-break was fixed in advance, so both are reported: field_check gives 39.3% [33.4, 45.1] on held-out and always gives 47.7%. Picking always now would choose the rule from held-out results. Under either rule, multi_agent is significantly behind the single agent, unlike with qwen3:32b. On intact recovery packets the rules give 36.0% (strict), 52.0% (field_check) and 60.0% (always).

**Repeatability (20 questions, 5 repeats).** qwen3.5:122b is far more stable than qwen3:32b. direct_llm scored 75% in every repeat and single_agent 55% in every repeat, with identical answers across repeats. multi_agent averaged 27% (20-30%), abstained in 54% of runs and errored in 32%.

**Recovery.** Removing one needed record cut the single agent from 74% to 30% (−44.0 points [−60.8, −26.5]), the same pattern as qwen3:32b. With the record omitted, multi_agent (22%) and single_agent are no longer significantly different (−8.0 [−23.8, +8.9]).

## gpt-oss:120b: tool-using results not comparable

gpt-oss:120b's direct baseline (71.0%) is the best score of any system on any model, but its tool-using results mostly measure Ollama rejecting its tool calls, not the systems.

Ollama 0.40.13 answers a gpt-oss reply whose tool-call arguments are not valid JSON with HTTP 500, so the call never reaches the agent. Most rejected calls are Python code arguments written as `"code":"…"]` with a stray bracket.

| First attempts that failed (of 300) | single_agent | multi_agent |
| --- | --- | --- |
| Tool call rejected by Ollama | 50 | 103 |
| Turn limit reached without a final reply | 0 | 85 |
| Output format or validation error | 1 | 37 |
| Context limit | 1 | 2 |
| **Total** | **52** | **227** |

- **Fix attempted:** commit a5cd295 sends the rejection reason back to the model and asks it to call again. It did not help: at temperature 0 the model repeated the same malformed call on all five tries.
- **Client-side repair:** removing a stray bracket after a string value would fix 411 of the 541 rejected calls saved so far (76%). The rest are varied (unescaped quotes, plain text instead of JSON).
- **Release rule:** always scored best on dev (22.0%, against 20.0% field_check and 10.0% strict) and gives 17.3% on held-out; most multi_agent runs fail before reaching an answer.
- **Repeatability (20 questions, 5 repeats):** direct_llm 69% (65-70%), single_agent 56% (50-60%) with 18% errors, multi_agent 1% with 83% errors.

The 85 turn-limit failures are gpt-oss's own behaviour and would remain under any fix. A fair rerun would serve gpt-oss with vLLM, which implements its tool-call format directly, or add the client-side repair; neither has been run.

## Run integrity

The first full run (3 November) is invalid for both tool-using systems and is excluded; these results come from the rerun.

- **Cause:** RIS Compute2 ships bubblewrap 0.4.1, which rejects the `--clearenv` option the Python sandbox passed. Every sandbox call failed, touching 290 of 300 single_agent questions and all 300 multi_agent questions. direct_llm uses no tools and was unaffected (56.3% then, 55.3% in the rerun).
- **Fix:** commit 42eb617 drops `--clearenv`, since the sandbox already receives a minimal explicit environment. The real sandbox tests (no network, no workspace access, no inherited environment) pass on the cluster. The Slurm job now refuses to start if the sandbox fails a self-test.
- **Check:** the rerun's results contain zero sandbox failures in every arm.
- **Fingerprints:** five LangGraph dependencies are now pinned, so laptop and cluster produce the same code fingerprint (55ae0ef…).
- **Retries:** the three follow-up jobs retried failed questions: 69 retries for multi_agent, 15 for single_agent, 7 for direct_llm. Remaining failures count as wrong.

The invalid results are kept on the cluster under `output/experiments/qwen3:32b/held_out_old_bwrap_sandbox_broken`.

## Caveats and next steps

These results describe one 32B model with a 32k context; they do not show how the systems compare with a stronger model.

- **Context limit:** with 32k tokens, some patients' records do not fit, which hurts the systems that load whole record sets.
- **One model judges nothing here:** accuracy comes from the typed scorer, so no LLM grades its own answers.
- **Retries:** accuracy uses each question's first attempt, as in the official analysis; the follow-up jobs' retries (69 for multi_agent) only add to the cost totals.
- **Verifier variant:** three release rules were fixed in advance: strict (as built), field_check (withhold only on a deterministic field-check issue) and always. always scored best on the 50-question dev split and, applied once to the saved held-out runs (`scripts/release_policy.py`), gives 46.7%; no agent rerun was needed.
- **Stronger model:** qwen3.5:122b confirms the qwen3:32b ranking (see Larger models). gpt-oss:120b's tool-using results are not comparable until it is served without Ollama's tool-call rejections.
