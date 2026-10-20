# Diagnosis and final design (V8): does a verified LLM team beat a tool-equipped single agent on FHIR-AgentBench?

Branch `experiments/diagnosis-v8`. This document consolidates seven completed
design iterations, a root-cause audit of their shared failures, and the one
remaining prespecified attempt (V8) with its companion experiments. Numbers
marked **TBD** are filled in from `benchmarks/design-v8-<date>/` after the
runs in [`launch.md`](launch.md) complete; nothing else in this file depends
on them.

## 1. Short answer

On gpt-6-luna, the proposal's hypothesis is **not supported**. Across seven
team designs on the same 300 questions, the team never beat a fresh
single-agent control (best: −1.0 pts, 95 % CI [−5.2, +3.2], at ~1.6× tokens).
The reason is structural, not a tuning problem: 82 % of the errors that every
design shares are *benchmark conventions the question does not state*
(which FHIR field "specimen name" means, that "first drug" is a tie set,
that counts are over the full prescription closure). A verifier built from
the same model holds the same priors and therefore finds nothing — which is
exactly what V5, V6 and V7 measured (0, 0 and 1 complementary corrections).

V8 replaces same-model verification with deterministic, audit-derived
contract checks routed through the proposal's fixed controller, plus an
optional heterogeneous-model audit arm. Its acceptance gate is frozen in
[`design_v8.txt`](design_v8.txt) before any result exists.

**V8 result (run C): gate failed.** Team 241/300 (80.3 %) vs fresh single
246/300 (82.0 %): −1.7 pts, 95 % CI [−4.8, +1.4], at 0.82× the single's
tokens. The checks fired on only 19 questions; the 6 released repairs were
3 wrong→right and 3 wrong→wrong with no harm, but the inherited V3d solver
revisions turned 6 right drafts wrong, so the arm's net transitions are 5 vs 6.
No confirmation is run ([`outcome.txt`](../../benchmarks/design-v8-20261016/outcome.txt)).

## 2. Completed paired runs (300 `held_out` questions, gpt-6-luna, 200k tokens / 30 tools)

All first attempts; terminal errors count as wrong; patient-block bootstrap
CIs from each branch's frozen `outcome.txt`.

| Design (branch) | Team | Single control | Δ team − single (95 % CI) | Team terminal errors | Mechanism |
|---|---|---|---|---|---|
| Archived fixed controller (`main`) | 162 (54.0 %) | 223 (74.3 %) | −20.3 | 42 | 30 budget stops; verifier abstentions |
| V2 verified team | 198 (66.0 %) | 231 (77.0 %) | −11.0 [−16.1, −6.0] | 7 | discarded 34 correct drafts, recovered 12; 98.6k vs 72.7k tokens |
| V3d query contract + static audit | 232 (77.3 %) | 238 (79.3 %)* | −2.0 [−5.2, +1.1] | 2 | audit proposed 0 repairs |
| V4 plan-conditioned | 218 (72.7 %) | 239 (79.7 %) | −7.0 [−10.7, −3.4] | 0 | 18 revisions: 2 helped, 0 harmed; plan exposure hurt the solver |
| V5 executable critic | 225 (75.0 %) | 237 (79.0 %)* | −4.0 [−8.1, −0.3] | 0 | 0 accepted repairs; 64 source-valid-but-wrong |
| V6 independent recompute | 231 (77.0 %) | 234 (78.0 %)* | −1.0 [−5.2, +3.2] | 5 | 0 A-wrong/B-right among 37 eligible |
| V7 residual trajectories | 233 (77.7 %) | 238 (79.3 %)* | −1.7 [−5.0, +1.6] | 6 | 1 A-wrong/B-right among 184 eligible; 115.0k vs 72.2k tokens; A revisions 3 neutral / 2 harmful |
| **V8 convention-verified** | **241 (80.3 %)** | **246 (82.0 %)*** | **−1.7 [−4.8, +1.4]** | 0 | checks fired on 19; repairs 6 released (3 W→R, 3 W→W), 10 rejected, 3 failed; 276 primaries preserved, 18 null; 60.7k vs 74.0k tokens |
| V8 + heterogeneous verifier | not run | same control | — | — | needs a second provider key |
| V9 dual-solver (`experiments/dual-solver-v9`) | 249 (83.0 %) | 244 (81.3 %)* | +1.7 [−1.4, +4.9] | 0 | gate passes, but controller net 0: 298/300 released the primary (which alone scores 249); 2 adjudicated changes, 1 W→R / 1 R→W; 9/18 adjudications hit the phase ceiling; 110.0k vs 75.1k tokens; see §5a |

\* fresh single run under that design's fingerprint. Direct LLM (no tools):
202/300 (67.3 %) archived.

Cost: every competitive team (V3d–V7) spent 1.5–1.6× the single's tokens for
a statistically indistinguishable or worse score. An accuracy-vs-tokens plot
over all arms is produced by `scripts/experiment_plots.py` once V8 outputs
exist (`scripts/run_experiments.py analyze`).

## 3. Where the errors are (Phase A)

`scripts/harvest_prior_runs.py` pulls the 13 first-attempt arms above (3,900
rows) from the frozen branches; `scripts/error_taxonomy.py` joins them with
the typed scorer and template manifest ([`taxonomy.md`](taxonomy.md)).

| Stability set (≥ 5 of 7 runs) | n | share of 300 |
|---|---|---|
| both right | 114 | 38 % |
| flaky (correct in some runs) | 137 | 46 % — 49 are terminal errors (provider timeouts), the rest split wrong_value / false_abstention |
| **both wrong core** | **45** | **15 %** |
| single-only wins | 3 | |
| team-only wins | 1 | |

The both-wrong core is dominated by five templates (30/45): T054 specimen
name (11), T051/T052 first/last drug or route (9), T001 count distinct drugs
(5), T043 admission type (4), T055 organism (3). Manual audit of all 45
against the gold SQL ([`root_causes.csv`](root_causes.csv), exposure recorded
in [`exposure.md`](exposure.md)):

| Root cause | n | Could same-model verification fix it? |
|---|---|---|
| convention — benchmark means a specific field / tie rule / closure the question never states | 28 | no |
| retrieval_gap — needed resource never fetched (hasMember children, preview cap) | 8 | yes, if the verifier knows what to fetch |
| label_issue — gold not representable in the released FHIR (e.g. `admission_type` collapsed into `Encounter.priority`) | 7 | no |
| format — right values, un-typeable shape | 2 | no |

37/45 = 82 % ≥ the 60 % threshold fixed in the plan → decision rule selects
"verifier that does not share the analyst's priors"
([`decision.md`](decision.md)).

Why the gap between team and single exists at all: it is not reasoning
quality (both are wrong on the same 45) but **budget pressure and verifier
side effects** — V2 overwrote 34 correct drafts; V4's exposed plan degraded
the solver; V3d–V7 changed nothing but spent the budget. The 3 single-only
wins are all false abstentions by the team.

## 4. What changed for the final attempt (Phases B3, C)

**B3 shared-guide fix** (`agent/ehr_semantics.CONVENTION_GUIDANCE`): the
audited conventions are stated once and given to *all three* systems —
appended to the single/team tool guide and injected into the direct LLM
system prompt (which previously had none). This is a fairness fix, not a
team mechanism; it changes the code fingerprint, so every arm re-runs fresh.

**V8** (`agent/convention_verified_agent.py`, strategy `multi_agent_v8`) keeps
V3d unchanged through its release decision, then runs five deterministic
checks on the released draft (`agent/convention_checks.py`: tie
completeness, specimen field, organism children, distinct-count scope,
timestamp field; an admission-type check was dropped after live
calibration showed its reference is not representable). Each issue is expressed in the
controller vocabulary (`missing_evidence` → retrieval, `wrong_field` /
`wrong_filter` → analysis) and routed by the existing `FixedController` to
exactly one repair that continues the solver's own trajectory (45k tokens /
6 tools inside the shared 200k / 30 ceiling). The repair is released only if
the source check passes, the triggering checks clear, and the answer is
non-null; otherwise the original draft stands — never an abstention.
Proposal shape preserved: fixed controller, three contexts, one targeted
repair.

**Heterogeneous verifier arm** (`--verifier_model`): only V3's static
program audit runs on another model family. Declared deviation from "same
base LLM in three contexts", justified by §3.

Offline verification: 605 tests pass (`tests/test_convention_checks.py`,
`tests/test_convention_verified_agent.py`, `tests/test_ehr_semantics.py`
added). V3 phase caps are asserted frozen and untouched.

## 5. Prespecified gate and companion experiments

Gate (frozen before launch; see `design_v8.txt`): team > fresh single
**and** CI lower bound of (team − single) > −2 pts **and** net released
repairs ≥ 0. Pass → one confirmation on a family-stratified 300 drawn once
from the protected questions (not in any split, patient-disjoint from all
exposed patients), scored once. Fail → reported here as a negative result;
no further dev-set iteration.

| Experiment | What it tests (proposal §) | Status | Result |
|---|---|---|---|
| C — paired 300 under the B3 fingerprint | H1: team > single | done 2026-10-16, 900/900, 0 errors | **gate failed**: −1.7 [−4.8, +1.4]; transitions 5 W→R / 6 R→W; direct 69.3 % |
| C' — heterogeneous verifier | breaks shared-model error correlation | prepared (§3) | **TBD** |
| B1 — recovery, 50 × intact/omitted × single/team | §5.3: does a team recover an omitted resource better? Archived hint: single −14 pts, team −4 pts, confounded by a 54 % team | prepared (§4) | **TBD** |
| B2 — weaker base model, 900 runs | verification pays when the base model has headroom (direct LLM is already 67 % on Luna) | model fixed before launch: `gpt-5-mini` (first candidate; check_setup and 5-question pilot smoke passed 2026-10-16); done 2026-10-17, 900 attempts, 3 terminal errors (2 direct invalid output, 1 team provider 520) counted wrong | **not supported**: team 230 (76.7 %) vs single 234 (78.0 %), −1.3 [−4.6, +2.1], wins/losses 12/16; direct 221 (73.7 %). V8 repairs 6 released (5 W→R, 1 W→W, 0 R→W) but team abstained on 51 vs single 40; 53.8k vs 60.5k tokens; median latency 110 s vs 40 s |
| Confirmation | generalisation off the burned 300 | not run: gate failed | — |

## 5a. Run-C ablation and the V9 design

Plan V9 starts from run C's per-question rows instead of a new architecture.
`scripts/replay_v9_controller.py predict` takes the single agent's released
answer as the primary and the V8 solver's candidate (`revision.primary`) as an
independent recompute. Counts only:

| Run C, 300 dev q, gpt-6-luna | n |
|---|---|
| single released / V8 solver candidate / V8 solver draft | 246 / 244 / 242 |
| both right / both wrong / single-only right / solver-only right | 231 / 41 / 15 / 13 |
| oracle pick between the two | 259 (+13) |
| primary null, recompute non-null | 12, of which 11 are single budget stops |
| single tokens mean / median / p90 | 74.0k / 63.3k / 146k |
| V8 solver-phase tokens mean / median / p90 | 52.6k / 38.4k / 118k |

The two scaffolds disagree in a complementary way. V6/V7's same-scaffold
recomputes found 0–1 such cases. V9 (`multi_agent_v9`, `design_v9.txt`) therefore
runs the unchanged single agent first on the full allowance. The query-contract
solver then recomputes on the residual minus a 15k adjudication reserve
(skipped below 25k). A fixed controller releases the primary on agreement,
applies a null rule, and sends both-non-null disagreements to one adjudication
call whose cited field must verify.

Replaying the controller (steps 3–4 exact, adjudication defaulting to the
primary) predicts **246, i.e. +0** (perfect-adjudicator bound 250). Policy
counts: agree 218, recompute skipped or truncated 64, adjudicate 13, null rule
primary 4 / recompute 1. The null-rule headroom sits on the single's budget
stops, and those are exactly the questions with no residual for the recompute.
With an unlimited recompute budget the replay gives 251, but that breaks the
equal-ceiling constraint. The prediction was written into `design_v9.txt`
before the run, and the run was launched as the prespecified test regardless
([`launch_v9.md`](launch_v9.md)).

Result (`benchmarks/design-v9-20261019/outcome.txt`): team 249 vs single 244,
+1.7 pts [−1.4, +4.9], so all three frozen gate items pass. The mechanism table
says the pass is not the treatment's doing. The team released the primary's
answer on 298 of 300 questions; its primary (the unchanged single scaffold)
scores 249 on its own, and the two adjudicated replacements cancelled (1 W→R,
1 R→W). The +5 is run-to-run variation between two runs of the same
single-agent code path, consistent with the pre-registered +0. Recompute ran on
282/300 (estimate 236), but 33 recomputes returned null, and half of the 18
adjudications defaulted because their prompt no longer fit the remaining
allowance. Cost: 1.47× the single's tokens.

## 6. Caveats

- The 300 `held_out` questions have been reused for eight design
  iterations; every number in §2 and any V8 number is exploratory. Only the
  confirmation run is confirmatory.
- Gold SQL was inspected for 45 development questions (36 patients) during
  the audit; agents never see it. The conventions in `CONVENTION_GUIDANCE`
  are therefore dev-set-derived and may not transfer to templates absent
  from the core — the confirmation run is the only test of that.
- 3–14 terminal errors per arm in every run were provider timeouts, not
  design failures; first-attempt scoring keeps them as wrong for
  comparability. Report both when writing up.
- No run in this document was launched from the diagnosis machine (no
  credentials or FHIR store); all V8/B1/B2 numbers come from the team's
  credentialed launcher following `launch.md`.

## 7. Files

| | |
|---|---|
| `scripts/harvest_prior_runs.py`, `scripts/error_taxonomy.py` | reproduce §3 from the frozen branches (writes `output/diagnosis/`, gitignored) |
| `taxonomy.md`, `root_causes.csv`, `audited_subject_ids.json`, `exposure.md` | Phase A artifacts |
| `decision.md` | decision rule applied, D1–D4 |
| `design_v8.txt` | text frozen by `record_design.py freeze` |
| `launch.md` | flag-checked commands for pre-flight, freeze, C, C', B1, B2, analysis, confirmation |
| `agent/convention_checks.py`, `agent/convention_verified_agent.py` | V8 |
| `agent/ehr_semantics.py`, `agent/direct_llm_agent.py` | B3 shared guide |
| `agent/dual_solver_agent.py`, `scripts/replay_v9_controller.py`, `design_v9.txt`, `launch_v9.md` | V9 (§5a) |
