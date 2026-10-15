# V8 decision memo

Written before any new LLM run. Inputs: `taxonomy.md` (automatic stable
sets over the seven completed paired runs) and `root_causes.csv` (manual
audit of the 45-question both-wrong core plus the 4 asymmetric wins).
Patients whose gold SQL/reference was inspected are listed in
`audited_subject_ids.json`; they were already development-exposed patients.

## 1. What the audit found

Both-wrong core = 45 questions (15 pts). Root causes:

| root cause | n | fix type | verification could help? |
|---|---|---|---|
| convention (benchmark convention not stated anywhere in the prompt) | 28 | shared guide | no |
| retrieval_gap (needed resource never fetched / preview cap) | 8 | tool + guide | yes |
| label_issue (gold arguably wrong or not representable in FHIR) | 7 | none | no |
| format (correct values, un-typeable output shape) | 2 | shared guide | no |

Convention + label + format = 37/45 = **82 %** (≥ 60 % threshold of the
plan's decision rule). Five templates account for 30 of the 45:

- **T054 specimen name (11)** — gold is `spec_type_desc`, i.e. the
  `Specimen.type` display. Agents answer the test name from
  `Observation.code`.
- **T051/T052 first/last drug or route (7+2)** — gold returns the *whole tie
  set* of distinct drugs sharing the selected `starttime`; agents return one.
- **T001 count distinct drugs (5)** — gold counts 29–97 distinct drug strings;
  agents count a 32-record retrieval preview.
- **T043 admission type (4)** — gold is `admissions.admission_type`
  (`EW EMER.`, `OBSERVATION ADMIT`, …). The released Encounter carries it only
  as a normalized `priority` coding (v3-ActPriority) that collapses categories
  (`scripts/packet_query.py` refuses `admission_type` for the same reason), so
  the raw string is not recoverable; agents answer `Encounter.class`. Counted
  as label_issue.
- **T055 organisms (3)** — organisms are child Observations reachable only
  through `hasMember`; agents never fetch them.

The remaining tail (T064/T073 timestamp field at minute precision, T056/T050
exact drug-string match, T069 tie-break, T017 window-scoped existence,
T031 per-period output shape, T030/T039 label issues) follows the same
pattern: the data was retrievable and the arithmetic trivial; what was
missing is *which field the benchmark means*.

## 2. Why seven same-model verifiers found nothing

Every design from V2 to V7 asked gpt-6-luna to check gpt-6-luna. Conventions
are shared priors: an analyst that reads `Encounter.class` for "admission
type" is verified by a model that holds the same belief. Blind recomputation
(V6), executable critics (V5) and independent trajectories (V7) therefore
produced 0, 0 and 1 complementary corrections. The 8 retrieval-gap questions
are the only subset where an independent check could have helped, and even
there the verifier needs to know *what* to look for (child organisms, the
full medication closure).

Consequence: more orchestration on the same model cannot move the 15-point
core. The gap between team and single (−2 to −7 pts) is not reasoning error
either; it is budget pressure and verifier-induced abstention/overwrites
(V2: 34 lost drafts; V4: plan exposure).

## 3. Decisions

### D1. Shared-guide fix (B3) — applies to all three systems

`agent/ehr_semantics.SEMANTIC_GUIDANCE` gains data-representation rules for
the convention findings: coded-display verbatim answers (`Specimen.type`,
`Encounter.priority`), tie-set completeness for first/last-by-time selections,
exact drug/route string matching, distinct-count over the full Medication
closure, `charttime` ≙ parent test `effectiveDateTime`, window-scoped
existence, plain-list output for per-period aggregates. These are statements
about how MIMIC fields are released in FHIR and how the question language maps
to them; they do not contain reference answers or per-question information.
They apply to direct, single and team alike, so no arm benefits unilaterally.
This is a development exposure and is recorded in this directory.

Expected effect: the single agent should gain as much as the team. The B3
fix is a fairness correction, not the hypothesis test.

### D2. V8 design — rule branch "≥ 60 % convention"

Per the plan's decision rule the final design is a **convention-aware
verifier that does not share the analyst's priors**:

1. **Deterministic contract checks** (`agent/convention_checks.py`), run by
   the host with no model call on the analyst's executed draft:
   - `tie_completeness`: a first/last-by-time selection over drugs/routes
     must return every distinct value sharing the selected timestamp.
   - `verbatim_code`: specimen-name questions must answer a display string
     that occurs verbatim in a cited `Specimen.type`; admission-type answers
     must come from `Encounter.priority`, not `Encounter.class`.
   - `timestamp_field`: a "when" answer over prescriptions / microbiology must
     equal `validityPeriod.start` / the parent test `effectiveDateTime` of a
     cited record (minute precision).
   - `distinct_count_preview`: a distinct-count answer equal to or bounded by
     the retrieval preview size is flagged as `missing_evidence`.
   - `child_observation`: an organism question whose cited test has
     `hasMember` entries that were never retrieved is `missing_evidence`.
   Each check emits an issue in the controller's existing vocabulary
   (`wrong_field`, `wrong_filter`, `missing_evidence`) so `FixedController.route`
   sends it to the responsible role (analysis or retrieval) for the single
   permitted repair. This keeps the proposal's shape: fixed controller, three
   contexts, one targeted repair.
2. **Heterogeneous verifier (optional arm)**: `--verifier_model` lets the
   model-based verification phase run on a different model family while
   retrieval and analysis stay on the base model. Token ceilings are counted
   in aggregate. This is a stated deviation from the proposal's "same base
   LLM in three contexts" and is justified by the measured same-model error
   correlation above. It is an *arm*, not the default, so the paper can
   report both.

The override gate stays conservative (V5 discipline): a repaired answer
replaces the draft only if it passes the source check and the deterministic
check that triggered the repair; otherwise the original source-valid draft
is released, never an abstention.

### D3. Weaker-model arm (B2)

Direct LLM already scores 67 % on gpt-6-luna; the hypothesis predicts that a
verification team pays off when the base model has headroom. B2 runs
direct / single / team on the 300 dev questions with the smallest Azure
deployment that passes `scripts/check_setup.py` and the five-question smoke
test (candidate order: gpt-5-mini, gpt-5.4-mini). Fixed before launch.

### D4. Recovery rerun (B1)

The only hint of a team advantage (archived: single −14 pts, team −4 pts
under omission) was measured with a 54 % team. B1 re-runs the four recovery
arms with the V3 team and the fresh single under the B3 guide.

## 4. Prespecified gate for V8 (copied into `benchmarks/design-v8-<date>/design.txt`)

On the 300 development questions, paired with a fresh single under the same
code fingerprint:

- team typed-accuracy > fresh single, **and**
- patient-block bootstrap 95 % CI lower bound of (team − single) > −2 pts, **and**
- net revisions (helpful − harmful) ≥ 0.

Only if all three hold: one confirmation run on a family-stratified
300-question sample of the protected 1,397 set, patient-disjoint from every
exposed patient, scored once. Otherwise the result is reported as negative.

## 5. Run order (all on the team's credentialed machine)

Exact, flag-checked commands live in [`launch.md`](launch.md); the design
text frozen before launch is [`design_v8.txt`](design_v8.txt). Order:
pre-flight smoke (pilot split) → `record_design.py freeze` → C paired 300
(`--split held_out --arms ordinary --team_strategy multi_agent_v8`) →
heterogeneous arm (`--verifier_model`) → B1 (`--arms recovery`) → B2
(`--model <weaker>`) → `analyze` / `compare_runs.py` → gate → confirmation
only if the gate passes. Note the burned "300 development questions" are
`split.json` → `held_out`; `--split dev` is the 50-question smoke split.
