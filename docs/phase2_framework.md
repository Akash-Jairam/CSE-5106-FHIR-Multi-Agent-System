# Phase 2 Agent Framework (P2-11)

## Decision

The multi-agent system reuses **LangGraph 1.2.14** for execution and state
message passing, as promised in the proposal. `FixedWorkflow` in
[`agent/framework.py`](../agent/framework.py) builds a `StateGraph` whose
nodes are ordinary Python role handlers and whose edges are fixed or
deterministically conditional. The project still implements the team
protocol, JSON message contracts, and repair classification itself.

LangGraph does not require LangChain chat models: its
[overview](https://docs.langchain.com/oss/python/langgraph/overview) and
[Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
describe Python function nodes and deterministic edges. The earlier
in-house-only decision incorrectly treated optional framework features as
mandatory. The selected [pinned release](https://pypi.org/project/langgraph/1.2.14/)
supports Python 3.10 and newer and is installed through `requirements.txt`.
There is no framework-specific model credential or cloud service requirement.

The adapters preserve Phase 1's constraints:

- **One budget per question.** Phase 1 budget controls (#13) cap
  completions, count automatic checks, and refuse unaccounted usage. They
  only work if every model call goes through `BaseAgent._llm_call`. A
  framework only invokes Python nodes; it never owns the LLM loop.
- **Sandboxed code only.** Generated Python must run in
  `tools/python_executor.py` (OS sandbox, copied data, real deadline).
  The graph uses our existing registry adapter, not a built-in code executor.
- **A deterministic controller.** P2-12 requires a non-LLM controller, so
  ordinary conditional edges execute the controller's own routing policy.
- **The common contract and resume identity.** The system must still
  return `AgentOutput` through `BaseAgent.run()`, and `run_agent.py`
  fingerprints `agent/`, `tools/`, `utils/`, and `requirements.txt` for
  resume, including installed versions of declared dependencies and the
  graph runtime's transitive packages. The adapter and protocol remain
  in those directories; changing an installed framework version invalidates
  saved result identity even when the Python source is unchanged.
- **Local-only graph execution.** Checkpointing, graph retries, caching,
  remote runtime calls, and external tracing are disabled. A
  `tracing_context(enabled=False)` overrides a developer's tracing environment
  without modifying it, so question and record data are not sent to LangSmith.

## The Adapters

- **`FixedWorkflow`** executes role nodes through a compiled LangGraph
  `StateGraph`. It has no checkpointer, retry policy, or cache. A small
  graph-step ceiling guards accidental controller cycles independently of
  the per-question token/tool ceilings.

- **`Bus`** logs structured `Message(sender, recipient, kind, payload)`
  values. Payloads must be finite JSON objects and are copied when sent, so
  returning a message or exporting the log also gives a copy, so callers
  cannot rewrite prior messages. The log becomes the coordination
  record for revision tracking (P3-22).
- **`Toolbox`** is one role's allow-listed view of `tools/registry.py`.
  Model arguments are limited to each tool's declared parameters (so a model
  cannot pass `global_vars` or `resources`). Question context is bound by
  the system. Patient-wide retrieval is limited to the question's patient.
  Fetched records that belong to another patient, or are dated after the
  assumed current time, are left out, and the tool reply counts them.
  Retrieval updates the role's working records, which `execute_python_code`
  receives as `retrieved_resources`. Roles that should see the same records
  share the dict; the verifier gets its own.
- **`Role`** is one LLM role with a system prompt and toolbox. Each
  `run()` starts a fresh context, which the verifier (P2-15) needs. Model
  calls, tool-call counts, and trace messages go to the host `BaseAgent`, so
  all roles answer to one ceiling and `tool_calls` covers the whole system.

The multi-agent system will be a `BaseAgent` subclass whose `_run()` is the
fixed controller (P2-12). It creates roles (P2-13 to P2-15) and routes
`Bus` messages, including repairs (P2-17).

The P2-11 correction supplies the graph adapter and compatibility spike;
the separate P2-12 correction adds the controller execution below. Neither
adds clinical role handlers or later repair classification. The existing
`Toolbox` API is unchanged for those dependent PRs.

## Fixed Controller (P2-12)

[`agent/controller.py`](../agent/controller.py) defines `FixedController`,
a `BaseAgent` whose `_run()` executes the fixed LangGraph pipeline without
a controller model:

1. It parses the shared context (question, patient FHIR ID, assumed current
   time, and the dataset's assumption text) and sends it with every task.
2. It runs retrieval → analysis → verification. Each role's result goes to
   the next role over the `Bus`.
3. If verification reports issues, it runs **at most one repair**. The
   repair starts from the role chosen by repair routing (below):
   `retrieval` reruns retrieval and then analysis, and `analysis` reruns
   analysis only. The repair task includes the issues and the previous
   output. The controller then verifies again.
4. An answer is released only after a clean verification. Otherwise the
   system abstains, and the limitation lists the remaining issues.

Retrieval, analysis, verification, and repair classification are graph
nodes. Conditional edges stop on a role failure or route exactly one
repair, then terminate after its verification. Framework retries and
checkpoint resume are disabled, so no hidden second attempt can bypass
the common token/tool ceiling. Revision tracking uses an explicit flag
after a successful repaired analysis, not object identity in graph state.
The repair node calls P2-17's type-based `route(issues, draft)` (below),
sends its decisions as a `routing` bus message, and saves them in
`revision.routing`.

Roles are handlers (`task -> result`, both JSON objects) supplied by
`build_roles()`. They are expected to follow these message contracts:

| Role | Result |
|---|---|
| retrieval | any object, such as record IDs (records stay in the workspace) |
| analysis | `{"answer", "evidence": [...], "limitation"}` |
| verification | `{"issues": [{"type", "record", "field", "responsible_role", "action"}]}` |

If a handler returns `{"error": ...}`, raises an exception, or breaks its
contract, the run stops with that error. A budget stop becomes an
abstention, as it does for the single agent. The `revision` field records
the draft, both checks, the repair role, the final candidate, and the full
message log; the runner already saves this field. P2-13 to P2-15 provide the
LLM roles. The controller is not yet a `run_agent.py` strategy.

### Repair Routing (P2-17)

`classify()` assigns each issue a repair role by its **type**, so the
verifier model's `responsible_role` only matters for types outside the
fixed vocabulary:

| Issue | Role | Why |
|---|---|---|
| `missing_evidence` | retrieval | the records the answer needs were never retrieved |
| `recompute_mismatch` on a draft that abstained | retrieval | the verifier found an answer that analysis couldn't, so records are likely missing |
| field-check types (`value_mismatch`, `bad_path`, `missing_resource`, `malformed`, `time_field`, `unit`, `encounter_link`, `encounter_scope`, `current_time`, `time_window`, `patient_scope`), `no_evidence`, `recompute_mismatch`, `wrong_field`, `wrong_filter`, `wrong_calculation` | analysis | the records were there but were cited, filtered, or computed wrongly |
| `source_check` | none | the evidence could not be checked at all |
| any other type citing a record that was never retrieved | retrieval | missing evidence |
| any other type | the verifier's `responsible_role`, else none | |

`route()` picks the earliest role only when every issue is repairable. A
retrieval repair reruns analysis too, so repairable mixed cases start from
retrieval. If any issue is unrepairable, the system abstains without a repair. Each
decision (type, record, role, reason) goes to the `revision.routing` field
and to a `routing` bus message. After the repair the controller verifies
again and releases the answer only if that check is clean; otherwise it
releases a limitation. The verifier's prompt asks it to use
`missing_evidence`, `wrong_filter`, `wrong_calculation`, and `wrong_field`
so that its own issues route deterministically.

## Retrieval Agent (P2-13)

[`agent/retrieval_agent.py`](../agent/retrieval_agent.py) defines
`RetrievalAgent`, the controller's retrieval handler:

- **Same starting point as the single agent.** It loads the deterministic
  starting packet into the workspace it shares with the analysis role and
  shows the packet text to the model.
- **Resource-type selection.** The model chooses what to fetch, using only
  the two read-only searches (`get_resources_by_patient_fhir_id`,
  `get_resources_by_resource_id`). It cannot run Python or send raw FHIR
  queries.
- **Scope resolution.** The `Toolbox` limits records to the question's
  patient and current time. Missing patient/time provenance is excluded
  with an explicit limitation, not treated as an established empty result.
  Prescription start, rather than order time, governs the cutoff. Shared
  Medication and Location records must be linked from retrieved evidence.
  Encounter fetches expose dates and visit identifiers so the model can
  resolve scope even when the packet omitted an encounter. The model reports which encounters the
  question is about. IDs that aren't among the retrieved encounters are
  dropped and noted in the limitation.
- **Result.** Record IDs by type, the relevant resource types, the scope
  (patient, current time, encounter IDs), and a limitation. The records
  themselves stay in the shared workspace, so the bus log stays small.
- **Evidence preservation.** Both searches merge records into the
  workspace. Fetched IDs replace earlier copies with original source
  records; records outside the fetched result are retained.
- **Repair.** On a repair task the agent keeps the workspace, sees the
  verifier's issues and what it already holds, and doesn't reload the
  packet.

## Analysis Agent (P2-14)

[`agent/analysis_agent.py`](../agent/analysis_agent.py) defines
`AnalysisAgent`, the controller's analysis handler:

- **Compute in Python only.** Its one tool is `execute_python_code`, which
  runs over the workspace shared with retrieval. It cannot fetch records; if
  evidence is missing, it abstains and says what is missing, so repair
  routing can send the issue to retrieval.
- **What the model sees.** The question and assumptions, the retrieval
  report (relevant types, scope, limitation), and for each resource type a
  record count and one pruned sample record. The full records stay in the
  sandbox, not the prompt.
- **Draft.** It extends the common answer object as follows:

  ```json
  {"answer": ..., "evidence": [{"resource_id", "field_path", "value"}],
   "operation": "how the answer was computed",
   "contributing_ids": ["Observation/o1", ...], "limitation": null}
  ```

  `contributing_ids` explicitly lists every record the computation used,
  including all cited records. Typed IDs must belong to the analysis
  workspace; citation IDs alone cannot substitute for aggregate inputs.
  Non-abstaining answers require a non-empty operation, and every draft
  value must be finite JSON. Zero/empty results can have an explicit empty
  contributor list. A reply without a well-formed draft is an error, as for
  the direct baseline. Source lookups do not require an extra Python call.
- **Repair.** A repair task shows the previous draft, the verifier's issues,
  and the current record counts, which include any records retrieval
  re-fetched.

## Verification Agent (P2-15)

[`agent/verification_agent.py`](../agent/verification_agent.py) defines
`VerificationAgent`, the controller's verification handler. It never sees
the other roles' conversations or workspace, and its own workspace is
cleared before every check.

1. **Re-fetch cited fields.** `check_fields` re-reads every citation from
   the store. It checks the value, the time field, the unit, the patient,
   the encounter scope, and the current time. As for the single agent, this
   counts as one tool call. If the tool-call budget is used up, the result
   is an unrepairable `source_check` issue, so the system abstains.
2. **Recompute.** Acceptance requires a successful independent source fetch
   followed by a successful sandbox calculation, not just the model's
   assertion. A new retrieval or calculation attempt invalidates the
   previous calculation even if the later attempt fails. The reported
   answer must match the actual Python result, including an explicitly
   assigned null result. The model reports issues the field check can't
   see, such as missing records (retrieval) or a wrong calculation (analysis).
3. **Merge.** The field-check issues, the model's issues, and a
   `recompute_mismatch` issue are combined. The mismatch is added when the
   recomputed answer differs from the draft after normalization: numbers
   are rounded, text is case-folded, lists are unordered, and booleans never
   equal numbers. Timestamps use the primary scorer's wall-clock parsing
   and the question's requested day, minute, or second precision.

Every issue is `{type, record, field, responsible_role, action}`.
`responsible_role` is `retrieval`, `analysis`, or `null` (no role can
repair it). Failed source checks, malformed verdicts, failed independent
calculations, unresolved source scope, and budget-refused checks produce
unrepairable issues while preserving earlier deterministic findings. A model
duplicate cannot reroute a deterministic issue to another role.

Zero, false, null, or an empty set may legitimately have no citations only
when successful independent retrieval and computation establish that result.
Unknown patient/time exclusions or a refused later tool do not establish an
empty search. Per-check computation and source-audit state are reset before
each verification, so a prior successful calculation cannot certify a later
failed check. Completing all requested checks exactly at the ceiling is valid.

## Shared Field Check (P2-16)

`tools/field_check.py` is the deterministic check shared by the single agent
(its `check_fields` tool and automatic final check) and the verifier. Phase
1 already checked the time field (`issued` vs `effectiveDateTime`,
`authoredOn` vs the prescription start), units, the encounter link, the
patient, and the current time.

Phase 2 adds a **time-window** check. Of the 370 frozen questions, the
encounter-scope check already recognizes 88 of the 89 that name a visit,
and none ask for a unit with "in <unit>". About 170, though, restrict
events to a time window, and nothing checked it. `question_time_window()`
parses every window phrasing in the dataset and follows its SQL semantics:

| Phrasing | Window |
|---|---|
| `since 05/2137`, `since 08/11/2146`, `since 2157` | from that month, day, or year, to the current time |
| `since N days/months/years ago` | from midnight of the current date minus N, with SQLite month arithmetic (12-31 minus 1 month is 12-01) |
| `in 11/2136`, `in 2142`, `on 12/06/2183` | that month, year, or day |
| `[MM/[DD/]]this year`, `… last year` | that calendar year, month, or day |
| `this month[/DD]`, `last month[/DD]` | that calendar month or day |
| `today`, `yesterday` | that day |

A primary event field outside the window gets a `time_window` warning,
which the verifier turns into an analysis issue. Supporting encounter
metadata is not rejected simply because admission precedes an in-window
event. The check normalizes wall-clock timestamps, and Conditions are dated
by their encounter. A prescription with no start gets an unknown-event-time
warning: `authoredOn` does not establish when treatment began.

Encounter scope independently distinguishes the earliest begun visit, the
latest completed visit by the reference time, and an ongoing current visit.
Future cited timestamps are checked even when the event began earlier.

**Validation.** The original PR reported a non-held-out gold-citation audit
(618 questions and 14,123 citations) under its earlier prescription exemption.
That historical result is not validation of this correction. The corrected
194-test suite uses synthetic scope, window, missing-date, and timestamp
regressions with mocked sources; no held-out model run was performed.

## Missing-Evidence Perturbations (P2-18)

`scripts/build_perturbations.py` writes `manifests/perturbations.json`: 50
held-out questions, each with one answer-changing resource left out of its
starting packet. The chosen packet is strictly reduced by one resource;
ranking/truncation are not rerun and no replacement evidence fills the gap.
Headers are adjusted without identifying the omitted record.
**The FHIR store is unchanged**, so a system can recover the
evidence by fetching it. At run time,
`tools.perturbation.get_perturbed_starting_packet()` verifies the original
live packet and returns its frozen omission. `load_perturbations()` validates
the manifests, code hashes, packet version, patient/time scope and exact
one-record resource-ID difference, retaining each entry's packet hashes.
Wiring both
into `run_agent.py` is part of run orchestration (P3-19).
**Eligibility.** A question is eligible when it has a non-empty reference
answer, and the released SQL query reproduces it from the **exact rendered
packet fields**. `scripts/packet_query.py` projects those fields into a
read-only in-memory SQLite database. The same query over the one-resource
omission must return a different result. SQL supplies the operation and
label/time/encounter filters; an aggregate is never inferred by matching
gold values. The omitted result must also be incorrect under the existing
typed scorer, so insignificant changes within its fixed tolerance are not
counted as answer changes. Unrepresented fields and unsupported relations
fail closed. Normalized FHIR priority cannot recover all MIMIC admission
types, and collapsed microbiology resources cannot prove raw row counts.
Antibiotic susceptibility labels are not organism names; missing parent
tests or hospital membership are unknown, not SQL-null evidence.

Candidates are from the query's primary relation. Gold IDs prioritize
candidates but are not completeness or answer-change proofs. A candidate
with incoming clinical references is excluded if hiding its ID would also
erase another record's linking or ordering information. Other clinical
records and metadata remain unchanged; retained records/metadata must not
literally supply a removed selection value. Shared Patient, Medication and
Location resources are not omission targets.

Multi-value answers are excluded. The 50 questions are sampled with the
split seed, in proportion to each family's eligible count. Synthetic
development regressions cover redundant non-gold events, tied extrema,
actual query operations, date/encounter filters, strict omissions and
runtime drift. SQL, templates and labels are confined to this offline
construction harness; agents see only the perturbed starting packet.
On held-out, the script prints aggregate counts, never answer values.
The old annotation-based manifest is superseded by the version 2 manifest
over packet version 3. Fixed rules were validated on all 70 pilot/dev
questions (15 eligible) before the held-out build, with no subsequent rule
changes. Of 300 held-out questions, 55 were eligible; 50 were sampled across
27 patients: 15 lookup/counting, 15 numerical aggregation, and 20 temporal
selection questions. The selected operations are max (2), min (5), sum (8),
count (4), existence (11), and selection (20).
Version 2 records hashes of dataset, split, families, packets, builder,
oracle, omitter and scorer, plus before/after query-result hashes for every
entry. Independent regeneration produced a byte-identical manifest.

All 50 selected cases can be rebuilt from local public source data without
model/FHIR-store calls:

```sh
python scripts/build_perturbations.py --fhir-dir "$FHIR_NDJSON_DIR" \
  --verify-manifest manifests/perturbations.json
FHIR_NDJSON_DIR="$FHIR_NDJSON_DIR" python -m unittest discover -s tests \
  -p 'test_phase2_perturbations*.py'
```

Verification covers the source freeze, original and omitted packet hashes,
the actual visible SQL results, typed incorrectness after omission, exact
one-resource removal, and unchanged remaining clinical fields. Only counts
are reported. Rule validation uses pilot/dev before the held-out build;
held-out labels are not used to revise the fixed eligibility rules.
All 50 source rebuilds passed, and all 32 perturbation tests passed with
`FHIR_NDJSON_DIR` set, including the all-case source/runtime verification;
none were skipped.

## Spike Results

`tests/test_phase2_framework.py` runs a two-role retrieval → analysis
LangGraph pipeline through `BaseAgent.run()` with a mocked model and FHIR client. It
uses the real registry tools and the real Python sandbox. It verifies that:

- records fetched by the retrieval role reach the analysis role's sandboxed
  code, and the output contract lists both roles' tool calls;
- a tool-call ceiling of 1 is shared across roles, and the second role's call
  is refused with the standard budget reply;
- tools outside a role's allow list, undeclared model arguments, and
  retrieval for another patient are refused without running;
- system-bound arguments override the model's;
- each role run starts from a fresh context, and role errors and turn limits
  are reported;
- bus payloads are copied on input and output and must be JSON;
- the actual graph has no checkpointer or retry policy, a failed node is
  invoked once, and tracing stays disabled even with tracing environment flags.

Run with:

```sh
python -m unittest discover -s tests -p 'test_phase*.py' -v
```

No live model or FHIR-store run was part of this spike. Phase 1 agents are
unchanged. `MultiTurnCodeResourceAgent` still has its own tool dispatch, so
the baselines keep their Phase 1 behavior. Adding a file under `agent/`
changes `run_agent.py`'s code fingerprint, so result files from before this
change cannot be resumed and must be rerun under a new output path.

Shared token accounting also normalizes real SDK message objects through their
structured serialization APIs before counting subsequent prompts. This fixes
bounded multi-turn calls without introducing a personal model configuration.

The scoped P2-11 correction passed all 136 available Phase 1 and framework
tests with mocked model/store calls and real macOS sandbox execution. A
Linux sandbox backend still requires its own platform verification.
The separate P2-12 correction, including that prerequisite, passed all 148
available tests. Its regressions cover actual graph nodes, early failures,
one bounded repair, abstention after failed verification, and the original
scalar repair-route API.
