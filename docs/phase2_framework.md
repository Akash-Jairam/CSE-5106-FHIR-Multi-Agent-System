# Phase 2 Agent Framework (P2-11)

## Decision

The multi-agent system uses a thin in-house message-passing layer,
[`agent/framework.py`](../agent/framework.py), built on `BaseAgent`, LiteLLM,
and the existing tool registry. It adds no dependency.

| Option | Why not |
|---|---|
| LangGraph | Its graph and checkpointing are more than a fixed controller with at most one repair needs. Its LLM calls go through LangChain chat models, which bypass `BaseAgent._llm_call`, so the per-question token ceiling and usage records would have to be rebuilt through callbacks. |
| AutoGen / AG2 | Its conversation and LLM loop would replace ours. An LLM group chat or selector conflicts with the proposal's non-LLM controller. Its code executors run outside `tools/python_executor.py`'s sandbox. |
| CrewAI | It uses LLM delegation and its own prompts and tool wrappers. That makes routing nondeterministic and adds a second set of prompts to control. |
| OpenAI Agents SDK | It is built around one provider's Responses API and its own tool loop. The project runs several providers through LiteLLM. |

The deciding constraints come from Phase 1:

- **One budget per question.** Phase 1 budget controls (#13) cap
  completions, count automatic checks, and refuse unaccounted usage. They
  only work if every model call goes through `BaseAgent._llm_call`. A
  framework that owns the LLM loop would need these rules reimplemented
  and checked again.
- **Sandboxed code only.** Generated Python must run in
  `tools/python_executor.py` (OS sandbox, copied data, real deadline).
  None of the frameworks' built-in executors meet that bar.
- **A deterministic controller.** P2-12 requires a non-LLM controller, so
  a framework's main feature, LLM-driven routing, is unused.
- **The common contract and resume identity.** The system must still
  return `AgentOutput` through `BaseAgent.run()`, and `run_agent.py`
  fingerprints `agent/`, `tools/`, and `utils/` for resume. Pinned
  in-repo code keeps that fingerprint meaningful.

## The Layer

- **`Bus`** logs structured `Message(sender, recipient, kind, payload)`
  values. Payloads must be finite JSON objects and are copied when sent, so
  roles never share mutable objects. The log becomes the coordination
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

## Fixed Controller (P2-12)

[`agent/controller.py`](../agent/controller.py) defines `FixedController`,
a `BaseAgent` whose `_run()` decides the order of work without a model:

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

`route()` picks the earliest role among the issues. A retrieval repair
reruns analysis too, so mixed or ambiguous cases start from retrieval. If
no issue is repairable, the system abstains without a repair. Each
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
  patient and current time. The model reports which encounters the
  question is about. IDs that aren't among the retrieved encounters are
  dropped and noted in the limitation.
- **Result.** Record IDs by type, the relevant resource types, the scope
  (patient, current time, encounter IDs), and a limitation. The records
  themselves stay in the shared workspace, so the bus log stays small.
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

  `contributing_ids` lists every record the computation used. It defaults to
  the cited records. A reply without a well-formed draft is an error, as for
  the direct baseline.
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
2. **Recompute.** The model re-searches the patient's records (read-only,
   within scope) and recomputes the answer in the sandbox. It reports
   issues the field check can't see, such as records the draft missed
   (retrieval) or a wrong filter or calculation (analysis).
3. **Merge.** The field-check issues, the model's issues, and a
   `recompute_mismatch` issue are combined. The mismatch is added when the
   recomputed answer differs from the draft after normalization: numbers
   are rounded, text is case-folded, timestamps accept either `T` or a
   space, lists are unordered, and booleans never equal numbers.

Every issue is `{type, record, field, responsible_role, action}`.
`responsible_role` is `retrieval`, `analysis`, or `null` (no role can
repair it). A reply without a JSON verdict, or with malformed issues, is an
error, so a draft is never released unchecked.

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

A cited resource dated outside the window gets a `time_window` warning,
which the verifier turns into an analysis issue. The check uses
wall-clock time, and Conditions are dated by their encounter. A
prescription without a start time isn't checked, because its order time
(`authoredOn`) is not what the window filters.

**Validation.** No held-out patient was used. On the 618 eligible
questions from non-held-out patients that have a window, all 14,123 gold
evidence citations pass the check. Before the prescription rule above was
added, the only one flagged was a prescription with only an order time.

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
gold values. Unrepresented fields and unsupported relations fail closed.

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
The old 83-eligible/50-selected counts used annotation-based rules and are
superseded; regenerate the manifest after the corrected packet freeze.
Version 2 records hashes of dataset, split, families, packets, builder,
oracle and omitter, plus before/after query-result hashes for every entry.

## Spike Results

`tests/test_phase2_framework.py` runs a two-role retrieval → analysis
pipeline through `BaseAgent.run()` with a mocked model and FHIR client. It
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
- bus payloads are copied and must be JSON.

Run with:

```sh
python -m unittest discover -s tests -p 'test_phase*.py' -v
```

No live model or FHIR-store run was part of this spike. Phase 1 agents are
unchanged. `MultiTurnCodeResourceAgent` still has its own tool dispatch, so
the baselines keep their Phase 1 behavior. Adding a file under `agent/`
changes `run_agent.py`'s code fingerprint, so result files from before this
change cannot be resumed and must be rerun under a new output path.
