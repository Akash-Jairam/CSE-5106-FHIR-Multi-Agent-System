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
   repair starts from the earliest responsible role: `retrieval` reruns
   retrieval and then analysis, and `analysis` reruns analysis only. The
   repair task includes the issues and the previous output. The controller
   then verifies again.
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
LLM roles, and P2-17 refines `route()`. The controller is not yet a
`run_agent.py` strategy because those roles don't exist yet.

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
