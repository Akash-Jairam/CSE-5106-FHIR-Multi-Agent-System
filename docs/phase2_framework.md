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
   repair starts from the earliest responsible role: `retrieval` reruns
   retrieval and then analysis, and `analysis` reruns analysis only. The
   repair task includes the issues and the previous output. The controller
   then verifies again.
4. An answer is released only after a clean verification. Otherwise the
   system abstains, and the limitation lists the remaining issues.

Retrieval, analysis, verification, and repair classification are graph
nodes. Conditional edges stop on a role failure or route exactly one
repair, then terminate after its verification. Framework retries and
checkpoint resume are disabled, so no hidden second attempt can bypass
the common token/tool ceiling. Revision tracking uses an explicit flag
after a successful repaired analysis, not object identity in graph state.
The original `route(issues) -> role or None` interface is retained here;
P2-17's later error classification and routing diagnostics are not part of
this controller correction.

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

The scoped P2-11 correction passed all 134 available Phase 1 and framework
tests with mocked model/store calls and real macOS sandbox execution. A
Linux sandbox backend still requires its own platform verification.
The separate P2-12 correction, including that prerequisite, passed all 148
available tests. Its regressions cover actual graph nodes, early failures,
one bounded repair, abstention after failed verification, and the original
scalar repair-route API.
