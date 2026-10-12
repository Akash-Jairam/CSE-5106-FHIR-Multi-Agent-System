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
