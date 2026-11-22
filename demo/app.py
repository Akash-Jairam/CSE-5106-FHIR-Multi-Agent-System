"""Demo interface (P3-23): ask one patient's FHIR record a question through the evaluated pipeline.

    streamlit run demo/app.py -- --max_total_tokens 200000 --max_tool_calls 30

Use the same settings as scripts/run_experiments.py. The header then shows
whether the code, model, and budget match the evaluated run. Example
questions come from the pilot and dev patients (`--splits` to change).
Reference answers are never shown.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo import backend  # noqa: E402
from utils import get_default_base_url, get_default_model, validate_config  # noqa: E402

BADGES = {"verified": (st.success, "✓"), "unverified": (st.warning, "⚠"), "unchecked": (st.info, "○"),
          "abstained": (st.info, "–"), "error": (st.error, "✕")}


def settings_from_argv(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default=get_default_model())
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_total_tokens", type=int, default=None)
    parser.add_argument("--max_tool_calls", type=int, default=None)
    parser.add_argument("--splits", nargs="+", choices=["pilot", "dev", "held_out"], default=list(backend.DEFAULT_SPLITS))
    parser.add_argument("--output_dir", default=None, help="Evaluated results; default as in run_experiments.py")
    parser.add_argument("--input", default=os.path.join(backend.ROOT, "final_dataset", "questions_answers_sql_fhir.csv"))
    parser.add_argument("--split_manifest", default=os.path.join(backend.ROOT, "manifests", "split.json"))
    args, _ = parser.parse_known_args(argv)
    args.base_url = args.base_url or get_default_base_url(args.model)
    for key in ("max_total_tokens", "max_tool_calls"):
        if getattr(args, key) is not None and getattr(args, key) < 0:
            parser.error(f"--{key} must be non-negative")
    args.output_dir = args.output_dir or os.path.join(backend.ROOT, "output", "experiments",
                                                      args.model.replace("/", "_"), "held_out")
    return args


@st.cache_data
def patients(input_file, split_manifest, splits, input_hash, split_hash):
    return backend.load_patients(input_file, split_manifest, splits)


def show_answer(value, empty_result=False):
    if value is None:
        return "No matching value" if empty_result else "No answer"
    return value if isinstance(value, str) else json.dumps(value)


args = settings_from_argv(sys.argv[1:])
settings = {key: getattr(args, key) for key in ("model", "base_url", "max_total_tokens", "max_tool_calls")}
st.set_page_config(page_title="FHIR record Q&A demo", layout="wide")
st.title("Ask a patient's FHIR record")

with st.sidebar:
    system = st.radio("System", list(backend.SYSTEMS), format_func=backend.SYSTEMS.get)
    pool = {p["patient_fhir_id"]: p for p in patients(args.input, args.split_manifest, tuple(args.splits),
                                                    backend._file_hash(args.input), backend._file_hash(args.split_manifest))}
    if not pool:
        st.warning("No patients have example questions in these splits.")
        st.stop()
    patient = pool[st.selectbox("Patient", list(pool), format_func=lambda i: f"{pool[i]['subject_id']} ({pool[i]['split']})")]
    st.caption(f"{len(patient['questions'])} example questions · FHIR ID {patient['patient_fhir_id']}")

budget = ", ".join(f"{k.removeprefix('max_').replace('_', ' ')} ≤ {v:,}" for k, v in settings.items()
                   if k.startswith("max_") and v is not None) or "no budget ceilings"
evaluated = backend.evaluated_run(system, settings, args.output_dir, args.input, args.split_manifest)
if not evaluated["found"]:
    match = "no evaluated run found for these settings"
elif evaluated["all_match"]:
    match = "✓ same code, model, and budget as the evaluated run"
elif not evaluated["valid"]:
    match = evaluated.get("reason", "no complete evaluated result batch found")
else:
    differing = [key.replace("_hash", "") for key, ok in evaluated["matches"].items() if not ok]
    match = "⚠ differs from the evaluated run: " + ", ".join(differing)
st.caption(f"{backend.SYSTEMS[system]} · {args.model} · {budget} · tool cache on · {match}")

mode = st.radio("Question", ["Example", "Custom"], horizontal=True, label_visibility="collapsed")
if mode == "Example":
    examples = {q["question_id"]: q for q in patient["questions"]}
    example = examples[st.selectbox(f"Example question for patient {patient['subject_id']}", list(examples),
                                    format_func=lambda i: examples[i]["question"])]
    question, current_time = example["question"], example["current_time"]
    st.caption(f"Assumed current time: {current_time or 'not stated'}")
else:
    times = sorted(q["current_time"] for q in patient["questions"] if q["current_time"])
    question = st.text_area("Your question", placeholder=f"What was the last weight of patient {patient['subject_id']}?")
    current_time = st.text_input("Assumed current time (optional)", value=times[-1] if times else "")
asked = st.button("Ask", type="primary")
st.divider()
request_context = (system, patient["patient_fhir_id"], mode, question, current_time or None, tuple(settings.items()))
if st.session_state.get("result", {}).get("context") != request_context:
    st.session_state.pop("result", None)

if asked:
    st.session_state.pop("result", None)
    try:
        input_str = backend.build_input(patient["patient_fhir_id"], question, current_time or None)
    except ValueError as exc:
        st.error(str(exc))
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            configured = validate_config(args.model, args.base_url)
        if not configured:
            st.error("The model API key or the FHIR store is not configured. See the README's setup section.")
        else:
            with st.spinner(f"Running the {backend.SYSTEMS[system].lower()}…"):
                output = backend.ask(system, input_str, settings)
            st.session_state["result"] = {"system": system, "question": question, "patient": patient["subject_id"],
                                          "current_time": current_time, "output": output, "context": request_context}

result = st.session_state.get("result")
if result:
    output, status = result["output"], backend.verification_status(result["system"], result["output"])
    st.subheader(result["question"])
    st.caption(f"Patient {result['patient']} · {backend.SYSTEMS[result['system']]}"
               + (f" · current time {result['current_time']}" if result["current_time"] else ""))
    show, icon = BADGES[status["state"]]
    show(f"{icon} **{status['label']}** — {status['detail']}")

    left, right = st.columns([2, 3])
    with left:
        st.metric("Answer", show_answer(output.get("agent_answer"), status.get("empty_result", False)))
        if output.get("limitation"):
            st.caption(f"Limitation: {output['limitation']}")
    with right:
        st.markdown("**Cited fields**")
        rows = backend.cited_fields(output)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.caption("No fields cited.")
    if status["issues"]:
        st.markdown("**Remaining issues**")
        st.dataframe(status["issues"], hide_index=True, use_container_width=True)

    revision = output.get("revision") if isinstance(output.get("revision"), dict) else None
    if revision:
        with st.expander("Draft and repair"):
            draft = revision.get("draft") or {}
            st.write(f"Draft answer: {show_answer(draft.get('answer'))}")
            st.write(f"Revised: {'yes' if revision.get('revised') else 'no'}"
                     + (f" (repair started from {revision['repair_role']})" if revision.get("repair_role") else ""))
            if revision.get("check_issues"):
                st.write("First check found:")
                st.dataframe([backend.describe_issue(i) for i in revision["check_issues"]], hide_index=True,
                             use_container_width=True)
    with st.expander("Usage"):
        tokens, run_budget = output.get("tokens") or {}, output.get("budget") or {}
        cost = tokens.get("cost")
        total = "unavailable" if tokens.get("total") is None else f"{tokens['total']:,}"
        calls = "unavailable" if tokens.get("llm_calls") is None else str(tokens["llm_calls"])
        tools = "unavailable" if run_budget.get("tool_calls_run") is None else str(run_budget["tool_calls_run"])
        latency = "unavailable" if output.get("latency") is None else f"{output['latency']:.1f}s"
        st.write(f"{total} tokens · {calls} model calls · {tools} tool calls · latency {latency}"
                 + (f" · ${cost:.4f}" if cost is not None else " · cost unavailable"))
        names = [c.get("name") for c in output.get("tool_calls") or [] if isinstance(c, dict)]
        if names:
            st.caption("Tools: " + ", ".join(names))
else:
    st.info("Choose a patient and a question, then press Ask.")
