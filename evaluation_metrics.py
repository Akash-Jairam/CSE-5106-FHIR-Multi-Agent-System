import os
import sys
import ast
import json
import re
import multiprocessing as mp
from functools import partial
import yaml
import argparse
import datetime
import csv
from collections import Counter
from utils import is_reasoning_llm

import pandas as pd
from tqdm import tqdm
import numpy as np
import litellm


from utils.metrics_utils import (
    retrieval_recall,
    retrieval_precision
)
from utils.answer_scorer import parse_reference, score_answer
from utils.evidence_scorer import cited_refs, fetch_from_ndjson, fetch_from_store, parse_evidence, score_evidence
from config import (
    QUESTION_SQL_FHIR_OUTPUT_CSV
)

# Argparser
parser = argparse.ArgumentParser(description="Evaluate FHIR agent answers.")
parser.add_argument("--input",
                    type=str,
                    required=True,
                    help="Path to the input CSV or JSON file with agent results.")
parser.add_argument("--output_dir",
                    type=str,
                    default='output_eval',
                    required=False,
                    help="Path to the output directory.")
parser.add_argument("--model",
                    type=str,
                    default="o4-mini",
                    required=False,
                    help="Model to use for evaluation.")
parser.add_argument("--num_processes",
                    type=int,
                    default=-1,
                    help="Number of processes to use for parallel processing.")
parser.add_argument("--force",
                    action="store_true",
                    help="Force re-evaluation even if results already exist.")
parser.add_argument("--evidence_source",
                    choices=["store", "ndjson"],
                    default="store",
                    help="Where to resolve cited evidence: the FHIR store (config.yml) or the local NDJSON files.")
parser.add_argument("--fhir_dir",
                    default="physionet.org/files/mimic-iv-fhir-demo/2.1.0/fhir",
                    help="MIMIC-IV FHIR NDJSON directory, used with --evidence_source ndjson.")
parser.add_argument("--skip_llm_judge",
                    action="store_true",
                    help="Only run the deterministic typed scorer (no LLM calls or API key needed).")
args = parser.parse_args()


# Setup LiteLLM
from utils.core_utils import setup_api_keys

def setup_environment():
    """Setup environment and configuration"""
    if args.num_processes == -1:
        args.num_processes = os.cpu_count()
    if args.skip_llm_judge:
        return
    if not setup_api_keys():
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to config.yml")
    
    # Configure LiteLLM
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

# ============================================================================
# LLM and Processing Functions
# ============================================================================

def call_model(model, prompt):
    """Call LLM with appropriate settings"""
    try:
        model_mapping = {
            "o4-mini": "o4-mini"
        }
        
        litellm_model = model_mapping.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        
        if is_reasoning_llm(model):
            response = litellm.completion(model=litellm_model, messages=messages)
        else:
            response = litellm.completion(model=litellm_model, messages=messages, temperature=0.0)
            
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Failed to call model {model}: {str(e)}")


def process_in_parallel(data_tuples, worker_func, desc="Processing", num_processes=1):
    """Process data in parallel or sequentially"""
    if num_processes == 1:
        results = []
        for data_tuple in tqdm(data_tuples, desc=f"{desc} (single-core)"):
            results.append(worker_func(data_tuple))
        return results
    else:
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(worker_func, data_tuples),
                total=len(data_tuples),
                desc=f"{desc} (multi-core)"
            ))
        return results


# Worker functions for multiprocessing
def worker_normalize_answer(data_tuple):
    idx, agent_answer, question, model = data_tuple
    try:
        result = normalize_null_answers_with_llm(agent_answer, question, model)
        return idx, result
    except Exception as e:
        print(f"Error processing during normalize_answer: row {idx}: {e}")
        return idx, "question answered"


def worker_check_correctness(data_tuple):
    idx, agent_answer, true_answer, question, model, agent_answer_normalized, error = data_tuple
    null_answer_check = re.sub(r'[^\w\s]', '', agent_answer_normalized).strip().lower()
    try:
        error_mask = (
            pd.isnull(error) or
            (error == "") or
            (error == "nan") or
            (error.strip() == "")
        )
        # Error cases - wrong
        if not error_mask:
            return idx, 0
        # Token exceeded - wrong
        if pd.notnull(agent_answer) and 'Input tokens exceeded' in agent_answer:
            return idx, 0
        # If no answer outputted by both true answer and null answer check, correct
        if null_answer_check == true_answer == "no answer":
            return idx, 1
        # Otherwise if agent answer was none or null, wrong
        if pd.isnull(agent_answer):
            return idx, 0
        # Final check for all other answer types
        result = check_answer_correctness_with_llm(agent_answer, true_answer, question, model)
        return idx, result
    except Exception as e:
        print(f"Error processing during check_correctness: row {idx}: {e}")
        return idx, 0

# ============================================================================
# LLM Evaluation Functions
# ============================================================================

def check_answer_correctness_with_llm(answer: str, ref_answer: str, question: str, model: str = None) -> int:
    prompt = f"""You are a helpful assistant that evaluates whether a model answer to a question is correct, by comparing it to the true answer.

Your task:
- Return 1 if the model answer is correct.
- Return 0 if the model answer is incorrect.
- Never return anything other than 0 or 1.

The model answer may be more verbose or formatted differently from the true answer. Focus on correctness of content, not exact formatting.

---

### Core Rules:

1. Null or no-answer cases  
   - If the true answer is `[]` or `'null'` or 'no answer', this means there is no answer or the answer is effectively 0.  
   - If the model gives any conflicting or non-empty answer, return 0.  
   - If the model also implies no answer, return 1.
   - If the model also implies no answer, or returns 0 for numerical questions, return 1.

2. Yes/No answers 
   - True answers may appear as `[[0]]` (No) or `[[1]]` (Yes), with flexible formatting (e.g., `[0]`, `'[[0]]'` are equivalent).
   - Evaluate based on meaning, not syntax.
   - Ignore differences in variable names or context if the Yes/No meaning aligns.

3. Numerical answers
   - Match on value, rounding both sides to the nearest integer, except for days (which should be rounded to the nearest whole day).
   - Be lenient if the model answer has the true answer in the breakdown but returns a different total aggregated value.
   - Ignore decimal formatting (`1.0` = `1.` = `1`).
   - Ignore units (`1850` = `1850 mL`).

4. Date answers
    - Match on dates, you can ignore time and timezone differences unless specifically stated in the question.
    - A time difference of up to a minute is acceptable, as the values may be rounded.

5. List answers
   - If the true answer is a list (e.g., `['or ebl', 'or urine']`), the model must include all listed values and no extra medical values.  
   - Ignore harmless extra context (e.g., time references, phrasing).

6. Detail and verbosity
   - Extra correct details in the model answer are fine if they align with the true answer.

7. Formatting leniency
   - Be lenient with brackets, quotes, spacing, and style.
   - As long as the model’s answer semantically matches the true answer, return 1.

---

### Examples:

EXAMPLE 1:
- question : Please tell me the sex of patient C  
- true answer : [['F']]  
- model answer : The patient’s sex is female.  
- reasoning : The true answer indicates "female" (F). The model states the patient’s sex is female, which directly matches.  
You will return : 1

---

EXAMPLE 2:  
- question : Compared to last measured on the first icu visit, is the glucose measurement value of patient B less than second to last measured on the first icu visit?  
- true answer : '
- model answer : The patient’s last glucose measurement on the current hospital visit (87 mg/dL) was greater than the first measurement (86 mg/dL).  
- reasoning : The true answer is No (`[[0]]`). The model states the value is greater, which also implies it is not less. This aligns with the true answer meaning.  
You will return : 1

---

EXAMPLE 3:
- question : Are patient A's bicarbonate levels second measured on the first hospital visit less than the levels first measured on the first hospital visit?  
- true answer : [[0]]
- model answer: The second bicarbonate measurement (26 mEq/L at 2161-04-28 02:15) is not less than the first measurement (23 mEq/L at 2161-04-27 12:09).  
- reasoning : The true answer is No (`[[0]]`). The model explicitly says the second value is not less than the first, which matches the true answer.  
You will return : 1

---

EXAMPLE 4:
- question : "Hey, what was patient 10019172's minimum ['fibrinogen, functional'] value in 11/this year?"  
- true answer : [[201.]]  
- model answer: The minimum fibrinogen functional value for patient 10019172 in November of this year is 201.  
- reasoning : The true answer is 201. The model provides the same value (ignoring decimals). They match.  
You will return : 1

---

EXAMPLE 5:  
- question : Show me the total amount of patient's output on this month/02.  
- true answer : [[1850]] 
- model answer: The total patient output on 2110-12-02 was 1850 mL.  
- reasoning : The true answer is 1850. The model answer gives the same value and adds "mL", which is acceptable because units are disregarded.  
You will return : 1

---

EXAMPLE 6:
- question : What was the change in ['platelet count'] in patient 10029484 second measured on the first hospital visit compared to the first value measured on the first hospital visit?
- true answer : [[-38]]
- model answer : The platelet count on the first hospital visit went from 261 on the first measurement to 223 on the second—a drop of 38.
- reasoning : The true answer is -38, indicating a decrease. The model states the count dropped by 38, which matches the true answer meaning.
You will return : 1

---

EXAMPLE 7:
- question : When was patient 10004422's last maximum ['cholesterol, total'] value on their last hospital encounter?
- true answer : [['2111-01-16 07:02:00']]
- model answer : The highest total cholesterol value on the patient’s last hospital encounter was 121 mg/dL, recorded on 2111-01-16.
- reasoning : The true answer is a timestamp. The model provides the same timestamp and mentions the maximum cholesterol value, which aligns with the true answer context. Here there is additional information (the max value) but the answer is still correct. It is okay that the answer does not contain the exact time. 
You will return : 1

--

EXAMPLE 8:
- question : Tell me the total ['acetaminophen'] dose that patient 10037975 has been prescribed?
- true answer : [[1300]]
- model answer : "Total acetaminophen prescribed: 3,300 mg (3.3 g).

Breakdown:
- Acetaminophen IV: 2 prescriptions × 1,000 mg = 2,000 mg  
- Acetaminophen: 2 prescriptions × 650 mg = 1,300 mg"
- reasoning : The true answer only shows the total dose for the basic variation of Acetaminophen, whereas the model provides the total dose for all variations, breaking down the count for the basic one.
You will return : 1

--

### Final Input:
- Question: {question}
- True answer: {ref_answer}
- Model answer: {answer}

### Output:
Return only 0 or 1. Do not explain your reasoning.

"""
    eval_model = model or args.model
    while True:
        response = call_model(eval_model, prompt).strip("'")
        if response in ['0', '1']:
            return int(response)


def normalize_null_answers_with_llm(answer: str, question: str, model: str = None) -> str:
    """Normalize generated answer to reference answer"""
    prompt = f"""
You are a helpful assistant that converts free-text answers into a standardized format.

Your task: Determine if the model's answer successfully answers the question or if it indicates insufficient information.

### Instructions:
1. If the answer shows that the question CANNOT be answered (e.g., not enough data, unknown, missing information), return exactly:
    no answer

2. If the answer DOES answer the question, return exactly:
    question answered

Important:
- Return ONLY one of these two values: 'no answer' or 'question answered'.
- Do NOT include any additional text, punctuation, or explanation.

---

### Input:

Question:
{question}

Answer:
{answer}

---

### Examples:

EXAMPLE 1:
Question: Please tell me the sex of patient C  
Answer: The patient's sex is female.  
Return: question answered

---

EXAMPLE 2:
Question: Compared to last measured on the first ICU visit, is the glucose measurement value of patient B less than second to last measured on the first ICU visit?  
Answer: The patient does not have glucose measurements.
Return: no answer

"""
    eval_model = model or args.model
    return call_model(eval_model, prompt)


# ============================================================================
# Data Processing Functions
# ============================================================================

def qualified_resource_ids(resources):
    """Flatten type -> ids without dropping false-positive resource types."""
    if not isinstance(resources, dict):
        return []
    qualified = []
    for rtype, ids in resources.items():
        ids = [ids] if isinstance(ids, str) else ids
        if not isinstance(ids, (list, tuple, set)):
            continue
        qualified.extend(rid if rid.startswith(f"{rtype}/") else f"{rtype}/{rid}"
                         for rid in ids if isinstance(rid, str))
    return qualified


def extract_agent_resources(row):
    resources = row.get("agent_fhir_resources")
    if isinstance(resources, str):
        if resources.strip() in ("", "nan"):
            return []
        try:
            resources = json.loads(resources)
        except json.JSONDecodeError:
            try:
                resources = ast.literal_eval(resources)
            except (ValueError, SyntaxError):
                return []
    return qualified_resource_ids(resources)


# ============================================================================
# Main Pipeline Functions
# ============================================================================

def load_and_merge_data():
    """Load input data and merge with ground truth"""
    print("=" * 60)
    print("READING DATA...")
    print("=" * 60)
    
    # Load ground truth
    ground_truth_df = pd.read_csv(QUESTION_SQL_FHIR_OUTPUT_CSV)
    ground_truth_df["true_fhir_ids"] = ground_truth_df["true_fhir_ids"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else dict()
    )
    
    # Load input data
    if args.input.endswith('.json'):
        with open(args.input, 'r', encoding='utf-8') as f:
            input_df = pd.DataFrame(json.load(f))
        print(f"Read JSON file with {len(input_df)} records")
    else:
        input_df = pd.read_csv(args.input)
        print(f"Read CSV file with {len(input_df)} records")
    
    # Merge data
    # Always score against the dataset labels; predictions cannot replace them.
    overlaps = [col for col in ground_truth_df.columns if col != "question_id" and col in input_df.columns]
    eval_df = input_df.drop(columns=overlaps).merge(
        ground_truth_df, on="question_id", how="left", validate="many_to_one", indicator=True)
    unknown = eval_df["_merge"] != "both"
    if unknown.any():
        raise ValueError(f"{unknown.sum()} prediction rows have question IDs absent from the ground-truth dataset")
    eval_df = eval_df.drop(columns="_merge")
    
    print(f"Unique questions: {input_df['question_id'].nunique()}")
    print(f"After merge: {eval_df.shape}")
    return eval_df


def filter_evaluation_data(eval_df):
    """Filter evaluation data to remove invalid entries"""
    print("=" * 60)
    print("FILTERING DATA...")
    print("=" * 60)
    
    original_shape = eval_df.shape
    print(f"Original shape: {original_shape}")
    
    # Remove rows without answers
    # eval_df = eval_df[eval_df['agent_answer'].notnull()]
    print(f"# rows without answers: {eval_df['agent_answer'].isnull().sum()}")
    
    # Remove rows with errors (keep NaN and empty strings)
    if 'error' in eval_df.columns:
        error_mask = (eval_df["error"].isnull() | 
                     (eval_df["error"] == "") | 
                     (eval_df["error"] == "nan") | 
                     (eval_df["error"].astype(str).str.strip() == ""))
        
        print(f"# rows with errors: {len(eval_df.loc[~error_mask])}")
    
    if len(eval_df) == 0:
        print("ERROR: No data left after filtering!")
        sys.exit(1)
        
    return eval_df


def calculate_retrieval_metrics(eval_df):
    """Calculate retrieval precision and recall metrics"""
    print("=" * 60)
    print("CALCULATING RETRIEVAL METRICS...")
    print("=" * 60)
    
    # Precision includes all retrieved identities, including unrelated types.
    eval_df["metric_relevant_resources"] = eval_df.apply(extract_agent_resources, axis=1)
    eval_df["true_fhir_ids_list"] = eval_df["true_fhir_ids"].apply(
        qualified_resource_ids
    )
    
    # Add resource types column
    eval_df["true_fhir_resource_types"] = eval_df["true_fhir_ids"].apply(
        lambda d: "{" + ", ".join(sorted(d.keys())) + "}" if isinstance(d, dict) else "{}"
    )
    
    # Calculate metrics
    eval_df["recall"] = eval_df.apply(
        lambda row: retrieval_recall(row["metric_relevant_resources"], row["true_fhir_ids_list"]), axis=1)
    eval_df["precision"] = eval_df.apply(
        lambda row: retrieval_precision(row["metric_relevant_resources"], row["true_fhir_ids_list"], method="continuous"), axis=1)
    
    print(f"Retrieval Precision: {eval_df['precision'].mean():.4f}")
    print(f"Retrieval Recall:    {eval_df['recall'].mean():.4f}")

    if "evidence" in eval_df.columns:
        eval_df = calculate_evidence_metrics(eval_df)
    return eval_df


def calculate_evidence_metrics(eval_df):
    """Resolve each answer's cited {resource_id, field_path, value} and compare cited ids to the gold ids"""
    evidence = eval_df["evidence"].apply(parse_evidence)
    refs = cited_refs(evidence)
    print(f"Resolving {len(refs)} cited resources from the {args.evidence_source}...")
    if args.evidence_source == "ndjson":
        resources = fetch_from_ndjson(refs, args.fhir_dir)
    else:
        resources = fetch_from_store(refs)

    scores = pd.DataFrame([score_evidence(ev, resources, gold) for ev, gold in
                           zip(evidence, eval_df["true_fhir_ids"])], index=eval_df.index)
    for col in scores.columns:
        eval_df[col] = scores[col]

    total_cited = eval_df["evidence_cited"].sum()
    print(f"Valid evidence:      {eval_df['evidence_valid'].sum()}/{total_cited} citations; "
          f"mean per answer {eval_df['valid_evidence_fraction'].mean():.4f}")
    print(f"Evidence Precision:  {eval_df['evidence_precision'].mean():.4f}")
    print(f"Evidence Recall:     {eval_df['evidence_recall'].mean():.4f}")
    return eval_df


def has_agent_error(row):
    error = row.get("error")
    if pd.notnull(error) and str(error).strip() not in ("", "nan"):
        return True
    answer = row.get("agent_answer")
    return isinstance(answer, str) and "Input tokens exceeded" in answer


def calculate_typed_metrics(eval_df):
    """Primary metric: deterministic typed answer scoring (utils/answer_scorer.py)"""
    print("=" * 60)
    print("CALCULATING TYPED ANSWER METRICS...")
    print("=" * 60)

    def score_row(row):
        result = score_answer(row["agent_answer"], row["true_answer"], row["question"], row.get("template", ""))
        if has_agent_error(row):
            result.update(typed_correct=0, typed_reason="agent error")
        return pd.Series(result)

    typed = eval_df.apply(score_row, axis=1)
    for col in ["typed_correct", "answer_type", "typed_reason"]:
        eval_df[col] = typed[col]
    print(f"Typed accuracy: {eval_df['typed_correct'].mean():.4f}")
    return eval_df


def answer_for_judge(answer):
    """The LLM judge reads text, so structured answers (bool, number, list) are passed as JSON."""
    if answer is None or isinstance(answer, str) or (isinstance(answer, float) and pd.isna(answer)):
        return answer
    return json.dumps(answer)


def calculate_answer_metrics(eval_df, num_processes=1):
    """Secondary metric: answer correctness judged by an LLM"""
    print("=" * 60)
    print("CALCULATING ANSWER METRICS...")
    print("=" * 60)
    
    # Set up true answers
    eval_df['true_answer_extracted'] = eval_df["true_answer"]
    empty_reference = eval_df["true_answer"].apply(
        lambda answer: not any(value is not None for value in parse_reference(answer)))
    eval_df.loc[empty_reference, "true_answer_extracted"] = "no answer"

    # Normalize answers
    print("Normalizing answers...")
    normalize_data = [(idx, answer_for_judge(row["agent_answer"]), row["question"], args.model) 
                     for idx, row in eval_df.iterrows()]
    normalize_results = process_in_parallel(normalize_data, worker_normalize_answer, 
                                          "Normalizing answers", num_processes)
    normalize_results.sort(key=lambda x: x[0])
    eval_df["agent_answer_normalized"] = [result[1] for result in normalize_results]

    # Check correctness
    print("Checking correctness...")
    correctness_data = [(idx, answer_for_judge(row["agent_answer"]), row["true_answer_extracted"], row["question"], args.model, row["agent_answer_normalized"], row.get("error"))
                       for idx, row in eval_df.iterrows()]
    correctness_results = process_in_parallel(correctness_data, worker_check_correctness, 
                                            "Checking correctness", num_processes)
    correctness_results.sort(key=lambda x: x[0])
    eval_df["answer_correctness"] = [result[1] for result in correctness_results]
    
    return eval_df


def save_results(eval_df):
    """Save evaluation results to JSON file"""
    print("=" * 60)
    print("SAVING RESULTS...")
    print("=" * 60)
    
    save_cols = [
        "question_id", "question", "assumption", "question_with_context", 
        "true_answer", "agent_answer", "typed_correct", "answer_type", "typed_reason", "typed_parsed",
        "answer_correctness",
        "trace", "true_answer_extracted", "agent_answer_normalized", "usage",
        "limitation", "error", "tokens", "tool_calls", "latency", "budget", "revision",
        "precision", "recall", "true_fhir_resource_types", "template",
        "evidence", "evidence_cited", "evidence_valid", "valid_evidence_fraction", "evidence_issues",
        "evidence_precision", "evidence_recall"
    ]
    save_cols = [col for col in save_cols if col in eval_df.columns]

    os.makedirs(args.output_dir, exist_ok=True)
    
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f'{input_base}_eval.json')
    
    eval_df[save_cols].to_json(output_path, orient='records', indent=2)
    print(f"Results saved to: {output_path}")
    return output_path




def summarize_metrics(df):
    """Answer accuracy (typed = primary, LLM judge = secondary) and retrieval metrics for a group of rows"""
    prec, rec = df['precision'].mean(), df['recall'].mean()
    mean = lambda col: df[col].mean() if col in df else float('nan')
    return {
        "questions": len(df),
        "typed_accuracy": mean('typed_correct'),
        "accuracy": mean('answer_correctness'),
        "precision": prec,
        "recall": rec,
        "f1_score": 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0,
        "valid_evidence_fraction": mean('valid_evidence_fraction'),
        "evidence_precision": mean('evidence_precision'),
        "evidence_recall": mean('evidence_recall'),
    }


def save_performance_summary_to_csv(eval_df, input_file, output_eval_file, output_dir="output_eval"):
    """Save performance summary to a CSV file, one row per resource type (including Overall).
    `accuracy` is the LLM judge; `typed_accuracy` is the deterministic typed scorer."""
    summary_csv = os.path.join(output_dir, "performance_summary.csv")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {"date": now, "input_file": input_file, "output_eval_file": output_eval_file}

    rows = [{**meta, "fhir_resource_type": "Overall", **summarize_metrics(eval_df)}]
    if 'true_fhir_resource_types' in eval_df.columns:
        for resource_type, group in eval_df.groupby('true_fhir_resource_types'):
            rows.append({**meta, "fhir_resource_type": resource_type, **summarize_metrics(group)})

    fieldnames = [
        "date", "input_file", "output_eval_file", "fhir_resource_type",
        "questions", "precision", "recall", "f1_score", "accuracy", "typed_accuracy",
        "valid_evidence_fraction", "evidence_precision", "evidence_recall"
    ]
    os.makedirs(output_dir, exist_ok=True)
    new_rows = pd.DataFrame(rows, columns=fieldnames)
    if os.path.exists(summary_csv):
        existing = pd.read_csv(summary_csv)
        # Files written before typed/evidence scoring existed lack those columns
        new_rows = pd.concat([existing.reindex(columns=fieldnames), new_rows], ignore_index=True)
    new_rows.to_csv(summary_csv, index=False)


def print_performance_summary(eval_df):
    """Print comprehensive performance summary"""
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    overall = summarize_metrics(eval_df)
    total_questions = overall["questions"]
    print(f"Total Questions Evaluated: {total_questions}")
    print()
    print("ANSWER CORRECTNESS:")
    if 'typed_correct' in eval_df.columns:
        print(f"  Typed scorer (primary): {eval_df['typed_correct'].sum()}/{total_questions} "
              f"= {overall['typed_accuracy']:.4f}")
    if 'answer_correctness' in eval_df.columns:
        print(f"  LLM judge (secondary):  {eval_df['answer_correctness'].sum()}/{total_questions} "
              f"= {overall['accuracy']:.4f}")
    if {'typed_correct', 'answer_correctness'} <= set(eval_df.columns):
        agreement = (eval_df['typed_correct'] == eval_df['answer_correctness']).mean()
        print(f"  Typed vs LLM agreement: {agreement:.4f}")
    print()
    print("RETRIEVAL PERFORMANCE:")
    print(f"  Average Precision: {overall['precision']:.4f}")
    print(f"  Average Recall: {overall['recall']:.4f}")
    print(f"  F1 Score: {overall['f1_score']:.4f}")
    print()
    if 'evidence_cited' in eval_df.columns:
        print("EVIDENCE:")
        print(f"  Valid citations: {eval_df['evidence_valid'].sum()}/{eval_df['evidence_cited'].sum()}; "
              f"mean valid fraction per answer: {overall['valid_evidence_fraction']:.4f}")
        issues = sum((Counter(i) for i in eval_df['evidence_issues'] if isinstance(i, dict)), Counter())
        if issues:
            print(f"  Invalid citations by reason: {dict(issues)}")
        print(f"  Evidence Precision: {overall['evidence_precision']:.4f}")
        print(f"  Evidence Recall: {overall['evidence_recall']:.4f}")
        print()

    if 'answer_type' in eval_df.columns:
        print("TYPED ACCURACY BY ANSWER TYPE:")
        for answer_type, group in eval_df.groupby('answer_type'):
            unparsed = (group['typed_reason'] == 'unparsed').sum()
            print(f"  {answer_type}: {group['typed_correct'].sum()}/{len(group)} = {group['typed_correct'].mean():.4f}"
                  + (f" ({unparsed} unparsed)" if unparsed else ""))
        print()

    if 'true_fhir_resource_types' in eval_df.columns:
        print("PERFORMANCE BY RESOURCE TYPE:")
        for resource_type, group in eval_df.groupby('true_fhir_resource_types'):
            m = summarize_metrics(group)
            print(f"  {resource_type}:")
            print(f"    Questions: {m['questions']}, Typed accuracy: {m['typed_accuracy']:.4f}, "
                  f"LLM accuracy: {m['accuracy']:.4f}")
            print(f"    Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
        print()

    print("=" * 60)


def check_existing_results():
    """Check if evaluation results already exist"""
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f'{input_base}_eval.json')
    
    if os.path.exists(output_path) and not args.force:
        print("=" * 60)
        print("EXISTING RESULTS FOUND!")
        print("=" * 60)
        print(f"Results file already exists: {output_path}")
        print("Use --force to re-evaluate or press Enter to view existing results.")
        
        response = input("Do you want to re-evaluate? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Loading existing results...")
            existing_df = pd.read_json(output_path)
            print_performance_summary(existing_df)
            return True, existing_df
    elif os.path.exists(output_path) and args.force:
        print(f"Force mode: Overwriting existing results at {output_path}")
    
    return False, None


def main():
    """Main evaluation pipeline"""
    setup_environment()
    
    # Check if results already exist
    exists, existing_df = check_existing_results()
    if exists:
        # If loaded from existing, we don't know the output file path, so skip CSV save
        print_performance_summary(existing_df)
        return
    
    # Run full evaluation pipeline
    eval_df = load_and_merge_data()
    eval_df = filter_evaluation_data(eval_df)
    eval_df = calculate_retrieval_metrics(eval_df)
    eval_df = calculate_typed_metrics(eval_df)
    if not args.skip_llm_judge:
        eval_df = calculate_answer_metrics(eval_df, args.num_processes)
    output_path = save_results(eval_df)
    
    # Print performance summary and save to CSV
    print_performance_summary(eval_df)

    # Save summary to CSV
    save_performance_summary_to_csv(eval_df, args.input, output_path, args.output_dir)    
    print("=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print()



if __name__ == "__main__":
    main()
