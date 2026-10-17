"""Private fixed-checkpoint MEASUREMENTS ONLY audit; no scorer/analyzer imports.

Standard JSON parsing is followed only by approved measurement field access.
Answer/reference/correctness fields are never inspected, printed or evaluated.
Tool-call arguments are not inspected; only the array length is counted.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RUN = ROOT / 'output/experiments/gpt-6-luna/v3_full_2931_20261015'
OUT = Path('/private/tmp/single-measurement-checkpoint-891')
EXPECTED = 891
ALLOWED = {'question_id','patient_fhir_id','latency','tokens','budget'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def project(raw):
    """Standard JSON parse followed only by approved measurement field access."""
    for original in json.loads(raw):
        result = {key:original.get(key) for key in ALLOWED}
        calls = original.get('tool_calls')
        result['tool_calls_recorded'] = len(calls) if isinstance(calls,list) else None
        result['retry_history_present'] = bool(original.get('attempt_history'))
        yield result


def numeric(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0


def scalar_stats(values):
    selected = [v for v in values if numeric(v)]
    missing = len(values) - len(selected)
    if selected:
        ordered = sorted(selected)
        position = (len(ordered)-1)*.95
        low, high = math.floor(position),math.ceil(position)
        observed = {'total':sum(selected),'mean':statistics.fmean(selected),
                    'median':statistics.median(selected),'p95':ordered[low]+(ordered[high]-ordered[low])*(position-low),
                    'sample_stdev':statistics.stdev(selected) if len(selected)>1 else None,
                    'min':min(selected),'max':max(selected)}
    else:
        observed = dict.fromkeys(('total','mean','median','p95','sample_stdev','min','max'))
    return {'n':len(values),'observed_n':len(selected),'missing_n':missing,
            **{k:None if missing else v for k,v in observed.items()},'observed_only':observed}


def main():
    OUT.mkdir(exist_ok=True)
    output = OUT / 'per_question_measurements.jsonl'
    require(not output.exists(),'Measurement output already exists; do not overwrite')
    files = [RUN/'full_single_agent.json',RUN/'acceptance.json',RUN/'pricing.json']
    started = datetime.now(timezone.utc).isoformat()
    before = {str(path):sha(path) for path in files}
    ledger = json.loads(files[1].read_bytes())
    selections = ledger['arms']['single_agent']
    require(len(selections) == EXPECTED,'Checkpoint changed; refuse without restart')
    pricing = json.loads(files[2].read_bytes())
    require(pricing['per_million_tokens'] == {'input':.1,'cached_input':.01,'output':.5},'Official proxy schedule differs')
    proof_path = ROOT/'benchmarks/single-agent-full-run-20261017/sealed_round10_first_clean_validation.json'
    proof = json.loads(proof_path.read_bytes())
    require(proof['selected_accepted'] == EXPECTED and proof['canonical_first_clean_verified'] is True
            and proof['accepted_reruns'] == 0 and proof['replayed_counts'] == ledger['counts']['single_agent'],
            'Prior first-clean proof does not match checkpoint')
    sealed = {b['round']:b for b in proof['sealed_batches']}
    indexed = {}
    for row in project(files[0].read_bytes()):
        qid = row.get('question_id')
        require(qid in selections and qid not in indexed and row.get('patient_fhir_id'),'Canonical identity missing or duplicated')
        require(not row.get('retry_history_present'),'Canonical retry history present')
        selection = selections[qid]
        require(selection['runtime_clean'] is True and selection['selection'] in ('runtime_clean_terminal','intrinsic_terminal_error'),
                'Non-runtime-clean selected row')
        indexed[qid] = row
    require(set(indexed) == set(selections),'Canonical and selected ledger cohorts differ')
    grouped = defaultdict(set)
    for qid, entry in selections.items():
        grouped[entry['attempt_file']].add(qid)
    raw_bindings = {}
    for relative,qids in grouped.items():
        path = (RUN/relative).resolve()
        require(path.is_relative_to(RUN) and path.is_file(),'Selected raw pointer missing or escaping run')
        rounds = {selections[qid]['round'] for qid in qids}
        require(len(rounds) == 1,'Raw file has inconsistent round')
        round_number = next(iter(rounds))
        digest = sha(path)
        require(digest == sealed[round_number]['result_sha256'],'Sealed selected raw attempt changed')
        raw_bindings[relative] = {'sha256':digest,'selected_rows':len(qids),'round':round_number}
        seen = set()
        for row in project(path.read_bytes()):
            qid = row.get('question_id')
            if qid in qids:
                require(qid not in seen and row == indexed[qid],'Canonical measurement projection differs from selected raw attempt: ' + ','.join(k for k in row if row[k] != indexed[qid].get(k)))
                seen.add(qid)
        require(seen == qids,'Selected raw measurement rows missing')
    measurements,unknown_rows,total_calls = [],0,0
    for qid,row in sorted(indexed.items()):
        tokens,budget = row['tokens'],row['budget']
        require(set(tokens) <= {'prompt','completion','total','cost','llm_calls','cached_prompt','repeated_prompt','per_call'},
                'Unexpected token schema; avoid decoding any new content')
        calls = tokens.get('per_call')
        require(isinstance(calls,list),'Per-call usage missing')
        for call in calls:
            require(isinstance(call,dict) and set(call) <= {'prompt','completion','cached','usage_available'},'Unexpected per-call schema')
            require(type(call.get('usage_available')) is bool,'Per-call availability missing')
            require(all(numeric(call.get(k)) for k in ('prompt','completion','cached')) and call['cached'] <= call['prompt'],
                    'Per-call usage invalid or cached subset exceeds input')
        unknown = any(call['usage_available'] is False for call in calls)
        unknown_rows += unknown
        total_calls += len(calls)
        for key in ('prompt','completion','total','llm_calls','cached_prompt','repeated_prompt'):
            require(numeric(tokens.get(key)),'Question token/call measurement missing')
        require(tokens['llm_calls'] == len(calls),'Question call count differs from per-call list')
        for key,aggregate in (('prompt','prompt'),('completion','completion'),('cached','cached_prompt')):
            require(sum(c[key] for c in calls) == tokens[aggregate],'Token aggregate differs from per-call sum')
        require(tokens['prompt']+tokens['completion'] == tokens['total'],'Total token balance differs')
        repeated = sum(min(cur['prompt'],prev['prompt']+prev['completion']) for prev,cur in zip(calls,calls[1:]))
        require(tokens['repeated_prompt'] == repeated,'Repeated input estimate differs')
        require(numeric(row['latency']) and numeric(budget.get('tool_calls_run')),'Latency/executed tool measurement missing')
        require(type(row.get('tool_calls_recorded')) is int and row['tool_calls_recorded'] >= 0,'Recorded tool measurement missing')
        require(tokens['cost'] is None or numeric(tokens['cost']),'Reported bill invalid')
        p,c,o = tokens['prompt'],tokens['cached_prompt'],tokens['completion']
        estimate = None if unknown else ((p-c)*.1+c*.01+o*.5)/1e6
        measurements.append({'question_id':qid,'patient_fhir_id':row['patient_fhir_id'],
                             'latency_seconds':row['latency'],'prompt_tokens':None if unknown else p,
                             'completion_tokens':None if unknown else o,'total_tokens':None if unknown else tokens['total'],
                             'cached_prompt_tokens':None if unknown else c,'uncached_prompt_tokens':None if unknown else p-c,
                             'repeated_prompt_tokens':None if unknown else repeated,'llm_calls':tokens['llm_calls'],
                             'per_call_usage':calls,'provider_usage_available':not unknown,
                             'tool_calls_executed':budget['tool_calls_run'],'tool_calls_recorded':row['tool_calls_recorded'],
                             'estimated_cost_usd':estimate,'reported_cost_usd':tokens['cost']})
    after = {str(path):sha(path) for path in files}
    require(before == after,'Live canonical/ledger/pricing changed; refuse snapshot without restart')
    for relative, detail in raw_bindings.items():
        require(sha(RUN/relative) == detail['sha256'],'Selected raw attempt changed during QA')
    numeric_fields = [key for key in measurements[0] if key not in {'question_id','patient_fhir_id','per_call_usage','provider_usage_available'}]
    output.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in measurements))
    evidence = {'status':'PASS','label':'IN_PROGRESS_NOT_FINAL','accepted_measurements':EXPECTED,'questions_in_full_cohort':2931,
                'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
                'source_hashes_before':before,'source_hashes_after':after,'source_stable':True,
                'selected_raw_measurement_bindings':raw_bindings,'first_clean_proof_sha256':sha(proof_path),
                'runtime_failed_attempts_excluded':ledger['counts']['single_agent']['runtime_failures'],
                'measurement_rows_from_selected_canonical_only':True,'per_call_measurements':total_calls,
                'provider_usage_unknown_rows':unknown_rows,'reported_billing_unknown_rows':sum(r['reported_cost_usd'] is None for r in measurements),
                'metrics':{key:scalar_stats([r[key] for r in measurements]) for key in numeric_fields},
                'pricing':pricing,'partial_accuracy_evaluated':False,'answer_reference_prediction_values_inspected':False,
                'answer_reference_prediction_values_printed':False,'scorer_or_analyzer_calls':False,'provider_calls':False,
                'comparison_or_extrapolation_performed':False,'repo_or_live_files_modified':False,
                'measurement_projection_method':'Standard JSON parsing followed solely by allowlisted measurement/ID access; answer/reference/correctness values unexamined; tool arguments unexamined.',
                'limitations':['891 accepted measurements are a partial progress checkpoint, not final statistics or a comparison.',
                               'Raw first-clean classification is established by the separately reviewed round10 proof; this audit independently matches only selected measurement projections.',
                               'Estimated dollar costs use the frozen official OpenAI rates as an Azure proxy; actual billing is unknown when absent.'],
                'measurement_file_sha256':sha(output),'script_sha256':sha(Path(__file__))}
    (OUT/'safe_evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','label':evidence['label'],'accepted_measurements':EXPECTED,'per_call_measurements':total_calls,
                      'provider_usage_unknown_rows':unknown_rows,'source_stable':True,'script_sha256':evidence['script_sha256']}))


if __name__ == '__main__':
    main()
