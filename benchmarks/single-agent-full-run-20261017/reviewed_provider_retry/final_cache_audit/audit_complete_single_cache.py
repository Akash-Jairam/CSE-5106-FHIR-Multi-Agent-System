"""Metadata-only cache provenance audit of a completed selected Single Agent cohort.

Refuses incomplete cohorts before canonical capture. No answer access, scoring,
provider calls, inference launches, source edits, or metric corrections.
Unique token-sequence matches are observed associations, not stored keyed joins.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

from completed_gate import completion_gate

ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RUN = ROOT/'output/experiments/gpt-6-luna/v3_full_2931_20261015'



def require(test,message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def count_cache(events):
    result = {'null':0,'zero':0,'positive':0,'invalid':0}
    for event in events:
        value = event.get('usage',{}).get('cached_prompt_tokens')
        if value is None:
            result['null'] += 1
        elif type(value) is not int or value < 0:
            result['invalid'] += 1
        elif value == 0:
            result['zero'] += 1
        else:
            result['positive'] += 1
    return result


def signature(call):
    return (call['prompt'],call['completion'])


def event_signature(event):
    usage = event.get('usage') or {}
    return (usage.get('prompt_tokens'),usage.get('completion_tokens')) if event['outcome']=='success' else None


def main():
    global ROOT,RUN,OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--completion-proof',required=True,type=Path,
                        help='Saved JSON output of the reviewed monitor verify --final-arm')
    parser.add_argument('--completion-binding',required=True,type=Path,
                        help='Stable before/after bindings saved beside the captured final monitor proof')
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    gate = completion_gate(args.project_root,args.run_dir,args.completion_proof,args.completion_binding,args.output_dir)
    ROOT,RUN,OUT = gate['root'],gate['run'],gate['output']
    canonical,proof_path = gate['canonical'],gate['proof_path']
    selections,round_proofs,fixed = gate['selections'],gate['rounds'],gate['fixed']
    canonical_expected = fixed[str(canonical)]
    reader_source = gate['reader_source']
    spec = importlib.util.spec_from_file_location('frozen_result_stream_reader',reader_source)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    selected,by_round = {},defaultdict(dict)
    for original in reader.iter_result_rows(canonical):
        qid = original.get('question_id')
        require(qid in selections and qid not in selected and original.get('patient_fhir_id')==gate['patients'][qid],
                'Canonical baseline identity differs')
        require(not original.get('attempt_history'),'Canonical baseline retry history present')
        tokens = original.get('tokens') or {}
        calls = tokens.get('per_call')
        require(isinstance(calls,list) and calls,'Canonical baseline per-call usage missing')
        for call in calls:
            require(set(call)<={'prompt','completion','cached','usage_available'} and call.get('usage_available') is True,
                    'Unexpected canonical baseline per-call measurement schema/unknown usage')
            require(all(type(call.get(k)) is int and call[k]>=0 for k in ('prompt','completion','cached'))
                    and call['cached']<=call['prompt'],'Invalid canonical numeric usage/cache subset')
        row = {'question_id':qid,'patient_fhir_id':original['patient_fhir_id'],'latency_seconds':original.get('latency'),
               'per_call_usage':calls,'prompt_tokens':tokens.get('prompt'),'completion_tokens':tokens.get('completion'),
               'cached_prompt_tokens':tokens.get('cached_prompt'),'total_tokens':tokens.get('total'),'llm_calls':tokens.get('llm_calls')}
        require(row['llm_calls']==len(calls) and sum(x['prompt'] for x in calls)==row['prompt_tokens']
                and sum(x['completion'] for x in calls)==row['completion_tokens']
                and sum(x['cached'] for x in calls)==row['cached_prompt_tokens']
                and row['total_tokens']==row['prompt_tokens']+row['completion_tokens'],'Baseline per-call aggregate balance differs')
        require(selections[qid]['round'] in round_proofs,'Selected baseline round absent from final receipt')
        selected[qid] = row
        by_round[selections[qid]['round']][qid] = row
    require(set(selected)==set(selections) and len(selected)==2931,'Complete canonical measurement cohort required')
    total_calls = sum(len(r['per_call_usage']) for r in selected.values())
    raw_bindings,raw_call_totals = {},{}
    for number,summary in sorted(round_proofs.items()):
        relative = f'attempts/single_agent/round-{number:04}.json'
        path = RUN/relative
        seal_path = RUN/f'attempts/single_agent/round-{number:04}.batch.seal.json'
        require(sha(seal_path)==summary['seal_sha256'],'Selected sealed-round receipt differs')
        fixed[str(seal_path)] = summary['seal_sha256']
        expected = summary['result_sha256']
        if expected is None:
            require(not path.exists() and not by_round[number],'Sealed absent result unexpectedly exists/has accepted rows')
            raw_call_totals[number] = {}
            continue
        require(sha(path)==expected,'Sealed baseline raw differs from completion receipt')
        raw_bindings[str(path)] = expected
        found,raw_calls = set(),Counter()
        for original in reader.iter_result_rows(path):
            usage = (original.get('tokens') or {}).get('per_call') or []
            for call in usage:
                require(type(call.get('usage_available')) is bool,'Raw baseline call availability missing')
                raw_calls['calls'] += 1
                raw_calls['usage_available' if call['usage_available'] else 'usage_unavailable'] += 1
            qid = original.get('question_id')
            if qid not in by_round[number]:
                continue
            row = by_round[number][qid]
            require(qid not in found and selections[qid]['attempt_file']==relative,'Selected baseline raw pointer/identity differs')
            tokens = original.get('tokens') or {}
            require(original.get('patient_fhir_id')==row['patient_fhir_id'] and original.get('latency')==row['latency_seconds']
                    and tokens.get('per_call')==row['per_call_usage'] and tokens.get('total')==row['total_tokens'],
                    'Selected baseline raw measurement projection differs')
            found.add(qid)
        require(found==set(by_round[number]),'Selected baseline raw measurement missing')
        raw_call_totals[number] = dict(raw_calls)
    fixed.update(raw_bindings)
    gate_source = Path(__file__).with_name('completed_gate.py')
    fixed[str(gate_source)] = sha(gate_source)
    # Output capture begins only after the exact full cohort/receipt/raw gates.
    OUT.mkdir(parents=True)
    measurement_path = OUT/'per_question_measurements.jsonl'
    measurement_path.write_text(''.join(json.dumps(selected[qid],allow_nan=False)+'\n' for qid in sorted(selected)))
    round_evidence, all_events, journal_hashes, streams = {},[],{},defaultdict(list)
    for number in sorted(round_proofs):
        folder = RUN/'diagnostics/single_agent'/f'round-{number:04}'
        paths = sorted(folder.glob('azure-diagnostics-*.jsonl'))
        # Missing retained journals are recorded honestly, not presumed complete.
        events = []
        file_info = []
        for path in paths:
            before = sha(path)
            raw = path.read_bytes()
            require(raw.endswith(b'\n'),'Diagnostic journal lacks final newline')
            safe = []
            for line in raw.splitlines():
                entry = json.loads(line)
                record = {k:entry.get(k) for k in ('event','pid','process_session','counter','started_utc','finished_utc',
                          'outcome','duration_seconds','http_status','request_ids','usage')}
                record.update(round=number,file=str(path.relative_to(RUN)))
                require(record['event'] in ('start','end') and type(record['counter']) is int and record['counter']>0,'Unexpected diagnostic event identity')
                safe.append(record)
            require(sha(path)==before==hashlib.sha256(raw).hexdigest(),'Sealed journal changed during read')
            destination = OUT/'raw_diagnostics'/f'round-{number:04}'/path.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(raw)
            require(sha(destination)==before,'Diagnostic preservation differs')
            journal_hashes[str(path)] = before
            starts = Counter((x['pid'],x['process_session'],x['counter']) for x in safe if x['event']=='start')
            ends = Counter((x['pid'],x['process_session'],x['counter']) for x in safe if x['event']=='end')
            keys = set(starts)|set(ends)
            counter_values = {key[2] for key in keys}
            gaps = sorted(set(range(1,max(counter_values)+1))-counter_values) if counter_values else []
            file_info.append({'path':str(path.relative_to(RUN)),'sha256':before,'events':len(safe),
                              'start_events':sum(starts.values()),'end_events':sum(ends.values()),
                              'starts_without_end':[list(k) for k in starts.keys()-ends.keys()],
                              'ends_without_start':[list(k) for k in ends.keys()-starts.keys()],
                              'duplicate_start_keys':sum(v-1 for v in starts.values()),'duplicate_end_keys':sum(v-1 for v in ends.values()),
                              'counter_min':min(counter_values),'counter_max':max(counter_values),'counter_gaps':gaps})
            for entry in safe:
                if entry['event']=='end':
                    streams[number].append(entry)
            events.extend(safe)
        all_events.extend(events)
        successes = [e for e in events if e['event']=='end' and e['outcome']=='success']
        round_evidence[number] = {'selected_questions':len(by_round[number]),
                                  'selected_calls':sum(len(r['per_call_usage']) for r in by_round[number].values()),
                                  'start_events':sum(e['event']=='start' for e in events),'end_success':len(successes),
                                  'end_error':sum(e['event']=='end' and e['outcome']=='error' for e in events),
                                  'successful_event_cache_counts':count_cache(successes),'journals':file_info}
        round_evidence[number]['all_sealed_raw_attempt_call_counts'] = raw_call_totals[number]
        round_evidence[number]['recorded_end_count_minus_raw_calls'] = len([e for e in events if e['event']=='end'])-raw_call_totals[number].get('calls',0)
        round_evidence[number]['recorded_success_count_minus_raw_known_usage_calls'] = len(successes)-raw_call_totals[number].get('usage_available',0)
    associations, candidate_counts, question_candidates, cache_comparisons = [],Counter(),{},Counter()
    claimed = defaultdict(list)
    for number,questions in by_round.items():
        worker_streams = defaultdict(list)
        signatures = defaultdict(list)
        for event in streams[number]:
            worker_streams[(event['pid'],event['process_session'])].append(event)
            if event['outcome']=='success':
                signatures[event_signature(event)].append(event)
        for events in worker_streams.values():
            events.sort(key=lambda e:e['counter'])
        for qid,row in sorted(questions.items()):
            calls = row['per_call_usage']
            targets = [signature(c) for c in calls]
            for target in targets:
                candidate_counts['single_call_no_candidate' if not signatures[target] else 'single_call_unique' if len(signatures[target])==1 else 'single_call_ambiguous'] += 1
            matches = []
            for events in worker_streams.values():
                observed = [event_signature(e) for e in events]
                for i in range(len(events)-len(targets)+1):
                    chunk = events[i:i+len(targets)]
                    if observed[i:i+len(targets)]==targets and all(b['counter']==a['counter']+1 for a,b in zip(chunk,chunk[1:])):
                        matches.append(chunk)
            question_candidates[qid] = {'round':number,'calls':len(calls),'observed_sequence_candidates':len(matches)}
            candidate_counts['question_no_sequence' if not matches else 'question_unique_sequence' if len(matches)==1 else 'question_ambiguous_sequence'] += 1
            if len(matches)!=1:
                continue
            match = matches[0]
            duration = sum(e['duration_seconds'] for e in match)
            require(duration<=row['latency_seconds']+.1,'Matched provider duration exceeds accepted question latency')
            for i,(call,event) in enumerate(zip(calls,match)):
                key = (number,event['pid'],event['process_session'],event['counter'])
                claimed[key].append((qid,i))
                value = event['usage'].get('cached_prompt_tokens')
                cache_comparisons['diagnostic_null' if value is None else 'diagnostic_explicit_zero' if value==0 else 'diagnostic_positive'] += 1
                cache_comparisons['normalized_cache_agrees' if call['cached']==(value if value is not None else 0) else 'normalized_cache_disagrees'] += 1
                associations.append({'question_id':qid,'call_index':i,'round':number,
                                     'association':'unique_observed_contiguous_token_sequence_not_stored_keyed_join',
                                     **{k:event[k] for k in ('file','pid','process_session','counter','started_utc','finished_utc','request_ids','usage')},
                                     'canonical_normalized_cached':call['cached']})
    collisions = {str(key):value for key,value in claimed.items() if len(value)>1}
    success = [e for e in all_events if e['event']=='end' and e['outcome']=='success']
    request_ids = [tuple(sorted((e.get('request_ids') or {}).items())) for e in success]
    final_hashes = {str(path):sha(path) for path in map(Path,journal_hashes)}
    require(journal_hashes==final_hashes,'Diagnostic source changed during audit')
    for name,digest in fixed.items():
        require(sha(name)==digest,'Fixed completed baseline input changed during audit')
    from bound_completion import snapshot
    require(snapshot(ROOT,RUN)==gate['bound_snapshot'],'Completed round/byte inventory changed during cache audit')
    output = OUT/'accepted_call_metadata_associations.jsonl'
    output.write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in associations))
    write(OUT/'per_question_candidate_counts.json',question_candidates)
    write(OUT/'safe_evidence.json',{
        'status':'METADATA_ONLY_OBSERVED_ASSOCIATION_AUDIT_COMPLETE','label':'COMPLETE_SINGLE_AGENT_COHORT_METADATA_PROVENANCE_AUDIT',
        'at_utc':datetime.now(timezone.utc).isoformat(),'selected_questions':2931,'selected_calls':total_calls,
        'fixed_completed_baseline_input_hashes':fixed,'completed_first_clean_proof_sha256':sha(proof_path),
        'completed_proof_binding_sha256':sha(gate['binding_path']),
        'completed_single_agent_canonical_read':True,'canonical_sha256':canonical_expected,'v3_canonical_or_state_modified':False,
        'rounds':round_evidence,'successful_diagnostic_events':len(success),'successful_event_cache_counts':count_cache(success),
        'diagnostic_source_sha256_before':journal_hashes,'diagnostic_source_sha256_after':final_hashes,
        'source_stable':True,'matched_call_records':len(associations),'unique_matched_event_keys':len(claimed),
        'overlapping_question_event_associations':collisions,'join_candidate_counts':dict(candidate_counts),
        'matched_cache_metadata':dict(cache_comparisons),
        'successful_events_without_request_id':sum(not x for x in request_ids),
        'duplicate_nonempty_success_request_id_sets':sum(v-1 for k,v in Counter(request_ids).items() if k),
        'stored_request_id_qid_join_available':False,
        'full_direct_identity_join_proven':False,
        'conclusion':'Observed unique contiguous token-count sequences associate selected calls with journal events. These are metadata associations, not a persisted question/request-ID identity join. Cache presence can be described at observed-event level and for those associations, with the logging/identity limitations retained.',
        'limitations':['Canonical cache zero is lossy and does not by itself prove explicit provider zero.',
                       'Diagnostics are best-effort. Contiguous counters and start/end coverage describe recorded events; failed logging before counter increment may leave no observable gap.',
                       'Numeric sequence uniqueness is relative to the retained journals; question IDs and request IDs were not persisted together in canonical per_call.',
                       'Missing journals, interrupted work and unmatched starts are reported; observed event totals are not automatically accepted-cohort call totals.',
                       'Outcome cohort completeness is mandatory. Diagnostic association coverage is not assumed: missing journals, ambiguous sequences and nullable cache fields are reported without invented zeros.',
                       'No correction to frozen metrics or estimates was made; an absent cache detail is never inferred to be zero.'],
        'no_answers_references_correctness_inspected':True,'no_scoring_or_model_provider_calls':True,
        'no_repo_live_or_frozen_metrics_edits':True,'measurement_projection_sha256':sha(measurement_path),'script_sha256':sha(Path(__file__)),
        'association_file_sha256':sha(output)})
    print(json.dumps({'status':'PASS','selected_questions':2931,'selected_calls':total_calls,'matched_calls':len(associations),
                      'successful_diagnostic_events':len(success),'cache_counts':count_cache(success),
                      'candidate_counts':dict(candidate_counts),'overlap_count':len(collisions)}))


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,KeyError) as error:
        print(json.dumps({'status':'REFUSED','reason':str(error) if isinstance(error,ValueError) else 'Required completed metadata unavailable or malformed'}))
        raise SystemExit(2)
