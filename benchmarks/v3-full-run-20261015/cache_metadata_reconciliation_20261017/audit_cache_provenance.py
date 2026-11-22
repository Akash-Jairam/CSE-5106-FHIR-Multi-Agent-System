"""Metadata-only cache provenance audit of the complete immutable V3 cohort.

No baseline canonical read, answer access, scoring, provider calls, or source edits.
Unique token-sequence matches are observed associations, not stored keyed joins.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System')
RUN = ROOT/'output/experiments/gpt-6-luna/v3_full_2931_20261015'
EXPORT = ROOT/'output/experiments/gpt-6-luna/v3_complete_2931_20261016'
PACKAGE = ROOT/'benchmarks/v3-full-run-20261015/completed_v3'
OUT = Path('/private/tmp/cache-provenance-v3-complete')


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
    require(not (OUT/'safe_evidence.json').exists(),'Refuse overwrite')
    canonical = RUN/'full_multi_agent.json'
    canonical_expected = '1c8c974ccff99aae802f698ad043f4d959bb1808b3abee0f9e27c9c3ffa8097c'
    require(sha(canonical)==canonical_expected,'Completed V3 canonical changed')
    package_inventory = json.loads((PACKAGE/'ARTIFACT_SHA256SUMS.json').read_bytes())
    checked_names = ['V3_COMPLETE_CONTROLS_PENDING.json','accepted_selection_ledger_at_v3_completion.json',
                     'run_manifest_used_for_v3_export.json','verified_provenance.json','independent_reconciliation.json']
    fixed = {}
    for name in checked_names:
        path = PACKAGE/name
        require(sha(path)==package_inventory[name]['sha256'],'Completed package artifact differs')
        fixed[str(path)] = sha(path)
    receipt_path = EXPORT/'V3_COMPLETE_CONTROLS_PENDING.json'
    require(sha(receipt_path)==fixed[str(PACKAGE/'V3_COMPLETE_CONTROLS_PENDING.json')],'Original complete receipt differs')
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt['status']=='V3_COMPLETE_CONTROLS_PENDING' and receipt['questions']==2931 and receipt['v3_complete'] is True,
            'Complete V3 receipt required')
    require(receipt['v3_canonical_sha256_after']==canonical_expected and receipt['v3_canonical_sha256_before']==canonical_expected,
            'Complete receipt canonical binding differs')
    require(receipt['sealed_v3_rounds']==list(range(1,34)) and receipt['raw_v3_attempt_counts']=={'raw_attempts':3012,'runtime_failures':81,'intrinsic_errors':0},
            'Historical first-clean replay scope differs')
    independent = json.loads((PACKAGE/'independent_reconciliation.json').read_bytes())
    require(independent['status']=='PASS' and independent['questions']==2931 and independent['canonical_sha256']==canonical_expected,
            'Independent completed V3 reconciliation required')
    ledger_path = EXPORT/'validated_v3_input_snapshot/acceptance.json'
    require(sha(ledger_path)==fixed[str(PACKAGE/'accepted_selection_ledger_at_v3_completion.json')], 'Immutable V3 ledger differs')
    ledger = json.loads(ledger_path.read_bytes())
    selections = ledger['arms']['multi_agent']
    ledger_slice = {'selections':selections,'counts':ledger['counts']['multi_agent']}
    semantic = hashlib.sha256(json.dumps(ledger_slice,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(semantic==receipt['v3_ledger_slice_sha256_after'] and len(selections)==2931,'Completed V3 ledger binding differs')
    source_manifest = json.loads((PACKAGE/'run_manifest_used_for_v3_export.json').read_bytes())
    require(sha(PACKAGE/'run_manifest_used_for_v3_export.json')==receipt['source_manifest_sha256'],'Historical source manifest differs')
    reader_source = ROOT/'scripts/full_benchmark.py'
    require(sha(reader_source)==source_manifest['source_sha256']['scripts/full_benchmark.py'],'Frozen stream reader source differs')
    spec = importlib.util.spec_from_file_location('frozen_result_stream_reader',reader_source)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    selected, by_round = {},defaultdict(dict)
    for original in reader.iter_result_rows(canonical):
        qid = original.get('question_id')
        require(qid in selections and qid not in selected and original.get('patient_fhir_id'),'Canonical V3 identity differs')
        require(not original.get('attempt_history'),'Canonical V3 retry history present')
        selection = selections[qid]
        require(selection['runtime_clean'] is True and selection['selection'] in ('runtime_clean_terminal','intrinsic_terminal_error'),
                'Non-first-clean selected V3 row')
        tokens = original.get('tokens') or {}
        calls = tokens.get('per_call')
        require(isinstance(calls,list) and calls,'Canonical V3 per-call usage missing')
        for call in calls:
            require(set(call)<={'prompt','completion','cached','usage_available'} and call.get('usage_available') is True,
                    'Unexpected canonical V3 per-call measurement schema/unknown usage')
        row = {'question_id':qid,'patient_fhir_id':original['patient_fhir_id'],'latency_seconds':original.get('latency'),
               'per_call_usage':calls,'prompt_tokens':tokens.get('prompt'),'completion_tokens':tokens.get('completion'),
               'cached_prompt_tokens':tokens.get('cached_prompt'),'total_tokens':tokens.get('total'),'llm_calls':tokens.get('llm_calls')}
        require(row['llm_calls']==len(calls) and sum(x['prompt'] for x in calls)==row['prompt_tokens']
                and sum(x['completion'] for x in calls)==row['completion_tokens']
                and sum(x['cached'] for x in calls)==row['cached_prompt_tokens']
                and row['total_tokens']==row['prompt_tokens']+row['completion_tokens'],'V3 per-call aggregate balance differs')
        selected[qid] = row
        by_round[selection['round']][qid] = row
    require(set(selected)==set(selections) and len(selected)==2931,'Complete canonical measurement cohort required')
    total_calls = sum(len(r['per_call_usage']) for r in selected.values())
    provenance = json.loads((PACKAGE/'verified_provenance.json').read_bytes())
    raw_bindings, raw_call_totals = {},{}
    for number in range(1,34):
        relative = f'attempts/multi_agent/round-{number:04}.json'
        path = RUN/relative
        require(str(path) in provenance and sha(path)==provenance[str(path)]['sha256'],'Sealed V3 raw differs from verified provenance')
        raw_bindings[str(path)] = sha(path)
        found = set()
        raw_calls = Counter()
        for original in reader.iter_result_rows(path):
            usage = (original.get('tokens') or {}).get('per_call') or []
            for call in usage:
                require(type(call.get('usage_available')) is bool,'Raw V3 call availability missing')
                raw_calls['calls'] += 1
                raw_calls['usage_available' if call['usage_available'] else 'usage_unavailable'] += 1
            qid = original.get('question_id')
            if qid not in by_round[number]:
                continue
            row = by_round[number][qid]
            require(qid not in found and selections[qid]['attempt_file']==relative,'Selected V3 raw pointer/identity differs')
            tokens = original.get('tokens') or {}
            require(original.get('patient_fhir_id')==row['patient_fhir_id'] and original.get('latency')==row['latency_seconds']
                    and tokens.get('per_call')==row['per_call_usage'] and tokens.get('total')==row['total_tokens'],
                    'Selected V3 raw measurement projection differs')
            found.add(qid)
        require(found==set(by_round[number]),'Selected V3 raw measurement missing')
        raw_call_totals[number] = dict(raw_calls)
    measurement_path = OUT/'per_question_measurements.jsonl'
    measurement_path.write_text(''.join(json.dumps(selected[qid],allow_nan=False)+'\n' for qid in sorted(selected)))
    fixed[str(canonical)] = canonical_expected
    fixed[str(receipt_path)] = sha(receipt_path)
    fixed[str(ledger_path)] = sha(ledger_path)
    fixed[str(reader_source)] = sha(reader_source)
    fixed.update(raw_bindings)
    round_evidence, all_events, journal_hashes, streams = {},[],{},defaultdict(list)
    for number in range(1,34):
        folder = RUN/'diagnostics/multi_agent'/f'round-{number:04}'
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
        require(sha(name)==digest,'Fixed completed V3 input changed during audit')
    output = OUT/'accepted_call_metadata_associations.jsonl'
    output.write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in associations))
    write(OUT/'per_question_candidate_counts.json',question_candidates)
    write(OUT/'safe_evidence.json',{
        'status':'METADATA_ONLY_OBSERVED_ASSOCIATION_AUDIT_COMPLETE','label':'COMPLETE_V3_COHORT_METADATA_PROVENANCE_AUDIT',
        'at_utc':datetime.now(timezone.utc).isoformat(),'selected_questions':2931,'selected_calls':total_calls,
        'fixed_completed_v3_input_hashes':fixed,'first_clean_receipt_sha256':sha(receipt_path),
        'immutable_complete_v3_canonical_read':True,'canonical_sha256':canonical_expected,'baseline_canonical_or_state_read':False,
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
                       'No correction to frozen metrics or estimates was made; an absent cache detail is never inferred to be zero.'],
        'no_answers_references_correctness_inspected':True,'no_scoring_or_model_provider_calls':True,
        'no_repo_live_or_frozen_metrics_edits':True,'measurement_projection_sha256':sha(measurement_path),'script_sha256':sha(Path(__file__)),
        'association_file_sha256':sha(output)})
    print(json.dumps({'status':'PASS','selected_questions':2931,'selected_calls':total_calls,'matched_calls':len(associations),
                      'successful_diagnostic_events':len(success),'cache_counts':count_cache(success),
                      'candidate_counts':dict(candidate_counts),'overlap_count':len(collisions)}))


if __name__=='__main__':
    main()
