"""Metadata-only cache provenance audit of the immutable 891-row capture.

No live canonical read, answer access, scoring, provider calls, or source edits.
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
CAPTURE = Path('/private/tmp/single-measurement-checkpoint-891')
OUT = Path('/private/tmp/cache-provenance-checkpoint-891')


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
    fixed = {'safe_evidence.json':'76fee805787376b052773d8d924b0b3ae95a7220b1e32e8148b627e8138f8074',
             'per_question_measurements.jsonl':'961b39cef4f3f8b566ab6c98c24160e3362e9e5e6574f538b51ce942a205bc3b',
             'audit_measurements.py':'48aa1253426f2822b522a83528b30c97ef3aa1bc4167a887004ae73569ebe2e6'}
    for name,digest in fixed.items():
        require(sha(CAPTURE/name)==digest,'Immutable capture changed: '+name)
    capture = json.loads((CAPTURE/'safe_evidence.json').read_bytes())
    require(capture['accepted_measurements']==891 and capture['source_stable'] and capture['runtime_failed_attempts_excluded']==21,'891 capture is not valid')
    proof_path = ROOT/'benchmarks/single-agent-full-run-20261017/sealed_round10_first_clean_validation.json'
    require(sha(proof_path)=='32cee962f1bbc37af3bda55fa3a7893fed3d3cef81c9d42d18aa55004f94c77c','Proof differs')
    proof = json.loads(proof_path.read_bytes())
    require(proof['selected_accepted']==891 and proof['canonical_first_clean_verified'] and proof['accepted_reruns']==0,'First-clean proof differs')
    selected = {}
    for line in (CAPTURE/'per_question_measurements.jsonl').read_text().splitlines():
        row = json.loads(line)
        require(row['question_id'] not in selected,'Duplicate captured question')
        selected[row['question_id']] = row
    require(len(selected)==891 and sum(len(r['per_call_usage']) for r in selected.values())==5858,'Captured call cohort differs')
    spec = importlib.util.spec_from_file_location('reviewed_numeric_projection',CAPTURE/'audit_measurements.py')
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    by_round = defaultdict(dict)
    for relative,binding in capture['selected_raw_measurement_bindings'].items():
        path = RUN/relative
        require(sha(path)==binding['sha256'],'Sealed raw changed')
        found = 0
        for row in projection.project(path.read_bytes()):
            qid = row.get('question_id')
            saved = selected.get(qid)
            if saved is None:
                continue
            # All inspected fields are IDs or numeric measurement structures.
            if row['patient_fhir_id']==saved['patient_fhir_id'] and row['latency']==saved['latency_seconds'] and row['tokens']['per_call']==saved['per_call_usage']:
                require(all(qid not in other for other in by_round.values()),'Captured selection matches multiple sealed attempts')
                by_round[binding['round']][qid] = saved
                found += 1
        require(found==binding['selected_rows'],'Captured selected round count differs')
    require(sum(len(x) for x in by_round.values())==891,'Round assignment incomplete')
    round_evidence, all_events, journal_hashes, streams = {},[],{},defaultdict(list)
    for number in range(1,11):
        folder = RUN/'diagnostics/single_agent'/f'round-{number:04}'
        paths = sorted(folder.glob('azure-diagnostics-*.jsonl'))
        require(paths,'Diagnostic files absent')
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
        require(sha(CAPTURE/name)==digest,'Fixed capture changed during audit')
    output = OUT/'accepted_call_metadata_associations.jsonl'
    output.write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in associations))
    write(OUT/'per_question_candidate_counts.json',question_candidates)
    write(OUT/'safe_evidence.json',{
        'status':'METADATA_ONLY_OBSERVED_ASSOCIATION_AUDIT_COMPLETE','label':'IN_PROGRESS_NOT_FINAL',
        'at_utc':datetime.now(timezone.utc).isoformat(),'selected_questions':891,'selected_calls':5858,
        'fixed_capture_hashes':fixed,'first_clean_proof_sha256':sha(proof_path),
        'current_live_canonical_read':False,'used_immutable_historical_891_capture':True,
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
                       'Round1 contains interrupted unpersisted work and unmatched start events; observed event totals are not automatically accepted-cohort call totals.',
                       'No correction to frozen metrics or estimates was made; an absent cache detail is never inferred to be zero.'],
        'no_answers_references_correctness_inspected':True,'no_scoring_or_model_provider_calls':True,
        'no_repo_live_or_frozen_metrics_edits':True,'script_sha256':sha(Path(__file__)),
        'association_file_sha256':sha(output)})
    print(json.dumps({'status':'PASS','selected_questions':891,'selected_calls':5858,'matched_calls':len(associations),
                      'successful_diagnostic_events':len(success),'cache_counts':count_cache(success),
                      'candidate_counts':dict(candidate_counts),'overlap_count':len(collisions)}))


if __name__=='__main__':
    main()
