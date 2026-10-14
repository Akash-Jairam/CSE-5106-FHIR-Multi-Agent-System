"""Post-run mechanism ledger; never supplies feedback to benchmark agents."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    from scripts.full_benchmark import iter_result_rows
    from scripts.variability import first_attempt, is_failed
    from utils.answer_scorer import score_answer
    source_path = root / 'final_dataset/questions_answers_sql_fhir.csv'
    with source_path.open(newline='') as stream:
        source = {row['question_id']: row for row in csv.DictReader(stream)}
    with (root / 'manifests/question_families.csv').open(newline='') as stream:
        families = {row['question_id']: row for row in csv.DictReader(stream)}
    expected = (set(source) if args.full else set(json.loads(
        (root / 'manifests/split.json').read_text())['splits']['held_out']['question_ids']))
    records, summaries = [], {}
    def score(value, qid):
        ref = source[qid]
        return score_answer(value, ref['true_answer'], ref['question'], families[qid]['template'])
    def issues(value):
        return sorted({item.get('type', 'malformed_issue') for item in value or []
                       if isinstance(item, dict)})
    for specification in args.arm:
        name, separator, filename = specification.partition('=')
        if not separator or name in summaries:
            raise ValueError('Arms must have unique NAME=PATH specifications')
        rows = []
        seen = set()
        for original in iter_result_rows(filename):
            qid = original['question_id']
            if qid in seen or qid not in expected:
                raise ValueError(f'{name}: duplicate or unexpected question')
            seen.add(qid)
            row = first_attempt(original)
            failed = is_failed(row)
            if row.get('latency') is None and not failed:
                raise ValueError(f'{name}: incomplete first attempt')
            revision = row.get('revision') or {}
            draft = revision.get('draft')
            messages = revision.get('messages') or []
            if messages and not isinstance(messages, list):
                raise ValueError('Restore external audit messages before mechanism analysis')
            verdicts = [message['payload'] for message in messages if isinstance(message, dict)
                        and message.get('sender') == 'verification'
                        and message.get('kind') == 'verdict' and isinstance(message.get('payload'), dict)]
            independent = [verdict['independent_computation'] for verdict in verdicts
                           if isinstance(verdict.get('independent_computation'), dict)]
            final_score = score(row.get('agent_answer'), qid)
            released = not failed and revision.get('final_check_issues') == []
            rec = {'arm': name, 'question_id': qid, 'family': families[qid]['family'],
                   'terminal_error': failed, 'correct': bool(not failed and final_score['typed_correct']),
                   'reference_empty': final_score['answer_type'] == 'empty',
                   'draft_correct': bool(score(draft.get('answer'), qid)['typed_correct']) if isinstance(draft, dict) else None,
                   'independent_correct': bool(score(independent[-1].get('answer'), qid)['typed_correct']) if independent else None,
                   'independent_artifacts': len(independent), 'independent_reused': any(
                       bool(verdict.get('independent_check_reused')) for verdict in verdicts),
                   'released_after_clean_check': released, 'revised': bool(revision.get('revised')),
                   'initial_issue_types': '|'.join(issues(revision.get('check_issues'))),
                   'final_issue_types': '|'.join(issues(revision.get('final_check_issues'))),
                   'budget_stop': (row.get('budget') or {}).get('stopped'),
                   'context_compactions': len(revision.get('context_compactions') or [])}
            rows.append(rec)
        if seen != expected:
            raise ValueError(f'{name}: incomplete scheduled cohort')
        count = lambda predicate: sum(bool(predicate(row)) for row in rows)
        summaries[name] = {'questions': len(rows), 'terminal_errors': count(lambda r:r['terminal_error']),
                           'correct': count(lambda r:r['correct']),
                           'empty_reference_count': count(lambda r:r['reference_empty']),
                           'empty_reference_correct': count(lambda r:r['reference_empty'] and r['correct']),
                           'nonempty_reference_correct': count(lambda r:not r['reference_empty'] and r['correct']),
                           'correct_draft_lost': count(lambda r:r['draft_correct'] is True and not r['correct']),
                           'correct_draft_lost_nonterminal': count(lambda r:r['draft_correct'] is True and not r['correct'] and not r['terminal_error']),
                           'correct_draft_terminal_error': count(lambda r:r['draft_correct'] is True and r['terminal_error']),
                           'incorrect_draft_recovered': count(lambda r:r['draft_correct'] is False and r['correct']),
                           'correct_independent_with_incorrect_final': count(lambda r:r['independent_correct'] is True and not r['correct']),
                           'incorrect_independent_with_correct_draft': count(lambda r:r['independent_correct'] is False and r['draft_correct'] is True),
                           'clean_release_count': count(lambda r:r['released_after_clean_check']),
                           'credited_without_clean_release': count(lambda r:r['correct'] and not r['released_after_clean_check']),
                           'initial_issue_types': dict(Counter(kind for r in rows for kind in r['initial_issue_types'].split('|') if kind)),
                           'final_issue_types': dict(Counter(kind for r in rows for kind in r['final_issue_types'].split('|') if kind))}
        records.extend(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'mechanisms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {'scope': 'Post-hoc descriptive mechanisms; no inference-time gold feedback.',
              'scorer': 'Unchanged typed scorer; first attempts; terminal errors incorrect.',
              'caveat': 'Source-valid or independent agreement does not establish semantic correctness or retrieval completeness. The primary scorer may credit unverified null on empty references.',
              'dataset_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'arms': summaries}
    (output / 'mechanisms.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
