"""Score human-anchored key characters, never whole-line CER or inferred gold text."""
import argparse
import json
from pathlib import Path

from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SCHEMA = 'revision-keychar-review'
TASKS = ('verbatim', 'deletion')
MODES = ('B', 'C', 'D')
KNOWN = {'outside', 'inside', 'wrong_outside', 'wrong_inside', 'missing'}
REVISION = {'unreviewed', 'yes', 'no', 'uncertain', 'duplicate'}
DAMAGE = {'unreviewed', 'yes', 'no', 'uncertain'}


def validate(annotation, row_ids, dataset_id):
    require(annotation.get('schema') == SCHEMA, 'wrong annotation schema')
    require(annotation.get('dataset_id') == dataset_id, 'annotation belongs to another pack')
    rows = annotation.get('rows', {})
    require(isinstance(rows, dict) and set(rows) <= row_ids, 'unknown annotation rows')
    for row in rows.values():
        require(row.get('revision') in REVISION, 'unknown revision status')
        require(isinstance(row.get('reviewed'), bool), 'missing row review flag')
        keys = row.get('keys', [])
        require(isinstance(keys, list), 'keys must be a list')
        require(len({k['id'] for k in keys}) == len(keys), 'duplicate key character id')
        if row['reviewed'] and row['revision'] == 'yes':
            require(row.get('key_region_complete') is True, 'key region has not been fully reviewed')
            require(bool(keys), 'confirmed positive requires at least one anchored revision')
        for key in keys:
            require(key.get('kind') in ('cancelled', 'valid'), 'unknown character kind')
            require(isinstance(key.get('char'), str) and bool(key['char'].strip()), 'missing character label')
            require(isinstance(key.get('readable'), bool), 'missing readability flag')
            box = key.get('box', [])
            require(len(box) == 4 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in box)
                    and box[0] < box[2] and box[1] < box[3], 'invalid normalized character box')
            require(key.get('whitening_damage') in DAMAGE, 'unknown whitening damage status')
            judgments = key.get('judgments', {})
            require(set(judgments) <= set(TASKS), 'unknown test in judgments')
            for result in judgments.values():
                require(set(result) <= set(MODES) and all(x in KNOWN | {'', 'uncertain'} for x in result.values()),
                        'unknown character judgment')
            if row['reviewed'] and row['revision'] == 'yes' and key['readable'] and key['char'].strip() != '□':
                require(all(judgments.get(t, {}).get(m) in KNOWN | {'uncertain'} for t in TASKS for m in MODES),
                        'confirmed character is missing judgments')
                require(key['kind'] != 'valid' or key['whitening_damage'] != 'unreviewed',
                        'valid ink damage has not been reviewed')
    return annotation


def fraction(count, denominator):
    return {'count': count, 'denominator': denominator, 'rate': count/denominator if denominator else None}


def score(annotation, total_rows):
    rows = list(annotation.get('rows', {}).values())
    positive = [r for r in rows if r.get('reviewed') and r.get('revision') == 'yes'
                and r.get('key_region_complete')]
    keys = [k for r in positive for k in r.get('keys', [])]
    readable = [k for k in keys if k.get('readable') and k.get('char', '').strip() not in ('', '□')]
    cancelled = [k for k in readable if k['kind'] == 'cancelled']
    valid = [k for k in readable if k['kind'] == 'valid']
    matched = [k for k in cancelled if all(k.get('judgments', {}).get('deletion', {}).get(m) in KNOWN for m in MODES)]
    leakage = {m: fraction(sum(k['judgments']['deletion'][m] in ('outside', 'wrong_outside') for k in matched),
                           len(matched)) for m in MODES}
    cut_off = {}
    for task in TASKS:
        correct = {'outside', 'inside'} if task == 'verbatim' else {'outside'}
        assessed = [k for k in valid if k.get('whitening_damage') in ('yes', 'no')
                    and all(k.get('judgments', {}).get(task, {}).get(m) in KNOWN for m in ('B', 'C'))]
        baseline = [k for k in assessed if k['judgments'][task]['B'] in correct]
        broken = [k for k in baseline if k['judgments'][task]['C'] not in correct and k['whitening_damage'] == 'yes']
        cut_off[task] = {**fraction(len(broken), len(baseline)), 'assessed_valid_keys': len(assessed),
                         'unassessed_valid_keys': len(valid)-len(assessed)}
    return {'coverage': {'total_rows': total_rows, 'reviewed_rows': sum(bool(r.get('reviewed')) for r in rows),
                         'confirmed_positive_rows': len(positive),
                         'excluded_rows': sum(bool(r.get('reviewed')) and r.get('revision') in ('no', 'duplicate') for r in rows),
                         'uncertain_rows': sum(bool(r.get('reviewed')) and r.get('revision') == 'uncertain' for r in rows),
                         'unreadable_keys': len(keys)-len(readable), 'readable_cancelled_keys': len(cancelled),
                         'readable_valid_keys': len(valid), 'unmatched_cancelled_keys': len(cancelled)-len(matched)},
            'cancelled_ink_leakage': leakage, 'valid_ink_cut_off': cut_off}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    path = args.pack/'pack.json'
    pack = json.loads(path.read_text())
    require(pack.get('selection', {}).get('kind') == 'revision-only', 'requires revision-only pack')
    row_ids = {r['id'] for page in pack['pages'] for r in page['rows']}
    annotation = validate(json.loads(args.annotations.read_text()), row_ids, digest(path))
    metrics = score(annotation, len(row_ids))
    out = make_output_dir(args.output_dir)
    write_json(out/'metrics.json', metrics)
    write_manifest(out, config=to_jsonable(args), inputs={str(x): digest(x) for x in (path, args.annotations)},
                   metric_scope='human-confirmed revision cuts; annotated key characters only; no CER')
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
