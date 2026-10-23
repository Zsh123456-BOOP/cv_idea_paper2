"""The questions the user annotates for paper 2: whole questions with much writing and several revised lines, per split of paper 1.

    python -m revrender.data.annotation_set --mapping index_manifest.json --same-student pairs.json --output-dir OUT
        [--lines test=400 val=200 train=200] [--min-lines 12] [--min-edited 3] [--pilot 5]

A question image is a candidate when its writer is in paper 1's training, development or test split (``prior_annotations.writer_splits``,
the two ids of one student counted as one: development is paper 2's validation split; the writers paper 1 did not use go to no split
here), it has at least ``--min-lines`` lines in the catalogue and at least ``--min-edited`` of them were judged revised.  Per split,
questions are taken in a fixed order (a hash of the image id, seed 42) one question number after another, so every question of the exam
is drawn from before any is drawn twice, and a writer gives a second question only when every writer of the split has given one; a split
stops at the first question that brings its lines to the target.  The first ``--pilot`` questions of the test split are the pilot.

A labmate's writers never come in (``data.allowed``: the catalogue holds only allowed writers); ``--exclude`` leaves out the
questions of earlier selections, ``--exclude-export`` those an annotator already worked on in a review export, and ``SPLIT=all`` takes
every question that qualifies.

OUT/selection.json lists the questions (image id, split, writer, question, lines, revised lines, pilot) and the counts per split; only ids
and counts, no text.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from . import lines as lines_module
from .allowed import refuse_outside
from .prior_annotations import writer_splits

SPLITS = {'test': 'test', 'dev': 'val', 'train': 'train'}     # paper 1's split -> paper 2's name
TARGETS = {'test': 400, 'val': 200, 'train': 200}


def question_of(image_id):
    """The question number of an image id (``<writer>_<question>_<part>``; the supplementary scans write ``s<question>``)."""
    return int(image_id.split('_')[1].lstrip('s'))


def candidates(catalogue, writers, min_lines, min_edited):
    """{split: [{image_id, writer, question, lines, edited}]} of the questions that qualify, in paper 2's split names."""
    pages = defaultdict(list)
    for record in catalogue:
        pages[record['image_id']].append(record)
    out = defaultdict(list)
    for image_id, records in sorted(pages.items()):
        writer = records[0].get('group_uuid') or image_id.split('_')[0]
        split = SPLITS.get(writers.get(writer))
        edited = sum(bool(r.get('has_edit')) for r in records)
        if split and len(records) >= min_lines and edited >= min_edited:
            out[split].append({'image_id': image_id, 'writer': writer, 'question': question_of(image_id), 'lines': len(records), 'edited': edited})
    return dict(out)


def rank(image_id, seed=SEED):
    return hashlib.sha256(f'annotation-set:{seed}:{image_id}'.encode()).hexdigest()


def choose(pool, target):
    """Questions of one split until their lines reach ``target``: round robin over question numbers, a writer's second question only
    after every writer has given one."""
    by_question = defaultdict(list)
    for page in sorted(pool, key=lambda p: rank(p['image_id'])):
        by_question[page['question']].append(page)
    order = sorted(by_question, key=lambda q: rank(f'question-{q}'))
    chosen, used, lines = [], defaultdict(int), 0
    while lines < target and any(by_question.values()):
        progress = False
        for question in order:
            queue = by_question[question]
            if not queue:
                continue
            fewest = min(used[p['writer']] for p in queue)
            page = next(p for p in queue if used[p['writer']] == fewest)
            if fewest > min(used[p['writer']] for p in pool):          # every writer of the split gives one question before any gives two
                continue
            queue.remove(page)
            chosen.append(page)
            used[page['writer']] += 1
            lines += page['lines']
            progress = True
            if lines >= target:
                break
        if not progress:                                                # only writers who already gave more are left: take them anyway
            for question in order:
                if by_question[question]:
                    page = by_question[question].pop(0)
                    chosen.append(page)
                    used[page['writer']] += 1
                    lines += page['lines']
                    break
    return chosen


def annotated(exports):
    """The questions of review exports where the annotator confirmed, edited or wrote anything (a library export lists every question)."""
    out = set()
    for path in exports:
        for image_id, page in json.loads(path.read_text())['pages'].items():
            if any(r.get('confirmed') or r.get('ops') or r.get('text_confirmed') or r.get('deleted') for r in page.get('revisions', [])):
                out.add(image_id)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--catalogue', type=Path, help='the line catalogue (default: configs/paths.toml)')
    parser.add_argument('--mapping', type=Path, required=True, help="paper 1's split mapping (index_manifest.json)")
    parser.add_argument('--same-student', type=Path, help='json of students with two writer ids')
    parser.add_argument('--lines', nargs='+', default=[f'{k}={v}' for k, v in TARGETS.items()], help='SPLIT=LINES targets')
    parser.add_argument('--min-lines', type=int, default=12)
    parser.add_argument('--min-edited', type=int, default=3)
    parser.add_argument('--pilot', type=int, default=5, help='the first N test questions are the pilot')
    parser.add_argument('--exclude', type=Path, action='append', default=[], help="an earlier selection.json whose questions are left out")
    parser.add_argument('--exclude-export', type=Path, action='append', default=[], help='a review export whose annotated questions are left out')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    targets = {k: v for k, v in (s.split('=') for s in args.lines)}
    require(set(targets) <= set(SPLITS.values()), f'splits are {sorted(SPLITS.values())}')
    catalogue = lines_module.read_catalogue(args.catalogue) if args.catalogue else lines_module.default_catalogue()
    writers = writer_splits(args.mapping, args.same_student)
    excluded = {p['image_id'] for e in args.exclude for p in json.loads(e.read_text())['pages']} | annotated(args.exclude_export)
    pools = {split: [p for p in pool if p['image_id'] not in excluded]
             for split, pool in candidates(catalogue, writers, args.min_lines, args.min_edited).items()}
    pages = []
    for split, target in targets.items():
        target = sum(p['lines'] for p in pools.get(split, [])) if target == 'all' else int(target)
        chosen = choose(pools.get(split, []), target)
        require(sum(p['lines'] for p in chosen) >= target, f'{split}: not enough questions for {target} lines')
        pages += [{**p, 'split': split, 'pilot': split == 'test' and n < args.pilot} for n, p in enumerate(chosen)]
    by_writer = defaultdict(set)
    for p in pages:
        by_writer[p['writer']].add(p['split'])
    require(all(len(s) == 1 for s in by_writer.values()), 'a writer is in two splits')
    summary = {split: {'questions': sum(p['split'] == split for p in pages), 'lines': sum(p['lines'] for p in pages if p['split'] == split),
                       'revised_lines': sum(p['edited'] for p in pages if p['split'] == split), 'writers': len({p['writer'] for p in pages if p['split'] == split}),
                       'question_numbers': len({p['question'] for p in pages if p['split'] == split}), 'candidates': len(pools.get(split, []))}
               for split in targets}
    out = make_output_dir(args.output_dir)
    write_json(out/'selection.json', {'pages': pages, 'summary': summary, 'min_lines': args.min_lines, 'min_edited': args.min_edited, 'seed': SEED})
    refuse_outside([p['image_id'] for p in pages], 'question')
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.mapping, *args.exclude, *args.exclude_export)},
                   summary=summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
