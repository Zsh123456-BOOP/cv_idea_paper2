"""Select an exact-size review batch of whole training questions, without repeats."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ... import pages
from ...data.clean_pages import review_clean
from ...data.prior_annotations import twins, writer_splits
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .page_prefill import from_revrender, numbered


def exact_questions(questions, count):
    """Prefer many writers; never truncate a selected question to hit the budget."""
    choices = {0: []}
    for i, q in enumerate(questions):
        size = len(q['revisions'])
        for total, picked in sorted(list(choices.items()), reverse=True):
            new = total + size
            if new <= count and (new not in choices or len(picked)+1 > len(choices[new])):
                choices[new] = picked + [i]
    require(count in choices, f'cannot make exactly {count} regions from complete questions')
    return [questions[i] for i in choices[count]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--same-student', type=Path, required=True)
    p.add_argument('--exclude-export', type=Path, action='append', default=[])
    p.add_argument('--pages', type=Path, required=True)
    p.add_argument('--revrender', type=Path, required=True)
    p.add_argument('--count', type=int, default=80)
    p.add_argument('--max-per-page', type=int, default=6)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    require(args.count > 0 and args.max_per_page >= 2, 'invalid batch size')
    splits = writer_splits(args.mapping, args.same_student); other = twins(args.same_student)
    excluded = set()
    for path in args.exclude_export:
        excluded.update(key.split('_')[0] for key in json.loads(path.read_text())['pages'])
    excluded |= {other[w] for w in list(excluded) if w in other}
    entries = json.loads(args.selection.read_text())['pages']
    inputs = {str(f): digest(f) for f in [args.selection,args.mapping,args.same_student,*args.exclude_export,
                                         args.revrender/'run_manifest.json',args.pages/'run_manifest.json']}
    candidates = []; used = set(); by_id = {}; scanned = 0
    for entry in entries:
        key = entry['image_id']; writer = key.split('_')[0]; canonical = min(writer,other.get(writer,writer))
        if entry['split'] != 'train' or writer in excluded or canonical in used:
            continue
        if splits.get(writer) != 'train' or splits.get(other.get(writer,writer),'train') != 'train':
            continue
        path = args.pages/(key+'.png')
        if not path.exists() or not (args.revrender/(key+'.json')).exists():
            continue
        with Image.open(path) as im:
            grey = review_clean(np.asarray(im.convert('L')))
        record, instances = pages.read_output(args.revrender,key)
        require(tuple(record['size']) == grey.shape[::-1], 'scan/prediction dimensions differ')
        revisions, _ = numbered([r for r in from_revrender(grey,record,instances) if r['kind']=='line'])
        scanned += 1
        if not 2 <= len(revisions) <= args.max_per_page:
            continue
        used.add(canonical); by_id[key] = entry
        candidates.append({'id':key,'question':entry['question'],'split':'train','pilot':False,
                           'width':grey.shape[1],'height':grey.shape[0],'source':'revrender',
                           'revisions':revisions,'writer_group':canonical})
        for f in [path,args.revrender/(key+'.json'),args.revrender/(key+'.npz')]:
            inputs[str(f)] = digest(f)
    selected = exact_questions(candidates,args.count)
    out = make_output_dir(args.output_dir)
    write_json(out/'prefill.json',{'which':'train','questions':selected})
    write_json(out/'selection.json',{'pages':[by_id[q['id']] for q in selected], 'seed':42})
    summary = {'questions':len(selected),'regions':sum(len(q['revisions']) for q in selected),
               'writers':len({q['writer_group'] for q in selected}),
               'question_numbers':dict(Counter(q['question'] for q in selected)),
               'scanned_pages':scanned,'candidate_pages':len(candidates),'human_labels':False}
    write_manifest(out,config=to_jsonable(args),inputs=inputs,**summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
