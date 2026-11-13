"""The paper-2 dataset as annotated so far: every confirmed region of the annotator's exports, merged, and split by writer.

    python -m revrender.data.struck_dataset --exports DIR --same-student PAIRS --mapping INDEX_MANIFEST --output-dir OUT

The exports (pixel reviews, ``pixel_masks.SCHEMA``) are read in the order of their file names (oldest first).  On one question a region
is the same region when it carries the same id or its box overlaps a held one by an IoU of at least one half; the latest export decides
it (a later deletion removes it), and a region counts once it is confirmed in pixels or text.  Pages of writers outside the allowed set
(``data.allowed``: a labmate's) are left out.  The split (user 2026-11-13: 8:1:1) is by writer, a student's two ids one writer: the
writers in a seed-42 order fill the test, then the validation part up to a tenth of the regions each, the rest train.  Only a writer
paper 1 did not train on (its test writers, or writers it never used) may be tested, since the detector saw paper 1's training writers;
each writer's split in paper 1 is recorded beside it.

OUT/regions.json has a row per region (id, page, writer, split, box, polygon, mask runs, mode, text, readable, judgment, the annotator's
edits and seconds, the model that prefilled it, the export it came from); OUT/summary.json the counts and the statistics of the
summary.  Student text: OUT stays out of Git.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest
from .allowed import allowed
from .pixel_masks import SCHEMA, decode, has_uncertainty
from .prior_annotations import twins, writer_splits

SHARES = {'test': .1, 'val': .1}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1-x0)*max(0, y1-y0); union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def merged(exports):
    """{page: [the latest record of each region]} over the exports, oldest first."""
    pages = {}
    for export in exports:
        data = json.loads(export.read_text())
        if data.get('schema') != SCHEMA: continue
        for page, p in data['pages'].items():
            if not allowed(page): continue
            held = pages.setdefault(page, [])
            for r in p.get('revisions', []):
                if not r.get('box'): continue
                key = r['id'].split('#')[-1]
                hit = next((x for x in held if x['key'] == key or iou(x['box'], r['box']) >= .5), None)
                if r.get('deleted'):
                    if hit is not None: held.remove(hit)
                    continue
                if not (r.get('confirmed') or r.get('text_confirmed')): continue
                if hit is None: hit = {'key': key, 'first_export': export.name}; held.append(hit)
                pixels, text, judged = hit.get('pixels'), hit.get('text'), hit.get('judged')
                hit.update(box=r['box'], record=r, export=export.name, image_source=data.get('image_source'))
                hit['pixels'] = r if r.get('confirmed') else pixels       # the latest confirmed pixels and text
                hit['text'] = r if r.get('text_confirmed') else text
                hit['judged'] = r if r.get('judgment_confirmed') else judged
    return pages


def split_writers(counts, testable=None, seed=SEED):
    """{writer: split}: writers in a seeded order fill test (only those in ``testable``, when given), then val, to a tenth of the
    regions each; the rest train."""
    order = sorted(counts); random.Random(seed).shuffle(order)
    total = sum(counts.values()); filled = Counter(); out = {}
    for w in order:
        part = next((s for s in SHARES if filled[s] < SHARES[s]*total and (s != 'test' or testable is None or w in testable)), 'train')
        out[w] = part; filled[part] += counts[w]
    return out


def quartiles(values):
    v = np.asarray([x for x in values if x is not None], float)
    return [float(np.percentile(v, q)) for q in (25, 50, 75)] if len(v) else None


def build(args):
    exports = sorted(p for p in args.exports.glob('*.json'))
    require(exports, 'no exports'); other = twins(args.same_student); canon = lambda w: min(w, other.get(w, w))
    paper1 = writer_splits(args.mapping, args.same_student)
    rows = []
    for page, held in sorted(merged(exports).items()):
        for x in held:
            px, tx, r = x.get('pixels'), x.get('text'), x['record']
            l, t, rr, b = px['box'] if px else r['box']
            strike = int(decode(px['runs'], (b-t, rr-l)).sum()) if px and px.get('runs') else 0
            rows.append({'id': page+'#'+x['key'], 'page': page, 'writer': canon(page.split('_')[0]), 'box': px['box'] if px else r['box'],
                         'region': (px or r).get('region'), 'runs': px.get('runs') if px else None, 'mode': px.get('annotationMode') if px else None,
                         'pixels_confirmed': px is not None, 'pixels_uncertain': bool(px and has_uncertainty(px)), 'not_revision': bool(px and px.get('not_revision')),
                         'strike_pixels': strike, 'text': tx.get('text') if tx else None, 'text_confirmed': tx is not None,
                         'readable': bool(tx) and not tx.get('unreadable'), 'judgment': x['judged'].get('judgment') if x.get('judged') else None,
                         'edits': r.get('ops'), 'seconds': r.get('seconds'), 'prefill_model': r.get('model'), 'export': x['export'],
                         'first_export': x['first_export'], 'image_source': x['image_source']})
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate region id')
    counts = Counter(r['writer'] for r in rows); parts = split_writers(counts, {w for w in counts if paper1.get(w, 'none') != 'train'})
    for r in rows: r['split'] = parts[r['writer']]; r['paper1_split'] = paper1.get(r['writer'], 'none')
    out = make_output_dir(args.output_dir); write_json(out/'regions.json', rows)
    write_json(out/'splits.json', {'seed': SEED, 'shares': SHARES, 'writers': {w: {'split': parts[w], 'paper1_split': paper1.get(w, 'none'), 'regions': counts[w]}
                                                                               for w in sorted(parts)}})
    def describe(sel):
        text = [r for r in sel if r['text_confirmed']]
        return {'regions': len(sel), 'questions': len({r['page'] for r in sel}), 'writers': len({r['writer'] for r in sel}),
                'pixels_confirmed': sum(r['pixels_confirmed'] for r in sel), 'text_confirmed': len(text), 'readable': sum(r['readable'] for r in text),
                'unreadable': sum(not r['readable'] for r in text), 'pixels_and_readable_text': sum(r['pixels_confirmed'] and r['readable'] for r in sel),
                'judged': sum(r['judgment'] is not None for r in sel), 'modes': dict(Counter(r['mode'] for r in sel if r['mode'])),
                'paper1_split_of_regions': dict(Counter(r['paper1_split'] for r in sel))}
    summary = {'all': describe(rows), **{s: describe([r for r in rows if r['split'] == s]) for s in ('train', 'val', 'test')},
               'by_export': dict(Counter(r['export'] for r in rows)), 'prefilled_by': dict(Counter(r['prefill_model'] or 'not recorded' for r in rows)),
               'statistics': {'strike_pixels_quartiles': quartiles(r['strike_pixels'] for r in rows if r['pixels_confirmed']),
                              'box_width_quartiles': quartiles(r['box'][2]-r['box'][0] for r in rows),
                              'box_height_quartiles': quartiles(r['box'][3]-r['box'][1] for r in rows),
                              'text_characters_quartiles': quartiles(len(r['text']) for r in rows if r['readable']),
                              'edits_quartiles': quartiles(r['edits'] for r in rows), 'seconds_quartiles': quartiles(r['seconds'] for r in rows),
                              'seconds_total_hours': sum(r['seconds'] or 0 for r in rows)/3600, 'regions_per_writer_quartiles': quartiles(counts.values())}}
    write_json(out/'summary.json', summary)
    write_manifest(out, config={'exports': str(args.exports), 'same_student': str(args.same_student), 'mapping': str(args.mapping)},
                   inputs={str(p): digest(p) for p in [*exports, args.same_student, args.mapping]}, regions=len(rows))
    print(json.dumps({k: summary[k] for k in ('all', 'train', 'val', 'test')}, ensure_ascii=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exports', 'same-student', 'mapping', 'output-dir'): p.add_argument('--'+key, type=Path, required=True)
    build(p.parse_args(argv))


if __name__ == '__main__':
    main()
