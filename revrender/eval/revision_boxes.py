"""A model's revisions on whole questions read against the user's boxes (the annotation page's exports, ``paper2-revision-box-v1``).

    python -m revrender.eval.revision_boxes boxes --labels EXPORT.json [--labels ...] --selection selection.json --split val
        --model NAME=OUT_OF_revrender.pages [--model ...] --pages DIR [--pages DIR ...] [--cuts .2 .3 ...] [--cut-from SUMMARY] --output-dir OUT
    python -m revrender.eval.revision_boxes read --pack OUT/pack --readings predictions.jsonl --output read.json
    python -m revrender.eval.revision_boxes pixels --geometry PAGE_LABELS_OUT --model NAME=DIR [--model ...] --pages DIR [...]
        [--cut C | --cut-from SUMMARY] --output pixels.json

``boxes``: the model's removal on a question at a score cut is its instances at or above the cut, pasted on the page and cut to the
page's non-white pixels (``tighten.non_white``).  A revision is found when the removal takes at least ``FOUND`` of the ink in its box
(and ``MIN_FOUND`` pixels); the boxes cover every revision of a question, so a piece of removed ink of at least ``FALSE_PLACE`` pixels
outside every box (grown by ``EDGE``) is a place the model removed valid writing.  Per cut: revisions found (recall), places found
against false places (precision), F1, and the share of the ink outside the boxes removed; the cut with the best F1 is reported (on the
validation split this chooses it; on the test split it is only read, and ``--cut-from`` -- the validation's summary -- gives each model
the cut chosen there).  Several models (``--model NAME=DIR``, a ``revrender.pages`` run each) are read on the same boxes, and ``scan``
-- nothing removed -- with them.  OUT/summary.json (per model), OUT/questions.json (per model and question at its cut) and OUT/pack: a
line pack (``recognize_lines``, condition ``erased_neutral``) with a row per model and revision whose struck text was written
(``<model>|<image id>#<revision>``): the box with a margin, the model's removal whited out (``scan``: as scanned).

``read``: what a reader read on those crops against the struck text: the share of the text's characters a minimal edit alignment of
the two keeps (``eval.coverage``; a ``?`` of the label is not counted), per model and condition, for struck lines and blackouts apart.
``pixels``: on the questions a person labelled stroke by stroke (``data.page_labels``'s output: the pixels of every struck line, a
struck line stopping where it crosses writing), counted on ink inside those questions' struck-line regions: of what a model removes,
the share that is the strike (precision; the rest is writing it should have kept), and of the strike, the share it removes (recall).
Student work: OUT stays out of Git.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import pages as pages_module
from ..data.page_labels import polygon_mask
from ..tighten import non_white
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .coverage import alignment, prepare
from .ink import INK_BELOW

SCHEMA = 'paper2-revision-box-v1'
FOUND = .05                # a revision is found when the removal takes this share of the ink in its box ...
MIN_FOUND = 15             # ... and at least this many pixels
EDGE = 4                   # the boxes are grown by this much before what lies outside them counts
FALSE_PLACE = 30           # removed ink outside the boxes in a piece this big is a false place
CUTS = (.1, .2, .3, .4, .5, .6, .7)
PAD = 6                    # a revision's crop is its box and this margin


def read_labels(paths):
    """{image id: question} from the page's exports, a later file winning for a question it also holds."""
    pages = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        require(value.get('schema') == SCHEMA, f'{path}: not an export of the box annotation page')
        pages.update(value['pages'])
    return pages


def removal(instances, shape, cut, soft):
    """The pixels a model removes at a score cut: its instances at or above the cut on the page, cut to non-white."""
    out = np.zeros(shape, bool)
    for instance in instances:
        if instance['score'] >= cut:
            l, t, r, b = instance['box']
            out[t:b, l:r] |= instance['mask']
    return out & soft


def question_reading(removed, ink, regions):
    """(found flags per revision, false places, removed ink outside the boxes, ink outside the boxes) of one question."""
    found = []
    for region in regions:
        inside = int((ink & region).sum())
        taken = int((removed & ink & region).sum())
        found.append(taken >= max(MIN_FOUND, FOUND*inside))
    boxes = ndimage.binary_dilation(np.logical_or.reduce(regions), iterations=EDGE) if regions else np.zeros(ink.shape, bool)
    outside = removed & ink & ~boxes
    labels, count = ndimage.label(outside, np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel())[1:] if count else np.zeros(0, int)
    return found, int((sizes >= FALSE_PLACE).sum()), int(outside.sum()), int((ink & ~boxes).sum())


def totals(rows):
    """Recall, precision and F1 of a cut over questions (rows of question_reading with the revisions' kinds)."""
    revisions = sum(len(r['found']) for r in rows)
    found = sum(sum(r['found']) for r in rows)
    false = sum(r['false_places'] for r in rows)
    recall, precision = found/max(1, revisions), found/max(1, found+false)
    out = {'revisions': revisions, 'found': found, 'false_places': false, 'recall': recall, 'precision': precision,
           'f1': 2*recall*precision/max(1e-9, recall+precision),
           'removed_outside_share': sum(r['removed_outside'] for r in rows)/max(1, sum(r['ink_outside'] for r in rows))}
    for kind in ('line', 'blot'):
        flags = [f for r in rows for f, k in zip(r['found'], r['kinds']) if k == kind]
        out[f'recall_{kind}'] = sum(flags)/max(1, len(flags))
        out[f'revisions_{kind}'] = len(flags)
    return out


def boxes(args):
    selection = {p['image_id']: p for p in json.loads(args.selection.read_text())['pages']}
    labels = read_labels(args.labels)
    chosen = sorted(i for i, q in labels.items() if selection.get(i, {}).get('split') == args.split and (q.get('done') or args.all))
    require(chosen, f'no labelled {args.split} question (only questions marked done count unless --all)')
    models = dict(m.split('=', 1) for m in args.model)
    require(len(models) == len(args.model) and 'scan' not in models, 'model names must differ and not be scan')
    chosen_cuts = {m: r['best_cut'] for m, r in json.loads(args.cut_from.read_text())['models'].items()} if args.cut_from else {}
    scans = pages_module.page_files(args.pages, chosen)
    out = make_output_dir(args.output_dir)
    (out/'pack'/'crops').mkdir(parents=True)
    per_cut = {m: defaultdict(list) for m in models}
    questions, rows, summary = defaultdict(list), [], {}
    pages_cache = {}
    for image_id in chosen:
        with Image.open(scans[image_id]) as opened:
            rgb = np.array(opened.convert('RGB'))
        grey = np.asarray(Image.fromarray(rgb).convert('L'))
        size = (grey.shape[1], grey.shape[0])
        ink, soft = grey < INK_BELOW, non_white(grey)
        revisions = labels[image_id]['revisions']
        regions = [polygon_mask(r['region'], size) for r in revisions]
        for name, directory in models.items():
            record, instances = pages_module.read_output(Path(directory), image_id)
            require(tuple(record['size']) == size, f'{image_id}: {name} read another size of this page')
            for cut in args.cuts:
                found, false, outside, ink_outside = question_reading(removal(instances, grey.shape, cut, soft), ink, regions)
                per_cut[name][cut].append({'image_id': image_id, 'found': found, 'kinds': [r['kind'] for r in revisions], 'false_places': false,
                                           'removed_outside': outside, 'ink_outside': ink_outside})
        pages_cache[image_id] = (rgb, soft, revisions)
    for name in models:
        readings = {str(cut): totals(rows_) for cut, rows_ in per_cut[name].items()}
        best = max(args.cuts, key=lambda c: (readings[str(c)]['f1'], -c))
        cut = chosen_cuts.get(name, best)
        require(cut in args.cuts, f'{name}: the cut {cut} is not among --cuts')
        summary[name] = {'best_cut': best, 'pack_cut': cut, 'at_pack_cut': readings[str(cut)], 'by_cut': readings}
        questions[name] = per_cut[name][cut]
    for image_id, (rgb, soft, revisions) in pages_cache.items():
        pictures = {'scan': rgb}
        for name, directory in models.items():
            _, instances = pages_module.read_output(Path(directory), image_id)
            pictures[name] = rgb.copy()
            pictures[name][removal(instances, soft.shape, summary[name]['pack_cut'], soft)] = 255
        for r in revisions:
            if r.get('unreadable') or not prepare(r.get('text', '')).replace('?', ''):
                continue
            xs, ys = [p[0] for p in r['region']], [p[1] for p in r['region']]
            box = (max(0, min(xs)-PAD), max(0, min(ys)-PAD), min(rgb.shape[1], max(xs)+PAD), min(rgb.shape[0], max(ys)+PAD))
            for name, picture in pictures.items():
                path = f'crops/{len(rows):05d}.png'
                Image.fromarray(picture).crop(box).save(out/'pack'/path)
                rows.append({'id': f'{name}|{image_id}#{r["id"]}', 'line_id': f'{image_id}#{r["id"]}', 'model': name, 'image_id': image_id,
                             'kind': r['kind'], 'size': [box[2]-box[0], box[3]-box[1]], 'boxes': [], 'images': {'erased': path}, 'text': r['text']})
    write_json(out/'questions.json', questions)
    write_json(out/'pack'/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    result = {'split': args.split, 'questions': len(chosen), 'pack_rows': len(rows), 'models': summary,
              'rule': {'found': FOUND, 'min_found': MIN_FOUND, 'edge': EDGE, 'false_place': FALSE_PLACE}}
    write_json(out/'summary.json', result)
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [*args.labels, args.selection,
                   *(Path(d)/'run_manifest.json' for d in models.values()), *([args.cut_from] if args.cut_from else [])]},
                   questions=len(chosen), pack_rows=len(rows), cuts={m: r['pack_cut'] for m, r in summary.items()})
    for name, r in summary.items():
        print(json.dumps({'model': name, 'cut': r['pack_cut'], **{k: round(v, 3) if isinstance(v, float) else v for k, v in r['at_pack_cut'].items()}}))
    return 0


def held(label, reading):
    """The share of a struck text's characters (its ? not counted) that a minimal edit alignment with the reading keeps."""
    x, y = prepare(label), prepare(reading)
    kept, _ = alignment(x, y)
    counted = [i for i, c in enumerate(x) if c != '?']
    return sum(i in kept for i in counted)/len(counted) if counted else None


def read(args):
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    shares, exact = defaultdict(list), defaultdict(list)
    for line in args.readings.read_text(encoding='utf-8').splitlines():
        reading = json.loads(line)
        row = rows.get(reading['id'])
        if row is None:
            continue
        share = held(row['text'], reading['text'])
        if share is not None:
            for kind in (row['kind'], 'all'):
                key = (row.get('model', ''), reading['condition'], kind)
                shares[key].append(share)
                # Legacy whole means all labelled characters survived, not exact transcription.
                # Partially unreadable labels have no exact-match denominator.
                if '?' not in prepare(row['text']):
                    exact[key].append(prepare(row['text']) == prepare(reading['text']))
    result = {f'{key[0]}/{key[1]}/{key[2]}': {'revisions': len(v), 'held': float(np.mean(v)),
              'whole': float(np.mean([s == 1 for s in v])), 'exact': float(np.mean(exact[key])) if exact[key] else None,
              'exact_revisions': len(exact[key])} for key, v in sorted(shares.items())}
    write_json(args.output, {'pack': str(args.pack), 'readings': str(args.readings), 'held': result})
    print(json.dumps(result, ensure_ascii=False))
    return 0


def unpacked(array, width):
    return np.unpackbits(array, axis=-1)[:, :width].astype(bool)


def pixels(args):
    models = dict(m.split('=', 1) for m in args.model)
    cuts = {m: r['pack_cut'] for m, r in json.loads(args.cut_from.read_text())['models'].items()} if args.cut_from else {}
    questions = sorted(p.stem for p in args.geometry.glob('*.json') if p.stem not in ('run_manifest', 'summary'))
    require(questions, f'{args.geometry}: no question')
    scans = pages_module.page_files(args.pages, questions)
    counts = {m: defaultdict(int) for m in models}
    for image_id in questions:
        record = json.loads((args.geometry/f'{image_id}.json').read_text())
        with Image.open(scans[image_id]) as opened:
            grey = np.asarray(opened.convert('L'))
        ink, soft = grey < INK_BELOW, non_white(grey)
        lines = [n for n, r in enumerate(record['revisions']) if r['kind'] == 'line']
        with np.load(args.geometry/f'{image_id}.npz') as arrays:
            strike = np.logical_or.reduce([unpacked(arrays[f'revision_{n}'], grey.shape[1]) for n in lines]) if lines else np.zeros(grey.shape, bool)
            domain = np.logical_or.reduce([unpacked(arrays[f'region_{n}'], grey.shape[1]) for n in lines]) if lines else np.zeros(grey.shape, bool)
        for name, directory in models.items():
            _, instances = pages_module.read_output(Path(directory), image_id)
            removed = removal(instances, grey.shape, cuts.get(name, args.cut), soft) & domain & ink
            c = counts[name]
            c['removed'] += int(removed.sum())
            c['strike'] += int((strike & ink).sum())
            c['hit'] += int((removed & strike).sum())
    result = {m: {**c, 'precision': c['hit']/max(1, c['removed']), 'recall': c['hit']/max(1, c['strike']), 'cut': cuts.get(m, args.cut)}
              for m, c in counts.items()}
    write_json(args.output, {'geometry': str(args.geometry), 'questions': len(questions), 'models': result})
    for m, r in result.items():
        print(json.dumps({'model': m, 'precision': round(r['precision'], 3), 'recall': round(r['recall'], 3), 'removed': r['removed'], 'strike': r['strike']}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('boxes', help="the model's removal against the boxes, and crops for reading")
    b.add_argument('--labels', type=Path, action='append', required=True)
    b.add_argument('--selection', type=Path, required=True)
    b.add_argument('--split', choices=('val', 'test', 'train'), required=True)
    b.add_argument('--model', action='append', required=True, help='NAME=DIR: a revrender.pages run of a model on these questions (repeatable)')
    b.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    b.add_argument('--cuts', type=float, nargs='+', default=list(CUTS))
    b.add_argument('--cut-from', type=Path, help="the validation's summary: each model's crops at the cut chosen there")
    b.add_argument('--all', action='store_true', help='also questions not marked done')
    b.add_argument('--output-dir', type=Path, required=True)
    r = sub.add_parser('read', help='what a reader read on the crops against the struck text')
    r.add_argument('--pack', type=Path, required=True)
    r.add_argument('--readings', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    x = sub.add_parser('pixels', help="what each model removes against the strokes a person drew, on ink")
    x.add_argument('--geometry', type=Path, required=True, help='data.page_labels output of the questions labelled stroke by stroke')
    x.add_argument('--model', action='append', required=True, help='NAME=DIR: a run of a model on these questions (repeatable)')
    x.add_argument('--pages', type=Path, action='append', required=True)
    x.add_argument('--cut', type=float, default=0., help='the score cut (unless --cut-from gives the model one)')
    x.add_argument('--cut-from', type=Path, help="the validation's summary: each model's cut chosen there")
    x.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    return {'boxes': boxes, 'read': read, 'pixels': pixels}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
