"""A model's revisions on whole questions read against the user's boxes (the annotation page's exports, ``paper2-revision-box-v1``).

    python -m revrender.eval.revision_boxes boxes --labels EXPORT.json [--labels ...] --selection selection.json --split val
        --pages-output OUT_OF_revrender.pages --pages DIR [--pages DIR ...] [--cuts .2 .3 ...] [--pack-cut C] --output-dir OUT
    python -m revrender.eval.revision_boxes read --pack OUT/pack --readings predictions.jsonl --output read.json

``boxes``: the model's removal on a question at a score cut is its instances at or above the cut, pasted on the page and cut to the
page's non-white pixels (``tighten.non_white``).  A revision is found when the removal takes at least ``FOUND`` of the ink in its box
(and ``MIN_FOUND`` pixels); the boxes cover every revision of a question, so a piece of removed ink of at least ``FALSE_PLACE`` pixels
outside every box (grown by ``EDGE``) is a place the model removed valid writing.  Per cut: revisions found (recall), places found
against false places (precision), F1, and the share of the ink outside the boxes removed; the cut with the best F1 is reported (on the
validation split this chooses it; on the test split it is only read).  OUT/summary.json, OUT/questions.json (per question at that cut)
and OUT/pack (``--pack-cut``, default that cut): a line pack (``recognize_lines``) with each revision whose struck text was written,
its box with a margin as scanned (``raw``) and with the removal whited out (``erased``).

``read``: what a reader read on those crops against the struck text: the share of the text's characters a minimal edit alignment of
the two keeps (``eval.coverage``; a ``?`` of the label is not counted), per condition, for struck lines and blackouts apart.
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
    scans = pages_module.page_files(args.pages, chosen)
    out = make_output_dir(args.output_dir)
    per_cut, kept = defaultdict(list), {}
    for image_id in chosen:
        with Image.open(scans[image_id]) as opened:
            rgb = np.array(opened.convert('RGB'))
        grey = np.asarray(Image.fromarray(rgb).convert('L'))
        size = (grey.shape[1], grey.shape[0])
        ink, soft = grey < INK_BELOW, non_white(grey)
        record, instances = pages_module.read_output(args.pages_output, image_id)
        require(tuple(record['size']) == size, f'{image_id}: the model read another size of this page')
        revisions = labels[image_id]['revisions']
        regions = [polygon_mask(r['region'], size) for r in revisions]
        for cut in args.cuts:
            removed = removal(instances, grey.shape, cut, soft)
            found, false, outside, ink_outside = question_reading(removed, ink, regions)
            per_cut[cut].append({'image_id': image_id, 'found': found, 'kinds': [r['kind'] for r in revisions], 'false_places': false,
                                 'removed_outside': outside, 'ink_outside': ink_outside})
        kept[image_id] = (rgb, soft, instances, revisions)
    readings = {str(cut): totals(rows) for cut, rows in per_cut.items()}
    best = max(args.cuts, key=lambda c: (readings[str(c)]['f1'], -c))
    pack_cut = args.pack_cut if args.pack_cut is not None else best
    write_json(out/'questions.json', per_cut[best])
    (out/'pack'/'crops').mkdir(parents=True)
    rows = []
    for image_id, (rgb, soft, instances, revisions) in kept.items():
        erased = rgb.copy()
        erased[removal(instances, soft.shape, pack_cut, soft)] = 255
        for r in revisions:
            if r.get('unreadable') or not prepare(r.get('text', '')).replace('?', ''):
                continue
            xs, ys = [p[0] for p in r['region']], [p[1] for p in r['region']]
            box = (max(0, min(xs)-PAD), max(0, min(ys)-PAD), min(rgb.shape[1], max(xs)+PAD), min(rgb.shape[0], max(ys)+PAD))
            images = {}
            for name, picture in (('raw', rgb), ('erased', erased)):
                images[name] = f'crops/{len(rows):05d}_{name}.png'
                Image.fromarray(picture).crop(box).save(out/'pack'/images[name])
            rows.append({'id': f'{image_id}#{r["id"]}', 'line_id': f'{image_id}#{r["id"]}', 'image_id': image_id, 'kind': r['kind'],
                         'size': [box[2]-box[0], box[3]-box[1]], 'boxes': [], 'images': images, 'text': r['text']})
    write_json(out/'pack'/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    summary = {'split': args.split, 'questions': len(chosen), 'best_cut': best, 'pack_cut': pack_cut, 'pack_rows': len(rows), 'by_cut': readings,
               'rule': {'found': FOUND, 'min_found': MIN_FOUND, 'edge': EDGE, 'false_place': FALSE_PLACE}}
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [*args.labels, args.selection, args.pages_output/'run_manifest.json']},
                   **{k: v for k, v in summary.items() if k != 'by_cut'}, best=readings[str(best)])
    print(json.dumps({'best_cut': best, **readings[str(best)]}))
    return 0


def held(label, reading):
    """The share of a struck text's characters (its ? not counted) that a minimal edit alignment with the reading keeps."""
    x, y = prepare(label), prepare(reading)
    kept, _ = alignment(x, y)
    counted = [i for i, c in enumerate(x) if c != '?']
    return sum(i in kept for i in counted)/len(counted) if counted else None


def read(args):
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    shares = defaultdict(list)
    for line in args.readings.read_text(encoding='utf-8').splitlines():
        reading = json.loads(line)
        row = rows.get(reading['id'])
        if row is None:
            continue
        share = held(row['text'], reading['text'])
        if share is not None:
            shares[(reading['condition'], row['kind'])].append(share)
            shares[(reading['condition'], 'all')].append(share)
    result = {f'{condition}/{kind}': {'revisions': len(v), 'held': float(np.mean(v)), 'whole': float(np.mean([s == 1 for s in v]))}
              for (condition, kind), v in sorted(shares.items())}
    write_json(args.output, {'pack': str(args.pack), 'readings': str(args.readings), 'held': result})
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('boxes', help="the model's removal against the boxes, and crops for reading")
    b.add_argument('--labels', type=Path, action='append', required=True)
    b.add_argument('--selection', type=Path, required=True)
    b.add_argument('--split', choices=('val', 'test', 'train'), required=True)
    b.add_argument('--pages-output', type=Path, required=True, help='revrender.pages run with the model on these questions')
    b.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    b.add_argument('--cuts', type=float, nargs='+', default=list(CUTS))
    b.add_argument('--pack-cut', type=float, help='the score cut of the crops (default: the best one)')
    b.add_argument('--all', action='store_true', help='also questions not marked done')
    b.add_argument('--output-dir', type=Path, required=True)
    r = sub.add_parser('read', help='what a reader read on the crops against the struck text')
    r.add_argument('--pack', type=Path, required=True)
    r.add_argument('--readings', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    return boxes(args) if args.command == 'boxes' else read(args)


if __name__ == '__main__':
    raise SystemExit(main())
