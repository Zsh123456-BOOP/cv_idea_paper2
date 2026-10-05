"""The struck pixels inside the boxes a person drew, read by a RevRender trained on box crops (``data.synthetic_set --crops``).

    python -m revrender.box_remover --checkpoint step_<n>.pt --point MASK NMS SCORE --labels EXPORT.json [--labels ...]
        --selection selection.json --split val --pages DIR [--pages DIR ...] --output-dir OUT

Every struck-line box of the labelled questions of ``--split`` (a blackout's box is left alone: what it covers is taken as unreadable)
is cropped with ``eval.revision_boxes.PAD`` round it, letterboxed onto the 96 x 768 strip the way the remover was trained, read at
the operating point (a low score cut keeps what a later reading cuts at its own) and its delivered masks put back on the page.  The
output is laid out like ``revrender.pages``'s (OUT/<image_id>.json and .npz, the boxes as its pieces, source ``box``), so
``eval.revision_boxes`` reads it like any model.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import infer
from . import pages as pages_module
from .data.gold import letterbox, unletterbox
from .eval.revision_boxes import PAD, read_labels
from .util import digest, make_output_dir, require, to_jsonable, write_manifest

STRIP_H, STRIP_W = 96, 768


def crop_of(region, size):
    """The crop (left, top, right, bottom) round a box drawn as a polygon, ``PAD`` wider on every side, inside the page."""
    xs, ys = [p[0] for p in region], [p[1] for p in region]
    return [max(0, int(min(xs))-PAD), max(0, int(min(ys))-PAD), min(size[0], int(max(xs))+PAD), min(size[1], int(max(ys))+PAD)]


def read_boxes(model, point, grey, boxes):
    """The instances the remover delivers in each box of a grey page: [{line, score, box, mask (on the box)}]."""
    strips = [np.asarray(letterbox(Image.fromarray(grey).crop(tuple(box)), STRIP_W, STRIP_H, 255)) for box in boxes]
    instances = []
    for k, (box, delivered) in enumerate(zip(boxes, infer.predict_lines(model, strips, point))):
        for mask, score in delivered:
            crop = unletterbox(mask, box[2]-box[0], box[3]-box[1])
            if crop.any():
                instances.append({'line': k, 'score': float(score), 'box': list(box), 'mask': crop})
    return instances


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--point', type=float, nargs=3, metavar=('MASK', 'NMS', 'SCORE'), required=True)
    parser.add_argument('--labels', type=Path, action='append', required=True, help="the box annotation page's exports")
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test', 'train'), required=True)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--all', action='store_true', help='also questions not marked done')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    selection = {p['image_id']: p for p in json.loads(args.selection.read_text())['pages']}
    labels = read_labels(args.labels)
    chosen = sorted(i for i, q in labels.items() if selection.get(i, {}).get('split') == args.split and (q.get('done') or args.all))
    require(chosen, f'no labelled {args.split} question')
    scans = pages_module.page_files(args.pages, chosen)
    out = make_output_dir(args.output_dir)
    point = dict(zip(('mask_threshold', 'nms', 'score_threshold'), args.point))
    model, payload = infer.load_checkpoint(args.checkpoint)
    totals = {'pages': 0, 'boxes': 0, 'instances': 0, 'seconds': 0.}
    for image_id in chosen:
        started = time.monotonic()
        with Image.open(scans[image_id]) as opened:
            grey = np.asarray(opened.convert('L'))
        size = (grey.shape[1], grey.shape[0])
        boxes = [crop_of(r['region'], size) for r in labels[image_id]['revisions'] if r['kind'] == 'line']
        instances = read_boxes(model, point, grey, boxes) if boxes else []
        pages_module.write_page(out, image_id, size, [{'box': box, 'source': 'box', 'ocr_type': None, 'ocr_text': ''} for box in boxes], instances)
        totals['pages'] += 1
        totals['boxes'] += len(boxes)
        totals['instances'] += len(instances)
        totals['seconds'] += time.monotonic()-started
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.checkpoint, *args.labels, args.selection]},
                   point=point, checkpoint_step=payload.get('step'), **totals)
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in totals.items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
