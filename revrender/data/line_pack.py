"""The human-labelled lines as images a VLM can be asked to read: the line as it is, with the revision erased, and with it marked in red.

For every line of ``data.prior_annotations`` that is not a test writer's -- a line whose revision a person described (``pair``: the struck
text, the text that replaced it) or judged to have none (``no_edit``, the controls) -- the exact catalogue crop is cut from the scan, read by
RevRender at the checkpoint's frozen point (paper 1's own setting: the crop letterboxed to a 96x768 strip), and the delivered masks are cut
to ink (``tighten.non_white``).  Three images and the places of the masks go to the pack:

  raw      the crop as scanned
  erased   the crop with the masked ink whited out
  overlay  the crop with the masked ink tinted red (one image: a second image would be read as more text)
  boxes    each mask's box, in thousandths of the crop's width and height, for a hint in words

    python -m revrender.data.line_pack --pairs pairs.jsonl --catalogue FILE --pages DIR [--pages DIR ...] --checkpoint best.pt
        --freeze freeze.json --output-dir OUT [--limit N]

OUT/pack.json lists the rows (the labels, the split, the images, the boxes) and OUT/assets holds the png files.  A line of a test writer is
never put in the pack.  The masks are the model's, not the annotators': a line the model finds nothing on has an erased and an overlay image
equal to the raw one, and a control the model marks is one where erasing does harm.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import infer, pages, tighten
from ..eval.freeze import frozen_point
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from . import lines as lines_module
from .gold import CANVAS, letterbox, unletterbox
from .prior_annotations import lines as read_jsonl

KINDS = ('pair', 'no_edit')
RED, ALPHA = (220, 0, 0), .6


def select(rows, limit=0):
    """The rows a pack is made of: ``pair`` and ``no_edit`` lines of writers who are not test writers, by line id; a pair must hold text."""
    chosen = [r for r in rows if r['kind'] in KINDS and r['split'] != 'test' and (r['kind'] == 'no_edit' or r.get('text_before') or r.get('text_after'))]
    chosen.sort(key=lambda r: r['line_id'])
    return chosen[:limit] if limit else chosen


def erase(rgb, mask):
    out = np.array(rgb)
    out[mask] = 255
    return Image.fromarray(out)


def tint(rgb, mask):
    out = np.array(rgb, np.float32)
    out[mask] = (1-ALPHA)*out[mask]+ALPHA*np.array(RED, np.float32)
    return Image.fromarray(out.astype(np.uint8))


def boxes_of(masks, size):
    """Each mask's box in thousandths of the crop (left, top, right, bottom), left to right."""
    width, height = size
    out = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if len(ys):
            out.append([round(1000*xs.min()/width), round(1000*ys.min()/height), round(1000*(xs.max()+1)/width), round(1000*(ys.max()+1)/height)])
    return sorted(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pairs', type=Path, required=True, help='pairs.jsonl of revrender.data.prior_annotations (with the split of each line)')
    parser.add_argument('--catalogue', type=Path, required=True, help='the line catalogue (exact_crops jsonl)')
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha')
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='only the first N lines (a smoke test)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = select(read_jsonl(args.pairs), args.limit)
    require(rows, 'no line to put in the pack')
    index = lines_module.index_by_line(lines_module.read_catalogue(args.catalogue))
    missing = [r['line_id'] for r in rows if r['line_id'] not in index]
    require(not missing, f'{len(missing)} line(s) not in the catalogue, e.g. {missing[:2]}')
    files = pages.page_files(args.pages, sorted({r['image_id'] for r in rows}))
    crops, strips = [], []
    for r in rows:
        with Image.open(files[r['image_id']]) as opened:
            box = lines_module.crop_box(index[r['line_id']]['box_1000'], *opened.size)
            rgb = opened.convert('RGB').crop(box)
        grey = np.asarray(rgb.convert('L'))
        crops.append((rgb, grey))
        strips.append(np.asarray(letterbox(Image.fromarray(grey), CANVAS[1], CANVAS[0], 255)))
    point = frozen_point(args.freeze, None)
    model, payload = infer.load_checkpoint(args.checkpoint, expected_sha=args.checkpoint_sha)
    delivered = infer.predict_lines(model, strips, point)
    out = make_output_dir(args.output_dir)
    (out/'assets').mkdir()
    packed = []
    for n, (r, (rgb, grey), found) in enumerate(zip(rows, crops, delivered)):
        width, height = rgb.size
        white = tighten.non_white(grey)
        masks = [unletterbox(mask, width, height) & white for mask, _ in found]
        masks = [m for m in masks if m.any()]
        union = np.logical_or.reduce(masks) if masks else np.zeros((height, width), bool)
        stem = f'{n:04d}'
        rgb.save(out/'assets'/f'{stem}_raw.png')
        erase(rgb, union).save(out/'assets'/f'{stem}_erased.png')
        tint(rgb, union).save(out/'assets'/f'{stem}_overlay.png')
        keep = ('line_id', 'image_id', 'kind', 'split', 'ink_kind', 'text_before', 'text_after', 'no_rewrite', 'before_unreadable')
        packed.append({**{k: r.get(k) for k in keep}, 'id': stem, 'size': [width, height], 'masked_pixels': int(union.sum()), 'boxes': boxes_of(masks, (width, height)),
                       'images': {k: f'assets/{stem}_{k}.png' for k in ('raw', 'erased', 'overlay')}})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'seed': 42, 'rows': packed})
    write_manifest(out, config=to_jsonable(args), inputs={str(args.pairs): digest(args.pairs), str(args.checkpoint): digest(args.checkpoint), str(args.freeze): digest(args.freeze)},
                   point={k: point[k] for k in ('mask_threshold', 'nms', 'score_threshold')}, checkpoint_step=payload.get('step'), rows=len(packed),
                   pairs=sum(r['kind'] == 'pair' for r in packed), controls=sum(r['kind'] == 'no_edit' for r in packed),
                   with_mask=sum(r['masked_pixels'] > 0 for r in packed))
    print(json.dumps({'rows': len(packed), 'pairs': sum(r['kind'] == 'pair' for r in packed), 'controls': sum(r['kind'] == 'no_edit' for r in packed),
                      'with_mask': sum(r['masked_pixels'] > 0 for r in packed)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
