"""Real revision marks cut out of paper 1's gold: what ``data.paste_strikes`` draws its synthetic revisions with besides its own strokes.

    python -m revrender.data.strike_bank --gold GOLD_PAGES --pages DIR [--pages DIR ...] --mapping index_manifest.json
        --same-student pairs.json --output-dir OUT

``--gold`` is paper 1's gold put back on its pages and cut to ink (``data.page_gold``, then ``tighten``).  Every instance of a writer of
paper 1's training split is cut from its scan -- never one of a writer whose other id (``--same-student``) is in another split -- as the
grey of the scan inside the instance's mask (paper outside it) on the mask's box, with the grey of its own ink (``ink_level``: the median
of its darkest third), so it can be drawn in another writer's ink.  A thin, long instance is a struck line, any other a blackout (paper
1's overwrite boxes, under 1% of its marks, fall among the blackouts).  ``line_height`` is the height of the annotated line the instance
lies on: a piece is scaled to a host line by the ratio of the two lines' heights.

OUT/bank.npz holds ``piece_<n>`` (uint8 grey); OUT/pieces.json lists the pieces (id, image_id, writer, kind, size, line_height, ink_level,
pixels); OUT/sheet_line.png and OUT/sheet_blot.png show a sample of each kind.  The pieces are student work: OUT stays out of Git.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import pages as pages_module
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .prior_annotations import twins, writer_splits

THIN = .45                 # a mark whose typical height is under this share of its line's height ...
LONG = 2.5                 # ... and that is this many times longer than high is a struck line
KINDS = ('line', 'blot')
SHEET = 48                 # pieces on a sheet


def kind_of(mask, line_height):
    """'line' for a thin, long mark (a bool mask on its line), else 'blot'."""
    ys, xs = np.nonzero(mask)
    columns = np.unique(xs)
    typical = float(np.median([int(ys[xs == c].max()-ys[xs == c].min())+1 for c in columns]))
    return 'line' if typical <= THIN*line_height and xs.max()-xs.min()+1 >= LONG*typical else 'blot'


def training_writers(mapping, same_student):
    """The writers whose marks the bank takes: paper 1's training split, without a student whose other id is in another split."""
    splits, other = writer_splits(mapping, same_student), twins(same_student)
    return {w for w, s in splits.items() if s == 'train' and splits.get(other.get(w), 'train') == 'train'}


def cut(grey, instance):
    """(grey piece on the box of the instance's mask, paper outside the mask, its pixels), or None for an empty mask."""
    l, t, _, _ = instance['box']
    ys, xs = np.nonzero(instance['mask'])
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    inside = instance['mask'][y0:y1, x0:x1]
    piece = np.where(inside, grey[t+y0:t+y1, l+x0:l+x1], 255).astype(np.uint8)
    return piece, inside


def ink_level(piece, inside):
    """The grey of a piece's own ink: the median of its darkest third."""
    values = np.sort(piece[inside])
    return int(np.median(values[:max(1, len(values)//3)]))


def sheet(bank, pieces, path, rng):
    """A sample of pieces, each on a white tile with its kind and size, to look at."""
    chosen = rng.sample(pieces, min(SHEET, len(pieces)))
    tile_w, tile_h, cols = 240, 90, 6
    out = Image.new('L', (cols*tile_w, ((len(chosen)+cols-1)//cols)*tile_h), 255)
    draw = ImageDraw.Draw(out)
    for k, p in enumerate(chosen):
        image = Image.fromarray(bank[p['id']])
        image.thumbnail((tile_w-10, tile_h-20))
        x, y = (k % cols)*tile_w, (k//cols)*tile_h
        out.paste(image, (x+5, y+16))
        draw.text((x+5, y+2), f'{p["kind"]} {p["size"][0]}x{p["size"][1]} h{p["line_height"]}', fill=0)
    out.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--gold', type=Path, required=True, help="paper 1's gold on its pages, cut to ink")
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--mapping', type=Path, required=True, help="paper 1's split mapping (index_manifest.json)")
    parser.add_argument('--same-student', type=Path, required=True, help='json of students with two writer ids')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    allowed = training_writers(args.mapping, args.same_student)
    ids = sorted(p.stem for p in args.gold.glob('*.json') if p.stem not in ('run_manifest', 'summary') and p.stem.split('_')[0] in allowed)
    require(ids, 'no gold page of a training writer')
    scans = pages_module.page_files(args.pages, ids)
    bank, pieces = {}, []
    for image_id in ids:
        record, instances = pages_module.read_output(args.gold, image_id)
        if not instances:
            continue
        with Image.open(scans[image_id]) as opened:
            grey = np.asarray(opened.convert('L'))
        require(grey.shape == tuple(record['size'][::-1]), f'{image_id}: the scan is not the size of the gold')
        for instance in instances:
            made = cut(grey, instance)
            if made is None:
                continue
            piece, inside = made
            line = record['lines'][instance['line']]['box']
            n = f'piece_{len(pieces)}'
            bank[n] = piece
            pieces.append({'id': n, 'image_id': image_id, 'writer': image_id.split('_')[0], 'kind': kind_of(instance['mask'], line[3]-line[1]),
                           'size': [int(piece.shape[1]), int(piece.shape[0])], 'line_height': int(line[3]-line[1]),
                           'ink_level': ink_level(piece, inside), 'pixels': int(inside.sum())})
    require(pieces, 'no piece')
    out = make_output_dir(args.output_dir)
    np.savez_compressed(out/'bank.npz', **bank)
    write_json(out/'pieces.json', pieces)
    rng = random.Random(SEED)
    for kind in KINDS:
        sheet(bank, [p for p in pieces if p['kind'] == kind], out/f'sheet_{kind}.png', rng)
    summary = {'pages': len(ids), 'pieces': len(pieces), 'writers': len({p['writer'] for p in pieces}),
               'by_kind': {k: sum(p['kind'] == k for p in pieces) for k in KINDS}}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.mapping, args.same_student, args.gold/'run_manifest.json']}, **summary)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
