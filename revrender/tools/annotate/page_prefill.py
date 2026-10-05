"""Where the questions the user annotates start: per question its revisions -- the region of the writing each voids and whether it is struck
through or blacked out -- and a crop of every revision for a VLM to read what is struck there.

    python -m revrender.tools.annotate.page_prefill --selection selection.json --pages DIR [--pages DIR ...] --revrender PAGES_OUT
        [--which pilot|test|val|train|all] [--previous EXPORT.json ...] --output-dir OUT

A question starts from RevRender's instances on it (``--revrender``, the output of ``revrender.pages``): an instance's ink, grown through
the ink it touches within a window round it, is the region of the writing it voids; a thin, long instance is a struck line and any other a
blackout; regions that overlap are one revision.  An instance on a piece of leftover ink (not an OCR line) that is taller than ``TALL``
text lines is left out: it lies on the lines of a figure.  A question an earlier export of the geometry page holds (``--previous``: strokes grouped
into revisions, each with its region) starts from those revisions instead, their regions as the annotator left them.  Revisions are in
reading order (rows from the top, left to right in a row).

OUT/prefill.json: the questions (id, question, split, pilot, size, revisions).  OUT/pack.json and OUT/crops/: a line pack
(``recognize_lines``) with a row per revision, the region and a margin round it, for the ``raw_struck`` reading the page starts each text
from.  The crops are student work: OUT stays under work/, never in Git.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ... import pages as pages_module
from ...data.page_labels import SCHEMA as GEOMETRY_SCHEMA, line_mask, polygon_mask
from ...eval.ink import INK_BELOW
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

THIN = .45                 # an instance whose typical height is under this share of its piece's height ...
LONG = 2.5                 # ... and that is this many times longer than high is a struck line
WINDOW_X, WINDOW_Y = 14, 18    # the writing a revision voids is looked for this far round its ink (pixels)
MARGIN = 3                 # a region is this much larger than the ink it holds
REACH = 2.5                # ink this close to a drawn struck line is the stroke's
OVERLAP = .3               # two regions are one revision when they share this much of the smaller one
CROP = (30, 12)            # a revision's crop is its region and this margin (x, y)
TALL = 1.5                 # an instance on leftover ink this many OCR lines high is on a figure


def kind_of(mask, line_height):
    """'line' for a thin, long instance (its mask on its piece), else 'blot'."""
    ys, xs = np.nonzero(mask)
    columns = np.unique(xs)
    typical = float(np.median([int(ys[xs == c].max()-ys[xs == c].min())+1 for c in columns]))
    return 'line' if typical <= THIN*line_height and len(columns) and xs.max()-xs.min()+1 >= LONG*typical else 'blot'


def region_of(seeds, ink):
    """The box [left, top, right, bottom] of the writing a revision voids: its ``seeds`` and the ink connected to them within a window round
    them, with a margin.  None without seeds."""
    ys, xs = np.nonzero(seeds)
    if not len(ys):
        return None
    height, width = ink.shape
    x0, x1 = max(0, xs.min()-WINDOW_X), min(width, xs.max()+1+WINDOW_X)
    y0, y1 = max(0, ys.min()-WINDOW_Y), min(height, ys.max()+1+WINDOW_Y)
    window, seeded = ink[y0:y1, x0:x1], seeds[y0:y1, x0:x1]
    labels, _ = ndimage.label(window, np.ones((3, 3), bool))
    touched = np.unique(labels[seeded & window])
    ry, rx = np.nonzero(np.isin(labels, touched[touched > 0]) | seeded)
    return [int(max(0, x0+rx.min()-MARGIN)), int(max(0, y0+ry.min()-MARGIN)), int(min(width, x0+rx.max()+1+MARGIN)), int(min(height, y0+ry.max()+1+MARGIN))]


def corners(box):
    l, t, r, b = box
    return [[l, t], [r, t], [r, b], [l, b]]


def box_of(polygon):
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def area(box):
    return max(0, box[2]-box[0])*max(0, box[3]-box[1])


def merge(found):
    """(kind, box, pixels) pieces with overlapping boxes merged into one: the union box, the kind of the piece with the most pixels."""
    found = [list(f) for f in found]
    merged = True
    while merged:
        merged = False
        for i in range(len(found)):
            for j in range(i+1, len(found)):
                a, b = found[i][1], found[j][1]
                shared = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
                if shared >= OVERLAP*min(area(a), area(b)):
                    kind = found[i][0] if found[i][2] >= found[j][2] else found[j][0]
                    found[i] = [kind, [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], found[i][2]+found[j][2]]
                    del found[j]
                    merged = True
                    break
            if merged:
                break
    return found


def reading_order(revisions):
    """Revisions in rows from the top (a revision joins a row when its middle lies within the row's first region), left to right in a row."""
    rows = []
    for r in sorted(revisions, key=lambda r: box_of(r['region'])[1]):
        l, t, _, b = box_of(r['region'])
        for row in rows:
            _, rt, _, rb = box_of(row[0]['region'])
            if rt <= (t+b)/2 <= rb:
                row.append(r)
                break
        else:
            rows.append([r])
    return [r for row in rows for r in sorted(row, key=lambda r: box_of(r['region'])[0])]


def numbered(revisions):
    """Revisions in reading order with ids r1, r2, ... and the state's next id."""
    out = [{**r, 'id': f'r{n}'} for n, r in enumerate(reading_order(revisions), 1)]
    return out, len(out)


def from_revrender(grey, record, instances):
    """Revisions from RevRender's instances on a page (``grey``, its grey scan; ``record``, RevRender's record of the page)."""
    ink = grey < INK_BELOW
    heights = [p['box'][3]-p['box'][1] for p in record['lines'] if p['source'] != 'ink']
    line_height = float(np.median(heights)) if heights else float('inf')
    found = []
    for instance in instances:
        l, t, r, b = instance['box']
        rows = np.nonzero(instance['mask'].any(1))[0]
        if record['lines'][instance['line']]['source'] == 'ink' and rows[-1]-rows[0]+1 > TALL*line_height:
            continue
        mask = np.zeros(grey.shape, bool)
        mask[t:b, l:r] = instance['mask']
        seeds = mask & ink if (mask & ink).any() else mask
        box = region_of(seeds, ink)
        if box:
            found.append((kind_of(instance['mask'], b-t), box, int(seeds.sum())))
    return [{'kind': kind, 'region': corners(box)} for kind, box, _ in merge(found)]


def from_geometry(state, grey):
    """Revisions from a question of the geometry page's export: each group's region (computed from its strokes when it has none) and kind
    (a blackout when every stroke is one)."""
    ink, size = grey < INK_BELOW, (grey.shape[1], grey.shape[0])
    strokes = {s['id']: s for s in state['strokes']}
    revisions = []
    for g in state['groups']:
        members = [strokes[i] for i in g['strokes'] if i in strokes]
        if not members:
            continue
        region = g.get('region')
        if not region:
            seeds = np.logical_or.reduce([polygon_mask(m['path'], size) if m['kind'] == 'blot' else line_mask(m['path'], size, 2*REACH+1) for m in members]) & ink
            box = region_of(seeds, ink)
            if not box:
                continue
            region = corners(box)
        revisions.append({'kind': 'blot' if all(m['kind'] == 'blot' for m in members) else 'line', 'region': [[int(x), int(y)] for x, y in region]})
    return revisions


def read_previous(paths):
    pages = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        require(value.get('schema') == GEOMETRY_SCHEMA, f'{path}: not an export of the geometry page')
        pages.update(value['pages'])
    return pages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--revrender', type=Path, required=True, help='the output of revrender.pages on the selected questions')
    parser.add_argument('--which', choices=('pilot', 'test', 'val', 'train', 'all'), default='test')
    parser.add_argument('--previous', type=Path, action='append', default=[], help='an export of the geometry page whose questions start from it')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    selection = json.loads(args.selection.read_text())
    entries = [p for p in selection['pages'] if args.which == 'all' or (p['pilot'] if args.which == 'pilot' else p['split'] == args.which)]
    require(entries, 'no question to prefill')
    previous = read_previous(args.previous)
    scans = pages_module.page_files(args.pages, [p['image_id'] for p in entries])
    out = make_output_dir(args.output_dir)
    (out/'crops').mkdir()
    questions, rows = [], []
    for entry in entries:
        image_id = entry['image_id']
        with Image.open(scans[image_id]) as opened:
            rgb = opened.convert('RGB')
        grey = np.asarray(rgb.convert('L'))
        if image_id in previous:
            revisions, source = from_geometry(previous[image_id], grey), 'previous'
        else:
            record, instances = pages_module.read_output(args.revrender, image_id)
            require(grey.shape == tuple(record['size'][::-1]), f'{image_id}: the scan is not the size RevRender read')
            revisions, source = from_revrender(grey, record, instances), 'revrender'
        revisions, last = numbered(revisions)
        for r in revisions:
            l, t, rr, b = box_of(r['region'])
            crop = (max(0, l-CROP[0]), max(0, t-CROP[1]), min(rgb.width, rr+CROP[0]), min(rgb.height, b+CROP[1]))
            name = f'crops/{len(rows):05d}.png'
            rgb.crop(crop).save(out/name)
            w, h = crop[2]-crop[0], crop[3]-crop[1]
            rows.append({'id': f'{image_id}#{r["id"]}', 'line_id': f'{image_id}#{r["id"]}', 'image_id': image_id, 'kind': r['kind'], 'split': entry['split'],
                         'size': [w, h], 'images': {'raw': name},
                         'boxes': [[round(1000*(l-crop[0])/w), round(1000*(t-crop[1])/h), round(1000*(rr-crop[0])/w), round(1000*(b-crop[1])/h)]]})
        questions.append({'id': image_id, 'question': entry['question'], 'split': entry['split'], 'pilot': entry['pilot'], 'width': rgb.width,
                          'height': rgb.height, 'source': source, 'revisions': revisions, 'next': last})
    write_json(out/'prefill.json', {'which': args.which, 'questions': questions})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    summary = {'questions': len(questions), 'revisions': len(rows), 'blots': sum(r['kind'] == 'blot' for r in rows),
               'from_previous': sum(q['source'] == 'previous' for q in questions)}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.selection, *args.previous]}, **summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
