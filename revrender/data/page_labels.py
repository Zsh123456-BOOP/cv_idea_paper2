"""The user's revision annotations of whole questions, read back: per stroke and per revision the pixels to remove, per revision the region
of writing it voids, and per question the scan with the revisions removed.

    python -m revrender.data.page_labels --export 涂改标注_….json [--export …] --selection selection.json --pages DIR [--pages DIR ...]
        --output-dir OUT [--all]

``--export`` is what the geometry version of the annotation page (git history, commit 1be20db) downloaded: the test questions labelled to
the pixel, a small check of what strike removal leaves.  Several exports are merged, a later file winning for
a question it also holds.  Only questions marked done are read unless ``--all``.  A struck line was drawn along the stroke and stopped where
the stroke crosses writing, so its pixels are the ink within half the stroke's width of the line (the width measured on the ink under it),
cut to ink with its soft edge (``tighten.non_white``); a blackout's pixels are the ink inside its polygon.  The removed scan sets those
pixels white: the writing a struck line crossed keeps the pixels the annotator skipped, so it is whole.  A revision's region is the polygon
of the writing it voids (computed from the ink on that page; the annotator may have moved it).

Per question OUT/<image_id>.json (strokes and revisions with their kinds and pixel counts), OUT/<image_id>.npz (``stroke_<n>``,
``revision_<n>``, ``region_<n>``: masks on the page, packed bits) and OUT/removed/<image_id>.png; OUT/summary.json the counts per split.
The scans are student work: OUT stays out of Git.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .. import pages as pages_module
from ..eval.ink import INK_BELOW
from ..tighten import non_white
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SCHEMA = 'paper2-revision-annotation-v1'
MIN_WIDTH = 2.             # a struck line is taken at least this wide (pixels)


def polygon_mask(points, size):
    canvas = Image.new('L', size, 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=255)
    return np.asarray(canvas) > 0


def line_mask(path, size, width):
    """A polyline ``width`` wide with square-cut ends: it stops where the annotator stopped it, never past."""
    canvas = Image.new('L', size, 0)
    ImageDraw.Draw(canvas).line([tuple(p) for p in path], fill=255, width=max(1, int(round(width))), joint='curve')
    return np.asarray(canvas) > 0


def stroke_width(path, ink, depth, size):
    """The width of the ink a struck line was drawn along: twice the median depth (``depth``: the ink's distance transform) of the ink
    under the line, at least ``MIN_WIDTH``."""
    on = line_mask(path, size, 1) & ink
    return max(MIN_WIDTH, 2*float(np.median(depth[on]))) if on.any() else MIN_WIDTH


def stroke_pixels(stroke, ink, soft, depth, size):
    """The pixels a stroke removes: a blackout's ink inside it (with the soft edge); a struck line's ink within half its width of the line,
    with the faint edge round that ink (never more of the writing it stopped at)."""
    if stroke['kind'] == 'blot':
        return polygon_mask(stroke['path'], size) & soft
    core = line_mask(stroke['path'], size, stroke_width(stroke['path'], ink, depth, size)+1) & ink
    if not core.any():
        return core
    return core | ndimage.binary_dilation(core, np.ones((3, 3), bool), iterations=2, mask=soft & ~ink)


def merge_exports(paths):
    """{image id: question state} from editor exports, a later file winning for a question it also holds."""
    pages = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        require(value.get('schema') == SCHEMA, f'{path}: not an export of the revision annotation page')
        pages.update(value['pages'])
    return pages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--export', type=Path, action='append', required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--all', action='store_true', help='also the questions not marked done')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    selection = {p['image_id']: p for p in json.loads(args.selection.read_text())['pages']}
    pages = merge_exports(args.export)
    unknown = [i for i in pages if i not in selection]
    require(not unknown, f'{len(unknown)} annotated question(s) are not in the selection, e.g. {unknown[:2]}')
    chosen = sorted(i for i, state in pages.items() if args.all or state.get('done'))
    require(chosen, 'no question to read (none is marked done; --all reads the rest)')
    scans = pages_module.page_files(args.pages, chosen)
    out = make_output_dir(args.output_dir)
    (out/'removed').mkdir()
    counts = defaultdict(Counter)
    for image_id in chosen:
        state, split = pages[image_id], selection[image_id]['split']
        with Image.open(scans[image_id]) as opened:
            rgb = np.array(opened.convert('RGB'))
        grey = np.asarray(Image.fromarray(rgb).convert('L'))
        size = (grey.shape[1], grey.shape[0])
        ink, soft = grey < INK_BELOW, non_white(grey)
        depth = ndimage.distance_transform_edt(ink)
        by_id = {s['id']: s for s in state['strokes']}
        stroke_masks = {s['id']: stroke_pixels(s, ink, soft, depth, size) for s in state['strokes']}
        arrays, revisions = {}, []
        for n, s in enumerate(state['strokes']):
            arrays[f'stroke_{n}'] = np.packbits(stroke_masks[s['id']], axis=-1)
        for n, g in enumerate(state['groups']):
            members = [by_id[i] for i in g['strokes'] if i in by_id]
            mask = np.logical_or.reduce([stroke_masks[m['id']] for m in members]) if members else np.zeros(grey.shape, bool)
            region = polygon_mask(g['region'], size) if g.get('region') else np.zeros(grey.shape, bool)
            kinds = {m['kind'] for m in members}
            kind = 'line' if kinds == {'line'} else 'blot' if kinds == {'blot'} else 'mixed'
            arrays[f'revision_{n}'], arrays[f'region_{n}'] = np.packbits(mask, axis=-1), np.packbits(region, axis=-1)
            revisions.append({'id': g['id'], 'strokes': g['strokes'], 'kind': kind, 'region': g.get('region'), 'region_auto': g.get('region_auto', True),
                              'pixels': int(mask.sum()), 'region_pixels': int(region.sum()), 'ink_in_region': int((ink & region).sum())})
            counts[split][f'revisions_{kind}'] += 1
            counts[split]['revisions_without_region'] += not g.get('region')
            counts[split]['regions_moved_by_hand'] += bool(g.get('region')) and not g.get('region_auto', True)
        removal = np.logical_or.reduce(list(stroke_masks.values())) if stroke_masks else np.zeros(grey.shape, bool)
        removed = rgb.copy()
        removed[removal] = 255
        Image.fromarray(removed).save(out/'removed'/f'{image_id}.png')
        np.savez_compressed(out/f'{image_id}.npz', **arrays)
        write_json(out/f'{image_id}.json', {'image_id': image_id, 'split': split, 'size': list(size),
                                             'strokes': [{**s, 'pixels': int(stroke_masks[s['id']].sum())} for s in state['strokes']], 'revisions': revisions})
        counts[split]['questions'] += 1
        counts[split]['strokes_line'] += sum(s['kind'] == 'line' for s in state['strokes'])
        counts[split]['strokes_blot'] += sum(s['kind'] == 'blot' for s in state['strokes'])
        counts[split]['strokes_without_ink'] += sum(not m.any() for m in stroke_masks.values())
        counts[split]['removed_pixels'] += int(removal.sum())
    summary = {split: dict(c) for split, c in sorted(counts.items())}
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [*args.export, args.selection]}, questions=len(chosen), summary=summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
