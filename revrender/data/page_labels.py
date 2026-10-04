"""The user's annotations of whole questions, read back: per line its polygon, final text, struck stretches and revision marks.

    python -m revrender.data.page_labels --export 标注_paper2-test_….json [--export …] --selection selection.json --pages DIR [--pages DIR ...]
        --output-dir OUT [--all]

``--export`` is what the page editor (``tools/annotate/page_editor.html``) downloads; several exports are merged, a later file winning
for a question it also holds.  Only questions marked done are read unless ``--all``.  Each line's text is read by the editor's own rules
(``[[ ]]`` around what was struck, ``?`` for a character that cannot be read): the final text, the struck stretches and whether each is
legible.  Each revision mark goes to the line that holds its centre (the editor's rule); it is rasterised on the page -- a box as it is, a
struck line as a polyline ``STROKE_PX`` wide -- and cut to ink (``tighten.non_white``), the shape paper 1's masks have.

OUT/labels.jsonl has a row per line (image id, split, line number in reading order, polygon, text as typed, final text, struck stretches,
flags, the indexes of its marks); OUT/marks/<image_id>.npz holds each mark's mask on the page (packed bits, ``mask_<n>``) and
OUT/marks/<image_id>.json the marks (type, shape, line); OUT/summary.json the counts per split.  A struck stretch and a mark of one line
are paired left to right; a line where their numbers differ is counted (``unpaired``), never fixed.  The text is student work: OUT stays
out of Git.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import pages as pages_module
from ..tighten import non_white
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

SCHEMA = 'paper2-page-annotation-v1'
STROKE_PX = 5              # a struck line drawn by the annotator is rasterised this wide, then cut to ink
OPEN, CLOSE = '[[', ']]'


def parse_line(text):
    """(final text, [{text, legible}], error or None): the Python twin of ``page_rules.js``'s ``parseLine`` (tests/test_page_editor.py
    checks that the two agree)."""
    final, struck, rest, error = '', [], text or '', None
    while rest:
        opening, closing = rest.find(OPEN), rest.find(CLOSE)
        if closing != -1 and (opening == -1 or closing < opening):
            return final+rest, struck, '多了 ]]'
        if opening == -1:
            return final+rest, struck, error
        final += rest[:opening]
        after = rest[opening+len(OPEN):]
        end, nested = after.find(CLOSE), after.find(OPEN)
        if end == -1:
            return final+rest[opening:], struck, '[[ 没有配对的 ]]'
        if nested != -1 and nested < end:
            return final+rest[opening:], struck, '[[ 里面又有 [['
        inside = after[:end]
        plain = re.sub(r'\s+', '', inside)
        if not plain:
            error = error or '空的 [[ ]]'
        struck.append({'text': inside, 'legible': 'no' if re.fullmatch(r'\?+', plain) else 'partly' if '?' in plain else 'yes'})
        rest = after[end+len(CLOSE):]
    return final, struck, error


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for (xi, yi), (xj, yj) in zip(polygon, polygon[-1:]+polygon[:-1]):
        if (yi > y) != (yj > y) and x < (xj-xi)*(y-yi)/(yj-yi)+xi:
            inside = not inside
    return inside


def mark_points(mark):
    if mark['shape'] == 'box':
        return [[mark['x'], mark['y']], [mark['x']+mark['w'], mark['y']], [mark['x']+mark['w'], mark['y']+mark['h']], [mark['x'], mark['y']+mark['h']]]
    return mark['path']


def centre(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2)


def assign(lines, marks):
    """([mark indexes per line, left to right], [mark indexes in no line]): the editor's ``assignMarks``."""
    per_line, outside = [[] for _ in lines], []
    for m, mark in enumerate(marks):
        c = centre(mark_points(mark))
        holders = [k for k, line in enumerate(lines) if not line.get('not_writing') and point_in_polygon(c, line['poly'])]
        if not holders:
            outside.append(m)
            continue
        per_line[min(holders, key=lambda k: np.hypot(c[0]-centre(lines[k]['poly'])[0], c[1]-centre(lines[k]['poly'])[1]))].append(m)
    for indexes in per_line:
        indexes.sort(key=lambda m: centre(mark_points(marks[m]))[0])
    return per_line, outside


def rasterise(mark, size, ink):
    """A mark's mask on the page (``size`` = width, height), cut to ``ink``."""
    canvas = Image.new('L', size, 0)
    draw = ImageDraw.Draw(canvas)
    if mark['shape'] == 'box':
        draw.rectangle([mark['x'], mark['y'], mark['x']+mark['w']-1, mark['y']+mark['h']-1], fill=255)
    else:
        draw.line([tuple(p) for p in mark['path']], fill=255, width=STROKE_PX, joint='curve')
        for x, y in mark['path']:
            draw.ellipse([x-STROKE_PX/2, y-STROKE_PX/2, x+STROKE_PX/2, y+STROKE_PX/2], fill=255)
    return (np.asarray(canvas) > 0) & ink


def merge_exports(paths):
    """{image id: page state} from editor exports, a later file winning for a question it also holds."""
    pages = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        require(value.get('schema') == SCHEMA, f'{path}: not an export of the page editor')
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
    (out/'marks').mkdir()
    counts = defaultdict(Counter)
    with (out/'labels.jsonl').open('x', encoding='utf-8') as stream:
        for image_id in chosen:
            state, split = pages[image_id], selection[image_id]['split']
            with Image.open(scans[image_id]) as opened:
                grey = np.asarray(opened.convert('L'))
            ink = non_white(grey)
            per_line, outside = assign(state['lines'], state['marks'])
            masks = [rasterise(mark, (grey.shape[1], grey.shape[0]), ink) for mark in state['marks']]
            for k, line in enumerate(state['lines']):
                final, struck, error = parse_line(line.get('text', ''))
                writing = not line.get('not_writing') and not line.get('unreadable')
                stream.write(json.dumps({'image_id': image_id, 'split': split, 'line': k, 'poly': line['poly'], 'source': line.get('source'),
                                         'catalogue_line_id': line.get('line_id'), 'text': line.get('text', ''), 'final': final, 'struck': struck, 'error': error,
                                         'not_writing': bool(line.get('not_writing')), 'unreadable': bool(line.get('unreadable')), 'marks': per_line[k],
                                         'paired': writing and not error and len(struck) == len(per_line[k])}, ensure_ascii=False)+'\n')
                if writing:
                    counts[split]['lines'] += 1
                    counts[split]['revised_lines'] += bool(struck or per_line[k])
                    counts[split]['struck_stretches'] += len(struck)
                    counts[split]['unpaired_lines'] += bool(error or len(struck) != len(per_line[k]))
                    for s in struck:
                        counts[split][f'legible_{s["legible"]}'] += 1
                counts[split]['not_writing'] += bool(line.get('not_writing'))
                counts[split]['unreadable_lines'] += bool(line.get('unreadable'))
            counts[split]['questions'] += 1
            counts[split]['marks'] += len(state['marks'])
            counts[split]['marks_outside_lines'] += len(outside)
            counts[split]['empty_marks'] += sum(not m.any() for m in masks)
            np.savez_compressed(out/'marks'/f'{image_id}.npz', **{f'mask_{n}': np.packbits(m, axis=-1) for n, m in enumerate(masks)})
            line_of = {m: k for k, indexes in enumerate(per_line) for m in indexes}
            write_json(out/'marks'/f'{image_id}.json', {'image_id': image_id, 'size': [grey.shape[1], grey.shape[0]],
                                                         'marks': [{**mark, 'line': line_of.get(n)} for n, mark in enumerate(state['marks'])]})
    summary = {split: dict(c) for split, c in sorted(counts.items())}
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [*args.export, args.selection]}, questions=len(chosen), summary=summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
