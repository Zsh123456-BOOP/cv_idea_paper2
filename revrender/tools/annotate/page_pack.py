"""Fold the questions the user annotates into one self-contained page: the scans and their revision strokes, prefilled.

    python -m revrender.tools.annotate.page_pack --selection selection.json --pages DIR [--pages DIR ...] --revrender PAGES_OUT
        [--which pilot|test|val|train|all] [--first N] [--previous EXPORT.json ...] --output-dir OUT

``--selection`` is the output of ``revrender.data.annotation_set``, ``--revrender`` the output of ``revrender.pages`` on those questions.
Each question starts with one stroke per RevRender instance, cut to its ink, in a revision of its own: a thin, long instance as a struck
line (a polyline through its middle), any other as a blackout (the convex hull of its ink, simplified to a few corners).  The page works
out each revision's region from the ink when it opens (``page_rules.js``).  ``--previous`` starts the questions it holds from what the
annotator already did instead: an export of this page (taken as it is) or of the earlier line-and-text page (its marks become strokes,
each a revision of its own; its texts are left out).

The editor (``page_editor.html``) and its rules (``page_rules.js``) are inlined with the scans, so OUT/标注.html opens from anywhere
without a server.  The pack's id depends only on ``--which``: a rebuilt pack keeps what the annotator already did in the browser.  The
output is student handwriting: it belongs under work/, never in Git.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ... import pages as pages_module
from ...eval.ink import INK_BELOW
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

HERE = Path(__file__).parent
EDITOR, RULES = HERE/'page_editor.html', HERE/'page_rules.js'
SCHEMA, OLD_SCHEMA = 'paper2-revision-annotation-v1', 'paper2-page-annotation-v1'
THIN = .45                 # an instance whose typical height is under this share of its piece's height ...
LONG = 2.5                 # ... and that is this many times longer than high is a struck line
CORNERS = 1.5              # a blackout's hull keeps the corners that stand out of it by more than this (pixels)


def slot(text, name, payload):
    """Replace the <!--NAME-->...<!--/NAME--> block of the editor with an inline script."""
    begin, end = f'<!--{name}-->', f'<!--/{name}-->'
    require(text.count(begin) == 1 and text.count(end) == 1, f'the editor has no single {name} slot')
    head, rest = text.split(begin)
    return head+'<script>'+payload.replace('</', '<\\/')+'</script>'+rest.split(end)[1]


def hull(points):
    """The convex hull of integer points (x, y), counter-clockwise, by the monotone chain."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 2:
        return [list(p) for p in points]
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1]+upper[:-1]]


def simplify(polygon, tolerance=CORNERS):
    """A closed polygon without the corners that stand out of the line between their neighbours by less than ``tolerance``."""
    pts = [list(p) for p in polygon]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i-1], pts[i], pts[(i+1) % len(pts)]
            length = np.hypot(c[0]-a[0], c[1]-a[1])
            height = abs((c[0]-a[0])*(a[1]-b[1])-(a[0]-b[0])*(c[1]-a[1]))/length if length else 0.
            if height < tolerance:
                del pts[i]
                changed = True
                break
    return pts


def stroke_of(mask, box, line_height):
    """(kind, path) of one instance (its mask on ``box``, left/top/right/bottom on the page, cut to ink by the caller), or None."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    left, top = box[0], box[1]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
    typical = float(np.median([int(ys[xs == c].max()-ys[xs == c].min())+1 for c in np.unique(xs)]))
    if typical <= THIN*line_height and x1-x0 >= LONG*typical:
        steps = int(np.clip(round((x1-x0)/max(1., line_height)), 1, 7))
        edges = np.linspace(x0, x1, steps+1)
        path = []
        for a, b in zip(edges[:-1], edges[1:]):
            inside = (xs >= a) & (xs < b) if b < x1 else (xs >= a)
            if inside.any():
                path.append([int(round(left+(a+b)/2)), int(round(top+np.median(ys[inside])))])
        if len(path) == 1:
            path.append(list(path[0]))
        path[0][0], path[-1][0] = left+x0, left+x1-1                   # the line runs from the first column of the mark to the last
        return 'line', path
    corners = [[left+x, top+y] for x, y in hull([[x+dx, y+dy] for x, y in zip(xs.tolist(), ys.tolist()) for dx in (0, 1) for dy in (0, 1)])]
    return 'blot', simplify(corners)


def prefill(scan, record, instances):
    """{strokes, groups, next} of a question from RevRender's instances: one stroke and one revision per instance."""
    with Image.open(scan) as opened:
        grey = np.asarray(opened.convert('L'))
    require(grey.shape == tuple(record['size'][::-1]), f'{record["image_id"]}: the scan is not the size RevRender read')
    state = {'strokes': [], 'groups': [], 'next': 0}
    for instance in instances:
        l, t, r, b = instance['box']
        ink = instance['mask'] & (grey[t:b, l:r] < INK_BELOW)
        made = stroke_of(ink if ink.any() else instance['mask'], instance['box'], b-t)
        if made:
            add(state, *made)
    return state


def add(state, kind, path):
    """A stroke in a revision of its own (the editor's ``addStroke``)."""
    state['next'] += 1
    stroke = {'id': f's{state["next"]}', 'kind': kind, 'path': [[int(round(x)), int(round(y))] for x, y in path]}
    state['next'] += 1
    state['strokes'].append(stroke)
    state['groups'].append({'id': f'g{state["next"]}', 'strokes': [stroke['id']], 'region': None, 'region_auto': True})


def from_previous(page):
    """A question's state from an earlier export: this page's as it is; the line-and-text page's marks as strokes."""
    if 'strokes' in page:
        return {k: page[k] for k in ('strokes', 'groups', 'next') if k in page}
    state = {'strokes': [], 'groups': [], 'next': 0}
    for mark in page.get('marks', []):
        if mark.get('shape') == 'box':
            x, y, w, h = mark['x'], mark['y'], mark['w'], mark['h']
            add(state, 'blot', [[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        elif len(mark.get('path', [])) >= 2:
            add(state, 'line', mark['path'])
    return state


def read_previous(paths):
    pages = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        require(value.get('schema') in (SCHEMA, OLD_SCHEMA), f'{path}: not an export of an annotation page')
        pages.update(value['pages'])
    return pages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--revrender', type=Path, required=True, help='the output of revrender.pages on the selected questions')
    parser.add_argument('--which', choices=('pilot', 'test', 'val', 'train', 'all'), default='pilot')
    parser.add_argument('--first', type=int, default=0, help='only the first N questions of --which (in the order of the selection)')
    parser.add_argument('--previous', type=Path, action='append', default=[], help='an earlier export whose questions start from what was done')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    selection = json.loads(args.selection.read_text())
    entries = [p for p in selection['pages'] if args.which == 'all' or (p['pilot'] if args.which == 'pilot' else p['split'] == args.which)]
    entries = entries[:args.first or None]
    require(entries, 'no question to pack')
    previous = read_previous(args.previous)
    scans = pages_module.page_files(args.pages, [p['image_id'] for p in entries])
    records = []
    for entry in entries:
        record, instances = pages_module.read_output(args.revrender, entry['image_id'])
        start = from_previous(previous[entry['image_id']]) if entry['image_id'] in previous else prefill(scans[entry['image_id']], record, instances)
        width, height = record['size']
        records.append({'id': entry['image_id'], 'question': entry['question'], 'pilot': entry['pilot'], 'width': width, 'height': height,
                        'image': 'data:image/jpeg;base64,'+base64.b64encode(scans[entry['image_id']].read_bytes()).decode(),
                        'from_previous': entry['image_id'] in previous, **start})
    pack = {'id': f'paper2-revisions-{args.which}', 'schema': 'paper2-revision-annotation-pack-v1', 'pages': records}
    html = slot(slot(EDITOR.read_text(encoding='utf-8'), 'RULES', RULES.read_text(encoding='utf-8')), 'PACK', 'window.PACK = '+json.dumps(pack, ensure_ascii=False)+';')
    out = make_output_dir(args.output_dir)
    (out/'标注.html').write_text(html, encoding='utf-8')
    summary = {'questions': len(records), 'strokes': sum(len(r['strokes']) for r in records), 'revisions': sum(len(r['groups']) for r in records),
               'struck_lines': sum(s['kind'] == 'line' for r in records for s in r['strokes']), 'from_previous': sum(r['from_previous'] for r in records),
               'bytes': (out/'标注.html').stat().st_size}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.selection, EDITOR, RULES, *args.previous]}, **summary)
    write_json(out/'questions.json', [{'id': r['id'], 'question': r['question'], 'pilot': r['pilot']} for r in records])
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
