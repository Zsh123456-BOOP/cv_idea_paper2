"""Fold the questions the user annotates into one self-contained page: the scans, their lines and their revision marks, all prefilled.

    python -m revrender.tools.annotate.page_pack --selection selection.json --pages DIR [--pages DIR ...] --revrender PAGES_OUT
        [--which pilot|test|val|train|all] --output-dir OUT

``--selection`` is the output of ``revrender.data.annotation_set``, ``--revrender`` the output of ``revrender.pages`` on those questions
(the pieces each question was cut into and the instances RevRender delivered).  For each question the page carries:

  lines   every piece as a four-corner polygon, in reading order (top to bottom, then left to right); the text of an OCR line is what the
          strong model read on that catalogue line (Qwen3.5-35B, asked to keep crossed-out writing; Infinity-Parser2 when it has none,
          the OCR's own text after that), the text of any other piece the OCR's; headers, footers and titles start as "not writing"
  marks   every RevRender instance cut to its ink: a thin, long one as a struck line (a polyline through its middle), any other as a
          blackout box; "written over" is never guessed

The editor (``page_editor.html``) and its rules (``page_rules.js``) are inlined with the scans, so OUT/标注.html opens from anywhere
without a server.  The pack's id depends only on ``--which``: a rebuilt pack keeps what the annotator already did in the browser.
The output is student handwriting: it belongs under work/, never in Git.
"""
import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ... import pages as pages_module
from ...config import expand, load_paths
from ...data import lines as lines_module
from ...data.prior_annotations import read_line_text
from ...eval.ink import INK_BELOW
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

HERE = Path(__file__).parent
EDITOR, RULES = HERE/'page_editor.html', HERE/'page_rules.js'
NOT_WRITING = ('header', 'footer', 'title')
THIN = .45                 # a mark whose typical height is under this share of its line's height ...
LONG = 2.5                 # ... and that is this many times longer than high is a struck line


def slot(text, name, payload):
    """Replace the <!--NAME-->...<!--/NAME--> block of the editor with an inline script."""
    begin, end = f'<!--{name}-->', f'<!--/{name}-->'
    require(text.count(begin) == 1 and text.count(end) == 1, f'the editor has no single {name} slot')
    head, rest = text.split(begin)
    return head+'<script>'+payload.replace('</', '<\\/')+'</script>'+rest.split(end)[1]


def tidy(text):
    """A reading as the annotator edits it: the strong model spaces every character out (``2 0 . ( 1 ) 解``); a space stays only between
    two ASCII letters (``\\geq a``, ``A P``), so LaTeX keeps its meaning and the rest closes up."""
    parts = re.split(r'(\s+)', (text or '').strip())
    out = []
    for k, part in enumerate(parts):
        if k % 2 == 0:
            out.append(part)
            continue
        before, after = parts[k-1][-1:], parts[k+1][:1]
        out.append(' ' if before.isascii() and before.isalpha() and after.isascii() and after.isalpha() else '')
    return ''.join(out)


def reading_order(pieces):
    """Pieces top to bottom by their centre, then left to right (the annotator fixes the order where a row has two pieces)."""
    return sorted(pieces, key=lambda p: ((p['box'][1]+p['box'][3])/2, p['box'][0]))


def mark_of(mask, box, line_height):
    """An editor mark for one instance: its mask (on ``box``, left/top/right/bottom on the page), cut to ink by the caller."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    left, top = box[0], box[1]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max())+1, int(ys.min()), int(ys.max())+1
    extents = [int(ys[xs == c].max()-ys[xs == c].min())+1 for c in np.unique(xs)]
    typical = float(np.median(extents))
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
        return {'type': 'line', 'shape': 'line', 'path': path, 'from': 'revrender'}
    return {'type': 'blot', 'shape': 'box', 'x': left+x0, 'y': top+y0, 'w': x1-x0, 'h': y1-y0, 'from': 'revrender'}


def page_record(entry, scan, record, instances, catalogue_lines, readings):
    """The editor's record of one question."""
    width, height = record['size']
    with Image.open(scan) as opened:
        grey = np.asarray(opened.convert('L'))
    require(grey.shape == (height, width), f'{entry["image_id"]}: the scan is not the size RevRender read')
    by_box = {tuple(lines_module.crop_box(c['box_1000'], width, height)): c['representative_line_id'] for c in catalogue_lines}
    pieces = reading_order(record['lines'])
    lines = []
    for piece in pieces:
        l, t, r, b = piece['box']
        line_id = by_box.get(tuple(piece['box'])) if piece['source'] == 'ocr' else None
        text = next((readings[m][line_id] for m in ('qwen35_35b', 'infinity') if line_id and readings[m].get(line_id)), None) or piece.get('ocr_text') or ''
        lines.append({'poly': [[l, t], [r, t], [r, b], [l, b]], 'text': tidy(text) if line_id else text.strip(), 'source': piece['source'], 'line_id': line_id,
                      'not_writing': piece.get('ocr_type') in NOT_WRITING, 'unreadable': False})
    marks = []
    for instance in instances:
        l, t, r, b = instance['box']
        ink = instance['mask'] & (grey[t:b, l:r] < INK_BELOW)
        mark = mark_of(ink if ink.any() else instance['mask'], instance['box'], b-t)
        if mark:
            marks.append({**mark, 'score': round(instance['score'], 3)})
    return {'id': entry['image_id'], 'question': entry['question'], 'pilot': entry['pilot'], 'width': width, 'height': height,
            'image': 'data:image/jpeg;base64,'+base64.b64encode(Path(scan).read_bytes()).decode(), 'lines': lines, 'marks': marks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    parser.add_argument('--revrender', type=Path, required=True, help='the output of revrender.pages on the selected questions')
    parser.add_argument('--which', choices=('pilot', 'test', 'val', 'train', 'all'), default='pilot')
    parser.add_argument('--catalogue', type=Path, help='the line catalogue (default: configs/paths.toml)')
    parser.add_argument('--root', type=Path, help="the earlier project's folder, for the strong models' readings (default: configs/paths.toml [prior] root)")
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    selection = json.loads(args.selection.read_text())
    entries = [p for p in selection['pages'] if args.which == 'all' or (p['pilot'] if args.which == 'pilot' else p['split'] == args.which)]
    require(entries, 'no question to pack')
    catalogue = lines_module.read_catalogue(args.catalogue) if args.catalogue else lines_module.default_catalogue()
    wanted = {p['image_id'] for p in entries}
    catalogue = [c for c in catalogue if c['image_id'] in wanted]
    root = args.root or expand(load_paths().get('prior', {}).get('root', ''))
    readings = read_line_text(root, {c['representative_line_id'] for c in catalogue}) if root and Path(root).is_dir() else {'qwen35_35b': {}, 'infinity': {}}
    scans = pages_module.page_files(args.pages, [p['image_id'] for p in entries])
    records = []
    for entry in entries:
        record, instances = pages_module.read_output(args.revrender, entry['image_id'])
        records.append(page_record(entry, scans[entry['image_id']], record, instances, [c for c in catalogue if c['image_id'] == entry['image_id']], readings))
    pack = {'id': f'paper2-{args.which}', 'schema': 'paper2-page-annotation-pack-v1', 'pages': records}
    html = slot(slot(EDITOR.read_text(encoding='utf-8'), 'RULES', RULES.read_text(encoding='utf-8')), 'PACK', 'window.PACK = '+json.dumps(pack, ensure_ascii=False)+';')
    out = make_output_dir(args.output_dir)
    (out/'标注.html').write_text(html, encoding='utf-8')
    summary = {'questions': len(records), 'lines': sum(len(r['lines']) for r in records), 'marks': sum(len(r['marks']) for r in records),
               'struck_lines_prefilled': sum(m['shape'] == 'line' for r in records for m in r['marks']),
               'texts_from_strong_model': sum(bool(l['line_id']) for r in records for l in r['lines']), 'bytes': (out/'标注.html').stat().st_size}
    write_manifest(out, config=to_jsonable(args), inputs={str(args.selection): digest(args.selection), str(EDITOR): digest(EDITOR), str(RULES): digest(RULES)}, **summary)
    write_json(out/'questions.json', [{'id': r['id'], 'question': r['question'], 'pilot': r['pilot']} for r in records])
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
