"""An offline page on which the annotator draws, round each struck region, the whole line it stands in, and corrects the struck text.

    python -m revrender.tools.annotate.line_review build --dataset regions.json --ids PACK.json|IDS.txt --pages DIR [...] \
        [--reading LABEL=DIR[,DIR...][@CONDITION] ...] [--check LABEL] --katex KATEX_DIST --name NAME --output-dir OUT

The unit a reader is scored on becomes a line (user 2026-11-15): each region belongs to one line, one polygon per line shared by its
regions, and a region that is the whole line marked so (its line is then its own polygon).  The page shows each question's clean image
(the one its export showed the annotator: the dataset's ``image_source`` when that folder is here, else the first of ``--pages`` holding
it) with the regions' polygons (read only) and a prefilled line polygon per line, which the annotator drags, extends, redraws or makes
anew; and per region its crops in the readers' window (as ``data.struck_pack.layered``: the scan and the scan with the strike pixels white,
white outside the polygon), the readings named by ``--reading`` (only to find mistakes), its struck text and the readable mark to
correct, the whole-line mark and a seen mark.  The region polygons and strike pixels can be changed too: a polygon dragged, a region
added or taken out, the strike pixels painted (B) or erased (E) in the window; the crops are cut in the page from the page image, as
``layered`` cuts them, and follow every change.

The prefill: a region goes to the line of the catalogue (``data.lines``) that covers most of its height; regions no catalogue line
covers get a band of their own height stretched sideways along the ink (gaps up to ``GAP`` heights), merged into a line they overlap.
A line's polygon is the box of its catalogue line (or band) and its regions.  A region alone on its line whose polygon holds at least
``WHOLE`` of the line's ink is guessed to be the whole line.

The page keeps every change in IndexedDB and localStorage (read back, the place shown) and, once bound, in a backup file; it exports
and imports the progress as JSON (``schema`` ``paper2-line-review-v1``).  A new build of the page (another pack id) takes over what the
builds named by ``--carry-over`` saved in the same browser, while it has nothing of its own.  The output is student work: it stays out of Git.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...data.lines import crop_box, default_catalogue, lines_by_image
from ...data.pixel_masks import region_pixels
from ...eval.transcription import counts, refused
from ...util import digest, make_output_dir, require, write_json, write_manifest
from .pixel_offline import katex_inline, script_json

HERE = Path(__file__).parent
ASSETS = ('line_editor.html', 'page_rules.js', 'pixel_rules.js', 'pixel_overview.js', 'library_store.js', 'symbol_picker.js', 'line_rules.js', 'line_editor.js')
INK = 200      # grey below which a pixel is ink
GAP = 2.        # a band follows the ink sideways across gaps of up to this many region heights
COVER = .5     # a catalogue line takes a region when it covers at least this share of the region's height
WHOLE = .85    # a region alone on its line holding this share of the line's ink is guessed to be the whole line
PAD = 3


def rectangle(l, t, r, b):
    return [[l, t], [r, t], [r, b], [l, b]]


def union(*boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def band(ink, box):
    """The box of a region's own rows stretched left and right along the ink, across gaps of up to GAP region heights."""
    l, t, r, b = box; columns = ink[t:b].any(0); gap = int(GAP*(b-t)); left, right = l, r
    for step, edge in ((-1, l), (1, r-1)):
        x, empty = edge, 0
        while 0 <= x+step < len(columns) and empty <= gap:
            x += step
            if columns[x]: empty = 0; left, right = (x, right) if step < 0 else (left, x+1)
            else: empty += 1
    return [left, t, right, b]


def overlap(a, b, axis):
    return min(a[axis+2], b[axis+2])-max(a[axis], b[axis])


def estimate_lines(grey, regions, catalogue):
    """[{regions, box, source}] of a question: the regions grouped into lines, each with the box its polygon starts from."""
    h, w = grey.shape; ink = grey < INK
    lines = [{'regions': [], 'box': list(crop_box(e['box_1000'], w, h)), 'source': 'catalogue'} for e in catalogue]
    loose = []
    for r in regions:
        box = r['box']; height = box[3]-box[1]
        scored = [(overlap(box, line['box'], 1)/height-(0 if overlap(box, line['box'], 0) > 0 else .5), i) for i, line in enumerate(lines)]
        best = max(scored, default=(0, None))
        if best[0] >= COVER: lines[best[1]]['regions'].append(r)
        else: loose.append(r)
    lines = [line for line in lines if line['regions']]
    for r in loose:
        own = band(ink, r['box'])
        near = [line for line in lines if overlap(own, line['box'], 1) > .5*min(own[3]-own[1], line['box'][3]-line['box'][1]) and overlap(own, line['box'], 0) > 0]
        if near: near[0]['regions'].append(r)
        else: lines.append({'regions': [r], 'box': own, 'source': 'ink'})
    for line in lines:
        l, t, r, b = union(line['box'], *(x['box'] for x in line['regions']))
        line['box'] = [max(0, l-PAD), max(0, t-PAD), min(w, r+PAD), min(h, b+PAD)]
    return sorted(lines, key=lambda line: (line['box'][1], line['box'][0]))


def whole_share(grey, region, line_box):
    """The share of the ink inside a line's box that lies inside the region's polygon."""
    l, t, r, b = line_box; ink = grey[t:b, l:r] < INK
    rl, rt, rr, rb = region['box']; inside = np.zeros_like(ink)          # the line's box holds the region's (estimate_lines)
    inside[rt-t:rb-t, rl-l:rr-l] = region_pixels(region['region'], region['box'])
    return float((ink & inside).sum()/max(1, ink.sum()))


def data_uri(image):
    buffer = io.BytesIO(); image.save(buffer, 'PNG', optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def readings(specs, ids):
    """{label: {region id: text}} of the readings named LABEL=DIR[,DIR...][@CONDITION]; a label must read every region."""
    out = {}
    for spec in specs:
        label, rest = spec.split('=', 1); folders, _, condition = rest.partition('@'); got = {}
        for folder in folders.split(','):
            for line in (Path(folder)/'predictions.jsonl').read_text().splitlines():
                g = json.loads(line)
                if g['id'] in ids and (not condition or g['condition'] == condition): got[g['id']] = g['text']
        require(set(got) == ids, f'the reading {label} does not cover every region')
        out[label] = got
    return out


def wrong(row, reading):
    if not row['readable']: return not refused(reading)
    return not counts(row['text'], reading)['exact']


def chosen_ids(path):
    text = Path(path).read_text()
    if path.suffix == '.json': return {r['id'] for r in json.loads(text)['rows']}
    return {line.strip() for line in text.splitlines() if line.strip()}


def page_file(region, folders):
    for folder in ([Path(region['image_source'])] if region['image_source'] else [])+list(folders):
        if (folder/(region['page']+'.png')).exists(): return folder/(region['page']+'.png')
    raise SystemExit(f'no image of {region["page"]} in --pages')


def build(args):
    dataset = json.loads(args.dataset.read_text()); ids = chosen_ids(args.ids)
    chosen = [r for r in dataset if r['id'] in ids]
    require(len(chosen) == len(ids), 'regions named by --ids are missing from the dataset')
    read = readings(args.reading, ids); check = args.check or (args.reading[-1].split('=')[0] if args.reading else None)
    catalogue = lines_by_image(default_catalogue())
    pages, inputs, stats = [], {str(p): digest(p) for p in [args.dataset, args.ids, *(HERE/n for n in ASSETS), Path(__file__)]}, []
    for page_id in sorted({r['page'] for r in chosen}):
        here = sorted((r for r in chosen if r['page'] == page_id), key=lambda r: (r['box'][1], r['box'][0]))
        path = page_file(here[0], args.pages); require(all(page_file(r, args.pages) == path for r in here), 'regions of one page on two images')
        inputs[str(path)] = digest(path)
        with Image.open(path) as im:
            image = im.convert('L'); grey = np.asarray(image)
        lines = estimate_lines(grey, here, catalogue.get(page_id, []))
        rows = []
        for r in here:
            row = {'id': r['id'], 'region': r['region'], 'box': r['box'], 'runs': r['runs'], 'mode': r['mode'] or 'erase', 'text': r['text'] or '',
                   'readable': bool(r['readable']), 'readings': {label: got[r['id']] for label, got in read.items()}}
            if check: row['wrong'] = wrong({'readable': bool(r['readable']), 'text': r['text'] or ''}, read[check][r['id']])
            rows.append(row)
        out_lines = []
        for n, line in enumerate(lines, 1):
            line_id = f'{page_id}#L{n}'; members = [x['id'] for x in line['regions']]; estimate = rectangle(*line['box'])
            whole = len(members) == 1 and whole_share(grey, line['regions'][0], line['box']) >= WHOLE
            out_lines.append({'id': line_id, 'regions': members, 'estimate': estimate, 'source': line['source'], 'whole': whole,
                              'polygon': line['regions'][0]['region'] if whole else estimate})
        others = [{'id': r['id'], 'region': r['region']} for r in dataset if r['page'] == page_id and r['id'] not in ids and r['region']]
        pages.append({'id': page_id, 'size': [image.width, image.height], 'image': data_uri(image), 'regions': rows, 'lines': out_lines, 'others': others})
        stats.append({'page': page_id, 'regions': len(rows), 'lines': len(out_lines), 'whole': sum(l['whole'] for l in out_lines),
                      'from_ink': sum(l['source'] == 'ink' for l in out_lines)})
    revision = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    pack = {'schema': 'paper2-line-review-pack-v1', 'id': f'{args.name}-{revision}', 'name': args.name, 'pages': pages,
            'labels': dict(label.split('=', 1) for label in args.label) if args.label else {}, 'check': check, 'carry': args.carry_over}
    out = make_output_dir(args.output_dir)
    html = (HERE/'line_editor.html').read_text()
    style, script = katex_inline(args.katex)
    html = html.replace('<!--KATEX-->', style+script).replace('<!--PACK-->', '<script id="line-pack" type="application/json">'+script_json(pack)+'</script>')
    for name in ASSETS[1:]:
        html = html.replace(f'<script src="{name}"></script>', '<script>'+(HERE/name).read_text().replace('</script', '<\\/script')+'</script>')
    require('<script src=' not in html and '<!--' not in html, 'an asset or the pack was not inlined')
    (out/'整行框选.html').write_text(html)
    summary = {'pack': pack['id'], 'pages': len(pages), 'regions': sum(s['regions'] for s in stats), 'lines': sum(s['lines'] for s in stats),
               'whole_guessed': sum(s['whole'] for s in stats), 'lines_from_ink': sum(s['from_ink'] for s in stats),
               'pages_without_catalogue': sum(p not in catalogue for p in {r['page'] for r in chosen}),
               'checked_wrong': sum(r.get('wrong', False) for p in pages for r in p['regions'])}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs=inputs, katex=str(args.katex), per_page=stats, **summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--dataset', type=Path, required=True, help='regions.json of data.struck_dataset')
    b.add_argument('--ids', type=Path, required=True, help='a pack.json (its rows) or a text file of region ids')
    b.add_argument('--pages', type=Path, action='append', default=[], help='folders of clean question images, for regions whose image_source is not here')
    b.add_argument('--reading', action='append', default=[], help='LABEL=DIR[,DIR...][@CONDITION]: predictions.jsonl shown beside each region')
    b.add_argument('--label', action='append', default=[], help='LABEL=what the page calls that reading')
    b.add_argument('--check', help='the reading whose mistakes the page can filter by (default the last)')
    b.add_argument('--katex', type=Path, required=True, help='a KaTeX dist folder to inline (formulas render offline)')
    b.add_argument('--carry-over', action='append', default=[], help='the pack id of an earlier build whose progress this browser keeps: taken over while this build has none')
    b.add_argument('--name', required=True); b.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    build(args)


if __name__ == '__main__':
    main()
