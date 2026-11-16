"""The struck regions by the line they stand in: the annotator's line review merged into the dataset, and each line's two images.

    python -m revrender.data.line_set merge --dataset regions.json --folds FOLDS_PACK --review EXPORT.json [--fixes FIXES.json] --output-dir OUT
    python -m revrender.data.line_set unchanged --old regions.json --new MERGED/regions.json --output OUT.json
    python -m revrender.data.line_set pack --dataset OUT/regions.json --lines OUT/lines.json --pages DIR [...] --output-dir PACK

``merge``: the export of the line page (``tools.annotate.line_review``, ``paper2-line-review-v1``) laid over the dataset
(``data.struck_dataset``) as a later export: each region it holds takes the annotator's text, readable mark, polygon, box and strike
pixels (the earlier values kept under ``line_review``), a region it added is a new row of its question's writer, and a region it took
out leaves the set the line page showed (it stays in the file, ``deleted``).  Every region the export holds carries the fold and the val
mark of its writer in the folds pack (``data.struck_pack folds``) the page was built from, so a region added keeps the fold of its
question.  OUT/lines.json: a row per line (page, writer, fold, val, the line's polygon, its regions from left to right by the centre of
their polygons' boxes, whether the one region is the whole line).  ``--fixes``: the corrections made on the page that checks the line
readings (``paper2-line-reading-fix-v1``), laid over that: a region's text and readable mark, and its strike pixels (runs over its box)
where they were painted again; each correction must start from what the region holds, and the values it replaces are kept under
``reading_fix``.

``pack``: a line pack (``data.line_pack`` schema) of the lines with a struck text to read, a row per line, and beside it, in
PACK/regions, the pack of every region of those lines as ``data.struck_pack folds`` cuts it (the scan, the erased crop and the layer
map, white outside the polygon).  A line's window is its polygon's box and six pixels, white outside the polygon; ``before`` has the
strike pixels of its readable regions white (the struck writing left) and every unreadable region's polygon white, ``after`` every
region's polygon white.  A line whose one region is the whole line has no ``after`` (nothing would be left).  The label of a line is the
struck text of its readable regions, left to right, in the spelling of ``eval.coverage.prepare`` (``?`` dropped).  A line none of
whose regions can be read is left out, and a line holding a readable region with no text.  Student text and pages: OUT and PACK stay
out of Git.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..util import digest, make_output_dir, require, write_json, write_manifest
from .pixel_masks import decode, region_pixels, reviewed_crop
from .struck_pack import layered, window

REVIEW = 'paper2-line-review-v1'
FIXES = 'paper2-line-reading-fix-v1'


def centre(polygon):
    xs = [x for x, _ in polygon]
    return (min(xs)+max(xs))/2


def merge(args):
    dataset = json.loads(args.dataset.read_text()); review = json.loads(args.review.read_text())
    require(review.get('schema') == REVIEW, f'not an export of the line page ({REVIEW})')
    folds = {}
    for r in json.loads((args.folds/'pack.json').read_text())['rows']:
        require(folds.setdefault(r['writer'], (r['fold'], r['val'])) == (r['fold'], r['val']), f"writer {r['writer']}: two folds")
    by_id = {r['id']: r for r in dataset}; page_rows = {}
    for r in dataset:
        page_rows.setdefault(r['page'], []).append(r)
    name = args.review.name; rows = [dict(r) for r in dataset]; index = {r['id']: i for i, r in enumerate(rows)}; counts = Counter()
    for e in review['regions']:
        if e['added']:
            mates = page_rows.get(e['page']); require(mates, f"{e['id']}: its question has no region in the dataset")
            writer = mates[0]['writer']; require({m['writer'] for m in mates} == {writer}, f"{e['page']}: two writers")
            base = {k: mates[0][k] for k in ('page', 'writer', 'split', 'paper1_split', 'image_source')}
            row = {**base, 'id': e['id'], 'export': name, 'first_export': name, 'judgment': None, 'edits': None, 'seconds': None, 'prefill_model': 'manual',
                   'not_revision': False, 'pixels_uncertain': False, 'pixels_confirmed': True, 'text_confirmed': True}
            index[e['id']] = len(rows); rows.append(row); counts['added'] += 1
        else:
            require(e['id'] in by_id, f"{e['id']}: not in the dataset"); row = rows[index[e['id']]]
            row['line_review'] = {'text': row['text'], 'readable': row['readable'], 'region': row['region'], 'box': row['box'], 'runs': row['runs']}
            row['export'] = name; counts['reviewed'] += 1
        require(row['writer'] in folds, f"{e['id']}: its writer is not in the folds pack")
        mask = decode(e['runs'], (e['box'][3]-e['box'][1], e['box'][2]-e['box'][0]))
        row.update({'text': e['text'], 'readable': e['readable'], 'region': e['region'], 'box': e['box'], 'runs': e['runs'], 'mode': e['mode'],
                    'strike_pixels': int(mask.sum()), 'deleted': e['deleted'], 'line': e['line'], 'whole_line': e['whole_line'],
                    'fold': folds[row['writer']][0], 'val': folds[row['writer']][1]})
        counts['deleted'] += e['deleted']; counts['text_changed'] += bool(not e['added'] and e['corrected'])
        counts['region_changed'] += bool(not e['added'] and e['region_changed']); counts['pixels_changed'] += bool(not e['added'] and e['pixels_changed'])
    for fix in json.loads(args.fixes.read_text())['fixes'] if args.fixes else []:
        require(fix['region'] in index, f"{fix['region']}: not in the dataset"); row = rows[index[fix['region']]]
        require((row['text'], row['readable']) == (fix['original_text'], fix['original_readable']), f"{fix['region']}: the correction starts from another text")
        row['reading_fix'] = {'text': row['text'], 'readable': row['readable'], 'box': row['box'], 'runs': row['runs']}
        row.update({'text': fix['text'], 'readable': fix['readable']}); counts['fix_text'] += fix['text_changed']
        counts['fix_unreadable'] += bool(fix['original_readable'] and not fix['readable'])
        if fix['pixels_changed']:
            require(fix['box'] == row['box'], f"{fix['region']}: the correction's box is not the region's")
            row.update({'runs': fix['runs'], 'strike_pixels': int(decode(fix['runs'], (row['box'][3]-row['box'][1], row['box'][2]-row['box'][0])).sum())})
            counts['fix_pixels'] += 1
    regions = {r['id']: r for r in rows}; lines = []
    for line in review['lines']:
        inside = sorted(line['regions'], key=lambda k: centre(regions[k]['region']))
        require(inside and all(regions[k]['line'] == line['id'] and not regions[k]['deleted'] for k in inside), f"{line['id']}: its regions do not name it")
        require(not line['whole_line'] or len(inside) == 1, f"{line['id']}: a whole line of several regions")
        writer = regions[inside[0]]['writer']
        lines.append({'id': line['id'], 'page': line['page'], 'writer': writer, 'fold': folds[writer][0], 'val': folds[writer][1], 'polygon': line['polygon'],
                      'regions': inside, 'whole_line': line['whole_line'], 'adjusted': line['adjusted']})
    placed = Counter(k for line in lines for k in line['regions'])
    require(all(placed[e['id']] == (0 if e['deleted'] else 1) for e in review['regions']), 'a region in no line or in two')
    out = make_output_dir(args.output_dir); write_json(out/'regions.json', rows); write_json(out/'lines.json', lines)
    summary = {'regions': len(rows), 'in_review': len(review['regions']), **counts, 'lines': len(lines), 'whole_lines': sum(l['whole_line'] for l in lines)}
    write_manifest(out, config={k: str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in (args.dataset, args.review, args.folds/'pack.json', *([args.fixes] if args.fixes else []))}, **summary)
    print(json.dumps(summary), flush=True)


def target(region):
    """The struck text a reader should give for a readable region, in the spelling the scores are counted on."""
    from ..eval.coverage import prepare
    return prepare(region['text'] or '').replace('?', '')


def line_images(page, line, regions):
    """(before, after or None) of a line, grey uint8 images of its window."""
    grey = np.asarray(page.convert('L')); h, w = grey.shape
    l, t, r, b = window(line['polygon'], page.size)
    keep = region_pixels(line['polygon'], [l, t, r, b]); before = np.where(keep, grey[t:b, l:r], 255).astype(np.uint8); after = before.copy()
    for region in regions:
        bl, bt, br, bb = region['box']; x0, y0, x1, y1 = max(bl, l), max(bt, t), min(br, r), min(bb, b)
        if x0 >= x1 or y0 >= y1:
            continue
        inside = region_pixels(region['region'], region['box'])[y0-bt:y1-bt, x0-bl:x1-bl]
        if region['readable']:
            _, strike = reviewed_crop(page, {'box': region['box'], 'runs': region['runs'], 'region': region['region']})
            before[y0-t:y1-t, x0-l:x1-l][strike[y0-bt:y1-bt, x0-bl:x1-bl].astype(bool)] = 255
        else:
            before[y0-t:y1-t, x0-l:x1-l][inside] = 255
        after[y0-t:y1-t, x0-l:x1-l][inside] = 255
    return before, None if line['whole_line'] else after


def chosen_lines(lines, regions):
    """(the lines to read, each with its label; the counts of the lines left out and why)."""
    chosen, skipped = [], Counter()
    for line in lines:
        inside = [regions[k] for k in line['regions']]; readable = [r for r in inside if r['readable']]
        if not readable:
            skipped['nothing_readable'] += 1
        elif any(not target(r) for r in readable):
            skipped['readable_without_text'] += 1
        else:
            chosen.append({**line, 'text': ''.join(target(r) for r in readable), 'readable_regions': [r['id'] for r in readable]})
    return chosen, skipped


def page_files(chosen, regions, pages):
    """{page: file}: the page its export showed the annotator (the dataset's ``image_source`` when that folder is here, else the first of
    ``pages`` holding it), as ``struck_pack.chosen_regions`` finds it."""
    source = {}
    for line in chosen:
        for k in line['regions']:
            folder = regions[k]['image_source']; folder = folder if folder and Path(folder).is_dir() else ''
            require(source.setdefault(line['page'], folder) == folder, f"{line['page']}: its regions were drawn on two images")
    files = {}
    for folder in set(source.values()):
        files.update(pages_module.page_files([folder] if folder else pages, sorted(k for k, v in source.items() if v == folder)))
    return files


def line_pack(args):
    regions = {r['id']: r for r in json.loads(args.dataset.read_text())}; lines = json.loads(args.lines.read_text())
    chosen, skipped = chosen_lines(lines, regions)
    files = page_files(chosen, regions, args.pages)
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); (out/'regions').mkdir(); (out/'regions'/'crops').mkdir(); rows, region_rows = [], []
    for page_id in sorted(files):
        with Image.open(files[page_id]) as page:
            for line in (x for x in chosen if x['page'] == page_id):
                inside = [regions[k] for k in line['regions']]; made = line_images(page, line, inside)
                images = {name: f'crops/{len(rows):05d}_{name}.png' for name, image in zip(('before', 'after'), made) if image is not None}
                for name, image in zip(('before', 'after'), made):
                    if image is not None:
                        Image.fromarray(image).save(out/images[name])
                rows.append({'id': line['id'], 'line_id': line['id'], 'kind': 'whole' if line['whole_line'] else 'partial', 'page': page_id,
                             'writer': line['writer'], 'fold': line['fold'], 'val': line['val'], 'readable': True, 'text': line['text'], 'target': line['text'],
                             'regions': line['regions'], 'readable_regions': line['readable_regions'], 'images': images, 'size': list(made[0].shape[::-1]), 'boxes': []})
                for r in inside:
                    crops = {name: f'crops/{len(region_rows):05d}_{name}.png' for name in ('raw', 'erased', 'map')}
                    for name, image in zip(crops, layered(page, r)):
                        image.save(out/'regions'/crops[name])
                    region_rows.append({'id': r['id'], 'line_id': r['id'], 'kind': 'line', 'line': line['id'], 'writer': r['writer'], 'fold': r['fold'], 'val': r['val'],
                                        'readable': bool(r['readable']), 'text': r['text'] if r['readable'] else '', 'images': crops, 'size': list(image.size), 'boxes': [],
                                        'strike_pixels': r['strike_pixels'], 'mode': r['mode']})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_json(out/'regions'/'pack.json', {'schema': 'revrender-line-pack', 'rows': region_rows})
    summary = {'lines': len(rows), 'whole': sum(r['kind'] == 'whole' for r in rows), 'partial': sum(r['kind'] == 'partial' for r in rows),
               'regions': len(region_rows), 'readable_regions': sum(r['readable'] for r in region_rows), 'writers': len({r['writer'] for r in rows}),
               'left_out': dict(skipped)}
    write_manifest(out, config={k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in vars(args).items() if k != 'command'},
                   inputs={str(p): digest(p) for p in (args.dataset, args.lines)}, **summary)
    print(json.dumps(summary), flush=True)


def unchanged(args):
    """The regions readable in both datasets whose box, polygon and strike pixels are the same (``pixels``), and of those the ones whose
    text is too (``all``): where a reading scored against both labels shows what changing the labels did, apart from the regions."""
    old = {r['id']: r for r in json.loads(args.old.read_text())}; new = {r['id']: r for r in json.loads(args.new.read_text()) if not r.get('deleted')}
    same = lambda a, b, keys: all(a[k] == b[k] for k in keys)
    pixels = sorted(k for k in old.keys() & new.keys() if old[k]['readable'] and new[k]['readable'] and same(old[k], new[k], ('box', 'region', 'runs')))
    result = {'pixels': pixels, 'all': [k for k in pixels if old[k]['text'] == new[k]['text']]}
    require(not args.output.exists(), 'output exists'); write_json(args.output, {**result, 'inputs': {str(p): digest(p) for p in (args.old, args.new)}})
    print(json.dumps({k: len(v) for k, v in result.items()}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('merge')
    for key in ('dataset', 'folds', 'review', 'output-dir'):
        m.add_argument('--'+key, type=Path, required=True)
    m.add_argument('--fixes', type=Path, help='the corrections of the page that checks the line readings (paper2-line-reading-fix-v1)')
    k = sub.add_parser('pack')
    for key in ('dataset', 'lines', 'output-dir'):
        k.add_argument('--'+key, type=Path, required=True)
    k.add_argument('--pages', type=Path, action='append', required=True)
    u = sub.add_parser('unchanged')
    for key in ('old', 'new', 'output'):
        u.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv)
    {'merge': merge, 'pack': line_pack, 'unchanged': unchanged}[args.command](args)


if __name__ == '__main__':
    main()
