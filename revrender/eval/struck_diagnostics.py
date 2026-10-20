"""Why do VLMs misread struck content?  The diagnostic study of docs/struck_diagnostics_plan.md: nothing is trained, no test writer is
read.

    python -m revrender.eval.struck_diagnostics build --config configs/struck_diagnostics.toml --output-dir BUILD
    python -m revrender.eval.struck_diagnostics score --config configs/struck_diagnostics.toml --build BUILD
        --readings NAME=predictions.jsonl [--readings ...] --output-dir SCORE

``build``: the regions -- set P, the pixel-and-text refinements (``[data] pixel_export``: confirmed pixels and text, readable, a strike),
and set V, the validation box labels (``box_export``: readable struck lines) -- each drawn in eight versions (``VARIANTS``) and cut to
three scopes (``SCOPES``), a blue box round the region in every image.  Versions change the region only: R the page as it is; B the
region white; E* / M* the annotator's strike pixels white / red; E^p1 / M^p1 paper 1's RevRender masks (on the region grown by
``mask_reach``) white / red; E^syn / M^syn the synthetic-only box remover's mask the same way.  Set V has no strike pixels, so no E* / M*.
Scopes: S1 the region's box and ``crop_pad``; S2 the lines of paper 1's line cut that overlap the region (``line_band``); S3 the whole
answer image.  BUILD/pack_read and BUILD/pack_marked are line packs for ``recognize_lines --conditions boxed_read`` / ``boxed_marked``
(a row per region, version and scope); pack_*_s1 hold S1 x R, B, E*, M* only (the secondary models).  BUILD/regions.json: per region
its set, split, label and the counts the report groups by; BUILD/contact/: sheets of sampled regions to look at before reading.

``score``: per model, set, version and scope the share of struck characters held (``eval.revision_boxes.held``), exact readings, whole
readings, extra and empty readings; paired differences with a bootstrap over answer images; the decisions of ``[decisions]``.
"""
import argparse
import json
import random
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .. import pages as pages_module
from ..data.pixel_masks import SCHEMA as PIXEL_SCHEMA, decode, region_pixels
from ..data.prior_annotations import writer_splits
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .coverage import alignment, prepare
from .revision_boxes import read_labels

VARIANTS = ('R', 'B', 'E*', 'M*', 'Ep1', 'Mp1', 'Esyn', 'Msyn')
MARKED = {'M*', 'Mp1', 'Msyn'}
ORACLE = {'E*', 'M*'}
SCOPES = ('S1', 'S2', 'S3')
SECONDARY = {'R', 'B', 'E*', 'M*'}
INK = 245


def load_config(path):
    with open(path, 'rb') as handle:
        return tomllib.load(handle)


def bounds(region):
    xs, ys = [p[0] for p in region], [p[1] for p in region]
    return [int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))]


def page_mask(region, size):
    """The region's pixels on a page of ``size`` (width, height)."""
    l, t, r, b = bounds(region)
    w, h = size
    l, t, r, b = max(0, l-1), max(0, t-1), min(w, r+1), min(h, b+1)
    out = np.zeros((h, w), bool)
    out[t:b, l:r] = region_pixels(region, [l, t, r, b])
    return out


def runs_on_page(revision, size):
    l, t, r, b = revision['box']
    out = np.zeros((size[1], size[0]), bool)
    out[t:b, l:r] = decode(revision['runs'], (b-t, r-l))
    return out


def version(grey, variant, region, strike, colours):
    """An RGB page of one version: ``strike`` the pixels the version takes as the strike (the annotator's or a model's)."""
    rgb = np.repeat(grey[..., None], 3, axis=2).copy()
    if variant == 'B':
        rgb[region] = 255
    elif variant in MARKED:
        rgb[strike & (grey < INK)] = colours['red']
    elif variant != 'R':
        rgb[strike] = 255
    return rgb


def draw_box(rgb, box, gap, width, colour):
    image = Image.fromarray(rgb)
    l, t, r, b = box
    ImageDraw.Draw(image).rectangle([l-gap-width, t-gap-width, r+gap+width-1, b+gap+width-1], outline=tuple(colour), width=width)
    return np.asarray(image)


def line_band(box, lines, size, pad, overlap):
    """(crop, rule) of scope S2: the lines that overlap the region's box over ``overlap`` of the lower of the two heights, with the box,
    grown by ``pad``; without such a line, the box's own band (half its height up and down) across the page."""
    l, t, r, b = box
    w, h = size
    chosen = []
    for line in lines:
        x0, y0, x1, y1 = line['box']
        shared = min(b, y1)-max(t, y0)
        if shared > 0 and shared >= overlap*min(b-t, y1-y0):
            chosen.append(line['box'])
    if chosen:
        xs0, ys0, xs1, ys1 = zip(*chosen)
        crop, rule = [min(l, *xs0), min(t, *ys0), max(r, *xs1), max(b, *ys1)], f'{len(chosen)} line(s)'
    else:
        half = (b-t)//2
        crop, rule = [0, t-half, w, b+half], 'band'
    return [max(0, crop[0]-pad), max(0, crop[1]-pad), min(w, crop[2]+pad), min(h, crop[3]+pad)], rule


def scope_crop(scope, box, lines, size, cfg):
    l, t, r, b = box
    w, h = size
    pad = cfg['crop_pad']
    if scope == 'S1':
        return [max(0, l-pad), max(0, t-pad), min(w, r+pad), min(h, b+pad)], 'box'
    if scope == 'S2':
        return line_band(box, lines, size, pad, cfg['line_overlap'])
    return [0, 0, w, h], 'page'


def pixel_regions(cfg):
    """Set P: the confirmed, readable strike regions of the pixel export, with the split of their page."""
    export = json.loads(Path(cfg['pixel_export']).read_text())
    require(export.get('schema') == PIXEL_SCHEMA, 'not a pixel review export')
    split = {r['page']: r['split'] for r in json.loads(Path(cfg['page_splits']).read_text())}
    regions = []
    for page, q in export['pages'].items():
        for r in q.get('revisions', []):
            text = (r.get('text') or '').strip()
            if not (r.get('confirmed') and r.get('text_confirmed')) or r.get('unreadable') or r.get('not_revision') or not text.replace('?', '').strip():
                continue
            regions.append({'id': r['id'], 'set': 'P', 'page': page, 'split': split[page], 'region': r['region'], 'text': text,
                            'mode': r.get('annotationMode') or 'erase', 'manual': '#manual-' in r['id'], 'revision': r})
    return regions


def box_regions(cfg):
    """Set V: the readable struck lines of the validation box labels."""
    regions = []
    for page, q in read_labels([cfg['box_export']]).items():
        for r in q.get('revisions', []):
            text = (r.get('text') or '').strip()
            if r.get('kind') != 'line' or r.get('unreadable') or not text.replace('?', '').strip():
                continue
            regions.append({'id': f'{page}#{r["id"]}', 'set': 'V', 'page': page, 'split': 'val', 'region': r['region'], 'text': text,
                            'mode': 'box', 'manual': False, 'revision': None})
    return regions


def refuse_test_writers(regions, mapping, same_student):
    splits = writer_splits(mapping, same_student)
    test = sorted({r['page'] for r in regions if splits.get(r['page'].split('_')[0]) == 'test'})
    require(not test, f'{len(test)} page(s) of test writers, e.g. {test[:2]}: the test set is not read here')


def build(args):
    cfg = load_config(args.config)
    # a rebuilt detector or box remover (2026-10-21: the models trained without a labmate's data) replaces the configured one
    cfg['data'].update({k: str(v) for k, v in (('detector', args.detector), ('box_remover', args.box_remover)) if v})
    data, images = cfg['data'], cfg['images']
    regions = pixel_regions(data)+box_regions(data)
    refuse_test_writers(regions, data['mapping'], data['same_student'])
    out = make_output_dir(args.output_dir)
    for d in ('images', 'contact'):
        (out/d).mkdir()
    from .. import infer
    from ..box_remover import crop_of, read_boxes
    remover, _ = infer.load_checkpoint(Path(data['box_remover']))
    point = dict(zip(('mask_threshold', 'nms', 'score_threshold'), images['remover_point']))
    by_page = defaultdict(list)
    for r in regions:
        by_page[r['page']].append(r)
    rows, records, inputs = [], [], [Path(args.config), Path(data['pixel_export']), Path(data['page_splits']), Path(data['box_export']),
                                     Path(data['box_remover']), Path(data['mapping']), Path(data['same_student'])]
    for page in sorted(by_page):
        path = Path(data['pages'])/f'{page}.png'
        with Image.open(path) as opened:
            grey = np.asarray(opened.convert('L'))
        size = (grey.shape[1], grey.shape[0])
        record, instances = pages_module.read_output(data['detector'], page)
        require(record['size'] == list(size), f'{page}: the detector read another size')
        inputs += [path, Path(data['detector'])/f'{page}.json']
        detected = np.zeros(grey.shape, bool)
        for inst in instances:
            l, t, r, b = inst['box']
            detected[t:b, l:r] |= inst['mask']
        for reg in by_page[page]:
            region = page_mask(reg['region'], size)
            near = ndimage.binary_dilation(region, iterations=images['mask_reach'])
            box = bounds(reg['region'])
            synthetic = np.zeros(grey.shape, bool)
            for inst in read_boxes(remover, point, grey, [crop_of(reg['region'], size)], top=1):
                l, t, r, b = inst['box']
                synthetic[t:b, l:r] |= inst['mask']
            strikes = {'Ep1': detected & near, 'Mp1': detected & near, 'Esyn': synthetic & near, 'Msyn': synthetic & near}
            if reg['revision'] is not None:
                drawn = runs_on_page(reg['revision'], size) & region
                strikes['E*'] = strikes['M*'] = drawn
            k = len(records)
            ink = region & (grey < INK)
            strike_pixels = int(strikes['E*'].sum()) if 'E*' in strikes else None
            records.append({k2: v for k2, v in reg.items() if k2 not in ('revision',)} | {
                'index': k, 'box': box, 'ink_pixels': int(ink.sum()), 'strike_pixels': strike_pixels,
                'strike_share': strike_pixels/max(1, int(ink.sum())) if strike_pixels is not None else None,
                'p1_pixels': int(strikes['Ep1'].sum()), 'syn_pixels': int(strikes['Esyn'].sum()),
                'text_length': len(prepare(reg['text'])), 'has_math': '$' in reg['text']})
            for variant in VARIANTS:
                if variant in ORACLE and reg['revision'] is None:
                    continue
                rgb = draw_box(version(grey, variant, region, strikes.get(variant, np.zeros(grey.shape, bool)), images), box,
                               images['box_gap'], images['box_width'], images['blue'])
                for scope in SCOPES:
                    crop, rule = scope_crop(scope, box, record['lines'], size, images)
                    name = f'images/{k:05d}_{variant.replace("*", "o")}_{scope}.png'
                    Image.fromarray(rgb[crop[1]:crop[3], crop[0]:crop[2]]).save(out/name)
                    rows.append({'id': f'{reg["id"]}|{variant}|{scope}', 'line_id': reg['id'], 'image_id': page, 'set': reg['set'],
                                 'split': reg['split'], 'variant': variant, 'scope': scope, 'rule': rule, 'text': reg['text'], 'kind': 'line',
                                 'size': [crop[2]-crop[0], crop[3]-crop[1]], 'images': {'raw': name}, 'boxes': [[0, 0, 1000, 1000]]})
        print(json.dumps({'page': page, 'regions': len(by_page[page]), 'rows': len(rows)}), flush=True)
    packs = {'pack_read': lambda r: r['variant'] not in MARKED, 'pack_marked': lambda r: r['variant'] in MARKED}
    for name, keep in packs.items():
        for suffix, extra in (('', lambda r: True), ('_s1', lambda r: r['scope'] == 'S1' and r['variant'] in SECONDARY)):
            chosen = [r for r in rows if keep(r) and extra(r)]
            (out/(name+suffix)).mkdir()
            write_json(out/(name+suffix)/'pack.json', {'schema': 'revrender-line-pack', 'rows': [{**r, 'images': {'raw': '../'+r['images']['raw']}} for r in chosen]})
    write_json(out/'regions.json', records)
    contact(out, records, rows, cfg)
    summary = {'regions': {s: sum(r['set'] == s for r in records) for s in ('P', 'V')}, 'rows': len(rows),
               'splits': {s: sum(r['split'] == s for r in records) for s in ('train', 'val')},
               'modes': {m: sum(r['mode'] == m for r in records) for m in ('erase', 'keep', 'box')}}
    write_manifest(out, config=to_jsonable(args), recipe=cfg, inputs={str(p): digest(p) for p in inputs}, **summary)
    print(json.dumps(summary), flush=True)


def contact(out, records, rows, cfg):
    """Sheets of sampled P regions: every version at S1 side by side, and R at the three scopes."""
    sample = random.Random(SEED).sample([r for r in records if r['set'] == 'P'], min(cfg['images']['contact_regions'], sum(r['set'] == 'P' for r in records)))
    by = {(r['line_id'], r['variant'], r['scope']): out/r['images']['raw'] for r in rows}
    for n in range(0, len(sample), 6):
        tiles = []
        for reg in sample[n:n+6]:
            row = [Image.open(by[(reg['id'], v, 'S1')]).convert('RGB') for v in VARIANTS]
            height = 90
            row = [im.resize((max(1, round(im.width*height/im.height)), height)) for im in row]
            tiles.append(row)
        width = max(sum(im.width+8 for im in row) for row in tiles)
        sheet = Image.new('RGB', (width, len(tiles)*98), (235, 235, 235))
        for y, row in enumerate(tiles):
            x = 0
            for im in row:
                sheet.paste(im, (x, y*98))
                x += im.width+8
        sheet.save(out/'contact'/f'versions_{n//6:02d}.png')
    for reg in sample[:6]:
        ims = [Image.open(by[(reg['id'], 'R', s)]).convert('RGB') for s in SCOPES]
        ims = [im.resize((900, max(1, round(im.height*900/im.width)))) if im.width > 900 else im for im in ims]
        sheet = Image.new('RGB', (max(im.width for im in ims), sum(im.height+10 for im in ims)), (235, 235, 235))
        y = 0
        for im in ims:
            sheet.paste(im, (0, y))
            y += im.height+10
        sheet.save(out/'contact'/f'scopes_{reg["index"]:05d}.png')


def measures(label, reading):
    """held, exact (None for a label with ?), whole, extra (unaligned read characters per label character) and empty of one reading."""
    reading = '' if reading.strip() in ('', '[空]') else reading
    x, y = prepare(label), prepare(reading)
    kept_x, kept_y = alignment(x, y)
    counted = [i for i, c in enumerate(x) if c != '?']
    held = sum(i in kept_x for i in counted)/len(counted)
    return {'held': held, 'whole': float(held == 1), 'exact': None if '?' in x else float(x == y),
            'extra': (len(y)-len(kept_y))/len(counted), 'empty': float(not y)}


def bootstrap(values, clusters, resamples, seed=SEED):
    """(mean, low, high): a 95% interval of the mean of ``values`` resampling whole clusters."""
    groups = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    keys = sorted(groups)
    sums, counts = np.array([sum(groups[k]) for k in keys]), np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    means = sums[draws].sum(1)/counts[draws].sum(1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(scores, model, a, b, scope, regions, resamples, measure='held', subset='P'):
    """Mean and interval of b - a over the regions both were read on (b and a are versions, or (version, scope) pairs)."""
    va, vb = (a, scope) if isinstance(a, str) else a, (b, scope) if isinstance(b, str) else b
    ids = [r['id'] for r in regions if r['set'] == subset and (model, r['id'], *va) in scores and (model, r['id'], *vb) in scores]
    if not ids:
        return None
    diffs = [scores[(model, i, *vb)][measure]-scores[(model, i, *va)][measure] for i in ids]
    page = {r['id']: r['page'] for r in regions}
    mean, low, high = bootstrap(diffs, [page[i] for i in ids], resamples)
    return {'mean': mean, 'low': low, 'high': high, 'regions': len(ids)}


def decide(scores, regions, models, cfg):
    d, n = cfg['decisions'], cfg['score']['bootstrap']
    primary = [m for m in d['primary'] if m in models]
    out = {'primary': primary, 'rules': {}}
    table = lambda m, v, s, k: mean_of(scores, regions, m, v, s, k)
    per = defaultdict(dict)
    for m in primary:
        per[m]['d1'] = table(m, 'R', 'S1', 'held') < d['d1_held_below'] and (table(m, 'R', 'S1', 'exact') or 0) < d['d1_exact_below']
        gain = paired(scores, m, 'R', 'E*', 'S1', regions, n)
        per[m]['d2'] = gain['mean'] >= d['d2_gain'] and gain['low'] > 0
        per[m]['d2_none'] = gain['mean'] < d['d2_none']
        mark = paired(scores, m, 'E*', 'M*', 'S1', regions, n)
        per[m]['d3'] = mark['mean'] >= d['d3_margin'] and mark['low'] > 0
        per[m]['d3_erase'] = mark['mean'] <= -d['d3_margin'] and mark['high'] < 0
        for v in ('R', 'E*'):
            context = [scores[(m, r['id'], v, 'S3')]['held']-scores[(m, r['id'], v, 'S1')]['held']
                       -(scores[(m, r['id'], 'B', 'S3')]['held']-scores[(m, r['id'], 'B', 'S1')]['held'])
                       for r in regions if r['set'] == 'P' and all((m, r['id'], w, s) in scores for w in (v, 'B') for s in ('S1', 'S3'))]
            per[m][f'd5_{v}'] = float(np.mean(context)) if context else None
        per[m]['d2_detail'], per[m]['d3_detail'] = gain, mark
    count = lambda key: sum(bool(per[m][key]) for m in primary)
    majority = d['majority']
    out['rules'] = {
        'D1 problem': count('d1') >= majority,
        'D2 separation is the bottleneck': count('d2') >= majority,
        "D2' recognition is the bottleneck": count('d2_none') >= majority,
        'D3 keeping (marking) beats erasing': count('d3') >= majority,
        "D3' erasing beats keeping": count('d3_erase') >= majority,
        'D5 whole-question context helps (R)': sum((per[m]['d5_R'] or 0) >= d['d5_gain'] for m in primary) >= majority,
        'D5 whole-question context helps (E*)': sum((per[m]['d5_E*'] or 0) >= d['d5_gain'] for m in primary) >= majority}
    out['per_model'] = per
    return out


def mean_of(scores, regions, model, variant, scope, measure, subset='P', keep=None):
    values = [scores[(model, r['id'], variant, scope)][measure] for r in regions
              if r['set'] == subset and (keep is None or keep(r)) and (model, r['id'], variant, scope) in scores]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def score(args):
    cfg = load_config(args.config)
    regions = json.loads((args.build/'regions.json').read_text())
    rows = {}
    for pack in ('pack_read', 'pack_marked'):
        rows.update({r['id']: r for r in json.loads((args.build/pack/'pack.json').read_text())['rows']})
    scores, raw, models = {}, defaultdict(dict), []
    for item in args.readings:
        model, path = item.split('=', 1)
        if model not in models:          # a model's plain and marked readings come as two files
            models.append(model)
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            reading = json.loads(line)
            row = rows.get(reading['id'])
            if row is None or reading['condition'] != ('boxed_marked' if row['variant'] in MARKED else 'boxed_read'):
                continue
            scores[(model, row['line_id'], row['variant'], row['scope'])] = measures(row['text'], reading['text'])
            raw[model][reading['id']] = reading['text']
    table = {}
    for model in models:
        for subset in ('P', 'V'):
            for variant in VARIANTS:
                for scope in SCOPES:
                    cell = {k: mean_of(scores, regions, model, variant, scope, k, subset) for k in ('held', 'exact', 'whole', 'extra', 'empty')}
                    if cell['held'] is not None:
                        cell['regions'] = sum((model, r['id'], variant, scope) in scores for r in regions if r['set'] == subset)
                        table[f'{model}/{subset}/{variant}/{scope}'] = cell
    n = cfg['score']['bootstrap']
    comparisons = {}
    for model in models:
        for scope in SCOPES:
            for a, b in (('R', 'E*'), ('R', 'M*'), ('E*', 'M*'), ('Ep1', 'E*'), ('Esyn', 'E*'), ('Mp1', 'M*'), ('Msyn', 'M*'), ('R', 'Ep1'), ('R', 'Esyn'), ('B', 'R')):
                c = paired(scores, model, a, b, scope, regions, n)
                if c:
                    comparisons[f'{model}/{scope}/{b}-{a}'] = c
            for a, b in (('R', 'Ep1'), ('R', 'Mp1'), ('R', 'Esyn'), ('R', 'Msyn')):
                c = paired(scores, model, a, b, scope, regions, n, subset='V')
                if c:
                    comparisons[f'{model}/V/{scope}/{b}-{a}'] = c
        for v in ('R', 'E*', 'B'):
            c = paired(scores, model, (v, 'S1'), (v, 'S3'), None, regions, n)
            if c:
                comparisons[f'{model}/{v}/S3-S1'] = c
    groups = {}
    tercile = np.percentile([r['strike_share'] for r in regions if r['set'] == 'P'], [100/3, 200/3])
    splits = {'mode': lambda r: r['mode'], 'split': lambda r: r['split'], 'manual': lambda r: 'manual' if r['manual'] else 'detected',
              'math': lambda r: 'math' if r['has_math'] else 'plain',
              'length': lambda r: 'short' if r['text_length'] <= 3 else 'medium' if r['text_length'] <= 8 else 'long',
              'strike_share': lambda r: 'low' if r['strike_share'] <= tercile[0] else 'mid' if r['strike_share'] <= tercile[1] else 'high'}
    for model in models:
        for name, key in splits.items():
            for value in sorted({key(r) for r in regions if r['set'] == 'P'}):
                for variant in ('R', 'E*', 'M*'):
                    held = mean_of(scores, regions, model, variant, 'S1', 'held', keep=lambda r: key(r) == value)
                    if held is not None:
                        groups[f'{model}/{name}={value}/{variant}/S1'] = {'held': held, 'regions': sum(key(r) == value for r in regions if r['set'] == 'P')}
    decisions = decide(scores, regions, models, cfg)
    out = make_output_dir(args.output_dir)
    summary = {'table': table, 'comparisons': comparisons, 'groups': groups, 'decisions': decisions, 'models': models}
    write_json(out/'summary.json', summary)
    (out/'report.md').write_text(report(summary, regions, rows, raw), encoding='utf-8')
    write_manifest(out, config=to_jsonable(args), recipe=cfg, inputs={str(p): digest(Path(p.split('=', 1)[1])) for p in args.readings} |
                   {str(args.build/'regions.json'): digest(args.build/'regions.json')})
    print(json.dumps(decisions['rules'], ensure_ascii=False, indent=1))


def report(summary, regions, rows, raw):
    pct = lambda v: '—' if v is None else f'{100*v:.1f}'
    lines = ['# 诊断实验结果', '', '按 docs/struck_diagnostics_plan.md 运行；读回率、严格全对等均为百分比。', '', '## 判定（§8）', '']
    for rule, held in summary['decisions']['rules'].items():
        lines.append(f'- {rule}：{"成立" if held else "不成立"}')
    for model in summary['models']:
        lines += ['', f'## {model}', '', '| 集合 | 版本 | S1 读回 / 全对 | S2 读回 / 全对 | S3 读回 / 全对 | S1 多读 | S1 空读 |', '|---|---|---|---|---|---|---|']
        for subset in ('P', 'V'):
            for variant in VARIANTS:
                cells = [summary['table'].get(f'{model}/{subset}/{variant}/{s}') for s in SCOPES]
                if not any(cells):
                    continue
                show = lambda c: '—' if not c else f'{pct(c["held"])} / {pct(c["exact"])}'
                lines.append(f'| {subset} | {variant} | {" | ".join(show(c) for c in cells)} | {pct(cells[0]["extra"]) if cells[0] else "—"} | '
                             f'{pct(cells[0]["empty"]) if cells[0] else "—"} |')
        lines += ['', '| 比较 | 差（读回） | 95% 区间 | 处数 |', '|---|---|---|---|']
        for key, c in summary['comparisons'].items():
            if key.startswith(model+'/'):
                lines.append(f'| {key[len(model)+1:]} | {pct(c["mean"])} | {pct(c["low"])} ~ {pct(c["high"])} | {c["regions"]} |')
    lines += ['', '## 分组（S1 读回）', '', '| 模型 / 分组 / 版本 | 读回 | 处数 |', '|---|---|---|']
    lines += [f'| {k} | {pct(v["held"])} | {v["regions"]} |' for k, v in summary['groups'].items()]
    lines += ['', '## 逐处样例（S1）', '']
    for reg in random.Random(SEED).sample([r for r in regions if r['set'] == 'P'], min(12, len(regions))):
        lines.append(f'- `{reg["id"]}` 答案：{reg["text"]}')
        for model in summary['models']:
            reads = '；'.join(f'{v}：' + raw[model].get('|'.join((reg['id'], v, 'S1')), '—') for v in ('R', 'E*', 'M*'))
            lines.append(f'  - {model}：{reads}')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--config', type=Path, required=True)
    b.add_argument('--output-dir', type=Path, required=True)
    b.add_argument('--detector', type=Path, help='page detections in place of [data] detector')
    b.add_argument('--box-remover', type=Path, help='a box remover checkpoint in place of [data] box_remover')
    s = sub.add_parser('score')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('--build', type=Path, required=True)
    s.add_argument('--readings', action='append', required=True, help='NAME=predictions.jsonl of recognize_lines on the build packs')
    s.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    build(args) if args.command == 'build' else score(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
