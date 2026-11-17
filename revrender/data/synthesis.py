"""Strike materials for synthesis and their checks (plan of 2026-11-17, "按一笔来"; rules written before any reading in
work/synthesis_plan20261116/合成数据_第二版_按一笔来_方案与判定规则_2026-11-17.md).  No training here.

    python -m revrender.data.synthesis materials --set PREFILL_SET --bank BANK --dataset regions.json --catalogue LINES.jsonl
        --mapping index_manifest.json --same-student pairs.json --templates templates.npz --pages DIR --output-dir OUT
    python -m revrender.data.synthesis read-check --pack OUT/reader --predictions READ/predictions.jsonl --output check4.json

Three sources (``data.one_stroke`` makes the units): H the annotator's erased pixels of the prefill set's training regions, cut into
lines, groups and blobs and repaired; P paper 1's bank lines, regrown on their page and repaired; D drawn ones, put where an H unit of
their kind was (its frame, measures and shape).  No material or host of an excluded writer (the prefill set's held-out and fixed
validation writers, paper 1's test writers, and each one's other id); readable regions only (a blackout read as unreadable is not used).

``materials``: every unit (OUT/materials.json, the arrays in OUT/materials.npz: ``<id>|repaired`` and ``<id>|before`` as uint8 opacity
on the unit's origin window); ``LOOK`` of each source (seed 42) each put once on a host (up to ``one_stroke`` tries of ``TRIES`` host
lines of a pool drawn writer by writer); the numbers of checks 1, 2, 3 and 5 (OUT/summary.json), check 4's reader pack (OUT/reader:
each unit alone on white) and the page to look at (OUT/check_page.html: origin, the repaired unit straightened and put back, put on
its host).  ``read-check``: check 4's shares.  Everything under OUT is student writing: it stays out of Git.
"""
import argparse
import base64
import html
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .. import pages
from ..eval.ink import INK_BELOW
from ..util import SEED, digest, make_output_dir, to_jsonable, write_json, write_manifest
from . import one_stroke as S
from .clean_pages import review_clean
from .handwriting import group_key
from .lines import crop_box, read_catalogue
from .prior_annotations import twins, writer_splits
from .strike_bank import INK_ALPHA, ink_level, line_measures, opacity

SOURCES = ('H', 'P', 'D')
LOOK = 200                    # units of each source checked (seed 42)
TRIES = 20                    # hosts tried for a unit before it is counted as not put
POOL = 1500                   # host lines in the pool, drawn writer by writer
NOMINAL = 40                  # a drawn unit is drawn for writing this high
DRAWN = {'line': .2, 'double': .1, 'wave': .1, 'zigzag': .2, 'cross': .15, 'scribble': .25}   # D's styles (the 11-16 weights)
KIND_OF = {'line': 'line', 'wave': 'line', 'double': 'group', 'cross': 'group', 'zigzag': 'group', 'scribble': 'blob'}
SHARE_CUTS = (.23, .35)       # light, middle, heavy by the strike's share of the ink (eval.prefill's tiers)
HOLES = .3                    # a blob with more writing in its holes than this share of its pixels is left out (as on 11-16)
FRAGMENTS = .4                # or with more of its pixels in specks
PRESSED = .05                 # a unit with less of its pixels on writing at its origin is not a strike on writing (a figure's line)
TALL = 1.2                    # a bank line's struck writing taller than this many times its line (paper 1's) is not a row of writing
ROW = 3.                      # a catalogue line more than this many times as high as the region is not its row (the window keeps to the region)
BIG = .25                     # a region whose box is more than this share of its page is not one revision (a whole answer)
CHECK = {'fill_median': 1.10, 'fill_p95': 1.30, 'cover_median': .95, 'cover_below_0.8': .10, 'back_covered': .90, 'back_on_paper': .05,
         'read_over_control': .05, 'put': .80}


def tier(share):
    return 'light' if share < SHARE_CUTS[0] else 'middle' if share < SHARE_CUTS[1] else 'heavy'


def canonical(writer, other):
    return min(writer, other.get(writer, writer))


def excluded_writers(set_dir, mapping, same_student):
    """The writers no material or host may come from: the held-out and fixed-validation writers of the prefill set, paper 1's test
    writers, and each one's other id."""
    summary = json.loads((set_dir/'summary.json').read_text()); other = twins(same_student); splits = writer_splits(mapping, same_student)
    out = set(summary['validation_writers'])|set(summary['fixed_writers'])|{w for w, s in splits.items() if s == 'test'}
    return out|{other[w] for w in out if w in other}


def specks(mask):
    labels, n = ndimage.label(mask, np.ones((3, 3)))
    if not n:
        return 1.
    sizes = np.bincount(labels.ravel())[1:]; width = max(2., 2*float(np.median(ndimage.distance_transform_edt(mask)[mask])))
    return float(sizes[sizes < max(12, 2*width*width)].sum())/max(1, sizes.sum())


def line_at(lines, page, box):
    """The catalogue line on the page that holds most of a box (its pixel box), or None."""
    best, found = 0, None
    for line in lines.get(page, []):
        l, t, r, b = line['px']; over = max(0, min(r, box[2])-max(l, box[0]))*max(0, min(b, box[3])-max(t, box[1]))
        if over > best:
            best, found = over, line['px']
    return found


def shape_of(unit, frame):
    """Where a unit lies in its frame (body heights): the box of its ink, and a line's ends (for D's drawing)."""
    ys, xs = np.nonzero(unit['repaired'] >= INK_ALPHA); u, v = S.in_frame(ys, xs, frame)
    out = {'extent': [float(u.min()), float(v.min()), float(u.max()), float(v.max())]}
    if unit['segments']:
        g = unit['segments'][0]; t = g['t0']+np.array([0, len(g['ribbon'])-1])
        p = [g['mean']+x*g['a']+np.polyval(g['coef'], x)*g['c'] for x in t]
        e = [S.in_frame(np.array([q[1]]), np.array([q[0]]), frame) for q in p]
        out['ends'] = [[float(e[0][0][0]), float(e[0][1][0])], [float(e[1][0][0]), float(e[1][1][0])]]
    return out


def placed_unit(unit, grey, writing, base):
    """A unit's place on the writing it struck, or the reason it is left out."""
    if unit['kind'] == 'blob' and (unit['holes'] > HOLES or specks(unit['repaired'] >= INK_ALPHA) > FRAGMENTS):
        return 'holes' if unit['holes'] > HOLES else 'specks'
    found = S.struck(writing, unit['repaired'] >= INK_ALPHA)
    if found is None:
        return 'not_on_writing'
    frame = found[0]
    why = S.crowded(writing, found[1], frame) if base['source'] == 'P' else None    # the annotator's mask holds every stroke of H's
    if why:
        return why
    here = S.measures(unit['repaired'], writing, frame)
    if here['press'] < PRESSED:
        return 'not_on_writing'                           # a line of a figure, a mark above the writing
    if here['height'] is None or not 0 <= here['height'] <= 1:
        return 'not_across_the_body'
    ys, xs = np.nonzero(unit['repaired'] >= INK_ALPHA); u, v = S.in_frame(ys, xs, frame)
    if max(np.ptp(u), np.ptp(v)) < S.TINY:
        return 'tiny'
    pen = S.pen_width(writing & np.isin(S.pieces(writing)[1], found[1]))
    return {**base, 'kind': unit['kind'], 'frame': frame, 'writing_pen': pen, 'measures': here, 'crosses_rows': bool(found[2]), 'width': S.pen_width(unit['repaired'] >= INK_ALPHA),
            'strokes': len(unit['segments']), 'clean': round(min([g['clean'] for g in unit['segments']] or [1.]), 3),
            **shape_of(unit, frame), 'arrays': {'origin': grey, 'writing': writing, 'repaired': unit['repaired'], 'before': unit['before'],
                                                 'reference': unit['reference'], 'ribbons': [g['ribbon'] for g in unit['segments']],
                                                 'segments': unit['segments']}}


def human_units(set_dir, readable, files, lines, excluded):
    """Source H: the units of the strikes the annotator erased in the prefill set's training regions (``one_stroke.units_of``), on a
    window of the clean page round the region and its catalogue line (the region's own crop and mask put in it)."""
    rows = json.loads((set_dir/'training_native.json').read_text()); arrays = np.load(set_dir/'training_native.npz')
    made, left, opened = [], Counter(), {}
    for k, r in enumerate(rows):
        if r['group_uuid'] in excluded: left['excluded_writer'] += 1; continue
        if readable.get(r['id']) is False: left['unreadable'] += 1; continue
        mask = arrays[f'mask_{k}'] > 0
        if mask.sum() < 12: left['too_few_pixels'] += 1; continue
        l, t, rr, b = r['box']
        with Image.open(files[r['page']]) as im: size = im.size
        if (rr-l)*(b-t) > BIG*size[0]*size[1]: left['region_too_big'] += 1; continue
        if r['page'] not in opened:
            opened.clear()
            with Image.open(files[r['page']]) as im: opened[r['page']] = review_clean(np.asarray(im.convert('L')))
        page = opened[r['page']]; l, t, rr, b = r['box']; h = b-t
        line = line_at(lines, r['page'], r['box'])
        if line is None or line[3]-line[1] > ROW*h: line = r['box']     # a catalogue "line" round a whole answer is no row
        L, T = max(0, min(l, line[0])-h), max(0, min(t, line[1])-h//2)
        R, B = min(page.shape[1], max(rr, line[2])+h), min(page.shape[0], max(b, line[3])+h//2)
        grey = page[T:B, L:R].copy(); grey[t-T:b-T, l-L:rr-L] = arrays[f'image_{k}']
        strike = np.zeros(grey.shape, bool); strike[t-T:b-T, l-L:rr-L] = mask
        level = ink_level(grey, strike); units, why = S.units_of(grey, strike, level)
        left.update(f'stroke_{w}' for w in why)
        if not units: left['no_unit'] += 1; continue
        writing = S.writing_of(grey, strike)
        for n, unit in enumerate(units):
            base = {'id': f"H:{r['id']}#{n}", 'source': 'H', 'writer': r['group_uuid'], 'page': r['page'], 'window': [L, T, R, B], 'level': int(level)}
            got = placed_unit(unit, grey, writing, base)
            if isinstance(got, str): left[got] += 1
            else: made.append(got)
    return made, dict(left)


def beside(frame, offset, boxes):
    """Whether another mark (a box on the page) lies on the body of the writing a unit struck (``frame`` on a window at ``offset``):
    paper 1's bank holds one stroke of a mark, its other strokes are not writing."""
    c, s = np.cos(frame['theta']), np.sin(frame['theta']); ax, ay = frame['anchor']; h = frame['height']
    xs = [ax+u*c-v*s+offset[0] for u in (0, frame['span']) for v in (0, h)]; ys = [ay+u*s+v*c+offset[1] for u in (0, frame['span']) for v in (0, h)]
    l, t, r, b = min(xs), min(ys), max(xs), max(ys)
    return any(min(r, q[2])-max(l, q[0]) > 0 and min(b, q[3])-max(t, q[1]) > 0 for q in boxes)


def bank_units(bank_dir, files, excluded, other):
    """Source P: paper 1's bank lines of writers not excluded, regrown on their page (``one_stroke.regrown``) and repaired (a line
    breaking on paper gives a unit for each piece long enough); left out where another of paper 1's marks lies on the writing it
    struck (``beside``), or that writing holds another strike or a smear (``one_stroke.crowded``)."""
    pieces = json.loads((bank_dir/'pieces.json').read_text()); arrays = np.load(bank_dir/'bank.npz')
    marks = {}                                    # every gold mark of paper 1 on a page, kept in the bank or not (its box)
    for q in pieces:
        marks.setdefault(q['image_id'], []).append((q['id'], q['box']))
    for q in json.loads((bank_dir/'left_out.json').read_text()):
        marks.setdefault(q['image_id'], []).append((None, q['window']))
    made, left, opened = [], Counter(), {}
    for p in pieces:
        if canonical(p['writer'], other) in excluded or p['writer'] in excluded: left['excluded_writer'] += 1; continue
        if p['image_id'] not in opened:
            opened.clear()
            with Image.open(files[p['image_id']]) as im: opened[p['image_id']] = np.asarray(im.convert('L'))
        page = opened[p['image_id']]; l, t, r, b = p['box']; m = p['line_height']
        L, T, R, B = max(0, l-m), max(0, t-m), min(page.shape[1], r+m), min(page.shape[0], b+m)
        grey = page[T:B, L:R]; piece = np.zeros(grey.shape, bool); piece[t-T:b-T, l-L:r-L] = arrays[p['id']] < 255
        own = S.regrown(grey, piece)
        if own.sum() < 6: left['no_ink_on_page'] += 1; continue
        level = ink_level(grey, own); line = own & ~S.crossed(grey, own)
        segments, why = S.ribbon(grey, own, line, level)
        left.update(f'stroke_{w}' for w in why)
        if not segments: left['no_unit'] += 1; continue
        writing = S.writing_of(grey, line)
        for n, g in enumerate(segments):
            unit = {'kind': 'line', 'repaired': S.repaired([g], opacity(grey, level, own), grey.shape), 'before': opacity(grey, p['ink_level'], piece), 'reference': piece,
                    'regrown': own, 'segments': [g], 'width': g['width'], 'holes': 0.}
            base = {'id': f"P:{p['id']}#{n}", 'source': 'P', 'writer': canonical(p['writer'], other), 'page': p['image_id'], 'window': [L, T, R, B], 'level': int(level)}
            got = placed_unit(unit, grey, writing, base)
            if not isinstance(got, str) and got['frame']['height'] > TALL*p['line_height']:
                got = 'not_a_row_of_writing'                  # a figure's or table's line: what it lies on is taller than paper 1's line
            if not isinstance(got, str) and beside(got['frame'], (L, T), [b for i, b in marks[p['image_id']] if i != p['id']]):
                got = 'another_strike'
            if isinstance(got, str): left[got] += 1
            else: got['arrays']['regrown'] = own; made.append(got)
    return made, dict(left)


def drawn_unit(rng, n, human):
    """Source D: a unit drawn in one of ``DRAWN``'s styles where an H unit of its kind lay (its frame, width and shape, in body heights),
    for writing ``NOMINAL`` high; its measures those of it put at that unit's place on that unit's writing."""
    from .paste_strikes import strokes
    style = rng.choices(list(DRAWN), weights=list(DRAWN.values()))[0]; kind = KIND_OF[style]
    pool = [m for m in human if m['kind'] == kind] or [m for m in human if m['kind'] == 'line']
    like = rng.choice(pool); f = like['frame']; h = NOMINAL; super_ = 3
    u0, v0, u1, v1 = like['extent']; pad = h
    anchor = (pad-u0*h, pad-v0*h); size = (int((u1-u0)*h+2*pad), int((v1-v0)*h+2*pad))
    width = max(2., like['width']/f['height']*h)
    at = lambda x, y: ((anchor[0]+x*h)*super_, (anchor[1]+y*h)*super_)
    big = Image.new('L', (size[0]*super_, size[1]*super_), 0); draw = ImageDraw.Draw(big)
    line = lambda points: draw.line([at(*q) for q in points], fill=255, width=int(round(width*super_)), joint='curve')
    if style in ('line', 'wave'):
        (xa, ya), (xb, yb) = like.get('ends') or [[u0, (v0+v1)/2], [u1, (v0+v1)/2]]
        steps = np.linspace(0, 1, 60); length = np.hypot(xb-xa, yb-ya); nx, ny = -(yb-ya)/max(1e-6, length), (xb-xa)/max(1e-6, length)
        if style == 'line':
            bow = rng.uniform(-.03, .03)*length; off = bow*4*steps*(1-steps)
        else:
            off = rng.uniform(.08, .18)*np.sin(2*np.pi*steps*length/rng.uniform(.3, .6)+rng.uniform(0, 2*np.pi))
        line([(xa+s*(xb-xa)+o*nx, ya+s*(yb-ya)+o*ny) for s, o in zip(steps, off)])
    elif style == 'double':
        for y in (v0+.3*(v1-v0), v0+.7*(v1-v0)):
            tilt = rng.uniform(-.05, .05)*(u1-u0); line([(u0, y-tilt), (u1, y+tilt)])
    elif style == 'cross':
        line([(u0, v0), (u1, v1)]); line([(u0, v1), (u1, v0)])
    elif style == 'zigzag':
        k = rng.randint(6, 10); line([(x, (v0, v1)[i % 2]+rng.uniform(-.05, .05)) for i, x in enumerate(np.linspace(u0, u1, k+1))])
    alpha = np.asarray(big.resize(size, Image.LANCZOS), np.float32)/255.
    if style == 'scribble':
        w, hh = max(4, int((u1-u0)*h)), max(4, int((v1-v0)*h))
        _, a = strokes((w, hh), 'scribble', rng, (0, 0, 0), width, 0)
        x0, y0 = int(round(anchor[0]+u0*h)), int(round(anchor[1]+v0*h)); alpha[y0:y0+a.shape[0], x0:x0+a.shape[1]] = a[:size[1]-y0, :size[0]-x0]
    frame = {'theta': 0., 'anchor': list(anchor), 'height': float(h), 'span': f['span']/f['height']*h, 'chars': f['chars']}
    there = S.warp(alpha, S.carry(frame, f), like['arrays']['origin'].shape)          # as if drawn at its H unit's place, on its writing
    return {'id': f'D:{n}', 'source': 'D', 'writer': None, 'style': style, 'kind': kind, 'frame': frame, 'crosses_rows': like['crosses_rows'],
            'measures': S.measures(there, like['arrays']['writing'], f),
            'width': width, 'drawn': True, 'like': like['id'], 'strokes': 0, 'clean': 1., 'level': 0,
            'arrays': {'origin': None, 'repaired': alpha, 'before': alpha, 'reference': None, 'ribbons': [], 'segments': []}}


def clean_line(page, box, others):
    """Why a host line's box is not clean, or None: another line's box on the page overlaps it where there is ink, or a piece of ink
    it holds part of has its centre outside it (a neighbour's stroke reaching in, or its own reaching out)."""
    l, t, r, b = box; ink = page < INK_BELOW
    for o in others:
        x0, y0, x1, y1 = max(l, o[0]), max(t, o[1]), min(r, o[2]), min(b, o[3])
        if x1 > x0 and y1 > y0 and ink[y0:y1, x0:x1].any():
            return 'boxes_overlap'
    h = b-t; L, T, R, B = max(0, l-h), max(0, t-h), min(page.shape[1], r+h), min(page.shape[0], b+h)
    labels, _ = ndimage.label(ink[T:B, L:R], np.ones((3, 3)))
    inside = np.zeros(labels.shape, bool); inside[t-T:b-T, l-L:r-L] = True
    for k in np.unique(labels[inside & (labels > 0)]):
        ys, xs = np.nonzero(labels == k)
        if len(ys) >= S.SPECK and not (t <= ys.mean()+T < b and l <= xs.mean()+L < r):
            return 'piece_across_the_box'
    return None


def host_pool(catalogue, files, excluded, other, templates, page_lines):
    """``POOL`` host lines drawn writer by writer (seed 42) from the clean ones: lines a person looked at one by one and found without a
    revision (the catalogue's ``label_origin`` ``human_individual_review``, ``has_edit`` false), of text or an equation, of allowed
    writers, whose box is clean (``clean_line``) and whose writing ``one_stroke.host_row`` finds.  Each: the clean page round it (its
    height above and below, as much to each side), its template, the height of its writing and the boxes of the page's lines with a
    revision (``has_edit``) in the window.  Also the count of lines and writers left after each condition."""
    steps = {}
    def count(name, rows):
        steps[name] = {'lines': len(rows), 'writers': len({canonical(r['group_uuid'], other) for r in rows})}
    lines = [r for r in catalogue if not r.get('has_edit') and canonical(r['group_uuid'], other) not in excluded
             and r['group_uuid'] not in excluded and r['image_id'] in files]
    count('without_revision_allowed_writers', lines)
    lines = [r for r in lines if r.get('label_origin') == 'human_individual_review']; count('looked_at_by_a_person', lines)
    lines = [r for r in lines if r.get('ocr_type') in ('text', 'equation')]; count('text_or_equation', lines)
    clean, why, by_page = [], Counter(), {}
    for r in lines:
        by_page.setdefault(r['image_id'], []).append(r)
    for page_id in sorted(by_page):
        with Image.open(files[page_id]) as im: page = review_clean(np.asarray(im.convert('L')))
        for r in by_page[page_id]:
            box = crop_box(r['box_1000'], page.shape[1], page.shape[0])
            others = [e['px'] for e in page_lines.get(page_id, []) if e is not r and e['px'] != box]
            reason = clean_line(page, box, others)
            if reason: why[reason] += 1
            else: clean.append(r)
    count('clean_box', clean)
    by_writer = {}
    for r in clean:
        by_writer.setdefault(canonical(r['group_uuid'], other), []).append(r)
    rng = random.Random(SEED); writers = sorted(by_writer); rng.shuffle(writers)
    for w in writers:
        rng.shuffle(by_writer[w])
    order = [by_writer[w][i] for i in range(max(map(len, by_writer.values()))) for w in writers if i < len(by_writer[w])]
    pool, opened = [], {}
    for r in order:
        if len(pool) >= POOL:
            break
        if r['image_id'] not in opened:
            if len(opened) > 32: opened.clear()
            with Image.open(files[r['image_id']]) as im: opened[r['image_id']] = review_clean(np.asarray(im.convert('L')))
        page = opened[r['image_id']]; l, t, rr, b = crop_box(r['box_1000'], page.shape[1], page.shape[0]); h = b-t
        L, T, R, B = max(0, l-h), max(0, t-h), min(page.shape[1], rr+h), min(page.shape[0], b+h)
        grey = page[T:B, L:R].copy()
        row = S.host_row(grey, (l-L, t-T, rr-L, b-T))
        if row is None:
            why['no_row_found'] += 1; continue
        keys = [k for c in row['chars'] for k in c[2]]
        template = templates.get(group_key(r['image_id'], (page.shape[1], page.shape[0])))
        edited = [[e['px'][0]-L, e['px'][1]-T, e['px'][2]-L, e['px'][3]-T] for e in page_lines.get(r['image_id'], [])
                  if e.get('has_edit') and e['px'][2] > L and e['px'][0] < R and e['px'][3] > T and e['px'][1] < B]
        height = S.body(S.joined(row['found'], keys), row['theta'])[1]
        pool.append({'grey': grey, 'template': None if template is None else template[T:B, L:R].copy(), 'line': (l-L, t-T, rr-L, b-T),
                     'page': r['image_id'], 'writer': canonical(r['group_uuid'], other), 'window': [L, T, R, B],
                     'pen_per_height': row['pen']/height, 'height': height, 'edited': edited})
    writers_in_pool = Counter(h['writer'] for h in pool)
    steps['pool'] = {'lines': len(pool), 'writers': len(writers_in_pool),
                     'writer_max_share': round(max(writers_in_pool.values())/max(1, len(pool)), 4) if pool else None}
    return pool, steps, dict(why)


def put_unit(unit, pool, rng):
    """A unit put on up to ``TRIES`` hosts of the pool, drawn among those whose pen is like the unit's for their writing's height (the
    unit's width over its writing's body height, within ``one_stroke.PEN`` of theirs, and their writing at most ``one_stroke.MAX_SCALE``
    times as high: the checks ``one_stroke.put`` makes on the run it strikes) when there are ``TRIES`` such: (result with its host, or None; the reasons of the tries that failed)."""
    why = Counter(); mine = unit['width']/unit['frame']['height']
    like = [h for h in pool if S.PEN[0] <= mine/h['pen_per_height'] <= S.PEN[1] and h['height'] <= S.MAX_SCALE*unit['frame']['height']]
    like = like if len(like) >= TRIES else pool
    for _ in range(TRIES):
        host = rng.choice(like); row = S.host_row(host['grey'], host['line'])
        got, reason = S.put(unit, unit['arrays']['repaired'], {**host, 'row': row}, rng)
        if got is not None:
            return {**got, 'host': host}, why
        why[reason] += 1
    return None, why


# ---- the checks ------------------------------------------------------------------------------------------------------------------
def stroke_measures(unit):
    """Checks 1 and 2 on each stroke of a line or group: ``strike_bank.line_measures`` of the stroke alone, repaired."""
    if unit['source'] == 'D':
        return [line_measures(unit['arrays']['repaired'])] if unit['kind'] == 'line' else []
    return [line_measures(S.render(g, unit['arrays']['origin'].shape)) for g in unit['arrays']['segments']]


def put_back(unit, reference):
    """Check 3: the share of the reference pixels within a pixel of the repaired unit, and of the unit's pixels where its page has no ink."""
    on = unit['arrays']['repaired'] >= INK_ALPHA; ref = reference
    covered = float((ref & ndimage.binary_dilation(on)).sum())/max(1, ref.sum())
    paper = float((on & (unit['arrays']['origin'] >= INK_BELOW)).sum())/max(1, on.sum())
    return covered, paper


def summary_of(values, keys):
    if not values:
        return {'n': 0}
    a = np.array(values, float)
    return {'n': len(values), **{k: round(float(f(a)), 3) for k, f in keys.items()}}


def rendered(alpha, level=40):
    return np.rint(255-(255-level)*np.clip(alpha, 0, 1)).astype(np.uint8)


def png64(array, scale):
    image = Image.fromarray(array); image = image.resize((max(1, image.width*scale), max(1, image.height*scale)), Image.NEAREST)
    buffer = io.BytesIO(); image.save(buffer, 'PNG'); return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def overlay(grey, alpha, colour=(220, 30, 30), box=None):
    """``grey`` with ``alpha`` laid on it in a colour (and a box drawn), RGB."""
    rgb = np.stack([grey]*3, -1).astype(np.float32)
    if alpha is not None:
        a = np.clip(alpha, 0, 1)[..., None]*.75; rgb = rgb*(1-a)+np.array(colour, np.float32)*a
    out = Image.fromarray(np.rint(rgb).astype(np.uint8))
    if box is not None:
        ImageDraw.Draw(out).rectangle([box[0], box[1], box[2]-1, box[3]-1], outline=(30, 90, 230))
    return np.asarray(out)


def look_page(chosen, path, key):
    """The page of the human check: per unit its origin, the repaired unit (its ribbons straightened, and put back in red on its
    origin), and put on its host (the region's box in blue), all three times; two boxes to tick (material, placement) kept in the
    browser, counted at the top and exportable."""
    rows = []
    for m in chosen:
        a = m['arrays']; scale = 3
        origin = png64(a['origin'], scale) if a['origin'] is not None else None
        if a['ribbons']:
            strips = [rendered(r.T) for r in a['ribbons']]                          # a ribbon is sections x across: shown lying down
            w = max(s.shape[1] for s in strips)
            straight = np.concatenate([np.pad(s, ((2, 2), (0, w-s.shape[1])), constant_values=255) for s in strips], 0)
        else:
            ys, xs = np.nonzero(a['repaired'] > 0)
            straight = rendered(a['repaired'][ys.min():ys.max()+1, xs.min():xs.max()+1]) if len(ys) else None
        back = png64(overlay(a['origin'], a['repaired']), scale) if a['origin'] is not None else None
        put = m.get('put')
        if put:
            host = put['host']; L, T, R, B = put['box']; g = host['grey'].copy(); g[T:B, L:R] = put['image']
            m0 = int(put['frame']['height']); y0, y1 = max(0, T-m0), min(g.shape[0], B+m0); x0, x1 = max(0, L-2*m0), min(g.shape[1], R+2*m0)
            shown = png64(overlay(g[y0:y1, x0:x1], None, box=(L-x0, T-y0, R-x0, B-y0)), scale)
        info = ' '.join(f'{k}={v}' for k, v in m['shown'].items())
        cells = [f'<img src="{origin}">' if origin else '（程序画的，没有原处）',
                 (f'<img src="{png64(straight, scale)}"><br>' if straight is not None else '')+(f'<img src="{back}">' if back else ''),
                 f'<img src="{shown}">' if put else f'没贴上：{html.escape(m["shown"].get("why", ""))}']
        rows.append(f'<tr data-id="{html.escape(m["id"])}"><td class="id">{html.escape(m["id"])}<br><small>{html.escape(info)}</small></td>'
                    + ''.join(f'<td>{c}</td>' for c in cells)
                    + '<td class="tick"><label><input type="checkbox" data-k="material">素材有问题</label><br>'
                      '<label><input type="checkbox" data-k="placement">贴错了</label></td></tr>')
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>合成素材检查</title>
<style>body{{font:14px sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse;width:100%;table-layout:fixed}}
td{{border:1px solid #ddd;padding:4px;vertical-align:middle;overflow:hidden}}td.id{{width:190px;word-break:break-all;font-size:11px}}td.tick{{width:110px}}
img{{image-rendering:pixelated;max-width:100%;display:block;margin:2px 0}}#top{{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #ccc;z-index:1}}
h2{{margin-top:28px}}</style>
<div id="top">勾选：素材有问题 / 贴错了。计数（按来源）：<span id="count"></span> <button id="export">导出勾选</button></div>
<p>每行（放大 3 倍）：原处｜补好的这一笔（上：拉直的带子，一笔一条；下：红色是放回原处的样子）｜贴到宿主上（蓝框是区域框）。
线、一组拉直补齐；团不拉直，只按第一版第 2 轮清洗。D 是程序画的，照一份 H 素材在原处的位置画。</p>"""
    for source in SOURCES:
        part = [r for r, m in zip(rows, chosen) if m['source'] == source]
        page += (f'<h2>来源 {source}（{len(part)} 条）</h2><table><tr><th class="id">素材</th><th>原处</th><th>补好的这一笔</th><th>贴到宿主上</th>'
                 f'<th class="tick">看出</th></tr>{"".join(part)}</table>')
    page += f"<script>const K='synth-materials-check-{key}';"+"""let s={};try{s=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
function count(){const c={};for(const tr of document.querySelectorAll('tr[data-id]')){const src=tr.dataset.id[0];c[src]=c[src]||{n:0,material:0,placement:0};c[src].n++;
for(const b of tr.querySelectorAll('input'))if(b.checked)c[src][b.dataset.k]++}
document.getElementById('count').textContent=Object.entries(c).map(([k,v])=>`${k}: 素材有问题 ${v.material}（${(100*v.material/v.n).toFixed(1)}%）、贴错了 ${v.placement}（${(100*v.placement/v.n).toFixed(1)}%）/ ${v.n}`).join('；')}
for(const tr of document.querySelectorAll('tr[data-id]'))for(const b of tr.querySelectorAll('input')){b.checked=!!(s[tr.dataset.id]||{})[b.dataset.k];
b.onchange=()=>{(s[tr.dataset.id]=s[tr.dataset.id]||{})[b.dataset.k]=b.checked;try{localStorage.setItem(K,JSON.stringify(s))}catch(e){}count()}}
document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(s,null,1)],{type:'application/json'}));a.download='synth_materials_check_'+K.slice(22)+'.json';a.click()};count()</script>"""
    path.write_text(page)


def materials(args):
    """See the module."""
    other = twins(args.same_student); excluded = excluded_writers(args.set, args.mapping, args.same_student)
    readable = {r['id']: r.get('readable') for r in json.loads(args.dataset.read_text())}
    catalogue = read_catalogue(args.catalogue); files = pages.page_files(args.pages)
    lines, sizes = {}, {}
    for r in catalogue:
        if r['image_id'] in files: lines.setdefault(r['image_id'], []).append(r)
    for page, rows in lines.items():
        with Image.open(files[page]) as im: sizes[page] = im.size
        for r in rows: r['px'] = crop_box(r['box_1000'], *sizes[page])
    human, human_left = human_units(args.set, readable, files, lines, excluded)
    bank, bank_left = bank_units(args.bank, files, excluded, other)
    rng = random.Random(SEED); drawn = [drawn_unit(rng, n, human) for n in range(LOOK)]
    out = make_output_dir(args.output_dir)
    store = {}
    for m in human+bank:
        for key in ('repaired', 'before'): store[f"{m['id']}|{key}"] = np.rint(np.clip(m['arrays'][key], 0, 1)*255).astype(np.uint8)
    np.savez_compressed(out/'materials.npz', **store)
    write_json(out/'materials.json', [{k: v for k, v in m.items() if k != 'arrays'} for m in human+bank])
    templates = dict(np.load(args.templates))
    pool, host_steps, host_left = host_pool(catalogue, files, excluded, other, templates, lines)
    chosen = [m for pool_ in (human, bank, drawn) for m in random.Random(SEED).sample(pool_, min(LOOK, len(pool_)))]
    pack_rows = []; (out/'reader'/'assets').mkdir(parents=True)
    checks = {s: {'fill': [], 'cover': [], 'covered': {}, 'paper': {}, 'put': 0, 'tried': 0, 'why': Counter(), 'tiers': Counter(),
                  'kinds': Counter(), 'hosts': Counter()} for s in SOURCES}
    for n, m in enumerate(chosen):
        c = checks[m['source']]; c['kinds'][m['kind']] += 1; c['tried'] += 1
        got, why = put_unit(m, pool, random.Random(f'{SEED}:{m["id"]}'))
        c['why'].update(why)
        if got:
            c['put'] += 1; c['tiers'][tier(got['strike_share'])] += 1; c['hosts'][got['host']['writer']] += 1; m['put'] = got
        if m['kind'] != 'blob':
            for x in stroke_measures(m):
                if 'fill' in x: c['fill'].append(x['fill']); c['cover'].append(x['cover'])
        if m['source'] != 'D':
            reference = m['arrays']['regrown'] if m['source'] == 'P' else m['arrays']['reference']   # P: its stroke found again (user 2026-11-17)
            covered, paper = put_back(m, reference)
            c['covered'].setdefault(m['kind'], []).append(covered); c['paper'].setdefault(m['kind'], []).append(paper)
            if m['source'] == 'P':
                covered, paper = put_back(m, m['arrays']['reference'])
                c['covered'].setdefault('line_bank_pixels', []).append(covered); c['paper'].setdefault('line_bank_pixels', []).append(paper)
        m['shown'] = {'kind': m['kind'], 'chars': m['frame']['chars'], **({'style': m['style'], 'like': m['like']} if 'style' in m else {}),
                      'clean': m['clean'], 'origin': {k: (round(v, 2) if v is not None else None) for k, v in m['measures'].items()},
                      **({'here': {k: (round(v, 2) if v is not None else None) for k, v in got['measures'].items()}, 'share': round(got['strike_share'], 2),
                          'scale': round(got['scale'], 2), 'stretch': round(got['stretch'], 2)} if got else {'why': ', '.join(f'{k}×{v}' for k, v in why.items())})}
        name = f"{m['source']}_{n}.png"
        a = m['arrays']['repaired']; ys, xs = np.nonzero(a >= .05)
        alone = rendered(a[ys.min():ys.max()+1, xs.min():xs.max()+1]) if len(ys) else np.full((8, 8), 255, np.uint8)
        Image.fromarray(np.pad(alone, 8, constant_values=255)).convert('RGB').save(out/'reader'/'assets'/name)
        pack_rows.append({'id': m['id'], 'line_id': m['id'], 'source': m['source'], 'kind': m['kind'], 'images': {'raw': f'assets/{name}'}})
    write_json(out/'reader'/'pack.json', {'schema': 'revrender-line-pack', 'seed': SEED, 'rows': pack_rows})
    look_page(chosen, out/'check_page.html', out.name)
    median, p95 = lambda a: np.median(a), lambda a: np.percentile(a, 95)
    result = {}
    for s, c in checks.items():
        hosts = c['hosts']
        result[s] = {'looked': c['tried'], 'kinds': dict(c['kinds']),
                     'check_1_fill': summary_of(c['fill'], {'median': median, 'p95': p95}),
                     'check_2_cover': summary_of(c['cover'], {'median': median, 'below_0.8': lambda a: (a < .8).mean()}),
                     'check_3_covered': {k: summary_of(v, {'median': median}) for k, v in c['covered'].items()},
                     'check_3_on_paper': {k: summary_of(v, {'median': median}) for k, v in c['paper'].items()},
                     'check_5_put': round(c['put']/max(1, c['tried']), 3), 'failed_tries': dict(c['why']), 'tiers': dict(c['tiers']),
                     'host_writers': len(hosts), 'host_writer_max_share': round(max(hosts.values())/max(1, sum(hosts.values())), 4) if hosts else None}
    produced = {s: {'units': len(pool_), 'writers': len({m['writer'] for m in pool_}), 'kinds': dict(Counter(m['kind'] for m in pool_)),
                    'kind_writers': {k: len({m['writer'] for m in pool_ if m['kind'] == k}) for k in ('line', 'group', 'blob')}} for s, pool_ in (('H', human), ('P', bank))}
    summary = {'checks': result, 'produced': produced, 'left_out': {'H': human_left, 'P': bank_left},
               'hosts': {'steps': host_steps, 'left_out': host_left},
               'excluded_writers': len(excluded), 'bars': CHECK}
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [args.set/'training_native.npz', args.bank/'bank.npz', args.dataset,
                   args.catalogue, args.mapping, args.same_student, args.templates]}, **summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def read_check(pack, predictions, output):
    """Check 4: per source and kind, the share of units (alone on white) a reader that was never taught read at least one letter, digit or
    Chinese character in (not the refusal, not empty; a LaTeX command's name is not read writing: ``\\cancel{}`` reads nothing).  Lines
    (a line or a group) of H and P against D's lines; blobs reported, not judged."""
    rows = {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows']}
    readings = [json.loads(s) for s in predictions.read_text().splitlines() if s.strip()]
    found, total = Counter(), Counter()
    for p in readings:
        r = rows[p['id']]; text = re.sub(r'\\[A-Za-z]+', '', (p.get('text') or '').replace('[读不出]', '').replace('[空]', ''))
        hit = bool(re.search(r'[0-9A-Za-z一-鿿]', text))
        for key in (f"{r['source']}_{r['kind']}", f"{r['source']}_lines" if r['kind'] != 'blob' else None):
            if key: total[key] += 1; found[key] += hit
    shares = {k: round(found[k]/total[k], 3) for k in total}
    control = shares.get('D_lines', 0.)
    out = {'read_share': shares, 'counts': dict(total), 'control_D_lines': control,
           'within_5_points_of_control': {s: shares[f'{s}_lines'] <= control+CHECK['read_over_control'] for s in ('H', 'P') if f'{s}_lines' in shares}}
    write_json(output, out); print(json.dumps(out), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('materials', description=materials.__doc__)
    for name in ('set', 'bank', 'dataset', 'catalogue', 'mapping', 'same-student', 'templates', 'output-dir'):
        m.add_argument('--'+name, type=Path, required=True)
    m.add_argument('--pages', type=Path, action='append', required=True, help='clean pages (hosts, H windows) and the scans the bank was cut from')
    r = sub.add_parser('read-check', description=read_check.__doc__)
    for name in ('pack', 'predictions', 'output'):
        r.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'materials':
        materials(a)
    else:
        read_check(a.pack, a.predictions, a.output)


if __name__ == '__main__':
    main()
