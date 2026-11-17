"""The checks of the synthetic materials beside ``data.synthesis`` (run through its command line): the row rule a host line's own ink is
taken by, examined against the annotator's polygons (plan of 2026-11-17, 行内墨迹归属_先考规则), and check 6, the strike's ink against
the writing's in a pasted set and in real regions (plan of 2026-11-17, 墨色对齐).
"""
import hashlib
import html
import json
import random
from collections import Counter

import numpy as np
from PIL import Image

from .. import pages
from ..eval.ink import INK_BELOW
from ..util import SEED, digest, make_output_dir, to_jsonable, write_json, write_manifest
from . import one_stroke as S
from .clean_pages import review_clean
from .lines import crop_box, read_catalogue
from .prior_annotations import twins
from .synthesis import SOURCES, canonical, overlay, png64


# ---- the row rule against the annotator's polygons (plan of 2026-11-17, 行内墨迹归属_先考规则) -------------------------------------
ROW_WRONG = (.80, .10)        # a region is wrong below this IoU, or with more than this share of what the rule took not the row's


def row_half(writer):
    """'debug' or 'exam' by the writer's hash (seed 42)."""
    return 'debug' if int(hashlib.sha256(f'row-rule:{SEED}:{writer}'.encode()).hexdigest()[:8], 16) % 2 == 0 else 'exam'


def adjusted(polygon):
    """Whether the annotator moved a region's outline: five points or more, or not an upright rectangle."""
    if len(polygon) != 4:
        return True
    xs, ys = sorted({x for x, _ in polygon}), sorted({y for _, y in polygon})
    return not (len(xs) == 2 and len(ys) == 2)


def row_check(args):
    """The rule that takes a host line's own ink (``one_stroke.host_row``: what synthesis round 4 strikes) against the annotator's
    polygons: per confirmed region off the test pages, on the window of its line's box (half its height more above and below) across
    the polygon's columns, the ink the rule takes against the ink inside the polygon (precision, recall, IoU); the line's whole box as
    the control.  One half of the writers (``row_half``) at a time."""
    from .page_labels import polygon_mask
    other = twins(args.same_student)
    regions = [r for r in json.loads(args.dataset.read_text()) if r.get('split') != 'test' and r.get('pixels_confirmed')]
    catalogue = read_catalogue(args.catalogue); files = pages.page_files(args.pages)
    by_page = {}
    for r in catalogue:
        by_page.setdefault(r['image_id'], []).append(r)
    rows, left = [], Counter()
    for page_id in sorted({r['page'] for r in regions}):
        mine = [r for r in regions if r['page'] == page_id and row_half(canonical(r['writer'], other)) == args.half]
        if not mine:
            continue
        if page_id not in files: left['no_page'] += len(mine); continue
        with Image.open(files[page_id]) as im: page = review_clean(np.asarray(im.convert('L')))
        boxes = [crop_box(x['box_1000'], page.shape[1], page.shape[0]) for x in by_page.get(page_id, [])]
        ink = page < INK_BELOW
        for r in mine:
            poly = np.array(r['region'], float); x0, y0 = poly.min(0); x1, y1 = poly.max(0)
            over = [max(0, min(b[2], x1)-max(b[0], x0))*max(0, min(b[3], y1)-max(b[1], y0)) for b in boxes]
            if not over or max(over) <= 0: left['no_line_box'] += 1; continue
            l, t, rr, b = boxes[int(np.argmax(over))]; h = b-t
            L, T, R, B = max(0, l-h), max(0, t-h), min(page.shape[1], rr+h), min(page.shape[0], b+h)
            row = S.host_row(page[T:B, L:R], (l-L, t-T, rr-L, b-T))
            taken = np.zeros(page.shape, bool)
            if row is not None:
                taken[T:B, L:R] = row['own']
            wy0, wy1 = max(0, int(t-h/2)), min(page.shape[0], int(b+h/2)); wx0, wx1 = int(x0)+2, int(np.ceil(x1))-2
            if wx1 <= wx0: left['too_narrow'] += 1; continue
            inside = polygon_mask(r['region'], (page.shape[1], page.shape[0]))[wy0:wy1, wx0:wx1]
            win = ink[wy0:wy1, wx0:wx1]; answer = win & inside
            if not answer.any(): left['no_ink_in_polygon'] += 1; continue
            control = np.zeros(win.shape, bool); control[max(0, t-wy0):b-wy0, :] = True
            def scores(said):
                said = said & win; hit = (said & answer).sum(); iou = hit/max(1, (said | answer).sum())
                extra = (said & ~answer).sum()/max(1, said.sum())
                return {'precision': float(hit/max(1, said.sum())), 'recall': float(hit/answer.sum()), 'iou': float(iou), 'extra': float(extra),
                        'wrong': bool(iou < ROW_WRONG[0] or extra > ROW_WRONG[1])}
            rule = scores(taken[wy0:wy1, wx0:wx1])
            rows.append({'id': r['id'], 'writer': canonical(r['writer'], other), 'adjusted': adjusted(r['region']), 'rule': rule,
                         'control': scores(control), 'window': [wx0, wy0, wx1, wy1], 'found_row': row is not None,
                         '_arrays': (page[wy0:wy1, wx0:wx1], answer, taken[wy0:wy1, wx0:wx1] & win)})
    def summary(sel, who):
        if not sel:
            return {'n': 0}
        v = {k: np.array([x[who][k] for x in sel]) for k in ('precision', 'recall', 'iou')}
        wrong = np.array([x[who]['wrong'] for x in sel], float); writers = sorted({x['writer'] for x in sel})
        idx = {w: [i for i, x in enumerate(sel) if x['writer'] == w] for w in writers}; rng = np.random.default_rng(SEED); boot = []
        for _ in range(1000):
            pick = np.concatenate([idx[w] for w in rng.choice(writers, len(writers))]); boot.append(wrong[pick].mean())
        return {'n': len(sel), 'writers': len(writers), 'wrong': round(float(wrong.mean()), 3),
                'wrong_95': [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                **{k: [round(float(np.percentile(a, q)), 3) for q in (25, 50, 75)] for k, a in v.items()}}
    result = {'half': args.half, 'regions': len(rows), 'writers': len({x['writer'] for x in rows}), 'left_out': dict(left),
              'bars': {'wrong_at_most': .10, 'iou_median_at_least': .90},
              **{group: {who: summary(sel, who) for who in ('rule', 'control')}
                 for group, sel in (('all', rows), ('adjusted', [x for x in rows if x['adjusted']]), ('rectangle', [x for x in rows if not x['adjusted']]))}}
    out = make_output_dir(args.output_dir)
    write_json(out/'regions.json', [{k: v for k, v in x.items() if k != '_arrays'} for x in rows])
    write_json(out/'summary.json', result)
    wrong = [x for x in rows if x['rule']['wrong']]
    cells = []
    for x in wrong:
        grey, answer, said = x['_arrays']
        user = overlay(grey, answer.astype(np.float32), colour=(30, 160, 30))
        rgb = np.stack([grey]*3, -1).copy(); rgb[said & ~answer] = (220, 30, 30); rgb[answer & ~said] = (30, 90, 230)
        cells.append(f'<tr><td class="id">{html.escape(x["id"])}<br><small>IoU {x["rule"]["iou"]:.2f} 多拿 {x["rule"]["extra"]:.2f} '
                     f'（整框：IoU {x["control"]["iou"]:.2f}）</small></td>' + ''.join(f'<td><img src="{png64(a, 3)}"></td>' for a in (grey, user, rgb)) + '</tr>')
    (out/'wrong_page.html').write_text(f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>行内墨迹归属：分错的区域</title>
<style>body{{font:14px sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse}}td{{border:1px solid #ddd;padding:4px;vertical-align:middle}}
td.id{{width:180px;word-break:break-all;font-size:11px}}img{{image-rendering:pixelated;max-width:360px;max-height:260px}}</style>
<p>{args.half} 半分错的区域 {len(wrong)} / {len(rows)}。每行：窗口原图｜用户的范围（绿）｜规则的回答（红 = 多拿，蓝 = 漏拿，黑 = 拿对）。</p>
<table>{''.join(cells)}</table>""")
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.dataset, args.catalogue, args.same_student)}, **result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


INK_CHECK = {50: 3, 25: 5, 75: 5, 5: 10, 95: 10}   # check 6: a synthetic set's difference percentile within this of the real one's


SEEN = 20                     # a strike pixel is seen when the paste made it this much darker than the host (or than paper)
SEEN_LEAST = 30               # a region's strike is seen with at least this many such pixels and half its pixels
UNSEEN_AT_MOST = .01          # check 6 (added 2026-11-17): at most this share of the regions with their strike not seen


def crops(paste_dir, arm):
    """(region, image, target) of each region of a pasted set, its crop put back together from its 256 tiles
    (``restoration_pairs.canvas_tiles``: centred on white when shorter, stride 192)."""
    from .restoration_pairs import SIZE
    rows = json.loads((paste_dir/arm/'records.json').read_text()); regions = {r['id']: r for r in json.loads((paste_dir/'regions.json').read_text())}
    x = np.load(paste_dir/arm/'gold_pixels.npy', mmap_mode='r'); y = np.load(paste_dir/arm/'targets.npy', mmap_mode='r')
    by_region = {}
    for i, r in enumerate(rows):
        by_region.setdefault(r['id'].split('~')[0], []).append(i)
    at = lambda n: sorted(set(list(range(0, n-SIZE+1, 192))+[n-SIZE]))
    for rid, idx in by_region.items():
        L, T, R, B = regions[rid]['box_on_page']; h, w = B-T, R-L; ph, pw = max(SIZE, h), max(SIZE, w); oy, ox = (ph-h)//2, (pw-w)//2
        image, target = np.full((ph, pw), 255, np.uint8), np.full((ph, pw), 255, np.uint8)
        for i, (ty, tx) in zip(idx, [(a, b) for a in at(ph) for b in at(pw)]):
            image[ty:ty+SIZE, tx:tx+SIZE] = x[i]; target[ty:ty+SIZE, tx:tx+SIZE] = y[i]
        yield regions[rid], image[oy:oy+h, ox:ox+w], target[oy:oy+h, ox:ox+w]


def seen(image, reference, strike, within=None):
    """Whether a strike shows: at least ``SEEN_LEAST`` pixels (of ``within``, all without), and as many as half the strike's, ``SEEN``
    darker than ``reference``."""
    dark = image.astype(int) < reference.astype(int)-SEEN
    dark = int((dark if within is None else dark & within).sum())
    return dark >= SEEN_LEAST and dark >= strike.sum()/2


def unseen(paste_dir, arm):
    """The regions of a pasted set whose strike does not show (``seen``: the crop's pixels the paste made darker than the host, against
    the host's paper pixels it darkened at all, a little more than ``remove``), by source, kind and tier."""
    out = {}
    for region, image, target in crops(paste_dir, arm):
        strike = (target >= 235) & (image < target)
        miss = not seen(image, target, strike)
        for key in (region['source'], f"{region['source']}_{region['kind']}", f"{region['source']}_{region['tier']}", 'all'):
            out.setdefault(key, [0, 0]); out[key][0] += 1; out[key][1] += miss
    return {k: {'regions': v[0], 'not_seen': v[1], 'share': round(v[1]/max(1, v[0]), 4)} for k, v in sorted(out.items())}


def tile_differences(paste_dir, arm):
    """Per source, the strike-less-writing core grey difference of each region of a pasted set kept without opacity (the set pasted
    before the ink was matched): the strike the host's paper pixels the paste darkened, the writing the host's ink it left as it was."""
    out = {s: [] for s in SOURCES}
    for region, image, target in crops(paste_dir, arm):
        d = S.ink_difference(image, (target >= 235) & (image < target), (target < INK_BELOW) & (image == target))
        if d is not None:
            out[region['source']].append(d)
    return out


def ink_check(args):
    """Check 6 (plan of 2026-11-17, 墨色对齐): the strike-less-writing core grey differences of a pasted set against the real ones, all
    sources together within ``INK_CHECK`` at the 5th to 95th percentiles (H, P and D also apart), for the new set and the set before;
    and a page of 100 regions (seed 42) at three times their size, without and with their box."""
    arrays = np.load(args.set/'training_native.npz'); rows = json.loads((args.set/'training_native.json').read_text())
    real = S.real_differences([arrays[f'image_{k}'] for k in range(len(rows))], [arrays[f'mask_{k}'] > 0 for k in range(len(rows))])
    quantiles = lambda d: {q: round(float(np.percentile(d, q)), 1) for q in INK_CHECK} if d else None
    want = quantiles(real)
    def judged(d):
        got = quantiles(d)
        return {'regions': len(d), 'percentiles': got, 'off': {q: round(got[q]-want[q], 1) for q in INK_CHECK},
                'pass': all(abs(got[q]-want[q]) <= INK_CHECK[q] for q in INK_CHECK)} if d else None
    result = {'real': {'regions': len(real), 'percentiles': want}, 'bars': INK_CHECK}
    regions = json.loads((args.paste/'regions.json').read_text())
    for arm, key in (('S', 'ink_difference'), ('S0', 'ink_difference_plain')):
        per = {s: [r[key] for r in regions if r['source'] == s and r[key] is not None] for s in SOURCES}
        result[f'new_{arm}'] = {'all': judged([v for s in SOURCES for v in per[s]]), **{s: judged(per[s]) for s in SOURCES},
                                'measured': sum(r[key] is not None for r in regions), 'not_measured': sum(r[key] is None for r in regions),
                                'not_measured_why': f'fewer than {S.CORE_LEAST} ink pixels of the strike on paper or of the writing it leaves untouched'}
        result[f'unseen_{arm}'] = unseen(args.paste, arm)
    real_seen = [seen(arrays[f'image_{k}'], np.full(arrays[f'image_{k}'].shape, 255, np.uint8), arrays[f'mask_{k}'] > 0, arrays[f'mask_{k}'] > 0)
                 for k in range(len(rows))]
    result['unseen_real'] = {'regions': len(real_seen), 'not_seen': int(len(real_seen)-sum(real_seen)),
                             'share': round(1-sum(real_seen)/max(1, len(real_seen)), 4), 'reference': 'paper (255): the annotator erased pixels'}
    result['unseen_bar'] = UNSEEN_AT_MOST
    if args.before:
        result['unseen_before_S'] = unseen(args.before, 'S'); result['unseen_before_S0'] = unseen(args.before, 'S0')
    if args.before:                     # the set before has no opacity kept: its writing is the host's ink the paste left as it was
        for arm in ('S', 'S0'):
            per = tile_differences(args.before, arm)
            result[f'before_{arm}'] = {'all': judged([v for s in SOURCES for v in per[s]]), **{s: judged(per[s]) for s in SOURCES},
                                       'writing': 'host ink left unchanged (tiles; no opacity kept)'}
    result['pass'] = bool(result['new_S']['all']['pass'] and result['new_S0']['all']['pass'] and result['unseen_S']['all']['share'] <= UNSEEN_AT_MOST)
    out = make_output_dir(args.output_dir); write_json(out/'ink_check.json', result)
    chosen = {r['id'] for r in random.Random(SEED).sample(regions, min(100, len(regions)))}
    pasted = {r['id']: image for r, image, _ in crops(args.paste, 'S') if r['id'] in chosen}
    files = pages.page_files(args.pages); cells = []
    for r in (x for x in regions if x['id'] in chosen):
        with Image.open(files[r['host_page']]) as im: page = review_clean(np.asarray(im.convert('L')))
        L, T, R, B = r['box_on_page']; page = page.copy(); page[T:B, L:R] = pasted[r['id']]
        m = max(B-T, 20); y0, y1, x0, x1 = max(0, T-m), min(page.shape[0], B+m), max(0, L-m), min(page.shape[1], R+m)
        view = page[y0:y1, x0:x1]
        cells.append(f'<tr><td class="id">{html.escape(r["id"])} {r["source"]} {r["kind"]} {r["tier"]}</td><td><img src="{png64(view, 3)}"></td>'
                     f'<td><img src="{png64(overlay(view, None, box=(L-x0, T-y0, R-x0, B-y0)), 3)}"></td></tr>')
    (out/'ink_page.html').write_text(f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>合成划线墨色检查</title>
<style>body{{font:14px sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse}}td{{border:1px solid #ddd;padding:4px;vertical-align:middle}}
td.id{{width:160px;word-break:break-all;font-size:11px}}img{{image-rendering:pixelated;max-width:620px}}</style>
<p>随机 100 处（种子 42），原始分辨率放大 3 倍：左边不画框，右边画出区域框。检查 ⑥：{'过' if result['pass'] else '没过'}。</p><table>{''.join(cells)}</table>""")
    write_manifest(out, config=to_jsonable(args), inputs={str(args.paste/'regions.json'): digest(args.paste/'regions.json')}, **{k: v for k, v in result.items()})
    print(json.dumps(result, ensure_ascii=False), flush=True)
