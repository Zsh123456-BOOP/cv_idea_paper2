"""From which block does the strike reach the tokens it does not cover?  The offline measurement of the plan of 2026-11-22
(划线影响的逐层扩散): forward passes of Qwen3-VL-8B's frozen tower (and, reported only, each fold's V-修 adapter) -- nothing taught,
nothing read.

    python -m revrender.eval.strike_spread measure --model MODEL --pack VIEWS --raw-pack RAW_VIEWS --students PATTERN --records PATTERN
        [--threads 8] [--limit N] --output-dir OUT
    python -m revrender.eval.strike_spread summary --measured OUT --pack VIEWS --shares SHARES --readings H=DIR[,DIR...]
        --readings V=DIR[,DIR...] --output-dir OUT2

Per region, two pairs: the window (the pack's raw crop -> its erased crop; the strike the pack's ``map``, the human mask, which is the
pixels the two differ in) and the line (the raw pack's context -> the pack's; the strike the pixels they differ in, the window's
placed).  x the scan's features, c the erased image's, the distance 1 - cos(x, c).  The places: the input of the first block (the patch
embedding with its positions), every block's output (a token per patch, in the processor's 2 x 2 merge-group order), the merged tokens
and the three deepstack levels (a token per 32-pixel cell, in row order).

Each patch (cell) classed by the strike mask carried to the processor's resized image (a patch covers the source rectangle its 16
resized pixels -- 32 for a cell -- come from): struck (a mask pixel inside), next to struck (not struck, one of its eight neighbours
struck), far (the rest); the classes put in the tokens' order (``models.maa.merge_order``).  Per region, place and class: the tokens, the
mean distance, the mean energy ratio |x - c|^2 / |c|^2, the summed |x - c|^2 (OUT/regions.jsonl); on the four outputs also the mean
cosine of x and of the region's own fold's V-修 adapter on the scan (s) to c.  ``summary``: the spread ratio U = (the regions' mean of
the unstruck tokens' mean distance) / (the same of the struck), the regions with both classes; a writer bootstrap (1,000, seed 42,
95 %); the rule (the window, the readable regions): after 9 blocks (block 8's output, where deepstack 1 is taken) U's upper end at most
0.25 -> 局部修有位置; after 3 blocks (block 2's output) U's lower end at least 0.5 -> 放下; else 其余.
Checks (OUT/checks.json): 1 one grid for both images, every region; 2 the first block's input differs (x != c) at exactly the patches
whose input pixels differ; 3 the far patches' pixels the same in both images, and the share of the next-to-struck patches whose pixels
differ; 4 the merged tokens' raw closeness (the touched tokens, as ``train_clean_view distance``) within 0.005 of V-修's records."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

CLASSES = ('struck', 'next', 'far')
OUTPUTS = ('merged', 'deepstack1', 'deepstack2', 'deepstack3')


def classes(mask, rows, cols):
    """[rows, cols] 0 struck / 1 next / 2 far: a cell struck when the source rectangle it is resized from holds a mask pixel."""
    h0, w0 = mask.shape
    ys, xs = np.nonzero(mask)
    struck = np.zeros((rows, cols), bool)
    if len(ys):
        # a source pixel [y, y + 1) falls in the cells whose source rectangles [i h0 / rows, (i + 1) h0 / rows) it overlaps
        top = np.floor(ys*rows/h0).astype(int); bottom = np.ceil((ys+1)*rows/h0).astype(int)-1
        left = np.floor(xs*cols/w0).astype(int); right = np.ceil((xs+1)*cols/w0).astype(int)-1
        for t, b, l, r in zip(top, bottom, left, right):
            struck[t:b+1, l:r+1] = True
    near = np.zeros_like(struck)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(struck)
            shifted[max(dy, 0):rows+min(dy, 0), max(dx, 0):cols+min(dx, 0)] = struck[max(-dy, 0):rows+min(-dy, 0), max(-dx, 0):cols+min(-dx, 0)]
            near |= shifted
    return np.where(struck, 0, np.where(near, 1, 2))


def merge_ordered(grid, merge):
    """A [rows, cols] class grid in the tokens' 2 x 2 merge-group order (``models.maa.merge_order``)."""
    import torch
    from ..models.maa import merge_order
    rows, cols = grid.shape
    return merge_order(torch.as_tensor(grid).reshape(1, rows, cols, 1), merge).reshape(-1).numpy()


def per_class(x, c, labels):
    """Per class: tokens, mean distance, mean energy ratio, summed |x - c|^2 (float64)."""
    x, c = x.astype(np.float64), c.astype(np.float64)
    cos = (x*c).sum(1)/np.maximum(np.linalg.norm(x, axis=1)*np.linalg.norm(c, axis=1), 1e-300)
    diff = ((x-c)**2).sum(1); ratio = diff/np.maximum((c*c).sum(1), 1e-300)
    out = {}
    for k, name in enumerate(CLASSES):
        sel = labels == k
        out[name] = {'tokens': int(sel.sum()), 'distance': float((1-cos[sel]).mean()) if sel.any() else None,
                     'energy_ratio': float(ratio[sel].mean()) if sel.any() else None, 'energy': float(diff[sel].sum())}
    return out


def measure(args):
    import torch
    from ..models.maa import student_view
    from ..models.vlm_tokens import processor_for
    from ..train_clean_view import TOUCHED, levels, vision_tower
    torch.set_num_threads(args.threads)
    out = make_output_dir(args.output_dir)
    rows = json.loads((args.pack/'pack.json').read_text())['rows']
    folds = sorted({r['fold'] for r in rows})
    if args.limit:
        rows = [r for k in folds for r in [x for x in rows if x['fold'] == k][:args.limit]]
    raws = {r['id']: r for r in json.loads((args.raw_pack/'pack.json').read_text())['rows']}
    processor = processor_for(args.model); tower = vision_tower(args.model).float().eval()
    students = {k: Path(args.students.format(fold=k)) for k in folds}
    view = student_view(tower, {f'f{k}': p for k, p in students.items()}); view.eval()
    merge, patch = tower.spatial_merge_size, tower.config.patch_size
    places = ['embed', *(f'block{b}' for b in range(len(tower.blocks)))]
    captured = []
    tower.blocks[0].register_forward_pre_hook(lambda m, a, kw: captured.append(a[0].detach().float().numpy()) if a else None, with_kwargs=True)
    for block in tower.blocks:
        block.register_forward_hook(lambda m, i, o: captured.append(o.detach().float().numpy()))
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.raw_pack/'pack.json',
                   *(w for k in folds for w in sorted(students[k].glob('*.safetensors'))))}, touched_below=TOUCHED, regions=len(rows),
                   places=places+list(OUTPUTS), classes=CLASSES, dtype='float32', threads=args.threads)
    opened = lambda path: Image.open(path).convert('RGB')
    checks = {'check1_grids_differ': [], 'check2_mismatch': [], 'check3_far_pixels_differ': [], 'next_patches': 0, 'next_patches_pixels_differ': 0,
              'struck_patches': 0, 'struck_patches_pixels_same': 0}
    touched_raw = {}
    with (out/'regions.jsonl').open('x', encoding='utf-8') as stream:
        for n, r in enumerate(rows):
            k, q = r['fold'], raws[r['id']]
            pairs = {'window': (opened(args.pack/r['images']['raw']), opened(args.pack/r['images']['erased']),
                                np.asarray(Image.open(args.pack/r['images']['map']).convert('L')) > 0),
                     'line': (opened(args.raw_pack/q['images']['context']), opened(args.pack/r['images']['context']), None)}
            entry = {'id': r['id'], 'fold': k}
            for view_name, (scan, clean, mask) in pairs.items():
                if mask is None:
                    mask = (np.asarray(scan).astype(int) != np.asarray(clean).astype(int)).any(-1)
                batch = processor.image_processor(images=[scan, clean], return_tensors='pt')
                pv, g = batch['pixel_values'].float(), batch['image_grid_thw']
                if not torch.equal(g[0], g[1]):
                    checks['check1_grids_differ'].append([r['id'], view_name]); continue
                t, h, w = g[0].tolist(); size = t*h*w
                captured.clear()
                with torch.no_grad(), view.disable_adapter():
                    outs = levels(view, pv, g)
                at = {p: (a[:size], a[size:]) for p, a in zip(places, captured)}
                fine = merge_ordered(classes(mask, h, w), merge); coarse = classes(mask, h//merge, w//merge).reshape(-1)
                px, pc = pv[:size].numpy(), pv[size:].numpy(); pixels_differ = (px != pc).any(1)
                differs = (at['embed'][0] != at['embed'][1]).any(1)
                if not np.array_equal(differs, pixels_differ):
                    checks['check2_mismatch'].append([r['id'], view_name, int((differs != pixels_differ).sum())])
                far = fine == 2
                if pixels_differ[far].any():
                    checks['check3_far_pixels_differ'].append([r['id'], view_name, int(pixels_differ[far].sum())])
                checks['next_patches'] += int((fine == 1).sum()); checks['next_patches_pixels_differ'] += int(pixels_differ[fine == 1].sum())
                checks['struck_patches'] += int((fine == 0).sum()); checks['struck_patches_pixels_same'] += int((~pixels_differ[fine == 0]).sum())
                record = {'grid': [h, w], 'places': {p: per_class(a, b, fine) for p, (a, b) in at.items()}}
                m = h*w//merge**2
                view.set_adapter(f'f{k}')
                with torch.no_grad():
                    taught = levels(view, pv[:size], g[:1])
                for name, level, s in zip(OUTPUTS, outs, taught):
                    a, b, s = level[:m].numpy(), level[m:].numpy(), s.numpy()
                    e = per_class(a, b, coarse)
                    cos_s = (s.astype(np.float64)*b).sum(1)/np.maximum(np.linalg.norm(s, axis=1)*np.linalg.norm(b, axis=1), 1e-300)
                    cos_x = (a.astype(np.float64)*b).sum(1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-300)
                    for j, cls in enumerate(CLASSES):
                        sel = coarse == j
                        e[cls].update(cos_x=float(cos_x[sel].mean()) if sel.any() else None, cos_s=float(cos_s[sel].mean()) if sel.any() else None)
                    record['places'][name] = e
                    if name == 'merged' and view_name == 'window':
                        touched = cos_x < TOUCHED
                        touched_raw[r['id']] = float(cos_x[touched].mean()) if touched.any() else None
                record['class_shares'] = {'patches': {c: int((fine == j).sum()) for j, c in enumerate(CLASSES)}, 'cells': {c: int((coarse == j).sum()) for j, c in enumerate(CLASSES)}}
                entry[view_name] = record
            stream.write(json.dumps(entry)+'\n'); stream.flush()
            if n % 50 == 0: print(json.dumps({'regions': n+1, 'of': len(rows)}), flush=True)
    records = {}
    for k in folds:
        for line in (Path(args.records.format(fold=k))/'distances.jsonl').read_text().splitlines():
            g = json.loads(line); records[g['id']] = (g['window']['merged'] or {}).get('raw')
    ids = [i for i in touched_raw if touched_raw[i] is not None and records.get(i) is not None]
    this, rec = float(np.mean([touched_raw[i] for i in ids])), float(np.mean([records[i] for i in ids]))
    checks['check3_next_share_pixels_differ'] = checks['next_patches_pixels_differ']/max(1, checks['next_patches'])
    checks['check4_raw_closeness'] = {'this_run': this, 'records': rec, 'regions': len(ids), 'difference': this-rec, 'within_0.005': abs(this-rec) <= .005}
    checks['passed'] = (not checks['check1_grids_differ'] and not checks['check2_mismatch'] and not checks['check3_far_pixels_differ']
                        and checks['check4_raw_closeness']['within_0.005'])
    write_json(out/'checks.json', checks); print(json.dumps({k: v for k, v in checks.items() if not isinstance(v, list) or len(v) < 5}), flush=True)
    require(args.limit or checks['passed'], f'a check failed: {json.dumps(checks)[:600]}')
    write_json(out/'complete.json', {'regions': len(rows)})


def exact_by_region(spec, rows):
    from .transcription import counts, read_arm
    _, readings, _ = read_arm(spec)
    return {k: counts(rows[k]['text'], readings[k])['exact'] for k in readings if rows[k]['readable']}


def summary(args):
    out = make_output_dir(args.output_dir)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    half = {k: v.get('half') for k, v in json.loads(args.shares.read_text())['regions'].items()}
    readings = dict(spec.split('=', 1) for spec in args.readings)
    h, v = (exact_by_region(f'{n}={readings[n]}', rows) for n in ('H', 'V'))
    measured = [json.loads(l) for l in (args.measured/'regions.jsonl').read_text().splitlines()]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.measured/'regions.jsonl', args.pack/'pack.json', args.shares)},
                   bootstrap={'resamples': 1000, 'seed': SEED, 'by': 'writer'})
    readable = lambda i: rows[i]['readable']
    parts = {'readable': readable, 'heavy': lambda i: readable(i) and half.get(i) == 'heavy', 'light': lambda i: readable(i) and half.get(i) == 'light',
             'mode:erase': lambda i: readable(i) and rows[i]['mode'] == 'erase', 'mode:keep': lambda i: readable(i) and rows[i]['mode'] == 'keep',
             'H_right_V_wrong': lambda i: readable(i) and h.get(i) == 1 and v.get(i) == 0,
             'other': lambda i: readable(i) and not (h.get(i) == 1 and v.get(i) == 0), 'all': lambda i: True}
    places = list(measured[0]['window']['places'])

    def ratio(sel, place, view_, num, den, key='distance'):
        """(the regions' mean of num's per-region value) / (of den's), on regions with both, with the writer interval."""
        vals = []
        for m in sel:
            p = m[view_]['places'][place]
            a = _pool(p, num, key); b = _pool(p, den, key)
            if a is not None and b is not None:
                vals.append((rows[m['id']]['writer'], a, b))
        if not vals:
            return {'regions': 0}
        writers = sorted({w for w, _, _ in vals}); by = {w: [(a, b) for x, a, b in vals if x == w] for w in writers}
        point = np.mean([a for _, a, _ in vals])/np.mean([b for _, _, b in vals])
        rng = np.random.default_rng(SEED); boots = []
        for _ in range(1000):
            pick = [ab for w in rng.choice(writers, len(writers)) for ab in by[w]]
            boots.append(np.mean([a for a, _ in pick])/np.mean([b for _, b in pick]))
        return {'regions': len(vals), 'writers': len(writers), 'value': float(point), 'interval': [float(np.percentile(boots, q)) for q in (2.5, 97.5)]}

    result = {}
    for view_ in ('window', 'line'):
        for part, inside in parts.items():
            sel = [m for m in measured if view_ in m and inside(m['id'])]
            e = {'regions': len(sel), 'places': {}}
            for place in places:
                pe = {'U': ratio(sel, place, view_, ('next', 'far'), ('struck',)),
                      'U_next': ratio(sel, place, view_, ('next',), ('struck',)), 'U_far': ratio(sel, place, view_, ('far',), ('struck',)),
                      'U_energy': ratio(sel, place, view_, ('next', 'far'), ('struck',), 'energy_ratio')}
                shares = [_share(m[view_]['places'][place]) for m in sel]; shares = [s for s in shares if s is not None]
                pe['unstruck_energy_share'] = float(np.mean(shares)) if shares else None
                for cls in CLASSES:
                    d = [m[view_]['places'][place][cls]['distance'] for m in sel if m[view_]['places'][place][cls]['distance'] is not None]
                    pe[f'distance_{cls}'] = float(np.mean(d)) if d else None
                    pe[f'regions_with_{cls}'] = len(d)
                    if place in OUTPUTS:
                        for q in ('cos_x', 'cos_s'):
                            vals = [m[view_]['places'][place][cls][q] for m in sel if m[view_]['places'][place][cls].get(q) is not None]
                            pe[f'{q}_{cls}'] = float(np.mean(vals)) if vals else None
                e['places'][place] = pe
            e['class_shares'] = {unit: {c: int(sum(m[view_]['class_shares'][unit][c] for m in sel)) for c in CLASSES} for unit in ('patches', 'cells')}
            result[f'{view_}:{part}'] = e
    w = result['window:readable']['places']; u9, u3 = w['block8']['U'], w['block2']['U']
    verdict = '局部修有位置' if u9['interval'][1] <= .25 else '放下' if u3['interval'][0] >= .5 else '其余'
    result['rule'] = {'after_9_blocks_U': u9, 'after_3_blocks_U': u3, 'verdict': verdict}
    write_json(out/'summary.json', result); print(json.dumps(result['rule'], ensure_ascii=False), flush=True)


def _pool(place, names, key):
    """The token-weighted mean of ``key`` over the classes ``names`` of one region's place (None when they hold no token)."""
    n = sum(place[c]['tokens'] for c in names)
    return sum(place[c][key]*place[c]['tokens'] for c in names if place[c][key] is not None)/n if n else None


def _share(place):
    total = sum(place[c]['energy'] for c in CLASSES)
    return (place['next']['energy']+place['far']['energy'])/total if total > 0 else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure')
    for key in ('model', 'pack', 'raw-pack', 'output-dir'): m.add_argument('--'+key, type=Path, required=True)
    m.add_argument('--students', required=True, help="V-修's adapter of a fold, {fold} for the fold")
    m.add_argument('--records', required=True, help="V-修's distance run of a fold (distances.jsonl inside), {fold} for the fold")
    m.add_argument('--threads', type=int, default=8); m.add_argument('--limit', type=int, help='a dry run: the first LIMIT regions of each fold')
    s = sub.add_parser('summary')
    for key in ('measured', 'pack', 'shares', 'output-dir'): s.add_argument('--'+key, type=Path, required=True)
    s.add_argument('--readings', action='append', required=True, help='H=DIR[,DIR...] and V=DIR[,DIR...]: whose readings split H right / V wrong')
    args = p.parse_args(argv)
    (measure if args.command == 'measure' else summary)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
