"""Is the strike a few directions in the tower's features?  The offline measurement of the plan of 2026-11-22 (划线方向_离线测量):
forward passes of Qwen3-VL-8B's frozen tower and of each fold's V-修 adapter only -- nothing taught, nothing read.

    python -m revrender.tools.strike_direction --model MODEL --pack VIEWS --raw-pack RAW_VIEWS --students PATTERN --shares SHARES
        --records RECORDS_PATTERN [--synthetic SYNTH] [--threads 8] [--device cpu] --output-dir OUT

The pairs: each region's window (the pack's raw crop -> its erased crop) and line (the raw pack's context -> the pack's), as
``train_clean_view distance`` takes them; x the scan's features, c the erased crop's, d = x - c, s the region's own fold's adapter on
the scan.  Where: the merged tokens (what the language model takes) and the output of every block (a token per patch).  Fold K's fit:
the regions V-修's fold-K adapter was taught on (the other folds without its validation writers, as ``train_clean_view`` takes them);
both images' tokens.  Fold K is held out.

1. Concentration (every block; the merged tokens): the directions of the fit's d (its energy, not centred); E(k) the share of the
   held-out d's energy they hold, k in ``KS``; beside it the share of the held-out centred c its own fit's directions hold.
2. Linear erasure (the merged tokens): d predicted from x by reduced-rank ridge regression -- the full ridge solution (ridge 1e-3 x the
   mean diagonal of x's covariance), its fitted values projected on their first k directions -- f = x - d_hat; the closeness of f, x
   and s to c: a region's mean cosine over its touched window tokens (the cosine of x and c under 0.99, as ``distance``); g(k) =
   (mean f - mean x) / (mean s - mean x), the means over the held-out regions of all folds, and per strike half.  The lines beside.
3. Reported: rank-1 LEACE (the concept: a merged window token's 32 x 32 cell holds erased ink, ``synth_views.window_stats``'s strike),
   fitted on the fit's window tokens; per block the held-out normalised residual of the full-rank linear map; with ``--synthetic``,
   E(k) of each fold's synthetic pairs under the real fit's directions (``synthetic_real_directions``).
Checks (OUT/checks.json): 1 the scan and its erased crop give one grid, every region; 2 the first 20 regions' x and s closeness beside
V-修's records (``--records``, the sixth round's ``distance`` runs, bf16 on a card); 3 no held-out region in its fold's fit.
OUT holds the statistics only (student work: never in Git)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import make_output_dir, require, to_jsonable, write_json, write_manifest

KS = (1, 2, 4, 8, 16, 32, 64, 128)
TOUCHED = .99


class Moments:
    """Running sums over tokens: n, sum x, sum d, sum c, x'x, x'd, d'd, c'c (float64)."""

    def __init__(self, width):
        z = lambda *s: np.zeros(s, np.float64)
        self.n, self.x, self.d, self.c = 0, z(width), z(width), z(width)
        self.xx, self.xd, self.dd, self.cc = z(width, width), z(width, width), z(width, width), z(width, width)

    def add(self, x, c):
        x, c = x.astype(np.float64), c.astype(np.float64); d = x-c
        self.n += len(x); self.x += x.sum(0); self.d += d.sum(0); self.c += c.sum(0)
        self.xx += x.T@x; self.xd += x.T@d; self.dd += d.T@d; self.cc += c.T@c

    def combine(self, others, minus=()):
        out = Moments(len(self.x))
        for key in ('n', 'x', 'd', 'c', 'xx', 'xd', 'dd', 'cc'):
            setattr(out, key, sum(getattr(o, key) for o in others)-sum((getattr(o, key) for o in minus), 0*getattr(self, key)))
        return out


def top(matrix, k):
    """The first k eigenvectors of a symmetric matrix, largest first (only those computed)."""
    from scipy.linalg import eigh
    n = len(matrix)
    return eigh(matrix, subset_by_index=[n-k, n-1])[1][:, ::-1]


def concentration(fit, held):
    """E(k) of the held-out d under the fit's d directions, and of the held-out centred c under the fit's centred c directions."""
    dirs = top(fit.dd, max(KS)); energy = np.trace(held.dd)
    centred = lambda m: m.cc-np.outer(m.c, m.c)/m.n
    cdirs, ch = top(centred(fit), max(KS)), centred(held)
    return ({k: float(np.trace(dirs[:, :k].T@held.dd@dirs[:, :k])/energy) for k in KS},
            {k: float(np.trace(cdirs[:, :k].T@ch@cdirs[:, :k])/np.trace(ch)) for k in KS})


def ridge(fit):
    """(W, mean x, mean d, the fitted values' directions): the centred ridge regression of d on x."""
    mx, md = fit.x/fit.n, fit.d/fit.n
    sxx = fit.xx-fit.n*np.outer(mx, mx); sxd = fit.xd-fit.n*np.outer(mx, md)
    w = np.linalg.solve(sxx+1e-3*np.mean(np.diag(sxx))*np.eye(len(mx)), sxd)       # 1e-3 x the mean diagonal (of the sums, so of the covariance x n)
    return w, mx, md, top(w.T@sxx@w, max(KS))


def residual(fitted, held):
    """The held-out normalised residual of the full-rank ridge map of x to d (``fitted``: ``ridge``'s): |d - d_hat|^2 / |d|^2."""
    w, mx, md, _ = fitted
    hx, hd = held.x/held.n, held.d/held.n
    sxx = held.xx-held.n*np.outer(hx, hx); sxd = held.xd-held.n*np.outer(hx, hd); sdd = held.dd-held.n*np.outer(hd, hd)
    shift = hd-md-(hx-mx)@w
    return float((np.trace(sdd)-2*np.trace(w.T@sxd)+np.trace(w.T@sxx@w)+held.n*shift@shift)/np.trace(held.dd))


def leace(xs, zs):
    """Rank-1 LEACE (Belrose et al., NeurIPS 2023) fitted on tokens xs [n, w] with a binary concept zs: x -> x - W+ u u' W (x - mean)."""
    mean = xs.mean(0); xc = xs-mean; cov = xc.T@xc/len(xs); cross = xc.T@(zs-zs.mean())/len(xs)
    values, vectors = np.linalg.eigh(cov); keep = values > values.max()*1e-8
    v, lam = vectors[:, keep], values[keep]
    whiten, unwhiten = (v/np.sqrt(lam))@v.T, (v*np.sqrt(lam))@v.T
    u = whiten@cross; u = u/np.linalg.norm(u)
    return lambda x: x-((x-mean)@whiten@np.outer(u, u))@unwhiten


def cos_rows(a, b):
    return np.sum(a*b, 1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12)


def closeness(f, c, touched):
    return float(cos_rows(f, c)[touched].mean()) if touched.any() else None


def synthetic(args):
    """The synthetic pairs, reported only (revision one): each fold's synthetic set (``SYNTH/fold_K_raw`` -> ``fold_K_clean``, window
    and line, the frozen tower; a region shared by several folds' sets computed once); per block and the merged tokens, (a) E(k) of the
    fold's synthetic d under the real fit's d directions, (b) E(k) of the real held-out fold's d under the fold's synthetic d directions."""
    import torch
    from ..models.vlm_tokens import processor_for
    from ..train_clean_view import encode, vision_tower
    torch.set_num_threads(args.threads)
    out = make_output_dir(args.output_dir); real = np.load(args.real/'real_d.npz')
    folds = sorted(int(p.name.split('_')[1]) for p in args.synthetic.glob('fold_*_raw'))
    packs = {k: ({r['id']: r for r in json.loads((args.synthetic/f'fold_{k}_raw'/'pack.json').read_text())['rows']},
                 {r['id']: r for r in json.loads((args.synthetic/f'fold_{k}_clean'/'pack.json').read_text())['rows']}) for k in folds}
    sets = {k: set(packs[k][0]) for k in folds}; ids = sorted(set().union(*sets.values()))
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    processor = processor_for(args.model); tower = vision_tower(args.model).to(args.device, dtype).eval()
    depth, merge = len(tower.blocks), tower.spatial_merge_size; captured = []
    for block in tower.blocks:
        block.register_forward_hook(lambda m, i, o: captured.append(o.detach().float().cpu().numpy()))
    write_manifest(out, config=to_jsonable(args), regions=len(ids), per_fold={k: len(v) for k, v in sets.items()}, ks=KS, dtype=str(dtype))
    dd = {k: [np.zeros((tower.config.hidden_size,)*2) for _ in range(depth)]+[np.zeros((tower.config.out_hidden_size,)*2)] for k in folds}
    opened = lambda path: Image.open(path).convert('RGB')

    def features(images):
        x, g = encode(processor, images, args.device); captured.clear()
        with torch.no_grad():
            merged_out = tower(x.to(dtype), grid_thw=g)[0]
        return [c for c in captured]+[merged_out.float().cpu().numpy()], g.tolist()
    for n, i in enumerate(ids):
        k0 = next(k for k in folds if i in sets[k]); raw, clean = packs[k0][0][i], packs[k0][1][i]
        (xs, gx), (cs, gc) = features([opened(raw['images'][key]) for key in ('erased', 'context')]), features([opened(clean['images'][key]) for key in ('erased', 'context')])
        require(gx == gc, f'{i}: the synthetic pair gives two grids')
        for k in folds:
            if i in sets[k]:
                for b in range(depth+1):
                    d = (xs[b]-cs[b]).astype(np.float64); dd[k][b] += d.T@d
        if n % 200 == 0: print(json.dumps({'regions': n+1, 'of': len(ids)}), flush=True)
    share = lambda dirs, m, q: float(np.trace(dirs[:, :q].T@m@dirs[:, :q])/np.trace(m))
    result = {}
    for b in range(depth+1):
        name = 'merged' if b == depth else b
        a = {q: float(np.mean([share(real[f'{name}_f{k}_directions'].astype(np.float64), dd[k][b], q) for k in folds])) for q in KS}
        own = {k: top(dd[k][b], max(KS)) for k in folds}
        bb = {q: float(np.mean([share(own[k], real[f'{name}_f{k}_held_dd'].astype(np.float64), q) for k in folds])) for q in KS}
        result[str(name)] = {'synthetic_d_under_real_directions': a, 'real_held_out_d_under_synthetic_directions': bb}
    write_json(out/'result.json', result); print(json.dumps({'merged': result['merged']}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ('model', 'pack', 'raw-pack', 'shares', 'output-dir'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--students', required=True, help="V-修's adapter of a fold, {fold} for the fold")
    p.add_argument('--records', required=True, help="V-修's distance run of a fold (distances.jsonl inside), {fold} for the fold")
    p.add_argument('--synthetic', type=Path, help='a data.synth_views build folder: E(k) of its pairs (reported; with --real)')
    p.add_argument('--real', type=Path, help="the real pairs' run (its real_d.npz), for --synthetic")
    p.add_argument('--threads', type=int, default=8); p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, help='a dry run of the code: the first LIMIT regions of each fold')
    args = p.parse_args(argv)
    if args.synthetic:
        return synthetic(args)
    import torch
    from ..models.maa import student_view
    from ..models.vlm_tokens import cell_shares, processor_for
    from ..train_clean_view import encode, regions_of, validation_writers, vision_tower
    torch.set_num_threads(args.threads)
    out = make_output_dir(args.output_dir)
    rows = json.loads((args.pack/'pack.json').read_text())['rows']
    if args.limit:
        rows = [r for k in sorted({r['fold'] for r in rows}) for r in [x for x in rows if x['fold'] == k][:args.limit]]
    raws = {r['id']: r for r in json.loads((args.raw_pack/'pack.json').read_text())['rows']}
    folds = sorted({r['fold'] for r in rows}); half = {k: v.get('half') for k, v in json.loads(args.shares.read_text())['regions'].items()}
    fit_ids, left = {}, {}
    for k in folds:                                                      # what V-修's fold-K adapter was taught on
        real = regions_of(args.pack, args.raw_pack, lambda r: r['fold'] != k); held = validation_writers(real)      # the whole pack, as V-修
        kept = {r['id'] for r in rows}
        fit_ids[k] = {r[0] for r in real if r[1] not in held and r[0] in kept}; left[k] = {r[0] for r in real if r[1] in held and r[0] in kept}
    fold_of = {r['id']: r['fold'] for r in rows}
    checks = {'check3_held_out_in_fit': {k: sorted(i for i in fit_ids[k] if fold_of[i] == k) for k in folds}}
    require(not any(checks['check3_held_out_in_fit'].values()), 'check 3: a held-out region in its own fit')
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    processor = processor_for(args.model); tower = vision_tower(args.model).to(args.device, dtype).eval()
    view = student_view(tower, {f'f{k}': Path(args.students.format(fold=k)) for k in folds}); view.eval()
    depth, merge = len(tower.blocks), tower.spatial_merge_size; captured = []
    for block in tower.blocks:
        block.register_forward_hook(lambda m, i, o: captured.append(o.detach().float().cpu().numpy()))
    write_manifest(out, config=to_jsonable(args), fits={k: len(v) for k, v in fit_ids.items()}, ks=KS, touched_below=TOUCHED, dtype=str(dtype),
                   threads=args.threads)
    wm, wb = tower.config.out_hidden_size, tower.config.hidden_size
    # per fold: its own regions' moments, and those of the regions of the other folds that V-修's fold-K adapter left to validation
    own = {k: ([Moments(wb) for _ in range(depth)], Moments(wm)) for k in folds}
    val = {k: ([Moments(wb) for _ in range(depth)], Moments(wm)) for k in folds}
    windows, per_region, grids_differ = {k: [] for k in folds}, {}, []
    opened = lambda path: Image.open(path).convert('RGB')

    def run(images, adapter=None):
        x, g = encode(processor, images, args.device); x = x.to(dtype); captured.clear()
        with torch.no_grad():
            if adapter is None:
                with view.disable_adapter():
                    merged_out = view(x, grid_thw=g)[0]
            else:
                view.set_adapter(adapter); merged_out = view(x, grid_thw=g)[0]
        sizes = [int(t*h*w) for t, h, w in g.tolist()]
        blocks = [np.split(c, np.cumsum(sizes)[:-1]) for c in captured] if adapter is None else None
        return np.split(merged_out.float().cpu().numpy(), np.cumsum([n//merge**2 for n in sizes])[:-1]), blocks, g.tolist()

    def add(target, xb, cb, xm, cm):
        blocks, merged = target
        for b in range(depth):
            for v in range(2): blocks[b].add(xb[b][v], cb[b][v])
        for v in range(2): merged.add(xm[v], cm[v])
    for n, r in enumerate(rows):
        k, q = r['fold'], raws[r['id']]
        scan = [opened(args.pack/r['images']['raw']), opened(args.raw_pack/q['images']['context'])]
        clean = [opened(args.pack/r['images']['erased']), opened(args.pack/r['images']['context'])]
        xm, xb, gx = run(scan); cm, cb, gc = run(clean)
        if gx != gc:
            grids_differ.append(r['id']); continue
        sm, _, _ = run(scan, f'f{k}')
        add(own[k], xb, cb, xm, cm)
        for j in folds:
            if r['id'] in left[j]: add(val[j], xb, cb, xm, cm)
        s_, c_ = (np.asarray(im.convert('L'), np.float32)/255 for im in (scan[0], clean[0]))
        strike = (c_-s_ > .2) & (s_ < 200/255)                              # window_stats's strike
        t, h, w = gx[0]; z = (cell_shares(strike, h//merge, w//merge) > 0).astype(np.float64)
        touched, line_touched = cos_rows(xm[0], cm[0]) < TOUCHED, cos_rows(xm[1], cm[1]) < TOUCHED
        windows[k].append((r['id'], xm[0], cm[0], z, touched, xm[1], cm[1], line_touched))
        per_region[r['id']] = {'fold': k, 'half': half.get(r['id']), 'readable': r['readable'], 'touched': int(touched.sum()), 'tokens': int(len(touched)),
                               'raw': closeness(xm[0], cm[0], touched), 'student': closeness(sm[0], cm[0], touched),
                               'line_raw': closeness(xm[1], cm[1], line_touched), 'line_student': closeness(sm[1], cm[1], line_touched)}
        if n % 50 == 0: print(json.dumps({'regions': n+1}), flush=True)
    checks['check1_grids_differ'] = grids_differ
    require(not grids_differ, f'check 1: grids differ for {grids_differ[:3]}')
    records = {}
    for k in folds:
        for line in (Path(args.records.format(fold=k))/'distances.jsonl').read_text().splitlines():
            g = json.loads(line); records[g['id']] = (g['window']['merged'] or {})
    first = [r['id'] for r in rows[:20]]
    checks['check2_against_records'] = [{'id': i, 'raw': per_region[i]['raw'], 'raw_record': records[i].get('raw'), 'student': per_region[i]['student'],
                                         'student_record': records[i].get('student')} for i in first]
    diffs = [abs(c[a]-c[a+'_record']) for c in checks['check2_against_records'] for a in ('raw', 'student') if c[a] is not None and c[a+'_record'] is not None]
    checks['check2_largest_difference'] = max(diffs) if diffs else None
    # the wiring check (revision one): over the five folds the means of this run's x and s closeness within 0.005 of the records'
    both = {a: [i for i in per_region if per_region[i][a] is not None and records.get(i, {}).get(a) is not None] for a in ('raw', 'student')}
    means = {a: {'this_run': float(np.mean([per_region[i][a] for i in both[a]])), 'records': float(np.mean([records[i][a] for i in both[a]])), 'regions': len(both[a])}
             for a in ('raw', 'student')}
    checks['check2_means'] = {a: {**m, 'difference': m['this_run']-m['records'], 'within_0.005': abs(m['this_run']-m['records']) <= .005} for a, m in means.items()}
    write_json(out/'checks.json', checks); write_json(out/'per_region.json', per_region)
    require(args.limit or all(v['within_0.005'] for v in checks['check2_means'].values()),
            f"check 2: the mean closeness is more than 0.005 off the records (an adapter, image or token set wired wrong): {checks['check2_means']}")

    # the folds' fits and the measures
    result = {'layers': {}, 'merged': {}, 'erasure': {}, 'leace': {}}
    kept, ridges = {}, {}                                                # for the synthetic pairs' run: each fit's d directions, each held-out d'd
    for b in list(range(depth))+['merged']:
        pick = (lambda m: m[1]) if b == 'merged' else (lambda m, b=b: m[0][b])
        e, ref, res = {q: [] for q in KS}, {q: [] for q in KS}, []
        heldouts = []
        for k in folds:
            fit = Moments(wm if b == 'merged' else wb).combine([pick(own[j]) for j in folds if j != k], [pick(val[k])])
            held = pick(own[k]); ek, rk = concentration(fit, held); heldouts.append(np.trace(held.dd))
            kept[f'{b}_f{k}_directions'] = top(fit.dd, max(KS)).astype(np.float32); kept[f'{b}_f{k}_held_dd'] = held.dd.astype(np.float32)
            for q in KS: e[q].append(ek[q]); ref[q].append(rk[q])
            fitted = ridge(fit); res.append(residual(fitted, held))
            if b == 'merged': ridges[k] = fitted
        weights = np.array(heldouts)/sum(heldouts)                         # the folds' E(k) weighted by their held-out energy
        block = {'E': {q: float(np.dot(weights, e[q])) for q in KS}, 'E_reference_c': {q: float(np.mean(ref[q])) for q in KS},
                 'E_by_fold': {q: e[q] for q in KS}, 'full_rank_residual': float(np.mean(res))}
        (result['merged'].update(block) if b == 'merged' else result['layers'].__setitem__(b, block))
    # linear erasure and LEACE on the merged window tokens of each held-out fold
    gains = {q: [] for q in (*KS, 'full')}; rows_out = []
    for k in folds:
        w, mx, md, dirs = ridges[k]
        maps = {q: w@dirs[:, :q]@dirs[:, :q].T for q in KS}; maps['full'] = w
        fit_tokens = [t for j in folds if j != k for t in windows[j] if t[0] in fit_ids[k]]
        eraser = leace(np.concatenate([t[1] for t in fit_tokens]).astype(np.float64), np.concatenate([t[3] for t in fit_tokens]))
        for rid, x, c, z, touched, xl, cl, lt in windows[k]:
            row = {'id': rid, 'half': half.get(rid), 'raw': per_region[rid]['raw'], 'student': per_region[rid]['student'], 'line_raw': per_region[rid]['line_raw'],
                   'line_student': per_region[rid]['line_student']}
            for q, m in maps.items():
                row[f'linear_{q}'] = closeness(x-((x-mx)@m+md), c, touched)
                row[f'line_linear_{q}'] = closeness(xl-((xl-mx)@m+md), cl, lt)
            row['leace'] = closeness(eraser(x.astype(np.float64)), c, touched)
            rows_out.append(row)

    def g_of(part, key, raw='raw', student='student'):
        sel = [r for r in rows_out if r[raw] is not None and r[key] is not None and (part == 'all' or r['half'] == part)]
        mean = lambda f: float(np.mean([r[f] for r in sel]))
        return {'regions': len(sel), 'raw': mean(raw), 'student': mean(student), 'erased': mean(key),
                'g': (mean(key)-mean(raw))/(mean(student)-mean(raw))}
    result['erasure'] = {part: {str(q): g_of(part, f'linear_{q}') for q in (*KS, 'full')} for part in ('all', 'heavy', 'light')}
    result['erasure_lines'] = {str(q): g_of('all', f'line_linear_{q}', 'line_raw', 'line_student') for q in (*KS, 'full')}
    result['leace'] = {part: g_of(part, 'leace') for part in ('all', 'heavy', 'light')}
    with (out/'erasure_by_region.jsonl').open('x') as stream:
        for r in rows_out: stream.write(json.dumps(r)+'\n')
    write_json(out/'result.json', result)
    np.savez(out/'real_d.npz', **kept)
    print(json.dumps({'merged_E': result['merged']['E'], 'g_all': {q: v['g'] for q, v in result['erasure']['all'].items()},
                      'check2_largest_difference': checks['check2_largest_difference']}), flush=True)


if __name__ == '__main__':
    main()
