"""Is the plug-in's feature shrunk?  T3 of the plan of 2026-11-22 (三个不训练的检验, revision one), with the chord's closeness of T2:
forward passes of Qwen3-VL-8B's frozen tower and of each fold's V-修 adapter only -- nothing taught, nothing read.

    python -m revrender.eval.feature_scale measure --model MODEL --pack VIEWS --raw-pack RAW_VIEWS --students PATTERN --records PATTERN
        [--threads 8] [--device cpu] [--limit N] --output-dir OUT
    python -m revrender.eval.feature_scale summary --measured OUT --shares SHARES --output-dir OUT2

Per region (each fold's own regions, its own fold's adapter), per view -- the window (the pack's raw crop -> its erased crop) and the line
(the raw pack's context -> the pack's) --, per level (the merged tokens, deepstack 1-3), on the touched tokens (the merged tokens' cosine
of x and c under 0.99, as ``train_clean_view distance``): x the frozen tower on the scan, c on the erased image, s the adapter on the
scan, M and M' the matched points of T2 (``models.feature_mix.matched_alpha``: s farther than x -> x, or s).

- the norm ratio: the region's mean over its tokens of |v| / |c|, v = s, x, M;
- the decomposition: a = <v, c> / <c, c> and |v - a c| / |c|, per token, the region's mean;
- the closeness: the mean cosine of v and c (v = x, s, M, M', and the chord's points at alpha = 0, 0.25, 0.5, 0.75, 1) and the share
  of tokens taken back (s farther than x);
- for the variance: the tokens, each quantity's sum vector and sum of squared norms (OUT/{window,line}.npz, x / s / c / M; on the
  server only), from which ``summary`` pools a half's tokens -- the per-dimension variances summed, R_v = v's / c's -- and the
  within-region version (each region centred on itself), with a writer bootstrap (1,000, seed 42, 95 %).
Check 6 (OUT/checks.json): the five folds' mean x and s closeness (window, merged) within 0.005 of V-修's records."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

LEVELS = ('merged', 'deepstack1', 'deepstack2', 'deepstack3')
QUANTITIES = ('x', 's', 'c', 'M')
ALPHAS = (0., .25, .5, .75, 1.)


def region_stats(x, c, s, touched):
    """One view, one level: the region's statistics on its touched tokens, and (n, sums [4, w], sums of squared norms [4])."""
    import torch
    from ..models.feature_mix import matched_alpha
    x, c, s = (torch.as_tensor(v, dtype=torch.float32) for v in (x, c, s))
    alpha, farther = matched_alpha(x, c, s); a = alpha.float()[:, None]
    chord = (1-a)*x+a*c
    m, m2 = chord, torch.where(farther[:, None], s, chord)
    t = torch.as_tensor(touched)
    cos = lambda u, v: torch.nn.functional.cosine_similarity(u.double(), v.double(), dim=-1)
    out = {'taken_back_all': float(farther.double().mean()), 'taken_back_touched': float(farther[t].double().mean()) if t.any() else None}
    if not t.any():
        return out, None
    xt, ct, st, mt, m2t = (v[t].double() for v in (x, c, s, m, m2))
    cn = ct.norm(dim=-1); cc = (ct*ct).sum(-1)
    for name, v in (('x', xt), ('s', st), ('M', mt)):
        coef = (v*ct).sum(-1)/cc
        out[f'norm_{name}'] = float((v.norm(dim=-1)/cn).mean())
        out[f'proj_{name}'] = float(coef.mean()); out[f'perp_{name}'] = float(((v-coef[:, None]*ct).norm(dim=-1)/cn).mean())
    for name, v in (('x', xt), ('s', st), ('M', mt), ("M'", m2t)):
        out[f'cos_{name}'] = float(cos(v, ct).mean())
    for al in ALPHAS:
        out[f'cos_alpha_{al}'] = float(cos((1-al)*xt+al*ct, ct).mean())
    pooled = (int(t.sum()), np.stack([v.sum(0).numpy() for v in (xt, st, ct, mt)]).astype(np.float32),
              np.array([float((v*v).sum()) for v in (xt, st, ct, mt)]))
    return out, pooled


def measure(args):
    import torch
    from ..models.maa import student_view
    from ..models.vlm_tokens import processor_for
    from ..train_clean_view import TOUCHED, encode, levels, vision_tower
    torch.set_num_threads(args.threads)
    out = make_output_dir(args.output_dir)
    rows = json.loads((args.pack/'pack.json').read_text())['rows']
    folds = sorted({r['fold'] for r in rows})
    if args.limit:
        rows = [r for k in folds for r in [x for x in rows if x['fold'] == k][:args.limit]]
    raws = {r['id']: r for r in json.loads((args.raw_pack/'pack.json').read_text())['rows']}
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    processor = processor_for(args.model); tower = vision_tower(args.model).to(args.device, dtype).eval()
    students = {k: Path(args.students.format(fold=k)) for k in folds}
    view = student_view(tower, {f'f{k}': p for k, p in students.items()}); view.eval()
    merge = tower.spatial_merge_size
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.raw_pack/'pack.json',
                   *(w for k in folds for w in sorted(students[k].glob('*.safetensors'))))}, touched_below=TOUCHED, dtype=str(dtype), alphas=ALPHAS, regions=len(rows))
    opened = lambda path: Image.open(path).convert('RGB')

    def run(images, adapter=None):
        x, g = encode(processor, images, args.device); x = x.to(dtype)
        with torch.no_grad():
            if adapter is None:
                with view.disable_adapter():
                    got = levels(view, x, g)
            else:
                view.set_adapter(adapter); got = levels(view, x, g)
        sizes = [int(t*h*w)//merge**2 for t, h, w in g.tolist()]
        return [[lv.cpu().numpy() for lv in level.split(sizes)] for level in got], g.tolist()     # [level][image]

    per_region, grids_differ = {}, []
    pooled = {view_: {'n': [], 'sums': [], 'squares': []} for view_ in ('window', 'line')}; kept = []
    for n, r in enumerate(rows):
        k, q = r['fold'], raws[r['id']]
        scan = [opened(args.pack/r['images']['raw']), opened(args.raw_pack/q['images']['context'])]
        clean = [opened(args.pack/r['images']['erased']), opened(args.pack/r['images']['context'])]
        xs, gx = run(scan); cs, gc = run(clean)
        if gx != gc:
            grids_differ.append(r['id']); continue
        ss, _ = run(scan, f'f{k}')
        entry = {'fold': k, 'writer': r['writer'], 'readable': r['readable'], 'mode': r['mode']}
        stats = {}
        for i, view_ in enumerate(('window', 'line')):
            touched = np.asarray(torch.nn.functional.cosine_similarity(torch.as_tensor(xs[0][i]), torch.as_tensor(cs[0][i]), dim=-1) < TOUCHED)
            entry[view_] = {'tokens': int(len(touched)), 'touched': int(touched.sum())}
            stats[view_] = []
            for lv, name in enumerate(LEVELS):
                entry[view_][name], p = region_stats(xs[lv][i], cs[lv][i], ss[lv][i], touched)
                stats[view_].append(p)
        per_region[r['id']] = entry; kept.append(r['id'])
        for view_ in ('window', 'line'):
            width = xs[0][0].shape[1]
            ps = stats[view_]
            pooled[view_]['n'].append(ps[0][0] if ps[0] else 0)
            pooled[view_]['sums'].append(np.stack([p[1] if p else np.zeros((len(QUANTITIES), width), np.float32) for p in ps]))
            pooled[view_]['squares'].append(np.stack([p[2] if p else np.zeros(len(QUANTITIES)) for p in ps]))
        if n % 50 == 0: print(json.dumps({'regions': n+1, 'of': len(rows)}), flush=True)
    require(not grids_differ, f'grids differ for {grids_differ[:3]}')
    for view_, d in pooled.items():                                      # the variance's sufficient statistics (server only)
        np.savez(out/f'{view_}.npz', ids=np.array(kept), n=np.array(d['n']), sums=np.stack(d['sums']), squares=np.stack(d['squares']),
                 levels=np.array(LEVELS), quantities=np.array(QUANTITIES))
    write_json(out/'per_region.json', per_region)
    records = {}
    for k in folds:
        for line in (Path(args.records.format(fold=k))/'distances.jsonl').read_text().splitlines():
            g = json.loads(line); records[g['id']] = g['window']['merged'] or {}
    pairs = {'raw': 'cos_x', 'student': 'cos_s'}; checks = {'grids_differ': grids_differ}
    for a, mine in pairs.items():
        ids = [i for i in per_region if per_region[i]['window']['merged'].get(mine) is not None and records.get(i, {}).get(a) is not None]
        this, rec = float(np.mean([per_region[i]['window']['merged'][mine] for i in ids])), float(np.mean([records[i][a] for i in ids]))
        checks[f'check6_{a}'] = {'this_run': this, 'records': rec, 'regions': len(ids), 'difference': this-rec, 'within_0.005': abs(this-rec) <= .005}
    write_json(out/'checks.json', checks); print(json.dumps(checks), flush=True)
    require(args.limit or all(v['within_0.005'] for k, v in checks.items() if k.startswith('check6')), f'check 6: the mean closeness is more than 0.005 off the records: {checks}')
    write_json(out/'complete.json', {'regions': len(kept)})


def mean(values):
    return float(np.mean(values)) if values else None


def summary(args):
    out = make_output_dir(args.output_dir)
    per_region = json.loads((args.measured/'per_region.json').read_text())
    half = {k: v.get('half') for k, v in json.loads(args.shares.read_text())['regions'].items()}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.measured/'per_region.json', args.measured/'window.npz', args.measured/'line.npz', args.shares)},
                   bootstrap={'resamples': 1000, 'seed': SEED, 'by': 'writer'})
    parts = {'heavy': lambda i: half.get(i) == 'heavy' and per_region[i]['readable'], 'light': lambda i: half.get(i) == 'light' and per_region[i]['readable'],
             'readable': lambda i: per_region[i]['readable'], 'unreadable': lambda i: not per_region[i]['readable'], 'all': lambda i: True}
    result = {}
    for view_ in ('window', 'line'):
        z = np.load(args.measured/f'{view_}.npz'); ids = list(z['ids']); n, sums, squares = z['n'], z['sums'], z['squares']      # sums [R, level, q, w]
        for part, inside in parts.items():
            sel = [j for j, i in enumerate(ids) if inside(i) and n[j] > 0]
            if not sel:
                result[f'{view_}:{part}'] = {'regions': 0}; continue
            writers = sorted({per_region[ids[j]]['writer'] for j in sel}); index = {w: m for m, w in enumerate(writers)}
            wn = np.zeros(len(writers)); ws = np.zeros((len(writers),)+sums.shape[1:]); wq = np.zeros((len(writers),)+squares.shape[1:])
            win = np.zeros((len(writers),)+squares.shape[1:])
            for j in sel:
                m = index[per_region[ids[j]]['writer']]; wn[m] += n[j]; ws[m] += sums[j]; wq[m] += squares[j]
                win[m] += squares[j]-(sums[j].astype(np.float64)**2).sum(-1)/n[j]
            ratio = lambda counts: _ratios(counts, wn, ws, wq, win)
            point = ratio(np.ones(len(writers)))
            rng = np.random.default_rng(SEED); boots = [ratio(np.bincount(rng.integers(0, len(writers), len(writers)), minlength=len(writers))) for _ in range(1000)]
            entry = {'regions': len(sel), 'writers': len(writers), 'tokens': int(n[sel].sum()), 'levels': {}}
            for lv, name in enumerate(LEVELS):
                e = {key: {'value': float(point[key][lv]), 'interval': [float(np.percentile([b[key][lv] for b in boots], q)) for q in (2.5, 97.5)]} for key in point}
                for stat in ('norm', 'proj', 'perp'):
                    for v in ('x', 's', 'M'):
                        vals = [per_region[ids[j]][view_][name].get(f'{stat}_{v}') for j in sel]
                        e[f'{stat}_{v}'] = mean([x for x in vals if x is not None])
                for key in ('cos_x', 'cos_s', 'cos_M', "cos_M'", *(f'cos_alpha_{a}' for a in ALPHAS), 'taken_back_touched', 'taken_back_all'):
                    vals = [per_region[ids[j]][view_][name].get(key) for j in sel]
                    e[key] = mean([x for x in vals if x is not None])
                entry['levels'][name] = e
            result[f'{view_}:{part}'] = entry
    rs = result['window:heavy']['levels']['merged']['R_s']
    verdict = ('缩了' if rs['value'] < .9 and rs['interval'][1] < 1 else '幅度相当' if .9 <= rs['value'] <= 1.1 else '更大' if rs['value'] > 1.1 else '其余')
    result['rule_T3'] = {'R_s': rs, 'verdict': verdict}
    write_json(out/'summary.json', result); print(json.dumps(result['rule_T3'], ensure_ascii=False), flush=True)


def _ratios(counts, wn, ws, wq, win):
    """R_v (pooled and within regions) of the writers weighted by ``counts`` (a bootstrap draw), per level; v = x, s, M over c."""
    total = counts@wn
    s = np.tensordot(counts, ws, 1); sq = np.tensordot(counts, wq, 1); within = np.tensordot(counts, win, 1)/total
    pooled = sq/total-(s*s).sum(-1)/total**2                                 # [level, q]
    q = {v: i for i, v in enumerate(QUANTITIES)}
    return {**{f'R_{v}': pooled[:, q[v]]/pooled[:, q['c']] for v in ('s', 'x', 'M')},
            **{f'R_within_{v}': within[:, q[v]]/within[:, q['c']] for v in ('s', 'x', 'M')}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure')
    for key in ('model', 'pack', 'raw-pack', 'output-dir'): m.add_argument('--'+key, type=Path, required=True)
    m.add_argument('--students', required=True, help="V-修's adapter of a fold, {fold} for the fold")
    m.add_argument('--records', required=True, help="V-修's distance run of a fold (distances.jsonl inside), {fold} for the fold")
    m.add_argument('--threads', type=int, default=8); m.add_argument('--device', default='cpu')
    m.add_argument('--limit', type=int, help='a dry run of the code: the first LIMIT regions of each fold')
    s = sub.add_parser('summary')
    for key in ('measured', 'shares', 'output-dir'): s.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv)
    (measure if args.command == 'measure' else summary)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
