"""Zero-training diagnostics of the restorers' read-back (plan "只减不加", 2026-10-25, section 2), read in one OCR batch.

    python -m revrender.eval.restoration_diagnostics pack --evaluation EVAL --upstream UP --dev-export EXPORT --dev-pages DIR --output-dir OUT
    python -m revrender.eval.restoration_diagnostics score --diagnostics OUT --readings OUT/qwen8b [OUT/qwen8b_repeat] --output OUT/scores.json

``pack`` takes a finished unified evaluation (``eval.unified_restoration``: its chosen versions and their restored OCR crops) and writes
a line pack of the scan, the chosen RevSep, RevSep-both, NAFNet and DE-GAN as they are, and:

- D1: RevSep's erase probability used as a soft lightening, scan + sigmoid(erase/T)*(1-scan), at T = 1 and 2 (no H1, no crossing);
- D2: on the development strikes whose pixels the annotator confirmed, RevSep-both with its writing (scan ink outside the annotator's
  erased pixels) put back to the scan, and RevSep with its strike pixels taken from RevSep-both; and (added after the first reading,
  not in the plan's rules) every one of the four with its writing put back, the read-back each loses to what it does to the writing;
- D3: each of NAFNet, DE-GAN and RevSep averaged with the scan, (scan + output)/2.

It also measures, on those development strikes, what the pixel scores do not see: each version's mean change of the writing's grey
(no threshold, grey levels) and the mean gradient on the writing's edge relative to the scan's.  ``score`` gives each version's
read-back, paired with RevSep's on the strikes both read, its 95% interval by resampling writers and by resampling strikes (1,000,
seed 42), and with a second reading of the same pack the reading's own spread.  Nothing is trained.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest

REFERENCE = 'RevSep'
FAMILIES = ('RevSep', 'RevSep-both', 'NAFNet', 'DE-GAN')
TEMPERATURES = (1., 2.)


def grey_of(path):
    with Image.open(path) as im: return np.asarray(im.convert('L'), np.float32)/255


def strike_masks(rows, export, pages):
    """{line_id: the annotator's erased pixels on the OCR crop} for the development strikes whose pixels are confirmed."""
    from ..data.pixel_masks import has_uncertainty, reviewed_crop
    data = json.loads(Path(export).read_text()); masks = {}
    for r in rows:
        if r['split'] != 'dev': continue
        image_id, rid = r['line_id'].split('#', 1); revision = next(x for x in data['pages'][image_id]['revisions'] if x['id'] == rid)
        if not revision.get('confirmed') or has_uncertainty(revision): continue
        with Image.open(Path(pages)/(image_id+'.png')) as page:
            _, mask = reviewed_crop(page, revision); full = np.zeros((page.height, page.width), bool)
        l, t = revision['box'][:2]; full[t:t+mask.shape[0], l:l+mask.shape[1]] = mask.astype(bool)
        points = np.asarray(revision['region']); wl, wt = np.maximum(0, points.min(0)-6).astype(int)
        wr, wb = np.minimum([full.shape[1], full.shape[0]], points.max(0)+6).astype(int)
        require([int(wr-wl), int(wb-wt)] == r['size'], 'not the window of the evaluation crop')
        masks[r['line_id']] = full[wt:wb, wl:wr]
    return masks


def edge_gradient(image, edge):
    return float(np.hypot(ndimage.sobel(image, 0), ndimage.sobel(image, 1))[edge].mean()) if edge.any() else None


def pack(args):
    from ..baselines.restoration import restore_native
    from ..baselines.restoration_export import selected_predictor
    root = args.evaluation; chosen = json.loads((root/'chosen/selection.json').read_text())
    names = {f: chosen[f]['chosen']['name'] for f in FAMILIES}
    rows = json.loads((root/'pack/pack.json').read_text())['rows']
    by = {(r['model'].split('__', 1)[1], r['split'], r['line_id']): r for r in rows}
    scans = [r for r in json.loads((root/'inputs/pack.json').read_text())['rows']]
    hints = np.load(root/'inputs/hints.npz'); masks = strike_masks(scans, args.dev_export, args.dev_pages)
    out = make_output_dir(args.output_dir); (out/'crops').mkdir(); pack_rows = []; measures = {}
    def add(name, r, image):
        path = f'crops/{len(pack_rows):05d}.png'; Image.fromarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)).save(out/path)
        pack_rows.append({**r, 'id': f"{r['split']}|{name}|{r['line_id']}", 'model': f"{r['split']}__{name}", 'images': {'erased': path}})
    spec = chosen['RevSep']['spec']
    soft = {T: selected_predictor(Path(spec['run']), args.upstream, spec['checkpoint'], soften=T)[0] for T in TEMPERATURES}
    for n, r in enumerate(scans):
        scan = grey_of(root/'inputs'/r['images']['erased']); hint = hints[f'ocr_{n}'].astype(np.float32)
        outputs = {'scan': scan, **{f: grey_of(by[(names[f], r['split'], r['line_id'])]['images']['erased']) for f in FAMILIES}}
        outputs.update({f'D1-T{int(T)}': restore_native(soft[T], scan, hint) for T in TEMPERATURES})
        outputs.update({f'D3-half-{f}': (scan+outputs[f])/2 for f in ('NAFNet', 'DE-GAN', 'RevSep')})
        strike = masks.get(r['line_id'])
        if strike is not None:
            writing = (scan < 200/255) & ~strike
            outputs['D2-both-writing-kept'] = np.where(writing, scan, outputs['RevSep-both'])
            outputs['D2-revsep-strike-from-both'] = np.where(strike, outputs['RevSep-both'], outputs['RevSep'])
            outputs.update({f'D2-kept-{f}': np.where(writing, scan, outputs[f]) for f in FAMILIES if f != 'RevSep-both'})
            edge = writing & ndimage.binary_dilation(scan >= 200/255)
            for name, image in outputs.items():
                m = measures.setdefault(name, {'writing_change': [], 'edge_gradient_ratio': []})
                if writing.any(): m['writing_change'].append(float((image-scan)[writing].mean())*255)
                g, g0 = edge_gradient(image, edge), edge_gradient(scan, edge)
                if g is not None and g0: m['edge_gradient_ratio'].append(g/g0)
        for name, image in outputs.items(): add(name, r, image)
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': pack_rows})
    summary = {name: {k: float(np.mean(v)) if v else None for k, v in m.items()} | {'strikes': len(m['writing_change'])} for name, m in measures.items()}
    write_json(out/'measures.json', {'definitions': {'writing': 'scan ink (<200) outside the annotator erased pixels, development strikes with confirmed pixels',
                                                     'writing_change': 'mean (output - scan) on the writing, grey levels', 'edge_gradient_ratio':
                                                     'mean Sobel gradient on the writing edge (writing next to paper), output over scan'}, 'versions': summary})
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in [root/'chosen/selection.json', root/'pack/pack.json', args.dev_export]},
                   chosen=names, rows=len(pack_rows), development_strikes_with_pixels=len(masks))
    print(json.dumps({'rows': len(pack_rows), 'strikes': len(scans), 'with_pixels': len(masks)}), flush=True)


def scores_of(readings, pack_rows):
    from .revision_boxes import held
    rows = {r['id']: r for r in pack_rows}; scores = {}
    for line in (readings/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        g = json.loads(line); r = rows[g['id']]; h = held(r['text'], g['text'])
        if h is not None: scores.setdefault(r['model'].split('__', 1)[1], {})[r['line_id']] = h
    return scores


def paired(x, y, rng):
    lines = sorted(set(x) & set(y)); writers = sorted({l.split('_')[0] for l in lines}); by = {w: [l for l in lines if l.startswith(w+'_')] for w in writers}
    diff = lambda ls: float(np.mean([y[l]-x[l] for l in ls]))
    by_writer = [diff([l for w in rng.choice(writers, len(writers)) for l in by[w]]) for _ in range(1000)]
    by_strike = [diff(list(rng.choice(lines, len(lines)))) for _ in range(1000)]
    return {'reference': float(np.mean([x[l] for l in lines])), 'version': float(np.mean([y[l] for l in lines])), 'difference': diff(lines),
            'writer_interval': [float(np.percentile(by_writer, 2.5)), float(np.percentile(by_writer, 97.5))],
            'strike_interval': [float(np.percentile(by_strike, 2.5)), float(np.percentile(by_strike, 97.5))], 'strikes': len(lines), 'writers': len(writers)}


def score(args):
    require(not args.output.exists(), 'output exists')
    pack_rows = json.loads((args.diagnostics/'pack.json').read_text())['rows']
    readings = [scores_of(r, pack_rows) for r in args.readings]
    mean = {m: {l: float(np.mean([s[m][l] for s in readings])) for l in readings[0][m]} for m in readings[0]}
    rng = np.random.default_rng(SEED); result = {}
    for name in sorted(mean):
        result[name] = {'read_back': float(np.mean(list(mean[name].values()))), 'strikes': len(mean[name])}
        if name != REFERENCE: result[name]['against_' + REFERENCE] = paired(mean[REFERENCE], mean[name], rng)
        if len(readings) > 1:
            result[name]['readings'] = [float(np.mean(list(s[name].values()))) for s in readings]
            result[name]['strikes_read_differently'] = sum(abs(readings[0][name][l]-readings[1][name][l]) > 1e-9 for l in readings[0][name])
    measures = json.loads((args.diagnostics/'measures.json').read_text())['versions']
    for name in result: result[name]['measures'] = measures.get(name)
    write_json(args.output, {'reference': REFERENCE, 'versions': result,
                             'readings': [str(r) for r in args.readings], 'inputs': {str(r/'predictions.jsonl'): digest(r/'predictions.jsonl') for r in args.readings}})
    for name, r in result.items():
        a = r.get('against_'+REFERENCE)
        print(f"{name:28s} {100*r['read_back']:5.1f}" + (f"  vs RevSep {100*a['difference']:+5.1f} writers ({100*a['writer_interval'][0]:+.1f}, {100*a['writer_interval'][1]:+.1f})"
              f" strikes ({100*a['strike_interval'][0]:+.1f}, {100*a['strike_interval'][1]:+.1f}) n={a['strikes']}" if a else '') +
              (f"  readings {[round(100*v, 1) for v in r['readings']]}" if 'readings' in r else ''))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    k = sub.add_parser('pack')
    for key in ('evaluation', 'upstream', 'dev-export', 'dev-pages', 'output-dir'): k.add_argument('--'+key, type=Path, required=True)
    s = sub.add_parser('score'); s.add_argument('--diagnostics', type=Path, required=True)
    s.add_argument('--readings', type=Path, nargs='+', required=True); s.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv); {'pack': pack, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
