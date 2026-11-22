"""The transplant without the reader (the premise check of 2026-11-22, 早期局部修_前提检验, revision one, section five): on the server's
CPU, Qwen3-VL-8B's frozen tower in float32, per region and per image the reader transplants (the window, image 2, and the line, image
3, of context_revision), the eleven arms' four outputs beside the erased image's -- nothing taught, nothing read.

    python -m revrender.eval.transplant_closeness measure --model MODEL --pack RAW_VIEWS --transplant-pack VIEWS --nafnet-pack ERASED_VIEWS
        --spread SPREAD [--threads 8] [--check-regions 10] [--limit N] --output-dir OUT
    python -m revrender.eval.transplant_closeness summary --measured OUT --pack VIEWS --output-dir OUT2

x the scan's features (the read pack's image), c the transplant pack's erased image's; an arm (``ARMS``) puts c's block-K output in
place at its chosen patches after K blocks (``models.transplant``: struck, or struck and next, by the human strike -- the pixels the two
differ in --; one arm by NAFNet's fold-out erased image under the removal metrics' rule) and runs the rest.  All the arms of a region's
image go through the tower as one batch of copies of the scan, each copy's block K hooked (one image never sees another: the blocks
attend within an image).  Per region, image, arm, output and class of the 32-pixel cells (the human strike, ``classes``): the tokens and
the mean cos(z, c); the raw's cos(x, c) beside them (OUT/regions.jsonl), and the patches each arm took.  NAFNet's places beside the
human's: per class set (struck; struck and next) the patches of both, of NAFNet, of the human; the arm's chosen patches, those
on the human's struck, the human's struck, the human's struck and next (``overlap``).

Checks (OUT/checks.json; the bit-for-bit ones are ``models.transplant``'s on the card): 1 one grid for the scan and the erased image;
2 the read pack's window is the transplant pack's raw crop, pixel for pixel (so the strike is the pack's map, as the spread had it);
3 per region the struck and next patches the spread's record counted (window and line); 4 on the first ``--check-regions`` regions of
each fold, every patch transplanted after K blocks gives c's outputs from block K on (the earlier deepstack levels x's) and none gives
x's, to float32's batch precision (the largest |difference| over the largest |value| at most 1e-4)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..models.transplant import classes, patch_labels, selection, strike_pixels
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

AFTER = (3, 6, 9, 13, 17)
ARMS = (*((f'k{k}_{s}', k, s, 'human') for k in AFTER for s in ('struck', 'next')), ('k9_next_NAFNet', 9, 'next', 'nafnet'))
CHECK_ARMS = (*((f'check_all_k{k}', k, 'all', 'human') for k in AFTER), ('check_none', 9, 'none', 'human'))
OUTPUTS = ('merged', 'deepstack1', 'deepstack2', 'deepstack3')
CLASSES = ('struck', 'next', 'far')
IMAGES = (('window', 'erased'), ('line', 'context'))          # context_revision's images 2 and 3, by their keys
TOLERANCE = 1e-4


def cosines(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a*b).sum(1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-300)


def by_class(cos, cells):
    out = {'all': {'tokens': int(len(cos)), 'cos': float(cos.mean())}}
    for j, name in enumerate(CLASSES):
        sel = cells == j
        out[name] = {'tokens': int(sel.sum()), 'cos': float(cos[sel].mean()) if sel.any() else None}
    return out


def overlap(naf, human):
    """Patch counts of NAFNet's and the human's places, per class set (struck; struck and next): of both, of NAFNet, of the human; and
    the NAFNet arm's chosen patches (its struck and next), those of them on the human's struck patches, the human's struck, the human's
    struck and next."""
    out = {name: {'both': int(((naf <= top) & (human <= top)).sum()),
                  'nafnet': int((naf <= top).sum()), 'human': int((human <= top).sum())} for name, top in (('struck', 0), ('next', 1))}
    out['arm'] = {'chosen': int((naf <= 1).sum()), 'chosen_on_human_struck': int(((naf <= 1) & (human == 0)).sum()),
                  'human_struck': int((human == 0).sum()), 'human_struck_and_next': int((human <= 1).sum())}
    return out


def measure(args):
    import torch
    from ..models.vlm_tokens import processor_for
    from ..train_clean_view import levels, vision_tower
    torch.set_num_threads(args.threads)
    out = make_output_dir(args.output_dir)
    load = lambda pack: {r['id']: r for r in json.loads((pack/'pack.json').read_text())['rows']}
    reads, clean, naf = load(args.pack), load(args.transplant_pack), load(args.nafnet_pack)
    require(set(reads) == set(clean) == set(naf), 'the three packs hold other regions')
    spread = {g['id']: g for g in map(json.loads, (args.spread/'regions.jsonl').read_text().splitlines())}
    rows = sorted(reads.values(), key=lambda r: (r['fold'], r['id']))
    folds = sorted({r['fold'] for r in rows})
    if args.limit:
        rows = [r for k in folds for r in [x for x in rows if x['fold'] == k][:args.limit]]
    checked = {r['id'] for k in folds for r in [x for x in rows if x['fold'] == k][:args.check_regions]}
    processor = processor_for(args.model); tower = vision_tower(args.model).float().eval()
    merge, blocks = tower.spatial_merge_size, len(tower.blocks)
    deep_after = [d+1 for d in tower.deepstack_visual_indexes]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.pack/'pack.json', args.transplant_pack/'pack.json',
                   args.nafnet_pack/'pack.json', args.spread/'regions.jsonl', *sorted(args.model.glob('*.safetensors')))},
                   arms=[list(a) for a in ARMS], check_arms=[list(a) for a in CHECK_ARMS], deepstack_after_blocks=deep_after, regions=len(rows),
                   checked_regions=len(checked), dtype='float32', threads=args.threads, rule_nafnet={'brighter_than': .2, 'ink_below': 200})
    opened = lambda path: Image.open(path).convert('RGB')
    checks = {'check1_grids_differ': [], 'check2_window_not_the_raw_crop': [], 'check3_counts_differ': [], 'check4': {}}
    with (out/'regions.jsonl').open('x', encoding='utf-8') as stream:
        for n, r in enumerate(rows):
            entry = {'id': r['id'], 'fold': r['fold']}
            c_row, n_row = clean[r['id']], naf[r['id']]
            if not np.array_equal(np.asarray(opened(args.pack/r['images']['erased'])), np.asarray(opened(args.transplant_pack/c_row['images']['raw']))):
                checks['check2_window_not_the_raw_crop'].append(r['id'])
            arms = ARMS+(CHECK_ARMS if r['id'] in checked else ())
            for view, key in IMAGES:
                scan, erased, nafnet = (opened(p/q['images'][key]) for p, q in ((args.pack, r), (args.transplant_pack, c_row), (args.nafnet_pack, n_row)))
                batch = processor.image_processor(images=[scan, erased], return_tensors='pt')
                pv, g = batch['pixel_values'].float(), batch['image_grid_thw']
                if not torch.equal(g[0], g[1]):
                    checks['check1_grids_differ'].append([r['id'], view]); continue
                t, h, w = g[0].tolist(); size = t*h*w; m = size//merge**2
                human = patch_labels(strike_pixels(scan, erased, 'any'), h, w, merge)
                nafl = patch_labels(strike_pixels(scan, nafnet, 'brighter'), h, w, merge)
                cells = classes(strike_pixels(scan, erased, 'any'), h//merge, w//merge).reshape(-1)
                record = spread[r['id']][view]['class_shares']['patches']
                counted = {'struck': int((human == 0).sum()), 'next': int((human == 1).sum())}
                if counted != {k: record[k] for k in counted}:
                    checks['check3_counts_differ'].append([r['id'], view, counted, {k: record[k] for k in counted}])
                caught = []
                hooks = [b.register_forward_hook(lambda mod, i, o: caught.append(o.detach()[size:].clone())) for b in tower.blocks]
                with torch.no_grad():
                    both = levels(tower, pv, g)
                for hook in hooks:
                    hook.remove()
                x, c = [lv[:m].numpy() for lv in both], [lv[m:].numpy() for lv in both]
                chosen = [selection(nafl if places == 'nafnet' else human, select) for _, _, select, places in arms]

                def swap(b):
                    def hook(mod, inputs, output):
                        output = output.clone()
                        for j, (_, k, _, _) in enumerate(arms):
                            if k-1 == b:
                                rows_ = torch.as_tensor(j*size+np.flatnonzero(chosen[j]), dtype=torch.long)
                                output[rows_] = caught[b][rows_-j*size]
                        return output
                    return hook
                hooks = [tower.blocks[b].register_forward_hook(swap(b)) for b in sorted({k-1 for _, k, _, _ in arms})]
                with torch.no_grad():
                    planted = levels(tower, pv[:size].repeat(len(arms), 1), g[:1].repeat(len(arms), 1))
                for hook in hooks:
                    hook.remove()
                e = {'grid': [h, w], 'patches': {'struck': counted['struck'], 'next': counted['next'], 'all': size},
                     'cells': {cls: int((cells == j).sum()) for j, cls in enumerate(CLASSES)},
                     'nafnet_places': overlap(nafl, human), 'taken': {}, 'raw': {}, 'arms': {}}
                for name, xl, cl in zip(OUTPUTS, x, c):
                    e['raw'][name] = by_class(cosines(xl, cl), cells)
                for j, (arm, k, select, _) in enumerate(arms):
                    z = [lv[j*m:(j+1)*m].numpy() for lv in planted]
                    e['taken'][arm] = int(chosen[j].sum())
                    if arm.startswith('check'):
                        for level, (name, zl) in enumerate(zip(OUTPUTS, z)):
                            want = x[level] if select == 'none' or (level > 0 and deep_after[level-1] < k) else c[level]
                            gap = float(np.abs(zl-want).max()/max(np.abs(want).max(), 1e-30))
                            f = checks['check4'].setdefault(f'{arm}:{name}', {'images': 0, 'largest_relative_gap': 0., 'wanted': 'x' if want is x[level] else 'c'})
                            f['images'] += 1; f['largest_relative_gap'] = max(f['largest_relative_gap'], gap)
                        continue
                    e['arms'][arm] = {name: by_class(cosines(zl, cl), cells) for name, zl, cl in zip(OUTPUTS, z, c)}
                entry[view] = e
            stream.write(json.dumps(entry)+'\n'); stream.flush()
            if n % 50 == 0: print(json.dumps({'regions': n+1, 'of': len(rows)}), flush=True)
    for f in checks['check4'].values():
        f['within'] = f['largest_relative_gap'] <= TOLERANCE
    checks['passed'] = (not checks['check1_grids_differ'] and not checks['check2_window_not_the_raw_crop'] and not checks['check3_counts_differ']
                        and bool(checks['check4']) and all(f['within'] for f in checks['check4'].values()))
    write_json(out/'checks.json', checks); print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in checks.items()}), flush=True)
    require(args.limit or checks['passed'], f'a check failed: {json.dumps(checks)[:800]}')
    write_json(out/'complete.json', {'regions': len(rows)})


def summary(args):
    """Per image, part (the readable regions, all) and arm: the regions' mean cos(z, c) per output and class, the raw's beside them, the
    patches taken (share of the image's), and NAFNet's places against the human's (summed patches: recall = both / human, precision =
    both / NAFNet)."""
    out = make_output_dir(args.output_dir)
    rows = {r['id']: r for r in json.loads((args.pack/'pack.json').read_text())['rows']}
    measured = [json.loads(l) for l in (args.measured/'regions.jsonl').read_text().splitlines()]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.measured/'regions.jsonl', args.pack/'pack.json')})
    mean = lambda v: float(np.mean(v)) if v else None
    result = {}
    for view, _ in IMAGES:
        for part, inside in (('readable', lambda i: rows[i]['readable']), ('all', lambda i: True)):
            sel = [m[view] for m in measured if view in m and inside(m['id'])]
            e = {'regions': len(sel), 'raw': {}, 'arms': {}}
            cos = lambda get: {name: {cls: mean([v for v in (get(s)[name][cls]['cos'] for s in sel) if v is not None]) for cls in ('all', *CLASSES)} for name in OUTPUTS}
            e['raw'] = cos(lambda s: s['raw'])
            for arm, *_ in ARMS:
                e['arms'][arm] = {'closeness': cos(lambda s, a=arm: s['arms'][a]),
                                  'share_taken': mean([s['taken'][arm]/s['patches']['all'] for s in sel])}
            places = {k: {q: int(sum(s['nafnet_places'][k][q] for s in sel)) for q in ('both', 'nafnet', 'human')} for k in ('struck', 'next')}
            for v in places.values():
                v.update(recall=v['both']/max(1, v['human']), precision=v['both']/max(1, v['nafnet']))
            e['nafnet_places'] = places
            result[f'{view}:{part}'] = e
    write_json(out/'summary.json', result)
    w = result['window:readable']
    print(json.dumps({'regions': w['regions'], 'raw_merged': w['raw']['merged']['all'],
                      'merged_all': {a: w['arms'][a]['closeness']['merged']['all'] for a in w['arms']}, 'nafnet_places': w['nafnet_places']}, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure')
    for key in ('model', 'pack', 'transplant-pack', 'nafnet-pack', 'spread', 'output-dir'): m.add_argument('--'+key, type=Path, required=True)
    m.add_argument('--threads', type=int, default=8); m.add_argument('--check-regions', type=int, default=10, help='per fold, the regions check 4 runs on')
    m.add_argument('--limit', type=int, help='a dry run: the first LIMIT regions of each fold')
    s = sub.add_parser('summary')
    for key in ('measured', 'pack', 'output-dir'): s.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(argv)
    (measure if args.command == 'measure' else summary)(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
