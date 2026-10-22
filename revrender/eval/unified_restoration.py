"""Frozen-model comparison on shared native pixel and OCR validation/test inputs.

Every restorer receives identical source crops; architecture-specific inference geometry remains as trained.  ``choose`` picks each
family's version from candidate versions evaluated together (read-back on validation), ``compare`` pairs two such choices.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages
from ..baselines.restoration import removal_counts, removal_metrics, restore_native
from ..data.clean_pages import review_clean
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .revision_boxes import prepare as prepare_text, read_labels, unpacked


PROTOCOL = {
    'pixel_ink': 'source grayscale < 200 (both prediction and target)',
    'pixel_removed': 'restored minus source > 0.2 on [0,1]',
    'pixel_scope': 'native validation crops; test polygon union counted once per page',
    'ocr_scope': 'readable confirmed val/test boxes with identical six-pixel context',
    'ocr': 'one Qwen3-VL-8B-Instruct TP2 run, erased_neutral, greedy, seed42, max256 tokens',
    'preprocessing': 'review_clean source; no UI residual cleanup; same native output coordinates',
    'selection': 'all checkpoints frozen before inference; no selection from new test results',
    'own_inference': 'trained 768x96 letterbox, top1, mask0.4, nms0.5, score0',
    'baseline_inference': 'trained native256 tiling with stride192',
}


def freeze_specs(specs):
    require(len({s['name'] for s in specs}) == len(specs), 'duplicate model name')
    frozen = []
    for spec in specs:
        require(spec['name'] != 'scan' and spec['name'].replace('-', '').replace('_', '').isalnum(), 'invalid model name')
        item = dict(spec)
        if spec['type'] == 'own':
            files = [Path(spec['checkpoint']), *map(Path, spec.get('provenance_files', []))]
        else:
            require(spec['type'] == 'baseline', 'unsupported model type')
            run = Path(spec['run']); selection = json.loads((run/'selection.json').read_text())
            # an explicit step directory of the run (a candidate version), else the run's own selection
            checkpoint = run/(spec.get('checkpoint') or selection['checkpoint'])
            model = json.loads((run/'run_manifest.json').read_text())['config']['model']
            files = [run/'selection.json', run/'run_manifest.json', checkpoint/('generator.h5' if model == 'DE-GAN' else 'weights.pt')]
            item['selection'] = selection
        item['sha256'] = {str(p): digest(p) for p in files}
        frozen.append(item)
    return frozen


def verify_files(hashes):
    for path, expected in hashes.items():
        require(digest(path) == expected, f'frozen input changed: {path}')


def extend_inputs(args):
    """Append frozen candidates without regenerating or changing scored inputs."""
    base=args.base; source=base/'inputs'
    require((base/'COMPLETED').is_file(), 'base comparison is incomplete')
    verify_files(json.loads((source/'prepared_sha256.json').read_text()))
    prior=json.loads((source/'models.json').read_text())
    added=freeze_specs(json.loads(args.models.read_text()))
    require(not {s['name'] for s in prior}&{s['name'] for s in added}, 'candidate already evaluated')
    for spec in prior: verify_files(spec['sha256'])
    out=make_output_dir(args.output_dir); inputs=make_output_dir(out/'inputs'); runs=make_output_dir(out/'runs')
    copied={}
    for name in ('crops','protocol.json','pack.json','pixels.json','pixels.npz','counts.json'):
        src=source/name; dst=inputs/name
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copyfile(src,dst)
        files=sorted(src.rglob('*')) if src.is_dir() else [src]
        for file in files:
            if file.is_file():
                dest=inputs/file.relative_to(source)
                require(digest(dest)==digest(file), 'copied input differs')
                copied[str(file)]=digest(file)
    write_json(inputs/'models.json',prior+added)
    for spec in prior:
        src=base/'runs'/spec['name']; manifest=json.loads((src/'run_manifest.json').read_text())
        require(manifest['model']==spec, 'cached inference used different model')
        require(manifest['inputs'][str(source/'prepared_sha256.json')]==digest(source/'prepared_sha256.json'), 'cached inference used different inputs')
        shutil.copytree(src,runs/spec['name'])
        for file in sorted(src.rglob('*')):
            if file.is_file():
                expected=digest(file)
                require(digest(runs/spec['name']/file.relative_to(src))==expected, 'copied inference differs')
                copied[str(file)]=expected
    write_json(out/'reused_sha256.json',copied)
    generated=[p for p in inputs.rglob('*') if p.is_file()]
    write_json(inputs/'prepared_sha256.json',{str(p):digest(p) for p in generated})
    write_manifest(inputs, config=to_jsonable(args), inputs={str(source/'prepared_sha256.json'):digest(source/'prepared_sha256.json'),
        str(args.models):digest(args.models),str(out/'reused_sha256.json'):digest(out/'reused_sha256.json')},
        reused_models=[s['name'] for s in prior],new_models=[s['name'] for s in added],protocol=PROTOCOL)


def prepare_inputs(args):
    out = make_output_dir(args.output_dir)
    specs = freeze_specs(json.loads(args.models.read_text()))
    # Freeze choices before reading any held-out image or label.
    write_json(out/'models.json', specs); write_json(out/'protocol.json', PROTOCOL)
    selection = {r['image_id']: r for r in json.loads(args.selection.read_text())['pages']}
    labels = read_labels(args.labels)
    revisions = [(selection[i]['split'], i, r) for i, q in sorted(labels.items())
                 if q.get('done') and selection.get(i, {}).get('split') in ('val', 'test')
                 for r in q['revisions'] if not r.get('deleted') and not r.get('unreadable')
                 and prepare_text(r.get('text', '')).replace('?', '')]
    # the old seven pixel test pages (--geometry) are optional: that test set was retired on 2026-10-18 (it held a labmate's pages)
    geometry = [] if args.geometry is None else [*args.geometry.glob('*.json'), *args.geometry.glob('*.npz')]
    geometry_ids = sorted(p.stem for p in geometry if p.suffix == '.json' and p.stem not in ('summary', 'run_manifest'))
    require(all(selection[i]['split'] == 'test' for i in geometry_ids), 'pixel test page outside test split')
    files = pages.page_files(args.pages, sorted({i for _, i, _ in revisions} | set(geometry_ids)))
    scans = {}
    def scan(i):
        if i not in scans:
            with Image.open(files[i]) as im: scans[i] = review_clean(np.asarray(im.convert('L')))
        return scans[i]
    (out/'crops').mkdir(); rows = []
    for split, image_id, r in revisions:
        grey = scan(image_id); points = np.asarray(r['region'])
        l, t = np.maximum(0, points.min(0)-6).astype(int)
        rr, b = np.minimum([grey.shape[1], grey.shape[0]], points.max(0)+6).astype(int)
        require(rr > l and b > t, 'empty OCR crop')
        path = f'crops/{len(rows):05d}.png'; Image.fromarray(grey[t:b, l:rr]).save(out/path)
        key = image_id+'#'+r['id']
        rows.append(dict(id=split+'|scan|'+key, line_id=key, split=split, model=split+'__scan', image_id=image_id,
                         kind=r['kind'], text=r['text'], boxes=[], size=[int(rr-l), int(b-t)], images={'erased':path}))
    write_json(out/'pack.json', {'schema':'revrender-line-pack', 'rows':rows})
    native_rows = json.loads(args.human_validation.read_text()); native = np.load(args.human_validation.with_suffix('.npz'), allow_pickle=False)
    arrays = {}; cases = []
    for i, row in enumerate(native_rows):
        grey = native[f'image_{i}']; shape = grey.shape; k = len(cases)
        arrays[f'image_{k}'] = grey; arrays[f'gold_{k}'] = native[f'mask_{i}'].astype(bool)
        arrays[f'domain_{k}'] = np.ones(shape, bool); arrays[f'region_{k}_0'] = np.ones(shape, bool)
        cases.append(dict(id=row['id'], split='val', windows=[[0, 0, shape[1], shape[0]]]))
    for image_id in geometry_ids:
        grey = scan(image_id); k = len(cases); record = json.loads((args.geometry/f'{image_id}.json').read_text())
        masks = np.load(args.geometry/f'{image_id}.npz', allow_pickle=False)
        domain = np.zeros(grey.shape, bool); gold = np.zeros(grey.shape, bool); windows = []
        for n, r in enumerate(record['revisions']):
            if r['kind'] != 'line': continue
            region = unpacked(masks[f'region_{n}'], grey.shape[1]); ys, xs = np.nonzero(region)
            if not len(xs): continue
            l = max(0, int(xs.min())-6); t = max(0, int(ys.min())-6)
            rr = min(grey.shape[1], int(xs.max())+7); b = min(grey.shape[0], int(ys.max())+7)
            arrays[f'region_{k}_{len(windows)}'] = region[t:b, l:rr]
            windows.append([l, t, rr, b]); domain |= region
            gold |= unpacked(masks[f'revision_{n}'], grey.shape[1])
        arrays[f'image_{k}'] = grey; arrays[f'gold_{k}'] = gold; arrays[f'domain_{k}'] = domain
        cases.append(dict(id=image_id, split='test', windows=windows))
    write_json(out/'pixels.json', cases); np.savez_compressed(out/'pixels.npz', **arrays)
    counts = {split:dict(pixel_cases=sum(c['split']==split for c in cases),
                        pixel_regions=sum(len(c['windows']) for c in cases if c['split']==split),
                        ocr_regions=sum(r['split']==split for r in rows),
                        ocr_lines=sum(r['split']==split and r['kind']=='line' for r in rows)) for split in ('val','test')}
    require(counts['val']['pixel_regions']==16 and counts['val']['ocr_lines']==41, 'unexpected validation population')
    inputs = [args.models, args.selection, *args.labels, args.human_validation, args.human_validation.with_suffix('.npz'),
              *files.values(), *geometry]
    write_json(out/'counts.json', counts)
    generated = [out/'models.json', out/'protocol.json', out/'pack.json', out/'pixels.json', out/'pixels.npz', *sorted((out/'crops').glob('*.png'))]
    write_json(out/'prepared_sha256.json', {str(p):digest(p) for p in generated})
    write_manifest(out, config=to_jsonable(args), inputs={str(p):digest(p) for p in inputs}, counts=counts, protocol=PROTOCOL)
    print(json.dumps(counts), flush=True)


def restore_case(restore, grey, windows, regions):
    raw = grey.astype(np.float32)/255; restored = raw.copy()
    for window, region in zip(windows, regions):
        l, t, r, b = window; value = restore(raw[t:b, l:r])
        require(value.shape == region.shape and np.isfinite(value).all(), 'invalid restored crop')
        restored[t:b, l:r][region] = value[region]
    return restored


def infer_model(args):
    inputs = args.inputs
    verify_files(json.loads((inputs/'prepared_sha256.json').read_text()))
    spec = next(s for s in json.loads((inputs/'models.json').read_text()) if s['name']==args.name)
    verify_files(spec['sha256']); out = make_output_dir(args.output_dir)
    if spec['type'] == 'own':
        from ..infer import load_checkpoint
        from .review_removal import predict_crops
        model, _ = load_checkpoint(Path(spec['checkpoint']))
        def restore(raw):
            grey = np.rint(raw*255).astype(np.uint8)
            mask = predict_crops(model, [grey])[0]
            return np.where(mask, 1., raw).astype(np.float32)
    else:
        from ..baselines.restoration_export import selected_predictor
        predictor, _, _ = selected_predictor(Path(spec['run']), Path(spec['upstream']), spec.get('checkpoint'))
        def restore(raw): return restore_native(predictor, raw)
    cases = json.loads((inputs/'pixels.json').read_text()); archive = np.load(inputs/'pixels.npz', allow_pickle=False)
    totals = {s:np.zeros(5, np.int64) for s in ('val','test')}; detail = []; outputs = {}
    for k, case in enumerate(cases):
        grey = archive[f'image_{k}']; domain = archive[f'domain_{k}']; gold = archive[f'gold_{k}']
        value = restore_case(restore, grey, case['windows'], [archive[f'region_{k}_{j}'] for j in range(len(case['windows']))])
        count = removal_counts(grey[domain].astype(np.float32)/255, value[domain], gold[domain]); totals[case['split']] += count
        detail.append({**case, **removal_metrics(count), 'valid_ink':int(count[3])})
        outputs[f'image_{k}'] = np.rint(value*255).astype(np.uint8)
    np.savez_compressed(out/'restored_pixels.npz', **outputs)
    write_json(out/'pixels.json', {'model':args.name, 'splits':{s:{**removal_metrics(c),'valid_ink':int(c[3])} for s,c in totals.items()}, 'cases':detail})
    pack = make_output_dir(out/'pack'); (pack/'crops').mkdir(); rows = []
    for r in json.loads((inputs/'pack.json').read_text())['rows']:
        source = inputs/r['images']['erased']
        with Image.open(source) as im: raw = np.asarray(im.convert('L'), np.float32)/255
        value = restore(raw); require(value.shape==raw.shape and np.isfinite(value).all(), 'invalid OCR image')
        path = f'crops/{len(rows):05d}.png'; Image.fromarray(np.rint(value*255).astype(np.uint8)).save(pack/path)
        rows.append({**r, 'id':r['split']+'|'+args.name+'|'+r['line_id'], 'model':r['split']+'__'+args.name, 'images':{'erased':path}})
    write_json(pack/'pack.json', {'schema':'revrender-line-pack','rows':rows})
    write_manifest(out, config=to_jsonable(args), inputs={str(inputs/'prepared_sha256.json'):digest(inputs/'prepared_sha256.json'),**spec['sha256']}, model=spec, protocol=PROTOCOL)
    print(json.dumps({'model':args.name,'splits':{s:removal_metrics(c) for s,c in totals.items()},'ocr_images':len(rows)}), flush=True)


def merge(args):
    out = make_output_dir(args.output_dir); rows = []; expected = json.loads((args.inputs/'pack.json').read_text())['rows']
    specs = json.loads((args.inputs/'models.json').read_text())
    for directory in [args.inputs, *[args.runs/s['name']/'pack' for s in specs]]:
        part = json.loads((directory/'pack.json').read_text())['rows']
        require([(r['split'],r['line_id'],r['text'],r['size']) for r in part] == [(r['split'],r['line_id'],r['text'],r['size']) for r in expected], 'unmatched OCR population')
        rows.extend({**r,'images':{'erased':str((directory/r['images']['erased']).resolve())}} for r in part)
    # Keep every model's version of each source region in the same OCR batch.
    order = {r['split']+'|'+r['line_id']:i for i,r in enumerate(expected)}
    rows.sort(key=lambda r:(order[r['split']+'|'+r['line_id']],r['model']))
    require(len({r['id'] for r in rows})==len(rows),'duplicate OCR request')
    write_json(out/'pack.json',{'schema':'revrender-line-pack','rows':rows})
    write_manifest(out, config=to_jsonable(args), inputs={str(args.inputs/'models.json'):digest(args.inputs/'models.json')}, rows=len(rows))


def candidates(args):
    """The versions to choose from, as a model list for ``eval_unified_restoration.sh``: a box remover's kept checkpoints (``--own``,
    every ``--every`` steps and the last, as a restorer keeps one each evaluation it improves; all of them without ``--every``) and every
    step directory a restorer's run kept (``--run FAMILY=RUN``)."""
    import re
    def step_of(name):          # step_<n>[_raw|_ema]: an EMA version after the raw one of its step
        m = re.fullmatch(r'step_(\d+)(?:_(raw|ema))?(?:\.pt)?', name); return int(m[1])*10+(m[2] == 'ema')
    label = lambda n: f'{n//10}'+('ema' if n % 10 else '')
    specs = []
    for own in args.own or []:
        family, folder = own.split('=', 1)
        mine = sorted(Path(folder).glob('step_*.pt'), key=lambda p: step_of(p.name))
        if args.every:
            mine = [p for p in mine if step_of(p.name)//10 % args.every == 0] + [mine[-1]]*(step_of(mine[-1].name)//10 % args.every != 0)
        specs += [dict(name=f'{family}-s{label(step_of(p.name))}', family=family, step=step_of(p.name), type='own', checkpoint=str(p)) for p in mine]
    for run in args.run or []:
        family, folder = run.split('=', 1); root = Path(folder)
        model = json.loads((root/'run_manifest.json').read_text())['config']['model']
        upstream = args.stain_upstream if model == 'StainRestorer' else args.upstream
        for d in sorted((d for d in root.glob('step_*') if d.is_dir()), key=lambda d: step_of(d.name)):
            specs.append(dict(name=f'{family}-s{label(step_of(d.name))}', family=family, step=step_of(d.name), type='baseline',
                              run=str(root), upstream=str(upstream), checkpoint=d.name))
    require(specs, 'no versions'); require(not args.output.exists(), 'output exists')
    write_json(args.output, specs); print(len(specs), 'versions', flush=True)


def row_scores(root):
    """Per model, {line_id: share of the struck characters held} on the validation lines (``eval.revision_boxes.held``)."""
    from .revision_boxes import held
    rows = {r['id']: r for r in json.loads((root/'pack/pack.json').read_text())['rows']}
    scores = {}
    for line in (root/'qwen8b/predictions.jsonl').read_text(encoding='utf-8').splitlines():
        reading = json.loads(line); r = rows[reading['id']]
        if r['split'] != 'val' or r['kind'] != 'line': continue
        share = held(r['text'], reading['text'])
        if share is not None: scores.setdefault(r['model'].split('__', 1)[1], {})[r['line_id']] = share
    return scores


def choose(args):
    """Each family's version by the rule of 2026-10-22: its candidates are the versions whose pixel F1 on the 16 fixed validation
    regions beat every earlier one, and its last (the versions a restorer's own training keeps); of these, the highest read-back on
    the 41 validation strikes, the earlier step on a tie.  A family is ``family`` in the model list, a version's order ``step``."""
    root = args.evaluation; specs = json.loads((root/'inputs/models.json').read_text()); scores = row_scores(root)
    families = {}
    for spec in specs: families.setdefault(spec['family'], []).append(spec)
    chosen = {}
    for family, versions in sorted(families.items()):
        versions.sort(key=lambda s: s['step']); best = -1; table = []
        for i, spec in enumerate(versions):
            f1 = json.loads((root/'runs'/spec['name']/'pixels.json').read_text())['splits']['val']['f1']
            held = float(np.mean(list(scores[spec['name']].values())))
            candidate = f1 > best or i == len(versions)-1; best = max(best, f1)
            table.append(dict(name=spec['name'], step=spec['step'], pixel_f1=f1, read_back=held, candidate=candidate))
        pick = max((t for t in table if t['candidate']), key=lambda t: (t['read_back'], -t['step']))
        chosen[family] = dict(chosen=pick, versions=table, spec=next(s for s in versions if s['name'] == pick['name']))
    out = make_output_dir(args.output_dir); write_json(out/'selection.json', chosen)
    write_manifest(out, config={'evaluation': str(root)}, inputs={str(p): digest(p) for p in [root/'inputs/models.json', root/'qwen8b/predictions.jsonl']},
                   rule='candidates: pixel-F1 improvements on the 16 fixed validation regions and the last version; chosen: highest read-back on the 41 validation strikes, earlier step on a tie')
    print(json.dumps({f: c['chosen'] for f, c in chosen.items()}), flush=True)


def compare(args):
    """The paired difference in read-back between two chosen versions of every family (``choose`` outputs: before and after), on the
    validation strikes both read, with 95% intervals by resampling writers (seed 42)."""
    before = json.loads((args.before/'selection.json').read_text()); after = json.loads((args.after/'selection.json').read_text())
    a_scores = row_scores(Path(json.loads((args.before/'run_manifest.json').read_text())['config']['evaluation']))
    b_scores = row_scores(Path(json.loads((args.after/'run_manifest.json').read_text())['config']['evaluation']))
    rng = np.random.default_rng(42); result = {}
    for family in sorted(set(before) & set(after)):
        x = a_scores[before[family]['chosen']['name']]; y = b_scores[after[family]['chosen']['name']]
        lines = sorted(set(x) & set(y)); writers = sorted({l.split('_')[0] for l in lines})
        by = {w: [l for l in lines if l.startswith(w+'_')] for w in writers}
        diff = lambda ls: float(np.mean([y[l]-x[l] for l in ls]))
        draws = [diff([l for w in rng.choice(writers, len(writers)) for l in by[w]]) for _ in range(1000)]
        result[family] = dict(before=float(np.mean([x[l] for l in lines])), after=float(np.mean([y[l] for l in lines])),
                              difference=diff(lines), interval=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                              strikes=len(lines), writers=len(writers))
    # every family together: the mean of the families' paired differences, writers resampled jointly (the decision rule's pooled test)
    families = sorted(result); lines = {f: sorted(set(a_scores[before[f]['chosen']['name']]) & set(b_scores[after[f]['chosen']['name']])) for f in families}
    writers = sorted({l.split('_')[0] for f in families for l in lines[f]})
    def pooled(chosen_writers):
        diffs = []
        for f in families:
            x = a_scores[before[f]['chosen']['name']]; y = b_scores[after[f]['chosen']['name']]
            ls = [l for w in chosen_writers for l in lines[f] if l.startswith(w+'_')]
            if ls: diffs.append(np.mean([y[l]-x[l] for l in ls]))
        return float(np.mean(diffs))
    draws = [pooled(rng.choice(writers, len(writers))) for _ in range(1000)]
    result['_all'] = dict(difference=pooled(writers), interval=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                          families=len(families), improved=sum(result[f]['difference'] > 0 for f in families), writers=len(writers))
    out = make_output_dir(args.output_dir); write_json(out/'comparison.json', result)
    write_manifest(out, config={'before': str(args.before), 'after': str(args.after)}, inputs={str(p/'selection.json'): digest(p/'selection.json') for p in (args.before, args.after)})
    print(json.dumps(result), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    for key in ('models','selection','human-validation','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--geometry',type=Path)
    for key in ('labels','pages'):p.add_argument('--'+key,type=Path,action='append',required=True)
    p = sub.add_parser('infer')
    for key in ('inputs','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--name',required=True)
    p = sub.add_parser('merge')
    for key in ('inputs','runs','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p = sub.add_parser('extend')
    for key in ('base','models','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p = sub.add_parser('candidates')
    p.add_argument('--own', action='append', help='FAMILY=FOLDER of step_<n>.pt'); p.add_argument('--every', type=int)
    p.add_argument('--run', action='append', help="FAMILY=RUN of a restorer (its kept step_<n> directories)")
    for key in ('upstream', 'stain-upstream', 'output'): p.add_argument('--'+key, type=Path, required=key == 'output')
    p = sub.add_parser('choose')
    for key in ('evaluation','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p = sub.add_parser('compare')
    for key in ('before','after','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args(argv)
    {'prepare':prepare_inputs,'infer':infer_model,'merge':merge,'extend':extend_inputs,'candidates':candidates,'choose':choose,'compare':compare}[args.command](args)


if __name__ == '__main__': main()
