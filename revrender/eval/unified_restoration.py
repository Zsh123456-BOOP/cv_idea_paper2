"""Frozen-model comparison on shared native pixel and OCR validation/test inputs.

No optimization or checkpoint selection. Every restorer receives identical
source crops; architecture-specific inference geometry remains as trained.
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
            checkpoint = run/selection['checkpoint']
            files = [run/'selection.json', run/'run_manifest.json', checkpoint/('generator.h5' if spec['name'] == 'DE-GAN' else 'weights.pt')]
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
        predictor, _, _ = selected_predictor(Path(spec['run']), Path(spec['upstream']))
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
    args = parser.parse_args(argv)
    {'prepare':prepare_inputs,'infer':infer_model,'merge':merge,'extend':extend_inputs}[args.command](args)


if __name__ == '__main__': main()
