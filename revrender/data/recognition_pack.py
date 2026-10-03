"""Prepare a writer-diverse development pilot using the existing, completed page cuts.

The original line indices and boxes are retained. Ordering changes presentation only;
no page is segmented again. The strata describe predicted mask geometry, not human labels.
Images and the resulting pack belong under work/ or the server outputs, never in Git.
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages
from ..util import SEED, digest, make_output_dir, require, write_json, write_manifest, to_jsonable
from .ocr_lines import reorder


def geometry(instances):
    if not instances:
        return 'none'
    shapes = []
    for item in instances:
        ys, xs = np.nonzero(item['mask'])
        if not len(ys):
            continue
        height, width = int(ys.max()-ys.min()+1), int(xs.max()-xs.min()+1)
        density = len(ys)/(height*width)
        shapes.append((len(ys), 'thin' if width/height >= 5 else 'block' if density >= .55 else 'mixed'))
    return max(shapes)[1] if shapes else 'none'


def choose(candidates, count=40):
    """Equal predicted-shape strata, balancing writers first and question/source second."""
    require(count > 0 and count % 4 == 0, 'count must be a positive multiple of four')
    pool = sorted(candidates, key=lambda x: x['image_id'])
    random.Random(SEED).shuffle(pool)
    rank = {p['image_id']: i for i, p in enumerate(pool)}
    writers, questions, sources = Counter(), Counter(), Counter()
    selected = []
    for _ in range(count//4):
        for stratum in ('none', 'thin', 'block', 'mixed'):
            available = [p for p in pool if p['stratum'] == stratum]
            require(available, f'not enough development pages in predicted stratum {stratum}')
            def key(p):
                writer, question = p['image_id'].split('_')[:2]
                return writers[writer], questions[question.lstrip('s')], sources[question.startswith('s')], rank[p['image_id']]
            page = min(available, key=key)
            pool.remove(page)
            selected.append(page)
            writer, question = page['image_id'].split('_')[:2]
            writers[writer] += 1
            questions[question.lstrip('s')] += 1
            sources[question.startswith('s')] += 1
    return selected


def ordered_lines(record):
    """Existing cuts in reading order, with their original indices intact."""
    width, height = record['size']
    temporary = []
    for line in record['lines']:
        l, t, r, b = line['box']
        temporary.append({'box': [l/width*1000, t/height*1000, r/width*1000, b/height*1000], 'line': line})
    return [x['line'] for x in reorder(temporary)]


def page_variants(image, instances):
    """B: original clean scan; C: exact-mask whitening; D: auxiliary red mask on the scan."""
    base = np.asarray(image.convert('RGB')).copy()
    union = np.zeros(base.shape[:2], bool)
    for item in instances:
        l, t, r, b = item['box']
        union[t:b, l:r] |= item['mask']
    erased, hint = base.copy(), base.copy()
    erased[union] = 255
    hint[union] = np.rint(.45*np.array([220, 30, 45])+.55*base[union]).astype(np.uint8)
    return {'raw': Image.fromarray(base), 'erased': Image.fromarray(erased), 'hint': Image.fromarray(hint)}, union


def revision_subset(source, count, output):
    """Freeze detector-positive cuts without looking at any recognition outputs."""
    pack_path = source/'pack.json'
    pack = json.loads(pack_path.read_text())
    require(all(p['split'] == 'dev' for p in pack['pages']), 'development pages only')
    candidates = [(p, r) for p in pack['pages'] for r in p['rows'] if r['masked_pixels'] > 0]
    groups = {}
    for p, r in candidates:
        groups.setdefault(p['image_id'], []).append(r['id'])
    require(len(groups) <= count <= len(candidates), 'count must cover each positive page and fit the candidate pool')
    rng = random.Random(SEED)
    # Each positive page has one randomly chosen cut; round-robin adds the rest.
    queues = [sorted(ids) for _, ids in sorted(groups.items())]
    for queue in queues:
        rng.shuffle(queue)
    chosen = []
    while len(chosen) < count:
        rng.shuffle(queues)
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop())
    chosen = set(chosen)
    selected = [{**p, 'rows': [r for r in p['rows'] if r['id'] in chosen]} for p in pack['pages']]
    selected = [p for p in selected if p['rows']]
    out = make_output_dir(output)
    files = {path for p in selected for item in [p, *p['rows']] for path in item['images'].values()}
    hashes = {}
    for relative in sorted(files):
        dest = out/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, dest)
        hashes[relative] = digest(dest)
    selection = {'kind': 'revision-only', 'candidate_rows': len(candidates), 'selected_rows': count,
                 'excluded_rows': [r['id'] for _, r in candidates if r['id'] not in chosen],
                 'rule': 'masked_pixels > 0; seed 42; page-balanced round-robin; no OCR outputs used',
                 'human_positive_confirmed': False}
    payload = {**pack, 'pages': selected, 'selection': selection,
               'note': '仅检测阳性候选；实际是否涂改及关键字符由人工核实；保留原有切块坐标。'}
    write_json(out/'pack.json', payload)
    write_manifest(out, config={'source': str(source), 'count': count},
                   inputs={str(pack_path): digest(pack_path)}, asset_sha256=hashes,
                   pack_sha256=digest(out/'pack.json'), pages=len(selected), rows=count, selection=selection)
    print(json.dumps({'pages': len(selected), 'candidate_rows': len(candidates), 'rows': count}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument('--source-pack', type=Path, help='select only predicted-positive cuts from an existing pack')
    p.add_argument('--pages-output', type=Path)
    p.add_argument('--images', type=Path)
    p.add_argument('--ids', type=Path, help='development page IDs only')
    p.add_argument('--tags', type=Path)
    p.add_argument('--page-index', type=Path, help='pages_view pages.json with the existing answer-length filter')
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    if args.source_pack:
        revision_subset(args.source_pack, args.count, args.output_dir)
        return 0
    require(all((args.pages_output, args.images, args.ids, args.tags, args.page_index)), 'page-building inputs required')
    ids = args.ids.read_text().splitlines()
    tags = json.loads(args.tags.read_text())
    index = json.loads(args.page_index.read_text())['pages']
    candidates = []
    for image_id in ids:
        require(tags.get(image_id, {}).get('split') == 'dev', f'{image_id}: not a declared development page')
        if not index[image_id]['kept']:
            continue
        record, instances = pages.read_output(args.pages_output, image_id)
        if record['lines']:
            candidates.append({'image_id': image_id, 'stratum': geometry(instances)})
    selected = choose(candidates, args.count)
    out = make_output_dir(args.output_dir)
    assets = out/'assets'
    assets.mkdir()
    packed = []
    inputs = {str(path): digest(path) for path in (args.ids, args.tags, args.page_index, args.pages_output/'run_manifest.json')}
    for number, candidate in enumerate(selected, 1):
        image_id = candidate['image_id']
        image_path = args.images/f'{image_id}.png'
        record, instances = pages.read_output(args.pages_output, image_id)
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
        require(list(image.size) == record['size'], 'page size differs from the recorded cuts')
        variants, union = page_variants(image, instances)
        files = {}
        for kind, value in variants.items():
            name = f'assets/{image_id}_{kind}.png'
            value.save(out/name)
            files[kind] = name
        rows = []
        for order, line in enumerate(ordered_lines(record), 1):
            row_id = f'{image_id}:L{line["index"]+1:03d}'
            l, t, r, b = line['box']
            crops = {}
            for kind, value in variants.items():
                name = f'assets/{image_id}_L{line["index"]+1:03d}_{kind}.png'
                value.crop((l, t, r, b)).save(out/name)
                crops[kind] = name
            rows.append({'id': row_id, 'order': order, 'source_index': line['index'], 'box': line['box'],
                         'source': line['source'], 'ocr_type': line['ocr_type'], 'ocr_draft': line['ocr_text'],
                         'masked_pixels': int(union[t:b, l:r].sum()), 'images': crops})
        packed.append({**candidate, 'number': number, 'size': record['size'], 'images': files, 'rows': rows,
                       'question': image_id.split('_')[1], 'answer_chars': index[image_id]['chars'],
                       'predicted_instances': len(instances), 'split': 'dev'})
        for path in (image_path, args.pages_output/f'{image_id}.json', args.pages_output/f'{image_id}.npz'):
            inputs[str(path)] = digest(path)
    payload = {'schema': 'revrender-recognition-pack', 'seed': SEED, 'pages': packed,
               'note': '开发集探索样本；预测几何分层不是真实涂改类别；原切块和索引完整保留。'}
    write_json(out/'pack.json', payload)
    (out/'ids.txt').write_text(''.join(x['image_id']+'\n' for x in packed))
    summary = {'pages': len(packed), 'rows': sum(len(x['rows']) for x in packed),
               'writers': len({x['image_id'].split('_')[0] for x in packed}),
               'strata': dict(Counter(x['stratum'] for x in packed)),
               'sources': dict(Counter(r['source'] for x in packed for r in x['rows']))}
    write_manifest(out, config=to_jsonable(args), inputs=inputs, pack_sha256=digest(out/'pack.json'), **summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
