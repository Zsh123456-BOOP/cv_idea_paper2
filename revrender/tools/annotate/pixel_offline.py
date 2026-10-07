"""Bundle native scans, human progress and the shared editor in one offline HTML."""
import base64
import copy
import hashlib
import json
import shutil

import numpy as np
from PIL import Image

from ...data.clean_pages import review_clean
from ...data.pixel_masks import decode, encode
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest


def script_json(value):
    return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')


def standalone(args):
    from .pixel_review import HERE, ASSETS, write_assets
    payload = json.loads((args.pack/'pack.json').read_text())
    original = json.loads(args.review_export.read_text())
    require(original.get('schema') == 'paper2-pixel-review-v1' and original.get('pack') == payload['id'],
            'annotation file does not belong to this pack')
    known = {p['id'] for p in payload['pages']}
    require(set(original['pages']) <= known, 'unknown annotation pages')
    initial = copy.deepcopy(original)
    inputs = {str(p): digest(p) for p in [args.pack/'pack.json', args.review_export, *(HERE/n for n in ASSETS),
                                        HERE/'pixel_offline.py', HERE.parent.parent/'data/clean_pages.py',
                                        HERE.parent.parent/'tighten.py']}
    images, stats, seen = [], [], set()
    for pi, page in enumerate(payload['pages']):
        path = args.pack/page['image']
        require('/' not in page['id'] and '\\' not in page['id'], 'invalid image ID')
        with Image.open(path) as im:
            require(list(im.size) == page['size'], 'page dimensions changed')
            grey = np.asarray(im.convert('L'))
        clean = review_clean(grey)
        inputs[str(path)] = digest(path)
        human = initial['pages'].get(page['id'], {})
        records = [r for r in payload['rows'] if r['page'] == pi]
        human_records = human.get('revisions', []) + human.get('deleted_revisions', [])
        removed, reconfirm = 0, 0
        for row in human_records:
            require(row['id'] not in seen, 'duplicate annotation region')
            seen.add(row['id'])
            base = next((r for r in payload['rows'] if r['id'] == row['id']), None)
            require(base is None or base['page'] == pi, 'region belongs to another page')
        for row in records + human_records:
            box = row['box']; w,h = page['size']
            require(len(box) == 4 and all(type(v) is int for v in box), 'invalid box')
            l,t,r,b = box
            require(0 <= l < r <= w and 0 <= t < b <= h, 'box outside image')
            region = row['region']
            require(len(region) >= 3 and all(len(p) == 2 and 0 <= p[0] <= w and 0 <= p[1] <= h for p in region),
                    'invalid region')
            mask = decode(row['runs'], (b-t,r-l))
            kept = mask & (clean[t:b,l:r] < 245)
            count = int((mask & ~kept).sum())
            row['runs'] = encode(kept)
            if count and any(row is item for item in human_records):
                removed += count
                reconfirm += bool(row.get('confirmed'))
                row['needs_clean_review'] = bool(row.get('confirmed')) or row.get('needs_clean_review', False)
                row['confirmed'] = False
                row['cleaned_pixels'] = row.get('cleaned_pixels', 0) + count
                human['done'] = False
        images.append(clean)
        stats.append({'page':page['id'], 'whitened_pixels':int((clean != grey).sum()),
                      'removed_human_mask_pixels':removed, 'confirmations_to_review':reconfirm})
        page['image'] = 'images/'+page['id']+'.png'
    revision = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    out = make_output_dir(args.output_dir)
    (out/'images').mkdir()
    for page, image in zip(payload['pages'], images):
        Image.fromarray(image).convert('RGB').save(out/page['image'])
    payload.update(image_revision=revision, image_source=str((out/'images').resolve()))
    initial.update(image_revision=revision, image_source=payload['image_source'])
    # Exact original remains recoverable, even when pale mask pixels need reconfirmation.
    shutil.copyfile(args.review_export, out/'source_annotations.json')
    write_json(out/'imported_annotations.json', initial)
    payload['initial_export'] = initial
    write_json(out/'pack.json', payload)
    write_assets(out)
    inline = copy.deepcopy(payload)
    inline['standalone'] = True
    for page in inline['pages']:
        page['image'] = 'data:image/png;base64,'+base64.b64encode((out/page['image']).read_bytes()).decode()
    html = (HERE/'pixel_editor.html').read_text()
    data = '<script id="review-pack" type="application/json">'+script_json(inline)+'</script>'
    data += '<script id="review-draft" type="application/json">null</script>'
    html = html.replace('<script src="page_rules.js">', data+'<script src="page_rules.js">')
    for name in ASSETS:
        if name.endswith('.js'):
            code = (HERE/name).read_text().replace('</script', '<\\/script')
            html = html.replace(f'<script src="{name}"></script>', '<script>'+code+'</script>')
    (out/'标注.html').write_text(html)
    write_manifest(out, config=to_jsonable(args), inputs=inputs, cleaning='connected-ink-soft-edge',
                   pages=stats, revisions=sum(len(p.get('revisions', [])) for p in initial['pages'].values()),
                   completed_questions=sum(bool(p.get('done')) for p in initial['pages'].values()))
    print(json.dumps({'output':str(out/'标注.html'), 'pages':stats}, ensure_ascii=False))
