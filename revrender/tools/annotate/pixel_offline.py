"""Bundle native scans, human progress and the shared editor in one offline HTML."""
import base64
import copy
from html.parser import HTMLParser
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
    original = json.loads(args.review_export.read_text()) if args.review_export else {
        'schema':'paper2-pixel-review-v1','pack':payload['id'],'task':payload.get('task','strike_only'),'pages':{}}
    require(original.get('schema') == 'paper2-pixel-review-v1' and original.get('pack') == payload['id'],
            'annotation file does not belong to this pack')
    known = {p['id'] for p in payload['pages']}
    require(set(original['pages']) <= known, 'unknown annotation pages')
    initial = copy.deepcopy(original)
    inputs = {str(p): digest(p) for p in [args.pack/'pack.json', *([args.review_export] if args.review_export else []), *(HERE/n for n in ASSETS),
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
    if args.review_export:
        shutil.copyfile(args.review_export, out/'source_annotations.json')
    write_json(out/'imported_annotations.json', initial)
    payload['initial_export'] = initial
    write_json(out/'pack.json', payload)
    write_assets(out)
    inline = copy.deepcopy(payload)
    inline['standalone'] = True
    for page in inline['pages']:
        page['image'] = 'data:image/png;base64,'+base64.b64encode((out/page['image']).read_bytes()).decode()
    (out/'标注.html').write_text(render_html(inline))
    write_manifest(out, config=to_jsonable(args), inputs=inputs, cleaning='connected-ink-soft-edge',
                   pages=stats, revisions=len(payload['rows']), imported_revisions=sum(len(p.get('revisions', [])) for p in initial['pages'].values()),
                   completed_questions=sum(bool(p.get('done')) for p in initial['pages'].values()))
    print(json.dumps({'output':str(out/'标注.html'), 'pages':stats}, ensure_ascii=False))


KATEX_CSS = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">'
KATEX_JS = '<script async src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js" onload="window.katexLoaded&&window.katexLoaded()"></script>'


def katex_inline(folder):
    """(style, script) of a KaTeX dist folder (katex.min.css, katex.min.js, fonts/*.woff2) with its fonts as data URIs: formulas render
    on a page opened offline, not only when the CDN answers."""
    import re
    css = (folder/'katex.min.css').read_text()
    def font(m):
        data = base64.b64encode((folder/'fonts'/(m.group(1)+'.woff2')).read_bytes()).decode()
        return f'src:url(data:font/woff2;base64,{data}) format("woff2")'
    css = re.sub(r'src:url\(fonts/([A-Za-z0-9_-]+)\.woff2\) format\("woff2"\)[^;}]*', font, css)
    require('url(fonts/' not in css, 'a KaTeX font the stylesheet names is not inlined')
    return '<style>'+css+'</style>', '<script>'+(folder/'katex.min.js').read_text().replace('</script', '<\\/script')+'</script>'


def render_html(payload, draft=None, katex=None, katex_href=None):
    """The offline editor page; ``katex``: a KaTeX dist folder inlined, or ``katex_href``: a folder beside the page holding
    katex.min.css (its fonts inside) and katex.min.js, linked (pages of one library share one copy)."""
    from .pixel_review import HERE, ASSETS
    html = (HERE/'pixel_editor.html').read_text()
    if katex or katex_href:
        require(KATEX_CSS in html and KATEX_JS in html, 'the editor no longer loads KaTeX from the CDN this replaces')
        style, script = katex_inline(katex) if katex else (f'<link rel="stylesheet" href="{katex_href}/katex.min.css">',
                                                           f'<script src="{katex_href}/katex.min.js"></script>')
        html = html.replace(KATEX_CSS, style).replace(KATEX_JS, script)
    data = '<script id="review-pack" type="application/json">'+script_json(payload)+'</script>'
    data += '<script id="review-draft" type="application/json">'+script_json(draft)+'</script>'
    html = html.replace('<script src="page_rules.js">', data+'<script src="page_rules.js">')
    for name in ASSETS:
        if name.endswith('.js'):
            code = (HERE/name).read_text().replace('</script', '<\\/script')
            html = html.replace(f'<script src="{name}"></script>', '<script>'+code+'</script>')
    return html


class EmbeddedReview(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.values = {}; self.active = None; self.feed(html)

    def handle_starttag(self, tag, attrs):
        info = dict(attrs)
        if tag == 'script' and info.get('id') in ('review-pack', 'review-draft'):
            self.active = info['id']; self.values[self.active] = ''

    def handle_data(self, data):
        if self.active: self.values[self.active] += data

    def handle_endtag(self, tag):
        if tag == 'script': self.active = None


def update_html(args):
    """Replace editor assets only; preserve images, pack identity and embedded drafts."""
    from .pixel_review import HERE, ASSETS
    data = EmbeddedReview(args.source.read_text()).values
    require('review-pack' in data, 'not a standalone review HTML')
    payload = json.loads(data['review-pack']); draft = json.loads(data.get('review-draft', 'null'))
    require(payload.get('standalone'), 'not an offline pack')
    out = make_output_dir(args.output_dir)
    (out/'标注.html').write_text(render_html(payload, draft))
    write_manifest(out, config=to_jsonable(args), inputs={str(p):digest(p) for p in
                   [args.source, HERE/'pixel_offline.py', *(HERE/n for n in ASSETS)]},
                   pack=payload['id'], image_revision=payload['image_revision'], embedded_progress_preserved=True)
    print(str(out/'标注.html'))
