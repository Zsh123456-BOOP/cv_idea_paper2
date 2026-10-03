"""A static HTML viewer for the output of ``revrender.pages``: the pages with their recognised revisions stitched back, and the pages
a short-answer filter set aside.

    python -m revrender.tools.pages_view --pages-output OUT --images DIR [--images DIR ...] --answer-text FILE [--answer-text FILE ...]
        [--tags tags.json] [--min-chars 10] --output-dir SITE

A page is set aside when the student wrote fewer than ``--min-chars`` characters on it: ``data.ocr_lines.answer_chars`` of the page's
student-only transcription (no printed question, no crossed-out writing), which a page that is blank apart from its printed question
gets 0 or a few for.  SITE/kept.html shows the other pages with every recognised revision drawn over the page, SITE/filtered.html the pages
set aside with their character count and transcription, SITE/index.html the counts.  A page and, when something was recognised on it, a
transparent mask are copied to SITE/assets, so the folder opens from anywhere without a server.  SITE/kept_ids.txt and filtered_ids.txt
list the two sets.  ``--tags`` is a json {image_id: {split, rows}}, shown as badges (the writer's split in paper 1 and its line rows).
"""
import argparse
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..data.ocr_lines import answer_chars, answer_texts
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

TEMPLATE = Path(__file__).with_name('pages_view.html')
RED, FILL_ALPHA, EDGE_ALPHA = (220, 0, 0), 70, 235
TEXT_LIMIT = 1200


def read_page(output_dir, image_id):
    """(the page's record, the union of its instances in the page frame or None, each instance's tight box (l, t, r, b))."""
    record = json.loads((Path(output_dir)/f'{image_id}.json').read_text())
    masks = np.load(Path(output_dir)/f'{image_id}.npz')
    width, height = record['size']
    union, boxes = np.zeros((height, width), bool), []
    for n, instance in enumerate(record['instances']):
        left, top, right, bottom = instance['box']
        crop = np.unpackbits(masks[f'mask_{n}'], axis=-1)[:, :right-left].astype(bool)
        ys, xs = np.nonzero(crop)
        boxes.append([int(left+xs.min()), int(top+ys.min()), int(left+xs.max()+1), int(top+ys.max()+1)])
        union[top:bottom, left:right] |= crop
    return record, (union if boxes else None), boxes


def mask_image(union):
    """A transparent RGBA overlay: the recognised revisions in soft red with a firm edge."""
    edge = union & ~ndimage.binary_erosion(union, iterations=2)
    rgba = np.zeros((*union.shape, 4), np.uint8)
    rgba[union] = (*RED, FILL_ALPHA)
    rgba[edge] = (*RED, EDGE_ALPHA)
    return Image.fromarray(rgba, 'RGBA')


def question_of(image_id):
    """(question number, whether the page is from the supplementary pack) of ``<writer>_<q>_<part>`` / ``<writer>_s<q>_<part>``."""
    middle = image_id.split('_')[1]
    return int(middle.lstrip('s')), middle.startswith('s')


def page_file(directories, image_id):
    for directory in directories:
        path = Path(directory)/f'{image_id}.jpg'
        if path.is_file():
            return path
    raise FileNotFoundError(f'page image {image_id} not found')


def embed(payload):
    """JSON for a <script type="application/json"> block."""
    return json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')


def index_page(summary):
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k, v in summary['rows'])
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>整道题的涂改识别</title>
<style>body{{font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#1e1e1c;background:#f5f5f2}}
table{{border-collapse:collapse;width:100%;background:#fff}}td{{border:1px solid #dcdcd6;padding:6px 10px}}td:last-child{{text-align:right}}
a.big{{display:inline-block;margin:8px 12px 8px 0;padding:10px 18px;border-radius:8px;background:#c62828;color:#fff;text-decoration:none}}
a.big.alt{{background:#555}}p{{color:#555}}</style></head><body>
<h1>整道题的涂改识别</h1>
<p>{html.escape(summary['note'])}</p>
<p><a class="big" href="kept.html">识别完并拼回整页的（{summary['kept']} 页）</a><a class="big alt" href="filtered.html">被过滤掉的（{summary['filtered']} 页）</a></p>
<table>{rows}</table>
</body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pages-output', type=Path, required=True, help='the output folder of revrender.pages')
    parser.add_argument('--images', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--answer-text', type=Path, action='append', required=True, help='jsonl of {image_id, text}: the student-only transcription')
    parser.add_argument('--tags', type=Path, help='json {image_id: {split, rows}} shown as badges')
    parser.add_argument('--min-chars', type=int, default=10, help='a page with fewer characters than this is set aside')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    ids = sorted(p.stem for p in args.pages_output.glob('*.json') if p.name != 'run_manifest.json')
    texts = answer_texts(args.answer_text)
    missing = [i for i in ids if i not in texts]
    require(not missing, f'{len(missing)} page(s) have no transcription record, e.g. {missing[:2]}')
    tags = json.loads(args.tags.read_text()) if args.tags else {}
    out = make_output_dir(args.output_dir)
    (out/'assets').mkdir()
    cards = {'kept': [], 'filtered': []}
    for image_id in ids:
        record, union, boxes = read_page(args.pages_output, image_id)
        shutil.copyfile(page_file(args.images, image_id), out/'assets'/f'{image_id}.jpg')
        if union is not None:
            mask_image(union).save(out/'assets'/f'{image_id}.png', optimize=True)
        chars = answer_chars(texts[image_id])
        question, supplementary = question_of(image_id)
        tag = tags.get(image_id, {})
        cards['kept' if chars >= args.min_chars else 'filtered'].append({
            'id': image_id, 'w': record['size'][0], 'h': record['size'][1], 'q': question, 'supp': supplementary, 'chars': chars,
            'n': len(boxes), 'scores': [round(i['score'], 4) for i in record['instances']], 'boxes': boxes, 'mask': union is not None,
            'lines': [[line['box'], line['source']] for line in record['lines']], 'text': texts[image_id][:TEXT_LIMIT],
            'split': tag.get('split', 'none'), 'rows': tag.get('rows') or {}})
    template = TEMPLATE.read_text(encoding='utf-8')
    titles = {'kept': '识别完并拼回整页的', 'filtered': '被过滤掉的（学生写得太少）'}
    for mode, group in cards.items():
        payload = {'mode': mode, 'min_chars': args.min_chars, 'cards': group}
        (out/f'{mode}.html').write_text(template.replace('__TITLE__', titles[mode]).replace('__MODE__', mode).replace('__DATA__', embed(payload)),
                                        encoding='utf-8')
        (out/f'{mode}_ids.txt').write_text(''.join(c['id']+'\n' for c in group))
    everything = cards['kept']+cards['filtered']
    rows = [('页数', len(everything)), ('识别完拼回的', len(cards['kept'])), ('被过滤的（字数少于 %d）' % args.min_chars, len(cards['filtered'])),
            ('识别出的涂改（总处数）', sum(c['n'] for c in everything)),
            ('识别完拼回的里：至少检出一处的页数', sum(1 for c in cards['kept'] if c['n'])),
            ('被过滤的里：检出了涂改的页数', sum(1 for c in cards['filtered'] if c['n'])),
            ('主扫描 / 补充资料包', '%d / %d' % (sum(not c['supp'] for c in everything), sum(c['supp'] for c in everything)))]
    for split, label in (('train', '第一篇训练书写者的页'), ('dev', '第一篇验证书写者的页'), ('test', '第一篇测试书写者的页'), ('none', '第一篇没用到的书写者的页')):
        rows.append((label, '%d（过滤 %d）' % (sum(c['split'] == split for c in everything), sum(c['split'] == split for c in cards['filtered']))))
    note = ('每页先由 Unlimited-OCR 切成行，每行交给第一篇的最优模型找涂改，再把结果放回整页。字数是学生手写解答转写的字数'
            '（不含印刷的题干和被划掉的内容），少于 %d 个字的页被过滤。' % args.min_chars)
    (out/'index.html').write_text(index_page({'rows': rows, 'kept': len(cards['kept']), 'filtered': len(cards['filtered']), 'note': note}), encoding='utf-8')
    write_json(out/'pages.json', {'min_chars': args.min_chars, 'pages': {c['id']: {'chars': c['chars'], 'instances': c['n'], 'lines': len(c['lines']),
                                                                                   'kept': c['chars'] >= args.min_chars} for c in everything}})
    write_manifest(out, config=to_jsonable(args), inputs={**{str(p): digest(p) for p in args.answer_text},
                                                          **({str(args.tags): digest(args.tags)} if args.tags else {}),
                                                          str(args.pages_output/'run_manifest.json'): digest(args.pages_output/'run_manifest.json')},
                   pages=len(everything), kept=len(cards['kept']), filtered=len(cards['filtered']),
                   chars=dict(Counter(min(c['chars']//10*10, 100) for c in everything)))
    print(json.dumps({'pages': len(everything), 'kept': len(cards['kept']), 'filtered': len(cards['filtered']), 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
