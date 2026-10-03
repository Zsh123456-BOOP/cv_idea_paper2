"""A static HTML viewer for masks cut to ink: every mask before and after ``revrender.tighten``, the model's and the gold, with the page
erased where the tightened mask is.

    python -m revrender.tools.tighten_view --model BEFORE AFTER [--gold BEFORE AFTER] --images DIR [--images DIR ...] [--ids FILE]
        [--scores BEFORE.json AFTER.json] --output-dir SITE

BEFORE and AFTER are the outputs of ``revrender.pages`` (or ``data.page_gold``) and of ``revrender.tighten`` on it.  SITE/compare.html has a
card per mask: the page crop, the mask as it was, the tightened mask (red: the ink it keeps, blue: the paper it gives up), the crop with the
tightened mask erased (white) and, for comparison, the crop with only the ink of the mask (grey below ``eval.ink.INK_BELOW``, no soft edge)
erased; a switch shows the crops at high contrast so that what an erasure leaves behind can be seen.  SITE/index.html has the counts;
``--scores`` (two ``revrender.eval.pages_score`` files) adds how the model reads against the gold before and after.  The cards come in a
fixed random order (seed 42).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import pages
from ..eval.ink import INK_BELOW
from ..tighten import listed_ids
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_manifest
from .pages_view import embed

TEMPLATE = Path(__file__).with_name('tighten_view.html')
MARGIN = 12                                  # pixels of page around a mask
RED, BLUE, WHITE = (220, 0, 0), (30, 100, 255), (255, 255, 255)
LINE_ASPECT, BLOCK_FILL = 5., .55            # a long thin mark is a line; a mark that fills its box is a block
VISIBLE = 235                                # a pixel darker than this is seen as not white
SPECK = 3                                    # a connected piece of a tightened mask this small (pixels) is a speck
LABELS = {'model': '模型', 'gold': '金标'}


def window(instance, region):
    """The instance's mask on a region (left, top, right, bottom) of the page, clipped to it."""
    left, top, right, bottom = instance['box']
    x0, y0, x1, y1 = region
    out = np.zeros((y1-y0, x1-x0), bool)
    ix0, iy0, ix1, iy1 = max(left, x0), max(top, y0), min(right, x1), min(bottom, y1)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = instance['mask'][iy0-top:iy1-top, ix0-left:ix1-left]
    return out


def tight_box(instance):
    """The mask's own box (left, top, right, bottom) on the page."""
    ys, xs = np.nonzero(instance['mask'])
    left, top = instance['box'][:2]
    return [left+int(xs.min()), top+int(ys.min()), left+int(xs.max())+1, top+int(ys.max())+1]


def shape_of(mask):
    """'line' for a long thin mark (box at least ``LINE_ASPECT`` times wider than tall), 'block' for one that fills its box, else 'other'."""
    ys, xs = np.nonzero(mask)
    width, height = int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1
    return 'line' if width/height >= LINE_ASPECT else ('block' if mask.sum()/(width*height) >= BLOCK_FILL else 'other')


def layer(shape, *parts):
    """An RGBA image of the given size with each (mask, colour, alpha) drawn in turn."""
    rgba = np.zeros((*shape, 4), np.uint8)
    for mask, colour, alpha in parts:
        rgba[mask] = (*colour, alpha)
    return Image.fromarray(rgba, 'RGBA')


def cards_of(name, before_dir, after_dir, ids, files, assets):
    """The cards of one set: a card per mask of BEFORE, with its tightened mask (the AFTER instance that names it) when it has one."""
    cards = []
    for image_id in ids:
        record, before = pages.read_output(before_dir, image_id)
        _, after = pages.read_output(after_dir, image_id)
        cut = {i['from_instance']: i for i in after}
        if not before:
            continue
        with Image.open(files[image_id]) as opened:
            grey = np.asarray(opened.convert('L'))
        width, height = record['size']
        for n, instance in enumerate(before):
            box = tight_box(instance)
            region = (max(0, box[0]-MARGIN), max(0, box[1]-MARGIN), min(width, box[2]+MARGIN), min(height, box[3]+MARGIN))
            old = window(instance, region)
            new = window(cut[n], region) if n in cut else np.zeros_like(old)
            crop = grey[region[1]:region[3], region[0]:region[2]]
            core = old & (crop < INK_BELOW)
            pieces, count = ndimage.label(new, np.ones((3, 3), bool))
            stem = f'{name}_{image_id}_{n}'
            Image.fromarray(crop).save(assets/f'{stem}_c.png', optimize=True)
            layer(old.shape, (old, RED, 120)).save(assets/f'{stem}_b.png', optimize=True)
            layer(old.shape, (old & ~new, BLUE, 150), (new, RED, 150)).save(assets/f'{stem}_a.png', optimize=True)
            layer(old.shape, (new, WHITE, 255)).save(assets/f'{stem}_e.png', optimize=True)
            layer(old.shape, (core, WHITE, 255)).save(assets/f'{stem}_k.png', optimize=True)
            cards.append({'set': name, 'id': image_id, 'n': n, 'file': stem, 'w': region[2]-region[0], 'h': region[3]-region[1],
                          'score': round(float(instance['score']), 4), 'before': int(old.sum()), 'after': int(new.sum()),
                          'ghost_ink': int((old & ~core & (crop < VISIBLE)).sum()), 'ghost': int((old & ~new & (crop < VISIBLE)).sum()),
                          'specks': int((ndimage.sum(new, pieces, range(1, count+1)) <= SPECK).sum()) if count else 0,
                          'shape': shape_of(new) if new.any() else 'gone'})
    return cards


def summary_rows(name, cards):
    """The counts of one set for the index: masks, those with no ink, pixels before and after, the share of a mask given up."""
    if not cards:
        return []
    before, after = sum(c['before'] for c in cards), sum(c['after'] for c in cards)
    shares = np.array([1.-c['after']/c['before'] for c in cards])
    label = LABELS[name]
    return [(f'{label}：掩膜个数', len(cards)), (f'{label}：收紧后没有墨迹、被丢弃的', sum(c['shape'] == 'gone' for c in cards)),
            (f'{label}：掩膜总像素 原来 → 收紧后', f'{before:,} → {after:,}'), (f'{label}：总像素里落在白纸上的', f'{1.-after/before:.1%}'),
            (f'{label}：每个掩膜白纸占比 中位数 / 第 90 百分位', f'{np.quantile(shares, .5):.1%} / {np.quantile(shares, .9):.1%}'),
            (f'{label}：白纸占比 ≥ 30% 的掩膜', f'{int((shares >= .3).sum())}（{(shares >= .3).mean():.1%}）'),
            (f'{label}：擦除后还看得见（灰度 &lt; {VISIBLE}）的残留，占原掩膜像素 只取墨迹 → 收紧后',
             f'{sum(c["ghost_ink"] for c in cards)/before:.1%} → {sum(c["ghost"] for c in cards)/before:.1%}'),
            (f'{label}：收紧后孤立的小杂点（≤ {SPECK} 像素的连通块），每个掩膜平均', f'{sum(c["specks"] for c in cards)/len(cards):.2f} 个')]


def score_rows(before, after):
    """The scoring rows: how the model reads against the gold with the masks as they were and cut to ink."""
    show = lambda v: f'{v:.1%}' if isinstance(v, float) else str(v)
    names = (('gold_components', '金标处数'), ('found', '找到（位置对，形状不一定准）'), ('drawn_right', '画对（IoU ≥ 0.5）'), ('false_positive', '多画的'))
    return [(label, f'{show(before["scores"][key])} → {show(after["scores"][key])}') for key, label in names]


def index_page(rows, scores, note):
    cells = lambda items: ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in items)
    extra = f'<h2>模型对金标（验证集的整页，原来 → 两边都收紧后）</h2><table>{cells(scores)}</table>' if scores else ''
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>掩膜收紧到墨迹</title>
<style>body{{font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:820px;margin:40px auto;padding:0 16px;color:#1e1e1c;background:#f5f5f2}}
table{{border-collapse:collapse;width:100%;background:#fff;margin-bottom:8px}}td{{border:1px solid #dcdcd6;padding:6px 10px}}td:last-child{{text-align:right}}
a.big{{display:inline-block;margin:8px 0;padding:10px 18px;border-radius:8px;background:#c62828;color:#fff;text-decoration:none}}p{{color:#555}}</style></head><body>
<h1>掩膜收紧到墨迹：前后对比</h1><p>{note}</p><p><a class="big" href="compare.html">看每个掩膜的前后对比</a></p>
<table>{cells(rows)}</table>{extra}</body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--model', type=Path, nargs=2, metavar=('BEFORE', 'AFTER'), required=True)
    parser.add_argument('--gold', type=Path, nargs=2, metavar=('BEFORE', 'AFTER'))
    parser.add_argument('--images', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--ids', type=Path, help='a file of image ids, one per line (default: the pages of the model output)')
    parser.add_argument('--scores', type=Path, nargs=2, metavar=('BEFORE', 'AFTER'), help='two revrender.eval.pages_score files')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    sets = {'model': args.model, **({'gold': args.gold} if args.gold else {})}
    ids = listed_ids(args.model[0], args.ids)
    files = pages.page_files(args.images, ids)
    out = make_output_dir(args.output_dir)
    (out/'assets').mkdir()
    cards, rows = [], []
    for name, (before, after) in sets.items():
        shown = [i for i in ids if (before/f'{i}.json').is_file()]
        require(shown, f'no page of {before} is among the pages to show')
        part = cards_of(name, before, after, shown, files, out/'assets')
        cards += part
        rows += summary_rows(name, part)
    random.Random(SEED).shuffle(cards)
    scores = score_rows(*(json.loads(p.read_text()) for p in args.scores)) if args.scores else []
    (out/'compare.html').write_text(TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', embed({'cards': cards, 'labels': LABELS})), encoding='utf-8')
    note = ('收紧 = 原掩膜 ∩ 页面上的非白像素：墨迹（灰度 &lt; 200），加上从墨迹出发、只经过灰度 &lt; 240 的像素、最多走 2 步够得着的软边；'
            '不挨着笔画的浅灰孤点不算。JPEG 在笔画旁留下 220–254 的浅灰杂点，所以不能用“&lt; 255”，也不能只调低灰度线去杂点：那会先吃掉笔画的软边。'
            '收紧只会去掉像素，不会多出像素。')
    (out/'index.html').write_text(index_page([('页数', len(ids)), *rows], scores, note), encoding='utf-8')
    write_manifest(out, config=to_jsonable(args), inputs={str(d/'run_manifest.json'): digest(d/'run_manifest.json') for pair in sets.values() for d in pair},
                   pages=len(ids), cards=len(cards))
    print(json.dumps({'pages': len(ids), 'cards': len(cards), 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
