"""What is written under each strike region of the pixel review: the regions the annotator confirmed there, with a VLM's starting
text, in the pixel review page's text mode, where the annotator types what was struck and corrects pixels and boxes in one place.

    python -m revrender.tools.annotate.struck_text crops --export EXPORT.json --review REVIEW_PACK_DIR --pages DIR [--pages DIR ...]
        --split train [--split val ...] --output-dir OUT
    python -m revrender.tools.annotate.struck_text page --crops OUT --export EXPORT.json --readings predictions.jsonl
        --questions questions_with_answers.jsonl --answer-groups answer_groups.jsonl --output-dir PAGE
    python -m revrender.tools.annotate.struck_text erased --crops OUT --export EXPORT.json --review REVIEW_PACK_DIR --output-dir ERASED
    python -m revrender.tools.annotate.struck_text update --source PAGE.html [--readings predictions.jsonl --condition erased_neutral]
        [--judgments DRAFT.json] --output-dir NEW_PAGE

``crops``: every confirmed, not deleted region of the review export (``data.pixel_masks`` schema) on a page of the review pack
(``tools.annotate.pixel_library infer``) whose split is one of ``--split`` -- the pack's split already folds in paper 1's writers, the
same-student pairs and the frozen validation writers.  Per region: the scan cropped round its box (``CROP``) for ``recognize_lines
--conditions raw_struck`` (OUT/pack.json, a line pack, as ``page_prefill`` writes for the box annotation), and the review's cleaned page
cropped the same way twice, as it is and with the annotator's erased pixels white; per page the cleaned page.  OUT/regions.json lists
them in page order, regions top to bottom.

``page``: a pixel review pack (task ``strike_text``) of those regions -- the cleaned pages, each page's exam question with its derived
standard solution (paper 1's response index maps a page to its question), each region with the annotator's pixels, polygon and marks from
``--export`` and its starting text (``page_pack.starting_text`` of the reading) -- and its offline page (``pixel_offline.standalone``):
PAGE/pack/ and PAGE/site/标注.html.  The pixel editor's text mode (``pixel_text.js``, ``symbol_picker.js``) shows the struck text beside the
pixels and confirms both at once; on first opening it takes over what the earlier struck-text page kept in the browser
(``text_source_pack``: that page's id, from the same region ids).  Later editor changes reach an existing page with ``pixel_review
update-html`` (its storage key holds the image revision).

``erased``: the same regions as the annotator left them in ``--export``, each cut to its polygon (outside white, ``EDGE`` pixels of paper
round it) from the cleaned page with the annotator's erased pixels white: what is left is the writing under the strike.  ERASED/pack.json
is a line pack for ``recognize_lines --conditions erased_neutral`` (transcribe what is there; no word of striking).

``update``: an existing page (``--source``) with its pack id, image revision and embedded progress kept, so the browser's progress
stays, and the editor's current code; with ``--readings``, new starting texts (the editor replaces a region's text with the new one only
where the annotator has neither confirmed nor changed it: it still equals a starting text it had, kept as ``previous_prefill``); with
``--judgments``, drafted judgments of each revision (``paper2-revision-judgment-draft-v1``: was the struck content right, is what
replaced it right, the kind of revision, why), which the editor shows for the annotator to confirm or correct.  Crops, pages and the page are student work: never in Git.
"""
import argparse
import hashlib
import json
import re
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from ... import pages as pages_module
from ...data.pixel_masks import SCHEMA, decode
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from ...data.pixel_masks import region_pixels
from .page_pack import read_readings
from .pixel_offline import EmbeddedReview, render_html, standalone

EDGE = 4               # paper kept round a region's polygon in the erased crops
CROP = (30, 12)        # a crop is the region's box and this margin (x, y), as page_prefill cuts the box annotation's crops
UNKNOWN_RUN = re.compile(r'\?{4,}')   # a reader that loops on unreadable characters starts as three of them


def confirmed_regions(export, unconfirmed=False):
    """{page id: [revision]} of an export: the confirmed revisions that are not deleted (all that are not deleted with
    ``unconfirmed``: a model's prefill, ``pixel_library prefill-export``, to be annotated in one pass)."""
    require(export.get('schema') == SCHEMA, 'not a pixel review export')
    pages = {}
    for page, q in export['pages'].items():
        kept = [r for r in q.get('revisions', []) if (unconfirmed or r.get('confirmed')) and not r.get('deleted')]
        if kept:
            pages[page] = kept
    return pages


def crop_box(box, size):
    l, t, r, b = box
    return (max(0, l-CROP[0]), max(0, t-CROP[1]), min(size[0], r+CROP[0]), min(size[1], b+CROP[1]))


def erased(grey, revision):
    """The cleaned page with the annotator's erased pixels (``runs`` over the revision's box) white."""
    l, t, r, b = revision['box']
    out = grey.copy()
    out[t:b, l:r][decode(revision['runs'], (b-t, r-l))] = 255
    return out


def crops(args):
    export = json.loads(args.export.read_text())
    review = json.loads((args.review/'pack.json').read_text())
    split = {p['id']: p['split'] for p in review['pages']}
    regions = confirmed_regions(export, args.unconfirmed)
    unknown = sorted(set(regions)-set(split))
    require(not unknown, f'{len(unknown)} page(s) of the export are not in the review pack, e.g. {unknown[:2]}')
    chosen = sorted(p for p in regions if split[p] in args.split)
    require(chosen, 'no confirmed region on a page of the chosen splits')
    scans = pages_module.page_files(args.pages, chosen)
    out = make_output_dir(args.output_dir)
    for d in ('crops', 'pages'):
        (out/d).mkdir()
    rows, records = [], []
    for page in chosen:
        with Image.open(scans[page]) as opened:
            rgb = opened.convert('RGB')
        with Image.open(args.review/'images'/f'{page}.png') as opened:
            grey = np.asarray(opened.convert('L'))
        require(grey.shape == (rgb.height, rgb.width), f'{page}: the review page is not the size of the scan')
        Image.fromarray(grey).save(out/'pages'/f'{page}.png')
        for revision in sorted(regions[page], key=lambda r: (r['box'][1], r['box'][0])):
            box = crop_box(revision['box'], rgb.size)
            k = len(rows)
            names = {kind: f'crops/{k:05d}_{kind}.png' for kind in ('scan', 'clean', 'erased')}
            rgb.crop(box).save(out/names['scan'])
            Image.fromarray(grey).crop(box).save(out/names['clean'])
            Image.fromarray(erased(grey, revision)).crop(box).save(out/names['erased'])
            w, h = box[2]-box[0], box[3]-box[1]
            l, t, r, b = revision['box']
            rows.append({'id': revision['id'], 'line_id': revision['id'], 'image_id': page, 'kind': 'line', 'split': split[page], 'size': [w, h],
                         'images': {'raw': names['scan']},
                         'boxes': [[round(1000*(l-box[0])/w), round(1000*(t-box[1])/h), round(1000*(r-box[0])/w), round(1000*(b-box[1])/h)]]})
            records.append({'id': revision['id'], 'page': page, 'split': split[page], 'box': revision['box'], 'region': revision.get('region'),
                            'crop': list(box), 'mode': revision.get('annotationMode'), 'unreadable': bool(revision.get('unreadable')),
                            'images': {'clean': names['clean'], 'erased': names['erased'], 'page': f'pages/{page}.png'},
                            'page_size': [rgb.width, rgb.height]})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_json(out/'regions.json', records)
    summary = {'pages': len(chosen), 'regions': len(records), 'splits': {s: sum(r['split'] == s for r in records) for s in args.split},
               'left_out': sum(len(v) for p, v in regions.items() if split[p] not in args.split)}
    inputs = [args.export, args.review/'pack.json', *(args.review/'images'/f'{p}.png' for p in chosen), *(scans[p] for p in chosen)]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, **summary)
    print(json.dumps(summary, ensure_ascii=False))


def question_of(answer_groups):
    """{image id: question_global_id} from paper 1's response index."""
    found = {}
    for line in Path(answer_groups).read_text(encoding='utf-8').splitlines():
        group = json.loads(line)
        for image in group['image_ids']:
            found[image] = group['question_global_id']
    return found


def identity(records, kind):
    return f'paper2-{kind}-'+hashlib.sha256('|'.join(r['id'] for r in records).encode()).hexdigest()[:12]


def page(args):
    records = json.loads((args.crops/'regions.json').read_text())
    export = json.loads(args.export.read_text())
    require(export.get('schema') == SCHEMA, 'not a pixel review export')
    human = {r['id']: r for q in export['pages'].values() for r in q.get('revisions', [])}
    missing = [r['id'] for r in records if r['id'] not in human]
    require(not missing, f'{len(missing)} region(s) of the crops are not in the export, e.g. {missing[:2]}')
    texts = read_readings(args.readings, args.condition)
    questions = {q['question_global_id']: q for q in map(json.loads, args.questions.read_text(encoding='utf-8').splitlines())}
    of = question_of(args.answer_groups)
    lost = sorted({r['page'] for r in records if of.get(r['page']) not in questions})
    require(not lost, f'{len(lost)} page(s) without an exam question, e.g. {lost[:2]}')
    out = make_output_dir(args.output_dir)
    (out/'pack'/'images').mkdir(parents=True)
    pages, index, rows, initial = [], {}, [], {}
    for r in records:
        if r['page'] not in index:
            index[r['page']] = len(pages)
            source = args.crops/r['images']['page']
            with Image.open(source) as opened:
                require(list(opened.size) == r['page_size'], f'{r["page"]}: the cleaned page is not at the scan\'s size')
            shutil.copyfile(source, out/'pack'/'images'/f'{r["page"]}.png')
            pages.append({'id': r['page'], 'question': r['page'].split('_')[-2], 'image': f'images/{r["page"]}.png', 'size': r['page_size'],
                          'split': r['split'], 'exam_question': of[r['page']]})
        prefill = UNKNOWN_RUN.sub('???', texts.get(r['id'], ''))
        h = human[r['id']]
        rows.append({'id': r['id'], 'page': index[r['page']], 'box': h['box'], 'region': h['region'], 'runs': h['runs'], 'text': prefill,
                     'prefill': prefill})
        page_record = initial.setdefault(r['page'], {'revisions': [], 'done': False})
        page_record['revisions'].append({**h, 'text': h.get('text') or prefill})
    used = sorted({p['exam_question'] for p in pages})
    exam = {q: {k: questions[q][k] for k in ('question_global_id', 'printed_problem_no', 'source_exam_title', 'problem_statement_markdown',
                                             'standard_solution_markdown', 'final_answer')} for q in used}
    pack_id = identity(records, 'struck-pixels')
    write_json(out/'pack'/'pack.json', {'id': pack_id, 'schema': 'paper2-pixel-review-pack-v1', 'task': 'strike_text', 'pages': pages,
                                         'rows': rows, 'questions': exam, 'text_source_pack': identity(records, 'struck-text')})
    write_json(out/'initial.json', {'schema': SCHEMA, 'pack': pack_id, 'task': 'strike_text', 'pages': initial})
    standalone(Namespace(pack=out/'pack', review_export=out/'initial.json', output_dir=out/'site'))
    summary = {'id': pack_id, 'regions': len(rows), 'pages': len(pages), 'questions': len(exam), 'with_prefill': sum(bool(r['prefill']) for r in rows)}
    inputs = [args.crops/'regions.json', args.crops/'run_manifest.json', args.export, args.readings, args.questions, args.answer_groups]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, **summary)
    print(json.dumps(summary, ensure_ascii=False))


def erased_crops(args):
    records = json.loads((args.crops/'regions.json').read_text())
    export = json.loads(args.export.read_text())
    require(export.get('schema') == SCHEMA, 'not a pixel review export')
    human = {r['id']: r for q in export['pages'].values() for r in q.get('revisions', [])}
    out = make_output_dir(args.output_dir)
    (out/'crops').mkdir()
    rows, pages = [], {}
    for k, record in enumerate(records):
        r = human[record['id']]
        if record['page'] not in pages:
            with Image.open(args.review/'images'/f'{record["page"]}.png') as opened:
                pages[record['page']] = np.asarray(opened.convert('L'))
        grey = pages[record['page']].copy()
        l, t, rr, b = r['box']
        grey[t:b, l:rr][decode(r['runs'], (b-t, rr-l))] = 255
        xs, ys = [p[0] for p in r['region']], [p[1] for p in r['region']]
        box = [max(0, int(min(xs))-EDGE), max(0, int(min(ys))-EDGE), min(grey.shape[1], int(max(xs))+EDGE), min(grey.shape[0], int(max(ys))+EDGE)]
        crop = grey[box[1]:box[3], box[0]:box[2]].copy()
        crop[~region_pixels(r['region'], box)] = 255
        name = f'crops/{k:05d}.png'
        Image.fromarray(crop).save(out/name)
        rows.append({'id': record['id'], 'line_id': record['id'], 'image_id': record['page'], 'kind': 'line', 'split': record['split'],
                     'size': [crop.shape[1], crop.shape[0]], 'images': {'erased': name}, 'boxes': [[0, 0, 1000, 1000]]})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    inputs = [args.crops/'regions.json', args.export, *(args.review/'images'/f'{p}.png' for p in pages)]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, regions=len(rows), pages=len(pages))
    print(json.dumps({'regions': len(rows), 'pages': len(pages)}))


JUDGMENT_SCHEMA = 'paper2-revision-judgment-draft-v1'


def update(args):
    data = EmbeddedReview(args.source.read_text()).values
    require('review-pack' in data, 'not an offline review page')
    payload, draft = json.loads(data['review-pack']), json.loads(data.get('review-draft') or 'null')
    require(payload.get('task') == 'strike_text', 'not a struck-text page')
    texts = read_readings(args.readings, args.condition) if args.readings else {}
    if args.judgments:
        value = json.loads(args.judgments.read_text(encoding='utf-8'))
        require(value.get('schema') == JUDGMENT_SCHEMA, 'not a judgment draft')
        payload['judgment_drafts'] = value['judgments']
        payload['judgment_values'] = value['values']
    changed = 0
    for row in payload['rows']:
        if row['id'] in texts:
            new = UNKNOWN_RUN.sub('???', texts[row['id']])
            if new != row.get('prefill'):
                # Every starting text a region has had: the editor replaces only a text still equal to one of them.
                row['previous_prefill'] = [*row.get('previous_prefill', []), row.get('prefill', '')]
                row['prefill'] = new
                changed += 1
    out = make_output_dir(args.output_dir)
    (out/'标注.html').write_text(render_html(payload, draft))
    inputs = [p for p in (args.source, args.readings, args.judgments) if p]
    summary = {'regions': len(payload['rows']), 'new_prefill': changed, 'judgments': len(payload.get('judgment_drafts', {}))}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, pack=payload['id'],
                   image_revision=payload.get('image_revision'), embedded_progress_preserved=True, **summary)
    print(json.dumps({'pack': payload['id'], **summary}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('crops')
    c.add_argument('--export', type=Path, required=True, help='the pixel review export')
    c.add_argument('--review', type=Path, required=True, help='the review pack directory (pack.json, images/)')
    c.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    c.add_argument('--split', action='append', required=True, choices=('train', 'val', 'test'))
    c.add_argument('--unconfirmed', action='store_true', help="also the regions nobody confirmed (a model's prefill)")
    c.add_argument('--output-dir', type=Path, required=True)
    p = sub.add_parser('page')
    p.add_argument('--crops', type=Path, required=True, help='the output of crops')
    p.add_argument('--export', type=Path, required=True, help='the pixel review export the crops were cut from')
    p.add_argument('--readings', type=Path, required=True, help='recognize_lines predictions on the crops (raw_struck)')
    p.add_argument('--condition', default='raw_struck', help='the condition of the readings (erased_neutral for the erased crops)')
    p.add_argument('--questions', type=Path, required=True, help='questions_with_answers.jsonl (paper 1)')
    p.add_argument('--answer-groups', type=Path, required=True, help="answer_groups.jsonl of paper 1's response index")
    p.add_argument('--output-dir', type=Path, required=True)
    e = sub.add_parser('erased')
    e.add_argument('--crops', type=Path, required=True, help='the output of crops (its regions)')
    e.add_argument('--export', type=Path, required=True, help='the pixel review export with the regions as the annotator left them')
    e.add_argument('--review', type=Path, required=True, help='the review pack directory (images/)')
    e.add_argument('--output-dir', type=Path, required=True)
    d = sub.add_parser('update')
    d.add_argument('--source', type=Path, required=True, help='the page now in use')
    d.add_argument('--readings', type=Path)
    d.add_argument('--judgments', type=Path, help='a judgment draft of the revisions')
    d.add_argument('--condition', default='erased_neutral')
    d.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    {'crops': crops, 'page': page, 'erased': erased_crops, 'update': update}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
