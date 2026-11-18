"""What is written under each strike region of the pixel review: the regions the annotator confirmed there, with a VLM's starting
text, in the pixel review page's text mode, where the annotator types what was struck and corrects pixels and boxes in one place.

    python -m revrender.tools.annotate.struck_text crops --export EXPORT.json --review REVIEW_PACK_DIR --pages DIR [--pages DIR ...]
        --split train [--split val ...] --output-dir OUT
    python -m revrender.tools.annotate.struck_text page --crops OUT --export EXPORT.json --readings predictions.jsonl
        --questions questions_with_answers.jsonl --answer-groups answer_groups.jsonl --output-dir PAGE
    python -m revrender.tools.annotate.struck_text erased --crops OUT --export EXPORT.json --review REVIEW_PACK_DIR --output-dir ERASED
    python -m revrender.tools.annotate.struck_text update --source PAGE.html [--readings predictions.jsonl --condition erased_neutral]
        [--judgments DRAFT.json] [--runs PREFILL] --output-dir NEW_PAGE
    python -m revrender.tools.annotate.struck_text prefill --source PAGE.html --export EXPORT.json --restorer RUN --restorer-checkpoint STEP
        --upstream UP --output-dir PREFILL
    python -m revrender.tools.annotate.struck_text sheets --source PAGE.html --export EXPORT.json --output-dir SHEETS

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
``--runs`` (``prefill``), new pixels for the regions it lists and the prefill's ``model_revision``: the editor gives them to a region
only where the annotator has neither confirmed nor changed it (no edit, not deleted), its text and marks kept, the draft before it kept
in the browser; with ``--drafts`` (``sheets``' folder with a drafts.json beside its index: code -> text, unreadable, sure,
pixels_wrong), drafted texts for those regions, each with a note against the page's starting text (``draft_note``): the editor writes
one once into a region nobody confirmed, edited or deleted whose text is still a starting text it had, the note only where it has none;
with ``--judgments``, drafted judgments of each revision (``paper2-revision-judgment-draft-v1``: was the struck content right, is what
replaced it right, the kind of revision, why), which the editor shows for the annotator to confirm or correct.  Crops, pages and the page are student work: never in Git.

``prefill``: a restorer's pixels (``pixel_library.predictor_of``: the ink it brightens by more than ``REMOVED`` inside the region's polygon,
as the library prefills a hinted region) for every region of ``--export`` nobody touched (``untouched``), on the page's own embedded images, in the region's box and polygon as the export has them: PREFILL/runs.json.

``sheets``: contact sheets for drafting the struck text of the regions nobody touched (``untouched``): per region its window (the
polygon's box and ``EDGE`` pixels, white outside) as scanned and with the page's prefill white, ``SHEET_SIZE`` regions a
sheet, each under a short code; SHEETS/index.json maps a code to its region, its sheet and the starting text.  ``REFERENCES`` regions
whose text was confirmed (seed 42) make reference sheets, their texts in SHEETS/reference.json.  Student work: never in Git.
"""
import argparse
from collections import Counter
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
SHEET_SIZE = 10        # regions a contact sheet
REFERENCES = 30        # confirmed regions shown with their text


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
    # with --unconfirmed the whole question is annotated: a page whose regions were all deleted stays, so a missed one can be added
    empty = sorted(p for p in export['pages'] if args.unconfirmed and p in split and split[p] in args.split and p not in regions)
    require(chosen, 'no confirmed region on a page of the chosen splits')
    scans = pages_module.page_files(args.pages, chosen+empty)
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
    blank = []
    for page in empty:
        with Image.open(args.review/'images'/f'{page}.png') as opened:
            opened.convert('L').save(out/'pages'/f'{page}.png')
            blank.append({'page': page, 'split': split[page], 'page_size': list(opened.size), 'image': f'pages/{page}.png'})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_json(out/'regions.json', records)
    write_json(out/'empty_pages.json', blank)
    summary = {'pages': len(chosen), 'empty_pages': len(empty), 'regions': len(records), 'splits': {s: sum(r['split'] == s for r in records) for s in args.split},
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
    empty = json.loads((args.crops/'empty_pages.json').read_text()) if (args.crops/'empty_pages.json').exists() else []
    lost = sorted({e['page'] for e in empty if of.get(e['page']) not in questions})
    require(not lost, f'{len(lost)} page(s) without an exam question, e.g. {lost[:2]}')
    for r in sorted(records+[{**e, 'empty': True} for e in empty], key=lambda r: r['page']):
        if r.get('empty'):
            index[r['page']] = len(pages)
            shutil.copyfile(args.crops/r['image'], out/'pack'/'images'/f'{r["page"]}.png')
            pages.append({'id': r['page'], 'question': r['page'].split('_')[-2], 'image': f'images/{r["page"]}.png', 'size': r['page_size'],
                          'split': r['split'], 'exam_question': of[r['page']]})
            initial[r['page']] = {'revisions': [], 'done': False}
            continue
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
        page_record['revisions'].append({**h, 'text': h['text'] if 'text' in h else prefill})   # the annotator's text, even empty
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


def draft_note(draft, before):
    """The note a drafted text carries (plan of 2026-11-18): against the page's earlier starting text in the spelling of
    ``eval.coverage.prepare``, 两稿一致 or 两稿不同, or 初稿：读不出; ·没把握 when the drafter was not sure, ·像素要看 when it saw the pixels wrong."""
    from ...eval.coverage import prepare
    head = '初稿：读不出' if draft['unreadable'] else '两稿一致' if prepare(draft['text']) == prepare(before or '') else '两稿不同'
    return head+('' if draft['sure'] else '·没把握')+('·像素要看' if draft['pixels_wrong'] else '')


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
    if args.runs:   # new pixels for the regions nobody touched; the editor applies them where the browser's draft agrees
        value = json.loads((args.runs/'runs.json').read_text())
        require(value['pack'] == payload['id'], 'the prefill is of another page')
        for row in payload['rows']:
            if row['id'] in value['runs']: row['runs'] = value['runs'][row['id']]
        payload['model_revision'] = value['model_revision']; payload['prefill_model'] = value['model']
    notes = Counter()
    if args.drafts:   # drafted texts of the contact sheets' regions, each with its note
        index = json.loads((args.drafts/'index.json').read_text()); drafts = json.loads((args.drafts/'drafts.json').read_text())
        require(set(index) == set(drafts), 'the drafts are not of the sheets')
        by_id = {row['id']: row for row in payload['rows']}
        for code, draft in drafts.items():
            row = by_id[index[code]['id']]; require((row.get('prefill') or '') == index[code]['draft_32b'], f'{code}: the page is not the sheets\' page')
            require(not (draft['unreadable'] and draft['text']), f'{code}: unreadable with a text')
            note = draft_note(draft, row.get('prefill')); notes[note.split('·')[0]] += 1
            row['previous_prefill'] = [*row.get('previous_prefill', []), row.get('prefill') or '']
            row.update(prefill='' if draft['unreadable'] else draft['text'], unreadable_prefill=bool(draft['unreadable']), prefill_note=note)
        payload['text_revision'] = digest(args.drafts/'drafts.json')
    for row in payload['rows']:
        if row['id'] in texts:
            new = UNKNOWN_RUN.sub('???', texts[row['id']])
            if new != row.get('prefill'):
                # Every starting text a region has had: the editor replaces only a text still equal to one of them.
                row['previous_prefill'] = [*row.get('previous_prefill', []), row.get('prefill', '')]
                row['prefill'] = new
                changed += 1
    out = make_output_dir(args.output_dir)
    (out/'标注.html').write_text(render_html(payload, draft, args.katex))
    inputs = [p for p in (args.source, args.readings, args.judgments, args.runs and args.runs/'runs.json', args.drafts and args.drafts/'drafts.json') if p]
    if args.katex: inputs += [args.katex/'katex.min.js', args.katex/'katex.min.css']
    summary = {'regions': len(payload['rows']), 'new_prefill': changed, 'judgments': len(payload.get('judgment_drafts', {})),
               'new_pixels': len(value['runs']) if args.runs else 0, 'drafted': sum(notes.values()), 'notes': dict(notes)}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, pack=payload['id'],
                   image_revision=payload.get('image_revision'), embedded_progress_preserved=True, **summary)
    print(json.dumps({'pack': payload['id'], **summary}, ensure_ascii=False))


def page_images(payload):
    """{page index: the page's embedded image as grey} of an offline page's pack."""
    import base64, io
    out = {}
    for k, p in enumerate(payload['pages']):
        require(p['image'].startswith('data:image/png;base64,'), 'not an offline page with its images')
        with Image.open(io.BytesIO(base64.b64decode(p['image'].split(',', 1)[1]))) as opened: out[k] = np.asarray(opened.convert('L'))
    return out


def human_of(export, payload):
    export = json.loads(export.read_text()) if isinstance(export, Path) else export
    require(export.get('schema') == SCHEMA and export.get('pack') == payload['id'] and export.get('image_revision') == payload.get('image_revision'),
            'the export is not of this page')
    return {r['id']: r for q in export['pages'].values() for r in q.get('revisions', [])}


def untouched(h, row):
    """Whether nobody touched a region: not confirmed (pixels or text), no edit, not deleted, nothing marked unreadable, no note, its
    text still the page's starting text."""
    return (not h.get('confirmed') and not h.get('text_confirmed') and not h.get('ops') and not h.get('deleted') and not h.get('unreadable')
            and not h.get('note') and (h.get('text') or '') == (row.get('prefill') or ''))


def prefill(args):
    from argparse import Namespace as Args
    from ...data.pixel_masks import encode
    from .pixel_library import predictor_of
    payload = json.loads(EmbeddedReview(args.source.read_text()).values['review-pack']); human = human_of(args.export, payload)
    rows = [r for r in payload['rows'] if r['id'] in human and untouched(human[r['id']], r)]
    masks_of, selected, checkpoint = predictor_of(Args(restorer=args.restorer, restorer_checkpoint=args.restorer_checkpoint, upstream=args.upstream))
    images, runs = page_images(payload), {}
    for k in sorted({r['page'] for r in rows}):
        chosen = [human[r['id']] for r in rows if r['page'] == k]; grey = images[k]
        crops = [np.where(region_pixels(h['region'], h['box']), grey[h['box'][1]:h['box'][3], h['box'][0]:h['box'][2]], 255).astype(np.uint8) for h in chosen]
        for h, mask in zip(chosen, masks_of(crops)): runs[h['id']] = encode(mask & region_pixels(h['region'], h['box']))
    out = make_output_dir(args.output_dir)
    value = {'pack': payload['id'], 'model': selected['selected'], 'model_revision': digest(checkpoint), 'runs': runs}
    write_json(out/'runs.json', value)
    summary = {'regions': len(payload['rows']), 'in_export': len(human), 'new_pixels': len(runs), 'kept': len(human)-len(runs),
               'confirmed': sum(bool(h.get('confirmed')) for h in human.values()), 'changed_unconfirmed': sum(not h.get('confirmed') and bool(h.get('ops')) for h in human.values()),
               'zero_masks': sum(not v for v in runs.values()), 'model': selected['selected']}
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in (args.source, args.export, checkpoint)}, **summary)
    print(json.dumps(summary, ensure_ascii=False))


def window_pair(grey, h, runs):
    """(scan, erased) of a region's window: its polygon's box and ``EDGE`` pixels, white outside the polygon; ``runs`` white in erased."""
    xs, ys = [p[0] for p in h['region']], [p[1] for p in h['region']]
    box = [max(0, int(min(xs))-EDGE), max(0, int(min(ys))-EDGE), min(grey.shape[1], int(max(xs))+EDGE), min(grey.shape[0], int(max(ys))+EDGE)]
    gone = erased(grey, {'box': h['box'], 'runs': runs})
    outside = ~region_pixels(h['region'], box)
    return [np.where(outside, 255, a[box[1]:box[3], box[0]:box[2]]).astype(np.uint8) for a in (grey, gone)]


def sheet(cells, path):
    """A contact sheet: per cell its code, then its scan and erased windows side by side, each scaled to a readable height."""
    from PIL import ImageDraw
    height = lambda a: min(160, max(96, a.shape[0]))
    rendered = []
    for code, scan, gone in cells:
        k = height(scan)/scan.shape[0]; w = max(1, round(scan.shape[1]*k)); w = min(w, 700); k2 = w/scan.shape[1]; hh = max(1, round(scan.shape[0]*k2))
        pair = [Image.fromarray(a).resize((w, hh), Image.Resampling.LANCZOS) for a in (scan, gone)]
        rendered.append((code, pair, hh, w))
    width = max(2*w+60 for _, _, _, w in rendered); total = sum(hh+40 for _, _, hh, _ in rendered)
    canvas = Image.new('L', (width, total), 255); draw = ImageDraw.Draw(canvas); y = 0
    for code, pair, hh, w in rendered:
        draw.text((6, y+4), code, fill=0); y += 22
        canvas.paste(pair[0], (10, y)); canvas.paste(pair[1], (w+40, y))
        draw.rectangle((9, y-1, w+10, y+hh), outline=160); draw.rectangle((w+39, y-1, 2*w+40, y+hh), outline=160); y += hh+18
    canvas.save(path)


def sheets(args):
    import random
    payload = json.loads(EmbeddedReview(args.source.read_text()).values['review-pack']); human = human_of(args.export, payload)
    rows = sorted((r for r in payload['rows'] if r['id'] in human and not human[r['id']].get('deleted')), key=lambda r: (r['page'], r['box'][1], r['box'][0]))
    runs_of = lambda r: human[r['id']]['runs'] if not untouched(human[r['id']], r) else r['runs']
    draft = [r for r in rows if untouched(human[r['id']], r)]
    changed = [r['id'] for r in rows if not human[r['id']].get('text_confirmed') and r not in draft]
    reference = random.Random(42).sample([r for r in rows if human[r['id']].get('text_confirmed')], min(REFERENCES, sum(bool(human[r['id']].get('text_confirmed')) for r in rows)))
    images = page_images(payload); out = make_output_dir(args.output_dir); index, refs = {}, {}
    for kind, chosen, letter in (('draft', draft, 'T'), ('reference', reference, 'R')):
        for n in range(0, len(chosen), SHEET_SIZE):
            name = f'{kind}_{n//SHEET_SIZE+1:02d}.png'; cells = []
            for k, r in enumerate(chosen[n:n+SHEET_SIZE], n+1):
                code = f'{letter}{k:03d}'; h = human[r['id']]; cells.append((code, *window_pair(images[r['page']], h, runs_of(r))))
                if kind == 'draft': index[code] = {'id': r['id'], 'sheet': name, 'draft_32b': r.get('prefill') or '', 'pixels': 'prefill'}
                else: refs[code] = {'id': r['id'], 'sheet': name, 'text': h.get('text') or '', 'unreadable': bool(h.get('unreadable'))}
            sheet(cells, out/name)
    write_json(out/'index.json', index); write_json(out/'reference.json', refs)
    summary = {'drafted': len(index), 'draft_sheets': -(-len(index)//SHEET_SIZE), 'reference': len(refs), 'reference_sheets': -(-len(refs)//SHEET_SIZE),
               'text_changed_unconfirmed_left_out': len(changed), 'model': payload.get('prefill_model')}
    write_json(out/'summary.json', {**summary, 'left_out_ids': changed})
    print(json.dumps(summary, ensure_ascii=False))


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
    d.add_argument('--katex', type=Path, help='a KaTeX dist folder to inline (formulas render offline)')
    d.add_argument('--runs', type=Path, help="a prefill's output (prefill): new pixels for the regions nobody touched")
    d.add_argument('--drafts', type=Path, help="sheets' folder with drafts.json: drafted texts for its regions, with their notes")
    d.add_argument('--output-dir', type=Path, required=True)
    f = sub.add_parser('prefill')
    for key in ('source', 'export', 'restorer', 'upstream', 'output-dir'): f.add_argument('--'+key, type=Path, required=True)
    f.add_argument('--restorer-checkpoint', required=True)
    g = sub.add_parser('sheets')
    for key in ('source', 'export', 'output-dir'): g.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(argv)
    {'crops': crops, 'page': page, 'erased': erased_crops, 'update': update, 'prefill': prefill, 'sheets': sheets}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
