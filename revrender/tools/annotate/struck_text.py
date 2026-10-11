"""What is written under each strike region of the pixel review: the regions the annotator confirmed there, packed with a VLM's
starting text into one offline page where the annotator types what was struck.

    python -m revrender.tools.annotate.struck_text crops --export EXPORT.json --review REVIEW_PACK_DIR --pages DIR [--pages DIR ...]
        --split train [--split val ...] --output-dir OUT
    python -m revrender.tools.annotate.struck_text page --crops OUT --readings predictions.jsonl --questions questions_with_answers.jsonl
        --answer-groups answer_groups.jsonl --output-dir PAGE

``crops``: every confirmed, not deleted region of the review export (``data.pixel_masks`` schema) on a page of the review pack
(``tools.annotate.pixel_library infer``) whose split is one of ``--split`` -- the pack's split already folds in paper 1's writers, the
same-student pairs and the frozen validation writers.  Per region: the scan cropped round its box (``CROP``) for ``recognize_lines
--conditions raw_struck`` (OUT/pack.json, a line pack, as ``page_prefill`` writes for the box annotation), and the review's cleaned page
cropped the same way twice, as it is and with the annotator's erased pixels white; per page the cleaned page, at most ``PAGE_WIDTH``
wide.  OUT/regions.json lists them in page order, regions top to bottom.

``page``: the regions, their crops and pages inlined with their starting texts (``page_pack.starting_text`` of the reading), and each
page's exam question with its derived standard solution (paper 1's response index maps a page to its question), into PAGE/补文字.html
(``struck_text.html``; text rules from ``page_rules.js``).  The pack id depends only on the region ids, so a rebuilt page keeps what the
browser already saved.  Crops and pages are student work: OUT and PAGE stay under work/ or the server, never in Git.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ... import pages as pages_module
from ...data.pixel_masks import SCHEMA, decode
from ...util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .page_pack import RULES, read_readings, slot

HERE = Path(__file__).parent
EDITOR = HERE/'struck_text.html'
CROP = (30, 12)        # a crop is the region's box and this margin (x, y), as page_prefill cuts the box annotation's crops
PAGE_WIDTH = 1400      # pages wider than this are shown scaled down (the crops keep the scan's pixels)
UNKNOWN_RUN = re.compile(r'\?{4,}')   # a reader that loops on unreadable characters starts as three of them


def confirmed_regions(export):
    """{page id: [revision]} of an export: the confirmed revisions that are not deleted."""
    require(export.get('schema') == SCHEMA, 'not a pixel review export')
    pages = {}
    for page, q in export['pages'].items():
        kept = [r for r in q.get('revisions', []) if r.get('confirmed') and not r.get('deleted')]
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
    regions = confirmed_regions(export)
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
        shown = Image.fromarray(grey)
        if shown.width > PAGE_WIDTH:
            shown = shown.resize((PAGE_WIDTH, round(grey.shape[0]*PAGE_WIDTH/grey.shape[1])), Image.LANCZOS)
        shown.save(out/'pages'/f'{page}.png')
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


def data_url(path):
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def question_of(answer_groups):
    """{image id: question_global_id} from paper 1's response index."""
    found = {}
    for line in Path(answer_groups).read_text(encoding='utf-8').splitlines():
        group = json.loads(line)
        for image in group['image_ids']:
            found[image] = group['question_global_id']
    return found


def page(args):
    records = json.loads((args.crops/'regions.json').read_text())
    texts = read_readings(args.readings)
    questions = {q['question_global_id']: q for q in map(json.loads, args.questions.read_text(encoding='utf-8').splitlines())}
    of = question_of(args.answer_groups)
    missing = sorted({r['page'] for r in records if of.get(r['page']) not in questions})
    require(not missing, f'{len(missing)} page(s) without an exam question, e.g. {missing[:2]}')
    pages, used = {}, set()
    for r in records:
        if r['page'] not in pages:
            pages[r['page']] = {'image': data_url(args.crops/r['images']['page']), 'size': r['page_size'], 'question': of[r['page']]}
            used.add(of[r['page']])
    regions = [{**{k: r[k] for k in ('id', 'page', 'split', 'box', 'region', 'crop', 'mode')},
                'prefill': UNKNOWN_RUN.sub('???', texts.get(r['id'], '')), 'unreadable': r['unreadable'],
                'clean': data_url(args.crops/r['images']['clean']), 'erased': data_url(args.crops/r['images']['erased'])} for r in records]
    exam = {q: {k: questions[q][k] for k in ('question_global_id', 'printed_problem_no', 'source_exam_title', 'problem_statement_markdown',
                                             'standard_solution_markdown', 'final_answer')} for q in sorted(used)}
    identity = 'paper2-struck-text-'+hashlib.sha256('|'.join(r['id'] for r in records).encode()).hexdigest()[:12]
    pack = {'id': identity, 'schema': 'paper2-struck-text-pack-v1', 'regions': regions, 'pages': pages, 'questions': exam}
    html = slot(slot(EDITOR.read_text(encoding='utf-8'), 'RULES', RULES.read_text(encoding='utf-8')), 'PACK',
                'window.PACK = '+json.dumps(pack, ensure_ascii=False)+';')
    out = make_output_dir(args.output_dir)
    (out/'补文字.html').write_text(html, encoding='utf-8')
    summary = {'id': identity, 'regions': len(regions), 'pages': len(pages), 'questions': len(exam),
               'with_prefill': sum(bool(r['prefill']) for r in regions), 'bytes': (out/'补文字.html').stat().st_size}
    inputs = [args.crops/'regions.json', args.crops/'run_manifest.json', args.readings, args.questions, args.answer_groups, EDITOR, RULES]
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for p in inputs}, **summary)
    print(json.dumps(summary, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('crops')
    c.add_argument('--export', type=Path, required=True, help='the pixel review export')
    c.add_argument('--review', type=Path, required=True, help='the review pack directory (pack.json, images/)')
    c.add_argument('--pages', type=Path, action='append', required=True, help='a folder of scans <image_id>.jpg (repeatable)')
    c.add_argument('--split', action='append', required=True, choices=('train', 'val', 'test'))
    c.add_argument('--output-dir', type=Path, required=True)
    p = sub.add_parser('page')
    p.add_argument('--crops', type=Path, required=True, help='the output of crops')
    p.add_argument('--readings', type=Path, required=True, help='recognize_lines predictions on the crops (raw_struck)')
    p.add_argument('--questions', type=Path, required=True, help='questions_with_answers.jsonl (paper 1)')
    p.add_argument('--answer-groups', type=Path, required=True, help="answer_groups.jsonl of paper 1's response index")
    p.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    crops(args) if args.command == 'crops' else page(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
