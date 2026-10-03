"""Whole-question images end to end: cut the page, read every piece with RevRender, paste the instances back.

    python -m revrender.pages --checkpoint best.pt --freeze freeze.json --pages DIR [--pages DIR ...] --output-dir OUT
        (--ocr-model MODEL_DIR | --ocr-text ocr.jsonl [--ocr-text ...]) [--templates templates.npz] [--frames frames.json] [--ids FILE] [--limit N] [--overlay]

A page is cut into pieces and each piece is read the way paper 1 read a line: cropped, letterboxed to the 96x768 strip, read by RevRender
at the checkpoint's frozen point (mask threshold, NMS, score cut), every delivered instance resized back and pasted into the page.  The
pieces (``page_lines``): the OCR model's own lines (``ocr``: the catalogue's lines, which paper 1 trained on); the lines of a region the
OCR returned as one block of several lines, cut where the ink has gaps (``split``); and the handwriting no line covers -- a line the OCR
skipped, a stroke that ran past the end of a line, a note in the margin (``ink``).  Where the handwriting is comes from the page against
the printed template of its question (``--templates``, made by ``python -m revrender.data.handwriting``).  On pages cleaned by
``python -m revrender.data.clean_pages`` (``--pages`` the cleaned folder, ``--frames`` its frames.json) a piece whose centre lies outside the
page's frame is not the question's and is dropped.  Per page OUT/<image_id>.json
lists the pieces and the instances (piece, score, box in pixels) and OUT/<image_id>.npz holds each instance's mask on its box (packed bits,
``mask_<n>``); ``--overlay`` also writes <image_id>.png.  ``--ocr-text`` takes OCR text the model already produced (jsonl of {image_id,
text}) instead of running it.  Only the pages named by ``--ids`` (default: every jpg in ``--pages``) are read.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import infer
from .data import handwriting
from .data import lines as lines_module
from .data import ocr_lines
from .data.gold import CANVAS, letterbox, unletterbox
from .eval.freeze import frozen_point
from .util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

STRIP_H, STRIP_W = CANVAS


def clamped(box):
    """A thousandths box inside the page, or None when nothing of it is left."""
    x0, y0, x1, y1 = (min(max(float(v), 0.), 1000.) for v in box)
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


def pixel_box(box, size):
    """A thousandths box as pixels (left, top, right, bottom) on a page of this size, or None."""
    box = clamped(box)
    return list(lines_module.crop_box(box, *size)) if box else None


def page_lines(text, size, ink):
    """The pieces a page is cut into: [{box (pixels), source, ocr_type, ocr_text}] -- see the module's docstring.  ``ink`` is the page's
    handwriting (``data.handwriting.handwriting``)."""
    out = []
    for line in ocr_lines.strict_lines(text):
        box = pixel_box(line['box'], size)
        if box:
            out.append({'box': box, 'source': 'ocr', 'ocr_type': line['ocr_type'], 'ocr_text': line['ocr_text']})
    groups = {}
    for line in ocr_lines.reorder(ocr_lines.parse_regions(text)):
        if line['inferred'] and line['student'] and line['box']:
            groups.setdefault(tuple(line['region']), []).append(line)
    for region, parts in groups.items():
        box = pixel_box(region, size)
        if box:
            for part, cut in zip(parts, handwriting.split_by_ink(box, parts[0]['parts'], ink)):
                if cut:
                    out.append({'box': cut, 'source': 'split', 'ocr_type': part['ocr_type'], 'ocr_text': part['ocr_text']})
    for box in handwriting.leftover_boxes(ink, [line['box'] for line in out]):
        out.append({'box': box, 'source': 'ink', 'ocr_type': None, 'ocr_text': ''})
    return out


def inside(frame, box):
    """Whether the centre of a piece ``box`` (left, top, right, bottom) lies in ``frame`` (left, top, right, bottom)."""
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
    return frame[0] <= x <= frame[2] and frame[1] <= y <= frame[3]


def read_page(model, point, image, lines, batch_size=infer.BATCH_SIZE):
    """The instances RevRender delivers on a grey page cut into ``lines`` (``page_lines``):
    [{line, score, box (left, top, right, bottom in pixels), mask (on the box)}], overlapping duplicates dropped."""
    strips = [np.asarray(letterbox(image.crop(tuple(line['box'])), STRIP_W, STRIP_H, 255)) for line in lines]
    instances = []
    for k, (line, delivered) in enumerate(zip(lines, infer.predict_lines(model, strips, point, batch_size))):
        box = line['box']
        for mask, score in delivered:
            crop = unletterbox(mask, box[2]-box[0], box[3]-box[1])
            if crop.any():
                instances.append({'line': k, 'score': float(score), 'box': list(box), 'mask': crop})
    return merge_duplicates(instances)


def overlap(a, b):
    """IoU of two instances on the page, computed where their boxes meet."""
    left, top = max(a['box'][0], b['box'][0]), max(a['box'][1], b['box'][1])
    right, bottom = min(a['box'][2], b['box'][2]), min(a['box'][3], b['box'][3])
    if right <= left or bottom <= top:
        return 0.
    part = lambda i: i['mask'][top-i['box'][1]:bottom-i['box'][1], left-i['box'][0]:right-i['box'][0]]
    inter = float((part(a) & part(b)).sum())
    return inter/(float(a['mask'].sum())+float(b['mask'].sum())-inter)


def merge_duplicates(instances, level=.5):
    """Instances in descending score; one that overlaps a kept instance by more than ``level`` (IoU on the page) is the same mark
    seen from a neighbouring piece and is dropped."""
    kept = []
    for instance in sorted(instances, key=lambda i: -i['score']):
        if all(overlap(instance, k) <= level for k in kept):
            kept.append(instance)
    return sorted(kept, key=lambda i: (i['line'], -i['score']))


def paste(instance, size):
    """One instance in the page frame (a boolean mask of the page's size)."""
    page = np.zeros((size[1], size[0]), bool)
    left, top, right, bottom = instance['box']
    page[top:bottom, left:right] = instance['mask']
    return page


def overlay(image, instances, alpha=.5):
    """The page with its instances tinted red."""
    rgb = np.stack([np.asarray(image)]*3, -1).astype(np.float32)
    if instances:
        union = np.logical_or.reduce([paste(i, image.size) for i in instances])
        rgb[union] = (1-alpha)*rgb[union] + alpha*np.array((220, 0, 0), np.float32)
    return Image.fromarray(rgb.astype(np.uint8))


def write_page(out, image_id, size, lines, instances):
    """OUT/<image_id>.json and .npz: the pieces the page was cut into and what RevRender delivered on them.  An instance that was
    made from another (``tighten``) also names it: ``from_instance``, its index in the output it came from."""
    write_json(Path(out)/f'{image_id}.json', {
        'image_id': image_id, 'size': list(size),
        'lines': [{'index': k, 'box': line['box'], 'source': line['source'], 'ocr_type': line['ocr_type'], 'ocr_text': line['ocr_text']}
                  for k, line in enumerate(lines)],
        'instances': [{'line': i['line'], 'score': i['score'], 'box': i['box'], **({'from_instance': i['from_instance']} if 'from_instance' in i else {})}
                      for i in instances]})
    np.savez_compressed(Path(out)/f'{image_id}.npz', **{f'mask_{n}': np.packbits(i['mask'], axis=-1) for n, i in enumerate(instances)})


def read_output(directory, image_id):
    """(the page's record, its instances) from OUT/<image_id>.json and .npz: each instance is its entry of the record with ``mask``, a
    boolean array on its ``box``, added."""
    record = json.loads((Path(directory)/f'{image_id}.json').read_text())
    instances = []
    with np.load(Path(directory)/f'{image_id}.npz') as masks:
        for n, entry in enumerate(record['instances']):
            width = entry['box'][2]-entry['box'][0]
            instances.append({**entry, 'mask': np.unpackbits(masks[f'mask_{n}'], axis=-1)[:, :width].astype(bool)})
    return record, instances


def read_ocr_text(files):
    """{image_id: OCR text} from jsonl files of {image_id, text}."""
    text = {}
    for path in files:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                text[record['image_id']] = record.get('text') or ''
    return text


def page_files(directories, ids=None):
    """{image_id: file} for the ids (default: every page) found in the directories: a page is a jpg (the scan) or a png (a cleaned page)."""
    found = {}
    for directory in directories:
        for suffix in ('*.jpg', '*.png'):
            for path in sorted(Path(directory).glob(suffix)):
                found.setdefault(path.stem, path)
    if ids is None:
        return found
    missing = [i for i in ids if i not in found]
    require(not missing, f'{len(missing)} page(s) not found, e.g. {missing[:2]}')
    return {i: found[i] for i in ids}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha')
    parser.add_argument('--freeze', type=Path, required=True, help="the checkpoint's freeze record: its mask threshold, NMS and score cut")
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--ids', type=Path, help='a file of image ids, one per line (default: every jpg in --pages)')
    parser.add_argument('--ocr-model', type=Path, help='the Unlimited-OCR folder: run the model on pages without stored text')
    parser.add_argument('--ocr-text', type=Path, action='append', default=[], help='jsonl of {image_id, text} the model already produced')
    parser.add_argument('--templates', type=Path, help='printed templates (python -m revrender.data.handwriting): without them the printed ink counts as handwriting')
    parser.add_argument('--frames', type=Path, help='frames.json of data.clean_pages: pieces whose centre is outside their page\'s frame are dropped')
    parser.add_argument('--limit', type=int, default=0, help='only the first N pages')
    parser.add_argument('--overlay', action='store_true', help='also write <image_id>.png')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.ocr_model or args.ocr_text, 'give --ocr-model or --ocr-text')
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()] if args.ids else None
    files = page_files(args.pages, ids)
    names = sorted(files)[:args.limit or None] if ids is None else ids[:args.limit or None]
    stored = read_ocr_text(args.ocr_text)
    out = make_output_dir(args.output_dir)
    point = frozen_point(args.freeze, None)
    model, payload = infer.load_checkpoint(args.checkpoint, expected_sha=args.checkpoint_sha)
    ocr = None
    templates = dict(np.load(args.templates)) if args.templates else {}
    frames = json.loads(args.frames.read_text()) if args.frames else {}
    totals = {'pages': 0, 'lines': 0, 'lines_ocr': 0, 'lines_split': 0, 'lines_ink': 0, 'lines_outside': 0, 'instances': 0, 'ocr_seconds': 0., 'read_seconds': 0.}
    for image_id in names:
        started = time.monotonic()
        if image_id in stored:
            text = stored[image_id]
        else:
            require(args.ocr_model, f'{image_id}: no stored OCR text and no --ocr-model')
            if ocr is None:
                from .ocr import UnlimitedOCR
                ocr = UnlimitedOCR(args.ocr_model)
            text = ocr.read(files[image_id])
        totals['ocr_seconds'] += time.monotonic()-started
        started = time.monotonic()
        with Image.open(files[image_id]) as opened:
            image = opened.convert('L')
        figures = [pixel_box(box, image.size) for kind, box in ocr_lines.regions(text) if kind in ocr_lines.PRINTED]
        ink = handwriting.handwriting(image, templates.get(handwriting.group_key(image_id, image.size)), [b for b in figures if b])
        lines = page_lines(text, image.size, ink)
        if frames.get(image_id):
            kept = [line for line in lines if inside(frames[image_id], line['box'])]
            totals['lines_outside'] += len(lines)-len(kept)
            lines = kept
        instances = read_page(model, point, image, lines)
        write_page(out, image_id, image.size, lines, instances)
        if args.overlay:
            overlay(image, instances).save(out/f'{image_id}.png')
        totals['read_seconds'] += time.monotonic()-started
        totals['pages'] += 1
        totals['lines'] += len(lines)
        for source in ('ocr', 'split', 'ink'):
            totals[f'lines_{source}'] += sum(line['source'] == source for line in lines)
        totals['instances'] += len(instances)
        if totals['pages'] % 25 == 0:
            print(json.dumps({'pages_done': totals['pages'], **{k: round(v, 1) for k, v in totals.items()}}), flush=True)
    weights = sorted(Path(args.ocr_model).glob('*.safetensors')) if ocr is not None else []
    write_manifest(out, config=to_jsonable(args),
                   inputs={str(args.checkpoint): digest(args.checkpoint), str(args.freeze): digest(args.freeze),
                           **{str(p): digest(p) for p in args.ocr_text}, **{str(p): digest(p) for p in weights},
                           **({str(args.templates): digest(args.templates)} if args.templates else {}),
                           **({str(args.frames): digest(args.frames)} if args.frames else {})},
                   point={k: point[k] for k in ('mask_threshold', 'nms', 'score_threshold')}, checkpoint_step=payload.get('step'),
                   **totals)
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in totals.items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
