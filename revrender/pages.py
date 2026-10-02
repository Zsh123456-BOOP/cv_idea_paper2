"""Whole-question images end to end: cut the lines with the OCR model, read each line with RevRender, paste the instances back.

    python -m revrender.pages --checkpoint best.pt --freeze freeze.json --pages DIR [--pages DIR ...] --output-dir OUT
        (--ocr-model MODEL_DIR | --ocr-text ocr.jsonl [--ocr-text ...]) [--ids FILE] [--limit N] [--overlay]

A page takes the route paper 1 trained for: Unlimited-OCR cuts it into regions (``ocr.py``), ``data.ocr_lines`` into the lines of the
catalogue, each line is cropped by its box and letterboxed to the 96x768 strip, RevRender reads the strip at the checkpoint's frozen
point (mask threshold, NMS, score cut), and every delivered instance is resized back to its crop and pasted into the page.  Per page
OUT/<image_id>.json lists the lines and the instances (line, score, box in pixels) and OUT/<image_id>.npz holds each instance's mask on
its crop (packed bits, ``mask_<n>``); ``--overlay`` also writes <image_id>.png.  ``--ocr-text`` takes OCR text already produced by the
model (jsonl of {image_id, text}) instead of running it.  Only the pages named by ``--ids`` (default: every jpg in ``--pages``) are read.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import infer
from .data import lines as lines_module
from .data.gold import CANVAS, letterbox, unletterbox
from .data.ocr_lines import strict_lines
from .eval.freeze import frozen_point
from .util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

STRIP_H, STRIP_W = CANVAS


def clamped(box):
    """A thousandths box inside the page, or None when nothing of it is left."""
    x0, y0, x1, y1 = (min(max(float(v), 0.), 1000.) for v in box)
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


def read_page(model, point, image, lines, batch_size=infer.BATCH_SIZE):
    """The instances RevRender delivers on a grey page cut into ``lines`` (``strict_lines``):
    [{line, score, box (left, top, right, bottom in pixels), mask (on the crop)}]."""
    width, height = image.size
    cuts = []
    for k, line in enumerate(lines):
        box = clamped(line['box'])
        if box:
            cuts.append((k, lines_module.crop_box(box, width, height)))
    strips = [np.asarray(letterbox(image.crop(box), STRIP_W, STRIP_H, 255)) for _, box in cuts]
    instances = []
    for (k, box), delivered in zip(cuts, infer.predict_lines(model, strips, point, batch_size)):
        for mask, score in delivered:
            crop = unletterbox(mask, box[2]-box[0], box[3]-box[1])
            if crop.any():
                instances.append({'line': k, 'score': float(score), 'box': list(box), 'mask': crop})
    return instances


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
    """OUT/<image_id>.json and .npz: the lines the page was cut into and what RevRender delivered on them."""
    write_json(Path(out)/f'{image_id}.json', {
        'image_id': image_id, 'size': list(size),
        'lines': [{'index': k, 'box_1000': line['box'], 'ocr_type': line['ocr_type'], 'ocr_text': line['ocr_text']}
                  for k, line in enumerate(lines)],
        'instances': [{'line': i['line'], 'score': i['score'], 'box': i['box']} for i in instances]})
    np.savez_compressed(Path(out)/f'{image_id}.npz', **{f'mask_{n}': np.packbits(i['mask'], axis=-1) for n, i in enumerate(instances)})


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
    """{image_id: file} for the ids (default: every jpg) found in the directories."""
    found = {path.stem: path for directory in directories for path in sorted(Path(directory).glob('*.jpg'))}
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
    totals = {'pages': 0, 'lines': 0, 'instances': 0, 'ocr_seconds': 0., 'read_seconds': 0.}
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
        lines = strict_lines(text)
        instances = read_page(model, point, image, lines)
        write_page(out, image_id, image.size, lines, instances)
        if args.overlay:
            overlay(image, instances).save(out/f'{image_id}.png')
        totals['read_seconds'] += time.monotonic()-started
        totals['pages'] += 1
        totals['lines'] += len(lines)
        totals['instances'] += len(instances)
        if totals['pages'] % 25 == 0:
            print(json.dumps({'pages_done': totals['pages'], **{k: round(v, 1) for k, v in totals.items()}}), flush=True)
    weights = sorted(Path(args.ocr_model).glob('*.safetensors')) if ocr is not None else []
    write_manifest(out, config=to_jsonable(args),
                   inputs={str(args.checkpoint): digest(args.checkpoint), str(args.freeze): digest(args.freeze),
                           **{str(p): digest(p) for p in args.ocr_text}, **{str(p): digest(p) for p in weights}},
                   point={k: point[k] for k in ('mask_threshold', 'nms', 'score_threshold')}, checkpoint_step=payload.get('step'),
                   **totals)
    print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in totals.items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
