"""What the strike GAN learns from: lines a student struck out (real) and clean lines for its generator to strike, letterboxed onto
paper 1's strip.

    python -m revrender.data.gan_sets --detections PAGES_OUT [--detections ...] --pages DIR [--pages DIR ...] --mapping index_manifest.json
        --same-student pairs.json [--catalogue FILE] [--missing ids.txt] --output-dir OUT

Lines of the catalogue (``data.lines``) of the writers training may use: paper 1's training writers and the writers it did not use,
never a student with an id in its development or test split.  Every line was judged revised or not (``has_edit``, people and models,
97% right).  ``--detections`` are paper 1's RevRender run on the pages (``revrender.pages``, cut to ink by ``tighten``): an instance
lies on a line when its box's centre does.  Real: a revised line whose instances are all struck lines (thin and long:
``strike_bank.kind_of``), at least one -- a line with a blackout or an overwrite is left out, as the generator strikes only.  Hosts:
an unrevised line on which RevRender found nothing either.  A page with no detections gives neither; ``--missing`` writes the ids
of such pages that hold a revised line of these writers, for RevRender to be run on them first.

OUT/real.npy and OUT/hosts.npy ((N, 96, 768) uint8 grey), OUT/real.json and OUT/hosts.json (line id, page, writer, its split in paper
1 -- train or unused --, box on the page, its height on the strip), OUT/sheet_real.png and OUT/sheet_hosts.png.  Student
handwriting: OUT stays out of Git.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .. import pages as pages_module
from ..models.strike_gan import CANVAS
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from . import lines as lines_module
from .gold import letterbox
from .prior_annotations import twins, writer_splits
from .strike_bank import kind_of


def usable_writers(mapping, same_student, catalogue):
    """{writer: 'train' | 'unused'} of the catalogue's writers training may use: no id of a student in paper 1's dev or test split."""
    splits, other = writer_splits(mapping, same_student), twins(same_student)
    out = {}
    for record in catalogue:
        writer = record.get('group_uuid') or record['image_id'].split('_')[0]
        mine, twin = splits.get(writer, 'unused'), splits.get(other.get(writer), 'unused')
        if mine in ('train', 'unused') and twin in ('train', 'unused'):
            out[writer] = mine
    return out


def strip_height(box):
    """A line's height on the strip it is letterboxed onto."""
    width, height = box[2]-box[0], box[3]-box[1]
    return height*min(CANVAS[1]/width, CANVAS[0]/height)


def sort_lines(rows, instances, size):
    """(real, hosts) of a page's catalogue rows given RevRender's instances on it (page pixels)."""
    real, hosts = [], []
    for r in rows:
        box = lines_module.crop_box(r['box_1000'], *size)
        mine = [i for i in instances if box[0] <= (i['box'][0]+i['box'][2])/2 <= box[2] and box[1] <= (i['box'][1]+i['box'][3])/2 <= box[3]]
        if r.get('has_edit'):
            if mine and all(kind_of(i['mask'], box[3]-box[1]) == 'line' for i in mine):
                real.append((r, box))
        elif not mine:
            hosts.append((r, box))
    return real, hosts


def sheet(canvases, path, count=24):
    chosen = random.Random(0).sample(range(len(canvases)), min(count, len(canvases)))
    height, width = CANVAS
    out = Image.new('L', (width, len(chosen)*(height+4)), 160)
    for k, i in enumerate(chosen):
        out.paste(Image.fromarray(canvases[i]), (0, k*(height+4)))
    out.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--detections', type=Path, action='append', required=True, help='revrender.pages output cut to ink (repeatable)')
    parser.add_argument('--pages', type=Path, action='append', required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--same-student', type=Path, required=True)
    parser.add_argument('--catalogue', type=Path, help='the line catalogue (default: configs/paths.toml)')
    parser.add_argument('--missing', type=Path, help='only write the ids of pages with a revised line of these writers and no detections')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    catalogue = lines_module.read_catalogue(args.catalogue) if args.catalogue else lines_module.default_catalogue()
    writers = usable_writers(args.mapping, args.same_student, catalogue)
    detected = {}
    for folder in args.detections:
        for p in folder.glob('*.json'):
            if p.stem not in ('run_manifest', 'summary'):
                detected.setdefault(p.stem, folder)
    by_page = defaultdict(list)
    for r in catalogue:
        if (r.get('group_uuid') or r['image_id'].split('_')[0]) in writers:
            by_page[r['image_id']].append(r)
    if args.missing:
        ids = sorted(i for i, rows in by_page.items() if i not in detected and any(r.get('has_edit') for r in rows))
        args.missing.write_text(''.join(i+'\n' for i in ids))
        print(json.dumps({'missing_pages': len(ids)}))
        return 0
    require(args.output_dir, 'give --output-dir (or --missing)')
    scans = pages_module.page_files(args.pages, sorted(i for i in by_page if i in detected))
    real, hosts, real_records, host_records = [], [], [], []
    for image_id in sorted(scans):
        record, instances = pages_module.read_output(detected[image_id], image_id)
        page_real, page_hosts = sort_lines(by_page[image_id], instances, record['size'])
        if not page_real and not page_hosts:
            continue
        with Image.open(scans[image_id]) as opened:
            grey = opened.convert('L')
        for rows, canvases, records in ((page_real, real, real_records), (page_hosts, hosts, host_records)):
            for r, box in rows:
                canvases.append(np.asarray(letterbox(grey.crop(tuple(box)), CANVAS[1], CANVAS[0], 255)))
                writer = r.get('group_uuid') or image_id.split('_')[0]
                records.append({'line_id': r['representative_line_id'], 'image_id': image_id, 'writer': writer, 'paper1_split': writers[writer],
                                'box': [int(v) for v in box], 'strip_height': strip_height(box)})
    require(real and hosts, 'no real line or no host line')
    out = make_output_dir(args.output_dir)
    np.save(out/'real.npy', np.stack(real))
    np.save(out/'hosts.npy', np.stack(hosts))
    write_json(out/'real.json', real_records)
    write_json(out/'hosts.json', host_records)
    sheet(real, out/'sheet_real.png')
    sheet(hosts, out/'sheet_hosts.png')
    summary = {'real': len(real), 'hosts': len(hosts), 'real_writers': len({r['writer'] for r in real_records}),
               'host_writers': len({r['writer'] for r in host_records}), 'pages': len(scans),
               'by_split': {s: {'real': sum(r['paper1_split'] == s for r in real_records), 'hosts': sum(r['paper1_split'] == s for r in host_records)}
                            for s in ('train', 'unused')}}
    files = {name: digest(out/name) for name in ('real.npy', 'hosts.npy', 'real.json', 'hosts.json')}
    write_manifest(out, config=to_jsonable(args), inputs={str(args.mapping): digest(args.mapping), str(args.same_student): digest(args.same_student)},
                   files=files, **summary)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
