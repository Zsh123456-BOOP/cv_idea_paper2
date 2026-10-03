"""The paper-1 gold on the whole-question page.

Paper 1 annotated lines: a gold revision is a component of the mask of a 96x768 strip (``data.gold``).  The line sits in its whole-question
image at the catalogue's box, so a gold instance is the strip's component resized back to the line's crop and pasted at that box.  Written
in the layout of ``revrender.pages`` (OUT/<image_id>.json and .npz, score 1, the annotated lines as pieces of source ``gold``), so what
reads a page's output reads its gold too.

    python -m revrender.data.page_gold --bundle B --bundle-sha S --catalogue FILE --pages DIR [--pages DIR ...] [--split dev] --output-dir OUT

``--split`` (repeatable: train, dev) names the bundle's rows to put back; a bundle holds no test row.
"""
import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .. import pages
from ..util import digest, make_output_dir, require, to_jsonable, write_manifest
from . import lines as lines_module
from .bundle import load_bundle
from .gold import unletterbox


def page_gold(rows, index, size):
    """(pieces, instances) of one page from its bundle ``rows``: a piece per annotated line (in line order), an instance per gold
    component, each on its line's crop box; ``index`` is ``data.lines.index_by_line`` and ``size`` the page's (width, height)."""
    pieces, instances = [], []
    for k, row in enumerate(sorted(rows, key=lambda r: r['line_id'])):
        box = list(lines_module.crop_box(index[row['line_id']]['box_1000'], *size))
        pieces.append({'box': box, 'source': 'gold', 'ocr_type': None, 'ocr_text': ''})
        for component in row['instances']:
            crop = unletterbox(component, box[2]-box[0], box[3]-box[1])
            if crop.any():
                instances.append({'line': k, 'score': 1., 'box': box, 'mask': crop})
    return pieces, instances


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--catalogue', type=Path, required=True, help='the line catalogue (exact_crops jsonl)')
    parser.add_argument('--pages', type=Path, action='append', required=True, help='a folder of <image_id>.jpg (repeatable)')
    parser.add_argument('--split', action='append', choices=('train', 'dev'), help='which rows to put back (repeatable; default dev)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    splits = args.split or ['dev']
    bundle = load_bundle(args.bundle, args.bundle_sha)
    index = lines_module.index_by_line(lines_module.read_catalogue(args.catalogue))
    rows = defaultdict(list)
    for split in splits:
        for row in bundle[split]:
            require(row['line_id'] in index, f'{row["line_id"]} is not in the catalogue')
            rows[row['image_id']].append(row)
    ids = sorted(rows)
    files = pages.page_files(args.pages, ids)
    out = make_output_dir(args.output_dir)
    totals = {'pages': 0, 'lines': 0, 'instances': 0}
    for image_id in ids:
        with Image.open(files[image_id]) as opened:
            size = opened.size
        pieces, instances = page_gold(rows[image_id], index, size)
        pages.write_page(out, image_id, size, pieces, instances)
        totals['pages'] += 1
        totals['lines'] += len(pieces)
        totals['instances'] += len(instances)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.bundle/'run_manifest.json'): digest(args.bundle/'run_manifest.json'),
                                                           str(args.catalogue): digest(args.catalogue)},
                   splits=splits, **totals)
    print(totals)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
