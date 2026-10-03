"""What a page's output delivered, against the gold put back on the page.

    python -m revrender.eval.pages_score --pages-output OUT --gold-output GOLD --output scores.json [--ids FILE]

OUT is the output of ``revrender.pages`` and GOLD that of ``data.page_gold`` (each also as cut to ink by ``revrender.tighten``).  Every
instance OUT holds counts as delivered -- the model's frozen point already cut it -- and is read against the gold components of its page with
the paper-1 fates (``eval.fates``: matched at IoU 0.50, located, missed).  The pages scored are those of both outputs (or ``--ids``); a page
with gold lines and nothing delivered counts, and so does a delivered mask on a page whose lines hold no gold.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import pages
from ..tighten import listed_ids
from ..util import digest, provenance, require, to_jsonable
from . import fates

POINT = {'nms': 1., 'score_threshold': 0.}      # nothing is cut again: the output already is what was delivered


def score_page(delivered, truth, size):
    """The fates of one page: ``delivered`` and ``truth`` are lists of instances (``pages.read_output``) on a page of this size."""
    masks = [pages.paste(i, size) for i in delivered]
    scores = np.array([i['score'] for i in delivered], np.float64)
    return fates.line_record(masks, scores, [pages.paste(t, size) for t in truth], POINT) if masks else \
        {'tp': 0, 'fp': 0, 'gold': [{'fate': 'missed', 'iou': 0.} for _ in truth], 'predicted': []}


def score(pages_output, gold_output, ids=None):
    """(the summary of ``fates.summarise`` over the pages in both outputs, those pages' ids)."""
    delivered_ids = set(listed_ids(pages_output))
    chosen = [i for i in listed_ids(gold_output, ids) if i in delivered_ids]
    records = []
    for image_id in chosen:
        _, delivered = pages.read_output(pages_output, image_id)
        record, truth = pages.read_output(gold_output, image_id)
        records.append(score_page(delivered, truth, record['size']))
    return fates.summarise(records), chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pages-output', type=Path, required=True)
    parser.add_argument('--gold-output', type=Path, required=True)
    parser.add_argument('--ids', type=Path, help='a file of image ids, one per line (default: the pages of both outputs)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'create-only output exists', FileExistsError)
    summary, chosen = score(args.pages_output, args.gold_output, args.ids)
    require(chosen, 'no page is in both outputs')
    with args.output.open('x') as f:
        f.write(json.dumps({'scores': summary, 'pages': len(chosen), 'provenance': provenance(to_jsonable(args), {
            str(d): digest(d/'run_manifest.json') for d in (args.pages_output, args.gold_output)})}, indent=1, sort_keys=True)+'\n')
    print(json.dumps({'pages': len(chosen), 'gold': summary['gold_components'], 'found': round(summary['found'], 4),
                      'drawn_right': round(summary['drawn_right'], 4), 'false_positive': summary['false_positive']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
