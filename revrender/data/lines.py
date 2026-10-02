"""The line catalogue: where every handwritten line sits inside its whole-question image.

``exact_crops_46902.jsonl`` has one record per line (46,902 lines from 4,990 question images): ``image_id``
(``<group_uuid>_<question>_<part>``), ``representative_line_id`` (``<image_id>:L0001``), ``alias_line_ids``,
``box_1000`` (the line's box in the question image, in thousandths of its width and height), ``question_no``,
``ocr_text`` and ``group_uuid`` (the writer).  The file is student data; it is read from the path in
configs/paths.toml and never copied.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from ..util import digest, require


def read_catalogue(path, expected_sha=None):
    """All records, in file order."""
    path = Path(path).expanduser()
    if expected_sha is not None:
        require(digest(path) == expected_sha, 'line catalogue changed')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def index_by_line(entries):
    """{line id: record}, reached by the representative id or any alias."""
    index = {}
    for entry in entries:
        index[entry['representative_line_id']] = entry
        for alias in entry.get('alias_line_ids') or []:
            index.setdefault(alias, entry)
    return index


def lines_by_image(entries):
    """{image_id: [records]} in line order: the lines of each whole question."""
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry['image_id']].append(entry)
    for lines in grouped.values():
        lines.sort(key=lambda e: e['representative_line_id'])
    return dict(grouped)


def crop_box(box_1000, width, height):
    """Pixel box (left, top, right, bottom) of a thousandths box on an image of this size."""
    require(len(box_1000) == 4, 'box_1000 must have four values')
    x0, y0, x1, y1 = (float(v) for v in box_1000)
    require(0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000, 'box_1000 must be ordered within [0,1000]')
    left = max(0, min(width - 1, math.floor(x0*width/1000)))
    top = max(0, min(height - 1, math.floor(y0*height/1000)))
    right = max(left + 1, min(width, math.ceil(x1*width/1000)))
    bottom = max(top + 1, min(height, math.ceil(y1*height/1000)))
    return left, top, right, bottom


def crop_line(image, box_1000):
    """The line's crop (a PIL image) cut from its whole-question image."""
    return image.crop(crop_box(box_1000, *image.size))
