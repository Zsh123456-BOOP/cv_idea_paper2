"""Whose pages this project may use (user 2026-10-18; paper 1 decided the same on 2026-10-15).

A labmate's annotated set (the scans in ``标注好数据``, ``full1026_v1`` on the server: 146 writers) is never used: no image, line, label,
crop, synthetic host or stroke piece of it.  Only the writers of the unlabeled set the user was given may be read; their uuids are in
``configs/allowed_writers.txt`` (499 ids, no student number).  The guard sits where every page and line enters: ``pages.page_files``
refuses an id outside the list and leaves such files out of a folder listing, ``data.lines.read_catalogue`` drops their lines, and the
builders that start from paper 1's records (``page_gold``, ``strike_bank``, ``prior_annotations``) leave them out.

    python -m revrender.data.allowed scan DIR [DIR ...]     # the page ids of other writers named in a folder's file names or json text
"""
import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from ..util import ROOT, require

LIST = ROOT/'configs'/'allowed_writers.txt'
PAGE = re.compile(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(s?\d+)_(\d+)')


@lru_cache(maxsize=None)
def allowed_writers(path=None):
    """The writer uuids whose pages may be used."""
    found = frozenset(x.strip() for x in Path(path or LIST).read_text().split() if x.strip())
    require(found, f'{path or LIST}: no writer')
    return found


def writer_of(image_id):
    """The writer of an image id (``<writer>_<question>_<part>``), a line id (``<image_id>:L0001``) or a file stem."""
    return str(image_id).split('_')[0].split(':')[0]


UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def allowed(image_id, path=None):
    """Whether the id's writer may be used.  Every real writer is a uuid; a name that is not one (test fixtures) is no student."""
    writer = writer_of(image_id)
    return not UUID.match(writer) or writer in allowed_writers(path)


def refuse_outside(ids, what='page', path=None):
    """Refuse any id of a writer outside the allowed set."""
    outside = sorted({i for i in ids if not allowed(i, path)})
    require(not outside, f'{len(outside)} {what}(s) of writers outside the allowed set (a labmate\'s data), e.g. {outside[:2]}')


def scan(paths, limit=400_000_000):
    """{folder: sorted ids of other writers} named in the file names or the json/jsonl/txt/html text under each folder."""
    out = {}
    for root in paths:
        found = set()
        for directory, _, files in os.walk(root):
            for name in files:
                found |= {'_'.join(m) for m in PAGE.findall(name)}
                path = Path(directory)/name
                if path.suffix in ('.json', '.jsonl', '.txt', '.html', '.csv', '.md') and path.stat().st_size <= limit:
                    found |= {'_'.join(m) for m in PAGE.findall(path.read_text(errors='ignore'))}
        out[str(root)] = sorted(p for p in found if not allowed(p))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('scan')
    s.add_argument('paths', type=Path, nargs='+')
    args = parser.parse_args(argv)
    for root, ids in scan(args.paths).items():
        print(json.dumps({'folder': root, 'other_writer_pages': len(ids), 'writers': len({writer_of(i) for i in ids}), 'example': ids[:2]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
