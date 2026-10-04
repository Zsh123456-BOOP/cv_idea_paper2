"""A static HTML viewer for readings of the labelled lines: the scan, the labels and what each run read, with its leak and valid shares.

    python -m revrender.tools.reader_view --pack PACK --run NAME=PREDICTIONS.jsonl:CONDITION [--run ...] --output-dir SITE

``--pack`` is a ``revrender.data.line_pack`` folder; a run is a condition of a ``revrender.recognize_lines`` predictions file under a name
(``base=out/predictions.jsonl:raw_final``).  SITE/index.html has a card for every line of the pack that carries the text before the strike
(``eval.leakage.labelled``): the scan and the crop with the columns under the masks whited out, the text before and after the strike, and
for each run the reading with the shares ``eval.coverage`` gives it.  Cards sort by how far the first run's leak is from the last's.  The
images are copied to SITE/assets, so the folder opens from anywhere without a server.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from ..eval.leakage import flags, labelled
from ..util import digest, make_output_dir, require, to_jsonable, write_manifest
from .pages_view import embed

TEMPLATE = Path(__file__).with_name('reader_view.html')


def parse_run(spec):
    """(name, predictions path, condition) from ``NAME=PATH:CONDITION``."""
    name, _, rest = spec.partition('=')
    path, _, condition = rest.rpartition(':')
    require(name and path and condition, f'a run is NAME=PREDICTIONS.jsonl:CONDITION, not {spec!r}')
    return name, Path(path), condition


def readings(path, condition):
    """{line id of the pack: reading} of one condition of a predictions file."""
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line) if line.strip() else None
        if row and row['condition'] == condition:
            out[row['id']] = row['text']
    return out


def cards(rows, runs):
    """The cards of the labelled lines: the labels and, per run, the reading and its shares (None where the line has no such characters)."""
    out = []
    for row in rows:
        if not labelled(row):
            continue
        shown = {}
        for name, texts in runs.items():
            text = texts.get(row['id'])
            shown[name] = None if text is None else {'text': text, **{k: v for k, v in flags(row, text).items() if k != 'blank'}}
        first, last = (shown[n] for n in (list(runs)[0], list(runs)[-1]))
        gap = (first['leak'] or 0)-(last['leak'] or 0) if first and last else 0.
        out.append({'id': row['id'], 'kind': row.get('ink_kind') or 'unknown', 'masked': row.get('masked_pixels', 0) > 0, 'before': row.get('text_before') or '',
                    'after': row.get('text_after') or '', 'no_rewrite': bool(row.get('no_rewrite')), 'size': row['size'], 'runs': shown, 'gap': gap})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, help='NAME=PREDICTIONS.jsonl:CONDITION (repeatable)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    specs = [parse_run(s) for s in args.run]
    require(len({n for n, _, _ in specs}) == len(specs), 'duplicate run name')
    pack = json.loads((args.pack/'pack.json').read_text())
    runs = {name: readings(path, condition) for name, path, condition in specs}
    made = cards(pack['rows'], runs)
    require(made, 'no labelled line in the pack')
    out = make_output_dir(args.output_dir)
    (out/'assets').mkdir()
    by_id = {r['id']: r for r in pack['rows']}
    for card in made:
        for kind in ('raw', 'blanked'):
            shutil.copyfile(args.pack/by_id[card['id']]['images'][kind], out/'assets'/f'{card["id"]}_{kind}.png')
    (out/'index.html').write_text(TEMPLATE.read_text(encoding='utf-8').replace('__DATA__', embed({'cards': made, 'runs': [n for n, _, _ in specs]})), encoding='utf-8')
    write_manifest(out, config=to_jsonable(args), inputs={str(p): digest(p) for _, p, _ in specs}, lines=len(made))
    print(json.dumps({'lines': len(made), 'runs': [n for n, _, _ in specs], 'output': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
