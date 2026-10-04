"""Recipes of synthetic revisions on clean lines: the training corpus of the reader (``data.paste_strikes`` draws them).

    python -m revrender.data.reader_corpus --mapping index_manifest.json --same-student pairs.json --exclude-pack PACK --output-dir OUT

A host is a line of the catalogue that a person or a consensus of models judged to have no edit, of a writer in paper 1's training split,
whose text two strong models read alike (``AGREE``; the reading of the first is the text the reader is taught to give).  A host is never
a test writer's, nor a writer of any line of the evaluation pack (``--exclude-pack``: a reader taught on the writer's own hand reads that
writer's lines better for it), nor the other id of a student who has two (``--same-student``).  Each host is made into one recipe per entry of ``VARIANTS``: ``clean`` (the line itself), ``whole`` (the
whole line struck out: the target is nothing), ``paste`` (struck writing of the same writer's other lines put in: the target is the
host's text) and ``stack`` (the same, above or below a word of the host).  A recipe says which lines, which kind, which seed; the image is drawn when it is read (``data.paste_strikes.compose``), so
the corpus is a file of text.  A small share of the writers, chosen by hash, is the corpus's own validation split.

OUT/corpus.jsonl has a row per recipe (id, split, kind, line_id, image_id, writer, text, donors, seed) and OUT/summary.json the counts.
The text is student work: OUT stays out of Git.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from ..config import expand, load_paths
from ..eval.coverage import prepare, similarity
from ..util import SEED, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from . import lines as lines_module
from .prior_annotations import read_line_text, twins, writer_splits

AGREE = .9                 # the two readings of a host are at least this alike (difflib's ratio after normalising)
MIN_CHARS = 4              # and the text has at least this many characters
VARIANTS = ('clean', 'whole', 'paste', 'paste', 'stack')
DONORS = 6                 # other hosts of the writer a paste may take writing from
VAL_SHARE = .05            # of the writers
TYPES = ('text', 'equation')


def val_writer(writer, share=VAL_SHARE):
    """Whether a writer is in the corpus's validation split: a hash of the writer, so the choice does not move when the corpus does."""
    return int(hashlib.sha256(f'reader-val:{writer}'.encode()).hexdigest()[:8], 16)/16**8 < share


def hosts(catalogue, readings, writers, excluded):
    """The catalogue's clean lines the corpus is made from: [{line_id, image_id, writer, text}], in catalogue order.  ``readings`` is
    {model: {line id: text}} with the models ``qwen35_35b`` and ``infinity``; ``writers`` the writer -> split map of paper 1."""
    out = []
    for record in catalogue:
        writer = record.get('group_uuid') or record['image_id'].split('_')[0]
        if record.get('has_edit') or record.get('ocr_type') not in TYPES or writers.get(writer) != 'train' or writer in excluded:
            continue
        line_id = record['representative_line_id']
        first, second = readings['qwen35_35b'].get(line_id), readings['infinity'].get(line_id)
        if not first or not second or len(prepare(first)) < MIN_CHARS or similarity(first, second) < AGREE:
            continue
        out.append({'line_id': line_id, 'image_id': record['image_id'], 'writer': writer, 'text': first.strip(), 'box_1000': record['box_1000']})
    return out


def recipes(chosen, variants=VARIANTS, donors=DONORS, seed=SEED):
    """One recipe per host and variant.  A paste's donors are other hosts of the same writer (fewer when the writer has fewer; a writer with one
    host line has no paste recipes)."""
    by_writer = defaultdict(list)
    for h in chosen:
        by_writer[h['writer']].append(h['line_id'])
    out = []
    for h in chosen:
        others = [i for i in by_writer[h['writer']] if i != h['line_id']]
        for n, kind in enumerate(variants):
            if kind in ('paste', 'stack') and not others:
                continue
            draw = int(hashlib.sha256(f'{seed}:{h["line_id"]}:{n}'.encode()).hexdigest()[:8], 16)
            rng = random.Random(draw)
            out.append({'id': f'{h["line_id"]}#{n}', 'split': 'val' if val_writer(h['writer']) else 'train', 'kind': kind, 'line_id': h['line_id'],
                        'image_id': h['image_id'], 'writer': h['writer'], 'text': h['text'] if kind != 'whole' else '', 'host_text': h['text'],
                        'box_1000': h['box_1000'], 'donors': rng.sample(others, min(donors, len(others))) if kind in ('paste', 'stack') else [], 'seed': draw})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--catalogue', type=Path, help='the line catalogue (default: configs/paths.toml)')
    parser.add_argument('--root', type=Path, help="the earlier project's folder (default: configs/paths.toml [prior] root)")
    parser.add_argument('--mapping', type=Path, required=True, help="paper 1's split mapping (index_manifest.json)")
    parser.add_argument('--same-student', type=Path, help='json of students with two writer ids')
    parser.add_argument('--exclude-pack', type=Path, action='append', default=[], help='a line pack (folder) whose writers are left out (repeatable)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    catalogue = lines_module.read_catalogue(args.catalogue) if args.catalogue else lines_module.default_catalogue()
    root = args.root or expand(load_paths().get('prior', {}).get('root', ''))
    require(root and Path(root).is_dir(), 'give --root or set [prior] root in configs/paths.toml')
    writers = writer_splits(args.mapping, args.same_student)
    excluded = {r['image_id'].split('_')[0] for pack in args.exclude_pack for r in json.loads((pack/'pack.json').read_text())['rows']}
    excluded |= {twin for w, twin in twins(args.same_student).items() if w in excluded}      # a student with two writer ids has one hand
    readings = read_line_text(root, {r['representative_line_id'] for r in catalogue})
    chosen = hosts(catalogue, readings, writers, excluded)
    require(chosen, 'no host line')
    rows = recipes(chosen)
    require(not any(writers.get(r['writer']) == 'test' or r['writer'] in excluded for r in rows), 'a test or evaluation writer is in the corpus')
    out = make_output_dir(args.output_dir)
    with (out/'corpus.jsonl').open('x', encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {'hosts': len(chosen), 'writers': len({h['writer'] for h in chosen}), 'recipes': len(rows), 'excluded_writers': len(excluded),
               'by_kind': {k: sum(r['kind'] == k for r in rows) for k in sorted(set(VARIANTS))}, 'by_split': {s: sum(r['split'] == s for r in rows) for s in ('train', 'val')},
               'val_writers': len({r['writer'] for r in rows if r['split'] == 'val'}), 'agree': AGREE, 'min_chars': MIN_CHARS, 'variants': list(VARIANTS)}
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={str(args.mapping): digest(args.mapping), str(out/'corpus.jsonl'): digest(out/'corpus.jsonl')}, **summary)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
