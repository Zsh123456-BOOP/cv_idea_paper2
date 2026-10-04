"""What the earlier project already holds on the same lines: the human before/after pairs, the blind no-edit judgements and the text three
models read for every line.

Every record is keyed by the catalogue's line id (``<image_id>:L<nnnn>``), the id this project cuts pages with, so none of it has to be
mapped.  The sources are files of the earlier project (``configs/paths.toml`` ``[prior] root``; read-only, never copied, the text is
student work and stays out of Git):

  revision pair annotation (300 lines)   text_before = the struck text, text_after = what replaced it, ink_kind, and the escapes
                                         no_revision / illegible / before_unreadable / no_rewrite
  combined task B (200 lines, 103 filled) the same fields with edit_kind (strike_rewrite, strike_only, whole_line, overwrite)
  combined task A (500 lines, blind)     has_edit true / false: the human-confirmed no-edit lines are the controls
  line readings (all 46,902 lines)       Qwen3.5-35B-A3B-LaTeX-OCR and Infinity-Parser2-Pro on the exact line crop, prompted to keep
                                         crossed-out writing, and Unlimited-OCR's text (the catalogue's)

    python -m revrender.data.prior_annotations [--root DIR] [--tags tags.json] --output-dir OUT

OUT/pairs.jsonl has one record per annotated line, OUT/summary.json the counts and what the three models did on the revised lines.
``--mapping`` is paper 1's split mapping (``index_manifest.json`` of the data bundle: it names the split of every writer, the external
set's included) and ``--same-student`` the json of students who have two writer ids (a student whose other id is a test writer is a
test writer); without them ``--tags`` (a json {image_id: {split, rows}}) gives the split of the pages it knows.  Lines of a test writer are
counted but ``summary.json`` reads the models on the others only.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from ..config import expand, load_paths
from ..eval.fragments import contains, similarity
from ..util import digest, make_output_dir, require, to_jsonable, write_json, write_manifest

FILES = {
    'pair_key': 'outputs/revision_pair_annotation_20260822_v1/key.jsonl',
    'pair_returned': 'gold/revision_region_annotation_exports_20260829_v1/returned_annotations/revision_pair_returned_v1.json',
    'combined_export': 'outputs/combined_annotation_export_20260820_v1/export.json',
    'combined_b_key': 'outputs/combined_annotation_page_v3/task_b_key.jsonl',
    'audit_sample': 'outputs/policy_label_audit_sample_v1/sample_blind.jsonl',
}
LINE_READINGS = {
    'qwen35_35b': 'outputs/ocr_line_all46902_results_20260816_v1/server_original/qwen_v1/crop_predictions.jsonl',
    'infinity': 'outputs/ocr_line_all46902_results_20260816_v1/server_original/infinity_v1/crop_predictions.jsonl',
}
MODELS = ('ulocr', *LINE_READINGS)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def read_pairs(root):
    """{line_id: record} from the three human sources: line_id, image_id, sources, and the fields each source holds."""
    root = Path(root).expanduser()
    out = {}

    def add(line_id, image_id, source, **fields):
        record = out.setdefault(line_id, {'line_id': line_id, 'image_id': image_id, 'sources': []})
        record['sources'].append(source)
        record.update({k: v for k, v in fields.items() if v is not None})

    key = {k['uid']: k for k in lines(root/FILES['pair_key'])}
    for uid, r in json.loads((root/FILES['pair_returned']).read_text(encoding='utf-8'))['records'].items():
        add(key[uid]['line_id'], key[uid]['image_id'], 'pair_v1', text_before=r.get('text_before'), text_after=r.get('text_after'),
            ink_kind=r.get('ink_kind'), no_rewrite=r.get('no_rewrite'), before_unreadable=r.get('before_unreadable'),
            no_revision=r.get('no_revision'), illegible=r.get('illegible'))
    export = json.loads((root/FILES['combined_export']).read_text(encoding='utf-8'))['records']
    for k in lines(root/FILES['combined_b_key']):
        r = export.get(k['uid'])
        if r:
            add(k['line_id'], k['image_id'], 'combined_B', text_before=r.get('text_before'), text_after=r.get('text_after'),
                ink_kind=r.get('edit_kind'), no_rewrite=r.get('no_rewrite'), before_unreadable=r.get('before_unreadable'))
    sample = {s['audit_id']: s for s in lines(root/FILES['audit_sample'])}
    for uid, r in export.items():
        if uid.startswith('A_') and 'has_edit' in r:
            s = sample[uid[2:]]
            add(s['representative_line_id'], s['image_id'], 'combined_A', has_edit=r['has_edit'] == 'true')
    return out


def kind_of(record):
    """'pair' (a struck or rewritten fragment is labelled), 'no_edit' (blind judgement: none), 'edit_no_text' (blind judgement: an edit,
    nothing written), 'no_revision' (the pair annotator found none), 'other' (illegible and the like)."""
    if record.get('text_before') or record.get('text_after'):
        return 'pair'
    if record.get('has_edit') is False:
        return 'no_edit'
    if record.get('has_edit'):
        return 'edit_no_text'
    return 'no_revision' if record.get('no_revision') else 'other'


def read_line_text(root, wanted=None):
    """{model: {line_id: text}} for 'ulocr' and the two models of ``LINE_READINGS`` (only the line ids in ``wanted`` when given)."""
    root = Path(root).expanduser()
    out = {model: {} for model in MODELS}
    for model, relative in LINE_READINGS.items():
        for r in lines(root/relative):
            for line_id in {r['representative_line_id'], *r.get('alias_line_ids', [])}:
                if wanted is None or line_id in wanted:
                    out[model][line_id] = r['model_text']
            if model == 'qwen35_35b' and (wanted is None or r['representative_line_id'] in wanted):
                out['ulocr'][r['representative_line_id']] = r['ulocr_text']
    return out


def writer_splits(mapping, same_student=None):
    """{writer: split} from paper 1's split mapping (``index_manifest.json``); of two ids of one student, one a test writer, both are."""
    splits = {row['group_uuid']: row['split'] for row in json.loads(Path(mapping).read_text(encoding='utf-8'))['rows']}
    if same_student:
        for pair in json.loads(Path(same_student).read_text(encoding='utf-8'))['pairs']:
            ids = list(pair.values())
            if any(splits.get(i) == 'test' for i in ids):
                splits.update({i: 'test' for i in ids})
    return splits


def split_of(record, tags, writers=None):
    """The writer's split in paper 1 ('none': not used there): from the writer mapping when given, else from the page tags."""
    if writers is not None:
        return writers.get(record['image_id'].split('_')[0], 'none')
    return tags.get(record['image_id'], {}).get('split', 'none')


def summarise(pairs, texts, tags, writers=None):
    """Counts by kind and split, and for the revised lines of the non-test writers what each model's reading holds of the labelled
    fragments; the agreement of the two models on revised lines against the no-edit lines."""
    counts = defaultdict(Counter)
    for r in pairs.values():
        counts[kind_of(r)][split_of(r, tags, writers)] += 1
    revised = [r for r in pairs.values() if kind_of(r) == 'pair' and split_of(r, tags, writers) != 'test']
    controls = [r for r in pairs.values() if kind_of(r) == 'no_edit' and split_of(r, tags, writers) != 'test']
    readings = {}
    for model in MODELS:
        seen = Counter()
        for r in revised:
            reading = texts[model].get(r['line_id'])
            if reading is None:
                continue
            before = contains(reading, r['text_before']) if r.get('text_before') else None
            after = contains(reading, r['text_after']) if r.get('text_after') else None
            seen['before_labelled'] += before is not None
            seen['before_in_reading'] += bool(before)
            seen['after_labelled'] += after is not None
            seen['after_in_reading'] += bool(after)
        readings[model] = dict(seen)
    agree = {}
    for label, group in (('revised', revised), ('no_edit', controls)):
        s = [similarity(texts['qwen35_35b'][r['line_id']], texts['infinity'][r['line_id']]) for r in group
             if r['line_id'] in texts['qwen35_35b'] and r['line_id'] in texts['infinity']]
        agree[label] = {'lines': len(s), 'median': median(s) if s else None, 'share_at_least_0.9': sum(x >= .9 for x in s)/len(s) if s else None}
    return {'by_kind_and_split': {k: dict(v) for k, v in counts.items()}, 'revised_non_test_lines': len(revised), 'no_edit_non_test_lines': len(controls),
            'ink_kinds_of_revised': dict(Counter(str(r.get('ink_kind')) for r in revised)), 'readings_of_labelled_fragments': readings,
            'qwen35_35b_vs_infinity': agree}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--root', type=Path, help="the earlier project's folder (default: configs/paths.toml [prior] root)")
    parser.add_argument('--tags', type=Path, help='json {image_id: {split, rows}}: the writer\'s split in paper 1')
    parser.add_argument('--mapping', type=Path, help="paper 1's split mapping (index_manifest.json): the split of every writer, the external set's too")
    parser.add_argument('--same-student', type=Path, help='json of students with two writer ids (the pairs a test writer taints)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root or expand(load_paths().get('prior', {}).get('root', ''))
    require(root and Path(root).is_dir(), 'give --root or set [prior] root in configs/paths.toml')
    tags = json.loads(args.tags.read_text()) if args.tags else {}
    writers = writer_splits(args.mapping, args.same_student) if args.mapping else None
    pairs = read_pairs(root)
    texts = read_line_text(root, set(pairs))
    out = make_output_dir(args.output_dir)
    with (out/'pairs.jsonl').open('x', encoding='utf-8') as f:
        for record in pairs.values():
            f.write(json.dumps({**record, 'kind': kind_of(record), 'split': split_of(record, tags, writers),
                                **{model: texts[model].get(record['line_id']) for model in MODELS}}, ensure_ascii=False)+'\n')
    summary = summarise(pairs, texts, tags, writers)
    write_json(out/'summary.json', summary)
    write_manifest(out, config=to_jsonable(args), inputs={**{str(Path(root)/relative): digest(Path(root)/relative)
                                                              for relative in (*FILES.values(), *LINE_READINGS.values())},
                                                           **({str(args.mapping): digest(args.mapping)} if args.mapping else {})}, lines=len(pairs))
    print(json.dumps({'lines': len(pairs), **{k: sum(v.values()) for k, v in summary['by_kind_and_split'].items()}}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
