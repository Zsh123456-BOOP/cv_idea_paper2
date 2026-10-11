"""Phase 2 of the diagnostic study (docs/struck_diagnostics_plan.md §12.1): does knowing the struck content help a VLM judge a revision?

    python -m revrender.eval.revision_judgment build --config configs/revision_judgment.toml --output-dir PACK
    python -m revrender.recognize_lines --pack PACK --conditions from_pack --model MODEL ... --output-dir READ
    python -m revrender.eval.revision_judgment score --config configs/revision_judgment.toml --pack PACK --judgments NAME=READ/predictions.jsonl ...
        --output-dir SCORE

Every confirmed revision of the diagnostic set P is judged four times on the same whole-page image (its blue box, the R / S3 image of the
diagnostic build) with the question, its standard solution and answer; the four differ in one sentence: (a) no struck content, (b) the
annotator's struck text, (c1) Qwen3-VL-8B's reading of the raw crop (R / S1), (c2) its reading of the refined erased crop (E* / S1).  The
judge answers three fields -- the struck content right or wrong, the rewrite after it right or wrong, the kind of revision -- against the
annotator's confirmed judgments.  Nothing is trained; the diagnostic build already refused test writers.
"""
import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..util import RefusedError, digest, make_output_dir, require, to_jsonable, write_json, write_manifest
from .struck_diagnostics import bootstrap, load_config

CONDITIONS = ('a', 'b', 'c1', 'c2')
NAMES = {'a': '不给被划内容', 'b': '人工被划文字', 'c1': '8B 读原图', 'c2': '8B 读擦线图'}
READ_FROM = {'c1': 'R', 'c2': 'E*'}           # the diagnostic version each model reading comes from (scope S1)
FIELDS = ('struck', 'after', 'type')
FIELD_NAMES = {'struck': '被划内容', 'after': '改后', 'type': '类型'}
RIGHT = ('对', '错', '不确定', '不适用')
TYPES = ('纠错', '改错', '错改错', '抵消', '笔误重写', '等价改写', '换思路', '删除多余', '放弃', '不确定')
SYSTEM = 'You are a careful mathematics teacher grading handwritten student work.'
PROMPT = '''图中是一名学生对下面这道题的整页作答。蓝色框标出了学生的一处修改：框里的内容被学生划掉（作废）了。

【题目】
{problem}

【标准解答】
{solution}

【答案】{answer}

{struck}

请对照标准解答和学生的整份作答，判断这处修改：
1. struck：框里被划掉的内容本身对不对（对／错／不确定／不适用）。被划掉的不是数学内容，或内容不完整、无法判断对错时选“不适用”。
2. after：划掉之后学生改写成的内容对不对（对／错／不确定／不适用）。划掉后没有改写，或改写的不是数学内容时选“不适用”。
3. type：修改类型，从下面选一个：纠错（原来错，改后对）、改错（原来对，改后错）、错改错（前后都错）、抵消（相消的项划掉，是演算记号）、笔误重写（字写错或写一半，重写同样内容）、等价改写（换个写法，意思相同）、换思路（放弃这步、调顺序或换方法）、删除多余（多写的删掉）、放弃（划掉后没有再写）、不确定。
只输出 <judgment>{{"struck": "…", "after": "…", "type": "…"}}</judgment>，不要解释。'''
GIVEN = '框里被划掉的内容（转写，公式在 $ $ 里）：{text}'
NOT_GIVEN = '框里被划掉的内容没有另外提供，请直接看图。'


def plain(markdown):
    """Question markdown for a prompt: an image link becomes [题图]."""
    return re.sub(r'!\[[^\]]*\]\([^)]*\)', '[题图]', markdown or '').strip()


def labels(path):
    """Region id -> the annotator's confirmed judgment (fields struck, after, type)."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for page in data['pages'].values():
        for r in page['revisions']:
            j = r.get('judgment')
            if j and r.get('judgment_confirmed'):
                out[r['id']] = {k: j[k] for k in FIELDS}
    return out


def build(args):
    cfg = load_config(args.config)
    d = cfg['data']
    diag = Path(d['diagnostics'])
    regions = [r for r in json.loads((diag/'regions.json').read_text()) if r['set'] == 'P']
    gold = labels(d['labels'])
    pages = {}
    for line in open(d['answer_groups'], encoding='utf-8'):
        group = json.loads(line)
        pages.update({i: group['question_global_id'] for i in group['image_ids']})
    questions = {q['question_global_id']: q for q in map(json.loads, open(d['questions'], encoding='utf-8'))}
    images = {r['line_id']: diag/'pack_read'/r['images']['raw'] for r in json.loads((diag/'pack_read'/'pack.json').read_text())['rows']
              if r['variant'] == 'R' and r['scope'] == 'S3'}
    readings = {}
    for line in open(d['readings'], encoding='utf-8'):
        p = json.loads(line)
        line_id, variant, scope = p['id'].split('|')
        if scope == 'S1' and variant in READ_FROM.values():
            readings[(line_id, variant)] = p['text']
    missing = [r['id'] for r in regions if r['id'] not in gold]
    require(len(missing) <= 1, f'regions without a confirmed judgment: {missing}')   # the one region the drafts skipped (not a revision)
    out = make_output_dir(args.output_dir)
    (out/'images').mkdir()
    rows, kept = [], []
    for k, r in enumerate(r for r in regions if r['id'] in gold):
        if r['split'] == 'test':
            raise RefusedError(f'{r["id"]} belongs to a test writer')
        q = questions[pages[r['page']]]
        image = f'images/{k:05d}.png'
        shutil.copyfile(images[r['id']], out/image)
        given = {'a': None, 'b': r['text'], **{c: readings[(r['id'], v)] for c, v in READ_FROM.items()}}
        for c in CONDITIONS:
            struck = NOT_GIVEN if given[c] is None else GIVEN.format(text=given[c].strip() or '（空）')
            rows.append({'id': f'{r["id"]}|{c}', 'line_id': r['id'], 'image_id': r['page'], 'kind': 'line', 'condition': c, 'split': r['split'],
                         'images': {'raw': image}, 'system': SYSTEM, 'boxes': [],
                         'prompt': PROMPT.format(problem=plain(q['problem_statement_markdown']), solution=plain(q['standard_solution_markdown']),
                                                 answer=plain(q['final_answer']), struck=struck)})
        kept.append({'id': r['id'], 'page': r['page'], 'question': q['question_global_id'], 'split': r['split'], 'gold': gold[r['id']]})
    write_json(out/'pack.json', {'schema': 'revrender-line-pack', 'rows': rows})
    write_json(out/'labels.json', kept)
    write_manifest(out, config=to_jsonable(args), settings=cfg, inputs={str(p): digest(p) for p in (
        Path(d['labels']), diag/'regions.json', Path(d['readings']), Path(d['questions']), Path(d['answer_groups']))},
        regions=len(kept), rows=len(rows), splits=dict(Counter(r['split'] for r in kept)))
    print(json.dumps({'regions': len(kept), 'rows': len(rows)}))


def parse(raw):
    """The three fields of a judge's output (None for a field it did not give, or gave outside the choices)."""
    raw = raw.rsplit('</think>', 1)[-1]
    found = re.search(r'<judgment>(.*?)(?:</judgment>|$)', raw, re.S)
    body = (found.group(1) if found else raw).strip()
    braces = re.search(r'\{.*\}', body, re.S)
    try:
        value = json.loads(braces.group(0)) if braces else {}
    except json.JSONDecodeError:
        value = {}
    choices = {'struck': RIGHT, 'after': RIGHT, 'type': TYPES}
    return {k: value.get(k) if value.get(k) in choices[k] else None for k in FIELDS}


def macro_f1(gold, guess):
    """F1 averaged over the classes the labels hold."""
    scores = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in zip(gold, guess))
        fp = sum(g != c and p == c for g, p in zip(gold, guess))
        fn = sum(g == c and p != c for g, p in zip(gold, guess))
        scores.append(2*tp/(2*tp+fp+fn) if tp else 0.)
    return float(np.mean(scores))


def score(args):
    cfg = load_config(args.config)
    kept = json.loads((args.pack/'labels.json').read_text())
    by_id = {r['id']: r for r in kept}
    guesses, parsed = defaultdict(dict), Counter()
    models = []
    for item in args.judgments:
        model, path = item.split('=', 1)
        models.append(model)
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            p = json.loads(line)
            region, condition = p['id'].rsplit('|', 1)
            fields = parse(p['raw'])
            guesses[(model, condition)][region] = fields
            parsed[(model, condition)] += all(fields.values())
    n, resamples = len(kept), cfg['score']['bootstrap']
    table, comparisons = {}, {}
    for model in models:
        for c in CONDITIONS:
            g = guesses[(model, c)]
            require(len(g) == n, f'{model} {c}: {len(g)} judgments for {n} regions')
            entry = {'parsed': parsed[(model, c)]/n}
            for f in FIELDS:
                gold, guess = [r['gold'][f] for r in kept], [g[r['id']][f] for r in kept]
                entry[f] = {'accuracy': float(np.mean([a == b for a, b in zip(gold, guess)])), 'macro_f1': macro_f1(gold, guess)}
            table[f'{model}/{c}'] = entry
        for f in FIELDS:
            right = {c: [float(guesses[(model, c)][r['id']][f] == r['gold'][f]) for r in kept] for c in CONDITIONS}
            for a, b in (('a', 'b'), ('a', 'c1'), ('a', 'c2'), ('c1', 'b'), ('c1', 'c2'), ('c2', 'b')):
                mean, low, high = bootstrap([y-x for x, y in zip(right[a], right[b])], [r['page'] for r in kept], resamples)
                comparisons[f'{model}/{f}/{b}-{a}'] = {'mean': mean, 'low': low, 'high': high}
    d = cfg['decisions']
    decision = None
    if d['primary'] in models:
        gain = comparisons[f'{d["primary"]}/{d["field"]}/b-a']
        accuracy = {c: table[f'{d["primary"]}/{c}'][d['field']]['accuracy'] for c in CONDITIONS}
        span = accuracy['b']-accuracy['a']
        decision = {'H4 knowing the struck content matters': gain['mean'] >= d['gain'] and gain['low'] > 0, 'gain': gain,
                    'position': {c: (accuracy[c]-accuracy['a'])/span if span else None for c in ('c1', 'c2')}}
    gold_counts = {f: dict(Counter(r['gold'][f] for r in kept)) for f in FIELDS}
    summary = {'regions': n, 'models': models, 'table': table, 'comparisons': comparisons, 'decision': decision, 'gold': gold_counts}
    out = make_output_dir(args.output_dir)
    write_json(out/'summary.json', summary)
    (out/'report.md').write_text(report(summary, kept, guesses), encoding='utf-8')
    write_manifest(out, config=to_jsonable(args), settings=cfg, inputs={**{str(args.pack/'labels.json'): digest(args.pack/'labels.json')},
                                                                     **{i.split('=', 1)[1]: digest(Path(i.split('=', 1)[1])) for i in args.judgments}})
    print(json.dumps({'decision': decision}, ensure_ascii=False))


def report(summary, kept, guesses):
    pct = lambda x: f'{100*x:.1f}'
    lines = ['# 第二阶段：读对被划内容对判断修改有多大帮助', '', f'{summary["regions"]} 处，标签为用户确认的对错。准确率 / 宏 F1 均为百分比。', '']
    if summary['decision']:
        d = summary['decision']
        g = d['gain']
        lines += ['## 判定（§12.1，32B，被划内容准确率）', '',
                  f'- (b) − (a) = {pct(g["mean"])}（{pct(g["low"])} ~ {pct(g["high"])}）：' + ('成立' if d['H4 knowing the struck content matters'] else '不成立'),
                  '- (c − a) / (b − a)：' + '，'.join(f'{c} {"—" if v is None else f"{v:.2f}"}' for c, v in d['position'].items()), '']
    lines += ['## 标签分布', '']
    for f, counts in summary['gold'].items():
        lines.append(f'- {FIELD_NAMES[f]}：' + '，'.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    for model in summary['models']:
        lines += ['', f'## {model}', '', '| 条件 | 解析率 | 被划内容 | 改后 | 类型 |', '|---|---|---|---|---|']
        for c in CONDITIONS:
            e = summary['table'][f'{model}/{c}']
            lines.append(f'| ({c}) {NAMES[c]} | {pct(e["parsed"])} | ' + ' | '.join(f'{pct(e[f]["accuracy"])} / {pct(e[f]["macro_f1"])}' for f in FIELDS) + ' |')
        lines += ['', '| 比较（准确率差） | 被划内容 | 改后 | 类型 |', '|---|---|---|---|']
        for pair in ('b-a', 'c1-a', 'c2-a', 'b-c1', 'c2-c1', 'b-c2'):
            cells = [summary['comparisons'][f'{model}/{f}/{pair}'] for f in FIELDS]
            lines.append(f'| {pair} | ' + ' | '.join(f'{pct(x["mean"])}（{pct(x["low"])} ~ {pct(x["high"])}）' for x in cells) + ' |')
        lines += ['', '被划内容的混淆（行：标签；列：(a) → (b) 的判断）', '', '| 标签 | ' + ' | '.join(RIGHT) + ' | 无 |', '|---|' + '---|'*(len(RIGHT)+1)]
        for g in RIGHT:
            cells = []
            for p in (*RIGHT, None):
                cells.append('→'.join(str(sum(r['gold']['struck'] == g and guesses[(model, c)][r['id']]['struck'] == p for r in kept)) for c in ('a', 'b')))
            lines.append(f'| {g} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--config', type=Path, required=True)
    b.add_argument('--output-dir', type=Path, required=True)
    s = sub.add_parser('score')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('--pack', type=Path, required=True)
    s.add_argument('--judgments', action='append', required=True, help='NAME=predictions.jsonl of recognize_lines --conditions from_pack')
    s.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    {'build': build, 'score': score}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
