"""Phase 2 of the diagnostic study: the judge's output, the measures and the decision (eval.revision_judgment)."""
import json

import pytest

from revrender import recognize_lines as R
from revrender.eval import revision_judgment as J


def test_the_judgment_is_parsed_and_off_list_values_are_none():
    raw = '<think>maybe <judgment>{"struck": "错"}</judgment></think><judgment>{"struck": "对", "after": "错", "type": "纠错"}</judgment>'
    assert J.parse(raw) == {'struck': '对', 'after': '错', 'type': '纠错'}
    assert J.parse('<judgment>{"struck": "正确", "after": "错", "type": "纠错"}</judgment>')['struck'] is None
    assert J.parse('no json at all') == {'struck': None, 'after': None, 'type': None}


def test_macro_f1_averages_over_the_label_classes():
    assert J.macro_f1(['对', '对', '错'], ['对', '对', '错']) == 1.
    assert J.macro_f1(['对', '错'], ['对', '对']) == pytest.approx((2/3+0)/2)


def test_a_pack_row_brings_its_own_words_and_system_prompt():
    row = {'prompt': '判断', 'system': 'teacher'}
    assert R.prompt('from_pack', row) == '判断'
    assert R.messages(__import__('PIL.Image').Image.new('RGB', (2, 2)), '判断', 'teacher')[0]['content'] == 'teacher'


def test_score_takes_the_decision(tmp_path):
    kept = [{'id': f'r{k}', 'page': f'p{k // 2}', 'question': 'q', 'split': 'train',
             'gold': {'struck': '错' if k % 2 else '对', 'after': '对', 'type': '纠错'}} for k in range(20)]
    pack = tmp_path/'pack'
    pack.mkdir()
    (pack/'labels.json').write_text(json.dumps(kept, ensure_ascii=False))
    rows = []
    for r in kept:
        for c in J.CONDITIONS:
            struck = r['gold']['struck'] if c != 'a' else '对'        # without the struck text the judge always says 对
            rows.append(json.dumps({'id': f'{r["id"]}|{c}', 'raw': '<judgment>' + json.dumps({'struck': struck, 'after': '对', 'type': '纠错'}, ensure_ascii=False) + '</judgment>'}, ensure_ascii=False))
    (tmp_path/'p.jsonl').write_text('\n'.join(rows))
    config = tmp_path/'c.toml'
    config.write_text((J.Path(__file__).parents[1]/'configs'/'revision_judgment.toml').read_text().replace('bootstrap = 10000', 'bootstrap = 200'))
    J.main(['score', '--config', str(config), '--pack', str(pack), '--judgments', f'qwen3vl32b={tmp_path/"p.jsonl"}', '--output-dir', str(tmp_path/'s')])
    summary = json.loads((tmp_path/'s'/'summary.json').read_text())
    assert summary['table']['qwen3vl32b/a']['struck']['accuracy'] == .5 and summary['table']['qwen3vl32b/b']['struck']['accuracy'] == 1.
    assert summary['decision']['H4 knowing the struck content matters'] and summary['decision']['position']['c1'] == 1.
    assert '## 判定' in (tmp_path/'s'/'report.md').read_text()
