import json
from types import SimpleNamespace

from revrender.data import struck_pack


def test_method_split_fills_validation_by_regions_then_readable(tmp_path):
    rows = [{'id': f'{w}#{k}', 'writer': w, 'readable': k % 3 != 0} for w in (f'w{i:02d}' for i in range(30)) for k in range(4)]
    (tmp_path/'pack').mkdir(); (tmp_path/'pack'/'pack.json').write_text(json.dumps({'rows': rows}))
    struck_pack.method_split(SimpleNamespace(pack=tmp_path/'pack', output=tmp_path/'split.json'))
    split = json.loads((tmp_path/'split.json').read_text())
    order = sorted({r['writer'] for r in rows}, key=struck_pack.split_order)
    val = set(split['validation']); held = [r for r in rows if r['writer'] in val]
    assert set(order[:len(val)]) == val and not val & set(split['training'])
    assert len(held) >= struck_pack.SPLIT_REGIONS and sum(r['readable'] for r in held) >= struck_pack.SPLIT_READABLE
    fewer = [r for r in rows if r['writer'] in order[:len(val)-1]]
    assert len(fewer) < struck_pack.SPLIT_REGIONS or sum(r['readable'] for r in fewer) < struck_pack.SPLIT_READABLE
