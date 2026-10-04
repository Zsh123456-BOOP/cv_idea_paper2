"""The reader viewer: a card per labelled line with the labels, each run's reading and its shares; the images copied beside the page."""
import json

import pytest
from PIL import Image

from revrender.tools import reader_view as V


def pack_with_images(tmp_path):
    folder = tmp_path/'pack'
    (folder/'assets').mkdir(parents=True)
    rows = []
    for n, (before, after, kind, masked) in enumerate([('x=3,y=5', 'x=7,y=5', 'strike_rewrite', 40), ('y=2', '', 'whole_line', 0)]):
        for k in ('raw', 'blanked'):
            Image.new('RGB', (30, 10), 'white').save(folder/'assets'/f'{n}_{k}.png')
        rows.append({'id': str(n), 'kind': 'pair', 'image_id': f'w{n}_3_1', 'text_before': before, 'text_after': after, 'ink_kind': kind, 'masked_pixels': masked, 'size': [30, 10],
                     'no_rewrite': not after, 'images': {'raw': f'assets/{n}_raw.png', 'blanked': f'assets/{n}_blanked.png'}})
    rows.append({'id': '2', 'kind': 'no_edit', 'image_id': 'w2_3_1', 'size': [30, 10], 'images': {'raw': 'assets/0_raw.png', 'blanked': 'assets/0_blanked.png'}})
    (folder/'pack.json').write_text(json.dumps({'schema': 'revrender-line-pack', 'rows': rows}))
    return folder


def predictions(path, condition, texts):
    path.write_text(''.join(json.dumps({'condition': condition, 'id': str(i), 'text': t}, ensure_ascii=False)+'\n' for i, t in enumerate(texts)))


def test_a_run_is_a_name_a_file_and_a_condition():
    assert V.parse_run('base=out/p.jsonl:raw_final') == ('base', V.Path('out/p.jsonl'), 'raw_final')
    with pytest.raises(Exception, match='NAME=PREDICTIONS'):
        V.parse_run('nonsense')


def test_the_site_has_a_card_for_each_labelled_line_with_the_shares_of_every_run(tmp_path):
    pack = pack_with_images(tmp_path)
    first, second = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    predictions(first, 'raw_final', ['x=3,y=5', 'y=2', 'a'])
    predictions(second, 'raw_final', ['x=7,y=5', '', 'a'])
    out = tmp_path/'site'
    assert V.main(['--pack', str(pack), '--run', f'base={first}:raw_final', '--run', f'adapter={second}:raw_final', '--output-dir', str(out)]) == 0
    page = (out/'index.html').read_text(encoding='utf-8')
    data = json.loads(page.split('<script type="application/json" id="data">')[1].split('</script>')[0].replace('<\\/', '</'))
    assert data['runs'] == ['base', 'adapter'] and [c['id'] for c in data['cards']] == ['0', '1']
    zero = data['cards'][0]
    assert zero['runs']['base']['leak'] == 1. and zero['runs']['adapter']['leak'] == 0. and zero['gap'] == 1. and zero['masked'] and not data['cards'][1]['masked']
    assert (out/'assets'/'0_raw.png').exists() and (out/'assets'/'1_blanked.png').exists() and json.loads((out/'run_manifest.json').read_text())['lines'] == 2
    with pytest.raises(FileExistsError):
        V.main(['--pack', str(pack), '--run', f'base={first}:raw_final', '--output-dir', str(out)])
