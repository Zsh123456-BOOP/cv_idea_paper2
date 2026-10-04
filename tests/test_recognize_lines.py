"""The conditions a line is read under, and the scores of struck-text leakage, rewrite keeping and harm to clean lines."""
import json

import pytest
from PIL import Image

from revrender import recognize_lines as R
from revrender.eval import leakage


def test_each_condition_says_one_thing_more_than_the_skeleton_and_a_hint_names_the_places():
    row = {'boxes': [[100, 200, 400, 800], [500, 100, 900, 900]]}
    neutral, final = R.prompt('raw_neutral', row), R.prompt('raw_final', row)
    assert neutral == R.SKELETON and final == R.SKELETON+'\n'+R.FINAL and R.prompt('erased_final', row) == final and R.prompt('erased_neutral', row) == neutral
    assert R.OVERLAY in R.prompt('overlay_final', row) and R.FINAL in R.prompt('overlay_final', row)
    hint = R.prompt('hint_final', row)
    assert '[100, 200, 400, 800]' in hint and '[500, 100, 900, 900]' in hint and hint.endswith(R.FINAL)
    assert '没有检测到涂改区域' in R.prompt('hint_final', {'boxes': []})
    assert [R.CONDITIONS[c][0] for c in ('raw_neutral', 'erased_final', 'overlay_final', 'hint_final')] == ['raw', 'erased', 'overlay', 'raw']


def test_a_request_has_one_image_and_the_reading_is_what_is_inside_the_tags():
    request = R.messages(Image.new('RGB', (8, 4), 'white'), 'x')
    content = request[1]['content']
    assert request[0]['role'] == 'system' and [c['type'] for c in content] == ['image_url', 'text'] and content[0]['image_url']['url'].startswith('data:image/png;base64,')
    assert R.reading_of('<transcription>\n$x=3$\n</transcription>') == '$x=3$' and R.reading_of('<transcription>x=') == 'x='
    assert R.reading_of('plain text') == 'plain text' and R.reading_of('<transcription>[空]</transcription>') == '' and R.reading_of('') == ''


PACK = [
    {'id': '0', 'line_id': 'a:L1', 'image_id': 'a_3_1', 'kind': 'pair', 'text_before': 'x=3', 'text_after': 'x=-3', 'ink_kind': 'strike_rewrite'},
    {'id': '1', 'line_id': 'a:L2', 'image_id': 'a_3_1', 'kind': 'pair', 'text_before': 'y=2', 'no_rewrite': True, 'ink_kind': 'whole_line'},
    {'id': '2', 'line_id': 'b:L1', 'image_id': 'b_4_1', 'kind': 'pair', 'text_before': 'z=1', 'text_after': 'z=5', 'ink_kind': 'strike_rewrite'},
    {'id': '3', 'line_id': 'c:L1', 'image_id': 'c_5_1', 'kind': 'no_edit'},
]


def predictions(model, condition, texts):
    return [{'model': model, 'condition': condition, 'id': str(i), 'text': t} for i, t in enumerate(texts)]


def test_leakage_keeping_blank_and_harm_to_clean_lines_are_counted_per_model_and_condition():
    rows = (predictions('m', 'raw_neutral', ['x=3 then x=-3', 'y=2', 'z=1 z=5', 'a=b'])
            + predictions('m', 'raw_final', ['x=-3', 'y=2', 'z=5', 'a=b'])
            + predictions('m', 'erased_final', ['x=-3', '', 'z', 'a=c']))
    scores = leakage.score(PACK, rows)['m']
    raw = scores['raw_neutral']
    assert raw['leak']['n'] == 3 and raw['leak']['share'][0] == 1. and raw['keep']['share'][0] == 1. and raw['blank']['share'][0] == 0.
    final = scores['raw_final']
    assert final['leak']['share'][0] == 1/3 and final['keep']['share'][0] == 1. and final['controls']['mean_change'] == 0.        # y=2 still read from the struck line
    erased = scores['erased_final']
    assert erased['leak']['share'][0] == 0. and erased['keep']['share'][0] == .5 and erased['blank']['share'][0] == 1.
    assert erased['controls']['share_moved'] == 1. and erased['controls']['mean_change'] > leakage.MOVED
    assert erased['leak']['difference_from_raw_final'][0] == pytest.approx(-1/3) and 'difference_from_raw_final' not in final['leak']
    assert erased['by_ink_kind']['whole_line']['leak'] == 0. and raw['by_ink_kind']['strike_rewrite']['lines'] == 2
    assert leakage.flags({'text_before': 'x=3', 'before_unreadable': True}, 'x=3')['leak'] is None and leakage.flags({'text_after': 'q'}, 'q')['keep'] is None


def test_the_tool_scores_files_and_is_create_only(tmp_path):
    pack = tmp_path/'pack'
    pack.mkdir()
    (pack/'pack.json').write_text(json.dumps({'rows': PACK}))
    path = tmp_path/'predictions.jsonl'
    path.write_text(''.join(json.dumps(p)+'\n' for p in predictions('m', 'raw_neutral', ['x=3', 'y=2', 'z=1', 'a=b'])))
    out = tmp_path/'scores.json'
    assert leakage.main(['--pack', str(pack), '--predictions', str(path), '--output', str(out)]) == 0
    assert json.loads(out.read_text())['scores']['m']['raw_neutral']['leak']['n'] == 3
    with pytest.raises(FileExistsError):
        leakage.main(['--pack', str(pack), '--predictions', str(path), '--output', str(out)])
