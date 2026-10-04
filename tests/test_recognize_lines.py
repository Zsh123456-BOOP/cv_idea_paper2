"""The conditions a line is read under, and the scores of struck-text leakage, valid-text loss and harm to clean lines."""
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
    assert [R.CONDITIONS[c][0] for c in ('raw_neutral', 'erased_final', 'overlay_final', 'hint_final', 'blanked_final')] == ['raw', 'erased', 'overlay', 'raw', 'blanked']
    assert R.prompt('blanked_neutral', row) == neutral and R.prompt('blanked_final', row) == final
    assert [R.CONDITIONS[c][2] for c in ('raw_neutral', 'hide_band_neutral', 'hide_ink_neutral', 'soft_band_neutral', 'blanked_hide_band_neutral')] == [None, 'band', 'ink', 'soft_band', 'band']
    assert R.prompt('hide_band_neutral', row) == neutral and R.TOKEN_RULES == {'hide_band_neutral', 'hide_ink_neutral', 'soft_band_neutral', 'blanked_hide_band_neutral'}


def test_a_token_condition_needs_the_transformers_backend(tmp_path):
    with pytest.raises(Exception, match='--backend hf'):
        R.main(['--pack', str(tmp_path), '--model', str(tmp_path), '--conditions', 'hide_band_neutral', '--output-dir', str(tmp_path/'out')])


def test_a_request_has_one_image_and_the_reading_is_what_is_inside_the_tags():
    request = R.messages(Image.new('RGB', (8, 4), 'white'), 'x')
    content = request[1]['content']
    assert request[0]['role'] == 'system' and [c['type'] for c in content] == ['image_url', 'text'] and content[0]['image_url']['url'].startswith('data:image/png;base64,')
    assert R.reading_of('<transcription>\n$x=3$\n</transcription>') == '$x=3$' and R.reading_of('<transcription>x=') == 'x='
    assert R.reading_of('plain text') == 'plain text' and R.reading_of('<transcription>[空]</transcription>') == '' and R.reading_of('') == ''


PACK = [
    {'id': '0', 'line_id': 'a:L1', 'image_id': 'a_3_1', 'kind': 'pair', 'text_before': 'x=3,y=5', 'text_after': 'x=7,y=5', 'ink_kind': 'strike_rewrite'},
    {'id': '1', 'line_id': 'a:L2', 'image_id': 'a_3_1', 'kind': 'pair', 'text_before': 'y=2', 'text_after': '', 'no_rewrite': True, 'ink_kind': 'whole_line'},
    {'id': '2', 'line_id': 'b:L1', 'image_id': 'b_4_1', 'kind': 'pair', 'text_before': 'z=1+w', 'text_after': 'z=5+w', 'ink_kind': 'strike_rewrite'},
    {'id': '3', 'line_id': 'c:L1', 'image_id': 'c_5_1', 'kind': 'no_edit'},
]


def predictions(model, condition, texts):
    return [{'model': model, 'condition': condition, 'id': str(i), 'text': t} for i, t in enumerate(texts)]


def test_leakage_valid_loss_blank_and_harm_to_clean_lines_are_counted_per_model_and_condition():
    rows = (predictions('m', 'raw_neutral', ['x=3,y=5', 'y=2', 'z=1+w', 'a=b'])
            + predictions('m', 'raw_final', ['x=7,y=5', 'y=2', 'z=5+w', 'a=b'])
            + predictions('m', 'erased_final', ['x=7', '', 'z=+w', 'a=c']))
    scores = leakage.score(PACK, rows)['m']
    raw = scores['raw_neutral']
    assert raw['leak']['n'] == 3 and raw['leak']['mean'][0] == 1. and raw['rewrite']['mean'][0] == 0. and raw['valid']['n'] == 2 and raw['valid']['mean'][0] == 1.
    assert raw['blank']['n'] == 1 and raw['blank']['mean'][0] == 0.
    final = scores['raw_final']
    assert final['leak']['mean'][0] == pytest.approx(1/3) and final['rewrite']['mean'][0] == 1. and final['controls']['mean_change'] == 0.   # y=2 still read from the struck line
    erased = scores['erased_final']
    assert erased['leak']['mean'][0] == 0. and erased['rewrite']['mean'][0] == .5 and erased['blank']['mean'][0] == 1.
    assert erased['valid']['mean'][0] == pytest.approx((2/6+1.)/2)                     # 'x=7' holds two of the six valid characters of the first line
    assert erased['controls']['share_moved'] == 1. and erased['controls']['mean_change'] > leakage.MOVED
    assert erased['leak']['difference_from_raw_final'][0] == pytest.approx(-1/3) and 'difference_from_raw_final' not in final['leak']
    assert erased['by_ink_kind']['whole_line']['leak'] == 0. and raw['by_ink_kind']['strike_rewrite']['lines'] == 2
    assert raw['leak']['pooled'] == 1. and raw['leak']['masked'] == {'n': 0, 'mean': None} and raw['leak']['unmasked']['n'] == 3 and raw['abstain'] == 0.
    assert erased['abstain'] == 1/3 and erased['valid']['change_from_raw_final']['n'] == 2 and erased['valid']['change_from_raw_final']['lose'] == .5
    assert raw['leak']['strict'] <= raw['leak']['mean'][0] <= raw['leak']['lenient']
    assert not leakage.labelled({'kind': 'pair', 'text_before': 'x=3', 'before_unreadable': True}) and not leakage.labelled({'kind': 'pair', 'text_before': ''})
    assert not leakage.labelled({'kind': 'no_edit'}) and leakage.labelled(PACK[0])


def test_the_references_are_what_the_labels_allow_not_what_a_model_reads():
    refs = leakage.references(PACK)
    assert refs['exactly_after']['leak']['mean'] == 0. and refs['exactly_after']['rewrite']['mean'] == 1. and refs['exactly_after']['valid']['mean'] == 1.
    assert refs['exactly_before']['leak']['mean'] == 1. and refs['empty']['valid']['mean'] == 0. and refs['empty']['leak']['mean'] == 0.
    assert set(refs) == {'exactly_after', 'exactly_before', 'empty', 'other_line_of_the_same_length'}
    assert refs['other_line_of_the_same_length']['leak']['n'] == 3 and 0. <= refs['other_line_of_the_same_length']['leak']['mean'] < 1.


def test_the_tool_scores_files_and_is_create_only(tmp_path):
    pack = tmp_path/'pack'
    pack.mkdir()
    (pack/'pack.json').write_text(json.dumps({'rows': PACK}))
    path = tmp_path/'predictions.jsonl'
    path.write_text(''.join(json.dumps(p)+'\n' for p in predictions('m', 'raw_neutral', ['x=3,y=5', 'y=2', 'z=1+w', 'a=b'])))
    out = tmp_path/'scores.json'
    assert leakage.main(['--pack', str(pack), '--predictions', str(path), '--output', str(out)]) == 0
    assert json.loads(out.read_text())['scores']['m']['raw_neutral']['leak']['n'] == 3 and 'exactly_after' in json.loads(out.read_text())['references']
    with pytest.raises(FileExistsError):
        leakage.main(['--pack', str(pack), '--predictions', str(path), '--output', str(out)])
